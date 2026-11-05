"""Land-cover grounding of named SAE atoms.

For every land-cover class (water, forest, ...), the atoms whose names map to that class score each token
by their maximum activation. That score is evaluated with AUROC / AP against "class present", taken from
the 2020 land-cover maps (LC / NEG_LC). Baseline: atoms of a randomly initialised SAE, named the same way.
The MS-CLIP zero-shot baseline lives in `landcover_zeroshot.py`.

See `landcover_utils.py` for the token <-> LC pixel alignment.
"""
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.concept_bottleneck.eval.landcover_utils import (
    N_LC_IDS, LandCoverTokenDataset, auroc_ap, build_group_matrix, class_text_embeddings, compute_class_metrics,
    load_year_dataset, nanmean, plot_auroc_per_class, ungrouped_lc_ids,
)
from experiments.concept_bottleneck.eval.utils import build_untrained_sae, compute_text_dist, load_img_model, load_trained_sae, log_metrics
from experiments.concept_bottleneck.sae.act_datamodule import OnlineActDataModule
from experiments.concept_bottleneck.sae.sae_module import plSAE
from experiments.concept_bottleneck.sae.utils import get_pylogger
from overcomplete.sae.train import extract_input
from src.constants import CONFIG_PATH

log = get_pylogger(__name__)

DICTIONARIES = ("trained", "untrained")
ACTIVATIONS = ("post", "pre")  # post: after ReLU + TopK, pre: encoder pre-codes

# Reference categorical palette (fixed order, one slot per method).
_METHOD_COLORS = {
    "trained_post": "#2a78d6",
    "trained_pre": "#eb6834",
    "untrained_post": "#1baf7a",
    "untrained_pre": "#eda100",
}


# ----------------------------------------------------------------------------------------------------------
# Atom -> class mapping
# ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def atom_class_similarity(
    sae: plSAE, atom_indices: torch.Tensor, class_emb: torch.Tensor, source: str
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Cosine of each atom (or its nearest vocab term) to each class: [n_atoms, n_classes]."""
    dictionary = sae.net.get_dictionary().detach().float().cpu()[atom_indices]
    vocab_emb = sae.label_vocab_emb.float().cpu()
    term_sim, term_idx = compute_text_dist(dictionary, vocab_emb)
    if source == "term":
        query = vocab_emb[term_idx]
    elif source == "atom":
        query = dictionary
    else:
        raise ValueError(f"Unknown mapping source {source}")
    sim = F.normalize(query, dim=-1) @ class_emb.cpu().T
    return sim, term_idx.numpy(), term_sim.numpy()


def assign_atoms(sim: torch.Tensor, threshold: float, exclusive: bool) -> torch.Tensor:
    """Boolean [n_atoms, n_classes] assignment of atoms to classes."""
    assigned = sim >= threshold
    if exclusive:
        assigned &= F.one_hot(sim.argmax(dim=1), sim.shape[1]).bool()
    return assigned


def save_mapping(
    path: Path,
    atom_indices: torch.Tensor,
    sim: torch.Tensor,
    term_idx: np.ndarray,
    term_sim: np.ndarray,
    vocab_terms: Optional[List[str]],
    class_names: List[str],
    assigned: torch.Tensor,
) -> None:
    df = pd.DataFrame({
        "atom": atom_indices.numpy(),
        "term": [vocab_terms[i] for i in term_idx] if vocab_terms is not None else term_idx,
        "atom_term_cosine": term_sim,
    })
    for class_idx, name in enumerate(class_names):
        df[f"cos_{name}"] = sim[:, class_idx].numpy()
    df["best_class"] = [class_names[i] for i in sim.argmax(dim=1).numpy()]
    df["best_class_cosine"] = sim.max(dim=1).values.numpy()
    df["assigned_class"] = [";".join(class_names[i] for i in np.flatnonzero(row)) for row in assigned.numpy()]
    df.sort_values("best_class_cosine", ascending=False).to_csv(path, index=False)


# ----------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval_landcover_alignment")
def evaluate_landcover_alignment(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir) / f"landcover_alignment_{cfg.lc_year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    lc_classes = OmegaConf.to_container(cfg.lc_classes, resolve=True)
    class_keys = list(lc_classes)
    class_names = [lc_classes[key]["name"] for key in class_keys]
    group_matrix = build_group_matrix(lc_classes)
    mapping_cfg = cfg.mapping
    # Below we combine the sweep thresholds and the sim_threshold to ensure that we always evaluate the sim_threshold.
    thresholds = sorted(set(float(t) for t in mapping_cfg.coverage_sweep) | {float(mapping_cfg.sim_threshold)})

    ### Model ###
    log.info(f"Instantiating MS-CLIP Model at <{cfg.model_path}>")
    model = load_img_model(cfg, device, logger=log)
    assert cfg.MODEL.test_max_seq_len is None, "For evaluation, test_max_seq_len must be None."
    grid, patch_size = model.model.H_patch, model.model.patch_size
    assert model.model.H_patch == model.model.W_patch

    ### Data: tiles of lc_year ###
    dataset = load_year_dataset(cfg)
    log.info(f"{len(dataset.data_paths)} tiles of year {cfg.lc_year} ({grid}x{grid} tokens of {patch_size}px)")

    lc_dataset = LandCoverTokenDataset(
        dataset, list(cfg.lc_dirs), group_matrix, cfg.MODEL.input_img_res, cfg.MODEL.img_res, grid, patch_size
    )
    loader = DataLoader(lc_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    act_datamodule = OnlineActDataModule(
        model=model, train_dataset=dataset, device=str(device), batch_size=1, test_batch_size=1
    )
    act_datamodule.load_input_norm(cfg.sae_ckpt_path)

    ### SAEs and atom -> class mapping ###
    saes = {"trained": load_trained_sae(cfg, device, logger=log), "untrained": build_untrained_sae(cfg, device)}
    if cfg.alive_feature_path is not None and Path(cfg.alive_feature_path).exists():
        trained_alive = torch.load(cfg.alive_feature_path, map_location="cpu").bool()
    else:
        log.info("No alive mask found, using all trained atoms")
        trained_alive = torch.ones(saes["trained"].net.nb_concepts, dtype=torch.bool)
    alive_indices = {
        "trained": torch.where(trained_alive)[0],
        "untrained": torch.arange(saes["untrained"].net.nb_concepts),
    }

    text_emb = class_text_embeddings(model, class_names, cfg.prompt_template, device)
    vocab_terms = pd.read_csv(cfg.vocab_csv_path)["term"].astype(str).tolist() if cfg.get("vocab_csv_path") else None

    # assignments[dict_name][threshold] -> {class_idx: LongTensor of SAE atom indices}
    assignments: Dict[str, Dict[float, Dict[int, torch.Tensor]]] = {}
    for dict_name in DICTIONARIES:
        sim, term_idx, term_sim = atom_class_similarity(
            saes[dict_name], alive_indices[dict_name], text_emb[mapping_cfg.class_text], mapping_cfg.source
        )
        # Best cosine per atom, for logging the distribution of best-class similarities. Threshold at top-percentile.
        best = sim.max(dim=1).values.numpy()
        log.info(
            f"[{dict_name}] best-class cosine quantiles (5/25/50/75/95/max): "
            + " / ".join(f"{q:.3f}" for q in np.quantile(best, [0.05, 0.25, 0.5, 0.75, 0.95, 1.0]))
        )
        assignments[dict_name] = {}
        for t in thresholds:
            assigned = assign_atoms(sim, t, mapping_cfg.exclusive) # Exclusion is important
            # This provide the mapping of each class to the atoms that are assigned to it, for each threshold.
            assignments[dict_name][t] = {
                c: alive_indices[dict_name][assigned[:, c]] for c in range(len(class_keys)) if assigned[:, c].any()
            }
        save_mapping(
            output_dir / f"atom_class_mapping_{dict_name}.csv", alive_indices[dict_name], sim, term_idx, term_sim,
            vocab_terms, class_keys, assign_atoms(sim, float(mapping_cfg.sim_threshold), mapping_cfg.exclusive),
        )

    ### One pass over the tokens ###
    # scores[(dict_name, activation, threshold, class_idx)] -> list of per-batch arrays
    scores: Dict[tuple, List[np.ndarray]] = {}
    fracs: List[np.ndarray] = []
    lc_pixel_counts = np.zeros(N_LC_IDS, dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Scoring {cfg.lc_year} tokens"):
            valid = batch["lc_valid"].reshape(-1)
            frac = batch["lc_frac"].reshape(-1, len(class_keys))
            lc_pixel_counts += batch["lc_counts"].sum(dim=0).numpy()
            act_batch = act_datamodule.on_after_batch_transfer(batch, 0)
            x = extract_input({k: v.to(device) for k, v in act_batch.items()})  # [P, D]
            x = x[valid.to(device)]
            fracs.append(frac[valid].numpy())

            for dict_name in DICTIONARIES:
                pre, z, _ = saes[dict_name](x)
                codes = {"post": z, "pre": pre}
                for t, class_atoms in assignments[dict_name].items():
                    for c, atoms in class_atoms.items():
                        atoms = atoms.to(device)
                        for act in ACTIVATIONS:
                            scores.setdefault((dict_name, act, t, c), []).append(
                                codes[act][:, atoms].max(dim=-1).values.float().cpu().numpy()
                            )

    frac_all = np.concatenate(fracs, axis=0)  # [N, n_classes]
    scores_all = {k: np.concatenate(v, axis=0) for k, v in scores.items()}
    log.info(f"{frac_all.shape[0]} valid tokens")

    ungrouped = ungrouped_lc_ids(lc_pixel_counts, group_matrix)
    if ungrouped:
        log.warning(f"Land-cover ids present but not evaluated (pixels): {ungrouped}")

    ### Metrics ###
    pos_t, sweep = float(cfg.pos_threshold), [float(t) for t in cfg.threshold_sweep]
    sim_t = float(mapping_cfg.sim_threshold)
    rows = []
    for c, key in enumerate(class_keys):
        for dict_name in DICTIONARIES:
            for act in ACTIVATIONS:
                key_scores = (dict_name, act, sim_t, c)
                n_atoms = len(assignments[dict_name][sim_t].get(c, []))
                if key_scores in scores_all:
                    metrics = compute_class_metrics(scores_all[key_scores], frac_all[:, c], pos_t, sweep, act == "post")
                else:
                    metrics = {}
                rows.append({"class": key, "method": f"{dict_name}_{act}", "n_atoms": n_atoms, **metrics})
    per_class = pd.DataFrame(rows)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)

    # Coverage sweep over mapping thresholds
    sweep_rows = []
    for dict_name in DICTIONARIES:
        for t in thresholds:
            covered = sorted(assignments[dict_name][t])
            row = {"dictionary": dict_name, "sim_threshold": t, "coverage": len(covered),
                   "mean_atoms_per_covered_class": nanmean([len(assignments[dict_name][t][c]) for c in covered])}
            for act in ACTIVATIONS:
                aurocs = [auroc_ap(scores_all[(dict_name, act, t, c)], frac_all[:, c], pos_t)[0] for c in covered]
                row[f"macro_auroc_{act}"] = nanmean(aurocs)
            sweep_rows.append(row)
    coverage_sweep = pd.DataFrame(sweep_rows)
    coverage_sweep.to_csv(output_dir / "coverage_sweep.csv", index=False)

    ### Summary ###
    n_classes = len(class_keys)

    def macro(method: str, metric: str) -> float:
        df = per_class[per_class["method"] == method]
        return nanmean(df[metric].tolist()) if metric in df else float("nan")

    summary: Dict[str, Dict[str, tuple]] = {
        "Setup": {
            "caveat": (f"year {cfg.lc_year} is in the '{cfg.source_split}' split used to train the SAE "
                       "(unsupervised, never saw LC labels)", ""),
            "n_tiles": (len(dataset.data_paths), ""),
            "n_valid_tokens": (int(frac_all.shape[0]), ""),
            "token_pixels": (f"{patch_size}x{patch_size} LC pixels (10 m)", ""),
            "positive_rule": (f"fraction >= {pos_t:g} vs fraction == 0 (mixed tokens dropped)", ""),
            "mapping": (f"source={mapping_cfg.source}, class_text={mapping_cfg.class_text}, "
                        f"cos >= {sim_t:g}, exclusive={mapping_cfg.exclusive}", ""),
            "prompt_template": (cfg.prompt_template, ""),
            "ungrouped_lc_ids": (str(ungrouped) if ungrouped else "none", ""),
        },
    }
    for dict_name in DICTIONARIES:
        covered = [class_keys[c] for c in sorted(assignments[dict_name][sim_t])]
        uncovered = [k for k in class_keys if k not in covered]
        section = {
            "coverage": (f"{len(covered)}/{n_classes}", "classes with at least one named atom"),
            "covered_classes": (", ".join(covered) if covered else "none", ""),
            "uncovered_classes": (", ".join(uncovered) if uncovered else "none", ""),
        }
        for act in ACTIVATIONS:
            method = f"{dict_name}_{act}"
            section[f"macro_auroc_{act}"] = (macro(method, "auroc"), "mean over covered classes")
            section[f"macro_ap_{act}"] = (macro(method, "ap"), "mean over covered classes")
        summary[f"{dict_name.capitalize()} dictionary"] = section
    summary["Coverage sweep"] = {
        f"{r.dictionary}@{r.sim_threshold:g}": (
            f"coverage {r.coverage}/{n_classes}, atoms/class {r.mean_atoms_per_covered_class:.1f}, "
            f"AUROC post {r.macro_auroc_post:.3f}, pre {r.macro_auroc_pre:.3f}", "")
        for r in coverage_sweep.itertuples()
    }
    log_metrics(output_dir, "landcover_alignment", summary)

    if cfg.get("plot", True):
        plot_auroc_per_class(per_class, class_keys, output_dir / "auroc_per_class.png", _METHOD_COLORS,
                             title="Land-cover AUROC per class (missing bar = no named atom)")
    log.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    evaluate_landcover_alignment()
