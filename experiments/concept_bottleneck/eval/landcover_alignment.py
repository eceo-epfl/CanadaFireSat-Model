"""Land-cover grounding of SAE atoms, with the same metrics as the MS-CLIP baseline (`landcover_zeroshot.py`).

For every land-cover class (water, forest, ...), alive atoms are ranked by the cosine between the class text and
either the atom's nearest vocabulary term ("term") or its dictionary vector ("atom"). Rankings are exclusive: an
atom only ranks for its closest class, so a class can have fewer than n atoms. The top-n atoms of a class score each
token by their maximum activation, after (post) or before (pre) ReLU + TopK. Baseline: a randomly initialised SAE.

Reported per method `{dictionary}_{source}_{class_text}_top{n}_{activation}`:
  - per class: AUROC / AP against "class present" (fraction >= pos_threshold vs fraction == 0)
  - across classes: accuracy / balanced accuracy, argmax over the class scores, on tokens whose dominant class covers
    >= accuracy_min_fraction of the token. Post tokens with no active atom have no prediction: they count as errors
    in the all-token metrics and are excluded from the *_active metrics.
  - post only, per class: precision of the tokens where the top-n set is active, and how often it is active
  - pre only, per class: precision of the tokens above the top_percentiles of the class score, as in zero-shot

With activation_norm="zscore", each atom's activations are z-scored (mean and std over the evaluated tokens) before
taking the max over a set, so that atoms with different scales are comparable. See `normalize_codes`.

Ground truth: 2020 land-cover maps (LC / NEG_LC). See `landcover_utils.py` for the token <-> pixel alignment.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.concept_bottleneck.eval.landcover_utils import (
    N_LC_IDS, LandCoverTokenDataset, build_group_matrix, class_text_embeddings, compute_class_metrics,
    load_year_dataset, multiclass_accuracy, nanmean, top_percentile_precision, ungrouped_lc_ids,
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
_EMPTY_CLASS_SCORE = {"post": 0.0, "pre": -float("inf")}  # A class without atoms can never be predicted

# Reference categorical palette (fixed order, one slot per line).
_SERIES_COLORS = {
    ("trained", "post"): "#2a78d6",
    ("trained", "pre"): "#eb6834",
    ("untrained", "post"): "#1baf7a",
    ("untrained", "pre"): "#eda100",
}
_TEXT_COLOR = "#1f2328"
_MUTED_COLOR = "#6e7781"


# ----------------------------------------------------------------------------------------------------------
# Atom -> class ranking
# ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def atom_class_similarity(
    sae: plSAE, atom_indices: torch.Tensor, class_emb: torch.Tensor, source: str
) -> tuple[torch.Tensor, Optional[np.ndarray], Optional[np.ndarray]]:
    """Cosine of each atom (or its nearest vocab term) to each class: [n_atoms, n_classes].

    Also returns each atom's nearest term and its cosine, for display, or None when the SAE has no vocabulary.
    """
    dictionary = sae.net.get_dictionary().detach().float().cpu()[atom_indices]
    term_sim, term_idx, vocab_emb = None, None, None
    if sae.label_vocab_emb is not None:
        vocab_emb = sae.label_vocab_emb.float().cpu()
        term_sim, term_idx = compute_text_dist(dictionary, vocab_emb)
        term_sim, term_idx = term_sim.numpy(), term_idx.numpy()

    if source == "term":
        if vocab_emb is None:
            raise ValueError('mapping.sources contains "term" but this SAE has no vocabulary (label_path): use ["atom"]')
        query = vocab_emb[term_idx]
    elif source == "atom":
        query = dictionary
    else:
        raise ValueError(f"Unknown mapping source {source}")
    sim = F.normalize(query, dim=-1) @ class_emb.cpu().T
    return sim, term_idx, term_sim


def rank_atoms(sim: np.ndarray, max_n: int) -> List[np.ndarray]:
    """Per class, positions of the atoms closest to that class, best first: up to max_n, possibly fewer or none."""
    closest_class = sim.argmax(axis=1)
    ranked = []
    for c in range(sim.shape[1]):
        atoms = np.flatnonzero(closest_class == c)
        # Stable sort: atoms with the same cosine (same nearest term with source="term") keep their index order.
        order = np.argsort(-sim[atoms, c], kind="stable")
        ranked.append(atoms[order][:max_n])
    return ranked


def padded_ranking(ranked: List[np.ndarray], atom_indices: np.ndarray, max_n: int) -> torch.Tensor:
    """[n_classes, max_n] SAE atom indices. Short lists repeat their last atom, which leaves the running max unchanged."""
    rows = []
    for atoms in ranked:
        sae_atoms = atom_indices[atoms] if len(atoms) else np.zeros(1, dtype=np.int64)  # Empty classes are overwritten
        rows.append(np.pad(sae_atoms, (0, max_n - len(sae_atoms)), mode="edge"))
    return torch.as_tensor(np.stack(rows), dtype=torch.long)


def save_ranking(
    path: Path,
    ranked: List[np.ndarray],
    atom_indices: np.ndarray,
    sim: np.ndarray,
    term_idx: Optional[np.ndarray],
    term_sim: Optional[np.ndarray],
    vocab_terms: Optional[List[str]],
    class_keys: List[str],
) -> None:
    rows = []
    for c, key in enumerate(class_keys):
        for rank, pos in enumerate(ranked[c]):
            row = {"class": key, "rank": rank + 1, "atom": int(atom_indices[pos]), "cos_class": float(sim[pos, c])}
            if term_idx is not None:  # The nearest term is only informative here, it plays no role in the ranking
                row["term"] = vocab_terms[term_idx[pos]] if vocab_terms is not None else int(term_idx[pos])
                row["atom_term_cosine"] = float(term_sim[pos])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


# ----------------------------------------------------------------------------------------------------------
# Per-atom normalization
# ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def sae_inputs(batch: Dict[str, Any], act_datamodule: OnlineActDataModule, device: torch.device) -> torch.Tensor:
    """Patch embeddings [P_valid, D] of the tokens with a land-cover label."""
    act_batch = act_datamodule.on_after_batch_transfer(batch, 0)
    x = extract_input({k: v.to(device) for k, v in act_batch.items()})  # [P, D]
    return x[batch["lc_valid"].reshape(-1).to(device)]


@torch.no_grad()
def compute_atom_stats(
    loader: DataLoader, act_datamodule: OnlineActDataModule, saes: Dict[str, plSAE], device: torch.device
) -> Dict[tuple, Dict[str, torch.Tensor]]:
    """Per-atom mean and std of the post and pre activations, over the evaluated tokens."""
    totals: Dict[tuple, List[Any]] = {}
    for batch in tqdm(loader, desc="Computing per-atom statistics"):
        x = sae_inputs(batch, act_datamodule, device)
        for dict_name, sae in saes.items():
            pre, z, _ = sae(x)
            for act, codes in (("post", z), ("pre", pre)):
                codes = codes.double()
                total = totals.setdefault((dict_name, act), [0, torch.zeros_like(codes[0]), torch.zeros_like(codes[0])])
                total[0] += codes.shape[0]
                total[1] += codes.sum(dim=0)
                total[2] += (codes ** 2).sum(dim=0)

    stats = {}
    for key, (count, sum_codes, sum_squares) in totals.items():
        mean = sum_codes / count
        std = (sum_squares / count - mean ** 2).clamp_min(0).sqrt()
        std = torch.where(std < 1e-8, torch.ones_like(std), std)  # Constant atoms (e.g. never active)
        # Below any active post z-score, which is at least -mean / std
        shift = (-mean / std).min() - 1.0
        stats[key] = {"mean": mean.float(), "std": std.float(), "shift": shift.float()}
    return stats


def normalize_codes(codes: torch.Tensor, stats: Dict[str, torch.Tensor], activation: str) -> torch.Tensor:
    """Z-score each atom. Post: inactive atoms stay at 0 and active ones are > 0, so "active" keeps its meaning.

    A plain z-score would turn a zero into -mean / std, which differs per atom and lets an inactive atom win the max.
    Subtracting one constant from all the active z-scores keeps their order within and across atoms.
    """
    zscores = (codes - stats["mean"]) / stats["std"]
    if activation == "pre":
        return zscores
    return torch.where(codes > 0, zscores - stats["shift"], torch.zeros_like(codes))


# ----------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------
def predict_classes(scores: np.ndarray, activation: str) -> np.ndarray:
    """Argmax over the class scores [N, n_classes]; -1 for post tokens with no active atom in any class."""
    pred = scores.argmax(axis=1)
    if activation == "post":
        pred = np.where(scores.max(axis=1) > 0, pred, -1)
    return pred


def active_precision(scores: np.ndarray, frac: np.ndarray, pos_threshold: float) -> Dict[str, float]:
    """Precision of the tokens where the top-n set is active, and how often it is active."""
    active = scores > 0
    n_active = int(active.sum())
    base_rate = float((frac >= pos_threshold).mean())
    rate = n_active / len(scores)
    precision = float((frac[active] >= pos_threshold).mean()) if n_active else float("nan")
    return {
        "base_rate": base_rate,
        "n_active": n_active,
        "activation_rate": rate,
        "equiv_percentile": 100.0 * (1.0 - rate),  # comparable to the zero-shot top_percentiles
        "prec@active": precision,  # active tokens whose class fraction >= pos_threshold
        "prec_any@active": float((frac[active] > 0).mean()) if n_active else float("nan"),
        "mean_frac@active": float(frac[active].mean()) if n_active else float("nan"),
        "lift@active": precision / base_rate if base_rate > 0 and n_active else float("nan"),
    }


def method_class_metrics(
    scores: np.ndarray, frac: np.ndarray, activation: str, has_atoms: bool, pos_threshold: float, sweep: List[float],
    percentiles: List[float],
) -> Dict[str, float]:
    """Metrics of one class for one method (empty dict for a class without atoms)."""
    if not has_atoms:
        return {}
    metrics = compute_class_metrics(scores, frac, pos_threshold, sweep, activation == "post")
    if activation == "post":
        metrics.update(active_precision(scores, frac, pos_threshold))
    else:  # Pre-activations are continuous, so a percentile always selects the requested share of tokens
        metrics.update(top_percentile_precision(scores, frac, pos_threshold, percentiles))
    return metrics


# ----------------------------------------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------------------------------------
def plot_macro_auroc_vs_topn(summary: pd.DataFrame, output_path: Path) -> None:
    sources = list(summary["source"].unique())
    class_texts = list(summary["class_text"].unique())
    top_n = sorted(summary["top_n"].unique())
    fig, axes = plt.subplots(
        len(sources), len(class_texts), figsize=(4.2 * len(class_texts), 3.0 * len(sources) + 0.8),
        sharex=True, sharey=True, squeeze=False, facecolor="white",
    )
    x = np.arange(len(top_n))  # Categorical axis: the levels are not evenly spaced
    for row, source in enumerate(sources):
        for col, class_text in enumerate(class_texts):
            ax = axes[row, col]
            panel = summary[(summary["source"] == source) & (summary["class_text"] == class_text)]
            for (dictionary, activation), color in _SERIES_COLORS.items():
                series = panel[(panel["dictionary"] == dictionary) & (panel["activation"] == activation)]
                values = series.set_index("top_n").reindex(top_n)["macro_auroc"].to_numpy()
                ax.plot(x, values, color=color, linewidth=2, marker="o", markersize=6,
                        markeredgecolor="white", label=f"{dictionary} {activation}")
            ax.axhline(0.5, color=_MUTED_COLOR, linewidth=1, linestyle="--")
            ax.set_title(f"{source} ranking, {class_text} class text", loc="left", fontsize=10, color=_TEXT_COLOR)
            ax.set_xticks(x, [f"top-{n}" for n in top_n], fontsize=9, color=_TEXT_COLOR)
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(colors=_MUTED_COLOR)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if col == 0:
                ax.set_ylabel("Macro AUROC", color=_TEXT_COLOR)
    axes[0, -1].text(len(top_n) - 1, 0.5, "chance", ha="right", va="bottom", fontsize=8, color=_MUTED_COLOR)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=4, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval_landcover_alignment")
def evaluate_landcover_alignment(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    activation_norm = cfg.get("activation_norm", "none")
    assert activation_norm in ("none", "zscore"), f"Unknown activation_norm {activation_norm}"
    suffix = "" if activation_norm == "none" else f"_{activation_norm}"  # Never overwrites the raw-activation results
    output_dir = Path(cfg.output_dir) / f"landcover_alignment_{cfg.lc_year}{suffix}"
    (output_dir / "confusion").mkdir(parents=True, exist_ok=True)

    lc_classes = OmegaConf.to_container(cfg.lc_classes, resolve=True)
    class_keys = list(lc_classes)
    class_names = [lc_classes[key]["name"] for key in class_keys]
    n_classes = len(class_keys)
    group_matrix = build_group_matrix(lc_classes)
    mapping_cfg = cfg.mapping
    top_n = sorted(int(n) for n in mapping_cfg.top_n)
    max_n = top_n[-1]

    ### Model ###
    log.info(f"Instantiating MS-CLIP Model at <{cfg.model_path}>")
    model = load_img_model(cfg, device, logger=log)
    assert cfg.MODEL.test_max_seq_len is None, "For evaluation, test_max_seq_len must be None."
    grid, patch_size = model.model.H_patch, model.model.patch_size
    assert model.model.H_patch == model.model.W_patch
    if grid * patch_size != cfg.MODEL.img_res:
        log.warning(f"img_res={cfg.MODEL.img_res} but the ViT only sees {grid * patch_size}px: "
                    "the token grid is not centered in the tile")

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

    ### SAEs and atom -> class rankings ###
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

    # rankings[(dictionary, source, class_text)] -> per-class atoms, their cosines and the padded [n_classes, max_n] tensor
    rankings: Dict[tuple, Dict[str, Any]] = {}
    for dict_name in DICTIONARIES:
        atom_indices = alive_indices[dict_name].numpy()
        for source in mapping_cfg.sources:
            for class_text in mapping_cfg.class_texts:
                sim, term_idx, term_sim = atom_class_similarity(
                    saes[dict_name], alive_indices[dict_name], text_emb[class_text], source
                )
                sim = sim.numpy()
                ranked = rank_atoms(sim, max_n)
                rankings[(dict_name, source, class_text)] = {
                    "cosines": [sim[atoms, c] for c, atoms in enumerate(ranked)],
                    "atoms": padded_ranking(ranked, atom_indices, max_n).to(device),
                    "has_atoms": torch.as_tensor([len(atoms) > 0 for atoms in ranked], device=device),
                }
                save_ranking(
                    output_dir / f"atom_class_ranking_{dict_name}_{class_text}_{source}.csv", ranked, atom_indices,
                    sim, term_idx, term_sim, vocab_terms, class_keys,
                )

    ### Per-atom statistics (one extra pass over the tokens) ###
    atom_stats = compute_atom_stats(loader, act_datamodule, saes, device) if activation_norm == "zscore" else None

    ### One pass over the tokens ###
    level_idx = torch.as_tensor([n - 1 for n in top_n], device=device)
    # scores[(dictionary, source, class_text, activation)] -> per-batch arrays [P, n_classes, n_levels]
    scores: Dict[tuple, List[np.ndarray]] = {}
    fracs: List[np.ndarray] = []
    lc_pixel_counts = np.zeros(N_LC_IDS, dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Scoring {cfg.lc_year} tokens"):
            valid = batch["lc_valid"].reshape(-1)
            frac = batch["lc_frac"].reshape(-1, n_classes)
            lc_pixel_counts += batch["lc_counts"].sum(dim=0).numpy()
            x = sae_inputs(batch, act_datamodule, device)
            fracs.append(frac[valid].numpy())

            for dict_name in DICTIONARIES:
                pre, z, _ = saes[dict_name](x)
                codes = {"post": z, "pre": pre}
                if atom_stats is not None:
                    codes = {act: normalize_codes(codes[act], atom_stats[(dict_name, act)], act) for act in ACTIVATIONS}
                for (ranking_dict, source, class_text), ranking in rankings.items():
                    if ranking_dict != dict_name:
                        continue
                    for act in ACTIVATIONS:
                        # Running max over the ranked atoms: position n - 1 is the max over the top-n set.
                        running_max = torch.cummax(codes[act][:, ranking["atoms"]], dim=2).values
                        running_max[:, ~ranking["has_atoms"]] = _EMPTY_CLASS_SCORE[act]
                        scores.setdefault((dict_name, source, class_text, act), []).append(
                            running_max[:, :, level_idx].float().cpu().numpy()
                        )

    frac_all = np.concatenate(fracs, axis=0)  # [N, n_classes]
    scores_all = {k: np.concatenate(v, axis=0) for k, v in scores.items()}  # [N, n_classes, n_levels]
    log.info(f"{frac_all.shape[0]} valid tokens")

    ungrouped = ungrouped_lc_ids(lc_pixel_counts, group_matrix)
    if ungrouped:
        log.warning(f"Land-cover ids present but not evaluated (pixels): {ungrouped}")

    ### Metrics ###
    pos_t, sweep = float(cfg.pos_threshold), [float(t) for t in cfg.threshold_sweep]
    percentiles = [float(q) for q in cfg.top_percentiles]
    methods = [
        {"method": f"{dict_name}_{source}_{class_text}_top{n}_{act}", "dictionary": dict_name, "source": source,
         "class_text": class_text, "top_n": n, "activation": act, "level": level}
        for (dict_name, source, class_text, act) in scores_all for level, n in enumerate(top_n)
    ]

    def method_scores(m: Dict[str, Any]) -> np.ndarray:
        return scores_all[(m["dictionary"], m["source"], m["class_text"], m["activation"])][:, :, m["level"]]

    def set_cosines(m: Dict[str, Any], c: int) -> np.ndarray:
        return rankings[(m["dictionary"], m["source"], m["class_text"])]["cosines"][c][: m["top_n"]]

    class_metrics = Parallel(n_jobs=cfg.metric_workers, prefer="threads")(
        delayed(method_class_metrics)(
            method_scores(m)[:, c], frac_all[:, c], m["activation"], len(set_cosines(m, c)) > 0, pos_t, sweep,
            percentiles,
        )
        for m in methods for c in range(n_classes)
    )

    rows, acc_rows = [], []
    for method_idx, m in enumerate(methods):
        identity = {k: m[k] for k in ("method", "dictionary", "source", "class_text", "top_n", "activation")}
        acc_summary, acc_per_class, confusion = multiclass_accuracy(
            predict_classes(method_scores(m), m["activation"]), frac_all, float(cfg.accuracy_min_fraction), class_keys
        )
        confusion.to_csv(output_dir / "confusion" / f"confusion_{m['method']}.csv")
        acc_rows.append({**identity, **acc_summary})
        acc_per_class = acc_per_class.set_index("class")
        for c, key in enumerate(class_keys):
            cosines = set_cosines(m, c)
            rows.append({
                "class": key, **identity,
                "n_atoms": len(cosines),
                "filled": len(cosines) == m["top_n"],
                "mean_cos_topn": float(cosines.mean()) if len(cosines) else float("nan"),
                "min_cos_topn": float(cosines.min()) if len(cosines) else float("nan"),
                **class_metrics[method_idx * n_classes + c],
                **acc_per_class.loc[key].to_dict(),
            })
    per_class = pd.DataFrame(rows)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    accuracy = pd.DataFrame(acc_rows)
    accuracy.to_csv(output_dir / "accuracy.csv", index=False)

    ### Summary ###
    summary: Dict[str, Dict[str, tuple]] = {
        "Setup": {
            "caveat": (f"year {cfg.lc_year} is in the '{cfg.source_split}' split used to train the SAE "
                       "(unsupervised, never saw LC labels)", ""),
            "n_tiles": (len(dataset.data_paths), ""),
            "n_valid_tokens": (int(frac_all.shape[0]), ""),
            "image_window": (f"center crop {cfg.MODEL.input_img_res}->{cfg.MODEL.img_res}, ViT sees "
                             f"{grid * patch_size}px", ""),
            "token_pixels": (f"{patch_size}x{patch_size} LC pixels (10 m)", ""),
            "positive_rule": (f"fraction >= {pos_t:g} vs fraction == 0 (mixed tokens dropped)", ""),
            "accuracy_rule": (f"argmax over class scores, tokens with dominant fraction >= "
                              f"{float(cfg.accuracy_min_fraction):g}; post tokens with no active atom have no "
                              "prediction", ""),
            "active_precision_rule": (f"post only: share of tokens with score > 0 whose fraction >= {pos_t:g}", ""),
            "top_percentile_rule": (f"pre only: share of tokens above the percentile of the class score whose "
                                    f"fraction >= {pos_t:g}", ""),
            "ranking": ("exclusive: an atom only ranks for its closest class", ""),
            "activation_norm": (activation_norm, "per-atom normalization applied before the max over a set"),
            "prompts": (f"raw = class name, template = '{cfg.prompt_template}'", ""),
            "ungrouped_lc_ids": (str(ungrouped) if ungrouped else "none", ""),
        },
    }
    summary_rows = []
    all_underfilled = set()
    for m in methods:
        df = per_class[per_class["method"] == m["method"]]
        acc = accuracy.set_index("method").loc[m["method"]]
        underfilled = df[(df["n_atoms"] < m["top_n"])]
        all_underfilled.add((m["dictionary"], m["source"], m["class_text"], m["top_n"], len(underfilled)))
        coverage = int((df["n_atoms"] > 0).sum())
        row = {
            **{k: m[k] for k in ("method", "dictionary", "source", "class_text", "top_n", "activation")},
            "coverage": coverage,
            "n_underfilled_classes": len(underfilled),
            "macro_auroc": nanmean(df["auroc"].tolist()),
            "macro_ap": nanmean(df["ap"].tolist()),
            "macro_prevalence": nanmean(df["prevalence"].tolist()),
            **{k: float(acc[k]) for k in ("n_eval_tokens", "accuracy", "balanced_accuracy", "majority_accuracy",
                                          "chance_balanced_accuracy", "abstain_rate", "n_active_eval",
                                          "accuracy_active", "balanced_accuracy_active",
                                          "majority_accuracy_active")},
        }
        section = {
            "coverage": (f"{coverage}/{n_classes}", "classes with at least one atom"),
            "underfilled": (", ".join(f"{r['class']} {r['n_atoms']}/{m['top_n']}" for _, r in underfilled.iterrows())
                            or "none", ""),
            "macro_auroc": (row["macro_auroc"], "mean over classes with atoms and positives"),
            "macro_ap": (row["macro_ap"], "mean over classes with atoms and positives"),
            "macro_prevalence": (row["macro_prevalence"], "AP of a random score"),
            "accuracy": (row["accuracy"], f"{int(row['n_eval_tokens'])} tokens, no prediction = wrong"),
            "balanced_accuracy": (row["balanced_accuracy"], "mean per-class recall, no prediction = wrong"),
            "majority_accuracy": (row["majority_accuracy"], "always predicting the most frequent class"),
            "chance_balanced_accuracy": (row["chance_balanced_accuracy"], ""),
            "abstain_rate": (row["abstain_rate"], "evaluated tokens with no prediction"),
            "accuracy_active": (row["accuracy_active"], f"{int(row['n_active_eval'])} tokens with a prediction"),
            "balanced_accuracy_active": (row["balanced_accuracy_active"], "mean per-class recall on those tokens"),
            "majority_accuracy_active": (row["majority_accuracy_active"], "most frequent class on those tokens"),
        }
        if m["activation"] == "post":
            present = df[df["base_rate"] > 0]
            row.update({
                "mean_activation_rate": nanmean(df["activation_rate"].tolist()),
                "macro_prec@active": nanmean(present["prec@active"].tolist()),
                "macro_lift@active": nanmean(present["lift@active"].tolist()),
            })
            section.update({
                "mean_activation_rate": (row["mean_activation_rate"], "share of all valid tokens, mean over classes"),
                "macro_prec@active": (row["macro_prec@active"], "mean over classes with positives"),
                "macro_lift@active": (row["macro_lift@active"], "precision / base rate"),
            })
        else:
            present = df[df["base_rate"] > 0]
            for q in percentiles:
                row[f"macro_prec@p{q:g}"] = nanmean(present[f"prec@p{q:g}"].tolist())
                row[f"macro_lift@p{q:g}"] = nanmean(present[f"lift@p{q:g}"].tolist())
                section[f"macro_prec@p{q:g}"] = (row[f"macro_prec@p{q:g}"], "mean over classes with positives")
                section[f"macro_lift@p{q:g}"] = (row[f"macro_lift@p{q:g}"], "precision / base rate")
        summary_rows.append(row)
        summary[m["method"]] = section
    summary_by_topn = pd.DataFrame(summary_rows)
    summary_by_topn.to_csv(output_dir / "summary_by_topn.csv", index=False)
    log_metrics(output_dir, "landcover_alignment", summary)

    underfilled_sets = sorted(entry for entry in all_underfilled if entry[-1] > 0)
    if underfilled_sets:
        log.warning("Top-n sets not filled (dictionary, source, class_text, n, classes affected): "
                    f"{underfilled_sets}. Details in per_class_metrics.csv (n_atoms, filled).")

    if cfg.get("plot", True):
        plot_macro_auroc_vs_topn(summary_by_topn, output_dir / "macro_auroc_vs_topn.png")
    log.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    evaluate_landcover_alignment()
