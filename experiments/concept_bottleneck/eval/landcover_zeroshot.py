"""MS-CLIP zero-shot land-cover baseline.

Each token is scored by the cosine between its patch embedding and the text embedding of each land-cover
class (raw class name, and the class name in a prompt template). Patch embeddings are taken with and without
the temporal aggregation:
  - temporal:   temporal-mixer output at the last valid frame (what the SAE sees)
  - frame_last: per-frame MS-CLIP patch embedding of the last valid frame (no mixer)
  - frame_mean: per-frame embeddings averaged over the valid frames (no learned aggregation)

Reported per (variant, prompt) method:
  - per class: AUROC / AP of the cosine against "class present" (fraction >= pos_threshold vs fraction == 0)
  - per class: precision of the top-percentile tokens (cosine >= q-th percentile over all tokens of that class)
  - across classes: zero-shot accuracy / balanced accuracy, argmax over class cosines, on tokens whose
    dominant class covers >= accuracy_min_fraction of the token

Ground truth: 2020 land-cover maps (LC / NEG_LC). See `landcover_utils.py` for the token <-> pixel alignment.
"""
from pathlib import Path
from typing import Dict, List

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
    N_LC_IDS, LandCoverTokenDataset, build_group_matrix, class_text_embeddings, compute_class_metrics,
    load_year_dataset, multiclass_accuracy, nanmean, plot_auroc_per_class, top_percentile_precision,
    ungrouped_lc_ids,
)
from experiments.concept_bottleneck.eval.utils import load_img_model, log_metrics
from experiments.concept_bottleneck.sae.utils import get_pylogger
from src.constants import CONFIG_PATH

log = get_pylogger(__name__)

VARIANTS = ("temporal", "frame_last", "frame_mean")
PROMPTS = ("raw", "template")

# Reference categorical palette (fixed order, one slot per method).
_METHOD_COLORS = {
    "temporal_raw": "#2a78d6",
    "temporal_template": "#eb6834",
    "frame_last_raw": "#1baf7a",
    "frame_last_template": "#eda100",
    "frame_mean_raw": "#e87ba4",
    "frame_mean_template": "#008300",
}


# ----------------------------------------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def patch_embeddings(model, batch: Dict[str, torch.Tensor], variants: List[str], device) -> Dict[str, torch.Tensor]:
    """[P, D] patch embeddings of one tile for each requested variant."""
    inputs = batch["inputs"].to(device)  # [1, T, C, H, W]
    doy = batch["doy"].to(device)
    seq_len = batch["seq_lengths"].reshape(-1).to(device)  # [1]
    t_last = int(seq_len[0]) - 1
    out = {}
    if "temporal" in variants:
        out["temporal"] = model.model.encode_patches(inputs, use_temp=True, doy=doy, seq_len=seq_len)[0]
    if "frame_last" in variants or "frame_mean" in variants:
        frames = model.model.encode_patches(inputs, use_temp=False)[0]  # [T, P, D]
        if "frame_last" in variants:
            out["frame_last"] = frames[t_last]
        if "frame_mean" in variants:
            out["frame_mean"] = frames[: t_last + 1].mean(dim=0)
    return out


# ----------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval_landcover_zeroshot")
def evaluate_landcover_zeroshot(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir) / f"landcover_zeroshot_{cfg.lc_year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    lc_classes = OmegaConf.to_container(cfg.lc_classes, resolve=True)
    class_keys = list(lc_classes)
    class_names = [lc_classes[key]["name"] for key in class_keys]
    group_matrix = build_group_matrix(lc_classes)
    variants = [v for v in VARIANTS if v in cfg.variants]
    assert variants, f"variants must be a subset of {VARIANTS}"
    methods = [f"{v}_{p}" for v in variants for p in PROMPTS]

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
    text_emb = class_text_embeddings(model, class_names, cfg.prompt_template, device)

    ### One pass over the tokens ###
    scores: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    fracs: List[np.ndarray] = []
    lc_pixel_counts = np.zeros(N_LC_IDS, dtype=np.int64)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Scoring {cfg.lc_year} tokens"):
            valid = batch["lc_valid"].reshape(-1)
            fracs.append(batch["lc_frac"].reshape(-1, len(class_keys))[valid].numpy())
            lc_pixel_counts += batch["lc_counts"].sum(dim=0).numpy()
            valid_dev = valid.to(device)
            for variant, emb in patch_embeddings(model, batch, variants, device).items():
                emb = F.normalize(emb[valid_dev].float(), dim=-1)
                for prompt in PROMPTS:
                    scores[f"{variant}_{prompt}"].append((emb @ text_emb[prompt].T).cpu().numpy())

    frac_all = np.concatenate(fracs, axis=0)  # [N, n_classes]
    scores_all = {m: np.concatenate(v, axis=0) for m, v in scores.items()}  # [N, n_classes] cosines
    log.info(f"{frac_all.shape[0]} valid tokens")
    ungrouped = ungrouped_lc_ids(lc_pixel_counts, group_matrix)
    if ungrouped:
        log.warning(f"Land-cover ids present but not evaluated (pixels): {ungrouped}")

    ### Metrics ###
    pos_t, sweep = float(cfg.pos_threshold), [float(t) for t in cfg.threshold_sweep]
    percentiles = [float(q) for q in cfg.top_percentiles]
    rows, acc_rows = [], []
    for method in methods:
        acc_summary, acc_per_class, confusion = multiclass_accuracy(
            scores_all[method].argmax(axis=1), frac_all, float(cfg.accuracy_min_fraction), class_keys
        )
        confusion.to_csv(output_dir / f"confusion_{method}.csv")
        acc_rows.append({"method": method, **acc_summary})
        acc_per_class = acc_per_class.set_index("class")
        for c, key in enumerate(class_keys):
            rows.append({
                "class": key, "method": method,
                **compute_class_metrics(scores_all[method][:, c], frac_all[:, c], pos_t, sweep, False),
                **top_percentile_precision(scores_all[method][:, c], frac_all[:, c], pos_t, percentiles),
                **acc_per_class.loc[key].to_dict(),
            })
    per_class = pd.DataFrame(rows)
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)
    accuracy = pd.DataFrame(acc_rows)
    accuracy.to_csv(output_dir / "accuracy.csv", index=False)

    ### Summary ###
    summary: Dict[str, Dict[str, tuple]] = {
        "Setup": {
            "caveat": (f"year {cfg.lc_year} is in the '{cfg.source_split}' split", ""),
            "n_tiles": (len(dataset.data_paths), ""),
            "n_valid_tokens": (int(frac_all.shape[0]), ""),
            "image_window": (f"center crop {cfg.MODEL.input_img_res}->{cfg.MODEL.img_res}, ViT sees "
                             f"{grid * patch_size}px", ""),
            "token_pixels": (f"{patch_size}x{patch_size} LC pixels (10 m)", ""),
            "positive_rule": (f"fraction >= {pos_t:g} vs fraction == 0 (mixed tokens dropped)", ""),
            "top_percentile_rule": (f"precision = share of top tokens with fraction >= {pos_t:g}", ""),
            "accuracy_rule": (f"argmax over class cosines, tokens with dominant fraction >= "
                              f"{float(cfg.accuracy_min_fraction):g}", ""),
            "prompts": (f"raw = class name, template = '{cfg.prompt_template}'", ""),
            "ungrouped_lc_ids": (str(ungrouped) if ungrouped else "none", ""),
        },
    }
    for method in methods:
        df = per_class[per_class["method"] == method]
        acc = accuracy.set_index("method").loc[method]
        section = {
            "macro_auroc": (nanmean(df["auroc"].tolist()), "mean over classes with positives"),
            "macro_ap": (nanmean(df["ap"].tolist()), "mean over classes with positives"),
            "macro_prevalence": (nanmean(df["prevalence"].tolist()), "AP of a random score"),
            "accuracy": (float(acc["accuracy"]), f"{int(acc['n_eval_tokens'])} tokens"),
            "balanced_accuracy": (float(acc["balanced_accuracy"]), "mean per-class recall"),
            "majority_accuracy": (float(acc["majority_accuracy"]), "always predicting the most frequent class"),
            "chance_balanced_accuracy": (float(acc["chance_balanced_accuracy"]), ""),
        }
        present = df[df["base_rate"] > 0]
        for q in percentiles:
            section[f"macro_prec@p{q:g}"] = (nanmean(present[f"prec@p{q:g}"].tolist()),
                                             "mean over classes with positives")
            section[f"macro_lift@p{q:g}"] = (nanmean(present[f"lift@p{q:g}"].tolist()), "precision / base rate")
        summary[method] = section
    log_metrics(output_dir, "landcover_zeroshot", summary)

    if cfg.get("plot", True):
        plot_auroc_per_class(per_class, class_keys, output_dir / "auroc_per_class.png", _METHOD_COLORS,
                             title="MS-CLIP zero-shot land-cover AUROC per class")
    log.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    evaluate_landcover_zeroshot()
