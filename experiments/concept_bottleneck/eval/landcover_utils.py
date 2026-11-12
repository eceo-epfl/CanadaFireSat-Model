"""Shared pieces of the land-cover evaluations (SAE dictionaries and MS-CLIP zero-shot baseline).

Token <-> pixel alignment: the eval transform center-crops input_img_res -> img_res, and the ViT stem
(Conv2d, kernel = stride = patch_size, no padding) only sees the top-left H_patch * patch_size pixels of
that crop (all of it when img_res = H_patch * patch_size, e.g. 256). Each token therefore covers a
patch_size x patch_size block of 10 m LC pixels; the class fraction inside that block defines the token label.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset

from experiments.concept_bottleneck.sae.dataset import OnlineActivationDataset
from src.constants import LAND_COVER_DICT_IDS
from src.data import get_data

N_LC_IDS = max(LAND_COVER_DICT_IDS) + 1
UNKNOWN_LC_ID = 0

_TEXT_COLOR = "#1f2328"
_MUTED_COLOR = "#6e7781"


# ----------------------------------------------------------------------------------------------------------
# Land-cover token labels
# ----------------------------------------------------------------------------------------------------------
# This simply map the lc class ids to something coarser (e.g. water, forest, urban). The mapping is defined in the config file.
def build_group_matrix(lc_classes: Dict[str, Dict[str, Any]]) -> np.ndarray:
    """[N_LC_IDS, n_classes] binary matrix mapping fine LC ids to the evaluated class groups."""
    group = np.zeros((N_LC_IDS, len(lc_classes)), dtype=np.float32)
    for class_idx, class_cfg in enumerate(lc_classes.values()):
        for lc_id in class_cfg["ids"]:
            if lc_id == UNKNOWN_LC_ID or lc_id >= N_LC_IDS:
                raise ValueError(f"Invalid land-cover id {lc_id} in lc_classes")
            group[lc_id, class_idx] = 1.0
    if (group.sum(axis=1) > 1).any():
        raise ValueError("A land-cover id belongs to several classes in lc_classes")
    return group


def lc_token_counts(lc: np.ndarray, offset: int, grid: int, patch_size: int) -> np.ndarray:
    """Per-token pixel counts of each LC id: [grid * grid, N_LC_IDS], tokens in row-major order."""
    side = grid * patch_size
    # Center Crop the LC map
    lc = lc[offset:offset + side, offset:offset + side].astype(np.int64)
    lc = np.where((lc < 0) | (lc >= N_LC_IDS), UNKNOWN_LC_ID, lc)

    # Reshape LC from 256,256 to 16, 16, 16, 16 and flatten to 256 (patches), 16*16 (pixels).
    blocks = lc.reshape(grid, patch_size, grid, patch_size).transpose(0, 2, 1, 3).reshape(grid * grid, -1)

    # Count the number of pixels of each LC id in each token (row).
    one_hot = np.zeros((blocks.shape[0], N_LC_IDS), dtype=np.int32)
    np.add.at(one_hot, (np.arange(blocks.shape[0])[:, None], blocks), 1)
    return one_hot


class LandCoverTokenDataset(OnlineActivationDataset):
    """OnlineActivationDataset with per-token land-cover fractions of the evaluated classes."""

    def __init__(
        self,
        input_dataset: Dataset,
        lc_dirs: List[str],
        group_matrix: np.ndarray,
        input_img_res: int,
        img_res: int,
        grid: int,
        patch_size: int,
    ):
        super().__init__(input_dataset)
        self.lc_dirs = lc_dirs
        self.group_matrix = group_matrix
        self.input_img_res = input_img_res
        self.offset = (input_img_res - img_res) // 2  # same as the eval center Crop
        self.grid = grid
        self.patch_size = patch_size
        assert grid * patch_size <= img_res, "Token grid exceeds the cropped image"

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        row = self.input_dataset.data_paths.loc[idx]
        lc = np.load(Path(self.lc_dirs[int(row["file_id"])]) / str(row["tile_id"]) / "lc.npy")
        assert lc.shape == (self.input_img_res, self.input_img_res), f"Unexpected LC shape {lc.shape}"

        counts = lc_token_counts(lc, self.offset, self.grid, self.patch_size)  # [P, N_LC_IDS]
        known = counts.sum(axis=1) - counts[:, UNKNOWN_LC_ID]
        frac = (counts @ self.group_matrix) / np.maximum(known, 1)[:, None]
        item["lc_frac"] = torch.from_numpy(frac.astype(np.float32))  # [P, n_classes]
        item["lc_valid"] = torch.from_numpy(known > 0)  # [P]
        item["lc_counts"] = torch.from_numpy(counts.sum(axis=0))  # [N_LC_IDS], for the coverage report
        return item


def load_year_dataset(cfg: DictConfig) -> Dataset:
    """Image dataset of `source_split`, restricted to the tiles of `lc_year` (optionally capped)."""
    dataset = get_data(cfg).test_dataloader(split=cfg.source_split).dataset  # Has to be train with the current land-cover
    dataset.data_paths = dataset.data_paths[dataset.data_paths["year"] == cfg.lc_year].reset_index(drop=True)
    if cfg.get("max_samples"):
        dataset.data_paths = dataset.data_paths.iloc[: cfg.max_samples].reset_index(drop=True)
    assert len(dataset.data_paths) > 0, f"No sample of year {cfg.lc_year} in split {cfg.source_split}"
    return dataset


def ungrouped_lc_ids(lc_pixel_counts: np.ndarray, group_matrix: np.ndarray) -> List[tuple]:
    """LC ids present in the data but in none of the evaluated classes, with their pixel counts."""
    return [
        (LAND_COVER_DICT_IDS[i], int(n)) for i, n in enumerate(lc_pixel_counts)
        if n > 0 and i != UNKNOWN_LC_ID and group_matrix[i].sum() == 0
    ]


# ----------------------------------------------------------------------------------------------------------
# Text embeddings
# ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def encode_texts(model, texts: List[str], device: torch.device) -> torch.Tensor:
    toks = model.model.tokenizer(texts).to(device)
    return F.normalize(model.model.msclip_model.inference_text(toks).float(), dim=-1)


def class_text_embeddings(model, class_names: List[str], prompt_template: str, device) -> Dict[str, torch.Tensor]:
    """Class embeddings from the raw class name and from the prompt template, both [n_classes, D]."""
    return {
        "raw": encode_texts(model, class_names, device),
        "template": encode_texts(model, [prompt_template.format(name) for name in class_names], device),
    }


# ----------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------
def binary_split(frac: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Positives: fraction >= tau (> 0 when tau == 0). Negatives: fraction == 0. Mixed tokens dropped."""
    pos = frac > 0 if tau <= 0 else frac >= tau
    neg = frac == 0
    keep = pos | neg
    return keep, pos[keep]


def auroc_ap(scores: np.ndarray, frac: np.ndarray, tau: float) -> tuple[float, float, int, int]:
    keep, y = binary_split(frac, tau)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), n_pos, n_neg
    s = scores[keep].astype(np.float64)
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s)), n_pos, n_neg


def compute_class_metrics(
    scores: np.ndarray, frac: np.ndarray, pos_threshold: float, threshold_sweep: List[float], is_post: bool
) -> Dict[str, float]:
    auroc, ap, n_pos, n_neg = auroc_ap(scores, frac, pos_threshold)
    keep, _ = binary_split(frac, pos_threshold)
    metrics = {
        "auroc": auroc,
        "ap": ap,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": n_pos / (n_pos + n_neg) if n_pos + n_neg else float("nan"),
        "spearman": float(spearmanr(scores, frac).statistic) if np.ptp(scores) > 0 and np.ptp(frac) > 0
        else float("nan"),
        "frac_tied_at_zero": float((scores[keep] == 0).mean()) if is_post and keep.any() else float("nan"),
    }
    for tau in threshold_sweep:
        metrics[f"auroc@{tau:g}"] = auroc_ap(scores, frac, tau)[0]
    return metrics


def nanmean(values: List[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def top_percentile_precision(
    scores: np.ndarray, frac: np.ndarray, pos_threshold: float, percentiles: List[float]
) -> Dict[str, float]:
    """Precision of the tokens whose score is above the q-th percentile of this class's scores."""
    base_rate = float((frac >= pos_threshold).mean())
    metrics = {"base_rate": base_rate}
    for q in percentiles:
        top = scores >= np.percentile(scores, q)
        precision = float((frac[top] >= pos_threshold).mean())
        metrics[f"n_top@p{q:g}"] = int(top.sum())
        metrics[f"prec@p{q:g}"] = precision  # top tokens whose class fraction >= pos_threshold
        metrics[f"prec_any@p{q:g}"] = float((frac[top] > 0).mean())  # top tokens containing the class at all
        metrics[f"mean_frac@p{q:g}"] = float(frac[top].mean())
        metrics[f"lift@p{q:g}"] = precision / base_rate if base_rate > 0 else float("nan")
    return metrics


def multiclass_accuracy(
    y_pred: np.ndarray, frac: np.ndarray, min_fraction: float, class_keys: List[str]
) -> tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Accuracy on tokens whose dominant class covers >= min_fraction of the token.

    y_pred holds one class index per token, or -1 for "no prediction" (e.g. no active atom). No prediction counts
    as an error in the all-token metrics, and is excluded from the *_active metrics.
    """
    n_classes = len(class_keys)
    keep = frac.max(axis=1) >= min_fraction
    y_true = frac[keep].argmax(axis=1)
    pred = np.where(y_pred[keep] < 0, n_classes, y_pred[keep])  # last column: no prediction
    # Rows: true class, columns: predicted class + no prediction
    cm = confusion_matrix(y_true, pred, labels=np.arange(n_classes + 1))[:n_classes]
    correct = np.diag(cm[:, :n_classes])
    support, predicted = cm.sum(axis=1), cm[:, :n_classes].sum(axis=0)
    support_active = cm[:, :n_classes].sum(axis=1)
    recall = np.where(support > 0, correct / np.maximum(support, 1), np.nan)
    recall_active = np.where(support_active > 0, correct / np.maximum(support_active, 1), np.nan)
    precision = np.where(predicted > 0, correct / np.maximum(predicted, 1), np.nan)
    n_eval, n_active_eval = int(support.sum()), int(support_active.sum())
    summary = {
        "n_eval_tokens": n_eval,
        "accuracy": float(correct.sum() / n_eval) if n_eval else float("nan"),
        "balanced_accuracy": nanmean(recall.tolist()),  # mean recall over classes present / same as macro-accuracy
        "majority_accuracy": float(support.max() / n_eval) if n_eval else float("nan"),
        "chance_balanced_accuracy": 1.0 / max(int((support > 0).sum()), 1),
        "abstain_rate": 1.0 - n_active_eval / n_eval if n_eval else float("nan"),
        "n_active_eval": n_active_eval,
        "accuracy_active": float(correct.sum() / n_active_eval) if n_active_eval else float("nan"),
        "balanced_accuracy_active": nanmean(recall_active.tolist()),
        "majority_accuracy_active": float(support_active.max() / n_active_eval) if n_active_eval else float("nan"),
    }
    per_class = pd.DataFrame({"class": class_keys, "support": support, "n_predicted": predicted,
                              "recall": recall, "precision_cls": precision})
    confusion = pd.DataFrame(cm, index=[f"true_{k}" for k in class_keys],
                             columns=[f"pred_{k}" for k in class_keys] + ["pred_none"])
    return summary, per_class, confusion


# ----------------------------------------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------------------------------------
def plot_auroc_per_class(
    per_class: pd.DataFrame, class_names: List[str], output_path: Path, method_colors: Dict[str, str],
    title: str = "Land-cover AUROC per class",
) -> None:
    methods = [m for m in method_colors if m in per_class["method"].unique()]
    n_methods = len(methods)
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(1.3 * len(class_names) + 2, 4.2), facecolor="white")
    x = np.arange(len(class_names))
    for method_idx, method in enumerate(methods):
        values = (
            per_class[per_class["method"] == method].set_index("class").reindex(class_names)["auroc"].to_numpy()
        )
        # Bars grow from the 0.5 chance line: above = better than chance, below = worse.
        ax.bar(
            x + (method_idx - (n_methods - 1) / 2) * width, np.nan_to_num(values - 0.5, nan=0.0), bottom=0.5,
            width=width * 0.9, color=method_colors[method], label=method, edgecolor="white", linewidth=1,
        )
    ax.axhline(0.5, color=_MUTED_COLOR, linewidth=1, linestyle="--")
    ax.text(len(class_names) - 0.5, 0.505, "chance", ha="right", va="bottom", fontsize=8, color=_MUTED_COLOR)
    ax.set_xticks(x, class_names, fontsize=9, color=_TEXT_COLOR)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC", color=_TEXT_COLOR)
    ax.set_title(title, loc="left", fontsize=11, color=_TEXT_COLOR)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=_MUTED_COLOR)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
