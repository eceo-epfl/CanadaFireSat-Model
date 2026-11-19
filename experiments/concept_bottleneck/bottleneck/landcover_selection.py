"""Water grounding of the probe's alive SAE features against the 2020 WorldCover maps.

One pass over the `lc_year` tiles stores, for every token and every active alive feature, the post-TopK activation
(the codes the probe uses) and the water fraction of the token. Post codes are sparse (k non-zeros per token), so
this fits in memory and lets every score below be computed from the same entries:

  - corr:      Pearson correlation between the activation (zeros included) and the water fraction.
  - mass:      activation-weighted mean water fraction, sum(z * frac) / sum(z): the share of the feature's
               contribution to the fire logit that lands on water.
  - precision: share of water tokens (frac >= pos_threshold) among the feature's top (100 - percentile)% tokens.
               A feature active on fewer tokens than that uses all its active tokens.
  - recall:    share of the water tokens on which the feature is active (reported, not used to select).

Token <-> pixel alignment: see `experiments/concept_bottleneck/eval/landcover_utils.py`.
"""
import copy
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiments.concept_bottleneck.bottleneck.probe import SAEProbe
from experiments.concept_bottleneck.eval.landcover_utils import LandCoverTokenDataset, build_group_matrix
from experiments.concept_bottleneck.sae.act_datamodule import OnlineActDataModule
from experiments.concept_bottleneck.sae.utils import get_pylogger
from overcomplete.sae.train import extract_input

log = get_pylogger(__name__)

LC_SCORES = ("corr", "mass", "precision")


def year_subset(dataset: Dataset, lc_year: int, max_samples=None) -> Dataset:
    """Shallow copy of the dataset restricted to the tiles of lc_year (the original is left untouched)."""
    subset = copy.copy(dataset)
    subset.data_paths = dataset.data_paths[dataset.data_paths["year"] == lc_year].reset_index(drop=True)
    if max_samples:
        subset.data_paths = subset.data_paths.iloc[:max_samples].reset_index(drop=True)
    assert len(subset.data_paths) > 0, f"No tile of year {lc_year}"
    return subset


@torch.no_grad()
def collect_water_entries(
    probe: SAEProbe,
    act_datamodule: OnlineActDataModule,
    dataset: Dataset,
    lc_cfg: DictConfig,
    model,
    input_img_res: int,
    img_res: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Sparse (feature, activation, water fraction) entries of the active alive features, plus every token's fraction."""
    grid, patch_size = model.model.H_patch, model.model.patch_size
    group_matrix = build_group_matrix({"water": {"ids": list(lc_cfg.water_ids), "name": "water"}})
    lc_dataset = LandCoverTokenDataset(
        dataset, list(lc_cfg.lc_dirs), group_matrix, input_img_res, img_res, grid, patch_size
    )
    loader = DataLoader(lc_dataset, batch_size=1, shuffle=False, num_workers=lc_cfg.num_workers)

    feats, vals, entry_fracs, token_fracs = [], [], [], []
    for batch in tqdm(loader, desc=f"Water grounding on {lc_cfg.lc_year} tiles"):
        valid = batch["lc_valid"].reshape(-1)
        frac = batch["lc_frac"].reshape(-1)[valid].to(device)  # [P_valid]
        x = extract_input({k: v.to(device) for k, v in act_datamodule.on_after_batch_transfer(batch, 0).items()})
        z = probe.encode(x[valid.to(device)])  # [P_valid, n_alive], post-TopK as in the probe
        token, feat = torch.nonzero(z > 0, as_tuple=True)
        feats.append(feat.int().cpu())
        vals.append(z[token, feat].float().cpu())
        entry_fracs.append(frac[token].float().cpu())
        token_fracs.append(frac.float().cpu())
    return {
        "feat": torch.cat(feats).numpy(),
        "val": torch.cat(vals).numpy(),
        "entry_frac": torch.cat(entry_fracs).numpy(),
        "token_frac": torch.cat(token_fracs).numpy(),
    }


def water_feature_stats(
    entries: Dict[str, np.ndarray], n_alive: int, percentile: float, pos_threshold: float
) -> pd.DataFrame:
    """Per alive feature (row = alive position): activation counts and the water scores of the module docstring."""
    feat, val = entries["feat"].astype(np.int64), entries["val"].astype(np.float64)
    entry_frac, token_frac = entries["entry_frac"].astype(np.float64), entries["token_frac"].astype(np.float64)
    n_tokens = len(token_frac)
    entry_water = entry_frac >= pos_threshold
    n_water = int((token_frac >= pos_threshold).sum())

    n_active = np.bincount(feat, minlength=n_alive)
    sum_z = np.bincount(feat, weights=val, minlength=n_alive)
    sum_z2 = np.bincount(feat, weights=val ** 2, minlength=n_alive)
    sum_zf = np.bincount(feat, weights=val * entry_frac, minlength=n_alive)

    # Pearson over all tokens: inactive tokens contribute z = 0, so the sparse sums are enough
    mean_z, mean_f = sum_z / n_tokens, token_frac.mean()
    var_z, var_f = sum_z2 / n_tokens - mean_z ** 2, token_frac.var()
    cov = sum_zf / n_tokens - mean_z * mean_f
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(var_z > 0, cov / np.sqrt(var_z * var_f), np.nan)
        mass = np.where(sum_z > 0, sum_zf / sum_z, np.nan)

    # Top tokens of each feature: sort by (feature, activation desc) and keep the first n_top of every group
    n_top = int(np.ceil(round(n_tokens * (100.0 - percentile) / 100.0, 6)))  # round: 1 - 0.99 != 0.01 in floats
    order = np.lexsort((-val, feat))
    group_start = np.cumsum(n_active) - n_active
    rank = np.arange(len(order)) - group_start[feat[order]]
    top = order[rank < n_top]
    n_top_used = np.bincount(feat[top], minlength=n_alive)
    hits = np.bincount(feat[top], weights=entry_water[top], minlength=n_alive)
    base_rate = n_water / n_tokens
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(n_top_used > 0, hits / n_top_used, np.nan)
        recall = np.bincount(feat, weights=entry_water, minlength=n_alive) / max(n_water, 1)

    return pd.DataFrame({
        "lc_n_active": n_active,
        "lc_activation_rate": n_active / n_tokens,
        "lc_corr": corr,
        "lc_mass": mass,
        "lc_precision": precision,
        "lc_lift": precision / base_rate if base_rate > 0 else np.nan,
        "lc_recall": recall,
        "lc_n_top": n_top_used,
    })


def water_scores(
    probe: SAEProbe,
    act_datamodule: OnlineActDataModule,
    train_dataset: Dataset,
    lc_cfg: DictConfig,
    model,
    input_img_res: int,
    img_res: int,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    """Water stats of every alive feature, from the cached entries when available."""
    max_tag = f"_max{lc_cfg.max_samples}" if lc_cfg.get("max_samples") else ""
    cache_path = output_dir / f"water_lc_entries_{lc_cfg.lc_year}{max_tag}.npz"
    if cache_path.exists() and not lc_cfg.get("recompute", False):
        log.info(f"Loading cached water entries from {cache_path}")
        entries = dict(np.load(cache_path))
    else:
        dataset = year_subset(train_dataset, lc_cfg.lc_year, lc_cfg.get("max_samples"))
        log.info(f"Collecting water entries on {len(dataset.data_paths)} tiles of {lc_cfg.lc_year}")
        entries = collect_water_entries(probe, act_datamodule, dataset, lc_cfg, model, input_img_res, img_res, device)
        np.savez(cache_path, **entries)
        log.info(f"Cached water entries to {cache_path}")

    stats = water_feature_stats(entries, probe.n_alive, float(lc_cfg.percentile), float(lc_cfg.pos_threshold))
    log.info(
        f"{len(entries['token_frac'])} valid tokens, water base rate "
        f"{(entries['token_frac'] >= lc_cfg.pos_threshold).mean():.4f} (frac >= {lc_cfg.pos_threshold:g})"
    )
    return stats


def selection_score(stats: pd.DataFrame, score: str, min_active_tokens: int) -> torch.Tensor:
    """[n_alive] score to rank by; -inf for undefined scores and features active on too few tokens."""
    if score not in LC_SCORES:
        raise ValueError(f"Unknown edit.landcover.score {score}: choose from {LC_SCORES}")
    values = stats[f"lc_{score}"].to_numpy(dtype=np.float64)
    values = np.where(np.isfinite(values) & (stats["lc_n_active"].to_numpy() >= min_active_tokens), values, -np.inf)
    return torch.as_tensor(values, dtype=torch.float32)
