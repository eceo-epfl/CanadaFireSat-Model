"""Top S2 tiles of vocabulary-aligned SAE features, ranked by the number of patches above a pre-code percentile.

For ONE SAE and ONE column of a vocabulary CSV (``cfg.vocab_column``, e.g. ``term`` or ``class_name``; same CSV
format as ``feature_text_alignment_compare.py``), the unique values of the column are encoded with the SAE text
encoder and, for each vocabulary entry, its top-``cfg.top_n_features`` most similar alive features are selected
(inclusive: a feature can be selected by several entries, which is flagged on the figures). Run the script once per
column, e.g. ``vocab_column=term`` then ``vocab_column=class_name``.

Each selected feature is then evaluated on the tiles of ``cfg.eval_split``:

1. Threshold: the k-th largest pre-code (encoder output before ReLU + TopK) of the feature over the N patches of
   the split, with k = ceil((1 - ``cfg.percentile`` / 100) * N), i.e. the top 0.1% for the default 99.9. This is
   the ``cfg.percentile`` percentile of ``top_percentile_precision`` in ``landcover_alignment.py`` up to the
   interpolation of ``np.percentile`` (at most one neighbouring value apart).
2. Tile score: number of patches whose pre-code is >= the threshold (ties broken by the tile's max pre-code).
3. Plot: the top-``cfg.top_k_tiles`` tiles, first / middle / last frame in RGB, with the mask of the patches above
   the threshold, coloured by the land-cover class of the vocabulary entry.

Outputs (``cfg.output_dir/feature_mask_alignment_p<percentile>/<split>/<column>``):
  - ``<entry>/top<r>_feature_<idx>.png`` (and ``.pdf`` if ``cfg.save_pdf``)
  - ``feature_mask_alignment.csv``: one row per (entry, rank) with threshold, tiles and patch counts.
"""
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib.patches import Patch
from omegaconf import DictConfig, OmegaConf
from overcomplete.sae.train import extract_input
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from experiments.concept_bottleneck.eval.feature_image_alignment import _display_rgb
from experiments.concept_bottleneck.eval.feature_text_alignment_compare import encode_vocabularies
from experiments.concept_bottleneck.eval.plot_utils import PAPER_RC, class_color_map
from experiments.concept_bottleneck.eval.utils import load_img_model, load_trained_sae
from experiments.concept_bottleneck.sae.act_datamodule import OnlineActDataModule
from experiments.concept_bottleneck.sae.dataset import OnlineActivationDataset
from experiments.concept_bottleneck.sae.metrics import _cosine_similarity_matrix
from experiments.concept_bottleneck.sae.utils import get_pylogger
from src.constants import CONFIG_PATH
from src.data import get_data


log = get_pylogger(__name__)

_TEXT_COLOR = "#222222"
_MUTED_COLOR = "#666666"
_PANEL_SIZE = 1.9      # inches per image panel
_HEADER_HEIGHT = 1.8   # inches reserved above the grid for titles and the shared-feature note
_FOOTER_HEIGHT = 0.5   # inches reserved below the grid for the legend
_SAMPLE_GAP = 0.22     # spacer between tiles, relative to panel width


# ----------------------------------------------------------------------------------------------------------
# Feature selection
# ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def select_features(dictionary: torch.Tensor, alive_indices: np.ndarray, vocabs: Dict[str, dict], n: int) -> pd.DataFrame:
    """One row per (column, entry, rank) with the selected SAE feature and its cosine to the entry text."""
    rows = []
    for column, vocab in vocabs.items():
        cos = _cosine_similarity_matrix(vocab["emb"], dictionary)  # [n_entries, n_alive]
        sims, idx = torch.topk(cos, k=min(n, cos.shape[1]), dim=1)
        for e, (text, cls) in enumerate(zip(vocab["texts"], vocab["classes"])):
            for r in range(sims.shape[1]):
                rows.append({
                    "column": column, "entry": text, "vocab_class": cls, "rank": r + 1,
                    "feature": int(alive_indices[int(idx[e, r])]), "cosine": float(sims[e, r]),
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Passes over the tiles
# ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def iter_pre_codes(loader: DataLoader, act_datamodule: OnlineActDataModule, sae, features: torch.Tensor,
                   device: torch.device, desc: str, max_tiles: Optional[int] = None):
    """Yield (tile_id, pre-codes [P, F] of the selected features)."""
    total = len(loader) if max_tiles is None else min(max_tiles, len(loader))
    for tile_id, batch in enumerate(tqdm(loader, desc=desc, total=total)):
        if tile_id >= total:
            break
        act_batch = act_datamodule.on_after_batch_transfer(batch, 0)
        x = extract_input({k: v.to(device) for k, v in act_batch.items()})  # [P, D]
        pre = sae(x)[0]  # z_pre: encoder pre-codes before ReLU + TopK
        yield tile_id, pre[:, features].float()


def percentile_thresholds(loader, act_datamodule, sae, features, device, q: float, max_tiles=None) -> torch.Tensor:
    """Per feature, the k-th largest pre-code over all N patches, with k = ceil((1 - q / 100) * N).

    The patches >= this threshold are the top (100 - q)% of the feature (ties aside). Only the running top-k
    values are kept. N = n_tiles * patches per tile, since every tile has the same token grid.
    """
    n_tiles = len(loader) if max_tiles is None else min(max_tiles, len(loader))
    top, k = None, None
    for _, pre in iter_pre_codes(loader, act_datamodule, sae, features, device, "Pass 1/2: pre-code percentiles",
                                 max_tiles):
        if top is None:
            k = max(1, math.ceil((1 - q / 100) * n_tiles * pre.shape[0]))
            log.info(f"{n_tiles} tiles x {pre.shape[0]} patches: threshold = {k}-th largest pre-code per feature")
            top = pre[:0]
        merged = torch.cat([top, pre], dim=0)
        top = torch.topk(merged, k=min(k, merged.shape[0]), dim=0).values  # [<=k, F], descending
    return top[-1]


def top_tiles(loader, act_datamodule, sae, features, thresholds, k: int, device, max_tiles=None):
    """Per feature, the k tiles with the most patches >= threshold, with their masks.

    Returns counts [F, k], tile ids [F, k] (-1 if fewer than k tiles have a patch above the threshold),
    max pre-codes [F, k] and masks [F, k, P] (bool).
    """
    n_feat = len(features)
    key = torch.full((n_feat, k), -float("inf"), dtype=torch.float64, device=device)
    ids = torch.full((n_feat, k), -1, dtype=torch.long, device=device)
    counts = torch.zeros((n_feat, k), dtype=torch.long, device=device)
    maxes = torch.full((n_feat, k), float("nan"), device=device)
    masks = None
    for tile_id, pre in iter_pre_codes(loader, act_datamodule, sae, features, device,
                                       "Pass 2/2: patches above threshold", max_tiles):
        above = pre >= thresholds  # [P, F]
        if masks is None:
            masks = torch.zeros((n_feat, k, pre.shape[0]), dtype=torch.bool, device=device)
        count = above.sum(dim=0)
        tile_max = pre.max(dim=0).values
        # Count first, then the tile's max pre-code: sigmoid keeps the tie-breaker inside [0, 1).
        tile_key = torch.where(count > 0, count.double() + 0.99 * torch.sigmoid(tile_max.double()),
                               torch.full_like(count, -float("inf"), dtype=torch.float64))

        cand_key = torch.cat([key, tile_key[:, None]], dim=1)
        key, order = cand_key.topk(k, dim=1)
        ids = torch.cat([ids, torch.full((n_feat, 1), tile_id, device=device)], dim=1).gather(1, order)
        counts = torch.cat([counts, count[:, None]], dim=1).gather(1, order)
        maxes = torch.cat([maxes, tile_max[:, None]], dim=1).gather(1, order)
        masks = torch.cat([masks, above.T[:, None, :]], dim=1)
        masks = masks.gather(1, order[:, :, None].expand(-1, -1, masks.shape[-1]))

    ids = torch.where(torch.isfinite(key), ids, torch.full_like(ids, -1))
    return counts.cpu().numpy(), ids.cpu().numpy(), maxes.cpu().numpy(), masks.cpu().numpy()


def load_tile_frames(dataset, tile_ids: List[int], num_workers: int, mean, std) -> Dict[int, dict]:
    """RGB of the first / middle / last frame of the selected tiles only (no model forward)."""
    loader = DataLoader(Subset(OnlineActivationDataset(dataset), tile_ids), batch_size=1, shuffle=False,
                        num_workers=num_workers)
    frames = {}
    for tile_id, batch in zip(tile_ids, tqdm(loader, desc="Loading selected tiles", total=len(tile_ids))):
        inputs = batch["inputs"]
        if inputs.ndim == 5:
            inputs = inputs[0]
        if "seq_lengths" in batch:  # ignore temporal padding
            inputs = inputs[: int(batch["seq_lengths"].reshape(-1)[0])]
        t_idx = [0, inputs.shape[0] // 2, inputs.shape[0] - 1]
        frames[tile_id] = {"t": t_idx, "rgb": [_display_rgb(inputs[t], mean, std) for t in t_idx]}
    return frames


# ----------------------------------------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------------------------------------
def _clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_feature_masks(title: str, subtitle: str, tiles: List[dict], color: str, threshold: float, q: float,
                       output_stem: Path, save_pdf: bool = False, note: Optional[str] = None) -> None:
    """Two rows per tile group: RGB frames, and the same frames with the mask of the patches above the threshold.

    ``note``: optional line under the subtitle, drawn as a highlighted badge (e.g. the other entries sharing
    the feature).

    ``tiles``: dicts with "tile_id", "count", "n_patches", "mask" [P] and "frames" ({"t", "rgb"}).
    """
    n_tiles = len(tiles)
    width_ratios = []
    for i in range(n_tiles):
        width_ratios += [1, 1, 1] + ([_SAMPLE_GAP] if i < n_tiles - 1 else [])

    fig_w = _PANEL_SIZE * sum(width_ratios) + 0.6
    # Wrap the note to the figure width (~0.09 in per character at 11.5 pt) and grow the header to fit it.
    note_lines = textwrap.wrap(note, width=max(40, int((fig_w - 0.9) / 0.09))) if note else []
    extra = 0.21 * max(len(note_lines) - 1, 0)
    header_h = _HEADER_HEIGHT + extra

    with plt.rc_context(PAPER_RC):
        fig_h = _PANEL_SIZE * 2 + header_h + _FOOTER_HEIGHT
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        grid = fig.add_gridspec(
            2, len(width_ratios), width_ratios=width_ratios,
            left=0.45 / fig_w, right=1 - 0.15 / fig_w,
            top=1 - header_h / fig_h, bottom=_FOOTER_HEIGHT / fig_h, wspace=0.03, hspace=0.04,
        )
        mask_cmap = plt.matplotlib.colors.ListedColormap([color])

        for i, tile in enumerate(tiles):
            first_col = i * 4
            side = int(round(len(tile["mask"]) ** 0.5))
            mask = np.ma.masked_where(~tile["mask"].reshape(side, side), np.ones((side, side)))
            group_axes = []
            for c, (t, rgb) in enumerate(zip(tile["frames"]["t"], tile["frames"]["rgb"])):
                h, w = rgb.shape[:2]
                top = fig.add_subplot(grid[0, first_col + c])
                top.imshow(rgb)
                top.set_title(f"t = {t}", fontsize=10, color=_MUTED_COLOR, pad=3)
                bottom = fig.add_subplot(grid[1, first_col + c])
                bottom.imshow(rgb)
                # Stretch the patch grid over the image; patches below the threshold stay transparent.
                bottom.imshow(mask, cmap=mask_cmap, vmin=0, vmax=1, alpha=0.6, interpolation="nearest",
                              extent=(-0.5, w - 0.5, h - 0.5, -0.5))
                for ax in (top, bottom):
                    _clean_axis(ax)
                group_axes.append(top)
                if i == 0 and c == 0:
                    top.set_ylabel("RGB", fontsize=12, color=_MUTED_COLOR)
                    bottom.set_ylabel(f"Pre-code ≥ p{q:g}", fontsize=12, color=_MUTED_COLOR)

            left, right = group_axes[0].get_position().x0, group_axes[-1].get_position().x1
            fig.text((left + right) / 2, 1 - (1.47 + extra) / fig_h,
                     f"Top {i + 1}  ·  {tile['count']}/{tile['n_patches']} patches",
                     ha="center", va="bottom", fontsize=12, color=_TEXT_COLOR)

        fig.text(0.45 / fig_w, 1 - 0.18 / fig_h, title, ha="left", va="top", fontsize=17,
                 fontweight="bold", color=_TEXT_COLOR)
        fig.text(0.45 / fig_w, 1 - 0.55 / fig_h, subtitle, ha="left", va="top", fontsize=12, color=_MUTED_COLOR)
        if note_lines:
            fig.text(0.45 / fig_w + 0.04 / fig_w, 1 - 0.92 / fig_h, "\n".join(note_lines), linespacing=1.35, ha="left", va="top", fontsize=11.5,
                     color=_TEXT_COLOR,
                     bbox=dict(boxstyle="round,pad=0.35", facecolor="#fdf1dc", edgecolor="#eda100", linewidth=0.9))
        fig.legend(handles=[Patch(facecolor=color, alpha=0.6, edgecolor="none",
                                  label=f"Patches with pre-code ≥ p{q:g} of the feature  (τ = {threshold:.3g})")],
                   loc="lower left", bbox_to_anchor=(0.45 / fig_w, 0.0), frameon=False, fontsize=12,
                   handlelength=1.4, borderaxespad=0.2)

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_stem.with_suffix(".png"), dpi=150, facecolor="white")
        if save_pdf:
            fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)


def shared_feature_notes(selection: pd.DataFrame, max_per_column: int = 6) -> List[Optional[str]]:
    """For each selection row, the other (column, entry, rank) pairs that selected the same feature, or None.

    Entries of the row's own column come first; the other columns are prefixed by their name.
    """
    notes = []
    for row in selection.itertuples(index=False):
        others = selection[(selection["feature"] == row.feature)
                           & ~((selection["column"] == row.column) & (selection["entry"] == row.entry))]
        if others.empty:
            notes.append(None)
            continue
        columns = [row.column] + [c for c in dict.fromkeys(others["column"]) if c != row.column]
        parts = []
        for column in columns:
            group = others[others["column"] == column].sort_values(["rank", "entry"])
            if group.empty:
                continue
            items = [f"{e} (top-{r})" for e, r in zip(group["entry"], group["rank"])]
            if len(items) > max_per_column:
                items = items[:max_per_column] + [f"+{len(items) - max_per_column} more"]
            prefix = "" if column == row.column else f"{column}: "
            parts.append(prefix + ", ".join(items))
        notes.append("Shared feature, also selected by:  " + "   ·   ".join(parts))
    return notes


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "entry"


# ----------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval_feature_mask_alignment")
def evaluate_feature_mask_alignment(cfg: DictConfig):
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    q = float(cfg.percentile)
    k = int(cfg.top_k_tiles)
    max_tiles = cfg.get("max_tiles")
    num_workers = int(cfg.get("num_workers", 8))
    column = str(cfg.vocab_column)
    out_dir = Path(cfg.output_dir) / f"feature_mask_alignment_p{q:g}" / str(cfg.eval_split) / column
    out_dir.mkdir(parents=True, exist_ok=True)

    ### Vocabulary -> selected features ###
    vocab_df = pd.read_csv(cfg.vocab_csv_path)
    if column not in vocab_df.columns:
        raise ValueError(f"vocab_column={column!r} is not a column of {cfg.vocab_csv_path}: {list(vocab_df.columns)}")
    enc_cfg = OmegaConf.create({
        "vocab_columns": [column],
        "vocab_class_column": cfg.vocab_class_column,
        "text_batch_size": cfg.get("text_batch_size", 128),
        "sae": {"text_enc_kwargs": OmegaConf.to_container(cfg.sae.text_enc_kwargs, resolve=True)},
    })
    vocabs = encode_vocabularies(enc_cfg, vocab_df, device)
    class_order = list(dict.fromkeys(vocab_df[cfg.vocab_class_column].astype(str)))
    class_colors = class_color_map(class_order)

    sae = load_trained_sae(cfg, device, logger=log)
    n_concepts = sae.net.get_dictionary().shape[0]
    if cfg.get("alive_feature_path") and Path(cfg.alive_feature_path).exists():
        alive = torch.load(cfg.alive_feature_path, map_location="cpu").bool()
    else:
        log.warning("No alive mask found, using all features")
        alive = torch.ones(n_concepts, dtype=torch.bool)
    alive_indices = torch.where(alive)[0].numpy()
    dictionary = sae.net.get_dictionary().detach().float().cpu()[alive_indices]

    selection = select_features(dictionary, alive_indices, vocabs, int(cfg.top_n_features))
    features = np.unique(selection["feature"].to_numpy())
    log.info(f"{len(selection)} (entry, rank) pairs -> {len(features)} distinct features")
    feat_pos = {int(f): i for i, f in enumerate(features)}
    features_t = torch.as_tensor(features, device=device)

    ### Image model, tiles ###
    model = load_img_model(cfg, device, logger=log)
    datamodule = get_data(cfg)
    dataset = datamodule.test_dataloader(split=cfg.eval_split).dataset
    # The split is passed as test_dataset: its loader is unshuffled with one tile per batch, so the batch
    # index is the tile id in every pass.
    act_datamodule = OnlineActDataModule(model=model, train_dataset=dataset, test_dataset=dataset,
                                         device=str(device), batch_size=1, test_batch_size=1)
    act_datamodule.load_input_norm(cfg.sae_ckpt_path)
    normalizer = next(t for t in datamodule.test_transform.transforms if hasattr(t, "mean") and hasattr(t, "std"))
    norm_mean = torch.as_tensor(normalizer.mean, dtype=torch.float32)
    norm_std = torch.as_tensor(normalizer.std, dtype=torch.float32)

    loader = act_datamodule.test_dataloader(num_workers=num_workers)
    thresholds = percentile_thresholds(loader, act_datamodule, sae, features_t, device, q, max_tiles)
    counts, tile_ids, tile_max, masks = top_tiles(loader, act_datamodule, sae, features_t, thresholds, k, device,
                                                  max_tiles)
    thresholds = thresholds.cpu().numpy()
    n_patches = masks.shape[-1]

    needed = sorted({int(t) for t in tile_ids.reshape(-1) if t >= 0})
    frames = load_tile_frames(dataset, needed, num_workers, norm_mean, norm_std)

    ### Results table + plots ###
    notes = shared_feature_notes(selection)
    n_shared = sum(note is not None for note in notes)
    log.info(f"{n_shared}/{len(selection)} (entry, rank) pairs use a feature also selected by another entry")
    records = []
    for row, note in tqdm(zip(selection.itertuples(index=False), notes), total=len(selection), desc="Plotting"):
        i = feat_pos[row.feature]
        valid = tile_ids[i] >= 0
        records.append({
            **row._asdict(), "threshold": float(thresholds[i]),
            "tile_ids": " ".join(str(t) for t in tile_ids[i][valid]),
            "patch_counts": " ".join(str(c) for c in counts[i][valid]),
            "tile_max_pre": " ".join(f"{m:.4g}" for m in tile_max[i][valid]),
            "n_patches_per_tile": n_patches,
            "also_selected_by": (note or "").replace("Shared feature, also selected by:  ", ""),
        })
        if not valid.any():
            continue
        tiles = [{"tile_id": int(tile_ids[i, j]), "count": int(counts[i, j]), "n_patches": n_patches,
                  "mask": masks[i, j], "frames": frames[int(tile_ids[i, j])]} for j in np.flatnonzero(valid)]
        entry_label = row.entry if row.column != "class_name" else row.entry.capitalize()
        plot_feature_masks(
            title=entry_label,
            subtitle=(f"{row.column} · top-{row.rank} feature {row.feature}  ·  cos(feature, text) = {row.cosine:.3f}"
                      f"  ·  class: {row.vocab_class}"),
            tiles=tiles, color=class_colors.get(row.vocab_class, "#eb6834"), threshold=float(thresholds[i]), q=q,
            output_stem=out_dir / _slug(row.entry) / f"top{row.rank}_feature_{row.feature:05d}",
            save_pdf=bool(cfg.get("save_pdf", False)),
            note=note,
        )

    pd.DataFrame(records).to_csv(out_dir / "feature_mask_alignment.csv", index=False)
    log.info(f"Saved {len(records)} rows and plots to {out_dir}")


if __name__ == "__main__":
    evaluate_feature_mask_alignment()
