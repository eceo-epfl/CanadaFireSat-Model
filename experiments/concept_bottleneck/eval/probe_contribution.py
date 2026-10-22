from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from tqdm import tqdm

from experiments.concept_bottleneck.eval.utils import compute_text_dist, load_img_model
from experiments.concept_bottleneck.sae.act_datamodule import OnlineActDataModule
from experiments.concept_bottleneck.sae.sae_module import plSAE
from overcomplete.sae.train import extract_input
from src.constants import CANADA_REGIONS, CONFIG_PATH
from src.data import get_data
from src.models.module_img import ImgModule
from experiments.concept_bottleneck.bottleneck.probe import SAEProbe
from experiments.concept_bottleneck.sae.utils import get_pylogger


log = get_pylogger(__name__)


def feature_terms_from_vocab(
    probe: SAEProbe,
    vocab_csv_path: Optional[Path],
) -> tuple[list[str], Optional[np.ndarray], Optional[list[str]]]:
    """Return the matched term of each alive feature, its vocab index and the full vocab."""
    alive_indices = probe.alive_idx.detach().cpu()
    if vocab_csv_path is None:
        return [f"feature_{int(index)}" for index in alive_indices], None, None

    vocab_terms = pd.read_csv(vocab_csv_path)["term"].astype(str).tolist()
    _, vocab_indices = compute_text_dist(
        probe.sae.get_dictionary().cpu()[alive_indices],
        probe.label_vocab_emb.cpu(),
    )
    vocab_indices = vocab_indices.numpy()
    return [vocab_terms[int(index)] for index in vocab_indices], vocab_indices, vocab_terms


def name_term_cluster(
    member_indices: np.ndarray,
    vocab_terms: list[str],
    vocab_emb: np.ndarray,
    term_importance: np.ndarray,
    naming: str,
) -> str:
    """Pick a display name for a cluster of vocab terms, suffixed with the number of merged terms.

    - medoid: member with the highest mean cosine to the other members (most central term).
    - dominant: member carrying the largest total |contribution| (what the probe relies on most).
    - centroid: vocab term (possibly outside the cluster) closest to the mean member embedding,
      which can surface a broader term covering all members.
    """
    if member_indices.size == 1:
        return vocab_terms[int(member_indices[0])]

    member_emb = vocab_emb[member_indices]
    if naming == "medoid":
        representative = member_indices[np.argmax((member_emb @ member_emb.T).mean(axis=1))]
    elif naming == "dominant":
        representative = member_indices[np.argmax(term_importance)]
    elif naming == "centroid":
        centroid = member_emb.mean(axis=0)
        representative = np.argmax(vocab_emb @ (centroid / np.linalg.norm(centroid)))
    else:
        raise ValueError(f"Unknown merge naming strategy: {naming}")
    return f"{vocab_terms[int(representative)]} (+{member_indices.size - 1})"


def group_features_by_term(
    vocab_indices: np.ndarray,
    vocab_terms: list[str],
    vocab_emb: torch.Tensor,
    feature_importance: np.ndarray,
    cosine_threshold: Optional[float],
    linkage_method: str,
    naming: str,
    center_embeddings: bool,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Group features sharing a vocab term, then optionally merge terms whose MS-CLIP text
    embeddings are close (agglomerative clustering, cut at cosine distance 1 - threshold).
    Centering on the full-vocab mean removes the shared CLIP text direction, which otherwise puts
    the median pairwise cosine around 0.5 and the median nearest-neighbour cosine around 0.9.

    Returns the group id of every feature, the group display names and the "; "-joined members.
    """
    used_terms, feature_to_term = np.unique(vocab_indices, return_inverse=True)
    vocab_emb = vocab_emb.detach().cpu().double().numpy()
    if center_embeddings:
        vocab_emb = vocab_emb - vocab_emb.mean(axis=0, keepdims=True)
    vocab_emb = vocab_emb / np.linalg.norm(vocab_emb, axis=1, keepdims=True)

    if cosine_threshold is None or used_terms.size < 2:
        term_to_group = np.arange(used_terms.size)
    else:
        dendrogram = linkage(
            pdist(vocab_emb[used_terms], metric="cosine"), method=linkage_method
        )
        term_to_group = (
            fcluster(dendrogram, t=1.0 - cosine_threshold, criterion="distance") - 1
        )

    term_importance = np.bincount(
        feature_to_term, weights=feature_importance, minlength=used_terms.size
    )
    group_names, group_members = [], []
    for group in range(term_to_group.max() + 1):
        in_group = term_to_group == group
        member_indices = used_terms[in_group]
        group_names.append(
            name_term_cluster(
                member_indices, vocab_terms, vocab_emb, term_importance[in_group], naming
            )
        )
        group_members.append("; ".join(vocab_terms[int(index)] for index in member_indices))
    return term_to_group[feature_to_term], group_names, group_members


def merge_contributions(values: torch.Tensor, group_ids: np.ndarray, n_groups: int) -> torch.Tensor:
    """Sum per-feature contributions (last dim) within each group (exact, the probe logit is linear in w*z)."""
    index = torch.as_tensor(group_ids, device=values.device, dtype=torch.long)
    merged = torch.zeros(*values.shape[:-1], n_groups, dtype=values.dtype, device=values.device)
    return merged.index_add_(-1, index, values)


def aggregate_probe_contributions(
    probe: SAEProbe,
    dataloader: torch.utils.data.DataLoader,
    datamodule: OnlineActDataModule,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int, dict[str, torch.Tensor]]:
    """Return E[w*z | fire] and E[w*z | no-fire] for every alive feature, overall and per province.

    The per-province entries are [n_regions, n_features] means and [n_regions] patch counts,
    indexed like CANADA_REGIONS; patches with an unknown region only count towards the totals.
    """
    probe.eval()
    weights = probe.probe.weight.detach()[0]
    fire_sum = torch.zeros_like(weights)
    nofire_sum = torch.zeros_like(weights)
    fire_count = 0
    nofire_count = 0
    n_regions = len(CANADA_REGIONS)
    region_sum = {name: torch.zeros(n_regions, weights.shape[0], device=weights.device) for name in ("fire", "nofire")}
    region_count = {name: torch.zeros(n_regions, device=weights.device) for name in ("fire", "nofire")}

    for raw_batch in tqdm(dataloader, desc="Aggregating probe contributions"):
        batch = datamodule.on_after_batch_transfer(raw_batch, 0)
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch["labels"].reshape(-1).long()

        with torch.no_grad():
            _, codes, _ = probe.sae(extract_input(batch))
            alive_codes = codes[:, probe.alive_idx]
            contributions = alive_codes * weights.unsqueeze(0)

        fire_mask = labels == 1
        nofire_mask = labels == 0
        if fire_mask.any():
            fire_sum += contributions[fire_mask].sum(dim=0)
            fire_count += int(fire_mask.sum().item())
        if nofire_mask.any():
            nofire_sum += contributions[nofire_mask].sum(dim=0)
            nofire_count += int(nofire_mask.sum().item())

        if "region" in batch:
            regions = batch["region"].reshape(-1).long()
            for name, mask in (("fire", fire_mask), ("nofire", nofire_mask)):
                mask = mask & (regions >= 0)
                region_sum[name].index_add_(0, regions[mask], contributions[mask])
                region_count[name].index_add_(0, regions[mask], torch.ones_like(regions[mask], dtype=torch.float))

    fire_mean = fire_sum / max(fire_count, 1)
    nofire_mean = nofire_sum / max(nofire_count, 1)
    region_stats = {
        "fire_mean": region_sum["fire"] / region_count["fire"].clamp_min(1).unsqueeze(1),
        "nofire_mean": region_sum["nofire"] / region_count["nofire"].clamp_min(1).unsqueeze(1),
        "fire_count": region_count["fire"],
        "nofire_count": region_count["nofire"],
    }
    return fire_mean, nofire_mean, fire_count, nofire_count, region_stats


def save_contributions(
    output_dir: Path,
    eval_split: str,
    feature_names: list[str],
    feature_indices: torch.Tensor,
    weights: torch.Tensor,
    fire_mean: torch.Tensor,
    nofire_mean: torch.Tensor,
    group_ids: np.ndarray,
    group_names: list[str],
    group_members: list[str],
) -> None:
    table = pd.DataFrame(
        {
            "feature_index": feature_indices.cpu().numpy(),
            "term": feature_names,
            "group": group_ids,
            "group_term": [group_names[group] for group in group_ids],
            "probe_weight": weights.cpu().numpy(),
            "fire_mean_contribution": fire_mean.cpu().numpy(),
            "nofire_mean_contribution": nofire_mean.cpu().numpy(),
            "nofire_evidence": (-nofire_mean).cpu().numpy(),
        }
    )
    table.to_csv(
        output_dir / f"probe_feature_contributions_{eval_split}.csv",
        index=False,
    )

    merged = (
        table.groupby("group")
        .agg(
            n_features=("feature_index", "size"),
            feature_indices=("feature_index", lambda x: " ".join(map(str, x))),
            fire_mean_contribution=("fire_mean_contribution", "sum"),
            nofire_mean_contribution=("nofire_mean_contribution", "sum"),
            nofire_evidence=("nofire_evidence", "sum"),
        )
        .reset_index()
    )
    merged.insert(1, "group_term", [group_names[group] for group in merged["group"]])
    merged.insert(2, "members", [group_members[group] for group in merged["group"]])
    merged.sort_values("fire_mean_contribution", ascending=False).to_csv(
        output_dir / f"probe_term_contributions_{eval_split}.csv",
        index=False,
    )


def save_region_contributions(
    output_dir: Path,
    eval_split: str,
    group_names: list[str],
    region_fire_mean: torch.Tensor,
    region_nofire_mean: torch.Tensor,
    region_fire_count: torch.Tensor,
    region_nofire_count: torch.Tensor,
) -> None:
    """Long-format table: one row per (province, merged term), provinces without patches dropped."""
    tables = []
    for region_id, region in enumerate(CANADA_REGIONS):
        n_fire, n_nofire = int(region_fire_count[region_id]), int(region_nofire_count[region_id])
        if n_fire + n_nofire == 0:
            continue
        tables.append(
            pd.DataFrame(
                {
                    "region": region,
                    "fire_patches": n_fire,
                    "no_fire_patches": n_nofire,
                    "group_term": group_names,
                    "fire_mean_contribution": region_fire_mean[region_id].cpu().numpy(),
                    "nofire_mean_contribution": region_nofire_mean[region_id].cpu().numpy(),
                    "nofire_evidence": (-region_nofire_mean[region_id]).cpu().numpy(),
                }
            )
        )
    if not tables:
        log.warning("No region information in the batches; skipping the per-province contributions.")
        return
    pd.concat(tables).to_csv(output_dir / f"probe_term_contributions_by_region_{eval_split}.csv", index=False)


# Paper styling: palette slots 1 (blue) and 2 (orange) of the reference viz palette, CVD-safe pair.
TARGET_COLORS = {"fire": "#eb6834", "no_fire": "#2a78d6"}
TARGET_LABELS = {"fire": "Fire", "no_fire": "No fire"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
FONT_FAMILY = "Arial, Helvetica, sans-serif"
# Canada Post codes, keeps the province axis of the heatmaps compact
REGION_CODES = {
    "Alberta": "AB", "British Columbia": "BC", "Manitoba": "MB", "New Brunswick": "NB",
    "Newfoundland and Labrador": "NL", "Northwest Territories": "NT", "Nova Scotia": "NS", "Nunavut": "NU",
    "Ontario": "ON", "Prince Edward Island": "PE", "Quebec": "QC", "Saskatchewan": "SK", "Yukon": "YT",
}


def select_top_contributions(
    values: torch.Tensor, top_k: int, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (top indices sorted by decreasing contribution, their values, remaining positive values)."""
    contributions = values.detach().cpu().double().numpy()
    positive = np.flatnonzero(np.isfinite(contributions) & (contributions > threshold))
    positive = positive[np.argsort(contributions[positive])[::-1]]
    return positive[:top_k], contributions[positive[:top_k]], contributions[positive[top_k:]]


def hex_to_rgba(color: str, alpha: float) -> str:
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def write_sankey(
    output_dir: Path,
    eval_split: str,
    feature_names: list[str],
    values: torch.Tensor,
    target_name: str,
    top_k: int,
    threshold: float,
    region: Optional[tuple[str, int]] = None,
) -> None:
    """Sankey of the top-k terms flowing into the target; `region` = (province, patch count) restricts
    the title and file name to that province (written under by_region/)."""
    selected_indices, selected_values, rest = select_top_contributions(values, top_k, threshold)
    if selected_indices.size == 0:
        log.warning(f"No features passed the {target_name} Sankey threshold{f' in {region[0]}' if region else ''}.")
        return

    labels = [feature_names[index] for index in selected_indices]
    source_values = selected_values.tolist()
    source_colors = [hex_to_rgba(TARGET_COLORS[target_name], 0.9)] * len(labels)
    link_colors = [hex_to_rgba(TARGET_COLORS[target_name], 0.35)] * len(labels)

    total = float(np.sum(source_values))
    coverage = total / (total + rest.sum())
    n_sources = len(source_values)

    target_label = TARGET_LABELS[target_name]
    condition = "fire" if target_name == "fire" else "no-fire"
    scope = f" in {region[0]}" if region else ""
    patches = f"{region[1]:,} {condition} patches, " if region else ""
    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            valueformat=".3f",
            node={
                "label": [f"{label}  {value:.3f}" for label, value in zip(labels, source_values)]
                + [f"{target_label}  {total:.3f}"],
                "pad": 6,
                "thickness": 14,
                "color": source_colors + [TARGET_COLORS[target_name]],
                "line": {"width": 0},
                "hovertemplate": "%{label}<extra></extra>",
            },
            link={
                "source": list(range(n_sources)),
                "target": [n_sources] * n_sources,
                "value": source_values,
                "color": link_colors,
                "hovertemplate": "%{source.label} \u2192 %{target.label}: %{value}<extra></extra>",
            },
            textfont={"family": FONT_FAMILY, "size": 12, "color": TEXT_PRIMARY, "shadow": "none"},
        )
    )
    figure.update_layout(
        title={
            "text": f"Evidence for <b>{target_label.lower()}</b> by concept{scope}"
            f"<br><sup>Mean probe contribution w\u00b7z (logit units). Top {n_sources} of "
            f"{n_sources + rest.size} supporting terms, {coverage:.0%} of the positive evidence ({patches}{eval_split} split)</sup>",
            "x": 0.01,
            "xanchor": "left",
            "font": {"family": FONT_FAMILY, "size": 15, "color": TEXT_PRIMARY},
        },
        font={"family": FONT_FAMILY, "size": 12, "color": TEXT_SECONDARY},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        width=900,
        height=max(300, 26 * n_sources + 120),
    )
    output_stem = output_dir / f"probe_contributions_{target_name}_{eval_split}"
    if region:
        (output_dir / "by_region").mkdir(exist_ok=True)
        output_stem = output_dir / "by_region" / f"probe_contributions_{target_name}_{REGION_CODES[region[0]]}_{eval_split}"
    figure.write_html(output_stem.with_suffix(".html"))
    figure.write_image(output_stem.with_suffix(".png"), scale=3)
    figure.write_image(output_stem.with_suffix(".pdf"))


def write_contribution_bars(
    output_dir: Path,
    eval_split: str,
    feature_names: list[str],
    fire_values: torch.Tensor,
    nofire_values: torch.Tensor,
    top_k: int,
    threshold: float,
) -> None:
    """Two-panel ranked bar chart (fire / no-fire evidence) sized for a two-column paper figure."""
    panels = []
    for target_name, values in (("fire", fire_values), ("no_fire", nofire_values)):
        selected_indices, selected_values, _ = select_top_contributions(values, top_k, threshold)
        panels.append((target_name, [feature_names[index] for index in selected_indices], selected_values))
    n_rows = max(len(labels) for _, labels, _ in panels)
    if n_rows == 0:
        log.warning("No features passed the contribution bar threshold.")
        return

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_PRIMARY,
        "pdf.fonttype": 42,  # embed TrueType so the text stays editable in the PDF
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            1, 2, figsize=(7.0, 0.16 * n_rows + 0.7), constrained_layout=True
        )
        for axis, (target_name, labels, values) in zip(axes, panels):
            positions = np.arange(len(labels))
            axis.barh(positions, values, height=0.7, color=TARGET_COLORS[target_name], linewidth=0)
            for position, value in zip(positions, values):
                axis.text(value, position, f" {value:.3f}", va="center", ha="left",
                          fontsize=6, color=TEXT_SECONDARY)
            axis.set_yticks(positions, labels)
            axis.set_ylim(n_rows - 0.5, -0.5)
            axis.set_xlim(0, (values.max() if len(values) else 1.0) * 1.18)
            axis.tick_params(axis="y", length=0)
            axis.grid(axis="x", color="#e1e0d9", linewidth=0.5)
            axis.set_axisbelow(True)
            for side in ("top", "right", "left"):
                axis.spines[side].set_visible(False)
            axis.spines["bottom"].set_linewidth(0.5)
            axis.set_title(f"{TARGET_LABELS[target_name]} evidence", loc="left",
                           fontsize=8, fontweight="bold", color=TEXT_PRIMARY)
            sign = "" if target_name == "fire" else "\u2212"
            condition = "fire" if target_name == "fire" else "no fire"
            axis.set_xlabel(f"{sign}E[w\u00b7z | {condition}] (logit units)")

        output_stem = output_dir / f"probe_contributions_bars_{eval_split}"
        for suffix in (".pdf", ".svg"):
            figure.savefig(output_stem.with_suffix(suffix))
        figure.savefig(output_stem.with_suffix(".png"), dpi=300)
        plt.close(figure)


def format_count(count: float) -> str:
    """Compact patch count (3 significant digits) for narrow axis labels: 950, 14.4k, 986k, 1.44M."""
    for divisor, suffix in ((1e6, "M"), (1e3, "k")):
        if count >= divisor:
            return f"{count / divisor:.3g}{suffix}"
    return f"{int(count)}"


def write_region_heatmap(
    output_dir: Path,
    eval_split: str,
    feature_names: list[str],
    global_values: torch.Tensor,
    region_values: torch.Tensor,
    region_counts: torch.Tensor,
    target_name: str,
    top_k: int,
    threshold: float,
    min_patches: int,
) -> None:
    """Terms x provinces heatmap of the per-province evidence for the top-k terms of the global ranking.

    Diverging scale centred on 0: a term can support the target in one province and oppose it in
    another. Provinces with fewer than `min_patches` patches of the class are left out (too noisy).
    """
    selected_indices, _, _ = select_top_contributions(global_values, top_k, threshold)
    counts = region_counts.cpu().numpy()
    region_ids = [index for index in np.argsort(counts)[::-1] if counts[index] >= min_patches]
    if selected_indices.size == 0 or not region_ids:
        log.warning(f"Not enough terms or provinces for the {target_name} region heatmap.")
        return

    values = region_values.detach().cpu().double().numpy()[np.ix_(region_ids, selected_indices)].T
    limit = np.abs(values).max()
    colormap = LinearSegmentedColormap.from_list(
        "evidence", [TARGET_COLORS["no_fire" if target_name == "fire" else "fire"], "#f0efec", TARGET_COLORS[target_name]]
    )
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.edgecolor": TEXT_SECONDARY,
        "xtick.color": TEXT_PRIMARY,
        "ytick.color": TEXT_PRIMARY,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(
            figsize=(2.0 + 0.55 * len(region_ids), 0.16 * len(selected_indices) + 0.9), constrained_layout=True
        )
        image = axis.imshow(values, cmap=colormap, vmin=-limit, vmax=limit, aspect="auto")
        for row, column in np.ndindex(values.shape):
            axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=5,
                      color=TEXT_PRIMARY)
        axis.set_xticks(
            range(len(region_ids)),
            [f"{REGION_CODES[CANADA_REGIONS[index]]}\n{format_count(counts[index])}" for index in region_ids],
        )
        axis.xaxis.tick_top()
        axis.set_yticks(range(len(selected_indices)), [feature_names[index] for index in selected_indices])
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        # White cell borders
        axis.set_xticks(np.arange(len(region_ids) + 1) - 0.5, minor=True)
        axis.set_yticks(np.arange(len(selected_indices) + 1) - 0.5, minor=True)
        axis.grid(which="minor", color="white", linewidth=1)
        axis.tick_params(which="minor", length=0)

        sign = "" if target_name == "fire" else "−"
        condition = "fire" if target_name == "fire" else "no fire"
        colorbar = figure.colorbar(image, ax=axis, shrink=0.5, aspect=25)
        colorbar.set_label(f"{sign}E[w·z | {condition}, province] (logit units)")
        colorbar.outline.set_visible(False)
        axis.set_title(f"{TARGET_LABELS[target_name]} evidence by province (columns: code, {condition} patches)",
                       loc="left", fontsize=8, fontweight="bold", color=TEXT_PRIMARY)

        output_stem = output_dir / f"probe_contributions_{target_name}_by_region_{eval_split}"
        for suffix in (".pdf", ".svg"):
            figure.savefig(output_stem.with_suffix(suffix))
        figure.savefig(output_stem.with_suffix(".png"), dpi=300)
        plt.close(figure)


@hydra.main(version_base="1.2", config_path=str(CONFIG_PATH), config_name="eval_probe_contribution")
def evaluate_probe_contributions(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = load_img_model(cfg, device)

    assert cfg.MODEL.test_max_seq_len is None, "For evaluation, test_max_seq_len must be None to allow adaptive sequence lengths."
    datamodule = get_data(cfg)
    train_dataset = datamodule.test_dataloader(split="train").dataset
    val_dataset = datamodule.test_dataloader(split="val").dataset
    test_dataset = datamodule.test_dataloader(split="test").dataset
    probe_datamodule = OnlineActDataModule(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        batch_size=1,
        test_batch_size=1,
    )

    log.info(f"Loaded probe checkpoint from {cfg.probe_checkpoint_path}")
    probe: SAEProbe = hydra.utils.instantiate(cfg.probe)
    probe_checkpoint = torch.load(cfg.probe_checkpoint_path, map_location=device)
    mis_keys, un_keys = probe.load_state_dict(probe_checkpoint["state_dict"], strict=True)
    if mis_keys:
        log.warning(f"Missing keys in probe checkpoint: {mis_keys}")
    if un_keys:
        log.warning(f"Unexpected keys in probe checkpoint: {un_keys}")
    probe.to(device).eval()

    dataloaders = {
        "train": probe_datamodule.train_dataloader(),
        "val": probe_datamodule.val_dataloader(),
        "test": probe_datamodule.test_dataloader(),
    }
    fire_mean, nofire_mean, fire_count, nofire_count, region_stats = aggregate_probe_contributions(
        probe=probe,
        dataloader=dataloaders[cfg.eval_split],
        datamodule=probe_datamodule,
        device=device,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names, vocab_indices, vocab_terms = feature_terms_from_vocab(
        probe, Path(cfg.vocab_csv_path) if cfg.get("vocab_csv_path") else None
    )
    merge_cosine_threshold = cfg.get("merge_cosine_threshold")
    if vocab_indices is not None and cfg.get("merge_identical_terms", True):
        group_ids, group_names, group_members = group_features_by_term(
            vocab_indices=vocab_indices,
            vocab_terms=vocab_terms,
            vocab_emb=probe.label_vocab_emb,
            feature_importance=(fire_mean.abs() + nofire_mean.abs()).cpu().double().numpy(),
            cosine_threshold=merge_cosine_threshold,
            linkage_method=cfg.get("merge_linkage", "average"),
            naming=cfg.get("merge_naming", "medoid"),
            center_embeddings=cfg.get("merge_center_embeddings", True),
        )
    else:
        group_ids = np.arange(len(feature_names))
        group_names, group_members = list(feature_names), list(feature_names)
    log.info(f"Merged {len(feature_names)} alive features into {len(group_names)} terms.")

    save_contributions(
        output_dir=output_dir,
        eval_split=cfg.eval_split,
        feature_names=feature_names,
        feature_indices=probe.alive_idx.detach().cpu(),
        weights=probe.probe.weight.detach()[0].cpu(),
        fire_mean=fire_mean,
        nofire_mean=nofire_mean,
        group_ids=group_ids,
        group_names=group_names,
        group_members=group_members,
    )
    group_fire_mean = merge_contributions(fire_mean, group_ids, len(group_names))
    group_nofire_mean = merge_contributions(nofire_mean, group_ids, len(group_names))

    threshold = float(cfg.contribution_threshold)
    write_sankey(
        output_dir,
        cfg.eval_split,
        group_names,
        group_fire_mean,
        "fire",
        cfg.top_k_features,
        threshold,
    )
    write_sankey(
        output_dir,
        cfg.eval_split,
        group_names,
        -group_nofire_mean,
        "no_fire",
        cfg.top_k_features,
        threshold,
    )
    write_contribution_bars(
        output_dir,
        cfg.eval_split,
        group_names,
        group_fire_mean,
        -group_nofire_mean,
        cfg.top_k_features,
        threshold,
    )

    region_fire_mean = merge_contributions(region_stats["fire_mean"], group_ids, len(group_names))
    region_nofire_mean = merge_contributions(region_stats["nofire_mean"], group_ids, len(group_names))
    save_region_contributions(
        output_dir,
        cfg.eval_split,
        group_names,
        region_fire_mean,
        region_nofire_mean,
        region_stats["fire_count"],
        region_stats["nofire_count"],
    )
    min_region_patches = int(cfg.get("min_region_patches", 500))
    for target_name, global_values, region_values, region_counts in (
        ("fire", group_fire_mean, region_fire_mean, region_stats["fire_count"]),
        ("no_fire", -group_nofire_mean, -region_nofire_mean, region_stats["nofire_count"]),
    ):
        write_region_heatmap(
            output_dir,
            cfg.eval_split,
            group_names,
            global_values,
            region_values,
            region_counts,
            target_name,
            cfg.top_k_features,
            threshold,
            min_region_patches,
        )
        # One Sankey per province, ranked by that province's own top terms
        for region_id, region in enumerate(CANADA_REGIONS):
            if region_counts[region_id] < min_region_patches:
                continue
            write_sankey(
                output_dir,
                cfg.eval_split,
                group_names,
                region_values[region_id],
                target_name,
                int(cfg.get("region_top_k_features", 5)),
                threshold,
                region=(region, int(region_counts[region_id])),
            )

    with (output_dir / f"probe_contribution_summary_{cfg.eval_split}.txt").open(
        "w"
    ) as summary:
        summary.write(f"eval_split: {cfg.eval_split}\n")
        summary.write(f"fire_patches: {fire_count}\n")
        summary.write(f"no_fire_patches: {nofire_count}\n")
        summary.write(f"contribution_threshold: {threshold}\n")
        summary.write(f"top_k_features: {cfg.top_k_features}\n")
        summary.write(f"alive_features: {len(feature_names)}\n")
        summary.write(f"merged_terms: {len(group_names)}\n")
        summary.write(f"merge_cosine_threshold: {merge_cosine_threshold}\n")
        summary.write(f"merge_naming: {cfg.get('merge_naming', 'medoid')}\n")
        summary.write(f"min_region_patches: {min_region_patches}\n")
        for region_id, region in enumerate(CANADA_REGIONS):
            summary.write(
                f"  {region}: fire_patches={int(region_stats['fire_count'][region_id])}, "
                f"no_fire_patches={int(region_stats['nofire_count'][region_id])}\n"
            )


if __name__ == "__main__":
    evaluate_probe_contributions()
