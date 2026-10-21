from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from experiments.concept_bottleneck.eval.utils import compute_text_dist, load_img_model
from experiments.concept_bottleneck.sae.act_datamodule import OnlineActDataModule
from experiments.concept_bottleneck.sae.sae_module import plSAE
from overcomplete.sae.train import extract_input
from src.constants import CONFIG_PATH
from src.data import get_data
from src.models.module_img import ImgModule
from experiments.concept_bottleneck.bottleneck.probe import SAEProbe
from experiments.concept_bottleneck.sae.utils import get_pylogger


log = get_pylogger(__name__)


def feature_names_from_vocab(
    probe: SAEProbe,
    vocab_csv_path: Optional[Path],
) -> list[str]:
    alive_indices = probe.alive_idx.detach().cpu()
    fallback_names = [f"feature_{int(index)}" for index in alive_indices]
    if vocab_csv_path is None:
        return fallback_names

    vocab_terms = pd.read_csv(vocab_csv_path)["term"].astype(str).tolist()
    _, vocab_indices = compute_text_dist(
        probe.sae.get_dictionary().cpu()[alive_indices],
        probe.label_vocab_emb.cpu(),
    )
    return [vocab_terms[int(index)] for index in vocab_indices]


def aggregate_probe_contributions(
    probe: SAEProbe,
    dataloader: torch.utils.data.DataLoader,
    datamodule: OnlineActDataModule,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return E[w*z | fire] and E[w*z | no-fire] for every alive feature."""
    probe.eval()
    weights = probe.probe.weight.detach()[0]
    fire_sum = torch.zeros_like(weights)
    nofire_sum = torch.zeros_like(weights)
    fire_count = 0
    nofire_count = 0

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

    fire_mean = fire_sum / max(fire_count, 1)
    nofire_mean = nofire_sum / max(nofire_count, 1)
    return fire_mean, nofire_mean, fire_count, nofire_count


def save_contributions(
    output_dir: Path,
    eval_split: str,
    feature_names: list[str],
    feature_indices: torch.Tensor,
    weights: torch.Tensor,
    fire_mean: torch.Tensor,
    nofire_mean: torch.Tensor,
) -> None:
    table = pd.DataFrame(
        {
            "feature_index": feature_indices.cpu().numpy(),
            "term": feature_names,
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


def write_sankey(
    output_dir: Path,
    eval_split: str,
    feature_names: list[str],
    values: torch.Tensor,
    target_name: str,
    top_k: int,
    threshold: float,
) -> None:
    contributions = values.detach().cpu().numpy()
    positive = np.isfinite(contributions) & (contributions > threshold)
    selected_indices = np.flatnonzero(positive)
    selected_indices = selected_indices[np.argsort(contributions[selected_indices])[::-1]]
    selected_indices = selected_indices[:top_k]

    if selected_indices.size == 0:
        log.warning(f"No features passed the {target_name} Sankey threshold.")
        return

    labels = [feature_names[index] for index in selected_indices] + [target_name]
    target_index = len(selected_indices)
    figure = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node={"label": labels, "pad": 18, "thickness": 18},
            link={
                "source": list(range(len(selected_indices))),
                "target": [target_index] * len(selected_indices),
                "value": contributions[selected_indices].tolist(),
            },
        )
    )
    figure.update_layout(
        title=f"Probe contribution flow to {target_name}",
        font_size=11,
        width=1100,
        height=max(600, 24 * len(selected_indices)),
    )
    output_stem = (
        output_dir
        / f"probe_contributions_{target_name.lower().replace('-', '_')}_{eval_split}"
    )
    figure.write_html(output_stem.with_suffix(".html"))
    figure.write_image(output_stem.with_suffix(".png"))
    figure.write_image(output_stem.with_suffix(".pdf"))


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
    fire_mean, nofire_mean, fire_count, nofire_count = aggregate_probe_contributions(
        probe=probe,
        dataloader=dataloaders[cfg.eval_split],
        datamodule=probe_datamodule,
        device=device,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = feature_names_from_vocab(
        probe, Path(cfg.vocab_csv_path) if cfg.get("vocab_csv_path") else None
    )
    save_contributions(
        output_dir=output_dir,
        eval_split=cfg.eval_split,
        feature_names=feature_names,
        feature_indices=probe.alive_idx.detach().cpu(),
        weights=probe.probe.weight.detach()[0].cpu(),
        fire_mean=fire_mean,
        nofire_mean=nofire_mean,
    )

    threshold = float(cfg.contribution_threshold)
    write_sankey(
        output_dir,
        cfg.eval_split,
        feature_names,
        fire_mean,
        "fire",
        cfg.top_k_features,
        threshold,
    )
    write_sankey(
        output_dir,
        cfg.eval_split,
        feature_names,
        -nofire_mean,
        "no_fire",
        cfg.top_k_features,
        threshold,
    )

    with (output_dir / f"probe_contribution_summary_{cfg.eval_split}.txt").open(
        "w"
    ) as summary:
        summary.write(f"eval_split: {cfg.eval_split}\n")
        summary.write(f"fire_patches: {fire_count}\n")
        summary.write(f"no_fire_patches: {nofire_count}\n")
        summary.write(f"contribution_threshold: {threshold}\n")
        summary.write(f"top_k_features: {cfg.top_k_features}\n")


if __name__ == "__main__":
    evaluate_probe_contributions()
