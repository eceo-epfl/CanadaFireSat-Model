from pathlib import Path
from typing import Sequence

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from typing import Optional, Callable

from tqdm import tqdm
import yaml

from experiments.concept_bottleneck.eval.utils import compute_text_dist
from overcomplete.sae.train import extract_input

from experiments.concept_bottleneck.sae.act_datamodule import OnlineActDataModule
from experiments.concept_bottleneck.sae.metrics import _cosine_similarity_matrix
from experiments.concept_bottleneck.sae.sae_module import plSAE
from experiments.concept_bottleneck.sae.utils import get_pylogger
from src.constants import CONFIG_PATH
from src.data import get_data
import pytorch_lightning as pl

from src.models.module_img import ImgModule
import re


log = get_pylogger(__name__)


def summarize_water_activations(
    delta_mean_act: torch.Tensor,
    alive_feature: torch.Tensor,
    water_feature_indices: np.ndarray,
    matching_terms: list[str],
    output_dir: Path,
    eval_split: str,
    use_normalized_activations: bool,
) -> None:
    """Compute robust water-concept scores and save a text report."""
    alive_feature_indices = torch.where(alive_feature)[0].cpu().numpy()
    nonwater_feature_indices = np.setdiff1d(alive_feature_indices, water_feature_indices)

    water_delta_mean_act = delta_mean_act[
        torch.as_tensor(water_feature_indices, dtype=torch.long, device=delta_mean_act.device)
    ]
    water_delta_np = water_delta_mean_act.detach().cpu().numpy()
    water_delta_np = water_delta_np[np.isfinite(water_delta_np)]
    all_delta_np = delta_mean_act[alive_feature].detach().cpu().numpy()
    all_delta_np = all_delta_np[np.isfinite(all_delta_np)]
    nonwater_delta_np = delta_mean_act[
        torch.as_tensor(nonwater_feature_indices, dtype=torch.long, device=delta_mean_act.device)
    ].detach().cpu().numpy()
    nonwater_delta_np = nonwater_delta_np[np.isfinite(nonwater_delta_np)]

    if water_delta_np.size:
        mean_delta = float(np.mean(water_delta_np))
        median_delta = float(np.median(water_delta_np))
        mean_abs_delta = float(np.mean(np.abs(water_delta_np)))
        median_abs_delta = float(np.median(np.abs(water_delta_np)))
        positive_fraction = float(np.mean(water_delta_np > 0))
        selectivity_score = mean_delta / (mean_abs_delta + 1e-8)
        median_selectivity_score = median_delta / (median_abs_delta + 1e-8)
        all_mean_delta = float(np.mean(all_delta_np))
        all_median_delta = float(np.median(all_delta_np))
        nonwater_mean_delta = float(np.mean(nonwater_delta_np)) if nonwater_delta_np.size else float("nan")
        nonwater_median_delta = float(np.median(nonwater_delta_np)) if nonwater_delta_np.size else float("nan")
        nonwater_mad_delta = (
            float(np.median(np.abs(nonwater_delta_np - nonwater_median_delta)))
            if nonwater_delta_np.size
            else float("nan")
        )
        # Scale MAD to the Gaussian-equivalent standard deviation while remaining robust to outliers.
        nonwater_robust_scale = 1.4826 * nonwater_mad_delta
        water_vs_nonwater_delta = mean_delta - nonwater_mean_delta
        water_vs_nonwater_median_delta = median_delta - nonwater_median_delta
        water_vs_nonwater_score = water_vs_nonwater_delta / (nonwater_robust_scale + 1e-8)
        water_vs_nonwater_median_score = water_vs_nonwater_median_delta / (nonwater_robust_scale + 1e-8)
        # With normalized activations, the delta is already standardized per concept.
        primary_water_score = (
            water_vs_nonwater_median_delta
            if use_normalized_activations
            else water_vs_nonwater_median_score
        )
    else:
        mean_delta = median_delta = mean_abs_delta = median_abs_delta = float("nan")
        positive_fraction = selectivity_score = median_selectivity_score = float("nan")
        all_mean_delta = all_median_delta = float("nan")
        nonwater_mean_delta = nonwater_median_delta = float("nan")
        nonwater_mad_delta = nonwater_robust_scale = float("nan")
        water_vs_nonwater_delta = water_vs_nonwater_median_delta = float("nan")
        water_vs_nonwater_score = water_vs_nonwater_median_score = float("nan")
        primary_water_score = float("nan")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "water_delta_activation_summary.txt"
    with report_path.open("w") as report:
        report.write(f"eval_split: {eval_split}\n")
        report.write(f"n_water_features: {water_delta_np.size}\n")
        report.write(f"mean_delta_mean_act: {mean_delta:.6f}\n")
        report.write(f"median_delta_mean_act: {median_delta:.6f}\n")
        report.write(f"mean_absolute_delta: {mean_abs_delta:.6f}\n")
        report.write(f"median_absolute_delta: {median_abs_delta:.6f}\n")
        report.write(f"positive_fraction: {positive_fraction:.6f}\n")
        report.write(f"within_water_selectivity_score_mean: {selectivity_score:.6f}\n")
        report.write(f"within_water_selectivity_score_median: {median_selectivity_score:.6f}\n")
        report.write(f"all_alive_mean_delta: {all_mean_delta:.6f}\n")
        report.write(f"all_alive_median_delta: {all_median_delta:.6f}\n")
        report.write(f"nonwater_mean_delta: {nonwater_mean_delta:.6f}\n")
        report.write(f"nonwater_median_delta: {nonwater_median_delta:.6f}\n")
        report.write(f"nonwater_mad_delta: {nonwater_mad_delta:.6f}\n")
        report.write(f"nonwater_robust_scale: {nonwater_robust_scale:.6f}\n")
        report.write(f"water_vs_nonwater_delta: {water_vs_nonwater_delta:.6f}\n")
        report.write(f"water_vs_nonwater_median_delta: {water_vs_nonwater_median_delta:.6f}\n")
        report.write(f"water_vs_nonwater_score_mean: {water_vs_nonwater_score:.6f}\n")
        report.write(f"water_vs_nonwater_score_median: {water_vs_nonwater_median_score:.6f}\n")
        report.write(f"primary_water_score: {primary_water_score:.6f}\n")
        report.write(
            "primary_water_score_definition: "
            + ("median_delta_difference (activations normalized per concept)\n"
               if use_normalized_activations
               else "median_delta_difference divided by nonwater robust MAD scale\n")
        )
        report.write("feature_index\tterm\tdelta_mean_act\n")
        for feature_index, term, delta in zip(
            water_feature_indices, matching_terms, water_delta_mean_act.detach().cpu().tolist()
        ):
            report.write(f"{feature_index}\t{term}\t{delta:.6f}\n")

    log.info(f"Water activation summary saved to {report_path}")



@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval_water_structure")
def evaluate_water_structure(cfg: DictConfig):

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ### Loading the Finetuned MS-CLIP Model ~ EVAL ###
    log.info(f"Instantiating MS-CLIP Model at <{cfg.model_path}>")
    try:
        model = ImgModule.load_from_checkpoint(cfg.model_path)
    except (KeyError, RuntimeError):
        with open(Path(cfg.model_path).parent / cfg.config_name, "r") as f:
            model_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        model = ImgModule(model_cfg)
        mis_keys, un_keys = model.load_state_dict(torch.load(cfg.model_path), strict=True)
        log.info("Missing keys:", mis_keys)
        log.info("Unexpected keys:", un_keys)

    # Infer different image sizes
    if cfg.MODEL.out_H != model.model.out_H or cfg.MODEL.out_W != model.model.out_W:
        model.model.out_H = cfg.MODEL.out_H
        model.model.out_W = cfg.MODEL.out_W

    # Set model to evaluation mode
    model.eval()
    model.to(device)

    log.info(f"Instantiating SAE datamodule")
    assert cfg.MODEL.test_max_seq_len is None, "For evaluation, test_max_seq_len must be None to allow adaptive sequence lengths."

    datamodule = get_data(cfg)
    train_dataset = datamodule.test_dataloader(split="train").dataset
    val_dataset = datamodule.test_dataloader(split="val").dataset
    test_datamodule = get_data(cfg)
    test_dataset = test_datamodule.test_dataloader(split="test").dataset
    sae_datamodule = OnlineActDataModule(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                                            test_dataset=test_dataset, device=cfg.device if torch.cuda.is_available() else "cpu",
                                            batch_size=1, test_batch_size=1)

    log.info(f"Instantiating SAE <{cfg.sae._target_}>")
    sae: plSAE = hydra.utils.instantiate(cfg.sae)
    log.info(f"Loading SAE checkpoint from {cfg.sae_ckpt_path}")
    ckpt = torch.load(cfg.sae_ckpt_path, map_location=device)
    sae.to(device)

    if sae.hparams.get("arch_kwargs", None) is not None:
        sae.set_arch(sae.hparams["arch_kwargs"])

    missing, unexpected = sae.load_state_dict(ckpt["state_dict"], strict=True)
    log.info(f"Missing keys: {missing}")
    log.info(f"Unexpected keys: {unexpected}")

    sae.eval()
    dataloader_dict = {
        "train": sae_datamodule.train_dataloader(),
        "val": sae_datamodule.val_dataloader(),
        "test": sae_datamodule.test_dataloader()
    }

    if cfg.alive_feature_path is not None and Path(cfg.alive_feature_path).exists():
        alive_feature = torch.load(cfg.alive_feature_path, map_location=device)
    else:
        alive_feature = None

    vocab_df = pd.read_csv(cfg.vocab_csv_path)

    text_all_sim, text_all_indices = compute_text_dist(sae.net.get_dictionary()[alive_feature].to("cpu"), sae.label_vocab_emb.to("cpu"))
    text_all_sim, text_all_indices = text_all_sim.numpy(), text_all_indices.numpy()

    water_related_terms = ["river", "lake", "ocean", "sea", "water", "stream", "pond", "reservoir", "canal"]
    selected_vocab = vocab_df.iloc[text_all_indices].reset_index(drop=True)
    water_pattern = r"\b(?:" + "|".join(map(re.escape, water_related_terms)) + r")\b"
    water_mask = selected_vocab["term"].astype(str).str.contains(
        water_pattern,
        case=False,
        regex=True,
        na=False,
    )
    matching_terms = selected_vocab.loc[water_mask, "term"].tolist()
    alive_feature_indices = torch.where(alive_feature)[0].cpu().numpy()
    water_feature_indices = alive_feature_indices[water_mask.to_numpy()]

    log.info(f"Water-related terms found in the vocabulary: {matching_terms}")

    sae.to(device)

    mean_act = torch.zeros(sae.net.nb_concepts, device=device)
    fire_mean_act = torch.zeros(sae.net.nb_concepts, device=device)
    nofire_mean_act = torch.zeros(sae.net.nb_concepts, device=device)

    count_patch_fire = 0
    count_patch_nofire = 0

    for batch in tqdm(dataloader_dict[cfg.eval_split], desc="Computing Activations"):
        batch = sae_datamodule.on_after_batch_transfer(batch, 0)
        batch = {k: v.to(device) for k, v in batch.items()}

        labels = batch["labels"]
        with torch.no_grad():
            x = extract_input(batch)
            _, z, _ = sae(x)
            mean_act += z.mean(dim=0)

            fire_mean_act += z[labels == 1].sum(dim=0)
            nofire_mean_act += z[labels == 0].sum(dim=0)

            count_patch_fire += (labels == 1).sum().item()
            count_patch_nofire += (labels == 0).sum().item()

    mean_act = mean_act / len(dataloader_dict[cfg.eval_split].dataset)
    fire_mean_act = fire_mean_act / count_patch_fire
    nofire_mean_act = nofire_mean_act / count_patch_nofire
    delta_mean_act = fire_mean_act - nofire_mean_act

    if cfg.use_normalized_activations:
        std_act = torch.zeros(sae.net.nb_concepts, device=device)
        count_patch_fire = 0
        count_patch_nofire = 0
        for batch in tqdm(dataloader_dict[cfg.eval_split], desc="Computing Normalized Activations"):
            batch = sae_datamodule.on_after_batch_transfer(batch, 0)
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            with torch.no_grad():
                x = extract_input(batch)
                _, z, _ = sae(x)
                diff_act = z - mean_act
                std_act += (diff_act ** 2).sum(dim=0)
                count_patch_fire += (labels == 1).sum().item()
                count_patch_nofire += (labels == 0).sum().item()
        std_act = torch.sqrt(std_act / (count_patch_fire + count_patch_nofire))
        delta_mean_act = delta_mean_act / (std_act + 1e-8)

    summarize_water_activations(
        delta_mean_act=delta_mean_act,
        alive_feature=alive_feature,
        water_feature_indices=water_feature_indices,
        matching_terms=matching_terms,
        output_dir=Path(cfg.get("output_dir", cfg.paths.output_dir)) / "water_analysis",
        eval_split=cfg.eval_split,
        use_normalized_activations=cfg.use_normalized_activations,
    )

if __name__ == "__main__":
    evaluate_water_structure()