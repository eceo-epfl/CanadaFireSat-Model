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


log = get_pylogger(__name__)


def save_alignment_results(
    output_dir: Path,
    similarities: list[list[float]],
) -> None:
    """Save padded per-feature results and rank-wise similarity summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_features = len(similarities)
    max_k = max((len(values) for values in similarities), default=0)

    similarity_array = np.full((n_features, max_k), np.nan, dtype=np.float32)
    for feature_idx, feature_similarities in enumerate(similarities):
        size = len(feature_similarities)
        similarity_array[feature_idx, :size] = feature_similarities

    valid_scores = np.isfinite(similarity_array)
    per_rank_mean = np.full(max_k, np.nan, dtype=np.float32)
    if max_k:
        for rank in range(max_k):
            rank_scores = similarity_array[:, rank][valid_scores[:, rank]]
            if rank_scores.size:
                per_rank_mean[rank] = rank_scores.mean()

    output_path = output_dir / "alignment_results.txt"
    overall_mean = np.nanmean(similarity_array) if valid_scores.any() else np.nan
    with output_path.open("w") as output_file:
        output_file.write(f"nanmean_similarity_all_k: {overall_mean:.6f}\n")
        output_file.write(f"n_features: {n_features}\n\n")
        output_file.write("k\tnanmean_similarity\tn_features_with_image\tfraction_features_with_image\n")
        for rank, mean_similarity in enumerate(per_rank_mean, start=1):
            n_features_with_image = int(valid_scores[:, rank - 1].sum())
            fraction = n_features_with_image / n_features if n_features else np.nan
            output_file.write(
                f"{rank}\t{mean_similarity:.6f}\t{n_features_with_image}\t{fraction:.6f}\n"
            )


def _display_rgb(
    image: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    rgb_channels: tuple[int, int, int] = (2, 1, 0),
) -> np.ndarray:
    """Undo band normalization and convert a satellite image to RGB."""
    if mean is None:
        mean = torch.zeros(image.shape[0])
    if std is None:
        std = torch.ones(image.shape[0])
    reflectance = image * std[:, None, None] + mean[:, None, None]
    rgb = reflectance[list(rgb_channels)].detach().cpu().float().numpy()
    rgb = np.moveaxis(rgb, 0, -1)
    return np.clip(rgb, 0, 1)


@torch.no_grad()
def plot_topk_feature_images(
    sae: LightningModule,
    top_activations: list[list[float]],
    top_sample_ids: list[list[int]],
    alive_feature: torch.Tensor,
    raw_batches: dict[int, dict[str, torch.Tensor]],
    postprocess_batch: Callable[[dict, int], dict],
    output_dir: Path,
    vocab_names: Optional[Sequence[str]] = None,
    device: str | torch.device = "cuda",
    normalization_mean: Optional[torch.Tensor] = None,
    normalization_std: Optional[torch.Tensor] = None,
    plot_feature_indices: Optional[Sequence[int]] = None,
    rgb_channels: tuple[int, int, int] = (2, 1, 0),
) -> None:
    """Plot first, middle, and last frames for each feature's top-k samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sae.eval()
    alive_indices = torch.where(alive_feature)[0].tolist()
    selected_features = set(plot_feature_indices) if plot_feature_indices is not None else None

    for feature_idx, (activations, sample_ids) in enumerate(zip(top_activations, top_sample_ids)):
        if selected_features is not None and alive_indices[feature_idx] not in selected_features:
            continue
        if not sample_ids:
            continue

        feature_name = (
            vocab_names[feature_idx]
            if vocab_names is not None and feature_idx < len(vocab_names)
            else f"feature {alive_indices[feature_idx]}"
        )
        fig, axes = plt.subplots(2, len(sample_ids) * 3, figsize=(4 * len(sample_ids) * 3, 7))
        axes = np.asarray(axes).reshape(2, len(sample_ids) * 3)

        for sample_idx, (activation, batch_id) in enumerate(zip(activations, sample_ids)):
            raw_batch = raw_batches[batch_id]
            inputs = raw_batch["inputs"]
            if inputs.ndim == 5:
                inputs = inputs[0]
            frame_indices = [0, inputs.shape[0] // 2, inputs.shape[0] - 1]

            processed_batch = postprocess_batch(
                {key: value.clone() for key, value in raw_batch.items()}, batch_id
            )
            processed_batch = {key: value.to(device) for key, value in processed_batch.items()}
            x = extract_input(processed_batch)
            _, codes, _ = sae(x)
            feature_codes = codes[..., alive_indices[feature_idx]].reshape(-1)
            side = int(feature_codes.numel() ** 0.5)
            heatmap = feature_codes.reshape(side, side).detach().cpu().numpy()

            for column, frame_idx in enumerate(frame_indices):
                axis = axes[0, sample_idx * 3 + column]
                axis.imshow(_display_rgb(inputs[frame_idx], normalization_mean, normalization_std, rgb_channels))
                axis.axis("off")
                axis.set_title(f"t={frame_idx}", fontsize=9)

                axis = axes[1, sample_idx * 3 + column]
                axis.imshow(_display_rgb(inputs[frame_idx], normalization_mean, normalization_std, rgb_channels))
                axis.imshow(heatmap, cmap="magma", alpha=0.55, interpolation="nearest")
                axis.axis("off")
                if column == 0:
                    axis.set_ylabel(f"activation={activation:.3g}", fontsize=9)

        fig.suptitle(feature_name)
        fig.tight_layout()
        fig.savefig(output_dir / f"feature_{alive_indices[feature_idx]:05d}.png", dpi=150)
        plt.close(fig)


def set_alignment_topK_images(sae: LightningModule, dataloader: DataLoader, topK: int = 5, alive_feature: Optional[torch.Tensor] = None,
                              postprocess_batch: Optional[Callable[[dict, int], dict]] = None, device: str | torch.device = "cuda", use_vocab_emb: bool = True,
                              plot_dir: Optional[Path] = None, normalization_mean: Optional[torch.Tensor] = None, normalization_std: Optional[torch.Tensor] = None,
                              plot_feature_indices: Optional[Sequence[int]] = None,
                              vocab_csv_path: Optional[Path] = None, results_dir: Optional[Path] = None) -> tuple[list[list[float]], list[list[int]], list[list[float]]]:

    sae.eval()
    if alive_feature is None:
        alive_feature = torch.ones(sae.net.get_dictionary().shape[0], dtype=torch.bool, device=device)
    n_features = int(alive_feature.sum())

    vocab_names = None
    _, text_all_indices_no_dead_val = compute_text_dist(sae.net.get_dictionary()[alive_feature].to("cpu"), sae.label_vocab_emb.to("cpu"))
    text_all_indices_no_dead_val = text_all_indices_no_dead_val.numpy()

    if vocab_csv_path is not None:
        vocab_terms = pd.read_csv(vocab_csv_path)["term"].tolist()
        vocab_names = [vocab_terms[int(index)] for index in text_all_indices_no_dead_val]

    if use_vocab_emb:
        # Getting the label embeddings for alive features in order to compute cosine similarity with image features
        target_emb = sae.label_vocab_emb[
            text_all_indices_no_dead_val, :
        ].to(device)

    else:
        target_emb = sae.net.get_dictionary()[alive_feature].to(device)

    # Initializing tensors to store top-k activations, sample IDs, and similarities for each feature
    top_activations = torch.full(
        (n_features, topK), -torch.inf, device=device
    )
    top_sample_ids = torch.full(
        (n_features, topK), -1, dtype=torch.long, device=device
    )
    top_similarities = torch.full(
        (n_features, topK), torch.nan, device=device
    )
    raw_batches: dict[int, dict[str, torch.Tensor]] = {}

    for idx_sample, batch in tqdm(enumerate(dataloader), desc="Computing Activations", total=len(dataloader)):
        # Keep only the current batch temporarily; selected candidates are cached below.
        raw_batch = {
            key: value.detach().cpu().clone()
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        if postprocess_batch:
            batch = postprocess_batch(batch, idx_sample)

        batch = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            x = extract_input(batch)
            _, z, _ = sae(x)
            z_alive = z[..., alive_feature]
            total_activation = z_alive.sum(dim=0)
            sims = _cosine_similarity_matrix(
                x, target_emb
            )

            # Filter out similarities and sum it to only active features and count the total active patches for each feature across the batch
            active = z_alive > 0
            sim_sum = torch.where(
                active, sims, torch.zeros_like(sims)
            ).sum(dim=0)
            active_count = active.sum(dim=0)

            avg_sim = sim_sum / active_count.clamp(min=1)
            avg_sim = avg_sim.masked_fill(active_count == 0, float("nan"))
            total_activation = total_activation.masked_fill(active_count == 0, float("-inf"))

            # Add current sample to the running candidates
            candidates_act = torch.cat(
                [top_activations, total_activation[:, None]], dim=1
            )

            candidates_ids = torch.cat(
                [
                    top_sample_ids,
                    torch.full(
                        (n_features, 1), idx_sample,
                        dtype=torch.long, device=device
                    ),
                ],
                dim=1,
            )

            candidates_sim = torch.cat(
                [top_similarities, avg_sim[:, None]], dim=1
            )

            # Select top-k samples independently for each feature
            top_activations, indices = candidates_act.topk(topK, dim=1)
            top_sample_ids = candidates_ids.gather(1, indices)
            top_similarities = candidates_sim.gather(1, indices)

            selected_ids = set(top_sample_ids[top_sample_ids >= 0].detach().cpu().tolist())
            if idx_sample in selected_ids:
                raw_batches[idx_sample] = raw_batch
            for cached_id in set(raw_batches) - selected_ids:
                raw_batches.pop(cached_id)

    # Remove placeholders for features with fewer than topK valid candidates.
    valid_candidates = torch.isfinite(top_activations) & torch.isfinite(top_similarities)
    activations = [values[mask].detach().cpu().tolist() for values, mask in zip(top_activations, valid_candidates)]
    sample_ids = [values[mask].detach().cpu().tolist() for values, mask in zip(top_sample_ids, valid_candidates)]
    similarities = [values[mask].detach().cpu().tolist() for values, mask in zip(top_similarities, valid_candidates)]

    if results_dir is not None:
        save_alignment_results(
            output_dir=results_dir,
            similarities=similarities,
        )

    if plot_dir is not None and postprocess_batch is not None:
        plot_topk_feature_images(
            sae=sae,
            top_activations=activations,
            top_sample_ids=sample_ids,
            alive_feature=alive_feature,
            raw_batches=raw_batches,
            postprocess_batch=postprocess_batch,
            output_dir=plot_dir,
            vocab_names=vocab_names,
            device=device,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
            plot_feature_indices=plot_feature_indices,
        )

    return activations, sample_ids, similarities

@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval_image_alignment")
def evaluate_image_alignment(cfg: DictConfig):

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
    test_dataset = datamodule.test_dataloader(split="test").dataset
    sae_datamodule = OnlineActDataModule(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                                            test_dataset=test_dataset, device=cfg.device if torch.cuda.is_available() else "cpu",
                                            batch_size=1, test_batch_size=1)
    normalizer = next(
        transform for transform in datamodule.test_transform.transforms
        if hasattr(transform, "mean") and hasattr(transform, "std")
    )
    normalization_mean = torch.as_tensor(normalizer.mean, dtype=torch.float32)
    normalization_std = torch.as_tensor(normalizer.std, dtype=torch.float32)

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

    set_alignment_topK_images(
        sae,
        dataloader_dict[cfg.eval_split],
        topK=cfg.topK,
        alive_feature=alive_feature,
        postprocess_batch=sae_datamodule.on_after_batch_transfer,
        device=device,
        use_vocab_emb=cfg.use_vocab_emb,
        plot_dir=(Path(cfg.output_dir) / f"feature_alignment_vocab_{cfg.use_vocab_emb}_{cfg.eval_split}")
        if cfg.get("plot_topk", True)
        else None,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        plot_feature_indices=cfg.get("plot_feature_indices"),
        vocab_csv_path=(Path(cfg.vocab_csv_path) if cfg.get("vocab_csv_path") else None),
        results_dir=Path(cfg.output_dir) / f"feature_alignment_vocab_{cfg.use_vocab_emb}_{cfg.eval_split}",
    )

if __name__ == "__main__":
    evaluate_image_alignment()