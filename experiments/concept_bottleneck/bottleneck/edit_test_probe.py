from pathlib import Path
import re
from typing import Any, Dict

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from experiments.concept_bottleneck.eval.utils import load_img_model, feature_terms_from_vocab
from experiments.concept_bottleneck.sae.act_datamodule import OnlineActDataModule
from src.constants import CONFIG_PATH
from src.data import get_data
from experiments.concept_bottleneck.bottleneck.probe import SAEProbe
from experiments.concept_bottleneck.sae.utils import get_pylogger


log = get_pylogger(__name__)

WATER_RELATED_TERMS = ["river", "lake", "ocean", "sea", "water", "stream", "pond", "reservoir", "canal"]


def evaluate_probe(probe: SAEProbe, datamodule: OnlineActDataModule, device: torch.device) -> Dict[str, Any]:
    """Run validation and test as in train_probe.py and return the merged metrics."""
    trainer = pl.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    val_results = trainer.validate(model=probe, datamodule=datamodule)
    test_results = trainer.test(model=probe, datamodule=datamodule)
    return {**val_results[0], **test_results[0]}


@hydra.main(version_base="1.2", config_path=str(CONFIG_PATH), config_name="eval_edit_toy")
def edit_water_test_performance(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = load_img_model(cfg, device)

    ### For Test we want batch size of 1 and adaptive sequence length (no padding) ###
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
        val_batch_size=1,
        test_batch_size=1,
        flag_downsample=False,
    )
    probe_datamodule.load_input_norm(cfg.probe.sae_ckpt_path)

    log.info(f"Loaded probe checkpoint from {cfg.probe_checkpoint_path}")
    probe: SAEProbe = hydra.utils.instantiate(cfg.probe)
    probe_checkpoint = torch.load(cfg.probe_checkpoint_path, map_location=device)
    mis_keys, un_keys = probe.load_state_dict(probe_checkpoint["state_dict"], strict=True)
    if mis_keys:
        log.warning(f"Missing keys in probe checkpoint: {mis_keys}")
    if un_keys:
        log.warning(f"Unexpected keys in probe checkpoint: {un_keys}")
    probe.to(device).eval()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names, _, _ = feature_terms_from_vocab(
        probe, Path(cfg.vocab_csv_path) if cfg.get("vocab_csv_path") else None
    )

    # Water features: alive features whose matched vocab term contains a water-related word
    water_pattern = r"\b(?:" + "|".join(map(re.escape, WATER_RELATED_TERMS)) + r")\b"
    water_mask = torch.tensor(
        pd.Series(feature_names).str.contains(water_pattern, case=False, regex=True, na=False).to_numpy()
    )

    # Features used by the probe with a positive weight on the fire logit
    fire_row = 0 if probe.num_classes == 1 else 1
    fire_weight = probe.probe.weight.detach()[fire_row].cpu()
    positive_mask = fire_weight > probe.hparams.sparsity_eps
    edit_mask = water_mask & positive_mask

    alive_feature_indices = probe.alive_idx.detach().cpu()
    edited_features = pd.DataFrame(
        {
            "alive_position": torch.where(edit_mask)[0].numpy(),
            "sae_feature_index": alive_feature_indices[edit_mask].numpy(),
            "term": [feature_names[i] for i in torch.where(edit_mask)[0].tolist()],
            "original_weight": fire_weight[edit_mask].numpy(),
        }
    ).sort_values("original_weight", ascending=False)
    edited_features.to_csv(output_dir / "water_edited_features.csv", index=False)
    log.info(
        f"{int(water_mask.sum())} water-related features, {int(edit_mask.sum())} used with positive weight:\n"
        f"{edited_features.to_string(index=False)}"
    )

    # Zero out the probe weights of the selected water features on the fire logit
    with torch.no_grad():
        probe.probe.weight[fire_row, edit_mask.to(probe.probe.weight.device)] = 0.0

    log.info("Evaluating the water-edited probe...")
    results = evaluate_probe(probe, probe_datamodule, device)

    results_df = pd.Series(results, name="water_edited").to_frame()
    results_df.to_csv(output_dir / "water_edited_metrics.csv")
    log.info(f"Water-edited probe metrics:\n{results_df.to_string()}")


if __name__ == "__main__":
    edit_water_test_performance()
