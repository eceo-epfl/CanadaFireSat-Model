import os
from pathlib import Path
from pkgutil import get_data
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger
import yaml
import yaml

from experiments.concept_bottleneck.bottleneck.probe import compute_alive_mask, load_frozen_sae, SAEProbe
from experiments.concept_bottleneck.sae.act_datamodule import OnlineActDataModule
from experiments.concept_bottleneck.sae.sae_module import plSAE
from experiments.concept_bottleneck.sae.utils import get_pylogger, instantiate_callbacks, instantiate_loggers
from src.data import get_data
from src.models.module_img import ImgModule
from src.constants import CONFIG_PATH

log = get_pylogger(__name__)


def train(cfg: DictConfig):
    """Training Script for SAE training [with black-box model inference prior]

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """

    if not HydraConfig.initialized():
        HydraConfig.instance().clear()
        HydraConfig().set_config(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
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

    # Infer different image sizes
    if cfg.MODEL.out_H != model.model.out_H or cfg.MODEL.out_W != model.model.out_W:
        model.model.out_H = cfg.MODEL.out_H
        model.model.out_W = cfg.MODEL.out_W

    # Set model to evaluation mode
    model.eval()
    model.to(device)

    ### Instantiate the SAE Datamodule via MS-CLIP Model Inference ###
    log.info(f"Instantiating SAE datamodule")
    datamodule = get_data(cfg)
    train_dataset = datamodule.test_dataloader(split="train").dataset
    val_dataset = datamodule.test_dataloader(split="val").dataset

    ### For Test we want batch size of 1 and adaptive sequence length (no padding) ###
    if cfg.test_time_adaptive_seq_len:
        cfg.MODEL.test_max_seq_len = None
    test_datamodule = get_data(cfg)
    test_dataset = test_datamodule.test_dataloader(split="test").dataset
    probe_datamodule = OnlineActDataModule(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                                         test_dataset=test_dataset, device=cfg.device if torch.cuda.is_available() else "cpu",
                                         batch_size=cfg.probe_batch_size, test_batch_size=1, flag_downsample=False)

    log.info(f"Loading Pretrained SAE at <{cfg.probe.sae_ckpt_path}>")
    sae, _ = load_frozen_sae(cfg.probe.sae_ckpt_path, device=device)

    log.info("Computing alive mask for the SAE...")
    if not Path(cfg.probe.alive_mask_path).exists():
        alive = compute_alive_mask(sae, probe_datamodule, device=device)
        torch.save(alive.cpu(), Path(cfg.probe.alive_mask_path))

    log.info(f"Instantiating Probe Module <{cfg.probe._target_}>")
    probe: SAEProbe = hydra.utils.instantiate(cfg.probe)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer_probe._target_}>")
    trainer_probe: Trainer = hydra.utils.instantiate(cfg.trainer_probe, callbacks=callbacks, logger=logger)

    #  TEST-ONLY MODE (skip training and activations)
    if cfg.get("test_only", False):
        raise NotImplementedError("Test-only mode is not implemented for probe training. Please set test_only to False.")

    else:
        log.info("Running in TRAINING mode...")
        # Training the SAE
        trainer_probe.fit(model=probe, datamodule=probe_datamodule)

        # Rebuild validation data after training with adaptive sequence lengths.
        probe_datamodule.val_dataset = test_datamodule.test_dataloader(split="val").dataset
        probe_datamodule.val_batch_size = 1

        # Evaluate the best probe checkpoint on the adaptive validation split.
        trainer_probe.validate(model=probe, datamodule=probe_datamodule, ckpt_path="best")

        # Test the probe.
        trainer_probe.test(model=probe, datamodule=probe_datamodule, ckpt_path="best")

        rerun_label_dir = cfg.get("rerun_label_dir")
        if rerun_label_dir:
            log.info(f"Rerunning validation and test with labels from <{rerun_label_dir}>")

            rerun_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            rerun_cfg.DATASETS.eval.label_dir = rerun_label_dir
            rerun_cfg.DATASETS.kwargs.use_label_dir_neg = False # In toy labels, we want to use the new label dir for both positive and negative samples. But not for the original label dir.
            rerun_datamodule = get_data(rerun_cfg)

            ### For Test we want batch size of 1 and adaptive sequence length (no padding) ###
            if rerun_cfg.test_time_adaptive_seq_len:
                rerun_cfg.MODEL.test_max_seq_len = None

            rerun_val_dataset = rerun_datamodule.test_dataloader(split="val").dataset
            rerun_test_dataset = rerun_datamodule.test_dataloader(split="test").dataset

            rerun_probe_datamodule = OnlineActDataModule(
                model=model,
                train_dataset=probe_datamodule.train_dataset,
                val_dataset=rerun_val_dataset,
                test_dataset=rerun_test_dataset,
                device=cfg.device if torch.cuda.is_available() else "cpu",
                batch_size=cfg.probe_batch_size,
                val_batch_size=1,
                test_batch_size=1,
                flag_downsample=False,
            )

            # WandbLogger.name returns the project; the run name lives on the experiment
            base_run_name = logger[0].experiment.name if logger else "probe"

            def evaluate_rerun_split(split: str) -> List[Dict[str, Any]]:
                # WandbLogger reuses any active wandb.run, so close it to start a new run
                wandb.finish()

                rerun_logger_cfg = OmegaConf.create(
                    OmegaConf.to_container(cfg.get("logger"), resolve=True)
                )
                if "wandb" not in rerun_logger_cfg:
                    raise ValueError(
                        "A W&B logger is required for alternate-label evaluation."
                    )

                rerun_logger_cfg.wandb.name = f"{base_run_name}-new-label-{split}"
                rerun_logger_cfg.wandb.id = None
                rerun_logger_cfg.wandb.job_type = f"eval-new-label-{split}"
                rerun_logger = instantiate_loggers(rerun_logger_cfg)
                rerun_trainer: Trainer = hydra.utils.instantiate(
                    cfg.trainer_probe,
                    callbacks=[],
                    logger=rerun_logger,
                    enable_checkpointing=False,
                )
                if split == "val":
                    return rerun_trainer.validate(
                        model=probe,
                        datamodule=rerun_probe_datamodule,
                    )
                return rerun_trainer.test(
                    model=probe,
                    datamodule=rerun_probe_datamodule,
                )

            rerun_val_results = evaluate_rerun_split("val")
            rerun_test_results = evaluate_rerun_split("test")
            log.info(
                f"Alternate-label validation results logged to "
                f"{base_run_name}-new-label-val: {rerun_val_results}"
            )
            log.info(
                f"Alternate-label test results logged to "
                f"{base_run_name}-new-label-test: {rerun_test_results}"
            )
            wandb.finish()


@hydra.main(version_base="1.2", config_path=str(CONFIG_PATH), config_name="train_probe.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()