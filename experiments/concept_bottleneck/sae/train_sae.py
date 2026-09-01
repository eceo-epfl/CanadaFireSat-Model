import os
from pathlib import Path
from pkgutil import get_data
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger
import yaml
import yaml

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
    cfg.MODEL.test_max_seq_len = None
    test_datamodule = get_data(cfg)
    test_dataset = test_datamodule.test_dataloader(split="test").dataset
    sae_datamodule = OnlineActDataModule(model=model, train_dataset=train_dataset, val_dataset=val_dataset,
                                         test_dataset=test_dataset, device=cfg.device if torch.cuda.is_available() else "cpu",
                                         batch_size=cfg.sae_batch_size, test_batch_size=1)

    log.info(f"Instantiating SAE <{cfg.sae._target_}>")
    sae: plSAE = hydra.utils.instantiate(cfg.sae)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer_sae._target_}>")
    trainer_sae: Trainer = hydra.utils.instantiate(cfg.trainer_sae, callbacks=callbacks, logger=logger)

    if cfg.get("use_archetypical", False):
        sae.set_arch(arch_kwargs=cfg.sae.get("arch_kwargs"))

    """
    elif cfg.get("use_class_init", False):
        X = np.load(cfg.get("train_npy_path"), mmap_mode="r")
        y = np.load(cfg.get("train_label_path"), mmap_mode="r")
        sae.set_init_class(
            X=X, y=y, pos_class_budget=cfg.get("pos_class_budget", 0.5), mode=cfg.get("mode", "kmeans-under")
        )
    """

    # --------------------------------------------------
    #  TEST-ONLY MODE (skip training and activations)
    # --------------------------------------------------
    if cfg.get("test_only", False):
        log.info("Running in TEST ONLY mode...")
        log.info(f"Loading SAE checkpoint from {cfg.sae_ckpt_path}")
        sae = plSAE.load_from_checkpoint(cfg.sae_ckpt_path)
        trainer_sae.test(model=sae, datamodule=sae_datamodule)

    else:
        log.info("Running in TRAINING mode...")
        # Training the SAE
        trainer_sae.fit(model=sae, datamodule=sae_datamodule, ckpt_path=cfg.get("sae_ckpt_path"))

        # Test the SAE
        trainer_sae.test(model=sae, datamodule=sae_datamodule, ckpt_path="best")


@hydra.main(version_base="1.2", config_path=str(CONFIG_PATH), config_name="train_sae.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()