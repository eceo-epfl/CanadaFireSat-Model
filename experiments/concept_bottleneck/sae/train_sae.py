import os
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import pyrootutils
import pytorch_lightning as pl
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import utils

from src.data.act_datamodule import GeoNpyActDataModule
from src.model.sae_module import plSAE
from src.train.process_utils import save_activations_to_npy, save_labels_to_npy

log = utils.get_pylogger(__name__)


def train(cfg: DictConfig) -> Dict[Any, Any]:
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

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating SAE <{cfg.sae._target_}>")
    sae: plSAE = hydra.utils.instantiate(cfg.sae)

    log.info(f"Instantiating Backbone <{cfg.model.net._target_}>")
    net: nn.Module = hydra.utils.instantiate(cfg.model.net)

    log.info(f"Loading Trained Model <{cfg.model._target_}> at {cfg.ckpt_path}")
    model: LightningModule = hydra.utils.get_class(cfg.model._target_).load_from_checkpoint(
        cfg.ckpt_path, net=net
    )  # TODO: Should it be nn.Module
    model.set_target_shift(cfg.target_shift)
    model.eval()

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer_sae._target_}>")
    trainer_sae: Trainer = hydra.utils.instantiate(cfg.trainer_sae, callbacks=callbacks, logger=logger)

    # Compute Activations
    if cfg.get("mode_data") == "disk":
        for path, path_label, stage in zip(
            [cfg.get("train_npy_path"), cfg.get("val_npy_path"), cfg.get("test_npy_path")],
            [cfg.get("train_label_path"), cfg.get("val_label_path"), cfg.get("test_label_path")],
            ["train", "val", "test"],
        ):
            datamodule._has_setup = False
            if (path is not None) and (not os.path.exists(path)):
                path = path.replace("-act.npy", "")
                if not datamodule._has_setup:
                    datamodule.setup()  # This step load the WHOLE model in RAM
                    datamodule._has_setup = True

                if stage == "train":
                    dataloader = datamodule.train_dataloader(shuffle=False)
                elif stage == "val":
                    dataloader = datamodule.val_dataloader(shuffle=False)
                else:
                    dataloader = datamodule.test_dataloader()
                print(f"Saving Activations for stage {stage} at {path}")
                save_activations_to_npy(model, dataloader, path)
                if (path_label is not None) and (not os.path.exists(path_label)):
                    save_labels_to_npy(dataloader, path, size=(cfg.get("label_down")[0], cfg.get("label_down")[1]))

        if cfg.get("use_weighted_sampler", False):
            # Create weighted sampler to handle class imbalance
            y_train_resample = np.load(cfg.get("train_label_path"))
            y_train_resample = (
                np.argmax(y_train_resample, axis=1)
                if (len(y_train_resample.shape) > 1) and (y_train_resample.shape[2] > 1)
                else y_train_resample
            )
            pos_weight_factor = cfg.get("pos_weight_factor", 0.1)
            pos_ratio = sum(y_train_resample) / len(y_train_resample)
            pos_weight = (pos_weight_factor * (1 - pos_ratio)) / (pos_ratio * (1 - pos_weight_factor))
            class_weight = np.array([1.0, pos_weight])
        else:
            class_weight = None

        act_datamodule = GeoNpyActDataModule(
            batch_size=cfg.sae_batch_size,
            train_npy_path=cfg.get("train_npy_path"),
            train_loc_path=cfg.get("train_loc_path"),
            train_label_path=cfg.get("train_label_path"),
            val_npy_path=cfg.get("val_npy_path"),
            val_loc_path=cfg.get("val_loc_path"),
            val_label_path=cfg.get("val_label_path"),
            test_npy_path=cfg.get("test_npy_path"),
            test_loc_path=cfg.get("test_loc_path"),
            test_label_path=cfg.get("test_label_path"),
            class_weight=class_weight,
        )

    else:

        raise NotImplementedError("Online Data Pipeline Not Up To Date")

    if cfg.get("use_archetypical", False):
        X = np.load(cfg.get("train_npy_path"), mmap_mode="r")
        y = np.load(cfg.get("train_label_path"), mmap_mode="r")
        sae.set_arch(X=X, y=y, arch_kwargs=cfg.sae.get("arch_kwargs", {"ext_type": "all"}))
    elif cfg.get("use_class_init", False):
        X = np.load(cfg.get("train_npy_path"), mmap_mode="r")
        y = np.load(cfg.get("train_label_path"), mmap_mode="r")
        sae.set_init_class(
            X=X, y=y, pos_class_budget=cfg.get("pos_class_budget", 0.5), mode=cfg.get("mode", "kmeans-under")
        )

    # --------------------------------------------------
    #  TEST-ONLY MODE (skip training and activations)
    # --------------------------------------------------
    if cfg.get("test_only", False):
        log.info("Running in TEST ONLY mode...")
        log.info(f"Loading SAE checkpoint from {cfg.sae_ckpt_path}")
        sae = plSAE.load_from_checkpoint(cfg.sae_ckpt_path)
        trainer_sae.test(model=sae, datamodule=act_datamodule)

    else:
        log.info("Running in TRAINING mode...")
        # Training the SAE
        trainer_sae.fit(model=sae, datamodule=act_datamodule, ckpt_path=cfg.get("sae_ckpt_path"))

        # Test the SAE
        trainer_sae.test(model=sae, datamodule=act_datamodule, ckpt_path="best")


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()