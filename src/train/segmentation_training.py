"""Script for launching PL training."""
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from deepsat.utils.config_files_utils import copy_yaml
from deepsat.utils.torch_utils import get_device
from src.constants import CONFIG_PATH
from src.data import get_data
from src.data.Canada.callback import FWICallback, WeightLossCallback, SwitchAllCallback
from src.models import get_model
from src.utils.torch_utils import load_from_checkpoint

sys.path.insert(0, os.getcwd())


@hydra.main(version_base=None, config_path=str(CONFIG_PATH))
def train_and_evaluate(cfg: DictConfig):
    """Training and Evaluation (Val) Endpoint"""

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Extract key variables from the config
    num_epochs = cfg["SOLVER"]["num_epochs"]
    save_steps = cfg["CHECKPOINT"]["save_steps"]
    save_path = cfg["CHECKPOINT"]["save_path"]
    save_path = Path(save_path) / cfg["CHECKPOINT"]["experiment_name"]
    cfg["CHECKPOINT"]["save_path"] = str(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint = cfg["CHECKPOINT"]["load_from_checkpoint"]
    local_device_ids = cfg["SET-UP"]["local_device_ids"]  # List of integers ids for GPUs
    arch_name = cfg["MODEL"]["architecture"]
    device = get_device(local_device_ids, allow_cpu=False)
    seed_everything(cfg["SET-UP"]["seed"])

    datamodule = get_data(cfg)
    cfg["SOLVER"]["num_steps_train"] = len(datamodule.train_dataloader())

    # Initialize model & Load weights
    net = get_model(cfg, device)
    if checkpoint:
        load_from_checkpoint(net, checkpoint)

    # Set-up model checkpoint & callbacks
    checkpoint_callback_IoU = ModelCheckpoint(
        monitor="fire_F1",  # Not sure name macro/IOU
        dirpath=save_path,
        filename=f"{arch_name}-" + "{epoch:02d}-f1-{fire_F1:.2f}",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    """
    checkpoint_callback_step = ModelCheckpoint(
        dirpath=save_path,
        filename=f"{arch_name}-" + "{epoch:02d}-step-{step:.2f}",
        save_top_k=-1,
        every_n_train_steps=save_steps,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback_IoU, checkpoint_callback_step]"""

    callbacks = [checkpoint_callback_IoU]
    reload_dataloaders_every_n_epochs = 0
    if "pos_epochs" in cfg["DATASETS"]["train"]:
        switch_callback = SwitchAllCallback(config=cfg)
        callbacks.append(switch_callback)
        reload_dataloaders_every_n_epochs = 1

    if "fwi_ths" in cfg["DATASETS"]["train"]:
        fwi_callback = FWICallback(config=cfg)
        callbacks.append(fwi_callback)
        reload_dataloaders_every_n_epochs = 1

    if "weights" in cfg["SOLVER"]:
        loss_callback = WeightLossCallback(config=cfg)
        callbacks.append(loss_callback)

    # Set-up Wandb logger
    wandb_logger = WandbLogger(
        project=cfg["CHECKPOINT"]["wandb_project"],
        entity=cfg["CHECKPOINT"]["wandb_user"],
        name=cfg["CHECKPOINT"]["experiment_name"],
    )

    # Copy the config file to the save_path and wandb
    copy_yaml(cfg)
    wandb_logger.log_hyperparams(cfg)

    # Set-up Trainer
    trainer = Trainer(
        max_epochs=num_epochs,
        devices=local_device_ids,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        accelerator="gpu",
        # strategy="ddp",
        accumulate_grad_batches=cfg["SOLVER"]["accumulate_grad_batches"],
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
    )

    # Start training
    trainer.fit(model=net, datamodule=datamodule)


if __name__ == "__main__":
    train_and_evaluate()
