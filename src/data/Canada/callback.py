"""Callback classes for specific sampling strategy during training"""

import copy
from typing import Any, Dict

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from deepsat.metrics.loss_functions import get_loss


class SwitchAllCallback(Callback):
    """Callback to change from only using Positive to All samples"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        train_config = config["DATASETS"]["train"]
        self.pos_epochs = train_config["pos_epochs"]

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.pos_epochs - 1:
            trainer.datamodule.target_file_id = None


class FWICallback(Callback):
    """Callback to change the FWI bucket for selecting samples"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.fwi_ths = config["DATASETS"]["train"]["fwi_ths"]

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if (trainer.current_epoch + 1) in self.fwi_ths:
            trainer.datamodule.fwi_th = self.fwi_ths[trainer.current_epoch + 1]


class WeightLossCallback(Callback):
    """Callback to change from classification loss to segmentation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.weights = config["SOLVER"]["weights"]
        self.config = copy.deepcopy(config)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.current_epoch in self.weights:
            self.config["SOLVER"]["pos_weight"] = self.weights[trainer.current_epoch]
            pl_module.loss_fn = {
                "all": get_loss(self.config, "cuda" if torch.cuda.is_available() else "cpu", reduction=None),
                "mean": get_loss(self.config, "cuda" if torch.cuda.is_available() else "cpu", reduction="mean"),
            }
