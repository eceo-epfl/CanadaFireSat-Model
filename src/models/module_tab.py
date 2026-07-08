"""Script for multi-modal and ENV only lightning module"""

from math import ceil
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn

from deepsat.metrics.loss_functions import get_loss
from deepsat.metrics.numpy_metrics import get_classification_metrics
from deepsat.metrics.torch_metrics import get_binary_metrics, get_mean_metrics
from deepsat.utils.lr_scheduler import build_scheduler_pytorch
from src.baseline.seas_unet import UNet
from src.baseline.seas_utae import UTAEWrap
from src.data.utils import segmentation_ground_truths
from src.eval.utils import get_pr_auc_scores
from src.models.convlstm import TabConvLSTMNet
from src.models.multivit import MultiViTFactorizeModel
from src.models.resnet import EnvResNetConvLSTM, MixResNetConvLSTM, TabResNetConvLSTM
from src.models.tabtsvit import TabTSViTDown
from src.models.vit import EnvViTFactorizeModel, TabViTFactorizeModel
from src.utils.torch_utils import get_trainable_params


class TabModule(LightningModule):
    """Lightning Module for multi-modal and ENV only training"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model_type = config["MODEL"]["architecture"]
        self.model = self.model_factory(config["MODEL"])
        self.loss_input_fn = segmentation_ground_truths
        self.loss_fn = {
            "all": get_loss(config, "cuda" if torch.cuda.is_available() else "cpu", reduction=None),
            "mean": get_loss(config, "cuda" if torch.cuda.is_available() else "cpu", reduction="mean"),
        }

        self.train_metrics_steps = config["CHECKPOINT"]["train_metrics_steps"]
        self.num_classes = config["MODEL"]["num_classes"]
        self.threshold = config["MODEL"]["threshold"]
        self.lr = config["SOLVER"]["lr_base"]
        self.lr_ratio = config["SOLVER"]["lr_ratio"] if "lr_ratio" in config["SOLVER"] else 1.0
        self.lr_mode = config["SOLVER"]["lr_mode"] if "lr_mode" in config["SOLVER"] else "full"
        self.weight_decay = config["SOLVER"]["weight_decay"]
        self.scheduler_config = {"SOLVER": config["SOLVER"]}
        # self.params_scheduler = config["SOLVER"]["params_scheduler"] if "params_scheduler" in config["SOLVER"] else {}
        self.num_steps_train = ceil(config["SOLVER"]["num_steps_train"] / config["SOLVER"]["accumulate_grad_batches"])
        self.interval = config["SOLVER"]["interval"] if "interval" in config["SOLVER"] else "step"
        self.save_hyperparameters()
        self.validation_step_outputs = []

    @staticmethod
    def model_factory(model_config: Dict[str, Any]) -> nn.Module:
        """Callable function for model initialization"""
        if model_config["architecture"] == "TabTSViT":
            return TabTSViTDown(model_config)
        if model_config["architecture"] == "TabConvLSTM":
            return TabConvLSTMNet(**model_config)
        if model_config["architecture"] == "MixResNet":
            return MixResNetConvLSTM(**model_config)
        if model_config["architecture"] == "EnvResNet":
            return EnvResNetConvLSTM(**model_config)
        if model_config["architecture"] == "TabResNetConvLSTM":
            return TabResNetConvLSTM(**model_config)
        if model_config["architecture"] == "TabViTFactorizeModel":
            return TabViTFactorizeModel(**model_config)
        if model_config["architecture"] == "MultiViTFactorizeModel":
            return MultiViTFactorizeModel(**model_config)
        if model_config["architecture"] == "EnvViTFactorizeModel":
            return EnvViTFactorizeModel(**model_config)
        elif model_config["architecture"] == "UNet":
            return UNet(**model_config)
        elif model_config["architecture"] == "UTAE":
            return UTAEWrap(**model_config)
        raise NameError(f"Model architecture {model_config['architecture']} not found")

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        xtab: Optional[torch.Tensor] = None,
        masktab: Optional[torch.Tensor] = None,
        xmid: Optional[torch.Tensor] = None,
        xlow: Optional[torch.Tensor] = None,
        m_mid: Optional[torch.Tensor] = None,
        m_low: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward call for the module"""

        if self.model_type in [
            "TabTSViT",
            "TabConvLSTM",
            "TabResNetConvLSTM",
            "TabViTFactorizeModel",
            "MultiViTFactorizeModel",
        ]:
            return self.model(x, xtab, masktab)

        if self.model_type in ["EnvResNet", "EnvViTFactorizeModel", "UNet", "UTAE"]:
            return self.model(xmid, xlow, m_mid, m_low)

        return self.model(x, xmid, xlow, m_mid, m_low)

    def training_step(
        self,
        batch: Union[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:

        if self.model_type in [
            "TabTSViT",
            "TabConvLSTM",
            "TabResNetConvLSTM",
            "TabViTFactorizeModel",
            "MultiViTFactorizeModel",
        ]:
            outputs = self.model(batch[0]["inputs"], batch[1]["tab_inputs"], batch[1]["mask"])
        elif self.model_type in ["EnvResNet", "EnvViTFactorizeModel", "UNet", "UTAE"]:
            outputs = self.model(
                batch["mid_inputs"], batch["low_inputs"], batch["mid_inputs_mask"], batch["low_inputs_mask"]
            )
        else:
            outputs = self.model(
                batch[0]["inputs"],
                batch[1]["mid_inputs"],
                batch[1]["low_inputs"],
                batch[1]["mid_inputs_mask"],
                batch[1]["low_inputs_mask"],
            )

        outputs = outputs.permute(0, 2, 3, 1)
        if isinstance(batch, list):
            ground_truth = self.loss_input_fn(batch[0])
        else:
            ground_truth = self.loss_input_fn(batch)

        # loss = self.loss_fn["mean"](outputs, ground_truth)
        if self.num_classes == 1:
            loss = self.loss_fn["mean"](
                outputs.reshape(-1, self.num_classes), ground_truth[0].reshape(-1, self.num_classes)
            )
        else:
            loss = self.loss_fn["mean"](outputs.reshape(-1, self.num_classes), ground_truth[0].reshape(-1).long())
        logits = outputs

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.global_step % self.train_metrics_steps == 0:
            labels, unk_masks = ground_truth
            logits = logits.permute(0, 3, 1, 2)  # Logits need to be (N, D, H, W)
            if self.num_classes == 1:
                batch_metrics = get_binary_metrics(logits=logits, labels=labels, thresh=self.threshold, name="step_")
            else:
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, n_classes=self.num_classes, unk_masks=unk_masks, name="step_"
                )

            self.log_dict(batch_metrics, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        # self.scheduler.step_update(self.global_step)

        return {"loss": loss, "logits": outputs, "ground_truths": ground_truth}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:

        if self.model_type in [
            "TabTSViT",
            "TabConvLSTM",
            "TabResNetConvLSTM",
            "TabViTFactorizeModel",
            "MultiViTFactorizeModel",
        ]:
            logits = self.model(batch[0]["inputs"], batch[1]["tab_inputs"], batch[1]["mask"])
        elif self.model_type in ["EnvResNet", "EnvViTFactorizeModel", "UNet", "UTAE"]:
            logits = self.model(
                batch["mid_inputs"], batch["low_inputs"], batch["mid_inputs_mask"], batch["low_inputs_mask"]
            )
        else:
            logits = self.model(
                batch[0]["inputs"],
                batch[1]["mid_inputs"],
                batch[1]["low_inputs"],
                batch[1]["mid_inputs_mask"],
                batch[1]["low_inputs_mask"],
            )

        logits = logits.permute(0, 2, 3, 1)
        if isinstance(batch, list):
            ground_truth = self.loss_input_fn(batch[0])
        else:
            ground_truth = self.loss_input_fn(batch)

        if self.num_classes == 1:
            probs = torch.nn.functional.sigmoid(logits)
            predicted = (probs > self.threshold).to(torch.float32)
            loss = self.loss_fn["mean"](
                logits.reshape(-1, self.num_classes), ground_truth[0].reshape(-1, self.num_classes)
            )
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, predicted = torch.max(logits.data, -1)
            loss = self.loss_fn["mean"](logits.reshape(-1, self.num_classes), ground_truth[0].reshape(-1).long())

        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        labels, unk_masks = ground_truth

        if unk_masks is not None:
            probs = probs.view(-1, self.num_classes)[unk_masks.view(-1)].cpu().numpy()
            preds = predicted.view(-1)[unk_masks.view(-1)].cpu().numpy()
            labels = labels.view(-1)[unk_masks.view(-1)].cpu().numpy()
        else:
            probs = probs.view(-1, self.num_classes).cpu().numpy()
            preds = predicted.view(-1).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()
        loss = loss.view(-1).cpu().detach().numpy()
        self.validation_step_outputs.append({"predictions": preds, "labels": labels, "probs": probs, "loss": loss})
        return {"predictions": preds, "labels": labels, "probs": probs, "loss": loss}

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        predicted_classes = np.concatenate([out["predictions"] for out in outputs])
        target_classes = np.concatenate([out["labels"] for out in outputs])
        probs_classes = np.concatenate([out["probs"] for out in outputs])
        loss = np.array([out["loss"] for out in outputs])
        self.validation_step_outputs.clear()

        eval_metrics = get_classification_metrics(
            predicted=predicted_classes,
            labels=target_classes,
            n_classes=self.num_classes + 1 if self.num_classes == 1 else self.num_classes,
            unk_masks=None,
        )

        micro_auc, macro_auc, class_auc = get_pr_auc_scores(
            scores=probs_classes,
            labels=target_classes,
            n_classes=self.num_classes + 1 if self.num_classes == 1 else self.num_classes,
        )

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics["micro"]
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics["macro"]
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics["class"]

        val_metrics = {
            "val_loss_epoch": loss.mean(),
            "macro_Accuracy": macro_acc,
            "macro_Precision": macro_precision,
            "macro_Recall": macro_recall,
            "macro_F1": macro_F1,
            "macro_IOU": macro_IOU,
            "macro_AUC": macro_auc,
            "micro_Accuracy": micro_acc,
            "micro_Precision": micro_precision,
            "micro_Recall": micro_recall,
            "micro_F1": micro_F1,
            "micro_IOU": micro_IOU,
            "micro_AUC": micro_auc,
            "fire_Accuracy": class_acc[1],
            "fire_Precision": class_precision[1],
            "fire_Recall": class_recall[1],
            "fire_F1": class_F1[1],
            "fire_IOU": class_IOU[1],
            "fire_AUC": class_auc[1],
        }
        self.log_dict(val_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        trainable_params = get_trainable_params(
            self.model, model_type=self.model_type, lr=self.lr, lr_ratio=self.lr_ratio, mode=self.lr_mode
        )
        optimizer = torch.optim.AdamW(trainable_params, weight_decay=self.weight_decay)
        scheduler = build_scheduler_pytorch(
            config=self.scheduler_config,
            optimizer=optimizer,
            n_iter_per_epoch=self.num_steps_train,
            interval=self.interval,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.interval,
            },
        }
