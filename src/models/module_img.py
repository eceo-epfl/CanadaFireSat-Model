"""Script for SITS only lightning module"""

from math import ceil
from typing import Any, Dict, Optional

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn

from deepsat.metrics.loss_functions import get_loss
from deepsat.metrics.numpy_metrics import get_classification_metrics
from deepsat.metrics.torch_metrics import get_binary_metrics, get_mean_metrics
from deepsat.models.TSViT.TSViTdense import TSViTDown
from deepsat.utils.lr_scheduler import build_scheduler_pytorch
from experiments.concept_bottleneck.models.mscliptemporal import MSClipTemporalCBM
from src.data.utils import segmentation_ground_truths
from src.eval.utils import get_pr_auc_scores
from src.models.convlstm import ConvLSTMNet
from src.models.resnet import ResNetConvLSTM
from src.models.vit import ViTFactorizeModel, ViTModel
from src.utils.torch_utils import get_alpha, get_trainable_params


class ImgModule(LightningModule):
    """Lightning Module for SITS only training."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model_type = config["MODEL"]["architecture"]
        self.model = self.model_factory(config["MODEL"])
        self.multi_head = config["MODEL"]["multi_head"] if "multi_head" in config["MODEL"] else False
        self.loss_input_fn = segmentation_ground_truths
        self.loss_fn = {"mean": get_loss(config, "cuda" if torch.cuda.is_available() else "cpu", reduction="mean")}

        if self.multi_head:
            self.loss_fn["class"] = nn.CrossEntropyLoss()

        self.train_metrics_steps = config["CHECKPOINT"]["train_metrics_steps"]
        self.num_classes = config["MODEL"]["num_classes"]
        self.threshold = config["MODEL"]["threshold"]
        self.lr = config["SOLVER"]["lr_base"]
        self.lr_ratio = config["SOLVER"]["lr_ratio"] if "lr_ratio" in config["SOLVER"] else 1.0
        self.lr_mode = config["SOLVER"]["lr_mode"] if "lr_mode" in config["SOLVER"] else "full"
        self.weight_decay = config["SOLVER"]["weight_decay"]
        self.scheduler_config = {"SOLVER": config["SOLVER"]}
        self.num_steps_train = ceil(config["SOLVER"]["num_steps_train"] / config["SOLVER"]["accumulate_grad_batches"])
        self.interval = config["SOLVER"]["interval"] if "interval" in config["SOLVER"] else "step"
        self.save_hyperparameters()
        self.validation_step_outputs = []

    @staticmethod
    def model_factory(model_config: Dict[str, Any]) -> nn.Module:
        """Callable function for model initialization"""
        if model_config["architecture"] == "TSViT":
            return TSViTDown(model_config)
        if model_config["architecture"] == "ConvLSTM":
            return ConvLSTMNet(**model_config)
        if model_config["architecture"] == "ResNet":
            return ResNetConvLSTM(**model_config)
        if model_config["architecture"] == "ViT":
            return ViTModel(**model_config)
        if model_config["architecture"] == "ViTFacto":
            return ViTFactorizeModel(**model_config)
        if model_config["architecture"] == "MSClipFacto":
            return MSClipTemporalCBM(**model_config)
        raise NameError(f"Model architecture {model_config['architecture']} not found")

    def forward(self, x: torch.Tensor, doy: Optional[torch.Tensor] = None, seq_len: Optional[int] = None) -> torch.Tensor:
        """Forward call for the module"""
        return self.model(x, doy=doy, seq_len=seq_len)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Step for training"""

        ground_truth = self.loss_input_fn(batch)
        loss = 0

        if self.multi_head:
            outputs, class_outputs = self.model(batch["inputs"], doy=batch["doy"], seq_len=batch["seq_lengths"])
            class_target = ground_truth[0].long().amax(dim=(1, 2)).reshape(-1)
            class_loss = self.loss_fn["class"](class_outputs, class_target)
            alpha = get_alpha(self.current_epoch)
            loss += alpha * class_loss
            seg_weight = 1 - alpha
            self.log("train_class_loss", class_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log("alpha", alpha, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            outputs = self.model(batch["inputs"], doy=batch["doy"], seq_len=batch["seq_lengths"])
            seg_weight = 1

        outputs = outputs.permute(0, 2, 3, 1)
        seg_loss = self.loss_fn["mean"](outputs.reshape(-1, self.num_classes), ground_truth[0].reshape(-1).long())
        self.log("train_seg_loss", seg_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        loss += seg_weight * seg_loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        logits = outputs
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

        return {"loss": loss, "logits": outputs, "ground_truths": ground_truth}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Step for validation"""

        ground_truth = self.loss_input_fn(batch)
        loss = 0

        if self.multi_head:
            logits, class_logits = self.model(batch["inputs"], doy=batch["doy"], seq_len=batch["seq_lengths"])
            class_target = ground_truth[0].long().amax(dim=(1, 2)).reshape(-1)
            class_loss = self.loss_fn["class"](class_logits, class_target)
            alpha = get_alpha(self.current_epoch)
            loss += alpha * class_loss
            seg_weight = 1 - alpha
        else:
            logits = self.model(batch["inputs"], doy=batch["doy"], seq_len=batch["seq_lengths"])
            seg_weight = 1
            class_loss = torch.tensor(0)

        logits = logits.permute(0, 2, 3, 1)

        if self.num_classes == 1:
            probs = torch.nn.functional.sigmoid(logits)
            predicted = (probs > self.threshold).to(torch.float32)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, predicted = torch.max(logits.data, -1)

        seg_loss = self.loss_fn["mean"](logits.reshape(-1, self.num_classes), ground_truth[0].reshape(-1).long())
        loss += seg_weight * seg_loss
        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        labels, unk_masks = ground_truth
        if unk_masks is not None:
            preds = predicted.view(-1)[unk_masks.view(-1)].cpu().numpy()
            probs = probs.view(-1, self.num_classes)[unk_masks.view(-1)].cpu().numpy()
            labels = labels.view(-1)[unk_masks.view(-1)].cpu().numpy()
        else:
            preds = predicted.view(-1).cpu().numpy()
            probs = probs.view(-1, self.num_classes).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()

        loss = loss.view(-1).cpu().detach().numpy()
        class_loss = class_loss.view(-1).cpu().detach().numpy()
        self.validation_step_outputs.append(
            {"predictions": preds, "labels": labels, "loss": loss, "probs": probs, "class_loss": class_loss}
        )
        return {"predictions": preds, "labels": labels, "loss": loss, "probs": probs}

    def on_validation_epoch_end(self):
        """Epoch end evaluation for validation"""
        outputs = self.validation_step_outputs
        predicted_classes = np.concatenate([out["predictions"] for out in outputs])
        target_classes = np.concatenate([out["labels"] for out in outputs])
        probs_classes = np.concatenate([out["probs"] for out in outputs])

        # losses = np.concatenate([out["losses"] for out in outputs])
        loss = np.array([out["loss"] for out in outputs])
        class_loss = np.array([out["class_loss"] for out in outputs])
        print(predicted_classes.shape, target_classes.shape, probs_classes.shape, loss.shape)
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

        if class_loss.mean() > 0:
            val_metrics["class_loss"] = class_loss.mean()

        self.log_dict(val_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Function to set-up the optimizer and scheduler"""
        trainable_params = get_trainable_params(
            self.model, model_type=self.model_type, lr=self.lr, lr_ratio=self.lr_ratio, mode=self.lr_mode,
            weight_decay=self.weight_decay
        )
        optimizer = torch.optim.AdamW(trainable_params)
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
