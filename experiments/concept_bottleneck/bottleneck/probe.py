"""
Linear probe with L1 regularisation on top of a frozen, pretrained SAE (plSAE).

Pipeline:  OnlineActDataModule -> {"data": [N, D], "label": [N]}
           -> frozen SAE encoder -> z [N, n_concepts] -> keep alive concepts
           -> nn.Linear(n_alive, num_classes)

Evaluation mirrors ImgModule (same helpers, same metric keys: macro_/micro_/fire_
Accuracy/Precision/Recall/F1/IOU/AUC, val_loss_epoch). Test keys get a "test_" prefix.

Usage
-----
    dm.setup("fit")
    sae = load_frozen_sae(sae_ckpt, device="cuda")
    alive = compute_alive_mask(sae, dm, max_batches=200, device="cuda")
    torch.save(alive.cpu(), "alive_mask.pt")

    probe = SAEProbe(sae_ckpt_path=sae_ckpt, alive_mask_path="alive_mask.pt",
                     num_classes=1, threshold=0.5, l1_lambda=1e-4,
                     loss_config=config)   # same config as ImgModule -> same loss
    trainer.fit(probe, datamodule=dm)
    trainer.test(probe, datamodule=dm)
"""
from typing import Any, Dict, List, Optional, Tuple, Union

from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from deepsat.metrics.numpy_metrics import get_classification_metrics
from experiments.concept_bottleneck.bottleneck.probe_utils import build_probe_loss
from src.eval.utils import get_pr_auc_scores

from src.data.utils import segmentation_ground_truths

from overcomplete.sae.train import extract_input
from experiments.concept_bottleneck.sae.sae_module import plSAE, NAME_CLASS
from experiments.concept_bottleneck.sae.sae_utils import optimizer_factory, scheduler_factory
from experiments.concept_bottleneck.sae.trackers import DeadCodeTracker
from deepsat.metrics.loss_functions import get_loss


# Helpers
def load_frozen_sae(ckpt_path: str, device: Union[str, torch.device] = "cpu") -> Tuple[nn.Module, Optional[torch.Tensor]]:

    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = dict(ckpt["hyper_parameters"])
    hp.pop("_instantiator", None)
    sae = plSAE(**hp).to(device)

    if sae.hparams.get("arch_kwargs", None) is not None:
        print(f"[load_frozen_sae] Setting SAE Dictionary from checkpoint: {sae.hparams['arch_kwargs']}")
        sae.set_arch(sae.hparams["arch_kwargs"])

    missing, unexpected = sae.load_state_dict(ckpt["state_dict"], strict=True)
    if missing or unexpected:
        print(f"[load_frozen_sae] missing: {missing} | unexpected: {unexpected}")

    net = sae.net
    net.to(device).eval()

    for p in net.parameters():
        p.requires_grad_(False)

    return net, sae.label_vocab_emb


@torch.no_grad()
def compute_alive_mask(
    sae: nn.Module,
    datamodule: pl.LightningDataModule,
    max_batches: Optional[int] = None,
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    datamodule.setup("fit")  # moves the backbone to its device (idempotent)
    sae = sae.to(device).eval()
    tracker = DeadCodeTracker(sae.get_dictionary().shape[0], None, device=device)
    loader = datamodule.train_dataloader()
    for i, raw_batch in enumerate(tqdm(loader, desc="Computing alive mask")):
        if max_batches is not None and i >= max_batches:
            break
        transformed = datamodule.on_after_batch_transfer(raw_batch, 0)
        x = extract_input(transformed)
        _, z, _ = sae(x)
        tracker.update(z)
    print(f"Dead concepts: {tracker.get_dead_ratio()}")
    return tracker.alive_features.cpu()


def _load_mask(path: str) -> torch.Tensor:
    if str(path).endswith(".npy"):
        return torch.from_numpy(np.load(path)).bool()
    return torch.as_tensor(torch.load(path, map_location="cpu")).bool()


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
class SAEProbe(pl.LightningModule):
    def __init__(
        self,
        sae_ckpt_path: str,
        alive_mask_path: str,
        num_classes: int,
        loss_config: Dict[str, Any],
        threshold: float = 0.5,
        l1_lambda: float = 1e-4,
        weight_decay: float = 0.0,  # L2; L1 is handled explicitly below
        optimizer_type: str = "adam",
        scheduler_type: Optional[str] = None,
        # name_class: Dict[int, str] = NAME_CLASS,
        sparsity_eps: float = 1e-6,
        out_H: int = 25,
        out_W: int = 25,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        sae_load_device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        # self.name_class = name_class
        self.num_classes = num_classes
        self.threshold = threshold

        # Frozen SAE (encoder only is used)
        self.sae, self.label_vocab_emb = load_frozen_sae(sae_ckpt_path, device=sae_load_device)

        # Alive-concept selection
        alive_mask = _load_mask(alive_mask_path)
        assert alive_mask.numel() == self.sae.nb_concepts, "Alive mask / SAE size mismatch"
        self.register_buffer("alive_idx", torch.where(alive_mask)[0], persistent=True)
        self.n_alive = int(alive_mask.sum().item())

        self.probe = nn.Linear(self.n_alive, num_classes, bias=True)
        nn.init.zeros_(self.probe.weight) # Why this initialization?
        nn.init.zeros_(self.probe.bias) # Why this initialization?

        # Same loss as the baseline when a config is given
        self.loss_fn = build_probe_loss(loss_type=loss_config["loss_type"], num_classes=num_classes,
                                        pos_weight=loss_config.get("pos_weight", None),
                                        ce_weight=loss_config.get("ce_weight", 0.5),
                                        dice_weight=loss_config.get("dice_weight", 0.5),
                                        reduction=loss_config.get("reduction", "mean"))
        self._val_outputs: List[Dict[str, np.ndarray]] = []
        self._test_outputs: List[Dict[str, np.ndarray]] = []

    def _batch_size(self, y):
        return y.numel() // (self.hparams.out_H * self.hparams.out_W)

    # Keep the SAE in eval mode even when Lightning calls .train()
    def train(self, mode: bool = True):
        super().train(mode)
        self.sae.eval()
        return self

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """SAE codes restricted to the alive concepts: [N, n_alive]."""
        _, z, _ = self.sae(x)
        return z[:, self.alive_idx]

    def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Forward pass of the probe."""
        out = self.probe(self.encode(x)) # B*P,C
        out = rearrange(
            out,
            '(b h w) c -> b c h w',
            b=batch_size,
            h=int((out.shape[0] // batch_size) ** 0.5),
            w=int((out.shape[0] // batch_size) ** 0.5),
        )
        out = F.interpolate(
            out, size=(self.hparams.out_H, self.hparams.out_W),
            mode="bilinear", align_corners=False
        )
        return rearrange(out, 'b c h w -> b h w c') # For segmentation tasks, the output is [B, H, W, C] where C is the number of classes.

    def _task_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits.reshape(-1, self.num_classes), y.reshape(-1).long())

    def _loss(self, logits: torch.Tensor, y: torch.Tensor):
        task = self._task_loss(logits, y)
        l1 = self.probe.weight.abs().sum()
        return task + self.hparams.l1_lambda * l1, task, l1

    def _probs_preds(self, logits: torch.Tensor):
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).to(torch.float32)
        else:
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
        return probs, preds

    # ------------------------------------------------------------------ #
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = extract_input(batch) # Same as batch["data"]
        y = segmentation_ground_truths(batch)[0].long()  # Same as batch["label"].long() just aligned with ImgModule
        logits = self.forward(x, self._batch_size(y))
        loss, task, l1 = self._loss(logits, y)
        self.log("train/total_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/task_loss", task, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/l1", l1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        w = self.probe.weight.detach()
        self.log("train/probe_nonzero_frac", (w.abs() > self.hparams.sparsity_eps).float().mean(), prog_bar=True)

    def _eval_step(self, batch: Dict[str, Any], buffer: List[Dict[str, np.ndarray]]):
        x = extract_input(batch) # Same as batch["data"]
        y = segmentation_ground_truths(batch)[0].long()  # Same as batch["label"].long() just aligned with ImgModule
        logits = self.forward(x, self._batch_size(y))
        loss, task, _ = self._loss(logits, y)
        probs, preds = self._probs_preds(logits)
        buffer.append(
            {
                "predictions": preds.view(-1).cpu().numpy(),
                "labels": y.view(-1).cpu().numpy(),
                "probs": probs.view(-1, self.num_classes).cpu().numpy(),
                "loss": loss.detach().view(-1).cpu().numpy(),
                "task": task.detach().view(-1).cpu().numpy(),
            }
        )
        return loss

    def _eval_epoch_end(self, buffer: List[Dict[str, np.ndarray]], name: str):
        preds = np.concatenate([o["predictions"] for o in buffer])
        labels = np.concatenate([o["labels"] for o in buffer])
        probs = np.concatenate([o["probs"] for o in buffer])
        loss = np.array([o["loss"] for o in buffer])
        task = np.concatenate([o["task"] for o in buffer])
        buffer.clear()

        n_classes = self.num_classes + 1 if self.num_classes == 1 else self.num_classes
        eval_metrics = get_classification_metrics(predicted=preds, labels=labels, n_classes=n_classes, unk_masks=None)
        micro_auc, macro_auc, class_auc = get_pr_auc_scores(scores=probs, labels=labels, n_classes=n_classes)

        micro_acc, micro_p, micro_r, micro_f1, micro_iou = eval_metrics["micro"]
        macro_acc, macro_p, macro_r, macro_f1, macro_iou = eval_metrics["macro"]
        c_acc, c_p, c_r, c_f1, c_iou = eval_metrics["class"]

        metrics = {
            f"{name}/total_loss": loss.mean(),
            f"{name}/task_loss": task.mean(),
            f"{name}/macro_Accuracy": macro_acc,
            f"{name}/macro_Precision": macro_p,
            f"{name}/macro_Recall": macro_r,
            f"{name}/macro_F1": macro_f1,
            f"{name}/macro_IOU": macro_iou,
            f"{name}/macro_AUC": macro_auc,
            f"{name}/micro_Accuracy": micro_acc,
            f"{name}/micro_Precision": micro_p,
            f"{name}/micro_Recall": micro_r,
            f"{name}/micro_F1": micro_f1,
            f"{name}/micro_IOU": micro_iou,
            f"{name}/micro_AUC": micro_auc,
            f"{name}/fire_Accuracy": c_acc[1],
            f"{name}/fire_Precision": c_p[1],
            f"{name}/fire_Recall": c_r[1],
            f"{name}/fire_F1": c_f1[1],
            f"{name}/fire_IOU": c_iou[1],
            f"{name}/fire_AUC": class_auc[1],
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        optimizers = self.trainer.optimizers
        if optimizers:
            current_lr = optimizers[0].param_groups[0]["lr"]
            self.log(f"{name}/learning_rate", current_lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, self._val_outputs)

    def on_validation_epoch_end(self):
        self._eval_epoch_end(self._val_outputs, name="val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, self._test_outputs)

    def on_test_epoch_end(self):
        self._eval_epoch_end(self._test_outputs, name="test")

    # ------------------------------------------------------------------ #
    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        # Only the probe is optimised; the SAE is frozen.
        optimizer = optimizer_factory(
            optim_type=self.hparams.optimizer_type,
            params=self.probe.parameters(),
            lr=self.hparams.scheduler_kwargs.get("lr_base", 1e-3),
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = scheduler_factory(scheduler_type=self.hparams.scheduler_type, optimizer=optimizer,
                                         scheduler_kwargs=self.hparams.scheduler_kwargs)
        if lr_scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"}, "monitor": "train/total_loss"}

    """
    @torch.no_grad()
    def selected_concepts(self, top_k: int = 20) -> Dict[str, Any]:
        # Original SAE concept indices ranked by |weight| (per output row).
        w = self.probe.weight.detach().cpu()  # [C, n_alive]
        out = {}
        for c in range(w.shape[0]):
            _, idx = w[c].abs().topk(min(top_k, w.shape[1]))
            key = self.name_class.get(c, str(c)) if self.num_classes > 1 else "fire_logit"
            out[key] = {"concept_idx": self.alive_idx.cpu()[idx].tolist(), "weight": w[c][idx].tolist()}
        return out"""