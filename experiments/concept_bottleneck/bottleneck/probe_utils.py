import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional
from deepsat.metrics.loss_functions import MaskedDiceLoss



def _to_two_class_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Single logit l -> [0, l], since softmax([0, l])[1] == sigmoid(l). Two-class logits pass through."""
    if num_classes == 1:
        return torch.cat([torch.zeros_like(logits), logits], dim=-1)
    return logits


class ProbeDiceLoss(nn.Module):
    """Wrapper around MaskedDiceLoss to handle single-logit case (num_classes=1) by converting to two-class logits.
    """

    def __init__(self, num_classes: int, reduction: str = "mean"):
        super().__init__()
        assert num_classes in (1, 2), "Dice loss (MaskedDiceLoss) is binary only"
        self.num_classes = num_classes
        self.dice = MaskedDiceLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.dice(_to_two_class_logits(logits, self.num_classes), y)


class WeightedCE(nn.Module):
    """Equivalent to weighted cross entropy for model finetuning. Takes a single logit and converts to two-class logits for binary classification.
    """

    def __init__(self, num_classes: int, pos_weight: Optional[float] = None, reduction: str = "mean"):
        super().__init__()
        assert num_classes in (1, 2), "WeightedCE is binary only"
        self.num_classes = num_classes
        w = None
        if pos_weight is not None:
            w = torch.tensor([1.0 / pos_weight, 1.0], dtype=torch.float)
            w = w / w.sum()
        self.register_buffer("weight", w, persistent=False)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = _to_two_class_logits(logits, self.num_classes)
        return F.cross_entropy(logits, y, weight=self.weight, reduction=self.reduction)


class CombinedLoss(nn.Module):
    """ce_weight * WeightedCE + dice_weight * Dice. Individual terms kept in `.components` for logging."""
    def __init__(self, ce: nn.Module, dice: nn.Module, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce, self.dice = ce, dice
        self.ce_weight, self.dice_weight = ce_weight, dice_weight

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ce, dice = self.ce(logits, y), self.dice(logits, y)
        return self.ce_weight * ce + self.dice_weight * dice


# Aim to replicate the loss functions used in model finetuning, which are weighted cross entropy and dice loss. The combined loss is a linear combination of the two.
def build_probe_loss(
    loss_type: str,
    num_classes: int,
    pos_weight: Optional[float] = None,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    reduction: str = "mean",
) -> nn.Module:
    """loss_type in {"wce", "dice", "wce_dice", "baseline"}. Every loss maps (logits [N, K], y [N]) -> scalar."""
    if loss_type == "weight_cross_entropy":
        return WeightedCE(num_classes, pos_weight, reduction=reduction)
    if loss_type == "masked_dice_loss":
        return ProbeDiceLoss(num_classes, reduction=reduction)
    if loss_type == "combined_dice_ce":
        return CombinedLoss(WeightedCE(num_classes, pos_weight, reduction=reduction), ProbeDiceLoss(num_classes, reduction=reduction), ce_weight, dice_weight)
    raise NotImplementedError(f"Unknown loss_type '{loss_type}'")
