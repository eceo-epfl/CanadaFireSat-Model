"""Utility functions for Training SAE"""

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from einops import rearrange
from overcomplete.metrics import l1, r2_score
from overcomplete.sae import SAE
from overcomplete.sae.losses import _mse_with_penalty


def mse_criterion(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre_codes: torch.Tensor,
    codes: torch.Tensor,
    dictionary: Any,
    aggregate_batch: bool = True,
) -> torch.Tensor:
    if not aggregate_batch:
        mse = (x - x_hat).square().mean(dim=1)
    else:
        mse = (x - x_hat).square().mean()
    return mse


def mse_dyn_th_criterion(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre_codes: torch.Tensor,
    codes: torch.Tensor,
    dictionary: Any,
    sae: SAE,
    desired_sparsity: float = 0.1,
) -> torch.Tensor:
    # here we directly use the thresholds of the model to control the sparsity
    loss = mse_criterion(x, x_hat, pre_codes, codes, dictionary)
    sparsity = (codes > 0).float().mean().detach()
    if sparsity > desired_sparsity:
        # if we are not sparse enough, increase the thresholds levels
        loss -= sae.thresholds.sum()
    return loss


def mse_reanim_criterion(
    x: torch.Tensor, x_hat: torch.Tensor, pre_codes: torch.Tensor, codes: torch.Tensor, dictionary: Any
) -> torch.Tensor:
    loss = mse_criterion(x, x_hat, pre_codes, codes, dictionary)
    # is dead of shape (k) (nb concepts) and is 1 iif
    # not a single code has fire in the batch
    is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
    # we push the pre_codes (before relu) towards the positive orthant
    reanim_loss = (pre_codes * is_dead[None, :]).mean()

    loss -= reanim_loss * 1e-3
    return loss


def criterion_factory(
    loss_type: str, **criterion_kwargs
) -> Callable[[torch.Tensor, torch.Tensor, Any, Any, Any], torch.Tensor]:
    if loss_type == "mse":
        return partial(mse_criterion, **criterion_kwargs)
    elif loss_type == "mse_th":
        return partial(mse_dyn_th_criterion, **criterion_kwargs)
    elif loss_type == "mse_auxk":
        return partial(top_k_auxiliary_loss, **criterion_kwargs)
    elif loss_type == "mse_l1":
        return partial(_mse_with_penalty, penalty_fn=partial(l1, dims=1), **criterion_kwargs)
    elif loss_type == "mse_l1_ghost":
        return partial(ghost_grad_loss, **criterion_kwargs)
    else:
        raise NotImplementedError


def optimizer_factory(optim_type: str, **kwargs) -> torch.optim.Optimizer:

    if optim_type == "adam":
        return torch.optim.Adam(**kwargs)
    else:
        raise NotImplementedError


def scheduler_factory(
    scheduler_type: str, optimizer: torch.optim.Optimizer
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if scheduler_type == "lr_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    else:
        return None


def region_mse_criterion(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre_codes: torch.Tensor,
    codes: torch.Tensor,
    dictionary: Any,
    lat: torch.Tensor,
    lat_filter: Tuple[float] = (-40, 60),
    aggrebate_batch: bool = True,
) -> torch.Tensor:

    if len(lat.shape) == 3:
        lat = rearrange(lat, "n h w -> (n h w)")
    elif len(lat.shape) == 2:
        lat = rearrange(lat, "n c -> (n c)")

    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n c w h -> (n w h) c")
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n t c -> (n t) c")
    else:
        x_flatten = x.clone()

    if len(x_hat.shape) == 4:
        x_hat_flatten = rearrange(x_hat, "n c w h -> (n w h) c")
    elif len(x_hat.shape) == 3:
        x_hat_flatten = rearrange(x_hat, "n t c -> (n t) c")
    else:
        x_hat_flatten = x_hat.clone()

    assert x_flatten.shape == x_hat_flatten.shape

    lat_mask = (lat <= lat_filter[0]) | (lat >= lat_filter[1])
    x_flatten = x_flatten[lat_mask]
    x_hat_flatten = x_hat_flatten[lat_mask]

    if x_flatten.shape[0] == 0:
        return None

    return mse_criterion(
        x_flatten, x_hat_flatten, pre_codes[lat_mask], codes[lat_mask], dictionary, aggregate_batch=aggrebate_batch
    )


def region_reconstruction_error(
    x: torch.Tensor, x_hat: torch.Tensor, lat: torch.Tensor, lat_filter: Tuple[float] = (-40, 60)
) -> float:

    if len(lat.shape) == 3:
        lat = rearrange(lat, "n h w -> (n h w)")
    elif len(lat.shape) == 2:
        lat = rearrange(lat, "n c -> (n c)")

    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n c w h -> (n w h) c")
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n t c -> (n t) c")
    else:
        x_flatten = x.clone()

    if len(x_hat.shape) == 4:
        x_hat_flatten = rearrange(x_hat, "n c w h -> (n w h) c")
    elif len(x_hat.shape) == 3:
        x_hat_flatten = rearrange(x_hat, "n t c -> (n t) c")
    else:
        x_hat_flatten = x_hat.clone()

    assert x_flatten.shape == x_hat_flatten.shape
    lat_mask = (lat <= lat_filter[0]) | (lat >= lat_filter[1])

    x_flatten = x_flatten[lat_mask]
    x_hat_flatten = x_hat_flatten[lat_mask]

    if x_flatten.shape[0] == 0:
        return None

    r2 = r2_score(x_flatten, x_hat_flatten)

    return r2.item()


def region_mse_bands_per_class(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre_codes: torch.Tensor,
    codes: torch.Tensor,
    dictionary: Any,
    lat: torch.Tensor,
    labels: torch.Tensor,
    band_width: int = 20,
    aggregate_batch: bool = True,
    class_names: List[str] = ["background", "TC", "AR"],
) -> Dict[str, float]:
    """
    Compute MSE for each latitude band of width `band_width` and return as dict.
    """

    # Flatten lat if needed
    if len(lat.shape) == 3:
        lat = rearrange(lat, "n h w -> (n h w)")
    elif len(lat.shape) == 2:
        lat = rearrange(lat, "n c -> (n c)")

    # Flatten labels if needed
    if len(labels.shape) == 3:
        labels = rearrange(labels, "n h w -> (n h w)")
    elif len(labels.shape) == 2:
        labels = rearrange(labels, "n c -> (n c)")

    # Flatten x and x_hat if needed
    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n c w h -> (n w h) c")
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n t c -> (n t) c")
    else:
        x_flatten = x.clone()

    if len(x_hat.shape) == 4:
        x_hat_flatten = rearrange(x_hat, "n c w h -> (n w h) c")
    elif len(x_hat.shape) == 3:
        x_hat_flatten = rearrange(x_hat, "n t c -> (n t) c")
    else:
        x_hat_flatten = x_hat.clone()

    assert x_flatten.shape == x_hat_flatten.shape

    results = {}
    classes = labels.unique()
    for start in range(-90, 90, band_width):
        end = start + band_width
        mask = (lat >= start) & (lat < end)
        if mask.sum() == 0:
            results[f"{start}_{end}"] = None
            for cls in classes:
                results[f"{start}_{end}_{class_names[cls.item()]}"] = None
            continue
        mse = mse_criterion(
            x_flatten[mask],
            x_hat_flatten[mask],
            pre_codes[mask],
            codes[mask],
            dictionary,
            aggregate_batch=aggregate_batch,
        )
        results[f"{start}_{end}"] = mse
        for cls in classes:
            cls_mask = mask & (labels == cls)
            if cls_mask.sum() == 0:
                results[f"{start}_{end}_class_{cls.item()}"] = None
                continue
            mse_cls = mse_criterion(
                x_flatten[cls_mask],
                x_hat_flatten[cls_mask],
                pre_codes[cls_mask],
                codes[cls_mask],
                dictionary,
                aggregate_batch=aggregate_batch,
            )
            results[f"{start}_{end}_{class_names[cls.item()]}"] = mse_cls

    for cls in classes:
        cls_mask = labels == cls
        if cls_mask.sum() == 0:
            results[f"class_{class_names[cls.item()]}"] = None
            continue
        mse_cls = mse_criterion(
            x_flatten[cls_mask],
            x_hat_flatten[cls_mask],
            pre_codes[cls_mask],
            codes[cls_mask],
            dictionary,
            aggregate_batch=aggregate_batch,
        )
        results[f"class_{class_names[cls.item()]}"] = mse_cls

    return results


def region_r2_bands_per_class(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    lat: torch.Tensor,
    labels: torch.Tensor,
    band_width: int = 20,
    class_names: List[str] = ["background", "AR", "TC"],  # ["no_fire", "fire"]
) -> Dict[str, float]:
    """
    Compute R2 for each latitude band of width `band_width` and return as dict.
    """
    # Flatten lat if needed
    if len(lat.shape) == 3:
        lat = rearrange(lat, "n h w -> (n h w)")
    elif len(lat.shape) == 2:
        lat = rearrange(lat, "n c -> (n c)")

    # Flatten labels if needed
    if len(labels.shape) == 3:
        labels = rearrange(labels, "n h w -> (n h w)")
    elif len(labels.shape) == 2:
        labels = rearrange(labels, "n c -> (n c)")

    # Flatten x and x_hat if needed
    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n c w h -> (n w h) c")
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, "n t c -> (n t) c")
    else:
        x_flatten = x.clone()

    if len(x_hat.shape) == 4:
        x_hat_flatten = rearrange(x_hat, "n c w h -> (n w h) c")
    elif len(x_hat.shape) == 3:
        x_hat_flatten = rearrange(x_hat, "n t c -> (n t) c")
    else:
        x_hat_flatten = x_hat.clone()

    assert x_flatten.shape == x_hat_flatten.shape

    results = {}
    classes = labels.unique()
    for start in range(-90, 90, band_width):
        end = start + band_width
        mask = (lat >= start) & (lat < end)
        if mask.sum() == 0:
            results[f"{start}_{end}"] = None
            for cls in classes:
                results[f"{start}_{end}_{class_names[cls.item()]}"] = None
            continue
        r2 = r2_score(x_flatten[mask], x_hat_flatten[mask])
        results[f"{start}_{end}"] = r2.item()
        for cls in classes:
            cls_mask = mask & (labels == cls)
            if cls_mask.sum() == 0:
                results[f"{start}_{end}_{class_names[cls.item()]}"] = None
                continue
            r2_cls = r2_score(x_flatten[cls_mask], x_hat_flatten[cls_mask])
            results[f"{start}_{end}_{class_names[cls.item()]}"] = r2_cls.item()

    for cls in classes:
        cls_mask = labels == cls
        if cls_mask.sum() == 0:
            results[f"class_{class_names[cls.item()]}"] = None
            continue
        r2_cls = r2_score(x_flatten[cls_mask], x_hat_flatten[cls_mask])
        results[f"class_{class_names[cls.item()]}"] = r2_cls.item()

    return results


def top_k_auxiliary_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre_codes: torch.Tensor,
    codes: torch.Tensor,
    dictionary: torch.Tensor,
    penalty: float = 0.1,
    l1_penalty: float = 0.0,
    scale: Optional[torch.Tensor] = None,
    shift: Optional[torch.Tensor] = None,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    The Top-K Auxiliary loss (AuxK).

    The loss is defined in the original Top-K SAE paper:
        "Scaling and evaluating sparse autoencoders"
        by Gao et al. (2024).

    Similar to Ghost-grads, it consist in trying to "revive" the dead codes
    by trying the predict the residual using the 50% of the top non choosen codes.

    Loss = ||x - x_hat||^2 + penalty * ||x - (x_hat D * top_half(z_pre - z)||^2

    @tfel the order actually matter here! residual is x - x_hat and
    should be in this specific order.
    """
    # select the 50% of non choosen codes and predict the residual
    # using those non choosen codes
    # the code choosen are the non-zero element of codes

    residual = x - x_hat
    mse = residual.square().mean()

    if l1_penalty > 0.0:
        l1_loss = l1_penalty * torch.mean(l1(codes, dims=1))
        mse += l1_loss

    pre_codes = torch.relu(pre_codes)
    pre_codes = pre_codes - codes  # removing the choosen codes

    auxiliary_topk = torch.topk(pre_codes, k=pre_codes.shape[1] // 2, dim=1)
    pre_codes = torch.zeros_like(codes).scatter(-1, auxiliary_topk.indices, auxiliary_topk.values)

    residual_hat = pre_codes @ dictionary
    if scale is not None and shift is not None:
        residual_hat = (residual_hat - shift) / (scale + 1e-8)
    if mean is not None and std is not None:
        residual_hat = residual_hat * (std + 1e-5) + mean  # TODO: Rebuttal where is it used ? For LOAN
    auxilary_mse = (residual - residual_hat).square().mean()

    loss = mse + penalty * auxilary_mse
    return loss


def ghost_grad_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    pre_codes: torch.Tensor,
    codes: torch.Tensor,
    dictionary: torch.Tensor,
    dead_neuron_mask: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    shift: Optional[torch.Tensor] = None,
    penalty: float = 0.1,
    ghost_penalty: float = 1.0,
) -> torch.Tensor:

    residual = x - x_hat
    mse = residual.square()
    l1_loss = penalty * torch.mean(l1(codes, dims=1))

    if dead_neuron_mask is not None and dead_neuron_mask.sum() > 0:

        l2_norm_residual = torch.norm(residual, dim=-1)
        feature_acts_dead_neurons_only = torch.exp(pre_codes[:, dead_neuron_mask])
        ghost_out = feature_acts_dead_neurons_only @ dictionary[dead_neuron_mask, :]

        if scale is not None and shift is not None:
            ghost_out = (ghost_out - shift) / (scale + 1e-8)

        l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
        norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
        ghost_out = ghost_out * norm_scaling_factor[:, None].detach()
        mse_loss_ghost_resid = (
            torch.pow((ghost_out - residual.detach().float()), 2)
            / (residual.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
        )

        mse_rescaling_factor = (mse / (mse_loss_ghost_resid + 1e-6)).detach()
        mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid
        mse_loss_ghost_resid = ghost_penalty * mse_loss_ghost_resid.mean()

    else:
        mse_loss_ghost_resid = torch.tensor(0.0, dtype=mse.dtype, device=x.device)

    mse = mse.mean()
    loss = mse + l1_loss + mse_loss_ghost_resid
    return loss