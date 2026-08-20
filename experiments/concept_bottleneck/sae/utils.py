from typing import Optional

import numpy as np
import torch
from imblearn.under_sampling import RandomUnderSampler
import logging
from pytorch_lightning.utilities import rank_zero_only
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only

def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger

log = get_pylogger(__name__)

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def points_ext(ext_type: str, X: np.ndarray, y: Optional[np.ndarray], **kwargs) -> torch.Tensor:

    assert len(X.shape) == 2, f"The input features has shape {X.shape}"

    if ext_type == "all":
        return torch.from_numpy(X)
    elif ext_type == "under":
        N_pos = np.sum(y)
        rus = RandomUnderSampler(
            random_state=kwargs.get("seed"), sampling_strategy={0: int(kwargs.get("ratio") * N_pos), 1: int(N_pos)}
        )  # TODO: extend to better method check LIME sampling
        X_train, _ = rus.fit_resample(X, y)
        return torch.from_numpy(X_train)
    elif ext_type == "kmeans":
        kmean = KMeans(n_clusters=kwargs.get("n_clusters"))
        kmean.fit(X)
        clusters = kmean.cluster_centers_
        return torch.from_numpy(clusters)
    elif ext_type == "kmeans-under":
        print("Under-sampling before KMeans")
        N_pos = np.sum(y)
        rus = RandomUnderSampler(
            random_state=kwargs.get("seed"), sampling_strategy={0: int(kwargs.get("ratio") * N_pos), 1: int(N_pos)}
        )  # TODO: extend to better method check LIME sampling
        X_train, _ = rus.fit_resample(X, y)
        kmean = MiniBatchKMeans(
            n_clusters=kwargs.get("n_clusters"),
            random_state=kwargs.get("seed"),
            batch_size=20480,
            verbose=2,
            max_iter=300,
        )
        kmean.fit(X_train)
        clusters = kmean.cluster_centers_
        return torch.from_numpy(clusters)
    else:
        raise NotImplementedError(f"The points extraction type: {ext_type}, is not implemented")
