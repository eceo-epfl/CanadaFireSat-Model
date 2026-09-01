import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
import torch.nn.functional as F

from msclip.inference.utils import build_model

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


def points_vocab(npy_path: Optional[os.PathLike] = None, csv_path: Optional[os.PathLike] = None,
                 vocab_size: Optional[int] = None, text_enc_kwargs: Optional[Dict[str, Any]] = {},
                 device: Optional[str] = "cuda", text_batch_size: int = 128,  **kwargs) -> torch.Tensor:

    if npy_path is not None:
        dict_emb = np.load(npy_path)
        if vocab_size is not None:
            dict_emb = dict_emb[:vocab_size]
        dict_emb = torch.from_numpy(dict_emb)
    elif csv_path is not None:

        dict_atom = pd.read_csv(csv_path).sort_values("frequency", ascending=False)["concept"].tolist()
        dict_atom = dict_atom[:vocab_size] if vocab_size is not None else dict_atom
        msclip_model, _, tokenizer = build_model(
                device=device, **text_enc_kwargs
        )
        msclip_model.to(device).eval()

        @torch.no_grad()
        def batch_encode_text(texts: List[str], batch_size: int) -> torch.Tensor:
            embs = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
                batch = texts[i:i + batch_size]
                toks = tokenizer(batch).to(msclip_model.device)
                e = msclip_model.inference_text(toks)
                # e = F.normalize(e, dim=-1) Attention: Normalization is not needed
                embs.append(e.cpu())
            return torch.cat(embs, dim=0)  # [N, D]

        dict_emb = batch_encode_text(dict_atom, text_batch_size)  # [P, D]
    else:
        raise ValueError("Either npy_path or csv_path must be provided.")

    return dict_emb
