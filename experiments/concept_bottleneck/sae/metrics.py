from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import torch
from torchmetrics import Metric
import torch.nn.functional as F
from esda import Moran
from libpysal.weights import DistanceBand
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

SEAS_AREA = 295871040000000  # In m^2 the area cover by SEASFIRE patches


def _cosine_similarity_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    assert x.shape[1] == y.shape[1], "Input vectors must have the same dimensionality"
    assert len(x.shape) == 2 and len(y.shape) == 2, "Input tensors must be 2D"

    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return x_norm @ y_norm.T


@torch.no_grad()
def compute_stable_rank(codes_dict: torch.Tensor) -> float:
    norm_f = torch.linalg.matrix_norm(codes_dict, ord="fro")
    norm_2 = torch.linalg.matrix_norm(codes_dict, ord=2)
    return (norm_f**2) / (norm_2**2).item()


@torch.no_grad()
def compute_effective_rank(codes_dict: torch.Tensor) -> float:
    sing_v = torch.linalg.svdvals(codes_dict)
    sing_v = sing_v / (sing_v.sum() + 1e-12)
    sing_v = torch.clamp(sing_v, min=1e-12)
    return torch.exp(-torch.sum(sing_v * torch.log(sing_v)))


@torch.no_grad()
def compute_coherence(codes_dict: torch.Tensor) -> float:
    cosine_matrix = _cosine_similarity_matrix(codes_dict, codes_dict).abs()
    cosine_matrix = cosine_matrix.fill_diagonal_(-float("inf"))
    return cosine_matrix.max().item()


@torch.no_grad()
def compute_text_alignment(codes_dict: torch.Tensor, label_vocab_emb: torch.Tensor) -> float:
    cosine_matrix = _cosine_similarity_matrix(codes_dict, label_vocab_emb).abs()
    max_cosine_per_atom, _ = torch.max(cosine_matrix, dim=1)
    return max_cosine_per_atom.mean().item()


@torch.no_grad()
def compute_connect(
    code_activations: torch.Tensor, label: Optional[torch.Tensor] = None
) -> Union[float, Tuple[float, dict]]:

    C = code_activations.T @ code_activations
    l0 = (C.abs() > 0).sum().item()

    if label is not None:
        class_score = {}
        unique_labels = torch.unique(label)
        for lbl in unique_labels:
            mask = (label == lbl).squeeze()
            masked_activations = code_activations[mask, :]
            C_lbl = masked_activations.T @ masked_activations
            l0_lbl = (C_lbl.abs() > 0).sum().item()
            class_score[lbl.item()] = 1 - l0_lbl / (masked_activations.shape[1] ** 2)

        return 1 - l0 / (code_activations.shape[1] ** 2), class_score

    return 1 - l0 / (code_activations.shape[1] ** 2)


@torch.no_grad()
def compute_neg_interference(
    codes_dict: torch.Tensor, code_activations: torch.Tensor, label: Optional[torch.Tensor] = None
) -> Union[float, Tuple[float, dict]]:
    c_comatrix = codes_dict @ codes_dict.T
    a_comatrix = code_activations.T @ code_activations

    product = a_comatrix * c_comatrix
    product = F.relu(-product)

    if label is not None:
        class_score = {}
        unique_labels = torch.unique(label)
        for lbl in unique_labels:
            mask = (label == lbl).squeeze()
            masked_activations = code_activations[mask, :]
            a_comatrix_lbl = masked_activations.T @ masked_activations
            product_lbl = a_comatrix_lbl * c_comatrix
            product_lbl = F.relu(-product_lbl)
            class_score[lbl.item()] = torch.linalg.matrix_norm(product_lbl, ord=2).item()

        return torch.linalg.matrix_norm(product, ord=2).item(), class_score

    return torch.linalg.matrix_norm(product, ord=2).item()


class OODMetric(Metric):
    """For each dictionary atom, tracks max cosine similarity to any activation
    seen so far across the epoch. compute() returns 1 - mean(per-atom max)."""
    full_state_update = True  # state (per-atom max) must persist and be updated across batches

    def __init__(self, nb_concepts: int, **kwargs):
        super().__init__(**kwargs)
        self.nb_concepts = nb_concepts
        self.add_state(
            "max_cosine",
            default=torch.full((nb_concepts,), -1.0),
            dist_reduce_fx="max",
        )

    def update(self, codes_dict: torch.Tensor, activations: torch.Tensor) -> None:
        cosine_matrix = _cosine_similarity_matrix(codes_dict, activations)  # (n_dict, n_batch)
        batch_max, _ = torch.max(cosine_matrix, dim=1)
        self.max_cosine = torch.maximum(self.max_cosine, batch_max)

    def compute(self, alive_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        max_cosine = self.max_cosine
        if alive_features is not None:
            max_cosine = max_cosine[alive_features]
        if max_cosine.numel() == 0:
            return torch.tensor(0.0, device=self.max_cosine.device)
        return 1 - max_cosine.mean()
