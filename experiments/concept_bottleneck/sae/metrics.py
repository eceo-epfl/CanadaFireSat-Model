from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import torch
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


def compute_binary_moran(lat: np.ndarray, lon: np.ndarray, active: np.ndarray, threshold_m: int = 500000) -> float:
    """Compute Moran's I for binary 0/1 activations."""
    lat = lat.reshape(-1)
    lon = lon.reshape(-1)
    if len(lat) < 2 or len(lon) < 2:
        return np.nan
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)
    gdf["active"] = active.astype(float)
    w = DistanceBand.from_dataframe(gdf, threshold=threshold_m, silence_warnings=True)
    mi = Moran(gdf["active"], w)
    return mi.I


def spherical_cartesian(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat = np.radians(lat)
    lon = np.radians(lon)
    R = 6371000
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    coords = np.vstack([x, y, z]).T
    return coords


def compute_anni(lat: np.ndarray, lon: np.ndarray, label: np.ndarray = None, area: float = SEAS_AREA) -> float:

    lat = lat.reshape(-1)
    lon = lon.reshape(-1)
    if len(lat) < 2 or len(lon) < 2:
        if label is not None:
            unique_labels = np.unique(label)
            return np.nan, {class_id: np.nan for class_id in unique_labels}
        return np.nan

    coords = spherical_cartesian(lat, lon)

    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)  # Second Nearest as each coords is present in the dataset.
    nn_distances = dists[:, 1]
    mean_nn = nn_distances.mean()

    n = len(coords)
    density = n / area
    expected_mean = 0.5 / np.sqrt(density)

    anni = mean_nn / expected_mean

    if label is not None:
        class_score = {}
        unique_labels = np.unique(label)
        for lbl in unique_labels:
            mask = np.squeeze(label == lbl)
            masked_coords = coords[mask, :]

            if len(masked_coords) < 2:
                class_score[int(lbl)] = np.nan
                continue

            tree = cKDTree(masked_coords)
            dists, _ = tree.query(masked_coords, k=2)  # Second Nearest as each coords is present in the dataset.
            nn_distances = dists[:, 1]
            mean_nn = nn_distances.mean()

            n = len(masked_coords)
            density = n / area
            expected_mean = 0.5 / np.sqrt(density)
            class_score[lbl.item()] = mean_nn / expected_mean

        return anni, class_score

    return anni


@torch.no_grad()
def compute_ood(codes_dict: torch.Tensor, activations: torch.Tensor) -> float:
    cosine_matrix = _cosine_similarity_matrix(codes_dict, activations)
    max_cosine_matrix, _ = torch.max(cosine_matrix, dim=1)
    return 1 - max_cosine_matrix.mean().item()


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