from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from imblearn.under_sampling import RandomUnderSampler
from imodels import (
    HSTreeClassifier,
    HSTreeClassifierCV,
    HSTreeRegressor,
    HSTreeRegressorCV,
    SkopeRulesClassifier,
)
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import f1_score
from sklearn.neighbors import BallTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch


def white_box_factory(
    model_name: str, model_params: Dict[str, Any], class_weight: Dict[int, int] = None
) -> BaseEstimator:

    if model_name == "HSTreeClassifierCV":
        if class_weight is not None:
            base_dt = DecisionTreeClassifier(
                max_leaf_nodes=model_params.pop("max_leaf_nodes", 20), class_weight=class_weight
            )
            model_params["estimator_"] = base_dt
        model = HSTreeClassifierCV(**model_params, scoring=f1_score)
    elif model_name == "HSTreeClassifier":
        if class_weight is not None:
            base_dt = DecisionTreeClassifier(
                max_leaf_nodes=model_params.pop("max_leaf_nodes", 20), class_weight=class_weight
            )
            model_params["estimator_"] = base_dt
        model = HSTreeClassifier(**model_params)
    elif model_name == "HSTreeRegressorCV":
        model = HSTreeRegressorCV(**model_params)
    elif model_name == "HSTreeRegressor":
        model = HSTreeRegressor(**model_params)
    elif model_name == "RRL":
        pass
    ### Extension
    elif model_name == "SkopeRulesClassifier":
        model = SkopeRulesClassifier(**model_params)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented in white_box_factory.")
    return model


class HSTreeClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for HSTreeClassifier recognized as a classifier by sklearn."""

    def __init__(self, hs_model: BaseEstimator):
        self.hs_model = hs_model

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: List[float] = None):
        self.hs_model.fit(X, y, sample_weight=sample_weight)
        if hasattr(self.hs_model, "classes_"):
            self.classes_ = self.hs_model.classes_
        else:
            self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray):
        return self.hs_model.predict(X)

    def predict_proba(self, X: np.ndarray):
        return self.hs_model.predict_proba(X)

    def __sklearn_tags__(self):
        # Provide a minimal but complete set of sklearn tags
        return Bunch(
            estimator_type="classifier",
            requires_y=True,
            requires_fit=True,
            binary_only=False,
            poor_score=False,
            allow_nan=False,
            stateless=False,
            multitask=False,
            non_deterministic=False,
            no_validation=False,
            X_types=["2darray"],
            input_tags=Bunch(sparse=False, pairwise=False),  # <- this prevents the sparse error
        )


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


def top_random_sampling(x: np.ndarray, k: int, seed: int = 42, threshold: Optional[float] = None) -> np.ndarray:

    threshold = 0 if threshold is None else threshold
    pos_idx = np.where(x > threshold)[0]
    neg_idx = np.where(x <= threshold)[0]

    pos_vals = x[pos_idx]
    topk_local_idx = np.argsort(pos_vals)[-k:]
    topk_pos_idx = pos_idx[topk_local_idx]

    pos_res_idx = np.setdiff1d(pos_idx, topk_pos_idx, assume_unique=True)
    rng = np.random.default_rng(seed=seed)
    rand_pos = rng.choice(pos_res_idx, size=k, replace=False)

    rand_neg = rng.choice(neg_idx, size=2 * k, replace=False)

    return np.concatenate([topk_pos_idx, rand_pos, rand_neg], axis=0)


def top_nn_sampling(
    x: np.ndarray,
    coords: np.ndarray,
    k: int,
    seed: int = 42,
    threshold: Optional[float] = None,
    radius_km: float = 1000.0,
) -> np.ndarray:
    threshold = 0 if threshold is None else threshold
    pos_idx = np.where(x > threshold)[0]
    neg_idx = np.where(x <= threshold)[0]

    pos_vals = x[pos_idx]
    topk_local_idx = np.argsort(pos_vals)[-k:]
    topk_pos_idx = pos_idx[topk_local_idx]
    pos_res_idx = np.setdiff1d(pos_idx, topk_pos_idx, assume_unique=True)
    rng = np.random.default_rng(seed=seed)
    rand_pos = rng.choice(pos_res_idx, size=k, replace=False)
    selected_pos_idx = np.concatenate([topk_pos_idx, rand_pos])

    neg_coords_rad = np.radians(coords[neg_idx])
    tree = BallTree(neg_coords_rad, metric="haversine")

    radius_rad = radius_km / 6371.0
    neg_samples = []
    for i in selected_pos_idx:
        pos_rad = np.radians(coords[i]).reshape(1, -1)
        local_idx = tree.query_radius(pos_rad, r=radius_rad)[0]
        if len(local_idx) == 0:
            continue
        local_neg_idx = neg_idx[local_idx]
        neg_samples.append(local_neg_idx)

    select_neg_idx = np.unique(np.concatenate(neg_samples, axis=0))
    rand_neg = rng.choice(select_neg_idx, size=2 * k, replace=False)

    return np.concatenate([topk_pos_idx, rand_pos, rand_neg], axis=0)


def average_tree_path_length(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    stack = [(0, 0)]  # (node_id, depth)
    leaf_depths = []

    while stack:
        node_id, depth = stack.pop()
        left = children_left[node_id]
        right = children_right[node_id]

        if left == -1:  # leaf
            leaf_depths.append(depth)
        else:
            stack.append((left, depth + 1))
            stack.append((right, depth + 1))

    return sum(leaf_depths) / len(leaf_depths)


def rule_to_tuple_set(rule_obj):
    return {
        (feature, symbol, round(float(value), -2))
        # (feature, symbol)
        for (feature, symbol), value in rule_obj.agg_dict.items()
    }


def jaccard_distance(A: List[Tuple[str, str, float]], B: List[Tuple[str, str, float]]) -> float:
    A = set(A)
    B = set(B)
    if len(A) == 0 and len(B) == 0:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return 1 - inter / union


def hungarian_rule_set_distance(
    list_setA: List[List[Tuple[str, str, float]]],
    list_setB: List[List[Tuple[str, str, float]]],
    dist_func: Callable,
    agg_type: str = "mean",
) -> float:
    nA = len(list_setA)
    nB = len(list_setB)
    n = max(nA, nB)

    if nA == 0 or nB == 0:
        return np.nan

    # pad with empty rules so both sets have size n
    paddedA = list_setA + [set()] * (n - nA)
    paddedB = list_setB + [set()] * (n - nB)

    # build the n×n cost matrix (pairwise distances)
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost[i, j] = dist_func(paddedA[i], paddedB[j])

    # Hungarian optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    # average matched cost
    return cost[row_ind, col_ind].mean() if agg_type == "mean" else cost[row_ind, col_ind].sum()


def literal_equal(
    l1: Tuple[str, str, float], l2: Tuple[str, str, float], threshold_tol: Optional[float] = None
) -> float:
    f1, op1, v1 = l1
    f2, op2, v2 = l2
    if f1 != f2 or op1 != op2:
        return False
    if threshold_tol is None:
        return v1 == v2
    return abs(v1 - v2) <= threshold_tol


def rule_edit_distance(
    ruleA: List[Tuple[str, str, float]], ruleB: List[Tuple[str, str, float]], tol: Optional[float] = None
) -> float:
    A = list(ruleA)
    B = list(ruleB)

    matched_B = set()
    matches = 0

    # Count matches (using threshold tolerance)
    for a in A:
        for j, b in enumerate(B):
            if j in matched_B:
                continue
            if literal_equal(a, b, tol):
                matched_B.add(j)
                matches += 1
                break

    # Remaining literals are insertions + deletions
    deletions = len(A) - matches
    insertions = len(B) - matches

    return deletions + insertions


def split_indices(
    N: int, N_prime: int, train_ratio: float = 0.8, seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if seed is not None:
        np.random.seed(seed)

    T = N + N_prime

    # Shuffle virtual combined indices
    perm = np.random.permutation(T)

    split = int(train_ratio * T)
    idx1 = perm[:split]
    idx2 = perm[split:]

    # Map back to A or B
    def map_indices(idx):
        idx_A = idx[idx < N]
        idx_B = idx[idx >= N] - N
        return idx_A, idx_B

    A1, B1 = map_indices(idx1)
    A2, B2 = map_indices(idx2)

    return ([A1, B1], [A2, B2])