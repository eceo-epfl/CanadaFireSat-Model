from typing import Optional, Any

from omegaconf import DictConfig
import torch
from datetime import datetime
from pathlib import Path
from scipy.stats import skew, kurtosis
import numpy as np
import yaml

from experiments.concept_bottleneck.sae.metrics import _cosine_similarity_matrix
from src.models.module_img import ImgModule


def load_img_model(cfg: DictConfig, device: torch.device, logger: Optional[Any] = None) -> ImgModule:
    try:
        model = ImgModule.load_from_checkpoint(cfg.model_path)
    except (KeyError, RuntimeError):
        with open(Path(cfg.model_path).parent / cfg.config_name, "r") as config_file:
            model_cfg = yaml.load(config_file, Loader=yaml.SafeLoader)
        model = ImgModule(model_cfg)
        mis_key, un_keys = model.load_state_dict(torch.load(cfg.model_path, map_location=device), strict=True)
        if logger is not None:
            logger.info(f"Missing keys: {mis_key}")
            logger.info(f"Unexpected keys: {un_keys}")

    model.model.out_H = cfg.MODEL.out_H
    model.model.out_W = cfg.MODEL.out_W
    return model.eval().to(device)



def bimodality_coefficient(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return float("nan")
    s = skew(x)                    # skewness
    k = kurtosis(x, fisher=True)   # excess kurtosis (Fisher definition, normal=0)
    # Correction factor for small samples
    correction = 3 * (n - 1)**2 / ((n - 2) * (n - 3))
    bc = (s**2 + 1) / (k + correction)
    return float(bc)


def log_metrics(output_dir: Path, output_name: str, metrics: dict):
    """
    Write all computed metrics to a txt file in output_dir.
    Appends timestamp and concept file name as header.
    """
    log_path = output_dir / f"{output_name}_metrics.txt"
    with open(log_path, "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Concept Dictionary Evaluation Metrics\n")
        f.write(f"Concept file:  {output_name}\n")
        f.write(f"Timestamp:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        for section, section_metrics in metrics.items():
            f.write(f"--- {section} ---\n")
            for key, (value, description) in section_metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key:<45s} {value:.4f}\n")
                    if description:
                        f.write(f"  {'':45s} ({description})\n")
                else:
                    f.write(f"  {key:<45s} {value}\n")
                    if description:
                        f.write(f"  {'':45s} ({description})\n")
            f.write("\n")

    print(f"[INFO] Metrics saved to {log_path}")
    return log_path


def babel_function(concept_embed: torch.Tensor, k: int) -> torch.Tensor:
    """
    Babel function μ_1(k, D):
    For each atom i, sum the k largest absolute cosine similarities
    to all other atoms j ≠ i. Return the maximum over all atoms.

    Formally:
        μ_1(k, D) = max_i  sum_{j in N_k(i), j≠i} |<d_i, d_j>|

    where N_k(i) are the k nearest neighbors of atom i.

    Interpretation:
    - Measures cumulative coherence: how much any single atom
      overlaps with its k nearest neighbors collectively.
    - Lower is better: high Babel function means clusters of
      near-duplicate atoms exist in the dictionary.
    - In compressed sensing theory, unique k-sparse recovery
      is guaranteed when μ_1(k-1, D) + μ_1(k, D) < 1.
    - Strictly generalizes mutual coherence:
      μ_1(1, D) == mutual coherence (max pairwise cosine).

    Args:
        concept_embed: [num_concepts, D] L2-normalized concept embeddings
        k: number of nearest neighbors to sum over (analogous to sparsity level)

    Returns:
        babel_k: scalar tensor, the Babel function value at level k
    """
    # Full pairwise cosine similarity matrix [num_concepts, num_concepts]
    pairwise_sim = concept_embed @ concept_embed.T  # [num_concepts, num_concepts]

    # Exclude self-similarity by setting diagonal to -inf
    pairwise_abs = pairwise_sim.abs()
    pairwise_abs = pairwise_abs.clone()
    pairwise_abs.fill_diagonal_(0.0)  # zero out diagonal before summing

    # For each atom i, sort absolute similarities to all other atoms descending
    # and take the sum of the top-k
    # [num_concepts, num_concepts-1] after excluding diagonal
    sorted_sims, _ = pairwise_abs.sort(dim=1, descending=True)  # [num_concepts, num_concepts]
    top_k_sims = sorted_sims[:, :k]                             # [num_concepts, k]
    per_atom_babel = top_k_sims.sum(dim=1)                      # [num_concepts]

    babel_k = per_atom_babel.max().item()                       # scalar
    return babel_k


def compute_image_entropy_normalized(
    similarity: np.ndarray,
) -> float:
    T, P, num_concepts = similarity.shape
    similarity_flat = similarity.reshape(T * P, num_concepts)
    assignments = similarity_flat.argmax(axis=1)

    _, counts = np.unique(assignments, return_counts=True)
    n_assigned = len(counts)

    if n_assigned <= 1:
        return 0.0  # single concept dominates entirely, entropy is 0

    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(n_assigned)  # entropy if uniform over n_assigned concepts

    return float(entropy / max_entropy)


@torch.no_grad()
def compute_text_dist(codes_dict: torch.Tensor, label_vocab_emb: torch.Tensor) -> float:
    cosine_matrix = _cosine_similarity_matrix(codes_dict, label_vocab_emb).abs()
    max_cosine_per_atom, max_i = torch.max(cosine_matrix, dim=1)
    return max_cosine_per_atom, max_i
