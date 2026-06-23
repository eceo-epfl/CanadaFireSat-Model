from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from fastkmeans import FastKMeans
from msclip.inference.utils import build_model
from src.constants import CONFIG_PATH
import pandas as pd
from typing import List, Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F


def minibatch_kmeans(
        X: np.ndarray,
        method: str,
        **kwargs
) -> torch.Tensor:

    assert len(X.shape) == 2, f"The input features has shape {X.shape}"

    if method == "torch":
        # X = torch.from_numpy(X).float().cuda()
        kmeans = FastKMeans(
            d=X.shape[1],
            k=kwargs.get("n_cluster"),
            tol=1e-4,
            niter=kwargs.get("max_iter"),
            seed=kwargs.get("seed"),
            verbose=True,
            gpu=True
        )

        kmeans.fit(X)
        clusters = kmeans.centroids
        cluster_ids = kmeans.fit_predict(X)
        return torch.from_numpy(clusters), torch.from_numpy(cluster_ids)

    else:

        kmeans = MiniBatchKMeans(n_clusters=kwargs.get("n_cluster"),
                                random_state=kwargs.get("seed"),
                                batch_size=kwargs.get("kmeans_batch"),
                                max_iter=kwargs.get("max_iter"),
                                verbose=2)
        kmeans.fit(X)
        clusters = kmeans.cluster_centers_
        cluster_ids = kmeans.labels_
        return torch.from_numpy(clusters), torch.from_numpy(cluster_ids)


def DEPRECATED_label_centroids_nearest(
    centroid: torch.Tensor,      # [n_clusters, D]
    dict_emb: torch.Tensor,      # [n_atoms, D] normalized
    dict_atom: List[str],        # [n_atoms] original phrases
) -> Tuple[List[str], np.ndarray]:
    centroids_norm = F.normalize(centroid, dim=-1)
    sim = centroids_norm @ dict_emb.T  # [n_clusters, n_atoms]
    nearest_idx = sim.argmax(dim=1)     # [n_clusters]
    label = [dict_atom[i] for i in nearest_idx.tolist()]
    return label, sim.max(dim=1).numpy()


def label_centroids_nearest(
    centroid: torch.Tensor,      # [n_clusters, D]
    dict_emb: torch.Tensor,      # [n_atoms, D] normalized
    dict_atom: List[str],        # [n_atoms] original phrases
    chunk_size: int = 2000,      # process this many centroids at a time
    device: str = "cuda",
) -> Tuple[List[str], np.ndarray]:

    centroids_norm = F.normalize(centroid, dim=-1).to(device)  # [n_clusters, D]
    dict_emb_norm  = F.normalize(dict_emb, dim=-1).to(device)  # [n_atoms, D]

    n_clusters = centroids_norm.shape[0]
    nearest_idx_all = []
    max_sim_all = []

    for start in tqdm(range(0, n_clusters, chunk_size), desc="Labeling centroids"):
        end = min(start + chunk_size, n_clusters)
        chunk = centroids_norm[start:end]               # [chunk, D]

        sim_chunk = chunk @ dict_emb_norm.T              # [chunk, n_atoms]

        max_sim, nearest_idx = sim_chunk.max(dim=1)      # both [chunk]

        nearest_idx_all.append(nearest_idx.cpu())
        max_sim_all.append(max_sim.cpu())

        del sim_chunk  # free memory explicitly
        torch.cuda.empty_cache() if device == "cuda" else None

    nearest_idx_all = torch.cat(nearest_idx_all)          # [n_clusters]
    max_sim_all = torch.cat(max_sim_all).numpy()          # [n_clusters]

    labels = [dict_atom[i] for i in nearest_idx_all.tolist()]
    return labels, max_sim_all


def label_centroids_by_frequency(
    cluster_label: np.ndarray,       # [n_atoms] cluster id per phrase
    dict_atom: List[str],            # [n_atoms]
    dict_freq: List[int],    # frequency from your candidate extraction step
    n_clusters: int,
) -> List[str]:
    labels = []
    for c in range(n_clusters):
        members_idx = np.where(cluster_label == c)[0]
        if len(members_idx) == 0:
            labels.append(None)  # empty cluster, handle separately
            continue
        members = [dict_atom[i] for i in members_idx]
        freqs = [dict_freq[i] for i in members_idx]
        best_idx = np.argmax(freqs)
        best = members[best_idx]
        labels.append(best)
    return labels


def compute_simplified_silhouette(
    dict_emb: torch.Tensor,        # [n_atoms, D]
    centroid_emb: torch.Tensor,    # [n_clusters, D]
    cluster_labels: torch.Tensor,  # [n_atoms]
    chunk_size: int = 2000,
    device: str = "cuda",
) -> np.ndarray:

    dict_norm     = F.normalize(dict_emb, dim=-1).to(device)      # [n_atoms, D]
    centroid_norm = F.normalize(centroid_emb, dim=-1).to(device)  # [n_clusters, D]
    labels = cluster_labels.to(device)                            # [n_atoms]

    n_atoms = dict_norm.shape[0]

    a_approx = torch.empty(n_atoms, device=device)
    b_approx = torch.empty(n_atoms, device=device)

    for start in tqdm(range(0, n_atoms, chunk_size), desc="Simplified silhouette"):
        end = min(start + chunk_size, n_atoms)
        chunk = dict_norm[start:end]                  # [chunk, D]
        chunk_labels = labels[start:end]               # [chunk]

        # Cosine distance to ALL centroids [chunk, n_clusters]
        sim_to_centroids = chunk @ centroid_norm.T
        dist_to_centroids = 1.0 - sim_to_centroids      # [chunk, n_clusters]

        # a_approx: distance to own cluster's centroid
        own_dist = dist_to_centroids.gather(
            1, chunk_labels.unsqueeze(1)
        ).squeeze(1)                                    # [chunk]

        # b_approx: distance to nearest OTHER cluster's centroid
        # mask out own cluster's column with +inf before taking min
        masked_dist = dist_to_centroids.clone()
        masked_dist.scatter_(
            1, chunk_labels.unsqueeze(1), float("inf")
        )
        nearest_other_dist = masked_dist.min(dim=1).values  # [chunk]

        a_approx[start:end] = own_dist
        b_approx[start:end] = nearest_other_dist

    denom = torch.maximum(a_approx, b_approx)
    s_approx = (b_approx - a_approx) / denom.clamp(min=1e-8)

    s_approx_np = s_approx.cpu().numpy()

    return s_approx_np


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="cluster_concept")
def cluster_concept(cfg: DictConfig):

    dict_atom = pd.read_csv(cfg.concept_df_path)["concept"].tolist()
    dict_freq = pd.read_csv(cfg.concept_df_path)["frequency"].tolist()
    print(f"[INFO] Using {len(dict_atom)} unique atoms as k-means inputs.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, tokenizer = build_model(
            model_name=cfg.model.model_name, pretrained=cfg.model.pretrained,
            ckpt_path=cfg.model.ckpt_path, device=device, channels=cfg.model.channels
    )
    model.to(device).eval()

    @torch.no_grad()
    def batch_encode_text(texts: List[str], batch_size: int) -> torch.Tensor:
        embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            toks = tokenizer(batch).to(model.device)
            e = model.inference_text(toks)
            e = F.normalize(e, dim=-1)
            embs.append(e.cpu())
        return torch.cat(embs, dim=0)  # [N, D]

    dict_emb = batch_encode_text(dict_atom, cfg.msclip_batch_size)  # [P, D]

    assert cfg.n_cluster < len(dict_atom), f"Number of clusters {cfg.n_cluster} for the number of atoms {len(dict_atom)}"
    centroid_emb, cluster_labels = minibatch_kmeans(dict_emb.cpu().numpy(), method=cfg.cluster_type, n_cluster=cfg.n_cluster,
                                                    **cfg.cluster_params)
    Path(cfg.output_dir).mkdir(exist_ok=True)
    np.save(Path(cfg.output_dir) / (Path(cfg.concept_df_path).stem + f"_{cfg.n_cluster}_centroids.npy"), centroid_emb.numpy())


    closest_label_centroid, distances_rep = label_centroids_nearest(centroid_emb, dict_emb, dict_atom)
    freq_label_centroid = label_centroids_by_frequency(cluster_labels.numpy(), dict_atom,  dict_freq, cfg.n_cluster)

    sil_score = compute_simplified_silhouette(dict_emb, centroid_emb, cluster_labels, chunk_size=5000) # TODO: Double check code and merge with label centroids.
    # sil_score = 0
    dist_summary = {
        "mean_sim_to_representative":   float(distances_rep.mean()),
        "median_sim_to_representative": float(np.median(distances_rep)),
        "std_sim_to_representative":    float(distances_rep.std()),
        "mean_silhouette_score": float(sil_score.mean()),
    }
    print("\n[METRIC] Distance to assigned cluster representative (cosine distance) & Silhouette:")
    for k, v in dist_summary.items():
        print(f"  {k:<40s} {v:.4f}")

    df = pd.DataFrame({"concept_closest": closest_label_centroid,
                       "concept_most_frequent": freq_label_centroid})
    df.to_csv(Path(cfg.output_dir) / (Path(cfg.concept_df_path).stem + f"_{cfg.n_cluster}_centroids_label.csv"), index=False)
    print(f"[INFO] Saved {len(df)} concepts to {Path(cfg.output_dir) / (Path(cfg.concept_df_path).stem + f'_{cfg.n_cluster}_centroids_label.csv')}")

    ### Save all summary metrics to a single txt log alongside the .npy files
    metrics_path = Path(cfg.output_dir) / f"{Path(cfg.concept_df_path).stem}_{cfg.n_cluster}_clustering_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Clustering Quality Metrics\n")
        f.write(f"Concept file:   {Path(cfg.concept_df_path).stem}\n")
        f.write(f"n_clusters:     {cfg.n_cluster}\n")
        f.write(f"n_atoms:        {len(dict_atom)}\n")
        f.write(f"cluster method: {cfg.cluster_type}\n\n")

        f.write("--- Distance to Assigned Representative & Silhouette ---\n")
        for k, v in dist_summary.items():
            f.write(f"  {k:<40s} {v:.4f}\n")

    print(f"\n[INFO] Saved clustering metrics to {metrics_path}")

if __name__ == "__main__":
    cluster_concept()
