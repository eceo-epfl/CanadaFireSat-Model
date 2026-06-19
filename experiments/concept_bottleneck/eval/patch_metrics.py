
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from pathlib import Path
from tqdm import tqdm
import yaml
import torch.nn.functional as F
from typing import List
import math

from src.constants import CONFIG_PATH
from src.data.utils import segmentation_ground_truths
from src.models.module_img import ImgModule
from src.data import get_data

from experiments.concept_bottleneck.eval.utils import compute_image_entropy_normalized, log_metrics, bimodality_coefficient, babel_function


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval_concept")
def compute_text_patch_metrics(cfg: DictConfig):

    cfg = OmegaConf.to_container(cfg, resolve=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Set seed
    seed_everything(cfg["seed"], workers=True)

    ### Extract Model (ONLY ImgModule for now)
    try:
        model = ImgModule.load_from_checkpoint(cfg["model_path"])
    except (KeyError, RuntimeError):
        with open(Path(cfg["model_path"]).parent / cfg["config_name"], "r") as f:
            model_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        model = ImgModule(model_cfg)
        mis_keys, un_keys = model.load_state_dict(torch.load(cfg["model_path"]), strict=True)
        print("Missing keys:", mis_keys)

    # Test different image sizes
    if cfg["MODEL"]["out_H"] != model.model.out_H or cfg["MODEL"]["out_W"] != model.model.out_W:
        model.model.out_H = cfg["MODEL"]["out_H"]
        model.model.out_W = cfg["MODEL"]["out_W"]

    if "ViT" in model.model_type: # Here Potentially need to check for MSCLIP
        if model.model.features.patch_embed.img_size != (cfg["MODEL"]["img_res"], cfg["MODEL"]["img_res"]):
            model.model.features.patch_embed.img_size = (cfg["MODEL"]["img_res"], cfg["MODEL"]["img_res"])

    model.eval()
    model.to(device)

    ### Create/Identify output directory
    if "test_max_seq_len" in cfg["MODEL"]:
        temp_size = str(cfg["MODEL"]["test_max_seq_len"])
    else:
        temp_size = "adapt"

    temp_size = (
        temp_size + f"_{cfg['DATASETS']['kwargs']['eval_sampling']}"
        if "eval_sampling" in cfg["DATASETS"]["kwargs"]
        else temp_size
    )
    spa_size = str(cfg["MODEL"]["img_res"]) if "img_res" in cfg["MODEL"] else str(cfg["MODEL"]["mid_input_res"])

    if cfg["DATASETS"]["eval"].get("hard"):
        output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}_hard"
    else:
        output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Load the dataset
    datamodule = get_data(cfg)
    dataset = datamodule.test_dataloader(split=cfg["split"]).dataset # Here we can specify Trian / Val / Test

    ### Extracting text concepts
    @torch.no_grad()
    def batch_encode_text(texts: List[str], batch_size: int) -> torch.Tensor:
        embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            toks = model.model.tokenizer(batch).to(model.device)
            e = model.model.msclip_model.inference_text(toks)
            e = F.normalize(e, dim=-1)
            embs.append(e.cpu())
        return torch.cat(embs, dim=0)  # [N, D]

    ### Concept Extraction Clustering vs Frequency Filtering
    if cfg.concept_ext == "freq":
        concept_df = pd.read_csv(cfg["concept_path"]).sort_values("frequency", ascending=False)
        concept_list = concept_df["concept"].tolist()[:cfg["dict_size"]]
        concept_embed = batch_encode_text(concept_list, batch_size=cfg["batch_size_text"]) # [num_concepts, D]
    else:
        concept_embed = np.load(cfg["concept_path"]) # TODO: Potentially use the label here and not the centroid vector.
    n_concepts = concept_embed.shape[0] # Check if D here is 512 or 768 -> 512

    pairwise_sim_concept = concept_embed @ concept_embed.T
    pairwise_sim_concept_diag_0 = pairwise_sim_concept.abs().fill_diagonal_(0)
    max_sim_concept, _ = pairwise_sim_concept_diag_0.max(dim=1)
    max_sim_concept = max_sim_concept.mean()
    coherence = pairwise_sim_concept_diag_0.max()
    avg_sim_concept = pairwise_sim_concept_diag_0.mean(dim=1).sum() / (pairwise_sim_concept_diag_0.shape[0] - 1)

    tot_max_similarities = []
    tot_fire_sim = []
    tot_no_fire_sim = []
    tot_image_entropy_norm = []
    tot_fire_mask_size = 0
    tot_no_fire_mask_size = 0

    for i in tqdm(range(len(dataset)), desc=f"Inferring on split {cfg['split']}", total=len(dataset)):
        data = dataset[i]
        # Extract Batch & Forward Pass
        with torch.no_grad():
            sample = data[0]
            # img_name_info = data[1]
            patch_embed = model.model.encode_patches(sample["inputs"].unsqueeze(0).to(device)) # [1, T, P, D]
            patch_embed = patch_embed.squeeze(0).cpu() # [T, P, D]
            patch_embed = F.normalize(patch_embed, dim=-1)

        # Compute similarity between patch embeddings and concept embeddings
        # Reshape patch_embed to [T*P, D] for matrix multiplication
        T, P, D = patch_embed.shape
        patch_embed_reshaped = patch_embed.view(T * P, D) # [T*P, D]
        similarity = patch_embed_reshaped @ concept_embed.T # [T*P, num_concepts]
        max_similarity, _ = similarity.max(dim=1) # [T*P]
        tot_max_similarities.append(max_similarity.numpy())

        ground_truth = segmentation_ground_truths(sample)
        labels, _ = ground_truth

        similarity = similarity.view(T, P, -1).numpy()
        labels = labels.cpu()

        labels_float = labels.float()
        out_size = int(math.sqrt(P))
        pooled_labels = F.adaptive_avg_pool2d(labels_float.unsqueeze(0), output_size=(out_size, out_size)).squeeze(0)
        pooled_labels = pooled_labels.numpy()
        pooled_labels = (pooled_labels >= cfg["threshold_patch"])

        fire_mask = (pooled_labels == 1).reshape(-1)   # [P]
        no_fire_mask = (pooled_labels == 0).reshape(-1)   # [P]

        # Count: number of (patch, timestep) pairs in each class
        tot_fire_mask_size += int(fire_mask.sum()) * T
        tot_no_fire_mask_size += int(no_fire_mask.sum()) * T

        # Sum over time AND spatial patches → [num_concepts]
        fire_sim = similarity[:, fire_mask, :].sum(axis=(0, 1))
        no_fire_sim = similarity[:, no_fire_mask, :].sum(axis=(0, 1))

        # Per-image Entropy
        img_entropy_norm = compute_image_entropy_normalized(similarity)
        tot_image_entropy_norm.append(img_entropy_norm)

        tot_fire_sim.append(fire_sim)
        tot_no_fire_sim.append(no_fire_sim)

    ### Global averages
    tot_fire_sim = np.stack(tot_fire_sim).sum(axis=0)    / tot_fire_mask_size
    tot_no_fire_sim = np.stack(tot_no_fire_sim).sum(axis=0) / tot_no_fire_mask_size
    disc = tot_fire_sim - tot_no_fire_sim         # [num_concepts]

    ### Coverage metrics
    tot_max_similarities = np.concatenate(tot_max_similarities)  # [total_T*P]
    avg_max_similarity = float(tot_max_similarities.mean())
    quantile_10 = float(np.quantile(tot_max_similarities, 0.1))
    quantile_25 = float(np.quantile(tot_max_similarities, 0.25))

    ### Discriminativity metrics
    abs_disc = float(np.abs(disc).mean())
    bc = bimodality_coefficient(disc)
    frac_fire = float((disc >  cfg["threshold_disc"]).mean())
    frac_no_fire = float((disc < - cfg["threshold_disc"]).mean())
    frac_neutral = float((np.abs(disc) <= cfg["threshold_disc"]).mean())

    ### Babel Function
    babel_k_values = cfg.get("babel_k_values", [2, 4, 8, 16])
    babel_results  = {}
    for k in babel_k_values:
        if k >= n_concepts:
            print(f"[WARN] k={k} >= n_concepts={n_concepts}, skipping Babel at k={k}")
            continue
        bk = babel_function(concept_embed, k=k)
        babel_results[k] = bk

    ### Average Per-Image Entropy
    tot_image_entropy_norm  = np.array(tot_image_entropy_norm)   # [num_images]
    avg_entropy_norm    = float(tot_image_entropy_norm.mean())

    ### Save .npy outputs
    output_name = Path(cfg["concept_path"]).stem
    np.save(output_dir / f"{output_name}_max_similarities.npy",    tot_max_similarities)
    np.save(output_dir / f"{output_name}_target_discriminative.npy", disc)

    ### Build metrics dict for logging
    # Structure: {section: {metric_name: (value, description)}}
    metrics = {
        "Dataset & Dictionary Info": {
            "Concept file":             (output_name,    ""),
            "Number of concepts":       (n_concepts,     ""),
            "Total fire patches":       (tot_fire_mask_size,    "patch × timestep pairs"),
            "Total non-fire patches":   (tot_no_fire_mask_size, "patch × timestep pairs"),
        },
        "Coverage (patch-to-dictionary alignment)": {
            "Average max cosine similarity":    (avg_max_similarity, "higher is better"),
            "10th percentile max similarity":   (quantile_10,        "tail coverage, higher is better"),
            "25th percentile max similarity":   (quantile_25,        "lower quartile coverage"),
        },
        "Intra-Dictionary Redundancy": {
            "Mutual coherence (mean max pairwise cosine)": (
                coherence,
                "lower is better, measures worst-case redundancy"
            ),
            "Average Max-Pairwise cosine similarity": (
                max_sim_concept,
                "lower is better, measures average redundancy"
            ),
            "Average pairwise cosine similarity": (
                avg_sim_concept,
                "lower is better, measures average redundancy"
            ),
            **{
            f"Babel function μ_1(k={k})": (
            float(score),
            f"cumulative coherence over {k} nearest neighbors; "
            f"lower is better; k=1 equals strict mutual coherence"
        )
        for k, score in babel_results.items()
    },
        },
        "Discriminativity (fire vs non-fire)": {
            "Average absolute discriminativity":  (
                abs_disc,
                "mean |fire_sim - no_fire_sim| per concept, higher = more discriminative"
            ),
            "Bimodality coefficient (BC)":        (
                bc,
                ">0.555 = bimodal distribution, i.e. concepts split into fire/non-fire groups"
            ),
            f"Fraction fire-associated (disc>{cfg['threshold_disc']})":   (
                frac_fire,
                "fraction of concepts more active on fire patches"
            ),
            f"Fraction fire-suppressing (disc<-{cfg['threshold_disc']})": (
                frac_no_fire,
                "fraction of concepts more active on non-fire patches"
            ),
            f"Fraction neutral (|disc|<={cfg['threshold_disc']})":        (
                frac_neutral,
                "fraction of concepts indifferent to fire presence"
            ),
        },
        "Image-Level Entropy (assigned concepts only)": {
            "Average normalized entropy [0,1]": (
                avg_entropy_norm,
                "entropy divided by log(n_assigned_concepts), "
                "comparable across images using different numbers of concepts"
            ),
        },
    }

    ### Print to console
    for section, section_metrics in metrics.items():
        print(f"\n--- {section} ---")
        for key, (value, description) in section_metrics.items():
            if isinstance(value, float):
                print(f"  {key:<45s} {value:.4f}"
                      + (f"  ({description})" if description else ""))
            else:
                print(f"  {key:<45s} {value}"
                      + (f"  ({description})" if description else ""))

    ### Write to txt file
    log_metrics(output_dir, output_name, metrics)

if __name__ == "__main__":
    compute_text_patch_metrics()
