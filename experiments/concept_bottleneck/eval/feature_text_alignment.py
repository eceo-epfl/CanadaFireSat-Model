from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch

from experiments.concept_bottleneck.eval.plot_utils import plot_alignment_distribution, plot_concept_duplication, plot_dictionary_embedding
from experiments.concept_bottleneck.eval.utils import compute_text_dist
from experiments.concept_bottleneck.sae.metrics import _cosine_similarity_matrix
from experiments.concept_bottleneck.sae.sae_module import plSAE
from experiments.concept_bottleneck.sae.utils import get_pylogger
from src.constants import CONFIG_PATH
import pytorch_lightning as pl
import matplotlib as mpl


mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


log = get_pylogger(__name__)


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval_text_alignment")
def evaluate_text_alignment(cfg: DictConfig):

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    log.info(f"Instantiating SAE <{cfg.sae._target_}>")
    sae: plSAE = hydra.utils.instantiate(cfg.sae)
    log.info(f"Loading SAE checkpoint from {cfg.sae_ckpt_path}")
    ckpt = torch.load(cfg.sae_ckpt_path, map_location=device)
    sae.to(device)

    if sae.hparams.get("arch_kwargs", None) is not None:
        sae.set_arch(sae.hparams["arch_kwargs"])

    missing, unexpected = sae.load_state_dict(ckpt["state_dict"], strict=True)
    log.info(f"Missing keys: {missing}")
    log.info(f"Unexpected keys: {unexpected}")

    sae.eval()

    if cfg.alive_feature_path is not None and Path(cfg.alive_feature_path).exists():
        # alive_feature = np.load(cfg.alive_feature_path)
        # alive_feature = torch.tensor(alive_feature, dtype=torch.bool, device=device)
        alive_feature = torch.load(cfg.alive_feature_path, map_location=device)
    else:
        alive_feature = torch.ones(sae.net.get_dictionary().shape[0], dtype=torch.bool, device=device)

    text_all_sim, text_all_indices = compute_text_dist(sae.net.get_dictionary()[alive_feature].to("cpu"), sae.label_vocab_emb.to("cpu"))
    text_all_sim, text_all_indices = text_all_sim.numpy(), text_all_indices.numpy()

    W = sae.net.dictionary.W[alive_feature, :]
    W = W.detach().cpu()
    vocab_emb = sae.label_vocab_emb.to("cpu")
    _, text_all_indices_max_W = torch.max(W, dim=1)
    vocab_max = vocab_emb[text_all_indices_max_W, :]

    cosine_matrix = _cosine_similarity_matrix(sae.net.get_dictionary()[alive_feature].to("cpu"), vocab_max).abs()
    text_all_sim_max_W = np.diagonal(cosine_matrix.detach().numpy())

    fig, ax = plt.subplots(
    figsize=(12, 6))

    plot_alignment_distribution(
        text_all_sim,
        ax=ax,
        highlight_top=100,
        median_line=False,
        label="RA-SAE"
    )

    fig.legend()
    # fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / "text_alignment_distribution.png", dpi=300)

    fig, ax = plt.subplots(
    figsize=(12, 6))

    plot_alignment_distribution(
        text_all_sim_max_W,
        ax=ax,
        highlight_top=100,
        median_line=False,
        linestyle=":",
        color="#e63946",
        label="W-RA-SAE"
    )

    fig.legend()
    # fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / "W-text_alignment_distribution.png", dpi=300)

    vocab_df = pd.read_csv(cfg.vocab_csv_path)

    fig, ax = plt.subplots(
        figsize=(12, 6))

    plot_concept_duplication(
        names=vocab_df.loc[text_all_indices]["term"].values,
        sim=text_all_sim,
        top_n=20,
        ax=ax)

    fig.legend()
    # fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / "captions_duplication.png", dpi=300)

    fig, ax = plt.subplots(
    figsize=(12, 6))

    plot_concept_duplication(
            names=vocab_df.loc[text_all_indices_max_W]["term"].values,
            sim=text_all_sim_max_W,
            top_n=20,
            ax=ax)

    fig.legend()
    # fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / "W-captions_duplication.png", dpi=300)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_dictionary_embedding(
        sae.net.get_dictionary()[alive_feature].detach().to("cpu").numpy(),
        names=vocab_df.loc[text_all_indices]["term"].values,
        top_n_captions=20,
        method="umap",
        ax=ax,
    )

    # fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / "dictionary_embedding.png", dpi=300)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_dictionary_embedding(
        sae.net.get_dictionary()[alive_feature].detach().to("cpu").numpy(),
        names=vocab_df.loc[text_all_indices_max_W]["term"].values,
        top_n_captions=20,
        method="umap",
        ax=ax,
    )

    # fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / "W-dictionary_embedding.png", dpi=300)



if __name__ == "__main__":
    evaluate_text_alignment()