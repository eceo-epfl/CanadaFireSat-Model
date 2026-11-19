"""Compare vocabulary-to-feature text alignment across several SAEs.

The vocabulary is read from a single CSV and encoded on the fly with the SAE's text encoder,
once per column in ``cfg.vocab_columns`` (e.g. ``term`` -> one entry per term,
``class_name`` -> one entry per class). For each column:

* ``text_alignment_distribution_topn_<column>``: for every vocabulary entry, the mean cosine
  similarity over its top-n most similar (alive) SAE features, for each n in ``cfg.top_n_list``,
  ranked across entries; all SAEs on one plot.
* ``dictionary_embedding_<column>_top<n>`` (per SAE, for each n in ``cfg.embedding_n_list``): a UMAP
  of the dictionary where features in some entry's top-n are coloured by that entry's class
  (``cfg.vocab_class_column``). The layout is computed once per SAE and shared by all these plots.

Outputs are written to every SAE's ``output_dir`` (and to ``cfg.output_dir`` if set).
"""
from pathlib import Path

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from experiments.concept_bottleneck.eval.plot_utils import (
    SAE_COLORS,
    embed_dictionary,
    plot_alignment_distribution,
    plot_dictionary_embedding_topn,
    plot_topn_alignment_comparison,
)
from experiments.concept_bottleneck.sae.metrics import _cosine_similarity_matrix
from experiments.concept_bottleneck.sae.sae_module import plSAE
from experiments.concept_bottleneck.sae.utils import batch_encode_text, get_pylogger
from msclip.inference.utils import build_model
from src.constants import CONFIG_PATH


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


def load_sae(cfg: DictConfig, run_cfg: DictConfig, device: torch.device) -> plSAE:
    """Instantiate the base SAE config merged with the run-specific overrides and load its checkpoint."""
    sae_cfg = OmegaConf.create(OmegaConf.to_container(cfg.sae, resolve=True))
    if run_cfg.get("sae_overrides"):
        sae_cfg = OmegaConf.merge(sae_cfg, run_cfg.sae_overrides)

    log.info(f"[{run_cfg.name}] Instantiating SAE <{sae_cfg._target_}>")
    sae: plSAE = hydra.utils.instantiate(sae_cfg)
    sae.to(device)
    if sae.hparams.get("arch_kwargs", None) is not None:
        sae.set_arch(sae.hparams["arch_kwargs"])

    log.info(f"[{run_cfg.name}] Loading checkpoint from {run_cfg.ckpt_path}")
    ckpt = torch.load(run_cfg.ckpt_path, map_location=device)
    missing, unexpected = sae.load_state_dict(ckpt["state_dict"], strict=True)
    log.info(f"[{run_cfg.name}] Missing keys: {missing} | Unexpected keys: {unexpected}")
    return sae.eval()


def load_alive_mask(run_cfg: DictConfig, n_features: int, device: torch.device) -> torch.Tensor:
    path = run_cfg.get("alive_feature_path")
    if path is not None and Path(path).exists():
        return torch.load(path, map_location=device).bool()
    log.warning(f"[{run_cfg.name}] No alive mask found at {path}; using all features.")
    return torch.ones(n_features, dtype=torch.bool, device=device)


def save_figure(fig, stem: Path, close: bool = True):
    """Save a figure as PNG (preview) and PDF (vector, for the paper)."""
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    if close:
        plt.close(fig)


@torch.no_grad()
def topn_term_alignment(dictionary: torch.Tensor, vocab_emb: torch.Tensor, n_max: int):
    """
    Returns
    -------
    top_sims : (n_vocab, n_max) cosine similarity of each term's n_max nearest features (descending).
    feature_rank : (n_features,) best rank (1..n_max) a feature reaches for any term, 0 if none.
    feature_term : (n_features,) vocabulary index of the term giving that best rank
        (ties broken by highest similarity), -1 if none.
    """
    cos = _cosine_similarity_matrix(vocab_emb, dictionary)  # (n_vocab, n_features)
    n_vocab, n_features = cos.shape
    n_max = min(n_max, n_features)
    top_sims, top_idx = torch.topk(cos, k=n_max, dim=1)
    top_sims, top_idx = top_sims.numpy(), top_idx.numpy()

    feats = top_idx.reshape(-1)
    ranks = np.tile(np.arange(1, n_max + 1), n_vocab)
    terms = np.repeat(np.arange(n_vocab), n_max)
    sims = top_sims.reshape(-1)
    # Best entry per feature: lowest rank first, then highest similarity.
    order = np.lexsort((-sims, ranks))
    uniq_feats, first = np.unique(feats[order], return_index=True)

    feature_rank = np.zeros(n_features, dtype=np.int64)
    feature_term = np.full(n_features, -1, dtype=np.int64)
    feature_rank[uniq_feats] = ranks[order][first]
    feature_term[uniq_feats] = terms[order][first]
    return top_sims, feature_rank, feature_term


def encode_vocabularies(cfg: DictConfig, vocab_df: pd.DataFrame, device: torch.device,
                        extra_texts: dict | None = None):
    """
    Encode one vocabulary per column of ``cfg.vocab_columns`` with the SAE text encoder.

    Returns
    -------
    vocabs : {column: {"texts": [...], "classes": np.ndarray[str], "emb": Tensor [N, D]}}.
        Entries are the unique values of the column (CSV order); each keeps the class of its first row.
    extra : {name: Tensor [N, D]} embeddings of ``extra_texts`` ({name: [texts]}), same encoder.
    """
    class_col = cfg.vocab_class_column
    text_enc_kwargs = OmegaConf.to_container(cfg.get("text_enc_kwargs") or cfg.sae.text_enc_kwargs, resolve=True)
    batch_size = int(cfg.get("text_batch_size") or cfg.sae.get("text_batch_size", 128))

    log.info(f"Building text encoder {text_enc_kwargs.get('model_name')}")
    msclip_model, _, tokenizer = build_model(device=str(device), **text_enc_kwargs)
    msclip_model.to(device).eval()

    vocabs = {}
    for column in cfg.vocab_columns:
        firsts = vocab_df.drop_duplicates(subset=column, keep="first")
        texts = firsts[column].astype(str).tolist()
        classes = firsts[class_col].astype(str).to_numpy()
        emb = batch_encode_text(texts, msclip_model, tokenizer, batch_size).float()
        log.info(f"Vocabulary '{column}': {len(texts)} entries encoded -> {tuple(emb.shape)}")
        vocabs[column] = {"texts": texts, "classes": classes, "emb": emb}

    extra = {}
    for name, texts in (extra_texts or {}).items():
        extra[name] = batch_encode_text(list(texts), msclip_model, tokenizer, batch_size).float()
        log.info(f"Extra vocabulary '{name}': {len(texts)} entries encoded -> {tuple(extra[name].shape)}")

    del msclip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return vocabs, extra


def plot_uncovered_alignment(uncovered_sims: dict, n: int, ax=None):
    """Ranked closest-term cosine similarity of the left-out features, one curve per SAE."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.6, 4.8))
    for i, (name, sims) in enumerate(uncovered_sims.items()):
        if len(sims) == 0:
            log.warning(f"[{name}] No uncovered features at top-{n}; skipped in the plot.")
            continue
        plot_alignment_distribution(
            sims,
            ax=ax,
            label=f"{name} ({len(sims)} features)",
            color=SAE_COLORS[i % len(SAE_COLORS)],
            title=f"Closest-term similarity of features outside every top-{n}",
            median_line=False,
            highlight_top=0,
        )
    ax.set_xlabel("Uncovered features, ranked by similarity")
    ax.set_ylabel("Cosine similarity to the closest term")
    ax.legend(loc="upper right", frameon=False)
    return ax


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval_text_alignment_compare")
def evaluate_text_alignment_compare(cfg: DictConfig):

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    n_list = sorted(int(n) for n in cfg.top_n_list)
    emb_n_list = sorted(int(n) for n in cfg.embedding_n_list)
    unc_n = cfg.get("uncovered_top_n")
    unc_n = int(unc_n) if unc_n is not None else None
    n_needed = max(n_list + emb_n_list + ([unc_n] if unc_n is not None else []))
    coverage_columns = list(cfg.get("coverage_vocab_columns") or cfg.vocab_columns)
    unknown = set(coverage_columns) - set(cfg.vocab_columns)
    if unknown:
        raise ValueError(f"coverage_vocab_columns {sorted(unknown)} not in vocab_columns {list(cfg.vocab_columns)}")
    method = cfg.get("embedding_method", "umap")
    seed = cfg.get("seed") or 0

    vocab_df = pd.read_csv(cfg.vocab_csv_path)
    extra_texts = {}
    if unc_n is not None and cfg.get("uncovered_vocab_csv_path"):
        unc_df = pd.read_csv(cfg.uncovered_vocab_csv_path)
        if "term" not in unc_df.columns:
            raise ValueError(f"{cfg.uncovered_vocab_csv_path} must have a 'term' column (got {list(unc_df.columns)})")
        extra_texts["uncovered"] = list(dict.fromkeys(unc_df["term"].dropna().astype(str)))
    vocabs, extra_embs = encode_vocabularies(cfg, vocab_df, device, extra_texts)
    unc_terms = extra_texts.get("uncovered")
    unc_emb = extra_embs.get("uncovered")
    class_order = list(dict.fromkeys(vocab_df[cfg.vocab_class_column].astype(str)))
    log.info(f"Classes ({cfg.vocab_class_column}): {class_order}")

    topn_sims = {column: {} for column in vocabs}  # column -> SAE name -> (n_vocab, n_needed)
    coverage_rows = []
    uncovered_sims = {}  # SAE name -> closest-term similarity of each uncovered feature
    for run_cfg in cfg.saes:
        sae = load_sae(cfg, run_cfg, device)
        dictionary = sae.net.get_dictionary().detach()
        alive = load_alive_mask(run_cfg, dictionary.shape[0], device)
        dictionary = dictionary[alive].float().cpu()
        alive_indices = alive.nonzero().squeeze(1).cpu().numpy()
        del sae
        if device.type == "cuda":
            torch.cuda.empty_cache()

        out_dir = Path(run_cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[{run_cfg.name}] {dictionary.shape[0]} alive features; computing {method.upper()} layout")
        layout = embed_dictionary(dictionary.numpy(), method=method, seed=seed)

        covered = {}
        for column, vocab in vocabs.items():
            if dictionary.shape[1] != vocab["emb"].shape[1]:
                raise ValueError(f"[{run_cfg.name}] Dictionary dim {dictionary.shape[1]} != "
                                 f"text embedding dim {vocab['emb'].shape[1]}.")
            top_sims, feature_rank, feature_term = topn_term_alignment(dictionary, vocab["emb"], n_needed)
            topn_sims[column][run_cfg.name] = top_sims
            if unc_n is not None:
                covered[column] = (feature_rank > 0) & (feature_rank <= unc_n)
            np.savez(
                out_dir / f"topn_text_alignment_{column}.npz",
                texts=np.asarray(vocab["texts"]),
                top_sims=top_sims,
                feature_rank=feature_rank,
                feature_term=feature_term,
                alive_indices=alive_indices,
            )

            # Dictionary embedding: colour = class of the best-matching vocabulary entry.
            for n in emb_n_list:
                in_top = (feature_rank > 0) & (feature_rank <= n)
                feature_class = np.where(in_top, vocab["classes"][np.clip(feature_term, 0, None)], "")
                log.info(f"[{run_cfg.name}] '{column}' top-{n}: {in_top.sum()} features "
                         f"{pd.Series(feature_class[in_top]).value_counts().to_dict()}")
                entry = "term" if column == "term" else column.replace("_", " ")
                ax, _ = plot_dictionary_embedding_topn(
                    dictionary.numpy(),
                    feature_class=feature_class,
                    class_order=class_order,
                    n_max=n,
                    method=method,
                    emb=layout,
                    legend_title=f"Class of best-matching {entry}\n(top-{n} features)",
                )
                save_figure(ax.figure, out_dir / f"dictionary_embedding_{column}_top{n}")

        if unc_n is not None:
            n_alive = dictionary.shape[0]
            union = np.zeros(n_alive, dtype=bool)
            for column in coverage_columns:
                union |= covered[column]
            for column, mask in [*covered.items(), ("union(" + "+".join(coverage_columns) + ")", union)]:
                n_cov = int(mask.sum())
                coverage_rows.append({"sae": run_cfg.name, "vocabulary": column, "top_n": unc_n,
                                      "n_alive": n_alive, "n_covered": n_cov,
                                      "frac_covered": n_cov / n_alive, "n_uncovered": n_alive - n_cov})
                log.info(f"[{run_cfg.name}] top-{unc_n} coverage '{column}': {n_cov}/{n_alive} "
                         f"({100 * n_cov / n_alive:.1f}%) features covered")

            if unc_emb is not None:
                if dictionary.shape[1] != unc_emb.shape[1]:
                    raise ValueError(f"[{run_cfg.name}] Dictionary dim {dictionary.shape[1]} != "
                                     f"uncovered vocab dim {unc_emb.shape[1]}.")
                left_out = ~union
                cos = _cosine_similarity_matrix(dictionary[left_out], unc_emb)  # (n_left_out, n_terms)
                if cos.shape[0] > 0:
                    best_sim, best_idx = cos.max(dim=1)
                    best_sim, best_idx = best_sim.numpy(), best_idx.numpy()
                else:
                    best_sim, best_idx = np.zeros(0), np.zeros(0, dtype=np.int64)
                uncovered_sims[run_cfg.name] = best_sim
                pd.DataFrame({
                    "feature": alive_indices[left_out],
                    "term": np.asarray(unc_terms, dtype=object)[best_idx],
                    "cosine": best_sim,
                }).sort_values("cosine", ascending=False).to_csv(
                    out_dir / f"uncovered_feature_terms_top{unc_n}.csv", index=False)
                if len(best_sim):
                    log.info(f"[{run_cfg.name}] {len(best_sim)} uncovered features: mean closest-term "
                             f"similarity {best_sim.mean():.4f}, {len(np.unique(best_idx))} distinct terms")

    # Comparison of ranked top-n alignment across SAEs and n: one figure per vocabulary column.
    out_dirs = [Path(r.output_dir) for r in cfg.saes]
    if cfg.get("output_dir"):
        out_dirs.append(Path(cfg.output_dir))
    out_dirs = list(dict.fromkeys(out_dirs))

    for column, sims_by_sae in topn_sims.items():
        col_n_list = [n for n in n_list if all(n <= s.shape[1] for s in sims_by_sae.values())]
        for name, sims in sims_by_sae.items():
            for n in col_n_list:
                log.info(f"[{name}] '{column}' mean top-{n} similarity: {sims[:, :n].mean(axis=1).mean():.4f}")
        ax = plot_topn_alignment_comparison(sims_by_sae, col_n_list)
        if column != "term":
            ax.set_xlabel(f"Vocabulary entries ({column}), ranked by similarity")
        for out_dir in out_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
            save_figure(ax.figure, out_dir / f"text_alignment_distribution_topn_{column}", close=False)
        plt.close(ax.figure)

    if unc_n is not None:
        coverage_df = pd.DataFrame(coverage_rows)
        log.info(f"Feature coverage at top-{unc_n}:\n{coverage_df.to_string(index=False)}")
        for out_dir in out_dirs:
            coverage_df.to_csv(out_dir / f"feature_coverage_top{unc_n}.csv", index=False)
        if uncovered_sims:
            ax = plot_uncovered_alignment(uncovered_sims, unc_n)
            for out_dir in out_dirs:
                save_figure(ax.figure, out_dir / f"uncovered_text_alignment_top{unc_n}", close=False)
            plt.close(ax.figure)
    log.info(f"Saved comparison plots to {[str(d) for d in out_dirs]}")


if __name__ == "__main__":
    evaluate_text_alignment_compare()
