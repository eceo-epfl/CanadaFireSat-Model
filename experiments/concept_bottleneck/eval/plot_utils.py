import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.manifold import TSNE
import umap


PRIMARY = "#1f4e8c"
ACCENT = "#d1495b"
NEUTRAL = "#8c8c8c"



def plot_alignment_distribution(
    text_all_sim,
    ax=None,
    label=None,
    linestyle="-",
    color="#1f4e8c",
    title="Ranked Cosine Similarity of Assigned Concepts",
    mean_line=True,
    median_line=True,
    highlight_top=10,
):
    """
    Paper-style ranked cosine similarity plot with richer visual structure.

    Parameters
    ----------
    text_all_sim : array-like
        1D array of cosine similarities, one per concept.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    label : str, optional
        Label for the curve.
    color : str
        Main plot color.
    title : str
        Figure title.
    mean_line : bool
        Show mean similarity.
    median_line : bool
        Show median similarity.
    highlight_top : int
        Number of top-ranked concepts to highlight.
    """

    sims = np.sort(np.asarray(text_all_sim))[::-1]
    x = np.linspace(0, 1, len(sims))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    mean_val = np.mean(sims)
    median_val = np.median(sims)

    # ---------------------------------------------------------
    # Subtle background regions
    # ---------------------------------------------------------
    # Region above the mean
    ax.axhspan(
        mean_val,
        sims.max(),
        color=color,
        alpha=0.035,
        zorder=0
    )

    # ---------------------------------------------------------
    # Filled area underneath curve
    # ---------------------------------------------------------
    ax.fill_between(
        x,
        sims,
        0,
        color=color,
        alpha=0.10,
        linewidth=0,
        zorder=1
    )

    # ---------------------------------------------------------
    # Main ranked curve
    # ---------------------------------------------------------
    ax.plot(
        x,
        sims,
        color=color,
        linestyle=linestyle,
        linewidth=2.2,
        solid_capstyle="round",
        label=label,
        zorder=3,
        alpha=0.5
    )

    # ---------------------------------------------------------
    # Highlight top concepts
    # ---------------------------------------------------------
    if highlight_top is not None and highlight_top > 0:
        n = min(highlight_top, len(sims))

        ax.scatter(
            x[:n],
            sims[:n],
            s=24,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=4
        )

    # ---------------------------------------------------------
    # Mean line
    # ---------------------------------------------------------
    if mean_line:
        ax.axhline(
            mean_val,
            color=color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            zorder=2
        )

        ax.text(
            0.99,
            mean_val,
            f" mean = {mean_val:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=color,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=1.5
            )
        )

    # ---------------------------------------------------------
    # Median line
    # ---------------------------------------------------------
    if median_line:
        ax.axhline(
            median_val,
            color="0.35",
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
            zorder=2
        )

        ax.text(
            0.99,
            median_val,
            f" median = {median_val:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="top",
            fontsize=8.5,
            color="0.35",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=1.5
            )
        )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------
    ax.set_xlabel(
        "Relative concept rank",
        fontsize=10
    )

    ax.set_ylabel(
        r"Cosine similarity $v_c$",
        fontsize=10
    )

    ax.set_title(
        title,
        fontsize=12,
        fontweight="semibold",
        pad=12
    )

    # ---------------------------------------------------------
    # Axes
    # ---------------------------------------------------------
    ax.set_xlim(-0.05, 1.05)

    # Give the plot a little breathing room vertically
    ax.set_ylim(0, 1.)

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------
    ax.grid(
        axis="y",
        alpha=0.18,
        linewidth=0.7
    )

    ax.grid(
        axis="x",
        visible=False
    )

    # ---------------------------------------------------------
    # Cleaner spines
    # ---------------------------------------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)

    # ---------------------------------------------------------
    # Tick styling
    # ---------------------------------------------------------
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=9,
        length=3
    )

    # ---------------------------------------------------------
    # Optional top-concepts annotation
    # ---------------------------------------------------------
    if highlight_top is not None and highlight_top > 0:
        n = min(highlight_top, len(sims))

        ax.annotate(
            f"Top {n}",
            xy=(x[n - 1], sims[n - 1]),
            xytext=(15, -25),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                alpha=0.6,
                lw=0.8
            )
        )

    return ax


def plot_concept_duplication(
    names,
    sim=None,
    top_n=25,
    ax=None,
    annotate=True
):
    """
    names: array-like (len = n_concepts) of assigned vocabulary strings.
    sim:   optional array-like (len = n_concepts), cosine alignment score
           per concept, used to color-code the bars by mean alignment
           of the duplicate group.
    top_n: number of most-duplicated names to display.
    """
    s = pd.Series(names, name="name")
    counts = s.value_counts()
    dup = counts[counts > 1].sort_values(ascending=False).head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 0.28 * len(dup) + 1))

    y = np.arange(len(dup))[::-1]

    if sim is not None:
        sim = np.asarray(sim)
        mean_sim = {
            name: sim[s.values == name].mean() for name in dup.index
        }
        colors = plt.cm.viridis(
            (np.array([mean_sim[n] for n in dup.index]) - np.nanmin(sim))
            / (np.nanmax(sim) - np.nanmin(sim) + 1e-8)
        )
    else:
        colors = PRIMARY

    ax.barh(y, dup.values, color=colors, edgecolor="white", linewidth=0.4, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(dup.index)
    ax.set_xlabel("Number of concepts assigned this caption")
    ax.set_title(f"Top {len(dup)} duplicated concept names", fontsize=11)
    ax.invert_yaxis()

    if sim is not None:
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=mpl.colors.Normalize(vmin=np.nanmin(sim), vmax=np.nanmax(sim)),
        )
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
        cbar.set_label(r"Mean cosine similarity $v_c$", fontsize=9)

    if annotate:
        n_total = len(s)
        n_unique = (counts == 1).sum()
        n_captions = len(counts)
        pct_unique = 100 * n_unique / n_captions
        ax.text(
            0.97, 0.97,
            (f"$n$ concepts = {n_total}\n"
                f"$n$ unique captions = {n_captions}\n"
                f"{pct_unique:.1f}% of captions used once"),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor=NEUTRAL, linewidth=0.6, alpha=0.9),
        )

    ax.set_title("Most-duplicated captions")

    return ax, dup


def plot_dictionary_embedding(
    dictionary_vectors,   # (n_concepts, d) — the W_D rows, pc
    names,
    top_n_captions=12,
    method="umap",        # "umap" or "tsne"
    ax=None,
    seed=0,
):
    if method == "umap":
        reducer = umap.UMAP(random_state=seed)
    else:
        reducer = TSNE(random_state=seed, init="pca")

    emb = reducer.fit_transform(dictionary_vectors)

    s = pd.Series(names)
    top_caps = s.value_counts().head(top_n_captions).index
    is_top = s.isin(top_caps).values

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))

    ax.scatter(emb[~is_top, 0], emb[~is_top, 1],
               s=6, color=NEUTRAL, alpha=0.35, label="other concepts")

    cmap = plt.cm.get_cmap("tab20", len(top_caps))
    for i, cap in enumerate(top_caps):
        mask = (s.values == cap)
        ax.scatter(emb[mask, 0], emb[mask, 1], s=28, color=cmap(i),
                   edgecolor="white", linewidth=0.4, label=cap)

    ax.set_xlabel(f"{method.upper()} first dimension")
    ax.set_ylabel(f"{method.upper()} second dimension")
    ax.set_title("Dictionary-vector geometry of duplicated captions", fontsize=11)
    ax.legend(fontsize=7, markerscale=1.2, ncol=2, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))

    return ax, emb
