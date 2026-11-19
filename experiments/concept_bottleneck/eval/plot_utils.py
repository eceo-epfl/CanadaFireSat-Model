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


# Categorical slots (fixed order) used to identify SAEs across comparison plots.
SAE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
# Single-hue blue ramp, dark (rank 1) -> light (rank n_max).
RANK_RAMP = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]

# Print-ready style for paper figures (applied locally via plt.rc_context).
PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "axes.labelcolor": "#333333",
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "legend.fontsize": 13,
    "legend.title_fontsize": 13,
    "text.color": "#222222",
    "pdf.fonttype": 42,   # embed TrueType fonts (required by most venues)
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def _shade(color, t, max_fade=0.6):
    """Blend ``color`` towards white by ``t * max_fade`` (t in [0, 1])."""
    from matplotlib.colors import to_rgb
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb * (1 - max_fade * t) + max_fade * t)


def plot_topn_alignment_comparison(topn_sims, n_list, ax=None):
    """
    Ranked per-term mean cosine similarity over the top-n most similar SAE features.

    One hue per SAE; within an SAE, darker / thicker lines are smaller n. A light band spans
    the SAE's curves (top-min(n) to top-max(n)). The legend is a table: SAEs as columns, n as rows.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    n_list = list(n_list)
    k = len(n_list)
    with plt.rc_context(PAPER_RC):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.6, 4.8))

        ax.grid(axis="y", color="#e6e6e6", linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)

        all_vals, handles, labels = [], [], []
        for i, (name, sims) in enumerate(topn_sims.items()):
            base = SAE_COLORS[i % len(SAE_COLORS)]
            curves = [np.sort(sims[:, :n].mean(axis=1))[::-1] for n in n_list]
            x = np.linspace(0, 1, len(curves[0]))
            ax.fill_between(x, curves[0], curves[-1], color=base, alpha=0.08, linewidth=0, zorder=1)

            handles.append(Line2D([], [], linestyle="none"))
            labels.append(name)
            for j, (n, curve) in enumerate(zip(n_list, curves)):
                t = 0.0 if k == 1 else j / (k - 1)
                few = len(curve) <= 30  # show the individual entries when the vocabulary is small
                (h,) = ax.plot(x, curve, color=_shade(base, t), linewidth=3.0 - 1.2 * t,
                               solid_capstyle="round", zorder=3 + k - j,
                               marker="o" if few else None, markersize=6 - 1.5 * t,
                               markeredgecolor="white", markeredgewidth=0.8)
                handles.append(h)
                labels.append(f"top-{n}")
                all_vals.append(curve)

        all_vals = np.concatenate(all_vals)
        lo = max(0.0, np.floor(all_vals.min() * 10) / 10 - 0.05)
        hi = min(1.0, np.ceil(all_vals.max() * 10) / 10 + 0.02)
        ax.set_ylim(lo, hi)
        ax.set_xlim(0, 1)
        ax.margins(x=0)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.spines["left"].set_visible(False)

        ax.set_xlabel("Vocabulary terms, ranked by similarity")
        ax.set_ylabel("Mean cosine similarity\nof the top-$n$ features")

        leg = ax.legend(handles, labels, ncol=len(topn_sims), loc="upper left",
                        bbox_to_anchor=(1.02, 1.0), frameon=False, handlelength=2.0,
                        columnspacing=1.4, borderaxespad=0.0, labelspacing=0.45)
        # Column headers: bold SAE names with no marker.
        for txt, h in zip(leg.get_texts(), leg.legend_handles):
            if txt.get_text() in topn_sims:
                txt.set_fontweight("bold")
                txt.set_position((-30, 0))
    return ax


# Land-cover-conventional hues, keyed by keyword found in the class name.
LANDCOVER_COLORS = {
    "water": "#2a78d6",
    "forest": "#1b6e2e",
    "shrub": "#a6761d",
    "grass": "#8cc63f",
    "crop": "#eda100",
    "wetland": "#1baf9f",
    "barren": "#c2a77f",
    "snow": "#7b4fbf",
    "ice": "#7b4fbf",
    "urban": "#d1495b",
}
FALLBACK_CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                         "#7b4fbf", "#8c6d4f", "#d1495b", "#17becf"]


def class_color_map(class_order):
    """Map each class to a hue: land-cover convention when recognised, fixed fallback order otherwise."""
    colors, used = {}, set()
    for cls in class_order:
        key = next((k for k in LANDCOVER_COLORS if k in cls.lower()), None)
        if key is not None and LANDCOVER_COLORS[key] not in used:
            colors[cls] = LANDCOVER_COLORS[key]
            used.add(colors[cls])
    fallback = iter(c for c in FALLBACK_CLASS_COLORS if c not in used)
    for cls in class_order:
        if cls not in colors:
            colors[cls] = next(fallback, NEUTRAL)
    return colors


def embed_dictionary(dictionary_vectors, method="umap", seed=0):
    """2-D layout of dictionary atoms (UMAP or t-SNE)."""
    if method == "umap":
        reducer = umap.UMAP(random_state=seed)
    else:
        reducer = TSNE(random_state=seed, init="pca")
    return reducer.fit_transform(dictionary_vectors)


def plot_dictionary_embedding_topn(
    dictionary_vectors,   # (n_features, d)
    feature_class,        # (n_features,) class of the best-matching term, "" if not in any top-n
    class_order,          # classes in legend order
    n_max,
    method="umap",
    ax=None,
    seed=0,
    title=None,
    emb=None,
    legend_title=None,
):
    """
    Embed dictionary atoms. Features within some term's top-``n_max`` take the colour of that
    term's class; the rest are grey. Pass ``emb`` (n_features, 2) to reuse a precomputed layout.
    """
    from matplotlib.lines import Line2D

    if emb is None:
        emb = embed_dictionary(dictionary_vectors, method=method, seed=seed)

    feature_class = np.asarray(feature_class).astype(str)
    colors = class_color_map(class_order)

    with plt.rc_context(PAPER_RC):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8.0, 6.0))

        assigned = feature_class != ""
        ax.scatter(emb[~assigned, 0], emb[~assigned, 1], s=7, color="#d0d0d0", alpha=0.6,
                   linewidth=0, zorder=1, rasterized=True)

        present = [c for c in class_order if (feature_class == c).any()]
        for c in present:
            m = feature_class == c
            ax.scatter(emb[m, 0], emb[m, 1], s=30, color=colors[c], edgecolor="white",
                       linewidth=0.5, alpha=0.95, zorder=2, rasterized=True)

        handles = [Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=colors[c],
                          markeredgecolor="white", label=c) for c in present]
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="#d0d0d0",
                              markeredgecolor="none", label=f"not in a top-{n_max}"))
        ax.legend(handles=handles, title=legend_title or f"Class of best-matching term\n(top-{n_max} features)",
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, alignment="left",
                  handletextpad=0.2, borderaxespad=0.0, labelspacing=0.5)

        # Embedding coordinates carry no units: drop ticks and spines.
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel(f"{method.upper()} 1", color="#777777")
        ax.set_ylabel(f"{method.upper()} 2", color="#777777")
        if title:
            ax.set_title(title, loc="left", fontweight="bold")
    return ax, emb
