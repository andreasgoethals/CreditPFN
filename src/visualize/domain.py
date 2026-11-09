"""Figures for the credit vs non-credit comparison (notebooks/general/03_domain_comparison)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.domain_comparison import TABLE_PROPERTIES, TARGET_PROPERTIES, normalised_target
from src.visualize import style

LABELS = {"log10_rows": "Rows (log10)", "features": "Features", "categorical_share": "Categorical share",
          "missing_share": "Missing share", "median_abs_skew": "Median |skew| of features",
          "heavy_tail_share": "Share heavy-tailed features", "integer_share": "Share integer features",
          "max_abs_spearman": "Max |Spearman| feature–target", "top5_abs_spearman": "Top-5 |Spearman|",
          "positive_rate": "Minority-class rate", "bounded_unit_interval": "Target within [0, 1]",
          "share_at_min": "Target mass at minimum", "share_at_max": "Target mass at maximum",
          "target_skew": "Target skewness", "target_bimodality": "Target bimodality",
          "unique_share": "Distinct target values / rows"}


def _scatter(ax, frame, x, y):
    for group, sub in frame.groupby("group"):
        ax.scatter(sub[x], sub[y], s=style.POINT_SIZE, color=style.DOMAIN_COLORS[group],
                   alpha=style.POINT_ALPHA, label=style.DOMAIN_LABELS[group], edgecolors="none")


def embedding_figure(embeddings: dict, title: str):
    """One panel per task: first two principal components of the standardised feature and
    target properties of every table, coloured by table group."""
    tracks = [t for t in ("pd", "lgd") if t in embeddings and not embeddings[t].empty]
    fig, axes = plt.subplots(1, len(tracks), squeeze=False, figsize=style.figsize(style.WIDTH_FULL, 0.45))
    for ax, track in zip(axes[0], tracks):
        emb = embeddings[track]
        _scatter(ax, emb, "pc1", "pc2")
        ax.set_xlabel(f"PC1 ({emb.pc1_explained.iloc[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({emb.pc2_explained.iloc[0]:.0%})")
        ax.set_title("PD versus binary non-credit" if track == "pd" else "LGD versus regression non-credit")
    axes[0, 0].legend()
    fig.suptitle(title)
    return fig


def property_figure(properties: pd.DataFrame, track: str, names=None):
    names = list(names or (*TARGET_PROPERTIES[track], *TABLE_PROPERTIES))
    columns = 4
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=style.rows_figsize(rows), squeeze=False)
    groups = [g for g in ("credit", "retention", "domain_control") if g in set(properties.group)]
    rng = np.random.default_rng(0)
    for ax, name in zip(axes.ravel(), names):
        for i, group in enumerate(groups):
            values = properties.loc[properties.group == group, name].astype(float)
            ax.scatter(i + rng.uniform(-style.JITTER_WIDTH, style.JITTER_WIDTH, len(values)), values,
                       s=style.TABLE_POINT_SIZE, color=style.DOMAIN_COLORS[group], alpha=style.POINT_ALPHA)
            ax.hlines(values.median(), i - 0.3, i + 0.3, color=style.DOMAIN_COLORS[group], lw=style.MEAN_CURVE_LINE)
        ax.set_xticks(range(len(groups)), [style.DOMAIN_LABELS[g].split()[0] if g == "credit" else
                                           ("Retention" if g == "retention" else "Control") for g in groups])
        ax.set_title(LABELS.get(name, name), fontsize=style.ANNOTATION_SIZE)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    fig.suptitle(("PD" if track == "pd" else "LGD") + " tables versus non-credit tables, per property")
    return fig


def target_figure(tables_lgd: list[dict], properties_lgd: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.45))
    grid = np.linspace(0, 1, 201)
    for t in tables_lgd:
        z = np.sort(normalised_target(t["y"]))
        ecdf = np.searchsorted(z, grid, side="right") / len(z)
        credit = t["group"] == "credit"   # credit curves on top: they are the subject
        axes[0].plot(grid, ecdf, color=style.DOMAIN_COLORS[t["group"]],
                     lw=style.TABLE_CURVE_LINE * (1.6 if credit else 1), alpha=0.9 if credit else 0.45,
                     zorder=3 if credit else 2)
    axes[0].set_xlabel("Target, rescaled to [0, 1] per table")
    axes[0].set_ylabel("Cumulative share of rows")
    axes[0].set_title("Target per table")
    _scatter(axes[1], properties_lgd, "share_at_min", "share_at_max")
    axes[1].set_xlabel("Share of rows at the minimum target")
    axes[1].set_ylabel("Share of rows at the maximum target")
    axes[1].set_title("Point masses at the bounds")
    axes[1].legend(frameon=False, fontsize=style.ANNOTATION_SIZE, loc="upper right")
    fig.suptitle("LGD targets versus non-credit regression targets")
    return fig
