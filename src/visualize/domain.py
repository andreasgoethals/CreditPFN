"""Figures for the credit vs non-credit comparison (notebooks/general/03_domain_comparison)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.domain_comparison import normalised_target
from src.visualize import style

def _scatter(ax, frame, x, y):
    for group in style.DOMAIN_MARKERS:
        sub = frame[frame.group == group]
        if sub.empty:
            continue
        ax.scatter(sub[x], sub[y], s=style.POINT_SIZE, color=style.DOMAIN_COLORS[group],
                   marker=style.DOMAIN_MARKERS[group], alpha=style.POINT_ALPHA,
                   label=style.DOMAIN_LABELS[group], edgecolors="none")


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
        ax.set_title("PD and binary tables" if track == "pd" else "LGD and regression tables")
    axes[0, 0].legend()
    fig.suptitle(title)
    return fig


def property_figure(properties: dict[str, pd.DataFrame]):
    """Compare size matching, missingness and univariate signal on common scales by task."""
    names = {"log10_rows": "Rows (log10)", "missing_share": "Missing fraction",
             "max_abs_spearman": "Max. |Spearman ρ|"}
    fig, axes = plt.subplots(2, len(names), sharey="col", figsize=style.rows_figsize(2))
    rng = np.random.default_rng(0)
    for row, track in enumerate(("pd", "lgd")):
        frame = properties[track]
        axes[row, 0].set_ylabel(track.upper())
        for ax, (name, label) in zip(axes[row], names.items()):
            if row == 0:
                ax.set_title(label)
            for i, group in enumerate(style.DOMAIN_MARKERS):
                values = frame.loc[frame.group == group, name].astype(float).dropna()
                if values.empty:
                    continue
                ax.scatter(i + rng.uniform(-style.JITTER_WIDTH, style.JITTER_WIDTH, len(values)), values,
                           s=style.TABLE_POINT_SIZE, color=style.DOMAIN_COLORS[group],
                           marker=style.DOMAIN_MARKERS[group], alpha=style.POINT_ALPHA,
                           label=style.DOMAIN_LABELS[group])
                ax.hlines(values.median(), i - 0.3, i + 0.3, color=style.DOMAIN_COLORS[group],
                          lw=style.MEAN_CURVE_LINE)
            ax.set_xticks([])
            ax.set_xlim(-0.6, len(style.DOMAIN_MARKERS) - 0.4)
    axes[0, 0].legend()
    fig.suptitle("Credit tables, matched controls and retention tables")
    return fig


def target_figure(tables_lgd: list[dict], properties_lgd: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL, 0.45))
    grid = np.linspace(0, 1, 201)
    for t in tables_lgd:
        z = np.sort(normalised_target(t["y"]))
        ecdf = np.searchsorted(z, grid, side="right") / len(z)
        credit = t["group"] == "credit"   # credit curves on top: they are the subject
        axes[0].plot(grid, ecdf, color=style.DOMAIN_COLORS[t["group"]], linestyle=style.DOMAIN_LINES[t["group"]],
                     lw=style.TABLE_CURVE_LINE * (1.6 if credit else 1), alpha=0.9 if credit else 0.45,
                     zorder=3 if credit else 2)
    axes[0].set_xlabel("Target, rescaled to [0, 1]")
    axes[0].set_ylabel("Cumulative share of rows")
    axes[0].set_title("Target per table")
    _scatter(axes[1], properties_lgd, "share_at_min", "share_at_max")
    axes[1].set_xlabel("Share of rows at the minimum")
    axes[1].set_ylabel("Share at the maximum")
    axes[1].set_title("Point masses at the bounds")
    axes[1].legend(frameon=False, fontsize=style.ANNOTATION_SIZE, loc="upper right")
    fig.suptitle("LGD targets versus non-credit regression targets")
    return fig
