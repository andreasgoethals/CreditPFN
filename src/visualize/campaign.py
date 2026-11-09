"""Notebook page objects: a figure, its paper caption and its tables, saved and displayed in order."""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style
from src.visualize.reporting import NotebookReport, page_text

__all__ = ["NotebookReport", "Page", "show"]


@dataclass
class Page:
    name: str
    figure: object
    caption: str
    tables: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        style.finish_figure(self.figure)
        # Keep the object for explicit inline display/export without accumulating
        # pyplot managers (or implicit duplicate notebook renders) across pages.
        plt.close(self.figure)


def show(sink, pages, *, track=None):
    """Save and display each page immediately, releasing it before the next section."""
    from IPython.display import Markdown, display
    pages = list(pages)
    if not pages:
        print("No measurements available for this section yet.")
        if getattr(sink, "report", None) is not None:
            sink.report._pending.append("No measurements available for this figure group.")
    for page in pages:
        if track is not None:
            if track not in ("pd", "lgd"):
                raise ValueError("Figure track must be pd or lgd")
            page.name = f"{track}_{page.name}"
            title = page.figure._suptitle
            if title is not None and not title.get_text().startswith(track.upper()):
                title.set_text(f"{track.upper()}: {title.get_text()}")
            if not page.caption.startswith(track.upper()):
                page.caption = f"{track.upper()}. {page.caption}"
        if not page.caption.strip():
            raise ValueError(f"Figure {page.name} needs a manuscript caption")
        if getattr(sink, "report", None) is not None:
            sink.report._pending.append(page_text(page))
        metadata = {"provenance": page.provenance} if page.provenance else {}
        sink.save(page.figure, page.name, caption=page.caption, **metadata)
        display(page.figure)
        display(Markdown(page.caption))
        plt.close(page.figure)


def _subplots(title, ncols=1, *, sharey=False):
    fig, axes = plt.subplots(1, ncols, figsize=style.figsize(style.WIDTH_FULL, style.PANEL_RATIO),
                             sharey=sharey, squeeze=False, layout="constrained")
    fig.suptitle(title)
    return fig, axes[0]


def _heatmap(matrix, title, *, diverging=True, limits=None, label="Effect"):
    fig, axes = _subplots(title)
    ax = axes[0]
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if limits is None:
        extent = max(float(np.max(np.abs(finite))) if len(finite) else 0, 1e-8)
        limits = (-extent, extent) if diverging else (0, max(extent, 1))
    cmap = plt.get_cmap(style.CMAP_DIVERGING if diverging else style.CMAP_SEQUENTIAL).copy()
    cmap.set_bad(style.color("annotation"), alpha=0.15)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=limits[0], vmax=limits[1], aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [str(x) for x in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), [str(x) for x in matrix.index])
    ax.grid(False)
    if values.size <= style.PAGE_ROWS * style.PAGE_COLUMNS:
        for y, x in np.ndindex(values.shape):
            if np.isfinite(values[y, x]):
                norm = (values[y, x] - limits[0]) / max(limits[1] - limits[0], 1e-12)
                red, green, blue, _ = cmap(norm)
                ink = style.GRID_TEXT_LIGHT if .299*red + .587*green + .114*blue < .5 else style.GRID_TEXT_DARK
                ax.text(x, y, f"{values[y,x]:.3g}", ha="center", va="center", fontsize=style.ANNOTATION_SIZE, color=ink)
    bar = fig.colorbar(image, ax=ax, label=label, shrink=.8)
    if not diverging and limits == (0, 1) and set(finite).issubset({0., 1.}):
        bar.set_ticks([0, 1])
    return fig
