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
