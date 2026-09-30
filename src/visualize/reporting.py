"""Publication figures and their complete, ordered text equivalents."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer

from src.data.dataset_names import display_frame, redact_private_names


def table_text(frame):
    """No pandas ellipses, row truncation or private identifiers in final summaries."""
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    frame = display_frame(frame.copy())
    frame.index = [redact_private_names(str(x)) for x in frame.index]
    frame.columns = [redact_private_names(str(x)) for x in frame.columns]
    return frame.to_string(index=True, max_rows=None, max_cols=None,
                           max_colwidth=None, float_format=lambda value: f"{value:.8g}",
                           na_rep="unavailable")


def figure_tables(fig):
    """Record every data artist, including histogram bins, as labelled numeric tables.

    Authors can attach semantic tables to fig._report_tables. Otherwise the
    tables record exactly the plotted coordinates with axis labels and categories.
    """
    tables = dict(getattr(fig, "_report_tables", {}))
    if tables:
        return tables
    fig.canvas.draw()  # Resolve tick labels before recording their categories.
    for index, ax in enumerate(fig.axes, 1):
        if not ax.get_visible() or ax.get_label() == "<colorbar>":
            continue
        prefix = f"Panel {index}" + (f": {ax.get_title()}" if ax.get_title() else "")
        xname, yname = ax.get_xlabel() or "x", ax.get_ylabel() or "y"
        xticks = dict(zip(ax.get_xticks(), (t.get_text() for t in ax.get_xticklabels())))
        yticks = dict(zip(ax.get_yticks(), (t.get_text() for t in ax.get_yticklabels())))
        for number, im in enumerate(ax.images, 1):
            array = np.ma.filled(im.get_array().astype(float), np.nan)
            if array.ndim == 2:
                tables[f"{prefix} / matrix {number}"] = pd.DataFrame(
                    array, index=[yticks.get(i, str(i)) for i in range(array.shape[0])],
                    columns=[xticks.get(i, str(i)) for i in range(array.shape[1])])
        for number, container in enumerate(ax.containers, 1):
            if isinstance(container, BarContainer):
                horizontal = container.orientation == "horizontal"
                rows = []
                for bar in container.patches:
                    low = bar.get_y() if horizontal else bar.get_x()
                    extent = bar.get_height() if horizontal else bar.get_width()
                    center = low + extent / 2
                    rows.append(dict(category=(yticks if horizontal else xticks).get(center, center),
                                     bin_start=low, bin_end=low + extent,
                                     value=bar.get_width() if horizontal else bar.get_height(),
                                     stack_start=bar.get_x() if horizontal else bar.get_y()))
                tables[f"{prefix} / bars {number}: {container.get_label()}"] = pd.DataFrame(rows)
        for number, line in enumerate(ax.lines, 1):
            if line.get_transform() != ax.transData:
                continue
            x, y = line.get_data()
            if not len(x):
                continue
            label = line.get_label()
            label = label if label and not label.startswith("_") else f"series {number}"
            tables[f"{prefix} / {label}"] = pd.DataFrame({xname: x, yname: y})
        for number, points in enumerate(ax.collections, 1):
            if not isinstance(points, PathCollection):
                continue
            offsets = np.ma.filled(points.get_offsets(), np.nan)
            if not len(offsets):
                continue
            frame = pd.DataFrame(offsets, columns=[xname, yname])
            if yticks and all(v in yticks for v in offsets[:, 1]):
                frame.insert(0, "y category", [yticks[v] for v in offsets[:, 1]])
            if xticks and all(v in xticks for v in offsets[:, 0]):
                frame.insert(0, "x category", [xticks[v] for v in offsets[:, 0]])
            tables[f"{prefix} / points {number}"] = frame
    return tables


def page_text(page):
    tables = page.tables or figure_tables(page.figure)
    if not tables:
        raise ValueError(f"Figure {page.name} has no text-equivalent data")
    parts = [f"Figure: {page.name}", page.caption]
    for label, frame in tables.items():
        parts.extend((str(label), table_text(frame)))
    return "\n".join(parts)


@dataclass
class NotebookReport:
    """Capture figures and displayed tables once, then print them in section order."""
    title: str
    sink: object = None
    sections: list = field(default_factory=list)
    _pending: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.sink is not None:
            self.sink.report = self

    def table(self, label, frame):
        from IPython.display import display
        safe = display_frame(frame)
        display(safe)
        self._pending.append(f"Table: {label}\n{table_text(safe)}")

    def add(self, title, text):
        self.sections.append((title, "\n\n".join([str(text), *self._pending])))
        self._pending.clear()

    def summary(self, sink):
        if self._pending:
            raise ValueError("Displayed evidence has not been assigned to a notebook section")
        parts = [self.title]
        for title, text in self.sections:
            parts.extend(("", title, text))
        parts.extend(("", sink.summary()))
        return redact_private_names("\n".join(parts))

