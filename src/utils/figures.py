"""Per-notebook PDF figure sink.

Every notebook in ``notebooks/`` drops the figures it produces under
``output/figures/<notebook_name>/*.pdf``. The folder is recreated
from scratch on each notebook run so stale figures never accumulate.

Usage from a notebook
---------------------

    from src.utils.figures import open_figure_sink
    sink = open_figure_sink("1.0_training_visualization")

    sink.save(plot_loss_curve(...),        "01_loss_curve")
    sink.save(plot_trial_dashboard(...),   "02_trial_dashboard")
    ...

``sink.save(fig, name)`` writes the figure to PDF and **returns**
the same figure so Jupyter's inline display still renders the
image. ``fig=None`` (e.g. the helper returned a stub when no data
was on disk) is silently ignored.

Why this is a class, not a free function
----------------------------------------
``open_figure_sink`` wipes the per-notebook directory **once** at
construction. That way the very first ``save`` call already starts
from a clean slate, every subsequent ``save`` just adds to it, and
the next time the user re-runs the notebook the wipe happens again
at the top. A free function would either need a module-level cache
or would re-wipe on every call.

Path conventions
----------------
* ``output/figures/<notebook_name>/`` resolves via
  :func:`src.utils.paths.resolve_output_path`, so the figures land
  in the same place as the rest of the code's outputs (durable
  storage on VSC, repo root locally).
* ``notebook_name`` is the slug, with dots/spaces normalised to
  underscores — see :func:`_normalise_name`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from src.utils.paths import resolve_output_path

if TYPE_CHECKING:                          # pragma: no cover
    from matplotlib.figure import Figure   # noqa: F401

LOGGER = logging.getLogger(__name__)


_NORMALISE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _normalise_name(name: str) -> str:
    """Collapse spaces / dots / weird chars into a stable directory name.

    ``"1.0 Training Visualization"`` → ``"1.0_Training_Visualization"``.
    Leading/trailing separators are stripped.
    """
    return _NORMALISE_RE.sub("_", name.strip()).strip("._-")


class FigureSink:
    """Write per-notebook PDFs to ``output/figures/<notebook>/``.

    Attributes
    ----------
    notebook_name
        Original name passed to :func:`open_figure_sink` (informational).
    dir
        Absolute path to the per-notebook directory.
    saved
        List of ``Path`` objects written so far, in call order.
    """

    def __init__(self, notebook_name: str) -> None:
        self.notebook_name = notebook_name
        slug = _normalise_name(notebook_name)
        if not slug:
            raise ValueError(
                f"notebook_name={notebook_name!r} normalises to empty string"
            )
        self.dir: Path = resolve_output_path("output/figures") / slug
        self._wipe_and_prepare()
        self.saved: list[Path] = []
        # Auto-name counter for `save(fig)` calls without an explicit name.
        self._counter: int = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _wipe_and_prepare(self) -> None:
        """Recreate the per-notebook directory empty.

        We don't ``shutil.rmtree`` because the directory may also hold
        user-added scratch files; we only delete the ``*.pdf`` we own.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        for p in self.dir.glob("*.pdf"):
            try:
                p.unlink()
            except OSError as exc:                                  # pragma: no cover
                LOGGER.warning("could not remove %s: %s", p, exc)

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #

    def save(self, fig: "Figure | None", name: str | None = None) -> "Figure | None":
        """Persist ``fig`` as ``<dir>/<name>.pdf`` and return ``fig``.

        Parameters
        ----------
        fig
            The matplotlib Figure object. ``None`` is accepted and
            returned unchanged — this lets you wrap a plot helper that
            may opt out of plotting in some edge case.
        name
            Filename stem (no extension). ``None`` ⇒ auto-numbered as
            ``figure_NN``. The stem is sanitised via
            :func:`_normalise_name`.

        Returns
        -------
        The same Figure (or None) — so Jupyter's inline backend still
        renders the figure cell-by-cell.
        """
        if fig is None:
            return None
        self._counter += 1
        if name is None:
            stem = f"figure_{self._counter:02d}"
        else:
            stem = _normalise_name(name) or f"figure_{self._counter:02d}"
        out = self.dir / f"{stem}.pdf"
        try:
            fig.savefig(out, format="pdf", bbox_inches="tight")
        except Exception as exc:                                    # pragma: no cover
            LOGGER.warning("could not write %s: %s", out, exc)
            return fig
        self.saved.append(out)
        return fig

    def summary(self) -> str:
        """Short one-line summary of how many figures were written."""
        if not self.saved:
            return f"FigureSink({self.notebook_name}): no figures saved yet."
        return (
            f"FigureSink({self.notebook_name}): "
            f"{len(self.saved)} figure(s) saved to {self.dir}"
        )

    # Pretty-print so a bare `sink` cell at the end of a notebook
    # surfaces the output directory and file count.
    def __repr__(self) -> str:                                      # pragma: no cover
        return self.summary()


def open_figure_sink(notebook_name: str) -> FigureSink:
    """Convenience entrypoint — see :class:`FigureSink`."""
    return FigureSink(notebook_name)
