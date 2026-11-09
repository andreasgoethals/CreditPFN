"""One publication tree, with disjoint notebook ownership and paper captions.

PDFs: output CreditPFN/figures/<group>/<group>__<notebook>__<NN>_<name>.pdf
Metadata: output CreditPFN/figures/<group>/_metadata/<notebook>.json

Constructing a saver removes that notebook's previous figures; other notebooks and
scientific evidence are never cleared by it.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.paths import REPO_ROOT
from src.visualize.paths import (
    PUBLICATION_GROUPS,
    figure_prefix,
    figures_dir,
    notebook_figures_dir,
    notebook_parts,
)

#: Vector already, but heatmaps and scatter clouds inside a PDF rasterise, so it still needs a
#: print DPI.
DPI = 300


def manifest_path(notebook: str) -> Path:
    _, name = notebook_parts(notebook)
    return notebook_figures_dir(notebook) / "_metadata" / f"{name.as_posix()}.json"


def read_manifest(notebook: str) -> list[dict]:
    """What this notebook drew last time it ran, in order. `[]` if it never has."""
    path = manifest_path(notebook)
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A run killed mid-write leaves truncated JSON: a missing manifest, not a crash.
        return []


def clear(notebook: str) -> int:
    """Remove this notebook's PDFs and metadata, preserving other notebooks and run evidence."""
    removed = 0
    folder = notebook_figures_dir(notebook)
    for path in folder.glob(figure_prefix(notebook) + "*.pdf"):
        _guard(path, folder)
        if path.is_file():
            path.unlink()
            removed += 1
    path = manifest_path(notebook)
    if path.is_file():
        path.unlink()
        removed += 1
    return removed


def _prune_empty(folder: Path) -> None:
    """Remove only empty generated directories, bounded by the output root."""
    root = figures_dir().parent.resolve()
    folder = folder.resolve()
    if not folder.is_relative_to(root) or folder == root:
        return
    while folder != root:
        try:
            folder.rmdir()
        except PermissionError:
            # Downloaded Windows directories can retain the ReadOnly attribute.
            # Change it only for an empty, non-symlink directory inside output.
            if os.name != "nt" or folder.is_symlink() or any(folder.iterdir()):
                break
            try:
                folder.chmod(folder.stat().st_mode | stat.S_IWRITE)
                folder.rmdir()
            except OSError:
                break
        except OSError:
            break
        folder = folder.parent


def clear_collection(notebooks: tuple[str, ...]) -> None:
    """Full reruns also retire figures from notebooks that were renamed or removed."""
    owners = set(notebooks)
    for group in PUBLICATION_GROUPS:
        root = figures_dir() / group / "_metadata"
        owners.update(group + "/" + p.relative_to(root).with_suffix("").as_posix()
                      for p in root.rglob("*.json"))
    for name in sorted(owners):
        clear(name)
    (figures_dir() / "CAPTIONS.md").unlink(missing_ok=True)
    # Only the full pre-execution cleanup may prune shared publication folders: an individual
    # worker must not remove an empty directory another saver is about to write into.
    for folder in sorted(figures_dir().rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if folder.is_dir():
            _prune_empty(folder)


def owned_pdfs(notebook: str) -> list[Path]:
    return sorted(notebook_figures_dir(notebook).glob(figure_prefix(notebook) + "*.pdf"))


class FigureSaver:
    """Saves every figure of ONE notebook, in order, with its caption.

    Construct it in the notebook's setup cell, before anything is drawn:

        from src.visualize import figures, style
        style.apply()
        save = figures.FigureSaver("example_analysis")   # clears its own PDFs here

        fig, ax = plt.subplots()
        ...
        save(fig, "target_distribution",
             caption="Histogram of the target, 40 bins, n = 12,043.")

    THE CAPTION IS THE PAPER'S CAPTION. Write it to be pasted straight under the figure in the
    manuscript: pure description — what is plotted, on what axes, from how much data. No
    interpretation, no conclusion. The argument goes in the body text, and a caption that argues has
    to be rewritten when the argument changes.

    Passed at save time rather than kept in a central registry, so a caption lives next to the
    figure it describes and cannot go stale when that figure is renamed.
    """

    def __init__(self, notebook: str, *, clear_first: bool = True) -> None:
        self.notebook = notebook
        self.prefix = figure_prefix(notebook)
        self.folder = notebook_figures_dir(notebook)
        #: `clear_first=False` is for one case only: re-running a single cell mid-session without
        #: wiping what the earlier cells wrote. Never pass it in the setup cell.
        if clear_first:
            clear(notebook)
        self.folder.mkdir(parents=True, exist_ok=True)
        manifest_path(notebook).parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = [] if clear_first else read_manifest(notebook)

    def __call__(self, fig: plt.Figure, name: str, caption: str = "", **kwargs) -> None:
        return self.save(fig, name, caption=caption, **kwargs)

    @property
    def last_path(self) -> Path | None:
        """Where the most recent `save` wrote, for the rare caller that needs it."""
        return self.folder / f"{self.entries[-1]['stem']}.pdf" if self.entries else None

    def save(self, fig: plt.Figure, name: str, *, caption: str = "", provenance: dict | None = None) -> None:
        """Write a notebook-prefixed, numbered PDF and record the caption.

        The figure is left open so it still displays in Jupyter — that inline render is the only
        raster copy there is, and the interactive run has to look the same as the runner's.

        Returns nothing: a bare `sink.save(...)` cell would otherwise echo an absolute path, with
        the author's username, into the saved notebook. Use `last_path` when the path is needed.
        """
        index = len(self.entries) + 1
        stem = f"{self.prefix}{index:02d}_{_slug(name)}"
        path = self.folder / f"{stem}.pdf"
        _guard(path, self.folder)
        fig.savefig(path, format="pdf", dpi=DPI)
        entry = {"index": index, "stem": stem, "name": name, "caption": caption.strip()}
        if provenance:
            entry["provenance"] = provenance
        self.entries.append(entry)
        # Rewritten after every figure, so a notebook that dies halfway still has a manifest
        # describing the figures it did produce.
        manifest_path(self.notebook).write_text(
            json.dumps(self.entries, indent=2), encoding="utf-8"
        )

    def summary(self) -> str:
        """What was saved, for the notebook's final `print` — so `All_Results.md` says what the run
        drew, not only what it computed."""
        if not self.entries:
            return f"{self.notebook}: no figures saved."
        # Repo-relative, never absolute: All_Results.md is shared and must not carry a home path.
        try:
            where = self.folder.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            where = self.folder.name           # outside the repo (staging): name only
        lines = [f"{self.notebook}: {len(self.entries)} figures -> {where}"]
        for e in self.entries:
            lines.append(f"  {e['index']:02d}  {e['name']}")
            if not e["caption"]:
                lines.append("      NO CAPTION — add one; CAPTIONS.md will flag it.")
        return "\n".join(lines)


def _slug(name: str) -> str:
    """A filename-safe version of a figure name. Keeps it readable, not opaque."""
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in name.strip().lower()]
    return "".join(keep).strip("_") or "figure"


def _guard(path: Path, folder: Path) -> None:
    """Refuse to write outside this notebook's own folder — a `..` in a figure name would put a
    generated file outside `output CreditPFN/`, the one rule the layout rests on."""
    if folder.resolve() != path.resolve().parent:
        raise ValueError(
            f"figure would be written to {path.resolve()}, outside {folder.resolve()}. "
            f"Figure names are plain names, not paths."
        )
