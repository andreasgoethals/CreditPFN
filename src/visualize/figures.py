"""One flat PDF collection, with disjoint notebook ownership and paper captions.

PDFs: output CreditPFN/figures/<experiment>__<notebook>__<NN>_<name>.pdf
Metadata: output CreditPFN/<experiment>/manifests/figures/<notebook>.json

Constructing a saver removes that notebook's previous figures, including the old
folder layout. Other notebooks and scientific evidence are never cleared by it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.paths import REPO_ROOT, EXPERIMENTS, manifests_dir
from src.utils.paths import figures_dir as legacy_figures_dir
from src.visualize.paths import figures_dir, figure_prefix

#: Vector already, but heatmaps and scatter clouds inside a PDF rasterise, so it still needs a
#: print DPI.
DPI = 300

# Only these generated files are removed from a legacy notebook folder.
_OWNED = ("*.pdf", "_figures.json", "_stdout.txt")

MANIFEST = "_figures.json"


def manifest_path(notebook: str) -> Path:
    from src.utils.paths import notebook_parts
    group, name = notebook_parts(notebook)
    return manifests_dir(group) / "figures" / f"{name.as_posix()}.json"


def read_manifest(notebook: str) -> list[dict]:
    """What this notebook drew last time it ran, in order. `[]` if it never has."""
    path = manifest_path(notebook)
    if not path.is_file():
        path = legacy_figures_dir(notebook) / MANIFEST
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A run killed mid-write leaves truncated JSON. That is a missing manifest, not a crash:
        # the figures are still on disk and the next run rewrites it.
        return []


def clear(notebook: str) -> int:
    """Remove owned PDFs and metadata, preserving other notebooks and run evidence."""
    removed = 0
    for folder, patterns in ((figures_dir(), (figure_prefix(notebook) + "*.pdf",)),
                             (legacy_figures_dir(notebook), _OWNED)):
        for pattern in patterns:
            for path in folder.glob(pattern):
                _guard(path, folder)
                if path.is_file():
                    path.unlink()
                    removed += 1
    path = manifest_path(notebook)
    if path.is_file():
        path.unlink()
        removed += 1
    return removed


def clear_collection(notebooks: tuple[str, ...]) -> None:
    """Full reruns also retire figures from notebooks that were renamed or removed."""
    owners = set(notebooks)
    for group in EXPERIMENTS:
        root = manifests_dir(group) / "figures"
        owners.update(group + "/" + p.relative_to(root).with_suffix("").as_posix()
                      for p in root.rglob("*.json"))
    for name in sorted(owners):
        clear(name)
    # Caption indexes are generated from scratch after execution. The old shared
    # index is removed during migration; no training/result directories are visited.
    for path in (figures_dir() / "CAPTIONS.md", legacy_figures_dir() / "CAPTIONS.md"):
        if path.is_file():
            path.unlink()


def owned_pdfs(notebook: str) -> list[Path]:
    return sorted(figures_dir().glob(figure_prefix(notebook) + "*.pdf"))


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
        self.folder = figures_dir()
        self.folder.mkdir(parents=True, exist_ok=True)
        manifest_path(notebook).parent.mkdir(parents=True, exist_ok=True)
        #: `clear_first=False` is for one case only: re-running a single cell mid-session without
        #: wiping what the earlier cells wrote. Never pass it in the setup cell.
        if clear_first:
            clear(notebook)
        self.entries: list[dict] = [] if clear_first else read_manifest(notebook)

    def __call__(self, fig: plt.Figure, name: str, caption: str = "", **kwargs) -> None:
        return self.save(fig, name, caption=caption, **kwargs)

    @property
    def last_path(self) -> Path | None:
        """Where the most recent `save` wrote, for the rare caller that needs it."""
        return self.folder / f"{self.entries[-1]['stem']}.pdf" if self.entries else None

    def save(self, fig: plt.Figure, name: str, *, caption: str = "") -> None:
        """Write a notebook-prefixed, numbered PDF and record the caption.

        The figure is left open so it still displays in Jupyter — that inline render is the only
        raster copy there is, and the interactive run has to look the same as the runner's.

        RETURNS NOTHING, deliberately. It used to return the `Path`, and since every notebook
        cell here is a bare `sink.save(...)`, Jupyter echoed that return value as the cell's
        result: an `Out[n]: WindowsPath('C:/Users/<name>/.../20_paper_zero_shot.pdf')` line
        under all 22 figures. That is noise beside every figure, and — now that the executed
        notebooks are saved with their outputs — it wrote an absolute path carrying the author's
        username into a tracked file. Use `last_path` when a caller genuinely needs the path.
        """
        index = len(self.entries) + 1
        stem = f"{self.prefix}{index:02d}_{_slug(name)}"
        path = self.folder / f"{stem}.pdf"
        _guard(path, self.folder)
        fig.savefig(path, format="pdf", dpi=DPI)
        self.entries.append(
            {"index": index, "stem": stem, "name": name, "caption": caption.strip()}
        )
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
        # REPO-RELATIVE, never absolute. `All_Results.md` is tracked and shared, so an
        # absolute path would publish the author's home directory and username — and would
        # differ on every machine, so the file churned in the diff after each run (locally
        # `C:\Users\<name>\...`, on the cluster `/data/leuven/383/<account>/...`).
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
