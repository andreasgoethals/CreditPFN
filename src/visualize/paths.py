"""Publication paths, independent of the frozen cluster training-path contract."""
from pathlib import Path

from src.utils.paths import notebook_parts as legacy_notebook_parts, outputs_dir, EXPERIMENTS

RQ_GROUPS = ("rq1", "rq2", "rq3", "rq4")
PUBLICATION_GROUPS = (*EXPERIMENTS, *RQ_GROUPS)


def notebook_parts(notebook: str) -> tuple[str, Path]:
    """RQ publication groups do not change historical scientific path routing."""
    path = Path(notebook)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Notebook name must be a relative path without parent traversal")
    if path.parts[0] in RQ_GROUPS:
        if len(path.parts) < 2:
            raise ValueError("RQ notebook name needs a filename")
        return path.parts[0], Path(*path.parts[1:])
    return legacy_notebook_parts(notebook)


def figures_dir() -> Path:
    """Root of the complete local publication collection."""
    return outputs_dir() / "figures"


def captions_path() -> Path:
    return figures_dir() / "CAPTIONS.md"


def all_results_path() -> Path:
    return outputs_dir() / "All_Results.md"


def notebook_figures_dir(notebook: str) -> Path:
    """Use the notebook group as the publication folder."""
    group, _ = notebook_parts(notebook)
    return figures_dir() / group


def figure_prefix(notebook: str) -> str:
    """Unique notebook ownership within a publication group."""
    group, name = notebook_parts(notebook)
    parts = (group, *name.parts)
    if len(parts) < 2 or any(not p or "__" in p or not all(
            c.isalnum() or c in "-_" for c in p) for p in parts):
        raise ValueError("Notebook components must be names without spaces or the reserved '__' separator")
    # Encode a nested path rather than joining it with the owner delimiter:
    # clearing notebook "a" must never match notebook "a/b" by prefix.
    return group + "__" + "%2F".join(name.parts) + "__"
