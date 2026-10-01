"""Publication paths, independent of the frozen cluster training-path contract."""
from pathlib import Path

from src.utils.paths import notebook_parts, outputs_dir


def figures_dir() -> Path:
    """Root of the complete local publication collection."""
    return outputs_dir() / "figures"


def captions_path() -> Path:
    return figures_dir() / "CAPTIONS.md"


def all_results_path() -> Path:
    return outputs_dir() / "All_Results.md"


def notebook_figures_dir(notebook: str) -> Path:
    """Use notebook folder order, including 00_general, for the publication tree."""
    group, _ = notebook_parts(notebook)
    return figures_dir() / ("00_general" if group == "general" else group)


def figure_prefix(notebook: str) -> str:
    """Unique notebook ownership within an experiment folder."""
    group, name = notebook_parts(notebook)
    parts = (group, *name.parts)
    if len(parts) < 2 or any(not p or "__" in p or not all(
            c.isalnum() or c in "-_" for c in p) for p in parts):
        raise ValueError("Notebook components must be names without spaces or the reserved '__' separator")
    # Encode a nested path rather than joining it with the owner delimiter:
    # clearing notebook "a" must never match notebook "a/b" by prefix.
    return group + "__" + "%2F".join(name.parts) + "__"
