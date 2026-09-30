"""Publication paths, independent of the frozen cluster training-path contract."""
from pathlib import Path

from src.utils.paths import notebook_parts, outputs_dir


def figures_dir() -> Path:
    """All notebook PDFs share one folder on the local analysis output tier."""
    return outputs_dir() / "figures"


def captions_path() -> Path:
    return figures_dir() / "CAPTIONS.md"


def figure_prefix(notebook: str) -> str:
    """Unique notebook ownership without subdirectories or ambiguous separators."""
    group, name = notebook_parts(notebook)
    parts = (group, *name.parts)
    if len(parts) < 2 or any(not p or "__" in p or not all(
            c.isalnum() or c in "-_" for c in p) for p in parts):
        raise ValueError("Notebook components must be names without spaces or the reserved '__' separator")
    # Encode a nested path rather than joining it with the owner delimiter:
    # clearing notebook "a" must never match notebook "a/b" by prefix.
    return group + "__" + "%2F".join(name.parts) + "__"
