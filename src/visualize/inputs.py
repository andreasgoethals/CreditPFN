"""Optional read-only analysis source; generated figures retain normal output paths."""
from __future__ import annotations

import os
from pathlib import Path


def analysis_root() -> Path | None:
    value = os.environ.get("CREDITPFN_ANALYSIS_ROOT")
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("CREDITPFN_ANALYSIS_ROOT must name an existing downloaded output folder")
    return root


def load_consolidated(run, table, **kwargs):
    from src.utils.consolidate_output import load_consolidated as read_snapshot
    root = analysis_root()
    if root is not None:
        kwargs.update(manifest_root=root / "manifests", result_root=root / "results",
                      snapshot_root=root / "consolidated")
    return read_snapshot(run, table, **kwargs)
