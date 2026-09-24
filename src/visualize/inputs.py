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
        from src.utils.paths import group_for_run
        root = root / group_for_run(run)
        kwargs.update(manifest_root=root / "manifests", result_root=root / "results",
                      training_root=root / "training", snapshot_root=root / "consolidated")
    else:
        from src.utils.paths import group_for_run, training_dir
        kwargs.setdefault("training_root", training_dir(experiment=group_for_run(run)))
    return read_snapshot(run, table, **kwargs)
