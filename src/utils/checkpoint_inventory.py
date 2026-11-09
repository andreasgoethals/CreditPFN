"""Find a recorded trained checkpoint on either storage tier, refusing ambiguous copies."""
import json
from pathlib import Path

from src.utils.paths import group_for_run, resolve_output_path, resolve_staging_path


def resolve_checkpoint(recorded: str, track: str) -> tuple[Path | None, dict]:
    old = Path(recorded)
    relative = Path("checkpoints/trained") / group_for_run(old.name) / track
    roots = {old.parent, resolve_staging_path(relative), resolve_output_path(relative)}
    candidates = {root / old.name for root in roots if (root / old.name).is_file()}
    if not candidates:
        return None, {}
    if old in candidates:
        path = old
    elif len(candidates) == 1:
        path = candidates.pop()
    else:
        raise ValueError(f"Multiple copies of {old.name}; reconcile staging/fallback before evaluation")
    sidecar = Path(str(path) + ".provenance.json")
    prov = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    return path, prov
