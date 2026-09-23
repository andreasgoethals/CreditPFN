"""Resolve moved/retagged checkpoint references without rewriting historical manifests."""
from pathlib import Path
import json

from src.utils.paths import resolve_output_path, resolve_staging_path
from src.utils.migrate_l2sp_checkpoints import retag


def resolve_checkpoint(recorded: str, track: str) -> tuple[Path | None, dict]:
    old = Path(recorded)
    roots = {old.parent, resolve_staging_path("checkpoints/trained") / track,
             resolve_output_path("checkpoints/trained") / track}
    candidates = {root / old.name for root in roots if (root / old.name).is_file()}
    if not candidates and "_l2sp" not in old.name:
        # Glob broadly, but verify exact spelling against provenance below.
        for root in roots:
            for candidate in root.glob(old.stem.split("_lr")[0] + "*_l2sp*.ckpt"):
                sidecar = Path(str(candidate) + ".provenance.json")
                if not sidecar.is_file():
                    continue
                prov = json.loads(sidecar.read_text(encoding="utf-8"))
                lam = prov.get("hyperparameters", {}).get("l2sp_lambda")
                if lam is not None and retag(old.name, lam) == candidate.name:
                    candidates.add(candidate)
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
