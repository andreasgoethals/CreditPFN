"""Resolve moved/retagged checkpoint references without rewriting historical manifests."""
from pathlib import Path
import json
import re

from src.utils.paths import group_for_run, resolve_output_path, resolve_staging_path


def _retag_l2sp(name: str, value: float) -> str:
    """Reconstruct a historical filename for lookup; never rename a file."""
    return re.sub(r"(_lora|_iclhead)?\.ckpt$",
                  lambda m: f"_l2sp{float(value):g}{m.group(1) or ''}.ckpt", name)


def resolve_checkpoint(recorded: str, track: str) -> tuple[Path | None, dict]:
    old = Path(recorded)
    relative = Path("checkpoints/trained") / group_for_run(old.name) / track
    roots = {old.parent, resolve_staging_path("checkpoints/trained") / track,
             resolve_output_path("checkpoints/trained") / track,
             resolve_staging_path(relative), resolve_output_path(relative)}
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
                if lam is not None and _retag_l2sp(old.name, lam) == candidate.name:
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
