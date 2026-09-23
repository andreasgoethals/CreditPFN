"""Publish checkpoint bytes atomically, with provenance as the completion marker."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import torch


def atomic_save(payload: dict, path: Path, provenance: dict | None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    pending = path.with_name(f".{path.name}.{token}.tmp")
    sidecar = Path(str(path) + ".provenance.json")
    pending_prov = sidecar.with_name(f".{sidecar.name}.{token}.tmp")
    try:
        with pending.open("wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        if provenance is not None:
            pending_prov.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
        # A crash between replacements must leave NO completion marker for new bytes.
        sidecar.unlink(missing_ok=True)
        os.replace(pending, path)
        if provenance is not None:
            os.replace(pending_prov, sidecar)
    finally:
        pending.unlink(missing_ok=True)
        pending_prov.unlink(missing_ok=True)
