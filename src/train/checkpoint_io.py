"""Publish checkpoint bytes atomically, with provenance as the completion marker."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import torch


def _replace(source: Path, destination: Path) -> None:
    """Keep atomic replacement while allowing brief Windows scanner/share locks.

    A permanent permission error still fails; no unlink/copy fallback can expose
    partial checkpoint bytes. Linux errors are never retried.
    """
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


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
        _replace(pending, path)
        if provenance is not None:
            _replace(pending_prov, sidecar)
    finally:
        pending.unlink(missing_ok=True)
        pending_prov.unlink(missing_ok=True)
