"""Small metadata publications: complete bytes or no published file."""
import json
import os
import time
import uuid
from pathlib import Path


def write_json(path: Path, payload, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(pending, path)  # atomic create, refuses to replace an existing plan
        else:
            # Windows readers and file indexers can briefly deny replacement. Retry the
            # same complete temporary file without exposing a partly written destination.
            for delay in (0.02, 0.05, 0.1, 0.2, 0.4, None):
                try:
                    os.replace(pending, path)
                    break
                except PermissionError as exc:
                    if delay is None or getattr(exc, "winerror", None) not in (5, 32, 33):
                        raise
                    time.sleep(delay)
    finally:
        pending.unlink(missing_ok=True)
