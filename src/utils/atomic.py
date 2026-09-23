"""Small metadata publications: complete bytes or no published file."""
import json
import os
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
            os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)
