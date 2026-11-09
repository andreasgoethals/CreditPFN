"""Stage a sealed, relocatable site-wrapper runtime on the compute cluster's native scratch.

Scratch cleanup follows access time, so staged files must not inherit the persistent
image's old timestamps (vsc-library/guides/STORAGE.md): files are copied without
preserving times, keeping only permission bits. Reuse is validated against the sealed
manifest (size and SHA-256), never against timestamps.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from src.utils.runtime_container import sha256


def _safe(name, kind):
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"Runtime {kind} escaped its directory")


def _copy_fresh(source, target):
    # copyfile writes new data (fresh mtime/atime); copymode keeps executable bits only.
    shutil.copyfile(source, target)
    shutil.copymode(source, target)


def _verify(destination, manifest, digest):
    """Fail closed when an existing cache differs from the sealed manifest."""
    saved = json.loads((destination/"CACHE.json").read_text())
    if saved["manifest_sha256"] != digest or sha256(destination/"runtime.json") != digest:
        raise RuntimeError("Runtime cache belongs to another image")
    for name, expected in manifest["files"].items():
        _safe(name, "artifact")
        path = destination/name
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            raise RuntimeError(f"Immutable runtime cache changed: {name} is missing") from None
        record = saved["files"].get(name)
        # Receipts list [size, sha256]; earlier receipts list [size, mtime_ns]. Only the
        # size is taken from the receipt; content is always checked against the manifest.
        if not record or record[0] != size or sha256(path) != expected:
            raise RuntimeError(f"Immutable runtime cache changed: {name}")
    for name, target in manifest["symlinks"].items():
        _safe(name, "link")
        if os.readlink(destination/name) != target:
            raise RuntimeError("Runtime cache symlink changed")
    return destination


def stage(runtime, destination):
    import fcntl  # Native Linux job setup only; no third-party dependency/bootstrap install.
    runtime, destination = Path(runtime).resolve(strict=True), Path(destination).resolve()
    if runtime == destination or runtime in destination.parents or destination in runtime.parents:
        raise ValueError("Runtime cache must be separate from persistent source")
    manifest = json.loads((runtime/"runtime.json").read_text())
    if not manifest.get("images") or not manifest.get("symlinks"):
        raise ValueError("Runtime image/symlink inventory is missing")
    digest = sha256(runtime/"runtime.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent/(destination.name+".lock")).open("a+") as lock:
        # Concurrent jobs verify an existing cache under a shared lock; only staging
        # is exclusive. The publish is an atomic rename of a completed directory.
        fcntl.flock(lock, fcntl.LOCK_SH)
        if destination.exists():
            return _verify(destination, manifest, digest)
        fcntl.flock(lock, fcntl.LOCK_UN)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():  # Another job published while this one waited.
            return _verify(destination, manifest, digest)
        pending = destination.parent/("."+destination.name+"."+uuid.uuid4().hex)
        pending.mkdir()
        # Keep failed owned work for diagnosis; never erase unknown scratch files.
        files = {}
        for name, expected in manifest["files"].items():
            _safe(name, "artifact")
            source, target = runtime/name, pending/name
            if sha256(source) != expected:
                raise RuntimeError("Persistent runtime changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_fresh(source, target)
            if sha256(target) != expected:
                raise RuntimeError("Runtime staging checksum failed")
            files[name] = [target.stat().st_size, expected]
        for name, target in manifest["symlinks"].items():
            _safe(name, "link")
            if os.readlink(runtime/name) != target:
                raise RuntimeError("Persistent runtime symlink changed")
            link = pending/name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)
        _copy_fresh(runtime/"runtime.json", pending/"runtime.json")
        if sha256(pending/"runtime.json") != digest:
            raise RuntimeError("Runtime staging checksum failed")
        (pending/"CACHE.json").write_text(json.dumps(dict(source=str(runtime), manifest_sha256=digest, files=files)))
        pending.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(stage(args.runtime, args.destination))


if __name__ == "__main__":
    main()
