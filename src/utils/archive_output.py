"""Verified, two-stage output archival. Preview by default; never archive weights/data.

Create with --apply, then separately --prune ARCHIVE --quiescent. Pruning requires
unchanged sources and a complete, re-read archive. Epoch CSVs additionally require
a published consolidated snapshot. Stop all writers before either operation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import tarfile
import uuid
from pathlib import Path, PurePosixPath

from src.utils.consolidate_output import sha256
from src.utils.paths import archives_dir, consolidated_dir, resolve_output_path

ALLOWED = ("logs", "manifests/resolved", "manifests/epochs")


def safe_source(root: Path, name: str) -> Path:
    parts = PurePosixPath(name)
    if parts.is_absolute() or ".." in parts.parts or "\\" in name:
        raise ValueError("Unsafe archive path")
    if not any(name.startswith(prefix + "/") for prefix in ALLOWED):
        raise ValueError("Only logs, resolved configs and epoch histories may be pruned")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Source must stay inside the output root and must not be a symlink")
    return path


def verify(archive: Path) -> dict:
    inventory = json.loads(Path(str(archive) + ".json").read_text(encoding="utf-8"))
    expected = {r["path"]: r for r in inventory["files"]}
    seen = set()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise RuntimeError("Unexpected archive member")
            digest = hashlib.sha256()
            stream = tf.extractfile(member)
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            row = expected[member.name]
            if member.size != row["size"] or digest.hexdigest() != row["sha256"]:
                raise RuntimeError(f"Archive verification failed: {member.name}")
            seen.add(member.name)
    if seen != set(expected):
        raise RuntimeError("Archive is incomplete")
    return inventory


def create(*, root: Path, destination: Path, apply: bool = False,
           include_epochs: bool = False) -> dict:
    prefixes = ALLOWED if include_epochs else ALLOWED[:2]
    paths = sorted(p for prefix in prefixes for p in (root / prefix).rglob("*") if p.is_file())
    for p in paths:
        safe_source(root, p.relative_to(root).as_posix())
    report = {"files": len(paths), "bytes": sum(p.stat().st_size for p in paths),
              "source_root": str(root.resolve()), "destination": str(destination), "applied": apply}
    if not apply:
        return report
    if not paths:
        raise ValueError("No files to archive")
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Use an archive destination outside the live output root")
    destination.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    final = destination / f"output-{stamp}.tar.gz"
    stage = destination / f".building-{stamp}.tar.gz"
    inventory = {"schema_version": 1, "source_root": str(root.resolve()), "files": []}
    with tarfile.open(stage, "w:gz", compresslevel=6) as tf:
        for path in paths:
            stat = path.stat()
            row = {"path": path.relative_to(root).as_posix(), "size": stat.st_size,
                   "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}
            tf.add(path, arcname=row["path"], recursive=False)
            inventory["files"].append(row)
    sidecar = Path(str(stage) + ".json")
    sidecar.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    verify(stage)
    for row in inventory["files"]:
        check_source(root, row)
    stage.rename(final)
    sidecar.rename(Path(str(final) + ".json"))
    report.update(archive=str(final), compressed_bytes=final.stat().st_size)
    return report


def check_source(root: Path, row: dict) -> Path:
    p = safe_source(root, row["path"])
    if not p.is_file() or (p.stat().st_size, p.stat().st_mtime_ns, sha256(p)) != (
        row["size"], row["mtime_ns"], row["sha256"]
    ):
        raise RuntimeError(f"Source changed or missing; refusing cleanup: {row['path']}")
    return p


def prune(archive: Path, *, root: Path, quiescent: bool, snapshots: Path | None = None) -> dict:
    if not quiescent:
        raise ValueError("Stop all writers and explicitly pass --quiescent before cleanup")
    info = verify(archive)
    if Path(info["source_root"]).resolve() != root.resolve():
        raise ValueError("Archive belongs to a different source root")
    # A verified archive is sufficient for logs/configs. Curves must also remain
    # directly usable by the notebooks after raw CSVs have been removed.
    covered = {}
    snapshots = snapshots or consolidated_dir()
    for pointer in snapshots.glob("*/LATEST.json"):
        folder = pointer.parent / json.loads(pointer.read_text(encoding="utf-8"))["snapshot"]
        data = json.loads((folder / "inventory.json").read_text(encoding="utf-8"))
        for track in ("pd", "lgd"):
            table = f"training_{track}"
            if sha256(folder / f"{table}.csv.gz") != data["tables"][table]:
                raise RuntimeError("Consolidated training table failed checksum verification")
            for row in data["sources"][table]:
                covered["manifests/" + row["path"]] = row["sha256"]
    for row in info["files"]:
        check_source(root, row)
        if row["path"].startswith("manifests/epochs/") and covered.get(row["path"]) != row["sha256"]:
            raise RuntimeError("Consolidate every run represented in the epoch archive before cleanup")
    count = 0
    for row in info["files"]:
        check_source(root, row).unlink()
        count += 1
    return {"removed_files": count, "archive": str(archive), "root": str(root)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=resolve_output_path("output"))
    parser.add_argument("--destination", type=Path, default=archives_dir())
    parser.add_argument("--include-epochs", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", type=Path)
    mode.add_argument("--prune", type=Path)
    parser.add_argument("--quiescent", action="store_true")
    args = parser.parse_args(argv)
    if args.prune:
        report = prune(args.prune, root=args.root, quiescent=args.quiescent)
    elif args.verify:
        report = {"verified_files": len(verify(args.verify)["files"])}
    else:
        report = create(root=args.root, destination=args.destination,
                        apply=args.apply, include_epochs=args.include_epochs)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
