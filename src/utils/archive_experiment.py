"""Compact historical evidence, excluding weights; separately retire indexed checkpoints.

Preview: --run exp1. Create: --run exp1 --write --quiescent.
Retirement preview: --retire ARCHIVE. Deletion: --retire ARCHIVE --apply --quiescent.
An evidence archive cannot restore model weights. It preserves their identities and results.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import tarfile
import uuid
from pathlib import Path

from src.utils.archive_output import verify, safe_source
from src.utils.consolidate_output import consolidate, matches_run, sha256, source_files
from src.utils.paths import archives_dir, consolidated_dir, manifests_dir, results_dir, resolve_output_path, resolve_staging_path


def checkpoint_roots() -> list[Path]:
    return sorted({resolve_staging_path("checkpoints/trained").resolve(),
                   resolve_output_path("checkpoints/trained").resolve()})


def belongs(name: str, run: str) -> bool:
    return any(matches_run(name, run, track=t) for t in ("pd", "lgd"))


def create(run: str, *, write=False, quiescent=False, destination=None,
           manifest_root=None, result_root=None, roots=None, snapshot_root=None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run) or ".." in run:
        raise ValueError("Use a plain experiment name")
    if write and not quiescent:
        raise ValueError("Stop experiment writers and pass --quiescent")
    manifest_root = manifest_root or manifests_dir()
    result_root = result_root or results_dir()
    snapshot_root = snapshot_root or consolidated_dir()
    roots = list(roots or checkpoint_roots())
    destination = destination or archives_dir()
    sources = source_files(run, manifest_root, result_root)
    files: dict[str, Path] = {}
    for kind, group in sources.items():
        root = result_root if kind.startswith("eval_") else manifest_root
        label = "results" if kind.startswith("eval_") else "manifests"
        for p in group:
            files[f"{label}/{p.relative_to(root).as_posix()}"] = p
    for track in ("pd", "lgd"):
        plan = manifest_root / "plans" / f"{run}_{track}.json"
        if plan.is_file():
            files[f"plans/{plan.name}"] = plan
    # Logs/config snapshots were not consistently run-tagged historically. Preserve them all;
    # never infer ownership from a timestamp. They compress well and remain separate from tables.
    for label, root in (("logs", manifest_root.parent / "logs"),
                        ("resolved", manifest_root / "resolved")):
        for p in root.rglob("*"):
            if p.is_file():
                files[f"{label}/{p.relative_to(root).as_posix()}"] = p
    weights = []
    for root in roots:
        for p in sorted(root.rglob("*.ckpt")):
            if not belongs(p.name, run):
                continue
            if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
                raise ValueError("Checkpoint path escapes its trained-checkpoint root")
            row = {"path": str(p.resolve()), "size": p.stat().st_size}
            if write:
                row["sha256"] = sha256(p)
            weights.append(row)
            sidecar = Path(str(p) + ".provenance.json")
            if sidecar.is_file():
                files[f"provenance/root{roots.index(root)}/{sidecar.relative_to(root).as_posix()}"] = sidecar
    report = {"run": run, "evidence_files": len(files),
              "evidence_bytes": sum(p.stat().st_size for p in files.values()),
              "indexed_checkpoints": len(weights), "checkpoint_bytes": sum(r["size"] for r in weights),
              "weights_in_archive": False, "written": write}
    if not write:
        return report
    if not any(sources.values()):
        raise ValueError("No measurements found for this run; refusing an evidence-free retirement archive")
    consolidate(run, apply=True, manifest_root=manifest_root, result_root=result_root, destination=snapshot_root)
    pointer = snapshot_root / run / "LATEST.json"
    folder = pointer.parent / json.loads(pointer.read_text(encoding="utf-8"))["snapshot"]
    for p in folder.iterdir():
        if p.is_file():
            files[f"consolidated/{p.name}"] = p
    repo = Path(__file__).resolve().parents[2]
    # This is the archiving checkout, not a claim about which code trained old checkpoints.
    for pattern in ("config/*.yaml", "docs/*.md", "README.md", "AGENTS.md", "pyproject.toml"):
        for p in repo.glob(pattern):
            files[f"archiving_checkout/{p.relative_to(repo).as_posix()}"] = p
    destination.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    final = destination / f"{run}-evidence-{stamp}.tar.gz"
    stage = destination / f".building-{stamp}.tar.gz"
    inventory = {"schema_version": 1, "run": run, "weights_in_archive": False,
                 "checkpoints": weights, "files": [],
                 "source_root": str(manifest_root.parent.resolve()),
                 "warning": "Historical protocol; results are not corrected-run measurements. "
                            "Weights cannot be restored from this archive."}
    with tarfile.open(stage, "w:gz", compresslevel=6) as archive:
        for name, p in sorted(files.items()):
            allowed_roots = [manifest_root, result_root, manifest_root.parent / "logs",
                             folder, repo / "config", repo / "docs", *roots]
            in_tree = any(p.resolve().is_relative_to(r.resolve()) for r in allowed_roots)
            if p.is_symlink() or (not in_tree and p.parent.resolve() != repo):
                raise ValueError("Evidence sources must not be symlinks")
            stat = p.stat()
            digest = sha256(p)
            archive.add(p, arcname=name, recursive=False)
            if (p.stat().st_size, p.stat().st_mtime_ns, sha256(p)) != (stat.st_size, stat.st_mtime_ns, digest):
                raise RuntimeError(f"Evidence changed during archiving: {p}")
            row = {"path": name, "size": stat.st_size, "sha256": digest}
            if p.resolve().is_relative_to(manifest_root.parent.resolve()):
                relative = p.relative_to(manifest_root.parent).as_posix()
                if relative.startswith(("logs/", "manifests/resolved/", "manifests/epochs/")):
                    safe_source(manifest_root.parent, relative)
                    row["prunable_source"] = relative
            inventory["files"].append(row)
    for row in weights:
        if (Path(row["path"]).stat().st_size, sha256(Path(row["path"]))) != (row["size"], row["sha256"]):
            raise RuntimeError("Checkpoint changed while archiving; archive was not published")
    index = Path(str(stage) + ".json")
    index.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    verify(stage)
    stage.rename(final)
    index.rename(Path(str(final) + ".json"))
    report.update(archive=str(final), archive_bytes=final.stat().st_size)
    return report


def prune_evidence(archive: Path, *, apply=False, quiescent=False, root=None) -> dict:
    """Remove only unchanged archived logs/config shards/epoch shards, never trial manifests."""
    inventory = verify(archive)
    if "checkpoints" not in inventory or not any(r["path"].startswith("consolidated/") for r in inventory["files"]):
        raise ValueError("Pruning requires a complete experiment evidence archive")
    root = root or resolve_output_path("output")
    if root.resolve() != Path(inventory["source_root"]).resolve():
        raise ValueError("The archive refers to a different live output root")
    pending = []
    for row in inventory["files"]:
        if "prunable_source" not in row:
            continue
        p = safe_source(root, row["prunable_source"])
        if not p.exists():
            continue
        if (p.stat().st_size, sha256(p)) != (row["size"], row["sha256"]):
            raise RuntimeError("Archived source changed; no pruning was performed")
        pending.append((p, row))
    report = {"files": len(pending), "bytes": sum(r["size"] for _, r in pending), "deleted": False}
    if apply:
        if not quiescent:
            raise ValueError("Stop all writers and pass --quiescent")
        for p, row in pending:
            if sha256(p) != row["sha256"]:
                raise RuntimeError("Source changed during pruning; stopped")
            p.unlink()
        report["deleted"] = True
    return report


def retire(archive: Path, *, apply=False, quiescent=False, roots=None) -> dict:
    inventory = verify(archive)
    if "checkpoints" not in inventory or inventory.get("weights_in_archive") is not False:
        raise ValueError("Use an archive produced by archive_experiment")
    roots = roots or checkpoint_roots()
    paths = []
    for row in inventory["checkpoints"]:
        p = Path(row["path"])
        if not p.exists():
            continue  # permits recovery after an interrupted, partially completed retirement
        if p.is_symlink() or not any(p.resolve().is_relative_to(r.resolve()) for r in roots):
            raise ValueError("Refusing to retire a file outside the configured trained-checkpoint roots")
        if p.suffix != ".ckpt" or not belongs(p.name, inventory["run"]):
            raise ValueError("Checkpoint does not belong to the archived experiment")
        if (p.stat().st_size, sha256(p)) != (row["size"], row["sha256"]):
            raise RuntimeError("Checkpoint changed since archival; no retirement was performed")
        paths.append((p, row))
    report = {"run": inventory["run"], "files": len(paths),
              "bytes": sum(r["size"] for _, r in paths), "paths": [str(p) for p, _ in paths],
              "deleted": False, "warning": "Deleting these weights is irreversible from this evidence archive."}
    if apply:
        if not quiescent:
            raise ValueError("Stop all experiment writers and pass --quiescent")
        for p, row in paths:
            if sha256(p) != row["sha256"]:
                raise RuntimeError("Checkpoint changed during retirement; stopped")
            p.unlink()
        # Keep tiny provenance sidecars; they remain useful without the weight files.
        report["deleted"] = True
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run")
    mode.add_argument("--retire", type=Path)
    mode.add_argument("--prune", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--quiescent", action="store_true")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    if args.apply and not (args.retire or args.prune):
        parser.error("--apply is only valid with --retire or --prune")
    if args.write and not args.run:
        parser.error("--write is only valid with --run")
    report = prune_evidence(args.prune, apply=args.apply, quiescent=args.quiescent) if args.prune else retire(args.retire, apply=args.apply, quiescent=args.quiescent) if args.retire else create(
        args.run, write=args.write, quiescent=args.quiescent, destination=args.destination)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
