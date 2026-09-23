"""Snapshot existing output without changing training writers or source files.

Run with --run exp1 to preview; --apply publishes a versioned snapshot atomically.
Checkpoints/predictions remain in place. Raw attempts and raw epoch names are retained:
an old filename is not proof of which weights survived the pre-fix L2-SP collision.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.utils.paths import consolidated_dir, manifests_dir, results_dir


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    """Portable compressed CSV, retaining round-trip float precision and empty tables."""
    try:
        return pd.read_csv(path, float_precision="round_trip", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@lru_cache(maxsize=12)
def _cached_table(path: Path, size: int, mtime_ns: int, digest: str) -> pd.DataFrame:
    if sha256(path) != digest:
        raise RuntimeError(f"Consolidated table checksum failed: {path}")
    return read_table(path)


def matches_run(name: str, run: str, *, track: str | None = None) -> bool:
    suffix = rf"_(?:s\d+_)?{re.escape(track)}(?:_|\.)" if track else r"_(?:s\d+_|\d{8}_)"
    return bool(re.match(re.escape(run) + suffix, name))


def source_files(run: str, manifest_root: Path, result_root: Path) -> dict[str, list[Path]]:
    files = {}
    for track in ("pd", "lgd"):
        files[f"attempts_{track}"] = [p for p in sorted(manifest_root.glob("*.csv"))
                                     if matches_run(p.name, run, track=track)]
        files[f"training_{track}"] = [p for p in sorted((manifest_root / "epochs" / track).glob("*.csv"))
                                     if matches_run(p.name, run, track=track)]
        files[f"eval_{track}"] = [p for p in sorted((result_root / track.upper()).glob("*/*.csv"))
                                 if matches_run(p.name, run)]
    return files


def read_frames(files: list[Path], *, root: Path, kind: str) -> pd.DataFrame:
    frames = []
    for path in files:
        df = read_table(path)
        if df.empty:
            continue
        df["source_file"] = path.relative_to(root).as_posix()
        df["source_row"] = range(len(df))
        if kind == "training":
            df["trial_name"] = path.name.removesuffix(".csv").removesuffix(".trajectory")
            if "record_type" not in df:
                df["record_type"] = "epoch"
        if kind == "eval":
            df["method_dir"] = path.parent.name
        split = re.search(r"_s(\d+)_", path.name)
        if split:
            df["split"] = int(split.group(1))
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def latest_trials(attempts: pd.DataFrame) -> pd.DataFrame:
    """One recorded outcome per HP identity, retaining attempt counts and all source rows elsewhere.

    Prefer a completed measurement over a later SKIP's empty metrics. FAIL/DIVERGED
    rows are not discarded merely because they lack a git hash.
    """
    if attempts.empty:
        return attempts.copy()
    df = attempts.copy()
    keys = [k for k in ("source_file", "track", "base_checkpoint", "learning_rate", "use_lora",
                       "query_fraction", "accumulate_grad_batches", "seed", "epoch_pass_mode",
                       "min_train_rows", "l2sp_lambda") if k in df]
    df["attempt_count"] = df.groupby(keys, dropna=False)["source_row"].transform("size")
    df["_skip"] = df["status"].eq("SKIP")
    df = df.sort_values(["_skip", "source_row"], ascending=[False, True])
    df = df.drop_duplicates(keys, keep="last").drop(columns="_skip")
    df["evidence_status"] = "recorded_only"
    return df.reset_index(drop=True)


def consolidate(run: str, *, apply: bool = False, manifest_root: Path | None = None,
                result_root: Path | None = None, destination: Path | None = None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run) or ".." in run:
        raise ValueError("run must be a plain experiment name")
    manifest_root = manifest_root or manifests_dir()
    result_root = result_root or results_dir()
    destination = destination or consolidated_dir()
    files = source_files(run, manifest_root, result_root)
    report = {"schema_version": 1, "run": run, "counts": {k: len(v) for k, v in files.items()},
              "source_roots": {"manifests": str(manifest_root.resolve()), "results": str(result_root.resolve())},
              "source_bytes": sum(p.stat().st_size for group in files.values() for p in group),
              "notes": ["Legacy records remain exploratory; L2-SP retagging does not repair sampling or accumulation.",
                        "Training curves retain recorded filenames. No inferred link to overwritten checkpoints.",
                        "Raw predictions and checkpoints are not moved or deleted."]}
    # Re-consolidating an archived run must not publish empty/partial histories
    # over its last complete snapshot. Restore the raw shards before rebuilding it.
    previous = destination / run / "LATEST.json"
    missing = []
    if previous.is_file():
        folder = previous.parent / json.loads(previous.read_text(encoding="utf-8"))["snapshot"]
        prior = json.loads((folder / "inventory.json").read_text(encoding="utf-8"))
        for name, records in prior["sources"].items():
            root = result_root if name.startswith("eval_") else manifest_root
            missing.extend(record["path"] for record in records if not (root / record["path"]).is_file())
    report["missing_previous_sources"] = len(missing)
    if not apply:
        return report
    if missing:
        raise RuntimeError("Previous snapshot has archived/missing raw sources. Restore the verified raw "
                           "shards before reconsolidating this run; its LATEST pointer was preserved.")
    if not any(files.values()):
        raise ValueError(f"No source records found for {run}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    parent = destination / run
    stage = parent / (".building-" + stamp)
    stage.mkdir(parents=True, exist_ok=False)
    sources = {}
    report["rows"] = {}
    report["tables"] = {}
    for name, paths in files.items():
        root = result_root if name.startswith("eval_") else manifest_root
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns, sha256(p)) for p in paths}
        kind, track = name.split("_", 1)
        df = read_frames(paths, root=root, kind=kind)
        for p in paths:
            size, mtime, digest = before[str(p)]
            if (p.stat().st_size, p.stat().st_mtime_ns, sha256(p)) != (size, mtime, digest):
                raise RuntimeError(f"Source changed during consolidation: {p}; stop writers and retry")
        sources[name] = [{"path": p.relative_to(root).as_posix(), "size": before[str(p)][0],
                          "mtime_ns": before[str(p)][1], "sha256": before[str(p)][2]} for p in paths]
        tables = {name: df}
        if kind == "attempts":
            tables[f"trials_{track}"] = latest_trials(df)
        for table, frame in tables.items():
            path = stage / f"{table}.csv.gz"
            frame.to_csv(path, index=False, compression="gzip")
            if len(read_table(path)) != len(frame):
                raise RuntimeError(f"Row-count verification failed: {table}")
            report["rows"][table] = len(frame)
            report["tables"][table] = sha256(path)
    if files != source_files(run, manifest_root, result_root):
        raise RuntimeError("Source inventory changed during consolidation; stop writers and retry")
    # Recheck the entire transaction, including groups read near the start.
    for name, records in sources.items():
        root = result_root if name.startswith("eval_") else manifest_root
        for record in records:
            p = root / record["path"]
            if (p.stat().st_size, p.stat().st_mtime_ns, sha256(p)) != (
                record["size"], record["mtime_ns"], record["sha256"]
            ):
                raise RuntimeError("Source changed before publication; stop writers and retry")
    report["sources"] = sources
    report["created_utc"] = stamp
    (stage / "inventory.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    final = parent / stamp
    stage.rename(final)
    pointer = parent / (".latest-" + uuid.uuid4().hex + ".json")
    pointer.write_text(json.dumps({"snapshot": stamp}), encoding="utf-8")
    os.replace(pointer, parent / "LATEST.json")
    report["snapshot"] = str(final)
    return report


def load_consolidated(run: str, table: str, *, manifest_root: Path | None = None,
                      result_root: Path | None = None) -> pd.DataFrame | None:
    """Read the published snapshot if raw sources are absent or still match its inventory."""
    parent = consolidated_dir() / run
    pointer = parent / "LATEST.json"
    if not pointer.exists():
        return None
    snapshot = parent / json.loads(pointer.read_text(encoding="utf-8"))["snapshot"]
    info = json.loads((snapshot / "inventory.json").read_text(encoding="utf-8"))
    source_key = table.replace("trials_", "attempts_")
    manifest_root = manifest_root or manifests_dir()
    result_root = result_root or results_dir()
    actual = source_files(run, manifest_root, result_root).get(source_key, [])
    root = result_root if source_key.startswith("eval_") else manifest_root
    root_kind = "results" if source_key.startswith("eval_") else "manifests"
    default_root = results_dir() if root_kind == "results" else manifests_dir()
    # Default roots support downloaded snapshots from another machine. A custom
    # root must explicitly belong to this snapshot, even when it contains no files.
    if (root.resolve() != default_root.resolve()
            and str(root.resolve()) != info.get("source_roots", {}).get(root_kind)):
        return None
    recorded = info["sources"].get(source_key, [])
    if actual:
        state = {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in actual}
        expected = {p["path"]: (p["size"], p["mtime_ns"]) for p in recorded}
        if state != expected:
            return None
    path = snapshot / f"{table}.csv.gz"
    stat = path.stat()
    return _cached_table(path, stat.st_size, stat.st_mtime_ns, info["tables"][table]).copy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    result = consolidate(args.run, apply=args.apply, manifest_root=args.manifest_root,
                         result_root=args.result_root, destination=args.destination)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "tables")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
