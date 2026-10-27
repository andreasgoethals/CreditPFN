"""Remove one model family's final-benchmark results before re-scoring it.

    python -m src.utils.reset_family_benchmark --experiment experiment1 --family tabicl
    python -m src.utils.reset_family_benchmark --experiment experiment1 --family tabicl --apply

Lists by default. With ``--apply`` it deletes the family's original-model and adapted-model
result folders (per-fold CSVs, receipts and saved predictions) for both tasks on both
storage tiers, and supersedes the experiment's consolidated snapshot pointers.

Why the snapshot pointers: consolidation refuses to replace a snapshot whose raw sources
have disappeared, which is exactly what deleting a family's CSVs causes. The old snapshot
folders are kept; only ``LATEST.json`` is renamed, so the next consolidation publishes a
fresh snapshot from the re-scored results.

Never touches checkpoints, training records, plans, other families or classical baselines.
The shared evaluation cache is left alone: its keys include the inference settings, so a
changed precision policy can never reuse an old entry.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

from src.utils.paths import OUTPUT_DIR_NAME, consolidated_dir, experiment_group, outputs_dir, resolve_staging_path

FAMILIES = ("tabicl", "tabpfn")
TRACK_DIRS = ("PD", "LGD")


def family_dirs(experiment: str, family: str) -> list[Path]:
    """Result folders of one family (``<family>-untuned__*`` and ``<family>-trained__*``)."""
    group = experiment_group(experiment)
    if group in ("general", "experiment0", "experiment4"):
        raise ValueError("Select a benchmarked research experiment (1-3 or 5-8)")
    if family not in FAMILIES:
        raise ValueError(f"Family must be one of {FAMILIES}")
    found = []
    for root in dict.fromkeys((outputs_dir(), resolve_staging_path(OUTPUT_DIR_NAME))):
        for track in TRACK_DIRS:
            folder = root / group / "results" / track
            if folder.is_dir():
                found += sorted(p for p in folder.iterdir()
                                if p.is_dir() and p.name.startswith((f"{family}-untuned__", f"{family}-trained__")))
    return list(dict.fromkeys(found))


def _marker(results_root: Path, family: str) -> Path:
    return results_root / f".reset-{family}.json"


def stale_results(experiment: str, family: str) -> list[Path]:
    """Family result files written before the last reset (all of them if never reset).

    Files written after the reset are the re-scored results of an interrupted run; they must
    not block its resubmission, which skips the cells they already complete.
    """
    import json
    stale = []
    for folder in family_dirs(experiment, family):
        marker = _marker(folder.parents[1], family)
        cutoff = json.loads(marker.read_text(encoding="utf-8"))["reset_ns"] if marker.is_file() else None
        stale += [f for f in folder.rglob("*") if f.is_file() and (cutoff is None or f.stat().st_mtime_ns < cutoff)]
    return stale


def snapshot_pointers(experiment: str) -> list[Path]:
    root = consolidated_dir(experiment_group(experiment))
    return sorted(root.glob("*/LATEST.json")) if root.is_dir() else []


def missing_snapshot_sources(experiment: str, family: str) -> list[str]:
    """Sources of the current snapshots, other than this family's results, that no longer exist.

    Superseding a pointer bypasses consolidation's protection against publishing a partial
    snapshot. That is only safe when the family's own CSVs are the sole missing sources.
    """
    import json
    from src.utils.paths import manifests_dir, results_dir, training_dir
    group = experiment_group(experiment)
    roots = {"results": results_dir(experiment=group), "training": training_dir(experiment=group),
             "manifests": manifests_dir(group)}
    missing = []
    for pointer in snapshot_pointers(experiment):
        folder = pointer.parent / json.loads(pointer.read_text(encoding="utf-8"))["snapshot"]
        inventory = json.loads((folder / "inventory.json").read_text(encoding="utf-8"))
        for name, records in inventory["sources"].items():
            kind = ("results" if name.startswith("eval_") else
                    "training" if name.startswith(("training_", "parameters_", "resources_")) else "manifests")
            for record in records:
                parts = Path(record["path"]).parts
                own = kind == "results" and any(part.startswith((f"{family}-untuned__", f"{family}-trained__"))
                                                for part in parts)
                if not own and not (roots[kind] / record["path"]).is_file():
                    missing.append(f"{pointer.parent.name}: {record['path']}")
    return missing


def _refuse_links(paths):
    for path in paths:
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise RuntimeError(f"Refusing to follow a link: {path}")


def reset(experiment: str, family: str, *, apply: bool = False) -> dict:
    folders, pointers = family_dirs(experiment, family), snapshot_pointers(experiment)
    _refuse_links(folders + pointers)
    files = sum(1 for folder in folders for p in folder.rglob("*") if p.is_file())
    missing = missing_snapshot_sources(experiment, family)
    report = dict(experiment=experiment, family=family, folders=[str(p) for p in folders], files=files,
                  snapshot_pointers=[str(p) for p in pointers], missing_snapshot_sources=missing, applied=False)
    if not apply:
        return report
    if missing:
        raise RuntimeError(f"{len(missing)} non-{family} sources of the consolidated snapshot are already missing "
                           "(e.g. archived raw shards). Restore them first; nothing was deleted.")
    from src.utils.clean_run import check_cluster_writers
    check_cluster_writers(experiment_group(experiment))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for pointer in pointers:
        pointer.rename(pointer.with_name(f"LATEST.superseded-{family}-{stamp}.json"))
    for folder in folders:
        shutil.rmtree(folder)
    import json
    import time
    group = experiment_group(experiment)
    for root in dict.fromkeys((outputs_dir(), resolve_staging_path(OUTPUT_DIR_NAME))):
        results_root = root / group / "results"
        if results_root.is_dir():
            # Written last: re-scored files are newer than this and never count as stale.
            _marker(results_root, family).write_text(json.dumps({"reset_ns": time.time_ns(), "utc": stamp}),
                                                     encoding="utf-8")
    report["applied"] = True
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--apply", action="store_true", help="delete; default only lists")
    args = parser.parse_args(argv)
    report = reset(args.experiment, args.family, apply=args.apply)
    action = "Deleted" if report["applied"] else "Would delete"
    print(f"{action} {len(report['folders'])} {args.family} result folders ({report['files']} files) "
          f"for {args.experiment}:")
    for folder in report["folders"]:
        print("  " + folder)
    verb = "Superseded" if report["applied"] else "Would supersede"
    for pointer in report["snapshot_pointers"]:
        print(f"{verb} consolidated pointer: {pointer}")
    if report["missing_snapshot_sources"]:
        print(f"WARNING: {len(report['missing_snapshot_sources'])} other snapshot sources are missing; --apply will refuse:")
        for item in report["missing_snapshot_sources"][:10]:
            print("  " + item)
    if not report["applied"]:
        print("Listing only. Re-run with --apply after stopping this experiment's jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
