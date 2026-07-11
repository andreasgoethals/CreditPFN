#!/usr/bin/env python
"""Delete all OUTPUTS of a previous CreditPFN run, across BOTH VSC storage tiers.

What it deletes
---------------
  * project STAGING (big files):  checkpoints/trained/ , output/results/
  * $VSC_DATA / output root:       logs/ , output/training/manifests/ ,
                                   output/training/epochs/ , output/figures/

What it NEVER touches (inputs — deleting these would force a re-upload):
  * data/  (raw + processed datasets)
  * checkpoints/*.ckpt  (the base Prior Labs weights — only checkpoints/TRAINED/ goes)

It resolves every path through ``src.utils.paths``, so it automatically hits
the right places in every environment:
  * on the VSC  → project staging (Lustre, e.g. /lustre1/project/stg_00211/CreditPFN)
                  for checkpoints+results, and $VSC_DATA/CreditPFN for logs+manifests;
  * locally     → everything under the repo root.

Usage
-----
    python scripts/clean_run.py            # DRY-RUN: preview what would be deleted
    python scripts/clean_run.py --yes      # actually delete
    python scripts/clean_run.py --yes --keep-logs    # delete outputs but keep logs/
    python scripts/clean_run.py --yes --fresh-data   # ALSO delete data/processed/
                                                     # (keeps data/raw) so the data
                                                     # stage regenerates it from scratch

Run it on a Genius login node (it can reach both Lustre staging and $VSC_DATA)
or on your laptop. It imports only src.utils.paths (no torch/tabpfn), so it's
instant and needs no GPU.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.utils.paths import (  # noqa: E402
    get_roots, is_vsc_environment, resolve_output_path, resolve_staging_path,
)

# (label, tier, relpath). tier selects the resolver:
#   "staging" → resolve_staging_path (project storage: big files)
#   "output"  → resolve_output_path  ($VSC_DATA: logs/manifests/figures)
#   "dataset" → resolve_data_path    (where datasets live; --fresh-data only)
_TARGETS = [
    ("trained checkpoints", "staging", "checkpoints/trained"),
    # CRITICAL second location (bug found 2026-07-11): when staging is not
    # writable from the Mindwell compute nodes, training saves checkpoints to
    # the $VSC_DATA FALLBACK dir (resolve_writable_staging_path). A "clean"
    # rerun that misses this dir silently SKIPs trials against the previous
    # run's checkpoints — exactly what contaminated the Jul-10 rerun (59/64
    # trials reused old FP16 checkpoints).
    ("trained checkpoints (fallback)", "output", "checkpoints/trained"),
    ("benchmark results",   "staging", "output/results"),
    ("run logs",            "output",  "logs"),
    ("training manifests",  "output",  "output/training/manifests"),
    ("per-epoch CSVs",      "output",  "output/training/epochs"),
    ("notebook figures",    "output",  "output/figures"),
    # Orchestration state: data_done/train_ok_* sentinels + generated
    # eval_submit_*.sh scripts. Stale sentinels can't gate a NEW run (the
    # submitter clears them) but they accumulate and confuse debugging.
    ("orchestration sentinels", "output", ".sentinels"),
]

# Hard safety denylist: a resolved target whose final component is any of these
# is refused — guards against ever wiping the RAW datasets, the base-checkpoint
# root, or a source tree. ("processed" is intentionally NOT here — it's a valid
# target only under the explicit --fresh-data flag; "raw" always stays.)
_FORBIDDEN_LEAVES = {"data", "raw", "checkpoints", "src", "scripts",
                     "config", "tests", "docs", "repositories", "papers"}


def _resolve(tier: str, rel: str) -> Path:
    if tier == "staging":
        return resolve_staging_path(rel)
    if tier == "dataset":
        from src.utils.paths import resolve_data_path
        return resolve_data_path(rel)
    return resolve_output_path(rel)          # "output"


def _dir_stats(p: Path) -> tuple[int, int]:
    n = total = 0
    for f in p.rglob("*"):
        if f.is_file():
            n += 1
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return n, total


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}PB"


def _is_safe(p: Path) -> bool:
    """Refuse to delete a filesystem root or an input directory."""
    if p == p.anchor or len(p.parts) <= 2:
        return False
    return p.name not in _FORBIDDEN_LEAVES


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Delete previous-run outputs across both VSC storage tiers "
                    "(dry-run unless --yes). Datasets + base checkpoints are preserved.",
    )
    ap.add_argument("--yes", action="store_true",
                    help="actually delete (default is a dry-run preview)")
    ap.add_argument("--keep-logs", action="store_true",
                    help="keep the logs/ directory (delete only checkpoints/results/etc.)")
    ap.add_argument("--fresh-data", action="store_true",
                    help="ALSO delete data/processed/ (keeps data/raw/) so the data "
                         "stage regenerates the sanitized CSVs from scratch — needed "
                         "when the sanitize logic changed since the last run.")
    args = ap.parse_args()

    targets = list(_TARGETS)
    if args.fresh_data:
        targets.append(("processed datasets", "dataset", "data/processed"))

    roots = get_roots()
    print("=" * 74)
    print(f"CreditPFN clean_run — environment: {'VSC' if is_vsc_environment() else 'local'}")
    print(f"  staging_root : {roots['staging_root']}   (checkpoints + results)")
    print(f"  output_root  : {roots['output_root']}   (logs + manifests + figures)")
    print("=" * 74)

    total_files = total_bytes = 0
    to_delete: list[tuple[str, Path, int, int]] = []
    seen_paths: set[str] = set()
    for label, tier, rel in targets:
        if args.keep_logs and rel == "logs":
            continue
        p = _resolve(tier, rel)
        # Locally, staging/output both resolve to the repo root — skip dupes.
        if str(p) in seen_paths:
            continue
        seen_paths.add(str(p))
        tag = tier.upper()
        if not _is_safe(p):
            print(f"  [REFUSE] {label:22s} unsafe path, skipping: {p}")
            continue
        if p.exists() and p.is_dir():
            n, b = _dir_stats(p)
            to_delete.append((label, p, n, b))
            total_files += n
            total_bytes += b
            verb = "DELETE" if args.yes else "would del"
            print(f"  [{tag:7s}] {verb:9s} {label:22s} {n:6d} files {_human(b):>10s}  {p}")
        else:
            print(f"  [{tag:7s}] absent    {label:22s}                        {p}")

    print("-" * 74)
    print(f"TOTAL: {total_files} files, {_human(total_bytes)}")
    print("PRESERVED (never deleted): data/raw/ (raw datasets) and "
          "checkpoints/*.ckpt (base weights)."
          + ("" if args.fresh_data else " Also data/processed/ (pass --fresh-data to rebuild it)."))

    if not args.yes:
        print("\nDRY-RUN — nothing deleted. Re-run with --yes to delete.")
        return 0

    for label, p, _, _ in to_delete:
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)   # recreate empty so the pipeline's mkdir -p is a no-op
        print(f"  deleted + recreated empty: {p}")
    print(f"\nDone. Deleted {total_files} files ({_human(total_bytes)}) across both tiers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
