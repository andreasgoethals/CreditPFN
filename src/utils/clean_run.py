"""Wipe what the previous run produced, so the next one starts clean.

    python -m src.utils.clean_run                          list what is there, delete nothing
    python -m src.utils.clean_run --clean                   delete it
    python -m src.utils.clean_run --clean --processed       ...and the data/processed cache too
    python -m src.utils.clean_run --clean --stages eval     only what the eval stage produced

Clears `output CreditPFN/` on both storage tiers, preserving a maintenance job's active log — so
one invocation is enough whether you are on a laptop or on the cluster. Off-cluster both tiers
collapse into the repository and it is simply `output CreditPFN/`.

`--processed` additionally clears `data/processed/`, the preprocessing cache. It is separate
because rebuilding that cache can cost far more than re-running the notebooks, so "clean the last
run" should not silently throw it away.

LISTS BY DEFAULT. The two mistakes are not symmetric: a listing you meant as a deletion costs one
more command, and a deletion you meant as a listing costs the run.

NEVER TOUCHES `data/raw/`, original base weights, or `tfm-library/`.
Trained weights ARE removed by a full clean. Stop all experiment writers first;
download any historical output you want to keep before using `--clean`.

CREDITPFN ADDS TWO TREES the generic version cannot know about, both outside `output CreditPFN/`:

  * `checkpoints/trained/` on **both** tiers. Trained checkpoints normally go to project
    storage, but `resolve_writable_staging_path` falls back to `$VSC_DATA` when staging is
    unwritable from the compute node — and a clean that misses the fallback leaves the
    resume-skip check pointing at the previous run's weights. That is exactly what
    contaminated the 10-07-2026 rerun: 59 of 64 trials silently reused stale checkpoints.
    Base `checkpoints/*.ckpt` are never touched — only the `trained/` subtree.
  * Legacy `.sentinels/`, if present. The named launcher uses Slurm dependencies and
    writes its submission state under `output CreditPFN/manifests/`; no new sentinels are created.

`--stages data,train,eval` narrows the wipe to what one stage produced, which is what you
want when only the last stage needs redoing — re-running eval is minutes, re-running the
data pipeline is not. Without it, everything goes. (This replaced a second cleaner,
`pipeline_clean.py`, on 11-08-2026: two scripts that both delete things is one too many,
and the stage list is an argument, not a program.)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.utils.paths import (
    checkpoints_dir,
    outputs_dir,
    processed_dir,
    resolve_output_path,
    resolve_staging_path,
    results_dir,
)

#: Tracked so an empty directory survives a clone. Not run output, so never counted or deleted —
#: removing them would leave a fresh clone with nowhere to write.
KEEP = frozenset({".gitkeep", ".gitignore"})

#: A resolved root whose last component is one of these is REFUSED, however it got there.
#: Cheap insurance against a mis-resolved root (an unset env var collapsing a path to the
#: repository) wiping the raw datasets, the base weights or the source tree.
FORBIDDEN_LEAVES = frozenset({
    "data", "raw", "checkpoints", "src", "scripts", "config", "tests", "docs", "tfm-library",
})


def is_safe(root: Path) -> bool:
    """False for a filesystem root, a suspiciously short path, or an input directory."""
    return (root != Path(root.anchor)
            and len(root.parts) > 2
            and root.name not in FORBIDDEN_LEAVES)


#: What each pipeline stage puts on disk, as (glob-root, patterns). A stage's targets are
#: files, not directories, because the three stages share directories: `output CreditPFN/logs/` holds
#: all three stages' logs and `output CreditPFN/manifests/` holds both the data manifests and the
#: training ones. Wiping by directory would take another stage's work with it.
STAGES = ("data", "train", "eval")


def stage_targets(stage: str) -> list[Path]:
    """Every file the named stage produced. See `STAGES` for why this is file-level."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; valid: {', '.join(STAGES)}")
    out_root, logs = outputs_dir(), outputs_dir() / "logs"
    found: list[Path] = []
    if stage == "data":
        found += list(processed_dir().glob("**/*.sanitized.csv"))
        found += list(processed_dir().glob("**/*.sanitized.feature_groups.json"))
        found += list((out_root / "manifests").glob("manifest_*.csv"))
        found += list(logs.glob("data_*.log"))
    elif stage == "train":
        for trained in (checkpoints_dir("trained"), resolve_output_path("checkpoints/trained")):
            # Includes provenance and optimizer recovery states on BOTH tiers.
            found += list(trained.glob("**/*"))
        found += [p for p in (out_root / "manifests").glob("*.csv")
                  if not p.name.startswith("manifest_")]
        found += list((out_root / "manifests" / "epochs").glob("**/*.csv"))
        found += list(logs.glob("train_*.log"))
    else:                                                       # eval
        found += list(results_dir().glob("**/*"))
        found += list(resolve_staging_path("output CreditPFN/evaluation_cache").glob("**/*"))
        found += list((out_root / "figures").glob("**/*"))
        found += list((out_root / "manifests" / "figures").glob("*.json"))
        found += list(logs.glob("notebook_*.log"))
        found += list(logs.glob("eval_*.log"))
    return list(dict.fromkeys(p for p in found if p.is_file() and p.name not in KEEP))


def roots(*, processed: bool = False) -> list[Path]:
    """Every tree to clear. Two `output CreditPFN/` roots on the cluster, one locally, plus the cache.

    Project storage also holds consolidated tables, reusable evaluation caches and
    any legacy archives. Clear its WHOLE output tree, not only output CreditPFN/results.
    """
    found = [outputs_dir()]
    project_output = resolve_staging_path("output CreditPFN")
    if not project_output.is_relative_to(found[0]):
        found.append(project_output)
    # CreditPFN: trained weights on project storage AND the $VSC_DATA fallback, plus the
    # cross-cluster sentinels. See the module docstring for why both locations matter.
    for extra in (checkpoints_dir("trained"),
                  resolve_output_path("checkpoints/trained"),
                  resolve_output_path(".sentinels")):
        if not any(extra == f or extra.is_relative_to(f) for f in found):
            found.append(extra)
    if processed:
        found.append(processed_dir())
    return found


def validate_tree(root: Path) -> None:
    """Preflight an entire deletion tree; never follow symlinks or Windows junctions."""
    if not is_safe(root) or not is_safe(root.resolve()):
        raise ValueError(f"Refusing unsafe cleanup root: {root}")
    anchor = root.resolve()

    def check(path: Path) -> None:
        if (path.is_symlink() or getattr(path, "is_junction", lambda: False)()
                or not path.resolve().is_relative_to(anchor)):
            raise ValueError(f"Refusing linked or escaped cleanup path: {path}")

    check(root)
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in folders + files:
            check(Path(directory) / name)


def measure(root: Path, *, keep_paths: frozenset[Path] = frozenset()) -> tuple[int, int]:
    """(files, bytes) under a root, ignoring the structure markers."""
    if not root.is_dir():
        return 0, 0
    files = [p for p in root.rglob("*")
             if p.is_file() and p.name not in KEEP and p.resolve() not in keep_paths]
    return len(files), sum(p.stat().st_size for p in files)


def wipe(root: Path, *, keep_paths: frozenset[Path] = frozenset()) -> int:
    """Delete everything under a root except the structure markers. Returns files removed.

    Two passes, and the order matters: files first, then empty directories bottom-up. That leaves
    exactly the directories holding a tracked `.gitkeep` and removes the per-run ones
    (`figures/<notebook>/`) that do not. An `rmtree` of the subtree would take
    `output CreditPFN/figures/.gitkeep` with it, and the next clone would have nowhere to write.
    """
    validate_tree(root)
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.rglob("*"):
        if path.is_file() and path.name not in KEEP and path.resolve() not in keep_paths:
            path.unlink()
            removed += 1
    for path in sorted((p for p in root.rglob("*") if p.is_dir()),
                       key=lambda p: len(p.parts), reverse=True):
        if not any(path.iterdir()):
            path.rmdir()
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clean", action="store_true", help="actually delete; default lists only")
    parser.add_argument("--processed", action="store_true",
                        help="also clear data/processed/, the preprocessing cache")
    parser.add_argument("--stages", default=None,
                        help="comma-separated subset of " + ",".join(STAGES) +
                             " — clear only what those stages produced, instead of the "
                             "whole output CreditPFN/ tree")
    args = parser.parse_args(argv)

    # Unlinking a running job's stdout on Linux loses the cleanup report itself.
    # Only the exact active .log inside our log directory may survive a reset.
    active_log = os.environ.get("CREDITPFN_ACTIVE_LOG")
    keep_paths = frozenset()
    if active_log:
        path = Path(active_log).resolve()
        if path.parent != (outputs_dir() / "logs").resolve() or path.suffix != ".log":
            parser.error("CREDITPFN_ACTIVE_LOG must name a .log directly inside output CreditPFN/logs")
        keep_paths = frozenset({path})
        print(f"Preserving active maintenance log: {path}")

    # Validate every tree before deleting ANY file, including stage-specific paths.
    # No writers may be active during a clean; this is not a concurrent deletion service.
    targets = roots(processed=args.processed or bool(
        args.stages and "data" in [name.strip() for name in args.stages.split(",")]))
    for root in targets:
        validate_tree(root)

    if args.stages:
        names = [x.strip() for x in args.stages.split(",") if x.strip()]
        bad = [x for x in names if x not in STAGES]
        if bad:
            parser.error(f"unknown stage(s) {bad}; valid: {', '.join(STAGES)}")
        files: list[Path] = []
        print("Output of stage(s) " + ", ".join(names) + ":\n")
        for name in names:
            got = [p for p in stage_targets(name) if p.resolve() not in keep_paths]
            size = sum(f.stat().st_size for f in got)
            print(f"  {len(got):>6} files  {size / 1e6:>9.1f} MB  {name}")
            files += got
        files = list(dict.fromkeys(files))
        total = sum(f.stat().st_size for f in files)
        print(f"\nTOTAL: {len(files)} files, {total / 1e9:.2f} GB")
        print("Never touched: data/raw/, the base checkpoints/*.ckpt, tfm-library/.")
        if not args.clean:
            print("\nNothing was deleted. Re-run with --clean to delete.")
            return 0
        for f in files:
            f.unlink(missing_ok=True)
        print(f"\nDeleted {len(files)} files. Clean.")
        return 0

    total_files = total_bytes = 0
    print("Output from the previous run:\n")
    for root in targets:
        files, size = measure(root, keep_paths=keep_paths)
        total_files += files
        total_bytes += size
        state = f"{files:>6} files  {size / 1e6:>9.1f} MB" if files else "         empty"
        print(f"  {state}  {root}")

    print(f"\nTOTAL: {total_files} files, {total_bytes / 1e9:.2f} GB")
    print("Never touched: data/raw/, the base checkpoints/*.ckpt, tfm-library/.")

    if not args.clean:
        if total_files:
            print("\nNothing was deleted. Re-run with --clean to delete.")
        return 0

    print("\nDeleting:")
    for root in targets:
        print(f"  removed {wipe(root, keep_paths=keep_paths):>6} files from {root}")
    print("\nClean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
