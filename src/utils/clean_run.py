"""Wipe what the previous run produced, so the next one starts clean.

    python -m src.utils.clean_run                          list what is there, delete nothing
    python -m src.utils.clean_run --clean                   delete it
    python -m src.utils.clean_run --clean --processed       ...and the data/processed cache too
    python -m src.utils.clean_run --clean --stages eval     only what the eval stage produced

Clears the **whole `output/` tree on both storage tiers** — `$VSC_DATA` and project storage — so
one invocation is enough whether you are on a laptop or on the cluster. Off-cluster both tiers
collapse into the repository and it is simply `output/`.

`--processed` additionally clears `data/processed/`, the preprocessing cache. It is separate
because rebuilding that cache can cost far more than re-running the notebooks, so "clean the last
run" should not silently throw it away.

LISTS BY DEFAULT. The two mistakes are not symmetric: a listing you meant as a deletion costs one
more command, and a deletion you meant as a listing costs the run.

NEVER TOUCHES `data/raw/` or `checkpoints/` or `tfm-library/` — the inputs are irreplaceable and
the weights are either downloaded or a training run to reproduce.

CREDITPFN ADDS TWO TREES the generic version cannot know about, both outside `output/`:

  * `checkpoints/trained/` on **both** tiers. Trained checkpoints normally go to project
    storage, but `resolve_writable_staging_path` falls back to `$VSC_DATA` when staging is
    unwritable from the compute node — and a clean that misses the fallback leaves the
    resume-skip check pointing at the previous run's weights. That is exactly what
    contaminated the 10-07-2026 rerun: 59 of 64 trials silently reused stale checkpoints.
    Base `checkpoints/*.ckpt` are never touched — only the `trained/` subtree.
  * `.sentinels/` — the `data_done` / `train_ok_<track>_<i>` files the cross-cluster gate
    polls, plus the generated `eval_submit_*.sh`. Stale sentinels cannot release a NEW run
    (the submitter clears them first) but they make a log impossible to read.

`--stages data,train,eval` narrows the wipe to what one stage produced, which is what you
want when only the last stage needs redoing — re-running eval is minutes, re-running the
data pipeline is not. Without it, everything goes. (This replaced a second cleaner,
`pipeline_clean.py`, on 11-08-2026: two scripts that both delete things is one too many,
and the stage list is an argument, not a program.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.paths import (
    checkpoints_dir,
    outputs_dir,
    processed_dir,
    resolve_output_path,
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
#: files, not directories, because the three stages share directories: `output/logs/` holds
#: all three stages' logs and `output/manifests/` holds both the data manifests and the
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
        found += list((out_root / "manifests" / "dedup").glob("*.csv"))
        found += list((out_root / "manifests").glob("manifest_*.csv"))
        found += list(logs.glob("data_*.log"))
    elif stage == "train":
        trained = checkpoints_dir("trained")
        found += [p for p in trained.glob("**/*.ckpt")]
        found += list(trained.glob("**/*.provenance.json"))
        found += list(resolve_output_path("checkpoints/trained").glob("**/*.ckpt"))
        found += [p for p in (out_root / "manifests").glob("*.csv")
                  if not p.name.startswith("manifest_")]
        found += list((out_root / "manifests" / "epochs").glob("**/*.csv"))
        found += list(logs.glob("train_*.log"))
    else:                                                       # eval
        found += list(results_dir().glob("**/*.csv"))
        found += list((out_root / "figures").glob("**/*"))
        found += list(logs.glob("eval_*.log"))
    return [p for p in found if p.is_file() and p.name not in KEEP]


def roots(*, processed: bool = False) -> list[Path]:
    """Every tree to clear. Two `output/` roots on the cluster, one locally, plus the cache.

    `results_dir()` is listed separately because on the cluster it is the one part of `output/`
    on project storage — clearing only `outputs_dir()` there would leave the largest files behind.
    """
    found = [outputs_dir()]
    results = results_dir()
    if not results.is_relative_to(found[0]):
        found.append(results)
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


def measure(root: Path) -> tuple[int, int]:
    """(files, bytes) under a root, ignoring the structure markers."""
    if not root.is_dir():
        return 0, 0
    files = [p for p in root.rglob("*") if p.is_file() and p.name not in KEEP]
    return len(files), sum(p.stat().st_size for p in files)


def wipe(root: Path) -> int:
    """Delete everything under a root except the structure markers. Returns files removed.

    Two passes, and the order matters: files first, then empty directories bottom-up. That leaves
    exactly the directories holding a tracked `.gitkeep` and removes the per-run ones
    (`figures/<notebook>/`) that do not. An `rmtree` of the subtree would take
    `output/figures/.gitkeep` with it, and the next clone would have nowhere to write.
    """
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.rglob("*"):
        if path.is_file() and path.name not in KEEP:
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
                             "whole output/ tree")
    args = parser.parse_args(argv)

    if args.stages:
        names = [x.strip() for x in args.stages.split(",") if x.strip()]
        bad = [x for x in names if x not in STAGES]
        if bad:
            parser.error(f"unknown stage(s) {bad}; valid: {', '.join(STAGES)}")
        files: list[Path] = []
        print("Output of stage(s) " + ", ".join(names) + ":\n")
        for name in names:
            got = stage_targets(name)
            size = sum(f.stat().st_size for f in got)
            print(f"  {len(got):>6} files  {size / 1e6:>9.1f} MB  {name}")
            files += got
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

    targets = []
    for root in roots(processed=args.processed):
        if is_safe(root):
            targets.append(root)
        else:
            print(f"  REFUSED (unsafe path, not a run output): {root}")
    total_files = total_bytes = 0
    print("Output from the previous run:\n")
    for root in targets:
        files, size = measure(root)
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
        print(f"  removed {wipe(root):>6} files from {root}")
    print("\nClean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
