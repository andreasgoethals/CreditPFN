"""Wipe what the previous run produced, so the next one starts clean.

    python -m src.utils.clean_run                          list what is there, delete nothing
    python -m src.utils.clean_run --experiment experiment5 --clean   delete one experiment
    python -m src.utils.clean_run --all-experiments --clean          explicit global deletion
    python -m src.utils.clean_run --all-experiments --clean --stages eval   all evaluations

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

  * `checkpoints/trained/` on **both** tiers. Current cluster jobs require project storage,
    but historical versions allowed a DATA fallback. Include any such remnants so a clean
    start cannot accidentally reuse their weights.
    Base `checkpoints/*.ckpt` are never touched — only the `trained/` subtree.
  * Legacy `.sentinels/`, if present. The named launcher uses Slurm dependencies and
    writes shared pool state under `output CreditPFN/general/manifests/`; no new sentinels are created.

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
    OUTPUT_DIR_NAME,
    checkpoints_dir,
    outputs_dir,
    processed_dir,
    resolve_output_path,
    resolve_staging_path,
    results_dir,
)

#: Preserve optional structure markers. Writers create missing directories themselves.
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


#: Stage selection operates on files because experiment log/manifest directories can contain
#: records from several stages. Removing those directories would erase another stage's work.
STAGES = ("data", "train", "eval")


def stage_targets(stage: str) -> list[Path]:
    """Every file the named stage produced. See `STAGES` for why this is file-level."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; valid: {', '.join(STAGES)}")
    from src.utils.paths import EXPERIMENTS, manifests_dir, logs_dir, training_dir
    found: list[Path] = []
    if stage == "data":
        found += list(processed_dir().glob("**/*.sanitized.csv"))
        found += list(processed_dir().glob("**/*.sanitized.feature_groups.json"))
        found += list(manifests_dir("general").glob("manifest_*.csv"))
        found += list(logs_dir("general").glob("data_*.log"))
    elif stage == "train":
        for trained in (checkpoints_dir("trained"), resolve_output_path("checkpoints/trained")):
            found += list(trained.glob("**/*"))
        for group in EXPERIMENTS:
            found += [p for p in manifests_dir(group).glob("*.csv") if not p.name.startswith("manifest_")]
            found += list(training_dir(experiment=group).glob("**/*"))
            found += list(logs_dir(group).glob("train_*.log"))
    else:
        for group in EXPERIMENTS:
            found += list(results_dir(experiment=group).glob("**/*"))
            found += list(resolve_staging_path(f"output CreditPFN/{group}/evaluation_cache").glob("**/*"))
            found += list((outputs_dir() / group / "figures").glob("**/*"))
            found += list((manifests_dir(group) / "figures").glob("**/*.json"))
            found += list(logs_dir(group).glob("eval_*.log"))
    return list(dict.fromkeys(p for p in found if p.is_file() and p.name not in KEEP))


def roots(*, processed: bool = False) -> list[Path]:
    """Every tree to clear. Two `output CreditPFN/` roots on the cluster, one locally, plus the cache.

    Project storage also holds consolidated tables, reusable evaluation caches and
    any legacy archives. Clear its whole output tree, not only result subdirectories.
    """
    found = [outputs_dir()]
    project_output = resolve_staging_path(OUTPUT_DIR_NAME)
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


def experiment_roots(experiment: str, scope: str = "all") -> list[Path]:
    """Explicit experiment ownership on both tiers; exclude shared caches/data."""
    from src.utils.paths import experiment_group
    group = experiment_group(experiment)
    if group == "general":
        raise ValueError("General contains shared submission state; select a numbered experiment")
    if scope not in {"all", "training", "benchmark", "figures"}:
        raise ValueError("Unknown cleanup scope")
    selected = []
    for root in (outputs_dir(), resolve_staging_path(OUTPUT_DIR_NAME)):
        if scope == "all":
            selected.append(root / group)
        elif scope == "training":
            selected.append(root / group / "training")
        elif scope == "benchmark":
            selected += [root / group / "results", root / group / "evaluation_cache"]
        if scope in {"all", "figures"}:
            selected.append(root / "figures" / group)
    if scope in {"all", "training"}:
        selected += [checkpoints_dir("trained", group), resolve_output_path(f"checkpoints/trained/{group}")]
    return list(dict.fromkeys(selected))


def check_cluster_writers(experiment=None):
    """Protect writers and cross-experiment inputs; unknown ownership fails closed."""
    if not os.environ.get("VSC_DATA"):
        return
    import subprocess
    own = os.environ.get("SLURM_JOB_ID", "")
    import re
    active = []
    consumers = {experiment}
    if experiment == "experiment0":
        # New jobs recheck these validation receipts, including on recovery.
        consumers.update({"experiment4", "experiment5", "experiment6", "experiment7", "experiment8"})
    elif experiment == "experiment1":
        # Context curves load the fixed experiment-1 checkpoints at every cell.
        consumers.add("experiment4")
    elif experiment == "experiment6":
        # The episode study reuses this exact large-table control and plan.
        consumers.add("experiment7")
    for cluster in ("mindwell", "wice"):
        result = subprocess.run(["squeue", "-M", cluster, "-u", os.environ["USER"],
                                 "-h", "-o", "%i|%j"], text=True, capture_output=True, timeout=45, check=True)
        seen = set()
        for line in result.stdout.splitlines():
            if "creditpfn" not in line.lower():
                continue
            job, name = line.split("|", 1)
            job = job.strip().split('_')[0]
            if job == own or job in seen:
                continue
            seen.add(job)
            owner = ("experiment4" if "context-eval" in name else
                     "experiment0" if "followup-" in name else None)
            if experiment and owner is None:
                details = subprocess.run(["scontrol", "-M", cluster, "show", "job", "-o", job],
                                         text=True, capture_output=True, timeout=45, check=True).stdout
                groups = set(re.findall(r"(?:config/|CREDITPFN_EXPERIMENT=)(experiment[0-8])(?:/|,|\s|$)", details))
                if len(groups) == 1:
                    owner = groups.pop()
            if experiment is None or owner is None or owner in consumers:
                active.append(job)
    if active:
        raise RuntimeError("CreditPFN jobs are still queued/running; stop the selected writers/input consumers before cleanup: "
                           + ", ".join(active[:12]))


def check_saved_consumers(experiment, scope):
    """Preserve a shared scientific control even when its consumer is not running."""
    if experiment != "experiment6" or scope == "figures":
        return
    import json
    plans = outputs_dir() / "experiment7" / "manifests" / "plans"
    for path in plans.glob("*.json"):
        if json.loads(path.read_text(encoding="utf-8")).get("shared_reference"):
            raise RuntimeError("Experiment 7 still references experiment 6. Preserve the shared control, "
                               "or explicitly retire experiment 7 first; scoped cleanup deleted nothing.")


def lora_workflow_roots(identifier):
    """Exact isolated workflow ownership on both tiers; no token/glob matching."""
    from src.utils.lora_checks import validate_id
    validate_id(identifier)
    selected = [root / "experiment0" / "lora" / identifier
                for root in (outputs_dir(), resolve_staging_path(OUTPUT_DIR_NAME))]
    selected += [root / "experiment0" / "lora" / identifier
                 for root in (checkpoints_dir("trained"), resolve_output_path("checkpoints/trained"))]
    return list(dict.fromkeys(selected))


def check_lora_writers(identifier):
    """Fail closed for an unknown owner or a job consuming this workflow's gate."""
    if not os.environ.get("VSC_DATA"):
        return
    import re
    import subprocess
    own = os.environ.get("SLURM_JOB_ID", "")
    active = []
    for cluster in ("mindwell", "wice"):
        queue = subprocess.run(["squeue", "-M", cluster, "-u", os.environ["USER"], "-h", "-o", "%i|%j"],
                               text=True, capture_output=True, timeout=45, check=True)
        seen = set()
        for line in queue.stdout.splitlines():
            if "creditpfn" not in line.lower():
                continue
            job = line.split("|", 1)[0].strip().split("_")[0]
            if job == own or job in seen:
                continue
            seen.add(job)
            details = subprocess.run(["scontrol", "-M", cluster, "show", "job", "-o", job],
                                     text=True, capture_output=True, timeout=45, check=True).stdout
            workflow = re.search(r"lora_(?:gpu|prepare)\.slurm\s+(?:(?:smoke|capacity)\s+)?([A-Za-z0-9]+)(?:\s|$)", details)
            groups = set(re.findall(r"(?:config/|CREDITPFN_EXPERIMENT=)(experiment[0-8])(?:/|,|\s|$)", details))
            if workflow:
                if workflow.group(1) == identifier:
                    active.append(job)
            elif len(groups) != 1 or "experiment8" in groups or "experiment0" in groups:
                # sbatch --export=ALL often hides the gate variable from SubmitLine.
                # All production LoRA consumers must therefore finish first.
                active.append(job)
    if active:
        raise RuntimeError("LoRA workflow writers/consumers or jobs of unknown ownership are active: "
                           + ", ".join(active[:12]))


def clean_lora_workflow(identifier, *, clean=False):
    import json
    targets = lora_workflow_roots(identifier)
    # Inspect every tier before deleting the first file. Reject linked ancestors,
    # unknown layouts and unknown IDs instead of searching similarly named runs.
    for root in targets:
        validate_tree(root)
    marker = targets[0] / "workflow.json"
    if not marker.is_file():
        marker = targets[0] / ".scope.json"
    if not marker.is_file():
        raise ValueError("Unknown LoRA workflow: its preparation/scope marker is missing")
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("kind") not in {"creditpfn_lora", "creditpfn_lora_scope"} or record.get("id") != identifier:
        raise ValueError("Unknown LoRA workflow: preparation marker does not match")
    keep = frozenset()
    if os.environ.get("CREDITPFN_ACTIVE_LOG"):
        active = Path(os.environ["CREDITPFN_ACTIVE_LOG"]).resolve()
        if active.suffix != ".log":
            raise ValueError("Active cleanup log must be a .log")
        # A maintenance log outside these exact roots is unaffected; an inside
        # log is retained, together with the marker so a later cleanup is possible.
        if any(active.is_relative_to(root.resolve()) for root in targets):
            keep = frozenset({active, marker.resolve()})
    if clean:
        check_lora_writers(identifier)
    total = 0
    for root in targets:
        count, size = measure(root, keep_paths=keep)
        total += count
        print(f"{count:>6} files {size / 1e6:>9.1f} MB  {root}")
    if not clean:
        print(f"LoRA workflow {identifier}: {total} files. Nothing deleted. Add --clean after inspection.")
        return 0
    for root in targets:
        wipe(root, keep_paths=keep)
    print(f"Deleted {total} files belonging only to LoRA workflow {identifier}.")
    return 0


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
    for parent in root.parents:
        if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
            raise ValueError(f"Refusing linked cleanup ancestor: {parent}")
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
    directories holding optional structure markers and removes empty per-run directories.
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
    parser.add_argument("--experiment", help="only this experiment on both storage tiers")
    parser.add_argument("--scope", choices=["all", "training", "benchmark", "figures"], default="all")
    parser.add_argument("--workflow", help="only the named experiment-0 follow-up workflow")
    parser.add_argument("--lora-workflow", help="only the named isolated experiment-0 LoRA workflow on both tiers")
    parser.add_argument("--all-experiments", action="store_true", help="explicitly permit a global clean")
    parser.add_argument("--processed", action="store_true",
                        help="also clear data/processed/, the preprocessing cache")
    parser.add_argument("--stages", default=None,
                        help="comma-separated subset of " + ",".join(STAGES) +
                             " — clear only what those stages produced, instead of the "
                             "whole output CreditPFN/ tree")
    args = parser.parse_args(argv)
    if args.lora_workflow:
        if (args.experiment not in (None, "experiment0") or args.workflow or args.processed
                or args.stages or args.all_experiments or args.scope != "all"):
            parser.error("--lora-workflow cannot be combined with another cleanup scope")
        return clean_lora_workflow(args.lora_workflow, clean=args.clean)
    if args.experiment and (args.processed or args.stages or args.all_experiments):
        parser.error("Experiment cleanup cannot include shared data/stages/all-experiments")
    if args.workflow and (args.experiment != "experiment0" or not args.workflow.isalnum()):
        parser.error("--workflow requires --experiment experiment0 and an alphanumeric ID")
    if args.scope != "all" and not args.experiment:
        parser.error("--scope requires --experiment")
    if args.clean and not args.experiment and not args.all_experiments:
        parser.error("Deletion requires --experiment or --all-experiments; preview is unchanged")
    if args.clean:
        check_cluster_writers(args.experiment)
        check_saved_consumers(args.experiment, args.scope)

    # Unlinking a running job's stdout on Linux loses the cleanup report itself.
    # Only the exact active .log inside our log directory may survive a reset.
    active_log = os.environ.get("CREDITPFN_ACTIVE_LOG")
    keep_paths = frozenset()
    if active_log:
        path = Path(active_log).resolve()
        from src.utils.paths import EXPERIMENTS
        allowed = {(outputs_dir() / g / "logs").resolve() for g in EXPERIMENTS}
        if path.parent not in allowed or path.suffix != ".log":
            parser.error("CREDITPFN_ACTIVE_LOG must name a .log inside output CreditPFN/<experiment>/logs")
        keep_paths = frozenset({path})
        print(f"Preserving active maintenance log: {path}")

    # Validate every tree before deleting ANY file, including stage-specific paths.
    # No writers may be active during a clean; this is not a concurrent deletion service.
    clean_processed = args.processed or bool(args.stages and "data" in args.stages.split(","))
    targets = experiment_roots(args.experiment, args.scope) if args.experiment else roots(processed=clean_processed)
    for root in targets:
        validate_tree(root)

    if args.workflow:
        # New workflow IDs appear in generated run names and workflow directories.
        # Match a complete path component or an underscore-delimited token only.
        import re
        pattern = re.compile(r"(^|_)" + re.escape(args.workflow) + r"(_|\.|$)")
        files = list(dict.fromkeys(p for root in targets for p in root.rglob("*")
                     if p.is_file() and p.name not in KEEP and p.resolve() not in keep_paths
                     and (args.workflow in p.parts or pattern.search(p.name))))
        print(f"Workflow {args.workflow}: {len(files)} files, {sum(p.stat().st_size for p in files)/1e6:.1f} MB")
        if args.clean:
            for path in files:
                path.unlink()
        else:
            for path in files:
                print(path)
            print("Nothing deleted. Add --clean after inspection.")
        return 0

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
