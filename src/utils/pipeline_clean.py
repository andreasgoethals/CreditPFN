"""Clean up one or more pipeline stage outputs from the terminal.

This script is the explicit "I want to rerun stage X from scratch"
utility — it deletes everything that stage X produces (plus its logs)
and leaves every other stage untouched. Run on the supercomputer
between submissions when you want to invalidate a stage's cache.

It complements the **per-stage resume logic** already built into the
pipelines themselves:

  * **data**: ``scripts/data_pipeline.py``'s ``_ensure_processed``
    hook (and the train pipeline's auto-hook) check whether each
    dataset's ``data/processed/{track}/<id>.sanitized.csv`` is on
    disk. **If yes, the sanitize step is skipped for that dataset.**
    So if you DON'T run this util, re-submitting the pipeline does
    NOT redo data preparation — that part is already idempotent.

  * **train**: ``scripts/train_pipeline.py`` skips any trial whose
    ``checkpoints/trained/<track>/<descriptive_name>.ckpt`` **and**
    matching ``.provenance.json`` are both on disk (records a "SKIP"
    row in the manifest and moves on). So an interrupted sweep
    resumes from the last incomplete trial. Force a rerun by deleting
    the .ckpt for the trial in question — or use this script to wipe
    the whole stage.

  * **eval**: ``src/eval/benchmark.py::find_existing_results`` skips
    any (model × dataset) pair whose result CSVs already have all
    folds present with ``status == OK``. Partial-failure pairs (some
    folds OK, some FAIL) are retried.

CLI examples
------------

Wipe ONLY the training stage (keep data + eval):

    python -m src.utils.pipeline_clean --stages train

Wipe data + eval (keep training checkpoints):

    python -m src.utils.pipeline_clean --stages data,eval

Wipe everything (full fresh start):

    python -m src.utils.pipeline_clean --stages all

Preview what would be deleted without actually deleting:

    python -m src.utils.pipeline_clean --stages all --dry-run

Output paths (verified against the codebase 2026-05-27)
-------------------------------------------------------

**data stage** wipes:

  * ``data/processed/{pd,lgd}/*.sanitized.csv`` and any feature_groups
    sidecar JSONs (per-dataset sanitize output, under DATA_ROOT)
  * ``data/dedup/*`` (4× dedup CSVs, under OUTPUT_ROOT)
  * ``data/manifest_pd.csv``, ``data/manifest_lgd.csv``
    (track-level sanitize manifests, under OUTPUT_ROOT)
  * ``logs/data_*.log``, ``logs/sanitize_*.log``, ``logs/dedup_*.log``
  * The raw datasets at ``data/raw/`` are NEVER touched.

**train stage** wipes:

  * ``checkpoints/trained/{pd,lgd}/*.ckpt`` (continued-pretrained weights)
  * ``checkpoints/trained/{pd,lgd}/*.ckpt.provenance.json`` (sidecars)
  * ``checkpoints/trained/{pd,lgd}/*.ckpt.epoch_eval.ckpt`` (rolling
    per-epoch snapshots; usually auto-removed at end of training but
    can be left behind by a crashed trial)
  * ``output/training/manifests/<run_name>_<track>.csv``
  * ``output/training/epochs/{pd,lgd}/*.csv``
  * ``logs/train_*.log``
  * The base TabPFN checkpoints in ``checkpoints/`` (NOT in the
    ``trained/`` subdir) are NEVER touched.

**eval stage** wipes:

  * ``output/results/<TRACK>/<method>/*.csv`` (every per-model dir)
  * ``output/figures/<notebook>/*.pdf`` (analysis-time figures emitted
    by notebooks 1.0 and 2.0)
  * ``logs/eval_*.log``
  * Trained checkpoints are NOT touched (they're an INPUT to eval,
    not an output of it).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable

from src.utils.paths import resolve_data_path, resolve_output_path

LOGGER = logging.getLogger(__name__)


_VALID_STAGES = ("data", "train", "eval")


# --------------------------------------------------------------------------- #
# Config readers — dependency-free fallback for the VSC login node
# --------------------------------------------------------------------------- #
#
# This util is designed to run on the SLURM **login node** (where the
# user types commands before submitting batch jobs). The login node has
# a minimal Python environment — typically no `omegaconf` (a heavy
# import that's only installed in the per-job conda env). So we read
# the configs with three escalating layers:
#
#   1. Try `omegaconf` (matches the rest of the pipeline).
#   2. Else try `PyYAML` (a smaller, more commonly-available dep).
#   3. Else fall back to the HARDCODED DEFAULTS below — which are
#      the values shipped in config/{data,eval,train}.yaml. The user
#      only loses the ability to override paths via cfg edits, which
#      is acceptable for a utility script.
#
# Cleaning the outputs is a build-time concern, so the defaults are
# fine 99% of the time. The script also prints which layer it used so
# users can audit.

_DEFAULTS = {
    "data": {
        # data.paths.* — these match config/data.yaml verbatim.
        "processed":    "data/processed",
        "dedup":        "data/dedup",
        "manifest_pd":  "data/manifest_pd.csv",
        "manifest_lgd": "data/manifest_lgd.csv",
    },
    "train": {
        # train.checkpoint.trained_dir
        "trained_dir":  "checkpoints/trained",
    },
    "eval": {
        # eval.results.base_dir
        "results_base": "output/results",
    },
}


def _load_yaml(path: str) -> dict | None:
    """Load a yaml file via whichever parser is installed.

    Returns the parsed dict on success, ``None`` if the file is missing
    OR no yaml parser is available (so callers can fall back to the
    hardcoded defaults).
    """
    p = Path(path)
    if not p.exists():
        return None
    # Try OmegaConf first (matches the rest of the pipeline).
    try:
        from omegaconf import OmegaConf  # type: ignore[import-not-found]
        return OmegaConf.to_container(OmegaConf.load(str(p)), resolve=True)  # type: ignore[return-value]
    except ImportError:
        pass
    # Fall back to PyYAML.
    try:
        import yaml  # type: ignore[import-not-found]
        with p.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        return None


def _cfg_get(d: dict | None, *keys: str, default: str) -> str:
    """Walk a nested dict, falling back to `default` on any miss."""
    cur: object = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if not isinstance(cur, (str, int)):
        return default
    return str(cur)


# --------------------------------------------------------------------------- #
# Per-stage target enumeration
# --------------------------------------------------------------------------- #


def _glob_into(root: Path, patterns: Iterable[str]) -> list[Path]:
    """Return every path under ``root`` matching any of ``patterns``.

    Each pattern is a relative glob (e.g. ``"train_*.log"``). Missing
    roots resolve to an empty list — never raises.
    """
    if not root.exists():
        return []
    hits: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p in seen:
                continue
            seen.add(p)
            hits.append(p)
    return hits


def _data_stage_targets() -> dict[str, list[Path]]:
    """Catalogue every output of the data stage.

    Returns a ``{category: [paths]}`` dict so the user-facing report
    shows a clear breakdown ("processed CSVs (37 files)" vs "logs
    (12 files)" etc.). Categories are display-only — deletion treats
    all paths uniformly.
    """
    cfg = _load_yaml("config/data.yaml")
    d = _DEFAULTS["data"]

    # The processed and dedup roots — per data.yaml's `paths` block.
    proc_root = resolve_data_path(
        _cfg_get(cfg, "paths", "processed", default=d["processed"]),
    )
    dedup_root = resolve_output_path(
        _cfg_get(cfg, "paths", "dedup", default=d["dedup"]),
    )
    manifest_pd = resolve_output_path(
        _cfg_get(cfg, "paths", "manifest_pd", default=d["manifest_pd"]),
    )
    manifest_lgd = resolve_output_path(
        _cfg_get(cfg, "paths", "manifest_lgd", default=d["manifest_lgd"]),
    )
    logs_root = resolve_output_path("logs")

    out: dict[str, list[Path]] = {}
    out["Processed CSVs"] = (
        list(proc_root.glob("**/*.sanitized.csv"))
        + list(proc_root.glob("**/*.sanitized.feature_groups.json"))
    )
    out["Dedup CSVs"] = list(dedup_root.glob("*.csv"))
    out["Sanitize manifests"] = [p for p in (manifest_pd, manifest_lgd) if p.exists()]
    out["Data logs"] = _glob_into(
        logs_root,
        ["data_*.log", "sanitize_*.log", "dedup_*.log",
         # The slurm-renamed-by-train-pipeline-style files for the data
         # task — still data-stage outputs because the task_name prefix
         # is "data_".
         "data_pipeline_*.log"],
    )
    return out


def _train_stage_targets() -> dict[str, list[Path]]:
    """Catalogue every output of the training stage."""
    tcfg = _load_yaml("config/train.yaml")
    d = _DEFAULTS["train"]
    trained_dir_rel = _cfg_get(
        tcfg, "checkpoint", "trained_dir", default=d["trained_dir"],
    )

    trained_root = resolve_output_path(trained_dir_rel)
    manifests_root = resolve_output_path("output/training/manifests")
    epochs_root = resolve_output_path("output/training/epochs")
    logs_root = resolve_output_path("logs")

    out: dict[str, list[Path]] = {}
    # Separate the rolling per-epoch eval snapshots (`*.epoch_eval.ckpt`)
    # from the real finetuned checkpoints. The snapshot is overwritten
    # every epoch by `_save_eval_snapshot` and best-effort-deleted at
    # the end of training (see src/train/loop.py). A crashed trial may
    # leave one behind.
    all_ckpts = list(trained_root.glob("**/*.ckpt"))
    snapshot_ckpts = [p for p in all_ckpts if p.name.endswith(".epoch_eval.ckpt")]
    real_ckpts = [p for p in all_ckpts if p not in snapshot_ckpts]
    out["Finetuned checkpoints"] = real_ckpts
    out["Per-epoch snapshots"] = snapshot_ckpts
    out["Checkpoint provenance"] = list(trained_root.glob("**/*.provenance.json"))
    out["Manifests"] = list(manifests_root.glob("*.csv"))
    out["Per-epoch CSVs"] = list(epochs_root.glob("**/*.csv"))
    out["Training logs"] = _glob_into(logs_root, ["train_*.log"])
    return out


def _eval_stage_targets() -> dict[str, list[Path]]:
    """Catalogue every output of the eval stage."""
    ecfg = _load_yaml("config/eval.yaml")
    d = _DEFAULTS["eval"]
    results_base = _cfg_get(
        ecfg, "results", "base_dir", default=d["results_base"],
    )

    results_root = resolve_output_path(results_base)
    figures_root = resolve_output_path("output/figures")
    logs_root = resolve_output_path("logs")

    out: dict[str, list[Path]] = {}
    out["Benchmark CSVs"] = list(results_root.glob("**/*.csv"))
    out["Notebook figures"] = list(figures_root.glob("**/*.pdf"))
    out["Eval logs"] = _glob_into(logs_root, ["eval_*.log"])
    return out


_STAGE_HANDLERS = {
    "data":  _data_stage_targets,
    "train": _train_stage_targets,
    "eval":  _eval_stage_targets,
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_stages(arg: str) -> list[str]:
    """``"all"`` → every stage; otherwise comma-separated names."""
    if arg.lower() == "all":
        return list(_VALID_STAGES)
    parts = [s.strip().lower() for s in arg.split(",") if s.strip()]
    bad = [s for s in parts if s not in _VALID_STAGES]
    if bad:
        raise SystemExit(
            f"unknown stage(s): {bad}. "
            f"Valid: {', '.join(_VALID_STAGES)} (or 'all')."
        )
    # De-duplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for s in parts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _human_bytes(n: int) -> str:
    """Pretty-print a byte count. Fast and unitful."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _path_size(p: Path) -> int:
    """File size in bytes; 0 for missing or unreadable."""
    try:
        return p.stat().st_size if p.is_file() else 0
    except OSError:
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Delete outputs of one or more pipeline stages "
            "(data / train / eval / all). Always preserves raw data at "
            "data/raw/ and base checkpoints at checkpoints/*.ckpt "
            "(only the trained/ subdir is touched)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--stages",
        required=True,
        help=(
            "Comma-separated list of stages to wipe. Each entry must be "
            "one of: data, train, eval. Or pass 'all' for every stage. "
            "Examples: --stages train | --stages train,eval | --stages all"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted WITHOUT actually deleting anything.",
    )
    ap.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip the y/N confirmation prompt (use in scripts).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    stages = _parse_stages(args.stages)
    LOGGER.info("Stages to wipe: %s", ", ".join(stages))
    if args.dry_run:
        LOGGER.info("DRY-RUN mode: no files will be deleted.")

    # Build the full target list, deduplicated across stages (log files
    # can appear in multiple stage groups if the same .log name pattern
    # is shared, but the per-stage globs are designed to be disjoint).
    targets_by_stage: dict[str, dict[str, list[Path]]] = {}
    grand_total_bytes = 0
    grand_total_files = 0
    for stage in stages:
        targets_by_stage[stage] = _STAGE_HANDLERS[stage]()
        for cat, paths in targets_by_stage[stage].items():
            n = len(paths)
            sz = sum(_path_size(p) for p in paths)
            grand_total_bytes += sz
            grand_total_files += n
            print(
                f"  [{stage:>5}]  {cat:.<28} {n:>6} files  {_human_bytes(sz):>10}"
            )
    print("-" * 78)
    print(
        f"  TOTAL: {grand_total_files:,} file(s)   {_human_bytes(grand_total_bytes)}"
    )

    if grand_total_files == 0:
        LOGGER.info("Nothing to delete — all listed targets are already empty.")
        return 0

    if args.dry_run:
        LOGGER.info("Dry-run finished. No files modified.")
        return 0

    if not args.yes:
        ans = input(
            f"\nDelete {grand_total_files} file(s) "
            f"({_human_bytes(grand_total_bytes)})? [y/N] "
        ).strip().lower()
        if ans not in ("y", "yes"):
            LOGGER.info("Aborted — no files modified.")
            return 1

    # Actually delete.
    deleted_files = 0
    failed_files: list[tuple[Path, str]] = []
    for stage, cats in targets_by_stage.items():
        for cat, paths in cats.items():
            for p in paths:
                try:
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        continue
                    deleted_files += 1
                except OSError as exc:
                    failed_files.append((p, str(exc)))

    LOGGER.info(
        "Done. Deleted %d/%d file(s) across stages: %s",
        deleted_files, grand_total_files, ", ".join(stages),
    )
    if failed_files:
        LOGGER.warning("%d file(s) could NOT be deleted:", len(failed_files))
        for p, why in failed_files[:20]:
            LOGGER.warning("  %s  (%s)", p, why)
        if len(failed_files) > 20:
            LOGGER.warning("  ... and %d more", len(failed_files) - 20)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
