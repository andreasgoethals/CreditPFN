"""End-to-end orchestrator for the five data-pipeline stages.

Calls, in order:

    1. dedup    --pass pre   on data/raw/{pd,lgd}/
    2. register              → data/manifest_{pd,lgd}.csv
    3. sanitize              → data/processed/{pd,lgd}/<id>.sanitized.csv
    4. dedup    --pass post  on data/processed/{pd,lgd}/
    5. dataset               → data/cached/{pd,lgd}/<id>/chunk_*.npz

The five stage modules are each callable on their own (``python -m
src.data.<name>``); this script is the convenience wrapper that
chains them and writes a single summary line per run to ``logs/``.

Public entry point
------------------
:func:`run` — the orchestration function. Parameters:

``fresh: bool`` (default ``False``)
    ``True`` → wipe ``data/dedup``, ``data/processed``, ``data/cached``
    and the two manifest CSVs *before* anything runs. Use when you
    want the corpus rebuilt from scratch.
    ``False`` → leave existing artefacts in place. Register,
    sanitize, and dedup refresh their outputs; dataset.py skips only
    cache entries whose fingerprint still matches the manifest row,
    processed CSV content, dataset config, and cache schema.
``datasets: list[str] | None``
    ``None`` or empty list → process every dataset_id registered in
    :data:`src.data.preprocessing.DATASET_METADATA`. Otherwise: only
    the supplied dataset_ids (e.g. ``["0001.gmsc", "0001.heloc"]``).
``log_path: Path | str | None``
    ``None`` (default, CLI usage) → a fresh ``logs/<timestamp>.log``
    is created and the run summary is appended to it.
    Supplied path → no new file is created; the summary line is
    appended to the path you provided. Use this when calling ``run``
    from another orchestrator that already owns its own log file.
``cfg``
    Optional pre-loaded OmegaConf config. Defaults to loading
    ``config/data.yaml``.

CLI usage::

    python scripts/data_pipeline.py                    # incremental, all datasets
    python scripts/data_pipeline.py --fresh            # wipe + rebuild
    python scripts/data_pipeline.py --datasets 0001.gmsc 0013.hmeq

Returns
-------
``int`` exit code (``0`` on full success, ``1`` if any stage
returned non-zero).
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

# Ensure the repo root is importable when this file is executed as a
# plain script (``python scripts/data_pipeline.py``) rather than as a
# module (``python -m scripts.data_pipeline``).
import sys as _sys
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from src.data import dedup, register, sanitize, dataset  # noqa: E402
from src.data.preprocessing import DATASET_METADATA  # noqa: E402
from src.utils.paths import resolve_data_path  # noqa: E402
from src.utils.run_log import resolve_run_log  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load("config/data.yaml")


def _wipe(cfg) -> None:
    """Delete intermediate / output artefacts, but never the raw data.

    Empties the directories rather than removing them — Windows often
    refuses to remove a directory that an editor has visited recently.
    """
    dirs = [
        resolve_data_path(cfg.paths.dedup),
        resolve_data_path(cfg.paths.processed),
        resolve_data_path(cfg.paths.cached),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        for child in d.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except PermissionError:
                    pass

    for f in (resolve_data_path(cfg.paths.manifest_pd), resolve_data_path(cfg.paths.manifest_lgd)):
        if f.exists():
            try:
                f.unlink()
            except PermissionError:
                pass


def _filter_dataset_ids(datasets: list[str] | None) -> set[str] | None:
    """Validate the user-supplied dataset list. Returns ``None`` to mean
    "process everything"; otherwise returns a set for fast lookup.
    """
    if not datasets:
        return None
    unknown = [d for d in datasets if d not in DATASET_METADATA]
    if unknown:
        raise ValueError(
            f"Unknown dataset_id(s): {unknown}. "
            f"Known IDs: {sorted(DATASET_METADATA.keys())}"
        )
    return set(datasets)


def _count_files(folder: Path, pattern: str = "*") -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for _ in folder.glob(pattern))


def _count_chunks(cache_root: Path) -> int:
    if not cache_root.is_dir():
        return 0
    return sum(1 for _ in cache_root.rglob("chunk_*.npz"))


def _count_doubles(dedup_dir: Path, track: str, pass_name: str) -> int:
    p = dedup_dir / f"doubles_{track}_{pass_name}.csv"
    if not p.exists():
        return 0
    # one header line, one row per duplicate.
    return max(0, sum(1 for _ in p.open(encoding="utf-8")) - 1)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def run(
    fresh: bool = False,
    datasets: list[str] | None = None,
    log_path: Path | str | None = None,
    cfg=None,
) -> int:
    """Run the full five-stage data pipeline. See module docstring."""
    if cfg is None:
        cfg = _load_cfg()
    log, _ = resolve_run_log(log_path)

    selected = _filter_dataset_ids(datasets)
    if selected is None:
        n_pd = sum(1 for v in DATASET_METADATA.values() if v["track"] == "pd")
        n_lgd = sum(1 for v in DATASET_METADATA.values() if v["track"] == "lgd")
    else:
        n_pd = sum(1 for d in selected if DATASET_METADATA[d]["track"] == "pd")
        n_lgd = sum(1 for d in selected if DATASET_METADATA[d]["track"] == "lgd")

    if fresh:
        _wipe(cfg)

    # NOTE: per-dataset filtering is applied by overriding DATASET_METADATA
    # at the module level for the duration of the run. This keeps the five
    # stage modules unchanged. We restore the original mapping in `finally`.
    if selected is not None:
        full_metadata = dict(DATASET_METADATA)  # snapshot
        # Patch the underlying dict (DATASET_METADATA is a MappingProxy).
        from src.data import preprocessing as _pp
        _pp._RAW_METADATA = {k: dict(v) for k, v in full_metadata.items()
                             if k in selected}
        # Re-freeze.
        from types import MappingProxyType
        _pp.DATASET_METADATA = MappingProxyType(
            {k: MappingProxyType(v) for k, v in _pp._RAW_METADATA.items()}
        )
        # The stage modules already imported `DATASET_METADATA` by name; we
        # need to refresh those bindings too.
        for mod in (dedup, register, sanitize, dataset):
            mod.DATASET_METADATA = _pp.DATASET_METADATA

    t0 = time.monotonic()
    failures: list[str] = []
    try:
        if dedup.main(cfg, pass_name="pre"):
            failures.append("dedup_pre")
        if register.main(cfg):
            failures.append("register")
        if sanitize.main(cfg):
            failures.append("sanitize")
        if dedup.main(cfg, pass_name="post"):
            failures.append("dedup_post")
        if dataset.main(cfg):
            failures.append("dataset")
    finally:
        if selected is not None:
            # Restore.
            from types import MappingProxyType
            from src.data import preprocessing as _pp
            _pp._RAW_METADATA = {k: dict(v) for k, v in full_metadata.items()}
            _pp.DATASET_METADATA = MappingProxyType(
                {k: MappingProxyType(v) for k, v in _pp._RAW_METADATA.items()}
            )
            for mod in (dedup, register, sanitize, dataset):
                mod.DATASET_METADATA = _pp.DATASET_METADATA

    elapsed = time.monotonic() - t0
    dedup_dir = resolve_data_path(cfg.paths.dedup)
    cache_root = resolve_data_path(cfg.paths.cached)

    summary = (
        f"data_pipeline: "
        f"status={'OK' if not failures else 'FAIL[' + ','.join(failures) + ']'}  "
        f"fresh={fresh}  "
        f"selected={'all' if selected is None else len(selected)} "
        f"(pd={n_pd}, lgd={n_lgd})  "
        f"doubles_pre=[pd:{_count_doubles(dedup_dir, 'pd', 'pre')}, "
        f"lgd:{_count_doubles(dedup_dir, 'lgd', 'pre')}]  "
        f"doubles_post=[pd:{_count_doubles(dedup_dir, 'pd', 'post')}, "
        f"lgd:{_count_doubles(dedup_dir, 'lgd', 'post')}]  "
        f"chunks={_count_chunks(cache_root)}  "
        f"elapsed={elapsed:.1f}s"
    )
    log.write(summary)
    print(summary)

    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full data pipeline (dedup/register/sanitize/"
                    "dedup/dataset) end-to-end.",
    )
    p.add_argument(
        "--fresh", action="store_true",
        help="Delete existing dedup/, processed/, cached/, and manifests "
             "before running. Default: incremental (skip valid cache).",
    )
    p.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset of dataset IDs to process (e.g. 0001.gmsc 0013.hmeq). "
             "Default: every ID registered in DATASET_METADATA.",
    )
    p.add_argument(
        "--log-path", default=None,
        help="Append the run summary to this log file instead of creating "
             "a fresh logs/<timestamp>.log file.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(run(
        fresh=args.fresh,
        datasets=args.datasets,
        log_path=args.log_path,
    ))
