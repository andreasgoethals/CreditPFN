"""Manifest construction (Stage 2 of the data pipeline).

For every dataset under ``data/raw/{pd,lgd}/<id>.csv``, applies the
surgical fixes from :mod:`src.data.preprocessing` and computes the
metadata row that downstream stages rely on:

  * ``dataset_id``           — same as the filename stem
  * ``track``                — "pd" or "lgd"
  * ``task_type``            — "classification" or "regression"
  * ``target_column``        — from preprocessing.DATASET_METADATA
  * ``source`` / ``source_url`` — from preprocessing.DATASET_METADATA
  * ``n_rows`` / ``n_cols``  — post-fix shape (excluding target)
  * ``n_categorical`` / ``n_numerical`` — feature-type counts
  * ``categorical_columns``  — semicolon-joined list (auto-inferred
    when DATASET_METADATA leaves it empty)
  * ``numerical_columns``    — semicolon-joined list
  * ``n_missing_total`` / ``missing_rate``
  * ``minority_class_ratio`` — n_minority / n_total for classification
    (binary or multi-class — minority = the *smallest* class).
    Empty string for regression rows.
  * ``target_mean`` / ``target_std`` — for regression only
  * ``date_added``           — UTC ISO date when the row was first
    inserted; preserved on idempotent re-runs
  * ``sha256_shape_cols``    — stable shape-aware hash used by
    :mod:`src.data.dedup`

Public entry point
------------------
``main(cfg) -> int``
    Idempotent. Re-running updates rows in-place keyed by
    ``dataset_id`` and never duplicates them.

    Reads
    -----
    * ``cfg.paths.raw/{pd,lgd}/*.csv`` — every raw file present.
    * ``DATASET_METADATA`` from :mod:`src.data.preprocessing`.
    * Existing ``cfg.paths.manifest_pd`` / ``manifest_lgd`` (to
      preserve ``date_added``).

    Writes
    ------
    * ``cfg.paths.manifest_pd`` (CSV, one row per PD dataset)
    * ``cfg.paths.manifest_lgd`` (CSV, one row per LGD dataset)

    Returns
    -------
    int
        ``0`` if every dataset registered cleanly; ``1`` if any
        dataset failed (logged with its ID; processing continues).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.preprocessing import DATASET_METADATA, apply_dataset_specific_fixes
from src.utils.paths import resolve_data_path, resolve_output_path

LOGGER = logging.getLogger(__name__)

MANIFEST_COLUMNS: list[str] = [
    "dataset_id", "track", "task_type", "target_column",
    "source", "source_url",
    "n_rows", "n_cols", "n_categorical", "n_numerical",
    "categorical_columns", "numerical_columns",
    "n_missing_total", "missing_rate",
    "minority_class_ratio", "target_mean", "target_std",
    "date_added", "sha256_shape_cols",
]


# =============================================================================
# Pure helpers
# =============================================================================


def shape_aware_sha256(n_rows: int, n_cols: int, columns: Iterable[str]) -> str:
    """Stable content-aware identifier for a dataset.

    Two datasets with the same row count, column count, and the same set
    of column *names* (order-independent) hash identically. Used by the
    dedup stage as a fast first-pass equivalence check.
    """
    payload = f"{n_rows}|{n_cols}|{'|'.join(sorted(columns))}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_string_like(dtype) -> bool:
    """True for any dtype that holds non-numeric "labels".

    Covers the three pandas dtype families in current use:

    * legacy ``object`` (pandas ≤ 2.x default for strings),
    * the new ``str`` / ``StringDtype`` (pandas 3.x default),
    * ``pd.CategoricalDtype`` (explicit categorical).

    Does NOT match ``bool`` / numeric / datetime / interval — those
    would be misclassified as categoricals here.
    """
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def infer_categorical_numerical(
    df: pd.DataFrame, target: str, hinted_categorical: list[str],
) -> tuple[list[str], list[str]]:
    """Resolve which feature columns are categorical vs. numerical.

    Two-rule cascade:

    1. **Hint wins.** Any column whose name is in ``hinted_categorical``
       AND that survived the surgical fix is categorical.
    2. **Dtype fallback.** A surviving column not in the hint list is
       categorical iff its dtype is string-like (object / str /
       category — see :func:`_is_string_like`). Everything else is
       numerical.
    """
    feature_cols = [c for c in df.columns if c != target]
    cats: list[str] = [c for c in hinted_categorical if c in feature_cols]
    auto_cat = [
        c for c in feature_cols
        if c not in cats and _is_string_like(df[c].dtype)
    ]
    cats.extend(auto_cat)
    nums = [c for c in feature_cols if c not in cats]
    return cats, nums


def compute_manifest_row(
    df: pd.DataFrame, dataset_id: str,
) -> dict:
    """Compute one manifest row (no I/O)."""
    meta = DATASET_METADATA[dataset_id]
    target = meta["target_column"]
    if target not in df.columns:
        raise ValueError(
            f"target column {target!r} missing from {dataset_id} after "
            f"surgical fixes; columns are {list(df.columns)[:10]}…"
        )

    cats, nums = infer_categorical_numerical(
        df, target, list(meta["categorical_columns"]),
    )

    feature_df = df.drop(columns=[target])
    n_rows, n_feat = feature_df.shape
    n_missing = int(feature_df.isna().sum().sum())
    cell_count = max(1, n_rows * n_feat)

    if meta["task_type"] == "classification":
        # n_minority / n_total — share of the rarest class in the data.
        # For binary problems with a 90 / 10 split this returns 0.10, so the
        # "smaller, more impressive" number reads naturally as a percentage.
        vc = df[target].dropna().value_counts()
        if len(vc) >= 1 and vc.sum() > 0:
            ratio = float(vc.min() / vc.sum())
        else:
            ratio = float("nan")
        target_mean = ""
        target_std = ""
        minority_class_ratio = f"{ratio:.6f}" if np.isfinite(ratio) else ""
    else:
        y = pd.to_numeric(df[target], errors="coerce")
        target_mean = f"{float(y.mean()):.6f}" if y.notna().any() else ""
        target_std = f"{float(y.std()):.6f}" if y.notna().any() else ""
        minority_class_ratio = ""

    return {
        "dataset_id": dataset_id,
        "track": meta["track"],
        "task_type": meta["task_type"],
        "target_column": target,
        "source": meta["source"],
        "source_url": meta["source_url"] or "",
        "n_rows": n_rows,
        "n_cols": n_feat,
        "n_categorical": len(cats),
        "n_numerical": len(nums),
        "categorical_columns": ";".join(cats),
        "numerical_columns": ";".join(nums),
        "n_missing_total": n_missing,
        "missing_rate": f"{n_missing / cell_count:.6f}",
        "minority_class_ratio": minority_class_ratio,
        "target_mean": target_mean,
        "target_std": target_std,
        "date_added": "",  # filled in main() to preserve existing rows
        "sha256_shape_cols": shape_aware_sha256(n_rows, n_feat, feature_df.columns),
    }


# =============================================================================
# Manifest I/O
# =============================================================================


def _read_existing_manifest(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)


def _write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df.to_csv(path, index=False)
    LOGGER.info("wrote %s (%d rows)", path, len(df))


# =============================================================================
# CLI
# =============================================================================


def _load_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load("config/data.yaml")


def main(cfg=None) -> int:  # noqa: C901
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if cfg is None:
        cfg = _load_cfg()
    raw_root = resolve_data_path(cfg.paths.raw)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()

    by_track: dict[str, list[dict]] = {"pd": [], "lgd": []}
    failures = 0

    # Preserve date_added from existing manifests on idempotent re-runs.
    existing: dict[str, dict[str, str]] = {}
    for track, manifest_path in [
        ("pd",  resolve_output_path(cfg.paths.manifest_pd)),
        ("lgd", resolve_output_path(cfg.paths.manifest_lgd)),
    ]:
        prev = _read_existing_manifest(manifest_path)
        for _, row in prev.iterrows():
            existing[row["dataset_id"]] = dict(row)

    for dataset_id, meta in DATASET_METADATA.items():
        track = meta["track"]
        path = raw_root / track / f"{dataset_id}.csv"
        if not path.exists():
            LOGGER.warning("missing raw file: %s — skipped", path)
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            df = apply_dataset_specific_fixes(df, dataset_id)
            row = compute_manifest_row(df, dataset_id)
            row["date_added"] = (
                existing.get(dataset_id, {}).get("date_added") or today
            )
            by_track[track].append(row)
            LOGGER.info(
                "registered %-26s n_rows=%d n_cols=%d cats=%d",
                dataset_id, row["n_rows"], row["n_cols"], row["n_categorical"],
            )
        except Exception as exc:
            LOGGER.error("%s failed: %s", dataset_id, exc, exc_info=True)
            failures += 1

    by_track["pd"].sort(key=lambda r: r["dataset_id"])
    by_track["lgd"].sort(key=lambda r: r["dataset_id"])

    _write_manifest(by_track["pd"],  resolve_output_path(cfg.paths.manifest_pd))
    _write_manifest(by_track["lgd"], resolve_output_path(cfg.paths.manifest_lgd))

    return 1 if failures else 0


def _parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Build per-track manifests from raw CSVs."
    ).parse_args()


if __name__ == "__main__":
    _parse_args()
    raise SystemExit(main())
