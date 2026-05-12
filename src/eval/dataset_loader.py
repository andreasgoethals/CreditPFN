"""Eval-time dataset loader: reads ``data/processed/{track}/<id>.sanitized.csv``
and produces ``(X, y, categorical_idx)`` for downstream model wrappers.

Why the eval pipeline doesn't reuse the cached chunks
-----------------------------------------------------
The cached ``.npz`` chunks under ``data/cached/`` are the TRAINING
input format: each chunk is at most ``cfg.dataset.max_rows_per_chunk``
(= 100,000 in the default config) rows, ordinal-encoded and split
into the 60/40 context/query that TabPFN's in-context learning
expects. That cap is correct for TabPFN training but wrong for
evaluation:

  * XGBoost / CatBoost have no row-count limit and would be
    underestimated if capped at the chunk size.
  * The 60/40 ctx/query split was computed at cache-write time with
    a single seed; eval needs K-fold cross-validation which is a
    different (and more rigorous) split.

So the eval loads the *processed* (sanitised, post-FeatureAgglomeration)
CSV directly. The processed CSVs are produced by ``src/data/sanitize.py``
and are the canonical "one row per observation, features + target,
categoricals as strings/objects" representation of every dataset.

Public surface
--------------
* :class:`ProcessedDataset`         — ``(X_df, y, categorical_columns)`` triple.
* :func:`load_processed_dataset`    — load one (track, dataset_id) pair.
* :func:`encode_for_model`          — apply per-method encoding rules
                                      (ordinal-encode cats with the
                                      train-fold-only fit, mirroring
                                      `src/data/dataset.py` exactly).
* :func:`subsample`                 — stratified row-count cap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.paths import resolve_data_path, resolve_output_path

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public dataclass
# --------------------------------------------------------------------------- #


@dataclass
class ProcessedDataset:
    """One sanitised dataset, eval-ready.

    * ``X``: pandas DataFrame, n_rows × n_features. Categoricals are
      still string/object columns (encoding happens later, per-method,
      with the train-fold-only fit).
    * ``y``: 1D numpy array — int64 for classification, float32 for
      regression.
    * ``categorical_columns``: column names known to be categorical
      (from the manifest's ``categorical_columns`` field).
    * ``task_type``: ``"classification"`` | ``"regression"``.
    * ``dataset_id`` / ``track``: pass-through for log messages.
    """
    X: pd.DataFrame
    y: np.ndarray
    categorical_columns: list[str]
    task_type: str
    dataset_id: str
    track: str

    @property
    def n_rows(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


# --------------------------------------------------------------------------- #
# Manifest helpers
# --------------------------------------------------------------------------- #


def _read_manifest_row(track: str, dataset_id: str, *,
                       manifest_template: str = "data/manifest_{track}.csv"
                       ) -> dict:
    """Look up one dataset_id's row in the per-track manifest."""
    p = resolve_output_path(manifest_template.format(track=track))
    if not p.exists():
        raise FileNotFoundError(
            f"Manifest not found at {p}. Run the data pipeline first."
        )
    df = pd.read_csv(p, dtype=str).fillna("")
    matches = df[df["dataset_id"] == dataset_id]
    if matches.empty:
        raise KeyError(
            f"dataset_id={dataset_id!r} not found in {p} "
            f"(have: {df['dataset_id'].head(5).tolist()}…)"
        )
    return matches.iloc[0].to_dict()


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #


def load_processed_dataset(
    track: str, dataset_id: str,
    *,
    processed_template: str = "data/processed/{track}/{dataset_id}.sanitized.csv",
) -> ProcessedDataset:
    """Read the sanitised CSV + manifest row → :class:`ProcessedDataset`."""
    if track not in ("pd", "lgd"):
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")

    csv_path = resolve_data_path(
        processed_template.format(track=track, dataset_id=dataset_id)
    )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed CSV not found: {csv_path}\n"
            "Run `python scripts/data_pipeline.py` first."
        )

    row = _read_manifest_row(track, dataset_id)
    target = row["target_column"]
    task_type = row["task_type"]
    cats_hint = (
        row["categorical_columns"].split(";")
        if row.get("categorical_columns") else []
    )

    df = pd.read_csv(csv_path, low_memory=False)
    if target not in df.columns:
        raise ValueError(
            f"target column {target!r} missing from {csv_path}"
        )

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].copy()
    y_raw = df[target]
    if task_type == "classification":
        y = pd.to_numeric(y_raw, errors="coerce").astype(np.int64).to_numpy()
    else:
        y = pd.to_numeric(y_raw, errors="coerce").astype(np.float32).to_numpy()

    cats_present = [c for c in cats_hint if c in feature_cols]

    LOGGER.info(
        "loaded processed: track=%s id=%s rows=%d feats=%d cats=%d task=%s",
        track, dataset_id, len(X), X.shape[1], len(cats_present), task_type,
    )
    return ProcessedDataset(
        X=X, y=y,
        categorical_columns=cats_present,
        task_type=task_type,
        dataset_id=dataset_id,
        track=track,
    )


# --------------------------------------------------------------------------- #
# Subsample (per-method row cap)
# --------------------------------------------------------------------------- #


def subsample(
    X: pd.DataFrame, y: np.ndarray, *,
    max_rows: int | None,
    seed: int,
    stratify: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """If ``max_rows`` is set and the dataset is larger, randomly sample
    down to ``max_rows`` (stratified for classification when feasible).

    Always returns a fresh DataFrame + numpy array; callers can treat
    them as independent of the original.
    """
    n = len(X)
    if max_rows is None or n <= max_rows:
        return X.reset_index(drop=True), np.asarray(y).copy()

    rng = np.random.default_rng(seed)
    if stratify and len(np.unique(y)) >= 2:
        # Stratified subsample: fraction per class.
        keep = np.zeros(n, dtype=bool)
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            n_keep = max(1, int(round(len(idx) * (max_rows / n))))
            chosen = rng.choice(idx, size=min(n_keep, len(idx)), replace=False)
            keep[chosen] = True
    else:
        chosen = rng.choice(n, size=max_rows, replace=False)
        keep = np.zeros(n, dtype=bool)
        keep[chosen] = True

    X_out = X.iloc[keep].reset_index(drop=True)
    y_out = np.asarray(y)[keep]
    return X_out, y_out


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def encode_for_model(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    *,
    categorical_columns: Iterable[str],
    unknown_value: int = -1,
    missing_value_sentinel: float = float("nan"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Ordinal-encode cats with a TRAIN-FOLD-ONLY fit and apply to all
    three splits.

    Mirrors ``src/data/dataset.py::_ordinal_encode_categoricals`` —
    fitting the encoder on the train fold only means categories that
    appear *only* in val/test are encoded as ``unknown_value`` (-1),
    matching the inference scenario TabPFN was trained for. NaN values
    map to NaN (sklearn's ``encoded_missing_value=np.nan``).

    Returns ``(X_train_arr, X_val_arr, X_test_arr, categorical_idx)``
    — every array is float32 with categorical columns in their
    original positions.
    """
    from sklearn.preprocessing import OrdinalEncoder

    cols = list(X_train.columns)
    cat_positions = [cols.index(c) for c in categorical_columns if c in cols]
    if not cat_positions:
        return (
            X_train.to_numpy(dtype=np.float32, na_value=np.nan),
            X_val.to_numpy(dtype=np.float32, na_value=np.nan),
            X_test.to_numpy(dtype=np.float32, na_value=np.nan),
            [],
        )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=unknown_value,
        encoded_missing_value=missing_value_sentinel,
    )
    cat_train = X_train.iloc[:, cat_positions].astype(object).where(
        X_train.iloc[:, cat_positions].notna(), other=np.nan,
    )
    cat_val = X_val.iloc[:, cat_positions].astype(object).where(
        X_val.iloc[:, cat_positions].notna(), other=np.nan,
    )
    cat_test = X_test.iloc[:, cat_positions].astype(object).where(
        X_test.iloc[:, cat_positions].notna(), other=np.nan,
    )
    encoder.fit(cat_train)
    enc_train = encoder.transform(cat_train)
    enc_val   = encoder.transform(cat_val)
    enc_test  = encoder.transform(cat_test)

    def _splice(X_df: pd.DataFrame, enc: np.ndarray) -> np.ndarray:
        out = X_df.to_numpy(dtype=object).copy()
        for write_pos, src_pos in enumerate(cat_positions):
            out[:, src_pos] = enc[:, write_pos]
        return out.astype(np.float32)

    return (
        _splice(X_train, enc_train),
        _splice(X_val,   enc_val),
        _splice(X_test,  enc_test),
        cat_positions,
    )
