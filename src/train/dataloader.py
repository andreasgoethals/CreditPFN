"""Training dataloader — read sanitized CSVs, subsample per epoch.

Per-step recipe (one "batch" = one dataset, batch_size fixed at 1 by
TabPFN's ``meta_dataset_collator`` assertion at
``repositories/TabPFN .txt:17665-17666``):

  1. Pick one parent dataset (one ``DatasetRef``) — every parent
     contributes exactly one step per epoch. No more chunk splitting.

  2. Load (memoised) the entire sanitized CSV. Cast features to a
     pandas DataFrame; cast the target to ``int64`` (classification)
     or ``float32`` (regression).

  3. **Per-epoch reshuffle**: each epoch draws a fresh random
     subsample of ``cfg.finetuning.max_rows_per_epoch`` rows from the
     full dataset. Smaller datasets (rows ≤ the cap) are passed
     through in full — the cap is non-binding. The RNG seed mixes
     ``(base_seed, epoch, dataset_idx)`` so two epochs see two
     different subsamples of the same large dataset, while the
     subsample is still reproducible end-to-end if the same seed
     is rerun.

  4. Ordinal-encode categoricals **on the context split only**
     (matching the train-fold-only-fit pattern that the eval pipeline
     also uses — see ``src/eval/dataset_loader.encode_for_model``).

  5. Random ``(1 − query_fraction) / query_fraction`` split between
     context and query, drawn from the subsample.

  6. Cast to ``torch.Tensor`` of shape ``(n_samples, batch_size=1, F)``.

The DataLoader caller invokes :meth:`ProcessedDatasetLoader.set_epoch`
at the top of each epoch so that ``__getitem__`` picks up the new
epoch number. The CSV-loading is memoised behind a module-level cache
so re-visiting a dataset doesn't re-read the CSV from disk every
epoch — only re-subsamples it.

For *test-time evaluation*, see :func:`prepare_eval_chunk` — it
ignores the random subsample completely and uses the full dataset
(callers cap rows externally via ``n_inference_subsample_samples``).
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.train.corpus import DatasetRef

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tensor batch container
# --------------------------------------------------------------------------- #


@dataclass
class TabPFNBatch:
    """One forward-pass-ready batch (batch_size=1).

    Tensor shapes match the TabPFN ``PerFeatureTransformer`` signature:

    * ``X_context``  — (n_ctx,   1, n_features)   float32
    * ``y_context``  — (n_ctx,   1, 1)            float32 / int64
    * ``X_query``    — (n_query, 1, n_features)   float32
    * ``y_query``    — (n_query, 1, 1)            float32 / int64
    * ``categorical_idx`` — list[int]
    """
    X_context: torch.Tensor
    y_context: torch.Tensor
    X_query:   torch.Tensor
    y_query:   torch.Tensor
    categorical_idx: list[int]
    task_type: str
    dataset_id: str

    def to(self, device: str) -> "TabPFNBatch":
        return TabPFNBatch(
            X_context=self.X_context.to(device, non_blocking=True),
            y_context=self.y_context.to(device, non_blocking=True),
            X_query=self.X_query.to(device, non_blocking=True),
            y_query=self.y_query.to(device, non_blocking=True),
            categorical_idx=self.categorical_idx,
            task_type=self.task_type,
            dataset_id=self.dataset_id,
        )


# --------------------------------------------------------------------------- #
# CSV → (X_df, y, cat_cols) loader (memoised)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _LoadedDataset:
    X: pd.DataFrame
    y: np.ndarray
    cat_columns: tuple[str, ...]
    task_type: str
    dataset_id: str


@functools.lru_cache(maxsize=64)
def _load_processed_csv(ref: DatasetRef) -> _LoadedDataset:
    """Load one sanitized CSV (memoised by ``DatasetRef`` identity).

    Idempotent and thread-safe inside a single process; the LRU cache
    means each parent dataset is read from disk **once per training
    process** even though the dataloader re-visits it every epoch.
    """
    df = pd.read_csv(ref.processed_csv, low_memory=False)
    if ref.target_column not in df.columns:
        raise ValueError(
            f"target column {ref.target_column!r} missing from "
            f"{ref.processed_csv}"
        )
    feature_cols = [c for c in df.columns if c != ref.target_column]
    X_df = df[feature_cols].copy()
    if ref.task_type == "classification":
        y = pd.to_numeric(df[ref.target_column], errors="coerce").astype(np.int64).to_numpy()
    else:
        y = pd.to_numeric(df[ref.target_column], errors="coerce").astype(np.float32).to_numpy()
    cats = tuple(c for c in ref.categorical_columns if c in feature_cols)
    return _LoadedDataset(
        X=X_df, y=y, cat_columns=cats,
        task_type=ref.task_type, dataset_id=ref.dataset_id,
    )


# --------------------------------------------------------------------------- #
# Per-step subsample + encode + split
# --------------------------------------------------------------------------- #


_NAN_IMPUTE_VALUE: float = 0.0
"""Sentinel that replaces ±inf / NaN in the encoded feature matrix.

WHY THIS EXISTS.  TabPFN-v2.5's transformer asserts ``embedded_x``
has no NaN (``tabpfn/architectures/base/transformer.py:520``) — see
the PD-run-2026-05-20 trial-16 traceback. Our ordinal encoder
intentionally leaves NaN for missing categorical values (and the
sanitize pipeline leaves NaN for missing numerical values), so the
raw `model.forward()` call on v2.5 fails fast.

The fix is to impute ±inf / NaN to a single sentinel value AFTER
ordinal encoding but BEFORE the tensor cast. v3 happens to tolerate
NaN through its own column-distribution embedder; imputing to 0.0
costs us the explicit missing-value signal for v3 but does not
otherwise change behaviour. The previous in-context-learning prior
already routinely sees zero-valued features, so the imputed values
are not out-of-distribution.

Lossy alternative considered and rejected: passing data through
TabPFN's NanHandlingEncoderStep, which adds a binary
missing-indicator column. That would change feature dimensionality
mid-training and require deeper refactor. The simple 0.0
imputation is correct enough for continued pretraining; the eval
pipeline uses the same encoding via ``src.eval.dataset_loader``
which now mirrors this step.
"""


def _ordinal_encode(
    X_full: pd.DataFrame,
    *,
    ctx_idx: np.ndarray,
    cat_cols: Sequence[str],
    unknown_value: int = -1,
) -> tuple[np.ndarray, list[int]]:
    """Ordinal-encode categorical columns with a context-only fit, then
    replace any remaining ±inf / NaN with :data:`_NAN_IMPUTE_VALUE`.

    Mirrors :func:`src.eval.dataset_loader.encode_for_model`: the
    encoder is fit on the *context* rows so any category seen only
    in the query rows is encoded as ``unknown_value`` (-1), matching
    the inference scenario the model was trained for.
    """
    from sklearn.preprocessing import OrdinalEncoder

    cols = list(X_full.columns)
    cat_positions = [cols.index(c) for c in cat_cols if c in cols]
    if not cat_positions:
        arr = X_full.to_numpy(dtype=np.float32, na_value=_NAN_IMPUTE_VALUE)
        np.nan_to_num(
            arr, copy=False, nan=_NAN_IMPUTE_VALUE,
            posinf=_NAN_IMPUTE_VALUE, neginf=_NAN_IMPUTE_VALUE,
        )
        return arr, []

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=unknown_value,
        encoded_missing_value=np.nan,
    )
    cat_block = X_full.iloc[:, cat_positions].astype(object)
    cat_block = cat_block.where(cat_block.notna(), other=np.nan)
    encoder.fit(cat_block.iloc[ctx_idx])
    encoded = encoder.transform(cat_block)

    out = X_full.to_numpy(dtype=object).copy()
    for write_pos, src_pos in enumerate(cat_positions):
        out[:, src_pos] = encoded[:, write_pos]
    out = out.astype(np.float32)
    # Single in-place sweep — every NaN/±inf becomes _NAN_IMPUTE_VALUE.
    np.nan_to_num(
        out, copy=False, nan=_NAN_IMPUTE_VALUE,
        posinf=_NAN_IMPUTE_VALUE, neginf=_NAN_IMPUTE_VALUE,
    )
    return out, cat_positions


def _build_step_batch(
    loaded: _LoadedDataset,
    *,
    n_total_target: int,
    query_fraction: float,
    rng: np.random.Generator,
) -> TabPFNBatch:
    """Subsample → 80/20 split → ordinal-encode → tensorise."""
    n = len(loaded.X)
    n_total = min(n_total_target, n)
    if n_total <= 1:
        # Pathological tiny dataset — fall through with whatever we have.
        n_total = n

    if n_total < n:
        sel = rng.choice(n, size=n_total, replace=False)
    else:
        sel = rng.permutation(n)

    X_sub = loaded.X.iloc[sel].reset_index(drop=True)
    y_sub = loaded.y[sel]

    n_query = max(1, int(round(n_total * query_fraction)))
    n_query = min(n_query, n_total - 1)
    n_ctx = n_total - n_query
    ctx_idx = np.arange(n_ctx)

    X_full_arr, cat_idx = _ordinal_encode(
        X_sub, ctx_idx=ctx_idx, cat_cols=loaded.cat_columns,
    )
    X_ctx = X_full_arr[:n_ctx]
    X_qry = X_full_arr[n_ctx:]
    y_ctx = y_sub[:n_ctx]
    y_qry = y_sub[n_ctx:]

    X_ctx_t = torch.from_numpy(np.ascontiguousarray(X_ctx)).unsqueeze(1)
    X_qry_t = torch.from_numpy(np.ascontiguousarray(X_qry)).unsqueeze(1)
    y_dtype = torch.int64 if loaded.task_type == "classification" else torch.float32
    y_ctx_t = torch.as_tensor(y_ctx, dtype=y_dtype).reshape(-1, 1, 1).contiguous()
    y_qry_t = torch.as_tensor(y_qry, dtype=y_dtype).reshape(-1, 1, 1).contiguous()

    return TabPFNBatch(
        X_context=X_ctx_t,
        y_context=y_ctx_t,
        X_query=X_qry_t,
        y_query=y_qry_t,
        categorical_idx=cat_idx,
        task_type=loaded.task_type,
        dataset_id=loaded.dataset_id,
    )


# --------------------------------------------------------------------------- #
# Public: training Dataset
# --------------------------------------------------------------------------- #


class ProcessedDatasetLoader(Dataset):
    """One ``__getitem__`` call → one TabPFNBatch from one sanitized CSV.

    Designed to be wrapped in ``torch.utils.data.DataLoader`` with
    ``batch_size=1`` and ``collate_fn=identity_collate``.

    The training loop must call :meth:`set_epoch` before each new
    epoch so the per-epoch reshuffle (epoch-aware RNG seed) produces
    a fresh random subsample of large datasets every epoch. The
    subsample is still deterministic given ``(base_seed, epoch, idx)``,
    so a re-run with the same seed is bit-for-bit reproducible.
    """

    def __init__(
        self,
        refs: Sequence[DatasetRef],
        *,
        max_rows_per_epoch: int,
        query_fraction: float,
        seed: int = 0,
    ) -> None:
        if len(refs) == 0:
            raise ValueError("ProcessedDatasetLoader received an empty refs list")
        self.refs = list(refs)
        self.max_rows_per_epoch = int(max_rows_per_epoch)
        self.query_fraction = float(query_fraction)
        self._base_seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Bump the epoch counter so the next __getitem__ reshuffles."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> TabPFNBatch:
        ref = self.refs[idx]
        loaded = _load_processed_csv(ref)
        # Epoch-aware seed: same chunk on epoch 0 ≠ epoch 1 ≠ …
        rng = np.random.default_rng(
            (
                self._base_seed * 1_000_003
                + self._epoch * 10_007
                + idx * 31
            ) & 0xFFFF_FFFF
        )
        return _build_step_batch(
            loaded,
            n_total_target=self.max_rows_per_epoch,
            query_fraction=self.query_fraction,
            rng=rng,
        )


def identity_collate(batch):
    """Keep TabPFN's batch_size=1 invariant.

    Our ``__getitem__`` already returns one full :class:`TabPFNBatch`
    (= one dataset). The DataLoader wraps it in a length-1 list — we
    just unwrap it.
    """
    if len(batch) != 1:
        raise ValueError(
            f"identity_collate expects batch_size=1; got {len(batch)} "
            "(TabPFN's meta_dataset_collator hard-asserts this — see "
            "repositories/TabPFN .txt:17666)"
        )
    return batch[0]


# --------------------------------------------------------------------------- #
# Public: test/eval batch preparation (deterministic)
# --------------------------------------------------------------------------- #


def prepare_eval_chunk(
    ref: DatasetRef,
    *,
    n_inference_subsample_samples: int,
    seed: int,
) -> TabPFNBatch:
    """Build a deterministic eval batch for one dataset.

    Uses a fixed 80 / 20 context / query split for monitoring purposes
    (the proper evaluation pipeline uses K-fold CV — this function is
    only invoked by the training loop's end-of-epoch quick eval). When
    ``n_inference_subsample_samples`` is smaller than the dataset, both
    splits are subsampled proportionally.
    """
    loaded = _load_processed_csv(ref)
    rng = np.random.default_rng(seed)
    n = len(loaded.X)

    if 0 < n_inference_subsample_samples < n:
        keep = rng.choice(n, size=n_inference_subsample_samples, replace=False)
        X_sub = loaded.X.iloc[keep].reset_index(drop=True)
        y_sub = loaded.y[keep]
    else:
        X_sub = loaded.X.reset_index(drop=True)
        y_sub = loaded.y

    n_total = len(X_sub)
    n_query = max(1, int(round(n_total * 0.20)))
    n_query = min(n_query, n_total - 1)
    n_ctx = n_total - n_query
    ctx_idx = np.arange(n_ctx)

    X_full_arr, cat_idx = _ordinal_encode(
        X_sub, ctx_idx=ctx_idx, cat_cols=loaded.cat_columns,
    )
    X_ctx = X_full_arr[:n_ctx]
    X_qry = X_full_arr[n_ctx:]
    y_ctx = y_sub[:n_ctx]
    y_qry = y_sub[n_ctx:]

    X_ctx_t = torch.from_numpy(np.ascontiguousarray(X_ctx)).unsqueeze(1)
    X_qry_t = torch.from_numpy(np.ascontiguousarray(X_qry)).unsqueeze(1)
    y_dtype = torch.int64 if loaded.task_type == "classification" else torch.float32
    y_ctx_t = torch.as_tensor(y_ctx, dtype=y_dtype).reshape(-1, 1, 1).contiguous()
    y_qry_t = torch.as_tensor(y_qry, dtype=y_dtype).reshape(-1, 1, 1).contiguous()

    return TabPFNBatch(
        X_context=X_ctx_t,
        y_context=y_ctx_t,
        X_query=X_qry_t,
        y_query=y_qry_t,
        categorical_idx=cat_idx,
        task_type=loaded.task_type,
        dataset_id=loaded.dataset_id,
    )


# Backwards-compat alias.
prepare_validation_chunk = prepare_eval_chunk
