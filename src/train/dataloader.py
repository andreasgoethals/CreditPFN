"""Training dataloader — read sanitized CSVs, subsample per epoch.

Per-step recipe (one "batch" = one dataset, batch_size fixed at 1 by
TabPFN's ``meta_dataset_collator`` assertion at
``tfm-library/repositories/TabPFN .txt``):

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
import math
from dataclasses import dataclass
from typing import Any, Sequence

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
    # Positive-class rate of the ORIGINAL sampled labels (see
    # TabICLTrainBatch.ctx_pos_rate for why it must be measured pre-shuffle).
    ctx_pos_rate: float = float("nan")

    def to(self, device: str) -> "TabPFNBatch":
        return TabPFNBatch(
            X_context=self.X_context.to(device, non_blocking=True),
            y_context=self.y_context.to(device, non_blocking=True),
            X_query=self.X_query.to(device, non_blocking=True),
            y_query=self.y_query.to(device, non_blocking=True),
            categorical_idx=self.categorical_idx,
            task_type=self.task_type,
            dataset_id=self.dataset_id,
            ctx_pos_rate=self.ctx_pos_rate,
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


def _stratified_subsample_indices(
    y: np.ndarray, n_total: int, rng: np.random.Generator,
    mode: str = "stratified",
) -> np.ndarray:
    """Per-class subsample of size ``n_total`` — the CONTEXT-CONSTRUCTION knob.

    ``mode``:

    * ``"stratified"`` (default) — keep each class's frequency proportional
      to its frequency in ``y``, like sklearn's ``StratifiedKFold`` but as a
      single one-shot draw.
    * ``"balanced"`` — give every class an EQUAL quota, capped by how many
      rows that class actually has, and spend any leftover budget on the
      classes that still have rows.

    Why ``balanced`` exists (Tanna et al. 2026, *Data Presentation Over
    Architecture*, in tfm-library): because a tabular foundation model
    predicts by in-context learning, how the context window is built matters
    enormously under class imbalance. They cross seven context strategies
    with 1K–50K context sizes on Home Credit and Lending Club — both in OUR
    training corpus — and find balanced/hybrid sampling worth **3–4 AUC
    points over uniform, a gap larger than the spread between TFM
    families**. Concretely here: proportional sampling of 26 000 rows from a
    1 %-default-rate dataset yields ~260 positives, while balanced yields
    every positive available (often thousands). The minority class is what
    the model has to learn to retrieve.

    Falls back to uniform random sampling for regression targets, for
    single-class targets, and whenever a per-class quota is impossible
    (``n_total`` < number of classes), so it stays a strict superset of the
    original behaviour.
    """
    if mode not in ("stratified", "balanced"):
        raise ValueError(
            f"context sampling mode must be 'stratified' or 'balanced'; "
            f"got {mode!r}"
        )
    n = len(y)
    if n_total >= n:
        return rng.permutation(n)

    # Detect classification-style y: small integer alphabet.
    is_classy = (
        np.issubdtype(y.dtype, np.integer)
        and len(np.unique(y)) >= 2
        and len(np.unique(y)) <= max(50, n // 2)
    )
    if not is_classy:
        return rng.choice(n, size=n_total, replace=False)

    classes, counts = np.unique(y, return_counts=True)
    # When the requested subsample is smaller than the number of classes,
    # the max(1,…) floor below makes a balanced per-class quota impossible
    # (every class wants >=1 row but n_total < n_classes). Fall back to a
    # plain random draw rather than spinning in the drift loop forever.
    # (Bug fixed 2026-06-23.)
    if n_total < len(classes):
        return rng.choice(n, size=n_total, replace=False)

    if mode == "balanced":
        # Equal quota per class, capped by availability; the budget freed by
        # a class that cannot fill its share is redistributed (repeatedly) to
        # the classes that still have rows. With a 1 % positive rate this
        # takes EVERY positive plus enough negatives to reach n_total, rather
        # than the ~1 % of n_total a proportional draw would take.
        quotas = np.zeros(len(classes), dtype=int)
        remaining = int(n_total)
        open_mask = np.ones(len(classes), dtype=bool)
        while remaining > 0 and open_mask.any():
            share = max(1, remaining // int(open_mask.sum()))
            progressed = False
            for i in np.where(open_mask)[0]:
                if remaining <= 0:
                    break
                room = int(counts[i]) - int(quotas[i])
                if room <= 0:
                    open_mask[i] = False
                    continue
                take = int(min(share, room, remaining))
                quotas[i] += take
                remaining -= take
                progressed = True
                if quotas[i] >= counts[i]:
                    open_mask[i] = False
            if not progressed:
                break
    else:
        # Target per-class quota proportional to class frequency, rounded.
        quotas = np.maximum(1, np.round(counts * n_total / n).astype(int))
    # Resolve rounding drift: total quotas must equal n_total. Adjust on
    # the largest class(es) so a tiny rounding off-by-one doesn't bias.
    # Balanced mode allocates exactly against real availability above, so it
    # has no rounding drift to fix — and running this block would hand a
    # class a quota larger than the rows it owns.
    drift = 0 if mode == "balanced" else int(quotas.sum()) - int(n_total)
    if drift != 0:
        order = np.argsort(-counts)               # largest classes first
        # Bound the down-adjust loop: if a full pass over all classes can't
        # shed any more (every quota already at its floor of 1), stop — the
        # remaining drift is absorbed by returning slightly fewer rows,
        # which is harmless. Prevents an infinite spin.
        idx = 0
        stalls = 0
        while drift > 0 and stalls < len(order):
            if quotas[order[idx]] > 1:
                quotas[order[idx]] -= 1
                drift -= 1
                stalls = 0
            else:
                stalls += 1
            idx = (idx + 1) % len(order)
        while drift < 0:
            quotas[order[idx]] += 1
            drift += 1
            idx = (idx + 1) % len(order)

    picks: list[np.ndarray] = []
    for cls, quota in zip(classes, quotas):
        cls_idx = np.where(y == cls)[0]
        # min() guards against quota > class frequency (extreme class imbalance).
        quota = int(min(quota, len(cls_idx)))
        if quota == 0:
            continue
        picks.append(rng.choice(cls_idx, size=quota, replace=False))
    selection = np.concatenate(picks)
    # Final shuffle so the context/query slice afterwards isn't class-ordered.
    rng.shuffle(selection)
    return selection


def _build_step_batch(
    loaded: _LoadedDataset,
    *,
    n_total_target: int,
    query_fraction: float,
    rng: np.random.Generator,
    context_sampling: str = "stratified",
) -> TabPFNBatch:
    """Subsample → context/query split → ordinal-encode → tensorise.

    For classification targets the subsample is stratified — each
    class's rows are picked proportionally so the per-step gradient
    sees roughly the same class balance as the full dataset. For
    regression targets and degenerate classification edge cases
    (single class, ultra-imbalanced) the function falls back to a
    plain uniform random sample. See :func:`_stratified_subsample_indices`.
    """
    n = len(loaded.X)
    n_total = min(n_total_target, n)
    if n_total <= 1:
        # Pathological tiny dataset — fall through with whatever we have.
        n_total = n

    sel = _stratified_subsample_indices(loaded.y, n_total, rng, mode=context_sampling)

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

    # Canonical context labels — this path applies no class permutation, but
    # measure it here anyway so every batch type reports the same quantity.
    ctx_pos_rate = (
        float((np.asarray(y_ctx) > 0).mean())
        if loaded.task_type == "classification" else float("nan")
    )
    return TabPFNBatch(
        X_context=X_ctx_t,
        y_context=y_ctx_t,
        X_query=X_qry_t,
        y_query=y_qry_t,
        categorical_idx=cat_idx,
        task_type=loaded.task_type,
        dataset_id=loaded.dataset_id,
        ctx_pos_rate=ctx_pos_rate,
    )


# --------------------------------------------------------------------------- #
# TabICLv2 step builder — uses tabicl's official finetuning preprocessing
# --------------------------------------------------------------------------- #


@dataclass
class TabICLTrainBatch:
    """One training step for the TabICLv2 family (2026-08-04).

    Thin wrapper around tabicl's own ``MetaBatch`` fields (built by their
    ``_finetune.data._build_meta_batch`` — context/query split, per-member
    ``EnsembleGenerator`` preprocessing, class-shuffle remap, regression
    z-norm by context stats), plus the metadata our loop needs. The ensemble
    dimension E is the leading batch dim; the model forward is
    ``model(X, y_train)`` with rows ``[:train_size]`` = context.
    """
    X: torch.Tensor              # (E, T, H) float32, context+query concatenated
    y_train: torch.Tensor        # (E, train_size) float32
    y_query: torch.Tensor        # (E, test_size) long (clf, shuffled space) / float32 (reg, z-normed)
    train_size: int
    y_scaler_mean: float | None  # regression-only context z-norm stats
    y_scaler_std: float | None
    task_type: str
    dataset_id: str
    # Positive-class rate of the ORIGINAL sampled labels, before tabicl's
    # per-member class-shuffle remap. Measured here because after the remap
    # `y_train > 0` counts a permuted label and is meaningless: on 25 %-positive
    # data a 2-member batch reported 74.7 %. This is what verifies that
    # `context_sampling: balanced` actually raised minority exposure.
    ctx_pos_rate: float = float("nan")

    def to(self, device: str) -> "TabICLTrainBatch":
        return TabICLTrainBatch(
            X=self.X.to(device, non_blocking=True),
            y_train=self.y_train.to(device, non_blocking=True),
            y_query=self.y_query.to(device, non_blocking=True),
            train_size=self.train_size,
            y_scaler_mean=self.y_scaler_mean,
            y_scaler_std=self.y_scaler_std,
            task_type=self.task_type,
            dataset_id=self.dataset_id,
            # Must be carried through .to(): the loop reads it AFTER the device
            # move, and a dropped field silently reverts to its NaN default —
            # which is exactly how the metric went missing the first time.
            ctx_pos_rate=self.ctx_pos_rate,
        )


def _build_tabicl_step_batch(
    loaded: _LoadedDataset,
    *,
    n_total_target: int,
    query_fraction: float,
    rng: np.random.Generator,
    n_estimators: int,
    epoch_seed: int,
    replica: int,
    preprocessing_seed: int,
    context_sampling: str = "stratified",
) -> TabICLTrainBatch:
    """Subsample → encode/impute → tabicl's official ``_build_meta_batch``.

    Faithfulness notes (vs. upstream ``FinetunedTabICL*``):

    * The stratified subsample reuses the SAME sampler as the TabPFN path
      (`_stratified_subsample_indices`) so the "context construction" axis
      (Tanna 2026 shows it can move AUC more than model choice) is held
      constant across model families.
    * Ordinal encoding + median imputation are fit on the WHOLE subsample.
      That matches the upstream user contract — their finetune wrappers
      expect X already numeric and imputed for the entire training set
      (their ``TransformToNumerical`` is not in the finetune path) — and the
      ctx/qry split happens INSIDE ``_build_meta_batch`` afterwards. NaNs
      must not reach the transformer (only their scalers are NaN-aware).
    * ``preprocessing_seed`` is FIXED across epochs per dataset while
      ``epoch_seed`` varies — their design: fresh chunk/split every epoch,
      stable coarse preprocessing choices.
    """
    from src.train.tabicl_compat import import_tabicl_finetune_data
    _, build_meta_batch = import_tabicl_finetune_data()

    n = len(loaded.X)
    n_total = min(n_total_target, n)
    if n_total <= 1:                                    # pathological tiny set
        n_total = n
    sel = _stratified_subsample_indices(loaded.y, n_total, rng, mode=context_sampling)

    X_sub = loaded.X.iloc[sel].reset_index(drop=True)
    y_sub = np.asarray(loaded.y[sel])

    # Encode over the full subsample (see faithfulness note above).
    all_idx = np.arange(len(X_sub))
    X_arr, _cat_idx = _ordinal_encode(
        X_sub, ctx_idx=all_idx, cat_cols=loaded.cat_columns,
    )
    X_arr = np.asarray(X_arr, dtype=np.float64)
    # ±inf → NaN first, so the median impute below absorbs it. TabPFN clips
    # inf inside its feature normaliser; TabICLv2 does NOT (its sklearn path
    # raises on inf, and the raw training path would carry it into the
    # scalers). Credit features hit this via zero-denominator ratios.
    if np.isinf(X_arr).any():
        X_arr[np.isinf(X_arr)] = np.nan
    # Median-impute NaNs; all-NaN columns fall back to 0.
    if np.isnan(X_arr).any():
        col_median = np.nanmedian(X_arr, axis=0)
        col_median = np.where(np.isfinite(col_median), col_median, 0.0)
        nan_r, nan_c = np.nonzero(np.isnan(X_arr))
        X_arr[nan_r, nan_c] = col_median[nan_c]

    classification = loaded.task_type == "classification"
    query_size = max(1, int(round(len(X_arr) * query_fraction)))
    query_size = min(query_size, len(X_arr) - 1)

    mb = build_meta_batch(
        X_arr,
        y_sub.astype(np.int64) if classification else y_sub.astype(np.float64),
        classification=classification,
        n_estimators=n_estimators,
        query_size=query_size,
        epoch_seed=int(epoch_seed) & 0x7FFF_FFFF,
        chunk_idx=int(replica),
        norm_methods=None,                    # upstream default ["none","power"]
        feat_shuffle_method="latin",
        class_shuffle_method="shift",
        outlier_threshold=4.0,
        preprocessing_seed=int(preprocessing_seed) & 0x7FFF_FFFF,
    )
    ctx_pos_rate = (
        float((y_sub > 0).mean()) if classification else float("nan")
    )
    return TabICLTrainBatch(
        X=mb.X,
        y_train=mb.y_train,
        y_query=mb.y_query,
        train_size=int(mb.train_size),
        y_scaler_mean=mb.y_scaler_mean,
        y_scaler_std=mb.y_scaler_std,
        task_type=loaded.task_type,
        dataset_id=loaded.dataset_id,
        ctx_pos_rate=ctx_pos_rate,
    )


# --------------------------------------------------------------------------- #
# Ensemble step builder — uses TabPFN's official preprocessing pipeline
# --------------------------------------------------------------------------- #


def _build_ensemble_step_batch(
    loaded: _LoadedDataset,
    *,
    n_total_target: int,
    query_fraction: float,
    rng: np.random.Generator,
    inference_config: Any,
    n_estimators: int,
    rng_seed: int,
    context_sampling: str = "stratified",
):
    """N-estimator step batch with TabPFN's official preprocessing.

    Two-phase pipeline (mirrors the official multi-dataset finetune at
    ``tfm-library/repositories/TabPFN .txt`` step-for-step):

      1. **Once per dataset** — ``clean_loaded_dataset`` runs TabPFN's
         ``clean_data`` to produce a numeric numpy array with
         categoricals ordinal-encoded (NaNs preserved). Result is
         cached in ``tabpfn_preprocessing._CLEAN_CACHE`` keyed by
         ``(dataset_id, shape, n_estimators)`` so each parent dataset
         is cleaned exactly once per training process (step-level seeds
         are excluded from the key to prevent O(steps) memory growth).
      2. **Per step** — subsample rows, split context/query, hand the
         already-cleaned numeric slices to ``build_ensemble_members``
         which calls ``TabPFNEnsemblePreprocessor.fit_transform_ensemble_members``.

    The earlier 2026-05-27 PD/LGD crash (``np.isnan`` on object dtype
    inside `encode_categorical_features_step`) was caused by skipping
    phase 1 — we passed a raw object-dtype slice directly to
    ``fit_transform_ensemble_members``. Fixed by routing through
    ``clean_loaded_dataset`` first.
    """
    from src.train.tabpfn_preprocessing import (
        build_ensemble_members, clean_loaded_dataset,
    )

    # Determine n_classes BEFORE the clean cache lookup — the cache key
    # doesn't include n_classes but the EnsembleConfig generation does
    # need it, so we resolve it here.
    if loaded.task_type == "classification":
        n_classes = int(len(np.unique(loaded.y)))
        if n_classes < 2:
            n_classes = 2                     # binary minimum
    else:
        n_classes = None

    # ---- Phase 1: clean_data + EnsembleConfig (cached per dataset) ----- #
    cleaned = clean_loaded_dataset(
        X_full_df=loaded.X,
        y_full=loaded.y,
        cat_columns=loaded.cat_columns,
        task_type=loaded.task_type,
        dataset_id=loaded.dataset_id,
        n_estimators=n_estimators,
        n_classes=n_classes,
        inference_config=inference_config,
        base_seed=int(rng_seed),
    )
    X_all = cleaned.X_clean          # (n_total_dataset, n_features) numeric
    y_all = cleaned.y

    # ---- Phase 2a: subsample + ctx/query split (per step) -------------- #
    n = X_all.shape[0]
    n_total = min(n_total_target, n)
    if n_total <= 1:
        n_total = n

    sel = _stratified_subsample_indices(y_all, n_total, rng, mode=context_sampling)
    X_sub = X_all[sel]
    y_sub = y_all[sel]

    n_query = max(1, int(round(n_total * query_fraction)))
    n_query = min(n_query, n_total - 1)
    n_ctx = n_total - n_query

    X_ctx = np.ascontiguousarray(X_sub[:n_ctx])
    X_qry = np.ascontiguousarray(X_sub[n_ctx:])
    y_ctx = y_sub[:n_ctx]
    y_qry = y_sub[n_ctx:]

    # ---- Phase 2b: per-step TabPFN preprocessing ----------------------- #
    return build_ensemble_members(
        X_ctx=X_ctx,
        y_ctx_raw=y_ctx,
        X_qry=X_qry,
        y_qry_raw=y_qry,
        feature_schema=cleaned.feature_schema,
        ensemble_configs=cleaned.ensemble_configs,
        outlier_removal_std=cleaned.outlier_removal_std,
        task_type=loaded.task_type,
        n_classes=n_classes,
        inference_config=inference_config,
        n_estimators=n_estimators,
        rng_seed=int(rng_seed),
        dataset_id=loaded.dataset_id,
    )


# --------------------------------------------------------------------------- #
# Public: training Dataset
# --------------------------------------------------------------------------- #


class ProcessedDatasetLoader(Dataset):
    """One ``__getitem__`` call → one batch (legacy ``TabPFNBatch`` OR
    new ``TabPFNEnsembleBatch``) from one sanitized CSV.

    Designed to be wrapped in ``torch.utils.data.DataLoader`` with
    ``batch_size=1`` and ``collate_fn=identity_collate``.

    The training loop must call :meth:`set_epoch` before each new
    epoch so the per-epoch reshuffle (epoch-aware RNG seed) produces
    a fresh random subsample of large datasets every epoch. The
    subsample is still deterministic given ``(base_seed, epoch, idx)``,
    so a re-run with the same seed is bit-for-bit reproducible.

    Parameters
    ----------
    inference_config
        TabPFN's ``InferenceConfig`` (returned by
        ``load_tabpfn_for_training``). When non-None, ``__getitem__``
        runs TabPFN's official preprocessor and returns a
        ``TabPFNEnsembleBatch`` with ``n_estimators_finetune``
        preprocessed views. When None, falls back to the legacy
        single-view ``TabPFNBatch`` (used by the mocked smoke test that
        doesn't have a real TabPFN checkpoint).
    n_estimators_finetune
        How many preprocessed views per step. TabPFN's official
        ``FinetunedTabPFNClassifier`` defaults to 2 — different feature
        shifts, different class permutations, gradient averaged. See
        ``tfm-library/repositories/TabPFN .txt``.
    """

    def __init__(
        self,
        refs: Sequence[DatasetRef],
        *,
        max_rows_per_epoch: int,
        query_fraction: float,
        seed: int = 0,
        inference_config: Any | None = None,
        n_estimators_finetune: int = 2,
        pass_mode: str = "one_sample",
        max_cells_per_epoch: int | None = None,
        model_family: str = "tabpfn",
        context_sampling: str = "stratified",
    ) -> None:
        if len(refs) == 0:
            raise ValueError("ProcessedDatasetLoader received an empty refs list")
        if model_family not in ("tabpfn", "tabicl"):
            raise ValueError(f"unknown model_family: {model_family!r}")
        self.model_family = model_family
        # Context-construction strategy for the per-step subsample; see
        # _stratified_subsample_indices and docs/METHOD.md.
        self.context_sampling = str(context_sampling)
        self.refs = list(refs)
        self.max_rows_per_epoch = int(max_rows_per_epoch)
        self.query_fraction = float(query_fraction)
        self._base_seed = int(seed)
        self._epoch = 0
        self._inference_config = inference_config
        self._n_estimators_finetune = max(1, int(n_estimators_finetune))
        self.pass_mode = str(pass_mode)
        # Optional cell budget. When set (>0), the per-step row count for a
        # dataset is min(max_rows_per_epoch, max_cells_per_epoch // n_features)
        # — so narrow datasets get more rows and wide ones fewer, at roughly
        # constant rows×features. This is the right control for TabPFN-3 (whose
        # capacity is a "cell-budget frontier", TabPFN-3 report §2) but NOT for
        # v2.6 (dual-attention cost is O(r²c + rc²) — quadratic in rows — so a
        # row cap is correct there). Off (None) → pure row cap (unchanged).
        self.max_cells_per_epoch = (
            int(max_cells_per_epoch) if max_cells_per_epoch else None
        )

        # Step plan: a list of (ref_idx, replica) pairs, one per training
        # step in an epoch.
        #
        #  * "one_sample" (default): exactly ONE step per dataset per epoch —
        #    each step a fresh random subsample of <= max_rows_per_epoch rows.
        #    This is the original behaviour; nothing changes.
        #
        #  * "full_pass": size-proportional steps. A dataset with n rows gets
        #    ceil(n / max_rows_per_epoch) steps per epoch, so a large dataset
        #    contributes MANY steps (each a different draw, seeded by the
        #    replica index) instead of just one. This exploits the unused data
        #    in the big datasets without repeating it within a step; the small
        #    datasets stay at 1 step (no extra repetition / overfit pressure).
        #    The loop's steps-per-epoch auto-scales via len(self).
        if self.pass_mode == "full_pass":
            self._plan: list[tuple[int, int]] = []
            for ref_idx, ref in enumerate(self.refs):
                try:
                    loaded = _load_processed_csv(ref)
                    n_rows = int(len(loaded.y))
                    eff_cap = self._effective_cap(loaded)
                except Exception:                                  # pragma: no cover
                    n_rows = eff_cap = self.max_rows_per_epoch
                k = max(1, math.ceil(n_rows / max(1, eff_cap)))
                self._plan.extend((ref_idx, r) for r in range(k))
        else:
            if self.pass_mode != "one_sample":
                LOGGER.warning(
                    "Unknown pass_mode=%r; falling back to 'one_sample'.",
                    self.pass_mode,
                )
                self.pass_mode = "one_sample"
            self._plan = [(i, 0) for i in range(len(self.refs))]

    def set_epoch(self, epoch: int) -> None:
        """Bump the epoch counter so the next __getitem__ reshuffles."""
        self._epoch = int(epoch)

    def _effective_cap(self, loaded: "_LoadedDataset") -> int:
        """Per-step row cap for one dataset.

        Pure row cap (``max_rows_per_epoch``) unless a cell budget is set,
        in which case it's ``min(max_rows_per_epoch, max_cells // n_features)``
        — narrow datasets get more rows, wide ones fewer, at ~constant cells.
        Floored at 256 rows so a very wide dataset still trains on something.
        """
        cap = self.max_rows_per_epoch
        if self.max_cells_per_epoch:
            try:
                n_feat = max(1, int(loaded.X.shape[1]))
            except Exception:                                      # pragma: no cover
                return cap
            cap = min(cap, max(256, self.max_cells_per_epoch // n_feat))
        return int(cap)

    def __len__(self) -> int:
        return len(self._plan)

    def __getitem__(self, idx: int):
        ref_idx, replica = self._plan[idx]
        ref = self.refs[ref_idx]
        loaded = _load_processed_csv(ref)
        n_total_target = self._effective_cap(loaded)
        # Epoch-aware seed: same chunk on epoch 0 ≠ epoch 1 ≠ …; the replica
        # term makes each of a dataset's full_pass steps draw a distinct
        # subsample within the same epoch.
        step_seed = (
            self._base_seed * 1_000_003
            + self._epoch * 10_007
            + ref_idx * 31
            + replica * 131_071
        ) & 0xFFFF_FFFF
        rng = np.random.default_rng(step_seed)

        # ---- TabICLv2 family path (official tabicl finetune preprocessing) --- #
        if self.model_family == "tabicl":
            return _build_tabicl_step_batch(
                loaded,
                n_total_target=n_total_target,
                query_fraction=self.query_fraction,
                rng=rng,
                n_estimators=self._n_estimators_finetune,
                epoch_seed=step_seed,
                replica=replica,
                context_sampling=self.context_sampling,
                # Fixed per dataset across epochs (their design): coarse
                # preprocessing choices stay stable, splits vary per epoch.
                preprocessing_seed=self._base_seed * 7_919 + ref_idx,
            )

        # Legacy single-view path (smoke tests, debug runs without a
        # real InferenceConfig).
        if self._inference_config is None:
            return _build_step_batch(
                loaded,
                n_total_target=n_total_target,
                query_fraction=self.query_fraction,
                rng=rng,
                context_sampling=self.context_sampling,
            )

        # ---- TabPFN-preprocessed N-estimator path ------------------- #
        # Mirrors `DatasetCollectionWithPreprocessing.__getitem__`
        # (`tfm-library/repositories/TabPFN .txt`).
        return _build_ensemble_step_batch(
            loaded,
            # Use the SAME effective cap as the legacy path and the full_pass
            # step-count math (`_effective_cap`), not the raw row cap —
            # otherwise the optional max_cells_per_epoch budget is silently
            # ignored on the real (ensemble) training path, and under
            # full_pass the per-step row count would disagree with the
            # k = ceil(n_rows / eff_cap) step count. (Bug fixed 2026-06-23.)
            n_total_target=n_total_target,
            query_fraction=self.query_fraction,
            rng=rng,
            inference_config=self._inference_config,
            context_sampling=self.context_sampling,
            n_estimators=self._n_estimators_finetune,
            rng_seed=int(step_seed),
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
            "tfm-library/repositories/TabPFN .txt)"
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
    query_fraction: float = 0.20,
) -> TabPFNBatch:
    """Build a deterministic eval batch for one dataset.

    Uses a ``(1 - query_fraction) / query_fraction`` context/query split,
    matching the trial's training-time split so the per-epoch monitor
    measures the same geometry the optimizer just learned on. When
    ``n_inference_subsample_samples`` is smaller than the dataset, both
    splits are subsampled proportionally.

    Defaults to a 80/20 split (qf=0.20) for back-compat with tests and
    callers that don't sweep the axis.
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
    n_query = max(1, int(round(n_total * float(query_fraction))))
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
        ctx_pos_rate=(
            float((np.asarray(y_ctx) > 0).mean())
            if loaded.task_type == "classification" else float("nan")
        ),
    )


# Backwards-compat alias.
prepare_validation_chunk = prepare_eval_chunk
