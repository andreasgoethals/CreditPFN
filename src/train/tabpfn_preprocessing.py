"""TabPFN preprocessing pipeline mirror for continued pretraining.

This module wires TabPFN's official preprocessor
(:class:`TabPFNEnsemblePreprocessor`) into our per-step training data
preparation. **Without this, training-time inputs do not match the
distribution the model was pretrained on, which is the root cause of the
calibration-collapse failure mode we observed on 2026-05-27** (audit in
chat 2026-05-27; verified against ``repositories/TabPFN .txt`` line
ranges below).

Why a separate module
---------------------
The preprocessing call surface is wide (`TabPFNEnsemblePreprocessor`
takes a dozen arguments threaded through `InferenceConfig`,
`EnsembleConfig` factories, and a `FeatureSchema`). Keeping it inline in
``dataloader.py`` would obscure the per-step subsampling logic. Here it
lives behind one entry point :func:`build_ensemble_members`.

What it does (mirrors `DatasetCollectionWithPreprocessing.__getitem__`)
----------------------------------------------------------------------
Given the per-step ``(X_ctx, y_ctx, X_qry, y_qry, cat_indices)`` tuple:

  1. Build per-estimator ``EnsembleConfig`` objects via TabPFN's
     ``generate_classification_ensemble_configs`` /
     ``generate_regression_ensemble_configs`` (``TabPFN .txt:31415,
     31490``). These carry per-estimator feature shifts, class
     permutations, target transforms, and outlier-removal std.

  2. Instantiate :class:`TabPFNEnsemblePreprocessor` with these configs
     and the active ``InferenceConfig`` knobs (FEATURE_SUBSAMPLING_*,
     SUBSAMPLE_SAMPLES, FINGERPRINT_FEATURE, …).

  3. Call ``fit_transform_ensemble_members(X_ctx, y_ctx)`` —
     returns ``list[TabPFNEnsembleMember]``. Each member has its own
     preprocessed ``X_train`` (and label-permuted ``y_train``).

  4. For each member, call ``member.transform_X_test(X_qry)`` to
     preprocess the query features with the SAME pipeline that was
     fit on context.

  5. Wrap the result in :class:`TabPFNEnsembleBatch` (one batch carries
     N ensemble members; the training loop runs N forward passes per
     step, one per member).

What it does NOT do
-------------------
* The **GPU outlier-removal step** (``TorchSoftClipOutliersStep``,
  default n_sigma=12 for classifier) is intentionally NOT applied here.
  It is the only "GPU" step in TabPFN's official pipeline and runs
  inside ``_call_model`` via ``_maybe_run_gpu_preprocessing``
  (``TabPFN .txt:9398``). We mirror this in our training loop: the
  outlier-clip is applied just before the forward pass against the live
  training model, on the same device as the model.

* It does **not** support ``cache_trainset_representation`` (TabPFN's
  v2.6 / v3 fit-with-cache mode). Each step re-fits the preprocessor on
  the context split — same as the official finetune.

Reference citations
-------------------
* `TabPFNEnsemblePreprocessor`: ``TabPFN .txt:30477-30733``.
* `fit_transform_ensemble_members`: ``TabPFN .txt:30721-30733``.
* `DatasetCollectionWithPreprocessing.__getitem__`: ``TabPFN .txt:26147-26319``
  — the official finetune's exact code path that we mirror here.
* `generate_classification_ensemble_configs`: ``TabPFN .txt:31415``.
* `generate_regression_ensemble_configs`: ``TabPFN .txt:31490``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Per-dataset clean_data cache
# --------------------------------------------------------------------------- #
#
# TabPFN's `clean_data` (verified at `repositories/TabPFN .txt:29818-29846`)
# takes (X, feature_schema) and returns (X_numpy_clean, ordinal_encoder,
# feature_schema). It runs `fix_dtypes` (ensures numeric ndarray, casts
# categoricals to `category` dtype) and `process_text_na_dataframe`
# (ordinal-encodes categoricals to integer codes; NaNs are preserved).
#
# The official multi-dataset finetune (`get_preprocessed_dataset_chunks`,
# `TabPFN .txt:26604-26635`) calls `_initialize_dataset_preprocessing`
# ONCE per parent dataset BEFORE the per-step DataLoader iteration —
# the result is cached in `ClassifierDatasetConfig.X_raw` (the name is
# misleading; the X stored there is ALREADY clean_data-cleaned). Then
# per-step preprocessing just slices that numeric array and runs the
# `TabPFNEnsemblePreprocessor`.
#
# We mirror that exactly with an LRU cache keyed by `dataset_id` so each
# parent dataset is cleaned once per training process. The cache holds
# the cleaned numpy array (numeric dtype, ordinal-encoded categoricals)
# alongside the feature_schema with which to drive the per-step
# preprocessor.


@dataclass(frozen=True)
class _CleanedDataset:
    """One ``clean_data``-cleaned dataset PLUS its per-estimator
    ``EnsembleConfig`` list — cached by ``dataset_id``.

    **Why ensemble_configs are stored per dataset (not regenerated per
    step):** the official multi-dataset finetune
    (`TabPFN .txt:26604-26635` and `__getitem__` at 26193-26203) builds
    the configs ONCE per dataset chunk and stores them on the
    ``DatasetConfig``. Per-step `__getitem__` reuses these — only the
    train/test SPLIT and per-step preprocessor RNG seed change.
    Re-rolling the EnsembleConfigs every step (our pre-2026-05-27
    behaviour) adds unnecessary gradient variance and deviates from
    the published methodology. Fixed by moving generation here.
    """
    X_clean: np.ndarray            # (n_total, n_features) float64, NaN-preserving
    y: np.ndarray                  # original (un-encoded) targets
    feature_schema: Any            # tabpfn.preprocessing.datamodel.FeatureSchema
    task_type: str                 # "classification" / "regression"
    dataset_id: str
    # Per-dataset ensemble configs (length == n_estimators_finetune):
    ensemble_configs: tuple        # tuple[ClassifierEnsembleConfig | RegressorEnsembleConfig, ...]
    outlier_removal_std: float | None


def _clean_one_dataset(
    *,
    X_full_df: pd.DataFrame,
    y_full: np.ndarray,
    cat_columns: Sequence[str],
    task_type: str,
    dataset_id: str,
    n_estimators: int,
    n_classes: int | None,
    inference_config: Any,
    rng_seed: int,
) -> _CleanedDataset:
    """Run TabPFN's ``clean_data`` once per parent dataset AND generate
    the per-dataset EnsembleConfig list (with class permutations,
    feature shifts, etc.) ONCE — mirrors
    ``_initialize_dataset_preprocessing`` at
    ``TabPFN .txt:7686-7733`` (classifier) and 13270-13298 (regressor).

    The ``rng_seed`` is derived from ``(base_seed, dataset_id)`` so each
    dataset's configs are deterministic but distinct across datasets.
    Across epochs, the per-step preprocessor's *internal* random_state
    varies (so quantile fits, SVD seeds, etc. change every step), but
    the COARSE choices (which class permutation, which feature shift)
    stay stable per dataset — exactly the published behaviour.
    """
    from tabpfn.preprocessing import (
        clean_data,
        generate_classification_ensemble_configs,
        generate_regression_ensemble_configs,
    )
    from tabpfn.preprocessing.datamodel import FeatureSchema
    try:
        from tabpfn.preprocessing.steps import (
            get_all_reshape_feature_distribution_preprocessors,
        )
    except ImportError:                                                # pragma: no cover
        from tabpfn.preprocessing.steps.reshape_feature_distributions_step import (  # type: ignore[import-not-found]
            get_all_reshape_feature_distribution_preprocessors,
        )

    # Silence pandas's noisy `invalid value encountered in cast` warnings
    # that fire inside `clean_data`'s `fix_dtypes` when checking whether
    # a float-with-NaN column can be safely cast to int. Cosmetic — does
    # not affect the result (the cast is wrapped in an `if` that
    # gracefully falls through). Filed as P4 in chat 2026-05-28.
    import warnings as _warnings
    _warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in cast",
        category=RuntimeWarning,
        module=r"pandas\.core\.dtypes\.cast",
    )

    # Vocab translation (see build_ensemble_members for the rationale).
    if task_type == "classification":
        tabpfn_task_type = "classifier"
    elif task_type == "regression":
        tabpfn_task_type = "regressor"
    else:
        raise ValueError(f"unknown task_type {task_type!r}")

    cols = list(X_full_df.columns)
    cat_positions = [cols.index(c) for c in cat_columns if c in cols]
    feature_schema = FeatureSchema.from_only_categorical_indices(
        cat_positions, num_columns=int(X_full_df.shape[1]),
    )
    X_clean, _ord_encoder, feature_schema = clean_data(
        X=X_full_df, feature_schema=feature_schema,
    )
    # X_clean is a 2-D numpy array, float64, with categoricals as
    # integer codes and NaNs preserved. Exactly what TabPFN's
    # ensemble preprocessor expects (line 26244-26258 of the dump).

    outlier_removal_std = inference_config.get_resolved_outlier_removal_std(
        estimator_type=tabpfn_task_type,
    )

    # Build per-dataset ensemble configs.
    if task_type == "classification":
        ensemble_configs = generate_classification_ensemble_configs(
            num_estimators=int(n_estimators),
            add_fingerprint_feature=inference_config.FINGERPRINT_FEATURE,
            feature_shift_decoder=inference_config.FEATURE_SHIFT_METHOD,
            polynomial_features=inference_config.POLYNOMIAL_FEATURES,
            preprocessor_configs=list(inference_config.PREPROCESS_TRANSFORMS),
            class_shift_method=inference_config.CLASS_SHIFT_METHOD,
            n_classes=int(n_classes if n_classes is not None else 2),
            random_state=int(rng_seed),
            num_models=1,
            outlier_removal_std=outlier_removal_std,
        )
    else:
        target_factories = get_all_reshape_feature_distribution_preprocessors(
            num_examples=int(X_full_df.shape[0]),
            random_state=int(rng_seed),
        )
        target_transforms = []
        for name in inference_config.REGRESSION_Y_PREPROCESS_TRANSFORMS:
            if name is None:
                target_transforms.append(None)
            else:
                target_transforms.append(target_factories[name])
        ensemble_configs = generate_regression_ensemble_configs(
            num_estimators=int(n_estimators),
            add_fingerprint_feature=inference_config.FINGERPRINT_FEATURE,
            feature_shift_decoder=inference_config.FEATURE_SHIFT_METHOD,
            polynomial_features=inference_config.POLYNOMIAL_FEATURES,
            preprocessor_configs=list(inference_config.PREPROCESS_TRANSFORMS),
            target_transforms=target_transforms,
            random_state=int(rng_seed),
            num_models=1,
            outlier_removal_std=outlier_removal_std,
        )

    return _CleanedDataset(
        X_clean=np.asarray(X_clean),
        y=np.asarray(y_full),
        feature_schema=feature_schema,
        task_type=task_type,
        dataset_id=dataset_id,
        ensemble_configs=tuple(ensemble_configs),
        outlier_removal_std=(
            float(outlier_removal_std) if outlier_removal_std is not None else None
        ),
    )


# Cache by (dataset_id, n_rows, n_cols, task_type, n_estimators, base_seed).
_CLEAN_CACHE: dict[tuple, _CleanedDataset] = {}


def clean_loaded_dataset(
    *,
    X_full_df: pd.DataFrame,
    y_full: np.ndarray,
    cat_columns: Sequence[str],
    task_type: str,
    dataset_id: str,
    n_estimators: int,
    n_classes: int | None,
    inference_config: Any,
    base_seed: int,
) -> _CleanedDataset:
    """Cached wrapper for :func:`_clean_one_dataset`.

    Cache key includes ``n_estimators`` and ``base_seed`` so a sweep
    that changes either of them gets a fresh cache entry (different
    EnsembleConfig list).
    """
    key = (
        dataset_id,
        int(X_full_df.shape[0]),
        int(X_full_df.shape[1]),
        task_type,
        int(n_estimators),
        int(base_seed),
    )
    if key not in _CLEAN_CACHE:
        # Derive a stable per-dataset rng_seed from (base_seed, dataset_id).
        # `hash` over a tuple of ints+str — deterministic in CPython 3.6+
        # via the PYTHONHASHSEED env (we don't seed it; this is just a
        # mixer to spread dataset_ids across the int32 range).
        import hashlib
        h = hashlib.blake2b(
            f"{base_seed}::{dataset_id}".encode("utf-8"),
            digest_size=8,
        ).digest()
        rng_seed = int.from_bytes(h, "big") & 0x7FFF_FFFF
        _CLEAN_CACHE[key] = _clean_one_dataset(
            X_full_df=X_full_df,
            y_full=y_full,
            cat_columns=cat_columns,
            task_type=task_type,
            dataset_id=dataset_id,
            n_estimators=n_estimators,
            n_classes=n_classes,
            inference_config=inference_config,
            rng_seed=rng_seed,
        )
    return _CLEAN_CACHE[key]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class _PerEstimatorView:
    """One preprocessed view of the (context, query) split — one of
    ``n_estimators_finetune`` per training step.

    All tensor shapes match TabPFN's ``PerFeatureTransformer`` signature
    (``repositories/TabPFN .txt:15198-15217``):

    * ``X_context`` — (n_ctx,   1, n_features_after_preproc)  float32
    * ``y_context`` — (n_ctx,   1, 1)                         long / float32
    * ``X_query``   — (n_query, 1, n_features_after_preproc)  float32
    * ``categorical_idx`` — list[int]  (positional indices into the
                             POST-preprocessing feature space)
    * ``class_permutation`` — np.ndarray | None
        For classifier members only: the per-estimator class-shuffle
        permutation, used by the loop to unscramble logits before the
        CE loss. ``None`` for regression and for identity-permutation
        members.
    * ``outlier_removal_std`` — float | None
        The σ threshold for the GPU soft-clip step that runs at forward
        time. The loop passes this to its own outlier-clip helper.
    """
    X_context: torch.Tensor
    y_context: torch.Tensor
    X_query:   torch.Tensor
    categorical_idx: list[int]
    class_permutation: np.ndarray | None
    outlier_removal_std: float | None


@dataclass
class TabPFNEnsembleBatch:
    """One training step's payload — N preprocessed views of the same
    (context, query) split.

    Carries enough info for the loop to:
      1. Forward each member through the model and stack logits as
         ``(Q, B, E, L)`` — matches the official
         ``FinetunedTabPFNClassifier._forward_with_loss`` shape at
         ``TabPFN .txt:26920-26941``.
      2. Apply the per-member class-permutation undo on each logit
         tensor before the CE loss sees it.
      3. Compute the CE / NLL loss against the canonical-class-order
         ``y_query`` (which is repeated E times across the batch dim).
    """
    members: list[_PerEstimatorView]
    y_query: torch.Tensor                 # (n_query, 1, 1) long / float32
    task_type: str                        # "classification" | "regression"
    dataset_id: str
    n_classes: int | None                 # None for regression

    # Regression-only: z-norm statistics applied to ``y`` BEFORE the
    # per-estimator target transform. The loop uses these to invert the
    # transform for RMSE / R² metrics (raw target units).
    znorm_mean: float | None = None
    znorm_std:  float | None = None

    def to(self, device: str) -> "TabPFNEnsembleBatch":
        new_members = [
            _PerEstimatorView(
                X_context=m.X_context.to(device, non_blocking=True),
                y_context=m.y_context.to(device, non_blocking=True),
                X_query=m.X_query.to(device, non_blocking=True),
                categorical_idx=m.categorical_idx,
                class_permutation=m.class_permutation,
                outlier_removal_std=m.outlier_removal_std,
            )
            for m in self.members
        ]
        return TabPFNEnsembleBatch(
            members=new_members,
            y_query=self.y_query.to(device, non_blocking=True),
            task_type=self.task_type,
            dataset_id=self.dataset_id,
            n_classes=self.n_classes,
            znorm_mean=self.znorm_mean,
            znorm_std=self.znorm_std,
        )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def build_ensemble_members(
    *,
    X_ctx: np.ndarray,             # (n_ctx, n_features) float64 — ALREADY clean_data-cleaned
    y_ctx_raw: np.ndarray,
    X_qry: np.ndarray,             # (n_qry, n_features) float64 — ALREADY clean_data-cleaned
    y_qry_raw: np.ndarray,
    feature_schema: Any,           # tabpfn.preprocessing.datamodel.FeatureSchema (cleaned)
    ensemble_configs: Sequence,    # per-dataset cached configs (from clean_loaded_dataset)
    outlier_removal_std: float | None,    # cached on the _CleanedDataset
    task_type: str,                # "classification" | "regression"
    n_classes: int | None,
    inference_config: Any,         # tabpfn.inference_config.InferenceConfig
    n_estimators: int,
    rng_seed: int,
    dataset_id: str,
) -> TabPFNEnsembleBatch:
    """Apply TabPFN's official preprocessing pipeline to one step's data.

    Returns one :class:`TabPFNEnsembleBatch` carrying ``n_estimators``
    preprocessed views. The CPU pipeline (per-feature squashing scaler,
    quantile transform, SVD, fingerprint, ordinal-shuffled categoricals,
    polynomial features when configured) is applied here. The GPU
    soft-clip outlier removal is deferred to forward time — the
    ``outlier_removal_std`` value travels with the batch.

    Mirrors ``DatasetCollectionWithPreprocessing.__getitem__`` at
    ``repositories/TabPFN .txt:26147-26319`` step-for-step.
    """
    # Lazy imports — TabPFN is a multi-hundred-MB dependency and we want
    # the test suite (which mocks load_tabpfn_for_training) to be able
    # to import this module without paying the cost.
    #
    # IMPORTANT — exact import paths verified against the installed
    # TabPFN source (mirrored at ``repositories/TabPFN .txt``):
    #
    #   * ``TabPFNEnsemblePreprocessor`` lives in
    #     ``tabpfn.preprocessing.ensemble`` (NOT re-exported from the
    #     top-level ``tabpfn.preprocessing`` — verified at
    #     ``repositories/TabPFN .txt:5876``).
    #   * ``FeatureSubsamplingMethod`` re-exported from
    #     ``tabpfn.preprocessing`` (``__all__`` at line 29763).
    #   * ``FeatureModality`` re-exported from
    #     ``tabpfn.preprocessing.datamodel``.
    #
    # `generate_*_ensemble_configs`, `get_all_reshape_feature_distribution_preprocessors`,
    # `clean_data`, `FeatureSchema` — used in `_clean_one_dataset`, NOT
    # here. Importing them here would just shadow the function-local
    # scope unnecessarily.
    from tabpfn.preprocessing import FeatureSubsamplingMethod
    from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor
    from tabpfn.preprocessing.datamodel import FeatureModality

    # ---- 0) translate task-type vocabulary --------------------------- #
    # **VOCAB GAP — fixed 2026-05-27.** Our codebase uses
    # ``"classification"`` / ``"regression"`` everywhere (matches
    # sklearn conventions, our YAML files, and `DatasetRef.task_type`).
    # TabPFN's API uses the shorter ``"classifier"`` / ``"regressor"``
    # (matches `BaseEstimator._estimator_type`). Passing our vocabulary
    # to TabPFN's helpers SILENTLY MISBEHAVES — e.g.
    # `get_resolved_outlier_removal_std("regression")` returns the
    # classifier default (12.0σ) because the comparison
    # `estimator_type == "regressor"` is False, falling through to the
    # classifier branch. Verified at `TabPFN .txt:10622-10637`.
    if task_type == "classification":
        tabpfn_task_type = "classifier"
    elif task_type == "regression":
        tabpfn_task_type = "regressor"
    else:                                                              # pragma: no cover
        raise ValueError(
            f"task_type must be 'classification' or 'regression'; got {task_type!r}"
        )

    # ---- 1) feature_schema is supplied by the caller ------------------- #
    # The caller has already run ``clean_data`` on the full dataset and
    # passed in the resulting numeric numpy arrays plus the
    # ``FeatureSchema``. Mirrors the official multi-dataset finetune
    # which stores the cleaned ``X_mod`` in
    # ``ClassifierDatasetConfig.X_raw`` (TabPFN .txt:26604-26635) —
    # the "raw" in the name is a misnomer; that array is already
    # ordinal-encoded and numeric.
    assert X_ctx.ndim == 2, f"X_ctx must be 2D, got shape {X_ctx.shape}"
    assert X_qry.ndim == 2, f"X_qry must be 2D, got shape {X_qry.shape}"
    assert X_ctx.shape[1] == X_qry.shape[1], (
        f"X_ctx and X_qry must have same n_features; "
        f"got {X_ctx.shape[1]} vs {X_qry.shape[1]}"
    )

    # ---- 2) ensemble configs come from the per-dataset cache ----------- #
    # Mirrors the official path: `_initialize_dataset_preprocessing` runs
    # ONCE per parent dataset (TabPFN .txt:7686-7733 cls, 13270-13298
    # reg) and stores configs on the `DatasetConfig`. Per-step
    # `__getitem__` reuses these — only the per-step RNG seed differs.
    # Re-rolling per step would change class permutations and feature
    # shifts every step, adding gradient noise the published pipeline
    # does not have.
    assert len(ensemble_configs) == int(n_estimators), (
        f"ensemble_configs has length {len(ensemble_configs)} but "
        f"n_estimators={n_estimators}; the per-dataset cache is stale "
        "for a different n_estimators_finetune setting"
    )

    # ---- 3) regression: pre-z-norm y on context-only stats ------------ #
    # Matches `DatasetCollectionWithPreprocessing.__getitem__` lines
    # 26220-26240 — the official multi-dataset finetune path. Critical
    # detail: when `train_std < 1e-8` we use 1e-8 (with a warning),
    # NOT a tiny epsilon like 1e-20. Adding 1e-20 to a near-zero std
    # produces z-scores of ~1e+20, which immediately overflow fp16
    # gradients (one of the diagnosed causes of the 2026-05-27 inf
    # grad_norm spikes — see chat).
    import warnings as _warnings
    znorm_mean: float | None = None
    znorm_std:  float | None = None
    if task_type == "regression":
        znorm_mean = float(np.mean(y_ctx_raw))
        train_std = float(np.std(y_ctx_raw))
        eps = 1e-8
        if train_std < eps:
            _warnings.warn(
                f"Target variable for dataset={dataset_id} has constant or "
                f"near-constant values (std={train_std:.2e}). Clamping to "
                f"eps={eps:.0e} to prevent division by zero in standardization.",
                UserWarning,
                stacklevel=2,
            )
            train_std = eps
        znorm_std = train_std
        y_ctx_for_pre = ((y_ctx_raw - znorm_mean) / znorm_std).astype(np.float32)
        y_qry_for_loss = ((y_qry_raw - znorm_mean) / znorm_std).astype(np.float32)
    else:
        y_ctx_for_pre = y_ctx_raw.astype(np.int64)
        y_qry_for_loss = y_qry_raw.astype(np.int64)

    # ---- 4) build the TabPFNEnsemblePreprocessor ---------------------- #
    # The constructor signature is wide; most kwargs read from
    # inference_config. n_preprocessing_jobs=1 keeps everything in this
    # process (we already have DataLoader workers disabled).
    # enable_gpu_preprocessing=False so the soft-clip step is built
    # but not run until the forward pass (we apply it manually in
    # `_forward_one_member`).
    #
    # **VOCAB GAP** — same translation as for `outlier_removal_std`
    # above. The constructor's ``task_type`` kwarg expects ``"classifier"``
    # or ``"regressor"`` (TabPFN .txt:30504).
    #
    # **ENUM WRAP** — ``feature_subsampling_method`` is typed as
    # ``FeatureSubsamplingMethod`` (a ``(str, Enum)`` at TabPFN
    # .txt:29979). The official call wraps the raw string from
    # inference_config in the enum constructor (TabPFN .txt:13444).
    # We mirror that — passing the raw string MAY work due to the str
    # mix-in, but explicit wrapping is safer across versions.
    preprocessor = TabPFNEnsemblePreprocessor(
        configs=ensemble_configs,
        n_samples=int(X_ctx.shape[0]),
        feature_schema=feature_schema,
        random_state=int(rng_seed),
        n_preprocessing_jobs=1,
        keep_fitted_cache=False,
        enable_gpu_preprocessing=False,
        feature_subsampling_method=FeatureSubsamplingMethod(
            inference_config.FEATURE_SUBSAMPLING_METHOD
        ),
        constant_feature_count=inference_config.FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT,
        subsample_samples=inference_config.SUBSAMPLE_SAMPLES,
        importance_top_k_count=inference_config.FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT,
        X_train=X_ctx,
        y_train=y_ctx_for_pre,
        task_type=tabpfn_task_type,
    )

    # ---- 5) fit on context, transform context AND query --------------- #
    members = preprocessor.fit_transform_ensemble_members(
        X_train=X_ctx,
        y_train=y_ctx_for_pre,
    )

    # ---- 6) tensorise per-estimator views ----------------------------- #
    per_estimator_views: list[_PerEstimatorView] = []
    for member, config in zip(members, ensemble_configs):
        # member.X_train: (n_ctx, n_features_after_preproc) float
        # member.y_train: (n_ctx,) — already class-permuted for classifier
        X_ctx_np = np.asarray(member.X_train, dtype=np.float32)
        y_ctx_np = np.asarray(member.y_train)

        # Apply the SAME CPU pipeline to query features via the member's
        # bound transform_X_test (line 26265 in the official finetune).
        X_qry_np = np.asarray(
            member.transform_X_test(X_qry), dtype=np.float32,
        )

        # Cast to torch tensors with shape (n, 1, F) / (n, 1, 1).
        X_ctx_t = torch.from_numpy(np.ascontiguousarray(X_ctx_np)).unsqueeze(1)
        X_qry_t = torch.from_numpy(np.ascontiguousarray(X_qry_np)).unsqueeze(1)
        if task_type == "classification":
            y_ctx_t = torch.as_tensor(y_ctx_np, dtype=torch.int64).reshape(-1, 1, 1).contiguous()
        else:
            y_ctx_t = torch.as_tensor(y_ctx_np, dtype=torch.float32).reshape(-1, 1, 1).contiguous()

        # Categorical indices in the POST-preprocessing feature space.
        # FeatureSchema is mutated by the pipeline (new columns added by
        # SVD / polynomial, columns dropped by subsampling). The
        # member's `feature_schema` reflects the final layout.
        post_schema = getattr(member, "feature_schema", feature_schema)
        try:
            cat_idx_post = list(
                post_schema.indices_for(FeatureModality.CATEGORICAL)
            )
        except Exception:                                              # pragma: no cover
            cat_idx_post = []

        # The class_permutation is on the EnsembleConfig (the same one
        # used to drive _transform_labels_one). Pull it back out so the
        # forward path can undo it on the logits.
        class_perm = (
            np.asarray(config.class_permutation)
            if task_type == "classification" and getattr(config, "class_permutation", None) is not None
            else None
        )

        per_estimator_views.append(_PerEstimatorView(
            X_context=X_ctx_t,
            y_context=y_ctx_t,
            X_query=X_qry_t,
            categorical_idx=cat_idx_post,
            class_permutation=class_perm,
            outlier_removal_std=(
                float(outlier_removal_std) if outlier_removal_std is not None else None
            ),
        ))

    # ---- 7) build the canonical-order y_query for the loss ----------- #
    # NOTE the y_query stays in CANONICAL class order (no permutation
    # applied). The per-member class_permutation is used only on the
    # logits side — to swap the logit columns back into canonical order
    # before the loss compares against y_query. This matches the
    # official forward path at TabPFN .txt:8504-8525.
    if task_type == "classification":
        y_qry_t = torch.as_tensor(y_qry_for_loss, dtype=torch.int64).reshape(-1, 1, 1).contiguous()
    else:
        y_qry_t = torch.as_tensor(y_qry_for_loss, dtype=torch.float32).reshape(-1, 1, 1).contiguous()

    return TabPFNEnsembleBatch(
        members=per_estimator_views,
        y_query=y_qry_t,
        task_type=task_type,
        dataset_id=dataset_id,
        n_classes=n_classes,
        znorm_mean=znorm_mean,
        znorm_std=znorm_std,
    )


# --------------------------------------------------------------------------- #
# Helper: GPU outlier-clip (applied at forward time)
# --------------------------------------------------------------------------- #


def apply_outlier_clip(
    x: torch.Tensor, *,
    n_sigma: float | None,
    categorical_idx: Sequence[int] | None = None,
) -> torch.Tensor:
    """Mirror of TabPFN's ``TorchSoftClipOutliersStep`` (see
    ``TabPFN .txt:35959-35967``).

    Applied per training step on the combined ``(context+query)`` tensor
    just before model forward. The official pipeline runs this on
    NUMERICAL columns only; categoricals pass through unmodified.

    Math: for each numerical column j, compute column μ, σ on the
    finite rows; soft-clip to ``±n_sigma·σ`` via ``z / sqrt(1 + (z/B)^2)``
    where ``z = (x - μ) / σ``, ``B = n_sigma``.

    ``n_sigma=None`` → no-op (regression default). Returns a new tensor
    when clipping is active; the input ``x`` when not.
    """
    if n_sigma is None or n_sigma <= 0:
        return x

    # Identify numerical columns. The categorical_idx is positional in
    # the POST-preprocessing feature space — see
    # `_PerEstimatorView.categorical_idx`.
    n_features = x.shape[-1]
    cat_set = set(int(i) for i in (categorical_idx or []))
    num_idx = [i for i in range(n_features) if i not in cat_set]
    if not num_idx:
        return x

    out = x.clone()
    num_tensor = out[..., num_idx].float()
    # Stats over the row axis (axis 0) per (batch, feature).
    # Standard `unbiased=False` to match numpy `np.std`.
    mu = num_tensor.nanmean(dim=0, keepdim=True)
    sd = (
        (num_tensor - mu).square().nanmean(dim=0, keepdim=True)
        .sqrt().clamp_min(1e-6)
    )
    z = (num_tensor - mu) / sd
    soft = z / torch.sqrt(1.0 + (z / float(n_sigma)) ** 2) * sd + mu
    out[..., num_idx] = soft.to(out.dtype)
    return out
