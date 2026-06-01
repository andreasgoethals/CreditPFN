"""Dataset-agnostic cleaning (Stage 3 of the data pipeline).

Sequence applied to every dataset (PD or LGD), in order — keys
correspond to the steps in ``cfg.sanitize`` in ``config/data.yaml``:

  (b) drop exact-duplicate feature columns
  (c) drop feature columns whose NaN rate exceeds ``max_missing_rate``
  (d) drop all-NaN feature columns                  (edge case of (c))
  (e) drop constant feature columns                 (TabPFN errors on these)
  (f) coerce object columns that are mostly numeric strings to numeric
  (g) cast numerical features to ``numeric_dtype``  (default float32)
  (h) replace ±inf with NaN                         (uniform NaN handling)
  (i) **unsupervised feature SELECTION** to at most ``max_columns``
       features (``sanitize.max_columns``, currently 64); restricted to
       numerical features,
       categoricals always pass through. Keeps a subset of the *real*
       columns (top by scale-free variance, greedily de-correlated at
       ``corr_threshold``) — NOT cluster means. This preserves real
       marginals + interactions so continued pretraining specialises the
       prior toward genuine credit features (the old FeatureAgglomeration
       averaged columns into synthetic means, which defeated that goal).
  (j) classification targets → contiguous ``int64`` labels
  (k) regression targets — left in their raw scale (TabPFN's
      ``RegressorBatch.znorm_space_bardist_`` standardises internally).
      LGD targets are domain-clipped to ``[0, 1]`` here because that
      bound is a definition of the metric, not a statistical operation.

What this module deliberately does NOT do:

* No outlier winsorisation. ``OUTLIER_REMOVAL_STD = 12.0`` (classifier)
  / ``None`` (regressor) inside TabPFN handles outliers with the
  correct semantics — see ``repositories/REPOSITORIES.md``.
* No PowerTransformer / QuantileTransformer / RobustScaler. Those run
  per-estimator inside TabPFN's inference ensemble; pre-applying any
  of them on disk would break the ensemble's diversity.
* No imputation. ``NanHandlingEncoderStep`` handles NaNs natively.

Input / output
--------------
Reads
  * ``cfg.paths.raw/{pd,lgd}/<id>.csv``            (raw CSVs)
  * ``cfg.paths.manifest_pd`` / ``manifest_lgd``   (categorical hints)

Writes
  * ``cfg.paths.processed/{pd,lgd}/<id>.sanitized.csv``

Public entry point
------------------
``main(cfg) -> int``
    Returns 0 on full success, 1 if any dataset failed (logged with
    its dataset_id; the script does not abort).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import DATASET_METADATA, apply_dataset_specific_fixes
from src.utils.paths import resolve_data_path, resolve_output_path

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Pure helpers (no I/O)
# =============================================================================


def _drop_exact_duplicate_feature_columns(
    df: pd.DataFrame, target: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop feature columns whose values exactly equal another column.

    Equality includes matching NaN positions. Iterates left-to-right and
    keeps the first occurrence.
    """
    feat = [c for c in df.columns if c != target]
    keep, dropped = [], []
    seen: dict[bytes, str] = {}
    for col in feat:
        # Hash the column's bytes (NaN-aware via repr-ish encoding).
        s = df[col]
        # Use pandas' .equals semantics by comparing tobytes after
        # filling NaNs with a sentinel; fast for moderate widths.
        sentinel = np.frombuffer(b"NaN_placeholder_xX", dtype=np.uint8)
        if s.dtype.kind in "biufc":
            buf = np.where(s.isna(), -np.float64(1e308), s.astype(np.float64)).tobytes()
        else:
            # Fill NaNs *before* casting to str. On pandas 2.x,
            # astype(str) silently converts NaN to the literal string
            # "nan", so fillna() afterwards finds nothing to fill —
            # the order matters for portability across pandas versions.
            buf = "\x00".join(s.fillna("__NAN__").astype(str).tolist()).encode()
        key = buf + sentinel.tobytes()
        if key in seen:
            dropped.append(col)
        else:
            seen[key] = col
            keep.append(col)
    new_cols = ([target] if target in df.columns else []) + keep
    return df[new_cols].copy(), dropped


def _drop_high_missing_columns(
    df: pd.DataFrame, target: str, max_missing_rate: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop feature columns whose NaN rate exceeds ``max_missing_rate``.

    ``max_missing_rate=0.9`` → drop columns with >90% NaN. Always keeps
    the target column regardless of its NaN rate.
    """
    feat = [c for c in df.columns if c != target]
    rates = df[feat].isna().mean()
    drop = rates[rates > max_missing_rate].index.tolist()
    keep = [c for c in df.columns if c not in drop]
    return df[keep].copy(), drop


def _drop_constant_columns(
    df: pd.DataFrame, target: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop feature columns with ≤ 1 unique non-NaN value."""
    feat = [c for c in df.columns if c != target]
    drop = [c for c in feat if df[c].dropna().nunique() <= 1]
    keep = [c for c in df.columns if c not in drop]
    return df[keep].copy(), drop


def _coerce_numeric_strings(
    df: pd.DataFrame, target: str, threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Where ≥ ``threshold`` of a string-like column's non-NaN values
    parse as numeric, commit the coercion. Targets are left untouched.

    Treats both legacy ``object`` and the new pandas-3.x ``str`` /
    ``StringDtype`` columns as candidates — a single ``is_object_dtype``
    check would silently miss strings on pandas 3.x.
    """
    coerced: list[str] = []
    for col in df.columns:
        if col == target:
            continue
        dtype = df[col].dtype
        is_string_like = (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        )
        if not is_string_like:
            continue
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            continue
        non_null_in = df[col].notna().sum()
        if non_null_in == 0:
            continue
        non_null_out = converted.notna().sum()
        if non_null_out / non_null_in >= threshold:
            df[col] = converted
            coerced.append(col)
    return df, coerced


def _cast_numericals_to(
    df: pd.DataFrame, target: str, numerical_columns: list[str], dtype: str,
) -> pd.DataFrame:
    """Cast every column listed in ``numerical_columns`` to ``dtype``.

    Existing NaNs are preserved (float dtypes only — ``int64``-typed
    targets are handled separately).

    When casting to a smaller float dtype (e.g. ``float32``), values
    above the target's range — e.g. credit-risk features like
    debt-to-income with a near-zero denominator — would silently
    overflow to ``±inf`` AND emit a noisy
    ``RuntimeWarning: overflow encountered in cast`` from NumPy.
    Both effects are unhelpful: we want those out-of-range values to
    become NaN (the standard "this is a data issue, treat as missing"
    contract) cleanly, with no warning. So we explicitly replace
    ``±inf`` with NaN at the float64 stage — before the cast that
    would have produced the overflow.
    """
    np_dtype = np.dtype(dtype)
    is_narrow_float = (
        np_dtype.kind == "f" and np_dtype.itemsize < 8
    )
    for col in numerical_columns:
        if col not in df.columns or col == target:
            continue
        # Force float64 first; integer columns become floats (NaN-compatible).
        coerced = pd.to_numeric(df[col], errors="coerce")
        if is_narrow_float:
            # Catch both pre-existing ±inf in the source AND any value
            # that would overflow the narrower target dtype.
            target_max = np.finfo(np_dtype).max
            mask = ~np.isfinite(coerced.to_numpy(dtype=np.float64, na_value=np.nan))
            mask |= coerced.abs().to_numpy(dtype=np.float64, na_value=0.0) > target_max
            if mask.any():
                coerced = coerced.mask(mask, np.nan)
        df[col] = coerced.astype(np_dtype)
    return df


def _replace_inf_with_nan(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """In every numeric feature column, replace ±inf with NaN."""
    feat = [c for c in df.columns if c != target]
    for col in feat:
        if pd.api.types.is_numeric_dtype(df[col]):
            mask = np.isinf(df[col].to_numpy(dtype=np.float64, na_value=np.nan))
            if mask.any():
                df.loc[mask, col] = np.nan
    return df


def _select_to_max_columns(
    df: pd.DataFrame,
    target: str,
    numerical_columns: list[str],
    categorical_columns: list[str],
    max_columns: int,
    *,
    corr_threshold: float = 0.95,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Reduce the feature count to ≤ ``max_columns`` by **unsupervised
    feature SELECTION** — keep a subset of the *real* columns (numerical
    AND categorical), rather than averaging them into cluster means.

    Why selection, not the old ``FeatureAgglomeration``: continued
    pretraining aims to adjust TabPFN's prior toward the *real* marginal
    distributions and feature interactions of credit-risk data. Averaging
    columns into per-cluster means destroyed both, which works against that
    goal and is inconsistent with what the model sees at inference.
    Selection keeps real columns with real distributions. It is
    **unsupervised** (never touches ``y``) — no label leak.

    Both feature types are eligible, so a categorical-heavy dataset is
    capped too (the earlier version only trimmed numericals and skipped
    when the categoricals alone exceeded the budget — which left the LGD
    ``base_model*`` sets uncapped). Three steps:

      1. **Score** each feature, scale-free: numerical → variance after
         min-max to ``[0, 1]``; categorical → Shannon entropy of the value
         distribution, normalised to ``[0, 1]``. Near-constant of either
         type → ~0.
      2. **De-correlate** the numerical block (greedy, best-score-first,
         bounded candidate set): drop a numerical whose ``|Pearson r|``
         with an already-kept numerical exceeds ``corr_threshold``.
      3. **Rank** all survivors (numerical + categorical) by score and keep
         the top ``max_columns``; each kept column keeps its real name and
         type.

    Returns ``(df_reduced, kept_numerical_columns, kept_categorical_columns)``
    — all ORIGINAL column names.
    """
    feat_count = len(numerical_columns) + len(categorical_columns)
    if feat_count <= max_columns:
        return df, numerical_columns, categorical_columns

    scores: dict[str, float] = {}

    # (1a) Numerical score: variance after per-column min-max to [0, 1]
    # (scale-free; near-constant → ~0).
    if numerical_columns:
        nb = df[numerical_columns].astype(np.float64)
        span = (nb.max() - nb.min()).replace(0.0, np.nan)       # constant -> NaN
        nvar = ((nb - nb.min()) / span).var(axis=0, skipna=True).fillna(0.0)
        for c in numerical_columns:
            scores[c] = float(nvar.get(c, 0.0))

    # (1b) Categorical score: Shannon entropy of the value distribution,
    # normalised to [0, 1] (near-constant → ~0; balanced → ~1). This lets a
    # categorical-heavy dataset be capped too — the previous version only
    # trimmed numericals and silently skipped when the categoricals alone
    # exceeded the budget, leaving e.g. the LGD base_model* sets uncapped.
    for c in categorical_columns:
        vc = df[c].astype("object").value_counts(normalize=True, dropna=True)
        if len(vc) <= 1:
            scores[c] = 0.0
        else:
            p = vc.to_numpy(dtype=np.float64)
            scores[c] = float(-(p * np.log(p)).sum() / np.log(len(vc)))

    # (2) Greedy de-correlation among NUMERICAL columns (best-score-first,
    # bounded candidate set so the O(k²) correlation stays cheap). Drops a
    # numerical whose |Pearson r| with an already-kept numerical exceeds the
    # threshold. (Pearson does not apply to categoricals.)
    drop_corr: set[str] = set()
    num_ranked = sorted(numerical_columns, key=lambda c: scores.get(c, 0.0),
                        reverse=True)
    if len(num_ranked) > 1:
        n_cand = min(len(num_ranked), max(2 * max_columns, max_columns + 1))
        cand = num_ranked[:n_cand]
        cb = df[cand].astype(np.float64)
        cb = cb.fillna(cb.mean()).fillna(0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.atleast_2d(np.corrcoef(cb.to_numpy(), rowvar=False))
        kept_pos: list[int] = []
        for i in range(len(cand)):
            keep = True
            for j in kept_pos:
                cval = corr[i, j]
                if not np.isnan(cval) and abs(cval) > corr_threshold:
                    keep = False
                    break
            if keep:
                kept_pos.append(i)
            else:
                drop_corr.add(cand[i])

    # (3) Allocate the budget proportionally to each type's count, then keep
    # the top-scored within each type. A single cross-type ranking would be
    # biased: numerical min-max variance (~0.02-0.1) and normalised
    # categorical entropy (~0.5-1) are NOT on a comparable scale, so ranking
    # them together would keep almost only categoricals. Per-type allocation
    # preserves the dataset's feature-type balance; within a type we keep the
    # most-informative (numericals by variance after de-correlation,
    # categoricals by entropy). Leftover budget (when a type runs out) is
    # redistributed to the other type.
    num_survivors = [c for c in num_ranked if c not in drop_corr]      # score-ordered
    cat_ranked = sorted(categorical_columns, key=lambda c: scores.get(c, 0.0),
                        reverse=True)
    n_num_keep = int(round(max_columns * len(numerical_columns) / feat_count))
    n_num_keep = min(n_num_keep, len(num_survivors))
    n_cat_keep = min(max_columns - n_num_keep, len(cat_ranked))
    n_num_keep = min(len(num_survivors), max_columns - n_cat_keep)     # refill if cats short

    new_nums = num_survivors[:n_num_keep]
    new_cats = cat_ranked[:n_cat_keep]
    keep_cols = new_cats + new_nums          # categoricals first, then numericals
    new_df = df[keep_cols].copy()
    if target in df.columns:
        new_df[target] = df[target].values
    LOGGER.info(
        "Feature selection: %d features (%d num + %d cat) → %d kept "
        "(%d num + %d cat); unsupervised, real columns.",
        feat_count, len(numerical_columns), len(categorical_columns),
        len(keep_cols), len(new_nums), len(new_cats),
    )
    return new_df, new_nums, new_cats


def _label_encode_classification_target(
    df: pd.DataFrame, target: str,
) -> pd.DataFrame:
    """Map the target column to contiguous ``int64`` labels in [0, K-1]."""
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df = df.dropna(subset=[target]).copy()
    df[target] = le.fit_transform(df[target].astype(str)).astype(np.int64)
    return df


def _clip_lgd_target(
    df: pd.DataFrame, target: str, lower: float, upper: float,
) -> pd.DataFrame:
    """Clip LGD target to ``[lower, upper]`` (domain constraint)."""
    df = df.copy()
    if target in df.columns:
        before_below = (df[target] < lower).sum()
        before_above = (df[target] > upper).sum()
        df[target] = df[target].clip(lower, upper)
        if before_below + before_above:
            LOGGER.info(
                "LGD target clip: %d below %g, %d above %g",
                int(before_below), lower, int(before_above), upper,
            )
    return df


# =============================================================================
# Per-dataset orchestrator
# =============================================================================


def sanitize_dataset(
    df: pd.DataFrame,
    dataset_id: str,
    *,
    manifest_row: dict,
    cfg,
) -> tuple[pd.DataFrame, dict]:
    """Apply the (b)–(k) pipeline to one dataset.

    Returns the cleaned DataFrame plus a small log dict tallying what
    was dropped.
    """
    target = manifest_row["target_column"]
    raw_cats = (
        manifest_row["categorical_columns"].split(";")
        if manifest_row["categorical_columns"] else []
    )
    raw_nums = (
        manifest_row["numerical_columns"].split(";")
        if manifest_row["numerical_columns"] else []
    )

    log: dict[str, list[str] | int] = {}
    n_rows_before = len(df)

    # --- (b) exact-duplicate columns (always on — TabPFN doesn't want them) -
    df, log["dropped_duplicate_cols"] = _drop_exact_duplicate_feature_columns(
        df, target,
    )
    # --- (c)/(d) high-missing-rate columns ---------------------------------
    df, log["dropped_high_missing_cols"] = _drop_high_missing_columns(
        df, target, cfg.sanitize.max_missing_rate,
    )

    # Resolve which surviving columns are categorical vs numerical *before*
    # any further transforms touch them.
    surviving = set(df.columns)
    cats = [c for c in raw_cats if c in surviving]
    nums = [c for c in raw_nums if c in surviving]
    extras = [c for c in surviving
              if c != target and c not in cats and c not in nums]
    nums.extend(extras)

    # --- (f) coerce numeric strings ----------------------------------------
    # Done BEFORE the constant-column drop so that columns whose values
    # become all-NaN under pd.to_numeric(errors="coerce") (a column of
    # garbage strings, say) get caught by step (e) and never reach
    # FeatureAgglomeration.
    if cfg.sanitize.coerce_numeric_strings:
        df, log["coerced_numeric_strings"] = _coerce_numeric_strings(
            df, target, cfg.sanitize.coerce_numeric_threshold,
        )

    # --- (g) numerical dtype cast — always float32 (TabPFN default) --------
    df = _cast_numericals_to(df, target, nums, "float32")

    # --- (h) ±inf → NaN — always on (downstream NaN handler cleans up) -----
    df = _replace_inf_with_nan(df, target)

    # --- (e) constant columns (always on — TabPFN's encoders error on them) -
    df, log["dropped_constant_cols"] = _drop_constant_columns(df, target)
    # Refresh column lists after the drop pass.
    surviving = set(df.columns)
    cats = [c for c in cats if c in surviving]
    nums = [c for c in nums if c in surviving]

    # --- (i) feature selection (cap feature count; keep REAL columns) ------
    # Reads cfg.sanitize.feature_selection.{enabled, corr_threshold}; falls
    # back to the legacy cfg.sanitize.agglomeration.enabled flag so old
    # configs still parse.
    fs_cfg = getattr(cfg.sanitize, "feature_selection", None)
    if fs_cfg is not None:
        fs_enabled = bool(getattr(fs_cfg, "enabled", True))
        corr_thr = float(getattr(fs_cfg, "corr_threshold", 0.95))
    else:                                                       # legacy config
        agg = getattr(cfg.sanitize, "agglomeration", None)
        fs_enabled = bool(getattr(agg, "enabled", True)) if agg is not None else True
        corr_thr = 0.95
    if fs_enabled and (len(nums) + len(cats)) > cfg.sanitize.max_columns:
        df, nums, cats = _select_to_max_columns(
            df, target, nums, cats,
            max_columns=cfg.sanitize.max_columns,
            corr_threshold=corr_thr,
            seed=cfg.seed,
        )

    # --- (j) / (k) target handling -----------------------------------------
    # Classification: always re-encode to contiguous int64 (TabPFN requires it).
    if manifest_row["task_type"] == "classification":
        df = _label_encode_classification_target(df, target)
    # LGD: single source of truth for the [0, 1] clip is here (per-dataset
    # surgical fixes don't clip themselves).
    if manifest_row["track"] == "lgd" and cfg.sanitize.lgd_target_clip.enabled:
        df = _clip_lgd_target(
            df, target,
            cfg.sanitize.lgd_target_clip.lower,
            cfg.sanitize.lgd_target_clip.upper,
        )

    log["n_rows_before"] = n_rows_before
    log["n_rows_after"] = len(df)
    log["n_cols_after_features"] = len([c for c in df.columns if c != target])
    log["surviving_categorical_columns"] = cats
    log["surviving_numerical_columns"] = nums
    return df, log


# =============================================================================
# CLI
# =============================================================================


def _load_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.load("config/data.yaml")


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def main(cfg=None) -> int:  # noqa: C901
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if cfg is None:
        cfg = _load_cfg()

    raw_root = resolve_data_path(cfg.paths.raw)
    proc_root = resolve_data_path(cfg.paths.processed)
    manifests = {
        "pd":  _read_manifest(resolve_output_path(cfg.paths.manifest_pd)),
        "lgd": _read_manifest(resolve_output_path(cfg.paths.manifest_lgd)),
    }
    # Only require the manifest(s) for tracks present in the current
    # DATASET_METADATA snapshot to be populated. On subset runs
    # (`scripts/data_pipeline.py --datasets 0001.gmsc`, or the train
    # pipeline's auto-process hook with a missing PD-only dataset)
    # the LGD manifest legitimately stays empty on a fresh checkout —
    # treating that as a hard error would block the subset run that
    # only needs the PD manifest. Required = "track has datasets in
    # the current snapshot AND its manifest is empty".
    tracks_in_scope = {meta["track"] for meta in DATASET_METADATA.values()}
    missing = [t for t in tracks_in_scope if manifests[t].empty]
    if missing:
        LOGGER.error(
            "Manifest(s) empty for track(s) we're about to sanitize: %s. "
            "Run `python -m src.data.register` first.", missing,
        )
        return 1

    failures = 0
    for dataset_id, meta in DATASET_METADATA.items():
        track = meta["track"]
        raw_path = raw_root / track / f"{dataset_id}.csv"
        if not raw_path.exists():
            LOGGER.warning("missing raw file: %s — skipped", raw_path)
            continue
        try:
            mrow = manifests[track]
            row = mrow[mrow["dataset_id"] == dataset_id]
            if row.empty:
                LOGGER.warning("%s: not in manifest, skipping", dataset_id)
                continue
            manifest_row = row.iloc[0].to_dict()

            df = pd.read_csv(raw_path, low_memory=False)
            df = apply_dataset_specific_fixes(df, dataset_id)
            df_clean, log = sanitize_dataset(
                df, dataset_id, manifest_row=manifest_row, cfg=cfg,
            )

            out_dir = proc_root / track
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{dataset_id}.sanitized.csv"
            df_clean.to_csv(out_path, index=False)
            LOGGER.info(
                "%-26s rows=%d→%d cols(features)=%d  → %s",
                dataset_id,
                log["n_rows_before"], log["n_rows_after"],
                log["n_cols_after_features"], out_path,
            )
        except Exception as exc:
            LOGGER.error("%s failed: %s", dataset_id, exc, exc_info=True)
            failures += 1

    return 1 if failures else 0


def _parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Apply dataset-agnostic sanitisation to every raw dataset."
    ).parse_args()


if __name__ == "__main__":
    _parse_args()
    raise SystemExit(main())
