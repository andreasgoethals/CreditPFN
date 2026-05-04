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
  (i) ``FeatureAgglomeration`` to at most ``max_columns`` features
       (default 128); restricted to numerical features. Categoricals
       always pass through. Distances are computed on
       ``StandardScaler``-scaled values; the final output features are
       the *unscaled* per-cluster means.
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
    """
    np_dtype = np.dtype(dtype)
    for col in numerical_columns:
        if col not in df.columns or col == target:
            continue
        # Force float; integer columns become floats (NaN-compatible).
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np_dtype)
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


def _agglomerate_to_max_columns(
    df: pd.DataFrame,
    target: str,
    numerical_columns: list[str],
    categorical_columns: list[str],
    max_columns: int,
    *,
    metric: str,
    linkage: str,
    standardize_for_distance_only: bool,
    seed: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Reduce the feature count to ≤ ``max_columns`` via Ward-linkage
    feature clustering on the numerical columns only.

    Categoricals always pass through. If ``len(categoricals) >=
    max_columns``, no reduction is performed (we cannot drop categorical
    information). Otherwise the numerical bucket is reduced to
    ``max_columns - len(categoricals)`` clusters; the output features
    are per-cluster means in the *unscaled* original space (``standardize_
    for_distance_only=True``).

    Returns ``(df_reduced, new_numerical_columns, categorical_columns)``.
    """
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.preprocessing import StandardScaler

    feat_count = len(numerical_columns) + len(categorical_columns)
    if feat_count <= max_columns:
        return df, numerical_columns, categorical_columns

    target_numerical_count = max_columns - len(categorical_columns)
    if target_numerical_count <= 0:
        LOGGER.warning(
            "FeatureAgglomeration skipped: %d categorical features already "
            "exceed max_columns=%d", len(categorical_columns), max_columns,
        )
        return df, numerical_columns, categorical_columns

    # Build the numerical block; impute NaN with column mean *only* for
    # the distance computation. The output features are means of the
    # unscaled, original (NaN-bearing) columns. The fillna(0) afterwards
    # is a defensive fallback: a column whose values are all NaN at this
    # point (which step (c) should have dropped, but float-coercion in
    # step (g) can produce in edge cases) would otherwise still feed
    # NaNs into FeatureAgglomeration, which sklearn rejects.
    raw_block = df[numerical_columns].astype(np.float64)
    col_means = raw_block.mean(numeric_only=True)
    imputed = raw_block.fillna(col_means).fillna(0.0)
    if standardize_for_distance_only:
        scaler = StandardScaler()
        cluster_input = scaler.fit_transform(imputed.to_numpy())
    else:
        cluster_input = imputed.to_numpy()

    fa = FeatureAgglomeration(
        n_clusters=target_numerical_count,
        metric=metric,
        linkage=linkage,
    )
    fa.fit(cluster_input)
    labels = fa.labels_

    # Output features = unscaled per-cluster means.
    new_numerical_columns: list[str] = []
    out_cols: dict[str, pd.Series] = {}
    for k in range(target_numerical_count):
        members = [numerical_columns[i] for i in range(len(numerical_columns))
                   if labels[i] == k]
        if not members:
            continue
        new_name = f"feat_agglo_{k:04d}"
        out_cols[new_name] = raw_block[members].mean(axis=1)
        new_numerical_columns.append(new_name)

    # Stitch: categoricals (untouched) + agglomerated numericals + target.
    keep_cats = df[categorical_columns].copy()
    new_df = pd.concat([keep_cats, pd.DataFrame(out_cols, index=df.index)], axis=1)
    if target in df.columns:
        new_df[target] = df[target].values
    LOGGER.info(
        "FeatureAgglomeration: %d numerical → %d clusters (cats kept: %d)",
        len(numerical_columns), len(new_numerical_columns), len(categorical_columns),
    )
    return new_df, new_numerical_columns, categorical_columns


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

    # --- (b) exact-duplicate columns ----------------------------------------
    if cfg.sanitize.drop_exact_duplicate_columns:
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

    # --- (g) numerical dtype cast ------------------------------------------
    df = _cast_numericals_to(df, target, nums, cfg.sanitize.numeric_dtype)

    # --- (h) ±inf → NaN -----------------------------------------------------
    if cfg.sanitize.replace_inf_with_nan:
        df = _replace_inf_with_nan(df, target)

    # --- (e) constant columns (now that all coercion is done) --------------
    if cfg.sanitize.drop_constant_columns:
        df, log["dropped_constant_cols"] = _drop_constant_columns(df, target)
        # Refresh column lists after the second drop pass.
        surviving = set(df.columns)
        cats = [c for c in cats if c in surviving]
        nums = [c for c in nums if c in surviving]

    # --- (i) FeatureAgglomeration ------------------------------------------
    if cfg.sanitize.agglomeration.enabled and (len(nums) + len(cats)) > cfg.sanitize.max_columns:
        df, nums, cats = _agglomerate_to_max_columns(
            df, target, nums, cats,
            max_columns=cfg.sanitize.max_columns,
            metric=cfg.sanitize.agglomeration.metric,
            linkage=cfg.sanitize.agglomeration.linkage,
            standardize_for_distance_only=cfg.sanitize.agglomeration.standardize_for_distance_only,
            seed=cfg.seed,
        )

    # --- (j) / (k) target handling -----------------------------------------
    if manifest_row["task_type"] == "classification" and \
            cfg.sanitize.classification_target_to_int64_contiguous:
        df = _label_encode_classification_target(df, target)
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

    raw_root = Path(cfg.paths.raw)
    proc_root = Path(cfg.paths.processed)
    manifests = {
        "pd": _read_manifest(Path(cfg.paths.manifest_pd)),
        "lgd": _read_manifest(Path(cfg.paths.manifest_lgd)),
    }
    if any(m.empty for m in manifests.values()):
        LOGGER.error(
            "Manifests are empty. Run `python -m src.data.register` first."
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
