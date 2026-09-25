"""Cross-model benchmark on the held-out test datasets.

For each test dataset (one entry per `dataset_id`) and each model in
the roster, this module:

  1. Loads the processed CSV via
     :func:`src.eval.dataset_loader.load_processed_dataset`.
  2. Runs `cfg.cv.n_folds` cross-validation on the **full** dataset.
     Inside each fold, context and validation are capped at
     the configured per-model row limit (foundation models only —
     see ``cfg.max_rows_per_model``). The held-out test partition is
     never capped; we call `predict_proba(X_test)` once on the full
     test fold and TabPFN-v3's internal row chunking handles it.
     Classical baselines (XGBoost/CatBoost/LogReg/LinReg) bypass the
     cap, fit the full inner training split and predict every test row.
  3. Per outer fold:

         outer:  80% train  /  20% test
         inner:  80% sub-train  /  20% validation     (split of the train fold)

     The validation split is used for:
       * Optuna HPO objective (boosting and linear controls)
       * F1-threshold tuning for classification (PD)
       * Platt/isotonic calibration for classification

     Final metrics are computed on the held-out test fold AT the
     threshold chosen on validation.

  4. Computes a comprehensive metric block per (model × dataset × fold):

         classification → roc_auc, log_loss, pr_auc, brier_score,
                          optimal_threshold, f1, accuracy, precision,
                          recall, specificity, balanced_accuracy,
                          mcc, cohen_kappa
         regression     → rmse, mae, median_ae, mape, r2,
                          explained_variance, pearson_r, spearman_r,
                          neg_nll (TabPFN only)

  5. Persists each model's results to its own CSV under
     ``results/<TRACK>/<method>/<run>_<timestamp>[_<task_tag>].csv``.

The CSV has one row per model × dataset × fold. Pair trained models to
their own bases within folds, then aggregate within dataset before
giving datasets equal weight; report missing/failed cells separately.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from src.eval.dataset_loader import (
    ProcessedDataset, encode_for_model, load_processed_dataset, subsample,
)
from src.eval.metrics import (
    _best_f1_threshold, _binary_ece, _posthoc_calibrated,
    _classification_metrics, _regression_metrics,
)
from src.model.base import ModelHandle
from src.model.tabpfn_models import TabPFNTrained
from src.train.model import load_provenance
from src.utils.paths import resolve_staging_path

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Output row (wide format — one row per model × dataset × fold)
# --------------------------------------------------------------------------- #


@dataclass
class EvalRow:
    """One row of the benchmark CSV.

    All metric columns are filled with NaN where not applicable
    (e.g. classification metrics on a regression model).
    """
    track:           str
    task_type:       str
    model_name:      str
    model_source:    str
    model_path:      str | None
    test_dataset_id: str
    fold_idx:        int

    n_train_rows:    int
    n_val_rows:      int
    n_test_rows:     int

    # Classification metrics (NaN for regression). The block below adds
    # — on top of the original Roc-AUC / log-loss / PR-AUC / F1 /
    # accuracy / precision / recall set — five more discriminator-quality
    # / calibration / class-imbalance-aware columns:
    #
    #   * brier_score      — proper score on positive-class probability
    #                        (lower = better; calibration + sharpness).
    #   * mcc              — Matthews Correlation Coefficient on the
    #                        F1-tuned binary preds (single number that
    #                        handles class imbalance; -1..+1, 0 = chance).
    #   * balanced_accuracy — macro-recall on the F1-tuned preds.
    #   * cohen_kappa      — chance-adjusted accuracy (-1..+1; > 0 ⇒
    #                        agreement above random).
    #   * specificity      — TN / (TN + FP) on the F1-tuned preds; the
    #                        binary "true negative rate" companion to
    #                        recall (= sensitivity / TPR).
    roc_auc:            float = float("nan")
    log_loss:           float = float("nan")
    pr_auc:             float = float("nan")
    brier_score:        float = float("nan")
    ece:                float = float("nan")     # expected calibration error (10-bin, binary)
    # Post-hoc recalibration, fitted on the inner-val split and applied to the
    # test fold WITHOUT retraining the model. Paired with the raw columns
    # above so `ece - ece_platt` is the recoverable share of any calibration
    # damage. NaN for multiclass or when the calibrator cannot be fitted.
    ece_platt:             float = float("nan")
    brier_score_platt:     float = float("nan")
    log_loss_platt:        float = float("nan")
    ece_isotonic:          float = float("nan")
    brier_score_isotonic:  float = float("nan")
    # Thresholds are selected on validation after each calibration mapping.
    f1_platt:              float = float("nan")
    f1_isotonic:           float = float("nan")
    roc_auc_isotonic:      float = float("nan")
    log_loss_isotonic:     float = float("nan")
    optimal_threshold:  float = float("nan")    # max-F1 on inner-val
    f1:                 float = float("nan")
    accuracy:           float = float("nan")
    precision:          float = float("nan")
    recall:             float = float("nan")
    specificity:        float = float("nan")
    balanced_accuracy:  float = float("nan")
    mcc:                float = float("nan")
    cohen_kappa:        float = float("nan")
    f1_at_05: float = float("nan")
    roc_auc_platt: float = float("nan")
    optimal_threshold_platt: float = float("nan")
    optimal_threshold_isotonic: float = float("nan")
    prediction_mean: float = float("nan")
    prediction_std: float = float("nan")
    prediction_entropy: float = float("nan")
    target_prevalence: float = float("nan")
    true_negative: float = float("nan")
    false_positive: float = float("nan")
    false_negative: float = float("nan")
    true_positive: float = float("nan")
    calibration_bins: str = "[]"
    calibration_bins_platt: str = "[]"
    calibration_bins_isotonic: str = "[]"

    # Regression metrics (NaN for classification). On top of RMSE / MAE
    # / R² / neg-NLL the block below adds:
    #
    #   * median_ae         — median absolute error (outlier-robust).
    #   * mape              — mean absolute percentage error in DECIMAL
    #                         units (not ×100; multiply downstream for %).
    #                         NaN where any y_true == 0.
    #   * explained_variance — sklearn's explained_variance_score.
    #   * pearson_r          — linear correlation pred vs target.
    #   * spearman_r         — rank correlation pred vs target.
    rmse:                float = float("nan")
    mae:                 float = float("nan")
    median_ae:           float = float("nan")
    mape:                float = float("nan")
    r2:                  float = float("nan")
    explained_variance:  float = float("nan")
    pearson_r:           float = float("nan")
    spearman_r:          float = float("nan")
    neg_nll:             float = float("nan")    # TabPFN-* only
    mean_pinball_loss: float = float("nan")
    quantile_crossing_fraction: float = float("nan")
    interval_coverage_80: float = float("nan")
    interval_coverage_90: float = float("nan")
    interval_width_80: float = float("nan")
    interval_width_90: float = float("nan")

    elapsed_sec:     float = 0.0
    fit_seconds: float = 0.0
    predict_score_seconds: float = 0.0
    domain: str = "credit"
    split_seed: int = 99
    timestamp:       str = ""
    status:          str = "OK"
    error:           str | None = None
    cache_hit:       bool = False
    cached_elapsed_sec: float = 0.0
    fitted_parameters: str = "{}"
    hpo_trials_requested: int = 0
    hpo_trials_completed: int = 0


# --------------------------------------------------------------------------- #
# Loading the trained-checkpoint roster from the training manifest
# --------------------------------------------------------------------------- #


def load_trained_handles(
    manifest_csv: Path | str,
    *,
    track: Literal["pd", "lgd"],
    device: str = "auto",
    n_estimators: int = 4,
    n_estimators_tabicl: int | None = None,
) -> list[tuple[ModelHandle, object]]:
    """Read the training manifest and build a trained-model handle per
    successful (status=OK, ckpt-on-disk) row — TabPFNTrained or
    TabICLTrained depending on the row's base_checkpoint family."""
    manifest_csv = Path(manifest_csv)
    if not manifest_csv.exists():
        LOGGER.warning("training manifest not found: %s — skipping TabPFN-trained",
                       manifest_csv)
        return []

    df = pd.read_csv(manifest_csv)
    df = df[df["track"] == track]
    # Include both "OK" (freshly trained this run) and "SKIP" (a previous
    # run already produced a valid checkpoint + provenance, so the trial
    # was skipped). Both have a usable checkpoint on disk. EXCLUDE "FAIL"
    # (no checkpoint) and "DIVERGED" (checkpoint exists but the weights
    # collapsed to random — scoring it would just add noise rows).
    df = df[df["final_ckpt_path"].notna() & (df["final_ckpt_path"] != "")]

    # Manifests are append-only across resumptions. A completed task can
    # therefore appear first as OK and later as SKIP when its checkpoint is
    # detected on disk. Without deduplication the same scientific model enters
    # the roster twice (and can even appear once under staging and once under
    # the $VSC_DATA fallback). Checkpoint basenames encode the full trial
    # identity, so retain only the most recent row for each basename.
    if not df.empty:
        df = df.copy()
        df["_checkpoint_basename"] = df["final_ckpt_path"].map(
            lambda p: Path(str(p)).name,
        )
        n_before = len(df)
        df = df.drop_duplicates(subset=["_checkpoint_basename"], keep="last")
        if len(df) != n_before:
            LOGGER.info(
                "Deduplicated %d repeated training-manifest row(s); "
                "using the latest row per checkpoint basename.",
                n_before - len(df),
            )
    df = df[df["status"].isin(["OK", "SKIP"])]

    out: list[tuple[ModelHandle, object]] = []
    task_type = "classification" if track == "pd" else "regression"
    seen_checkpoints = set()
    for _, row in df.iterrows():
        from src.utils.checkpoint_inventory import resolve_checkpoint
        resolved, prov = resolve_checkpoint(str(row["final_ckpt_path"]), track)
        ckpt = str(resolved) if resolved else str(row["final_ckpt_path"])
        if resolved is None:
            LOGGER.warning("trained checkpoint missing on disk: %s — skipped", ckpt)
            continue
        if prov.get("diverged") or resolved in seen_checkpoints:
            continue
        seen_checkpoints.add(resolved)
        # ``use_lora`` column is present on manifests produced after the
        # LoRA tuneable was added; older manifests don't have it. Default
        # missing → False so the eval still runs against legacy manifests.
        use_lora_raw = row.get("use_lora", False)
        if isinstance(use_lora_raw, str):
            use_lora_val = use_lora_raw.strip().lower() in ("true", "1", "yes")
        else:
            use_lora_val = bool(use_lora_raw)
        extra = {
            "trial_identity_sha256": prov.get("trial_identity", {}).get("sha256"),
            "adaptation_mode": prov.get("adaptation_mode"),
            "base_checkpoint":     row["base_checkpoint"],
            "learning_rate":       float(row["learning_rate"]),
            "use_lora":            use_lora_val,
            "seed":                int(row["seed"]),
            # Keeps full_pass vs one_sample checkpoints in separate result
            # dirs (legacy manifests lack the column → "one_sample").
            "epoch_pass_mode":     str(row.get("epoch_pass_mode", "one_sample")),
            # Same reason, for the corpus-size arm (swept since run-8). Legacy
            # manifests lack the column → 0, which produces the old dirname.
            "min_train_rows":      int(row.get("min_train_rows", 0) or 0),
            # Swept from run-9. The manifest has always written this column; it just was
            # never read, so the two anchor arms would have collided in one results dir.
            "l2sp_lambda":         row.get("l2sp_lambda", None),
        }
        # Saved effective values are authoritative after the historical L2-SP
        # collision and after SKIP records with incomplete HP columns.
        hp = prov.get("hyperparameters", {})
        for key in ("l2sp_lambda", "min_train_rows", "epoch_pass_mode", "learning_rate"):
            if key in hp:
                extra[key] = hp[key]
        # Family from the manifest's base_checkpoint column — the trained
        # ckpt filename inherits the base stem, but the base column is the
        # canonical record.
        from src.train.tabicl_compat import model_family
        if model_family(str(row["base_checkpoint"])) == "tabicl":
            from src.model.tabicl_models import TabICLTrained
            model: object = TabICLTrained(
                task_type=task_type, ckpt_path=ckpt,
                device=device,
                n_estimators=int(
                    n_estimators_tabicl if n_estimators_tabicl is not None
                    else n_estimators
                ),
                extra=extra,
            )
            source = "tabicl-trained"
        else:
            model = TabPFNTrained(
                task_type=task_type, ckpt_path=ckpt,
                device=device, n_estimators=n_estimators, extra=extra,
            )
            source = "tabpfn-trained"
        handle = ModelHandle(
            name=model.name, track=track, task_type=task_type,
            source=source, base_path=ckpt, extra=extra,
        )
        out.append((handle, model))
    return out


# --------------------------------------------------------------------------- #
# Test-dataset resolution per model (provenance vs cfg)
# --------------------------------------------------------------------------- #


def resolve_test_datasets(handle: ModelHandle,
                          *, cfg_test_dataset_ids: list[str]) -> list[str]:
    """Decide WHICH test datasets to score this model on.

    For ``tabpfn-trained``: read the checkpoint's
    ``.provenance.json`` sidecar and use its ``test_datasets`` list —
    that's the test set this checkpoint was actually trained against.

    For ``tabpfn-untuned`` and classical baselines: use the cfg-level
    test split (passed in by the caller). Same seed + same fractions
    → same datasets, so this stays consistent across all models.
    """
    if handle.source.endswith("-trained") and handle.base_path:
        prov = load_provenance(handle.base_path)
        if prov and prov.get("test_datasets"):
            if set(prov["test_datasets"]) & set(prov.get("training_datasets", prov.get("train_datasets", []))):
                raise ValueError("Checkpoint provenance overlaps training and test datasets")
            # Guard the PAIRED comparison: a trained checkpoint is scored on its
            # own provenance test set, while untuned/classical use the cfg split.
            # These are identical only when the corpus seed + fractions match. If
            # they diverge, trained-vs-untuned is silently scored on DIFFERENT
            # datasets and the headline delta is not apples-to-apples — warn loud.
            prov_ids = sorted(set(str(x) for x in prov["test_datasets"]))
            from src.data.retention import is_retention
            retention_ids = [d for d in cfg_test_dataset_ids if is_retention(d)]
            cfg_ids = sorted(set(str(x) for x in cfg_test_dataset_ids) - set(retention_ids))
            if cfg_ids and prov_ids != cfg_ids:
                LOGGER.warning(
                    "PAIRING RISK: trained model %s was held out on %s, but the "
                    "current cfg test split is %s. Trained-vs-untuned will be scored "
                    "on DIFFERENT datasets (not a paired comparison). Re-run training "
                    "against the current corpus split, or align config/train.yaml.",
                    handle.name, prov_ids, cfg_ids,
                )
            return list(prov["test_datasets"]) + retention_ids
        raise ValueError(f"Trained checkpoint {handle.name} has no verified test_datasets; "
                         "refusing a potentially contaminated config fallback")
    return list(cfg_test_dataset_ids)


def _regression_strata(y: np.ndarray, *, n_bins: int, min_count: int):
    """Quantile-bin a continuous target so regression CV can be *stratified*.

    LGD targets are strongly bimodal (mass at 0 and 1, sparse interior), so
    plain KFold can hand folds very different 0/1 fractions, inflating
    fold-to-fold RMSE variance and adding noise to the trained-vs-untuned
    delta. Binning y into quantile strata and stratifying on them balances the
    target distribution across folds. Returns integer bin labels, or ``None``
    when binning isn't viable (caller then falls back to plain KFold). The
    split stays symmetric across all models, so it does not bias the comparison.
    """
    import pandas as pd
    y = np.asarray(y, dtype=float)
    if len(y) < n_bins * min_count:
        return None
    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    except Exception:                                              # pragma: no cover
        return None
    bins = np.asarray(bins, dtype=float)
    if np.isnan(bins).any():
        return None
    bins = bins.astype(int)
    if len(np.unique(bins)) < 2 or int(np.bincount(bins).min()) < min_count:
        return None
    return bins


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #


def _make_outer_folds(y: np.ndarray, *, task_type: str,
                      n_folds: int, seed: int):
    """Outer K-fold (per dataset). Yields ``(train_idx, test_idx)``.

    The user contract: each outer fold is **80% train / 20% test**
    (with `n_folds=5`). Stratified for classification and feasible regression
    quantile strata; regression otherwise uses plain KFold.
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    n = len(y)
    if n_folds <= 1 or n_folds > n:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_test = max(1, n // 5)
        yield perm[n_test:], perm[:n_test]
        return
    if task_type == "classification" and len(np.unique(y)) >= 2:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr, te in skf.split(np.zeros(n), y):
            yield tr, te
    else:
        # Regression: stratify on quantile bins of the (bimodal) target when
        # feasible, else plain KFold. Lower-variance LGD estimates; symmetric
        # across models so the trained-vs-untuned comparison stays fair.
        strata = _regression_strata(y, n_bins=n_folds, min_count=n_folds)
        splits = None
        if strata is not None:
            try:
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                splits = list(skf.split(np.zeros(n), strata))
            except Exception:                                      # pragma: no cover
                splits = None
        if splits is None:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits = list(kf.split(np.zeros(n)))
        for tr, te in splits:
            yield tr, te


def _inner_split(train_idx: np.ndarray, y_train: np.ndarray, *,
                 task_type: str, val_fraction: float, seed: int):
    """Inner train/val split for HPO + threshold tuning.

    The user's contract: 20% of the train fold becomes the inner
    validation split (= 16% of the dataset at outer 80/20). The
    remaining 64% is what the model actually fits on.
    """
    from sklearn.model_selection import train_test_split
    if task_type == "classification" and len(np.unique(y_train)) >= 2:
        stratify = y_train
    elif task_type != "classification":
        # Match the outer folds: stratify the inner HPO/val split on target
        # quantile bins so XGBoost/CatBoost tune on a val set whose target
        # distribution matches the sub-train. Safe: the except below falls
        # back to an unstratified split if the bins don't support it.
        stratify = _regression_strata(y_train, n_bins=5, min_count=2)
    else:
        stratify = None
    try:
        sub_tr, sub_va = train_test_split(
            train_idx, test_size=val_fraction,
            random_state=seed, stratify=stratify,
        )
    except ValueError:
        sub_tr, sub_va = train_test_split(
            train_idx, test_size=val_fraction, random_state=seed,
        )
    return sub_tr, sub_va


# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #


_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]")
# Filename schema: ``tabpfn-<version>-<role>-<version>_<variant>``
# where <version> is ``v2.5`` / ``v2.6`` / ``v3`` (and any future
# ``v3.x``). The pre- and post-role version strings always match; we
# only capture once.
_BASE_RE = re.compile(
    r"tabpfn-(?P<v>v\d+(?:\.\d+)?)-(?:classifier|regressor)-v\d+(?:\.\d+)?_(?P<variant>.+)"
)
# TabICLv2 checkpoint stems: ``tabicl-<role>-v<major>-<yyyymmdd>``.
_TABICL_BASE_RE = re.compile(
    r"tabicl-(?:classifier|regressor)-(?P<v>v\d+)-\d+"
)


def _short_base_tag(base_path: str | None) -> str:
    if not base_path:
        return "unknown"
    stem = Path(base_path).stem
    m = _BASE_RE.match(stem)
    if m:
        return f"{m['v']}-{m['variant']}"
    m = _TABICL_BASE_RE.match(stem)
    if m:
        # Track dirs (PD/LGD) already separate classifier from regressor,
        # so the role is deliberately dropped: one tag per generation.
        return f"tabicl-{m['v']}"
    return stem.removeprefix("tabpfn-")


def _method_dirname(handle: ModelHandle) -> str:
    if handle.source == "baseline":
        return handle.name
    if handle.source.endswith("-untuned"):
        return f"{handle.source}__{_short_base_tag(handle.base_path)}"
    extra = handle.extra or {}
    short = _short_base_tag(extra.get("base_checkpoint"))
    lr = extra.get("learning_rate")
    # Modern provenance names the adaptation explicitly. Old records retain their
    # historical directory suffixes so archived evaluations remain discoverable.
    if extra.get("adaptation_mode") == "frozen_backbone":
        lora_tag = "__frozen"
    elif extra.get("use_lora"):
        lora_tag = ("__iclhead" if handle.source.startswith("tabicl")
                    else "__lora")
    else:
        lora_tag = ""
    # full_pass checkpoints get a distinct dir so they don't collide with
    # the one_sample variant of the same (base, lr, lora).
    fp_tag = {"full_pass": "__fullpass", "accumulate": "__accumulate"}.get(
        extra.get("epoch_pass_mode"), "")
    # Corpus-size arm (swept since run-8). Two trials that trained on different corpora
    # are different experiments and must not share a results directory — sharing one
    # would make their scores indistinguishable AND make skip-existing treat the second
    # arm as already scored. Absent / 0 reproduces the pre-run-8 dirname exactly.
    try:
        min_rows = int(extra.get("min_train_rows", 0) or 0)
    except (TypeError, ValueError):
        min_rows = 0
    rows_tag = f"__min{min_rows}" if min_rows else ""
    # Anchor strength, swept from run-9. Without this tag the lambda=0 and lambda=0.003
    # arms of one (base, lr) pair would share a results directory and silently average,
    # which is exactly what happened to the corpus arm in run-8.
    l2 = extra.get("l2sp_lambda", None)
    l2_tag = "" if l2 is None or l2 != l2 else f"__l2sp{float(l2):g}"
    if lr is not None:
        return f"{handle.source}__{short}__lr{lr:.0e}{fp_tag}{rows_tag}{l2_tag}{lora_tag}"
    return f"{handle.source}__{short}{fp_tag}{rows_tag}{lora_tag}"


def _output_path_for(
    handle: ModelHandle, *,
    track: str, run_name: str, timestamp: str,
    base_dir: str | Path,
    per_task_tag: str | None = None,
) -> Path:
    track_dir = "PD" if track == "pd" else "LGD"
    suffix = f"__{per_task_tag}" if per_task_tag else ""
    return (
        resolve_staging_path(base_dir) / track_dir / _method_dirname(handle)
        / f"{run_name}_{timestamp}{suffix}.csv"
    )


# --------------------------------------------------------------------------- #
# Rerun helper: has this (model × dataset) pair already been scored OK?
# --------------------------------------------------------------------------- #
#
# When re-running the eval (e.g. after adding a new trained checkpoint, or
# adding a new test dataset), we don't want to redo the heavy baselines
# (XGBoost / CatBoost Optuna studies, TabPFN-untuned inference) that already
# produced clean rows on disk. A (handle, dataset_id) is considered already
# scored iff some CSV under
#
#     <results_base>/<TRACK>/<method-dirname>/*.csv
#
# contains an OK row for **every fold** of that dataset (so partial-failure
# pairs — where, say, 1/5 folds succeeded — are NOT skipped on rerun and
# the missing folds get retried). The required fold count is taken from
# the caller's `n_folds_required`; when None, any single OK row counts
# (legacy behaviour, used only by the test suite).


def find_existing_results(
    handle: ModelHandle, dataset_id: str, *,
    track: str, results_base_dir: str | Path,
    n_folds_required: int | None = None,
    run_name: str | None = None,
    evaluation_key: str | None = None,
    require_predictions: bool = False,
) -> list[Path]:
    """Return contributing CSVs only when each required fold's latest row is OK.

    Files and rows are read in the same order as the visualization loader. A
    later failed retry invalidates an earlier success for that fold.
    """
    method_dir = (
        resolve_staging_path(results_base_dir)
        / ("PD" if track == "pd" else "LGD")
        / _method_dirname(handle)
    )
    if not method_dir.exists():
        return []

    hits: list[Path] = []
    latest: dict[int | str, str] = {}
    for csv_path in sorted(method_dir.glob("*.csv")):
        if run_name and not csv_path.name.startswith(run_name + "_"):
            continue
        if evaluation_key is not None or require_predictions:
            sidecar = Path(str(csv_path) + ".evaluation.json")
            if not sidecar.is_file():
                continue
            try:
                receipt = json.loads(sidecar.read_text(encoding="utf-8"))
                if evaluation_key is not None and receipt.get(dataset_id) != evaluation_key:
                    continue
                if require_predictions:
                    prediction = receipt.get("__predictions__") or {}
                    name = prediction.get("file", "")
                    if not name or Path(name).name != name:
                        continue
                    artifact = csv_path.parent / name
                    if not artifact.is_file() or artifact.stat().st_size != prediction.get("bytes"):
                        continue
            except (OSError, ValueError, AttributeError, TypeError):
                continue
        checkpoint = getattr(handle, "base_path", None)
        if checkpoint and Path(checkpoint).is_file() and csv_path.stat().st_mtime_ns < Path(checkpoint).stat().st_mtime_ns:
            continue
        statuses = _csv_fold_statuses(csv_path, dataset_id)
        if statuses:
            hits.append(csv_path)
            latest.update(statuses)

    ok_folds = {fold for fold, status in latest.items() if status == "OK"}
    if not ok_folds:
        return []
    if n_folds_required is None:
        return hits
    return hits if set(range(int(n_folds_required))).issubset(ok_folds) else []


def _csv_fold_statuses(csv_path: Path, dataset_id: str) -> dict[int | str, str]:
    folds: dict[int | str, str] = {}
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("test_dataset_id") == dataset_id:
                    try:
                        fold = int(row.get("fold_idx", ""))
                    except (TypeError, ValueError):
                        fold = row.get("fold_idx", "")
                    folds[fold] = row.get("status", "")
    except (OSError, csv.Error):
        return {}
    return folds


#: Fixed cross-family grid for mean pinball loss and 80%/90% interval coverage.
#: Nine quantiles do not establish an accurate full-distribution CRPS.
PRED_QUANTILE_LEVELS: tuple[float, ...] = (
    0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95)


def _predict_quantiles(model, X, levels: tuple[float, ...] = PRED_QUANTILE_LEVELS):
    """Predicted quantiles at `levels`, shape (n_rows, len(levels)), or None.

    Two families, two APIs, one output shape:

      * TabICLv2 exposes a quantile head directly — ``predict(X, output_type="quantiles")``.
      * TabPFN carries a full bar distribution — ``predict(X, output_type="full")`` returns its
        ``criterion`` plus ``logits``, and the criterion's inverse CDF turns those into
        quantiles on the same grid.

    Best-effort and version-tolerant by design: this runs inside a ~20M-credit campaign and
    must never be the reason a fold fails. Every unexpected shape or missing attribute returns
    None, and the caller then stores the point prediction alone, exactly as before.
    """
    import numpy as _np

    # --- TabICLv2 (and anything else with a quantile head) ---
    for kwargs in ({"output_type": "quantiles", "quantiles": list(levels)},
                   {"output_type": "quantiles", "alphas": list(levels)}):
        try:
            out = model.predict(X, **kwargs)
            arr = _np.column_stack(out) if isinstance(out, list) else _np.asarray(out, dtype=float)
        except Exception:                                          # noqa: BLE001
            continue
        if arr.shape == (len(X), len(levels)) and _np.isfinite(arr).all():
            return arr
    # --- TabPFN bar distribution ---
    try:
        import torch
        out = model.predict(X, output_type="full")
        if not isinstance(out, dict):
            return None
        crit, logits = out.get("criterion"), out.get("logits")
        if crit is None or logits is None:
            return None
        logits_t = logits if torch.is_tensor(logits) else torch.as_tensor(logits)
        cols = []
        for q in levels:
            icdf = getattr(crit, "icdf", None) or getattr(crit, "quantile", None)
            if icdf is None:
                return None
            v = icdf(logits_t, q)
            cols.append(_np.asarray(v.detach().cpu(), dtype=float).reshape(-1))
        arr = _np.stack(cols, axis=1)
        return arr if arr.ndim == 2 and arr.shape[1] == len(levels) else None
    except Exception:                                              # noqa: BLE001
        return None


def _write_predictions(
    records: list[dict],
    out_path: Path,
) -> Path | None:
    """Write one row per (dataset, fold, test row): the truth and the prediction.

    Publish atomically. Prefer Parquet; use compressed CSV only when no
    Parquet engine is installed. Storage failures must propagate to the job.
    """
    if not records:
        return None
    import pandas as pd
    df = pd.DataFrame.from_records(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import uuid
    target = out_path.with_suffix(".parquet")
    pending = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            df.to_parquet(pending, index=False)
        except ImportError:
            target = out_path.with_suffix(".csv.gz")
            df.to_csv(pending, index=False, compression="gzip")
        os.replace(pending, target)
        return target
    finally:
        pending.unlink(missing_ok=True)


def _write_csv(rows: list[EvalRow], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    # Write ATOMICALLY: many eval array tasks run concurrently and the
    # skip-existing scan (`find_existing_results`) reads every CSV in the
    # method dir. A plain open("w") leaves a window where another task can
    # read a half-written file (header but only some fold rows) and wrongly
    # conclude the pair is incomplete -> duplicate re-scoring. Writing to a
    # unique temp file and os.replace()-ing it in means readers only ever
    # see a complete file. (Bug fixed 2026-06-23.)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:                                        # pragma: no cover
                pass


# --------------------------------------------------------------------------- #
# Per-model bench loop
# --------------------------------------------------------------------------- #


def _bench_model_on_dataset(
    *,
    handle: ModelHandle, model,
    ds: ProcessedDataset,
    n_folds: int, inner_val_fraction: float,
    seed: int,
    timestamp: str,
    max_rows_for_handle: int | None = None,
    # Predictions are appended to a caller-owned list rather than returned, so the fold loop
    # stays a generator of metric rows and nothing has to change shape.
    pred_records: list[dict] | None = None,
) -> list[EvalRow]:
    """Run K-fold CV (with inner train/val split) of one model on one
    test dataset and return the per-fold rows.

    Subsampling policy
    ------------------
      * Outer K-fold runs on the **full** dataset — no pre-cap.
      * Inside each fold, if a per-model row-cap applies (TabPFN
        family only — see :func:`resolve_max_rows_for_handle`), the
        **train + inner-val** partitions are subsampled to that cap.
        The **test fold is NEVER capped** — we predict on the
        entirety of the held-out rows in one ``predict_proba`` /
        ``predict`` call, which TabPFN-v3 handles via its own
        ``inference_row_chunk_size`` machinery
        (``tfm-library/repositories/TabPFN .txt``).
      * Classical baselines (``max_rows_for_handle=None``) see the
        full train + val partitions and predict on the full test
        partition.
    """
    rows: list[EvalRow] = []
    X_full, y_full = ds.X, ds.y

    for fold_idx, (tr_idx, te_idx) in enumerate(_make_outer_folds(
        y_full, task_type=ds.task_type, n_folds=n_folds, seed=seed,
    )):
        # Inner split of the train fold for HPO + threshold tuning.
        sub_tr, sub_va = _inner_split(
            tr_idx, y_full[tr_idx],
            task_type=ds.task_type, val_fraction=inner_val_fraction,
            seed=seed + fold_idx,
        )

        X_tr_df = X_full.iloc[sub_tr].reset_index(drop=True)
        X_va_df = X_full.iloc[sub_va].reset_index(drop=True)
        X_te_df = X_full.iloc[te_idx].reset_index(drop=True)
        y_tr = y_full[sub_tr]
        y_va = y_full[sub_va]
        y_te = y_full[te_idx]

        # Per-model architectural cap — applied to train + val only.
        # The test partition stays at full size: TabPFN-v3 handles
        # arbitrarily large test sets via its internal row-chunked
        # inference path.
        if max_rows_for_handle is not None:
            if len(X_tr_df) > max_rows_for_handle:
                LOGGER.info(
                    "  ↳ %s cap: train %d → %d rows (architectural limit, fold %d)",
                    handle.name, len(X_tr_df), max_rows_for_handle, fold_idx,
                )
                X_tr_df, y_tr = _subsample_train(
                    X_tr_df, y_tr,
                    max_rows=max_rows_for_handle,
                    seed=seed + 1000 + fold_idx,
                    task_type=ds.task_type,
                )
            # Validation set should be small (~16% of dataset post 80/20
            # outer + 20% inner) but cap it too if a tiny `max_rows`
            # somehow yields a val split larger than the cap.
            if len(X_va_df) > max_rows_for_handle:
                X_va_df, y_va = _subsample_train(
                    X_va_df, y_va,
                    max_rows=max_rows_for_handle,
                    seed=seed + 2000 + fold_idx,
                    task_type=ds.task_type,
                )

        X_tr_arr, X_va_arr, X_te_arr, cat_idx = encode_for_model(
            X_tr_df, X_va_df, X_te_df,
            categorical_columns=ds.categorical_columns,
        )

        t0 = time.monotonic()
        status = "OK"
        error: str | None = None
        metrics: dict[str, float] = {}
        fit_seconds = 0.0
        quantiles = None

        try:
            # Pass the val split through to the model. Wrappers that do
            # HPO (XGBoost/CatBoost) use it as the Optuna objective.
            # Wrappers without HPO ignore the args.
            model.fit(X_tr_arr, y_tr, cat_idx,
                      X_val=X_va_arr, y_val=y_va)
            fit_seconds = time.monotonic() - t0
            if ds.task_type == "classification":
                proba_va = np.asarray(model.predict_proba(X_va_arr))
                proba_te = np.asarray(model.predict_proba(X_te_arr))
                # Wrappers return columns in encoded-label order. Missing
                # columns do not reveal which labels they represent; padding
                # them could silently score the opposite class as positive.
                classes = np.unique(y_tr)
                K_total = len(classes)
                if K_total < 2 or not np.array_equal(classes, np.arange(K_total)):
                    raise ValueError("Training labels cannot establish probability column order")
                for probabilities, n_rows in ((proba_va, len(y_va)), (proba_te, len(y_te))):
                    if probabilities.shape != (n_rows, K_total):
                        raise ValueError("Unexpected probability matrix shape for training classes")
                    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                            or (probabilities > 1).any()):
                        raise ValueError("Invalid class probability values")
                pred_te = proba_te[:, 1] if K_total == 2 else proba_te.argmax(axis=1)
                metrics = _classification_metrics(
                    proba_test=proba_te, y_test=y_te,
                    proba_val=proba_va,  y_val=y_va,
                    n_classes_seen=K_total,
                )
            else:
                distribution_fn = getattr(model, "predict_distribution", None)
                if callable(distribution_fn):
                    # Both foundation families expose point and requested quantile
                    # outputs in one ensemble forward. TabPFN also reuses its logits
                    # for density scoring. An incompatible API fails this fold visibly.
                    output = distribution_fn(X_te_arr, y_te, PRED_QUANTILE_LEVELS)
                    pred_te = np.asarray(output["mean"]).reshape(-1)
                    quantiles = np.asarray(output["quantiles"])
                    neg_nll = output.get("neg_nll")
                else:
                    pred_te = np.asarray(model.predict(X_te_arr)).reshape(-1)
                    nll_fn = getattr(model, "neg_log_likelihood", None)
                    neg_nll = nll_fn(X_te_arr, y_te) if callable(nll_fn) else None
                    quantiles = _predict_quantiles(model, X_te_arr)
                metrics = _regression_metrics(
                    pred_test=pred_te, y_test=y_te, neg_nll=neg_nll,
                    quantiles=quantiles, quantile_levels=PRED_QUANTILE_LEVELS,
                )
        except Exception as exc:                              # noqa: BLE001
            status = "FAIL"
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning(
                "  ↳ %s/%s fold %d FAIL: %s",
                ds.dataset_id, handle.name, fold_idx, error,
            )
            LOGGER.debug("traceback:\n%s", traceback.format_exc())

        # PREDICTIONS, kept so a future metric or a calibration study does not need the GPU
        # again. `pred_te` is the positive-class probability for classification and the point
        # prediction for regression; both are one float per test row.
        if pred_records is not None and status == "OK":
            try:
                _p = np.asarray(pred_te).reshape(-1)
                _y = np.asarray(y_te).reshape(-1)
                if _p.shape != _y.shape:
                    raise ValueError("Prediction and test-target shapes differ")
                fold_predictions = []
                # Regression only: capture the predictive DISTRIBUTION as a fixed quantile
                # grid for interval coverage and mean pinball loss. A sparse quantile grid
                # is not an exact CRPS calculation; models without quantiles store points only.
                _q = quantiles
                for i in range(_y.size):
                    _rec = {"test_dataset_id": ds.dataset_id, "model_name": handle.name,
                            "fold_idx": int(fold_idx), "row_idx": int(te_idx[i]),
                            "y_true": float(_y[i]), "y_pred": float(_p[i])}
                    if _q is not None:
                        _rec.update({f"q{int(round(lv * 100)):02d}": float(_q[i, j])
                                     for j, lv in enumerate(PRED_QUANTILE_LEVELS)})
                    fold_predictions.append(_rec)
                pred_records.extend(fold_predictions)
            except Exception as exc:
                status = "FAIL"
                error = f"Prediction recording failed: {type(exc).__name__}: {exc}"
                LOGGER.warning("%s/%s fold %d: %s", ds.dataset_id, handle.name, fold_idx, error)

        rows.append(EvalRow(
            track=ds.track,
            task_type=ds.task_type,
            model_name=handle.name,
            model_source=handle.source,
            model_path=handle.base_path,
            test_dataset_id=ds.dataset_id,
            fold_idx=fold_idx,
            # Use the *post-cap* sizes so the CSV records what the model
            # actually saw, not the pre-cap fold size. When a TabPFN model
            # hits its per-architecture cap (cfg.max_rows_per_model), the
            # earlier `_subsample_train` call shrinks X_tr_df / X_va_df —
            # `sub_tr` / `sub_va` are the pre-cap index arrays and would
            # over-state the training-context size. Reported by Codex
            # on 2026-05-21.
            n_train_rows=int(len(X_tr_df)),
            n_val_rows=int(len(X_va_df)),
            n_test_rows=int(len(te_idx)),
            elapsed_sec=time.monotonic() - t0,
            fit_seconds=fit_seconds,
            predict_score_seconds=time.monotonic() - t0 - fit_seconds,
            split_seed=seed,
            domain="noncredit" if ds.dataset_id.startswith(("openml_", "package_")) else "credit",
            timestamp=timestamp,
            status=status, error=error,
            fitted_parameters=json.dumps({**getattr(model, "_params", {}),
                **(getattr(model, "best_params", None) or {})}, sort_keys=True, default=str),
            hpo_trials_requested=int(getattr(model, "_hpo_trials", 0)),
            hpo_trials_completed=int(getattr(model, "hpo_trials_completed", 0)),
            **{k: v for k, v in metrics.items() if k in EvalRow.__dataclass_fields__},
        ))

    return rows


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def resolve_max_rows_for_handle(
    handle: ModelHandle, *,
    max_rows_per_model: dict[str, int] | None,
) -> int | None:
    """Look up the architectural row-cap for one handle.

    The cap applies to foundation models (in-context learning
    has a hard memory budget tied to the training-context size). For
    classical baselines (XGBoost / CatBoost / LogReg / LinReg) the
    cap is unset — they see the full training fold.

    Lookup key: the leading ``v<MAJOR>[.<MINOR>]`` of the base-stem,
    e.g. ``v3-default`` → look up ``"v3"`` then ``"v3-default"`` then
    fall back to ``"default"``. So one entry per generation is enough.

    Parameters
    ----------
    max_rows_per_model
        ``cfg.max_rows_per_model`` from ``config/eval.yaml`` (or any
        equivalent dict). ``None`` disables the cap entirely.

    Returns
    -------
    The per-model row-cap, or ``None`` if no cap applies. ``None``
    also means "no cap" for classical baselines.
    """
    if not max_rows_per_model:
        return None
    if handle.source == "baseline":
        return None

    # Most-specific to least-specific key lookup. Both -trained and
    # -untuned handles resolve through _short_base_tag so the SAME key is
    # produced for the same base generation. (Bug fixed 2026-08-04: the
    # untuned branch previously stripped a "tabpfn-untuned__" prefix from
    # handle.name, but untuned names use brackets — the strip never
    # matched, every untuned model fell through to the "default" cap, and
    # untuned-v3 was evaluated with 50k-row folds while trained-v3 got
    # 1M. Asymmetric caps silently un-pair the headline comparison on
    # datasets larger than the default cap.)
    if handle.source.endswith("-trained"):
        base_short = _short_base_tag((handle.extra or {}).get("base_checkpoint"))
    else:
        base_short = _short_base_tag(handle.base_path)
    base_short = base_short or ""
    candidates: list[str] = [base_short]
    # Strip variant suffix: "v3-default" → "v3"
    if "-" in base_short:
        candidates.append(base_short.split("-", 1)[0])
    candidates.append("default")

    for key in candidates:
        if key in max_rows_per_model:
            return int(max_rows_per_model[key])
    return None


def _subsample_train(
    X_df: pd.DataFrame, y: np.ndarray,
    *,
    max_rows: int | None,
    seed: int,
    task_type: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Cap a (train-partition) DataFrame to ``max_rows`` rows.

    The test partition is NEVER touched by this function — the user
    contract is "use the same training data of each fold to make
    predictions on the entirety of the test set" (chat 2026-05-20).
    """
    if max_rows is None or len(X_df) <= max_rows:
        return X_df, y
    X_cap, y_cap = subsample(
        X_df, y,
        max_rows=max_rows, seed=seed,
        stratify=(task_type == "classification"),
    )
    return X_cap, y_cap


def run_benchmark(
    *,
    test_dataset_ids: list[str],          # one entry per dataset (NOT per chunk)
    handles_and_models: Iterable[tuple[ModelHandle, object]],
    track: Literal["pd", "lgd"],
    run_name: str,
    n_folds: int = 5,
    inner_val_fraction: float = 0.20,
    seed: int = 42,
    results_base_dir: str | Path = "results",
    max_rows_per_model: dict[str, int] | None = None,
    per_task_tag: str | None = None,
    save_predictions: bool = False,
    evaluation_config: dict | None = None,
    use_control_cache: bool = False,
) -> list[EvalRow]:
    """Score every (model × test_dataset × fold) and persist per-method CSVs.

    Failures inside one cell don't stop the loop — they're recorded
    with ``status="FAIL"`` so the comparison table is robust to a
    single bad cell.

    ``max_rows_per_model`` supplies architecture-specific context caps
    from the evaluation configuration, looked up by base-stem key.
    Applied to the context and validation partitions — the test fold
    is **always full** and predict_proba is called on it in one go.
    Classical baselines bypass the cap entirely.
    """
    handles_and_models = list(handles_and_models)
    # One record per (dataset, model, fold, test row). Accumulated across the whole benchmark
    # and filtered per model at write time, so a model's predictions land beside its metrics.
    pred_records: list[dict] = []
    if not test_dataset_ids:
        LOGGER.warning("test_dataset_ids is empty — nothing to benchmark.")
        return []
    if not handles_and_models:
        LOGGER.warning("handles_and_models is empty — nothing to benchmark.")
        return []

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    track_label = "PD" if track == "pd" else "LGD"
    LOGGER.info(
        "Benchmark: track=%s, %d models × %d datasets × %d folds = %d cells, "
        "run_name=%s, timestamp=%s",
        track_label, len(handles_and_models), len(test_dataset_ids), n_folds,
        len(handles_and_models) * len(test_dataset_ids) * n_folds,
        run_name, timestamp,
    )

    # Pre-load datasets once to amortise disk I/O across models. There is
    # no pre-cap step: the per-fold logic caps only the training partition,
    # leaving the test partition full so we predict on all held-out rows.
    datasets_full: dict[str, ProcessedDataset] = {}
    for did in test_dataset_ids:
        try:
            datasets_full[did] = load_processed_dataset(track=track, dataset_id=did)
        except (FileNotFoundError, KeyError) as exc:
            raise RuntimeError(f"Required evaluation dataset unavailable: {did}") from exc

    rows: list[EvalRow] = []
    rows_by_model: dict[str, list[EvalRow]] = {}

    for m_idx, (handle, model) in enumerate(handles_and_models, start=1):
        evaluation_keys = {}
        rows_by_model.setdefault(handle.name, [])
        max_rows_for_handle = resolve_max_rows_for_handle(
            handle, max_rows_per_model=max_rows_per_model,
        )
        LOGGER.info(
            "model %d/%d  %s  (source=%s, cap=%s)",
            m_idx, len(handles_and_models), handle.name, handle.source,
            "none" if max_rows_for_handle is None else f"{max_rows_for_handle:,}",
        )

        for did, ds in datasets_full.items():
            LOGGER.info("  dataset %s  (n_rows=%d, n_features=%d)",
                        did, ds.n_rows, ds.n_features)
            from src.eval import cache
            key = cache.evaluation_key(handle, did, track=track, config=evaluation_config) if evaluation_config else None
            if key:
                evaluation_keys[did] = key
            cacheable = key and use_control_cache and not handle.source.endswith("-trained")
            cached = cache.load(results_base_dir, key, n_folds=n_folds) if cacheable else None
            cached_predictions = cache.load_predictions(results_base_dir, key) if cached is not None and save_predictions else None
            if save_predictions and (cached_predictions is None
                    or not cache.predictions_complete(cached, cached_predictions)):
                cached = None
            if cached is not None:
                from dataclasses import replace
                fold_rows = [replace(EvalRow(**r), elapsed_sec=0.0, timestamp=timestamp,
                                     cache_hit=True, cached_elapsed_sec=float(r["elapsed_sec"])) for r in cached]
                LOGGER.info("Reused fingerprint-matched control for %s", did)
                if cached_predictions is not None:
                    pred_records.extend(cached_predictions)
            else:
                prediction_start = len(pred_records)
                fold_rows = _bench_model_on_dataset(
                    handle=handle, model=model, ds=ds,
                    n_folds=n_folds, inner_val_fraction=inner_val_fraction,
                    seed=seed, timestamp=timestamp,
                    max_rows_for_handle=max_rows_for_handle,
                    pred_records=pred_records if save_predictions else None,
                )
                if cacheable:
                    cache.save(results_base_dir, key, fold_rows, n_folds=n_folds,
                               predictions=pred_records[prediction_start:] if save_predictions else None)
            rows.extend(fold_rows)
            rows_by_model[handle.name].extend(fold_rows)

            # One-line per (model × dataset) result summary in the log, so a
            # shared eval log shows the actual scores (mean over OK folds) and
            # any fold failures at a glance — no need to open the CSVs.
            ok_rows = [r for r in fold_rows if r.status == "OK"]
            n_fail_ds = len(fold_rows) - len(ok_rows)

            def _mean(attr):
                vals = [getattr(r, attr) for r in ok_rows]
                vals = [v for v in vals if v == v]          # drop NaN (NaN != NaN)
                return sum(vals) / len(vals) if vals else None

            if track == "pd":
                # Surface calibration (ECE/Brier) next to discrimination (AUC):
                # continued-pretraining can silently DEGRADE calibration while
                # AUC holds — the Basel-relevant property must be visible in the
                # shared log, not just the CSV.
                _au, _f1 = _mean("roc_auc"), _mean("f1")
                _ece, _brier = _mean("ece"), _mean("brier_score")
                _summ = (f"roc_auc={_au:.4f}" if _au is not None else "roc_auc=n/a") + \
                        (f" f1={_f1:.4f}" if _f1 is not None else "") + \
                        (f" ece={_ece:.4f}" if _ece is not None else "") + \
                        (f" brier={_brier:.4f}" if _brier is not None else "")
            else:
                _rmse, _r2, _nnll = _mean("rmse"), _mean("r2"), _mean("neg_nll")
                _summ = (f"rmse={_rmse:.4f}" if _rmse is not None else "rmse=n/a") + \
                        (f" r2={_r2:.4f}" if _r2 is not None else "") + \
                        (f" neg_nll={_nnll:.4f}" if _nnll is not None else "")
            LOGGER.info(
                "    ↳ %s × %s: %d/%d folds OK  %s%s",
                handle.name, did, len(ok_rows), len(fold_rows), _summ,
                f"  FAILED={n_fail_ds}" if n_fail_ds else "",
            )

        # Persist this model's rows. New file per (run_name, timestamp,
        # task_tag), so concurrent slurm tasks never write to the same file.
        out_path = _output_path_for(
            handle, track=track, run_name=run_name, timestamp=timestamp,
            base_dir=results_base_dir, per_task_tag=per_task_tag,
        )
        marker = Path(str(out_path) + ".evaluation.json")
        marker.unlink(missing_ok=True)
        _write_csv(rows_by_model[handle.name], out_path)
        if save_predictions:
            model_predictions = [r for r in pred_records if r["model_name"] == handle.name]
            if not cache.predictions_complete(rows_by_model[handle.name], model_predictions):
                raise RuntimeError("Incomplete predictions for successful evaluation folds")
            _pred_path = _write_predictions(model_predictions, out_path)
            if _pred_path is not None:
                evaluation_keys["__predictions__"] = {"file": _pred_path.name, "bytes": _pred_path.stat().st_size}
                LOGGER.info("  → wrote predictions to %s", _pred_path)
        # This receipt is the commit point: never publish it before required artifacts.
        if evaluation_keys:
            from src.utils.atomic import write_json
            write_json(marker, evaluation_keys)
        # Total wall-clock this model spent across all its folds/datasets —
        # surfaces which methods dominate eval cost (typically the XGBoost /
        # CatBoost per-fold Optuna HPO), so the eval log alone shows where to
        # optimize the benchmark stage.
        model_elapsed = sum(
            float(r.elapsed_sec) for r in rows_by_model[handle.name]
            if r.elapsed_sec == r.elapsed_sec   # drop NaN
        )
        LOGGER.info("  → wrote %d rows to %s  (model elapsed=%.1fs)",
                    len(rows_by_model[handle.name]), out_path, model_elapsed)

    return rows
