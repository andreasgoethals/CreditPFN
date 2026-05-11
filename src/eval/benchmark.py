"""Cross-model benchmark on the held-out test datasets.

For each test dataset (one entry per `dataset_id`, NOT per chunk) and
each model in the roster, this module:

  1. Loads the processed CSV via
     :func:`src.eval.dataset_loader.load_processed_dataset` — i.e. the
     full sanitised dataset, NOT a 20k-row chunk. The chunk format
     is a TabPFN-training artefact and the wrong granularity here.
  2. For TabPFN-* models ONLY: subsamples to `cfg.max_rows_tabpfn`
     (architectural cap; 100k by default on H100). Non-TabPFN
     baselines see the full dataset — they can train on millions
     of rows and pre-capping them would be misleading.
  3. Runs `cfg.cv.n_folds` cross-validation. Per outer fold:

         outer:  80% train  /  20% test
         inner:  80% sub-train  /  20% validation     (split of the train fold)

     The validation split is used for:
       * Optuna HPO objective (XGBoost / CatBoost)
       * F1-threshold tuning for classification (PD)

     Final metrics are computed on the held-out test fold AT the
     threshold chosen on validation.

  4. Computes a comprehensive metric block per (model × dataset × fold):

         classification → roc_auc, log_loss, pr_auc,
                          optimal_threshold, f1, accuracy,
                          precision, recall
         regression     → rmse, mae, r2, neg_nll (TabPFN only)

  5. Persists each model's results to its own CSV under
     ``results/<TRACK>/<method>/<run>_<timestamp>[_<task_tag>].csv``.

The CSV is wide-format (one row per model × dataset × fold, all
metric columns side-by-side); reviewers find this much easier to
aggregate than the previous long format. Aggregation:

    pd.read_csv(...).groupby(["model_name"])[["roc_auc", "f1"]].agg(["mean", "std"])
"""

from __future__ import annotations

import csv
import datetime as _dt
import logging
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from src.eval.dataset_loader import (
    ProcessedDataset, encode_for_model, load_processed_dataset, subsample,
)
from src.model.base import ModelHandle
from src.model.tabpfn_models import TabPFNTrained
from src.train.model import load_provenance
from src.utils.paths import resolve_output_path

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

    # Classification metrics (NaN for regression).
    roc_auc:            float = float("nan")
    log_loss:           float = float("nan")
    pr_auc:             float = float("nan")
    optimal_threshold:  float = float("nan")    # max-F1 on inner-val
    f1:                 float = float("nan")
    accuracy:           float = float("nan")
    precision:          float = float("nan")
    recall:             float = float("nan")

    # Regression metrics (NaN for classification).
    rmse:               float = float("nan")
    mae:                float = float("nan")
    r2:                 float = float("nan")
    neg_nll:            float = float("nan")    # TabPFN-* only

    elapsed_sec:     float = 0.0
    timestamp:       str = ""
    status:          str = "OK"
    error:           str | None = None


# --------------------------------------------------------------------------- #
# Loading the trained-checkpoint roster from the training manifest
# --------------------------------------------------------------------------- #


def load_trained_handles(
    manifest_csv: Path | str,
    *,
    track: Literal["pd", "lgd"],
    device: str = "auto",
    n_estimators: int = 4,
) -> list[tuple[ModelHandle, TabPFNTrained]]:
    """Read the training manifest and build a TabPFN-trained handle per
    successful (status=OK, ckpt-on-disk) row."""
    manifest_csv = Path(manifest_csv)
    if not manifest_csv.exists():
        LOGGER.warning("training manifest not found: %s — skipping TabPFN-trained",
                       manifest_csv)
        return []

    df = pd.read_csv(manifest_csv)
    df = df[df["track"] == track]
    df = df[df["status"] == "OK"]
    df = df[df["final_ckpt_path"].notna() & (df["final_ckpt_path"] != "")]

    out: list[tuple[ModelHandle, TabPFNTrained]] = []
    task_type = "classification" if track == "pd" else "regression"
    for _, row in df.iterrows():
        ckpt = str(row["final_ckpt_path"])
        if not Path(ckpt).exists():
            LOGGER.warning("trained checkpoint missing on disk: %s — skipped", ckpt)
            continue
        extra = {
            "base_checkpoint":     row["base_checkpoint"],
            "learning_rate":       float(row["learning_rate"]),
            "seed":                int(row["seed"]),
        }
        model = TabPFNTrained(
            task_type=task_type, ckpt_path=ckpt,
            device=device, n_estimators=n_estimators, extra=extra,
        )
        handle = ModelHandle(
            name=model.name, track=track, task_type=task_type,
            source="tabpfn-trained", base_path=ckpt, extra=extra,
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
    if handle.source == "tabpfn-trained" and handle.base_path:
        prov = load_provenance(handle.base_path)
        if prov and prov.get("test_datasets"):
            return list(prov["test_datasets"])
        LOGGER.warning(
            "tabpfn-trained %s has no test_datasets in provenance — "
            "falling back to cfg test split",
            handle.name,
        )
    return list(cfg_test_dataset_ids)


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #


def _make_outer_folds(y: np.ndarray, *, task_type: str,
                      n_folds: int, seed: int):
    """Outer K-fold (per dataset). Yields ``(train_idx, test_idx)``.

    The user contract: each outer fold is **80% train / 20% test**
    (with `n_folds=5`). Stratified for classification, plain KFold
    for regression.
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
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr, te in kf.split(np.zeros(n)):
            yield tr, te


def _inner_split(train_idx: np.ndarray, y_train: np.ndarray, *,
                 task_type: str, val_fraction: float, seed: int):
    """Inner train/val split for HPO + threshold tuning.

    The user's contract: 20% of the train fold becomes the inner
    validation split (= 16% of the dataset at outer 80/20). The
    remaining 64% is what the model actually fits on.
    """
    from sklearn.model_selection import train_test_split
    stratify = y_train if (
        task_type == "classification" and len(np.unique(y_train)) >= 2
    ) else None
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
# Metric computation
# --------------------------------------------------------------------------- #


def _best_f1_threshold(
    proba_val_pos: np.ndarray, y_val: np.ndarray,
) -> float:
    """Return the threshold τ ∈ [0,1] that maximises F1 on the val set.

    Uses ``sklearn.metrics.precision_recall_curve`` so we evaluate F1
    only at the O(n) breakpoints sklearn returns (sorted by predicted
    score). The old "np.unique over all probas" approach was O(n²) on
    large val sets — Gemini's #3 bottleneck.
    """
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(
        y_val, proba_val_pos,
    )
    # precision_recall_curve returns one fewer threshold than (p, r);
    # match them by dropping the final p/r pair (which has threshold = ∞).
    p, r = precisions[:-1], recalls[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * p * r / (p + r)
        f1 = np.nan_to_num(f1, nan=0.0)
    if len(thresholds) == 0 or f1.max() <= 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def _classification_metrics(
    proba_test: np.ndarray, y_test: np.ndarray,
    proba_val: np.ndarray, y_val: np.ndarray,
    n_classes_seen: int,
) -> dict[str, float]:
    """All classification metrics in one place. F1 / accuracy /
    precision / recall use the threshold that MAXIMISES F1 on the
    inner-validation split (binary only); multiclass returns NaN
    for those four columns.
    """
    from sklearn.metrics import (
        roc_auc_score, log_loss, average_precision_score,
        f1_score, accuracy_score, precision_score, recall_score,
    )
    out: dict[str, float] = {}
    K = proba_test.shape[1]

    # Threshold-free metrics on test fold.
    try:
        if K == 2:
            out["roc_auc"] = float(roc_auc_score(y_test, proba_test[:, 1]))
            out["pr_auc"]  = float(average_precision_score(y_test, proba_test[:, 1]))
        elif n_classes_seen >= 2:
            out["roc_auc"] = float(
                roc_auc_score(y_test, proba_test, multi_class="ovr", average="macro")
            )
            out["pr_auc"]  = float("nan")
        else:
            out["roc_auc"] = float("nan")
            out["pr_auc"]  = float("nan")
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"]  = float("nan")
    try:
        out["log_loss"] = float(
            log_loss(y_test, proba_test, labels=list(range(K)))
        )
    except ValueError:
        out["log_loss"] = float("nan")

    # Threshold-tuned metrics — binary only.
    if K == 2 and len(np.unique(y_val)) >= 2:
        best_th = _best_f1_threshold(proba_val[:, 1], y_val)
        out["optimal_threshold"] = best_th
        preds_t = (proba_test[:, 1] >= best_th).astype(int)
        out["f1"]        = float(f1_score(y_test, preds_t, zero_division=0))
        out["accuracy"]  = float(accuracy_score(y_test, preds_t))
        out["precision"] = float(precision_score(y_test, preds_t, zero_division=0))
        out["recall"]    = float(recall_score(y_test, preds_t, zero_division=0))
    else:
        for k in ("optimal_threshold", "f1", "accuracy", "precision", "recall"):
            out[k] = float("nan")

    return out


def _regression_metrics(
    pred_test: np.ndarray, y_test: np.ndarray,
    *, neg_nll: float | None,
) -> dict[str, float]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        "rmse":    float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "mae":     float(mean_absolute_error(y_test, pred_test)),
        "r2":      float(r2_score(y_test, pred_test)),
        "neg_nll": float("nan") if neg_nll is None else float(neg_nll),
    }


# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #


_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]")
_BASE_RE = re.compile(r"tabpfn-(?P<v>v\d\.\d)-(?:classifier|regressor)-v\d\.\d_(?P<variant>.+)")


def _short_base_tag(base_path: str | None) -> str:
    if not base_path:
        return "unknown"
    stem = Path(base_path).stem
    m = _BASE_RE.match(stem)
    if m:
        return f"{m['v']}-{m['variant']}"
    return stem.removeprefix("tabpfn-")


def _method_dirname(handle: ModelHandle) -> str:
    if handle.source == "baseline":
        return handle.name
    if handle.source == "tabpfn-untuned":
        return f"tabpfn-untuned__{_short_base_tag(handle.base_path)}"
    extra = handle.extra or {}
    short = _short_base_tag(extra.get("base_checkpoint"))
    lr = extra.get("learning_rate")
    if lr is not None:
        return f"tabpfn-trained__{short}__lr{lr:.0e}"
    return f"tabpfn-trained__{short}"


def _output_path_for(
    handle: ModelHandle, *,
    track: str, run_name: str, timestamp: str,
    base_dir: str | Path,
    per_task_tag: str | None = None,
) -> Path:
    track_dir = "PD" if track == "pd" else "LGD"
    suffix = f"__{per_task_tag}" if per_task_tag else ""
    return (
        resolve_output_path(base_dir) / track_dir / _method_dirname(handle)
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
# contains at least one row with `test_dataset_id == dataset_id` AND
# `status == "OK"`. FAIL rows do NOT count — those should be retried.


def find_existing_results(
    handle: ModelHandle, dataset_id: str, *,
    track: str, results_base_dir: str | Path,
) -> list[Path]:
    """Return CSVs that already contain an OK row for this (handle, dataset).

    Walks every CSV under the method's results directory; opens each one
    with ``csv.DictReader`` and looks for a matching ``test_dataset_id``
    with ``status == "OK"``. Empty list ⇒ no prior result; caller should
    score this pair from scratch.
    """
    method_dir = (
        resolve_output_path(results_base_dir)
        / ("PD" if track == "pd" else "LGD")
        / _method_dirname(handle)
    )
    if not method_dir.exists():
        return []
    hits: list[Path] = []
    # Two fast filename heuristics first (per-task slurm tag encodes the id);
    # fall back to opening the file only for CSVs that might match.
    needle = f"ds-{dataset_id}"
    for csv_path in sorted(method_dir.glob("*.csv")):
        if needle in csv_path.name:
            # Per-task slurm CSV: filename guarantees the dataset is in here.
            # Trust it without re-opening — but still confirm at least one
            # OK row exists, since the file may record only FAILs.
            if _csv_has_ok_row(csv_path, dataset_id):
                hits.append(csv_path)
        else:
            # Non-tagged CSV (single-process run): may contain rows for many
            # datasets; have to open it.
            if _csv_has_ok_row(csv_path, dataset_id):
                hits.append(csv_path)
    return hits


def _csv_has_ok_row(csv_path: Path, dataset_id: str) -> bool:
    """Quick membership check: does this CSV record at least one OK row
    for ``dataset_id``?"""
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if (row.get("test_dataset_id") == dataset_id
                        and row.get("status") == "OK"):
                    return True
    except (OSError, csv.Error):
        return False
    return False


def _write_csv(rows: list[EvalRow], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


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
) -> list[EvalRow]:
    """Run K-fold CV (with inner train/val split) of one model on one
    test dataset and return the per-fold rows.

    Subsampling policy (Gemini's #4 fix + user's HPO-only-cap design):

      * The dataset's TabPFN-architectural cap (``max_rows_tabpfn``,
        applied to TabPFN-* models only) has been pre-applied by
        :func:`run_benchmark` BEFORE this function is called. Inside
        the CV loop we don't subsample further — we want every fold
        to use the full data the model can handle.
      * The boosting wrappers (XGBoost, CatBoost) themselves apply
        ``hpo.<m>.max_rows`` to the HPO objective only — that cap
        never touches the final fit or the test fold.
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

        X_tr_df = X_full.iloc[sub_tr]
        X_va_df = X_full.iloc[sub_va]
        X_te_df = X_full.iloc[te_idx]
        y_tr = y_full[sub_tr]
        y_va = y_full[sub_va]
        y_te = y_full[te_idx]

        X_tr_arr, X_va_arr, X_te_arr, cat_idx = encode_for_model(
            X_tr_df, X_va_df, X_te_df,
            categorical_columns=ds.categorical_columns,
        )

        t0 = time.monotonic()
        status = "OK"
        error: str | None = None
        metrics: dict[str, float] = {}

        try:
            # Pass the val split through to the model. Wrappers that do
            # HPO (XGBoost/CatBoost) use it as the Optuna objective —
            # Gemini's #1 fix. Wrappers without HPO ignore the args.
            model.fit(X_tr_arr, y_tr, cat_idx,
                      X_val=X_va_arr, y_val=y_va)
            if ds.task_type == "classification":
                proba_va = np.asarray(model.predict_proba(X_va_arr))
                proba_te = np.asarray(model.predict_proba(X_te_arr))
                # K-classes: pad with zeros if the model only saw some.
                K_seen = max(int(proba_va.shape[1]), int(proba_te.shape[1]))
                metrics = _classification_metrics(
                    proba_test=proba_te, y_test=y_te,
                    proba_val=proba_va,  y_val=y_va,
                    n_classes_seen=K_seen,
                )
            else:
                pred_te = np.asarray(model.predict(X_te_arr)).reshape(-1)
                # Bar-distribution NLL is TabPFN-only. The current
                # sklearn-style wrapper doesn't expose it cleanly; we
                # record NaN and flag it as a known gap. Implementing
                # it requires accessing TabPFNRegressor's `predict`
                # `output_type="full"` and computing NLL from the
                # bar-distribution borders. Future work.
                metrics = _regression_metrics(
                    pred_test=pred_te, y_test=y_te, neg_nll=None,
                )
        except Exception as exc:                              # noqa: BLE001
            status = "FAIL"
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning(
                "  ↳ %s/%s fold %d FAIL: %s",
                ds.dataset_id, handle.name, fold_idx, error,
            )
            LOGGER.debug("traceback:\n%s", traceback.format_exc())

        rows.append(EvalRow(
            track=ds.track,
            task_type=ds.task_type,
            model_name=handle.name,
            model_source=handle.source,
            model_path=handle.base_path,
            test_dataset_id=ds.dataset_id,
            fold_idx=fold_idx,
            n_train_rows=int(len(sub_tr)),
            n_val_rows=int(len(sub_va)),
            n_test_rows=int(len(te_idx)),
            elapsed_sec=time.monotonic() - t0,
            timestamp=timestamp,
            status=status, error=error,
            **{k: metrics.get(k, float("nan")) for k in (
                "roc_auc", "log_loss", "pr_auc", "optimal_threshold",
                "f1", "accuracy", "precision", "recall",
                "rmse", "mae", "r2", "neg_nll",
            )},
        ))

    return rows


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def _maybe_apply_tabpfn_cap(
    ds: ProcessedDataset, *,
    max_rows_tabpfn: int | None, seed: int,
) -> ProcessedDataset:
    """Apply the TabPFN architectural cap to ``ds`` if it's larger.

    This is the ONE place where the eval subsamples before splitting
    into CV folds — and only because TabPFN's in-context learning
    has a hard memory limit. For every other model the full dataset
    is fed to K-fold splitting unchanged (per Gemini #4 + the user's
    HPO-only-subsample design).
    """
    if max_rows_tabpfn is None or ds.n_rows <= max_rows_tabpfn:
        return ds
    X_cap, y_cap = subsample(
        ds.X, ds.y,
        max_rows=max_rows_tabpfn, seed=seed,
        stratify=(ds.task_type == "classification"),
    )
    LOGGER.info(
        "TabPFN cap: %s subsampled %d → %d rows (architectural limit)",
        ds.dataset_id, ds.n_rows, len(X_cap),
    )
    return ProcessedDataset(
        X=X_cap, y=y_cap,
        categorical_columns=ds.categorical_columns,
        task_type=ds.task_type,
        dataset_id=ds.dataset_id,
        track=ds.track,
    )


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
    max_rows_tabpfn: int | None = None,
    per_task_tag: str | None = None,
) -> list[EvalRow]:
    """Score every (model × test_dataset × fold) and persist per-method CSVs.

    Failures inside one cell don't stop the loop — they're recorded
    with ``status="FAIL"`` so the comparison table is robust to a
    single bad cell.

    ``max_rows_tabpfn`` is applied to TabPFN-* models only (architectural
    cap; their in-context inference can't exceed it on a single H100).
    Classical baselines see the full dataset.
    """
    handles_and_models = list(handles_and_models)
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

    # Pre-load datasets once to amortise disk I/O across models.
    datasets_full: dict[str, ProcessedDataset] = {}
    for did in test_dataset_ids:
        try:
            datasets_full[did] = load_processed_dataset(track=track, dataset_id=did)
        except (FileNotFoundError, KeyError) as exc:
            LOGGER.warning("skipping %s: %s", did, exc)

    # Build a TabPFN-capped variant ONCE per dataset (saves recomputing
    # the stratified subsample for every TabPFN model that sees it).
    datasets_tabpfn: dict[str, ProcessedDataset] = {
        did: _maybe_apply_tabpfn_cap(ds, max_rows_tabpfn=max_rows_tabpfn, seed=seed)
        for did, ds in datasets_full.items()
    }

    rows: list[EvalRow] = []
    rows_by_model: dict[str, list[EvalRow]] = {}

    for m_idx, (handle, model) in enumerate(handles_and_models, start=1):
        rows_by_model.setdefault(handle.name, [])
        LOGGER.info("model %d/%d  %s  (source=%s)",
                    m_idx, len(handles_and_models), handle.name, handle.source)

        # TabPFN models see the architecturally-capped dataset; every
        # other model sees the full one.
        ds_pool = (
            datasets_tabpfn if handle.source.startswith("tabpfn-")
            else datasets_full
        )

        for did, ds in ds_pool.items():
            LOGGER.info("  dataset %s  (n_rows=%d, n_features=%d)",
                        did, ds.n_rows, ds.n_features)
            fold_rows = _bench_model_on_dataset(
                handle=handle, model=model, ds=ds,
                n_folds=n_folds, inner_val_fraction=inner_val_fraction,
                seed=seed, timestamp=timestamp,
            )
            rows.extend(fold_rows)
            rows_by_model[handle.name].extend(fold_rows)

        # Persist this model's rows. New file per (run_name, timestamp,
        # task_tag), so concurrent slurm tasks never write to the same file.
        out_path = _output_path_for(
            handle, track=track, run_name=run_name, timestamp=timestamp,
            base_dir=results_base_dir, per_task_tag=per_task_tag,
        )
        _write_csv(rows_by_model[handle.name], out_path)
        LOGGER.info("  → wrote %d rows to %s",
                    len(rows_by_model[handle.name]), out_path)

    return rows
