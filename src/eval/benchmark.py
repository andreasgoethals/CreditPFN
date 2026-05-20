"""Cross-model benchmark on the held-out test datasets.

For each test dataset (one entry per `dataset_id`) and each model in
the roster, this module:

  1. Loads the processed CSV via
     :func:`src.eval.dataset_loader.load_processed_dataset`.
  2. Runs `cfg.cv.n_folds` cross-validation on the **full** dataset.
     Inside each fold, **only the training partition is capped** at
     the architectural per-model row limit (TabPFN family only —
     see ``cfg.max_rows_per_model``). The held-out test partition is
     never capped; we call `predict_proba(X_test)` once on the full
     test fold and TabPFN-v3's internal row chunking handles it.
     Classical baselines (XGBoost/CatBoost/LogReg/LinReg) bypass the
     cap and see the full train + test rows.
  3. Per outer fold:

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
        # ``use_lora`` column is present on manifests produced after the
        # LoRA tuneable was added; older manifests don't have it. Default
        # missing → False so the eval still runs against legacy manifests.
        use_lora_raw = row.get("use_lora", False)
        if isinstance(use_lora_raw, str):
            use_lora_val = use_lora_raw.strip().lower() in ("true", "1", "yes")
        else:
            use_lora_val = bool(use_lora_raw)
        extra = {
            "base_checkpoint":     row["base_checkpoint"],
            "learning_rate":       float(row["learning_rate"]),
            "use_lora":            use_lora_val,
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
# Filename schema: ``tabpfn-<version>-<role>-<version>_<variant>``
# where <version> is ``v2.5`` / ``v2.6`` / ``v3`` (and any future
# ``v3.x``). The pre- and post-role version strings always match; we
# only capture once.
_BASE_RE = re.compile(
    r"tabpfn-(?P<v>v\d+(?:\.\d+)?)-(?:classifier|regressor)-v\d+(?:\.\d+)?_(?P<variant>.+)"
)


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
    lora_tag = "__lora" if extra.get("use_lora") else ""
    if lr is not None:
        return f"tabpfn-trained__{short}__lr{lr:.0e}{lora_tag}"
    return f"tabpfn-trained__{short}{lora_tag}"


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
# contains an OK row for **every fold** of that dataset (so partial-failure
# pairs — where, say, 1/5 folds succeeded — are NOT skipped on rerun and
# the missing folds get retried). The required fold count is taken from
# the caller's `n_folds_required`; when None, any single OK row counts
# (legacy behaviour, used only by the test suite).


def find_existing_results(
    handle: ModelHandle, dataset_id: str, *,
    track: str, results_base_dir: str | Path,
    n_folds_required: int | None = None,
) -> list[Path]:
    """Return CSVs that contribute OK rows for this (handle, dataset).

    Walks every CSV under the method's results directory; opens each
    one with ``csv.DictReader`` and collects the set of distinct
    ``fold_idx`` values with ``status == "OK"`` for ``dataset_id``.
    The pair is considered "complete" — and the returned list is
    non-empty — iff the OK fold count is at least ``n_folds_required``
    (or, when ``n_folds_required`` is None, at least one OK row exists).

    A pair with some failed folds will return an empty list, so the
    caller re-runs and the missing folds get retried.
    """
    method_dir = (
        resolve_output_path(results_base_dir)
        / ("PD" if track == "pd" else "LGD")
        / _method_dirname(handle)
    )
    if not method_dir.exists():
        return []

    hits: list[Path] = []
    ok_folds: set[int | str] = set()
    needle = f"ds-{dataset_id}"
    for csv_path in sorted(method_dir.glob("*.csv")):
        if needle not in csv_path.name and not _csv_might_have_dataset(csv_path, dataset_id):
            # Filename doesn't carry the id AND the file isn't a generic
            # multi-dataset CSV (skip the expensive open).
            continue
        new_folds = _csv_ok_folds_for(csv_path, dataset_id)
        if new_folds:
            hits.append(csv_path)
            ok_folds.update(new_folds)

    if not hits:
        return []
    if n_folds_required is None:
        return hits
    return hits if len(ok_folds) >= int(n_folds_required) else []


def _csv_might_have_dataset(csv_path: Path, dataset_id: str) -> bool:
    """Cheap pre-filter for non-tagged (single-process) CSVs.

    Per-task slurm filenames always encode ``ds-<id>`` so are matched by
    the caller's filename check. Non-tagged files may or may not contain
    the dataset; we open them only when there's at least a chance the
    test_dataset_id column is present.
    """
    return csv_path.suffix == ".csv"


def _csv_ok_folds_for(csv_path: Path, dataset_id: str) -> set[int | str]:
    """Return the set of distinct ``fold_idx`` values with status=OK
    for ``dataset_id`` in this CSV (empty if none / on read error)."""
    folds: set[int | str] = set()
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if (row.get("test_dataset_id") == dataset_id
                        and row.get("status") == "OK"):
                    try:
                        folds.add(int(row.get("fold_idx", "")))
                    except (TypeError, ValueError):
                        folds.add(row.get("fold_idx", ""))
    except (OSError, csv.Error):
        return set()
    return folds


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
    max_rows_for_handle: int | None = None,
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
        (``repositories/TabPFN .txt:17650``).
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

        try:
            # Pass the val split through to the model. Wrappers that do
            # HPO (XGBoost/CatBoost) use it as the Optuna objective —
            # Gemini's #1 fix. Wrappers without HPO ignore the args.
            model.fit(X_tr_arr, y_tr, cat_idx,
                      X_val=X_va_arr, y_val=y_va)
            if ds.task_type == "classification":
                proba_va = np.asarray(model.predict_proba(X_va_arr))
                proba_te = np.asarray(model.predict_proba(X_te_arr))
                # If the model only saw a subset of classes during fit,
                # its predict_proba returns fewer columns than the dataset
                # has classes. Pad with zero columns so the column index
                # matches the actual class label — required for log_loss
                # with labels=[0..K-1] and multiclass roc_auc.
                K_total = max(
                    int(proba_va.shape[1]),
                    int(proba_te.shape[1]),
                    int(y_va.max()) + 1 if len(y_va) else 0,
                    int(y_te.max()) + 1 if len(y_te) else 0,
                )

                def _pad(p: np.ndarray, K: int) -> np.ndarray:
                    if p.shape[1] >= K:
                        return p
                    pad_cols = np.zeros((p.shape[0], K - p.shape[1]),
                                        dtype=p.dtype)
                    return np.hstack([p, pad_cols])

                proba_va = _pad(proba_va, K_total)
                proba_te = _pad(proba_te, K_total)
                metrics = _classification_metrics(
                    proba_test=proba_te, y_test=y_te,
                    proba_val=proba_va,  y_val=y_va,
                    n_classes_seen=K_total,
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


def resolve_max_rows_for_handle(
    handle: ModelHandle, *,
    max_rows_per_model: dict[str, int] | None,
) -> int | None:
    """Look up the architectural row-cap for one handle.

    The cap applies to TabPFN-family models only (in-context learning
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
    if not handle.source.startswith("tabpfn-"):
        return None

    # Most-specific to least-specific key lookup.
    base_short = (
        _short_base_tag((handle.extra or {}).get("base_checkpoint"))
        if handle.source == "tabpfn-trained"
        else handle.name.removeprefix("tabpfn-untuned__")
        if handle.source == "tabpfn-untuned"
        else _short_base_tag(handle.base_path)
    )
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
) -> list[EvalRow]:
    """Score every (model × test_dataset × fold) and persist per-method CSVs.

    Failures inside one cell don't stop the loop — they're recorded
    with ``status="FAIL"`` so the comparison table is robust to a
    single bad cell.

    ``max_rows_per_model`` is the per-architecture training-context
    cap (TabPFN-v3 → 1 M, TabPFN-v2.x → 100 k, etc.). Looked up by the
    base-stem key. Applied to the training fold only — the test fold
    is **always full** and predict_proba is called on it in one go.
    Classical baselines bypass the cap entirely.
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

    # Pre-load datasets once to amortise disk I/O across models. There is
    # no pre-cap step: the per-fold logic caps only the training partition,
    # leaving the test partition full so we predict on all held-out rows.
    datasets_full: dict[str, ProcessedDataset] = {}
    for did in test_dataset_ids:
        try:
            datasets_full[did] = load_processed_dataset(track=track, dataset_id=did)
        except (FileNotFoundError, KeyError) as exc:
            LOGGER.warning("skipping %s: %s", did, exc)

    rows: list[EvalRow] = []
    rows_by_model: dict[str, list[EvalRow]] = {}

    for m_idx, (handle, model) in enumerate(handles_and_models, start=1):
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
            fold_rows = _bench_model_on_dataset(
                handle=handle, model=model, ds=ds,
                n_folds=n_folds, inner_val_fraction=inner_val_fraction,
                seed=seed, timestamp=timestamp,
                max_rows_for_handle=max_rows_for_handle,
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
