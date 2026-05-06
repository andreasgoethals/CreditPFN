"""Cross-model benchmark on the held-out test split.

For each chunk in the test bucket (computed by the same
:func:`src.train.corpus.split_corpus` the training pipeline used,
so the test set is identical), this module:

  1. Loads ``X_context, y_context, X_query, y_query, categorical_idx``
     from the cached ``.npz`` and concatenates all rows into
     ``(X_all, y_all)``.
  2. Splits ``(X_all, y_all)`` into ``cfg.cv.n_folds`` folds — stratified
     for classification, plain KFold for regression.
  3. For each (model × fold):
        model.fit(X_train_fold, y_train_fold, categorical_idx)
        score = metric(model, X_val_fold, y_val_fold)
     producing ``n_models × n_chunks × n_folds`` rows in the output.
  4. Persists each row to a long-format CSV at::

         results/<TRACK>/<method>/<run_name>_<timestamp>.csv

     Each model gets its OWN method directory; the ``<timestamp>``
     suffix means a fresh run never overwrites previous ones —
     every benchmark we ever ran is preserved on disk.

The CSV is the input to whatever plot / table the paper builds
afterwards. Long format means a single ``pd.read_csv`` +
``groupby([model_name]).agg(...)`` is enough for any aggregate.
"""

from __future__ import annotations

import csv
import datetime as _dt
import logging
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from src.model.base import ModelHandle
from src.model.tabpfn_models import TabPFNTrained
from src.train.corpus import ChunkRef
from src.utils.paths import resolve_output_path

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Output row
# --------------------------------------------------------------------------- #


@dataclass
class EvalRow:
    """One row of the long-format benchmark CSV."""
    track:           str            # "pd" | "lgd"
    task_type:       str
    model_name:      str
    model_source:    str             # "baseline" | "tabpfn-untuned" | "tabpfn-trained"
    model_path:      str | None
    test_dataset_id: str
    test_chunk_idx:  int
    fold_idx:        int
    n_train_rows:    int             # = len(train fold)
    n_test_rows:     int             # = len(val fold)
    metric_name:     str
    metric_value:    float
    elapsed_sec:     float
    timestamp:       str             # ISO8601 stamp common to the whole run
    status:          str             # "OK" | "FAIL"
    error:           str | None


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
            "multi_chunk_policy":  row["multi_chunk_policy"],
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
# Per-chunk loader + scorer
# --------------------------------------------------------------------------- #


def _load_chunk_concat(ref: ChunkRef) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load ``ref`` and concatenate context+query into one (X, y) pool.

    K-fold CV needs the full chunk — we build folds ourselves.
    """
    with np.load(ref.chunk_path) as data:
        X = np.concatenate([data["X_context"], data["X_query"]], axis=0)
        y = np.concatenate([data["y_context"], data["y_query"]], axis=0)
        cat_idx = data["categorical_idx"].tolist()
    return X, y, cat_idx


def _score(
    model, *,
    task_type: str,
    X_query: np.ndarray, y_query: np.ndarray,
    metric_name: str,
) -> float:
    """Compute the requested metric on one fold."""
    from sklearn.metrics import (
        log_loss as sk_log_loss,
        roc_auc_score, mean_squared_error,
    )
    if task_type == "classification":
        proba = model.predict_proba(X_query)
        K = proba.shape[1] if proba.ndim == 2 else 2
        if metric_name == "roc_auc":
            unique = np.unique(y_query)
            if len(unique) < 2:
                return float("nan")
            if K == 2:
                return float(roc_auc_score(y_query, proba[:, 1]))
            return float(roc_auc_score(
                y_query, proba, multi_class="ovr", average="macro",
            ))
        if metric_name == "log_loss":
            return float(sk_log_loss(y_query, proba, labels=list(range(K))))
        raise ValueError(f"unsupported classification metric {metric_name!r}")

    pred = np.asarray(model.predict(X_query)).reshape(-1)
    if metric_name == "rmse":
        return float(np.sqrt(mean_squared_error(y_query, pred)))
    if metric_name == "neg_nll":
        return float(-np.mean((pred - y_query) ** 2))
    raise ValueError(f"unsupported regression metric {metric_name!r}")


def _make_folds(y: np.ndarray, *, task_type: str, n_folds: int, seed: int):
    """Yield ``(train_idx, val_idx)`` pairs for K folds.

    With ``n_folds=5`` (the default), each fold holds out 20% of the
    chunk's rows for evaluation and uses the remaining 80% as the
    "training" / context data — exactly the structure the user
    specified. Folds are stratified for classification (preserves
    class proportions across folds) and plain shuffled for
    regression.

    HPO (when used by XGBoost / CatBoost) further splits the 80%
    training fold into 64% (model fit) + 16% (Optuna validation).
    See :meth:`src.model.boosting.XGBoostModel._maybe_hpo` for the
    inner split — it uses ``train_test_split(test_size=0.2)`` on
    the same training fold, so the user's "20% of training as
    validation for HPO" contract is satisfied. Optuna runs ``n_folds``
    studies per (model × chunk), one per CV fold.
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    n = len(y)
    if n_folds <= 1 or n_folds > n:
        # Degenerate: a single 80/20 split.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, n // 5)
        yield perm[n_val:], perm[:n_val]
        return
    if task_type == "classification" and len(np.unique(y)) >= 2:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr, va in skf.split(np.zeros(n), y):
            yield tr, va
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr, va in kf.split(np.zeros(n)):
            yield tr, va


# --------------------------------------------------------------------------- #
# Per-method output dir + filename
# --------------------------------------------------------------------------- #


_NAME_RE   = re.compile(r"[^A-Za-z0-9_.-]")
_BASE_RE   = re.compile(r"tabpfn-(?P<v>v\d\.\d)-(?:classifier|regressor)-v\d\.\d_(?P<variant>.+)")


def _short_base_tag(base_path: str | None) -> str:
    """Compress a base-checkpoint filename to a short, readable tag.

    The published Prior Labs filenames look like
    ``tabpfn-v2.6-classifier-v2.6_default.ckpt`` — verbose, with the
    track repeated ("classifier" / "regressor") and the version
    repeated. The track is already encoded in the parent
    ``results/PD|LGD/`` folder, so we drop it here.

    Examples:
        tabpfn-v2.6-classifier-v2.6_default.ckpt   → v2.6-default
        tabpfn-v2.5-regressor-v2.5_real.ckpt       → v2.5-real
        tabpfn-v2.5-classifier-v2.5_default-2.ckpt → v2.5-default-2
    """
    if not base_path:
        return "unknown"
    stem = Path(base_path).stem
    m = _BASE_RE.match(stem)
    if m:
        return f"{m['v']}-{m['variant']}"
    # Fall back: strip a possible "tabpfn-" prefix and call it a day.
    return stem.removeprefix("tabpfn-")


def _method_dirname(handle: ModelHandle) -> str:
    """Folder name for one model under ``results/<TRACK>/<here>/``.

    Schema:
        baseline        → just the name (``xgboost``, ``catboost``, …)
        tabpfn-untuned  → ``tabpfn-untuned__<short_base>``
                          (e.g. ``tabpfn-untuned__v2.6-default``)
        tabpfn-trained  → ``tabpfn-trained__<short_base>__lr<lr>__<policy>``
                          (e.g. ``tabpfn-trained__v2.6-default__lr1e-05__allchunks``)

    The short tag is built by :func:`_short_base_tag`; the
    track-specific "classifier"/"regressor" infix is dropped because
    the parent path already encodes the track. The ``seed`` is
    intentionally NOT in the dirname — it would multiply the number
    of folders without changing what's being compared. A different
    seed is a separate ``<run_name>_<timestamp>.csv`` *inside* the
    same folder.
    """
    if handle.source == "baseline":
        return handle.name
    if handle.source == "tabpfn-untuned":
        return f"tabpfn-untuned__{_short_base_tag(handle.base_path)}"
    # tabpfn-trained — read HPs out of `extra` (see registry / load_trained_handles).
    extra = handle.extra or {}
    short = _short_base_tag(extra.get("base_checkpoint"))
    lr = extra.get("learning_rate")
    policy = extra.get("multi_chunk_policy", "")
    policy_short = {
        "all_chunks_as_separate_datasets": "allchunks",
        "first_chunk_only":                "firstchunk",
    }.get(policy, _NAME_RE.sub("-", policy))
    if lr is not None:
        return f"tabpfn-trained__{short}__lr{lr:.0e}__{policy_short}"
    return f"tabpfn-trained__{short}"


def _output_path_for(
    handle: ModelHandle, *,
    track: str, run_name: str, timestamp: str,
    base_dir: str | Path,
    per_task_tag: str | None = None,
) -> Path:
    """``results/<TRACK>/<method>/<run_name>_<timestamp>[_<tag>].csv``.

    The ``per_task_tag`` is appended only when a slurm-array task is
    running — it embeds the dataset_id (and task index) so two
    concurrent tasks for the same method don't clobber each other.
    """
    track_dir = "PD" if track == "pd" else "LGD"
    suffix = f"_{per_task_tag}" if per_task_tag else ""
    return (
        resolve_output_path(base_dir) / track_dir / _method_dirname(handle)
        / f"{run_name}_{timestamp}{suffix}.csv"
    )


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def run_benchmark(
    *,
    test_chunks: list[ChunkRef],
    handles_and_models: Iterable[tuple[ModelHandle, object]],
    track: Literal["pd", "lgd"],
    metric_name: str,
    run_name: str,
    n_folds: int = 5,
    seed: int = 42,
    results_base_dir: str | Path = "results",
    per_task_tag: str | None = None,
) -> list[EvalRow]:
    """Score every (model × test-chunk × fold) and persist per-method CSVs.

    Failures inside one cell don't stop the loop — they're recorded
    with ``status="FAIL"``. Each model writes its own
    ``results/<TRACK>/<method>/<run_name>_<timestamp>.csv``.
    """
    handles_and_models = list(handles_and_models)
    if not test_chunks:
        LOGGER.warning("test_chunks is empty — nothing to benchmark.")
        return []
    if not handles_and_models:
        LOGGER.warning("handles_and_models is empty — nothing to benchmark.")
        return []

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    track_label = "PD" if track == "pd" else "LGD"
    LOGGER.info(
        "Benchmark: track=%s, %d models × %d chunks × %d folds = %d cells, "
        "metric=%s, run_name=%s, timestamp=%s",
        track_label, len(handles_and_models), len(test_chunks), n_folds,
        len(handles_and_models) * len(test_chunks) * n_folds,
        metric_name, run_name, timestamp,
    )

    # Pre-load every chunk once to amortise disk I/O across models.
    chunk_data = []
    for ref in test_chunks:
        X, y, cat_idx = _load_chunk_concat(ref)
        # Pre-build the fold splits per chunk; same seed across models
        # ⇒ every model sees the SAME folds for a given chunk, which
        # is what lets us pair them later.
        folds = list(_make_folds(
            y, task_type=ref.task_type, n_folds=n_folds, seed=seed,
        ))
        chunk_data.append((ref, X, y, cat_idx, folds))

    rows: list[EvalRow] = []
    rows_by_model: dict[str, list[EvalRow]] = {}

    for m_idx, (handle, model) in enumerate(handles_and_models, start=1):
        rows_by_model.setdefault(handle.name, [])
        LOGGER.info("model %d/%d  %s  (source=%s)",
                    m_idx, len(handles_and_models), handle.name, handle.source)

        for ref, X, y, cat_idx, folds in chunk_data:
            for fold_idx, (tr_idx, va_idx) in enumerate(folds):
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_va, y_va = X[va_idx], y[va_idx]

                t0 = time.monotonic()
                status = "OK"
                error: str | None = None
                metric_value = float("nan")
                try:
                    model.fit(X_tr, y_tr, cat_idx)
                    metric_value = _score(
                        model, task_type=handle.task_type,
                        X_query=X_va, y_query=y_va,
                        metric_name=metric_name,
                    )
                except Exception as exc:                          # noqa: BLE001
                    status = "FAIL"
                    error = f"{type(exc).__name__}: {exc}"
                    LOGGER.warning(
                        "  ↳ %s/%s fold %d FAIL: %s",
                        ref.dataset_id, handle.name, fold_idx, error,
                    )
                    LOGGER.debug("traceback:\n%s", traceback.format_exc())

                row = EvalRow(
                    track=track,
                    task_type=handle.task_type,
                    model_name=handle.name,
                    model_source=handle.source,
                    model_path=handle.base_path,
                    test_dataset_id=ref.dataset_id,
                    test_chunk_idx=ref.chunk_idx,
                    fold_idx=fold_idx,
                    n_train_rows=int(len(tr_idx)),
                    n_test_rows=int(len(va_idx)),
                    metric_name=metric_name,
                    metric_value=metric_value,
                    elapsed_sec=time.monotonic() - t0,
                    timestamp=timestamp,
                    status=status, error=error,
                )
                rows.append(row)
                rows_by_model[handle.name].append(row)

        # Persist this model's rows to its own per-method file. New file
        # per (run_name, timestamp), so previous runs are never clobbered.
        out_path = _output_path_for(
            handle, track=track, run_name=run_name, timestamp=timestamp,
            base_dir=results_base_dir, per_task_tag=per_task_tag,
        )
        _write_csv(rows_by_model[handle.name], out_path)
        LOGGER.info("  → wrote %d rows to %s",
                    len(rows_by_model[handle.name]), out_path)

    return rows


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
