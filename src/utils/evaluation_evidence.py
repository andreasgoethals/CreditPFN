"""Independently recompute scores and verify unchanged test cohorts before publication."""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, r2_score, mean_squared_error


def scores(target, prediction, track):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if prediction.ndim == 2 and prediction.shape[1] == 2:
        prediction = prediction[:, 1]
    prediction = prediction.reshape(-1)
    if target.shape != prediction.shape or not len(target) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Missing, malformed or nonfinite saved predictions")
    if track == "pd":
        if not np.isin(target, [0, 1]).all() or ((prediction < 0) | (prediction > 1)).any():
            raise ValueError("Invalid binary probabilities or labels")
        return dict(roc_auc=roc_auc_score(target, prediction), brier_score=brier_score_loss(target, prediction))
    return dict(r2=r2_score(target, prediction), rmse=np.sqrt(mean_squared_error(target, prediction)))


def check_scores(record, target, prediction, track):
    measured = scores(target, prediction, track)
    for key, value in measured.items():
        if not np.isfinite(record[key]) or not np.isclose(value, record[key], rtol=2e-6, atol=1e-8):
            raise ValueError(f"Saved predictions disagree with {key}")
    return measured


def inventory(root, family="tabicl"):
    """One immutable comparison identity per method, campaign partition and table."""
    found = {}
    for track in ("PD", "LGD"):
        if family not in ("tabicl", "tabpfn"):
            raise ValueError("Unknown foundation family")
        for file in sorted((Path(root)/track).glob(f"{family}-*/*.csv")):
            match = re.match(r"(.+_s\d{2})_\d{8}_", file.name)
            if not match:
                raise ValueError(f"Unrecognized benchmark filename: {file.name}")
            data = pd.read_csv(file)
            for (model, table), part in data.groupby(["model_name", "test_dataset_id"]):
                key = (track, file.parent.name, match[1], model, table)
                if key in found or len(part) != 5 or set(part.fold_idx) != set(range(5)) or not part.status.eq("OK").all():
                    raise ValueError("Duplicate, failed or incomplete five-fold comparison")
                found[key] = (file, part.sort_values("fold_idx"))
    if not found:
        raise ValueError(f"No completed {family} comparisons")
    return found


def predictions(file):
    receipt = json.loads(file.with_name(file.name+".evaluation.json").read_text())["__predictions__"]
    name = receipt["file"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Unsafe prediction receipt path")
    path = file.parent/name
    if path.stat().st_size != receipt["bytes"]:
        raise ValueError("Prediction receipt size mismatch")
    return pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)


def benchmark(original, replacement):
    old, new = inventory(original), inventory(replacement)
    if old.keys() != new.keys():
        raise ValueError(f"Comparison coverage changed: missing={len(old.keys()-new.keys())}, extra={len(new.keys()-old.keys())}")
    total = 0
    for key, (file, metrics) in new.items():
        old_file, old_metrics = old[key]
        protocol = ["fold_idx", "n_train_rows", "n_val_rows", "n_test_rows", "split_seed", "domain"]
        pd.testing.assert_frame_equal(metrics[protocol].reset_index(drop=True),
                                      old_metrics[protocol].reset_index(drop=True), check_dtype=False)
        actual, previous = predictions(file), predictions(old_file)
        selected = lambda d: d[d.model_name.eq(key[3]) & d.test_dataset_id.eq(key[4])]
        actual, previous = selected(actual), selected(previous)
        index = ["fold_idx", "row_idx"]
        if actual.duplicated(index).any() or previous.duplicated(index).any():
            raise ValueError("Repeated test row in a fold")
        # Five outer folds must cover the same complete test cohort, once per row.
        if actual.row_idx.duplicated().any():
            raise ValueError("Outer test folds overlap")
        pd.testing.assert_frame_equal(actual[index+["y_true"]].sort_values(index).reset_index(drop=True),
                                      previous[index+["y_true"]].sort_values(index).reset_index(drop=True), check_dtype=False)
        for record in metrics.to_dict("records"):
            part = actual[actual.fold_idx.eq(record["fold_idx"])]
            if len(part) != record["n_test_rows"]:
                raise ValueError("Saved prediction count differs from test fold")
            check_scores(record, part.y_true.to_numpy(), part.y_pred.to_numpy(), key[0].lower())
            total += len(part)
    return dict(comparisons=len(new), folds=5*len(new), predictions=total, unchanged_test_cohorts=True)


def context(plan, result_root, receipt_root):
    from src.eval.context_curve import budgets, fold_indices
    from src.eval.dataset_loader import load_processed_dataset
    from src.utils.experiment import digest_json
    count = rows = 0
    cfg = plan["config"]
    for index, task in enumerate(plan["tasks"]):
        token = f"task_{index:04d}"
        receipt = json.loads((Path(receipt_root)/(token+".json")).read_text())
        if receipt.get("passed") is not True or receipt.get("plan_sha256") != plan["sha256"] or receipt.get("task") != index:
            raise ValueError("Context completion receipt is missing or stale")
        dataset = load_processed_dataset(task["track"], task["dataset_id"])
        order, validation, test = fold_indices(dataset.y, dataset.task_type, task["fold"], cfg)
        expected = {(n, state) for n in budgets(len(order), cfg) for state in ("base", "cpt")}
        export = pd.read_csv(Path(result_root)/(token+".csv.gz"))
        if len(export) != len(expected) or set(zip(export.context_rows, export.state)) != expected:
            raise ValueError("Incomplete or duplicate context export")
        with sqlite3.connect((Path(result_root)/(token+".sqlite")).as_uri()+"?mode=ro", uri=True) as db:
            if db.execute("SELECT sha FROM identity").fetchall() != [(plan["sha256"],)]:
                raise ValueError("Context journal belongs to a different plan")
            if set(db.execute("SELECT rows,state FROM cells")) != expected:
                raise ValueError("Context journal is incomplete")
            for size, state, raw, blob in db.execute("SELECT rows,state,metrics,predictions FROM cells"):
                record = json.loads(raw)
                with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
                    for key, array in (("context", order[:size]), ("validation", validation), ("test", test), ("target", dataset.y[test])):
                        if not np.array_equal(saved[key], array):
                            raise ValueError(f"Context cell changed {key}")
                    measured = check_scores(record, saved["target"], saved["prediction"], task["track"])
                exported = export[export.context_rows.eq(size) & export.state.eq(state)].iloc[0]
                for key, value in measured.items():
                    if not np.isclose(exported[key], value, rtol=2e-6, atol=1e-8):
                        raise ValueError("Context export disagrees with saved predictions")
                if (exported.test_rows != len(test) or exported.validation_rows != len(validation)
                        or exported.test_indices_sha256 != digest_json(test.tolist())
                        or exported.context_indices_sha256 != digest_json(order[:size].tolist())):
                    raise ValueError("Context export changed test/context coverage")
                count += 1; rows += len(test)
        if receipt["cells"] != len(expected):
            raise ValueError("Context receipt cell count mismatch")
    return dict(tasks=len(plan["tasks"]), cells=count, predictions=rows, fixed_complete_test_rows=True)
