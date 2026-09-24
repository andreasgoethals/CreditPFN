"""GPU integration check: uninterrupted versus stop-at-update-5 and resumed training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from src.utils.atomic import write_json
from src.utils.paths import training_dir, results_dir


def make_configs(folder: Path) -> list[Path]:
    paths = []
    for track in ("pd", "lgd"):
        for arm in ("reference", "resumed"):
            cfg = OmegaConf.load(f"config/experiment0/recovery_{track}.yaml")
            cfg.run_name += "_" + arm
            path = folder / f"recovery_{track}_{arm}.yaml"
            OmegaConf.save(cfg, path)
            paths.append(path)
    return paths


def compare(reference: Path, resumed: Path) -> dict:
    from src.utils.audit_experiment import _inference_state
    # Saved training weights already share one serialization format in both arms.
    before, after = _inference_state(reference, None), _inference_state(resumed, None)
    keys_equal = before.keys() == after.keys()
    max_delta = 0.0
    different = []
    exact = True
    for key in before.keys() & after.keys():
        a, b = before[key], after[key]
        exact &= torch.equal(a, b)
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            different.append(key)
            continue
        delta = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
        max_delta = max(max_delta, delta)
        if not torch.allclose(a, b, rtol=1e-6, atol=1e-7):
            different.append(key)
    return {"passed": keys_equal and not different, "bitwise_equal": keys_equal and exact,
            "max_absolute_difference": max_delta, "different_tensors": different,
            "rtol": 1e-6, "atol": 1e-7}


def benchmark_smoke(path: Path, track: str, trial: int) -> dict:
    """Exercise the real five-fold scoring/prediction path on a small packaged table."""
    from src.eval.benchmark import _bench_model_on_dataset, _write_csv, _write_predictions
    from src.data.retention import processed_dataset
    from src.model.base import ModelHandle
    from src.train.tabicl_compat import model_family
    from src.model.tabpfn_models import TabPFNTrained
    from src.model.tabicl_models import TabICLTrained
    family = model_family(str(path))
    task = "classification" if track == "pd" else "regression"
    wrapper = TabICLTrained if family == "tabicl" else TabPFNTrained
    model = wrapper(task_type=task, ckpt_path=path, device="cuda", n_estimators=2)
    handle = ModelHandle(model.name, track, task, f"{family}-trained", str(path))
    dataset = "package_breast_cancer" if track == "pd" else "package_diabetes"
    ds = processed_dataset(track, dataset)
    predictions = []
    rows = _bench_model_on_dataset(handle=handle, model=model, ds=ds, n_folds=5,
        inner_val_fraction=.2, seed=99, timestamp="recovery_smoke", pred_records=predictions)
    folder = results_dir(track.upper(), "recovery_smoke", experiment="experiment0")
    _write_csv(rows, folder / f"base_{trial}.csv")
    _write_predictions(predictions, folder / f"base_{trial}.predictions")
    required = ("roc_auc", "f1", "brier_score") if track == "pd" else (
        "rmse", "mean_pinball_loss", "interval_coverage_80", "interval_width_90")
    passed = (len(rows) == 5 and all(row.status == "OK" for row in rows)
        and all(np.isfinite(getattr(row, name)) for row in rows for name in required)
        and sorted(p["row_idx"] for p in predictions) == list(range(len(ds.y))))
    return {"passed": bool(passed), "folds": len(rows), "predictions": len(predictions), "dataset": dataset}


def run(folder: Path, track: str, trial: int):
    from src.utils.audit_experiment import audit
    paths = {arm: folder / f"recovery_{track}_{arm}.yaml" for arm in ("reference", "resumed")}
    common = [sys.executable, "-u", "scripts/train_pipeline.py", "--split-index", "0", "--trial-index", str(trial)]
    if os.environ.get("CREDITPFN_ACTIVE_LOG"):
        common += ["--log-path", os.environ["CREDITPFN_ACTIVE_LOG"]]
    env = dict(os.environ)
    env.pop("CREDITPFN_STOP_AFTER_UPDATES", None)
    env.pop("CREDITPFN_STOP_REQUESTED", None)
    env["CREDITPFN_SEGMENT_SECONDS"] = "0"
    # Do not accidentally compare a skipped previous run to a newly resumed one.
    reports = {arm: audit(path) for arm, path in paths.items()}
    if any(report["trials"][trial]["status"] != "PENDING" for report in reports.values()):
        raise RuntimeError("Recovery canary already has completed output; use a new run name for a fresh canary")
    subprocess.run([*common, "--config", str(paths["reference"])], env=env, check=True)
    stopped = subprocess.run([*common, "--config", str(paths["resumed"])],
        env=dict(env, CREDITPFN_STOP_AFTER_UPDATES="5"))
    if stopped.returncode != 75:
        raise RuntimeError(f"Expected recovery exit 75 at update 5, got {stopped.returncode}")
    subprocess.run([*common, "--config", str(paths["resumed"])], env=env, check=True)
    reports = {arm: audit(path) for arm, path in paths.items()}
    rows = {arm: report["trials"][trial] for arm, report in reports.items()}
    if any(r["status"] != "OK" or r["successful_updates"] != 12 for r in rows.values()):
        raise RuntimeError("Recovery arm did not finish its 12 successful updates")
    report = compare(Path(rows["reference"]["path"]), Path(rows["resumed"]["path"]))
    curves = [pd.read_csv(training_dir(track, rows[arm]["trial"] + ".trajectory.csv", experiment="experiment0"))
              for arm in ("reference", "resumed")]
    columns = [c for c in curves[0] if c.startswith(("metric__", "score__"))]
    trajectories_equal = (all(c["successful_updates"].tolist() == [0, 5, 12] for c in curves)
        and bool(columns) and np.allclose(curves[0][columns].to_numpy(float),
            curves[1][columns].to_numpy(float), rtol=1e-5, atol=1e-7, equal_nan=True))
    report.update(track=track, trial=trial, interrupted_at=5, final_update=12,
                  trajectory_equal=bool(trajectories_equal))
    report["benchmark_smoke"] = benchmark_smoke(Path(rows["resumed"]["path"]), track, trial)
    report["passed"] &= bool(trajectories_equal) and report["benchmark_smoke"]["passed"]
    write_json(folder / f"recovery_{track}_{trial}.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("GPU recovery equivalence failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.task < 8:
        parser.error("Expected task 0..7")
    from src.utils.experiment0 import root, complete
    track, trial = ("pd" if args.task < 4 else "lgd"), args.task % 4
    rc = 1
    try:
        run(root() / args.id, track, trial)
        rc = 0
    finally:
        complete(args.id, "recovery", track, trial, rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
