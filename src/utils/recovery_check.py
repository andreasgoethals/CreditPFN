"""GPU recovery check, with CPU-only inspection of existing checkpoint pairs."""
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
    non_diagnostic_max = 0.0
    non_diagnostic_tensor = None
    diagnostic_deltas = {}
    deltas = []
    different = []
    exact = True
    for key in sorted(before.keys() & after.keys()):
        a, b = before[key], after[key]
        exact &= a.dtype == b.dtype and torch.equal(a, b)
        if (a.shape != b.shape or a.dtype != b.dtype
                or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            different.append(key)
            continue
        delta = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
        max_delta = max(max_delta, delta)
        if key == "criterion.losses_per_bucket":
            diagnostic_deltas[key] = delta
        elif delta > non_diagnostic_max:
            non_diagnostic_max, non_diagnostic_tensor = delta, key
        if delta:
            deltas.append({"tensor": key, "max_absolute_difference": delta})
        if not torch.allclose(a, b, rtol=1e-6, atol=1e-7):
            different.append(key)
    # FullSupportBarDistribution.forward updates this detached diagnostic outside
    # the model's recovery state. It affects neither returned losses nor inference.
    # Report its difference, but use the same inference-state scope as null audits.
    model_differences = [key for key in different if key != "criterion.losses_per_bucket"]
    return {"passed": keys_equal and not model_differences,
            "all_saved_state_close": keys_equal and not different,
            "bitwise_equal": keys_equal and exact,
            "max_absolute_difference": max_delta, "different_tensors": model_differences,
            "missing_tensors": sorted(before.keys() - after.keys()),
            "added_tensors": sorted(after.keys() - before.keys()),
            "max_non_diagnostic_difference": non_diagnostic_max,
            "max_non_diagnostic_tensor": non_diagnostic_tensor,
            "diagnostic_buffer_differences": diagnostic_deltas,
            "diagnostic_buffers_different": [key for key in different if key == "criterion.losses_per_bucket"],
            "largest_differences": sorted(deltas, key=lambda d: -d["max_absolute_difference"])[:5],
            "rtol": 1e-6, "atol": 1e-7}


def compare_trajectories(reference: pd.DataFrame, resumed: pd.DataFrame) -> dict:
    """Separate independent-run differences at update 5 from differences after resume."""
    columns = sorted(c for c in reference if c.startswith(("metric__", "score__")))
    other_columns = sorted(c for c in resumed if c.startswith(("metric__", "score__")))
    milestones = [0, 5, 12]
    problems = []
    if not columns or columns != other_columns:
        problems.append("Missing or different monitor columns")
    if any(frame.get("successful_updates", pd.Series(dtype=int)).tolist() != milestones
           for frame in (reference, resumed)):
        problems.append("Expected exactly one measurement at each of updates 0, 5 and 12")
    if problems:
        return {"passed": False, "problems": problems, "by_update": [],
                "pre_interruption_equal": None}
    updates = []
    for index, step in enumerate(milestones):
        a = reference[columns].iloc[index].to_numpy(float)
        b = resumed[columns].iloc[index].to_numpy(float)
        close = np.isclose(a, b, rtol=1e-5, atol=1e-7, equal_nan=True)
        # Optional scores may be unavailable in both arms; primary metrics must be finite.
        primary = np.array([c.startswith("metric__") for c in columns])
        close[primary] &= np.isfinite(a[primary]) & np.isfinite(b[primary])
        close &= ~np.isinf(a) & ~np.isinf(b)
        finite = np.isfinite(a) & np.isfinite(b)
        delta = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None
        updates.append({"successful_updates": step, "passed": bool(close.all()),
                        "different_metrics": int((~close).sum()),
                        "first_different_metrics": [c for c, ok in zip(columns, close) if not ok][:5],
                        "max_absolute_difference": delta})
    return {"passed": all(u["passed"] for u in updates), "problems": [],
            "pre_interruption_equal": all(u["passed"] for u in updates[:2]),
            "by_update": updates, "rtol": 1e-5, "atol": 1e-7}


def comparison_report(rows: dict, track: str, trial: int) -> dict:
    if any(r["status"] != "OK" or r.get("successful_updates") != 12 for r in rows.values()):
        raise RuntimeError("Recovery arm did not finish its 12 successful updates")
    report = compare(Path(rows["reference"]["path"]), Path(rows["resumed"]["path"]))
    curves = [pd.read_csv(training_dir(track, rows[arm]["trial"] + ".trajectory.csv", experiment="experiment0"))
              for arm in ("reference", "resumed")]
    trajectory = compare_trajectories(*curves)
    report.update(track=track, trial=trial, interrupted_at=5, final_update=12,
                  trajectory_equal=trajectory["passed"], trajectory_comparison=trajectory)
    report["passed"] &= trajectory["passed"]
    return report


def compact_report(report: dict) -> dict:
    """Keep full tensor lists in JSON artifacts, not repeated in every job log."""
    result = dict(report)
    different = result.pop("different_tensors", [])
    result["different_tensor_count"] = len(different)
    result["first_different_tensors"] = different[:5]
    return result


def selected_audit_problems(audits: dict, rows: dict) -> dict:
    # Other pairs in the same plan can still be pending in parallel GPU jobs.
    return {arm: selected for arm, audit in audits.items()
            if (selected := [p for p in audit["problems"] if p.startswith(rows[arm]["trial"] + ":")])}


def inspect_existing(folder: Path, tasks: list[int]) -> dict:
    """Read existing plans, checkpoints and trajectories; never train or advance a workflow."""
    from src.utils.audit_experiment import audit
    reports = {}
    comparisons = []
    for task in tasks:
        track, trial = ("pd" if task < 4 else "lgd"), task % 4
        if track not in reports:
            reports[track] = {arm: audit(folder / f"recovery_{track}_{arm}.yaml")
                              for arm in ("reference", "resumed")}
        audits = reports[track]
        rows = {arm: report["trials"][trial] for arm, report in audits.items()}
        report = comparison_report(rows, track, trial)
        report["arm_audit_problems"] = selected_audit_problems(audits, rows)
        report["arm_audits_passed"] = not report["arm_audit_problems"]
        recorded_path = folder / f"recovery_{track}_{trial}.json"
        recorded = json.loads(recorded_path.read_text(encoding="utf-8")) if recorded_path.exists() else {}
        report["recorded_recovery_passed"] = recorded.get("passed")
        report["recorded_benchmark_smoke"] = recorded.get("benchmark_smoke")
        report["passed"] &= report["arm_audits_passed"]
        comparisons.append(compact_report(report))
    return {"action": "inspect_existing", "workflow": folder.name,
            "diagnostic_completed": True, "workflow_modified": False,
            "note": "Inspection only; this is not a passing recovery receipt or a new GPU check.",
            "comparisons": comparisons}


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
    report = comparison_report(rows, track, trial)
    report["arm_audit_problems"] = selected_audit_problems(reports, rows)
    report["arm_audits_passed"] = not report["arm_audit_problems"]
    report["benchmark_smoke"] = benchmark_smoke(Path(rows["resumed"]["path"]), track, trial)
    report["passed"] &= report["arm_audits_passed"] and report["benchmark_smoke"]["passed"]
    write_json(folder / f"recovery_{track}_{trial}.json", report)
    print(json.dumps(compact_report(report), indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("GPU recovery equivalence failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--task", type=int, help="Pair 0..7; omit with --inspect to read all eight")
    parser.add_argument("--inspect", action="store_true", help="CPU-only, read-only inspection; no workflow callbacks")
    args = parser.parse_args(argv)
    if not args.id.isalnum():
        parser.error("Invalid workflow identifier")
    if args.task is not None and not 0 <= args.task < 8:
        parser.error("Expected task 0..7")
    from src.utils.experiment0 import root, complete
    if args.inspect:
        tasks = [args.task] if args.task is not None else list(range(8))
        print(json.dumps(inspect_existing(root() / args.id, tasks), indent=2), flush=True)
        return 0  # Inspection succeeded; the failed workflow/receipt remains untouched.
    if args.task is None:
        parser.error("--task is required for GPU recovery checks")
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
