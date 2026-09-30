"""Short GPU controls for seed changes and disjoint full-pass/accumulation recovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.train.config import load_train_config
from src.train.corpus import split_from_cfg
from src.utils.atomic import write_json
from src.utils.paths import REPO_ROOT, training_dir


def policy():
    cfg = OmegaConf.load(REPO_ROOT / "config/experiment0/auxiliary.yaml")
    if not 0 < cfg.interrupt_after < cfg.target_updates or cfg.row_cap < 8:
        raise ValueError("Invalid auxiliary budget or row cap")
    return cfg


def cases(track):
    # LGD's seed-42 arm already has the experiment-3 stratified policy.
    return ["seed42", "seed43", *(["one_sample"] if track == "pd" else []),
            "full_pass", "accumulate", "full_pass_resumed", "accumulate_resumed"]


def config_path(folder, track, case):
    return folder / f"auxiliary_{track}_{case}.yaml"


def select_tables(cfg, limits):
    """Restrict fold zero by size alone, preserving its training/held-out roles."""
    split = split_from_cfg(cfg)
    candidates = sorted((d for d in split.train if d.n_rows > limits.row_cap),
                        key=lambda d: (d.n_rows, d.dataset_id))
    train = candidates[:limits.training_tables]
    held = sorted(split.test, key=lambda d: (d.n_rows, d.dataset_id))[:limits.held_out_tables]
    if len(train) != limits.training_tables or len(held) != limits.held_out_tables:
        raise ValueError("Insufficient tables for the auxiliary multi-chunk check")
    if sum(d.n_rows for d in train) > limits.max_training_rows:
        raise ValueError("Auxiliary training corpus exceeds its CPU-checked size limit; no GPU work launched")
    if {d.dataset_id for d in train} & {d.dataset_id for d in held}:
        raise ValueError("Auxiliary training and held-out tables overlap")
    return train, held


def make_configs(folder):
    limits = policy()
    paths = []
    for track in ("pd", "lgd"):
        source = load_train_config(config_path=f"config/experiment3/{track}.yaml")
        train, held = select_tables(source, limits)
        for case in cases(track):
            seed_case = case.startswith("seed")
            cfg = load_train_config(config_path=f"config/experiment{2 if seed_case else 3}/{track}.yaml")
            cfg.seed = 43 if case == "seed43" else 42
            cfg.experiment.training_seeds = [cfg.seed]
            cfg.experiment.output_group = "experiment0"
            cfg.run_name = f"cpt_pilot_aux_{folder.name[:12]}_{case}"
            cfg.corpus.n_splits = 1
            cfg.corpus.train_dataset_ids = [d.dataset_id for d in train]
            cfg.corpus.test_dataset_ids = [d.dataset_id for d in held]
            cfg.checkpoint.trained_dir = "checkpoints/trained/experiment0"
            cfg.tunable.epoch_pass_modes = ["one_sample" if seed_case else case.removesuffix("_resumed")]
            cfg.train.deterministic = True
            cfg.train.max_rows_per_step = limits.row_cap
            cfg.train.target_total_steps = limits.target_updates
            cfg.train.trajectory_steps = [0, limits.interrupt_after, limits.target_updates]
            cfg.train.max_epochs_for_step_budget = 20
            cfg.train.epochs = 6
            cfg.train.retention_panel = "smoke"
            cfg.train.epoch_eval_subsample_samples = limits.monitor_rows
            cfg.train.epoch_eval_n_estimators = limits.monitor_members
            cfg.train.epoch_eval_n_estimators_tabicl = limits.monitor_members
            cfg.train.recovery_every_updates = 1
            path = config_path(folder, track, case)
            with path.open("x", encoding="utf-8") as stream:
                OmegaConf.save(cfg, stream)
            paths.append(path)
    return paths


def trajectory(row, track):
    return pd.read_csv(training_dir(track, row["trial"] + ".trajectory.csv", experiment="experiment0"))


def seed_report(before, after, curves):
    """Seeds must change optimization, but not the fixed monitor's update-zero input."""
    from src.utils.recovery_check import compare, compare_trajectories
    weights = compare(Path(before["path"]), Path(after["path"]))
    baseline = compare_trajectories(*(v[v.successful_updates.eq(0)] for v in curves), milestones=[0])
    changed = weights["max_non_diagnostic_difference"] > 0
    return {"passed": baseline["passed"] and changed,
            "fixed_baseline_equal": baseline["passed"], "trained_weights_differ": changed,
            "max_weight_difference": weights["max_non_diagnostic_difference"]}


def run(folder, track, trial):
    from src.train.recovery import configure_execution
    from src.utils.audit_experiment import audit
    from src.utils.recovery_check import compare, compare_trajectories, benchmark_smoke
    limits = policy()
    execution = configure_execution(True)
    paths = {case: config_path(folder, track, case) for case in cases(track)}
    # A fresh workflow is required; do not turn a skipped old arm into a passing test.
    if any(audit(path)["trials"][trial]["status"] != "PENDING" for path in paths.values()):
        raise RuntimeError("Auxiliary task already has output; inspect it instead of rerunning the same task")
    common = [sys.executable, "-u", "scripts/train_pipeline.py", "--split-index", "0", "--trial-index", str(trial)]
    if os.environ.get("CREDITPFN_ACTIVE_LOG"):
        common += ["--log-path", os.environ["CREDITPFN_ACTIVE_LOG"]]
    rows, curves, outcomes = {}, {}, []
    for case, path in paths.items():
        cfg = load_train_config(config_path=str(path))
        env = dict(os.environ, PYTHONHASHSEED=str(cfg.seed), CREDITPFN_SEGMENT_SECONDS="0")
        for key in ("CREDITPFN_STOP_AFTER_UPDATES", "CREDITPFN_STOP_REQUESTED", "CREDITPFN_FLOW_ID", "CREDITPFN_FLOW_PHASE"):
            env.pop(key, None)
        command = [*common, "--config", str(path)]
        if case.endswith("_resumed"):
            stopped = subprocess.run(command, env=dict(env, CREDITPFN_STOP_AFTER_UPDATES=str(limits.interrupt_after)))
            if stopped.returncode != 75:
                raise RuntimeError(f"Expected interruption exit 75, got {stopped.returncode}")
        subprocess.run(command, env=env, check=True)
        report = audit(path)
        row = rows[case] = report["trials"][trial]
        problems = [p for p in report["problems"] if p.startswith(row["trial"] + ":")]
        if row["status"] != "OK" or row.get("successful_updates") != limits.target_updates or problems:
            raise RuntimeError(f"Auxiliary arm {case} failed its saved-output audit: {problems}")
        curves[case] = trajectory(row, track)
        drift = float(curves[case].weight_drift.iloc[-1])
        if not np.isfinite(drift) or drift <= 0:
            raise RuntimeError(f"Auxiliary arm {case} did not record finite positive parameter movement")
        outcomes.append(dict(case=case, seed=int(cfg.seed), sampling=str(cfg.tunable.epoch_pass_modes[0]),
                             context_sampling=str(cfg.train.context_sampling), trial=row["trial"],
                             updates=row["successful_updates"], processed_rows=int(curves[case].processed_rows.iloc[-1])))
    seed = seed_report(rows["seed42"], rows["seed43"], [curves["seed42"], curves["seed43"]])
    recovery = {}
    for mode in ("full_pass", "accumulate"):
        weights = compare(Path(rows[mode]["path"]), Path(rows[mode+"_resumed"]["path"]))
        monitors = compare_trajectories(curves[mode], curves[mode+"_resumed"],
                                       milestones=[0, limits.interrupt_after, limits.target_updates])
        exposure_equal = curves[mode].processed_rows.tolist() == curves[mode+"_resumed"].processed_rows.tolist()
        recovery[mode] = dict(passed=weights["passed"] and monitors["passed"] and exposure_equal,
                              weights=weights, monitors=monitors, exposure_equal=exposure_equal)
    exposures = {r["case"]: r["processed_rows"] for r in outcomes}
    single = "one_sample" if track == "pd" else "seed42"
    exposure_passed = exposures["accumulate"] > exposures[single] >= exposures["full_pass"] > 0
    smoke = benchmark_smoke(Path(rows["seed43"]["path"]), track, trial, workflow=folder.name, kind="auxiliary_smoke")
    result = dict(track=track, trial=trial, execution=execution, arms=outcomes, seed=seed, recovery=recovery,
                  exposure_passed=exposure_passed, benchmark=smoke,
                  passed=seed["passed"] and all(r["passed"] for r in recovery.values()) and smoke["passed"] and exposure_passed)
    write_json(folder / f"auxiliary_{track}_{trial}.json", result)
    print(json.dumps({k: result[k] for k in ("track", "trial", "passed", "seed", "arms")}, indent=2), flush=True)
    if not result["passed"]:
        raise RuntimeError("Auxiliary GPU check failed; research experiments remain gated")


def audit_reports(folder):
    reports = []
    for track in ("pd", "lgd"):
        for trial in range(4):
            path = folder / f"auxiliary_{track}_{trial}.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            expected = set(cases(track))
            if (result.get("track") != track or result.get("trial") != trial or not result.get("passed")
                    or {r["case"] for r in result.get("arms", [])} != expected
                    or len(result["arms"]) != len(expected)
                    or not result.get("exposure_passed")
                    or not result["seed"]["passed"] or not result["benchmark"]["passed"]
                    or not all(result["recovery"][m]["passed"] for m in ("full_pass", "accumulate"))):
                raise RuntimeError("Auxiliary reports incomplete, mismatched or failed")
            reports.append(result)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--task", required=True, type=int, choices=range(8))
    args = parser.parse_args(argv)
    if not args.id.isalnum():
        parser.error("Invalid workflow identifier")
    from src.utils.experiment0 import root, complete, locked
    with locked(args.id) as (_, state):
        if state["phase"] != "auxiliary":
            raise RuntimeError("Task does not belong to the auxiliary workflow")
    track, trial = ("pd" if args.task < 4 else "lgd"), args.task % 4
    rc = 1
    try:
        run(root() / args.id, track, trial)
        rc = 0
    finally:
        complete(args.id, "auxiliary", track, trial, rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
