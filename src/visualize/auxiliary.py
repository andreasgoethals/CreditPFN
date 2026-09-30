"""Read experiment-0 auxiliary evidence without starting or repairing any job."""
from __future__ import annotations

import json
import pandas as pd

from src.visualize.inputs import analysis_location
from src.visualize import style, training_viz
from src.utils.paths import manifests_dir
from src.utils.auxiliary_check import cases, policy


def planned_checks():
    limits = policy()
    return pd.DataFrame([dict(track=track.upper(), case=case,
        bases=4, updates=int(limits.target_updates), interrupt_after=int(limits.interrupt_after) if case.endswith("resumed") else None,
        row_cap=int(limits.row_cap), purpose="Resume equivalence" if case.endswith("resumed") else "Seed sensitivity" if case.startswith("seed") else "Sampling execution")
        for track in ("pd", "lgd") for case in cases(track)])


def evidence():
    root = analysis_location("experiment0", "manifests", manifests_dir("experiment0")) / "workflow"
    states = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in root.glob("*/state.json")]
    states = [(path, state) for path, state in states if state.get("part") == "auxiliary"]
    if not states:
        return {"status": "Not run: no downloaded auxiliary workflow evidence."}, pd.DataFrame(), pd.DataFrame()
    path, state = max(states, key=lambda item: item[0].stat().st_mtime)
    checks, arms = [], []
    for file in path.parent.glob("auxiliary_*_*.json"):
        r = json.loads(file.read_text(encoding="utf-8"))
        parsed = training_viz.parse_trial_name(r["arms"][0]["trial"])
        base = style.model_label(training_viz.compact_base(parsed.base_short))
        checks.append(dict(track=r["track"].upper(), base=base, passed=r["passed"],
            initial_monitor_equal=r["seed"]["fixed_baseline_equal"], seed_weights_differ=r["seed"]["trained_weights_differ"],
            full_pass_recovery=r["recovery"]["full_pass"]["passed"], accumulate_recovery=r["recovery"]["accumulate"]["passed"],
            exposure_check=r["exposure_passed"],
            five_fold_benchmark=r["benchmark"]["passed"]))
        arms.extend(dict(track=r["track"].upper(), base=base, **arm) for arm in r["arms"])
    scope = {"workflow": state["id"], "status": state["status"], "reported_tasks": len(checks), "expected_tasks": 8,
             "fingerprint": state["fingerprint"]}
    return scope, pd.DataFrame(checks), pd.DataFrame(arms)
