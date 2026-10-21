"""Read isolated numerical-review receipts; incomplete evidence never means parity."""
from __future__ import annotations

import json

import pandas as pd

from src.data.dataset_names import display_name
from src.utils.experiment import digest_json
from src.utils.paths import manifests_dir
from src.visualize import campaign as cp, style
from src.visualize.inputs import analysis_location

NOTE = (
    "This inference-only diagnostic uses two tables with observed numerical sensitivity "
    "and one control, not a new representative benchmark. Original, unchanged-save and "
    "CPT checkpoints use identical contexts and all test rows of five outer folds. "
    "The boundary probes use a fixed 256-row query subset in one fold and are reported "
    "separately. A passing parity check validates saving/reloading; differences between "
    "precision policies are measurements, not automatic failures. FP32 denotes tensor "
    "dtype with FA3 disabled; the recorded TF32 setting is also relevant."
)


def evidence():
    root = analysis_location("experiment0", "manifests", manifests_dir("experiment0")) / "inference_review"
    plans = list(root.glob("*/plan.json"))
    if not plans:
        return pd.DataFrame([dict(status="not run", expected_tasks=15)]), pd.DataFrame(), pd.DataFrame()
    path = max(plans, key=lambda p: p.stat().st_mtime)
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("sha256") != digest_json({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("Numerical-review plan checksum mismatch")
    coverage, records = [], []
    for index, task in enumerate(plan["tasks"]):
        receipt_path = path.parent / f"task_{index:02d}.json"
        meta = dict(task=index, dataset=display_name(task["dataset_id"]), fold=task["fold"])
        if not receipt_path.is_file():
            coverage.append(dict(**meta, status="missing receipt"))
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["plan_sha256"] != plan["sha256"] or receipt["task"] != index:
            raise ValueError("Numerical-review receipt belongs to another task/plan")
        coverage.append(dict(**meta, status="passed" if receipt["passed"] else "parity failed"))
        records.extend(dict(**meta, **{k: v for k, v in row.items() if k != "fold"})
                       for row in receipt["cells"])
    data = pd.DataFrame(records)
    audit_path = path.parent / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    audited = (audit.get("passed") is True and audit.get("plan_sha256") == plan["sha256"]
               and audit.get("tasks") == [dict(task=i, passed=True) for i in range(len(plan["tasks"]))])
    coverage = pd.DataFrame(coverage).assign(project_journals_audited=audited)
    benchmark = data[data.scope.eq("benchmark")].copy() if not data.empty else pd.DataFrame()
    return coverage, benchmark, data[data.scope.eq("boundary")].copy() if not data.empty else pd.DataFrame()


def plot(coverage, data):
    if (data.empty or len(coverage) != 15 or not coverage.status.eq("passed").all()
            or "project_journals_audited" not in coverage
            or not coverage.project_journals_audited.eq(True).all()):
        return []
    keys = ["dataset", "fold", "policy", "state"]
    if data.duplicated(keys).any() or len(data) != 90:
        raise ValueError("Expected six native-context cells per task")
    means = data.groupby(["dataset", "policy", "state"]).roc_auc.mean().reset_index()
    datasets = sorted(means.dataset.unique())
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(datasets), figsize=style.figsize(style.WIDTH_FULL, style.SMALL_RATIO),
                             squeeze=False, layout="constrained")
    axes = axes[0]
    fig.suptitle("PD / TabICL: checkpoint effects and numerical policy")
    states = ["base", "saved", "cpt"]
    for ax, dataset in zip(axes, datasets):
        for policy, key, marker in (("auto", "tabicl", "o"), ("fp32_no_fa3", "reference", "s")):
            part = means[means.dataset.eq(dataset) & means.policy.eq(policy)].set_index("state")
            ax.plot(range(3), part.loc[states, "roc_auc"], color=style.color(key),
                    marker=marker, markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE,
                    label="Automatic" if policy == "auto" else "FP32 / no FA3")
        ax.set_xticks(range(3), ["Original", "Saved", "CPT"])
        ax.set_title(dataset)
        ax.set_ylabel("Mean five-fold AUC")
    axes[0].legend()
    return [cp.Page("tabicl_numerical_policy", fig,
        "TabICL five-fold AUC under automatic numerical settings and FP32 tensors with "
        "Flash Attention 3 disabled. The saved control writes unchanged original weights "
        "through the CPT serialization path. Contexts, preprocessing, inference views and "
        "test rows match between states and policies. The three diagnostic tables were "
        "chosen to investigate observed numerical sensitivity and do not estimate its "
        "prevalence across credit tasks. Boundary probes are excluded from this figure.",
        {"All native-context cells": data, "Five-fold policy means": means,
         "Completion and parity": coverage})]
