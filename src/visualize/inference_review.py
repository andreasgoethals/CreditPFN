"""Read isolated numerical-review receipts; incomplete evidence never means parity."""
from __future__ import annotations

import json

import numpy as np
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


def policy_deltas(data):
    """Subtract matched original AUC within policy/table/fold, before averaging."""
    keys = ["dataset", "fold", "policy", "state"]
    if data.duplicated(keys).any() or len(data) != 90:
        raise ValueError("Expected six native-context cells per task")
    expected = {(d, f, p, s) for d in data.dataset.unique() for f in range(5)
                for p in ("auto", "fp32_no_fa3") for s in ("base", "saved", "cpt")}
    if len(expected) != 90 or set(data[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Numerical-policy effects need three complete tables and five paired folds")
    if not np.isfinite(data.roc_auc.to_numpy(float)).all():
        raise ValueError("Nonfinite numerical-policy AUC")
    paired = data.pivot(index=["dataset", "fold", "policy"], columns="state", values="roc_auc").reset_index()
    paired.columns.name = None
    return paired.assign(effect=paired.cpt-paired.base, save_difference=paired.saved-paired.base)


def plot(coverage, data):
    if (data.empty or len(coverage) != 15 or not coverage.status.eq("passed").all()
            or "project_journals_audited" not in coverage
            or not coverage.project_journals_audited.eq(True).all()):
        return []
    paired = policy_deltas(data)
    means = data.groupby(["dataset", "policy", "state"]).roc_auc.mean().reset_index()
    effects = paired.groupby(["dataset", "policy"]).effect.mean().reset_index()
    datasets = sorted(means.dataset.unique())
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
    grid = fig.add_gridspec(2, len(datasets))
    axes = [fig.add_subplot(grid[0, i]) for i in range(len(datasets))]
    delta_ax = fig.add_subplot(grid[1, :])
    fig.suptitle("PD / TabICLv2: selected-table numerical diagnostic")
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
    for index, dataset in enumerate(datasets):
        table = paired[paired.dataset.eq(dataset)]
        folds = table.pivot(index="fold", columns="policy", values="effect")
        x = [index-style.PROTOCOL_POINT_OFFSET, index+style.PROTOCOL_POINT_OFFSET]
        for _, fold in folds.iterrows():
            delta_ax.plot(x, fold[["auto", "fp32_no_fa3"]], color=style.color("annotation"),
                alpha=style.TABLE_CURVE_ALPHA, linewidth=style.TABLE_CURVE_LINE)
        for position, (policy, key, marker) in zip(x, (("auto", "tabicl", "o"), ("fp32_no_fa3", "reference", "s"))):
            effect = effects.loc[effects.dataset.eq(dataset) & effects.policy.eq(policy), "effect"].iloc[0]
            delta_ax.plot(position, effect, marker=marker, color=style.color(key),
                markersize=style.CURVE_MARKER_SIZE, linestyle="")
    delta_ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
    delta_ax.set_xticks(range(len(datasets)), datasets)
    delta_ax.set_ylabel("CPT − original AUC")
    delta_ax.set_title("Paired effects under each numerical policy")
    axes[0].legend()
    return [cp.Page("tabicl_numerical_policy", fig,
        "TabICL five-fold AUC under automatic numerical settings and FP32 tensors with "
        "Flash Attention 3 disabled. The saved control writes unchanged original weights "
        "through the CPT serialization path. Contexts, preprocessing, inference views and "
        "test rows match between states and policies. The lower panel subtracts original from CPT AUC within policy and fold. "
        "Thin gray segments connect each of five paired fold effects; colored markers show their mean, not a confidence interval. "
        "The three post-hoc diagnostic tables were "
        "chosen to investigate observed numerical sensitivity and do not estimate its "
        "prevalence across credit tasks. Boundary probes are excluded from this figure.",
        {"All native-context cells": data, "Five-fold policy means": means,
         "Completion and parity": coverage, "Paired fold effects": paired, "Mean policy effects": effects},
        {"measurement": "selected_table_precision_diagnostic", "metric": "roc_auc", "outer_folds": 5,
         "tables": datasets, "selection": "post_hoc", "boundary_probes_included": False})]
