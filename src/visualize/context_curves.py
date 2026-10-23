"""Experiment-4 context curves from verified, completed task exports.

SQLite journals can contain additional committed cells, but only the small CSV
exports with matching DATA receipts enter publication. Neither input is modified.
All comparisons pair original/CPT first, require five folds, then weight tables
equally. Fixed-cohort panels avoid confounding context size with dataset mix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.dataset_names import display_name
from src.utils.experiment import digest_json
from src.utils.paths import REPO_ROOT, manifests_dir, results_dir
from src.visualize import campaign as cp, style
from src.visualize.inputs import analysis_location, read_csv

FAMILIES = {"tabpfn": "TabPFN v3", "tabicl": "TabICLv2"}
KEYS = ["track", "family", "partition", "dataset", "fold", "context_rows"]
COHORT_HORIZONS = (128, 2048, 50000)


@dataclass
class ContextStudy:
    config: dict
    cells: pd.DataFrame
    inventory: pd.DataFrame


def protocol_table(study):
    cfg = study.config
    recipe = cfg["source_recipe"]
    return pd.DataFrame([
        ("Models", "Explicit TabPFN v3 and TabICLv2 original/CPT pairs"),
        ("CPT source", f"{cfg['source_run']}; full updates; LR {recipe['learning_rate']:g}; L2-SP {recipe['l2sp_lambda']:g}; seed {recipe['seed']}; {recipe['successful_updates']} updates"),
        ("Dataset exclusion", "Each table uses its original experiment-1 held-out partition's checkpoint"),
        ("Outer test folds", cfg["outer_folds"]),
        ("Context rows", ", ".join(map(str, cfg["context_rows"]))),
        ("Additional endpoint", "Available context pool, capped at 50,000, without duplicate sizes"),
        ("Validation labels", f"Fixed {cfg['inner_validation_fraction']:.0%} of each outer training fold, separate from context"),
        ("Test rows", "All outer-test rows; unchanged across context budgets and original/CPT"),
        ("Inference views", str(cfg["inference_views"])),
        ("Aggregation", "Five paired folds per table; equal-weight table means; fixed cohort within each curve"),
    ], columns=["setting", "value"])


def _check_cells(frame, task, receipt, cfg):
    from src.eval.context_curve import budgets
    needed = {*KEYS} - {"dataset"} | {"dataset_id", "state", "available_context_rows", "test_rows",
        "validation_rows", "context_indices_sha256", "test_indices_sha256"}
    if not needed <= set(frame) or frame.empty or len(frame) != receipt["cells"]:
        raise ValueError("Incomplete context task export")
    for key in ("track", "family", "partition", "fold", "dataset_id"):
        if not frame[key].eq(task[key]).all():
            raise ValueError(f"Context task identity mismatch: {key}")
    if (frame.duplicated(["context_rows", "state"]).any()
            or set(frame.state) != {"base", "cpt"}):
        raise ValueError("Duplicate or unknown context states")
    for key in ("available_context_rows", "test_rows", "validation_rows", "test_indices_sha256"):
        if frame[key].nunique(dropna=False) != 1 or frame[key].isna().any():
            raise ValueError(f"Context budget changed the fixed fold: {key}")
    if (frame[["test_rows", "validation_rows", "available_context_rows"]] <= 0).any().any():
        raise ValueError("Nonpositive context fold sizes")
    expected = {(n, state) for n in budgets(int(frame.available_context_rows.iloc[0]), cfg) for state in ("base", "cpt")}
    if set(zip(frame.context_rows, frame.state)) != expected:
        raise ValueError("Context export does not contain its complete declared grid")
    if (frame.context_indices_sha256.isna().any()
            or not frame.groupby("context_rows").context_indices_sha256.nunique().eq(1).all()):
        raise ValueError("Original and CPT context rows differ")


def load_context_study():
    cfg = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "config/experiment4/context.yaml"), resolve=True)
    mroot = analysis_location("experiment4", "manifests", manifests_dir("experiment4"))
    rroot = analysis_location("experiment4", "results", results_dir(experiment="experiment4")) / cfg["run_name"]
    plan_path = mroot / "plans" / (cfg["run_name"] + ".json")
    exports = {p.name: p for p in rroot.glob("task_*.csv.gz")}
    journals = list(rroot.glob("task_*.sqlite"))
    if not plan_path.is_file():
        return ContextStudy(cfg, pd.DataFrame(), pd.DataFrame([dict(
            status="Plan unavailable", tasks=0, exports=len(exports), journals=len(journals))]))
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    if digest_json({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
        raise ValueError("Context analysis plan checksum failed")
    if plan["config"] != cfg:
        raise ValueError("Context plan differs from the notebook's declared protocol")
    records, rows = [], []
    for index, task in enumerate(plan["tasks"]):
        name = f"task_{index:04d}"
        export = exports.get(name + ".csv.gz")
        receipt_path = mroot / "completed" / cfg["run_name"] / (name + ".json")
        status = "Export unavailable" if export is None else "Receipt unavailable"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf8"))
            if (receipt.get("plan_sha256") != plan["sha256"] or receipt.get("task") != index
                    or receipt.get("passed") is not True):
                raise ValueError("Context completion receipt does not match its task/plan")
            if export is not None:
                frame = read_csv(export)
                _check_cells(frame, task, receipt, cfg)
                records.append(frame.rename(columns={"dataset_id": "dataset"}).assign(
                    dataset=frame.dataset_id.map(display_name)))
                status = "Verified export"
        rows.append(dict(task=index, track=task["track"], family=task["family"],
            partition=task["partition"], dataset=display_name(task["dataset_id"]), fold=task["fold"], status=status))
    expected_names = {f"task_{i:04d}.csv.gz" for i in range(len(plan["tasks"]))}
    if set(exports) - expected_names:
        raise ValueError("Unexpected task exports in the selected context run")
    return ContextStudy(cfg, pd.concat(records, ignore_index=True) if records else pd.DataFrame(), pd.DataFrame(rows))


def paired_tables(study, metric):
    """Fold pairing precedes table averaging; no four-of-five approximations."""
    data = study.cells
    if data.empty or metric not in data:
        return pd.DataFrame()
    data = data.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    if data.duplicated(KEYS + ["state"]).any():
        raise ValueError("Duplicate context metric cell")
    left, right = [data[data.state.eq(s)][KEYS + [metric]] for s in ("base", "cpt")]
    paired = left.merge(right, on=KEYS, validate="one_to_one", suffixes=("_base", "_cpt"))
    paired = paired[np.isfinite(paired[[metric+"_base", metric+"_cpt"]]).all(axis=1)]
    rows = []
    keys = [k for k in KEYS if k != "fold"]
    for key, group in paired.groupby(keys):
        if set(group.fold) != set(range(5)):
            continue
        original, adapted = group[[metric+"_base", metric+"_cpt"]].mean()
        sign = -1 if metric in ("brier_score", "log_loss", "ece", "elapsed_seconds", "fit_seconds", "rmse", "mae") else 1
        rows.append(dict(zip(keys, key), original=original, adapted=adapted,
                         gain=sign*(adapted-original), metric=metric, folds=5))
    return pd.DataFrame(rows)


def fixed_cohort(study, table, track, horizon):
    """Require the same datasets at every nominal budget and for both families."""
    if table.empty:
        return table.copy()
    sizes = [n for n in study.config["context_rows"] if n <= horizon]
    part = table[table.track.eq(track) & table.context_rows.isin(sizes)].copy()
    expected = {(f, n) for f in study.config["families"] for n in sizes}
    valid = [d for d, g in part.groupby("dataset") if set(zip(g.family, g.context_rows)) == expected]
    return part[part.dataset.isin(valid)].copy()


def plot_context_curves(study):
    pages = []
    for track, metric, label in (("pd", "roc_auc", "AUC"), ("lgd", "r2", "R²")):
        paired = paired_tables(study, metric)
        for horizon in COHORT_HORIZONS:
            data = fixed_cohort(study, paired, track, horizon)
            if data.empty:
                continue
            n = data.dataset.nunique()
            fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
            fig.suptitle(f"{track.upper()}: context up to {horizon:,} rows · {n} fixed table{'s' if n != 1 else ''}")
            curves = []
            for col, family in enumerate(study.config["families"]):
                part = data[data.family.eq(family)]
                curve = part.groupby("context_rows")[["original", "adapted", "gain"]].mean()
                color = style.color("v3" if family == "tabpfn" else "tabicl")
                for state, name, line in (("original", "Original", "--"), ("adapted", "CPT", "-")):
                    axes[0, col].plot(curve.index, curve[state], color=color, linestyle=line, linewidth=style.THIN_LINE, label=name)
                for _, table in part.groupby("dataset"):
                    table = table.sort_values("context_rows")
                    axes[1, col].plot(table.context_rows, table.gain, color=style.color("annotation"),
                        alpha=style.TABLE_CURVE_ALPHA, linewidth=style.TABLE_CURVE_LINE)
                axes[1, col].plot(curve.index, curve.gain, color=color, linewidth=style.MEAN_CURVE_LINE)
                axes[1, col].axhline(0, color=style.color("reference"), linestyle=":")
                axes[0, col].set_title(FAMILIES[family]); axes[0, col].set_ylabel(label)
                axes[1, col].set_ylabel("CPT − original " + label)
                for ax in axes[:, col]:
                    ax.set_xscale("log"); ax.set_xlabel("Labeled context rows")
                curves.append(curve.reset_index().assign(family=family, datasets=n))
            from matplotlib.lines import Line2D
            handles, labels = axes[0, 0].get_legend_handles_labels()
            handles += [Line2D([], [], color=style.color("annotation"), linewidth=style.TABLE_CURVE_LINE),
                        Line2D([], [], color=style.color("mean"), linewidth=style.MEAN_CURVE_LINE)]
            labels += ["Individual table effect", "Equal-table mean effect"]
            fig.legend(handles, labels, loc="outside lower center", ncol=2)
            pages.append(cp.Page(f"context_{track}_through_{horizon}", fig,
                f"{track.upper()}. Original and fixed-recipe CPT {label}, with their paired difference below. "
                f"The same {n} tables contribute at every displayed context size and to both model families. "
                "Scores are averaged over five paired outer folds within table, then equally across tables. "
                "In the lower panels, thin gray curves retain each table's paired effect; the thicker colored curve is their equal-table mean. "
                "These curves are not uncertainty bands or repeated-context variance estimates. "
                "Validation labels have a fixed additional budget; all outer-test rows are unchanged. "
                "The horizon identifies the table cohort, so levels across different horizon panels need not agree. "
                "TabICLv2 uses the recorded automatic numerical policy; the selected-table precision diagnostic qualifies transfer attribution.",
                {"Plotted means": pd.concat(curves, ignore_index=True), "Complete paired table values": data},
                {"measurement": "five_fold_context_benchmark", "metric": metric, "aggregation_unit": "table",
                 "context_horizon": int(horizon), "fixed_tables": sorted(data.dataset.unique()),
                 "additional_validation_fraction": float(study.config["inner_validation_fraction"])}))
    return pages


def plot_context_support(study):
    parts = []
    for track, metric in (("pd", "roc_auc"), ("lgd", "r2")):
        data = paired_tables(study, metric)
        if not data.empty:
            parts.append(data[data.track.eq(track) & data.context_rows.isin(study.config["context_rows"])])
    data = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if data.empty:
        return []
    support = data.groupby(["track", "family", "context_rows"]).dataset.nunique().rename("tables").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(style.WIDTH_FULL), layout="constrained")
    for ax, track in zip(axes, ("pd", "lgd")):
        group = support[support.track.eq(track)]
        if group.empty:
            ax.remove(); continue
        for index, (family, arm) in enumerate(group.groupby("family")):
            ax.plot(arm.context_rows, arm.tables, label=FAMILIES[family],
                    color=style.color("v3" if family == "tabpfn" else "tabicl"),
                    marker=style.PANEL_MARKERS[index], markersize=style.CURVE_MARKER_SIZE,
                    linestyle=style.FACTOR_ARM_STYLES[index])
        ax.set_title(track.upper()); ax.set_xscale("log"); ax.set_xlabel("Labeled context rows")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Tables with all five paired folds"); ax.legend()
    return [cp.Page("context_support", fig,
        "Number of tables with finite original/CPT measurements in all five folds at each nominal context size. "
        "Counts reflect both available table sizes and completed exports; they are not live scheduler status. "
        "Performance panels separately fix the cohort over their full displayed range.", {"Paired support": support})]


def secondary_tables(study):
    return {metric: paired_tables(study, metric) for metric in
            ("roc_auc", "r2", "brier_score", "log_loss", "ece", "f1", "rmse", "mae", "elapsed_seconds", "fit_seconds")}
