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
        ("Models", f"Explicit TabPFN {cfg['base_version']} original/CPT pairs" if cfg.get("base_version") else "Explicit TabPFN v3 and TabICLv2 original/CPT pairs"),
        ("CPT source", f"{cfg['source_run']}; full updates; LR {recipe['learning_rate']:g}; L2-SP {recipe['l2sp_lambda']:g}; seed {recipe['seed']}; {recipe['successful_updates']} updates"),
        ("Dataset exclusion", "Each table uses its original experiment-1 held-out partition's checkpoint"),
        ("Outer test folds", cfg["outer_folds"]),
        ("Context rows", ", ".join(map(str, cfg["context_rows"]))),
        ("Additional endpoint", f"Available context pool, capped at {cfg['max_context_rows']:,}, without duplicate sizes"),
        ("Validation labels", f"Fixed {cfg['inner_validation_fraction']:.0%} of each outer training fold, separate from context"),
        ("Test rows", "All outer-test rows; unchanged across context budgets and original/CPT"),
        ("Inference views", str(cfg["inference_views"])),
        ("TabICL inference", "not applicable" if "tabicl" not in cfg["families"] else
         ("FP32; FA3 disabled" if cfg.get("tabicl_inference_options") == {"use_amp":False,"use_fa3":False} else "automatic; precision sensitivity caveat applies")),
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


def load_context_study(*, cfg_override=None):
    cfg = OmegaConf.to_container(cfg_override if cfg_override is not None else OmegaConf.load(REPO_ROOT / "config/experiment4/context.yaml"), resolve=True)
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
    study = ContextStudy(cfg, pd.concat(records, ignore_index=True) if records else pd.DataFrame(), pd.DataFrame(rows))
    pointer = mroot / "active_context.json"
    if cfg_override is None and pointer.is_file():
        active = json.loads(pointer.read_text(encoding="utf-8"))
        if active.get("family") != "tabicl" or active["config"].get("tabicl_inference_options") != {"use_amp": False, "use_fa3": False}:
            raise ValueError("Invalid active context precision policy")
        replacement_plan = json.loads((mroot/"plans"/(active["config"]["run_name"]+".json")).read_text(encoding="utf-8"))
        if replacement_plan["sha256"] != active["plan_sha256"]:
            raise ValueError("Published context plan changed")
        from src.utils.experiment import file_digest
        debug = analysis_location("experiment0", "manifests", manifests_dir("experiment0"))
        inspection = debug/"evaluation_refresh"/active["config"]["refresh_id"]/"inspection.json"
        if file_digest(inspection) != active["inspection_sha256"]:
            raise ValueError("Published context inspection is missing or changed")
        replacement = load_context_study(cfg_override=OmegaConf.create(active["config"]))
        if not replacement.inventory.status.eq("Verified export").all():
            raise ValueError("Published FP32 context evidence is incomplete")
        keys = ["track", "family", "partition", "dataset", "fold", "context_rows", "state"]
        before = study.cells[study.cells.family.eq("tabicl")]
        if set(map(tuple, before[keys].to_numpy())) != set(map(tuple, replacement.cells[keys].to_numpy())):
            raise ValueError("Published FP32 context refresh changed comparison coverage")
        study.cells = pd.concat([study.cells[~study.cells.family.eq("tabicl")], replacement.cells], ignore_index=True)
        study.inventory = pd.concat([study.inventory[~study.inventory.family.eq("tabicl")], replacement.inventory], ignore_index=True)
        study.config = dict(cfg, tabicl_inference_options=active["config"]["tabicl_inference_options"], precision_refresh=active["config"]["refresh_id"])
    return study


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


def plot_context_curves(study, *, selected_track=None, horizons=None):
    """Two metrics, absolute scores and paired effects in four readable panels."""
    from matplotlib.lines import Line2D
    pages = []
    for track, metrics in (("pd", (("roc_auc", "AUC"), ("brier_score", "Brier score"))),
                           ("lgd", (("r2", "R²"), ("rmse", "RMSE")))):
        if selected_track is not None and track != selected_track:
            continue
        for horizon in sorted({min(n, study.config["max_context_rows"]) for n in (COHORT_HORIZONS if horizons is None else horizons)}):
            pairs = {m: fixed_cohort(study, paired_tables(study, m), track, horizon) for m, _ in metrics}
            available = [set(d.dataset) if not d.empty else set() for d in pairs.values()]
            cohort = set.intersection(*available) if available else set()
            if not cohort:
                continue
            n = len(cohort)
            fig, axes = plt.subplots(2, 2, squeeze=False, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
            fig.suptitle(f"{track.upper()}: context up to {horizon:,} rows · {n} fixed table{'s' if n != 1 else ''}")
            curves, values = [], []
            for row, (metric, label) in enumerate(metrics):
                data = pairs[metric]
                if data.empty:
                    axes[row, 0].text(.5, .5, f"{label} unavailable", transform=axes[row, 0].transAxes, ha="center")
                    axes[row, 1].set_visible(False)
                    continue
                data = data[data.dataset.isin(cohort)]
                values.append(data)
                for family in study.config["families"]:
                    part = data[data.family.eq(family)]
                    curve = part.groupby("context_rows")[["original", "adapted", "gain"]].mean()
                    color = style.color(study.config.get("base_version", "v3") if family == "tabpfn" else "tabicl")
                    for state, line in (("original", "--"), ("adapted", "-")):
                        axes[row, 0].plot(curve.index, curve[state], color=color, linestyle=line, linewidth=style.THIN_LINE)
                    for _, table in part.groupby("dataset"):
                        table = table.sort_values("context_rows")
                        axes[row, 1].plot(table.context_rows, table.gain, color=color,
                            alpha=style.TABLE_CURVE_ALPHA, linewidth=style.TABLE_CURVE_LINE)
                    axes[row, 1].plot(curve.index, curve.gain, color=color, linewidth=style.MEAN_CURVE_LINE)
                    curves.append(curve.reset_index().assign(family=family, datasets=n, metric=metric))
                axes[row, 0].set_ylabel(label)
                axes[row, 1].set_ylabel(("Original − CPT " if metric in ("brier_score", "rmse") else "CPT − original ") + label)
                axes[row, 1].axhline(0, color=style.color("reference"), linestyle=":")
            axes[0, 0].set_title("Original and CPT performance")
            axes[0, 1].set_title("Paired effect · positive favors CPT")
            for ax in axes.flat:
                ax.set_xscale("log"); ax.set_xlabel("Labeled context rows")
            handles = [Line2D([], [], color=style.color(study.config.get("base_version", "v3") if f == "tabpfn" else "tabicl"),
                label="TabPFN " + study.config.get("base_version", "v3") if f == "tabpfn" else FAMILIES[f]) for f in study.config["families"]]
            handles += [Line2D([], [], color=style.color("reference"), linestyle="--", label="Original"),
                        Line2D([], [], color=style.color("reference"), label="CPT"),
                        Line2D([], [], color=style.color("annotation"), linewidth=style.TABLE_CURVE_LINE, label="Individual table effect"),
                        Line2D([], [], color=style.color("mean"), linewidth=style.MEAN_CURVE_LINE, label="Equal-table mean effect")]
            fig.legend(handles=handles, loc="outside lower center", ncol=3)
            pages.append(cp.Page(f"context_{track}_through_{horizon}", fig,
                f"{track.upper()}. Original and fixed-recipe CPT {metrics[0][1]} and {metrics[1][1]} (left), with their paired effects (right). "
                f"The same {n} tables contribute at every displayed context size and to every displayed model. "
                "Scores are averaged over five paired outer folds within table, then equally across tables. "
                "Thin colored curves retain individual table effects; thick curves show their equal-table mean. "
                "These curves are not uncertainty bands or repeated-context variance estimates. "
                "Validation labels have a fixed additional budget; all outer-test rows are unchanged. "
                "The horizon identifies the table cohort, so levels across different horizon panels need not agree. "
                "The CPT checkpoint is the experiment-1 reference recipe (full updates, LR 3e-7, L2-SP 0.003) whose training "
                "partition excluded the table; positive differences favour CPT.",
                {"Plotted means": pd.concat(curves, ignore_index=True), "Complete paired table values": pd.concat(values, ignore_index=True)},
                {"measurement": "five_fold_context_benchmark", "metrics": [m for m, _ in metrics], "aggregation_unit": "table",
                 "context_horizon": int(horizon), "fixed_tables": sorted(cohort),
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
            ax.plot(arm.context_rows, arm.tables, label="TabPFN " + study.config.get("base_version", "v3") if family == "tabpfn" else FAMILIES[family],
                    color=style.color(study.config.get("base_version", "v3") if family == "tabpfn" else "tabicl"),
                    marker=style.PANEL_MARKERS[index], markersize=style.CURVE_MARKER_SIZE,
                    linestyle=style.FACTOR_ARM_STYLES[index])
        ax.set_title(track.upper()); ax.set_xscale("log"); ax.set_xlabel("Labeled context rows")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Tables with all five paired folds"); ax.legend()
    return [cp.Page("context_support", fig,
        "Number of tables with finite original/CPT measurements in all five folds at each nominal context size. "
        "Counts reflect both available table sizes and completed exports; they are not live scheduler status. "
        "Performance panels separately fix the cohort over their full displayed range.", {"Paired support": support})]


def plot_table_curves(study):
    """Named tables expose heterogeneous responses hidden by an equal-table mean."""
    from matplotlib.lines import Line2D
    pages = []
    for track, metrics in (("pd", (("roc_auc", "AUC"), ("brier_score", "Brier score"))),
                           ("lgd", (("r2", "R²"), ("rmse", "RMSE")))):
        frames = [paired_tables(study, metric) for metric, _ in metrics]
        frames = [f[f.track.eq(track)] for f in frames if not f.empty]
        if not frames:
            continue
        data = pd.concat(frames, ignore_index=True)
        datasets = sorted(data.dataset.unique())
        for start in range(0, len(datasets), style.CONTEXT_TABLES_PER_PAGE):
            selected = datasets[start:start + style.CONTEXT_TABLES_PER_PAGE]
            fig, axes = plt.subplots(len(selected), 2, squeeze=False, layout="constrained",
                figsize=style.figsize(style.WIDTH_FULL, style.CONTEXT_TABLE_PAGE_RATIO))
            fig.suptitle(f"{track.upper()}: paired context effects by table")
            for row, dataset in enumerate(selected):
                for col, (metric, label) in enumerate(metrics):
                    ax = axes[row, col]
                    for family in study.config["families"]:
                        part = data[data.dataset.eq(dataset) & data.metric.eq(metric) & data.family.eq(family)].sort_values("context_rows")
                        color = style.color(study.config.get("base_version", "v3") if family == "tabpfn" else "tabicl")
                        ax.plot(part.context_rows, part.gain, color=color, linewidth=style.THIN_LINE)
                    ax.set_title(dataset)
                    ax.set_ylabel("Δ" + label if metric in ("roc_auc", "r2") else ("Brier reduction" if metric == "brier_score" else "RMSE reduction"))
                    ax.set_xscale("log"); ax.set_xlabel("Labeled context rows")
                    ax.axhline(0, color=style.color("reference"), linestyle=":")
            handles = [Line2D([], [], color=style.color(study.config.get("base_version", "v3") if f == "tabpfn" else "tabicl"),
                label="TabPFN " + study.config.get("base_version", "v3") if f == "tabpfn" else FAMILIES[f]) for f in study.config["families"]]
            fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
            pages.append(cp.Page(f"context_{track}_tables_{start // style.CONTEXT_TABLES_PER_PAGE + 1:02d}", fig,
                f"{track.upper()}. Context-size effects for individually named held-out credit tables. "
                "Each line averages five paired outer-fold differences; positive values favor CPT. "
                "AUC and R² use CPT minus original; Brier and RMSE use original minus CPT. "
                "Every curve retains the same table, fold membership and complete test rows at all context sizes. "
                "Each fold uses nested context subsets and an unchanged validation set. Model-specific native context caps apply. "
                "The original and CPT absolute scores and every plotted difference are retained in the printed tables.",
                {"Named-table absolute scores and effects": data[data.dataset.isin(selected)]}))
    return pages


def secondary_tables(study):
    return {metric: paired_tables(study, metric) for metric in
            ("roc_auc", "r2", "brier_score", "log_loss", "ece", "f1", "rmse", "mae", "elapsed_seconds", "fit_seconds")}
