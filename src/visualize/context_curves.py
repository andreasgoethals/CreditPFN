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
    cfg = OmegaConf.to_container(cfg_override if cfg_override is not None else OmegaConf.load(REPO_ROOT / "config/experiment10/context.yaml"), resolve=True)
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


TRACK_METRICS = {"pd": (("roc_auc", "AUC"), ("brier_score", "Brier score")),
                 "lgd": (("r2", "R²"), ("rmse", "RMSE"))}
GAIN_SCALE = {"roc_auc": 100., "brier_score": 100., "r2": 1., "rmse": 1.}
GAIN_LABELS = {"roc_auc": "ΔAUC (pp)\nCPT − original", "brier_score": "Brier reduction (×100)\noriginal − CPT",
               "r2": "ΔR²\nCPT − original", "rmse": "RMSE reduction\noriginal − CPT"}


def _family_color(study, family):
    return style.color(study.config.get("base_version", "v3") if family == "tabpfn" else "tabicl")


def _family_label(study, family):
    return "TabPFN " + study.config.get("base_version", "v3") if family == "tabpfn" else FAMILIES[family]


def plot_context_curves(study, *, selected_track=None, horizons=None):
    """Per task: columns are context horizons (fixed cohorts), rows are level and paired effects."""
    from matplotlib.lines import Line2D
    pages = []
    for track, metrics in TRACK_METRICS.items():
        if selected_track is not None and track != selected_track:
            continue
        columns = []
        for horizon in sorted({min(n, study.config["max_context_rows"]) for n in (COHORT_HORIZONS if horizons is None else horizons)}):
            pairs = {m: fixed_cohort(study, paired_tables(study, m), track, horizon) for m, _ in metrics}
            if any(d.empty for d in pairs.values()):
                continue
            cohort = set.intersection(*(set(d.dataset) for d in pairs.values()))
            if cohort:
                columns.append((horizon, cohort, {m: d[d.dataset.isin(cohort)] for m, d in pairs.items()}))
        if not columns:
            continue
        (primary, primary_label), (secondary, _) = metrics
        rows = [("level", primary), ("gain", primary), ("gain", secondary)]
        fig, axes = plt.subplots(len(rows), len(columns), squeeze=False, sharex="col", layout="constrained",
                                 figsize=style.rows_figsize(len(rows)))
        fig.suptitle("Labelled target context: original versus CPT")
        curves, values = [], []
        for c, (horizon, cohort, pairs) in enumerate(columns):
            for metric, data in pairs.items():
                values.append(data.assign(horizon=int(horizon)))
            for r, (kind, metric) in enumerate(rows):
                ax = axes[r, c]
                data = pairs[metric]
                for family in study.config["families"]:
                    part = data[data.family.eq(family)]
                    curve = part.groupby("context_rows")[["original", "adapted", "gain"]].mean()
                    color = _family_color(study, family)
                    if kind == "level":
                        ax.plot(curve.index, curve.original, color=color, linestyle="--", linewidth=style.THIN_LINE * 1.2)
                        ax.plot(curve.index, curve.adapted, color=color, linewidth=style.THIN_LINE * 1.2)
                    else:
                        for _, table in part.groupby("dataset"):
                            table = table.sort_values("context_rows")
                            ax.plot(table.context_rows, table.gain * GAIN_SCALE[metric], color=color,
                                    alpha=style.TABLE_CURVE_ALPHA, linewidth=style.TABLE_CURVE_LINE)
                        ax.plot(curve.index, curve.gain * GAIN_SCALE[metric], color=color, linewidth=style.MEAN_CURVE_LINE)
                    if r == 0 or (kind == "gain" and metric != rows[r - 1][1]):
                        curves.append(curve.reset_index().assign(family=family, datasets=len(cohort), metric=metric,
                                                                 horizon=int(horizon)))
                if kind == "gain":
                    ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
                ax.set_xscale("log")
                if r == 0:
                    ax.set_title(f"Context ≤ {horizon:,} rows\n{len(cohort)} fixed table{'s' if len(cohort) != 1 else ''}")
                if c == 0:
                    ax.set_ylabel(primary_label if kind == "level" else GAIN_LABELS[metric])
                if r == len(rows) - 1:
                    ax.set_xlabel("Labelled context rows")
        handles = [Line2D([], [], color=_family_color(study, f), linewidth=style.MEAN_CURVE_LINE, label=_family_label(study, f))
                   for f in study.config["families"]]
        handles += [Line2D([], [], color=style.color("annotation"), linestyle="--", label="Original (top row)"),
                    Line2D([], [], color=style.color("annotation"), label="CPT (top row)"),
                    Line2D([], [], color=style.color("annotation"), alpha=.6, linewidth=style.TABLE_CURVE_LINE, label="Single table"),
                    Line2D([], [], color=style.color("mean"), linewidth=style.MEAN_CURVE_LINE, label="Equal-table mean")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3)
        pages.append(cp.Page(f"context_{track}", fig,
            f"Top: equal-table mean {primary_label} of the original model and the fixed-recipe CPT checkpoint as labelled context grows. "
            "Middle and bottom: paired CPT effects for the primary and secondary metric; positive values favour CPT, "
            "thin curves are individual tables and thick curves their equal-table mean. "
            "Each column fixes one table cohort across every context size and both models, so levels need not agree between columns. "
            "Scores average five paired outer folds within table before tables receive equal weight; curves are not uncertainty bands. "
            "Validation labels use a separate fixed budget and all outer-test rows are unchanged. "
            "The CPT checkpoint is the E1 reference recipe (full updates, LR 3×10⁻⁷, L2-SP 0.003) trained on a partition that excluded the table.",
            {"Plotted means": pd.concat(curves, ignore_index=True), "Complete paired table values": pd.concat(values, ignore_index=True)},
            {"measurement": "five_fold_context_benchmark", "metrics": [m for m, _ in metrics], "aggregation_unit": "table",
             "context_horizons": [int(h) for h, _, _ in columns],
             "fixed_tables": {str(int(h)): sorted(cohort) for h, cohort, _ in columns},
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
    fig, axes = plt.subplots(1, 2, figsize=style.rows_figsize(1, row_height=1.6), layout="constrained")
    fig.suptitle("Tables with all five paired folds")
    for ax, track in zip(axes, ("pd", "lgd")):
        group = support[support.track.eq(track)]
        if group.empty:
            ax.remove(); continue
        for index, (family, arm) in enumerate(group.groupby("family")):
            ax.plot(arm.context_rows, arm.tables, label=_family_label(study, family), color=_family_color(study, family),
                    marker=style.PANEL_MARKERS[index], markersize=style.CURVE_MARKER_SIZE * 1.3,
                    linestyle=style.FACTOR_ARM_STYLES[index], linewidth=style.THIN_LINE * 1.2)
        ax.set_title(f"{track.upper()} credit tables"); ax.set_xscale("log"); ax.set_xlabel("Labelled context rows")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Tables"); ax.legend()
    return [cp.Page("context_support", fig,
        "Number of held-out credit tables with finite original and CPT measurements in all five outer folds at each nominal context size. "
        "Counts fall when the context budget exceeds a table's available training rows; lines coincide where both models have the same support. "
        "Context-curve columns separately fix one cohort over their full displayed range.", {"Paired support": support})]


def plot_table_curves(study):
    """Every table × context size paired effect as a heatmap, one page per task."""
    from matplotlib.colors import TwoSlopeNorm
    pages = []
    for track, metrics in TRACK_METRICS.items():
        frames = [paired_tables(study, metric) for metric, _ in metrics]
        frames = [f[f.track.eq(track)] for f in frames if not f.empty]
        frames = [f for f in frames if not f.empty]
        if not frames:
            continue
        data = pd.concat(frames, ignore_index=True)
        families = [f for f in study.config["families"] if data.family.eq(f).any()]
        depth = data.groupby("dataset").context_rows.max().sort_values(ascending=False)
        datasets = list(depth.index)
        # Nominal sizes only: per-table available-pool endpoints stay in the printed tables.
        data = data[data.context_rows.isin(study.config["context_rows"])]
        sizes = np.array(sorted(data.context_rows.unique()), float)
        edges = np.concatenate([[sizes[0] / np.sqrt(sizes[1] / sizes[0]) if len(sizes) > 1 else sizes[0] / 2],
                                np.sqrt(sizes[:-1] * sizes[1:]),
                                [sizes[-1] * np.sqrt(sizes[-1] / sizes[-2]) if len(sizes) > 1 else sizes[0] * 2]])
        present = [(m, l) for m, l in metrics if data.metric.eq(m).any()]
        height = style.FIGURE_BAND + len(present) * (len(datasets) * .105 + .35)
        fig, axes = plt.subplots(len(present), len(families), squeeze=False, sharex=True, sharey=True,
                                 layout="constrained", figsize=(style.WIDTH_FULL, min(height, style.MAX_HEIGHT)))
        fig.suptitle("Paired CPT effect for every table and context size")
        for r, (metric, label) in enumerate(present):
            part = data[data.metric.eq(metric)]
            scaled = part.gain * GAIN_SCALE[metric]
            limit = float(np.nanquantile(np.abs(scaled), style.HEATMAP_COLOR_QUANTILE)) or 1.
            norm = TwoSlopeNorm(0., -limit, limit)
            for c, family in enumerate(families):
                ax = axes[r, c]
                cells = part[part.family.eq(family)].pivot(index="dataset", columns="context_rows", values="gain")
                cells = cells.reindex(index=datasets, columns=sizes) * GAIN_SCALE[metric]
                mesh = ax.pcolormesh(edges, np.arange(len(datasets) + 1), cells.to_numpy(float),
                                     cmap=style.CMAP_DIVERGING, norm=norm, shading="flat", rasterized=True)
                ax.set_facecolor(style.COLORS["untuned"] + "33")
                ax.set_xscale("log"); ax.set_ylim(len(datasets), 0)
                ax.set_yticks(np.arange(len(datasets)) + .5, datasets, fontsize=6.5)
                ax.tick_params(axis="y", length=0); ax.grid(False)
                if r == 0:
                    ax.set_title(_family_label(study, family), color=_family_color(study, family), fontweight="bold")
                if r == len(present) - 1:
                    ax.set_xlabel("Labelled context rows")
            bar = fig.colorbar(mesh, ax=list(axes[r]), shrink=.9, aspect=18, pad=.01, extend="both")
            bar.set_label(GAIN_LABELS[metric], fontsize=7)
            bar.ax.tick_params(labelsize=6.5)
        pages.append(cp.Page(f"context_{track}_tables", fig,
            "Paired CPT effect, averaged over five outer folds, for every held-out credit table (rows, ordered by available context) "
            "and nominal context size (columns); blue favours CPT and red the original model. "
            "Rows show the primary metric (top) and secondary metric (bottom). Colour limits are symmetric at the 95th percentile "
            "of absolute effects per metric and extreme cells saturate; printed values are exact. "
            "Grey cells exceed the table's available training rows. Each table keeps its fold membership, nested context subsets "
            "and complete test rows at every size.",
            {"Named-table absolute scores and effects": data}))
    return pages


def secondary_tables(study):
    return {metric: paired_tables(study, metric) for metric in
            ("roc_auc", "r2", "brier_score", "log_loss", "ece", "f1", "rmse", "mae", "elapsed_seconds", "fit_seconds")}
