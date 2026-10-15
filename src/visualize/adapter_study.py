"""DiagScale reporting: adapter dynamics and paired, precision-matched evaluation.

Original versus adapter within experiment 5 is the primary comparison. Previous
full-parameter runs differ in precision, episode size and sampling and are not
silently imported as a causal parameter-count control.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualize import campaign as cp, comparisons as cmp, dashboards as db, style
from src.visualize.paper_figures import effect_label


def protocol_table(run):
    return pd.DataFrame([
        ("Optimized parameters", "Zero-centered query-scale deltas only; original weights unchanged"),
        ("Training", "Cross-table CPT; target table excluded from checkpoint training"),
        ("Grid", "2 bases × 3 adapter rates × 4 dataset partitions per task; 48 trials across PD/LGD"),
        ("Adapter rates", ", ".join(f"{lr:g}" for lr in sorted(run.trials.learning_rate.unique()))),
        ("Episode cap", int(run.cfg.train.max_rows_per_step)),
        ("Precision", "FP32; TabICL original/adapter inference both disable AMP and FA3"),
        ("Budget", f"{run.target:,} successful updates; seed {run.cfg.seed}"),
        ("Regularization", "AdamW decay on zero-centered deltas; no L2-SP"),
        ("Parameter movement", "RMS query-scale delta; not relative backbone-weight displacement"),
        ("Interpretation", "Recipe-level comparison against the base; not a one-factor comparison to earlier full updates"),
    ], columns=["setting", "value"])


def _grid(title):
    fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
    fig.suptitle(title)
    return fig, list(axes.flat)


def _legend(fig):
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncol=3)
            return


def plot_optimization(run):
    fields = {"train_loss": "Data loss", "grad_norm_mean": "Adapter gradient norm",
              "weight_drift": "RMS query-scale delta"}
    fields.update({"clipped_frac": "Clipped optimizer updates"} if run.track == "pd"
                  else {"elapsed_sec": "Recorded elapsed time (s)"})
    frames = [db.complete_windows(run, f) for f in fields]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return []
    data = pd.concat(frames, ignore_index=True)
    pages = []
    for base, group in data.groupby("base"):
        if not group.value.notna().any():
            continue
        fig, axes = _grid(f"{run.track.upper()} / {base}: DiagScale optimization")
        tables = []
        for ax, (field, label) in zip(axes, fields.items()):
            part = group[group.metric.eq(field)]
            if not part.value.notna().any():
                ax.remove(); continue
            for lr, arm in part.groupby("learning_rate"):
                curve = arm.set_index("window_end").reindex(cp.optimization_grid(run, field))
                ax.plot(curve.index, curve.value, color=style.ADAPTER_LR_COLORS[lr],
                        linestyle=style.ADAPTER_LR_STYLES[lr],
                        marker="o", markersize=style.CURVE_MARKER_SIZE,
                        markevery=style.CURVE_MARKER_INTERVAL,
                        linewidth=style.THIN_LINE, label=f"LR {lr:g}")
                tables.append(curve.reset_index().assign(learning_rate=lr, metric=field))
            ax.set_ylabel(cp.loss_label(run.track, base) if field == "train_loss" else label)
            ax.set_xlabel("Successful updates")
            if field == "clipped_frac":
                from matplotlib.ticker import PercentFormatter
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
                if part.value.max() == 0:
                    ax.text(.5, .5, "No clipped updates", ha="center", transform=ax.transAxes)
        _legend(fig)
        pages.append(cp.Page(f"adapter_optimization_{run.track}_{base}", fig,
            f"{run.track.upper()}. Mean diagnostics across all four dataset partitions at each adapter rate. "
            "Lines connect recorded summaries, with periodically spaced markers; a missing partition leaves a gap. Data loss is the native query objective; incomplete table-coverage epochs are excluded. "
            "Gradient diagnostics concern optimizer-selected query-scale deltas. "
            "Movement is RMS delta from the identity scaling, not a relative change from zero-valued adapter parameters.",
            {"Plotted windows and support": pd.concat(tables, ignore_index=True)}))
    return pages


def plot_monitors(run):
    curves = []
    for split, domain in (("test", "Held-out credit"), ("ood", "Non-credit retention")):
        data = db.monitor_observations(run, split)
        if data.empty:
            continue
        for key, recipe in data.groupby(cp.FACTORS):
            curve = cp.trajectory_curve(run, recipe, split=split)
            curves.append(curve.assign(**dict(zip(cp.FACTORS, key)), domain=domain))
    if not curves:
        return []
    data = pd.concat(curves, ignore_index=True)
    if not data["mean"].notna().any():
        return []
    fig, axes = _grid(f"{run.track.upper()}: DiagScale transfer and retention")
    bases = sorted(run.trials.base.unique())
    for ax, (base, domain) in zip(axes, [(b, d) for d in ("Held-out credit", "Non-credit retention") for b in bases]):
        part = data[data.base.eq(base) & data.domain.eq(domain)]
        if not part["mean"].notna().any():
            ax.remove(); continue
        for lr, arm in part.groupby("learning_rate"):
            ax.plot(arm.updates, arm["mean"], label=f"LR {lr:g}",
                    color=style.ADAPTER_LR_COLORS[lr], linestyle=style.ADAPTER_LR_STYLES[lr],
                    marker="o", markersize=style.CURVE_MARKER_SIZE,
                    linewidth=style.THIN_LINE)
        ax.axhline(0, color=style.color("reference"), linestyle=":")
        ax.set_title(base + "\n" + domain); ax.set_ylabel("Change in " + db.monitor_label(run))
        ax.set_xlabel("Successful updates")
    _legend(fig)
    return [cp.Page(f"adapter_monitors_{run.track}", fig,
        f"{run.track.upper()}. Change in {db.monitor_label(run)} from each table's update-zero score. "
        "Repeated non-credit partitions are averaged within table; tables receive equal weight. "
        "All planned partitions and the unchanged baseline table cohort are required at every milestone. "
        "These fixed-context training monitors are distinct from the final five-fold benchmark.", {"Curves and support": data})]


def costs(run):
    fields = [c for c in ("trial_name", "base", "learning_rate", "partition", "status", "trainable_params",
        "total_params", "rows_seen", "elapsed_sec", "gpu_hours", "peak_gpu_gb", "final_drift") if c in run.trials]
    return run.trials[fields].rename(columns={"trainable_params": "optimizer_selected_params", "final_drift": "query_scale_delta_rms"})


def plot_resources(run):
    return db.plot_resources(run)


def benchmark_tables(run, metric=None, domain="credit"):
    metric = metric or db.monitor_metric(run)
    data = cmp.observations(run, domain=domain, benchmark=True, metric=metric)
    return cmp.dataset_means(data, run, ["effect", "untuned", "trained"])


def matched_rates(run, metric=None, domain="credit"):
    data = benchmark_tables(run, metric, domain)
    if data.empty:
        return data
    rates = set(run.trials.learning_rate)
    complete = data.groupby(["base", "dataset"]).learning_rate.agg(set)
    keys = complete[complete.map(lambda v: v == rates)].index
    return data.set_index(["base", "dataset"]).loc[keys].reset_index() if len(keys) else data.iloc[:0]


def plot_benchmark(run, metric=None):
    metric = metric or db.monitor_metric(run)
    parts = [matched_rates(run, metric, d) for d in ("credit", "ood")]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return []
    data = pd.concat(parts, ignore_index=True)
    fig, axes = _grid(f"{run.track.upper()}: DiagScale five-fold benchmark")
    tables = []
    bases = sorted(run.trials.base.unique())
    for ax, (base, domain) in zip(axes, [(b, d) for d in ("credit", "ood") for b in bases]):
        group = data[data.base.eq(base) & data.domain.eq(domain)]
        if group.empty:
            ax.remove(); continue
        summary = group.groupby("learning_rate").effect.agg(mean="mean", q25=lambda v: v.quantile(.25),
            q75=lambda v: v.quantile(.75), datasets="size").reset_index()
        for pos, row in enumerate(summary.itertuples()):
            values = group.loc[group.learning_rate.eq(row.learning_rate), "effect"].sort_values()
            color = style.ADAPTER_LR_COLORS[row.learning_rate]
            jitter = np.linspace(-style.JITTER_WIDTH, style.JITTER_WIDTH, len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(pos + jitter, values,
                       color=color, s=style.POINT_SIZE/2, alpha=style.POINT_ALPHA)
            ax.plot([pos, pos], [row.q25, row.q75], color=style.color("reference"), linewidth=style.THIN_LINE)
            ax.scatter([pos], [row.mean], color=style.color("reference"), marker="D", s=style.POINT_SIZE)
        ax.axhline(0, color=style.color("reference"), linestyle=":")
        ax.set_xticks(range(len(summary)), [f"{v:g}" for v in summary.learning_rate]); ax.set_xlabel("Adapter learning rate")
        ax.set_title(base + "\n" + cmp.DOMAINS[domain]); ax.set_ylabel(effect_label(metric))
        tables.append(summary.assign(base=base, domain=domain))
    return [cp.Page(f"adapter_benchmark_{run.track}_{metric}", fig,
        f"{run.track.upper()}. Five-fold paired DiagScale effects relative to the original checkpoint under matched inference precision. "
        "Points are tables, diamonds are equal-table means and vertical bars span the table interquartile range, not confidence intervals. "
        "Only tables measured at all three rates enter each model/domain panel; non-credit tables require all four training partitions. "
        "Every trained/control pair requires all five finite outer folds. Positive effects favor DiagScale.",
        {"Paired table values": data, "Plotted summaries": pd.concat(tables, ignore_index=True)})]


def absolute_scores(run):
    metric = db.monitor_metric(run)
    if run.evaluation.empty or metric not in run.evaluation:
        return pd.DataFrame()
    data = cp._complete_folds(run.evaluation, metric)
    keys = [c for c in ("domain", "method_name", "test_dataset_id", "split") if c in data]
    return data.groupby(keys, dropna=False)[metric].agg(mean="mean", folds="size").reset_index().assign(metric=metric)
