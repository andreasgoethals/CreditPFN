"""Compact, evidence-gated reports for the separate cross-table LoRA campaign.

No scientific observations are generated here. Pending campaigns yield tables,
not empty figures, and comparisons retain complete folds and matched table cohorts.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

from src.visualize import campaign as cp, comparisons as cmp, dashboards as db, style
from src.visualize import training_summary as ts
from src.visualize.paper_figures import effect_label


def _check_mode(run):
    if (OmegaConf.select(run.cfg, "train.adaptation_mode") != "lora"
            or not str(run.cfg.run_name).startswith("cpt_lora_")):
        raise ValueError("LoRA reports require an explicitly configured LoRA campaign")


def protocol_table(run):
    _check_mode(run)
    values = {
        "Run identity": str(run.cfg.run_name),
        "Task": run.track.upper(),
        "Adaptation": "Low-rank updates to all attention query/value projections; base weights fixed",
        "Rank": OmegaConf.select(run.cfg, "lora.r"),
        "Alpha": OmegaConf.select(run.cfg, "lora.alpha"),
        "Adapter dropout": OmegaConf.select(run.cfg, "lora.dropout"),
        "Learning rates": ", ".join(f"{v:g}" for v in sorted(run.trials.learning_rate.unique())),
        "L2-SP": ", ".join(f"{v:g}" for v in sorted(run.trials.l2sp_lambda.unique())),
        "Planned trials in this task": len(run.trials),
        "Dataset partitions": int(run.cfg.corpus.n_splits),
        "Training seeds": ", ".join(map(str, sorted(run.trials.seed.unique()))),
        "Successful-update budget": run.target,
        "Episode row cap": OmegaConf.select(run.cfg, "train.max_rows_per_step"),
        "Context sampling": OmegaConf.select(run.cfg, "train.context_sampling"),
        "Evaluation": "Original/LoRA weights on the same outer folds; TabICL FP32 without FA3",
        "Movement": "Frobenius norm of effective Q/V weight change / initial targeted Q/V weight norm",
    }
    return pd.DataFrame(values.items(), columns=["Setting", "Value"])


def coverage_table(run):
    """A pending roster is evidence of a plan, not a claim about live jobs."""
    _check_mode(run)
    rows = []
    for (base, rate), trials in run.trials.groupby(["base", "learning_rate"]):
        rows.append(dict(base=base, learning_rate=rate, planned=len(trials),
            completed=int(trials.status.eq("OK").sum()),
            recorded_outcomes=", ".join(f"{k}: {v}" for k, v in trials.status.value_counts().sort_index().items()),
            evidence="No recorded training outcome" if trials.status.eq("PENDING").all() else "Recorded local outcomes"))
    return pd.DataFrame(rows)


def _grid(title):
    fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO),
                             layout="constrained")
    fig.suptitle(title)
    return fig, list(axes.flat)


def _legend(fig):
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncol=min(len(handles), style.LEGEND_COLUMNS))
            break


def plot_optimization(run):
    """Keep native loss separate by architecture; omit uninformative zero penalties."""
    _check_mode(run)
    fields = {"train_loss": "Data loss", "grad_norm_mean": "Mean LoRA gradient norm",
              "weight_drift": "Effective Q/V displacement", "elapsed_sec": "Training time (seconds)"}
    frames = [db.complete_windows(run, field) for field in fields]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return []
    data = pd.concat(frames, ignore_index=True)
    pages = []
    for base, group in data.groupby("base"):
        if not group.value.notna().any():
            continue
        fig, axes = _grid(f"{run.track.upper()} / {style.model_label(base)}: LoRA optimization")
        plotted = []
        for ax, (field, label) in zip(axes, fields.items()):
            part = group[group.metric.eq(field)]
            if not part.value.notna().any():
                ax.remove()
                continue
            for rate, arm in part.groupby("learning_rate"):
                curve = arm.set_index("window_end").reindex(cp.optimization_grid(run, field))
                ax.plot(curve.index, curve.value, label=f"LR {rate:g}",
                    color=style.LORA_LR_COLORS[rate], linestyle=style.LORA_LR_STYLES[rate],
                    marker=style.PANEL_MARKERS[0], markersize=style.CURVE_MARKER_SIZE,
                    markevery=style.CURVE_MARKER_INTERVAL, linewidth=style.THIN_LINE)
                plotted.append(curve.reset_index().assign(metric=field, learning_rate=rate))
            ax.set_xlabel("Successful updates")
            ax.set_ylabel(cp.loss_label(run.track, base) if field == "train_loss" else label)
        _legend(fig)
        pages.append(cp.Page(f"lora_optimization_{run.track}_{base}", fig,
            ts.seed_caption(run) + "LoRA diagnostics averaged over every planned dataset partition at each learning rate. "
            "Missing partitions leave gaps. Data loss is the architecture's native query objective; epochs without "
            "complete training-table coverage are omitted. Gradient norms concern optimizer-selected LoRA factors. "
            "Effective displacement is the Frobenius norm of the induced query/value weight update divided by the "
            "initial targeted query/value weight norm, not relative displacement of zero-initialized adapter factors. "
            "Time is recorded training wall time and excludes queue wait. Losses are never pooled across architectures.",
            {"Displayed curves and partition support": pd.concat(plotted, ignore_index=True)}))
    return pages


def plot_monitors(run):
    _check_mode(run)
    curves = []
    for split, domain in (("test", "credit"), ("ood", "ood")):
        data = db.monitor_observations(run, split)
        if data.empty:
            continue
        for key, recipe in data.groupby(cp.FACTORS):
            curve = cp.trajectory_curve(run, recipe, split=split)
            curves.append(curve.assign(**dict(zip(cp.FACTORS, key)), domain=domain))
    if not curves:
        return []
    data = pd.concat(curves, ignore_index=True)
    if not (data["mean"].notna() & data.updates.gt(0)).any():
        return []
    fig, axes = _grid(f"{run.track.upper()}: LoRA transfer during training")
    for ax, (base, domain) in zip(axes, [(base, domain) for domain in ("credit", "ood")
                                        for base in sorted(run.trials.base.unique())]):
        part = data[data.base.eq(base) & data.domain.eq(domain)]
        if not (part["mean"].notna() & part.updates.gt(0)).any():
            ax.remove()
            continue
        for rate, arm in part.groupby("learning_rate"):
            ax.plot(arm.updates, arm["mean"], label=f"LR {rate:g}",
                color=style.LORA_LR_COLORS[rate], linestyle=style.LORA_LR_STYLES[rate],
                marker=style.PANEL_MARKERS[0], markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE)
        ax.axhline(0, color=style.color("reference"), linestyle=":")
        ax.set_title(style.model_label(base) + "\n" + cmp.DOMAINS[domain])
        ax.set_xlabel("Successful updates")
        ax.set_ylabel("Change in " + db.monitor_label(run))
    _legend(fig)
    return [cp.Page(f"lora_monitors_{run.track}", fig,
        ts.seed_caption(run) + f"Change in {db.monitor_label(run)} from each table's update-zero score during LoRA CPT. "
        "Repeated non-credit observations are averaged within table; tables receive equal weight. Every planned "
        "partition and the unchanged baseline cohort are required at each milestone. Rates can have different "
        "monitor cohorts if baseline measurements are absent; the support table identifies these omissions. "
        "These fixed-context training monitors are distinct from the final five-fold benchmark.",
        {"Monitor curves and support": data})]


def plot_schedule(runs):
    """A pending plan needs no standalone figure; keep observed-rate tables intact."""
    for run in runs:
        _check_mode(run)
    if not any(not run.histories.empty for run in runs):
        return []
    from src.visualize.pilots import plot_schedule as configured_schedule
    pages = configured_schedule(runs)
    for page in pages:
        rates = page.tables["Configured applied schedules"].peak_lr.drop_duplicates()
        for line, rate in zip(page.figure.axes[0].lines, rates):
            line.set_color(style.LORA_LR_COLORS[rate])
            line.set_linestyle(style.LORA_LR_STYLES[rate])
        for legend in list(page.figure.legends):
            legend.remove()
        page.figure.axes[0].legend()
        style.finish_figure(page.figure)
        page.name = "lora_" + page.name
    return pages


def costs(run):
    _check_mode(run)
    fields = [field for field in ("trial_name", "base", "learning_rate", "partition", "seed", "status",
        "trainable_params", "total_params", "rows_seen", "elapsed_sec", "gpu_hours", "peak_gpu_gb", "final_drift")
        if field in run.trials]
    return run.trials[fields].rename(columns={"trainable_params": "optimizer_selected_lora_parameters",
        "final_drift": "relative_effective_qv_displacement"})


def resource_tables(runs):
    sources, rows = [], []
    for run in runs:
        source = costs(run)
        source = source[source.status.eq("OK")].assign(track=run.track)
        sources.append(source)
        for field in ("gpu_hours", "peak_gpu_gb"):
            if field not in source:
                continue
            for (base, rate), group in source.groupby(["base", "learning_rate"]):
                values = pd.to_numeric(group[field], errors="coerce")
                values = values[np.isfinite(values)]
                if values.empty:
                    continue
                rows.append(dict(track=run.track, base=base, learning_rate=rate, measure=field,
                    median=values.median(), q25=values.quantile(.25), q75=values.quantile(.75),
                    completed_trials=len(values), planned_trials=int(((run.trials.base == base) &
                        (run.trials.learning_rate == rate)).sum())))
    return pd.DataFrame(rows), pd.concat(sources, ignore_index=True) if sources else pd.DataFrame()


def plot_costs(runs):
    data, sources = resource_tables(runs)
    if data.empty:
        return []
    measures = [field for field in ("gpu_hours", "peak_gpu_gb") if field in set(data.measure)]
    fig, axes = plt.subplots(1, len(measures), squeeze=False, layout="constrained",
                             figsize=style.figsize(style.WIDTH_FULL, style.SMALL_RATIO))
    fig.suptitle("LoRA training cost by task, model and rate")
    identities = sorted(set(zip(data.track, data.base)), key=lambda value: (value[0] != "pd", value[1]))
    rates = sorted(data.learning_rate.unique())
    labels = {"gpu_hours": "Recorded training time (GPU hours)",
              "peak_gpu_gb": "Peak allocated tensor memory (GB)"}
    for ax, measure in zip(axes.flat, measures):
        for index, (track, base) in enumerate(identities):
            for position, rate in enumerate(rates):
                part = data[data.track.eq(track) & data.base.eq(base) & data.learning_rate.eq(rate) & data.measure.eq(measure)]
                if part.empty:
                    continue
                point = part.iloc[0]
                ax.errorbar(point["median"], index+(position-(len(rates)-1)/2)*style.PROTOCOL_POINT_OFFSET,
                    xerr=[[point["median"]-point.q25], [point.q75-point["median"]]],
                    color=style.LORA_LR_COLORS[rate], marker=style.PANEL_MARKERS[position],
                    markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE)
        ax.set_yticks(range(len(identities)), [f"{track.upper()} / {style.model_label(base)}" for track, base in identities])
        ax.invert_yaxis()
        ax.set_xlabel(labels[measure])
        ax.set_xlim(left=0)
    handles = [Line2D([], [], marker=style.PANEL_MARKERS[i], color=style.LORA_LR_COLORS[rate],
                      linestyle="", label=f"LR {rate:g}") for i, rate in enumerate(rates)]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    return [cp.Page("lora_training_cost", fig,
        "LoRA training cost with only low-rank query/value factors optimized. "
        + " ".join(f"{run.track.upper()}: {ts.seed_caption(run).strip()}" for run in runs)
        + " Points and horizontal intervals show medians and interquartile ranges over completed dataset partitions "
        "within each task, model and learning rate, not confidence intervals. Each completed trial contributes once; "
        "counts accompany the figure. Missing or unfinished costs are not zeros. Time excludes queue wait, and "
        "memory is peak allocated tensor memory rather than total device capacity.",
        {"Displayed cost summaries": data, "Completed-trial source measurements": sources})]


def benchmark_tables(run, metric=None, domain="credit"):
    _check_mode(run)
    if not run.evaluation.empty and "adaptation_mode" in run.evaluation:
        adapted = run.evaluation[run.evaluation.source.str.endswith("-trained", na=False)]
        if not adapted.adaptation_mode.eq("lora").all():
            raise ValueError("A LoRA report cannot pool other adaptation modes")
    metric = metric or db.monitor_metric(run)
    data = cmp.observations(run, domain=domain, benchmark=True, metric=metric)
    if data.empty:
        return data
    # The same suffix was used for legacy adapters before the native campaign.
    # Interpret it only after the campaign/mode checks, never globally.
    if not data.method_dirname.str.endswith("__lora").all() or not data.l2sp_lambda.eq(0).all():
        raise ValueError("A LoRA report cannot pool other adaptation modes or anchoring settings")
    if not set(data.learning_rate).issubset(set(run.trials.learning_rate)):
        raise ValueError("LoRA results include an unplanned learning rate")
    return cmp.dataset_means(data, run, ["effect", "untuned", "trained"])


def matched_tables(run, metrics, domain="credit"):
    """Common tables at all rates AND metrics, with full folds/partitions upstream."""
    rates = set(run.trials.learning_rate)
    frames, common = [], None
    for metric in metrics:
        data = benchmark_tables(run, metric, domain)
        if data.empty:
            return pd.DataFrame()
        complete = data.groupby(["base", "dataset"]).learning_rate.agg(set)
        keys = set(complete[complete.map(lambda values: values == rates)].index)
        common = keys if common is None else common & keys
        frames.append(data)
    if not common:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    return data.set_index(["base", "dataset"]).loc[sorted(common)].reset_index()


def score_summary(data):
    if data.empty:
        return pd.DataFrame()
    return data.groupby(["domain", "metric", "base", "learning_rate"]).agg(
        original=("untuned", "mean"), lora=("trained", "mean"), effect=("effect", "mean"),
        q25=("effect", lambda values: values.quantile(.25)),
        q75=("effect", lambda values: values.quantile(.75)), datasets=("dataset", "nunique")).reset_index()


def _rate_axis(ax, rates):
    from matplotlib.ticker import NullLocator
    ax.set_xscale("log")
    ax.set_xticks(rates, [f"{rate:g}" for rate in rates])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("LoRA learning rate (log scale)")


def plot_primary(run):
    metric = db.monitor_metric(run)
    parts = [matched_tables(run, [metric], domain) for domain in ("credit", "ood")]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return []
    data = pd.concat(parts, ignore_index=True)
    summary = score_summary(data)
    fig, axes = _grid(f"{run.track.upper()}: original and LoRA five-fold {db.monitor_label(run)}")
    for ax, (base, domain) in zip(axes, [(base, domain) for domain in ("credit", "ood")
                                        for base in sorted(run.trials.base.unique())]):
        group = summary[summary.base.eq(base) & summary.domain.eq(domain)].sort_values("learning_rate")
        if group.empty:
            ax.remove()
            continue
        ax.plot(group.learning_rate, group.original, label="Original weights", linestyle="--",
                color=style.color("reference"), linewidth=style.THIN_LINE)
        ax.plot(group.learning_rate, group.lora, label="LoRA CPT", color=style.color(base),
                marker=style.PANEL_MARKERS[0], markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE)
        _rate_axis(ax, sorted(run.trials.learning_rate.unique()))
        ax.set_title(style.model_label(base) + "\n" + cmp.DOMAINS[domain])
        ax.set_ylabel(db.monitor_label(run))
    # Use neutral handles: model identity is specified by the panel title, not this legend.
    fig.legend(handles=[Line2D([], [], color=style.color("reference"), linestyle="--", label="Original weights"),
        Line2D([], [], color=style.color("reference"), marker=style.PANEL_MARKERS[0], label="LoRA CPT")],
        loc="outside lower center", ncol=2)
    return [cp.Page(f"lora_primary_{run.track}", fig,
        f"Five-fold {db.monitor_label(run)} for original and LoRA checkpoints on excluded credit tables and the "
        "non-credit retention panel. Lines connect equal-table means across the three LoRA rates; they are not "
        "learning curves over training time. All rates share the same complete table cohort within each panel. "
        "Each trained/original pair requires five finite matched folds; non-credit tables additionally require all "
        "training partitions before within-table averaging. Original and LoRA inference settings match within this "
        "experiment, including FP32 without FA3 for TabICL. Both absolute scores and paired effects are retained "
        "in the accompanying tables. The best observed rate is not a validated selected recipe.",
        {"Complete matched table scores and effects": data, "Displayed equal-table means": summary})]


def plot_secondary(run):
    metrics = ["brier_score", "log_loss"] if run.track == "pd" else ["rmse", "mae"]
    data = matched_tables(run, metrics)
    if data.empty:
        return []
    summary = score_summary(data)
    fig, axes = _grid(f"{run.track.upper()}: LoRA held-out credit prediction quality")
    for ax, (base, metric) in zip(axes, [(base, metric) for metric in metrics
                                        for base in sorted(run.trials.base.unique())]):
        group = summary[summary.base.eq(base) & summary.metric.eq(metric)].sort_values("learning_rate")
        if group.empty:
            ax.remove()
            continue
        ax.plot(group.learning_rate, group.effect, color=style.color(base), marker=style.PANEL_MARKERS[0],
                markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE)
        ax.fill_between(group.learning_rate, group.q25, group.q75, color=style.color(base), alpha=style.INTERVAL_ALPHA)
        ax.axhline(0, color=style.color("reference"), linestyle=":")
        _rate_axis(ax, sorted(run.trials.learning_rate.unique()))
        ax.set_title(style.model_label(base))
        ax.set_ylabel(effect_label(metric))
    return [cp.Page(f"lora_secondary_{run.track}", fig,
        "LoRA prediction-quality effects on held-out credit tables, paired with the same model's original weights. "
        "Positive effects indicate improvement. Lines show equal-table mean effects and shading the table "
        "interquartile range, not confidence intervals. Each model uses a common table cohort across all three "
        "rates and both displayed metrics, with complete five-fold pairs. RMSE is a fractional reduction relative "
        "to the original; Brier, log-loss and MAE effects are original minus adapted on their native metric scales. "
        "These summaries describe the declared grid without selecting a best rate on held-out scores.",
        {"Complete matched secondary scores and effects": data, "Displayed effect summaries": summary})]


def reference_protocol_audit(run, reference):
    """Expose candidate confounders; never silently import earlier full/frozen results."""
    _check_mode(run)
    if run.track != reference.track:
        raise ValueError("Reference comparisons require the same prediction task")
    paths = ("seed", "train.target_total_steps", "train.max_rows_per_step", "train.context_sampling",
             "train.amp", "optimizer.weight_decay", "scheduler", "corpus.n_folds", "corpus.split_seed",
             "evaluation.tabicl_inference_options")
    rows = []
    for path in paths:
        left, right = [OmegaConf.select(cfg, path) for cfg in (reference.cfg, run.cfg)]
        left = OmegaConf.to_container(left, resolve=True) if OmegaConf.is_config(left) else left
        right = OmegaConf.to_container(right, resolve=True) if OmegaConf.is_config(right) else right
        rows.append(dict(check=path, experiment1=str(left), experiment8=str(right), matches=left == right))
    rates = sorted(set(reference.trials.loc[reference.trials.l2sp_lambda.eq(0), "learning_rate"]) & set(run.trials.learning_rate))
    rows.append(dict(check="Candidate shared rates with L2-SP zero", experiment1=", ".join(map(str, rates)),
        experiment8=", ".join(map(str, sorted(run.trials.learning_rate.unique()))),
        matches=set(rates) == set(run.trials.learning_rate)))
    rows.append(dict(check="Direct cross-experiment comparison", experiment1="Full/frozen grid remains separate",
        experiment8="Unavailable: requires matching artifact provenance, base/data hashes, folds and inference protocol", matches=False))
    return pd.DataFrame(rows)
