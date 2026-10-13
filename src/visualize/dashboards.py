"""Compact training and endpoint reports with explicit averaging units.

Native objectives remain separated by architecture. Predictive monitors use AUC
and R², with within-dataset update-zero pairing and fixed observation support.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset_names import display_name
from src.visualize import campaign as cp, diagnostics as dg, style

FIELDS = {"train_loss": "Data loss", "l2sp_penalty": "Applied L2-SP penalty",
          "grad_norm_mean": "Mean gradient norm", "clipped_frac": "Clipped updates",
          "weight_drift": "Relative parameter displacement"}


def monitor_metric(run):
    return "roc_auc" if run.track == "pd" else "r2"


def monitor_label(run):
    return "AUC" if run.track == "pd" else "R²"


def monitor_observations(run, split="test", metric=None):
    """Read actual per-table secondary scores; never reinterpret RMSE as R²."""
    metric = metric or monitor_metric(run)
    frame = run.trajectories
    if frame.empty:
        return pd.DataFrame()
    prefix, suffix = f"score__{split}__", "__" + metric
    columns = {c: c[len(prefix):-len(suffix)] for c in frame
               if c.startswith(prefix) and c.endswith(suffix)}
    if not columns and metric == "roc_auc" and run.track == "pd":
        prefix = f"metric__{split}__"
        columns = {c: c[len(prefix):] for c in frame if c.startswith(prefix)}
    if not columns:
        return pd.DataFrame()
    keys = ["trial_name", "successful_updates", *(["processed_rows"] if "processed_rows" in frame else [])]
    data = frame[keys + list(columns)].melt(id_vars=keys, var_name="dataset", value_name="value")
    data["dataset"] = data.dataset.map(columns).map(display_name)
    data["value"] = pd.to_numeric(data.value, errors="coerce")
    baseline = data[data.successful_updates.eq(0)][["trial_name", "dataset", "value"]].rename(columns={"value": "initial"})
    data = data.merge(baseline, on=["trial_name", "dataset"], validate="many_to_one")
    data = data[np.isfinite(data.value) & np.isfinite(data.initial)].copy()
    data["effect"] = ((data.initial-data.value) if metric == "brier_score" else (data.value-data.initial))
    data = data.merge(run.trials[["trial_name", *cp.FACTORS, "partition", "seed"]],
                      on="trial_name", validate="many_to_one")
    return data.rename(columns={"trial_name": "trial", "successful_updates": "updates"})


def _grid(title):
    fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO),
                             layout="constrained")
    fig.suptitle(title)
    return fig, list(axes.flat)


def _legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=min(4, len(labels)))


def complete_windows(run, field):
    """One equally weighted trial per partition; require the entire planned recipe."""
    data = cp.optimization_curves(run, field)
    if data.empty:
        return data
    expected = run.trials.groupby(cp.FACTORS).size().rename("expected")
    data = data.groupby([*cp.FACTORS, "window_end"])[field].agg(value="mean", count="size").reset_index()
    data = data.merge(expected.reset_index(), on=cp.FACTORS, validate="many_to_one")
    data.loc[data["count"].ne(data.expected), "value"] = np.nan
    return data.assign(metric=field)


def plot_optimization(run):
    """Four panels per base; peak-LR curves pool the remaining planned knobs."""
    fields = ["train_loss", "l2sp_penalty", "grad_norm_mean",
              "clipped_frac" if run.track == "pd" else "weight_drift"]
    frames = [complete_windows(run, f) for f in fields]
    data = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    pages = []
    if data.empty:
        return pages
    for base, group in data.groupby("base"):
        fig, axes = _grid(f"{run.track.upper()} / {base}: optimization across dataset partitions")
        tables = []
        for ax, field in zip(axes, fields):
            part = group[group.metric.eq(field)]
            if field == "l2sp_penalty":
                part = part[part.l2sp_lambda.gt(0)]
            if part.empty or not part.value.notna().any():
                ax.remove(); continue
            for lr, arm in part.groupby("learning_rate"):
                curve = arm.groupby("window_end").value.agg(mean="mean", count="count", expected="size")
                curve.loc[curve["count"].ne(curve.expected), "mean"] = np.nan
                curve = curve.reindex(cp.optimization_grid(run, field))
                ax.plot(curve.index, curve["mean"], color=style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr))),
                        linewidth=style.THIN_LINE, label=f"LR {lr:.0e}")
                tables.append(curve.reset_index().assign(metric=field, learning_rate=lr))
            ax.set_ylabel(cp.loss_label(run.track, base) if field == "train_loss" else FIELDS[field])
            ax.set_xlabel("Successful optimizer updates")
            if field == "clipped_frac":
                from matplotlib.ticker import PercentFormatter
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
                maximum = part.value.max()
                if maximum > 0:
                    ax.set_ylim(0, min(style.CLIP_DISPLAY_CEILING, maximum*1.1))
                else:
                    ax.text(.5, .5, "No clipped updates", ha="center", transform=ax.transAxes)
            if field == "l2sp_penalty":
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        _legend(fig, axes[0])
        pages.append(cp.Page(f"optimization_dashboard_{run.track}_{base}", fig,
            "Lines give arithmetic means across all planned dataset partitions within a recipe, then equal-weight means "
            "over anchoring and adaptation settings at each peak learning rate. A missing partition or recipe produces a gap. "
            "Only the nonzero L2-SP arm enters the penalty panel. Data loss is the native query objective excluding "
            "L2-SP; incomplete table-coverage loss epochs are omitted. Epoch diagnostics use successful-update windows; "
            "parameter displacement uses exact milestones. Native losses are never averaged across model architectures.",
            {"Displayed means and coverage": pd.concat(tables, ignore_index=True), "Recipe means and partition support": group}))
    return pages


def factor_statistics(run, data, factor, value="value", step="window_end"):
    """Exact other-knob/partition matching, including complete planned support."""
    reference = {"l2sp_lambda": 0., "frozen": False, "learning_rate": 3e-7}[factor]
    keys = [f for f in cp.FACTORS if f != factor] + ["partition", "seed"]
    if "dataset" in data:
        keys += ["dataset"]
    rows = []
    plan = run.trials
    for alternative in sorted(set(plan[factor]) - {reference}):
        left, right = [data[data[factor].eq(v)] for v in (reference, alternative)]
        pair = left[keys+[step, value]].merge(right[keys+[step, value]], on=keys+[step],
                    suffixes=("_reference", "_alternative"), validate="one_to_one")
        for base in sorted(plan.base.unique()):
            part = pair[pair.base.eq(base)]
            wanted = plan[plan[factor].eq(reference) & plan.base.eq(base)][[k for k in keys if k != "dataset"]].merge(
                plan[plan[factor].eq(alternative) & plan.base.eq(base)][[k for k in keys if k != "dataset"]],
                validate="one_to_one")
            expected = len(wanted)
            if "dataset" in keys:
                expected = len(part[part[step].eq(0)])
                baseline_trials = set(data[data[factor].eq(reference) & data.base.eq(base) & data[step].eq(0)].trial)
                if baseline_trials != set(plan[plan[factor].eq(reference) & plan.base.eq(base)].trial_name):
                    expected = -1
            for update, group in part.groupby(step):
                complete = len(group) == expected and expected > 0
                a, b = group[value+"_reference"], group[value+"_alternative"]
                # Dataset effects have one partition per credit table; OOD tables
                # repeat by partition. Collapse repeats before assigning table weights.
                if "dataset" in keys:
                    values = group.groupby("dataset")[[value+"_reference", value+"_alternative"]].mean()
                    a, b = values.iloc[:, 0], values.iloc[:, 1]
                rows.append(dict(base=base, update=update, factor=factor, reference=reference, alternative=alternative,
                    reference_mean=a.mean() if complete else np.nan, alternative_mean=b.mean() if complete else np.nan,
                    difference=(b-a).mean() if complete else np.nan, pairs=len(group), expected_pairs=expected, complete=complete))
    return pd.DataFrame(rows)


def plot_factor_dynamics(run):
    """Means and paired differences share a compact dashboard, without dual axes."""
    pages = []
    metrics = ("train_loss", "l2sp_penalty", "grad_norm_mean",
               "clipped_frac" if run.track == "pd" else "weight_drift")
    for factor, label in (("l2sp_lambda", "L2-SP"), ("frozen", "Backbone freezing"), ("learning_rate", "Peak learning rate")):
        summaries = []
        for field in metrics:
            data = cp.optimization_curves(run, field)
            if data.empty:
                continue
            data = data.merge(run.trials[["trial_name", "partition", "seed"]], on="trial_name", validate="many_to_one")
            values = factor_statistics(run, data, factor, value=field)
            if values.empty or not values.difference.notna().any():
                continue
            summaries.append(values.assign(metric=field))
        if not summaries:
            continue
        combined = pd.concat(summaries, ignore_index=True)
        for base, part in combined.groupby("base"):
            fig, axes = _grid(f"{run.track.upper()} / {base}: {label} effects")
            for row, field in enumerate(("train_loss", "grad_norm_mean")):
                mean_ax, difference_ax = axes[row*2:row*2+2]
                selected = part[part.metric.eq(field)]
                if selected.empty:
                    mean_ax.remove(); difference_ax.remove(); continue
                first = True
                for alternative, curve in selected.groupby("alternative"):
                    color = style.TRAJECTORY_LR_COLORS.get(alternative, style.color("highlight")) if factor == "learning_rate" else style.color("highlight")
                    name = f"LR {alternative:.0e}" if factor == "learning_rate" else ("Frozen backbone" if factor == "frozen" else "L2-SP 0.003")
                    if first:
                        mean_ax.plot(curve["update"], curve.reference_mean, color=style.color("untuned"),
                            linestyle=style.FACTOR_ARM_STYLES[0], linewidth=style.THIN_LINE, label="Reference")
                        first = False
                    mean_ax.plot(curve["update"], curve.alternative_mean, color=color, linewidth=style.THIN_LINE, label=name)
                    difference_ax.plot(curve["update"], curve.difference, color=color, linewidth=style.THIN_LINE)
                difference_ax.axhline(0, color=style.color("reference"), linestyle=style.DIFFERENCE_STYLE)
                metric_label = cp.loss_label(run.track, base) if field == "train_loss" else FIELDS[field]
                mean_ax.set_ylabel(metric_label); difference_ax.set_ylabel("Alternative − reference")
                mean_ax.set_title("Matched arm means"); difference_ax.set_title("Paired differences")
                mean_ax.set_xlabel("Successful updates"); difference_ax.set_xlabel("Successful updates")
            _legend(fig, axes[0])
            pages.append(cp.Page(f"factor_dashboard_{run.track}_{factor}_{base}", fig,
                f"Matched arm means and paired differences for {label.lower()}. Each pair fixes the dataset partition, seed "
                "and every other recipe setting. Lines average all planned pairs with equal weight and omit incomplete windows. "
                "References are no anchoring, full parameter updates, or LR 3e-7. Penalty, displacement and clipping contrasts "
                "are tabulated alongside the displayed loss and gradient curves. Native objectives remain separate by architecture; "
                "smaller training loss is not itself evidence of improved predictive performance.",
                {"Arm means, paired differences and coverage for every diagnostic": part}))
    return pages


def plot_monitor_curves(run, split="test"):
    data = monitor_observations(run, split)
    if data.empty:
        return []
    curves = []
    for key, recipe in data.groupby(cp.FACTORS):
        curve = cp.trajectory_curve(run, recipe, split=split)
        curves.append(curve.assign(**dict(zip(cp.FACTORS, key))))
    values = pd.concat(curves, ignore_index=True)
    fig, axes = _grid(f"{run.track.upper()}: {'Non-credit retention' if split == 'ood' else 'Held-out credit'} — {monitor_label(run)} change")
    tables = []
    for ax, (base, part) in zip(axes, values.groupby("base")):
        factor = "sampling" if part.sampling.nunique() > 1 else "learning_rate"
        for arm, group in part.groupby(factor):
            curve = group.groupby("updates")["mean"].agg(mean="mean", recipes="count", expected="size")
            curve.loc[curve.recipes.ne(curve.expected), "mean"] = np.nan
            name = style.SAMPLING_LABELS[arm] if factor == "sampling" else f"LR {arm:.0e}"
            color = style.SAMPLING_COLORS[arm] if factor == "sampling" else style.TRAJECTORY_LR_COLORS.get(arm, style.color(str(arm)))
            ax.plot(curve.index, curve["mean"], label=name, color=color, linewidth=style.THIN_LINE)
            tables.append(curve.reset_index().assign(base=base, arm=name))
        ax.axhline(0, color=style.color("reference"), linestyle=":")
        ax.set_title(base); ax.set_ylabel("Change in " + monitor_label(run)); ax.set_xlabel("Successful updates")
    for ax in axes[values.base.nunique():]: ax.remove()
    _legend(fig, axes[0])
    return [cp.Page(f"monitor_dashboard_{run.track}_{split}", fig,
        "Each recipe first pairs each table's score with its update-zero value, averages repeated partitions within table, "
        "and gives tables equal weight. Lines then average remaining recipe settings at a peak learning rate or sampling protocol. "
        "The complete planned partition cohort and unchanged baseline table support are required at every milestone. Missing "
        "observations remain gaps. Positive changes indicate improvement. These fixed-context monitors are separate from five-fold evaluation.",
        {"Displayed curves": pd.concat(tables, ignore_index=True), "Complete recipe curves": values})]


def endpoint_levels(run, metric=None):
    data = monitor_observations(run, metric=metric)
    if data.empty:
        return data
    baseline = data[data.updates.eq(0)].groupby("trial").dataset.agg(set)
    data = data[data.updates.eq(run.target)]
    support = data.groupby("trial").dataset.agg(set)
    valid = [name for name, tables in support.items() if tables == baseline.get(name, set())]
    data = data[data.trial.isin(valid)]
    return data.groupby([*cp.FACTORS, "partition", "trial"], dropna=False)[["initial", "value", "effect"]].mean().reset_index()


def plot_followup_curves(run, *, reference=None, row_exposure=False):
    """One four-model dashboard per domain, with exact seed/protocol support."""
    from src.visualize import comparisons as cmp
    pages = []
    metric = monitor_metric(run)
    for domain, label in cmp.DOMAINS.items():
        if reference is not None:
            raw = cmp.seed_pairs(reference, run, domain=domain, metric=metric)
            data = cmp.dataset_means(raw, run, ["reference", "repeat", "difference"])
        else:
            data = monitor_observations(run, "ood" if domain == "ood" else "test")
        if data.empty: continue
        curves = []
        fig, axes = _grid(f"{run.track.upper()}: {label} — {'seed difference' if reference is not None else monitor_label(run)}")
        visible = 0
        for ax, (base, part) in zip(axes, data.groupby("base")):
            finite_curve = False
            if reference is not None:
                cohort = set(part.loc[part.updates.eq(0), "dataset"])
                support = part.groupby("updates").dataset.agg(lambda x: bool(cohort) and set(x) == cohort)
                curve = part.groupby("updates").difference.mean().where(support).reindex(list(run.cfg.train.trajectory_steps))
                ax.plot(curve.index, curve, color=style.color(base), linewidth=style.THIN_LINE, label="Seed 43 − seed 42")
                finite_curve = bool(np.isfinite(curve).any())
                curves.append(curve.rename("mean").rename_axis("updates").reset_index().assign(base=base))
                ax.set_ylabel("Difference in " + monitor_label(run) + " change")
            else:
                for mode, arm in part.groupby("sampling"):
                    curve = cp.trajectory_curve(run, arm, split="ood" if domain == "ood" else "test",
                                                x="processed_rows" if row_exposure else "updates")
                    ax.plot(curve.x, curve["mean"], color=style.SAMPLING_COLORS[mode],
                            linestyle=style.SAMPLING_STYLES[mode], linewidth=style.THIN_LINE, label=style.SAMPLING_LABELS[mode])
                    finite_curve |= bool(np.isfinite(curve["mean"]).any())
                    curves.append(curve.assign(base=base, sampling=mode))
                ax.set_ylabel("Change in " + monitor_label(run))
            ax.set_title(base); ax.set_xlabel("Processed row exposures" if row_exposure else "Successful updates")
            ax.axhline(0, color=style.color("reference"), linestyle=":")
            if not finite_curve:
                if row_exposure:
                    limit = pd.to_numeric(part.get("processed_rows", pd.Series(dtype=float)),
                                          errors="coerce").max()
                else:
                    limit = max(run.cfg.train.trajectory_steps)
                if np.isfinite(limit) and limit > 0:
                    ax.set_xlim(0, limit)
                ax.text(.5, .65, "Incomplete matched cohort", transform=ax.transAxes,
                        ha="center", va="center", fontsize=style.ANNOTATION_SIZE,
                        color=style.color("annotation"))
            visible += 1
        for ax in axes[visible:]: ax.remove()
        displayed = pd.concat(curves, ignore_index=True)
        if not np.isfinite(displayed["mean"]).any():
            plt.close(fig); continue
        _legend(fig, axes[0])
        pages.append(cp.Page(f"followup_curves_{run.track}_{domain}_{'seeds' if reference is not None else 'rows' if row_exposure else 'updates'}", fig,
            "Fixed-context monitoring changes use AUC for PD and R² for LGD. Seed contrasts match the same table and "
            "partition before averaging and require all repeated non-credit partitions. Protocol curves require every "
            "planned trial and unchanged baseline table support. Missing milestones remain gaps; panels without any "
            "complete matched cohort are labeled explicitly. Row exposure includes "
            "repeated rows and does not establish equal compute. These monitors are separate from five-fold evaluation.",
            {"Displayed curves": displayed, "Paired observations and dataset support": data}))
    return pages


def plot_resources(run):
    data = dg.load(run, "resources")
    if data.empty: return []
    data = data[data.phase.eq("training") & data.successful_updates.gt(0)].copy()
    fields = {"gpu_utilization_percent": "GPU utilization (%)", "power_watts": "Device power (W)",
              "device_memory_used_mib": "Device memory (MiB)"}
    width = cp.optimization_window(run)
    data["window_end"] = np.minimum(np.ceil(data.successful_updates/width)*width, run.target)
    pages = []
    for field, label in fields.items():
        if field not in data: continue
        data[field] = pd.to_numeric(data[field], errors="coerce")
        values = data.dropna(subset=[field]).groupby(["trial_name", "base", "window_end"])[field].median().reset_index()
        if values.empty: continue
        fig, axes = _grid(f"{run.track.upper()}: sampled {label.lower()}")
        tables = []
        for ax, (base, part) in zip(axes, values.groupby("base")):
            curve = part.groupby("window_end")[field].agg(median="median", count="count").reindex(cp.optimization_grid(run, "grad_norm_mean"))
            ax.plot(curve.index, curve["median"], color=style.color(base), linewidth=style.THIN_LINE)
            ax.set_title(base); ax.set_ylabel(label); ax.set_xlabel("Successful updates")
            tables.append(curve.reset_index().assign(base=base))
        for ax in axes[values.base.nunique():]: ax.remove()
        pages.append(cp.Page(f"resource_dashboard_{run.track}_{field}", fig,
            "Device counters during training, summarized first by a within-trial window median, then by the median "
            "across observed trials. Recipes and adaptation modes are pooled; trial counts accompany every value. "
            "Changing support can change these descriptive curves. Device samples are neither process-specific utilization "
            "nor integrated energy measurements. Missing counters remain gaps.", {"Displayed counters and trial support": pd.concat(tables, ignore_index=True)}))
    return pages


def plot_endpoints(run, *, by_partition=False):
    data = endpoint_levels(run)
    if data.empty:
        return []
    if not by_partition:
        data = data.groupby(cp.FACTORS).agg(initial=("initial", "mean"), value=("value", "mean"),
            partitions=("partition", "nunique")).reset_index()
        data = data[data.partitions.eq(int(run.cfg.corpus.n_splits))]
    if data.empty:
        return []
    fig, axes = _grid(f"{run.track.upper()}: initial and post-finetuning {monitor_label(run)}" + (" by partition" if by_partition else " — partition means"))
    for ax, (base, part) in zip(axes, data.groupby("base")):
        for (lr, lam, frozen), arm in part.groupby(["learning_rate", "l2sp_lambda", "frozen"]):
            ax.scatter(arm.initial, arm.value, color=style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr))),
                marker=style.PANEL_MARKERS[int(lam > 0)], s=style.POINT_SIZE,
                facecolors="none" if frozen else None, alpha=style.POINT_ALPHA)
            if by_partition:
                for row in arm.itertuples(): ax.annotate(str(row.partition+1), (row.initial, row.value), fontsize=style.ANNOTATION_SIZE)
        low, high = part[["initial", "value"]].min().min(), part[["initial", "value"]].max().max()
        ax.plot([low, high], [low, high], color=style.color("reference"), linestyle=":")
        ax.set_title(base); ax.set_xlabel("Initial " + monitor_label(run)); ax.set_ylabel("Post-finetuning " + monitor_label(run))
    for ax in axes[data.base.nunique():]: ax.remove()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr))), marker="o", linestyle="", label=f"LR {lr:.0e}") for lr in sorted(data.learning_rate.unique())]
    handles += [Line2D([], [], color=style.color("reference"), marker=style.PANEL_MARKERS[i], linestyle="", label=f"L2-SP {lam:g}") for i, lam in enumerate(sorted(data.l2sp_lambda.unique()))]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    return [cp.Page(f"endpoint_dashboard_{run.track}_{'partitions' if by_partition else 'means'}", fig,
        "Initial and post-finetuning scores for each completed configuration. "
        + ("Numbers identify individual training-dataset partitions. " if by_partition else "Each dot averages the four partition means with equal partition weight; incomplete configurations are omitted. ")
        + "Scores first average available held-out tables within each partition. Colors distinguish learning rates and marker shapes distinguish anchoring. "
        "Open marks freeze the transformer backbone; filled marks update all parameters. The diagonal indicates unchanged score. "
        "These fixed-context monitors do not substitute for final five-fold benchmark scores.", {"Displayed endpoints": data})]


def plot_endpoint_factors(run, *, benchmark=False):
    data = cp.benchmark_effects(run, monitor_metric(run)) if benchmark else monitor_observations(run)
    if data.empty:
        return []
    if not benchmark:
        data = data[data.updates.eq(run.target)]
    fig, axes = _grid(f"{run.track.upper()}: {'Benchmark' if benchmark else 'Monitoring'} effects of adaptation settings")
    tables = []
    for ax, factor in zip(axes, ("l2sp_lambda", "frozen", "learning_rate", "base")):
        if factor == "base":
            summary = data.groupby([*cp.FACTORS, "dataset"]).effect.mean().reset_index().groupby("base").effect.mean().reset_index()
            ax.barh(summary.base, summary.effect, color=[style.color(b) for b in summary.base]); ax.set_xlabel("Mean change in " + monitor_label(run))
            tables.append(summary.assign(factor=factor))
        else:
            ref = {"l2sp_lambda": 0., "frozen": False, "learning_rate": 3e-7}[factor]
            for i, alt in enumerate(sorted(set(data[factor])-{ref})):
                contrast = cp.factor_contrasts(data, factor, ref, alt)
                if contrast.empty: continue
                summary = contrast.groupby("base").contrast.mean().reset_index()
                ax.plot(summary.contrast, summary.base, marker=style.PANEL_MARKERS[i%4], linestyle="",
                    color=style.TRAJECTORY_LR_COLORS.get(alt, style.color("highlight")) if factor == "learning_rate" else style.color("highlight"),
                    label=f"{alt:.0e} − 3e-7" if factor == "learning_rate" else ("Frozen − all parameters" if factor == "frozen" else "0.003 − 0"))
                tables.append(summary.assign(factor=factor, reference=str(ref), alternative=str(alt)))
            ax.set_xlabel("Paired change in " + monitor_label(run)); ax.legend()
        ax.set_title({"l2sp_lambda": "L2-SP", "frozen": "Backbone freezing", "learning_rate": "Peak learning rate", "base": "Base-model average"}[factor])
        ax.axvline(0, color=style.color("reference"), linestyle=":")
    return [cp.Page(f"endpoint_factors_{run.track}_{benchmark}", fig,
        "Conditional differences match dataset and every other configuration setting before averaging within model. "
        "Peak-learning-rate contrasts use LR 3e-7 as reference. The base-model panel averages configuration effects relative "
        "to each model's own initialization; it is descriptive rather than a causal architecture contrast. Positive values favor "
        "the alternative setting. Dataset and configuration support are retained in the accompanying tables.",
        {"Displayed model summaries": pd.concat(tables, ignore_index=True), "Source dataset effects": data})]


def plot_brier(run):
    data = cp.benchmark_effects(run, "brier_score")
    source = "five-fold benchmark"
    if data.empty:
        data = monitor_observations(run, metric="brier_score")
        source = "fixed-context monitors"
        if not data.empty: data = data[data.updates.eq(run.target)]
    if data.empty: return []
    fig, axes = cp._subplots(f"PD: Brier-score improvement ({source})")
    values = data.groupby([*cp.FACTORS, "dataset"]).effect.mean().reset_index().groupby("base").effect.agg(mean="mean", count="size").reset_index()
    axes[0].barh(values.base, values["mean"], color=[style.color(b) for b in values.base])
    axes[0].axvline(0, color=style.color("reference"), linestyle=":")
    axes[0].set_xlabel("Initial minus post-finetuning Brier score")
    return [cp.Page("brier_dashboard", fig, f"Brier-score reductions from {source}, averaged with equal weight over "
        "observed configuration–dataset pairs. Positive values indicate better probability accuracy. Brier score combines "
        "calibration and discrimination; it is not a pure calibration metric. Missing comparisons are omitted, with support counts reported.",
        {"Displayed reductions": values, "Source Brier-score effects": data})]
