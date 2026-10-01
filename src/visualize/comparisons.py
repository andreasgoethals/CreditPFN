"""Paired seed/protocol analyses, retaining dataset partitions until after matching."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.experiment import scientific_config
from src.visualize import campaign as cp, style
from src.visualize.paper_figures import effect_label

DOMAINS = {"credit": "Held-out credit", "ood": "Non-credit retention"}
MODES = list(style.SAMPLING_LABELS)


def reference_campaign(run):
    """Select the predefined main recipe without mutating the loaded campaign."""
    trials = cp.reference_rows(run.trials)
    names = set(trials.trial_name)
    def selected(frame):
        return frame[frame.trial_name.isin(names)].copy() if "trial_name" in frame else frame.copy()
    return cp.Campaign(run.cfg, trials, selected(run.trajectories), selected(run.histories), run.evaluation)


def observations(run, *, domain="credit", benchmark=False, metric=None, endpoint=False):
    if domain not in DOMAINS:
        raise ValueError("Expected credit or ood")
    metric = metric or run.metric
    if benchmark:
        data = cp.benchmark_effects(run, metric, domain=domain, by_partition=True)
    else:
        if metric != run.metric:
            raise ValueError("Trajectory comparisons use the recorded primary monitor")
        data = cp.effects(run, "test" if domain == "credit" else "ood")
        if not data.empty:
            data = data.merge(run.trials[["trial_name", "partition"]], left_on="trial", right_on="trial_name",
                              validate="many_to_one")
            if endpoint:
                data = data[data.updates.eq(run.target)]
    if data.empty:
        return data
    keys = cp.FACTORS + ["dataset", "partition"] + ([] if benchmark else ["updates"])
    if data.duplicated(keys).any():
        raise ValueError("Duplicate observations in a paired comparison")
    return data.assign(domain=domain, metric=metric)


def seed_pairs(main, repeat, *, domain="credit", benchmark=False, metric=None, endpoint=False):
    """Match the same partition before averaging repeated non-credit datasets."""
    specs = [scientific_config(run.cfg) for run in (main, repeat)]
    for spec in specs:
        spec.pop("seed", None)
    if specs[0] != specs[1] or main.track != repeat.track or main.cfg.seed == repeat.cfg.seed:
        raise ValueError("Seed comparisons require distinct seeds and otherwise identical reference settings")
    left, right = [cp.reference_rows(observations(run, domain=domain, benchmark=benchmark,
                    metric=metric, endpoint=endpoint)) for run in (main, repeat)]
    if left.empty or right.empty:
        return pd.DataFrame()
    keys = cp.FACTORS + ["partition", "dataset", "domain", "metric"] + ([] if benchmark else ["updates"])
    pairs = left[keys+["effect"]].merge(right[keys+["effect"]], on=keys,
                                       suffixes=("_reference", "_repeat"), validate="one_to_one")
    pairs = pairs.rename(columns={"effect_reference": "reference", "effect_repeat": "repeat"})
    return pairs.assign(difference=pairs.repeat-pairs.reference,
                        seed_reference=int(main.cfg.seed), seed_repeat=int(repeat.cfg.seed))


def dataset_means(data, run, values):
    """One credit partition or all OOD partitions per dataset; no changing OOD cohort."""
    if data.empty:
        return pd.DataFrame()
    keys = [c for c in ("base", "learning_rate", "l2sp_lambda", "frozen", "sampling", "dataset",
                       "domain", "metric", "updates", "comparison", "seed_reference", "seed_repeat") if c in data]
    if data.duplicated(keys+["partition"]).any():
        raise ValueError("Repeated partition in a dataset comparison")
    rows = []
    for key, group in data.groupby(keys, dropna=False):
        domain = group.domain.iloc[0]
        valid = (set(group.partition) == set(range(int(run.cfg.corpus.n_folds)))) if domain == "ood" else len(group) == 1
        if valid and np.isfinite(group[values].to_numpy(float)).all():
            rows.append(dict(zip(keys, key), **{c: group[c].mean() for c in values}, partitions=len(group)))
    return pd.DataFrame(rows)


def descriptive_summary(data, value="effect"):
    if data.empty:
        return pd.DataFrame()
    keys = [c for c in ("domain", "metric", "base", "sampling", "comparison") if c in data]
    return data.groupby(keys, dropna=False)[value].agg(
        datasets="size", mean="mean", median="median", minimum="min", maximum="max").reset_index()


def benchmark_coverage(run):
    """Report completed trained cells and those with a complete untuned counterpart."""
    rows = []
    for domain in DOMAINS:
        paired = observations(run, domain=domain, benchmark=True)
        for base, trials in run.trials.groupby("base"):
            part = paired[paired.base.eq(base)] if not paired.empty else paired
            rows.append(dict(domain=domain, base=base, planned_checkpoints=len(trials),
                completed_checkpoints=int(trials.status.eq("OK").sum()),
                paired_five_fold_comparisons=len(part),
                paired_datasets=part.dataset.nunique() if not part.empty else 0))
    return pd.DataFrame(rows)


def benchmark_scores(run):
    """Absolute primary scores for adapted, untuned and classical model context."""
    if run.evaluation.empty or run.metric not in run.evaluation:
        return pd.DataFrame()
    data = cp._complete_folds(run.evaluation, run.metric)
    keys = [c for c in ("domain", "method_name", "test_dataset_id", "split") if c in data]
    return data.groupby(keys, dropna=False)[run.metric].agg(
        mean="mean", minimum="min", maximum="max", folds="size").reset_index().assign(metric=run.metric)


def plot_seed_endpoints(main, repeat, *, benchmark=False):
    pages = []
    stage = "benchmark" if benchmark else "monitor"
    for domain, label in DOMAINS.items():
        raw = seed_pairs(main, repeat, domain=domain, benchmark=benchmark, endpoint=True)
        data = dataset_means(raw, repeat, ["reference", "repeat", "difference"])
        if data.empty:
            continue
        fig, axes = cp._subplots(f"{repeat.track.upper()}: seed effects / {label.lower()}", 2)
        for i, (base, group) in enumerate(data.groupby("base")):
            axes[0].scatter(group.reference, group.repeat, label=style.model_label(base),
                            color=style.color(base), marker=style.PANEL_MARKERS[i % len(style.PANEL_MARKERS)], s=style.POINT_SIZE)
            y = i + np.linspace(-style.JITTER_WIDTH, style.JITTER_WIDTH, len(group))
            axes[1].scatter(group.difference, y, color=style.color(base), s=style.POINT_SIZE/2, alpha=style.POINT_ALPHA)
            axes[1].plot([group.difference.mean()]*2, [i-style.JITTER_WIDTH, i+style.JITTER_WIDTH],
                         color=style.color("reference"), linewidth=style.THIN_LINE)
        low = min(data.reference.min(), data.repeat.min(), 0)
        high = max(data.reference.max(), data.repeat.max(), 0)
        axes[0].plot([low, high], [low, high], linestyle=":", color=style.color("reference"))
        axes[0].set_xlabel(f"Seed {int(main.cfg.seed)}: effect")
        axes[0].set_ylabel(f"Seed {int(repeat.cfg.seed)}: effect")
        axes[0].legend()
        axes[1].set_yticks(range(data.base.nunique()), [style.model_label(b) for b in sorted(data.base.unique())])
        axes[1].axvline(0, linestyle=":", color=style.color("reference"))
        axes[1].set_xlabel("Seed 43 minus seed 42 effect")
        pages.append(cp.Page(f"seed_{stage}_{domain}", fig,
            f"{label}: {'five-fold benchmark' if benchmark else str(repeat.target)+ '-update monitor'} effects for the predefined full-update reference, "
            f"LR 3e-7 and L2-SP 0.003. Effects are {effect_label(repeat.metric)} relative to each model's original weights. "
            "Observations match on dataset and training partition before averaging. Each non-credit dataset requires all four matched partitions; "
            "credit datasets have one held-out partition. Points are datasets and vertical marks in the difference panel are dataset means. "
            "Positive differences favor seed 43; two seeds do not estimate a general seed distribution.",
            {"Matched partition observations": raw, "Dataset effects": data,
             "Descriptive seed differences": descriptive_summary(data, "difference")}))
    return pages


def plot_seed_curves(main, repeat):
    data = {}
    for domain in DOMAINS:
        raw = seed_pairs(main, repeat, domain=domain)
        data[domain] = dataset_means(raw, repeat, ["reference", "repeat", "difference"])
    pages = []
    for base in sorted(repeat.trials.base.unique()):
        domains = [d for d, frame in data.items() if not frame.empty and frame.base.eq(base).any()
                   and frame.loc[frame.base.eq(base), "updates"].gt(0).any()]
        if not domains:
            continue
        fig, axes = cp._subplots(f"{repeat.track.upper()} / {base}: seed difference through training", len(domains))
        tables = {}
        for ax, domain in zip(axes, domains):
            group = data[domain][data[domain].base.eq(base)]
            updates = list(repeat.cfg.train.trajectory_steps)
            for _, series in group.groupby("dataset"):
                values = series.set_index("updates").difference.reindex(updates)
                ax.plot(values.index, values, color=style.color("annotation"), linewidth=style.THIN_LINE, alpha=style.POINT_ALPHA)
            expected = set(group.loc[group.updates.eq(0), "dataset"])
            complete = group.groupby("updates").dataset.agg(lambda x: bool(expected) and set(x) == expected)
            mean = group.groupby("updates").difference.mean().where(complete).reindex(updates)
            ax.plot(mean.index, mean, color=style.color(base), linewidth=style.THIN_LINE, label="Dataset mean")
            ax.axhline(0, linestyle=":", color=style.color("reference"))
            ax.set_title(DOMAINS[domain]); ax.set_xlabel("Successful optimizer updates")
            ax.set_ylabel("Seed 43 minus seed 42 effect"); ax.legend()
            tables[domain+" matched dataset differences"] = group
            tables[domain+" dataset mean"] = mean.rename("difference").rename_axis("updates").reset_index()
        pages.append(cp.Page(f"seed_curves_{base}", fig,
            "Matched seed differences in fixed-update monitoring effects. Thin gray lines show datasets; the colored mean retains "
            "the same baseline dataset support at every milestone. Non-credit dataset effects require all four matched training partitions. "
            "Gaps remain gaps; these are small-context monitors, separate from the final five-fold benchmark.", tables))
    return pages


def sampling_matched(run, *, domain="credit", benchmark=False, metric=None, endpoint=True):
    """Require all three protocols on the same dataset and partition."""
    data = observations(run, domain=domain, benchmark=benchmark, metric=metric, endpoint=endpoint)
    if data.empty:
        return data
    keys = [c for c in cp.FACTORS if c != "sampling"] + ["dataset", "partition", "domain", "metric"]
    if not benchmark:
        keys += ["updates"]
    support = data.groupby(keys).sampling.agg(lambda x: set(x) == set(MODES)).rename("complete").reset_index()
    return data.merge(support[support.complete][keys], on=keys, validate="many_to_one")


def sampling_contrasts(data):
    if data.empty:
        return pd.DataFrame()
    keys = [c for c in cp.FACTORS if c != "sampling"] + ["dataset", "partition", "domain", "metric"]
    if "updates" in data:
        keys += ["updates"]
    rows = []
    reference = data[data.sampling.eq("one_sample")][keys+["effect"]]
    for mode in ("full_pass", "accumulate"):
        paired = data[data.sampling.eq(mode)][keys+["effect"]].merge(reference, on=keys,
                              suffixes=("_alternative", "_reference"), validate="one_to_one")
        rows.append(paired.assign(comparison=mode, difference=paired.effect_alternative-paired.effect_reference))
    return pd.concat(rows, ignore_index=True)


def _intervals(ax, data, *, value, factor, labels, colors):
    bases = sorted(data.base.unique())
    for offset, level in enumerate(labels):
        part = data[data[factor].eq(level)]
        for base, group in part.groupby("base"):
            y = bases.index(base) + (offset-(len(labels)-1)/2)*style.JITTER_WIDTH
            ax.plot([group[value].quantile(.25), group[value].quantile(.75)], [y, y], color=colors[level], linewidth=style.THIN_LINE)
            ax.scatter(group[value].mean(), y, color=colors[level], s=style.POINT_SIZE, label=labels[level])
    ax.set_yticks(range(len(bases)), [style.model_label(b) for b in bases])
    ax.axvline(0, linestyle=":", color=style.color("reference")); ax.legend()


def plot_sampling_endpoints(run, *, benchmark=False):
    pages = []
    stage = "benchmark" if benchmark else "monitor"
    for domain, label in DOMAINS.items():
        raw = sampling_matched(run, domain=domain, benchmark=benchmark)
        effects = dataset_means(raw, run, ["effect"])
        contrasts = dataset_means(sampling_contrasts(raw), run, ["difference"])
        if effects.empty or contrasts.empty:
            continue
        fig, axes = cp._subplots(f"{run.track.upper()}: sampling / {label.lower()}", 2)
        _intervals(axes[0], effects, value="effect", factor="sampling", labels=style.SAMPLING_LABELS, colors=style.SAMPLING_COLORS)
        _intervals(axes[1], contrasts, value="difference", factor="comparison",
                   labels={m: style.SAMPLING_LABELS[m] for m in MODES if m != "one_sample"}, colors=style.SAMPLING_COLORS)
        axes[0].set_title("Effect relative to original weights"); axes[0].set_xlabel(effect_label(run.metric))
        axes[1].set_title("Difference relative to one sample"); axes[1].set_xlabel("Difference in paired effect (+ better)")
        pages.append(cp.Page(f"sampling_{stage}_{domain}", fig,
            f"{label}: {'five-fold benchmark' if benchmark else str(run.target)+'-update monitor'} effects and within-experiment protocol contrasts. "
            "Points give equal-weight dataset means; horizontal segments give dataset interquartile ranges, not confidence intervals. "
            "All three protocols must be available on the same dataset/partition before comparison; non-credit datasets also require all four partitions. "
            f"Effects use {effect_label(run.metric)}. Every arm preserves the same experiment-3 prevalence policy.",
            {"Matched partition effects": raw, "Dataset effects": effects, "Dataset protocol contrasts": contrasts,
             "Descriptive effects": descriptive_summary(effects), "Descriptive contrasts": descriptive_summary(contrasts, "difference")}))
    return pages


def metric_tables(run, *, main=None):
    metrics = ("roc_auc", "pr_auc", "brier_score", "log_loss", "ece", "f1", "ece_platt", "ece_isotonic") if run.track == "pd" else ("rmse", "mae", "r2", "crps")
    raw, means = [], []
    for domain in DOMAINS:
        for metric in metrics:
            if main is not None:
                paired = seed_pairs(main, run, domain=domain, benchmark=True, metric=metric)
                summary = dataset_means(paired, run, ["reference", "repeat", "difference"])
            else:
                paired = sampling_contrasts(sampling_matched(run, domain=domain, benchmark=True, metric=metric))
                summary = dataset_means(paired, run, ["difference"])
            if not paired.empty:
                raw.append(paired)
            if not summary.empty:
                means.append(summary)
    return {"Matched benchmark metric contrasts": pd.concat(raw, ignore_index=True) if raw else pd.DataFrame(),
            "Dataset benchmark metric contrasts": pd.concat(means, ignore_index=True) if means else pd.DataFrame(),
            "Descriptive metric contrasts": descriptive_summary(pd.concat(means, ignore_index=True), "difference") if means else pd.DataFrame()}


def sampling_costs(run):
    columns = [c for c in ("gpu_hours", "rows_seen", "sec_per_step", "peak_gpu_gb") if c in run.trials]
    if not columns:
        return pd.DataFrame()
    data = run.trials.loc[run.trials.status.eq("OK"), [*cp.FACTORS, "partition", *columns]].copy()
    if data.empty:
        return pd.DataFrame()
    keys = [c for c in cp.FACTORS if c != "sampling"]+["partition"]
    complete = data.groupby(keys).sampling.agg(lambda x: set(x) == set(MODES)).rename("complete").reset_index()
    data = data.merge(complete[complete.complete][keys], on=keys, validate="many_to_one")
    reference = data[data.sampling.eq("one_sample")][keys+columns]
    data = data.merge(reference, on=keys, suffixes=("", "_reference"), validate="many_to_one")
    for column in columns:
        data[column+"_ratio"] = data[column] / data[column+"_reference"].where(data[column+"_reference"].gt(0))
    return data


def seed_costs(main, repeat):
    left, right = [reference_campaign(run).trials for run in (main, repeat)]
    columns = [c for c in ("gpu_hours", "rows_seen", "sec_per_step", "peak_gpu_gb", "final_drift") if c in left and c in right]
    if not columns:
        return pd.DataFrame()
    keys = cp.FACTORS+["partition"]
    return left.loc[left.status.eq("OK"), keys+columns].merge(
        right.loc[right.status.eq("OK"), keys+columns], on=keys,
        suffixes=("_seed42", "_seed43"), validate="one_to_one")


def plot_sampling_costs(run):
    data = sampling_costs(run)
    fields = [c for c in ("gpu_hours_ratio", "rows_seen_ratio") if c in data and np.isfinite(data[c]).any()]
    if not fields:
        return []
    fig, axes = cp._subplots(f"{run.track.upper()}: sampling cost relative to one sample", len(fields))
    for ax, field in zip(axes, fields):
        _intervals(ax, data, value=field, factor="sampling", labels=style.SAMPLING_LABELS, colors=style.SAMPLING_COLORS)
        ax.axvline(1, color=style.color("annotation"), linestyle=":")
        ax.set_xlabel("Recorded trial-time ratio" if field.startswith("gpu") else "Processed row-exposure ratio")
    return [cp.Page("sampling_costs", fig,
        "Ratios to the one-sample trial matched on base, recipe and dataset partition. Only partitions with all three completed protocols enter. "
        "Points are partition means and segments are partition interquartile ranges, not uncertainty intervals. Trial time includes recorded monitoring "
        "and excludes queueing; it is not billed GPU allocation time. Row exposure counts repeated rows. Absolute costs and peak memory accompany the ratios.",
        {"Matched partition costs and ratios": data})]


def plot_sampling_quality_cost(run):
    costs = sampling_costs(run)
    if costs.empty or "gpu_hours" not in costs:
        return []
    hours = pd.to_numeric(costs.gpu_hours, errors="coerce")
    costs = costs.loc[np.isfinite(hours) & hours.gt(0)].copy()
    if costs.empty:
        return []
    # A final efficiency point requires every planned training partition's cost.
    counts = costs.groupby(["base", "sampling"]).partition.nunique()
    eligible = counts[counts.eq(int(run.cfg.corpus.n_folds))].index
    costs = costs.set_index(["base", "sampling"]).loc[eligible].reset_index()
    if costs.empty:
        return []
    records = []
    for domain in DOMAINS:
        data = dataset_means(sampling_matched(run, domain=domain, benchmark=True), run, ["effect"])
        if data.empty:
            continue
        quality = data.groupby(["base", "sampling"]).agg(effect=("effect", "mean"), datasets=("dataset", "nunique")).reset_index()
        runtime = costs.groupby(["base", "sampling"]).gpu_hours.sum().rename("hours").reset_index()
        records.append(quality.merge(runtime, on=["base", "sampling"], validate="one_to_one").assign(domain=domain))
    if not records:
        return []
    shown = pd.concat(records, ignore_index=True)
    domains = [d for d in DOMAINS if shown.domain.eq(d).any()]
    fig, axes = cp._subplots(f"{run.track.upper()}: benchmark quality and training cost", len(domains))
    for ax, domain in zip(axes, domains):
        for i, (base, group) in enumerate(shown[shown.domain.eq(domain)].groupby("base")):
            marker = style.PANEL_MARKERS[i % len(style.PANEL_MARKERS)]
            ax.scatter([], [], color=style.color("annotation"), marker=marker, s=style.POINT_SIZE, label=style.model_label(base))
            for row in group.itertuples():
                ax.scatter(row.hours, row.effect, color=style.SAMPLING_COLORS[row.sampling],
                           marker=marker, s=style.POINT_SIZE)
        for mode in MODES:
            ax.scatter([], [], color=style.SAMPLING_COLORS[mode], s=style.POINT_SIZE, label=style.SAMPLING_LABELS[mode])
        ax.axhline(0, linestyle=":", color=style.color("reference")); ax.set_xscale("log")
        ax.set_title(DOMAINS[domain]); ax.set_xlabel("Sum of four recorded trial times (hours)")
        ax.set_ylabel(effect_label(run.metric)); ax.legend()
    return [cp.Page("sampling_benchmark_quality_cost", fig,
        "Final benchmark effects versus recorded training time summed over four dataset partitions. Quality gives equally weighted datasets in the "
        "common three-protocol cohort; its size is recorded alongside every point. Non-credit effects first average four matched partitions per dataset. "
        "Colors identify protocols and marker shapes identify model bases. Evaluation and queueing costs are excluded. "
        "This is an observed cost comparison, not a matched-compute experiment or an estimated optimal frontier.",
        {"Benchmark quality and cost": shown, "Matched partition costs": costs})]


def plot_sampling_curves(run, *, row_exposure=False):
    pages = []
    data = {d: observations(run, domain=d) for d in DOMAINS}
    for base in sorted(run.trials.base.unique()):
        curves = {}
        for domain, frame in data.items():
            if frame.empty:
                continue
            for mode, part in frame[frame.base.eq(base)].groupby("sampling"):
                curves[domain, mode] = cp.trajectory_curve(run, part, split="ood" if domain == "ood" else "test",
                                                     x="processed_rows" if row_exposure else "updates")
        domains = [d for d in DOMAINS if any(domain == d and np.isfinite(curve.loc[curve.updates.gt(0), "mean"]).any()
                                              for (domain, mode), curve in curves.items())]
        if not domains:
            continue
        fig, axes = cp._subplots(f"{run.track.upper()} / {base}: protocol trajectories", len(domains))
        for ax, domain in zip(axes, domains):
            for mode in MODES:
                curve = curves.get((domain, mode))
                if curve is None:
                    continue
                ax.plot(curve.x, curve["mean"], color=style.SAMPLING_COLORS[mode], linestyle=style.SAMPLING_STYLES[mode],
                        linewidth=style.THIN_LINE, label=style.SAMPLING_LABELS[mode])
            ax.axhline(0, color=style.color("reference"), linestyle=":")
            ax.set_title(DOMAINS[domain]); ax.set_ylabel(effect_label(run.metric))
            ax.set_xlabel("Median processed row exposures" if row_exposure else "Successful optimizer updates"); ax.legend()
        pages.append(cp.Page(f"sampling_curves_{base}_{'rows' if row_exposure else 'updates'}", fig,
            "Fixed-update monitoring effects on held-out credit and non-credit datasets. Each protocol curve requires every planned trial and its "
            "unchanged baseline dataset support at the plotted milestone. Missing milestones remain gaps. Row counts include repeated exposures and "
            "are medians across observed trials; connecting measurements does not create a matched-row or matched-compute experiment.",
            {f"{d} / {m} curve and coverage": curve for (d, m), curve in curves.items()}))
    return pages
