"""Compact training-choice and execution summaries with explicit matched support.

Training notebooks keep native objectives in their architecture-specific panels.
These summaries compare like-for-like monitored scores and recorded costs without
turning optimization diagnostics into claims about final predictive performance.
"""
from __future__ import annotations

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualize import campaign as cp, dashboards as db, diagnostics as dg, style


CHOICES = (("l2sp_lambda", 0., "Add L2-SP"),
           ("frozen", False, "Freeze backbone"),
           ("learning_rate", 3e-7, "Peak learning rate"))


def choice_label(factor, alternative):
    if factor == "learning_rate":
        return f"LR {alternative:.0e} vs {3e-7:.0e}"
    return "Add L2-SP" if factor == "l2sp_lambda" else "Freeze backbone"


def seed_caption(run):
    """Identify the observed training seeds without inventing missing provenance."""
    if "seed" not in run.trials:
        return "Training-seed provenance is unavailable. "
    seeds = sorted(pd.to_numeric(run.trials.seed, errors="coerce").dropna().unique())
    if not seeds:
        return "Training-seed provenance is unavailable. "
    label = "Training seed" if len(seeds) == 1 else "Training seeds"
    return f"{label}: {', '.join(f'{seed:g}' for seed in seeds)}. "


def monitor_choice_tables(run):
    """Require identical planned pairs and baseline table support at each milestone.

    Repeated recipes and non-credit partitions are averaged within each table
    before the equal-table mean; architecture effects are never pooled.
    """
    summaries, tables = [], []
    for domain, split in (("credit", "test"), ("retention", "ood")):
        data = db.monitor_observations(run, split)
        if data.empty:
            continue
        for factor, reference, _ in CHOICES:
            keys = [c for c in cp.FACTORS if c != factor] + ["partition", "seed"]
            for alternative in sorted(set(run.trials[factor]) - {reference}):
                plan = run.trials[run.trials[factor].eq(reference)][keys].merge(
                    run.trials[run.trials[factor].eq(alternative)][keys],
                    on=keys, validate="one_to_one")
                left, right = [data[data[factor].eq(v)] for v in (reference, alternative)]
                pair = left[keys + ["dataset", "updates", "effect"]].merge(
                    right[keys + ["dataset", "updates", "effect"]],
                    on=keys + ["dataset", "updates"], suffixes=("_reference", "_alternative"),
                    validate="one_to_one")
                for base, planned in plan.groupby("base"):
                    group = pair[pair.base.eq(base)]
                    initial = group[group.updates.eq(0)]
                    observed_keys = set(initial[keys].itertuples(index=False, name=None))
                    planned_keys = set(planned[keys].itertuples(index=False, name=None))
                    support_keys = keys + ["dataset"]
                    roster = data[data.updates.eq(0)][["partition", "dataset"]].drop_duplicates()
                    if domain == "retention":
                        # Retention tables must be present in every training partition.
                        roster = pd.MultiIndex.from_product(
                            [planned.partition.unique(), roster.dataset.unique()],
                            names=["partition", "dataset"]).to_frame(index=False)
                    wanted = planned.merge(roster, on="partition", validate="many_to_many")
                    support = set(wanted[support_keys].itertuples(index=False, name=None))
                    actual_initial = set(initial[support_keys].itertuples(index=False, name=None))
                    baseline_complete = (bool(support) and observed_keys == planned_keys
                                         and actual_initial == support)
                    for update in sorted(data.updates.unique()):
                        current = group[group.updates.eq(update)]
                        actual = set(current[support_keys].itertuples(index=False, name=None))
                        complete = baseline_complete and actual == support
                        values = current.groupby("dataset")[["effect_reference", "effect_alternative"]].mean()
                        values["difference"] = values.effect_alternative - values.effect_reference
                        common = dict(domain=domain, base=base, factor=factor, reference=reference,
                                      alternative=alternative, choice=choice_label(factor, alternative),
                                      updates=update, complete=complete)
                        summaries.append(dict(**common,
                            reference_mean=values.effect_reference.mean() if complete else np.nan,
                            alternative_mean=values.effect_alternative.mean() if complete else np.nan,
                            difference=values.difference.mean() if complete else np.nan,
                            tables=len(values), pairs=len(current), expected_pairs=len(support)))
                        if update == run.target and complete:
                            tables.append(values.reset_index().assign(**common))
    return (pd.DataFrame(summaries),
            pd.concat(tables, ignore_index=True) if tables else pd.DataFrame())


def optimization_choice_tables(run):
    """Retain all native-scale paired diagnostic curves as auditable tables."""
    frames = []
    fields = ("train_loss", "l2sp_penalty", "grad_norm_mean", "weight_drift", "clipped_frac")
    for field in fields:
        values = cp.optimization_curves(run, field)
        if values.empty:
            continue
        values = values.merge(run.trials[["trial_name", "partition", "seed"]],
                              on="trial_name", validate="many_to_one")
        for factor, _, _ in CHOICES:
            result = db.factor_statistics(run, values, factor, value=field)
            if not result.empty:
                frames.append(result.assign(metric=field))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_choices(run):
    curves, table_effects = monitor_choice_tables(run)
    if curves.empty:
        return []
    endpoint = curves[curves.updates.eq(run.target)]
    if endpoint.empty or not endpoint.difference.notna().any():
        return []
    choices = list(dict.fromkeys(endpoint.choice))
    bases = sorted(endpoint.base.unique())
    domains = [(d, title) for d, title in (("credit", "Held-out credit tables"),
               ("retention", "Non-credit retention tables"))
               if endpoint.loc[endpoint.domain.eq(d), "difference"].notna().any()]
    fig, axes = plt.subplots(len(domains), 1, squeeze=False, layout="constrained",
                             figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO))
    label = db.monitor_label(run)
    fig.suptitle(f"{run.track.upper()}: training-choice effects on monitored {label} at {run.target:,} updates")
    extent = max(float(endpoint.difference.abs().max()), np.finfo(float).eps)
    for ax, (domain, title) in zip(axes.flat, domains):
        values = endpoint[endpoint.domain.eq(domain)].pivot(index="base", columns="choice", values="difference")
        values = values.reindex(index=bases, columns=choices)
        picture = ax.imshow(values.to_numpy(), aspect="auto", cmap=style.CMAP_DIVERGING,
                            vmin=-extent, vmax=extent)
        ax.set_yticks(range(len(bases)), [style.model_label(b) for b in bases])
        ax.set_xticks(range(len(choices)), [c.replace(" vs ", "\nvs ") if c.startswith("LR ")
                     else c.replace(" ", "\n", 1) for c in choices])
        ax.set_title(title)
        for row in range(len(bases)):
            for column in range(len(choices)):
                value = values.iloc[row, column]
                if np.isfinite(value):
                    color = style.GRID_TEXT_LIGHT if abs(value) > extent*.65 else style.GRID_TEXT_DARK
                    ax.text(column, row, f"{value:+.3f}", ha="center", va="center",
                            color=color, fontsize=style.GRID_FONT)
    fig.colorbar(picture, ax=list(axes.flat), label=f"Paired change in {label}")
    support_text = "; ".join(f"{domain}: " + ", ".join(
        f"{style.model_label(base)} {int(group.tables.min())} tables"
        for base, group in subset.groupby("base"))
        for domain, subset in endpoint.groupby("domain"))
    return [cp.Page(f"training_choice_summary_{run.track}", fig,
        f"Fixed-context monitor differences at {run.target:,} successful updates. Each cell compares a changed "
        "training choice with its matched reference: L2-SP 0.003 versus 0, transformer-backbone freezing versus "
        "all-parameter updates, or the stated peak learning rate versus 3e-7. Positive values favor the changed "
        "choice. Each pair fixes base model, training partition, seed and every other recipe setting. Within "
        "the displayed metric and domain, all planned pairs and their unchanged update-zero table support are "
        "required; repeated recipes and partitions are "
        "averaged within table before the equal-table mean. Models remain separate. Counts and full paired "
        f"monitor trajectories accompany the values. Observed support: {support_text}. Native loss and optimizer contrasts are retained as tables. "
        "These held-out monitoring contexts are separate from the five-fold final benchmark; no recipe is selected here.",
        {"Displayed final monitor choice effects and support": endpoint,
         "Table-level final paired monitor effects": table_effects,
         "Paired monitor effects throughout training": curves,
         "Native-scale paired optimization dynamics": optimization_choice_tables(run)},
        {"measurement": "fixed_context_monitor", "metric": db.monitor_metric(run),
         "updates": int(run.target), "aggregation_unit": "table", "contrast": "matched_training_choice"})]


def movement_tables(run):
    """Aggregate only complete planned model/scope cohorts at exact milestones."""
    whole = cp.optimization_curves(run, "weight_drift")
    parameters = dg.load(run, "parameters")
    pieces = []
    if not whole.empty:
        pieces.append(whole[["trial_name", "base", "frozen", "window_end", "weight_drift"]]
                      .rename(columns={"window_end": "updates", "weight_drift": "value"})
                      .assign(measure="Whole-model relative displacement"))
    if not parameters.empty and "relative_change" in parameters:
        tensors = parameters.groupby(["trial_name", "base", "frozen", "successful_updates"]).relative_change.median().reset_index()
        pieces.append(tensors.rename(columns={"successful_updates": "updates", "relative_change": "value"})
                      .assign(measure="Median relative tensor displacement"))
    if not pieces:
        return pd.DataFrame()
    source = pd.concat(pieces, ignore_index=True)
    source["value"] = pd.to_numeric(source.value, errors="coerce")
    expected = run.trials.groupby(["base", "frozen"]).size().rename("expected_trials")
    result = source.groupby(["base", "frozen", "measure", "updates"]).value.agg(
        median="median", trials="count").reset_index().merge(expected.reset_index(), on=["base", "frozen"])
    result["complete"] = result.trials.eq(result.expected_trials)
    result.loc[~result.complete, "median"] = np.nan
    return result


def plot_movement(run):
    data = movement_tables(run)
    if data.empty or not data["median"].notna().any():
        return []
    measures = list(data.measure.unique())
    fig, axes = plt.subplots(1, len(measures), squeeze=False, layout="constrained",
                             figsize=style.figsize(style.WIDTH_FULL, style.SMALL_RATIO))
    fig.suptitle(f"{run.track.upper()}: parameter movement across models")
    handles = []
    for ax, measure in zip(axes.flat, measures):
        for (base, frozen), curve in data[data.measure.eq(measure)].groupby(["base", "frozen"]):
            ax.plot(curve.updates, curve["median"], color=style.color(base),
                    linestyle=style.FACTOR_ARM_STYLES[1-int(frozen)], linewidth=style.THIN_LINE)
        ax.set_title(measure.replace("relative ", ""))
        ax.set_xlabel("Successful updates"); ax.set_ylabel("Relative displacement")
        ax.set_ylim(bottom=0)
    for base in sorted(data.base.unique()):
        handles.append(Line2D([], [], color=style.color(base), label=style.model_label(base)))
    for frozen in sorted(data.frozen.unique()):
        handles.append(Line2D([], [], color=style.color("reference"),
            linestyle=style.FACTOR_ARM_STYLES[1-int(frozen)],
            label="Backbone frozen" if frozen else "All parameters updated"))
    fig.legend(handles=handles, loc="outside lower center", ncol=style.LEGEND_COLUMNS)
    return [cp.Page(f"movement_summary_{run.track}", fig,
        seed_caption(run) + "Parameter displacement at recorded milestones, shown together by model and update scope. Whole-model "
        "displacement is norm weighted; the tensor panel first takes the median across anchored tensors within "
        "a trial. Lines then take the median over the planned recipes and partitions within a model and scope. "
        "A milestone with an incomplete trial cohort remains a gap. Line color identifies the model and line "
        "style the update scope. The numerical values use different parameter sets across architectures and "
        "are descriptive movement diagnostics, not predictive-quality or causal comparisons.",
        {"Displayed displacement and complete-trial support": data})]


def resource_tables(run):
    """One completed trial contributes once, independent of sampling duration."""
    trials = run.trials[run.trials.status.eq("OK")].copy()
    fields = [f for f in ("gpu_hours", "peak_gpu_gb") if f in trials]
    if not fields:
        return pd.DataFrame(), trials
    rows = []
    for field in fields:
        trials[field] = pd.to_numeric(trials[field], errors="coerce")
        for (base, frozen), values in trials.groupby(["base", "frozen"]):
            finite = values.loc[np.isfinite(values[field]), field]
            if finite.empty:
                continue
            rows.append(dict(base=base, frozen=frozen, measure=field, median=finite.median(),
                q25=finite.quantile(.25), q75=finite.quantile(.75), trials=len(finite),
                expected_trials=int(((run.trials.base == base) & (run.trials.frozen == frozen)).sum())))
    provenance = [column for column in ("seed",) if column in trials]
    return pd.DataFrame(rows), trials[["trial_name", "base", "frozen", *provenance, *fields]]


def plot_resources(run):
    data, trials = resource_tables(run)
    if data.empty:
        return []
    fields = [f for f in ("gpu_hours", "peak_gpu_gb") if f in set(data.measure)]
    labels = {"gpu_hours": "Recorded training time (GPU hours)",
              "peak_gpu_gb": "Peak allocated tensor memory (GB)"}
    fig, axes = plt.subplots(1, len(fields), squeeze=False, layout="constrained",
                             figsize=style.figsize(style.WIDTH_FULL, style.SMALL_RATIO))
    fig.suptitle(f"{run.track.upper()}: training cost across models")
    bases = sorted(data.base.unique())
    for ax, field in zip(axes.flat, fields):
        for row, base in enumerate(bases):
            for frozen, group in data[(data.base.eq(base)) & data.measure.eq(field)].groupby("frozen"):
                point = group.iloc[0]
                offset = style.PROTOCOL_POINT_OFFSET * (-1 if frozen else 1) if data.frozen.nunique() > 1 else 0
                ax.errorbar(point["median"], row+offset,
                    xerr=[[point["median"]-point.q25], [point.q75-point["median"]]],
                    marker=style.PANEL_MARKERS[int(frozen)], color=style.color(base),
                    markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE,
                    markerfacecolor="none" if frozen else style.color(base))
        ax.set_yticks(range(len(bases)), [style.model_label(b) for b in bases])
        ax.set_xlabel(labels[field]); ax.set_xlim(left=0)
    handles = [Line2D([], [], marker=style.PANEL_MARKERS[int(frozen)], linestyle="",
        markerfacecolor="none" if frozen else style.color("reference"), color=style.color("reference"),
        label="Backbone frozen" if frozen else "All parameters updated") for frozen in sorted(data.frozen.unique())]
    if len(handles) > 1:
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    return [cp.Page(f"training_cost_summary_{run.track}", fig,
        seed_caption(run) + "Completed-trial training cost by model and update scope. Points show medians and horizontal intervals "
        "show interquartile ranges across observed recipes and training partitions, not confidence intervals. "
        "Each trial contributes once. Counts and all trial measurements accompany the plot. Time is recorded "
        "training wall time, excluding queue wait; memory is the process's recorded peak allocated tensor memory, "
        "rather than total device capacity or device-wide use. Recipe pooling is descriptive and does not "
        "establish an equal-compute predictive comparison. Missing or unfinished measurements are not zeros.",
        {"Displayed cost medians, interquartile ranges and trial support": data,
         "Completed-trial source measurements": trials})]


def plot_adapter_costs(runs):
    """Put both tasks' small DiagScale cost comparisons in one publication panel."""
    summaries, sources = [], []
    for run in runs:
        if str(run.cfg.train.adaptation_mode) != "diagscale":
            raise ValueError("The joint adapter-cost figure requires DiagScale campaigns")
        summary, source = resource_tables(run)
        if summary.empty:
            continue
        summaries.append(summary.drop(columns="frozen").assign(track=run.track))
        sources.append(source.drop(columns="frozen").assign(
            track=run.track, parameter_update_scope="DiagScale query-scale deltas only"))
    if not summaries:
        return []
    data = pd.concat(summaries, ignore_index=True)
    trials = pd.concat(sources, ignore_index=True)
    identities = [(track, base) for track in ("pd", "lgd")
                  for base in sorted(data.loc[data.track.eq(track), "base"].unique())]
    fields = [f for f in ("gpu_hours", "peak_gpu_gb") if f in set(data.measure)]
    labels = {"gpu_hours": "Recorded training time (GPU hours)",
              "peak_gpu_gb": "Peak allocated tensor memory (GB)"}
    fig, axes = plt.subplots(1, len(fields), squeeze=False, layout="constrained",
                             figsize=style.figsize(style.WIDTH_FULL, style.SMALL_RATIO))
    fig.suptitle("PD and LGD: DiagScale training cost")
    for ax, field in zip(axes.flat, fields):
        for row, (track, base) in enumerate(identities):
            selected = data[data.track.eq(track) & data.base.eq(base) & data.measure.eq(field)]
            if selected.empty:
                continue
            point = selected.iloc[0]
            ax.errorbar(point["median"], row,
                xerr=[[point["median"]-point.q25], [point.q75-point["median"]]],
                marker=style.PANEL_MARKERS[0], color=style.color(base),
                markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE)
        ax.set_yticks(range(len(identities)),
                     [f"{track.upper()} / {style.model_label(base)}" for track, base in identities])
        ax.invert_yaxis()
        ax.set_xlabel(labels[field]); ax.set_xlim(left=0)
    seeds = " ".join(f"{run.track.upper()}: {seed_caption(run).strip()}" for run in runs)
    return [cp.Page("diagscale_training_cost_pd_lgd", fig,
        "PD and LGD DiagScale training cost. Only query-scale deltas are optimized; original backbone "
        "weights remain unchanged. " + seeds + " Points show medians and horizontal intervals show "
        "interquartile ranges across adapter learning rates and training partitions within each task and model, "
        "not confidence intervals. Each completed trial contributes once; counts and all trial measurements "
        "are retained in the accompanying tables. Time is recorded training wall time, excluding queue wait. "
        "Memory is recorded peak allocated tensor memory, not total device capacity. No PD/LGD losses or "
        "predictive scores are pooled, and this cost summary is not an equal-compute comparison with full-parameter adaptation.",
        {"Displayed DiagScale cost medians, interquartile ranges and trial support": data,
         "Completed DiagScale trial source measurements": trials},
        {"measurement": "training_cost", "tracks": list(dict.fromkeys(data.track)),
         "adaptation_mode": "diagscale", "parameter_update_scope": "query_scale_deltas_only",
         "aggregation_unit": "completed_trial"})]
