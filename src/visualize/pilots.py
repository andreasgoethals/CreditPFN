"""Compact, task-labelled validation reports for experiment 0."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.visualize import campaign as cp, style
from src.visualize.paper_figures import effect_label


def coverage_table(campaigns):
    frames = []
    for run in campaigns:
        frame = run.trials.groupby(["base", "status"]).size().unstack(fill_value=0)
        frames.append(frame.rename_axis("base").reset_index().assign(track=run.track.upper(), updates=run.target))
    return pd.concat(frames, ignore_index=True).fillna(0)


def endpoint_table(campaign):
    """Complete paired cohort means; retain support counts alongside effects."""
    rows = []
    for split, domain in (("test", "Credit"), ("ood", "Non-credit")):
        data = cp.effects(campaign, split)
        if data.empty:
            continue
        for factors, recipe in data.groupby(cp.FACTORS):
            curve = cp.trajectory_curve(campaign, recipe, split=split)
            point = curve[curve.updates.eq(campaign.target)].iloc[0]
            rows.append(dict(zip(cp.FACTORS, factors), domain=domain, effect=point["mean"],
                             datasets=point.datasets, complete=point.complete))
    return pd.DataFrame(rows)


def plot_endpoints(campaign):
    data = endpoint_table(campaign)
    if data.empty or not np.isfinite(data.effect).any():
        return []
    data["recipe"] = [f"{base} / LR {lr:.0e}" for base, lr in zip(data.base, data.learning_rate)]
    data["arm"] = np.where(data.frozen, "Frozen / ", "Full / ") + data.domain
    matrix = data.pivot(index="recipe", columns="arm", values="effect")
    matrix = matrix.reindex(columns=[c for c in ("Full / Credit", "Frozen / Credit", "Full / Non-credit", "Frozen / Non-credit") if c in matrix])
    fig = cp._heatmap(matrix, f"{campaign.track.upper()}: changes after {campaign.target:,} updates",
                      label=effect_label(campaign.metric))
    return [cp.Page(f"pilot_endpoints_{campaign.track}", fig,
        f"{campaign.track.upper()} changes from each dataset's unmodified baseline after {campaign.target:,} successful updates. "
        "Cells are equally weighted dataset means, conditional on complete planned trial and dataset support. "
        "Positive values indicate improvement. Credit and non-credit columns use different dataset panels. "
        "Only baseline and endpoint measurements are available; no intermediate learning curve is inferred.",
        {"Endpoint effects and support": data, "Displayed effects": matrix})]


def plot_horizon(campaign, *, x="updates"):
    domains = []
    tables = []
    for split, label in (("test", "Held-out credit"), ("ood", "Non-credit retention")):
        data = cp.effects(campaign, split)
        curves = []
        if not data.empty:
            for base, group in data.groupby("base"):
                if len(group[cp.FACTORS].drop_duplicates()) != 1:
                    raise ValueError("A budget horizon requires one recipe per base")
                curve = cp.trajectory_curve(campaign, group, split=split, x=x).assign(base=base, domain=label)
                tables.append(curve)
                if np.isfinite(curve["mean"]).any():
                    curves.append((base, curve))
        if curves:
            domains.append((label, curves))
    if not domains:
        return []
    fig, axes = cp._subplots(f"{campaign.track.upper()}: budget-pilot learning horizon", len(domains), sharey=True)
    for ax, (label, curves) in zip(axes, domains):
        for i, (base, curve) in enumerate(curves):
            ax.plot(curve.x, curve["mean"], marker=style.PANEL_MARKERS[i % 4],
                    color=style.color(base), label=base, markersize=style.CURVE_MARKER_SIZE)
        ax.axhline(0, color=style.color("reference"), linestyle=":")
        ax.set_title(label)
        ax.set_xlabel("Successful updates" if x == "updates" else "Processed row exposures")
        ax.legend()
    axes[0].set_ylabel(effect_label(campaign.metric))
    return [cp.Page(f"budget_horizon_{campaign.track}_{x}", fig,
        f"{campaign.track.upper()} budget-pilot changes relative to the same trial and dataset at update zero. "
        "Each curve is one base model at the fixed pilot recipe; points show equally weighted dataset means "
        "and lines connect recorded milestones without smoothing. Missing scheduled measurements remain gaps. "
        "The two panels use distinct dataset sets. Row exposures count repeated rows. Historical TabPFN objective "
        "differences limit comparison with the corrected main experiment; these curves do not establish an optimal budget.",
        {"Milestone means, quartiles and support": pd.concat(tables, ignore_index=True)})]


def plot_loss(campaign):
    data = cp.optimization_curves(campaign, "train_loss")
    if data.empty:
        return []
    bases = sorted(data.base.unique())
    columns = min(2, len(bases))
    rows = int(np.ceil(len(bases) / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False, layout="constrained",
                             figsize=style.figsize(style.WIDTH_FULL, style.PANEL_RATIO))
    fig.suptitle(f"{campaign.track.upper()}: recorded training loss")
    tables = []
    width = cp.optimization_window(campaign)
    windows = np.unique(np.minimum(np.arange(width, campaign.target+width, width), campaign.target))
    for ax, base in zip(axes.flat, bases):
        group = data[data.base.eq(base)]
        for (lr, frozen), recipe in group.groupby(["learning_rate", "frozen"]):
            curve = recipe.groupby("window_end").train_loss.agg(mean="mean", trials="size").reindex(windows)
            ax.plot(curve.index, curve["mean"], color=style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr))),
                    linestyle="--" if frozen else "-", marker=style.SEED_MARKERS[int(frozen)],
                    markersize=style.CURVE_MARKER_SIZE,
                    label=f"LR {lr:.0e} / {'frozen' if frozen else 'full'}")
            tables.append(curve.rename_axis("window_end").reset_index().assign(base=base, learning_rate=lr, frozen=frozen))
        ax.set_title(base)
        ax.set_xlabel("Successful-update window end")
        ax.set_ylabel("Data loss")
        ax.legend()
    for ax in list(axes.flat)[len(bases):]:
        ax.remove()
    return [cp.Page(f"pilot_loss_{campaign.track}", fig,
        f"{campaign.track.upper()} data loss in recorded epochs, averaged within trial and update window before averaging trials. "
        f"Window width is {width:,} updates, no finer than the typical epoch recording interval. "
        "Markers show observed windows; missing windows remain gaps. L2-SP is excluded. PD uses cross-entropy; "
        "LGD uses TabPFN bar-distribution NLL or TabICLv2 quantile pinball loss. Separate vertical scales "
        "preserve each model's loss evolution; their numerical magnitudes are not comparable.",
        {"Window means and counts": pd.concat(tables, ignore_index=True)})]


def plot_schedule(campaigns):
    """Pool measured schedules instead of repeating the same recipe for each base."""
    tables = []
    curves = []
    for run in campaigns:
        data = cp.optimization_curves(run, "lr_applied")
        if data.empty:
            continue
        tables.append(data.assign(track=run.track.upper()))
        for lr, recipe in data.groupby("learning_rate"):
            curve = recipe.groupby("window_end").lr_applied.agg(
                median="median", q25=lambda v: v.quantile(.25), q75=lambda v: v.quantile(.75), trials="size")
            curves.append((run.track, lr, curve))
    if not curves:
        return []
    tracks = list(dict.fromkeys(track for track, _, _ in curves))
    # Merge tracks only when their recorded window statistics are identical.
    signatures = {track: [(lr, curve.to_json()) for tr, lr, curve in curves if tr == track] for track in tracks}
    same = len(tracks) > 1 and len({str(v) for v in signatures.values()}) == 1
    panels = [tracks[0]] if same else tracks
    fig, axes = cp._subplots("PD and LGD: applied learning rate" if len(tracks)>1 else f"{tracks[0].upper()}: applied learning rate", len(panels))
    stats = []
    for ax, track in zip(axes, panels):
        for tr, lr, curve in curves:
            if tr != track:
                continue
            color = style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr)))
            ax.plot(curve.index, curve["median"], color=color, label=f"Peak LR {lr:.0e}")
            ax.fill_between(curve.index.to_numpy(float), curve.q25.to_numpy(float), curve.q75.to_numpy(float), color=color, alpha=style.INTERVAL_ALPHA)
            stats.append(curve.rename_axis("window_end").reset_index().assign(track="PD + LGD" if same else track.upper(), peak_lr=lr))
        ax.set_title("PD + LGD" if same else track.upper())
        ax.set_xlabel("Successful-update window end"); ax.set_ylabel("Applied learning rate")
        ax.legend()
    return [cp.Page("shared_schedule_" + "_".join(tracks), fig,
        "Observed learning rates grouped by peak rate and task, pooling base models and adaptation modes. "
        "Lines and shading give trial-window medians and interquartile ranges; unequal epoch sampling can create "
        "small differences despite identical configured schedules. Identical task summaries share a panel. "
        "Full window observations and counts are retained in the text summary.",
        {"Schedule statistics": pd.concat(stats, ignore_index=True), "Observed trial windows": pd.concat(tables, ignore_index=True)})]


def health_table(campaign):
    rows = []
    h = campaign.histories
    if h.empty:
        return pd.DataFrame()
    h = h[h.successful_updates.gt(0)]
    fields = [f for f in ("train_loss", "l2sp_penalty", "grad_norm_mean", "clipped_frac", "weight_drift") if f in h]
    for trial, group in h.groupby("trial_name"):
        record = dict(trial_name=trial, epochs=len(group))
        values = group[fields].apply(pd.to_numeric, errors="coerce")
        record["nonfinite_diagnostics"] = int((~np.isfinite(values)).sum().sum())
        for field, name, reducer in (("grad_norm_mean", "peak_epoch_gradient", "max"),
                                     ("clipped_frac", "peak_epoch_clipped_fraction", "max"),
                                     ("l2sp_penalty", "peak_l2sp_penalty", "max"),
                                     ("weight_drift", "peak_weight_drift", "max"),
                                     ("amp_skipped_steps", "amp_skips", "sum"),
                                     ("data_skipped_steps", "data_skips", "sum")):
            if field in group:
                record[name] = getattr(pd.to_numeric(group[field], errors="coerce"), reducer)()
        rows.append(record)
    return campaign.trials[["trial_name", "base", "learning_rate", "frozen"]].merge(
        pd.DataFrame(rows), on="trial_name", validate="one_to_one").drop(columns="trial_name")


def trial_summary(campaign):
    columns = ["base", "learning_rate", "frozen", "status", "trainable_params", "total_params",
               "final_drift", "elapsed_sec", "gpu_hours", "peak_gpu_gb", "rows_seen"]
    return campaign.trials[[c for c in columns if c in campaign.trials]].copy()


def resource_summary(campaign):
    from src.visualize import diagnostics as dg
    data = dg.load(campaign, "resources")
    if data.empty:
        return pd.DataFrame()
    fields = [f for f in ("gpu_utilization_percent", "power_watts", "device_memory_used_mib") if f in data]
    data = data[data.phase.eq("training")].copy()
    data[fields] = data[fields].apply(pd.to_numeric, errors="coerce")
    medians = data.groupby(["trial_name", "base", "frozen"])[fields].median()
    summary = medians.groupby(["base", "frozen"])[fields].median().rename(columns={f: f+"_median" for f in fields})
    summary["trials"] = medians.groupby(["base", "frozen"]).size()
    return summary.reset_index()


def parameter_summary(campaign):
    from src.visualize import diagnostics as dg
    data = dg.load(campaign, "parameters")
    if data.empty:
        return pd.DataFrame()
    endpoint = data[data.successful_updates.eq(campaign.target)]
    medians = endpoint.groupby(["trial_name", "base", "frozen"]).relative_change.median()
    return medians.groupby(["base", "frozen"]).agg(trials="count", median_tensor_change="median",
                                                    min_trial_median="min", max_trial_median="max").reset_index()
