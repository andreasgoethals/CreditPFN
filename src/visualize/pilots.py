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
    metric = "change in AUC" if campaign.metric == "roc_auc" else "fractional RMSE reduction"
    fig = cp._heatmap(matrix, f"{campaign.track.upper()}: {metric} after {campaign.target:,} updates",
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
    seen = cp.effects(campaign, "train")
    gaps = []
    if not seen.empty:
        held = {base: curve for label, curves in domains if label == "Held-out credit" for base, curve in curves}
        for base, group in seen.groupby("base"):
            if base not in held:
                continue
            curve = cp.trajectory_curve(campaign, group, split="train", x=x)
            pair = curve.merge(held[base][["updates", "mean"]], on="updates", suffixes=("_train", "_held"), validate="one_to_one")
            pair["mean"] = pair.mean_train - pair.mean_held
            gaps.append((base, pair))
            tables.append(pair.assign(base=base, domain="Training minus held-out credit change"))
        if gaps:
            domains.append(("Train − held-out change", gaps))
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
        "Credit and non-credit panels use distinct dataset sets. The third panel subtracts mean held-out-credit improvement "
        "from mean training-credit improvement; positive values indicate greater adaptation on seen tables. This is a between-table "
        "transfer gap, not an unbiased overfitting estimate, since the training monitor reuses adaptation rows and table sets differ. "
        "Row exposures count repeated rows. Historical TabPFN objective "
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
                    markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE,
                    markevery=(int(frozen)*2, style.CURVE_MARKER_INTERVAL),
                    label=f"LR {lr:.0e} / {'frozen' if frozen else 'full'}")
            tables.append(curve.rename_axis("window_end").reset_index().assign(base=base, learning_rate=lr, frozen=frozen))
        ax.set_title(base)
        ax.set_xlabel("Successful-update window end")
        ax.set_ylabel(cp.loss_label(campaign.track, base))
        ax.legend()
    for ax in list(axes.flat)[len(bases):]:
        ax.remove()
    return [cp.Page(f"pilot_loss_{campaign.track}", fig,
        f"{campaign.track.upper()} data loss in recorded epochs, averaged within trial and update window before averaging trials. "
        f"Window width is {width:,} updates, no finer than the typical epoch recording interval. "
        "Markers show observed windows; missing windows remain gaps. L2-SP is excluded. PD uses cross-entropy; "
        "LGD uses TabPFN bar-distribution NLL or TabICLv2 quantile pinball loss. Separate vertical scales "
        "preserve each model's loss evolution; their numerical magnitudes are not comparable. "
        "Epochs with incomplete table coverage are excluded from the curves and retained in the accompanying tables. "
        "Markers are thinned for readability; all recorded window means remain in the line and text summary.",
        {"Window means and counts": pd.concat(tables, ignore_index=True),
         **cp.loss_coverage_tables(campaign)})]


def plot_schedule(campaigns):
    """Show identical configured schedules once; retain measured epoch values separately."""
    import torch
    from omegaconf import OmegaConf
    from src.train.loop import make_warmup_cosine_schedule

    schedules = {}
    measured = []
    for run in campaigns:
        observed = cp.optimization_curves(run, "lr_applied")
        if not observed.empty:
            measured.append(observed.assign(track=run.track.upper()))
        warmup = float(OmegaConf.select(run.cfg, "scheduler.warmup_fraction", default=.1))
        floor = float(OmegaConf.select(run.cfg, "scheduler.min_lr_fraction", default=.05))
        for lr in sorted(run.trials.learning_rate.unique()):
            schedules.setdefault((run.target, warmup, floor, float(lr)), []).append(run.track.upper())
    if not schedules:
        return []
    tracks = [t for t in ("PD", "LGD") if any(r.track.upper() == t for r in campaigns)]
    fig, (ax,) = cp._subplots("Applied learning-rate schedule: " + " + ".join(tracks))
    tables = []
    for (target, warmup, floor, lr), owners in schedules.items():
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=lr)
        schedule = make_warmup_cosine_schedule(optimizer, total_steps=target,
            warmup_fraction=warmup, min_lr_fraction=floor, schedule_type="warmup_cosine")
        updates = np.unique(np.r_[np.linspace(1, target, min(target, style.SCHEDULE_POINTS)).astype(int),
                                   min(2, target), min(target, max(1, round(target * warmup)) + 1)])
        rates = np.array([lr * schedule.lr_lambdas[0](int(u)-1) for u in updates])
        owners = [t for t in tracks if t in owners]
        table = pd.DataFrame(dict(successful_update=updates, applied_lr=rates,
            peak_lr=lr, tracks=" + ".join(owners), target=target,
            warmup_fraction=warmup, min_lr_fraction=floor))
        tables.append(table)
        label = f"Peak LR {lr:.0e}"
        if owners != tracks:
            label += " / " + " + ".join(owners)
        ax.plot(updates, np.where(rates > 0, rates, np.nan), label=label,
                color=style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr))))
    ax.set_yscale("log")
    ax.set_xlabel("Successful optimizer update")
    ax.set_ylabel("Applied learning rate (log scale)")
    ax.legend()
    return [cp.Page("shared_schedule_" + "_".join(t.lower() for t in tracks), fig,
        "Learning rates evaluated from the recorded campaign configuration using the training scheduler. "
        "Each distinct peak-rate schedule is drawn once, pooling tasks only when budget, warmup and decay floor agree. "
        "The rate for update u is the scheduler value before that update (index u minus one). "
        "The first update has zero learning rate and is omitted from the logarithmic axis, but retained in the table. "
        "These are configured schedules; measured epoch-end rates and their sampling intervals are reported separately.",
        {"Configured applied schedules": pd.concat(tables, ignore_index=True),
         "Observed trial windows": pd.concat(measured, ignore_index=True) if measured else pd.DataFrame()})]


def load_schedule_plan(experiment, track):
    """Read the companion task's schedule without loading its training histories."""
    cfg = cp.load_train_config(config_path=str(cp.config_path(experiment, track)))
    return cp.Campaign(cfg, cp.planned_trials(cfg), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def plot_process_overview(campaign):
    """Compact dynamics for fixed-recipe seed and sampling experiments."""
    if len(campaign.trials[["learning_rate", "l2sp_lambda", "frozen"]].drop_duplicates()) != 1:
        raise ValueError("Process overview requires one LR, L2-SP and adaptation recipe")
    fields = {"train_loss": "Data loss", "grad_norm_mean": "Gradient norm",
              "clipped_frac": "Clipped fraction", "weight_drift": "Relative weight drift"}
    curves = {field: cp.optimization_curves(campaign, field) for field in fields}
    curves = {field: data for field, data in curves.items() if not data.empty}
    if not curves:
        return []
    pages = []
    for base in sorted(campaign.trials.base.unique()):
        if not any(data.base.eq(base).any() for data in curves.values()):
            continue
        fig, axes = plt.subplots(2, 2, layout="constrained", figsize=style.figsize(style.WIDTH_FULL, style.PANEL_RATIO))
        fig.suptitle(f"{campaign.track.upper()} / {base}: optimization dynamics")
        tables = []
        for ax, (field, data) in zip(axes.flat, curves.items()):
            for mode, group in data[data.base.eq(base)].groupby("sampling"):
                curve = group.groupby("window_end")[field].agg(median="median", trials="size")
                curve = curve.reindex(cp.optimization_grid(campaign, field))
                ax.plot(curve.index, curve["median"], label=style.SAMPLING_LABELS[mode], color=style.SAMPLING_COLORS[mode],
                        linestyle=style.SAMPLING_STYLES[mode], linewidth=style.THIN_LINE)
                tables.append(curve.rename_axis("window_end").reset_index().assign(metric=field, sampling=mode))
            ax.set_xlabel("Successful-update window end")
            ax.set_ylabel(cp.loss_label(campaign.track, base) if field == "train_loss" else fields[field])
            if field == "clipped_frac":
                from matplotlib.ticker import PercentFormatter
                maximum = pd.to_numeric(data[field], errors="coerce").max()
                if maximum > 0:
                    ax.set_yscale("symlog", linthresh=style.CLIP_LINEAR_THRESHOLD)
                    ax.set_ylim(0, min(style.CLIP_DISPLAY_CEILING, maximum*1.1))
                    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
                else:
                    ax.text(.5, .5, "No clipped updates recorded", ha="center", va="center", transform=ax.transAxes)
                    ax.set_ylim(0,1)
            ax.legend()
        for ax in list(axes.flat)[len(curves):]:
            ax.remove()
        pages.append(cp.Page(f"process_{campaign.track}_{base}", fig,
            "Optimization diagnostics at the fixed reference recipe. Each curve is the median of trial-window "
            "means for one sampling protocol across available dataset partitions; counts are retained in the text. "
            "Incomplete table-coverage loss epochs are tabulated separately; missing windows remain gaps. "
            "Clipping uses a shared percentage range, linear through 0.1% and logarithmic above. Drift follows exact measured milestones. "
            "Loss excludes L2-SP and retains the native objective scale, which is not comparable across architectures.",
            {"Window medians and support": pd.concat(tables, ignore_index=True)}))
    return pages


def health_table(campaign):
    rows = []
    h = campaign.histories
    if h.empty:
        return pd.DataFrame()
    h = h[h.successful_updates.gt(0)]
    # Drift is a scheduled milestone measurement; its epoch NaNs mean "not sampled".
    fields = [f for f in ("train_loss", "l2sp_penalty", "grad_norm_mean", "clipped_frac") if f in h]
    drift = cp.optimization_curves(campaign, "weight_drift")
    for trial, group in h.groupby("trial_name"):
        record = dict(trial_name=trial, epochs=len(group))
        values = group[fields].apply(pd.to_numeric, errors="coerce")
        record["nonfinite_diagnostics"] = int((~np.isfinite(values)).sum().sum())
        for field, name, reducer in (("grad_norm_mean", "peak_epoch_gradient", "max"),
                                     ("clipped_frac", "peak_epoch_clipped_fraction", "max"),
                                     ("l2sp_penalty", "peak_l2sp_penalty", "max"),
                                     ("amp_skipped_steps", "amp_skips", "sum"),
                                     ("data_skipped_steps", "data_skips", "sum")):
            if field in group:
                record[name] = getattr(pd.to_numeric(group[field], errors="coerce"), reducer)()
        if not drift.empty:
            measured = drift.loc[drift.trial_name.eq(trial), "weight_drift"]
            record["peak_weight_drift"] = measured.max() if not measured.empty else np.nan
        rows.append(record)
    keys = ["trial_name", *cp.FACTORS, *[c for c in ("partition", "seed") if c in campaign.trials]]
    return campaign.trials[keys].merge(
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
