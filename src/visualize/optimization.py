"""Paired factor contrasts and compact parameter diagnostics during training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.visualize import campaign as cp, diagnostics as dg, style


def factor_curves(campaign, field, factor):
    """Pair identical partitions and other knobs before aggregating a factor effect.

    Require the full planned matched set at each window; a faster arm must never
    acquire a different weight merely because more of its jobs have finished.
    """
    reference = {"l2sp_lambda": 0., "learning_rate": 3e-7}[factor]
    data = cp.optimization_curves(campaign, field)
    if data.empty:
        return pd.DataFrame()
    keys = [c for c in cp.FACTORS if c != factor] + ["partition", "seed"]
    plan = campaign.trials
    data = data.merge(plan[["trial_name", "partition", "seed"]], on="trial_name", validate="many_to_one")
    outputs = []
    for alternative in sorted(set(plan[factor]) - {reference}):
        wanted = plan[plan[factor].eq(reference)][keys].merge(
            plan[plan[factor].eq(alternative)][keys], on=keys, validate="one_to_one")
        pair = data[data[factor].eq(reference)][keys+["window_end", field]].merge(
            data[data[factor].eq(alternative)][keys+["window_end", field]],
            on=keys+["window_end"], suffixes=("_reference", "_alternative"), validate="one_to_one")
        pair["difference"] = pair[field+"_alternative"] - pair[field+"_reference"]
        expected = wanted.groupby(["base", "frozen"]).size()
        for (base, frozen), count in expected.items():
            selected = pair[pair.base.eq(base) & pair.frozen.eq(frozen)]
            curve = selected.groupby("window_end").difference.agg(
                mean="mean", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), pairs="size")
            windows = cp.optimization_grid(campaign, field)
            curve = curve.reindex(windows)
            curve["pairs"] = curve.pairs.fillna(0).astype(int)
            curve["expected_pairs"] = int(count)
            curve["complete"] = curve.pairs.eq(count)
            curve.loc[~curve.complete, ["mean", "q25", "q75"]] = np.nan
            outputs.append(curve.rename_axis("window_end").reset_index().assign(
                base=base, frozen=frozen, factor=factor, reference=reference, alternative=alternative, metric=field))
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def plot_factor_effects(campaign):
    pages = []
    for factor, title in (("l2sp_lambda", "Adding L2-SP"), ("learning_rate", "Changing peak learning rate")):
        frames = [factor_curves(campaign, field, factor) for field in ("train_loss", "weight_drift")]
        data = pd.concat(frames, ignore_index=True)
        if data.empty:
            continue
        for base, group in data.groupby("base"):
            if not np.isfinite(group["mean"]).any():
                continue
            modes = sorted(group.loc[np.isfinite(group["mean"]), "frozen"].unique())
            fields = [f for f in ("train_loss", "weight_drift") if np.isfinite(group.loc[group.metric.eq(f), "mean"]).any()]
            fig, axes = plt.subplots(len(modes), len(fields), squeeze=False, layout="constrained",
                                     figsize=style.figsize(style.WIDTH_FULL, style.PANEL_RATIO))
            fig.suptitle(f"{campaign.track.upper()} / {base}: {title}")
            for row, frozen in enumerate(modes):
                for ax, field in zip(axes[row], fields):
                    arm = group[group.frozen.eq(frozen) & group.metric.eq(field)]
                    if not np.isfinite(arm["mean"]).any():
                        ax.set_visible(False)
                        continue
                    for alternative, curve in arm.groupby("alternative"):
                        if not np.isfinite(curve["mean"]).any():
                            continue
                        color = style.TRAJECTORY_LR_COLORS[alternative] if factor == "learning_rate" else style.color("highlight")
                        label = f"LR {alternative:.0e} − 3e-7" if factor == "learning_rate" else "L2-SP 0.003 − 0"
                        ax.plot(curve.window_end, curve["mean"], label=label, color=color, linewidth=style.THIN_LINE)
                        ax.fill_between(curve.window_end, curve.q25, curve.q75, color=color, alpha=style.INTERVAL_ALPHA)
                    ax.axhline(0, color=style.color("reference"), linestyle=":")
                    label = cp.loss_label(campaign.track, base) if field == "train_loss" else "Relative weight drift"
                    ax.set_ylabel("Change in " + label)
                    ax.set_xlabel("Successful updates" if field == "weight_drift" else "Successful-update window end")
                    ax.set_title(style.adaptation_label(frozen))
                    ax.legend()
            pages.append(cp.Page(f"factor_dynamics_{campaign.track}_{base}_{factor}", fig,
                "Paired changes in optimization diagnostics when one training factor changes. Each pair holds the base, "
                "dataset partition, training seed and remaining recipe settings fixed. Differences are alternative minus reference; "
                "a lower data loss or smaller drift is not itself evidence of better held-out performance. Each matched recipe–partition "
                "pair has equal weight. Lines show means and bands show interquartile spread of differences, not confidence intervals. "
                "A window is drawn only when every planned pair is present. Loss windows require full observed table coverage; "
                "epoch timing within a window can differ between recipes. Drift uses its exact recorded milestones. No architectures or tasks are pooled.",
                {"Matched factor differences and coverage": group.reset_index(drop=True)}))
    return pages


def plot_movement(campaign):
    whole = cp.optimization_curves(campaign, "weight_drift")
    tensors = dg.load(campaign, "parameters")
    frames = []
    if not whole.empty:
        frames.append(whole.rename(columns={"weight_drift": "value", "window_end": "updates"}).assign(metric="Whole-model relative drift"))
    if not tensors.empty and "relative_change" in tensors:
        tensor = tensors.groupby(["trial_name", "base", "frozen", "successful_updates"]).relative_change.median().reset_index()
        frames.append(tensor.rename(columns={"relative_change": "value", "successful_updates": "updates"}).assign(metric="Median relative tensor change"))
    if not frames:
        return []
    data = pd.concat(frames, ignore_index=True)
    pages = []
    for base, group in data.groupby("base"):
        metrics = group.metric.unique()
        fig, axes = cp._subplots(f"{campaign.track.upper()} / {base}: parameter movement", len(metrics))
        tables = []
        for ax, metric in zip(axes, metrics):
            for frozen, arm in group[group.metric.eq(metric)].groupby("frozen"):
                curve = arm.groupby("updates").value.agg(median="median", trials="count")
                ax.plot(curve.index, curve["median"], label=style.adaptation_label(frozen),
                        color=style.color("frozen" if frozen else "full"))
                tables.append(curve.reset_index().assign(metric=metric, frozen=frozen))
            ax.set_ylabel(metric); ax.set_xlabel("Successful updates"); ax.legend()
        pages.append(cp.Page(f"movement_{campaign.track}_{base}", fig,
            "Complementary summaries of parameter movement. Whole-model drift is norm weighted; tensor movement first takes "
            "the median across anchored tensors within a trial. Curves then take medians across available recipes and partitions "
            "within each adaptation arm; counts are reported alongside values. Both quantities use their recorded "
            "milestones. Changing trial support can affect these descriptive curves.",
            {"Parameter movement and trial support": pd.concat(tables, ignore_index=True)}))
    return pages
