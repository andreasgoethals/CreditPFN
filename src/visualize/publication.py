"""Paper-oriented views of coverage, paired effects, retention and resource use.

Each Page carries the complete numerical table used to draw it. Repeated folds
are aggregated within datasets; shaded/interval spreads are descriptive, not
confidence intervals over independent training runs.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.visualize import campaign as cp, style
from src.visualize.inputs import analysis_location
from src.visualize.paper_figures import effect_label
from src.utils.paths import manifests_dir


def availability(campaign):
    """Describe the evidence actually present, independently of training completion."""
    return pd.DataFrame([
        ("Planned trials", len(campaign.trials), "Config; not scheduler state"),
        ("Completed trials", int(campaign.trials.status.eq("OK").sum()), "DATA manifests"),
        ("Epoch rows", len(campaign.histories), "Project training diagnostics"),
        ("Milestone rows", len(campaign.trajectories), "Project trajectory diagnostics"),
        ("Benchmark folds", len(campaign.evaluation), "Project final evaluation"),
    ], columns=["evidence", "records", "origin"])


def cohort_inventory():
    """Keep generated debugging cohorts and their audit receipts distinct."""
    root = analysis_location("experiment0", "manifests", manifests_dir("experiment0"))
    rows = []
    for path in sorted((root / "workflow").glob("*_passed.json")):
        record = json.loads(path.read_text(encoding="utf8"))
        rows.append(dict(phase=record.get("part"), workflow=record.get("id"),
                         status=record.get("status"), successful_callbacks=len(record.get("done", [])),
                         failed_callbacks=len(record.get("failed", [])),
                         fingerprint=record.get("fingerprint", "")))
    return pd.DataFrame(rows)


def pilot_campaigns(track):
    """Use a receipt's exact generated config; never pool old and corrected objectives."""
    root = analysis_location("experiment0", "manifests", manifests_dir("experiment0"))
    receipt = root / "workflow" / "pilot_passed.json"
    if receipt.is_file():
        record = json.loads(receipt.read_text(encoding="utf8"))
        config = root / "workflow" / record["id"] / f"pilot_{track}.yaml"
        if config.is_file():
            from omegaconf import OmegaConf
            return cp.load_campaign(0, track, "pilot", cfg_override=OmegaConf.create(config.read_text(encoding="utf8")))
        raise ValueError("Pilot receipt exists but its generated configuration is missing")
    return cp.load_campaign(0, track, "pilot")


def plot_manifest_monitors(campaign):
    """Available with DATA only; never relabel fold-level means as per-table evidence."""
    data = campaign.trials
    required = ["baseline_test_metric", "final_test_metric"]
    if not set(required) <= set(data):
        return []
    data = data[data.status.eq("OK")].copy()
    data[required] = data[required].apply(pd.to_numeric, errors="coerce")
    data = data[np.isfinite(data[required]).all(axis=1)]
    if not data.empty and data.learning_rate.eq(0).all():
        data["monitor_change"] = data.final_test_metric - data.baseline_test_metric
        matrix = data.pivot_table(index="base", columns="frozen", values="monitor_change", aggfunc="mean")
        matrix = matrix.rename(columns={False: "Full updates", True: "Frozen backbone"})
        fig = cp._heatmap(matrix, f"{campaign.track.upper()}: null-control endpoint changes",
                         label=f"Endpoint minus initial {campaign.metric}")
        return [cp.Page(f"null_manifest_{campaign.track}", fig,
            "Change in the recorded held-out partition-mean monitor after zero-learning-rate updates. "
            "Each cell fixes base and adaptation. This aggregate check complements the exact saved-state and "
            "per-table prediction audits; an unchanged mean alone cannot establish either.",
            {"Monitor changes": matrix, "Initial and endpoint scores": data[
                ["partition", *cp.FACTORS, *required, "monitor_change"]].reset_index(drop=True)})]
    pages = []
    for base, group in data.groupby("base"):
        fig, axes = cp._subplots(f"{base}: held-out monitor endpoints", 2, sharey=True)
        lower, upper = group[required].min().min(), group[required].max().max()
        padding = max((upper - lower) * .06, .001)
        for ax, frozen in zip(axes, (False, True)):
            arm = group[group.frozen.eq(frozen)]
            for i, (lr, part) in enumerate(arm.groupby("learning_rate")):
                ax.scatter(part.baseline_test_metric, part.final_test_metric,
                           color=style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr))),
                           marker=style.PANEL_MARKERS[i % len(style.PANEL_MARKERS)],
                           alpha=style.POINT_ALPHA, s=style.POINT_SIZE, label=f"LR {lr:.0e}")
            ax.plot([lower-padding, upper+padding], [lower-padding, upper+padding],
                    color=style.color("reference"), linestyle=":")
            ax.set_xlim(lower-padding, upper+padding); ax.set_ylim(lower-padding, upper+padding)
            ax.set_xlabel("Initial monitor score")
            ax.set_title("Frozen backbone" if frozen else "Full updates")
            if not arm.empty:
                ax.legend()
        axes[0].set_ylabel("Endpoint monitor score")
        columns = ["partition", *cp.FACTORS, "baseline_test_metric", "final_test_metric"]
        pages.append(cp.Page(f"manifest_monitor_{campaign.track}_{base}", fig,
            f"Recorded {campaign.metric} before and after {campaign.target:,} updates for completed trials. "
            "Each point is a recipe and dataset partition; both L2-SP arms are included. The diagonal denotes unchanged score. "
            "Scores are the writer's means across that partition's held-out tables, not final five-fold estimates. "
            "In particular, these LGD means do not provide dataset-normalized effects or evidence about non-credit retention.",
            {"Completed partition-level monitors": group[columns].reset_index(drop=True)}))
    return pages


def plot_recipe_distributions(data, metric, *, prefix="effects"):
    """Dataset variation by recipe, with common support and no independent-fold error bars."""
    if data.empty:
        return []
    pages = []
    for (base, sampling), group in data.groupby(["base", "sampling"]):
        per = group.groupby([*cp.FACTORS, "dataset"], dropna=False).effect.mean().reset_index()
        recipes = per[cp.FACTORS].drop_duplicates()
        support = per.groupby("dataset").size()
        common = support[support.eq(len(recipes))].index
        per = per[per.dataset.isin(common)]
        if per.empty:
            continue
        summaries = per.groupby(cp.FACTORS).effect.agg(
            datasets="size", mean="mean", median="median",
            q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
            minimum="min", maximum="max").reset_index()
        fig, axes = cp._subplots(f"{base}: variation across datasets", 2, sharey=True)
        for ax, frozen in zip(axes, (False, True)):
            arm = summaries[summaries.frozen.eq(frozen)].sort_values(["learning_rate", "l2sp_lambda"])
            for position, row in enumerate(arm.itertuples()):
                values = per[(per.frozen == frozen) & (per.learning_rate == row.learning_rate)
                             & (per.l2sp_lambda == row.l2sp_lambda)].sort_values("dataset")
                jitter = np.linspace(-style.JITTER_WIDTH, style.JITTER_WIDTH, len(values))
                ax.scatter(values.effect, position+jitter, s=style.POINT_SIZE/2,
                           alpha=style.POINT_ALPHA, color=style.color(base))
                ax.plot([row.q25, row.q75], [position, position], color=style.color("reference"))
                ax.scatter([row.median], [position], marker="D", s=style.POINT_SIZE,
                           color=style.color("reference"))
            ax.set_yticks(range(len(arm)), [cp._recipe_label(lr, lam) for lr, lam in zip(arm.learning_rate, arm.l2sp_lambda)])
            ax.axvline(0, color=style.color("reference"), linestyle=":")
            ax.set_title("Frozen backbone" if frozen else "Full updates")
            ax.set_xlabel(effect_label(metric))
        axes[0].set_ylabel("Peak LR / L2-SP")
        pages.append(cp.Page(f"{prefix}_distribution_{base}_{sampling}", fig,
            f"Paired effects on the {len(common)} datasets available for every displayed recipe of {base}. "
            "Points are dataset means; diamonds and horizontal segments show the median and interquartile range across datasets. "
            "These intervals describe heterogeneity, not sampling uncertainty. Positive values favor adaptation. "
            "Unavailable recipes and datasets remain reported in coverage; no recipe is selected using these results.",
            {"Dataset effects": per.reset_index(drop=True), "Descriptive intervals": summaries}))
    return pages


def plot_retention_tradeoff(campaign):
    credit, retention = cp.endpoint_effects(campaign), cp.endpoint_effects(campaign, "ood")
    if credit.empty or retention.empty:
        return []
    keys = ["trial", *cp.FACTORS]
    left = credit.groupby(keys).effect.agg(credit_effect="mean", credit_tables="size").reset_index()
    right = retention.groupby(keys).effect.agg(retention_effect="mean", retention_tables="size").reset_index()
    paired = left.merge(right, on=keys, validate="one_to_one")
    pages = []
    for base, group in paired.groupby("base"):
        fig, axes = cp._subplots(f"{base}: credit transfer and non-credit retention", 2, sharey=True)
        for ax, frozen in zip(axes, (False, True)):
            for i, (lr, arm) in enumerate(group[group.frozen.eq(frozen)].groupby("learning_rate")):
                ax.scatter(arm.credit_effect, arm.retention_effect, label=f"LR {lr:.0e}",
                           color=style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr))),
                           marker=style.PANEL_MARKERS[i % 4], s=style.POINT_SIZE, alpha=style.POINT_ALPHA)
            ax.axhline(0, color=style.color("reference"), linestyle=":")
            ax.axvline(0, color=style.color("reference"), linestyle=":")
            ax.set_title("Frozen backbone" if frozen else "Full updates")
            ax.set_xlabel("Credit: "+effect_label(campaign.metric)); ax.legend()
        axes[0].set_ylabel("Non-credit: "+effect_label(campaign.metric))
        pages.append(cp.Page(f"retention_tradeoff_{campaign.track}_{base}", fig,
            "Endpoint credit and non-credit monitoring effects matched on the same trained trial. "
            "Each domain first pairs a dataset with its own update-zero result, then gives its datasets equal weight. "
            "Points in the lower-right quadrant combine credit gains with a decline on the fixed non-credit panel. "
            "The panel is finite and base-model pretraining exposure is unverified; it cannot establish universal retention.",
            {"Matched domains": group.reset_index(drop=True)}))
    return pages


def plot_trainable_fraction(campaign):
    data = campaign.trials
    if not {"trainable_params", "total_params"} <= set(data):
        return []
    data = data.dropna(subset=["trainable_params", "total_params"]).copy()
    data = data[data.total_params.gt(0)]
    if data.empty:
        return []
    values = data.groupby(["base", "frozen"])[["trainable_params", "total_params"]].median().reset_index()
    values["percent_trainable"] = 100 * values.trainable_params / values.total_params
    fig, axes = cp._subplots(f"{campaign.track.upper()}: trainable parameter fraction")
    labels = values.base + values.frozen.map({False: " / full", True: " / frozen"})
    axes[0].barh(labels, values.percent_trainable, color=[style.color(b) for b in values.base])
    axes[0].set_xlim(0, 100); axes[0].set_xlabel("Trainable parameters (%)")
    return [cp.Page(f"trainable_fraction_{campaign.track}", fig,
        "Trainable parameters divided by total model parameters for each base and adaptation mode. "
        "Counts are taken from observed trial provenance; median counts are used where recipes repeat a model. "
        "The frozen condition leaves family-specific surrounding modules trainable; it is not the same parameter subset across architectures.",
        {"Parameter counts": values})]


def plot_drift_effect(campaign):
    effect = cp.endpoint_effects(campaign)
    if effect.empty or "final_drift" not in campaign.trials:
        return []
    mean = effect.groupby(["trial", *cp.FACTORS]).effect.mean().reset_index()
    data = mean.merge(campaign.trials[["trial_name", "final_drift"]],
                      left_on="trial", right_on="trial_name", validate="one_to_one").dropna(subset=["final_drift"])
    pages = []
    for base, group in data.groupby("base"):
        fig, axes = cp._subplots(f"{base}: movement and held-out effect")
        for frozen, arm in group.groupby("frozen"):
            axes[0].scatter(arm.final_drift, arm.effect, label="Frozen" if frozen else "Full",
                           marker="s" if frozen else "o", color=style.color("frozen" if frozen else "full"),
                           s=style.POINT_SIZE, alpha=style.POINT_ALPHA)
        axes[0].axhline(0, color=style.color("reference"), linestyle=":")
        axes[0].set_xlabel("Relative parameter displacement")
        axes[0].set_ylabel(effect_label(campaign.metric)); axes[0].legend()
        pages.append(cp.Page(f"drift_effect_{campaign.track}_{base}", fig,
            "Endpoint held-out monitoring effect versus relative movement from initialization, matched by trial. "
            "Effects give each held-out dataset equal weight. Drift describes the anchored trainable subset, "
            "whose size differs by adaptation mode; association does not identify a causal effect of displacement.",
            {"Paired movement and monitoring": data[data.base.eq(base)].reset_index(drop=True)}))
    return pages


def plot_metric_profile(campaign):
    """Discrimination, probability quality and threshold behavior for one fixed recipe."""
    metrics = (("roc_auc", "pr_auc", "brier_score", "log_loss", "ece", "f1") if campaign.track == "pd"
               else ("rmse", "mae", "r2"))
    notes = {
        "f1": "F1 uses thresholds chosen on inner validation rows.",
        "brier_score": "Brier scores assess probabilistic accuracy, combining calibration and discrimination.",
        "log_loss": "Log loss assesses predicted probabilities and penalizes confidently incorrect predictions.",
        "ece": "ECE is a bin-dependent calibration diagnostic, not a complete measure of probabilistic accuracy.",
    }
    pages = []
    for metric in metrics:
        data = cp.reference_rows(cp.benchmark_effects(campaign, metric))
        if data.empty:
            continue
        matrix = data.pivot(index="dataset", columns="base", values="effect").sort_index()
        extent = max(float(np.nanmax(abs(matrix.to_numpy()))), 1e-8)
        for start in range(0, len(matrix), style.PAGE_ROWS):
            part = matrix.iloc[start:start+style.PAGE_ROWS]
            fig = cp._heatmap(part, f"Reference recipe: {metric}", limits=(-extent, extent), label=effect_label(metric))
            pages.append(cp.Page(f"reference_{metric}_{start//style.PAGE_ROWS+1}", fig,
                f"Dataset-level {metric} changes for the predefined full-update recipe, LR 3e-7 and L2-SP 0.003, "
                "relative to each model's untuned weights. Pairing occurs within outer folds and requires all five folds. "
                "Positive values indicate improvement; only RMSE is expressed as a fractional reduction. "
                + notes.get(metric, ""),
                {"Paired reference effects": part}))
    return pages


def plot_thresholds(campaign):
    if campaign.track != "pd" or campaign.evaluation.empty:
        return []
    data = campaign.evaluation.copy()
    column = next((c for c in ("threshold", "optimal_threshold", "threshold_f1") if c in data), None)
    if column is None:
        return []
    data = data[data.status.eq("OK") & data.source.str.endswith("-trained", na=False)]
    data = data[np.isclose(data.lr, 3e-7, rtol=1e-9, atol=0) & np.isclose(data.l2sp_lambda, .003)
                & ~data.use_lora.astype(bool) & data.epoch_pass_mode.eq("one_sample")]
    if "domain" in data:
        data = data[data.domain.eq("credit")]
    data = cp._complete_folds(data, "f1")
    if data.empty:
        return []
    pages = []
    for base, group in data.groupby("base_short"):
        matrix = group.pivot(index="test_dataset_id", columns="fold_idx", values=column).sort_index()
        for start in range(0,len(matrix),style.PAGE_ROWS):
            part = matrix.iloc[start:start+style.PAGE_ROWS]
            fig = cp._heatmap(part, f"{base}: validation-selected thresholds",
                              diverging=False, limits=(0,1), label="F1 threshold")
            pages.append(cp.Page(f"thresholds_{base}_{start//style.PAGE_ROWS+1}", fig,
                "F1-maximizing thresholds selected separately on each inner validation split for the predefined reference recipe. "
                "The displayed dataset rows have all five completed outer folds. Outer-test labels never choose thresholds, "
                "and threshold variation is not itself evidence of poor discrimination.",
                {"Thresholds by dataset and outer fold": part}))
    return pages


def cost_summary(campaign):
    cols = [c for c in ("base", "frozen", "elapsed_sec", "sec_per_step", "peak_gpu_gb",
                        "rows_seen", "trainable_params", "final_drift") if c in campaign.trials]
    d = campaign.trials.loc[campaign.trials.status.eq("OK"), cols]
    if len(cols) <= 2 or d.empty:
        return pd.DataFrame()
    return d.groupby(["base", "frozen"]).agg(["count", "median", "min", "max"])


def plot_sampling_endpoint(campaign, *, benchmark=False):
    data = cp.benchmark_effects(campaign) if benchmark else cp.endpoint_effects(campaign)
    if data.empty:
        return []
    pages = []
    for base, group in data.groupby("base"):
        matrix = group.pivot_table(index="dataset", columns="sampling", values="effect", aggfunc="mean")
        matrix = matrix.reindex(columns=list(style.SAMPLING_LABELS))
        for start in range(0,len(matrix),style.PAGE_ROWS):
            part = matrix.iloc[start:start+style.PAGE_ROWS]
            fig=cp._heatmap(part.rename(columns=style.SAMPLING_LABELS), f"{base}: protocol effects",
                            label=effect_label(campaign.metric))
            pages.append(cp.Page(f"sampling_endpoint_{campaign.track}_{base}_{benchmark}_{start//style.PAGE_ROWS+1}", fig,
                "Dataset-level endpoint effects for the three sampling protocols within experiment 3. "
                "Each effect is paired with that dataset's own base result. The shared proportional-PD sampling policy "
                "makes these within-experiment comparisons distinct from the balanced-PD main sweep. Missing cells are unavailable.",
                {"Sampling effects": part}))
    return pages

