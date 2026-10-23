"""Milestone parameter summaries and bounded resource views, separated from score curves."""
from __future__ import annotations

from src.visualize.inputs import read_csv

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from src.visualize import style
from src.visualize.campaign import Page, _subplots
from src.visualize.training_viz import _resolve_paths
from src.visualize.inputs import load_consolidated
from src.utils.consolidate_output import matches_run


def load(campaign, kind):
    if kind not in ("parameters", "resources"):
        raise ValueError(kind)
    cache = getattr(campaign, "_diagnostic_cache", None)
    if cache is None:
        cache = campaign._diagnostic_cache = {}
    if kind in cache:
        return cache[kind].copy()
    paths = _resolve_paths(campaign.cfg)
    frame = load_consolidated(paths["run_name"], f"{kind}_{campaign.track}")
    if frame is None:
        frames = []
        suffix = f".{kind}.csv" + (".gz" if kind == "parameters" else "")
        for path in sorted((paths["epoch_dir"] / campaign.track).glob("*" + suffix)):
            if matches_run(path.name, paths["run_name"], track=campaign.track):
                data = read_csv(path)
                data["trial_name"] = path.name.removesuffix(suffix)
                frames.append(data)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not frame.empty:
        frame = frame.merge(campaign.trials[["trial_name", "base", "frozen", "sampling"]],
                            on="trial_name", validate="many_to_one")
    cache[kind] = frame
    return frame.copy()


def plot_resource_curves(campaign):
    """Device counters along training, one trial contribution per update window."""
    from src.visualize.campaign import optimization_window
    data = load(campaign, "resources")
    if data.empty:
        return []
    data = data[data.phase.eq("training") & data.successful_updates.gt(0)].copy()
    width = optimization_window(campaign)
    data["window_end"] = np.minimum(np.ceil(data.successful_updates / width) * width, campaign.target)
    fields = {"gpu_utilization_percent": "Utilization (%)", "power_watts": "Power (W)",
              "device_memory_used_mib": "Memory in use (MiB)"}
    fields = {k: v for k, v in fields.items() if k in data and pd.to_numeric(data[k], errors="coerce").notna().any()}
    if not fields:
        return []
    pages = []
    factor = "sampling" if data.sampling.nunique() > 1 else "frozen"
    for base, group in data.groupby("base"):
        fig, axes = plt.subplots(len(fields), 1, sharex=True, squeeze=False, layout="constrained",
                                 figsize=style.figsize(style.WIDTH_FULL, style.PANEL_RATIO))
        fig.suptitle(f"{campaign.track.upper()} / {base}: device counters during training")
        tables = []
        for ax, (column, label) in zip(axes.flat, fields.items()):
            values = group.assign(**{column: pd.to_numeric(group[column], errors="coerce")})
            per_trial = values.groupby(["trial_name", factor, "window_end"])[column].median().reset_index()
            for arm, observations in per_trial.groupby(factor):
                curve = observations.groupby("window_end")[column].agg(median="median", count="count")
                windows = np.unique(np.minimum(np.arange(width, campaign.target+width, width), campaign.target))
                curve = curve.reindex(windows)
                name = style.SAMPLING_LABELS[arm] if factor == "sampling" else style.adaptation_label(arm)
                color = style.SAMPLING_COLORS[arm] if factor == "sampling" else style.color("frozen" if arm else "full")
                ax.plot(curve.index, curve["median"], label=name, color=color)
                tables.append(curve.rename_axis("window_end").reset_index().assign(metric=column, arm=name))
            ax.set_ylabel(label)
        axes.flat[0].legend()
        axes.flat[-1].set_xlabel("Successful-update window end")
        pages.append(Page(f"resource_curves_{campaign.track}_{base}", fig,
            f"Sampled training-phase device counters in {width:,}-update windows. Each line is the median of "
            "within-trial medians for an adaptation or sampling arm; learning-rate and anchoring recipes are pooled. "
            "Counts accompany every window in the text summary. Missing windows remain gaps; changing support "
            "can affect the curve. Counters are device-level samples, not integrated energy or billed allocation time.",
            {"Window medians and trial counts": pd.concat(tables, ignore_index=True)}))
    return pages


def summary(campaign):
    parameters, resources = load(campaign, "parameters"), load(campaign, "resources")
    text = f"{len(parameters):,} parameter-tensor measurements; {len(resources):,} resource samples."
    if not resources.empty:
        text += "\nGPU counter status: " + resources.gpu_status.value_counts().to_string()
    return text


def plot_resources(campaign):
    data = load(campaign, "resources")
    if data.empty:
        return []
    pages = []
    fields = {"gpu_utilization_percent": "Device utilization (%)", "power_watts": "Sampled device power (W)",
              "device_memory_used_mib": "Device memory in use (MiB)"}
    for column, label in fields.items():
        data[column] = pd.to_numeric(data[column], errors="coerce")
        good = data[data[column].notna() & data.phase.eq("training")]
        if good.empty:
            continue
        # One median per trial prevents long trials from dominating the plotted distribution.
        medians = good.groupby(["trial_name", "base", "frozen"])[column].median().reset_index()
        fig, axes = _subplots(f"{campaign.track.upper()}: {label}")
        labels = []
        for i, ((base, frozen), group) in enumerate(medians.groupby(["base", "frozen"])):
            labels.append(f"{base} / {style.adaptation_label(frozen)}")
            jitter = np.linspace(-style.JITTER_WIDTH, style.JITTER_WIDTH, len(group)) if len(group) > 1 else np.zeros(1)
            axes[0].scatter(group[column], i+jitter, color=style.color(base),
                            s=style.POINT_SIZE, alpha=style.POINT_ALPHA)
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].set_xlabel(label)
        pages.append(Page(f"resource_{campaign.track}_{column}", fig,
            f"Per-trial median {label.lower()} over sampled training-phase observations, grouped by base and adaptation. "
            "Vertical offsets separate coincident trials. Device counters are periodic samples, not precise process-level utilization or integrated energy. Missing counters are omitted.",
            {"Per-trial sampled medians": medians}))
    return pages


def plot_parameters(campaign):
    data = load(campaign, "parameters")
    if data.empty:
        return []
    data = data[data.relative_change.notna() & data.successful_updates.gt(0)]
    if data.empty:
        return []
    # Per-tensor summaries are retained on disk; plots avoid hundreds of overlaid layers.
    grouped = data.groupby(["trial_name", "base", "frozen", "successful_updates"]).relative_change.median().reset_index()
    pages = []
    for base, group in grouped.groupby("base"):
        fig, axes = _subplots(f"{base}: parameter movement")
        for frozen, arm in group.groupby("frozen"):
            curve = arm.groupby("successful_updates").relative_change.median()
            axes[0].plot(curve.index, curve, label=style.adaptation_label(frozen),
                         color=style.color("frozen" if frozen else "full"))
        axes[0].set_xlabel("Successful optimizer updates")
        axes[0].set_ylabel("Median relative tensor movement")
        axes[0].legend()
        pages.append(Page(f"parameter_movement_{campaign.track}_{base}", fig,
            "Relative parameter movement from initialization: first the median across anchored parameter tensors within each trial, "
            "then the median across available trials of an adaptation arm. This is an optimization summary across recipes; detailed tensor measurements "
            "and the norm-weighted whole-model drift are stored separately. Unanchored frozen tensors are excluded."))
    return pages


def plot_retention_benchmark(campaign):
    from src.visualize.campaign import benchmark_effects, plot_response_surfaces
    data = benchmark_effects(campaign, domain="noncredit")
    pages = plot_response_surfaces(data, campaign.metric)
    for page in pages:
        page.name = "retention_" + page.name
        page.caption = "Non-credit panel. " + page.caption
    return pages


def calibration_effects(campaign):
    """Fixed-reference proper-score effects on a common, fully paired table cohort."""
    from src.visualize.campaign import benchmark_effects, reference_rows
    if campaign.track != "pd":
        return pd.DataFrame()
    parts = []
    # Older exports may have the original Brier/ECE panel but no calibrated
    # log loss. Preserve those measured scores; never manufacture a third panel.
    metrics = [metric for metric in ("brier_score", "ece", "log_loss")
               if all(metric + suffix in campaign.evaluation for suffix in ("", "_platt", "_isotonic"))]
    for metric in metrics:
        for suffix, calibration in (("", "Raw"), ("_platt", "Platt"), ("_isotonic", "Isotonic")):
            effects = reference_rows(benchmark_effects(campaign, metric + suffix))
            if effects.empty:
                continue
            parts.append(effects.assign(metric=metric, calibration=calibration))
    if not parts:
        return pd.DataFrame()
    values = pd.concat(parts, ignore_index=True)
    keys = ["base", "dataset", "metric", "calibration"]
    if values.duplicated(keys).any():
        raise ValueError("Ambiguous calibration reference: multiple recipes or partitions per table")
    complete = values.groupby(["base", "dataset"]).size().eq(3*len(metrics))
    cohort = complete[complete].reset_index()[["base", "dataset"]]
    return values.merge(cohort, on=["base", "dataset"], validate="many_to_one")


def plot_calibration_overview(campaign):
    """Show all-table calibration effects without hundreds of reliability curves."""
    data = calibration_effects(campaign)
    if data.empty:
        return []
    metrics = [(m, label) for m, label in (("brier_score", "Brier reduction"), ("ece", "ECE reduction"),
                                          ("log_loss", "Log-loss reduction")) if m in set(data.metric)]
    fig, axes = plt.subplots(1, len(metrics), squeeze=False, layout="constrained",
                             figsize=style.figsize(style.WIDTH_FULL, style.PANEL_RATIO))
    fig.suptitle("PD: probability quality before and after calibration")
    rows = [(base, calibration) for base in sorted(data.base.unique())
            for calibration in ("Raw", "Platt", "Isotonic")]
    statistics = []
    for ax, (metric, label) in zip(axes.flat, metrics):
        labels = []
        for index, (base, calibration) in enumerate(rows):
            effects = data.loc[data.base.eq(base) & data.calibration.eq(calibration) & data.metric.eq(metric), "effect"]
            mean, q25, q75 = effects.mean(), effects.quantile(.25), effects.quantile(.75)
            ax.plot([q25, q75], [index, index], color=style.color(base), linewidth=style.THIN_LINE)
            ax.plot(mean, index, marker="D", color=style.color(base), markersize=style.CURVE_MARKER_SIZE)
            labels.append(f"{base} / {calibration} (n={len(effects)})")
            statistics.append(dict(base=base, calibration=calibration, metric=metric,
                                   tables=len(effects), mean=mean, q25=q25, q75=q75))
        ax.axvline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_yticks(range(len(rows)), labels)
        if metric != "brier_score":
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis()
        ax.set_xlabel(label + "\nPositive favors CPT")
    return [Page("calibration_overview", fig,
        "Changes in the displayed outer-test probability-quality scores for the predefined "
        "full-update reference (LR 3e-7, L2-SP 0.003, one-sample training) relative to its untuned base. "
        "Positive reductions favor continued pretraining (CPT). Raw probabilities and validation-fitted "
        f"Platt and isotonic calibration use a common dataset cohort within each base: all {3*len(metrics)} scores "
        "require the same five completed, matched outer folds. Diamonds show equal-dataset means; "
        "horizontal lines span dataset quartiles, not confidence intervals. Log loss, when exported, is in nats; ECE depends on the binning scheme.",
        {"Per-dataset probability-quality changes": data.reset_index(drop=True),
         "Plotted means and dataset quartiles": pd.DataFrame(statistics)})]


def plot_reliability(campaign, *, max_datasets=None):
    """Compare calibration methods side by side, requiring bins from every fold."""
    from src.visualize.campaign import _complete_folds
    from src.visualize.training_viz import compact_base
    import matplotlib.pyplot as plt
    if max_datasets is not None and (not isinstance(max_datasets, int) or max_datasets < 1):
        raise ValueError("max_datasets must be a positive integer or None")
    data = campaign.evaluation
    if campaign.track != "pd" or data.empty or "calibration_bins" not in data:
        return []
    data = data[data.status.eq("OK")].copy()
    if "domain" in data:
        data = data[data.domain.fillna("credit").eq("credit")]
    raw = data.source.str.endswith("-untuned", na=False)
    trained = (data.source.str.endswith("-trained", na=False)
               & np.isclose(data.lr, 3e-7, atol=0, rtol=1e-8)
               & np.isclose(data.l2sp_lambda, .003)
               & ~data.use_lora.astype(bool) & data.epoch_pass_mode.eq("one_sample"))
    data = _complete_folds(data[raw | trained], "roc_auc")
    pages = []
    for base, group in data.groupby("base_short"):
        all_records = []
        for suffix, calibration in (("", "raw"), ("_platt", "Platt"), ("_isotonic", "isotonic")):
            column = "calibration_bins" + suffix
            if column not in group:
                continue
            records = []
            for dataset, part in group.groupby("test_dataset_id"):
                if part[column].isna().any() or not part.source.str.endswith("-trained").any() or not part.source.str.endswith("-untuned").any():
                    continue
                pair_keys = [c for c in ("eval_run", "split", "fold_idx") if c in part]
                untreated = part[part.source.str.endswith("-untuned")]
                adapted = part[part.source.str.endswith("-trained")]
                if set(map(tuple, untreated[pair_keys].to_numpy())) != set(map(tuple, adapted[pair_keys].to_numpy())):
                    continue
                dataset_records = []
                for source, selected in part.groupby("source"):
                    fold_bins = [json.loads(value) for value in selected[column]]
                    # A successful AUC fold need not have a fitted calibrator.
                    # One missing calibration fold must not look like five-fold evidence.
                    if not all(fold_bins):
                        continue
                    bins = [b for items in fold_bins for b in items]
                    merged = pd.DataFrame(bins).groupby("bin").sum(numeric_only=True)
                    merged = merged[merged["count"].gt(0)].reset_index()
                    merged["predicted"] = merged.probability_sum / merged["count"]
                    merged["observed"] = merged.positive_count / merged["count"]
                    merged["dataset"], merged["role"] = dataset, "Untuned" if source.endswith("-untuned") else "Adapted reference"
                    merged["calibration"] = calibration
                    if not merged.empty:
                        dataset_records.append(merged)
                if len(dataset_records) == 2:
                    records.extend(dataset_records)
            all_records.extend(records)
        if not all_records:
            continue
        binned = pd.concat(all_records, ignore_index=True)
        eligible = sorted(binned.dataset.unique())
        roster = None
        if max_datasets is not None:
            if "n_test_rows" not in group:
                continue
            # Select by input size, never by the observed gain or calibration curve.
            counts = group.assign(n_test_rows=pd.to_numeric(group.n_test_rows, errors="coerce"))
            counts = counts[counts.n_test_rows.gt(0) & np.isfinite(counts.n_test_rows)]
            counts = counts.groupby(["test_dataset_id", "method_dirname"]).n_test_rows.agg(["sum", "count"])
            counts = counts[counts["count"].eq(5)].groupby(level=0)["sum"].max()
            roster = counts.rename("outer_test_rows").rename_axis("dataset").reset_index()
            roster = roster[roster.dataset.isin(eligible)].sort_values(["outer_test_rows", "dataset"])
            if roster.empty:
                continue
            indices = np.linspace(0, len(roster)-1, min(max_datasets, len(roster)), dtype=int)
            eligible = roster.iloc[indices].dataset.tolist()
            roster["selected"] = roster.dataset.isin(eligible)
        for page_no, (start, stop) in enumerate(style.page_slices(len(eligible), style.RELIABILITY_ROWS), 1):
            datasets = eligible[start:stop]
            ratio = style.PANEL_RATIO if len(datasets) > 1 else style.SMALL_RATIO
            fig, axes = plt.subplots(len(datasets), 3, squeeze=False,
                                     figsize=style.figsize(style.WIDTH_FULL, ratio),
                                     layout="constrained", sharex=True, sharey=True)
            fig.suptitle(f"PD / {compact_base(base)}: probability calibration")
            for row, dataset in enumerate(datasets):
                for col, calibration in enumerate(("raw", "Platt", "isotonic")):
                    ax = axes[row, col]
                    part = binned[binned.dataset.eq(dataset) & binned.calibration.eq(calibration)]
                    for role, bins in part.groupby("role"):
                        ax.plot(bins.predicted, bins.observed, marker="s" if role == "Untuned" else "o",
                                markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE,
                                linestyle="--" if role == "Untuned" else "-", label=role, color=style.color(role))
                    ax.plot([0,1], [0,1], color=style.color("reference"), linestyle=":")
                    ax.set_xlim(0,1); ax.set_ylim(0,1)
                    ax.set_title(f"{dataset}\n{calibration.capitalize()}", fontsize=style.ANNOTATION_SIZE)
                    if row == len(datasets)-1: ax.set_xlabel("Mean probability")
                    if col == 0: ax.set_ylabel("Observed fraction")
                    if part.empty:
                        ax.text(.5, .5, "Five paired folds\nunavailable", transform=ax.transAxes,
                                ha="center", va="center", fontsize=style.ANNOTATION_SIZE)
            handles = {}
            for ax in axes.flat:
                artists, labels = ax.get_legend_handles_labels()
                handles.update(zip(labels, artists))
            fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=2)
            selection = (" Examples are chosen at evenly spaced dataset-size ranks (smallest and largest "
                         "when two are shown), using total outer-test row counts independently of model scores. "
                         "They illustrate calibration shapes rather than the typical treatment effect. "
                         if max_datasets is not None else "")
            tables = {"Probability bins and counts": binned[binned.dataset.isin(datasets)].reset_index(drop=True)}
            if roster is not None:
                tables["Example-selection roster"] = roster.reset_index(drop=True)
            pages.append(Page(f"reliability_{compact_base(base)}_{page_no}", fig,
                    "Raw, Platt-scaled and isotonic positive-class reliability for the predefined full-update reference "
                    "(LR 3e-7, L2-SP 0.003, one-sample training) and its untuned base. "
                    "Rows identify datasets and columns identify calibration methods. Each populated panel requires "
                    "bin records from the same five completed outer folds for both models; missing panels are labeled. "
                    "Ten equal-width bins pool outer-test counts within the dataset; empty bins are omitted and bin counts "
                    "are retained in the text summary. Calibrators are fitted only on inner validation rows. "
                    "The diagonal denotes agreement of predicted and observed probabilities; these curves have no uncertainty intervals."
                    + selection, tables))
    return pages
