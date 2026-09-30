"""Milestone parameter summaries and bounded resource views, separated from score curves."""
from __future__ import annotations

from src.visualize.inputs import read_csv

import numpy as np
import pandas as pd
import json

from src.visualize import style
from src.visualize.campaign import Page, _subplots
from src.visualize.training_viz import _resolve_paths
from src.visualize.inputs import load_consolidated
from src.utils.consolidate_output import matches_run


def load(campaign, kind):
    if kind not in ("parameters", "resources"):
        raise ValueError(kind)
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
    if frame.empty:
        return frame
    return frame.merge(campaign.trials[["trial_name", "base", "frozen"]], on="trial_name", validate="many_to_one")


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
            labels.append(f"{base} / {'frozen' if frozen else 'full'}")
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
            axes[0].plot(curve.index, curve, label="Frozen backbone" if frozen else "Full updates",
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


def plot_reliability(campaign):
    """Bounded per-dataset reliability panels with paired, complete outer folds."""
    from src.visualize.campaign import _complete_folds
    from src.visualize.training_viz import compact_base
    import matplotlib.pyplot as plt
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
        for suffix, calibration in (("", "raw"), ("_platt", "Platt"), ("_isotonic", "isotonic")):
            column = "calibration_bins" + suffix
            if column not in group:
                continue
            records, eligible = [], []
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
                    bins = [b for value in selected[column] for b in json.loads(value)]
                    if not bins:
                        continue
                    merged = pd.DataFrame(bins).groupby("bin").sum(numeric_only=True)
                    merged = merged[merged["count"].gt(0)].reset_index()
                    merged["predicted"] = merged.probability_sum / merged["count"]
                    merged["observed"] = merged.positive_count / merged["count"]
                    merged["dataset"], merged["role"] = dataset, "Untuned" if source.endswith("-untuned") else "Adapted reference"
                    if not merged.empty:
                        dataset_records.append(merged)
                if len(dataset_records) == 2:
                    records.extend(dataset_records)
                    eligible.append(dataset)
            if not records:
                continue
            binned = pd.concat(records, ignore_index=True)
            for start in range(0, len(eligible), 4):
                datasets = eligible[start:start+4]
                fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.PANEL_RATIO),
                                         layout="constrained", sharex=True, sharey=True)
                fig.suptitle(f"{compact_base(base)}: {calibration} reliability")
                for ax, dataset in zip(axes.flat, datasets):
                    for role, bins in binned[binned.dataset.eq(dataset)].groupby("role"):
                        ax.plot(bins.predicted, bins.observed, marker="s" if role == "Untuned" else "o",
                                linestyle="--" if role == "Untuned" else "-", label=role, color=style.color(role))
                    ax.plot([0,1], [0,1], color=style.color("reference"), linestyle=":")
                    ax.set_xlim(0,1); ax.set_ylim(0,1)
                    ax.set_title(str(dataset), fontsize=style.ANNOTATION_SIZE)
                    ax.set_xlabel("Mean probability"); ax.set_ylabel("Observed fraction"); ax.legend()
                for ax in list(axes.flat)[len(datasets):]:
                    ax.set_visible(False)
                pages.append(Page(f"reliability_{compact_base(base)}_{calibration}_{start//4+1}", fig,
                    f"{calibration.capitalize()} positive-class reliability for the predefined full-update reference "
                    "(LR 3e-7, L2-SP 0.003, one-sample training) and its untuned base. "
                    "Each panel is a separate dataset, requiring the same five completed outer folds for both models. "
                    "Ten equal-width bins pool outer-test counts within the dataset; empty bins are omitted and bin counts "
                    "are retained in the text summary. Calibrators are fitted only on inner validation rows. "
                    "The diagonal denotes agreement of predicted and observed probabilities; these curves have no uncertainty intervals.",
                    {"Probability bins and counts": binned[binned.dataset.isin(datasets)].reset_index(drop=True)}))
    return pages
