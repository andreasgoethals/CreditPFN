"""Milestone parameter summaries and bounded resource views, separated from score curves."""
from __future__ import annotations

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
                data = pd.read_csv(path)
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
            axes[0].scatter(group[column], np.full(len(group), i), color=style.color(base),
                            s=style.POINT_SIZE, alpha=style.POINT_ALPHA)
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].set_xlabel(label)
        pages.append(Page(f"resource_{campaign.track}_{column}", fig,
            f"Per-trial median {label.lower()} over sampled training-phase observations, grouped by base and adaptation. "
            "Device counters are periodic samples, not precise process-level utilization or integrated energy. Missing counters are omitted."))
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
    """Reference recipe only: bound the view rather than overlaying the entire sweep."""
    data = campaign.evaluation
    if campaign.track != "pd" or data.empty or "calibration_bins" not in data:
        return []
    data = data[data.status.eq("OK")]
    if "domain" in data:
        data = data[data.domain.eq("credit")]
    raw = data.source.str.endswith("-untuned", na=False)
    trained = (data.source.str.endswith("-trained", na=False) & np.isclose(data.lr, 3e-7, atol=0, rtol=1e-8)
               & np.isclose(data.l2sp_lambda, .003) & ~data.use_lora.astype(bool))
    data = data[raw | trained]
    pages = []
    for (base, dataset), group in data.groupby(["base_short", "test_dataset_id"]):
        fig, axes = _subplots(f"{base}: {dataset}")
        for source, selected in group.groupby("source"):
            role = "Untuned" if source.endswith("-untuned") else "Adapted reference"
            for suffix, calibration in (("", "raw"), ("_platt", "Platt"), ("_isotonic", "isotonic")):
                column = "calibration_bins" + suffix
                if column not in selected:
                    continue
                records = [b for value in selected[column].dropna() for b in json.loads(value)]
                if not records:
                    continue
                bins = pd.DataFrame(records).groupby("bin").sum()
                bins = bins[bins["count"] > 0]
                axes[0].plot(bins.probability_sum/bins["count"], bins.positive_count/bins["count"],
                    marker="o", label=f"{role}: {calibration}",
                    color=style.color(role), linestyle={"raw":"-", "Platt":"--", "isotonic":":"}[calibration])
        axes[0].plot([0,1],[0,1],color=style.color("reference"),linestyle=":")
        axes[0].set_xlabel("Mean predicted probability")
        axes[0].set_ylabel("Observed positive fraction")
        axes[0].legend(ncol=2)
        pages.append(Page(f"reliability_{base}_{dataset}",fig,
            "Reliability curves for the predefined full-update reference (learning rate 3e-7, L2-SP 0.003) and its untuned base. "
            "Ten equal-width probability bins pool outer-test counts within this dataset. Platt and isotonic mappings are fitted on each inner validation set; "
            "no test labels select a threshold or mapping. Empty bins are omitted; these curves do not display uncertainty intervals."))
    return pages
