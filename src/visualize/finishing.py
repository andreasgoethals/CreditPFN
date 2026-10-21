"""Bounded paper-readiness analyses of existing evidence; no fitting or selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset_names import display_name, is_proprietary
from src.visualize import campaign as cp, style

UNCERTAINTY_NOTE = (
    "The recipe is fixed: full updates, LR 3e-7, L2-SP 0.003, seed 42. "
    "Five paired outer folds are averaged within each table. The 95% percentile "
    "bootstrap interval resamples tables (10,000 draws, seed 20261020), not folds, "
    "rows, recipes or overlapping training corpora. It describes sensitivity to "
    "the observed table collection conditional on the fitted checkpoints, under "
    "a table-exchangeability assumption; it is not a deployment guarantee or a "
    "training-seed confidence interval. Unknown shared populations can invalidate "
    "that assumption. No significance-based recipe selection is performed."
)
SOURCE_NOTE = (
    "A source URL identifies a release or platform, not independent borrowers. "
    "The ledger records documented shared sources and unresolved provenance; "
    "it does not certify absence of overlap. LendingClub PD/LGD and SBA PD/LGD "
    "share sources, but the two tasks have separate CPT corpora and are never "
    "pooled as independent replications. The documented-public-source sensitivity "
    "excludes unverified/local and restricted sources; it changes the population "
    "being summarized and cannot repair contamination in already trained weights."
)


def source_ledger(track):
    """Public provenance only; do not expose private IDs, providers or contents."""
    from src.data.preprocessing import DATASET_METADATA
    rows = []
    for identifier, meta in DATASET_METADATA.items():
        if meta.get("track") != track:
            continue
        name = display_name(identifier)
        private = is_proprietary(identifier)
        url = None if private else meta.get("source_url")
        source = "restricted" if private else meta.get("source", "unknown")
        group = ("LendingClub PD/LGD" if source == "lendingclub" else
                 "SBA PD/LGD" if url and "sba-loans-case" in url else name)
        rows.append(dict(dataset=name, track=track, source_group=group,
                         provenance="restricted" if private else "documented URL" if url else "unverified release",
                         source_url=url or "", documented_public=bool(url),
                         independence="not established"))
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def source_partition_audit(run):
    """Audit only documented source links and byte identities in the original plan."""
    import json
    from src.utils.experiment import digest_json
    from src.utils.paths import manifests_dir, group_for_run
    from src.visualize.inputs import analysis_location
    group = group_for_run(str(run.cfg.run_name))
    root = analysis_location(group, "manifests", manifests_dir(group))
    path = root / "plans" / f"{run.cfg.run_name}_{run.track}.json"
    ledger = source_ledger(run.track)
    if not path.is_file():
        return ledger.assign(plan_check="unavailable: prepared plan missing")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("sha256") != digest_json({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("Source audit: corrupt original plan")
    hashes = {}
    for key, spec in plan["identities"].items():
        if digest_json(spec) != key:
            raise ValueError("Source audit: corrupt original trial identity")
        for bucket in spec["datasets"].values():
            for item in bucket:
                name = display_name(item["id"])
                if name in hashes and hashes[name] != item["sha256"]:
                    raise ValueError("Source audit: inconsistent dataset bytes across planned trials")
                hashes[name] = item["sha256"]
    groups = ledger.set_index("dataset").source_group.to_dict()
    if set(hashes) != set(groups):
        raise ValueError("Source audit: dataset hashes do not cover the registered corpus")
    rows = []
    for part in plan["partitions"]:
        train, test = [set(map(display_name, part[key])) for key in ("train", "test")]
        if train & test:
            raise ValueError("Source audit: a test table entered its training corpus")
        if not (train | test) <= set(groups):
            raise ValueError("Source audit: unknown table")
        for name in sorted(test):
            rows.append(dict(dataset=name, holdout_partition=part["index"], plan_check="verified",
                             known_shared_source_in_training=sum(groups[d] == groups[name] for d in train),
                             identical_file_in_training=sum(hashes.get(d) == hashes.get(name) for d in train),
                             processed_sha256=hashes[name]))
    roster = pd.DataFrame(rows)
    if roster.dataset.duplicated().any() or set(roster.dataset) != set(ledger.dataset):
        raise ValueError("Source audit: holdouts do not cover each credit table exactly once")
    return ledger.merge(roster, on="dataset", validate="one_to_one")


def bootstrap_tables(values, *, draws=10000, seed=20261020):
    """Conditional equal-table mean interval; never treats CV folds as replications."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite, distinct table effects")
    if draws < 100:
        raise ValueError("Too few bootstrap draws")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    omitted = (values.sum()-values)/(len(values)-1)
    return dict(tables=len(values), mean=values.mean(), median=np.median(values),
                ci_low=low, ci_high=high, leave_one_out_low=omitted.min(),
                leave_one_out_high=omitted.max(), positive_tables=int((values > 0).sum()))


def uncertainty_tables(run):
    """Fixed-reference effects only, with an outcome-independent source sensitivity."""
    ledger = source_ledger(run.track)
    primary = ("roc_auc", "brier_score") if run.track == "pd" else ("r2", "rmse")
    observations, summaries = [], []
    for metric in primary:
        data = cp.reference_rows(cp.benchmark_effects(run, metric))
        if data.empty:
            continue
        if data.duplicated(["base", "dataset"]).any():
            raise ValueError("Expected one reference effect per base and held-out table")
        data = data.merge(ledger, on="dataset", how="left", validate="many_to_one")
        if data.provenance.isna().any():
            raise ValueError("Unknown table in provenance ledger")
        data["metric"] = metric
        observations.append(data)
        for base, group in data.groupby("base"):
            for scenario, subset in (
                ("All credit tables", group),
                ("Documented public sources", group[group.documented_public]),
            ):
                if len(subset) >= 2:
                    summaries.append(dict(base=base, metric=metric, scenario=scenario,
                                          **bootstrap_tables(subset.effect)))
    return (pd.concat(observations, ignore_index=True) if observations else pd.DataFrame(),
            pd.DataFrame(summaries), ledger)


def plot_uncertainty(run):
    data, estimates, ledger = uncertainty_tables(run)
    if estimates.empty:
        return []
    from src.visualize.paper_figures import effect_label
    metrics = list(dict.fromkeys(estimates.metric))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(metrics), figsize=style.figsize(style.WIDTH_FULL, style.SMALL_RATIO),
                             squeeze=False, layout="constrained")
    axes = axes[0]
    fig.suptitle(f"{run.track.upper()}: reference CPT and table uncertainty")
    for ax, metric in zip(axes, metrics):
        part = estimates[estimates.metric.eq(metric)]
        bases = sorted(part.base.unique())
        for i, base in enumerate(bases):
            for scenario, marker, offset in (
                ("All credit tables", "o", -style.JITTER_WIDTH/2),
                ("Documented public sources", "s", style.JITTER_WIDTH/2),
            ):
                sub = part[part.base.eq(base) & part.scenario.eq(scenario)]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                ax.plot([row.ci_low, row.ci_high], [i+offset]*2,
                        color=style.color(base), linewidth=style.THIN_LINE)
                ax.scatter([row["mean"]], [i+offset], marker=marker, s=style.POINT_SIZE,
                           facecolor=style.color(base) if marker == "o" else "none",
                           edgecolor=style.color(base), label=f"{scenario} (n={int(row.tables)})" if i == 0 else None)
        ax.axvline(0, color=style.color("reference"), linestyle=":")
        ax.set_yticks(range(len(bases)), [style.model_label(b) for b in bases])
        ax.set_xlabel(effect_label(metric))
    axes[0].legend(loc="best")
    return [cp.Page("reference_uncertainty", fig,
        "Mean paired benchmark effects of the predefined full-update CPT reference and "
        "95% conditional table-bootstrap intervals. Positive effects favor CPT; RMSE is "
        "reported as fractional error reduction. Filled circles include all credit tables; "
        "open squares retain documented public sources. Each table contributes the mean "
        "of five paired outer folds. Source restriction is based on provenance, not scores. "
        "Intervals do not capture training-seed uncertainty or establish source independence.",
        {"Reference per-table effects": data, "Intervals and leave-one-table-out sensitivity": estimates,
         "Source-population ledger and partition audit": source_partition_audit(run)})]


LGD_METRICS = {
    "r2": "R² (higher better)",
    "rmse": "RMSE (lower better)",
    "mean_pinball_loss": "Pinball loss (lower better)",
    "interval_coverage_90": "90% interval coverage",
    "interval_width_90": "90% interval width",
    "neg_nll": "Clipped log score (higher better)",
}


def lgd_distribution_scores(run):
    """Keep native metric levels, including the historical capped log score."""
    if run.track != "lgd":
        raise ValueError("Distribution diagnostic requires LGD")
    pieces = []
    for metric in LGD_METRICS:
        data = cp.reference_rows(cp.benchmark_effects(run, metric))
        if data.empty:
            continue
        data = data[data.base.eq("v2.6")]
        pieces.append(data[["dataset", "untuned", "trained"]].assign(
            metric=metric, difference=data.trained-data.untuned))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def boundary_errors(original, adapted):
    """Observed errors on target endpoints/interior, not theoretical estimator bias."""
    from src.visualize.prediction_diagnostics import paired_values, KEYS
    paired_values(original, adapted, "lgd")  # refuse foreign/incomplete row pairing
    left, right = [v.set_index(KEYS).sort_index() for v in (original, adapted)]
    y = left.y_true.to_numpy()
    if ((y < 0) | (y > 1)).any():
        raise ValueError("LGD boundary analysis requires targets in [0, 1]")
    rows = []
    for label, mask in (("Target = 0", y == 0), ("0 < target < 1", (y > 0) & (y < 1)), ("Target = 1", y == 1)):
        for state, frame in (("original", left), ("CPT", right)):
            error = frame.y_pred.to_numpy()[mask]-y[mask]
            rows.append(dict(region=label, state=state, rows=int(mask.sum()),
                             mean_error=float(error.mean()) if len(error) else np.nan,
                             rmse=float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
                             mae=float(np.mean(abs(error))) if len(error) else np.nan))
    return pd.DataFrame(rows)


def lgd_boundaries(run):
    from src.visualize.prediction_diagnostics import _PredictionReader, _candidates
    data = _candidates(run, True)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = cp._complete_folds(data[data.base_short.str.startswith("v2.6")], "rmse")
    reader = _PredictionReader(run)
    rows, coverage = [], []
    for (dataset, partition), group in data.groupby(["test_dataset_id", "split"]):
        original = group[group.source.str.endswith("-untuned")]
        adapted = group[group.source.str.endswith("-trained")]
        if len(original) != 5 or len(adapted) != 5:
            coverage.append(dict(dataset=dataset, status="unavailable: five paired folds required"))
            continue
        try:
            table = boundary_errors(reader.load(original), reader.load(adapted))
        except FileNotFoundError:
            coverage.append(dict(dataset=dataset, status="unavailable: predictions missing"))
            continue
        rows.append(table.assign(dataset=dataset, partition=partition))
        coverage.append(dict(dataset=dataset, status="verified paired predictions"))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(coverage))


def plot_lgd_distribution(run):
    data = lgd_distribution_scores(run)
    if data.empty:
        return []
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=style.figsize(style.WIDTH_FULL), constrained_layout=True)
    fig.suptitle("LGD / TabPFN v2.6: point and distribution scores")
    for ax, (metric, label) in zip(axes.flat, LGD_METRICS.items()):
        part = data[data.metric.eq(metric)]
        if part.empty:
            ax.set_visible(False)
            continue
        for row in part.itertuples():
            ax.plot([0, 1], [row.untuned, row.trained], color=style.color("annotation"),
                    linewidth=style.THIN_LINE, alpha=style.POINT_ALPHA)
        ax.plot([0, 1], [part.untuned.mean(), part.trained.mean()], marker="o",
                color=style.color("v2.6"), linewidth=style.THIN_LINE,
                markersize=style.CURVE_MARKER_SIZE)
        if metric == "interval_coverage_90":
            ax.axhline(.9, color=style.color("reference"), linestyle=":")
        ax.set_xticks([0, 1], ["Original", "CPT"])
        ax.set_title(label)
    return [cp.Page("v26_distribution_tradeoff", fig,
        "TabPFN v2.6 before and after the predefined CPT recipe. Thin lines connect "
        "each credit table's five-fold means; the colored line is their equal-table mean. "
        "The dotted coverage line marks the nominal 90% level. Log scores use the "
        "historical per-row NLL replacement/clipping at ±100 nats; they are not uncapped "
        "likelihood estimates.",
        {"Point and distribution scores": data})]
