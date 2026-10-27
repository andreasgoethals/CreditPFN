"""Final factorial response maps on recipe-complete, equally weighted tables.

Partitions and row folds are pairing keys, never extra independent observations.
No checkpoint is selected by its test result.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.visualize import campaign as cp, comparisons as cmp, style

METRICS = {
    "roc_auc": ("AUC change", "CPT minus original AUC"),
    "brier_score": ("Brier reduction", "Original minus CPT Brier score"),
    "r2": ("R² change", "CPT minus original R²"),
    "rmse": ("Relative RMSE gain", "Fractional RMSE reduction: (original − CPT) / original"),
}


def factorial_tables(run, metric, domain="credit"):
    """Require every planned recipe on one fixed table cohort within each base.

    For retention, all training partitions must first be complete for every
    table/recipe. Missing one fold, partition or arm removes that table from
    this map, rather than changing its weight at different learning rates.
    """
    values = cmp.observations(run, domain=domain, benchmark=True, metric=metric)
    values = cmp.dataset_means(values, run, ["effect"])
    if values.empty:
        return pd.DataFrame(), pd.DataFrame()
    recipe = [key for key in cp.FACTORS if key != "base"]
    planned = run.trials[cp.FACTORS].drop_duplicates()
    keep = []
    for base, expected in planned.groupby("base"):
        identities = set(expected[recipe].itertuples(index=False, name=None))
        part = values[values.base.eq(base)]
        for dataset, table in part.groupby("dataset"):
            if set(table[recipe].itertuples(index=False, name=None)) == identities:
                keep.append((base, dataset))
    cohort = pd.DataFrame(keep, columns=["base", "dataset"])
    if cohort.empty:
        return pd.DataFrame(), pd.DataFrame()
    values = values.merge(cohort, on=["base", "dataset"], validate="many_to_one")
    summary = values.groupby(cp.FACTORS).effect.agg(
        mean="mean", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
        tables="size").reset_index()
    return values, summary


def plot_factorial(run, metric, *, domain="credit"):
    """Two scope rows and one model column; anchor means and table IQRs."""
    values, summary = factorial_tables(run, metric, domain)
    if values.empty:
        return []
    bases = sorted(summary.base.unique())
    fig, axes = plt.subplots(2, len(bases), squeeze=False, sharex=True,
        figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
    label, definition = METRICS[metric]
    domain_label = cmp.DOMAINS[domain]
    fig.suptitle(f"{run.track.upper()} / {domain_label}: factorial {label}")
    for col, base in enumerate(bases):
        part = summary[summary.base.eq(base)]
        for row, frozen in enumerate((False, True)):
            ax = axes[row, col]
            arm = part[part.frozen.eq(frozen)]
            if arm.empty:
                style.note(ax, "No complete recipe cohort")
                continue
            for lam, group in arm.groupby("l2sp_lambda"):
                group = group.sort_values("learning_rate")
                color = style.color(base) if lam else style.color("untuned")
                ax.fill_between(group.learning_rate, group.q25, group.q75,
                                color=color, alpha=style.INTERVAL_ALPHA)
                ax.plot(group.learning_rate, group["mean"], color=color,
                        linestyle=style.ANCHOR_STYLES[bool(lam)],
                        marker=style.PANEL_MARKERS[int(bool(lam))],
                        markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE,
                        label=f"L2-SP {lam:g}")
            ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(style.model_label(base))
            else:
                ax.set_xlabel("Peak learning rate")
            if col == 0:
                scope = "All parameters" if not frozen else "Frozen backbone"
                ax.set_ylabel(scope + "\n" + label)
        # Compare scopes on the same scale without hiding smaller model effects.
        limits = [ax.get_ylim() for ax in axes[:, col]]
        low, high = min(x[0] for x in limits), max(x[1] for x in limits)
        for ax in axes[:, col]:
            ax.set_ylim(low, high)
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=style.color("untuned"), linestyle=style.ANCHOR_STYLES[False],
               marker=style.PANEL_MARKERS[0], label="L2-SP 0"),
        Line2D([], [], color=style.color("reference"), linestyle=style.ANCHOR_STYLES[True],
               marker=style.PANEL_MARKERS[1], label="L2-SP 0.003 (base color)")],
        loc="outside lower center", ncol=2)
    counts = values.groupby("base").dataset.nunique()
    cohort = "; ".join(f"{style.model_label(base)}: {n} tables" for base, n in counts.items())
    caption = (
        f"Completed five-fold final benchmark, {run.track.upper()} {domain_label.lower()}. {definition}. "
        f"Support: {cohort}. The same tables support every planned learning-rate, anchoring and update-scope combination within each base. "
        "Lines are equal-table means after pairing five outer folds; bands span the table interquartile range, not confidence intervals. "
        "The upper row updates all parameters; the lower freezes the dominant backbone while retaining the configured surrounding trainable components. "
        "Within each base, both rows share a vertical scale; scales can differ across bases. Positive values favor CPT. "
        "Peak learning rates are 3e-7, 1e-6, 1e-5 and 3e-5. "
        + ("Retention effects first average all four matched training partitions within table. " if domain == "ood" else "")
        + "These maps describe the fixed grid rather than a test-selected optimum. "
        + ("" if run.track == "pd" else "LGD RMSE reductions and R² changes retain distinct scales and are not native training losses.")
    )
    if run.track == "pd":
        caption += " Main-grid PD draws balance both context and query classes."
    return [cp.Page(f"factorial_{domain}_{metric}", fig, caption,
        {"Recipe-complete table effects": values, "Means and table interquartile ranges": summary},
        {"measurement": "five_fold_final_benchmark", "metric": metric, "domain": domain,
         "aggregation_unit": "table", "cohort": "all_planned_recipes_within_base",
         "tables_by_base": {str(k): int(v) for k, v in counts.items()}})]
