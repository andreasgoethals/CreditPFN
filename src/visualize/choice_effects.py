"""Compact descriptive CPT comparisons, with tables as the averaging unit.

Complete recipes and paired folds are required before contrasting choices.
Positive choice contrasts can mean less damage, rather than positive transfer.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import numpy as np
import pandas as pd

from src.visualize import campaign as cp, comparisons as cmp, dashboards as db
from src.visualize import interactions as ix, style

METRICS = {
    "roc_auc": (100., "AUC change (percentage points)"),
    "brier_score": (100., "Brier reduction (×100)"),
    "log_loss": (1., "Log-loss reduction (nats)"),
    "r2": (1., "R² change"),
    "rmse": (100., "Relative RMSE gain (%)"),
}


def _policy_note(track):
    return ("TabICLv2 PD effects retain the selected-table numerical-policy qualification."
            if track == "pd" else
            "Inference follows the recorded LGD policy.")


def _cohort_note(values):
    counts = values.groupby("base").dataset.nunique()
    if counts.nunique() == 1:
        return f"{int(counts.iloc[0])} complete tables per model. "
    return "Complete tables by model: " + ", ".join(
        f"{style.model_label(base)} {int(count)}" for base, count in counts.items()) + ". "


def benchmark_grid(run):
    """Retain one common complete table cohort across the displayed metrics."""
    metrics = ("roc_auc", "brier_score", "log_loss") if run.track == "pd" else ("r2", "rmse")
    frames = []
    for domain in ("credit", "ood"):
        parts = []
        for metric in metrics:
            values, _ = ix.factorial_tables(run, metric, domain)
            if not values.empty:
                parts.append(values.assign(metric=metric, domain=domain))
        if not parts:
            continue
        data = pd.concat(parts, ignore_index=True)
        support = data.groupby(["base", "dataset"]).metric.nunique()
        valid = support[support.eq(len(parts))].reset_index()[["base", "dataset"]]
        frames.append(data.merge(valid, on=["base", "dataset"], validate="many_to_one"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def recipe_summary(data):
    if data.empty:
        return pd.DataFrame()
    return data.groupby(["domain", "metric", *cp.FACTORS]).effect.agg(
        mean="mean", median="median", q25=lambda x: x.quantile(.25),
        q75=lambda x: x.quantile(.75), tables="size", improved=lambda x: int(x.gt(0).sum())
    ).reset_index()


def choice_tables(data):
    """Pair all remaining settings, average within table, then across tables."""
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pairs, levels = [], []
    for (domain, metric), part in data.groupby(["domain", "metric"]):
        for factor, reference in (("l2sp_lambda", 0.), ("frozen", False),
                                  ("learning_rate", part.learning_rate.min())):
            # These marginal levels are descriptive equal-recipe means, not
            # independent observations or estimates for a selected recipe.
            arms = part.groupby(["base", "dataset", factor]).effect.mean().reset_index()
            levels.append(arms.rename(columns={factor: "level"}).assign(
                domain=domain, metric=metric, factor=factor))
            for alternative in sorted(part[factor].unique()):
                if alternative == reference:
                    continue
                matched = cp.factor_contrasts(part, factor, reference, alternative)
                if matched.empty:
                    continue
                name = ("L2-SP\non − off" if factor == "l2sp_lambda" else
                        "Backbone frozen\n− all updated" if factor == "frozen" else
                        f"LR {alternative:.0e}\n− {reference:.0e}")
                within = matched.groupby(["base", "dataset"]).contrast.mean().reset_index()
                pairs.append(within.assign(domain=domain, metric=metric, choice=name,
                                           factor=factor, reference=reference, alternative=alternative))
    paired = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    marginal = pd.concat(levels, ignore_index=True) if levels else pd.DataFrame()
    summary = (paired.groupby(["domain", "metric", "base", "choice"], sort=False).contrast.agg(
        mean="mean", median="median", q25=lambda x: x.quantile(.25),
        q75=lambda x: x.quantile(.75), tables="size").reset_index()
        if not paired.empty else pd.DataFrame())
    return paired, summary, marginal


def _heatmap(ax, matrix, label, *, compact=False):
    array = matrix.to_numpy(float)
    limit = max(float(np.nanmax(np.abs(array))), np.finfo(float).eps)
    im = ax.imshow(array, aspect="auto", cmap=style.CMAP_DIVERGING, vmin=-limit, vmax=limit)
    ax.grid(False)
    ax.set_yticks(range(len(matrix)), [style.model_label(str(x)) for x in matrix.index])
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.tick_params(axis="both", labelsize=style.ANNOTATION_SIZE)
    for row, col in np.ndindex(array.shape):
        value = array[row, col]
        precision = 3 if "nats" in label or "R²" in label else 2
        text = "—" if not np.isfinite(value) else f"{value:+.{precision}f}"
        if np.isfinite(value) and 1e-12 < abs(value) < .5 * 10**(-precision):
            text = f"{value:+.2g}"  # Keep small effects visible rather than printing signed zero.
        color = style.GRID_TEXT_LIGHT if np.isfinite(value) and abs(value) > limit*.7 else style.GRID_TEXT_DARK
        ax.text(col, row, text, ha="center", va="center", fontsize=style.GRID_FONT, color=color)
    colorbar_label = label
    if compact:
        colorbar_label = {"AUC change (percentage points)": "AUC (pp)",
                          "Brier reduction (×100)": "Brier (×100)",
                          "Log-loss reduction (nats)": "Log loss (nats)",
                          "R² change": "R²", "Relative RMSE gain (%)": "RMSE (%)"}[label]
    ax.figure.colorbar(im, ax=ax, label=colorbar_label)


def plot_choices(data, track, domain="credit"):
    paired, summary, marginal = choice_tables(data)
    if summary.empty:
        return []
    selected = summary[summary.domain.eq(domain)]
    metrics = [m for m in METRICS if m in set(selected.metric)]
    if not metrics:
        return []
    fig, axes = plt.subplots(len(metrics), 1, squeeze=False, layout="constrained",
                             figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO))
    fig.suptitle(f"{track.upper()}: which CPT choices help held-out credit?" if domain == "credit"
                 else f"{track.upper()}: choices and non-credit retention")
    for ax, metric in zip(axes.flat, metrics):
        part = selected[selected.metric.eq(metric)]
        choices = list(dict.fromkeys(part.choice))
        matrix = part.pivot(index="base", columns="choice", values="mean").reindex(columns=choices)
        scale, label = METRICS[metric]
        _heatmap(ax, matrix*scale, label, compact=True)
    return [cp.Page(f"choice_contrasts_{domain}", fig,
        "Final five-fold benchmark contrasts between CPT choices. "
        + _cohort_note(data[data.domain.eq(domain)]) + "Each cell first pairs the same table and all other recipe settings, "
        "averages those matched differences within table, then weights tables equally. Positive values favor the named alternative. "
        "A favorable choice contrast can mean less deterioration; it does not establish a gain over original weights. "
        "Anchoring and scope contrasts average all four rates; each rate contrast averages both anchoring and scope settings. "
        "Models and metrics remain separate. Each model uses the same recipe-complete table cohort across displayed metrics. "
        + ("AUC contrasts are percentage points; Brier contrasts are multiplied by 100. Brier and log-loss signs are reversed so reductions are positive. "
           if track == "pd" else
           "R² contrasts use native units. RMSE contrasts divide the paired error difference by the original model's RMSE, not by the reference CPT arm; reductions are positive. ") +
        "Colors are symmetric about zero within each metric panel. These are descriptive fixed-grid contrasts, not causal architecture comparisons. "
        + _policy_note(track),
        {"Matched choice differences within each table": paired[paired.domain.eq(domain)],
         "Choice means and table spread": selected,
         "Marginal CPT-versus-original effects by choice level and table": marginal[marginal.domain.eq(domain)]},
        {"measurement": "five_fold_final_benchmark", "aggregation_unit": "table", "domain": domain})]


def plot_combinations(data, track, metric, domain="credit"):
    values = data[data.metric.eq(metric) & data.domain.eq(domain)] if not data.empty else data
    if values.empty:
        return []
    summary = recipe_summary(values)
    if summary.sampling.nunique() != 1:
        raise ValueError("Combination map needs a single sampling protocol")
    summary = summary.assign(row=summary.base.map(style.model_label) + summary.frozen.map(
        {False: " / all", True: " / frozen"}), column=summary.learning_rate.map(lambda x: f"{x:.0e}")
        + "\n" + summary.l2sp_lambda.map(lambda x: f"{x:g}"))
    matrix = summary.pivot(index="row", columns="column", values="mean")
    row_order = [style.model_label(b)+" / "+scope for b in sorted(summary.base.unique()) for scope in ("all", "frozen")]
    columns = [f"{lr:.0e}\n{lam:g}" for lr in sorted(summary.learning_rate.unique()) for lam in sorted(summary.l2sp_lambda.unique())]
    matrix = matrix.reindex(index=row_order, columns=columns)
    scale, label = METRICS[metric]
    fig, ax = plt.subplots(layout="constrained", figsize=style.figsize(style.WIDTH_FULL, style.GOLDEN))
    fig.suptitle(f"{track.upper()}: held-out credit CPT effects" if domain == "credit" else f"{track.upper()}: non-credit CPT effects")
    _heatmap(ax, matrix*scale, label)
    ax.set_xlabel("Peak learning rate (upper label) / L2-SP strength (lower label)")
    for boundary in range(2, len(matrix), 2):
        ax.axhline(boundary-.5, color=style.GRID_TEXT_LIGHT, linewidth=style.THIN_LINE)
    return [cp.Page(f"recipe_grid_{domain}_{metric}", fig,
        f"Final five-fold benchmark: {label}. " + _cohort_note(values)
        + "Each cell is the equal-table mean CPT effect relative to its own original weights; "
        "positive values favor CPT. All planned rate × anchoring × scope combinations are shown on one common table cohort within each model. "
        "Rows marked all update normally trainable parameters; frozen rows freeze the dominant transformer stack while retaining configured surrounding components. "
        "Neither mode uses LoRA. Colors share one symmetric scale; printed values retain small effects. "
        "Five outer folds are averaged within table. "
        + ("Retention additionally requires all four CPT partitions. " if domain == "ood" else "") +
        "Observed maxima are exploratory grid outcomes, not independently validated selected recipes. " + _policy_note(track),
        {"Every paired table and recipe effect": values, "Cell means and improvement counts": summary},
        {"measurement": "five_fold_final_benchmark", "metric": metric, "domain": domain, "aggregation_unit": "table"})]


def plot_conservative_recipes(data, track, metric):
    """Detail the two fixed low rates, without selecting cells by outcome."""
    rates = (3e-7, 1e-6)
    values = (data[data.domain.eq("credit") & data.metric.eq(metric)
                   & data.learning_rate.isin(rates)].copy() if not data.empty else data)
    if values.empty:
        return []
    summary = recipe_summary(values)
    if summary.sampling.nunique() != 1:
        raise ValueError("Conservative recipe view needs a single sampling protocol")
    scale, label = METRICS[metric]
    summary["plotted_mean"] = summary["mean"] * scale
    models = sorted(summary.base.unique())
    fig = plt.figure(figsize=style.figsize(style.WIDTH_FULL, style.PANEL_RATIO), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, style.CHOICE_LEGEND_RATIO))
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    settings = [(rate, anchor) for rate in rates for anchor in (0., .003)]
    for ax, frozen in zip(axes, (False, True)):
        for offset, (rate, anchor) in zip(style.CHOICE_POINT_OFFSETS, settings):
            part = summary[summary.frozen.eq(frozen) & summary.learning_rate.eq(rate)
                           & summary.l2sp_lambda.eq(anchor)]
            ax.scatter(part.plotted_mean, part.base.map(models.index) + offset,
                       marker=style.PANEL_MARKERS[int(anchor > 0)], s=style.POINT_SIZE,
                       color=style.TRAJECTORY_LR_COLORS[rate])
        ax.axvline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_yticks(range(len(models)), [style.model_label(base) for base in models])
        ax.set_ylim(len(models) - .5, -.5)
        ax.set_title("Backbone frozen" if frozen else "All parameters updated")
        ax.set_xlabel(label + " (+ better)")
    # Common limits retain every displayed mean, including harmful low-rate cells.
    limits = [ax.get_xlim() for ax in axes]
    for ax in axes:
        ax.set_xlim(min(x[0] for x in limits), max(x[1] for x in limits))
    legends = [
        ("Peak learning rate", [Line2D([], [], marker="o", linestyle="", color=style.TRAJECTORY_LR_COLORS[r], label=f"{r:.0e}") for r in rates]),
        ("L2-SP strength", [Line2D([], [], marker=style.PANEL_MARKERS[i], linestyle="", color=style.color("reference"), label=f"{v:g}") for i, v in enumerate((0., .003))]),
    ]
    for i, (title, handles) in enumerate(legends):
        ax = fig.add_subplot(grid[1, i]); ax.set_axis_off()
        legend = Legend(ax, handles, [h.get_label() for h in handles], title=title,
                        loc="center", ncol=2, frameon=True)
        ax.add_artist(legend)
    fig.suptitle(f"{track.upper()}: low-rate effects on held-out credit tables")
    return [cp.Page(f"conservative_recipes_credit_{metric}", fig,
        "Detail of the full five-fold benchmark grid at the two fixed lowest peak learning rates "
        "(3e-7 and 1e-6). All anchoring and parameter-scope combinations at these rates are included; "
        "cells are not selected by their outcomes. Each point is an equal-table mean CPT change relative to "
        "the same model's original weights, after averaging five folds within each table. "
        + _cohort_note(values) + "Color denotes rate and marker shape denotes L2-SP strength. "
        "Vertical offsets separate settings and have no quantitative meaning. Both panels share a scale "
        "that retains all displayed means and zero; positive values favor CPT. This focused mean view "
        "does not show uncertainty or table spread, and does not imply every table improves. The full grid "
        "retains the higher-rate outcomes. Neither parameter-scope mode uses LoRA. " + _policy_note(track),
        {"Every low-rate paired table effect": values, "Low-rate equal-table means and spread": summary},
        {"measurement": "five_fold_final_benchmark", "metric": metric, "domain": "credit", "aggregation_unit": "table"})]


def _recipe_panels(title, rates):
    fig = plt.figure(figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, style.CHOICE_LEGEND_RATIO))
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]
    band = grid[2, :].subgridspec(1, 3)
    groups = [
        ("Peak learning rate", [Line2D([], [], marker="o", linestyle="", color=style.TRAJECTORY_LR_COLORS[r], label=f"{r:.0e}") for r in rates], 2),
        ("L2-SP strength", [Line2D([], [], marker=style.PANEL_MARKERS[i], linestyle="", color=style.color("reference"), label=f"{v:g}") for i, v in enumerate((0., .003))], 1),
        ("Parameters updated", [Line2D([], [], marker="o", linestyle="", color=style.color("reference"), markerfacecolor="none" if f else style.color("reference"), label="Backbone frozen" if f else "All parameters") for f in (False, True)], 1),
    ]
    for i, (title_, handles, columns) in enumerate(groups):
        ax = fig.add_subplot(band[0, i]); ax.set_axis_off()
        legend = Legend(ax, handles, [h.get_label() for h in handles], title=title_, loc="center", ncol=columns,
                        frameon=True, fontsize=style.ANNOTATION_SIZE, title_fontsize=style.ANNOTATION_SIZE)
        ax.add_artist(legend)
    fig.suptitle(title)
    return fig, axes


def _mark(ax, row, x, y):
    color = style.TRAJECTORY_LR_COLORS[float(row.learning_rate)]
    ax.scatter(x, y, marker=style.PANEL_MARKERS[int(row.l2sp_lambda > 0)], s=style.POINT_SIZE,
               facecolors="none" if row.frozen else color, edgecolors=color)


def plot_monitor_changes(run):
    raw = cmp.observations(run, metric=db.monitor_metric(run), endpoint=True)
    values = cmp.dataset_means(raw, run, ["effect"])
    if values.empty:
        return []
    planned = run.trials[cp.FACTORS].drop_duplicates()
    recipe = [c for c in cp.FACTORS if c != "base"]
    keep = []
    for base, expected in planned.groupby("base"):
        identities = set(expected[recipe].itertuples(index=False, name=None))
        for dataset, table in values[values.base.eq(base)].groupby("dataset"):
            if set(table[recipe].itertuples(index=False, name=None)) == identities:
                keep.append((base, dataset))
    values = values.merge(pd.DataFrame(keep, columns=["base", "dataset"]), on=["base", "dataset"], validate="many_to_one")
    if values.empty:
        return []
    summary = values.groupby(cp.FACTORS).effect.agg(mean="mean", tables="size").reset_index()
    rates = sorted(summary.learning_rate.unique())
    fig, axes = _recipe_panels(f"{run.track.upper()}: held-out credit monitor changes", rates)
    scale, label = METRICS[db.monitor_metric(run)]
    for ax, (base, group) in zip(axes, summary.groupby("base")):
        for row in group.itertuples():
            offset = style.CHOICE_POINT_OFFSETS[2*int(row.frozen)+int(row.l2sp_lambda > 0)]
            _mark(ax, row, rates.index(row.learning_rate)+offset, row.mean*scale)
        ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_title(style.model_label(base)); ax.set_ylabel(label)
        ax.set_xticks(range(len(rates)), [f"{r:.0e}" for r in rates]); ax.set_xlabel("Peak learning rate")
    for ax in axes[summary.base.nunique():]: ax.set_visible(False)
    return [cp.Page("monitor_recipe_changes", fig,
        f"Held-out credit fixed-context monitors at {run.target:,} successful updates, paired with update zero. "
        + _cohort_note(values) +
        "Each mark is an equal-table mean for one rate, anchoring and update-scope combination on a recipe-complete cohort. "
        "Horizontal offsets separate the four combinations at each rate; they have no quantitative meaning. "
        "Colors encode rate, circle/square encodes L2-SP, and filled/open marks encode full/frozen-backbone updates. "
        "Zero means unchanged score. Vertical scales vary by base to expose small effects. "
        "These fixed-context monitoring measurements are separate from the final five-fold benchmark.",
        {"Equal-table monitor means": summary, "Per-table monitor effects": values,
         "All recorded partition and table monitor effects": raw},
        {"measurement": "fixed_context_monitor", "aggregation_unit": "table", "updates": int(run.target)})]


def plot_transfer_retention(data, track):
    metric = "roc_auc" if track == "pd" else "r2"
    summary = recipe_summary(data[data.metric.eq(metric)]) if not data.empty else pd.DataFrame()
    if summary.empty:
        return []
    a = summary[summary.domain.eq("credit")]
    b = summary[summary.domain.eq("ood")]
    paired = a.merge(b, on=["metric", *cp.FACTORS], suffixes=("_credit", "_retention"), validate="one_to_one")
    if paired.empty:
        return []
    fig, axes = _recipe_panels(f"{track.upper()}: credit gains and non-credit retention", sorted(paired.learning_rate.unique()))
    scale, label = METRICS[metric]
    for ax, (base, group) in zip(axes, paired.groupby("base")):
        for row in group.itertuples(): _mark(ax, row, row.mean_credit*scale, row.mean_retention*scale)
        ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.axvline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        short = "ΔAUC (pp)" if track == "pd" else "ΔR²"
        ax.set_title(style.model_label(base)); ax.set_xlabel("Credit "+short); ax.set_ylabel("Non-credit "+short)
    for ax in axes[paired.base.nunique():]: ax.set_visible(False)
    return [cp.Page("benchmark_transfer_retention", fig,
        "Final five-fold benchmark effects relative to original weights, matched by CPT recipe across the two domains. "
        + "Credit: " + _cohort_note(data[data.metric.eq(metric) & data.domain.eq("credit")])
        + "Non-credit: " + _cohort_note(data[data.metric.eq(metric) & data.domain.eq("ood")]) +
        "The upper-right quadrant improves both credit and non-credit means; the lower-right improves credit while reducing retention. "
        "Each domain weights its recipe-complete tables equally; non-credit effects first average all four matched CPT partitions. "
        "Axis limits vary across base models. Each of the 16 fixed-grid recipes remains visible through rate, anchoring and scope encodings; "
        "overlapping outcomes may coincide. The finite non-credit panel does not establish general retention. " + _policy_note(track),
        {"Matched credit and retention recipe means": paired},
        {"measurement": "five_fold_final_benchmark", "metric": metric, "aggregation_unit": "table"})]


def reference_profiles(run):
    metrics = ("roc_auc", "pr_auc", "brier_score", "log_loss", "ece", "f1") if run.track == "pd" else ("r2", "rmse", "mae")
    parts = [cp.reference_rows(cp.benchmark_effects(run, metric)).assign(metric=metric) for metric in metrics]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
