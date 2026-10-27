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


#: Which tested hyperparameter a choice contrast varies; drives label colours.
_CHOICE_PARAMETER = {"l2sp_lambda": "l2sp_lambda", "frozen": "scope", "learning_rate": "learning_rate"}


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


def _heatmap(ax, matrix, label, *, compact=False, counts=None):
    """Diverging effect map; ``counts`` adds an "improved/total tables" line per cell."""
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
        if counts is not None and np.isfinite(value):
            improved, total = counts[0].iat[row, col], counts[1].iat[row, col]
            text += f"\n{int(improved)}/{int(total)} up"
        color = style.GRID_TEXT_LIGHT if np.isfinite(value) and abs(value) > limit*.7 else style.GRID_TEXT_DARK
        ax.text(col, row, text, ha="center", va="center", fontsize=style.GRID_FONT - (1 if counts is not None else 0),
                color=color, linespacing=1.0)
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
    fig.suptitle(f"{track.upper()}: effect of each CPT choice (matched contrasts)" if domain == "credit"
                 else f"{track.upper()}: effect of each CPT choice on non-credit retention")
    factor_of = paired.drop_duplicates("choice").set_index("choice").factor
    for ax, metric in zip(axes.flat, metrics):
        part = selected[selected.metric.eq(metric)]
        choices = list(dict.fromkeys(part.choice))
        matrix = part.pivot(index="base", columns="choice", values="mean").reindex(columns=choices)
        scale, label = METRICS[metric]
        _heatmap(ax, matrix*scale, label, compact=True)
        for tick, choice in zip(ax.get_xticklabels(), choices):
            tick.set_color(style.PARAMETER_COLORS[_CHOICE_PARAMETER[factor_of[choice]]])
    return [cp.Page(f"choice_contrasts_{domain}", fig,
        "Experiment 1, final five-fold benchmark on "
        + ("held-out credit tables. " if domain == "credit" else "the non-credit retention panel. ")
        + _cohort_note(data[data.domain.eq(domain)]) + "Each column changes one CPT choice: L2-SP anchoring on (λ = 0.003) "
        "minus off (λ = 0), frozen transformer backbone minus all parameters updated, or a larger peak learning rate minus "
        "the lowest (3e-7). Each cell pairs the same table and all other recipe settings, averages those matched differences "
        "within table, then weights tables equally; positive values favour the named alternative. Anchoring and scope "
        "contrasts average over all four learning rates; each rate contrast averages over both anchoring and scope settings. "
        "A favourable contrast can mean less deterioration rather than a gain over the original weights (see the recipe grid). "
        "Column labels are coloured by the parameter varied (learning rate blue, L2-SP orange, update scope grey). "
        + ("AUC contrasts are in percentage points and Brier contrasts are multiplied by 100; Brier and log-loss signs are "
           "reversed so that reductions are positive. "
           if track == "pd" else
           "R² contrasts use native units. RMSE contrasts divide the paired error difference by the original model's RMSE; "
           "reductions are positive. ") +
        "Colours are symmetric about zero within each metric panel. These are descriptive contrasts over the fixed grid.",
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
        {False: " · all weights", True: " · frozen backbone"}), column=summary.learning_rate.map(lambda x: f"{x:.0e}")
        + "\n" + summary.l2sp_lambda.map(lambda x: f"{x:g}"))
    matrix = summary.pivot(index="row", columns="column", values="mean")
    row_order = [style.model_label(b)+scope for b in sorted(summary.base.unique()) for scope in (" · all weights", " · frozen backbone")]
    rates = sorted(summary.learning_rate.unique())
    anchors = sorted(summary.l2sp_lambda.unique())
    columns = [f"{lr:.0e}\n{lam:g}" for lr in rates for lam in anchors]
    matrix = matrix.reindex(index=row_order, columns=columns)
    improved = summary.pivot(index="row", columns="column", values="improved").reindex(index=row_order, columns=columns)
    tables = summary.pivot(index="row", columns="column", values="tables").reindex(index=row_order, columns=columns)
    scale, label = METRICS[metric]
    fig, ax = plt.subplots(layout="constrained", figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO))
    fig.suptitle(f"{track.upper()}: CPT − original, every recipe (held-out credit)" if domain == "credit"
                 else f"{track.upper()}: CPT − original, every recipe (non-credit retention)")
    _heatmap(ax, matrix*scale, label, counts=(improved, tables))
    # Two tick rows, one per hyperparameter, each in that parameter's colour: learning rate
    # on top, L2-SP strength below, so a reader can match a column to both settings at once.
    lr_color, anchor_color = style.PARAMETER_COLORS["learning_rate"], style.PARAMETER_COLORS["l2sp_lambda"]
    ax.set_xticks(range(len(columns)), [f"{lam:g}" for lr in rates for lam in anchors], color=anchor_color)
    ax.set_xlabel("L2-SP strength λ", color=anchor_color)
    top = ax.secondary_xaxis("top")
    top.set_xticks([i * len(anchors) + (len(anchors) - 1) / 2 for i in range(len(rates))], [f"{lr:.0e}" for lr in rates],
                   color=lr_color)
    top.set_xlabel("Peak learning rate", color=lr_color)
    top.tick_params(length=0, labelsize=style.ANNOTATION_SIZE)
    for boundary in range(len(anchors), len(columns), len(anchors)):
        ax.axvline(boundary-.5, color=lr_color, linewidth=style.THIN_LINE)
    for boundary in range(2, len(matrix), 2):
        ax.axhline(boundary-.5, color=style.GRID_TEXT_LIGHT, linewidth=style.MEAN_CURVE_LINE)
    band = {"roc_auc": "0 pp", "brier_score": "0", "log_loss": "0", "r2": "0", "rmse": "0%"}[metric]
    return [cp.Page(f"recipe_grid_{domain}_{metric}", fig,
        f"Experiment 1, final five-fold benchmark on {'held-out credit tables' if domain == 'credit' else 'the non-credit retention panel'}: "
        f"{label}, CPT minus the original model of the same base. " + _cohort_note(values)
        + "Each cell is the equal-table mean effect of one recipe, with the number of tables whose effect exceeds "
        f"{band} (\"up\") over the number of complete tables. Positive values favour CPT. Columns cross the four peak learning "
        "rates (top labels, blue) with L2-SP strength λ ∈ {0, 0.003} (bottom labels, orange); rows cross each base with the "
        "update scope (all trainable weights, or the dominant transformer stack frozen while embeddings and prediction "
        "components remain trainable). Every planned combination is shown on one common table cohort within each base. "
        "Colours share one symmetric scale; printed values retain small effects. Five outer folds are averaged within table "
        "before tables are weighted equally. "
        + ("Retention tables additionally average all four CPT partitions. " if domain == "ood" else "") +
        "The predefined reference recipe is full updates at LR 3e-7 and λ 0.003; any other cell with a higher value is an "
        "observed grid maximum, not an independently validated choice.",
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
            ys = part.base.map(models.index) + offset
            color = style.TRAJECTORY_LR_COLORS[rate]
            # Table interquartile range and median, so a mean driven by one or two tables is visible.
            ax.hlines(ys, part.q25 * scale, part.q75 * scale, color=color, linewidth=style.THIN_LINE)
            ax.scatter(part["median"] * scale, ys, marker="|", s=style.POINT_SIZE, color=color)
            ax.scatter(part.plotted_mean, ys,
                       marker=style.PANEL_MARKERS[int(anchor > 0)], s=style.POINT_SIZE,
                       color=color, edgecolors="black", linewidths=.3, zorder=3)
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
    fig.suptitle(f"{track.upper()}: low-rate recipes, mean and table spread")
    return [cp.Page(f"conservative_recipes_credit_{metric}", fig,
        "Experiment 1, final five-fold benchmark on held-out credit tables, at the two lowest peak learning rates "
        "(3e-7 and 1e-6). All anchoring and update-scope combinations at these rates are shown; none is selected by its "
        f"outcome. Values are {label.lower()}, CPT minus the original model of the same base. Markers give the equal-table "
        "mean (colour: learning rate; circle λ = 0, square λ = 0.003), vertical ticks the median table and horizontal "
        "segments the table interquartile range (heterogeneity across tables, not a confidence interval). "
        + _cohort_note(values) + "Five outer folds are averaged within table. Vertical offsets only separate settings. "
        "Both panels share one horizontal scale; positive values favour CPT. A mean far from its median indicates that a "
        "few tables drive the average.",
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
    fig, axes = _recipe_panels(f"{track.upper()}: credit transfer against non-credit retention", sorted(paired.learning_rate.unique()))
    scale, label = METRICS[metric]
    for ax, (base, group) in zip(axes, paired.groupby("base")):
        for row in group.itertuples(): _mark(ax, row, row.mean_credit*scale, row.mean_retention*scale)
        ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.axvline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        short = "ΔAUC (pp)" if track == "pd" else "ΔR²"
        ax.set_title(style.model_label(base)); ax.set_xlabel("Held-out credit "+short); ax.set_ylabel("Non-credit "+short)
    for ax in axes[paired.base.nunique():]: ax.set_visible(False)
    return [cp.Page("benchmark_transfer_retention", fig,
        "Experiment 1, final five-fold benchmark: each mark is one of the 16 recipes, placed at its equal-table mean change "
        "on held-out credit tables (horizontal) and on the same-task non-credit retention panel (vertical), both relative to "
        "the original model of the same base. "
        + "Credit: " + _cohort_note(data[data.metric.eq(metric) & data.domain.eq("credit")])
        + "Non-credit: " + _cohort_note(data[data.metric.eq(metric) & data.domain.eq("ood")]) +
        "Colour encodes peak learning rate, circle/square L2-SP λ = 0/0.003, filled/open all-weights/frozen-backbone updates. "
        "The upper-right quadrant improves both domains; the lower-right gains on credit while losing on the retention panel. "
        "Non-credit effects first average all four CPT training partitions per table. Axis limits differ between models; "
        "coinciding recipes overlap. A four-table retention panel does not establish general retention.",
        {"Matched credit and retention recipe means": paired},
        {"measurement": "five_fold_final_benchmark", "metric": metric, "aggregation_unit": "table"})]


def plot_table_recipes(data, track, metric):
    """Tables × recipes for each base and scope: which tables carry a recipe's mean."""
    values = data[data.metric.eq(metric) & data.domain.eq("credit")] if not data.empty else data
    if values.empty:
        return []
    scale, label = METRICS[metric]
    bases = sorted(values.base.unique())
    rates = sorted(values.learning_rate.unique())
    anchors = sorted(values.l2sp_lambda.unique())
    order = values.groupby("dataset").effect.mean().sort_values(ascending=False).index.tolist()
    finite = np.abs(values.effect.to_numpy(float) * scale)
    limit = max(float(np.quantile(finite, style.HEATMAP_COLOR_QUANTILE)), np.finfo(float).eps)
    pages = []
    lr_color, anchor_color = style.PARAMETER_COLORS["learning_rate"], style.PARAMETER_COLORS["l2sp_lambda"]
    for frozen in (False, True):
        part = values[values.frozen.eq(frozen)]
        if part.empty:
            continue
        fig, axes = plt.subplots(1, len(bases), sharey=True, layout="constrained",
                                 figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO))
        axes = np.atleast_1d(axes)
        for ax, base in zip(axes, bases):
            matrix = (part[part.base.eq(base)].pivot_table(index="dataset", columns=["learning_rate", "l2sp_lambda"],
                      values="effect").reindex(index=order, columns=[(r, a) for r in rates for a in anchors])) * scale
            image = ax.imshow(np.clip(matrix.to_numpy(float), -limit, limit), aspect="auto",
                              cmap=style.CMAP_DIVERGING, vmin=-limit, vmax=limit)
            ax.grid(False)
            ax.set_xticks(range(len(matrix.columns)), [f"{a:g}" for _, a in matrix.columns], color=anchor_color,
                          rotation=90, fontsize=style.GRID_FONT - 1)
            top = ax.secondary_xaxis("top")
            top.set_xticks([i * len(anchors) + (len(anchors) - 1) / 2 for i in range(len(rates))],
                           [f"{r:.0e}" for r in rates], color=lr_color, rotation=90, fontsize=style.GRID_FONT - 1)
            top.tick_params(length=0)
            for boundary in range(len(anchors), len(matrix.columns), len(anchors)):
                ax.axvline(boundary - .5, color=lr_color, linewidth=style.THIN_LINE)
            ax.set_title(style.model_label(base), pad=22)
            ax.set_xlabel("λ", color=anchor_color)
        axes[0].set_yticks(range(len(order)), order)
        for tick in axes[0].get_yticklabels():
            tick.set_fontsize(style.GRID_FONT)
        bar = fig.colorbar(image, ax=list(axes), extend="both", shrink=.8)
        bar.set_label(label)
        fig.suptitle(f"{track.upper()}: table-level CPT − original, {'frozen backbone' if frozen else 'all weights updated'}")
        pages.append(cp.Page(f"table_recipes_{'frozen' if frozen else 'full'}_{metric}", fig,
            f"Experiment 1, final five-fold benchmark: {label.lower()} of every held-out credit table (rows) for each "
            f"recipe with {'the dominant transformer backbone frozen' if frozen else 'all trainable weights updated'}, "
            "CPT minus the original model of the same base. Columns cross peak learning rate (upper labels, blue) with "
            "L2-SP strength λ (lower labels, orange). Rows are ordered by the table's mean effect over all recipes and bases, "
            "so tables that improve under many recipes are at the top. Five outer folds are averaged within table. Colours "
            f"are symmetric about zero and saturate at the 95th percentile of |effect| ({limit:.2f}); exact values are in the "
            "notebook summary.",
            {"Table × recipe effects": part[["base", "dataset", "learning_rate", "l2sp_lambda", "frozen", "effect"]]},
            {"measurement": "five_fold_final_benchmark", "metric": metric, "domain": "credit", "aggregation_unit": "table"}))
    return pages


def reference_profiles(run):
    metrics = ("roc_auc", "pr_auc", "brier_score", "log_loss", "ece", "f1") if run.track == "pd" else ("r2", "rmse", "mae")
    parts = [cp.reference_rows(cp.benchmark_effects(run, metric)).assign(metric=metric) for metric in metrics]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
