"""Research-question views of existing campaigns, without creating extra trials."""
from dataclasses import replace
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.utils.study_layout import NUMBERS, STUDIES
from src.visualize import campaign as cp, choice_effects as ce, style


def study_text(text):
    """Translate historical prose labels, never scientific IDs or file paths."""
    return re.sub(r"\b([Ee]xperiment) ([1-8])\b", lambda m:
                  f"{m[1]} {NUMBERS[int(m[2])]}", text)


def design_table():
    return pd.DataFrame([dict(experiment=n, rq=v[0], question=v[1],
                             historical_sources=", ".join(v[2]), analysis=v[3]) for n, v in STUDIES.items()])


def unique_training_inventory(studies):
    """Count shared controls and factorial checkpoints once across analysis views."""
    rows = []
    seen = set()
    for campaign, arm, run in studies:
        for trial in run.trials.to_dict("records"):
            key = (run.track, trial["trial_name"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(track=run.track, trial=trial["trial_name"],
                public_studies="1,3,4" if campaign == 1 else str(NUMBERS[campaign]),
                campaign=str(run.cfg.run_name), base=trial["base"], status=trial["status"],
                seed=int(run.cfg.seed), learning_rate=trial["learning_rate"],
                l2sp=trial["l2sp_lambda"], frozen=trial["frozen"]))
    return pd.DataFrame(rows)


def inventory_summary(inventory):
    if inventory.empty:
        return inventory.copy()
    return inventory.groupby(["public_studies", "campaign", "base", "status"]).size().rename("unique_trials").reset_index()


def plot_secondary_effects(primary, secondary, track):
    """The outcome-independent reference on both metrics, retaining individual tables."""
    from src.visualize import synthesis as sy
    keys = ["config", "base", "dataset"]
    a = primary[primary.reference][keys + ["original", "adapted", "effect"]]
    b = secondary[secondary.reference][keys + ["original", "adapted", "effect"]]
    data = a.merge(b, on=keys, suffixes=("_primary", "_secondary"), validate="one_to_one")
    if data.empty:
        return []
    means = data.groupby("base")[["effect_primary", "effect_secondary"]].mean().reset_index()
    first, second = sy.PRIMARY[track], sy.SECONDARY[track]
    fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
    bases = [base for base in sy.BASES if base in set(data.base)]
    for ax, base in zip(axes.flat, bases):
        part = data[data.base.eq(base)]
        mean = means[means.base.eq(base)].iloc[0]
        ax.scatter(part.effect_primary * sy.SCALE[first], part.effect_secondary * sy.SCALE[second],
                   color=style.color(base), s=style.POINT_SIZE)
        ax.scatter(mean.effect_primary * sy.SCALE[first], mean.effect_secondary * sy.SCALE[second],
                   color=style.color("reference"), marker="D", s=style.POINT_SIZE)
        ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.axvline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_title(style.model_label(base))
        ax.set_xlabel(sy.SHORT_UNITS[first]); ax.set_ylabel(sy.SHORT_UNITS[second])
    for ax in list(axes.flat)[len(bases):]:
        ax.remove()
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=style.color("annotation"), label="Individual table"),
                        Line2D([], [], marker="D", linestyle="", color=style.color("reference"), label="Equal-table mean")],
               loc="outside lower center", ncol=2)
    fig.suptitle(f"{track.upper()}: does the reference improve both metrics?")
    return [cp.Page("reference_two_metrics", fig,
        "The predefined full-update reference (peak LR 3e-7, L2-SP 0.003, seed 42) on held-out credit tables. "
        f"Horizontal values are {sy.UNITS[first]}; vertical values are {sy.UNITS[second]}. "
        "Each colored point represents one table with five complete paired outer folds for both metrics; the diamond is their equal-table mean. "
        "Both directions are oriented so that positive favors CPT. Axis ranges differ by base to retain small effects. "
        + ("AUC measures ranking; Brier score measures probability error, so their changes need not agree. " if track == "pd" else
           "R² and RMSE derive from squared error and are not independent confirmations; relative RMSE puts the size of error changes in context. "),
        {"Named-table original/CPT scores and paired effects": data, "Means on the same two-metric cohort": means},
        {"study_numbering": "v2", "measurement": "five_fold_final_benchmark", "aggregation_unit": "table"})]


def learning_rate_view(run):
    """E1 is the full-update, anchored slice of the shared factorial."""
    trials = run.trials[~run.trials.frozen & np.isclose(run.trials.l2sp_lambda, .003)].copy()
    names = set(trials.trial_name)
    frames = {}
    for key in ("trajectories", "histories"):
        frame = getattr(run, key)
        frames[key] = frame[frame.trial_name.isin(names)].copy() if "trial_name" in frame else frame.copy()
    return replace(run, trials=trials, **frames)


def factorial_contrasts(grid, experiment):
    """Require all four cells within each table/rate before differences of differences."""
    if experiment not in (3, 4):
        raise ValueError("Only E3 anchoring and E4 freezing use this contrast")
    if grid.empty:
        return pd.DataFrame()
    keys = ["domain", "metric", "base", "dataset", "learning_rate", "sampling"]
    if grid.duplicated(keys + ["frozen", "l2sp_lambda"]).any():
        raise ValueError("Duplicate factorial table/recipe")
    cells = grid.pivot(index=keys, columns=["frozen", "l2sp_lambda"], values="effect")
    expected = [(False, 0.), (False, .003), (True, 0.), (True, .003)]
    if any(key not in cells for key in expected):
        return pd.DataFrame()
    cells = cells[expected].dropna()
    a, b, c, d = (cells[key] for key in expected)
    if experiment == 3:
        effects = {"All parameters": b-a, "Frozen backbone": d-c}
    else:
        effects = {"L2-SP off": c-a, "L2-SP 0.003": d-b}
    effects["Interaction"] = (d-c)-(b-a)
    return pd.concat([values.rename("contrast").reset_index().assign(condition=label)
                      for label, values in effects.items()], ignore_index=True)


def plot_factorial(grid, track, experiment, domain="credit"):
    data = factorial_contrasts(grid, experiment)
    if data.empty:
        return []
    pages = []
    for metric in (("roc_auc", "brier_score") if track == "pd" else ("r2", "rmse")):
        part = data[data.domain.eq(domain) & data.metric.eq(metric)]
        if part.empty:
            continue
        # Fix the table cohort across all four rates within each base.
        rates = sorted(grid.learning_rate.unique())
        wanted = len(rates) * 3
        support = part.groupby(["base", "dataset"]).size()
        complete = support[support.eq(wanted)].reset_index()[["base", "dataset"]]
        part = part.merge(complete, on=["base", "dataset"], validate="many_to_one")
        if part.empty:
            continue
        means = part.groupby(["base", "learning_rate", "condition"], sort=False).contrast.mean().reset_index()
        fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
        title = "Adding L2-SP" if experiment == 3 else "Freezing the backbone"
        population = "held-out credit" if domain == "credit" else "non-credit retention"
        fig.suptitle(f"{track.upper()}: {title} · {population}")
        conditions = list(part.condition.unique())
        colors = [style.color("full"), style.color("frozen"), style.color("reference")]
        scale, label = ce.METRICS[metric]
        bases = [b for b in ("v2", "v2.6", "v3", "tabicl-v2") if b in set(part.base)]
        for ax, base in zip(axes.flat, bases):
            for i, condition in enumerate(conditions):
                curve = means[means.base.eq(base) & means.condition.eq(condition)].sort_values("learning_rate")
                ax.plot(curve.learning_rate, curve.contrast*scale, color=colors[i],
                    linestyle=":" if condition == "Interaction" else style.FACTOR_ARM_STYLES[i],
                    marker=style.PANEL_MARKERS[i], markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE)
            ax.axhline(0, color=style.color("reference"), linewidth=style.THIN_LINE, alpha=style.TABLE_CURVE_ALPHA)
            ax.set_xscale("log"); ax.set_xlabel("Peak learning rate"); ax.set_ylabel(label)
            ax.set_title(style.model_label(base))
        for ax in list(axes.flat)[len(bases):]:
            ax.remove()
        fig.legend(handles=[Line2D([], [], color=colors[i], linestyle=":" if c == "Interaction" else style.FACTOR_ARM_STYLES[i], label=c)
                    for i, c in enumerate(conditions)], loc="outside lower center", ncol=3)
        pages.append(cp.Page(f"factorial_e{experiment}_{domain}_{metric}", fig,
            f"{track.upper()}. {title} in the shared factorial, evaluated on {population} tables. "
            "Every contrast pairs the same table, base, rate and sampling recipe and requires all four full/frozen × unanchored/anchored cells. "
            "The dotted interaction is (anchored minus unanchored under freezing) minus the same contrast under full updates. "
            "Five row folds are averaged within table before equal-table means; table cohorts stay fixed across learning rates. "
            "Positive simple effects favor the changed choice; a positive interaction means anchoring helps more under freezing, not necessarily positive transfer. "
            "These are two analyses of the same 512 unique training runs; no extra independent replications are implied.",
            {"Matched table contrasts": part, "Equal-table means": means, "Four-cell recipe values": grid[grid.metric.eq(metric) & grid.domain.eq(domain)]},
            {"study_numbering": "v2", "shared_campaign": "cpt_main_v5", "experiment": experiment}))
    return pages


def plot_learning_rates(grid, track):
    data = grid[~grid.frozen & np.isclose(grid.l2sp_lambda, .003)] if not grid.empty else grid
    if data.empty:
        return []
    pages = []
    for metric in (("roc_auc", "brier_score") if track == "pd" else ("r2", "rmse")):
        part = data[data.domain.eq("credit") & data.metric.eq(metric)]
        if part.empty:
            continue
        rates = sorted(data.learning_rate.unique())
        support = part.groupby(["base", "dataset"]).learning_rate.nunique()
        complete = support[support.eq(len(rates))].reset_index()[["base", "dataset"]]
        part = part.merge(complete, on=["base", "dataset"], validate="many_to_one")
        if part.empty:
            continue
        summary = ce.recipe_summary(part)
        fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
        fig.suptitle(f"{track.upper()}: learning rate and transfer to held-out credit tables")
        scale, label = ce.METRICS[metric]
        bases = [b for b in ("v2", "v2.6", "v3", "tabicl-v2") if b in set(part.base)]
        for ax, base in zip(axes.flat, bases):
            curve = summary[summary.base.eq(base)].sort_values("learning_rate")
            color = style.color(base)
            ax.fill_between(curve.learning_rate, curve.q25*scale, curve.q75*scale, color=color, alpha=style.INTERVAL_ALPHA)
            ax.plot(curve.learning_rate, curve["mean"]*scale, color=color, linewidth=style.THIN_LINE,
                    marker=style.PANEL_MARKERS[0], markersize=style.CURVE_MARKER_SIZE)
            ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
            ax.set_title(style.model_label(base)); ax.set_xscale("log")
            ax.set_xlabel("Peak learning rate"); ax.set_ylabel(label)
        for ax in list(axes.flat)[len(bases):]:
            ax.remove()
        pages.append(cp.Page(f"learning_rate_credit_{metric}", fig,
            f"{track.upper()}. {label} after 10,000 successful updates, relative to the same original model, on held-out credit tables. "
            "All parameters are updated with L2-SP 0.003. Lines show equal-table means; shading is the interquartile range of table effects, not confidence intervals. "
            "All learning rates use the same complete table cohort within each base. Positive values favor CPT. "
            "Experiments 3 and 4 retain the remaining cells of the shared 512-run factorial and its anchoring/freezing interaction.",
            {"Paired table effects": part, "Mean and table spread": summary}, {"study_numbering": "v2"}))
    return pages


def plot_factor_training(run, experiment):
    from src.visualize import training_summary as ts
    factor = {3: "l2sp_lambda", 4: "frozen"}[experiment]
    summaries, tables = ts.monitor_choice_tables(run)
    if summaries.empty:
        return []
    pages = []
    for domain in ("credit", "retention"):
        data = summaries[summaries.factor.eq(factor) & summaries.domain.eq(domain)]
        if data.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
        label = "L2-SP" if experiment == 3 else "backbone freezing"
        fig.suptitle(f"{run.track.upper()}: {label} and {domain} monitor dynamics")
        bases = [b for b in ("v2", "v2.6", "v3", "tabicl-v2") if b in set(data.base)]
        names = ("L2-SP off", "L2-SP 0.003") if experiment == 3 else ("All parameters", "Frozen backbone")
        fields = [("reference_mean", names[0], "--"), ("alternative_mean", names[1], "-"),
                  ("difference", "Paired difference", ":")]
        for ax, base in zip(axes.flat, bases):
            group = data[data.base.eq(base)].sort_values("updates")
            for column, name, line in fields:
                ax.plot(group.updates, group[column], color=style.color(base), linestyle=line,
                        linewidth=style.THIN_LINE, label=name)
            ax.axhline(0, color=style.color("reference"), linewidth=style.THIN_LINE, alpha=style.TABLE_CURVE_ALPHA)
            ax.set_title(style.model_label(base)); ax.set_xlabel("Successful updates")
            ax.set_ylabel("Monitor ΔAUC" if run.track == "pd" else "Monitor ΔR²")
        for ax in list(axes.flat)[len(bases):]:
            ax.remove()
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
        pages.append(cp.Page(f"e{experiment}_monitor_{domain}", fig,
            f"{run.track.upper()}. Fixed-context {domain} monitoring trajectories for {label}, matched within the remaining factorial settings. "
            f"{names[0]} and {names[1]} lines show improvement from update zero; the dotted line is the latter minus the former. "
            "Recipe pairs are averaged within table, then tables equally; only complete planned pairs contribute. "
            "These trajectories reuse the original factorial training records and do not replace the five-fold final benchmark. "
            "The results notebook additionally stratifies by the other factor and reports its interaction.",
            {"Monitor means and paired differences": data, "Endpoint matched tables": tables[tables.factor.eq(factor) & tables.domain.eq(domain)] if not tables.empty else tables},
            {"study_numbering": "v2"}))
    return pages
