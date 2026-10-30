"""Compact RQ reports built from immutable campaign evidence.

Presentation groups are independent of experiment storage and training identity.
Tables remain the unit of benchmark averaging; shared controls enter only once.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.visualize import campaign as cp, dashboards as db, research_views as rv
from src.visualize import synthesis as sy, style, training_summary as ts

CAMPAIGNS = {1: (1, 2), 2: (1, 5, 8), 3: (3, 6, 7)}
TITLES = {1: "Transfer and training-seed sensitivity", 2: "Restrictions on parameter updates",
          3: "Training data and exposure", 4: "Labeled target context"}


def load(question, track):
    return sy.load_studies(track, campaigns=CAMPAIGNS[question])


def training_studies(studies, question):
    return [(e, arm, rv.learning_rate_view(run) if e == 1 and question == 1 else run)
            for e, arm, run in studies]


def catalog(studies, question, metric, domain="credit", *, reference_only=False):
    data = sy.catalog(studies, metric, domain=domain)
    if data.empty:
        return data
    if question == 1:
        data = data[data.experiment.ne(1) | (~data.frozen & np.isclose(data.l2sp_lambda, .003))]
    if reference_only:
        data = data[data.experiment.ne(1) | data.reference]
    return data.copy()


def coverage(studies):
    frames = []
    for e, arm, run in studies:
        frame = run.trials.copy()
        frame["track"], frame["campaign"], frame["arm"] = run.track, str(run.cfg.run_name), arm or ""
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    # An explicitly reused control is not a new trial.
    return pd.concat(frames, ignore_index=True).drop_duplicates(["track", "trial_name"])


def fixed_cohort(data):
    """Require the same tables for every displayed arm within each model."""
    if data.empty:
        return data.copy()
    if data.duplicated(["base", "config", "dataset"]).any():
        raise ValueError("Duplicate configuration/table in an RQ comparison")
    frames = []
    for _, part in data.groupby("base", sort=False):
        counts = part.groupby("dataset").config.nunique()
        eligible = counts[counts.eq(part.config.nunique())].index
        frames.append(part[part.dataset.isin(eligible)])
    return pd.concat(frames, ignore_index=True)


def _grid(title, bases):
    count = len(bases)
    rows, cols = (2, 2) if count > 2 else (1, count)
    fig, axes = plt.subplots(rows, cols, squeeze=False, layout="constrained",
                            figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO if rows > 1 else style.SMALL_RATIO))
    fig.suptitle(title)
    axes = list(axes.flat)
    for ax in axes[count:]:
        ax.remove()
    return fig, axes[:count]


def _bases(data):
    return [b for b in sy.BASES if b in set(data.base)]


def recipe_label(row):
    """Compact display text only; preserve the complete identity in report tables."""
    if row.experiment == 1:
        return f"E1 · LR {row.learning_rate:.0e}"
    if row.experiment == 2:
        return "E2 · reference, seed 43"
    return _label(row.experiment, row.arm, row)


def plot_effects(data, track, *, title, name, contrast=False, min_arms=1):
    """One metric, all models/arms: table points, IQR and an equal-table mean."""
    if data.empty:
        return []
    metric = data.metric.iloc[0]
    if data.metric.nunique() != 1:
        raise ValueError("Do not pool different metrics in an RQ effect panel")
    matched = fixed_cohort(data)
    available = matched.groupby('base').config.nunique() if not matched.empty else pd.Series(dtype=int)
    matched = matched[matched.base.isin(available[available.ge(min_arms)].index)] if not matched.empty else matched
    if matched.empty:
        return []
    bases = _bases(matched)
    order = list(dict.fromkeys(data.config)) if contrast else sy.config_order(data)
    order = [x for x in order if x in set(matched.config)]
    labels = dict(zip(data.config, data.config)) if contrast else {
        row.config: recipe_label(row) for row in data.drop_duplicates('config').itertuples()}
    if len(order) == 1:
        fig, axes_array = plt.subplots(1, len(bases), squeeze=False, layout="constrained",
                            figsize=style.figsize(style.WIDTH_FULL, style.CHOICE_LEGEND_RATIO))
        axes = list(axes_array.flat)
        fig.suptitle(f"{track.upper()}: {title}")
    else:
        fig, axes = _grid(f"{track.upper()}: {title}", bases)
    means = []
    for panel, (ax, base) in enumerate(zip(axes, bases)):
        local_order = [c for c in order if c in set(matched[matched.base.eq(base)].config)]
        for row, label in enumerate(local_order):
            part = matched[matched.base.eq(base) & matched.config.eq(label)]
            if part.empty:
                continue
            values = part.effect.to_numpy(float) * sy.SCALE[metric]
            offset = np.linspace(-style.PROTOCOL_POINT_OFFSET, style.PROTOCOL_POINT_OFFSET, len(values))
            ax.scatter(values, row + offset, color=style.color(base), alpha=style.POINT_ALPHA,
                       s=style.TABLE_POINT_SIZE, linewidths=0)
            low, high = np.quantile(values, [.25, .75])
            ax.plot([low, high], [row, row], color=style.color(base), linewidth=style.MEAN_CURVE_LINE)
            ax.plot(np.mean(values), row, marker="D", color=style.color(base), markersize=style.CURVE_MARKER_SIZE)
            means.append(dict(base=base, config=label, tables=len(values), mean=part.effect.mean(),
                              q25=part.effect.quantile(.25), q75=part.effect.quantile(.75)))
        ax.axvline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_title(style.model_label(base)); ax.set_xlabel(sy.SHORT_UNITS[metric])
        previous_order = ([c for c in order if c in set(matched[matched.base.eq(bases[panel - 1])].config)]
                          if panel else [])
        show_labels = panel == 0 or (len(order) > 1 and panel == 2) or local_order != previous_order
        ax.set_yticks(range(len(local_order)), [labels[c] for c in local_order] if show_labels else
                      [''] * len(local_order), fontsize=style.GRID_FONT)
        ax.set_ylim(len(local_order) - .5, -.5)
        from matplotlib.ticker import ScalarFormatter, MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=style.CONTINUOUS_TICKS - 1))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 3))
        ax.xaxis.set_major_formatter(formatter)
    fig.legend(handles=[Line2D([], [], color=style.color("reference"), marker="D", label="Mean and table IQR"),
                        Line2D([], [], color=style.color("annotation"), marker="o", linestyle="", label="Individual table")],
               loc="outside lower center", ncol=2)
    caption = (f"{track.upper()}. {title}. " +
               ("Paired differences between the named choices' CPT effects, on identical tables. Positive favors the alternative choice and can mean less degradation, rather than positive transfer. " if contrast else
                "Each CPT recipe is compared with its own original model; positive values favor CPT. ") +
               sy.NOTE_PAIRING + "Each model uses the same table cohort across all displayed arms; models can have different cohorts. "
               "Dots are individual tables, diamonds equal-table means and segments table interquartile ranges, not confidence intervals. "
               "Axes retain all finite values and can differ between models. Missing recipes are reported in coverage, not assigned zero. "
               + ("Bases without an available adapter comparison are omitted from this panel. " if min_arms > 1 else "")
               + ("Both choices' effects and their named-table differences are retained in the text summary." if contrast else
                  "Original scores, adapted scores and named-table effects are retained in the text summary."))
    return [cp.Page(name, fig, caption, {"Matched named-table values": matched,
                    "Equal-table means and heterogeneity": pd.DataFrame(means)}, {"study_numbering": "v2"})]


def choice_differences(data, question):
    """Only planned, interpretable contrasts; never a DiagScale-versus-LoRA claim."""
    if data.empty:
        return data.copy()
    if question == 1:
        pairs = [("E2 seed 43 − seed 42", data.reference, data.experiment.eq(2))]
    elif question == 3:
        sample = data.experiment.eq(3)
        large = data.experiment.eq(6) & data.arm.eq("Large tables")
        pairs = [
            ("E7 full pass − one sample", sample & data.sampling.eq("one_sample"), sample & data.sampling.eq("full_pass")),
            ("E7 accumulation − one sample", sample & data.sampling.eq("one_sample"), sample & data.sampling.eq("accumulate")),
            ("E8 large − all tables", data.experiment.eq(6) & data.arm.eq("All tables"), large),
            ("E9 2,048 − 8,192 rows", large, data.experiment.eq(7) & data.arm.eq("2,048 rows")),
            ("E9 upper row cap − 8,192", large, data.experiment.eq(7) & ~data.arm.eq("2,048 rows")),
        ]
    else:
        raise ValueError("No recipe-mechanism contrast is identified for RQ2")
    frames = []
    keys = ["base", "dataset", "metric", "domain"]
    for label, ref, alt in pairs:
        left, right = data[ref], data[alt]
        if left.empty or right.empty:
            continue
        joined = left[keys + ["effect"]].merge(right[keys + ["effect"]], on=keys,
                    suffixes=("_reference", "_alternative"), validate="one_to_one")
        joined["effect"] = joined.effect_alternative - joined.effect_reference
        joined["config"] = label
        frames.append(joined)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _label(experiment, arm, row):
    if experiment == 1:
        return f"E1 · LR {row.learning_rate:g}"
    if experiment == 2:
        return "E2 · seed 43"
    if experiment == 3:
        return "E7 · " + style.SAMPLING_LABELS[row.sampling]
    if experiment in (5, 8):
        return f"{'E5 DiagScale' if experiment == 5 else 'E6 LoRA'} · LR {row.learning_rate:g}"
    if experiment == 6:
        return "E8/E9 · large, 8,192 rows" if arm == "Large tables" else "E8 · all tables"
    return "E9 · " + str(arm)


def training_curves(studies, quantity, *, row_exposure=False):
    """Within-model recipe means with complete planned trial/table support."""
    frames = []
    for experiment, arm, run in studies:
        raw = cp.optimization_curves(run, "train_loss") if quantity == "loss" else db.monitor_observations(run, "ood" if quantity == "retention" else "test")
        if raw.empty:
            continue
        for key, part in raw.groupby(cp.FACTORS, sort=True, dropna=False):
            meta = dict(zip(cp.FACTORS, key))
            row = next(part.itertuples(index=False))
            if quantity == "loss":
                planned = run.trials.merge(pd.DataFrame([meta]), on=cp.FACTORS, validate="many_to_one")
                curve = part.groupby("window_end").train_loss.agg(mean="mean", observations="count").reindex(cp.optimization_grid(run, "train_loss"))
                curve["complete"] = curve.observations.eq(len(planned))
                curve.loc[~curve.complete, "mean"] = np.nan
                curve = curve.rename_axis("updates").reset_index()
                curve["x"] = curve.updates
            else:
                curve = cp.trajectory_curve(run, part, split="ood" if quantity == "retention" else "test",
                                             x="processed_rows" if row_exposure else "updates")
            frames.append(curve.assign(base=meta["base"], config=_label(experiment, arm, row),
                                       quantity=quantity, track=run.track))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty and result.duplicated(["base", "config", "updates"]).any():
        raise ValueError("A shared training curve was counted twice")
    return result


def plot_training(studies, quantity, *, title, name, row_exposure=False):
    data = training_curves(studies, quantity, row_exposure=row_exposure)
    if data.empty or not data["mean"].notna().any():
        return []
    track = data.track.iloc[0]
    bases = _bases(data)
    fig, axes = _grid(f"{track.upper()}: {title}", bases)
    labels = list(dict.fromkeys(data.config))
    colors = style.categorical(labels)
    for ax, base in zip(axes, bases):
        for label, part in data[data.base.eq(base)].groupby("config", sort=False):
            part = part.sort_values("updates")
            ax.plot(part.x, part["mean"], color=colors[label], linewidth=style.THIN_LINE,
                    linestyle="--" if "LoRA" in label or "seed 43" in label else "-")
        if quantity != "loss":
            ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_title(style.model_label(base))
        ax.set_xlabel("Processed row exposures" if row_exposure else "Successful updates")
        ax.set_ylabel(cp.loss_label(track, base) if quantity == "loss" else "Monitor ΔAUC" if track == "pd" else "Monitor ΔR²")
    fig.legend(handles=[Line2D([], [], color=colors[label], linestyle="--" if "LoRA" in label or "seed 43" in label else "-", label=label)
                        for label in labels], loc="outside lower center", ncol=min(3, len(labels)))
    return [cp.Page(name, fig, f"{track.upper()}. {title}. " +
        ("Native query objectives are averaged in successful-update windows, giving each planned trial one contribution; missing trial support remains a gap. Native LGD objectives differ across architectures and are not compared numerically. " if quantity == "loss" else
         f"Fixed-context {'non-credit retention' if quantity == 'retention' else 'held-out credit'} monitor changes from update zero, averaged across the same complete table/trial cohort at each milestone; these are not five-fold final benchmark results. ") +
        "The four training partitions are represented, and shared controls appear once. Colors identify the stated recipes. "
        "Equal updates do not imply equal rows or compute; row exposure includes repeated observations. Adapter curves compare specified recipes, not isolated mechanisms.",
        {"Displayed curves and completeness": data}, {"study_numbering": "v2"})]


def plot_factor_training(run):
    data, endpoints = ts.monitor_choice_tables(run)
    if data.empty:
        return []
    data = data[data.factor.isin(["l2sp_lambda", "frozen"])]
    fig, axes = plt.subplots(2, 2, layout="constrained", figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO))
    fig.suptitle(f"{run.track.upper()}: anchoring and freezing during training")
    for r, domain in enumerate(("credit", "retention")):
        for c, factor in enumerate(("l2sp_lambda", "frozen")):
            ax = axes[r, c]
            part = data[data.domain.eq(domain) & data.factor.eq(factor)]
            for base, curve in part.groupby("base"):
                curve = curve.sort_values("updates")
                ax.plot(curve.updates, curve.difference, color=style.color(base), linewidth=style.THIN_LINE)
            ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
            ax.set_title(("Add L2-SP" if c == 0 else "Freeze backbone") + " · " + domain)
            ax.set_xlabel("Successful updates"); ax.set_ylabel("Change in monitor ΔAUC" if run.track == "pd" else "Change in monitor ΔR²")
    fig.legend(handles=[Line2D([], [], color=style.color(b), label=style.model_label(b)) for b in _bases(data)],
               loc="outside lower center", ncol=4)
    return [cp.Page("factor_training", fig,
        "Matched changes in fixed-context credit and non-credit monitoring when adding L2-SP or freezing the backbone. "
        "Each curve is alternative minus reference after matching the remaining settings; recipes are averaged within table, then tables equally. "
        "Missing planned support stays missing. Original reference and alternative trajectories are retained numerically. "
        "Marginal effects share the same factorial; the final benchmark separately retains all four cells and their interaction.",
        {"Choice trajectories, reference and alternative": data, "Matched endpoint tables": endpoints}, {"study_numbering": "v2"})]


def plot_factorial(grid, track, domain="credit"):
    """All four factorial cells on one page per metric, rather than two repeated studies."""
    pages = []
    if grid.empty:
        return pages
    cells = [(False, 0.), (False, .003), (True, 0.), (True, .003)]
    for metric in (sy.PRIMARY[track], sy.SECONDARY[track]):
        part = grid[grid.metric.eq(metric) & grid.domain.eq(domain)].copy()
        if part.empty:
            continue
        keys = ["base", "dataset", "learning_rate", "sampling"]
        wide = part.pivot(index=keys, columns=["frozen", "l2sp_lambda"], values="effect")
        if not all(cell in wide for cell in cells):
            continue
        eligible = wide[cells].dropna().index.to_frame(index=False)
        part = part.merge(eligible, on=keys, validate="many_to_one")
        rates = grid.learning_rate.nunique()
        cohort = part.groupby(["base", "dataset"]).size()
        cohort = cohort[cohort.eq(4 * rates)].reset_index()[["base", "dataset"]]
        part = part.merge(cohort, on=["base", "dataset"], validate="many_to_one")
        if part.empty:
            continue
        means = part.groupby(["base", "learning_rate", "frozen", "l2sp_lambda"]).effect.mean().reset_index()
        population = 'held-out credit' if domain == 'credit' else 'non-credit retention'
        fig, axes = _grid(f"{track.upper()}: anchoring × freezing · {population}", _bases(part))
        for ax, base in zip(axes, _bases(part)):
            for frozen, lam in cells:
                line = means[means.base.eq(base) & means.frozen.eq(frozen) & means.l2sp_lambda.eq(lam)].sort_values("learning_rate")
                ax.plot(line.learning_rate, line.effect * sy.SCALE[metric],
                        color=style.color("frozen" if frozen else "full"), linestyle=style.ANCHOR_STYLES[lam > 0],
                        marker=style.PANEL_MARKERS[int(frozen)], markersize=style.CURVE_MARKER_SIZE, linewidth=style.THIN_LINE)
            ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
            ax.set_xscale("log"); ax.set_title(style.model_label(base)); ax.set_xlabel("Peak learning rate"); ax.set_ylabel(sy.SHORT_UNITS[metric])
        fig.legend(handles=[Line2D([], [], color=style.color("frozen" if f else "full"), linestyle=style.ANCHOR_STYLES[l > 0],
                    label=("Frozen" if f else "Full") + (" · L2-SP" if l else " · no L2-SP")) for f, l in cells],
                   loc="outside lower center", ncol=2)
        pages.append(cp.Page(f"factorial_{domain}_{metric}", fig,
            f"{track.upper()}. Complete full/frozen × unanchored/anchored factorial on {'held-out credit' if domain == 'credit' else 'non-credit retention'} tables. "
            "All four cells and all learning rates use a fixed complete table cohort within each model. Positive values favor CPT relative to its original checkpoint. "
            "Five folds are averaged within table; retention first averages the four training partitions, then tables receive equal weight. "
            "The full and frozen anchoring effects, both freezing effects and their difference-in-differences interaction are retained in the summary. "
            "E1/E3/E4 reuse one 512-run campaign, not independent repetitions.",
            {"All four paired cell values": part, "Equal-table means": means,
             "Anchoring simple effects and interaction": rv.factorial_contrasts(part, 3),
             "Freezing simple effects and interaction": rv.factorial_contrasts(part, 4)}, {"study_numbering": "v2"}))
    return pages
