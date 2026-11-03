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


def _bases(data):
    return [b for b in sy.BASES if b in set(data.base)]


def series(experiment, arm, row):
    """One label, colour, line style and order per recipe, shared by every RQ figure.

    Ordinal settings (learning rate, episode size) use ordinal colour ramps; the line
    style separates the alternative of a contrast (seed 43, LoRA, all-table corpus).
    """
    lr = float(getattr(row, "learning_rate", np.nan))
    if experiment == 1:
        reference = np.isclose(lr, sy.REFERENCE["learning_rate"])
        return dict(label=f"E1 · LR {style.lr_text(lr)}" + (" (reference)" if reference else ""),
                    color=style.LR_COLORS.get(lr, style.color("reference")), linestyle="-", order=(1, lr))
    if experiment == 2:
        return dict(label="E2 · reference, seed 43", color=style.LR_COLORS[sy.REFERENCE["learning_rate"]],
                    linestyle="--", order=(2, lr))
    if experiment == 3:
        key = row.sampling
        return dict(label="E7 · " + style.SAMPLING_LABELS[key], color=style.SAMPLING_COLORS[key],
                    linestyle="-", order=(7, list(style.SAMPLING_LABELS).index(key)))
    if experiment == 5:
        return dict(label=f"E5 DiagScale · LR {style.lr_text(lr)}",
                    color=style.DIAGSCALE_COLORS.get(lr, style.APPROACH_COLORS["DiagScale"]), linestyle="-", order=(5, lr))
    if experiment == 8:
        return dict(label=f"E6 LoRA · LR {style.lr_text(lr)}",
                    color=style.LORA_COLORS.get(lr, style.APPROACH_COLORS["LoRA"]), linestyle="--", order=(6, lr))
    if experiment == 6:
        large = arm == "Large tables"
        return dict(label="E8/E9 · large tables, 8,192 rows" if large else "E8 · all tables, 8,192 rows",
                    color=style.EPISODE_COLORS["8,192 rows"], linestyle="-" if large else "--", order=(8, int(large)))
    size = int(str(arm).split()[0].replace(",", ""))
    return dict(label=f"E9 · large tables, {arm}", color=style.EPISODE_COLORS.get(str(arm), style.color("highlight")),
                linestyle="-", order=(9, size))


def recipe_label(row):
    """Compact display text only; preserve the complete identity in report tables."""
    return series(row.experiment, row.arm, row)["label"]


QUANTITIES = {"loss": "Query loss", "credit": "Held-out credit", "retention": "Non-credit retention"}
DOMAINS = {"credit": "Held-out credit", "ood": "Non-credit retention"}
METRIC_NAMES = {"roc_auc": "AUC", "brier_score": "Brier score", "r2": "R²", "rmse": "RMSE"}
EFFECT_AXES = {"roc_auc": "ΔAUC (pp)", "brier_score": "Brier reduction (×100)",
               "r2": "ΔR²", "rmse": "RMSE reduction (%)"}


def _monitor_axis(track, quantity):
    return f"{QUANTITIES[quantity]}\nmonitor {'ΔAUC (pp)' if track == 'pd' else 'ΔR²'}"


def _zero(ax, vertical=False):
    (ax.axvline if vertical else ax.axhline)(0, color=style.color("reference"), linestyle=":",
                                               linewidth=style.THIN_LINE, zorder=1)


def _series_handles(data):
    looks = data.drop_duplicates("config")
    looks = looks.iloc[sorted(range(len(looks)), key=lambda i: tuple(looks.order.iloc[i]))]
    return [Line2D([], [], color=r.color, linestyle=r.linestyle, linewidth=style.MEAN_CURVE_LINE * .8, label=r.config)
            for r in looks.itertuples()]


def plot_training(studies, quantities=("loss", "credit", "retention"), *, title, name, row_exposure=False):
    """Recipe trajectories: rows are quantities, columns are models, colour/style are recipes."""
    quantities = (quantities,) if isinstance(quantities, str) else tuple(quantities)
    frames = [f for f in (training_curves(studies, q, row_exposure=row_exposure) for q in quantities) if not f.empty]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if data.empty or not data["mean"].notna().any():
        return []
    track = data.track.iloc[0]
    quantities = [q for q in quantities if data.quantity.eq(q).any()]
    bases = _bases(data)
    fig, axes = plt.subplots(len(quantities), len(bases), squeeze=False, sharex=True, layout="constrained",
                             figsize=style.rows_figsize(len(quantities)))
    fig.suptitle(title)
    scale = sy.SCALE[sy.PRIMARY[track]]
    for r, quantity in enumerate(quantities):
        first = None
        for c, base in enumerate(bases):
            ax = axes[r, c]
            part = data[data.base.eq(base) & data.quantity.eq(quantity)]
            for _, curve in part.groupby("config", sort=False):
                curve = curve.sort_values("x")
                look = curve.iloc[0]
                values = curve["mean"] * (1 if quantity == "loss" else scale)
                ax.plot(curve.x / (1e6 if row_exposure else 1), values, color=look.color, linestyle=look.linestyle, linewidth=style.THIN_LINE * 1.2)
            if quantity != "loss":
                _zero(ax)
            if row_exposure:
                ax.set_xscale("log")
            if r == 0:
                ax.set_title(style.model_label(base))
            if quantity == "loss":
                label = cp.loss_label(track, base)
            else:
                label = _monitor_axis(track, quantity)
            if c == 0 or label != first:
                ax.set_ylabel(label)
            if c == 0:
                first = label
            if r == len(quantities) - 1:
                ax.set_xlabel("Rows seen (millions)" if row_exposure else "Successful updates")
    handles = _series_handles(data)
    fig.legend(handles=handles, loc="outside lower center",
               ncol=style._legend_columns(fig, [handle.get_label() for handle in handles]))
    what = ", ".join(QUANTITIES[q].lower() for q in quantities)
    return [cp.Page(name, fig, f"{title}. Rows show {what}; columns show models. " +
        ("Query loss is the native training objective, averaged in successful-update windows with one contribution per planned trial; "
         "native LGD objectives differ across architectures and are compared only within a column. " if "loss" in quantities else "") +
        ("Monitor rows show changes from update zero on the fixed-context monitor, averaged over the same complete table and trial cohort "
         "at every milestone; they are training diagnostics, not five-fold benchmark results. Positive values favour CPT. "
         if set(quantities) - {"loss"} else "") +
        "The four training partitions contribute equally and shared controls appear once. Missing planned support is left as a gap. "
        + ("Rows seen count repeated observations, on a logarithmic axis; equal updates do not imply equal rows or compute." if row_exposure else
           "Equal update counts do not imply equal row exposure or compute."),
        {"Displayed curves and completeness": data}, {"study_numbering": "v2"})]


def _forest(ax, matched, order, base, scale):
    means = []
    for row, config in enumerate(order):
        part = matched[matched.config.eq(config)]
        if part.empty:
            continue
        values = part.effect.to_numpy(float) * scale
        offset = np.linspace(-style.PROTOCOL_POINT_OFFSET, style.PROTOCOL_POINT_OFFSET, len(values))
        ax.scatter(values, row + offset, color=style.color(base), alpha=style.POINT_ALPHA * .8,
                   s=style.TABLE_POINT_SIZE * .7, linewidths=0, zorder=2)
        low, high = np.quantile(values, [.25, .75])
        ax.plot([low, high], [row, row], color=style.color(base), linewidth=style.MEAN_CURVE_LINE * 1.3,
                solid_capstyle="butt", zorder=3)
        ax.plot(np.mean(values), row, marker="D", color="white", markeredgecolor=style.color("mean"),
                markeredgewidth=.8, markersize=3.6, zorder=4)
        means.append(dict(base=base, config=config, tables=len(values), mean=part.effect.mean(),
                          q25=part.effect.quantile(.25), q75=part.effect.quantile(.75)))
    return means


def plot_effects(panels, track, *, title, name, contrast=False, min_arms=1):
    """Paired effects for metric/population columns: rows are models, y ticks are recipes.

    `panels` is one catalog or a list of `(column heading, catalog)` pairs. Within each model
    and column, every displayed recipe shares one table cohort.
    """
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    if isinstance(panels, pd.DataFrame):
        panels = [(DOMAINS.get(panels.domain.iloc[0], "") if "domain" in panels and not panels.empty else "", panels)]
    prepared = []
    for heading, data in panels:
        if data.empty:
            continue
        if data.metric.nunique() != 1:
            raise ValueError("Do not pool different metrics in an RQ effect panel")
        matched = fixed_cohort(data)
        if matched.empty:
            continue
        available = matched.groupby("base").config.nunique()
        matched = matched[matched.base.isin(available[available.ge(min_arms)].index)]
        if not matched.empty:
            prepared.append((heading, data, matched))
    if not prepared:
        return []
    every = pd.concat([d for _, d, _ in prepared], ignore_index=True)
    order = list(dict.fromkeys(every.config)) if contrast else sy.config_order(every)
    shown = set().union(*(set(m.config) for _, _, m in prepared))
    order = [c for c in order if c in shown]
    labels = {c: c for c in order} if contrast else {
        r.config: recipe_label(r) for r in every.drop_duplicates("config").itertuples()}
    bases = [b for b in sy.BASES if any(b in set(m.base) for _, _, m in prepared)]
    per_base = {b: [c for c in order if any(c in set(m[m.base.eq(b)].config) for _, _, m in prepared)] for b in bases}
    # At least three recipe slots, so the rotated model label always fits its row.
    heights = [max(3, len(per_base[b])) for b in bases]
    total = sum(heights) * style.FOREST_ROW + .32 * len(bases)
    fig, axes = plt.subplots(len(bases), len(prepared), squeeze=False, sharey="row", layout="constrained",
                             gridspec_kw={"height_ratios": heights},
                             figsize=(style.WIDTH_FULL, min(style.FIGURE_BAND + total, style.MAX_HEIGHT)))
    fig.suptitle(title)
    means, tables = [], {}
    for c, (heading, data, matched) in enumerate(prepared):
        metric = data.metric.iloc[0]
        tables[f"{heading} {metric}: matched named-table values".strip()] = matched
        for r, base in enumerate(bases):
            ax = axes[r, c]
            local = per_base[base]
            means += [dict(m, column=heading, metric=metric) for m in
                      _forest(ax, matched[matched.base.eq(base)], local, base, sy.SCALE[metric])]
            _zero(ax, vertical=True)
            ax.set_ylim(len(local) - .5, -.5)
            ax.set_yticks(range(len(local)), [labels[x] for x in local])
            ax.tick_params(axis="y", length=0)
            ax.grid(False)
            if r == 0:
                ax.set_title((heading + "\n" if heading else "") + METRIC_NAMES[metric])
            if c == 0:
                ax.set_ylabel(style.model_label(base), color=style.color(base), fontweight="bold")
            if r == len(bases) - 1:
                ax.set_xlabel(("Difference in " if contrast else "") + EFFECT_AXES[metric])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_powerlimits((-3, 4))
            ax.xaxis.set_major_formatter(formatter)
    fig.legend(handles=[
        Line2D([], [], color=style.color("annotation"), marker="o", linestyle="", markersize=3, alpha=.8, label="Held-out table"),
        Line2D([], [], color=style.color("annotation"), linewidth=style.MEAN_CURVE_LINE * 1.3, label="Interquartile range over tables"),
        Line2D([], [], color="white", marker="D", markeredgecolor=style.color("mean"), linestyle="", label="Equal-table mean")],
        loc="outside lower center", ncol=3)
    caption = (f"{title}. " +
        ("Each value is the difference between two recipes' CPT-minus-original effects on the same table and model; "
         "positive favours the recipe named first and can mean less degradation rather than positive transfer. " if contrast else
         "Each CPT recipe is compared with its own original model; positive values favour CPT. ") +
        sy.NOTE_PAIRING +
        "Within a model and column, every displayed recipe uses the same tables; cohorts can differ between models and columns. "
        "Retention values first average the four training partitions. "
        "Horizontal axes are scaled per panel. Absent rows were not run for that model; they are not zero effects."
        + (" Models without at least two comparable recipes are omitted." if min_arms > 1 else ""))
    tables["Equal-table means and heterogeneity"] = pd.DataFrame(means)
    return [cp.Page(name, fig, caption, tables, {"study_numbering": "v2"})]


def effect_columns(catalogs, track, *, retention=True, secondary=True):
    """Standard RQ effect columns: credit primary and secondary metric, retention primary metric."""
    columns = [("Held-out credit", catalogs["credit", sy.PRIMARY[track]])]
    if secondary:
        columns.append(("Held-out credit", catalogs["credit", sy.SECONDARY[track]]))
    if retention:
        columns.append(("Non-credit retention", catalogs["ood", sy.PRIMARY[track]]))
    return columns


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
    return series(experiment, arm, row)["label"]


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
            look = series(experiment, arm, row)
            frames.append(curve.assign(base=meta["base"], config=look["label"], order=[look["order"]] * len(curve),
                                       color=look["color"], linestyle=look["linestyle"],
                                       experiment=int(experiment), quantity=quantity, track=run.track))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty and result.duplicated(["base", "config", "updates"]).any():
        raise ValueError("A shared training curve was counted twice")
    return result


def plot_factor_training(run):
    data, endpoints = ts.monitor_choice_tables(run)
    if data.empty:
        return []
    data = data[data.factor.isin(["l2sp_lambda", "frozen"])]
    scale = sy.SCALE[sy.PRIMARY[run.track]]
    unit = "ΔAUC (pp)" if run.track == "pd" else "ΔR²"
    fig, axes = plt.subplots(2, 2, sharex=True, layout="constrained", figsize=style.rows_figsize(2, row_height=1.45))
    fig.suptitle("Matched effect of anchoring and freezing during training")
    for r, domain in enumerate(("credit", "retention")):
        for c, factor in enumerate(("l2sp_lambda", "frozen")):
            ax = axes[r, c]
            part = data[data.domain.eq(domain) & data.factor.eq(factor)]
            for base in _bases(part):
                curve = part[part.base.eq(base)].sort_values("updates")
                ax.plot(curve.updates, curve.difference * scale, color=style.color(base), linewidth=style.THIN_LINE * 1.2)
            _zero(ax)
            if r == 0:
                ax.set_title("E3 · add L2-SP (λ 0 → 0.003)" if c == 0 else "E4 · freeze the backbone")
            if c == 0:
                ax.set_ylabel(f"{QUANTITIES[domain]}\nalternative − reference\n{unit}")
            if r == 1:
                ax.set_xlabel("Successful updates")
    fig.legend(handles=[Line2D([], [], color=style.color(b), label=style.model_label(b)) for b in _bases(data)],
               loc="outside lower center", ncol=4)
    return [cp.Page("factor_training", fig,
        "Monitor change of the alternative choice minus its reference after matching the learning rate and the other factor, "
        "on held-out credit (top) and non-credit retention (bottom); positive values mean the alternative transfers or retains more. "
        "Recipes are averaged within table, then tables equally; missing planned support stays missing. "
        "Both factors come from the same 512-run factorial; its four cells and their interaction are reported with the final benchmark.",
        {"Choice trajectories, reference and alternative": data, "Matched endpoint tables": endpoints}, {"study_numbering": "v2"})]


FACTORIAL_CELLS = [(False, 0.), (False, .003), (True, 0.), (True, .003)]


def _factorial_part(grid, metric, domain):
    part = grid[grid.metric.eq(metric) & grid.domain.eq(domain)].copy()
    if part.empty:
        return part
    keys = ["base", "dataset", "learning_rate", "sampling"]
    wide = part.pivot(index=keys, columns=["frozen", "l2sp_lambda"], values="effect")
    if not all(cell in wide for cell in FACTORIAL_CELLS):
        return part.iloc[:0]
    eligible = wide[FACTORIAL_CELLS].dropna().index.to_frame(index=False)
    part = part.merge(eligible, on=keys, validate="many_to_one")
    rates = grid.learning_rate.nunique()
    cohort = part.groupby(["base", "dataset"]).size()
    cohort = cohort[cohort.eq(4 * rates)].reset_index()[["base", "dataset"]]
    return part.merge(cohort, on=["base", "dataset"], validate="many_to_one")


def plot_factorial(grid, track, domain=None):
    """The complete full/frozen × λ factorial: rows are outcomes, columns are models."""
    if grid.empty:
        return []
    specs = [("credit", sy.PRIMARY[track]), ("credit", sy.SECONDARY[track]), ("ood", sy.PRIMARY[track])]
    if domain is not None:
        specs = [s for s in specs if s[0] == domain]
    parts = [(d, m, _factorial_part(grid, m, d)) for d, m in specs]
    parts = [p for p in parts if not p[2].empty]
    if not parts:
        return []
    bases = [b for b in sy.BASES if any(b in set(p.base) for _, _, p in parts)]
    fig, axes = plt.subplots(len(parts), len(bases), squeeze=False, sharex=True, layout="constrained",
                             figsize=style.rows_figsize(len(parts)))
    fig.suptitle("Anchoring × freezing across learning rates")
    tables = {}
    for r, (dom, metric, part) in enumerate(parts):
        means = part.groupby(["base", "learning_rate", "frozen", "l2sp_lambda"]).effect.mean().reset_index()
        prefix = f"{DOMAINS[dom]} {METRIC_NAMES[metric]}: "
        tables[prefix + "all four paired cell values"] = part
        tables[prefix + "equal-table means"] = means
        tables[prefix + "anchoring simple effects and interaction"] = rv.factorial_contrasts(part, 3)
        tables[prefix + "freezing simple effects and interaction"] = rv.factorial_contrasts(part, 4)
        for c, base in enumerate(bases):
            ax = axes[r, c]
            for frozen, lam in FACTORIAL_CELLS:
                line = means[means.base.eq(base) & means.frozen.eq(frozen) & means.l2sp_lambda.eq(lam)].sort_values("learning_rate")
                ax.plot(line.learning_rate, line.effect * sy.SCALE[metric],
                        color=style.color("frozen" if frozen else "full"), linestyle=style.ANCHOR_STYLES[lam > 0],
                        marker=style.PANEL_MARKERS[int(frozen)], markersize=style.CURVE_MARKER_SIZE * 1.3,
                        linewidth=style.THIN_LINE * 1.2)
            _zero(ax)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(style.model_label(base))
            if c == 0:
                ax.set_ylabel(f"{DOMAINS[dom]}\n{EFFECT_AXES[metric]}")
            if r == len(parts) - 1:
                ax.set_xlabel("Peak learning rate")
    fig.legend(handles=[Line2D([], [], color=style.color("frozen" if f else "full"), linestyle=style.ANCHOR_STYLES[l > 0],
                marker=style.PANEL_MARKERS[int(f)], markersize=style.CURVE_MARKER_SIZE * 1.3,
                label=("Backbone frozen" if f else "All parameters") + (", L2-SP λ 0.003" if l else ", no L2-SP"))
                for f, l in FACTORIAL_CELLS], loc="outside lower center", ncol=2)
    first = parts[0][2]
    tables["All four paired cell values"] = first
    tables["Anchoring simple effects and interaction"] = rv.factorial_contrasts(first, 3)
    tables["Freezing simple effects and interaction"] = rv.factorial_contrasts(first, 4)
    return [cp.Page("factorial" if domain is None else f"factorial_{domain}", fig,
        "Equal-table mean CPT-minus-original effect of each cell of the full/frozen × unanchored/anchored factorial "
        "(E3 anchoring, E4 freezing) at every peak learning rate; positive values favour CPT. "
        "Within a model and row, all four cells and all learning rates use one complete table cohort. "
        "Five folds are averaged within table; retention first averages the four training partitions. "
        "Simple effects and the difference-in-differences interaction are retained in the text summary. "
        "E1, E3 and E4 reuse one 512-run campaign; they are not independent repetitions.",
        tables, {"study_numbering": "v2"})]
