"""Cross-experiment synthesis: every CPT arm against its own original model.

The overview notebooks and the experiment-1 table views share these quantities, so a
number printed in one place is computed by the same pairing everywhere:

1. Pair each adapted model with the original checkpoint of the same base on the same
   evaluation split and outer fold (five complete folds on both sides, or the table is
   absent rather than estimated from fewer folds).
2. Average the five fold effects within table. For non-credit retention tables, first
   require and average all four CPT training partitions.
3. Weight tables equally across a cohort; never pool PD and LGD metrics.

Each study keeps its own original-model control and inference settings. Comparing arms
from different studies therefore compares recipes that can differ in more than the
named factor; ``comparability_table`` records those differences explicitly.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.visualize import campaign as cp, eval_viz, style, training_viz
from src.visualize.paper_figures import _effect

PRIMARY = {"pd": "roc_auc", "lgd": "r2"}
SECONDARY = {"pd": "brier_score", "lgd": "rmse"}
SCALE = {"roc_auc": 100., "r2": 1., "rmse": 100., "brier_score": 100.}
UNITS = {"roc_auc": "ΔAUC (percentage points)", "r2": "ΔR²",
         "rmse": "RMSE reduction (%)", "brier_score": "Brier reduction (×100)"}
SHORT_UNITS = {"roc_auc": "ΔAUC (pp)", "r2": "ΔR²", "rmse": "RMSE red. (%)", "brier_score": "Brier red. (×100)"}
BASES = ["v2", "v2.6", "v3", "tabicl-v2"]
APPROACHES = list(style.APPROACH_COLORS)
STUDIES = {1: "Main grid", 2: "Seed 43", 3: "Sampling", 5: "DiagScale", 6: "Corpus policy",
           7: "Episode size", 8: "LoRA"}
REFERENCE = dict(learning_rate=3e-7, l2sp_lambda=0.003)
NOTE_PAIRING = ("Each adapted model is paired with its own original checkpoint on the same split and "
                "outer fold; five folds are averaged within table and tables receive equal weight. ")


# --------------------------------------------------------------------------- #
# Paired table effects
# --------------------------------------------------------------------------- #

def fold_pairs(run, metric, *, domain="credit"):
    """One row per adapted method × table × split × fold, with its original score."""
    data = run.evaluation
    if data.empty or metric not in data:
        return pd.DataFrame()
    labels = (data.domain.fillna("credit").replace({"ood": "noncredit"})
              if "domain" in data else pd.Series("credit", index=data.index))
    data = data[labels.eq("noncredit" if domain == "ood" else "credit")]
    good = cp._complete_folds(data, metric, fractional_reference=metric == "rmse")
    if good.empty:
        return pd.DataFrame()
    keys = ["base_short", "test_dataset_id", *[c for c in ("eval_run", "split", "fold_idx") if c in good]]
    original = (good[good.source.str.endswith("-untuned", na=False)]
                .groupby(keys)[metric].mean().rename("original").reset_index())
    adapted = good[good.source.str.endswith("-trained", na=False)][["method_dirname", *keys, metric]]
    pairs = adapted.rename(columns={metric: "adapted"}).merge(original, on=keys, validate="many_to_one")
    pairs["effect"] = _effect(pairs.adapted, pairs.original, metric)
    return pairs[np.isfinite(pairs.effect)]


def _approach(experiment, meta):
    if experiment == 5:
        return "DiagScale"
    if experiment == 8:
        return "LoRA"
    return "Frozen backbone" if meta["use_lora"] else "Full CPT"


def _setting(experiment, arm, row):
    if experiment == 1:
        return f"LR {row.learning_rate:.0e} · λ {row.l2sp_lambda:g}"
    if experiment == 2:
        return "Reference · seed 43"
    if experiment == 3:
        return style.SAMPLING_LABELS[row.sampling]
    if experiment in (5, 8):
        return f"LR {row.learning_rate:g}"
    return str(arm)


def table_effects(run, metric, *, experiment, arm=None, domain="credit"):
    """Table-level paired effects with fold consistency and configuration identity."""
    pairs = fold_pairs(run, metric, domain=domain)
    if pairs.empty:
        return pd.DataFrame()
    n_folds = pairs.fold_idx.nunique()
    by_partition = pairs.groupby(["method_dirname", "base_short", "test_dataset_id", "split"]).agg(
        original=("original", "mean"), adapted=("adapted", "mean"), effect=("effect", "mean"),
        folds=("effect", "size"), folds_improved=("effect", lambda s: int(s.gt(0).sum()))).reset_index()
    by_partition = by_partition[by_partition.folds.eq(n_folds)]
    if domain == "ood":
        # Retention tables are scored once per CPT partition: require all of them.
        partitions = int(run.cfg.corpus.n_folds)
        complete = by_partition.groupby(["method_dirname", "test_dataset_id"]).split.transform("nunique").eq(partitions)
        by_partition = by_partition[complete]
    rows = by_partition.groupby(["method_dirname", "base_short", "test_dataset_id"]).agg(
        original=("original", "mean"), adapted=("adapted", "mean"), effect=("effect", "mean"),
        folds=("folds", "sum"), folds_improved=("folds_improved", "sum"),
        partitions=("split", "nunique")).reset_index()
    records = []
    for row in rows.itertuples(index=False):
        meta = eval_viz._decode_method_dirname(row.method_dirname)
        approach = _approach(experiment, meta)
        records.append(dict(row._asdict(), base=training_viz.compact_base(meta["base_short"]),
            dataset=row.test_dataset_id, learning_rate=float(meta["lr"]),
            l2sp_lambda=float(meta["l2sp_lambda"] or 0.), frozen=approach == "Frozen backbone",
            sampling=meta["epoch_pass_mode"], approach=approach, experiment=int(experiment),
            study=STUDIES[int(experiment)], arm=arm, metric=metric, domain=domain))
    data = pd.DataFrame(records)
    if data.empty:
        return data
    data["setting"] = [_setting(experiment, arm, r) for r in data.itertuples()]
    data["reference"] = (data.experiment.eq(1) & data.approach.eq("Full CPT")
                         & np.isclose(data.learning_rate, REFERENCE["learning_rate"])
                         & np.isclose(data.l2sp_lambda, REFERENCE["l2sp_lambda"]))
    data["config"] = config_labels(data)
    return data


def load_studies(track):
    """Every completed research study for a track, excluding experiment-0 controls.

    Experiment 7 reuses experiment 6's large-table checkpoints as its 8,192-row arm;
    that arm enters once, as experiment 6's large-table policy.
    """
    from src.visualize import size_studies as ss
    studies = [(1, None, cp.load_campaign(1, track)), (2, None, cp.load_campaign(2, track)),
               (3, None, cp.load_campaign(3, track)), (5, None, cp.load_campaign(5, track))]
    if track == "pd":
        studies += [(6, arm, run) for arm, run in ss.load_arms(6).items()]
        studies += [(7, arm, run) for arm, run in ss.load_arms(7).items() if arm != "8,192 rows"]
    studies.append((8, None, cp.load_campaign(8, track)))
    from src.visualize.legacy_extensions import studies as extension_studies
    studies.extend(extension_studies(track))
    return studies


def catalog(studies, metric, *, domain="credit"):
    frames = [table_effects(run, metric, experiment=e, arm=a, domain=domain) for e, a, run in studies]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["config"] = config_labels(data)
    if data.duplicated(["config", "base", "dataset"]).any():
        raise ValueError("Duplicate configuration/table effect in the synthesis catalog")
    return data


def config_labels(data):
    scope = data.approach.map({"Full CPT": "all", "Frozen backbone": "frozen", "DiagScale": "", "LoRA": ""})
    labels = []
    for row, sc in zip(data.itertuples(), scope):
        from src.utils.study_layout import NUMBERS
        prefix = f"E{NUMBERS[row.experiment]}"
        if row.experiment == 1:
            labels.append(f"Grid {sc} · {row.setting}")
        else:
            labels.append(f"{prefix} {row.setting}")
    return labels


def config_order(data):
    """Experiment, then approach, rate, anchoring and arm order: fixed by design."""
    arm_order = {"All tables": 0, "Large tables": 1, "2,048 rows": 2, "10,000 rows": 3, "11,000 rows": 4, "26,000 rows": 5,
                 "one_sample": 0, "full_pass": 1, "accumulate": 2}
    keys = data.drop_duplicates("config").assign(
        a=lambda f: f.approach.map(APPROACHES.index),
        s=lambda f: [arm_order.get(r.arm, arm_order.get(r.sampling, 0)) for r in f.itertuples()])
    from src.utils.study_layout import NUMBERS
    keys = keys.assign(public_study=keys.experiment.map(NUMBERS)).sort_values(
        ["public_study", "a", "learning_rate", "l2sp_lambda", "s"])
    return list(keys.config)


def study_arms(data):
    """Configurations whose selection does not depend on outcomes.

    Experiment 1 contributes its predefined reference recipe and the same recipe with a
    frozen backbone; every follow-up contributes all of its prescribed arms.
    """
    keep = data.experiment.ne(1) | (np.isclose(data.learning_rate, REFERENCE["learning_rate"])
                                    & np.isclose(data.l2sp_lambda, REFERENCE["l2sp_lambda"]))
    return data[keep]


def outcome(effect, metric):
    band = style.TIE_BAND[metric]
    return np.where(effect > band, "Improved", np.where(effect < -band, "Worse", "Negligible"))


def summarize(data):
    """Equal-table summaries per configuration and base, with explicit outcome counts."""
    if data.empty:
        return pd.DataFrame()
    data = data.assign(outcome=outcome(data.effect.to_numpy(float), data.metric.iloc[0]))
    keys = ["experiment", "study", "approach", "config", "base", "metric"]
    summary = data.groupby(keys).agg(
        tables=("dataset", "nunique"), original=("original", "mean"), adapted=("adapted", "mean"),
        mean=("effect", "mean"), median=("effect", "median"),
        q25=("effect", lambda x: x.quantile(.25)), q75=("effect", lambda x: x.quantile(.75)),
        minimum=("effect", "min"), maximum=("effect", "max"),
        improved=("outcome", lambda x: int(x.eq("Improved").sum())),
        negligible=("outcome", lambda x: int(x.eq("Negligible").sum())),
        worse=("outcome", lambda x: int(x.eq("Worse").sum())),
        folds_improved=("folds_improved", "sum"), folds=("folds", "sum")).reset_index()
    summary["share_improved"] = summary.improved / summary.tables
    return summary


def extremes(summary):
    """Highest/lowest mean and most consistent configuration per base.

    These are observed maxima over a grid and are labelled as such; they are not
    independently validated recipes.
    """
    rows = []
    for base, part in summary.groupby("base"):
        best, worst = part.loc[part["mean"].idxmax()], part.loc[part["mean"].idxmin()]
        consistent = part.sort_values(["share_improved", "median"], ascending=False).iloc[0]
        for role, row in (("Highest observed mean", best), ("Lowest observed mean", worst),
                          ("Most tables improved", consistent)):
            rows.append(dict(base=style.model_label(base), role=role, configuration=row.config,
                             mean=row["mean"], median=row["median"], improved=row.improved,
                             negligible=row.negligible, worse=row.worse, tables=row.tables))
    return pd.DataFrame(rows)


def selection_check(data):
    """Predefined reference, observed best recipe, and a leave-one-table-out selection.

    The observed best recipe maximizes the mean over the same tables it is reported on,
    so its mean is optimistically biased. The leave-one-table-out column selects the
    recipe with the highest mean on the other tables and records its effect on the
    held-out table; averaging those effects estimates what selecting a recipe by this
    rule changes these observed effects conditional on the fitted grid. This is an
    exploratory selection-sensitivity analysis, not a nested estimate of new-table
    generalization: fitted training partitions and the analyzed dataset collection
    are shared across the retrospective choices.
    """
    rows = []
    for base, part in data.groupby("base"):
        matrix = part.pivot_table(index="dataset", columns="config", values="effect")
        matrix = matrix.dropna(axis=1)
        if matrix.shape[1] < 2 or matrix.shape[0] < 3:
            continue
        means = matrix.mean()
        best = means.idxmax()
        reference = part.loc[part.reference, "config"]
        held, chosen = [], []
        for table in matrix.index:
            pick = matrix.drop(index=table).mean().idxmax()
            chosen.append(pick)
            held.append(matrix.at[table, pick])
        held = np.asarray(held)
        rows.append(dict(base=style.model_label(base), tables=len(matrix), recipes=matrix.shape[1],
            reference_recipe=reference.iloc[0] if len(reference) else "",
            reference_mean=means.get(reference.iloc[0], np.nan) if len(reference) else np.nan,
            best_observed_recipe=best, best_observed_mean=means[best],
            leave_one_table_out_mean=held.mean(),
            leave_one_table_out_improved=int((held > 0).sum()),
            recipes_chosen=", ".join(f"{k} ×{v}" for k, v in pd.Series(chosen).value_counts().items())))
    return pd.DataFrame(rows)


def plot_reference_tables(data, track):
    """The predefined experiment-1 reference recipe, table by table."""
    metric = PRIMARY[track]
    ref = data[data.reference]
    unit = "AUC (percentage points)" if track == "pd" else "R²"
    return plot_table_effects(ref, track, name="reference_tables",
        title=f"{track.upper()}: predefined reference recipe, table by table",
        caption=(f"Experiment 1, predefined reference recipe (all trainable weights updated, peak LR 3e-7, L2-SP λ 0.003, "
                 f"seed 42), final five-fold benchmark: paired change in {unit} on every held-out credit table relative to "
                 "the original model of the same base. Each axis label gives the equal-table mean and the number of tables "
                 f"improving by more than {style.TIE_BAND[metric] * SCALE[metric]:g} "
                 f"{'pp' if track == 'pd' else 'R²'}. Filled markers: at least four of the five outer folds change in the "
                 "direction of the table mean; open markers: the folds disagree. Tables are ordered by their mean effect "
                 "across the four bases; horizontal axes differ between models. " + NOTE_PAIRING))


def findings(credit, track):
    """Data-derived summary sentences, so narrative numbers cannot drift from the tables."""
    metric = PRIMARY[track]
    if credit.empty:
        return "No complete paired comparisons are available."
    unit = "pp AUC" if track == "pd" else "R²"
    scale = SCALE[metric]
    d = 3 if track == "pd" else 4
    summary = summarize(credit)
    lines = []
    ref = summary[summary.config.isin(credit.loc[credit.reference, "config"])]
    for row in ref.sort_values("base").itertuples():
        lines.append(f"Reference recipe, {style.model_label(row.base)}: mean {row.mean * scale:+.{d}f} {unit}, "
                     f"median {row.median * scale:+.{d}f}, {row.improved}/{row.tables} tables improved, "
                     f"{row.worse} worse (original {row.original:.4f} → adapted {row.adapted:.4f}).")
    for base, part in summary.groupby("base"):
        best, worst = part.loc[part["mean"].idxmax()], part.loc[part["mean"].idxmin()]
        positive = int(part["mean"].gt(style.TIE_BAND[metric]).sum())
        lines.append(f"{style.model_label(base)}: {positive} of {len(part)} configurations have a mean gain above "
                     f"{style.TIE_BAND[metric] * scale:g} {unit}; highest observed mean {best['mean'] * scale:+.{d}f} "
                     f"({best.config}), lowest {worst['mean'] * scale:+.{d}f} ({worst.config}).")
    grid = credit[credit.experiment.eq(1)]
    if not grid.empty:
        rates = grid.groupby(["base", "learning_rate"]).effect.mean().unstack()
        for base, row in rates.iterrows():
            lines.append(f"{style.model_label(base)}, shared factorial averaged over anchoring and scope: "
                         + ", ".join(f"LR {lr:.0e} {value * scale:+.{d}f}" for lr, value in row.items()) + f" {unit}.")
    return "\n".join(lines)


STUDY_NOTE = (
    "Experiment 0 (controls and pilots) is excluded. Each study pairs its adapted checkpoints with its own evaluation "
    "of the original model, under that study's inference settings, so every effect in this notebook is a within-study "
    "paired difference. Grid labels refer to the shared 512-run factorial analyzed by experiments 1, 3 and 4; "
    "each unique checkpoint enters the catalog once. The v2/v2.6 additions extend model coverage and remain explicitly exploratory. "
    "Experiment 10 reuses the predefined main-grid reference checkpoints at varying context sizes. Experiments 8 and 9 "
    "are PD-only. Cross-study contrasts can involve several recipe differences at once (see the comparability table: "
    "PD sampling, training precision, TabICLv2 inference precision, episode cap and learning-rate parameterization).")


def comparability_table(studies):
    """What differs between studies beyond the named factor."""
    rows, seen = [], set()
    for experiment, arm, run in studies:
        cfg = run.cfg
        bases = tuple(sorted(run.trials.base.unique()))
        key = (experiment, str(cfg.run_name), arm, bases)
        if key in seen:
            continue
        seen.add(key)
        options = cfg.get("evaluation", {}).get("tabicl_inference_options") if hasattr(cfg, "get") else None
        precision = active_precision(run, experiment)
        from src.utils.study_layout import NUMBERS
        rows.append(dict(
            experiments="1, 3, 4 (shared)" if experiment == 1 else str(NUMBERS[experiment]),
            study=STUDIES[experiment], campaign=str(cfg.run_name), arm=arm or "",
            bases=", ".join(style.model_label(b) for b in bases),
            learning_rates=", ".join(f"{v:g}" for v in sorted(run.trials.learning_rate.unique())),
            l2sp=", ".join(f"{v:g}" for v in sorted(run.trials.l2sp_lambda.unique())),
            pd_sampling=str(cfg.train.context_sampling) if run.track == "pd" else "uniform rows",
            training_amp=bool(cfg.train.amp),
            tabicl_inference=precision or ("FP32, no FA3" if options and not options.get("use_amp", True) else "automatic"),
            episode_rows=str(cfg.train.get("max_rows_per_step") or "measured cap"),
            seed=int(cfg.seed), trials=len(run.trials)))
    return pd.DataFrame(rows)


def active_precision(run, experiment):
    """Read the promoted evidence policy, never infer it from a future config change."""
    import json
    from src.utils.paths import manifests_dir
    from src.visualize.inputs import analysis_location
    if not any("tabicl" in str(base) for base in run.trials.base.unique()):
        return "not applicable"
    group = f"experiment{experiment}"
    folder = analysis_location(group, "manifests", manifests_dir(group))
    pointer = folder / "active_evaluation.json"
    if not pointer.is_file():
        return None
    record = json.loads(pointer.read_text(encoding="utf-8"))
    if record.get("family") != "tabicl" or record.get("options") != {"use_amp":False,"use_fa3":False}:
        raise ValueError("Invalid published TabICL precision policy")
    debug = analysis_location("experiment0", "manifests", manifests_dir("experiment0"))
    receipt = debug/"evaluation_refresh"/record["refresh_id"]/"inspection.json"
    from src.utils.experiment import file_digest
    if file_digest(receipt) != record["inspection_sha256"]:
        raise ValueError("Published precision inspection is missing or changed")
    return "FP32, no FA3 (audited refresh)"


def costs(studies):
    """Recorded training hours summed over the four CPT partitions of each configuration."""
    rows = []
    for experiment, arm, run in studies:
        trials = run.trials[run.trials.status.eq("OK")]
        if "gpu_hours" not in trials or trials.empty:
            continue
        if experiment in (6, 7) and arm is not None:
            pass  # `_select` already restricted the trials to this arm.
        group = trials.groupby(["base", "learning_rate", "l2sp_lambda", "frozen", "sampling"])
        for key, part in group:
            rows.append(dict(zip(["base", "learning_rate", "l2sp_lambda", "frozen", "sampling"], key),
                             experiment=experiment, arm=arm, gpu_hours=float(part.gpu_hours.sum()),
                             partitions=part.partition.nunique()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _symmetric_limit(values):
    values = np.abs(np.asarray(values, float))
    values = values[np.isfinite(values)]
    return max(float(np.quantile(values, style.HEATMAP_COLOR_QUANTILE)) if values.size else 0., 1e-9)


def _fmt(value, metric):
    if not np.isfinite(value):
        return "—"
    digits = 3 if metric == "r2" else 2
    if 0 < abs(value) < .5 * 10**(-digits):
        return f"{value:+.1g}"
    return f"{value:+.{digits}f}"


def plot_landscape(data, track):
    """Every configuration × base: mean paired effect, with improved/total tables."""
    metric = PRIMARY[track]
    summary = summarize(data)
    if summary.empty:
        return []
    order = config_order(data)
    bases = [b for b in BASES if b in set(summary.base)]
    mean = summary.pivot(index="config", columns="base", values="mean").reindex(index=order, columns=bases)
    improved = summary.pivot(index="config", columns="base", values="improved").reindex(index=order, columns=bases)
    tables = summary.pivot(index="config", columns="base", values="tables").reindex(index=order, columns=bases)
    scaled = mean * SCALE[metric]
    limit = _symmetric_limit(scaled.to_numpy())
    height = min(style.MAX_HEIGHT, 0.9 + style.OVERVIEW_ROW_HEIGHT * len(order))
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, height), layout="constrained")
    image = ax.imshow(np.clip(scaled.to_numpy(float), -limit, limit), aspect="auto", cmap=style.CMAP_DIVERGING,
                      norm=TwoSlopeNorm(0., -limit, limit))
    ax.grid(False)
    for (r, c), value in np.ndenumerate(scaled.to_numpy(float)):
        if not np.isfinite(value):
            ax.text(c, r, "not run", ha="center", va="center", fontsize=style.GRID_FONT - 1,
                    color=style.color("annotation"))
            continue
        text = f"{_fmt(value, metric)}  {int(improved.iat[r, c])}/{int(tables.iat[r, c])}"
        ax.text(c, r, text, ha="center", va="center", fontsize=style.GRID_FONT - 1,
                color=style.GRID_TEXT_LIGHT if abs(value) > .7 * limit else style.GRID_TEXT_DARK)
    configs = data.drop_duplicates("config").set_index("config")
    ax.set_yticks(range(len(order)), order)
    for tick, name in zip(ax.get_yticklabels(), order):
        tick.set_color(style.APPROACH_COLORS[configs.loc[name, "approach"]])
        tick.set_fontsize(style.GRID_FONT - 1)
        if bool(configs.loc[name, "reference"]):
            tick.set_fontweight("bold")
    ax.set_xticks(range(len(bases)), [style.model_label(b) for b in bases])
    ax.xaxis.tick_top()
    experiments = [int(configs.loc[name, "experiment"]) for name in order]
    for i in range(1, len(order)):
        if experiments[i] != experiments[i - 1]:
            ax.axhline(i - .5, color=style.color("reference"), linewidth=style.THIN_LINE)
    bar = fig.colorbar(image, ax=ax, extend="both", shrink=.6)
    bar.set_label(UNITS[metric] + ", CPT − original")
    handles = [Patch(color=style.APPROACH_COLORS[a], label=a) for a in APPROACHES if a in set(data.approach)]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), title="Row-label colour: adaptation approach")
    fig.suptitle(f"{track.upper()}: every CPT configuration vs. its original model")
    band = style.TIE_BAND[metric] * SCALE[metric]
    return [cp.Page("overview_landscape", fig,
        f"Held-out credit tables, final five-fold benchmark. Each cell is the equal-table mean change in "
        f"{'AUC (percentage points)' if track == 'pd' else 'R²'} of an adapted model relative to the original "
        "checkpoint of the same base, followed by the number of tables that improved by more than "
        f"{band:g} {'pp' if track == 'pd' else 'R²'} and the number of complete tables. " + NOTE_PAIRING +
        "Rows are grouped by experiment (E1 main grid: four learning rates × L2-SP λ ∈ {0, 0.003} × all or "
        "frozen-backbone updates; E2 the reference recipe with a second training seed; E3 sampling protocols; "
        "E5 DiagScale query-scaling adapters; E6 all versus large training tables; E7 episode sizes on the large-table "
        "corpus; E8 rank-16 query/value LoRA). The bold row is the predefined reference recipe (full updates, "
        "LR 3e-7, λ 0.003); every other row is reported in full rather than selected. Label colours identify the "
        "adaptation approach. Colours are symmetric about zero and saturate at the 95th percentile of |cell value| "
        f"({limit:.2f}); printed values are exact. 'Not run' marks base/approach combinations outside a study's design. "
        "Studies differ in more than their named factor (see the comparability table).",
        {"Configuration summaries": summary},
        {"measurement": "five_fold_final_benchmark", "aggregation_unit": "table", "metric": metric})]


def _box(ax, values, y, color, limits=None):
    values = np.asarray(values, float)
    q10, q25, q50, q75, q90 = np.clip(np.quantile(values, [.1, .25, .5, .75, .9]), *(limits or (-np.inf, np.inf)))
    ax.plot([q10, q25], [y, y], color=color, linewidth=style.THIN_LINE)
    ax.plot([q75, q90], [y, y], color=color, linewidth=style.THIN_LINE)
    ax.add_patch(plt.Rectangle((q25, y - .28), q75 - q25, .56, facecolor=color, alpha=.25,
                               edgecolor=color, linewidth=style.THIN_LINE))
    ax.plot([q50, q50], [y - .28, y + .28], color=color, linewidth=style.MEAN_CURVE_LINE)
    jitter = np.linspace(-.18, .18, len(values)) if len(values) > 1 else np.zeros(1)
    low, high = limits if limits is not None else (-np.inf, np.inf)
    inside = (values >= low) & (values <= high)
    ax.scatter(values[inside], (y + jitter)[inside], s=style.TABLE_POINT_SIZE * .6, color=color,
               alpha=style.POINT_ALPHA, linewidths=0, zorder=3)
    # Tables beyond the robust axis range stay visible as triangles at the axis edge.
    for edge, mask, marker in ((low, values < low, "<"), (high, values > high, ">")):
        if mask.any():
            ax.scatter(np.full(int(mask.sum()), edge), (y + jitter)[mask], marker=marker, s=style.TABLE_POINT_SIZE,
                       color=color, linewidths=0, zorder=3, clip_on=False)
    ax.scatter(np.clip(values.mean(), low, high), y, marker="D", s=style.POINT_SIZE * .8, color="white",
               edgecolors="black", linewidths=style.THIN_LINE, zorder=4, clip_on=False)


def plot_distributions(data, track):
    """Table-level effect distributions for the outcome-independent study arms."""
    metric = PRIMARY[track]
    arms = study_arms(data)
    if arms.empty:
        return []
    order = [c for c in config_order(data) if c in set(arms.config)]
    bases = [b for b in BASES if b in set(arms.base)]
    height = min(style.MAX_HEIGHT, 1.0 + .2 * len(order))
    fig, axes = plt.subplots(1, len(bases), sharey=True, figsize=(style.WIDTH_FULL, height), layout="constrained")
    axes = np.atleast_1d(axes)
    clipped = []
    for ax, base in zip(axes, bases):
        part = arms[arms.base.eq(base)]
        scaled = part.effect * SCALE[metric]
        # Robust axis: this base's 2nd-98th percentile of table effects plus zero, so one
        # catastrophic table does not flatten every other distribution.
        low, high = min(0., scaled.quantile(.02)), max(0., scaled.quantile(.98))
        pad = .08 * ((high - low) or 1.)
        limits = (low - pad, high + pad)
        clipped.append((base, int(((scaled < limits[0]) | (scaled > limits[1])).sum())))
        for i, config in enumerate(order):
            values = part.loc[part.config.eq(config), "effect"] * SCALE[metric]
            if values.empty:
                continue
            approach = part.loc[part.config.eq(config), "approach"].iloc[0]
            _box(ax, values.to_numpy(float), i, style.APPROACH_COLORS[approach], limits)
        ax.set_xlim(*limits)
        ax.axvline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_title(style.model_label(base))
        ax.set_xlabel(SHORT_UNITS[metric])
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_ylim(len(order) - .5, -.5)
    for tick in axes[0].get_yticklabels():
        tick.set_fontsize(style.GRID_FONT)
    handles = [Patch(facecolor=style.APPROACH_COLORS[a], alpha=.5, label=a) for a in APPROACHES if a in set(arms.approach)]
    handles.append(Line2D([], [], marker="D", linestyle="", markerfacecolor="white", markeredgecolor="black", label="Mean"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(f"{track.upper()}: CPT effects across held-out credit tables")
    return [cp.Page("overview_distributions", fig,
        f"Distribution over held-out credit tables of the paired change in {'AUC (percentage points)' if track == 'pd' else 'R²'} "
        "relative to each base's original model. Boxes span the table interquartile range, the thick line is the median, "
        "whiskers reach the 10th and 90th percentiles, dots are individual tables and the white diamond is the equal-table mean. "
        + NOTE_PAIRING + "Experiment 1 contributes only its predefined reference recipe (LR 3e-7, L2-SP λ 0.003) with all "
        "parameters updated and with the backbone frozen; every follow-up contributes all prescribed arms, so no row is chosen "
        "by its outcome. Horizontal axes differ between models and span each model's 2nd-98th percentile of table effects and zero; "
        "tables outside that range are drawn as triangles at the axis edge (" + ", ".join(f"{style.model_label(b)}: {n}" for b, n in clipped)
        + " tables) and keep their exact values in the summary. Spread describes heterogeneity across tables, not statistical uncertainty.",
        {"Table effects for the displayed arms": arms[["config", "base", "dataset", "original", "adapted", "effect",
                                                       "folds_improved", "folds"]]},
        {"measurement": "five_fold_final_benchmark", "aggregation_unit": "table", "metric": metric})]


def plot_outcomes(data, track):
    """Improved / negligible / worse table counts for the same arms."""
    metric = PRIMARY[track]
    arms = study_arms(data)
    if arms.empty:
        return []
    arms = arms.assign(outcome=outcome(arms.effect.to_numpy(float), metric))
    order = [c for c in config_order(data) if c in set(arms.config)]
    bases = [b for b in BASES if b in set(arms.base)]
    height = min(style.MAX_HEIGHT, 1.0 + .2 * len(order))
    fig, axes = plt.subplots(1, len(bases), sharey=True, sharex=True, figsize=(style.WIDTH_FULL, height),
                             layout="constrained")
    axes = np.atleast_1d(axes)
    counts = arms.groupby(["base", "config", "outcome"]).size().unstack(fill_value=0)
    for ax, base in zip(axes, bases):
        for i, config in enumerate(order):
            if (base, config) not in counts.index:
                continue
            row, left = counts.loc[(base, config)], 0
            total = row.sum()
            for label in ("Worse", "Negligible", "Improved"):
                share = row.get(label, 0) / total
                ax.barh(i, share, left=left, color=style.OUTCOME_COLORS[label], height=style.BAR_HEIGHT,
                        label=label if i == 0 else None)
                left += share
            ax.text(1.0, i, f" {int(row.get('Improved', 0))}/{int(total)}", va="center", ha="left",
                    fontsize=style.GRID_FONT - 1)
        ax.axvline(.5, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_xlim(0, 1.22)
        ax.set_xticks([0, .5, 1], ["0", "½", "1"])
        ax.set_title(style.model_label(base))
        ax.set_xlabel("Share of tables")
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_ylim(len(order) - .5, -.5)
    for tick in axes[0].get_yticklabels():
        tick.set_fontsize(style.GRID_FONT)
    handles = [Patch(color=style.OUTCOME_COLORS[k], label=k) for k in ("Improved", "Negligible", "Worse")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fig.suptitle(f"{track.upper()}: tables improved, unchanged or worse")
    band = style.TIE_BAND[metric] * SCALE[metric]
    return [cp.Page("overview_outcomes", fig,
        f"For the same arms as the distribution figure, the share of held-out credit tables whose paired "
        f"{'AUC' if track == 'pd' else 'R²'} change relative to the original model exceeds +{band:g} "
        f"{'percentage points' if track == 'pd' else ''} (improved), lies within ±{band:g} (negligible) or falls below "
        f"−{band:g} (worse); numbers give improved/total tables. The ±{band:g} band is a reporting convention for this study, "
        "not a statistical equivalence margin. " + NOTE_PAIRING + "The dotted line marks half of the tables.",
        {"Outcome counts": counts.reset_index()},
        {"measurement": "five_fold_final_benchmark", "aggregation_unit": "table", "metric": metric,
         "tie_band": style.TIE_BAND[metric]})]


def plot_table_map(data, track, base):
    """Tables × study arms for one base: which tables drive an average."""
    metric = PRIMARY[track]
    arms = study_arms(data)
    arms = arms[arms.base.eq(base)]
    if arms.empty:
        return []
    order = [c for c in config_order(data) if c in set(arms.config)]
    originals = arms.groupby("dataset").original.mean().sort_values()
    matrix = arms.pivot(index="dataset", columns="config", values="effect").reindex(
        index=originals.index, columns=order) * SCALE[metric]
    limit = _symmetric_limit(matrix.to_numpy())
    configs = data.drop_duplicates("config").set_index("config")
    chunks = [order[i:i + style.TABLE_MAP_COLUMNS] for i in range(0, len(order), style.TABLE_MAP_COLUMNS)]
    pages = []
    for index, columns in enumerate(chunks, 1):
        part = matrix[columns]
        fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
        image = ax.imshow(np.clip(part.to_numpy(float), -limit, limit), aspect="auto", cmap=style.CMAP_DIVERGING,
                          norm=TwoSlopeNorm(0., -limit, limit))
        ax.grid(False)
        for (r, c), value in np.ndenumerate(part.to_numpy(float)):
            ax.text(c, r, _fmt(value, metric) if np.isfinite(value) else "—", ha="center", va="center",
                    fontsize=style.GRID_FONT,
                    color=style.GRID_TEXT_LIGHT if np.isfinite(value) and abs(value) > .7 * limit else style.GRID_TEXT_DARK)
        ax.set_yticks(range(len(part)), [f"{d} ({originals[d]:.3f})" for d in part.index])
        ax.set_xticks(range(len(columns)), columns, rotation=55, ha="right")
        for tick, name in zip(ax.get_xticklabels(), columns):
            tick.set_color(style.APPROACH_COLORS[configs.loc[name, "approach"]])
        for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            tick.set_fontsize(style.GRID_FONT)
        bar = fig.colorbar(image, ax=ax, extend="both", shrink=.8)
        bar.set_label(SHORT_UNITS[metric])
        suffix = f" · {index}/{len(chunks)}" if len(chunks) > 1 else ""
        fig.suptitle(f"{track.upper()} / {style.model_label(base)}: table-level CPT effects{suffix}")
        pages.append(cp.Page(f"overview_tables_{base}_{index:02}", fig,
            f"{style.model_label(base)}, held-out credit tables: paired change in {'AUC (percentage points)' if track == 'pd' else 'R²'} "
            f"of each study arm relative to the original model (panel {index} of {len(chunks)}). Rows are tables, ordered by the original model's mean score "
            "(in parentheses, averaged over the available original evaluations; inference recipes can differ across studies); columns are the outcome-independent arms "
            "of the distribution figure, coloured by approach. " + NOTE_PAIRING +
            f"All panels for this base share a colour scale, symmetric about zero and saturated at the 95th percentile of |value| ({limit:.2f}); labels give rounded values. "
            "'—' marks a missing complete comparison.",
            {"Table × arm effects": part.reset_index()},
            {"measurement": "five_fold_final_benchmark", "aggregation_unit": "table", "metric": metric, "base": base}))
    return pages


def plot_rate_response(data, track):
    """Mean effect against learning rate, one line per approach and anchoring setting."""
    metric = PRIMARY[track]
    rated = data[data.experiment.isin([1, 5, 8])]
    if rated.empty:
        return []
    # A missing rate must not silently replace the table population along a curve.
    keys = ["base", "approach", "l2sp_lambda"]
    if rated.duplicated(keys + ["dataset", "learning_rate"]).any():
        raise ValueError("Duplicate table/rate in approach comparison")
    expected = rated.groupby(keys).learning_rate.nunique().rename("expected_rates")
    support = rated.groupby(keys + ["dataset"]).learning_rate.nunique().rename("observed_rates").reset_index()
    support = support.merge(expected.reset_index(), on=keys, validate="many_to_one")
    cohort = support[support.observed_rates.eq(support.expected_rates)][keys + ["dataset"]]
    rated = rated.merge(cohort, on=keys + ["dataset"], validate="many_to_one")
    if rated.empty:
        return []
    summary = rated.groupby(["base", "approach", "l2sp_lambda", "learning_rate"]).effect.agg(
        mean="mean", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75), tables="size").reset_index()
    bases = [b for b in BASES if b in set(summary.base)]
    fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO),
                             layout="constrained", sharex=True)
    axes = axes.ravel()
    for ax, base in zip(axes, bases):
        for (approach, lam), part in summary[summary.base.eq(base)].groupby(["approach", "l2sp_lambda"]):
            part = part.sort_values("learning_rate")
            color = style.APPROACH_COLORS[approach]
            linestyle = "--" if approach in ("Full CPT", "Frozen backbone") and lam == 0 else "-"
            label = approach + (f", λ {lam:g}" if approach in ("Full CPT", "Frozen backbone") else "")
            ax.fill_between(part.learning_rate, part.q25 * SCALE[metric], part.q75 * SCALE[metric],
                            color=color, alpha=style.INTERVAL_ALPHA, linewidth=0)
            ax.plot(part.learning_rate, part["mean"] * SCALE[metric], color=color, linestyle=linestyle,
                    marker=style.APPROACH_MARKERS[approach], markersize=3, linewidth=style.THIN_LINE, label=label)
        ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_xscale("log")
        ax.set_title(style.model_label(base))
        ax.set_ylabel(SHORT_UNITS[metric])
    for ax in axes[len(bases):]:
        ax.set_visible(False)
    for ax in axes[-2:]:
        ax.set_xlabel("Peak learning rate (log scale)")
    handles = []
    for approach in APPROACHES:
        if approach not in set(summary.approach):
            continue
        if approach in ("Full CPT", "Frozen backbone"):
            for lam, line in ((0.003, "-"), (0.0, "--")):
                handles.append(Line2D([], [], color=style.APPROACH_COLORS[approach], linestyle=line,
                                      marker=style.APPROACH_MARKERS[approach], markersize=3, label=f"{approach}, λ {lam:g}"))
        else:
            handles.append(Line2D([], [], color=style.APPROACH_COLORS[approach], marker=style.APPROACH_MARKERS[approach],
                                  markersize=3, label=approach))
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fig.suptitle(f"{track.upper()}: learning-rate response by approach")
    return [cp.Page("overview_rate_response", fig,
        f"Equal-table mean paired change in {'AUC (percentage points)' if track == 'pd' else 'R²'} relative to the original "
        "model, against peak learning rate, for full-parameter and frozen-backbone CPT (experiment 1; solid λ = 0.003, "
        "dashed λ = 0), DiagScale query-scaling adapters (experiment 5) and query/value LoRA (experiment 8). Shaded bands "
        "span the table interquartile range, not confidence intervals. Each curve retains the same complete table cohort across its rates. " + NOTE_PAIRING +
        "The rate scales are not interchangeable across approaches: DiagScale updates multiplicative query scales and LoRA "
        "updates low-rank factors, so their rates are shown on a common axis only to display each approach's own "
        "sensitivity. The studies also differ in PD sampling, training precision and TabICLv2 inference settings.",
        {"Rate response": summary, "Fixed-cohort table effects": rated},
        {"measurement": "five_fold_final_benchmark", "aggregation_unit": "table", "metric": metric})]


def plot_transfer_retention(credit, retention, track):
    """Credit gain against non-credit change per configuration (learning vs forgetting)."""
    metric = PRIMARY[track]
    if credit.empty or retention.empty:
        return []
    a = summarize(credit)[["config", "base", "approach", "experiment", "mean", "tables"]]
    b = summarize(retention)[["config", "base", "mean", "tables"]]
    paired = a.merge(b, on=["config", "base"], suffixes=("_credit", "_retention"), validate="one_to_one")
    if paired.empty:
        return []
    bases = [base for base in BASES if base in set(paired.base)]
    fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
    axes = axes.ravel()
    for ax, base in zip(axes, bases):
        for row in paired[paired.base.eq(base)].itertuples():
            ax.scatter(row.mean_credit * SCALE[metric], row.mean_retention * SCALE[metric],
                       color=style.APPROACH_COLORS[row.approach], marker=style.APPROACH_MARKERS[row.approach],
                       s=style.POINT_SIZE, edgecolors="black" if "λ 0.003" in row.config and row.experiment == 1 and "3e-07" in row.config else "none",
                       linewidths=style.THIN_LINE)
        ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.axvline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_title(style.model_label(base))
        ax.set_xlabel("Held-out credit " + SHORT_UNITS[metric])
        ax.set_ylabel("Non-credit " + SHORT_UNITS[metric])
    for ax in axes[len(bases):]:
        ax.set_visible(False)
    handles = [Line2D([], [], marker=style.APPROACH_MARKERS[a], linestyle="", color=style.APPROACH_COLORS[a], label=a)
               for a in APPROACHES if a in set(paired.approach)]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(f"{track.upper()}: credit transfer against non-credit retention")
    return [cp.Page("overview_transfer_retention", fig,
        "Each point is one configuration: its equal-table mean paired change on held-out credit tables (horizontal) and "
        "on the four non-credit retention tables of the same task (vertical), both relative to the original model of the same "
        "base. Non-credit tables first average all four CPT training partitions. The upper-right quadrant improves both; "
        "the lower-right gains on credit while losing on the retention panel. Points outlined in black are the experiment-1 "
        "reference recipes (full and frozen). Axis limits differ between models. A four-table retention panel does not "
        "establish general retention. " + NOTE_PAIRING,
        {"Credit and retention means": paired},
        {"measurement": "five_fold_final_benchmark", "aggregation_unit": "table", "metric": metric})]


def plot_headroom(data, track):
    """Reference-recipe effect against the original model's score, per table."""
    metric = PRIMARY[track]
    ref = data[data.reference]
    if ref.empty:
        return []
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_HALF * 1.6, .7), layout="constrained")
    for i, (base, part) in enumerate(ref.groupby("base")):
        ax.scatter(part.original, part.effect * SCALE[metric], color=style.color(base), s=style.POINT_SIZE,
                   marker=style.PANEL_MARKERS[i % len(style.PANEL_MARKERS)], label=style.model_label(base),
                   alpha=.85, linewidths=0)
    ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
    ax.set_xlabel(f"Original model {'AUC' if track == 'pd' else 'R²'} (table mean)")
    ax.set_ylabel(SHORT_UNITS[metric])
    ax.legend()
    fig.suptitle(f"{track.upper()}: reference effect vs. original score")
    return [cp.Page("overview_headroom", fig,
        "Predefined reference recipe (experiment 1: full updates, LR 3e-7, L2-SP λ 0.003): each point is one held-out credit "
        f"table and base, plotting the paired change in {'AUC (percentage points)' if track == 'pd' else 'R²'} against the "
        "original model's mean score on that table. " + NOTE_PAIRING + "The horizontal score appears in both quantities "
        "(the change is computed from it), so a trend can arise from ceiling effects or noise and is descriptive, not a mechanism.",
        {"Reference table effects": ref[["base", "dataset", "original", "adapted", "effect", "folds_improved", "folds"]]},
        {"measurement": "five_fold_final_benchmark", "aggregation_unit": "table", "metric": metric})]


def plot_cost_effect(data, cost, track):
    """Mean effect against recorded training hours for each configuration."""
    metric = PRIMARY[track]
    if data.empty or cost.empty:
        return []
    summary = summarize(data)
    keys = ["base", "learning_rate", "l2sp_lambda", "frozen", "sampling", "experiment"]
    configs = data.drop_duplicates(["config", "base"])[["config", "arm", *keys]]
    merged = configs.merge(cost, on=keys, suffixes=("", "_cost"))
    merged = merged[merged.arm.fillna("").eq(merged.arm_cost.fillna(""))] if "arm_cost" in merged else merged
    merged = merged.merge(summary[["config", "base", "approach", "mean"]], on=["config", "base"])
    merged = merged[merged.partitions.eq(4)]
    if merged.empty:
        return []
    bases = [b for b in BASES if b in set(merged.base)]
    fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, style.DASHBOARD_RATIO), layout="constrained")
    axes = axes.ravel()
    for ax, base in zip(axes, bases):
        for row in merged[merged.base.eq(base)].itertuples():
            ax.scatter(row.gpu_hours, row.mean * SCALE[metric], color=style.APPROACH_COLORS[row.approach],
                       marker=style.APPROACH_MARKERS[row.approach], s=style.POINT_SIZE * .8, linewidths=0)
        ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_xscale("log")
        low, high = ax.get_xlim()
        ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200) if low <= t <= high]
        if len(ticks) < 2:
            from matplotlib.ticker import MaxNLocator
            ticks = [t for t in MaxNLocator(nbins=style.CONTINUOUS_TICKS).tick_values(low, high) if low <= t <= high]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        from matplotlib.ticker import NullFormatter
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(style.model_label(base))
        ax.set_xlabel("Recorded training hours (sum of four partitions)")
        ax.set_ylabel(SHORT_UNITS[metric])
    for ax in axes[len(bases):]:
        ax.set_visible(False)
    handles = [Line2D([], [], marker=style.APPROACH_MARKERS[a], linestyle="", color=style.APPROACH_COLORS[a], label=a)
               for a in APPROACHES if a in set(merged.approach)]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(f"{track.upper()}: predictive change against training cost")
    return [cp.Page("overview_cost", fig,
        f"Each point is one configuration: its equal-table mean paired change in {'AUC (percentage points)' if track == 'pd' else 'R²'} "
        "on held-out credit tables against recorded training time summed over the four CPT partitions (logarithmic axis). "
        "Training time excludes evaluation and queueing and was recorded on the GPUs used for each study, so it is an observed "
        "cost, not a matched-compute comparison. " + NOTE_PAIRING,
        {"Cost and effect": merged},
        {"measurement": "five_fold_final_benchmark", "aggregation_unit": "table", "metric": metric})]


def plot_context_gain(study, track):
    """Experiment 4: CPT − original change against labelled context, per family."""
    from src.visualize import context_curves as cc
    metric = PRIMARY[track]
    paired = cc.paired_tables(study, metric)
    if paired.empty:
        return []
    horizons = [h for h in cc.COHORT_HORIZONS if not cc.fixed_cohort(study, paired, track, h).empty]
    if not horizons:
        return []
    families = list(study.config["families"])
    fig, axes = plt.subplots(1, len(families), sharey=True, figsize=style.figsize(style.WIDTH_FULL, style.SMALL_RATIO),
                             layout="constrained")
    axes = np.atleast_1d(axes)
    styles = dict(zip(horizons, ("-", "--", ":")))
    tables = []
    for ax, family in zip(axes, families):
        color = style.color("v3" if family == "tabpfn" else "tabicl")
        for horizon in horizons:
            cohort = cc.fixed_cohort(study, paired, track, horizon)
            part = cohort[cohort.family.eq(family)]
            curve = part.groupby("context_rows").gain.agg(mean="mean", q25=lambda x: x.quantile(.25),
                                                          q75=lambda x: x.quantile(.75), tables="size")
            n = part.dataset.nunique()
            if horizon == horizons[-1]:
                ax.fill_between(curve.index, curve.q25 * SCALE[metric], curve.q75 * SCALE[metric], color=color,
                                alpha=style.INTERVAL_ALPHA, linewidth=0)
            ax.plot(curve.index, curve["mean"] * SCALE[metric], color=color, linestyle=styles[horizon],
                    linewidth=style.THIN_LINE, label=f"{n} tables, ≤{horizon:,} rows")
            tables.append(curve.reset_index().assign(family=family, horizon=horizon, datasets=n))
        ax.axhline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_xscale("log")
        ax.set_title(cc.FAMILIES[family])
        ax.set_xlabel("Labelled context rows (log scale)")
        ax.set_ylabel(SHORT_UNITS[metric])
        ax.legend()
    fig.suptitle(f"{track.upper()}: CPT effect as labelled context grows")
    return [cp.Page("overview_context", fig,
        f"Experiment 4: paired change in {'AUC (percentage points)' if track == 'pd' else 'R²'} of the fixed reference CPT "
        "checkpoint (experiment 1: full updates, LR 3e-7, L2-SP λ 0.003) relative to its original model, as a function of the "
        "number of labelled context rows supplied at prediction time. Contexts are nested prefixes and test rows are identical "
        "for both weight states. Each line is a fixed table cohort that contributes at every size up to its horizon; the band "
        "spans the table interquartile range for the longest-horizon cohort. Five outer folds are averaged within table before "
        "tables are weighted equally. The curves do not estimate variation over independently resampled contexts.",
        {"Context gain curves": pd.concat(tables, ignore_index=True)},
        {"measurement": "five_fold_context_benchmark", "aggregation_unit": "table", "metric": metric})]


def plot_table_effects(data, track, *, title, name, caption):
    """Per-table paired effects, one panel per base, with fold consistency.

    Filled markers: at least four of five outer folds move in the direction of the table
    mean; open markers: the folds disagree. Tables are ordered by their mean effect across
    the displayed bases.
    """
    metric = PRIMARY[track]
    if data.empty:
        return []
    bases = [b for b in BASES if b in set(data.base)]
    order = data.groupby("dataset").effect.mean().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(1, len(bases), sharey=True, figsize=style.figsize(style.WIDTH_FULL, style.PANEL_RATIO),
                             layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, base in zip(axes, bases):
        part = data[data.base.eq(base)].set_index("dataset").reindex(order)
        values = part.effect * SCALE[metric]
        agree = np.where(values > 0, part.folds_improved, part.folds - part.folds_improved) >= np.ceil(.8 * part.folds)
        color = style.color(base)
        ys = np.arange(len(order))
        ax.hlines(ys, 0, values, color=color, linewidth=style.THIN_LINE, alpha=.6)
        ax.scatter(values[agree], ys[agree], color=color, s=style.POINT_SIZE * .8, zorder=3)
        ax.scatter(values[~agree], ys[~agree], facecolors="white", edgecolors=color, s=style.POINT_SIZE * .8,
                   linewidths=style.THIN_LINE, zorder=3)
        ax.axvline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_title(style.model_label(base))
        ax.set_xlabel(f"{SHORT_UNITS[metric]}\nmean {_fmt(values.mean(), metric)} · "
                      f"{int((part.effect > style.TIE_BAND[metric]).sum())}/{len(order)} up")
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_ylim(len(order) - .5, -.5)
    for tick in axes[0].get_yticklabels():
        tick.set_fontsize(style.GRID_FONT)
    handles = [Line2D([], [], marker="o", linestyle="", color=style.color("reference"), label="≥4 of 5 folds agree"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor="white", color=style.color("reference"),
                      label="Folds disagree")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    fig.suptitle(title)
    return [cp.Page(name, fig, caption,
        {"Displayed table effects": data[["base", "dataset", "original", "adapted", "effect", "folds_improved", "folds"]]},
        {"measurement": "five_fold_final_benchmark", "aggregation_unit": "table", "metric": metric})]
