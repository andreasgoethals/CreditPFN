"""Experiment-scoped, dataset-paired reports for the four research experiments.

Notebooks choose sections; this module performs joins and builds bounded publication pages.
Unavailable outcomes stay unavailable. Synthetic fixtures belong in tests, never reports.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.train.config import load_train_config, resolve_grid
from src.utils.experiment import apply_split_index
from src.utils.paths import REPO_ROOT, logs_dir
from src.visualize import eval_viz, style, training_viz
from src.visualize.paper_figures import effect_label, paired_deltas
from src.visualize.trajectories import load_trajectories, trajectory_effects
from src.visualize.reporting import NotebookReport, page_text

FACTORS = ["base", "learning_rate", "l2sp_lambda", "frozen", "sampling"]


@dataclass
class Campaign:
    cfg: object
    trials: pd.DataFrame
    trajectories: pd.DataFrame
    histories: pd.DataFrame
    evaluation: pd.DataFrame

    @property
    def track(self):
        return str(self.cfg.track)

    @property
    def metric(self):
        return "roc_auc" if self.track == "pd" else "rmse"

    @property
    def target(self):
        return int(self.cfg.train.target_total_steps)


@dataclass
class Page:
    name: str
    figure: object
    caption: str
    tables: dict = field(default_factory=dict)

    def __post_init__(self):
        style.finish_figure(self.figure)
        # Keep the object for explicit inline display/export without accumulating
        # pyplot managers (or implicit duplicate notebook renders) across pages.
        plt.close(self.figure)


def show(sink, pages, *, track=None):
    """Save and display each page immediately, releasing it before the next section."""
    from IPython.display import display, Markdown
    pages = list(pages)
    if not pages:
        print("No measurements available for this section yet.")
        if getattr(sink, "report", None) is not None:
            sink.report._pending.append("No measurements available for this figure group.")
    for page in pages:
        if track is not None:
            if track not in ("pd", "lgd"):
                raise ValueError("Figure track must be pd or lgd")
            page.name = f"{track}_{page.name}"
            title = page.figure._suptitle
            if title is not None and not title.get_text().startswith(track.upper()):
                title.set_text(f"{track.upper()}: {title.get_text()}")
            if not page.caption.startswith(track.upper()):
                page.caption = f"{track.upper()}. {page.caption}"
        if not page.caption.strip():
            raise ValueError(f"Figure {page.name} needs a manuscript caption")
        if getattr(sink, "report", None) is not None:
            sink.report._pending.append(page_text(page))
        sink.save(page.figure, page.name, caption=page.caption)
        display(page.figure)
        display(Markdown(page.caption))
        plt.close(page.figure)


def config_path(experiment: int, track: str, phase: str | None = None) -> Path:
    if track not in ("pd", "lgd") or experiment not in range(4):
        raise ValueError("Expected experiment 0–3 and track pd/lgd")
    if experiment == 0:
        if phase not in ("null", "pilot", "budget"):
            raise ValueError("Experiment 0 requires null, pilot or budget")
        name = f"{phase}_{track}.yaml"
    else:
        if phase is not None:
            raise ValueError("Only experiment 0 has named phases")
        name = f"{track}.yaml"
    return REPO_ROOT / "config" / f"experiment{experiment}" / name


def planned_trials(cfg) -> pd.DataFrame:
    """Use the actual launcher grid, including partition and seed, as the denominator."""
    from src.train.loop import descriptive_name
    rows = []
    for partition in range(int(cfg.corpus.n_splits)):
        current = apply_split_index(OmegaConf.create(OmegaConf.to_container(cfg)), partition)
        for base, lr, frozen, query, accum, sampling, min_rows, lam in resolve_grid(current, single=False):
            lam = float(current.optimizer.l2sp_lambda if lam is None else lam)
            name = descriptive_name(run_name=current.run_name, track=current.track, base_path=base,
                learning_rate=lr, seed=current.seed, use_lora=frozen, query_fraction=query,
                accumulate_grad_batches=accum, epoch_pass_mode=sampling, min_train_rows=min_rows,
                l2sp_lambda=lam, adaptation_mode="frozen_backbone" if frozen else "full").removesuffix(".ckpt")
            parsed = training_viz.parse_trial_name(name)
            rows.append(dict(trial_name=name, base=training_viz.compact_base(parsed.base_short),
                learning_rate=float(lr), l2sp_lambda=lam, frozen=bool(frozen), sampling=sampling,
                seed=int(current.seed), partition=partition))
    return pd.DataFrame(rows)


def load_campaign(experiment: int, track: str, phase: str | None = None, *, cfg_override=None) -> Campaign:
    cfg = cfg_override if cfg_override is not None else load_train_config(config_path=str(config_path(experiment, track, phase)))
    # Explicit selection prevents a notebook's seed comparison from changing later loaders.
    old_train, old_eval = training_viz._RUN_OVERRIDE, eval_viz._RUN_OVERRIDE
    try:
        training_viz.use_run(str(cfg.run_name))
        eval_viz.use_run(str(cfg.run_name))
        observed = training_viz.load_run_manifest(track, cfg)
        trajectories = load_trajectories(track, cfg)
        histories = training_viz.load_all_epoch_histories(track, cfg)
        evaluation = eval_viz.load_eval_results(track, include_retention=True)
    finally:
        training_viz.use_run(old_train)
        eval_viz.use_run(old_eval)
    trials = planned_trials(cfg)
    if not observed.empty:
        if observed.trial_name.duplicated().any():
            raise ValueError("Multiple current records for one trial")
        extras = observed[~observed.trial_name.isin(trials.trial_name)]
        if not extras.empty:
            raise ValueError("Recorded trials do not match this notebook's config; select the original config")
        columns = [c for c in observed if c == "trial_name" or c not in trials]
        trials = trials.merge(observed[columns], on="trial_name", how="left", validate="one_to_one")
    if "status" not in trials:
        trials["status"] = "PENDING"
    trials["status"] = trials.status.fillna("PENDING")
    history = pd.concat([v.assign(trial_name=k) for k, v in histories.items()], ignore_index=True) if histories else pd.DataFrame()
    return Campaign(cfg, trials, trajectories, history, evaluation)


def coverage_summary(campaign: Campaign) -> str:
    counts = campaign.trials.status.value_counts().sort_index()
    return (f"Run {campaign.cfg.run_name}; {campaign.track.upper()}; {len(campaign.trials)} planned trials; "
            f"target {campaign.target:,} successful updates.\n" + counts.to_string()
            + f"\n{len(campaign.trajectories)} trajectory records; {len(campaign.evaluation)} benchmark fold records."
            + "\nPENDING means no recorded outcome; it can include running or queued work. This is not live scheduler state.")


def effects(campaign: Campaign, split="test") -> pd.DataFrame:
    return trajectory_effects(campaign.trajectories, campaign.track, split=split)


def endpoint_effects(campaign: Campaign, split="test") -> pd.DataFrame:
    data = effects(campaign, split)
    return data[data.updates.eq(campaign.target)].copy() if not data.empty else data


def effect_summary(data: pd.DataFrame, column="effect") -> str:
    if data.empty or column not in data:
        return "No matched measurements available; absence is not a zero effect."
    group = [c for c in ("base", "sampling", "frozen") if c in data]
    if not group:
        return data[column].describe().to_string()
    return data.groupby(group, dropna=False)[column].agg(["count", "mean", "median", "min", "max"]).round(5).to_string()


def _subplots(title, ncols=1, *, sharey=False):
    fig, axes = plt.subplots(1, ncols, figsize=style.figsize(style.WIDTH_FULL, style.PANEL_RATIO),
                             sharey=sharey, squeeze=False, layout="constrained")
    fig.suptitle(title)
    return fig, axes[0]


def _recipe_label(lr, lam):
    return f"{float(lr):.0e} / {float(lam):g}"


def _heatmap(matrix, title, *, diverging=True, limits=None, label="Effect"):
    fig, axes = _subplots(title)
    ax = axes[0]
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if limits is None:
        extent = max(float(np.max(np.abs(finite))) if len(finite) else 0, 1e-8)
        limits = (-extent, extent) if diverging else (0, max(extent, 1))
    cmap = plt.get_cmap(style.CMAP_DIVERGING if diverging else style.CMAP_SEQUENTIAL).copy()
    cmap.set_bad(style.color("annotation"), alpha=0.15)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=limits[0], vmax=limits[1], aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [str(x) for x in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), [str(x) for x in matrix.index])
    ax.grid(False)
    if values.size <= style.PAGE_ROWS * style.PAGE_COLUMNS:
        for y, x in np.ndindex(values.shape):
            if np.isfinite(values[y, x]):
                norm = (values[y, x] - limits[0]) / max(limits[1] - limits[0], 1e-12)
                red, green, blue, _ = cmap(norm)
                ink = style.GRID_TEXT_LIGHT if .299*red + .587*green + .114*blue < .5 else style.GRID_TEXT_DARK
                ax.text(x, y, f"{values[y,x]:.3g}", ha="center", va="center", fontsize=style.ANNOTATION_SIZE, color=ink)
    bar = fig.colorbar(image, ax=ax, label=label, shrink=.8)
    if not diverging and limits == (0, 1) and set(finite).issubset({0., 1.}):
        bar.set_ticks([0, 1])
    return fig


def plot_coverage(campaign: Campaign) -> list[Page]:
    table = campaign.trials.groupby(["base", "status"]).size().unstack(fill_value=0)
    fig, axes = _subplots(f"{campaign.track.upper()}: planned trial coverage")
    left = np.zeros(len(table))
    for status in style.STATUS_COLORS:
        if status in table:
            axes[0].barh(table.index, table[status], left=left, label=status, color=style.STATUS_COLORS[status])
            left += table[status].to_numpy()
    axes[0].set_xlabel("Number of planned trials")
    axes[0].legend(ncol=3)
    return [Page(f"coverage_{campaign.track}", fig,
        f"Recorded status of {len(campaign.trials)} planned {campaign.track.upper()} trials by base model. PENDING denotes no manifest outcome and can include currently running or queued jobs. INTERRUPTED denotes saved progress, not a numerical failure. This figure does not query the scheduler.", {"Planned trial counts": table})]


def plot_coverage_grid(campaign: Campaign) -> list[Page]:
    pages = []
    data = campaign.trials.assign(recipe=[_recipe_label(lr, lam) for lr, lam in zip(campaign.trials.learning_rate, campaign.trials.l2sp_lambda)])
    for (base, frozen, sampling), group in data.groupby(["base", "frozen", "sampling"]):
        if group.status.eq("PENDING").all():
            continue  # The aggregate coverage figure already describes unstarted work.
        from matplotlib.colors import ListedColormap
        states = list(style.STATUS_COLORS)
        codes = dict(zip(states, range(len(states))))
        statuses = group.pivot(index="recipe", columns="partition", values="status")
        recipe_order = group.sort_values(["learning_rate", "l2sp_lambda"]).recipe.drop_duplicates()
        statuses = statuses.reindex(recipe_order)
        statuses.columns = [f"Fold {int(c)+1}" for c in statuses.columns]
        fig, axes = _subplots(f"{base}: {'frozen' if frozen else 'full'} / {style.SAMPLING_LABELS[sampling]}")
        matrix = statuses.apply(lambda column: column.map(codes)).to_numpy(dtype=float)
        color_map = ListedColormap([style.STATUS_COLORS[s] for s in states])
        image = axes[0].imshow(matrix, cmap=color_map, vmin=-.5, vmax=len(states)-.5, aspect="auto")
        axes[0].set_xticks(range(len(statuses.columns)), statuses.columns)
        axes[0].set_yticks(range(len(statuses.index)), statuses.index)
        axes[0].grid(False)
        for y,x in np.ndindex(matrix.shape):
            axes[0].text(x,y,statuses.iloc[y,x],ha="center",va="center",fontsize=style.ANNOTATION_SIZE)
        bar = fig.colorbar(image,ax=axes[0],ticks=range(len(states)),shrink=.8)
        bar.ax.set_yticklabels(states)
        pages.append(Page(f"coverage_grid_{base}_{frozen}_{sampling}", fig,
            "Latest status of each configured recipe and dataset partition. Row labels give peak learning rate / L2-SP; failed, divergent, interrupted and pending identities retain distinct labels. Entirely unstarted groups are shown in aggregate coverage rather than repeated empty grids."))
    return pages


def trajectory_curve(campaign, recipe, *, split="test", x="updates"):
    """Require the planned trials and the same finite baseline observations at every milestone."""
    planned = campaign.trials.merge(recipe[FACTORS].drop_duplicates(), on=FACTORS, validate="many_to_one")
    baseline = recipe[recipe.updates.eq(0)]
    expected = set(zip(baseline.trial, baseline.dataset))
    baseline_complete = set(baseline.trial) == set(planned.trial_name)
    count_column = {"test": "n_test_datasets", "train": "n_train_datasets"}.get(split)
    if count_column and count_column in planned and planned[count_column].notna().all():
        baseline_complete &= len(baseline) == int(planned[count_column].sum())
    rows = []
    for update in sorted({0, *map(int, campaign.cfg.train.trajectory_steps)}):
        point = recipe[recipe.updates.eq(update)]
        complete = baseline_complete and bool(expected) and set(zip(point.trial, point.dataset)) == expected
        values = point.groupby("dataset").effect.mean()
        position = update if x == "updates" else point[x].median()
        rows.append(dict(x=position, updates=update, mean=values.mean() if complete else np.nan,
                         q25=values.quantile(.25) if complete else np.nan,
                         q75=values.quantile(.75) if complete else np.nan,
                         datasets=len(values), observations=len(point), complete=complete))
    return pd.DataFrame(rows)


def plot_trajectory_pages(campaign: Campaign, *, x="updates", split="test") -> list[Page]:
    data = effects(campaign, split)
    if data.empty:
        return []
    pages = []
    for (base, frozen, sampling), group in data.groupby(["base", "frozen", "sampling"]):
        curves = {}
        for (lam, lr), recipe in group.groupby(["l2sp_lambda", "learning_rate"]):
            curves[(lam, lr)] = trajectory_curve(campaign, recipe, split=split, x=x).assign(learning_rate=lr, l2sp_lambda=lam)
        lambdas = sorted({lam for (lam, lr), curve in curves.items()
                          if np.isfinite(curve.loc[curve.updates.gt(0), "mean"]).any()})
        if not lambdas:
            continue
        fig, axes = _subplots(f"{base}: {'frozen backbone' if frozen else 'full updates'}", len(lambdas), sharey=True)
        records = list(curves.values())
        for ax, lam in zip(axes, lambdas):
            for i, (lr, recipe) in enumerate(group[group.l2sp_lambda.eq(lam)].groupby("learning_rate")):
                curve = curves[(lam, lr)]
                color = style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr)))
                ax.plot(curve.x, curve["mean"], marker=style.PANEL_MARKERS[i % 4], label=f"LR {lr:.0e}", color=color)
                ax.fill_between(curve.x.to_numpy(float), curve.q25.to_numpy(float), curve.q75.to_numpy(float),
                                color=color, alpha=style.INTERVAL_ALPHA)
            ax.axhline(0, color=style.color("reference"), linestyle=":")
            ax.set_title(f"L2-SP = {lam:g}")
            ax.set_xlabel("Successful updates" if x == "updates" else "Median processed row exposures")
            ax.legend()
        axes[0].set_ylabel(effect_label(campaign.metric))
        pages.append(Page(f"trajectory_{split}_{base}_{frozen}_{sampling}_{x}", fig,
            f"{campaign.track.upper()} monitoring on {'non-credit' if split == 'ood' else 'held-out' if split == 'test' else 'training'} tables, "
            "paired with update zero within trial and dataset. Lines show the equally weighted dataset mean; "
            "shading spans dataset quartiles and is not a confidence interval. Repeated observations are averaged within dataset first. "
            "A point requires baseline coverage of every planned trial in that recipe and unchanged observation support at the milestone. "
            "Missing points are gaps, not interpolated improvements. Processed row exposures include repeated rows.",
            {"Curve statistics and coverage": pd.concat(records, ignore_index=True),
             "Paired dataset observations": group.reset_index(drop=True)}))
    return pages


def plot_response_surfaces(data: pd.DataFrame, metric: str, *, value="effect") -> list[Page]:
    if data.empty:
        return []
    pages = []
    for (base, sampling), group in data.groupby(["base", "sampling"]):
        per = group.groupby(["frozen", "learning_rate", "l2sp_lambda", "dataset"])[value].mean().reset_index()
        means = per.groupby(["frozen", "learning_rate", "l2sp_lambda"])[value].mean()
        counts = per.groupby(["frozen", "learning_rate", "l2sp_lambda"]).dataset.nunique()
        extent = max(float(means.abs().max()), 1e-8)
        for frozen in sorted(group.frozen.unique()):
            matrix = means.loc[frozen].unstack("l2sp_lambda").sort_index()
            n = counts.loc[frozen].unstack("l2sp_lambda").reindex_like(matrix)
            matrix.index = [f"{lr:.0e}" for lr in matrix.index]
            matrix.columns = [f"L2-SP {lam:g}" for lam in matrix.columns]
            fig = _heatmap(matrix, f"{base}: {'frozen' if frozen else 'full'} / {style.SAMPLING_LABELS[sampling]}",
                           limits=(-extent, extent), label=effect_label(metric))
            present = np.isfinite(matrix.to_numpy())
            for text, count in zip(fig.axes[0].texts, n.to_numpy()[present]):
                text.set_text(text.get_text()+f"\nn={int(count)}")
            n.index, n.columns = matrix.index, matrix.columns
            pages.append(Page(f"response_{base}_{frozen}_{sampling}", fig,
                "Equally weighted means of paired dataset effects at the prescribed endpoint. "
                "Each cell reports its dataset count; unequal counts indicate partial support and prevent an unqualified recipe ranking. "
                "The full and frozen panels within each base share a symmetric color scale. Positive effects favor adaptation. "
                "Blank cells have no matched observations, not zero effect.",
                {"Mean effect": matrix, "Dataset counts": n}))
    return pages


def plot_dataset_pages(data: pd.DataFrame, metric: str, *, value="effect") -> list[Page]:
    if data.empty:
        return []
    pages = []
    for (base, frozen, sampling), group in data.groupby(["base", "frozen", "sampling"]):
        group = group.assign(recipe=[_recipe_label(lr, lam) for lr, lam in zip(group.learning_rate, group.l2sp_lambda)])
        matrix = group.pivot_table(index="dataset", columns="recipe", values=value, aggfunc="mean").sort_index()
        recipe_order = group.sort_values(["learning_rate", "l2sp_lambda"]).recipe.drop_duplicates()
        matrix = matrix.reindex(columns=recipe_order)
        extent = max(float(np.nanmax(np.abs(matrix.to_numpy()))), 1e-8)
        for page_no, (start, stop) in enumerate(style.page_slices(len(matrix)), 1):
            page = matrix.iloc[start:stop]
            fig = _heatmap(page, f"{base}: {'frozen' if frozen else 'full'} / datasets {start+1}–{start+len(page)}", limits=(-extent, extent), label=effect_label(metric))
            pages.append(Page(f"datasets_{base}_{frozen}_{sampling}_{page_no}", fig,
                f"Paired effects for {len(page)} datasets, one column per peak-learning-rate / L2-SP recipe. "
                "Panels fix model, adaptation and sampling; pages use a common scale within that combination. No dataset is discarded to shorten the display."))
    return pages


def factor_contrasts(data: pd.DataFrame, factor: str, reference, alternative, *, value="effect") -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    keys = [c for c in FACTORS + ["dataset", "seed", "partition", "fold_idx", "updates"] if c in data and c != factor]
    grouped = data.groupby(keys + [factor], dropna=False)[value].mean().reset_index()
    paired = grouped[grouped[factor].eq(reference)].merge(grouped[grouped[factor].eq(alternative)], on=keys, suffixes=("_reference", "_alternative"), validate="one_to_one")
    paired["contrast"] = paired[f"{value}_alternative"] - paired[f"{value}_reference"]
    return paired


def plot_contrasts(data: pd.DataFrame, metric: str, label: str) -> list[Page]:
    if data.empty:
        return []
    pages = []
    for base, group in data.groupby("base"):
        fig, axes = _subplots(f"{base}: {label}")
        by_dataset = group.groupby(["learning_rate", "dataset"]).contrast.mean().reset_index()
        lrs = sorted(by_dataset.learning_rate.unique())
        for i, lr in enumerate(lrs):
            vals = by_dataset.loc[by_dataset.learning_rate.eq(lr), "contrast"]
            axes[0].scatter(np.full(len(vals), i), vals, color=style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr))), alpha=style.POINT_ALPHA, s=style.POINT_SIZE)
            axes[0].plot([i-.18, i+.18], [vals.mean()]*2, color=style.color("reference"))
        axes[0].set_xticks(range(len(lrs)), [f"{lr:.0e}" for lr in lrs])
        axes[0].set_xlabel("Peak learning rate")
        axes[0].set_ylabel(f"Difference in {effect_label(metric)}")
        axes[0].axhline(0, color=style.color("reference"), linestyle=":")
        pages.append(Page(f"contrast_{label}_{base}", fig,
            f"{label}: matched changes in paired effects. Each point is a dataset, after averaging remaining repeated settings within dataset; short horizontal lines are dataset means. "
            "All other scientific factors are matched before computing differences. These are descriptive contrasts, not independent-replication confidence intervals."))
    return pages


def _complete_folds(data: pd.DataFrame, metric: str, *, fractional_reference=False) -> pd.DataFrame:
    """Require the exact configured fold set and finite, interpretable metric values."""
    expected_folds = set(range(int(OmegaConf.load(REPO_ROOT / "config/eval.yaml").cv.n_folds)))
    keys = [c for c in ("eval_run", "split", "method_dirname", "test_dataset_id") if c in data]
    if data.duplicated([*keys, "fold_idx"]).any():
        raise ValueError("Duplicate evaluation folds: reconcile outcomes before analysis")
    values = pd.to_numeric(data[metric], errors="coerce")
    valid = data.status.eq("OK") & np.isfinite(values)
    if fractional_reference:
        # A perfect untuned fold has no defined fractional reduction. Do not
        # silently turn its dataset into an average over the remaining folds.
        valid &= ~data.source.str.endswith("-untuned", na=False) | values.gt(0)
    good = data[valid].copy()
    good[metric] = values[valid]
    complete = good.groupby(keys).fold_idx.agg(lambda folds: set(folds) == expected_folds).rename("complete").reset_index()
    return good.merge(complete[complete.complete][keys], on=keys, how="inner", validate="many_to_one")


def benchmark_effects(campaign: Campaign, metric: str | None = None, *, domain="credit") -> pd.DataFrame:
    """Paired outer-fold effects with a complete-fold requirement per dataset/recipe."""
    metric = metric or campaign.metric
    data = campaign.evaluation
    if "domain" in data:
        data = data[data.domain.fillna("credit").eq(domain)]
    if data.empty or metric not in data:
        return pd.DataFrame()
    good = _complete_folds(data, metric, fractional_reference=metric == "rmse")
    paired = paired_deltas(good, metric)
    if paired.empty:
        return paired
    rows = []
    for _, row in paired.iterrows():
        meta = eval_viz._decode_method_dirname(row.method_dirname)
        rows.append(dict(row, dataset=row.test_dataset_id, base=training_viz.compact_base(meta["base_short"]),
            learning_rate=meta["lr"], frozen=meta["use_lora"], l2sp_lambda=meta["l2sp_lambda"],
            sampling=meta["epoch_pass_mode"], effect=row.delta, seed=int(campaign.cfg.seed)))
    return pd.DataFrame(rows)


def plot_eval_coverage(campaign: Campaign) -> list[Page]:
    data = campaign.evaluation
    if data.empty:
        return []
    data = data.copy()
    data["group"] = data.apply(lambda r: training_viz.compact_base(str(r.base_short)) if str(r.source).endswith(("-trained", "-untuned")) else str(r.source), axis=1)
    counts = data.groupby(["group", "status"]).size().unstack(fill_value=0)
    fig, axes = _subplots(f"{campaign.track.upper()}: recorded evaluation folds")
    left = np.zeros(len(counts))
    for status in counts:
        axes[0].barh(counts.index, counts[status], left=left, label=status, color=style.STATUS_COLORS.get(status, style.color("annotation")))
        left += counts[status].to_numpy()
    axes[0].legend()
    axes[0].set_xlabel("Recorded outer-fold outcomes")
    return [Page("evaluation_coverage", fig, "Recorded outer-fold outcomes by model group. This counts records rather than inferring completion from missing files. Subsequent effects require all configured outer folds for each model–dataset cell and its matching untuned control.")]


def diagnostics(campaign: Campaign) -> pd.DataFrame:
    data = campaign.trials.copy()
    if campaign.histories.empty:
        return data
    h = campaign.histories
    available = [c for c in ("training_seconds", "compute_seconds", "data_wait_seconds", "amp_skipped_steps", "data_skipped_steps") if c in h]
    totals = h.groupby("trial_name")[available].sum(min_count=1)
    if "training_seconds" in totals:
        for field in ("compute_seconds", "data_wait_seconds"):
            if field in totals:
                totals[field.replace("seconds", "fraction")] = totals[field] / totals.training_seconds.where(totals.training_seconds > 0)
    return data.merge(totals, on="trial_name", how="left", validate="one_to_one")


def plot_diagnostics(campaign: Campaign) -> list[Page]:
    data = diagnostics(campaign)
    fields = {"sec_per_step": "Seconds per optimizer update", "final_drift": "Final relative weight drift",
              "peak_gpu_gb": "Peak allocated tensor memory (GB)", "gpu_hours": "Training wall time (GPU hours)",
              "data_wait_fraction": "Data-wait / training time", "amp_skipped_steps": "Skipped AMP updates"}
    pages = []
    for column, label in fields.items():
        if column not in data or not pd.to_numeric(data[column], errors="coerce").notna().any():
            continue
        fig, axes = _subplots(f"{campaign.track.upper()}: {label}")
        labels = []
        for i, ((base, frozen), group) in enumerate(data.groupby(["base", "frozen"])):
            vals = pd.to_numeric(group[column], errors="coerce").dropna()
            labels.append(f"{base} / {'frozen' if frozen else 'full'}")
            jitter = np.linspace(-style.JITTER_WIDTH, style.JITTER_WIDTH, len(vals)) if len(vals)>1 else np.zeros(len(vals))
            axes[0].scatter(vals, i+jitter, color=style.color(base), alpha=style.POINT_ALPHA, s=style.POINT_SIZE)
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].set_xlabel(label)
        axes[0].set_xlim(left=0)
        pages.append(Page(f"diagnostic_{column}", fig,
            f"{label} by base and adaptation. Each point is a recorded trial, including unsuccessful attempts where the measurement is available. "
            "Vertical offsets separate coincident marks and have no numerical meaning. Unknown values are omitted, not replaced with zero.",
            {"Trial measurements": data[["trial_name","base","frozen","status",column]].reset_index(drop=True)}))
    return pages


def optimization_window(campaign: Campaign) -> int:
    """Avoid empty windows finer than the observed epoch recording interval."""
    history = campaign.histories
    width = max(1, int(np.ceil(campaign.target / style.CURVE_BINS)))
    if {"trial_name", "successful_updates"} <= set(history):
        gaps = history.sort_values(["trial_name", "successful_updates"]).groupby("trial_name").successful_updates.diff()
        gaps = gaps[gaps.gt(0)]
        if not gaps.empty:
            width = max(width, int(np.ceil(gaps.median())))
    return width


def loss_epoch_support(campaign: Campaign) -> pd.DataFrame:
    """Retain loss provenance without joining means over different table subsets."""
    h = campaign.histories
    if h.empty or "train_loss" not in h:
        return pd.DataFrame()
    data = h[["trial_name", "successful_updates", "train_loss"]].copy()
    columns = [c for c in h if c.startswith("loss__")]
    if columns:
        data["observed_tables"] = h[columns].notna().sum(axis=1)
        data["full_observed_tables"] = data.groupby("trial_name").observed_tables.transform("max")
        data["complete_pass"] = data.observed_tables.eq(data.full_observed_tables)
    else:
        data["observed_tables"] = np.nan
        data["full_observed_tables"] = np.nan
        data["complete_pass"] = True  # coverage unknown; do not infer a missing table
    data["train_loss"] = pd.to_numeric(data.train_loss, errors="coerce")
    data = data[np.isfinite(data.train_loss) & data.successful_updates.gt(0)]
    return data.merge(campaign.trials[["trial_name", *FACTORS]], on="trial_name", validate="many_to_one")


def loss_coverage_tables(campaign: Campaign) -> dict[str, pd.DataFrame]:
    """Report coverage and exceptions without duplicating the full epoch archive."""
    data = loss_epoch_support(campaign)
    if data.empty:
        return {"Loss-coverage availability": pd.DataFrame()}
    grouped = data.groupby(["trial_name", *FACTORS], dropna=False)
    coverage = grouped.agg(
        recorded_epochs=("successful_updates", "size"),
        full_coverage_epochs=("complete_pass", "sum"),
        fullest_observed_tables=("full_observed_tables", "max"),
        min_observed_tables=("observed_tables", "min"),
        last_update=("successful_updates", "max"),
    ).reset_index()
    coverage["partial_coverage_epochs"] = coverage.recorded_epochs - coverage.full_coverage_epochs
    return {"Loss coverage by trial": coverage,
            "Partial-coverage epoch losses": data[~data.complete_pass].reset_index(drop=True)}


def optimization_curves(campaign: Campaign, field: str) -> pd.DataFrame:
    """Bound dense epoch histories without giving prolific histories more trial weight."""
    h = campaign.histories
    if h.empty or not {field,"successful_updates","trial_name"} <= set(h):
        return pd.DataFrame()
    if field == "train_loss":
        support = loss_epoch_support(campaign)
        data = support[support.complete_pass].copy()
    else:
        data = h[["trial_name","successful_updates",field]].copy()
        data = data.merge(campaign.trials[["trial_name",*FACTORS]],on="trial_name",validate="many_to_one")
    data[field] = pd.to_numeric(data[field], errors="coerce")
    data = data[np.isfinite(data[field]) & data.successful_updates.gt(0)]
    width = optimization_window(campaign)
    data["window_end"] = np.minimum(np.ceil(data.successful_updates/width)*width, campaign.target)
    return data.groupby(["trial_name",*FACTORS,"window_end"],dropna=False)[field].mean().reset_index()


def plot_optimization(campaign: Campaign, field="train_loss") -> list[Page]:
    if field == "lr_applied":
        from src.visualize.pilots import plot_schedule
        return plot_schedule([campaign])
    labels = {"train_loss": "Data loss", "clipped_frac": "Fraction of clipped updates",
              "grad_norm_mean": "Mean gradient norm", "lr_applied": "Applied learning rate",
              "l2sp_penalty": "L2-SP penalty", "weight_drift": "Relative weight drift"}
    data = optimization_curves(campaign, field)
    if data.empty:
        return []
    pages = []
    for (base, frozen, sampling), group in data.groupby(["base", "frozen", "sampling"]):
        lambdas = sorted(group.l2sp_lambda.unique())
        fig, axes = _subplots(f"{base}: {labels[field]} / {'frozen' if frozen else 'full'}", len(lambdas), sharey=True)
        tables = []
        for ax, lam in zip(axes, lambdas):
            for i, (lr, recipe) in enumerate(group[group.l2sp_lambda.eq(lam)].groupby("learning_rate")):
                curve = recipe.groupby("window_end")[field].agg(
                    median="median", q25=lambda v: v.quantile(.25), q75=lambda v: v.quantile(.75), trials="size")
                width = optimization_window(campaign)
                curve = curve.reindex(np.unique(np.minimum(np.arange(width, campaign.target+width, width), campaign.target)))
                color = style.TRAJECTORY_LR_COLORS.get(lr, style.color(str(lr)))
                ax.plot(curve.index, curve["median"], marker=style.PANEL_MARKERS[i % 4], markersize=style.CURVE_MARKER_SIZE, label=f"LR {lr:.0e}", color=color)
                ax.fill_between(curve.index.to_numpy(float), curve.q25.to_numpy(float), curve.q75.to_numpy(float),
                                color=color, alpha=style.INTERVAL_ALPHA)
                tables.append(curve.rename_axis("window_end").reset_index().assign(learning_rate=lr, l2sp_lambda=lam))
            ax.set_title(f"L2-SP = {lam:g}"); ax.set_xlabel("Successful-update window end"); ax.legend()
        axes[0].set_ylabel(labels[field])
        if field == "clipped_frac":
            axes[0].set_ylim(0,1)
        pages.append(Page(f"optimization_{campaign.track}_{field}_{base}_{frozen}_{sampling}", fig,
            f"{labels[field]} summarized in successful-update windows of width {optimization_window(campaign):,}, with a shorter final window when necessary. "
            "Epoch values are averaged within trial and window; lines and bands show the median and interquartile range "
            "across available trials. Counts are retained in the text summary. Gaps are not interpolated. "
            "The data loss excludes the separately reported L2-SP term; numerical loss scales differ across tasks, architectures "
            "and target transforms, and a density NLL may be negative. Loss curves exclude epochs with fewer "
            "observed tables than the same trial's fullest recorded traversal; these partial-coverage values are tabulated separately.",
            {"Window statistics": pd.concat(tables, ignore_index=True)}))
    return pages


def plot_train_test(campaign: Campaign) -> list[Page]:
    held, seen = endpoint_effects(campaign), endpoint_effects(campaign, "train")
    if held.empty or seen.empty:
        return []
    keys = ["trial", *FACTORS]
    pair = held.groupby(keys).effect.mean().rename("held").reset_index().merge(
        seen.groupby(keys).effect.mean().rename("seen").reset_index(), on=keys, validate="one_to_one")
    pages = []
    for base, group in pair.groupby("base"):
        fig, axes = _subplots(f"{base}: seen versus held-out monitoring")
        for frozen, arm in group.groupby("frozen"):
            axes[0].scatter(arm.seen, arm.held, marker="s" if frozen else "o", label="Frozen" if frozen else "Full", s=style.POINT_SIZE, alpha=style.POINT_ALPHA)
        axes[0].axhline(0, color=style.color("reference"), linestyle=":")
        axes[0].axvline(0, color=style.color("reference"), linestyle=":")
        axes[0].set_xlabel("Mean effect on training tables")
        axes[0].set_ylabel("Mean effect on held-out tables")
        axes[0].legend()
        pages.append(Page(f"seen_held_{base}", fig,
            "Endpoint changes from each table's update-zero monitor, averaged separately over training and held-out tables. Each point is one trained trial. "
            "The two table sets differ, so this is a transfer diagnostic rather than a conventional within-dataset generalization gap."))
    return pages


def null_audits(campaigns: list[Campaign]) -> pd.DataFrame:
    """Read completed audit reports; never infer saved-weight equality from plotted drift."""
    wanted = {(str(c.cfg.run_name), c.track) for c in campaigns}
    reports = {}
    from src.visualize.inputs import analysis_root
    root = analysis_root()
    folder = root / "experiment0" / "logs" if root is not None else logs_dir("experiment0")
    for path in sorted(folder.glob("maintenance_*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"^\{", text, flags=re.MULTILINE):
            try:
                report, _ = json.JSONDecoder().raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            key = (report.get("run"), report.get("track"))
            if key in wanted and "trials" in report:
                reports[key] = (report, "END exit_code=0" in text)
    from src.utils.paths import manifests_dir
    workflow = root / "experiment0/manifests/workflow" if root is not None else manifests_dir("experiment0") / "workflow"
    for path in sorted(workflow.glob("*/null_audit.json"), key=lambda p: p.stat().st_mtime_ns):
        for report in json.loads(path.read_text(encoding="utf-8")):
            key = (report.get("run"), report.get("track"))
            if key in wanted:
                reports[key] = (report, bool(report.get("passed")))
    rows = []
    for (run, track), (report, exit_ok) in reports.items():
        for trial in report["trials"]:
            parsed = training_viz.parse_trial_name(trial["trial"])
            rows.append(dict(track=track, base=training_viz.compact_base(parsed.base_short), frozen=parsed.lora,
                weight_equal=trial.get("null_state", {}).get("equal", False),
                monitor_equal=trial.get("null_monitor_equal", False), updates=trial.get("successful_updates"),
                audit_passed=bool(report.get("passed") and exit_ok)))
    return pd.DataFrame(rows)


def plot_null_audits(frame: pd.DataFrame) -> list[Page]:
    if frame.empty:
        return []
    pages = []
    for track, data in frame.groupby("track"):
        matrix = data.assign(label=data.base + data.frozen.map({True: " / frozen", False: " / full"})).set_index("label")[["weight_equal", "monitor_equal", "audit_passed"]].astype(int)
        matrix.columns = ["Saved weights", "Monitor parity", "Audit passed"]
        fig = _heatmap(matrix, f"{track.upper()}: zero-LR controls", diverging=False, limits=(0,1), label="Passed (1) / failed (0)")
        pages.append(Page(f"null_audit_{track}", fig, "Recorded CPU audit checks on saved zero-learning-rate checkpoints: canonical model/inference-state equality, per-dataset monitoring parity and whole-audit outcome. Missing audit logs are not treated as successful checks."))
    return pages


def plot_null_monitor(campaign: Campaign) -> list[Page]:
    rows = []
    for split in ("train", "test"):
        data = endpoint_effects(campaign, split)
        if data.empty:
            continue
        for (base, frozen), group in data.groupby(["base", "frozen"]):
            rows.append(dict(label=f"{base} / {'frozen' if frozen else 'full'}", split=split, difference=group.effect.abs().max()))
    if not rows:
        return []
    matrix = pd.DataFrame(rows).pivot(index="label",columns="split",values="difference")
    fig = _heatmap(matrix,f"{campaign.track.upper()}: largest null-monitor change",diverging=False,label="Maximum absolute paired effect")
    return [Page(f"null_monitor_{campaign.track}",fig,"Largest absolute change from update zero over recorded datasets, separately for training and held-out tables. Zero indicates unchanged monitoring scores. This check complements rather than replaces exact saved-state audits.")]


def reference_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Predefined recipe, independent of observed performance."""
    if frame.empty:
        return frame
    return frame[np.isclose(frame.learning_rate, 3e-7, rtol=1e-10, atol=0) &
                 np.isclose(frame.l2sp_lambda, .003) & ~frame.frozen & frame.sampling.eq("one_sample")].copy()


def seed_pairs(main: Campaign, repeat: Campaign, *, benchmark=False) -> pd.DataFrame:
    left = reference_rows(benchmark_effects(main) if benchmark else effects(main))
    right = reference_rows(benchmark_effects(repeat) if benchmark else effects(repeat))
    if left.empty or right.empty:
        return pd.DataFrame()
    keys = [*FACTORS, "dataset"] + ([] if benchmark else ["updates"])
    left = left.groupby(keys, dropna=False).effect.mean().rename("reference").reset_index()
    right = right.groupby(keys, dropna=False).effect.mean().rename("repeat").reset_index()
    pair = left.merge(right, on=keys, validate="one_to_one")
    pair["difference"] = pair.repeat - pair.reference
    pair["seed_reference"], pair["seed_repeat"] = int(main.cfg.seed), int(repeat.cfg.seed)
    return pair


def plot_seed_pairs(pairs: pd.DataFrame, metric: str, *, endpoint: int | None = None) -> list[Page]:
    if pairs.empty:
        return []
    data = pairs[pairs.updates.eq(endpoint)] if endpoint is not None and "updates" in pairs else pairs
    pages = []
    groups = ((base, group.sort_values("dataset").iloc[start:stop], page_no)
              for base, group in data.groupby("base")
              for page_no, (start, stop) in enumerate(style.page_slices(len(group)), 1))
    for base, group, page_no in groups:
        fig, axes = _subplots(f"{base}: matched training seeds", 2)
        axes[0].scatter(group.reference, group.repeat, color=style.color(base), s=style.POINT_SIZE)
        low = min(group.reference.min(), group.repeat.min(), 0)
        high = max(group.reference.max(), group.repeat.max(), 0)
        if low == high:
            low, high = low-1e-5, high+1e-5
        axes[0].plot([low,high], [low,high], color=style.color("reference"), linestyle=":")
        axes[0].set_xlabel(f"Seed {int(group.seed_reference.iloc[0])}: effect")
        axes[0].set_ylabel(f"Seed {int(group.seed_repeat.iloc[0])}: effect")
        ordered = group.sort_values("dataset")
        axes[1].scatter(ordered.difference, range(len(ordered)), color=style.color(base), s=style.POINT_SIZE)
        axes[1].set_yticks(range(len(ordered)), ordered.dataset)
        axes[1].axvline(0, color=style.color("reference"), linestyle=":")
        axes[1].set_xlabel("Repeat minus reference effect")
        pages.append(Page(f"seed_pair_{base}_{page_no}", fig,
            f"Matched dataset effects at two training seeds for the predefined full-update, LR 3e-7, L2-SP 0.003 reference. Effects use {effect_label(metric)}. "
            "Both seeds must be available for a dataset. This is a sensitivity check at one recipe, not a seed-variance estimate for the grid."))
    return pages


def plot_seed_trajectories(pairs: pd.DataFrame, metric: str, *, milestones=None) -> list[Page]:
    if pairs.empty or "updates" not in pairs:
        return []
    pages = []
    for base, group in pairs.groupby("base"):
        fig, axes = _subplots(f"{base}: seed difference through training")
        updates = sorted(milestones if milestones is not None else pairs.updates.unique())
        for dataset, series in group.groupby("dataset"):
            values = series.set_index("updates").difference.reindex(updates)
            axes[0].plot(values.index, values.values, color=style.color("annotation"), alpha=style.POINT_ALPHA, linewidth=style.THIN_LINE)
        mean = group.groupby("updates").difference.mean()
        expected = set(group.loc[group.updates.eq(0), "dataset"])
        complete = group.groupby("updates").dataset.agg(lambda values: set(values) == expected and bool(expected))
        mean = mean.where(complete).reindex(updates)
        axes[0].plot(mean.index, mean, color=style.color(base), label="Dataset mean", marker="o")
        axes[0].axhline(0, color=style.color("reference"), linestyle=":")
        axes[0].set_xlabel("Successful optimizer updates")
        axes[0].set_ylabel("Repeat minus reference effect")
        axes[0].legend()
        pages.append(Page(f"seed_trajectory_{base}", fig, "Change in the paired seed difference across recorded update milestones. Thin gray lines are matched datasets; the colored line is their equally weighted mean. Positive values favor the additional seed, rather than indicating a larger average pretraining benefit. Missing milestones remain gaps.",
            {"Matched seed observations": group.reset_index(drop=True),
             "Dataset mean difference": mean.rename("difference").rename_axis("updates").reset_index()}))
    return pages


def plot_sampling_trajectories(campaign: Campaign, *, row_exposure=False) -> list[Page]:
    data = effects(campaign)
    if data.empty:
        return []
    pages = []
    for base, group in data.groupby("base"):
        fig, axes = _subplots(f"{base}: sampling protocols")
        records = []
        for mode in style.SAMPLING_LABELS:
            arm = group[group.sampling.eq(mode)]
            if arm.empty:
                continue
            curve = trajectory_curve(campaign, arm, x="processed_rows" if row_exposure else "updates")
            records.append(curve.assign(sampling=mode))
            axes[0].plot(curve.x, curve["mean"], marker="o", color=style.SAMPLING_COLORS[mode],
                         linestyle=style.SAMPLING_STYLES[mode], label=style.SAMPLING_LABELS[mode])
        axes[0].axhline(0, color=style.color("reference"), linestyle=":")
        axes[0].set_xlabel("Median processed row exposures" if row_exposure else "Successful optimizer updates")
        axes[0].set_ylabel(effect_label(campaign.metric))
        axes[0].legend()
        pages.append(Page(f"sampling_{base}_{'rows' if row_exposure else 'updates'}", fig,
            "Monitoring effects for one sampled batch, a disjoint full pass with per-batch updates, and a disjoint full pass with one averaged-gradient update. "
            "All three use the experiment-3 prevalence policy. Each point requires every planned trial and unchanged baseline observation coverage. "
            "Missing milestones remain gaps. Row exposures count repeats; equal optimizer-update budgets do not imply equal data or compute budgets.",
            {"Protocol curve statistics and coverage": pd.concat(records,ignore_index=True),
             "Paired dataset effects": group.reset_index(drop=True)}))
    return pages


def plot_sampling_cost(campaign: Campaign) -> list[Page]:
    d = campaign.trials[campaign.trials.status.eq("OK")]
    pages = []
    for column, label in (("sec_per_step", "Seconds per optimizer update"), ("rows_seen", "Processed row exposures"), ("gpu_hours", "Training wall time (GPU hours)")):
        if column not in d or d[column].dropna().empty:
            continue
        fig, axes = _subplots(f"{campaign.track.upper()}: {label}")
        bases = sorted(d.base.unique())
        for i, mode in enumerate(style.SAMPLING_LABELS):
            vals = d[d.sampling.eq(mode)].groupby("base")[column].median().reindex(bases)
            axes[0].bar(np.arange(len(bases))+(i-1)*.25, vals, width=.25, color=style.SAMPLING_COLORS[mode], label=style.SAMPLING_LABELS[mode])
        axes[0].set_xticks(range(len(bases)), bases)
        axes[0].set_ylabel(label)
        axes[0].legend()
        pages.append(Page(f"sampling_cost_{column}", fig, f"Median {label.lower()} over completed trials within each base and sampling protocol. Missing trials are excluded and completion is reported separately. Equal successful-update targets need not have equal cost."))
    return pages


def plot_secondary_tradeoff(campaign: Campaign) -> list[Page]:
    """Pair discrimination/point-error effects with a distinct secondary measure."""
    primary = benchmark_effects(campaign)
    secondary_metric = "brier_score" if campaign.track == "pd" else "mae"
    secondary = benchmark_effects(campaign, secondary_metric)
    if primary.empty or secondary.empty:
        return []
    keys = [*FACTORS, "dataset"]
    paired = primary.merge(secondary[keys+["effect"]], on=keys, suffixes=("_primary", "_secondary"), validate="one_to_one")
    pages = []
    for base, group in paired.groupby("base"):
        fig, axes = _subplots(f"{base}: primary and secondary effects", 2, sharey=True)
        for ax, frozen in zip(axes,(False,True)):
            arm = group[group.frozen.eq(frozen)]
            for lr, recipe in arm.groupby("learning_rate"):
                ax.scatter(recipe.effect_primary,recipe.effect_secondary,label=f"{lr:.0e}",s=style.POINT_SIZE,alpha=style.POINT_ALPHA,color=style.TRAJECTORY_LR_COLORS.get(lr,style.color(str(lr))))
            ax.axhline(0,color=style.color("reference"),linestyle=":")
            ax.axvline(0,color=style.color("reference"),linestyle=":")
            ax.set_xlabel(effect_label(campaign.metric))
            ax.set_title("Frozen backbone" if frozen else "Full updates")
            if not arm.empty:
                ax.legend(title="Peak LR",ncol=2)
        axes[0].set_ylabel(effect_label(secondary_metric))
        pages.append(Page(f"secondary_tradeoff_{base}",fig,
            f"Paired {campaign.metric} and {secondary_metric} effects for identical recipe–dataset observations. Positive directions indicate improvement on both axes. Each point averages the complete outer folds of one dataset; no metric is used to select a winning recipe."))
    return pages


def plot_control_context(campaign: Campaign) -> list[Page]:
    """Show untuned/classical context separately from the dense adapted grid."""
    d = campaign.evaluation
    if "domain" in d:
        d = d[d.domain.fillna("credit").eq("credit")]
    if d.empty or campaign.metric not in d:
        return []
    d = d[d.status.eq("OK") & ~d.source.str.endswith("-trained",na=False)].copy()
    if d.empty:
        return []
    d = _complete_folds(d, campaign.metric)
    d["label"] = d.apply(lambda r: f"Untuned {training_viz.compact_base(r.base_short)}" if r.source.endswith("-untuned") else r.method_name,axis=1)
    scores = d.pivot_table(index="test_dataset_id",columns="label",values=campaign.metric,aggfunc="mean").sort_index()
    # Rank only rows having every displayed control, so missing competitors cannot improve rank.
    scores = scores.dropna()
    if scores.empty:
        return []
    ranks = scores.rank(axis=1,ascending=campaign.track=="lgd",method="average")
    pages = []
    for page_no, (start, stop) in enumerate(style.page_slices(len(ranks)), 1):
        page = ranks.iloc[start:stop]
        fig = _heatmap(page,f"{campaign.track.upper()}: reference model context",diverging=False,limits=(1,len(ranks.columns)),label="Rank within dataset (1 = best)")
        pages.append(Page(f"control_context_{page_no}",fig,
            "Within-dataset ranks of untuned foundation models and classical controls, using complete outer-fold mean scores. Only datasets with every displayed control are included. Ranks provide corpus context; they do not quantify the size of the continued-pretraining effect."))
    return pages


def plot_cost_effect(campaign: Campaign) -> list[Page]:
    data = benchmark_effects(campaign)
    if data.empty or "gpu_hours" not in campaign.trials:
        return []
    costs = campaign.trials[campaign.trials.status.eq("OK")].groupby(FACTORS).gpu_hours.median().rename("gpu_hours").reset_index()
    means = data.groupby(FACTORS).effect.mean().reset_index().merge(costs,on=FACTORS,validate="one_to_one")
    pages = []
    for base, group in means.groupby("base"):
        group = group[group.gpu_hours.gt(0)]
        if group.empty:
            continue
        fig,axes = _subplots(f"{base}: training cost and benchmark effect")
        for frozen,arm in group.groupby("frozen"):
            axes[0].scatter(arm.gpu_hours,arm.effect,marker="s" if frozen else "o",label="Frozen" if frozen else "Full",s=style.POINT_SIZE)
        axes[0].set_xscale("log"); axes[0].set_xlabel("Median training wall time (hours)")
        axes[0].set_ylabel(effect_label(campaign.metric)); axes[0].legend()
        axes[0].axhline(0,color=style.color("reference"),linestyle=":")
        pages.append(Page(f"cost_effect_{base}",fig,"Dataset-mean paired benchmark effect versus median training wall time (hours) across completed partitions of each recipe. The horizontal axis excludes evaluation/HPO cost and is logarithmic. Partial coverage is shown separately; no efficiency frontier is selected."))
    return pages
