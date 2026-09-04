"""Final-benchmark visualisation helpers.

Consumes the wide-format CSVs written by ``scripts/eval_pipeline.py``
(via ``src.eval.benchmark.EvalRow``) at::

    output/results/<TRACK>/<method-dirname>/<run>_<ts>[__ds-<id>].csv

Each row is one ``(model × dataset × fold)`` tuple with all metric
columns side-by-side. We pool every CSV under one DataFrame, then
project / pivot / plot from there.

Method-dirname conventions (mirrored in ``src.eval.benchmark._method_dirname``):
    xgboost, catboost, logreg, linreg                  → classical baselines
    <family>-untuned__<short>                          → reference base weights, no finetune
    <family>-trained__<short>__lr<lr>[__fullpass][__lora|__iclhead]
                                                       → our continued-pretrained variants

``<family>`` is ``tabpfn`` or ``tabicl``. The adaptation tag is
``__lora`` for tabpfn and ``__iclhead`` for tabicl (that family's
freeze-backbone rendering of the same grid axis).

The visualisations here are deliberately exhaustive — the notebook
caller picks which to display.

Two design contracts (mirrors src/utils/training_viz):
    1. Every plot returns a matplotlib Figure.
    2. Empty-disk runs render stub figures with "(no data)"; loaders
       return empty DataFrames.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.visualize import style

LOGGER = logging.getLogger(__name__)

#: Metrics whose floor is 0.5 rather than 0 — a coin flip already scores 0.5, so a bar chart
#: that starts at 0 wastes half its width on unreachable space. Used only to pick an axis
#: floor, never to alter a value.
_CHANCE_AT_HALF = frozenset({"roc_auc", "auc", "accuracy", "balanced_accuracy"})

_REPO = Path(__file__).resolve().parents[2]


# =============================================================================
# Cfg + path resolution
# =============================================================================


def _load_eval_cfg():
    try:
        from omegaconf import OmegaConf
        return OmegaConf.load(_REPO / "config" / "eval.yaml")
    except Exception:  # pragma: no cover  — fallback for missing dep
        from types import SimpleNamespace as _NS

        from src.utils.paths import results_dir
        return _NS(results=_NS(base_dir=str(results_dir())))


def _resolve_paths():
    """Resolve durable-output roots used by the eval pipeline."""
    # Sync data_source so DATA_ROOT etc. matches the rest of the pipeline.
    try:
        from omegaconf import OmegaConf
        from src.utils.paths import apply_data_source_from_cfg
        apply_data_source_from_cfg(OmegaConf.load(_REPO / "config" / "data.yaml"))
    except Exception:  # pragma: no cover
        pass

    from src.utils.paths import resolve_staging_path, results_dir
    cfg = _load_eval_cfg()
    base = str(cfg.results.base_dir) if hasattr(cfg, "results") else str(results_dir())
    return {
        "benchmark_root": resolve_staging_path(base),
    }


# =============================================================================
# Method-name decoding
# =============================================================================


_CLASSICAL_BASELINES = {"xgboost", "catboost", "logreg", "linreg"}


def _decode_method_dirname(d: str) -> dict:
    """Unpack a method directory name into structured fields.

    Returns
    -------
    dict with keys ``source``, ``base_short``, ``lr``, ``use_lora``,
    where each is filled when the dirname encodes it.
    """
    if d in _CLASSICAL_BASELINES:
        return {"source": "baseline", "base_short": d, "lr": np.nan,
                "use_lora": False, "full_pass": False, "min_train_rows": 0, "l2sp_lambda": None}
    m_src = re.match(r"^(?P<src>(?:tabpfn|tabicl)-(?:untuned|trained))__", d)
    if m_src and m_src["src"].endswith("-untuned"):
        return {"source": m_src["src"],
                "base_short": d.removeprefix(m_src["src"] + "__"),
                "lr": np.nan, "use_lora": False, "full_pass": False,
                "min_train_rows": 0, "l2sp_lambda": None}
    if m_src:   # <family>-trained
        rest = d.removeprefix(m_src["src"] + "__")
        # Dirname layout:
        #   <base>[__lr<lr>][__fullpass][__min<rows>][__lora|__iclhead]
        # Tags are stripped BACK-TO-FRONT, so every tag has to be known here: an
        # unrecognised one is absorbed into `base_short` and takes the learning rate
        # with it, which mis-groups every figure instead of failing. (`__iclhead` is
        # the tabicl family's freeze-backbone rendering of the use_lora axis;
        # `__min<rows>` is the corpus-size arm swept since run-8.)
        lora = rest.endswith(("__lora", "__iclhead"))
        rest = rest.removesuffix("__lora").removesuffix("__iclhead")
        # Anchor strength, swept from run-9. Stripped BEFORE `__min` because
        # `_method_dirname` writes it after: ...__min5000__l2sp0.003__lora.
        m_l2 = re.search(r"__l2sp([0-9.eE+-]+)$", rest)
        l2sp_lambda = float(m_l2.group(1)) if m_l2 else None
        if m_l2:
            rest = rest[: m_l2.start()]
        m_rows = re.search(r"__min(\d+)$", rest)
        min_train_rows = int(m_rows.group(1)) if m_rows else 0
        if m_rows:
            rest = rest[: m_rows.start()]
        full_pass = rest.endswith("__fullpass")
        rest = rest.removesuffix("__fullpass")
        m = re.search(r"__lr([0-9eE.+\-]+)$", rest)
        if m:
            lr = float(m.group(1))
            base = rest[: m.start()]
        else:
            lr = np.nan
            base = rest
        return {"source": m_src["src"], "base_short": base, "lr": lr,
                "use_lora": lora, "full_pass": full_pass,
                "min_train_rows": min_train_rows, "l2sp_lambda": l2sp_lambda}
    return {"source": "unknown", "base_short": d, "lr": np.nan,
            "use_lora": False, "full_pass": False, "min_train_rows": 0, "l2sp_lambda": None}


def human_method_name(row: pd.Series) -> str:
    """Compact human-readable label from a row of :func:`load_eval_results`."""
    src = row.get("source", "unknown")
    base = row.get("base_short", "?")
    if src == "baseline":
        return base
    if src.endswith("-untuned"):
        return f"untuned ({base})"
    if src.endswith("-trained"):
        lr = row.get("lr", np.nan)
        # ONE label for the frozen arm, both families. Since 25-08-2026 `use_lora` selects the
        # same operation everywhere — freeze the repeated-block transformer stack, train the
        # embedders and head (src/train/freeze.py) — so labelling it "·LoRA" for TabPFN and
        # "·ICLhead" for TabICLv2 would put one scheme in two rows of every figure, under a
        # name (LoRA) for a technique this project no longer uses. The on-disk filename tags
        # still differ (`__lora` / `__iclhead`); `_method_series_name` already collapses both
        # to one boolean, and the manifest's `use_lora` column is the ground truth.
        adapt = " ·frozen" if row.get("use_lora") else ""
        fp = " ·fullpass" if row.get("full_pass") else ""
        # The corpus arm MUST appear: without it the filtered and unfiltered runs of the
        # same (base, lr) share a label, and every figure that groups by this name averages
        # them together. `·fullpass` is dropped when it is the only mode present, since a
        # constant tag on every bar is noise — see `compact_method_names`.
        mtr = row.get("min_train_rows", 0)
        arm = f" ·min{int(mtr) // 1000}k" if mtr and np.isfinite(mtr) else ""
        # Anchor strength, for the same reason the corpus arm is here: two trials that differ
        # only in lambda must not share a label, or every figure averages them together.
        l2 = row.get("l2sp_lambda", None)
        anch = "" if l2 is None or l2 != l2 else f" ·L2SP{float(l2):g}"
        if np.isfinite(lr):
            return f"trained ({base}) lr={lr:.0e}{fp}{adapt}{arm}{anch}"
        return f"trained ({base}){fp}{adapt}{arm}{anch}"
    return f"{src}({base})"


# =============================================================================
# Loaders
# =============================================================================


#: Restrict :func:`load_eval_results` to one run's result files. Eval writes
#: ``<run>_<ts>__task<k>_ds-<id>.csv`` (run is per-split, e.g. ``exp1_s03``), so a run is selected
#: by the ``<run>_`` filename prefix. A notebook sets ``eval_viz.use_run("exp1")`` so run-8's old
#: results in the same ``output/results/`` tree are not pooled in; ``CREDITPFN_VIZ_RUN`` does the
#: same for scripts. ``None`` (the default) pools everything, preserving the previous behaviour.
_RUN_OVERRIDE: str | None = None


def use_run(name: str | None) -> None:
    """Show only run ``name``'s eval results (matched by the ``<name>_`` filename prefix); ``None``
    pools every run. ``"exp1"`` matches its per-split files ``exp1_s00_…`` … ``exp1_s07_…`` and
    excludes ``exp0_…`` / ``creditpfn_…`` / ``run-8``."""
    global _RUN_OVERRIDE
    _RUN_OVERRIDE = str(name) if name else None


def load_eval_results(track: str) -> pd.DataFrame:
    """Pool every CSV under ``<benchmark_root>/<TRACK>/**/*.csv``.

    Adds structured columns derived from the parent directory name:
    ``method_dirname`` (raw dir name), ``source`` (baseline /
    ``<family>-untuned`` / ``<family>-trained`` / unknown), ``family``
    (tabpfn / tabicl / classical), ``base_short`` (the short tag, e.g.
    ``v3-default`` or ``tabicl-v2``), ``lr``, ``use_lora``,
    ``full_pass``, and a ``method_name`` column built by
    :func:`human_method_name`.

    Returns an empty DataFrame when nothing is on disk yet.
    """
    if track not in ("pd", "lgd"):
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")

    paths = _resolve_paths()
    track_dir = paths["benchmark_root"] / ("PD" if track == "pd" else "LGD")
    if not track_dir.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    run = _RUN_OVERRIDE or os.environ.get("CREDITPFN_VIZ_RUN")
    csv_files = sorted(track_dir.rglob("*.csv"))
    if run:
        # Eval names files ``<run>_<ts>__task…``; the ``<run>_`` prefix isolates one run's per-split
        # files (``exp1_s00_…``) and excludes ``exp0_…`` / ``creditpfn_…`` sharing this tree.
        csv_files = [c for c in csv_files if c.name.startswith(f"{run}_")]
    for csv in csv_files:
        try:
            df = pd.read_csv(csv)
        except Exception as exc:                          # pragma: no cover
            LOGGER.warning("could not read %s: %s", csv, exc)
            continue
        if df.empty:
            continue
        method_dir = csv.parent.name
        df["method_dirname"] = method_dir
        meta = _decode_method_dirname(method_dir)
        df["source"] = meta["source"]
        df["base_short"] = meta["base_short"]
        df["lr"] = meta["lr"]
        df["use_lora"] = meta["use_lora"]
        # ``full_pass`` was previously swallowed into base_short, which
        # averaged one_sample and full_pass rows of the same (base, lr)
        # into one point. Decoded explicitly since 2026-08-04.
        df["full_pass"] = meta["full_pass"]
        # ``min_train_rows`` is the run-8 corpus arm. It was decoded but never copied onto
        # the frame, so `human_method_name` could not see it and the two arms of every swept
        # (base, lr) pair collapsed to ONE label — 21 PD models showed as 14 rows, silently
        # averaging the corpus comparison that the run exists to make.
        df["min_train_rows"] = meta["min_train_rows"]
        df["l2sp_lambda"] = meta["l2sp_lambda"]
        df["family"] = (
            "tabicl" if meta["source"].startswith("tabicl")
            else "tabpfn" if meta["source"].startswith("tabpfn")
            else "classical"
        )
        df["source_file"] = str(csv.relative_to(paths["benchmark_root"]))
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)

    # Human-friendly method name (used as the legend label everywhere).
    full["method_name"] = full.apply(human_method_name, axis=1)
    full["method_name"] = _drop_constant_tags(full["method_name"])
    return full


#: Tags `human_method_name` can append. Any of them carried by EVERY trained method in the
#: frame distinguishes nothing, so it is removed — it only makes the labels longer, and label
#: width is what forces the y axis of a 21-method leaderboard off an A4 page.
_DROPPABLE_TAGS = (" ·fullpass", " ·ICLhead", " ·LoRA")
#: The anchor tag is dropped the same way when every trained method carries the
#: same lambda — which is every run before run-9.


def _drop_constant_tags(names: pd.Series) -> pd.Series:
    """Strip tags shared by every trained method — they carry no information.

    Run-8 set `epoch_pass_mode=full_pass` for all 32 trials, so `·fullpass` appeared on all
    16 trained labels and cost ~10 characters of y-axis width for nothing. A tag present on
    only SOME methods is the whole point of the label and is kept.
    """
    trained = names[names.str.startswith("trained (")]
    if trained.empty:
        return names
    out = names
    for tag in _DROPPABLE_TAGS:
        if trained.str.contains(tag, regex=False).all():
            out = out.str.replace(tag, "", regex=False)
    return out


def available_methods(track: str) -> list[str]:
    """List of distinct human method names with at least one row on disk."""
    df = load_eval_results(track)
    if df.empty:
        return []
    return sorted(df["method_name"].dropna().unique().tolist())


def available_datasets(track: str) -> list[str]:
    """Distinct ``test_dataset_id`` values across the benchmark CSVs."""
    df = load_eval_results(track)
    if df.empty:
        return []
    return sorted(df["test_dataset_id"].dropna().unique().tolist())


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ``status == 'OK'`` rows (defensive: status column may be missing)."""
    if "status" in df.columns:
        return df[df["status"] == "OK"].copy()
    return df.copy()


def primary_metric(track: str) -> str:
    """The primary monitoring metric for headline plots."""
    return "roc_auc" if track == "pd" else "rmse"


def metric_direction(metric: str) -> str:
    if metric in {"roc_auc", "pr_auc", "f1", "accuracy",
                  "precision", "recall", "r2", "neg_nll"}:
        return "max"
    return "min"


# =============================================================================
# Aggregations
# =============================================================================


def aggregate_per_method(
    track: str, *, metric: str | None = None,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Mean / median / std of ``metric`` per ``method_name``.

    ``metric`` defaults to the primary metric for the track.
    Aggregation pools all (dataset × fold) rows of each method.
    """
    metric = metric or primary_metric(track)
    if df is None:
        df = _ok(load_eval_results(track))
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("method_name")[metric].agg(["mean", "median", "std", "count"])
    direction = metric_direction(metric)
    grp = grp.sort_values("mean", ascending=(direction == "min"))
    return grp.reset_index()


def aggregate_per_method_per_dataset(
    track: str, *, metric: str | None = None,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pivot: index=method_name, columns=test_dataset_id, value=mean(metric).

    Averages over folds. Cells without data are NaN.
    """
    metric = metric or primary_metric(track)
    if df is None:
        df = _ok(load_eval_results(track))
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    return df.pivot_table(
        index="method_name", columns="test_dataset_id",
        values=metric, aggfunc="mean",
    )


def winrate_matrix(
    track: str, *, metric: str | None = None,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pairwise win-rate matrix (per-dataset comparison, à la TabPFN-3 Fig 3).

    For every (model_A, model_B) we compute the fraction of test
    datasets where mean(model_A) beats mean(model_B). Diagonal is
    NaN. Direction-aware (lower-is-better for rmse/log_loss/mae).
    """
    metric = metric or primary_metric(track)
    pivot = aggregate_per_method_per_dataset(track, metric=metric, df=df)
    if pivot.empty:
        return pd.DataFrame()
    methods = list(pivot.index)
    direction = metric_direction(metric)
    mat = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            if a == b:
                mat.loc[a, b] = np.nan
                continue
            va = pivot.loc[a]
            vb = pivot.loc[b]
            mask = va.notna() & vb.notna()
            if not mask.any():
                mat.loc[a, b] = np.nan
                continue
            wins = (va[mask] > vb[mask]) if direction == "max" else (va[mask] < vb[mask])
            mat.loc[a, b] = float(wins.mean())
    return mat


# =============================================================================
# Plot scaffolding
# =============================================================================


def _new_fig(title: str, *, figsize=style.figsize(style.WIDTH_FULL, ratio=0.611)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    style.title(ax, title)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    return fig, ax


def _no_data_fig(reason: str = "no data"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, ratio=0.333))
    ax.text(0.5, 0.5, f"({reason})", ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="#888")
    ax.set_axis_off()
    return fig


def _rows_figsize(n_rows: int) -> tuple[float, float]:
    """Figure size for a chart with ONE ROW PER METHOD, sized from the row count.

    `style.figsize` clamps to `MAX_HEIGHT`, so this cannot overflow the page; below that it
    gives each row a constant 0.19 in, which is what keeps an 8 pt label legible. The three
    boxplots here used to derive their height from `5.5 / (0.55 * n)` — a ratio that SHRINKS
    as methods are added, exactly backwards.
    """
    return style.figsize(style.WIDTH_FULL,
                         ratio=(0.19 * max(n_rows, 3) + 1.0) / style.WIDTH_FULL)


def _palette_for_methods(methods: Sequence[str]) -> dict[str, str]:
    """One colour per method, from `src/visualize/style.py`.

    Keyed on the method name rather than its index, so a method keeps its colour
    across every figure even when a plot drops or reorders arms.
    """
    from src.visualize.style import color
    return {m: color(_method_series_name(m)) for m in dict.fromkeys(methods)}


def _method_series_name(method: str) -> str:
    """Map an eval method dirname onto a registered `style.COLORS` name."""
    m = method.lower()
    for tag in ("tabicl", "v2.6", "v3", "xgboost", "lightgbm", "catboost"):
        if tag in m:
            return tag
    if "logreg" in m or "linreg" in m or "ridge" in m:
        return "linear"
    return method


# =============================================================================
# Headline plots
# =============================================================================


def plot_leaderboard(track: str, *, metric: str | None = None):
    """Sorted-bar leaderboard with mean ± std error bars."""
    metric = metric or primary_metric(track)
    agg = aggregate_per_method(track, metric=metric)
    if agg.empty:
        return _no_data_fig(f"no eval results on track={track}")
    direction = metric_direction(metric)
    fig, ax = _new_fig(
        f"Leaderboard — {metric} ({'higher is better' if direction == 'max' else 'lower is better'})  ·  track={track}",
        figsize=style.figsize(style.WIDTH_FULL, ratio=(max(4.5, 0.32 * len(agg))) / (11)),
    )
    palette = _palette_for_methods(agg["method_name"].tolist())
    colors = [palette[m] for m in agg["method_name"]]

    # STANDARD ERROR, NOT STANDARD DEVIATION. `std` here is the spread across datasets, which
    # is dominated by how hard each dataset is, not by the method: on run-8 it drew every bar
    # with a +-0.13 whisker on a field whose entire method spread is 0.06, so the figure said
    # "nothing differs" about data in which things do differ. The standard error of the mean
    # is the quantity that answers "how well do we know this bar's height".
    n = agg["count"] if "count" in agg.columns else pd.Series(np.nan, index=agg.index)
    err = (agg["std"] / np.sqrt(n.where(n > 0))).fillna(0) if n.notna().any() \
        else agg["std"].fillna(0)
    ax.barh(agg["method_name"], agg["mean"], xerr=err,
            color=colors, alpha=0.85, error_kw=dict(ecolor="black", capsize=2, alpha=0.6))
    ax.invert_yaxis()

    # START THE AXIS AT THE DATA, NOT AT ZERO. A bar chart of ROC-AUC from 0 spends 90 % of
    # its width on the region no model can occupy, and all 21 bars look identical; the real
    # spread here is 0.70-0.76. Bars still reach the axis floor, so nothing is exaggerated
    # beyond the honest statement "these are the values, magnified".
    lo, hi = float((agg["mean"] - err).min()), float((agg["mean"] + err).max())
    pad = max((hi - lo) * 0.12, 1e-6)
    floor = 0.5 if metric in _CHANCE_AT_HALF and direction == "max" else None
    left = max(lo - pad, floor) if floor is not None else lo - pad
    if left < hi:
        ax.set_xlim(left, hi + pad)
    ax.set_xlabel(f"mean ± s.e.  {metric}"
                  + ("  (axis starts at 0.5 = chance)" if left == floor else ""))
    ax.tick_params(axis="y", labelsize=8)
    style.note(ax, "error bars: standard error over datasets × folds")
    return fig


def plot_metric_boxplot(track: str, *, metric: str | None = None):
    """Boxplot of ``metric`` per method (across datasets × folds)."""
    metric = metric or primary_metric(track)
    df = _ok(load_eval_results(track))
    if df.empty or metric not in df.columns:
        return _no_data_fig(f"no results / metric={metric!r}")
    direction = metric_direction(metric)
    order = (
        df.groupby("method_name")[metric].median()
        .sort_values(ascending=(direction == "min"))
        .index.tolist()
    )
    palette = _palette_for_methods(order)
    fig, ax = _new_fig(
        f"{metric} by method",
        figsize=_rows_figsize(len(order)),
    )
    data = [df.loc[df["method_name"] == m, metric].dropna().values for m in order]
    # HORIZONTAL. Method names are 20-46 characters; on a vertical boxplot they become 21
    # rotated x tick labels that overlap each other and eat half the figure height. The
    # leaderboard already reads left-to-right, so this also makes the two figures comparable
    # row for row.
    bp = ax.boxplot(
        data, tick_labels=order, showmeans=True, patch_artist=True, vert=False,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=4),
        flierprops=dict(marker="x", markersize=3, alpha=0.4),
    )
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(palette[m])
        patch.set_alpha(0.75)
    ax.set_xlabel(metric)
    ax.invert_yaxis()                      # best at the top, as in the leaderboard
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    return fig


def plot_per_dataset_heatmap(track: str, *, metric: str | None = None):
    """Method × dataset heatmap of mean(metric).

    Direction-aware colourmap (``viridis`` for higher-is-better,
    ``viridis_r`` for lower-is-better).
    """
    import matplotlib.pyplot as plt
    metric = metric or primary_metric(track)
    pivot = aggregate_per_method_per_dataset(track, metric=metric)
    if pivot.empty:
        return _no_data_fig(f"no results / metric={metric!r}")
    direction = metric_direction(metric)
    # Sort methods by overall median (best first).
    order = (
        pivot.mean(axis=1).sort_values(ascending=(direction == "min")).index
    )
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(
        figsize=style.figsize(style.WIDTH_FULL, ratio=(max(5, 0.32 * pivot.shape[0])) / (max(8, 0.45 * pivot.shape[1]))),
    )
    # COLOUR THE GAP TO THE BEST METHOD ON EACH DATASET, NOT THE RAW VALUE. Absolute scores
    # are dominated by how hard the dataset is: on run-8 every column came out a single flat
    # colour, so the heatmap reported "myhom is hard, credit_risk is easy" — which the reader
    # already knows — and said nothing about the methods, which is what it is for. Per-column
    # normalisation makes each column a within-dataset comparison; the annotation still
    # carries the absolute number, so nothing is hidden.
    best = pivot.max(axis=0) if direction == "max" else pivot.min(axis=0)
    gap = (pivot - best) if direction == "max" else (best - pivot)   # <= 0, 0 = best
    im = ax.imshow(gap.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    style.thin_ticks(ax, 'y')
    style.title(ax, f"{metric} per method × dataset (colour = gap to best on that dataset)")
    # Annotate only while the cells are big enough to read at print size. Above that the
    # colour is the message and 300 numbers at 5 pt are a grey texture.
    if pivot.shape[0] * pivot.shape[1] <= 160:
        mid = float(np.nanmedian(gap.values))
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v, g = pivot.values[i, j], gap.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}".lstrip("0"), ha="center", va="center",
                            fontsize=7, color="white" if g < mid else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"{metric} − best on dataset", fontsize=7)
    # No tight_layout: style.py turns constrained_layout ON, and calling both makes
    # matplotlib warn and discard one of them. constrained_layout fits the content
    # INSIDE the declared A4 width, which is the whole point of drawing at final size.
    return fig


def plot_winrate_matrix(track: str, *, metric: str | None = None):
    """Pairwise win-rate matrix.

    Cells are the % of test datasets where the *row* method beat the
    *column* method.
    """
    import matplotlib.pyplot as plt
    metric = metric or primary_metric(track)
    mat = winrate_matrix(track, metric=metric)
    if mat.empty:
        return _no_data_fig(f"no results / metric={metric!r}")
    # Order by overall win rate (row mean).
    order = mat.mean(axis=1).sort_values(ascending=False).index
    mat = mat.loc[order, order]
    fig, ax = plt.subplots(
        figsize=style.figsize(style.WIDTH_FULL, ratio=(max(6, 0.5 * mat.shape[0])) / (max(7, 0.55 * mat.shape[0]))),
    )
    im = ax.imshow(mat.values * 100.0, vmin=0, vmax=100, cmap="RdYlGn")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    style.thin_ticks(ax, 'y')
    style.title(ax, f"Pairwise win rate — {metric}, row beats column (% of datasets)")
    # Annotate only while three digits fit in a cell. At 21x21 on an A4 width each cell is
    # ~14 pt wide, and "100" in the neighbouring cells ran together into "10010010080" —
    # strictly worse than no numbers, since the colour already carries the value.
    n = mat.shape[0]
    if n <= 12:
        for i in range(n):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                            fontsize=7, color="black" if 0.2 < v < 0.8 else "white")
    else:
        style.note(ax, f"{n} methods — cell values in the colour bar only")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="row wins  (%)")
    # No tight_layout: style.py turns constrained_layout ON, and calling both makes
    # matplotlib warn and discard one of them. constrained_layout fits the content
    # INSIDE the declared A4 width, which is the whole point of drawing at final size.
    return fig


def plot_method_vs_method_scatter(
    track: str, method_a: str, method_b: str, *,
    metric: str | None = None,
):
    """Per-dataset scatter: x = method_a, y = method_b.

    Each point is one test_dataset_id; the dashed y = x line marks
    parity. Above the line → ``method_b`` beats ``method_a``
    (for higher-is-better metrics).
    """
    metric = metric or primary_metric(track)
    pivot = aggregate_per_method_per_dataset(track, metric=metric)
    if pivot.empty or method_a not in pivot.index or method_b not in pivot.index:
        return _no_data_fig(f"need both methods present (have {len(pivot.index)})")
    a = pivot.loc[method_a]
    b = pivot.loc[method_b]
    mask = a.notna() & b.notna()
    if not mask.any():
        return _no_data_fig("no shared datasets between the two methods")
    fig, ax = _new_fig(
        f"{method_b} vs {method_a} — {metric} (track={track})",
        figsize=style.figsize(style.WIDTH_FULL, ratio=1.000),
    )
    ax.scatter(a[mask], b[mask], s=55, alpha=0.85, edgecolor="black", linewidth=0.4)
    for ds in a[mask].index:
        ax.annotate(ds, (a[ds], b[ds]),
                    fontsize=7, alpha=0.65,
                    xytext=(4, 4), textcoords="offset points")
    lo = min(a[mask].min(), b[mask].min())
    hi = max(a[mask].max(), b[mask].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.45, linewidth=0.9)
    ax.set_xlabel(f"{method_a}  {metric}")
    ax.set_ylabel(f"{method_b}  {metric}")
    return fig


def plot_trained_vs_untuned(
    track: str, *, metric: str | None = None,
):
    """For each (dataset, trained-checkpoint), scatter trained metric
    against the best untuned TabPFN of the same architecture.

    Trained > untuned (above the y=x line, for higher-is-better
    metrics) ⇒ continued pretraining helped.
    """
    import matplotlib.pyplot as plt
    metric = metric or primary_metric(track)
    df = _ok(load_eval_results(track))
    if df.empty or metric not in df.columns:
        return _no_data_fig(f"no results / metric={metric!r}")
    # base_short encodes the family (``v3-default`` vs ``tabicl-v2``), so
    # merging on it keeps the trained-vs-untuned comparison within-family.
    untuned = (
        df[df["source"].str.endswith("-untuned")]
        .groupby(["base_short", "test_dataset_id"])[metric]
        .mean()
        .rename("untuned")
        .reset_index()
    )
    trained = (
        df[df["source"].str.endswith("-trained")]
        .groupby(["base_short", "test_dataset_id", "lr", "use_lora", "full_pass"])[metric]
        .mean()
        .rename("trained")
        .reset_index()
    )
    if untuned.empty or trained.empty:
        return _no_data_fig("need both trained AND untuned rows")
    merged = trained.merge(untuned, on=["base_short", "test_dataset_id"], how="inner")
    if merged.empty:
        return _no_data_fig("no shared base × dataset between trained / untuned")

    fig, ax = _new_fig(
        f"Trained vs untuned TabPFN — {metric} (track={track})",
        figsize=style.figsize(style.WIDTH_FULL, ratio=1.000),
    )
    palette = _palette_for_methods(list(merged["base_short"].unique()))
    for base, grp in merged.groupby("base_short"):
        ax.scatter(
            grp["untuned"], grp["trained"],
            color=palette.get(base, (0.4, 0.4, 0.4)),
            s=55, alpha=0.85, edgecolor="black", linewidth=0.4,
            label=base,
        )
    lo = min(merged["untuned"].min(), merged["trained"].min())
    hi = max(merged["untuned"].max(), merged["trained"].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.45, linewidth=0.9)
    ax.set_xlabel(f"untuned TabPFN  {metric}")
    ax.set_ylabel(f"trained CreditPFN  {metric}")
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_fold_stability(track: str, *, metric: str | None = None):
    """Std across folds per (method, dataset) — distribution of
    fold-level variability per method. Tall boxes → unstable methods."""
    metric = metric or primary_metric(track)
    df = _ok(load_eval_results(track))
    if df.empty or metric not in df.columns:
        return _no_data_fig(f"no results / metric={metric!r}")
    stds = (
        df.groupby(["method_name", "test_dataset_id"])[metric]
        .std()
        .reset_index()
        .dropna()
    )
    if stds.empty:
        return _no_data_fig("not enough folds per (method, dataset) for std")
    order = stds.groupby("method_name")[metric].median().sort_values().index.tolist()
    palette = _palette_for_methods(order)
    fig, ax = _new_fig(
        f"Fold-level stability — std({metric}) per (method × dataset)",
        figsize=_rows_figsize(len(order)),
    )
    data = [stds.loc[stds["method_name"] == m, metric].values for m in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, vert=False,
                    flierprops=dict(marker="x", markersize=3, alpha=0.4))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(palette[m])
        patch.set_alpha(0.75)
    ax.set_xlabel(f"std({metric}) across folds")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelrotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
        lbl.set_fontsize(8)
    return fig


def plot_time_vs_metric(track: str, *, metric: str | None = None):
    """Per-row inference time (x) vs metric (y), coloured by method.

    Sanity check that "the best model" isn't 100× slower than the runner-up.
    """
    metric = metric or primary_metric(track)
    df = _ok(load_eval_results(track))
    if df.empty or metric not in df.columns or "elapsed_sec" not in df.columns:
        return _no_data_fig(f"no results / missing column")
    fig, ax = _new_fig(
        f"Inference time vs {metric}",
        figsize=style.figsize(style.WIDTH_FULL, ratio=0.611),
    )
    methods = sorted(df["method_name"].unique())
    palette = _palette_for_methods(methods)
    for m in methods:
        sub = df[df["method_name"] == m]
        ax.scatter(
            sub["elapsed_sec"], sub[metric],
            color=palette[m], alpha=0.7, s=30,
            edgecolor="black", linewidth=0.3, label=m,
        )
    ax.set_xscale("log")
    ax.set_xlabel("elapsed seconds (per fold)")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=7, ncol=2)
    return fig


def plot_metric_correlation(track: str):
    """Correlation matrix between the available metric columns.

    Useful to see e.g. whether high ROC-AUC always implies low log-loss.
    """
    import matplotlib.pyplot as plt
    df = _ok(load_eval_results(track))
    if df.empty:
        return _no_data_fig(f"no results on track={track}")
    metric_cols = [
        "roc_auc", "log_loss", "pr_auc", "brier_score", "ece", "f1",
        "accuracy", "precision", "recall", "rmse", "mae", "r2", "neg_nll",
    ]
    present = [c for c in metric_cols if c in df.columns]
    if len(present) < 2:
        return _no_data_fig("need ≥ 2 metric columns")
    corr = df[present].corr()
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, ratio=(0.6 * len(present) + 2) / (0.6 * len(present) + 2)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present)
    style.thin_ticks(ax, 'y')
    style.title(ax, f"Metric correlation matrix")
    for i in range(len(present)):
        for j in range(len(present)):
            v = corr.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # No tight_layout: style.py turns constrained_layout ON, and calling both makes
    # matplotlib warn and discard one of them. constrained_layout fits the content
    # INSIDE the declared A4 width, which is the whole point of drawing at final size.
    return fig


def plot_threshold_distribution(track: str):
    """Distribution of the F1-tuned thresholds per method (PD only)."""
    df = _ok(load_eval_results(track))
    if track != "pd" or df.empty or "optimal_threshold" not in df.columns:
        return _no_data_fig("threshold-tuning is PD-only (binary classification)")
    sub = df.dropna(subset=["optimal_threshold"])
    if sub.empty:
        return _no_data_fig("no optimal_threshold values")
    order = sorted(sub["method_name"].unique())
    palette = _palette_for_methods(order)
    fig, ax = _new_fig(
        "F1-tuned thresholds — per method (PD)",
        figsize=_rows_figsize(len(order)),
    )
    data = [sub.loc[sub["method_name"] == m, "optimal_threshold"].values for m in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, vert=False,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=4),
                    flierprops=dict(marker="x", markersize=3, alpha=0.4))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(palette[m])
        patch.set_alpha(0.75)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.45, linewidth=0.8)
    ax.set_xlabel("optimal threshold (max-F1 on val)")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    return fig


def plot_top_method_per_dataset(track: str, *, metric: str | None = None):
    """For each dataset: which method scored best?

    A coloured tile per (dataset, winning method) — quickly shows
    where each method dominates.
    """
    import matplotlib.pyplot as plt
    metric = metric or primary_metric(track)
    pivot = aggregate_per_method_per_dataset(track, metric=metric)
    if pivot.empty:
        return _no_data_fig(f"no results / metric={metric!r}")
    direction = metric_direction(metric)
    if direction == "max":
        winner = pivot.idxmax(axis=0)
    else:
        winner = pivot.idxmin(axis=0)
    counts = winner.value_counts()
    palette = _palette_for_methods(counts.index.tolist())
    fig, ax = _new_fig(
        f"Winning method per dataset — {metric} (track={track})",
        figsize=style.figsize(style.WIDTH_FULL, ratio=(4) / (max(8, 0.4 * len(winner)))),
    )
    for i, ds in enumerate(winner.index):
        m = winner.iloc[i]
        ax.bar(i, 1, color=palette.get(m, (0.4, 0.4, 0.4)), label=m, width=0.95)
    # Build a deduplicated legend.
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lbl in zip(handles, labels):
        seen.setdefault(lbl, h)
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=7)
    ax.set_xticks(range(len(winner)))
    ax.set_xticklabels(winner.index, rotation=60, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_ylim(0, 1.1)
    return fig


def plot_dataset_difficulty(track: str, *, metric: str | None = None):
    """Best-vs-worst per dataset, ordered by 'best score' — surface the
    easy datasets (everyone does well) vs the hard ones (best ≪ ideal).
    """
    metric = metric or primary_metric(track)
    pivot = aggregate_per_method_per_dataset(track, metric=metric)
    if pivot.empty:
        return _no_data_fig(f"no results / metric={metric!r}")
    direction = metric_direction(metric)
    if direction == "max":
        best = pivot.max(axis=0)
        worst = pivot.min(axis=0)
    else:
        best = pivot.min(axis=0)
        worst = pivot.max(axis=0)
    order = best.sort_values(ascending=(direction == "min")).index
    best = best.loc[order]
    worst = worst.loc[order]
    fig, ax = _new_fig(
        f"Per-dataset best vs worst — {metric}, track={track}",
        figsize=style.figsize(style.WIDTH_FULL, ratio=(5.5) / (max(8, 0.32 * len(order)))),
    )
    x = np.arange(len(order))
    ax.plot(x, best.values, marker="o", label="best method", linewidth=1.5)
    ax.plot(x, worst.values, marker="s", label="worst method",
            linewidth=1.5, alpha=0.8)
    ax.fill_between(x, worst.values, best.values, alpha=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_baselines_vs_tabpfn(track: str, *, metric: str | None = None):
    """Two boxplots side by side: 'classical baselines' and 'TabPFN-family'."""
    import matplotlib.pyplot as plt
    metric = metric or primary_metric(track)
    df = _ok(load_eval_results(track))
    if df.empty or metric not in df.columns:
        return _no_data_fig(f"no results / metric={metric!r}")

    def _group_of(src: str) -> str:
        if src == "baseline":
            return "classical"
        if src.startswith("tabpfn-"):
            return "tabpfn"
        if src.startswith("tabicl-"):
            return "tabicl"
        return "other"

    df["group"] = df["source"].map(_group_of)
    direction = metric_direction(metric)
    fig, ax = _new_fig(
        f"Classical baselines vs foundation models — {metric}",
        figsize=style.figsize(style.WIDTH_FULL, ratio=0.786),
    )
    _groups = ("classical", "tabpfn", "tabicl", "other")
    data = [df.loc[df["group"] == g, metric].dropna().values
            for g in _groups if (df["group"] == g).any()]
    labels = [g for g in _groups if (df["group"] == g).any()]
    bp = ax.boxplot(
        data, labels=labels, showmeans=True, patch_artist=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=5),
        flierprops=dict(marker="x", markersize=3, alpha=0.4),
    )
    palette = {"classical": (0.85, 0.5, 0.2),
               "tabpfn":    (0.3, 0.6, 0.8),
               "tabicl":    (0.45, 0.75, 0.45),
               "other":     (0.5, 0.5, 0.5)}
    for patch, lbl in zip(bp["boxes"], labels):
        patch.set_facecolor(palette.get(lbl, (0.5, 0.5, 0.5)))
        patch.set_alpha(0.8)
    ax.set_ylabel(metric)
    return fig


def failed_pairs(track: str) -> pd.DataFrame:
    """One row per (method, dataset, fold) with ``status != 'OK'``."""
    df = load_eval_results(track)
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in (
        "method_name", "test_dataset_id", "fold_idx",
        "status", "error", "elapsed_sec", "source_file",
    ) if c in df.columns]
    return df[df.get("status", "OK") != "OK"][cols].reset_index(drop=True)


def eval_leaderboard(track: str, *, metric: str | None = None) -> pd.DataFrame:
    """Sorted leaderboard DataFrame with mean / median / std / count.

    Sort is direction-aware (PD/roc_auc descends; LGD/rmse ascends).
    """
    metric = metric or primary_metric(track)
    return aggregate_per_method(track, metric=metric)
