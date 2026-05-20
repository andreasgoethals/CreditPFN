"""Final-benchmark visualisation helpers.

Consumes the wide-format CSVs written by ``scripts/eval_pipeline.py``
(via ``src.eval.benchmark.EvalRow``) at::

    output/results/<TRACK>/<method-dirname>/<run>_<ts>[__ds-<id>].csv

Each row is one ``(model × dataset × fold)`` tuple with all metric
columns side-by-side. We pool every CSV under one DataFrame, then
project / pivot / plot from there.

Method-dirname conventions (mirrored in ``src.eval.benchmark._method_dirname``):
    xgboost, catboost, logreg, linreg                  → classical baselines
    tabpfn-untuned__<short>                            → reference TabPFN with no fine-tune
    tabpfn-trained__<short>__lr<lr>[__lora]            → our continued-pretrained variants

The visualisations here are deliberately exhaustive — the notebook
caller picks which to display.

Two design contracts (mirrors src/utils/training_viz):
    1. Every plot returns a matplotlib Figure.
    2. Empty-disk runs render stub figures with "(no data)"; loaders
       return empty DataFrames.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

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
        return _NS(results=_NS(base_dir="output/results"))


def _resolve_paths():
    """Resolve durable-output roots used by the eval pipeline."""
    # Sync data_source so DATA_ROOT etc. matches the rest of the pipeline.
    try:
        from omegaconf import OmegaConf
        from src.utils.paths import apply_data_source_from_cfg
        apply_data_source_from_cfg(OmegaConf.load(_REPO / "config" / "data.yaml"))
    except Exception:  # pragma: no cover
        pass

    from src.utils.paths import resolve_output_path
    cfg = _load_eval_cfg()
    base = str(cfg.results.base_dir) if hasattr(cfg, "results") else "output/results"
    return {
        "benchmark_root": resolve_output_path(base),
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
        return {"source": "baseline", "base_short": d,
                "lr": np.nan, "use_lora": False}
    if d.startswith("tabpfn-untuned__"):
        return {"source": "tabpfn-untuned",
                "base_short": d.removeprefix("tabpfn-untuned__"),
                "lr": np.nan, "use_lora": False}
    if d.startswith("tabpfn-trained__"):
        rest = d.removeprefix("tabpfn-trained__")
        lora = rest.endswith("__lora")
        if lora:
            rest = rest.removesuffix("__lora")
        # The lr piece (if present) is the LAST ``__lr<tag>`` chunk.
        m = re.search(r"__lr([0-9eE.+\-]+)$", rest)
        if m:
            lr = float(m.group(1))
            base = rest[: m.start()]
        else:
            lr = np.nan
            base = rest
        return {"source": "tabpfn-trained", "base_short": base,
                "lr": lr, "use_lora": lora}
    return {"source": "unknown", "base_short": d,
            "lr": np.nan, "use_lora": False}


def human_method_name(row: pd.Series) -> str:
    """Compact human-readable label from a row of :func:`load_eval_results`."""
    src = row.get("source", "unknown")
    base = row.get("base_short", "?")
    if src == "baseline":
        return base
    if src == "tabpfn-untuned":
        return f"untuned ({base})"
    if src == "tabpfn-trained":
        lr = row.get("lr", np.nan)
        lora = " ·LoRA" if row.get("use_lora") else ""
        if np.isfinite(lr):
            return f"trained ({base}) lr={lr:.0e}{lora}"
        return f"trained ({base}){lora}"
    return f"{src}({base})"


# =============================================================================
# Loaders
# =============================================================================


def load_eval_results(track: str) -> pd.DataFrame:
    """Pool every CSV under ``<benchmark_root>/<TRACK>/**/*.csv``.

    Adds structured columns derived from the parent directory name:
    ``method_dirname`` (raw dir name), ``source`` (baseline /
    tabpfn-untuned / tabpfn-trained / unknown), ``base_short`` (the
    short tag, e.g. ``v3-default``), ``lr``, ``use_lora``, and a
    ``method_name`` column built by :func:`human_method_name`.

    Returns an empty DataFrame when nothing is on disk yet.
    """
    if track not in ("pd", "lgd"):
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")

    paths = _resolve_paths()
    track_dir = paths["benchmark_root"] / ("PD" if track == "pd" else "LGD")
    if not track_dir.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for csv in sorted(track_dir.rglob("*.csv")):
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
        df["source_file"] = str(csv.relative_to(paths["benchmark_root"]))
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)

    # Human-friendly method name (used as the legend label everywhere).
    full["method_name"] = full.apply(human_method_name, axis=1)
    return full


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


def _new_fig(title: str, *, figsize=(9, 5.5)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    return fig, ax


def _no_data_fig(reason: str = "no data"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, f"({reason})", ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="#888")
    ax.set_axis_off()
    return fig


def _palette_for_methods(methods: Sequence[str]) -> dict[str, tuple]:
    import matplotlib.cm as cm
    methods = list(dict.fromkeys(methods))
    if not methods:
        return {}
    cmap = cm.get_cmap("tab20", max(len(methods), 3))
    return {m: cmap(i % cmap.N)[:3] for i, m in enumerate(methods)}


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
        figsize=(11, max(4.5, 0.32 * len(agg))),
    )
    palette = _palette_for_methods(agg["method_name"].tolist())
    colors = [palette[m] for m in agg["method_name"]]
    ax.barh(agg["method_name"], agg["mean"], xerr=agg["std"].fillna(0),
            color=colors, alpha=0.85, error_kw=dict(ecolor="black", capsize=2, alpha=0.6))
    ax.invert_yaxis()
    ax.set_xlabel(f"mean ± std  {metric}")
    ax.tick_params(axis="y", labelsize=8)
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
        f"{metric} by method — track={track}",
        figsize=(max(8, 0.55 * len(order)), 5.5),
    )
    data = [df.loc[df["method_name"] == m, metric].dropna().values for m in order]
    bp = ax.boxplot(
        data, labels=order, showmeans=True, patch_artist=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=5),
        flierprops=dict(marker="x", markersize=3, alpha=0.4),
    )
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(palette[m])
        patch.set_alpha(0.75)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
        lbl.set_fontsize(8)
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
        figsize=(max(8, 0.45 * pivot.shape[1]),
                 max(5, 0.32 * pivot.shape[0])),
    )
    cmap = "viridis" if direction == "max" else "viridis_r"
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"{metric} per method × dataset — track={track}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white"
                        if (v < np.nanmedian(pivot.values)
                            if direction == "max"
                            else v > np.nanmedian(pivot.values))
                        else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
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
        figsize=(max(7, 0.55 * mat.shape[0]),
                 max(6, 0.5 * mat.shape[0])),
    )
    im = ax.imshow(mat.values * 100.0, vmin=0, vmax=100, cmap="RdYlGn")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_title(f"Pairwise win rate — {metric}, track={track}\n(row beats column, % of datasets)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v*100:.0f}",
                        ha="center", va="center",
                        fontsize=7,
                        color="black" if 0.2 < v < 0.8 else "white")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="row wins  (%)")
    fig.tight_layout()
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
        figsize=(6.5, 6.5),
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
    untuned = (
        df[df["source"] == "tabpfn-untuned"]
        .groupby(["base_short", "test_dataset_id"])[metric]
        .mean()
        .rename("untuned")
        .reset_index()
    )
    trained = (
        df[df["source"] == "tabpfn-trained"]
        .groupby(["base_short", "test_dataset_id", "lr", "use_lora"])[metric]
        .mean()
        .rename("trained")
        .reset_index()
    )
    if untuned.empty or trained.empty:
        return _no_data_fig("need both tabpfn-trained AND tabpfn-untuned rows")
    merged = trained.merge(untuned, on=["base_short", "test_dataset_id"], how="inner")
    if merged.empty:
        return _no_data_fig("no shared base × dataset between trained / untuned")

    fig, ax = _new_fig(
        f"Trained vs untuned TabPFN — {metric} (track={track})",
        figsize=(7, 7),
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
        figsize=(max(8, 0.55 * len(order)), 5.5),
    )
    data = [stds.loc[stds["method_name"] == m, metric].values for m in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True,
                    flierprops=dict(marker="x", markersize=3, alpha=0.4))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(palette[m])
        patch.set_alpha(0.75)
    ax.set_ylabel(f"std({metric}) across folds")
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
        f"Inference time vs {metric} — track={track}",
        figsize=(9, 5.5),
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
        "roc_auc", "log_loss", "pr_auc", "f1", "accuracy",
        "precision", "recall", "rmse", "mae", "r2", "neg_nll",
    ]
    present = [c for c in metric_cols if c in df.columns]
    if len(present) < 2:
        return _no_data_fig("need ≥ 2 metric columns")
    corr = df[present].corr()
    fig, ax = plt.subplots(figsize=(0.6 * len(present) + 2, 0.6 * len(present) + 2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present)
    ax.set_title(f"Metric correlation matrix — track={track}")
    for i in range(len(present)):
        for j in range(len(present)):
            v = corr.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
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
        figsize=(max(8, 0.55 * len(order)), 5.5),
    )
    data = [sub.loc[sub["method_name"] == m, "optimal_threshold"].values for m in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=4),
                    flierprops=dict(marker="x", markersize=3, alpha=0.4))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(palette[m])
        patch.set_alpha(0.75)
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.45, linewidth=0.8)
    ax.set_ylabel("optimal threshold (max-F1 on val)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
        lbl.set_fontsize(8)
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
        figsize=(max(8, 0.4 * len(winner)), 4),
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
        figsize=(max(8, 0.32 * len(order)), 5.5),
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
        if src in {"tabpfn-untuned", "tabpfn-trained"}:
            return "tabpfn"
        return "other"

    df["group"] = df["source"].map(_group_of)
    direction = metric_direction(metric)
    fig, ax = _new_fig(
        f"Classical baselines vs TabPFN-family — {metric}",
        figsize=(7, 5.5),
    )
    data = [df.loc[df["group"] == g, metric].dropna().values
            for g in ("classical", "tabpfn", "other") if (df["group"] == g).any()]
    labels = [g for g in ("classical", "tabpfn", "other") if (df["group"] == g).any()]
    bp = ax.boxplot(
        data, labels=labels, showmeans=True, patch_artist=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=5),
        flierprops=dict(marker="x", markersize=3, alpha=0.4),
    )
    palette = {"classical": (0.85, 0.5, 0.2),
               "tabpfn":    (0.3, 0.6, 0.8),
               "other":     (0.5, 0.5, 0.5)}
    for patch, lbl in zip(bp["boxes"], labels):
        patch.set_facecolor(palette[lbl]); patch.set_alpha(0.8)
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
