"""Data-exploration helpers for the CreditPFN corpus.

The pipeline produces three layers of data, each with a distinct shape
and a distinct exploration story. All helpers here are designed to
**scale to the 3 000-dataset corpus we will buy** — corpus-level views
use aggregate histograms / sortable tables; per-dataset views are
opt-in, paginated, and bounded by ``max_show``.

Three layers
------------

* **Raw**       — ``data/raw/{pd,lgd}/<id>.csv``.
                  Read by ``raw_*`` helpers / the
                  ``raw_data_exploration`` notebook. The "what did the
                  vendor actually deliver" view.
* **Processed** — ``data/processed/{pd,lgd}/<id>.sanitized.csv``.
                  Read by ``corpus_summary_table`` / the
                  ``processed_data_exploration`` notebook. The
                  "is the cleaning sound" view. Since 2026-05-20
                  this is also the format that the training pipeline
                  reads directly (no more `.npz` chunks).

Glossary
--------
* **missingness** (or ``missing_rate``) — *fraction of cells in the
  dataset that are NaN*, i.e. ``cell_count = n_rows × n_features``,
  ``missing_rate = n_NaN / cell_count``. This is dataset-level, not
  per-row. A row that has any NaN cell does NOT count as a "missing
  row" — it contributes proportionally to the cell-fraction. Always
  this denominator throughout the project, in the manifest as well
  as the plots.
* **minority_class_ratio** — for classification: ``n_minority /
  n_total`` (the share of the smallest class). For multiclass with K
  classes, perfect balance is ``1/K``; this column collapses
  imbalance into a single 0–1 number where lower = more imbalanced.
* **target_mean / target_std** — for regression: the target
  variable's empirical mean / standard deviation across the dataset.

Public surface — corpus-level (scales to 3 000 datasets)
--------------------------------------------------------
* :func:`raw_corpus_summary` — one row per raw CSV.
* :func:`corpus_summary_table` — one row per dataset, manifest +
  on-disk processed shapes side-by-side.
* :func:`plot_dataset_size_distribution` — ``track`` is required;
  one plot for one track at a time. Two histograms: rows and
  features.
* :func:`plot_missing_rate_distribution` — ``track`` required.
  Single histogram with explicit "% of cells" axis label.
* :func:`plot_class_imbalance_distribution` — PD only.
* :func:`plot_target_mean_distribution_lgd` — corpus-level histogram
  of LGD target means across all datasets.
* :func:`plot_source_breakdown` — dataset count per data source,
  split by track (corpus provenance).
* :func:`plot_size_scatter` — rows-vs-features scatter (log-log)
  mapping the corpus shape space; ``source`` selects raw / processed.
* :func:`plot_feature_type_distribution` — histogram of each
  dataset's categorical-feature share, per track.
* :func:`plot_feature_reduction` — raw vs post-sanitise feature
  count, visualising the FeatureAgglomeration 128-column cap.

Public surface — per-dataset (paginated for scale)
--------------------------------------------------
* :func:`plot_target_distribution_pd` — paginated grid; default
  ``max_show=30`` first IDs, override via ``dataset_ids=[…]``.
* :func:`plot_target_distribution_lgd` — same.

Public surface — error detection
--------------------------------
* :func:`find_anomalous_datasets` — flag corpus members with
  anomalous values on any of N indicators.

All plot helpers return the matplotlib ``Figure`` so the caller can
``fig.savefig(...)`` or further customise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]


# =============================================================================
# Cfg / path resolution
# =============================================================================


def _load_default_cfg():
    """Load ``config/data.yaml`` as the source-of-truth for paths.

    Falling back to a static dict if OmegaConf is unavailable (e.g.
    in a smoke-test environment) — but the production path is
    always to read the YAML, so cfg path overrides on the CLI also
    take effect for the exploration helpers.
    """
    try:
        from omegaconf import OmegaConf
        return OmegaConf.load(_REPO / "config" / "data.yaml")
    except Exception:  # pragma: no cover  — fallback for missing dep
        from types import SimpleNamespace as _NS
        return _NS(paths=_NS(
            raw=str(_REPO / "data" / "raw"),
            processed=str(_REPO / "data" / "processed"),
            manifest_pd=str(_REPO / "data" / "manifest_pd.csv"),
            manifest_lgd=str(_REPO / "data" / "manifest_lgd.csv"),
        ))


def _resolve_paths(cfg=None) -> dict[str, Path]:
    """Return absolute paths derived from ``cfg`` (or default cfg).

    Mirrors the same resolver split used by the data pipeline:

      * raw / processed → ``$CREDITPFN_DATA_ROOT`` (scratch on VSC)
      * manifest_*               → ``$CREDITPFN_OUTPUT_ROOT`` (durable on VSC)
    """
    from src.utils.paths import resolve_data_path, resolve_output_path
    if cfg is None:
        cfg = _load_default_cfg()

    raw_default       = "data/raw"
    return {
        "raw":          resolve_data_path(getattr(cfg.paths, "raw", raw_default)),
        "processed":    resolve_data_path(cfg.paths.processed),
        "manifest_pd":  resolve_output_path(cfg.paths.manifest_pd),
        "manifest_lgd": resolve_output_path(cfg.paths.manifest_lgd),
    }


# =============================================================================
# Loaders
# =============================================================================


def load_manifests(cfg=None) -> dict[str, pd.DataFrame]:
    """Return ``{"pd": ..., "lgd": ...}`` as DataFrames."""
    paths = _resolve_paths(cfg)
    return {
        "pd": pd.read_csv(paths["manifest_pd"]),
        "lgd": pd.read_csv(paths["manifest_lgd"]),
    }


def load_raw_dataset(track: str, dataset_id: str, cfg=None) -> pd.DataFrame:
    """Read ``<cfg.paths.raw>/{track}/<dataset_id>.csv``."""
    paths = _resolve_paths(cfg)
    p = paths["raw"] / track / f"{dataset_id}.csv"
    if not p.exists():
        raise FileNotFoundError(f"raw CSV not found at {p}")
    return pd.read_csv(p, low_memory=False)


def load_sanitized_dataset(
    track: str, dataset_id: str, cfg=None,
) -> pd.DataFrame:
    """Read ``<cfg.paths.processed>/{track}/<dataset_id>.sanitized.csv``."""
    paths = _resolve_paths(cfg)
    p = paths["processed"] / track / f"{dataset_id}.sanitized.csv"
    if not p.exists():
        raise FileNotFoundError(f"sanitised CSV not found at {p}")
    return pd.read_csv(p, low_memory=False)


# In-process memoisation: the loaders read every CSV on disk, which is
# slow for the wide datasets (algorithmwatch is 159 k × 2 987). Re-using
# the same DataFrame across notebook cells keeps the second-and-later
# cell render times in milliseconds. Pass ``refresh=True`` to bust the
# cache after a fresh pipeline rebuild.
_SUMMARY_CACHE: dict[tuple, pd.DataFrame] = {}


def _cache_key(name: str, cfg) -> tuple:
    """Hashable key for the per-cfg summary cache."""
    if cfg is None:
        return (name, None)
    # Stringify the cfg paths block — that's all we depend on.
    paths = _resolve_paths(cfg)
    return (name, tuple(sorted((k, str(v)) for k, v in paths.items())))


def clear_summary_cache() -> None:
    """Drop all memoised corpus summaries. Call this after re-running
    the data pipeline so subsequent plots see the fresh state."""
    _SUMMARY_CACHE.clear()


def _round_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Round summary-table numeric columns to a sane number of digits.

    Counts (rows, cols, n_*, n_chunks) stay as integers. Rates and
    means / stds get 4 decimals; file sizes get 2 decimals. Applied
    in-place-ish (returns a new DataFrame).
    """
    out = df.copy()
    rate_cols = [
        "missing_cells_rate", "missing_rate_raw",
        "minority_class_ratio", "target_mean", "target_std",
        "ctx_query_ratio", "unknown_sentinel_rate", "nan_rate_in_X",
    ]
    size_cols = ["file_mb", "total_size_mb", "mean_chunk_rows"]
    for c in rate_cols:
        if c in out.columns and pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(4)
    for c in size_cols:
        if c in out.columns and pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(2)
    return out


def raw_corpus_summary(cfg=None, *, refresh: bool = False) -> pd.DataFrame:
    """One row per raw CSV under ``data/raw/{pd,lgd}/``.

    Computed *without* applying any surgical fix — purely
    "what's-on-disk" view. Useful for the raw-data exploration
    notebook to spot files whose shapes don't match the manifest's
    expectations (a vendor delivered the wrong file, etc.).

    Memoised — the underlying CSV read is the bottleneck (>60 s on
    the wide algorithmwatch dataset), so subsequent calls within the
    same Python session reuse the cached DataFrame. Pass
    ``refresh=True`` to bust the cache after a pipeline rerun.

    Field reference
    ---------------
    ``track``               — "pd" or "lgd"
    ``dataset_id``          — same as the filename stem
    ``raw_rows`` /
    ``raw_cols``            — shape of the raw CSV before any fix
    ``missing_cells_rate``  — fraction of NaN cells in the raw CSV;
                              denominator = ``raw_rows × raw_cols``
    ``file_mb``             — CSV size on disk in megabytes
    ``target_in_raw``       — True if the metadata's target column
                              is present in the raw CSV
    ``raw_target_unique``   — number of distinct non-NaN values in
                              the target column
    ``source``              — the value of
                              ``DATASET_METADATA[dataset_id].source``
                              hardcoded in
                              ``src/data/preprocessing.py`` (e.g.
                              ``"kaggle"``, ``"uci"``,
                              ``"freddie-mac"``, ``"local"``)
    """
    key = _cache_key("raw_corpus_summary", cfg)
    if not refresh and key in _SUMMARY_CACHE:
        return _SUMMARY_CACHE[key].copy()

    from src.data.preprocessing import DATASET_METADATA
    paths = _resolve_paths(cfg)
    rows: list[dict] = []
    for dataset_id, meta in DATASET_METADATA.items():
        track = meta["track"]
        p = paths["raw"] / track / f"{dataset_id}.csv"
        if not p.exists():
            rows.append({
                "track": track, "dataset_id": dataset_id,
                "raw_rows": -1, "raw_cols": -1, "missing_cells_rate": np.nan,
                "file_mb": np.nan, "target_in_raw": False,
                "raw_target_unique": np.nan, "source": meta["source"],
            })
            continue
        df = pd.read_csv(p, low_memory=False)
        n_missing = int(df.isna().sum().sum())
        cells = max(1, df.shape[0] * df.shape[1])
        rows.append({
            "track": track,
            "dataset_id": dataset_id,
            "raw_rows": df.shape[0],
            "raw_cols": df.shape[1],
            "missing_cells_rate": n_missing / cells,
            "file_mb": p.stat().st_size / (1024 * 1024),
            "target_in_raw": meta["target_column"] in df.columns,
            "raw_target_unique": (
                int(df[meta["target_column"]].dropna().nunique())
                if meta["target_column"] in df.columns else np.nan
            ),
            "source": meta["source"],
        })
    out = _round_summary(pd.DataFrame(rows))
    _SUMMARY_CACHE[key] = out
    return out.copy()


def corpus_summary_table(
    track: str | None = None, cfg=None, *, refresh: bool = False,
) -> pd.DataFrame:
    """One row per dataset combining the manifest with on-disk processed.

    ``track`` filters to ``"pd"`` or ``"lgd"``; ``None`` returns both.
    Memoised — see :func:`raw_corpus_summary` for the rationale.

    The returned table has 15 columns covering raw shape, post-sanitise
    shape, n_categorical / n_numerical, missing rate, target stats
    (``minority_class_ratio`` for classification, ``target_mean`` /
    ``target_std`` for regression), and source provenance. Float
    columns are rounded to 4 decimals so the displayed table is
    readable.

    The ``source`` column is the ``DATASET_METADATA[id].source`` field
    hardcoded in ``src/data/preprocessing.py`` (e.g. ``"kaggle"``,
    ``"uci"``, ``"freddie-mac"``, ``"local"``); ``register.py`` carries
    it forward into the manifest verbatim.
    """
    key = _cache_key(f"corpus_summary_table:{track}", cfg)
    if not refresh and key in _SUMMARY_CACHE:
        return _SUMMARY_CACHE[key].copy()

    manifests = load_manifests(cfg)
    rows: list[dict] = []
    tracks = ["pd", "lgd"] if track is None else [track]
    for tr in tracks:
        for _, mrow in manifests[tr].iterrows():
            did = mrow["dataset_id"]
            try:
                df = load_sanitized_dataset(tr, did, cfg)
                post_rows, post_cols = df.shape
                target = mrow["target_column"]
                post_feature_cols = post_cols - (1 if target in df.columns else 0)
            except FileNotFoundError:
                post_rows = post_feature_cols = -1
            rows.append({
                "track": tr,
                "dataset_id": did,
                "task_type": mrow["task_type"],
                "target_column": mrow["target_column"],
                "raw_rows": int(mrow["n_rows"]),
                "raw_features": int(mrow["n_cols"]),
                "post_rows": post_rows,
                "post_features": post_feature_cols,
                "n_categorical": int(mrow["n_categorical"]),
                "n_numerical": int(mrow["n_numerical"]),
                "missing_rate_raw": float(mrow["missing_rate"]),
                "minority_class_ratio": (
                    float(mrow["minority_class_ratio"])
                    if pd.notna(mrow["minority_class_ratio"])
                    and mrow["minority_class_ratio"] != "" else np.nan
                ),
                "target_mean": (
                    float(mrow["target_mean"])
                    if pd.notna(mrow["target_mean"])
                    and mrow["target_mean"] != "" else np.nan
                ),
                "target_std": (
                    float(mrow["target_std"])
                    if pd.notna(mrow["target_std"])
                    and mrow["target_std"] != "" else np.nan
                ),
                "source": mrow["source"],
            })
    out = _round_summary(pd.DataFrame(rows))
    _SUMMARY_CACHE[key] = out
    return out.copy()


def _import_mpl():
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- #
# Plot style helpers (consistent across every corpus-level histogram)
# --------------------------------------------------------------------------- #


_TRACK_COLOR = {"pd": "#1f77b4", "lgd": "#ff7f0e"}     # tab:blue, tab:orange


def _apply_style(ax, *, title: str, xlabel: str, ylabel: str = "# datasets"):
    """Consistent grid + spine + label style across plots."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _log_bins(values, *, n_bins: int = 25, eps: float = 1.0):
    """Log-spaced bin edges spanning ``[min, max]``.

    Falls back to linear bins when the range collapses (all values
    equal). The ``eps`` shift handles the case where the data contains
    zero or one-row datasets that would crash a pure ``log10(0)``.
    """
    import numpy as _np
    v = _np.asarray(values, dtype=float)
    v = v[_np.isfinite(v)]
    if v.size == 0:
        return _np.linspace(0, 1, n_bins + 1)
    lo, hi = float(_np.maximum(v.min(), eps)), float(_np.maximum(v.max(), eps))
    if lo >= hi:
        return _np.linspace(lo - 1, hi + 1, n_bins + 1)
    return _np.logspace(_np.log10(lo), _np.log10(hi), n_bins + 1)


# --------------------------------------------------------------------------- #
# Corpus-level (scale to 3 000)
# --------------------------------------------------------------------------- #


def plot_dataset_size_distribution(
    track: str, *, source: str = "processed", cfg=None,
):
    """Per-track histograms of dataset rows and feature columns.

    Both panels use **log-spaced bins** with log-scaled x-axes so the
    bar widths stay consistent (the previous version used linear bins
    on a log axis, which produced wildly different bar widths across
    the range — datasets at 1k rows got narrow bars and datasets at
    300k rows got wide bars). Useful because our corpus spans 4–5
    orders of magnitude in both row count and feature count.

    ``source``: ``"raw"`` for ``data/raw/`` shapes (pre-fix), or
    ``"processed"`` for post-sanitise shapes. ``track`` is required —
    PD and LGD are plotted separately so the corpora can be compared
    side by side.
    """
    plt = _import_mpl()
    if track not in ("pd", "lgd"):
        raise ValueError("track must be 'pd' or 'lgd'")
    if source == "raw":
        summary = raw_corpus_summary(cfg)
        summary = summary[summary["track"] == track]
        rows_col, cols_col = "raw_rows", "raw_cols"
    elif source == "processed":
        summary = corpus_summary_table(track, cfg)
        rows_col, cols_col = "post_rows", "post_features"
    else:
        raise ValueError("source must be 'raw' or 'processed'")

    color = _TRACK_COLOR[track]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    rows = summary[rows_col].dropna()
    axes[0].hist(
        rows, bins=_log_bins(rows, n_bins=25),
        color=color, alpha=0.85, edgecolor="white", linewidth=0.6,
    )
    axes[0].set_xscale("log")
    _apply_style(
        axes[0],
        title=f"{track.upper()} — dataset rows ({source})",
        xlabel="rows  (log-scaled)",
    )

    cols = summary[cols_col].dropna()
    axes[1].hist(
        cols, bins=_log_bins(cols, n_bins=25),
        color=color, alpha=0.85, edgecolor="white", linewidth=0.6,
    )
    axes[1].set_xscale("log")
    _apply_style(
        axes[1],
        title=f"{track.upper()} — dataset features ({source})",
        xlabel="feature columns  (log-scaled)",
    )

    fig.tight_layout()
    return fig


def plot_missing_rate_distribution(
    track: str, *, source: str = "processed", cfg=None,
):
    """Per-track histogram of dataset-level missingness.

    "Missingness" here = **fraction of cells (rows × features) that
    are NaN**. Not "fraction of rows containing any NaN", which is a
    different metric. The y-axis is the number of datasets in each
    bin, and the x-axis is bounded ``[0, 1]``.
    """
    plt = _import_mpl()
    if track not in ("pd", "lgd"):
        raise ValueError("track must be 'pd' or 'lgd'")
    if source == "raw":
        summary = raw_corpus_summary(cfg)
        summary = summary[summary["track"] == track]
        col = "missing_cells_rate"
    elif source == "processed":
        summary = corpus_summary_table(track, cfg)
        col = "missing_rate_raw"
    else:
        raise ValueError("source must be 'raw' or 'processed'")

    color = _TRACK_COLOR[track]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(summary[col].dropna(),
            bins=np.linspace(0.0, 1.0, 31),
            color=color, alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.set_xlim(0.0, 1.0)
    _apply_style(
        ax,
        title=f"{track.upper()} — missingness ({source})",
        xlabel="missing rate  (fraction of NaN cells per dataset; "
               "denominator = rows × features)",
    )
    fig.tight_layout()
    return fig


def plot_class_imbalance_distribution(cfg=None):
    """PD-only: histogram of minority-class share across the corpus.

    For balanced binary classification this concentrates around 0.5;
    for credit-risk it typically clusters in 0.05–0.30. Useful to spot
    extreme outliers (datasets where the minority is < 1% — a hard
    signal that the labelling protocol differs from the rest of the
    corpus).
    """
    plt = _import_mpl()
    summary = corpus_summary_table("pd", cfg).dropna(subset=["minority_class_ratio"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(summary["minority_class_ratio"],
            bins=np.linspace(0.0, 0.5, 26),
            color=_TRACK_COLOR["pd"], alpha=0.85,
            edgecolor="white", linewidth=0.6)
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle=":",
               label="balanced (50%)")
    ax.set_xlim(0.0, 0.55)
    _apply_style(
        ax,
        title="PD — class-imbalance distribution across the corpus",
        xlabel="minority-class share (n_minority / n_total)",
    )
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def plot_target_mean_distribution_lgd(cfg=None):
    """LGD-only: histogram of target *means* across the corpus.

    Each dataset contributes its own LGD mean to the histogram. Useful
    to spot datasets whose mean LGD is suspiciously close to 0 or 1
    (i.e., the target is essentially constant — likely a labelling
    issue or a wrongly-set ``target_column`` in the metadata).
    """
    plt = _import_mpl()
    summary = corpus_summary_table("lgd", cfg).dropna(subset=["target_mean"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(summary["target_mean"],
            bins=np.linspace(0.0, 1.0, 26),
            color=_TRACK_COLOR["lgd"], alpha=0.85,
            edgecolor="white", linewidth=0.6)
    ax.set_xlim(0.0, 1.0)
    _apply_style(
        ax,
        title="LGD — target-mean distribution across the corpus",
        xlabel="dataset mean LGD",
    )
    fig.tight_layout()
    return fig


def plot_source_breakdown(cfg=None):
    """Corpus provenance: dataset count per source, split by track.

    Each raw dataset's ``source`` (``"kaggle"`` / ``"uci"`` /
    ``"freddie-mac"`` / ``"local"`` / …) comes from
    ``DATASET_METADATA`` and is carried into the manifest. A diverse
    source mix is a mild guard against the whole corpus inheriting one
    vendor's quirks.
    """
    plt = _import_mpl()
    summary = corpus_summary_table(cfg=cfg)
    if summary.empty or "source" not in summary.columns:
        return None
    counts = summary.groupby(["source", "track"]).size().unstack(fill_value=0)
    for tr in ("pd", "lgd"):
        if tr not in counts.columns:
            counts[tr] = 0
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    sources = list(counts.index)
    x = np.arange(len(sources))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(sources)), 4.5))
    ax.bar(x - width / 2, counts["pd"], width, label="PD",
           color=_TRACK_COLOR["pd"], alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.bar(x + width / 2, counts["lgd"], width, label="LGD",
           color=_TRACK_COLOR["lgd"], alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=20, ha="right")
    _apply_style(ax, title="Corpus provenance — datasets per source",
                 xlabel="source")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_size_scatter(
    track: str | None = None, *, source: str = "processed", cfg=None,
):
    """Per-dataset **features (y) vs rows (x)** shape scatter, shown as a
    **linear | log** pair in one row (1x2).

    Parameters
    ----------
    track
        ``"pd"`` or ``"lgd"`` for a single track; ``None`` (default) plots
        both tracks combined, coloured by track.
    source
        ``"raw"`` (pre-fix shapes) or ``"processed"`` (post-sanitise).

    Left panel = linear axes (absolute size); right panel = log-log (the
    corpus spans ~4-5 orders of magnitude in both axes). Tall-and-narrow
    datasets (many rows, few features) sit bottom-right; wide datasets sit
    top-left.
    """
    plt = _import_mpl()
    if source == "raw":
        summary = raw_corpus_summary(cfg)
        rcol, ccol = "raw_rows", "raw_cols"
    elif source == "processed":
        summary = corpus_summary_table(cfg=cfg)
        rcol, ccol = "post_rows", "post_features"
    else:
        raise ValueError("source must be 'raw' or 'processed'")
    if summary.empty:
        return None
    df = summary[(summary[rcol] > 0) & (summary[ccol] > 0)]
    if track is not None:
        if track not in ("pd", "lgd"):
            raise ValueError("track must be 'pd', 'lgd', or None")
        df = df[df["track"] == track]
    if df.empty:
        return None

    who = track.upper() if track else "PD + LGD"
    tracks = [track] if track else ("pd", "lgd")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, scale in zip(axes, ("linear", "log")):
        for tr in tracks:
            sub = df[df["track"] == tr]
            if sub.empty:
                continue
            ax.scatter(sub[rcol], sub[ccol], s=60, alpha=0.8,
                       color=_TRACK_COLOR[tr], edgecolor="black",
                       linewidth=0.4, label=tr.upper())
        if scale == "log":
            ax.set_xscale("log")
            ax.set_yscale("log")
            # Force explicit y-axis ticks spanning the actual data range so
            # the log panel always shows more than one label. Matplotlib's
            # auto locator can drop to a single tick (e.g. "10") when the
            # data sit close together in log-space.
            import math
            y_vals = df[ccol].replace(0, float("nan")).dropna()
            if not y_vals.empty:
                lo = 10 ** math.floor(math.log10(y_vals.min()))
                hi = 10 ** math.ceil(math.log10(y_vals.max()))
                import numpy as np
                ticks = [10**p for p in range(int(math.log10(lo)),
                                              int(math.log10(hi)) + 1)]
                if len(ticks) > 1:
                    ax.set_yticks(ticks)
                    ax.yaxis.set_major_formatter(
                        plt.matplotlib.ticker.FuncFormatter(
                            lambda v, _: f"{int(v):,}" if v >= 1 else str(v)))
                ax.set_ylim(lo * 0.5, hi * 2)
            _apply_style(ax, title="log", xlabel="rows  (log)", ylabel="features  (log)")
        else:
            _apply_style(ax, title="linear", xlabel="rows", ylabel="features")
        # Framed legend so the colour chips are clearly separated from the data
        leg = ax.legend(frameon=True, fancybox=True,
                        edgecolor="black", framealpha=0.9)
        leg.get_frame().set_linewidth(0.8)
    fig.suptitle(f"Shape space — {who}: features vs rows ({source})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_missing_cells_bar(
    track: str | None = None, *, source: str = "processed", cfg=None,
):
    """Per-dataset missing-cell rate as a sorted bar chart.

    ``track``: ``"pd"`` / ``"lgd"`` for one track, ``None`` for both
    (datasets coloured by track). Missing rate = fraction of NaN cells
    (denominator = rows × features). ``source``: ``"raw"`` / ``"processed"``.
    """
    plt = _import_mpl()
    if source == "raw":
        summary = raw_corpus_summary(cfg)
        col = "missing_cells_rate"
    elif source == "processed":
        summary = corpus_summary_table(cfg=cfg)
        col = "missing_rate_raw"
    else:
        raise ValueError("source must be 'raw' or 'processed'")
    if summary.empty or col not in summary.columns:
        return None
    df = summary.dropna(subset=[col]).copy()
    if track is not None:
        if track not in ("pd", "lgd"):
            raise ValueError("track must be 'pd', 'lgd', or None")
        df = df[df["track"] == track]
    if df.empty:
        return None
    df = df.sort_values(col, ascending=False)
    who = track.upper() if track else "PD + LGD"

    fig, ax = plt.subplots(figsize=(max(7, 0.32 * len(df)), 4.8))
    colors = [_TRACK_COLOR[t] for t in df["track"]]
    ax.bar(range(len(df)), df[col].to_numpy(), color=colors,
           alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["dataset_id"], rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0.0, max(0.01, float(df[col].max()) * 1.1))
    _apply_style(ax, title=f"Missing-cell rate per dataset — {who} ({source})",
                 xlabel="", ylabel="missing rate (NaN cells / rows×features)")
    if track is None:
        import matplotlib.patches as mpatches
        ax.legend(
            handles=[mpatches.Patch(color=_TRACK_COLOR["pd"], label="PD"),
                     mpatches.Patch(color=_TRACK_COLOR["lgd"], label="LGD")],
            frameon=False,
        )
    fig.tight_layout()
    return fig


def plot_feature_type_distribution(cfg=None):
    """Histogram of each dataset's categorical-feature share, per track.

    ``categorical share = n_categorical / (n_categorical + n_numerical)``.
    Credit-risk corpora are usually mostly-numerical, so this should
    cluster near 0; a dataset near 1 is worth a second look (likely a
    coding/labelling quirk).
    """
    plt = _import_mpl()
    summary = corpus_summary_table(cfg=cfg)
    if summary.empty:
        return None
    denom = (summary["n_categorical"] + summary["n_numerical"]).replace(0, np.nan)
    summary = summary.assign(cat_share=summary["n_categorical"] / denom)
    summary = summary.dropna(subset=["cat_share"])
    if summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0.0, 1.0, 21)
    for tr in ("pd", "lgd"):
        sub = summary[summary["track"] == tr]
        if sub.empty:
            continue
        ax.hist(sub["cat_share"], bins=bins, alpha=0.6, label=tr.upper(),
                color=_TRACK_COLOR[tr], edgecolor="white", linewidth=0.6)
    ax.set_xlim(0.0, 1.0)
    _apply_style(ax, title="Feature-type composition — categorical share per dataset",
                 xlabel="categorical share  (n_categorical / n_features)")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_feature_reduction(cfg=None):
    """Raw vs post-sanitise feature count — visualises feature selection.

    Points on the dashed ``y = x`` line were left untouched by sanitise.
    Points pulled down toward the red cap line had their feature count
    reduced by unsupervised **selection** (``sanitize.max_columns`` keeps
    the top-N *real* columns — not cluster means). The handful of
    >cap-feature datasets (e.g. ``algorithmwatch`` at ~2 987 cols) are
    the whole reason the selection step exists.
    """
    plt = _import_mpl()
    try:
        from omegaconf import OmegaConf
        cap = int(OmegaConf.load(_REPO / "config" / "data.yaml").sanitize.max_columns)
    except Exception:                                          # pragma: no cover
        cap = 64
    summary = corpus_summary_table(cfg=cfg)
    if summary.empty:
        return None
    df = summary[(summary["raw_features"] > 0) & (summary["post_features"] > 0)]
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for tr in ("pd", "lgd"):
        sub = df[df["track"] == tr]
        if sub.empty:
            continue
        ax.scatter(sub["raw_features"], sub["post_features"], s=55, alpha=0.8,
                   color=_TRACK_COLOR[tr], edgecolor="black",
                   linewidth=0.4, label=tr.upper())
    hi = float(df["raw_features"].max())
    ax.plot([1, hi], [1, hi], "k--", alpha=0.4, linewidth=0.8, label="y = x (unchanged)")
    ax.axhline(cap, color="red", linestyle=":", linewidth=1.0, alpha=0.7,
               label=f"{cap}-feature cap (selection)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _apply_style(ax, title="Feature reduction — raw vs post-sanitise feature count",
                 xlabel="raw features  (log-scaled)",
                 ylabel="post-sanitise features  (log-scaled)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_target_distribution_lgd(
    dataset_ids: Iterable[str] | None = None,
    *,
    max_show: int = 30,
    cfg=None,
):
    """Grid of LGD target histograms.

    Designed for the 3 000-dataset case: by default shows the first
    ``max_show=30`` datasets. To inspect a specific subset, pass
    ``dataset_ids=[...]`` explicitly.

    Two structural facts each subplot reports in its title:
    fraction of mass at LGD = 0 (full recovery), and fraction at
    LGD = 1 (total loss). The interior shape between those two
    spikes is what the regressor has to model.
    """
    plt = _import_mpl()
    summary = corpus_summary_table("lgd", cfg)
    if dataset_ids is not None:
        ids = list(dataset_ids)
        summary = summary[summary["dataset_id"].isin(ids)]
    summary = summary.head(max_show)
    n = len(summary)
    if n == 0:
        return None
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    for ax, (_, mrow) in zip(axes.flat, summary.iterrows()):
        df = load_sanitized_dataset("lgd", mrow["dataset_id"], cfg)
        y = pd.to_numeric(df[mrow["target_column"]], errors="coerce").dropna()
        ax.hist(y, bins=40, color="tab:orange", alpha=0.85)
        frac_zero = float((y == 0).mean())
        frac_one = float((y == 1).mean())
        ax.set_title(
            f"{mrow['dataset_id']}\n"
            f"n={len(y):,}, μ={y.mean():.3f}, σ={y.std():.3f}\n"
            f"P(LGD=0)={frac_zero:.2%}  P(LGD=1)={frac_one:.2%}",
            fontsize=9,
        )
        ax.set_xlabel("LGD")
        ax.set_ylabel("count")
        ax.set_xlim(-0.02, 1.02)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_target_distribution_pd(
    dataset_ids: Iterable[str] | None = None,
    *,
    max_show: int = 30,
    cfg=None,
):
    """Grid of PD class-proportion bar charts. See the LGD twin's
    docstring for the pagination semantics."""
    plt = _import_mpl()
    summary = corpus_summary_table("pd", cfg)
    if dataset_ids is not None:
        ids = list(dataset_ids)
        summary = summary[summary["dataset_id"].isin(ids)]
    summary = summary.head(max_show)
    n = len(summary)
    if n == 0:
        return None
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    for ax, (_, mrow) in zip(axes.flat, summary.iterrows()):
        df = load_sanitized_dataset("pd", mrow["dataset_id"], cfg)
        y = df[mrow["target_column"]].dropna()
        vc = y.value_counts(normalize=True).sort_index()
        ax.bar(vc.index.astype(str), vc.values, color="tab:blue", alpha=0.85)
        ax.set_title(
            f"{mrow['dataset_id']}\n"
            f"n={len(y):,}  classes={int(y.nunique())}\n"
            f"minority share={mrow['minority_class_ratio']:.3f}",
            fontsize=9,
        )
        ax.set_ylabel("share")
        ax.set_ylim(0, 1.05)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Cached: chunk-level views (per-dataset, paginated)
# --------------------------------------------------------------------------- #


def find_anomalous_datasets(
    cfg=None,
    *,
    max_missing_rate: float = 0.50,
    min_post_rows: int = 100,
    max_minority_share: float = 0.005,
) -> pd.DataFrame:
    """Flag corpus members with anomalous indicators.

    Returns a DataFrame whose rows are *only* the anomalous datasets,
    with one boolean column per indicator plus a ``reasons``
    semicolon-list column for at-a-glance triage.

    Indicators:

    * ``empty_processed`` — sanitisation produced 0 rows.
    * ``too_few_rows`` — fewer than ``min_post_rows`` rows after
      sanitisation. Default 100 is a soft floor below which TabPFN
      fine-tuning becomes pointless.
    * ``high_missing`` — more than ``max_missing_rate`` of cells are
      NaN. Default 50 % is well above the corpus norm and signals
      either bad source data or aggressive column drops.
    * ``severely_imbalanced`` — minority share below
      ``max_minority_share`` (PD only). Default 0.005 = 0.5 %.
    * ``constant_target`` — target column has 1 unique value.
    * ``feature_count_zero`` — zero non-target columns survived.
    """
    summary = corpus_summary_table(cfg=cfg)
    flags: list[dict] = []
    for _, row in summary.iterrows():
        reasons: list[str] = []
        if row["post_rows"] == 0:
            reasons.append("empty_processed")
        if 0 < row["post_rows"] < min_post_rows:
            reasons.append("too_few_rows")
        if row["missing_rate_raw"] > max_missing_rate:
            reasons.append("high_missing")
        if (row["task_type"] == "classification"
                and pd.notna(row["minority_class_ratio"])
                and row["minority_class_ratio"] < max_minority_share):
            reasons.append("severely_imbalanced")
        if row["post_features"] == 0:
            reasons.append("feature_count_zero")
        if row["task_type"] == "regression" and row["target_std"] == 0.0:
            reasons.append("constant_target")
        if reasons:
            flags.append({
                "track": row["track"], "dataset_id": row["dataset_id"],
                "reasons": ";".join(reasons),
                **{f"flag_{r}": True for r in reasons},
            })
    return pd.DataFrame(flags)


