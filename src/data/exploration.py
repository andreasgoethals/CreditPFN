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
                  "is the cleaning sound" view.
* **Cached**    — ``data/cached/{pd,lgd}/<id>/chunk_NNN.npz`` +
                  ``meta.json``. Read by ``cached_*`` helpers / the
                  ``cached_data_exploration`` notebook. The "is the
                  training input shape healthy" view.

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
* **chunk** — one ``.npz`` file under ``data/cached/...``. Each chunk
  is a self-contained ``(X_context, y_context, X_query, y_query)``
  tuple ready for the multi-table fine-tuning loop. Datasets larger
  than ``cfg.dataset.max_rows_per_chunk`` become multiple chunks.

Public surface — corpus-level (scales to 3 000 datasets)
--------------------------------------------------------
* :func:`raw_corpus_summary` — one row per raw CSV.
* :func:`corpus_summary_table` — one row per dataset, manifest +
  on-disk processed shapes side-by-side.
* :func:`cached_corpus_summary` — one row per cached dataset.
* :func:`plot_dataset_size_distribution` — ``track`` is required;
  one plot for one track at a time. Two histograms: rows and
  features.
* :func:`plot_missing_rate_distribution` — ``track`` required.
  Single histogram with explicit "% of cells" axis label.
* :func:`plot_class_imbalance_distribution` — PD only.
* :func:`plot_target_mean_distribution_lgd` — corpus-level histogram
  of LGD target means across all datasets.
* :func:`plot_chunk_count_distribution` — cached.
* :func:`plot_chunk_size_distribution` — cached.

Public surface — per-dataset (paginated for scale)
--------------------------------------------------
* :func:`plot_target_distribution_pd` — paginated grid; default
  ``max_show=30`` first IDs, override via ``dataset_ids=[…]``.
* :func:`plot_target_distribution_lgd` — same.

Public surface — error detection
--------------------------------
* :func:`find_anomalous_datasets` — flag corpus members with
  anomalous values on any of N indicators.
* :func:`find_anomalous_chunks` — same for cached chunks.

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
            cached=str(_REPO / "data" / "cached"),
            manifest_pd=str(_REPO / "data" / "manifest_pd.csv"),
            manifest_lgd=str(_REPO / "data" / "manifest_lgd.csv"),
        ))


def _resolve_paths(cfg=None) -> dict[str, Path]:
    """Return absolute paths derived from ``cfg`` (or default cfg).

    Mirrors the same resolver split used by the data pipeline:

      * raw / processed / cached → ``$CREDITPFN_DATA_ROOT`` (scratch on VSC)
      * manifest_*               → ``$CREDITPFN_OUTPUT_ROOT`` (durable on VSC)
    """
    from src.utils.paths import resolve_data_path, resolve_output_path
    if cfg is None:
        cfg = _load_default_cfg()

    raw_default       = "data/raw"
    cached_default    = "data/cached"
    return {
        "raw":          resolve_data_path(getattr(cfg.paths, "raw", raw_default)),
        "processed":    resolve_data_path(cfg.paths.processed),
        "cached":       resolve_data_path(getattr(cfg.paths, "cached", cached_default)),
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


def load_cached_meta(track: str, dataset_id: str, cfg=None) -> dict:
    """Read the meta.json sidecar for one cached dataset."""
    paths = _resolve_paths(cfg)
    p = paths["cached"] / track / dataset_id / "meta.json"
    if not p.exists():
        raise FileNotFoundError(f"meta.json not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def list_cached_chunks(track: str, dataset_id: str, cfg=None) -> list[Path]:
    """Return sorted list of ``chunk_NNN.npz`` paths for one dataset."""
    paths = _resolve_paths(cfg)
    folder = paths["cached"] / track / dataset_id
    if not folder.exists():
        return []
    return sorted(folder.glob("chunk_*.npz"))


def load_cached_chunk(path: Path) -> Mapping[str, np.ndarray]:
    """Load one ``.npz`` chunk as a dict-like read-only view."""
    return np.load(path)


# =============================================================================
# Corpus-level summaries
# =============================================================================
#
# Performance note: ``raw_corpus_summary`` and ``cached_corpus_summary``
# read every dataset on disk. With wide datasets (``algorithmwatch`` is
# 159 k × 2 987) a single pass is on the order of tens of seconds, and
# the exploration notebooks call these functions multiple times
# (once per plot). To avoid the resulting "every cell takes a minute"
# experience we memoise the result by ``(function, repr(cfg))`` so a
# notebook session re-uses the first computation.
#
# Pass ``refresh=True`` to bust the cache (e.g. after a rebuild).

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


def cached_corpus_summary(
    track: str | None = None, cfg=None, *, refresh: bool = False,
) -> pd.DataFrame:
    """One row per cached dataset. Memoised.

    For each dataset, summarises:

    * ``n_chunks`` — number of ``chunk_*.npz`` files on disk.
    * ``mean_chunk_rows`` — mean chunk size (context + query).
    * ``total_size_mb`` — sum of ``.npz`` file sizes.
    * ``ctx_query_ratio`` — actual ``len(X_context) / total`` averaged
      across chunks. Should be close to 0.60. Deviations indicate
      a stale cache vs. updated cfg.
    * ``unknown_sentinel_rate`` — fraction of cells in the *query*
      categorical columns that equal the encoder's ``unknown_value``
      (``-1`` by default). 0.0 means the dataset never tests TabPFN
      on unseen-in-context categories — fine for low-cardinality
      categoricals, suspicious for high-cardinality ones.
    * ``nan_rate_in_X`` — fraction of NaN cells across all chunk
      ``X_*`` arrays.
    """
    key = _cache_key(f"cached_corpus_summary:{track}", cfg)
    if not refresh and key in _SUMMARY_CACHE:
        return _SUMMARY_CACHE[key].copy()

    from src.data.preprocessing import DATASET_METADATA
    paths = _resolve_paths(cfg)
    rows: list[dict] = []
    tracks = ["pd", "lgd"] if track is None else [track]
    for did, meta in DATASET_METADATA.items():
        if meta["track"] not in tracks:
            continue
        chunks = list_cached_chunks(meta["track"], did, cfg)
        if not chunks:
            rows.append({
                "track": meta["track"], "dataset_id": did,
                "n_chunks": 0,
                "mean_chunk_rows": np.nan, "total_size_mb": np.nan,
                "ctx_query_ratio": np.nan, "unknown_sentinel_rate": np.nan,
                "nan_rate_in_X": np.nan, "task_type": meta["task_type"],
            })
            continue
        sizes = [p.stat().st_size for p in chunks]
        ratios: list[float] = []
        chunk_rows: list[int] = []
        unk_counts = unk_total = nan_count = nan_total = 0
        for p in chunks:
            d = load_cached_chunk(p)
            n_ctx, n_qry = len(d["X_context"]), len(d["X_query"])
            chunk_rows.append(n_ctx + n_qry)
            ratios.append(n_ctx / max(1, n_ctx + n_qry))
            cat_idx = d["categorical_idx"].tolist()
            if cat_idx:
                # Unknown-sentinel rate computed on the QUERY split only.
                qry_cats = d["X_query"][:, cat_idx]
                unk_counts += int(np.sum(qry_cats == -1.0))
                unk_total += qry_cats.size
            for arr_name in ("X_context", "X_query"):
                arr = d[arr_name]
                nan_count += int(np.isnan(arr).sum())
                nan_total += arr.size
        rows.append({
            "track": meta["track"],
            "dataset_id": did,
            "n_chunks": len(chunks),
            "mean_chunk_rows": float(np.mean(chunk_rows)),
            "total_size_mb": sum(sizes) / (1024 * 1024),
            "ctx_query_ratio": float(np.mean(ratios)),
            "unknown_sentinel_rate": (unk_counts / unk_total) if unk_total else 0.0,
            "nan_rate_in_X": (nan_count / nan_total) if nan_total else 0.0,
            "task_type": meta["task_type"],
        })
    out = _round_summary(pd.DataFrame(rows))
    _SUMMARY_CACHE[key] = out
    return out.copy()


# =============================================================================
# Plot helpers
# =============================================================================


def _import_mpl():
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- #
# Corpus-level (scale to 3 000)
# --------------------------------------------------------------------------- #


def plot_dataset_size_distribution(
    track: str, *, source: str = "processed", cfg=None,
):
    """Per-track histograms of dataset rows and feature columns.

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

    color = {"pd": "tab:blue", "lgd": "tab:orange"}[track]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(summary[rows_col], bins=30, color=color, alpha=0.85)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("rows  (log-scaled)")
    axes[0].set_ylabel("# datasets")
    axes[0].set_title(f"{track.upper()} — dataset rows ({source})")
    axes[1].hist(summary[cols_col], bins=30, color=color, alpha=0.85)
    axes[1].set_xlabel("feature columns")
    axes[1].set_ylabel("# datasets")
    axes[1].set_title(f"{track.upper()} — dataset features ({source})")
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

    color = {"pd": "tab:blue", "lgd": "tab:orange"}[track]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(summary[col].dropna(), bins=30, color=color, alpha=0.85,
            range=(0.0, 1.0))
    ax.set_xlabel("missing rate  (fraction of NaN cells per dataset, "
                  "denominator = rows × features)")
    ax.set_ylabel("# datasets")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"{track.upper()} — missingness ({source})")
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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(summary["minority_class_ratio"], bins=30, color="tab:blue",
            alpha=0.85, range=(0.0, 0.5))
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle=":",
               label="balanced (50%)")
    ax.set_xlim(0.0, 0.55)
    ax.set_xlabel("minority-class share (n_minority / n_total)")
    ax.set_ylabel("# datasets")
    ax.set_title("PD — class-imbalance distribution across the corpus")
    ax.legend()
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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(summary["target_mean"], bins=30, color="tab:orange",
            alpha=0.85, range=(0.0, 1.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("dataset mean LGD")
    ax.set_ylabel("# datasets")
    ax.set_title("LGD — target-mean distribution across the corpus")
    fig.tight_layout()
    return fig


def plot_chunk_count_distribution(cfg=None):
    """Histogram of chunks-per-dataset, faceted by track.

    Most credit-risk datasets are < ~20 k rows so most produce 1 chunk;
    a few large ones (`hackerearth`, `home_credit`, `algorithmwatch`)
    produce 8–30. Long-tail outliers indicate either very large
    parent datasets or a too-small ``max_rows_per_chunk``.
    """
    plt = _import_mpl()
    summary = cached_corpus_summary(cfg=cfg)
    fig, ax = plt.subplots(figsize=(8, 4))
    for tr, color in [("pd", "tab:blue"), ("lgd", "tab:orange")]:
        sub = summary[summary["track"] == tr]
        ax.hist(sub["n_chunks"], bins=30, alpha=0.6, label=tr.upper(),
                color=color)
    ax.set_xlabel("chunks per dataset")
    ax.set_ylabel("# datasets")
    ax.set_title("Cached — chunk count per dataset")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_chunk_size_distribution(cfg=None):
    """Histogram of mean chunk size in rows, faceted by track."""
    plt = _import_mpl()
    summary = cached_corpus_summary(cfg=cfg)
    fig, ax = plt.subplots(figsize=(8, 4))
    for tr, color in [("pd", "tab:blue"), ("lgd", "tab:orange")]:
        sub = summary[summary["track"] == tr].dropna(subset=["mean_chunk_rows"])
        ax.hist(sub["mean_chunk_rows"], bins=30, alpha=0.6, label=tr.upper(),
                color=color)
    ax.set_xlabel("mean chunk size  (rows in X_context + X_query)")
    ax.set_ylabel("# datasets")
    ax.set_title("Cached — mean chunk size per dataset")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_unknown_sentinel_rate(cfg=None):
    """Histogram of the per-dataset unknown-sentinel rate (cached).

    See the cached_corpus_summary docstring for what
    ``unknown_sentinel_rate`` means. A dataset at 0 means the encoder
    never had to substitute -1 in the query — fine for small
    categorical vocabularies, suspicious for wide ones (suggests
    every category is well-covered by chance, or your context split
    is too large). A dataset > 0.10 means a tenth of categorical
    cells in the query were unseen in context — that's exactly the
    scenario TabPFN must learn to handle.
    """
    plt = _import_mpl()
    summary = cached_corpus_summary(cfg=cfg)
    summary = summary[summary["unknown_sentinel_rate"].notna()]
    fig, ax = plt.subplots(figsize=(8, 4))
    for tr, color in [("pd", "tab:blue"), ("lgd", "tab:orange")]:
        sub = summary[summary["track"] == tr]
        if len(sub):
            ax.hist(sub["unknown_sentinel_rate"], bins=30, alpha=0.6,
                    label=tr.upper(), color=color)
    ax.set_xlabel("unknown-sentinel rate  (fraction of -1 in query categoricals)")
    ax.set_ylabel("# datasets")
    ax.set_title("Cached — unknown-sentinel rate per dataset")
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Per-dataset (paginated)
# --------------------------------------------------------------------------- #


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


def plot_chunk_target_consistency(
    track: str,
    dataset_ids: Iterable[str] | None = None,
    *,
    max_show: int = 12,
    cfg=None,
):
    """Per-dataset target stats per chunk.

    For PD: bar chart of class-1 fraction in each chunk's query split.
    Stratified chunking should make these almost identical across
    chunks of the same dataset; large drift means stratification
    failed for that dataset.

    For LGD: bar chart of mean target per chunk. Random chunking
    should keep means within ~1 sample-error of each other; an
    outlier chunk indicates suspicious mass concentration.
    """
    plt = _import_mpl()
    if track not in ("pd", "lgd"):
        raise ValueError("track must be 'pd' or 'lgd'")
    summary = cached_corpus_summary(track, cfg)
    summary = summary[summary["n_chunks"] > 1]
    if dataset_ids is not None:
        summary = summary[summary["dataset_id"].isin(list(dataset_ids))]
    summary = summary.head(max_show)
    n = len(summary)
    if n == 0:
        return None
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 2.5 * nrows),
                             squeeze=False)
    color = {"pd": "tab:blue", "lgd": "tab:orange"}[track]
    for ax, (_, row) in zip(axes.flat, summary.iterrows()):
        did = row["dataset_id"]
        chunks = list_cached_chunks(track, did, cfg)
        stats = []
        for p in chunks:
            d = load_cached_chunk(p)
            y = np.concatenate([d["y_context"], d["y_query"]])
            if track == "pd":
                stats.append(float((y == 1).mean()))
            else:
                stats.append(float(np.nanmean(y)))
        ax.bar(range(len(stats)), stats, color=color, alpha=0.85)
        if track == "pd":
            ax.set_ylabel("class-1 fraction")
            ax.set_ylim(0, max(0.5, max(stats) * 1.2))
        else:
            ax.set_ylabel("mean target")
            ax.set_ylim(0, 1)
        ax.set_xlabel("chunk index")
        ax.set_title(f"{did}  ({len(stats)} chunks)", fontsize=9)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig


# =============================================================================
# Error detection
# =============================================================================


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


def find_anomalous_chunks(
    cfg=None,
    *,
    min_chunks: int = 1,
    min_unknown_for_warning: float = 0.30,
    max_nan_rate: float = 0.50,
    ctx_query_tolerance: float = 0.10,
) -> pd.DataFrame:
    """Flag cached datasets with anomalous chunk-level indicators.

    Indicators (each becomes a ``flag_*`` column plus a token in
    the ``reasons`` semicolon-list):

    * ``no_chunks`` — fewer than ``min_chunks`` chunks on disk.
    * ``ctx_query_off`` — average context fraction differs from the
      configured ``cfg.dataset.context_fraction`` by more than
      ``ctx_query_tolerance`` (default ±0.10). Signals a stale cache
      relative to the current config.
    * ``high_unknown_sentinel`` — more than
      ``min_unknown_for_warning`` of query categorical cells were
      unseen in context. Not necessarily wrong (TabPFN can handle
      it), but noteworthy at very high rates.
    * ``too_many_nans`` — more than ``max_nan_rate`` of X-cells are
      NaN. Indicates the sanitisation didn't drop enough.
    """
    cfg = cfg or _load_default_cfg()
    summary = cached_corpus_summary(cfg=cfg)
    expected_ctx = float(cfg.dataset.context_fraction) if hasattr(cfg, "dataset") else 0.60
    flags: list[dict] = []
    for _, row in summary.iterrows():
        reasons: list[str] = []
        if row["n_chunks"] < min_chunks:
            reasons.append("no_chunks")
        if (pd.notna(row["ctx_query_ratio"])
                and abs(row["ctx_query_ratio"] - expected_ctx) > ctx_query_tolerance):
            reasons.append("ctx_query_off")
        if (pd.notna(row["unknown_sentinel_rate"])
                and row["unknown_sentinel_rate"] > min_unknown_for_warning):
            reasons.append("high_unknown_sentinel")
        if (pd.notna(row["nan_rate_in_X"])
                and row["nan_rate_in_X"] > max_nan_rate):
            reasons.append("too_many_nans")
        if reasons:
            flags.append({
                "track": row["track"], "dataset_id": row["dataset_id"],
                "reasons": ";".join(reasons),
                **{f"flag_{r}": True for r in reasons},
            })
    return pd.DataFrame(flags)
