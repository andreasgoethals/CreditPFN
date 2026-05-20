"""Training-pipeline visualisation helpers.

The training pipeline (``scripts/train_pipeline.py``) writes two
kinds of artefacts that this module consumes:

* **Per-trial epoch CSV** — one file per trial under
  ``output/training/epochs/<track>/<descriptive_name>.csv`` with columns::

      epoch, train_loss, lr, metric_name,
      train_metric, test_metric, epoch_time_sec, elapsed_sec

  (The writer is ``_on_epoch_end`` inside
  ``scripts/train_pipeline.py``.)

* **Run manifest CSV** — one file per track at
  ``output/training/manifests/<run_name>_<track>.csv``. Each row is one trial::

      track, base_checkpoint, learning_rate, use_lora, seed,
      n_train_datasets, n_test_datasets, n_train_chunks, n_test_chunks,
      final_ckpt_path, elapsed_sec, status, error

  (The dataclass is ``RunRow`` in the same script.)

The descriptive_name encodes the trial hyperparameters in the
filename (see ``src.train.loop.descriptive_name``)::

    <run_name>_<track>_<base-stem>_lr<lr_tag>_seed<seed>[_lora]

so we can recover ``(base, lr, seed, lora)`` from the filename
alone and treat the per-epoch CSV as self-describing.

Two design contracts
--------------------
1. **All plots return the matplotlib Figure.** Callers in notebooks
   can ``fig.savefig(...)`` or further customise without us having
   to thread `ax` kwargs everywhere.
2. **Empty-data graceful.** Every loader returns an empty
   DataFrame and every plot returns an empty Figure with an
   "(no data)" message when nothing is on disk yet — so the
   notebooks render even before the first training run finishes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]


# =============================================================================
# Cfg + path resolution
# =============================================================================


def _load_default_cfg():
    """Load ``config/train.yaml`` as the source-of-truth for run_name etc."""
    try:
        from omegaconf import OmegaConf
        return OmegaConf.load(_REPO / "config" / "train.yaml")
    except Exception:  # pragma: no cover  — local fallback
        from types import SimpleNamespace as _NS
        return _NS(run_name="creditpfn", track="pd",
                   checkpoint=_NS(trained_dir="checkpoints/trained"))


def _resolve_paths(cfg=None) -> dict[str, Path]:
    """Resolve the on-disk roots for the training artefacts.

    Mirrors :func:`src.data.exploration._resolve_paths`.
    """
    # Apply paths.data_source from config/data.yaml so the data root is
    # consistent with the rest of the pipeline. Cheap, idempotent.
    try:
        from omegaconf import OmegaConf
        from src.utils.paths import apply_data_source_from_cfg
        apply_data_source_from_cfg(OmegaConf.load(_REPO / "config" / "data.yaml"))
    except Exception:  # pragma: no cover  — local fallback
        pass

    from src.utils.paths import resolve_output_path
    if cfg is None:
        cfg = _load_default_cfg()

    run_name = str(getattr(cfg, "run_name", "creditpfn"))
    trained_dir = str(getattr(cfg, "checkpoint", _load_default_cfg().checkpoint)
                      .trained_dir if hasattr(cfg, "checkpoint")
                      else "checkpoints/trained")

    return {
        "epoch_dir":     resolve_output_path("output/training/epochs"),
        "manifest_dir":  resolve_output_path("output/training/manifests"),
        "trained_dir":   resolve_output_path(trained_dir),
        "run_name":      run_name,
    }


# =============================================================================
# Descriptive-name parsing
# =============================================================================


@dataclass(frozen=True)
class TrialId:
    """Parsed view of a descriptive_name filename.

    ``descriptive_name`` is::

        <run_name>_<track>_<base-stem>_lr<lr_tag>_seed<seed>[_lora]

    where ``base-stem`` itself contains underscores
    (``tabpfn-v3-classifier-v3_default``). We parse FROM THE END
    because the only fixed pieces are the suffixes.
    """
    name: str           # the full descriptive_name (no extension)
    run_name: str
    track: str          # "pd" | "lgd"
    base: str           # the base-stem, e.g. "tabpfn-v3-classifier-v3_default"
    lr: float
    seed: int
    lora: bool

    @property
    def base_short(self) -> str:
        """Human-friendlier label: drop the "tabpfn-" prefix and
        "-default" suffix, collapse "v3-classifier-v3" → "v3-classifier"."""
        s = self.base
        for prefix in ("tabpfn-",):
            if s.startswith(prefix):
                s = s[len(prefix):]
        s = re.sub(r"-v(\d+(?:\.\d+)?)_default(.*)$", r"-v\1\2", s)
        s = s.removesuffix("_default")
        return s

    @property
    def label(self) -> str:
        """Compact label for plot legends."""
        lora_tag = " ·LoRA" if self.lora else ""
        return f"{self.base_short}  lr={self.lr:.0e}{lora_tag}"


_NAME_RE = re.compile(
    r"^(?P<run>.+?)_(?P<track>pd|lgd)_"
    r"(?P<base>.+?)"
    r"_lr(?P<lr>[0-9eE.+\-]+)"
    r"_seed(?P<seed>\d+)"
    r"(?P<lora>_lora)?$"
)


def parse_trial_name(name: str) -> TrialId | None:
    """Parse a descriptive_name (with or without extension)."""
    stem = Path(name).stem      # strips .csv / .ckpt
    stem = stem.removesuffix(".ckpt")  # belt-and-braces
    m = _NAME_RE.match(stem)
    if not m:
        return None
    try:
        return TrialId(
            name=stem,
            run_name=m.group("run"),
            track=m.group("track"),
            base=m.group("base"),
            lr=float(m.group("lr")),
            seed=int(m.group("seed")),
            lora=bool(m.group("lora")),
        )
    except (TypeError, ValueError):                     # pragma: no cover
        return None


# =============================================================================
# Loaders
# =============================================================================


def load_run_manifest(track: str, cfg=None) -> pd.DataFrame:
    """Load ``manifests/<run_name>_<track>.csv`` as a DataFrame.

    Adds parsed columns ``base_short`` (humanised), ``trial_name``
    (descriptive_name with extension stripped) for convenience. Returns
    an empty DataFrame if the file doesn't exist yet (so notebook
    cells still render before any trial finishes).
    """
    paths = _resolve_paths(cfg)
    p = paths["manifest_dir"] / f"{paths['run_name']}_{track}.csv"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    if df.empty:
        return df

    # Derive trial_name + base_short from ckpt path (when available) or
    # rebuild from columns.
    def _stem(row) -> str:
        if isinstance(row["final_ckpt_path"], str) and row["final_ckpt_path"]:
            return Path(row["final_ckpt_path"]).stem
        # FAIL rows have no ckpt — reconstruct.
        base_stem = Path(str(row["base_checkpoint"])).stem
        lr_tag = f"{float(row['learning_rate']):.0e}".replace("+", "")
        lora_tag = "_lora" if bool(row.get("use_lora", False)) else ""
        return (
            f"{paths['run_name']}_{row['track']}_{base_stem}_"
            f"lr{lr_tag}_seed{int(row['seed'])}{lora_tag}"
        )

    df["trial_name"] = df.apply(_stem, axis=1)
    df["base_short"] = df["trial_name"].map(
        lambda n: (parse_trial_name(n).base_short if parse_trial_name(n) else "?")
    )
    df["lr_tag"] = df["learning_rate"].map(lambda x: f"{float(x):.0e}".replace("+", ""))
    return df


def load_epoch_history(trial_name: str, track: str, cfg=None) -> pd.DataFrame:
    """Load one trial's per-epoch CSV."""
    paths = _resolve_paths(cfg)
    stem = Path(trial_name).stem.removesuffix(".ckpt")
    p = paths["epoch_dir"] / track / f"{stem}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_all_epoch_histories(track: str, cfg=None) -> dict[str, pd.DataFrame]:
    """Load every per-epoch CSV under ``output/training/epochs/<track>/``.

    Returns a dict keyed by the file stem (== descriptive_name).
    """
    paths = _resolve_paths(cfg)
    dir_ = paths["epoch_dir"] / track
    if not dir_.exists():
        return {}
    out: dict[str, pd.DataFrame] = {}
    for csv in sorted(dir_.glob("*.csv")):
        try:
            out[csv.stem] = pd.read_csv(csv)
        except Exception as exc:                         # pragma: no cover
            LOGGER.warning("could not read %s: %s", csv, exc)
    return out


def training_overview(track: str, cfg=None) -> pd.DataFrame:
    """Wide overview table — one row per trial, joining the manifest
    with derived per-epoch stats (best test_metric, best epoch, etc.).

    Columns
    -------
    trial_name         — the descriptive_name (no extension)
    base_short         — humanised base checkpoint
    learning_rate      — float
    use_lora           — bool
    seed               — int
    status             — "OK" | "FAIL"
    n_epochs           — number of recorded epochs (NaN if FAIL or no CSV)
    final_train_loss   — last epoch's train_loss
    final_train_metric — last epoch's train_metric
    final_test_metric  — last epoch's test_metric
    best_test_metric   — best test_metric across epochs (max for AUC, min for RMSE)
    best_epoch         — argmax/argmin of test_metric
    metric_name        — "roc_auc" (PD) | "rmse" (LGD)
    elapsed_sec        — total training time
    mean_epoch_sec     — average epoch wall-clock (excludes setup)
    """
    manifest = load_run_manifest(track, cfg=cfg)
    if manifest.empty:
        return pd.DataFrame()

    histories = load_all_epoch_histories(track, cfg=cfg)
    rows: list[dict] = []
    for _, m_row in manifest.iterrows():
        trial = str(m_row["trial_name"])
        hist = histories.get(trial, pd.DataFrame())
        out = {
            "trial_name":     trial,
            "base_short":     m_row["base_short"],
            "base_checkpoint": m_row["base_checkpoint"],
            "learning_rate":  float(m_row["learning_rate"]),
            "use_lora":       bool(m_row.get("use_lora", False)),
            "seed":           int(m_row.get("seed", 0)),
            "status":         str(m_row.get("status", "")),
            "elapsed_sec":    float(m_row.get("elapsed_sec", 0.0)),
            "n_train_datasets": int(m_row.get("n_train_datasets", 0)),
            "n_test_datasets":  int(m_row.get("n_test_datasets", 0)),
        }
        if hist.empty:
            for col in (
                "n_epochs", "final_train_loss", "final_train_metric",
                "final_test_metric", "best_test_metric", "best_epoch",
                "metric_name", "mean_epoch_sec",
            ):
                out[col] = np.nan if col != "metric_name" else ""
            rows.append(out)
            continue

        last = hist.iloc[-1]
        metric_name = str(last.get("metric_name", ""))
        # Direction of improvement: higher-is-better for roc_auc / r2 /
        # accuracy / f1; lower-is-better for rmse / mae / log_loss / nll.
        higher_is_better = metric_name in {
            "roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall", "r2",
        }
        test_series = pd.to_numeric(hist["test_metric"], errors="coerce")
        if higher_is_better:
            best_idx = int(test_series.idxmax()) if test_series.notna().any() else -1
        else:
            best_idx = int(test_series.idxmin()) if test_series.notna().any() else -1
        best_epoch = int(hist["epoch"].iloc[best_idx]) if best_idx >= 0 else np.nan
        best_test = float(test_series.iloc[best_idx]) if best_idx >= 0 else np.nan

        out.update({
            "n_epochs":           int(hist["epoch"].nunique()),
            "final_train_loss":   float(last["train_loss"]),
            "final_train_metric": float(last.get("train_metric", np.nan)),
            "final_test_metric":  float(last.get("test_metric",  np.nan)),
            "best_test_metric":   best_test,
            "best_epoch":         best_epoch,
            "metric_name":        metric_name,
            "mean_epoch_sec":     float(hist["epoch_time_sec"].mean()
                                        if "epoch_time_sec" in hist.columns
                                        else np.nan),
        })
        rows.append(out)
    return pd.DataFrame(rows)


def metric_direction(track: str, history: pd.DataFrame | None = None) -> str:
    """Return ``"max"`` or ``"min"`` for the primary monitoring metric."""
    if history is not None and "metric_name" in history.columns and len(history):
        name = str(history["metric_name"].iloc[0])
    else:
        name = "roc_auc" if track == "pd" else "rmse"
    if name in {"roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall", "r2"}:
        return "max"
    return "min"


# =============================================================================
# Per-trial plots
# =============================================================================


def _new_fig(title: str, *, figsize=(8, 4.5)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    return fig, ax


def _no_data_fig(reason: str = "no data"):
    """Render a stub figure with a centred message — for notebooks that
    run before any training output is on disk."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, f"({reason})", ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="#888")
    ax.set_axis_off()
    return fig


def plot_loss_curve(trial_name: str, track: str, cfg=None):
    """Train loss vs epoch for one trial."""
    hist = load_epoch_history(trial_name, track, cfg=cfg)
    if hist.empty:
        return _no_data_fig(f"no epoch CSV for {trial_name}")
    fig, ax = _new_fig(f"Training loss — {trial_name}")
    ax.plot(hist["epoch"], hist["train_loss"],
            marker="o", linestyle="-", markersize=3, linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    return fig


def plot_lr_schedule(trial_name: str, track: str, cfg=None):
    """Learning rate vs epoch (warmup + cosine decay)."""
    hist = load_epoch_history(trial_name, track, cfg=cfg)
    if hist.empty:
        return _no_data_fig(f"no epoch CSV for {trial_name}")
    fig, ax = _new_fig(f"Learning-rate schedule — {trial_name}")
    ax.plot(hist["epoch"], hist["lr"], marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    return fig


def plot_metric_curves(trial_name: str, track: str, cfg=None):
    """Train vs test monitoring metric, one twin-axis figure."""
    hist = load_epoch_history(trial_name, track, cfg=cfg)
    if hist.empty:
        return _no_data_fig(f"no epoch CSV for {trial_name}")
    fig, ax = _new_fig(f"Train vs test metric — {trial_name}")
    metric_name = str(hist["metric_name"].iloc[0]) if "metric_name" in hist.columns else ""
    ax.plot(hist["epoch"], hist["train_metric"],
            label=f"train {metric_name}", marker="o", markersize=3, linewidth=1.5)
    ax.plot(hist["epoch"], hist["test_metric"],
            label=f"test {metric_name}", marker="s", markersize=3, linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_name or "metric")
    ax.legend(loc="best")
    return fig


def plot_epoch_time(trial_name: str, track: str, cfg=None):
    """Per-epoch wall-clock — useful to spot a slow chunk."""
    hist = load_epoch_history(trial_name, track, cfg=cfg)
    if hist.empty:
        return _no_data_fig(f"no epoch CSV for {trial_name}")
    fig, ax = _new_fig(f"Per-epoch wall-clock — {trial_name}")
    ax.bar(hist["epoch"], hist["epoch_time_sec"], width=0.9, alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds / epoch")
    return fig


def plot_trial_dashboard(trial_name: str, track: str, cfg=None):
    """2×2 dashboard: loss, lr, train/test metric, epoch time."""
    import matplotlib.pyplot as plt
    hist = load_epoch_history(trial_name, track, cfg=cfg)
    if hist.empty:
        return _no_data_fig(f"no epoch CSV for {trial_name}")
    metric_name = str(hist["metric_name"].iloc[0]) if "metric_name" in hist.columns else ""

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Trial dashboard — {trial_name}", fontsize=11)

    axes[0, 0].plot(hist["epoch"], hist["train_loss"],
                    marker="o", markersize=3, linewidth=1.5)
    axes[0, 0].set_title("train loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].grid(True, alpha=0.3, linestyle="--")

    axes[0, 1].plot(hist["epoch"], hist["lr"],
                    marker="o", markersize=3, linewidth=1.5, color="#d62728")
    axes[0, 1].set_title("learning rate")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.3, linestyle="--")

    axes[1, 0].plot(hist["epoch"], hist["train_metric"],
                    marker="o", markersize=3, linewidth=1.5, label="train")
    axes[1, 0].plot(hist["epoch"], hist["test_metric"],
                    marker="s", markersize=3, linewidth=1.5, label="test")
    axes[1, 0].set_title(f"train vs test {metric_name}")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend(loc="best")
    axes[1, 0].grid(True, alpha=0.3, linestyle="--")

    if "epoch_time_sec" in hist.columns:
        axes[1, 1].bar(hist["epoch"], hist["epoch_time_sec"], width=0.9, alpha=0.7)
        axes[1, 1].set_title("seconds / epoch")
        axes[1, 1].set_xlabel("epoch")
        axes[1, 1].grid(True, alpha=0.3, linestyle="--", axis="y")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# =============================================================================
# Cross-trial overlays
# =============================================================================


def _style_for(trial: TrialId, base_palette: dict[str, tuple]) -> dict:
    """Consistent style: color per base, linestyle per lora."""
    color = base_palette.get(trial.base, (0.4, 0.4, 0.4))
    linestyle = "-" if not trial.lora else "--"
    return dict(color=color, linestyle=linestyle, linewidth=1.6,
                alpha=0.95, label=trial.label)


def _palette_for_bases(bases: Sequence[str]) -> dict[str, tuple]:
    import matplotlib.cm as cm
    bases = list(dict.fromkeys(bases))   # de-dup, preserve order
    if not bases:
        return {}
    cmap = cm.get_cmap("tab10", max(len(bases), 3))
    return {b: cmap(i)[:3] for i, b in enumerate(bases)}


def plot_loss_overlay(track: str, *, only_ok: bool = True, cfg=None):
    """Overlay every trial's train-loss curve on one axes.

    Colors group by base checkpoint; LoRA trials use a dashed style.
    """
    histories = load_all_epoch_histories(track, cfg=cfg)
    parsed = {n: parse_trial_name(n) for n in histories}
    parsed = {n: t for n, t in parsed.items() if t is not None}
    if only_ok:
        # Drop trials whose run failed (no epoch rows or status≠OK in manifest).
        manifest = load_run_manifest(track, cfg=cfg)
        if not manifest.empty:
            ok = set(manifest.loc[manifest["status"] == "OK", "trial_name"])
            parsed = {n: t for n, t in parsed.items() if n in ok}
    if not parsed:
        return _no_data_fig(f"no training runs on track={track}")

    fig, ax = _new_fig(f"All trials — train loss (track={track})", figsize=(11, 6))
    palette = _palette_for_bases([t.base for t in parsed.values()])
    for name, trial in sorted(parsed.items(), key=lambda kv: (kv[1].base, kv[1].lr, kv[1].lora)):
        hist = histories[name]
        ax.plot(hist["epoch"], hist["train_loss"], **_style_for(trial, palette))
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.legend(loc="best", fontsize=7, ncol=2)
    return fig


def plot_metric_overlay(
    track: str, *, split: str = "test", only_ok: bool = True, cfg=None,
):
    """Overlay every trial's primary metric curve (train or test)."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    histories = load_all_epoch_histories(track, cfg=cfg)
    parsed = {n: parse_trial_name(n) for n in histories}
    parsed = {n: t for n, t in parsed.items() if t is not None}
    if only_ok:
        manifest = load_run_manifest(track, cfg=cfg)
        if not manifest.empty:
            ok = set(manifest.loc[manifest["status"] == "OK", "trial_name"])
            parsed = {n: t for n, t in parsed.items() if n in ok}
    if not parsed:
        return _no_data_fig(f"no training runs on track={track}")

    metric_name = ""
    for hist in histories.values():
        if "metric_name" in hist.columns and len(hist):
            metric_name = str(hist["metric_name"].iloc[0])
            break

    fig, ax = _new_fig(
        f"All trials — {split} {metric_name} (track={track})",
        figsize=(11, 6),
    )
    palette = _palette_for_bases([t.base for t in parsed.values()])
    for name, trial in sorted(parsed.items(), key=lambda kv: (kv[1].base, kv[1].lr, kv[1].lora)):
        hist = histories[name]
        col = f"{split}_metric"
        if col not in hist.columns:
            continue
        ax.plot(hist["epoch"], hist[col], **_style_for(trial, palette))
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{split} {metric_name}")
    ax.legend(loc="best", fontsize=7, ncol=2)
    return fig


def plot_overfitting_diagnostic(track: str, *, cfg=None):
    """``train_metric - test_metric`` over epochs, one line per trial.

    For higher-is-better metrics (PD: roc_auc) a positive value =
    optimism (train better than test). For lower-is-better (LGD: rmse)
    a negative value = optimism. The sign convention is preserved so
    each track's plot reads naturally.
    """
    histories = load_all_epoch_histories(track, cfg=cfg)
    parsed = {n: parse_trial_name(n) for n in histories}
    parsed = {n: t for n, t in parsed.items() if t is not None}
    if not parsed:
        return _no_data_fig(f"no training runs on track={track}")

    fig, ax = _new_fig(
        f"Overfitting gap (train − test) — track={track}",
        figsize=(11, 6),
    )
    palette = _palette_for_bases([t.base for t in parsed.values()])
    for name, trial in sorted(parsed.items(), key=lambda kv: (kv[1].base, kv[1].lr, kv[1].lora)):
        hist = histories[name]
        if "train_metric" not in hist.columns or "test_metric" not in hist.columns:
            continue
        gap = pd.to_numeric(hist["train_metric"], errors="coerce") - \
              pd.to_numeric(hist["test_metric"],  errors="coerce")
        ax.plot(hist["epoch"], gap, **_style_for(trial, palette))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train − test")
    ax.legend(loc="best", fontsize=7, ncol=2)
    return fig


# =============================================================================
# Final-metric comparisons
# =============================================================================


def plot_final_metric_bar(
    track: str, *, metric: str = "best_test_metric", cfg=None,
):
    """Sorted horizontal bar of one metric per trial.

    ``metric`` is one of the columns of :func:`training_overview`
    (typical: ``"best_test_metric"`` or ``"final_test_metric"``).
    """
    overview = training_overview(track, cfg=cfg)
    if overview.empty or metric not in overview.columns:
        return _no_data_fig(f"no overview / column {metric!r} on track={track}")
    df = overview.dropna(subset=[metric]).copy()
    if df.empty:
        return _no_data_fig(f"all NaN for {metric!r} on track={track}")
    direction = metric_direction(track)
    df = df.sort_values(metric, ascending=(direction == "min"))
    fig, ax = _new_fig(
        f"{metric} per trial — track={track} ({'lower is better' if direction == 'min' else 'higher is better'})",
        figsize=(11, max(4.5, 0.35 * len(df))),
    )
    palette = _palette_for_bases(list(df["base_short"].unique()))
    colors = [palette[b] for b in df["base_short"]]
    ax.barh(df["trial_name"], df[metric], color=colors, alpha=0.85)
    ax.set_xlabel(metric)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=7)
    return fig


def plot_lr_effect(
    track: str, *, metric: str = "best_test_metric", cfg=None,
):
    """Final metric vs learning rate, one line per (base × lora)."""
    import matplotlib.pyplot as plt
    overview = training_overview(track, cfg=cfg)
    if overview.empty or metric not in overview.columns:
        return _no_data_fig(f"no overview on track={track}")
    df = overview.dropna(subset=[metric]).copy()
    if df.empty:
        return _no_data_fig(f"all NaN for {metric!r}")

    fig, ax = _new_fig(
        f"Learning rate sweep — {metric} (track={track})",
        figsize=(9, 5.5),
    )
    palette = _palette_for_bases(list(df["base_short"].unique()))
    for (base, lora), grp in df.groupby(["base_short", "use_lora"], sort=True):
        grp = grp.sort_values("learning_rate")
        # Average across seeds (if any).
        agg = grp.groupby("learning_rate")[metric].mean().reset_index()
        ax.plot(
            agg["learning_rate"], agg[metric],
            marker="o" if not lora else "s",
            linestyle="-" if not lora else "--",
            color=palette.get(base, (0.4, 0.4, 0.4)),
            label=f"{base}{' ·LoRA' if lora else ''}",
            linewidth=1.6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_lora_effect(track: str, *, metric: str = "best_test_metric", cfg=None):
    """For each (base, lr) pair: paired no-LoRA vs LoRA scatter.

    Above the y = x line means LoRA improved the metric (for higher-is-better).
    """
    import matplotlib.pyplot as plt
    overview = training_overview(track, cfg=cfg)
    if overview.empty or metric not in overview.columns:
        return _no_data_fig(f"no overview on track={track}")
    pivot = (
        overview.pivot_table(
            index=["base_short", "learning_rate"],
            columns="use_lora",
            values=metric,
            aggfunc="mean",
        )
        .rename_axis(columns=None)
        .reset_index()
    )
    if pivot.empty or True not in pivot.columns or False not in pivot.columns:
        return _no_data_fig("need at least one with-LoRA and one without-LoRA trial")
    pivot = pivot.dropna(subset=[True, False])
    if pivot.empty:
        return _no_data_fig("no paired (LoRA, no-LoRA) trials")

    fig, ax = _new_fig(
        f"LoRA effect on {metric} — track={track}",
        figsize=(7, 7),
    )
    palette = _palette_for_bases(list(pivot["base_short"].unique()))
    for _, row in pivot.iterrows():
        ax.scatter(row[False], row[True],
                   color=palette.get(row["base_short"], (0.4, 0.4, 0.4)),
                   s=60, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.annotate(f"{row['base_short']}@{row['learning_rate']:.0e}",
                    (row[False], row[True]),
                    fontsize=7, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points")
    lo = min(pivot[False].min(), pivot[True].min())
    hi = max(pivot[False].max(), pivot[True].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel(f"{metric}  (no LoRA)")
    ax.set_ylabel(f"{metric}  (LoRA)")
    return fig


def plot_metric_heatmap(
    track: str, *, metric: str = "best_test_metric", cfg=None,
):
    """``base_short × learning_rate`` heatmap. One panel per LoRA setting."""
    import matplotlib.pyplot as plt
    overview = training_overview(track, cfg=cfg)
    if overview.empty or metric not in overview.columns:
        return _no_data_fig(f"no overview on track={track}")
    overview = overview.dropna(subset=[metric]).copy()
    if overview.empty:
        return _no_data_fig(f"all NaN for {metric!r}")

    direction = metric_direction(track)
    cmap = "viridis" if direction == "max" else "viridis_r"

    fig, axes = plt.subplots(
        1, 2, figsize=(12, max(4, 0.5 * overview["base_short"].nunique())),
        sharey=True,
    )
    fig.suptitle(f"{metric} heatmap — track={track}")
    for ax, lora_flag in zip(axes, (False, True)):
        sub = overview[overview["use_lora"] == lora_flag]
        if sub.empty:
            ax.text(0.5, 0.5, f"(no trials with LoRA={lora_flag})",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888")
            ax.set_title(f"LoRA={lora_flag}")
            ax.set_axis_off()
            continue
        mat = sub.pivot_table(
            index="base_short", columns="learning_rate",
            values=metric, aggfunc="mean",
        ).sort_index()
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([f"{c:.0e}" for c in mat.columns], rotation=30, ha="right")
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(mat.index)
        ax.set_title(f"LoRA={lora_flag}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=7, color="white"
                            if (np.isfinite(v) and v < (np.nanpercentile(mat.values, 60)
                                                        if direction == "max"
                                                        else np.nanpercentile(mat.values, 40)))
                            else "black")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_pareto_time_vs_metric(
    track: str, *, metric: str = "best_test_metric", cfg=None,
):
    """Total training time (x) vs final metric (y), one point per trial."""
    overview = training_overview(track, cfg=cfg)
    if overview.empty or metric not in overview.columns:
        return _no_data_fig(f"no overview on track={track}")
    df = overview.dropna(subset=[metric, "elapsed_sec"]).copy()
    if df.empty:
        return _no_data_fig(f"need {metric} AND elapsed_sec")

    fig, ax = _new_fig(
        f"Time / accuracy trade-off — track={track}", figsize=(9, 5.5),
    )
    palette = _palette_for_bases(list(df["base_short"].unique()))
    for base, grp in df.groupby("base_short"):
        for lora_flag, gg in grp.groupby("use_lora"):
            ax.scatter(
                gg["elapsed_sec"] / 60.0, gg[metric],
                color=palette.get(base, (0.4, 0.4, 0.4)),
                marker="o" if not lora_flag else "s",
                s=70, alpha=0.85, edgecolor="black", linewidth=0.5,
                label=f"{base}{' ·LoRA' if lora_flag else ''}",
            )
    ax.set_xlabel("total training time (minutes)")
    ax.set_ylabel(metric)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_base_ranking(
    track: str, *, metric: str = "best_test_metric", cfg=None,
):
    """Boxplot of one metric per base checkpoint (across LR / LoRA / seeds)."""
    import matplotlib.pyplot as plt
    overview = training_overview(track, cfg=cfg)
    if overview.empty or metric not in overview.columns:
        return _no_data_fig(f"no overview on track={track}")
    df = overview.dropna(subset=[metric]).copy()
    if df.empty:
        return _no_data_fig(f"all NaN for {metric!r}")

    direction = metric_direction(track)
    # Order bases by their median (best first).
    order = (
        df.groupby("base_short")[metric].median()
        .sort_values(ascending=(direction == "min"))
        .index.tolist()
    )
    palette = _palette_for_bases(order)
    fig, ax = _new_fig(
        f"{metric} by base checkpoint — track={track}",
        figsize=(max(7, 1.1 * len(order)), 5.5),
    )
    data = [df.loc[df["base_short"] == b, metric].values for b in order]
    bp = ax.boxplot(
        data, labels=order, showmeans=True, patch_artist=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
        flierprops=dict(marker="x", markersize=4, alpha=0.5),
    )
    for patch, base in zip(bp["boxes"], order):
        patch.set_facecolor(palette[base])
        patch.set_alpha(0.75)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=20)
    return fig


def plot_convergence_speed(track: str, *, cfg=None):
    """Histogram of the best epoch index across trials — are trials
    converging in the first half, or still improving at the cliff?"""
    overview = training_overview(track, cfg=cfg)
    if overview.empty:
        return _no_data_fig(f"no overview on track={track}")
    df = overview.dropna(subset=["best_epoch", "n_epochs"]).copy()
    if df.empty:
        return _no_data_fig("no best_epoch info")
    df["best_epoch_pct"] = df["best_epoch"] / df["n_epochs"].clip(lower=1)
    fig, ax = _new_fig(
        f"When does the best test metric land? — track={track}",
        figsize=(8.5, 4.5),
    )
    ax.hist(df["best_epoch_pct"] * 100, bins=20, alpha=0.8,
            edgecolor="black", linewidth=0.4)
    ax.set_xlabel("best epoch  (% of training budget)")
    ax.set_ylabel("trial count")
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    return fig


def plot_epoch_time_overlay(track: str, *, cfg=None):
    """Median seconds/epoch per trial — useful when one chunk is slow."""
    overview = training_overview(track, cfg=cfg)
    if overview.empty or "mean_epoch_sec" not in overview.columns:
        return _no_data_fig(f"no overview on track={track}")
    df = overview.dropna(subset=["mean_epoch_sec"]).copy()
    if df.empty:
        return _no_data_fig("no epoch timing")
    df = df.sort_values("mean_epoch_sec", ascending=False)
    fig, ax = _new_fig(
        f"Mean seconds / epoch per trial — track={track}",
        figsize=(11, max(4, 0.3 * len(df))),
    )
    palette = _palette_for_bases(list(df["base_short"].unique()))
    colors = [palette[b] for b in df["base_short"]]
    ax.barh(df["trial_name"], df["mean_epoch_sec"], color=colors, alpha=0.85)
    ax.set_xlabel("seconds / epoch (mean)")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=7)
    return fig


# =============================================================================
# Failure / status reporting
# =============================================================================


def failed_trials(track: str, cfg=None) -> pd.DataFrame:
    """One row per FAIL trial in the manifest. Empty if every trial succeeded."""
    manifest = load_run_manifest(track, cfg=cfg)
    if manifest.empty:
        return pd.DataFrame()
    return manifest[manifest["status"] != "OK"][
        ["trial_name", "base_short", "learning_rate", "use_lora",
         "seed", "elapsed_sec", "status", "error"]
    ].reset_index(drop=True)


def trial_leaderboard(track: str, *, cfg=None) -> pd.DataFrame:
    """Sorted leaderboard: best_test_metric per trial, with HPs alongside.

    Sort is direction-aware: PD (roc_auc) descends, LGD (rmse) ascends.
    """
    overview = training_overview(track, cfg=cfg)
    if overview.empty:
        return pd.DataFrame()
    direction = metric_direction(track)
    cols = [
        "trial_name", "base_short", "learning_rate", "use_lora", "seed",
        "metric_name", "best_test_metric", "best_epoch",
        "final_test_metric", "final_train_metric",
        "n_epochs", "mean_epoch_sec", "elapsed_sec", "status",
    ]
    cols = [c for c in cols if c in overview.columns]
    df = overview[cols].copy()
    if "best_test_metric" in df.columns:
        df = df.sort_values("best_test_metric", ascending=(direction == "min"))
    return df.reset_index(drop=True)
