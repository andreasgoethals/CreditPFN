"""Training-pipeline visualisation helpers.

The training pipeline (``scripts/train_pipeline.py``) writes two
kinds of artefacts that this module consumes:

* **Per-trial epoch CSV** — one file per trial under
  ``output/manifests/epochs/<track>/<descriptive_name>.csv`` with columns::

      epoch, train_loss, lr, metric_name,
      train_metric, test_metric, epoch_time_sec, elapsed_sec

  (The writer is ``_on_epoch_end`` inside
  ``scripts/train_pipeline.py``.)

* **Run manifest CSV** — one file per track at
  ``output/manifests/<run_name>_<track>.csv``. Each row is one trial::

      track, base_checkpoint, learning_rate, use_lora, seed,
      n_train_datasets, n_test_datasets,
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
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.visualize import style

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

        from src.utils.paths import checkpoints_dir
        return _NS(run_name="creditpfn", track="pd",
                   checkpoint=_NS(trained_dir=str(checkpoints_dir("trained"))))


#: Run to visualise, overriding ``config/train.yaml``'s ``run_name``. A notebook sets this once
#: (``training_viz.use_run("exp1")``) so every loader below reads that run's artefacts instead of
#: the default. The env var ``CREDITPFN_VIZ_RUN`` does the same for ``run_notebooks`` / scripts.
_RUN_OVERRIDE: str | None = None


def use_run(name: str | None) -> None:
    """Point every training loader at run ``name`` (e.g. ``"exp1"``); ``None`` restores the default.

    Experiment 1 is submitted per split, so its manifests are ``<run>_s00_<track>.csv`` …
    ``<run>_s07_<track>.csv`` rather than a single ``<run>_<track>.csv`` — :func:`load_run_manifest`
    handles both, and tags each row with a ``split`` column when the per-split layout is found.
    """
    global _RUN_OVERRIDE
    _RUN_OVERRIDE = str(name) if name else None


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

    from src.utils.paths import checkpoints_dir, manifests_dir, resolve_staging_path
    if cfg is None:
        cfg = _load_default_cfg()

    run_name = (_RUN_OVERRIDE or os.environ.get("CREDITPFN_VIZ_RUN")
                or str(getattr(cfg, "run_name", "creditpfn")))
    trained_dir = str(getattr(cfg, "checkpoint", _load_default_cfg().checkpoint)
                      .trained_dir if hasattr(cfg, "checkpoint")
                      else str(checkpoints_dir("trained")))

    return {
        "epoch_dir":     manifests_dir() / "epochs",
        "manifest_dir":  manifests_dir(),
        "trained_dir":   resolve_staging_path(trained_dir),
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
    #: Corpus-size arm (`corpus.min_train_rows`), 0 when the trial predates run-8.
    min_train_rows: int = 0
    #: L2-SP anchor strength. `None` when the trial predates run-9 and did not sweep it.
    l2sp_lambda: float | None = None

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


# Matches descriptive_name (src/train/loop.py):
#   <run>_<track>_<base>_lr<lr>_seed<seed>[_qf<qf>][_acc<K>][_fullpass][_lora]
# The qf / acc / fullpass segments are optional so this matches both the
# legacy name and the swept-axis names. (Before 2026-06-01 this regex
# stopped at _seed<seed>[_lora] and so returned None for every name that
# carried _qf/_acc — i.e. the entire current sweep — silently emptying the
# cross-trial training plots.)
_NAME_RE = re.compile(
    r"^(?P<run>.+?)_(?P<track>pd|lgd)_"
    r"(?P<base>.+?)"
    r"_lr(?P<lr>[0-9eE.+\-]+)"
    r"_seed(?P<seed>\d+)"
    r"(?:_qf(?P<qf>\d+))?"
    r"(?:_acc(?P<acc>\d+))?"
    r"(?P<fullpass>_fullpass)?"
    # Corpus-size arm, swept since run-8. Sits between the pass mode and the adapter
    # tag, exactly as `loop.descriptive_name` writes it.
    r"(?:_min(?P<min_rows>\d+))?"
    # Anchor strength, swept from run-9. Optional, so pre-run-9 names still match.
    r"(?:_l2sp(?P<l2sp>[0-9.eE+-]+))?"
    # `_lora` for TabPFN, `_iclhead` for TabICLv2 — ONE grid axis, two family
    # renderings. Omitting `_iclhead` here made every frozen-backbone TabICLv2 trial
    # unparseable, which is the same silent-drop failure the comment above describes.
    r"(?P<lora>_lora|_iclhead)?$"
)


def parse_trial_name(name: str) -> TrialId | None:
    """Parse a descriptive_name (with or without extension)."""
    # NOT Path(name).stem: it strips everything after the LAST dot, and the base stem
    # contains one — `tabpfn-v2.6-classifier-v2.6_default` becomes
    # `tabpfn-v2.6-classifier-v2`, taking the learning rate and seed with it. Every v2.6
    # trial then failed to parse and was labelled "?" in every training figure, which is
    # half the grid silently mislabelled and merged into one colour. Strip only the
    # extensions this project actually writes.
    stem = str(name)
    for ext in (".csv", ".ckpt", ".json"):
        stem = stem.removesuffix(ext)
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
            min_train_rows=int(m.group("min_rows") or 0),
            l2sp_lambda=(float(m.group("l2sp"))
                         if m.group("l2sp") is not None else None),
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
    import re
    paths = _resolve_paths(cfg)
    run, mdir = paths["run_name"], paths["manifest_dir"]

    # Two on-disk layouts. A single-run sweep writes one manifest ``<run>_<track>.csv``. Experiment 1
    # is submitted per dataset split (train_pipeline appends ``_s<NN>`` to run_name), so it writes
    # ``<run>_s00_<track>.csv`` … ``<run>_s07_<track>.csv``. Read whichever exists; for the per-split
    # layout, concatenate and tag each row with its ``split`` so downstream figures can aggregate.
    single = mdir / f"{run}_{track}.csv"
    if single.exists():
        parts: list[tuple[Path, int | None]] = [(single, None)]
    else:
        parts = []
        for pth in sorted(mdir.glob(f"{run}_s*_{track}.csv")):
            m = re.search(rf"_s(\d+)_{re.escape(track)}\.csv$", pth.name)
            parts.append((pth, int(m.group(1)) if m else None))

    frames = []
    for pth, split in parts:
        try:
            d = pd.read_csv(pth)
        except Exception:                                # pragma: no cover
            continue
        if d.empty:
            continue
        frames.append(d.assign(split=split) if split is not None else d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Drop stale pre-provenance rows. Re-submissions APPEND to the same per-split manifest, so it
    # accumulates rows from every code version the run has seen. Rows written before the pipeline
    # recorded git provenance carry an empty ``git_commit`` — in Experiment 1 these are the frozen-
    # TabPFN trials that died with the since-fixed ``ckpt_path`` NameError (fingerprint: empty commit
    # + ``l2sp_lambda`` NaN). They are pure noise — no checkpoint, no result — and would otherwise
    # inflate the failure rate by ~50 %. Current code always stamps the commit, so an empty one is
    # unambiguously old. (No ``git_commit`` column ⇒ a legacy single-run manifest; leave it as-is.)
    if "git_commit" in df.columns:
        commit = df["git_commit"].astype("string").str.strip()
        stale = commit.isna() | (commit == "")
        if bool(stale.any()):
            LOGGER.debug("load_run_manifest(%s): dropping %d stale pre-provenance row(s)",
                         track, int(stale.sum()))
            df = df.loc[~stale].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame()

    # Derive trial_name + base_short from ckpt path (when available) or
    # rebuild from columns.
    def _stem(row) -> str:
        if isinstance(row["final_ckpt_path"], str) and row["final_ckpt_path"]:
            return Path(row["final_ckpt_path"]).stem
        # FAIL rows have no ckpt — reconstruct (per-split run name when the split is known).
        run_i = (f"{run}_s{int(row['split']):02d}"
                 if "split" in row.index and pd.notna(row.get("split")) else run)
        base_stem = Path(str(row["base_checkpoint"])).stem
        lr_tag = f"{float(row['learning_rate']):.0e}".replace("+", "")
        lora_tag = "_lora" if bool(row.get("use_lora", False)) else ""
        return (
            f"{run_i}_{row['track']}_{base_stem}_"
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
    """Load every per-epoch CSV under ``output/manifests/epochs/<track>/``.

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


def _new_fig(title: str, *, figsize=style.figsize(style.WIDTH_FULL, ratio=0.562)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    # `style.title` keeps it on one or two lines; a title wider than the figure overlaps
    # the y tick labels. No grid call here — `style.apply()` owns the grid, and setting
    # it again is how two figures in one project end up looking different.
    style.title(ax, title)
    return fig, ax


def _no_data_fig(reason: str = "no data"):
    """Render a stub figure with a centred message — for notebooks that
    run before any training output is on disk."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, ratio=0.333))
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

    fig, axes = plt.subplots(2, 2, figsize=style.figsize(style.WIDTH_FULL, ratio=0.615))
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
    # No tight_layout: style.py turns constrained_layout ON, and calling both makes
    # matplotlib warn and discard one of them. constrained_layout fits the content
    # INSIDE the declared A4 width, which is the whole point of drawing at final size.
    return fig


# =============================================================================
# Cross-trial overlays
# =============================================================================


def _style_for(trial: TrialId, base_palette: dict[str, tuple],
               seen: set | None = None) -> dict:
    """Consistent style: colour per base, dashed for the adapter arm.

    `seen` collapses the legend to ONE ENTRY PER (base, adapter) combination instead of
    one per trial. With a 16-trial grid the per-trial legend was 16 long labels that
    overlapped each other whether it sat inside the axes (covering the curves) or outside
    (stacked past the figure) — 38 of the collisions in the 12-08-2026 audit. The colour
    and the dash already say everything the legend was repeating; the learning rate is
    what the reader reads off the curve, not off a label.
    """
    color = base_palette.get(trial.base, style.COLORS["annotation"])
    linestyle = "-" if not trial.lora else "--"
    key = (trial.base_short, trial.lora)
    label = None
    if seen is not None:
        if key not in seen:
            seen.add(key)
            label = f"{trial.base_short}{' · adapter' if trial.lora else ''}"
    else:
        label = trial.label
    return dict(color=color, linestyle=linestyle, linewidth=1.3,
                alpha=0.9, label=label)


def _palette_for_bases(bases: Sequence[str]) -> dict[str, str]:
    """One colour per base checkpoint, from `src/visualize/style.py`.

    Keyed on the base NAME, never on its position: the previous
    `cm.get_cmap("tab10", n)[i]` meant a figure that omitted one arm shifted every
    colour after it, so v2.6 was orange in one panel and green in the next.
    """
    from src.visualize.style import color
    return {b: color(_base_series_name(b)) for b in dict.fromkeys(bases)}


def _base_series_name(base: str) -> str:
    """Map a base checkpoint or short tag onto a registered `style.COLORS` name."""
    b = base.lower()
    if "tabicl" in b:
        return "tabicl"
    if "v2.6" in b or "v2_6" in b:
        return "v2.6"
    if "v3" in b:
        return "v3"
    return base


def _progress(hist: pd.DataFrame) -> tuple[pd.Series, str]:
    """The x axis for any CROSS-TRIAL curve: optimizer steps, falling back to epochs.

    `RESULTS.md` states the rule this implements — "never compare bases on epochs" — and the
    reason: steps per epoch is `sum(ceil(rows_i / cap))` over the training corpus, and the
    row cap differs per base (v3 26 000, v2.6 11 000), so epoch 50 is 9 135 steps for v2.6
    and 20 020 for v3. Overlaying curves against epoch silently stretches one base against
    another; every overlay in this module used to do exactly that.

    Per-trial figures keep epochs, where there is nothing to compare against.
    """
    # OPTIMIZER STEPS again (24-08, second change today). Epoch was briefly the axis because the
    # budget was set in epochs. It no longer is: experiment 1 budgets `target_total_steps: 6000`,
    # so the STEP is the controlled variable — every cell of the grid lands within 1 % of 6 000 —
    # while the derived epoch count ranges from 30 (PD v2.6 full_pass) to 1 000 (LGD accumulate).
    # Plotting against epoch would now stretch a 30-epoch curve across the same width as a
    # 1 000-epoch one and invite exactly the cross-base comparison this function exists to stop.
    if "optimizer_steps" in hist.columns:
        steps = pd.to_numeric(hist["optimizer_steps"], errors="coerce")
        # Per-epoch counts in older manifests, cumulative in newer ones. A cumulative series is
        # STRICTLY increasing (every epoch adds at least one update); a per-epoch series is
        # roughly constant. `is_monotonic_increasing` is the wrong test here — pandas counts a
        # constant series as monotonic, so [91, 91, 91] would be read as already cumulative.
        if steps.notna().any():
            d = steps.dropna().diff().dropna()
            if not (len(d) == 0 or (d > 0).all()):
                steps = steps.fillna(0).cumsum()
            if steps.iloc[-1] > 0:
                return steps, "optimizer steps"
    return hist["epoch"], "epoch"


def _finite(x: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Drop rows where y is missing, so a sparse series draws as a CONNECTED line.

    The monitor metric and the weight-drift columns are written every `monitor_every` epochs
    (5 in run-8), leaving NaN in between. `ax.plot` breaks a line at every NaN, so a curve of
    46 finite values inside 221 rows rendered as nothing at all except where two finite points
    happened to be adjacent — which is why the cross-trial overlays showed two short marks near
    the origin instead of fifteen curves, and why they looked "unclear and uninformative".
    """
    m = pd.to_numeric(y, errors="coerce").notna()
    return x[m], pd.to_numeric(y, errors="coerce")[m]


def compact_base(base_short: str) -> str:
    """A base label short enough for a tick: `tabicl-classifier-v2-20260212` → `tabicl-v2`.

    Axis labels are budgeted in inches, not characters. A 29-character tick label needs
    ~1.6 in of the 6.3 in width, which is what made `plot_metric_heatmap` overflow the
    default left margin and lose the start of every label. The task word is redundant on a
    per-track figure (a PD figure has only classifiers) and the checkpoint date belongs in
    `METHOD.md`, not on an axis. Matches the naming `eval_viz` uses, so the same checkpoint
    reads the same in both notebooks.
    """
    s = str(base_short)
    s = re.sub(r"-(classifier|regressor)", "", s)
    s = re.sub(r"-(\d{8})$", "", s)
    s = s.removesuffix("_default").removesuffix("-default")
    s = re.sub(r"^(v\d+(?:\.\d+)?)-\1$", r"\1", s)      # v3-v3 → v3
    s = re.sub(r"-(v\d+(?:\.\d+)?)-\1$", r"-\1", s)     # tabicl-v2-v2 → tabicl-v2
    return s


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

    fig, ax = _new_fig(f"All trials — train loss", figsize=style.figsize(style.WIDTH_FULL, ratio=0.545))
    palette = _palette_for_bases([t.base for t in parsed.values()])
    seen: set = set()
    xlabel = "epoch"
    xlabel = "epoch"
    for name, trial in sorted(parsed.items(), key=lambda kv: (kv[1].base, kv[1].lr, kv[1].lora)):
        hist = histories[name]
        x, xlabel = _progress(hist)
        xf, yf = _finite(x, hist["train_loss"])
        if xf.empty:
            continue
        ax.plot(xf, yf, **_style_for(trial, palette, seen))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("train loss")
    # TabICLv2 starts near 1.5 and TabPFN near 0.48, so a linear axis compresses every
    # TabPFN curve into a flat band at the bottom where the differences between learning
    # rates and corpus arms live. Log y separates them without hiding the descent.
    lo = min(float(histories[n]["train_loss"].min()) for n in parsed)
    hi = max(float(histories[n]["train_loss"].max()) for n in parsed)
    if lo > 0 and hi / lo >= 2.0:
        ax.set_yscale("log")
        ax.set_ylabel("train loss  (log)")
    ax.legend(loc="best", fontsize=7)
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
        f"All trials — {split} {metric_name}",
        figsize=style.figsize(style.WIDTH_FULL, ratio=0.545),
    )
    palette = _palette_for_bases([t.base for t in parsed.values()])
    seen: set = set()
    xlabel = "epoch"
    for name, trial in sorted(parsed.items(), key=lambda kv: (kv[1].base, kv[1].lr, kv[1].lora)):
        hist = histories[name]
        col = f"{split}_metric"
        if col not in hist.columns:
            continue
        x, xlabel = _progress(hist)
        xf, yf = _finite(x, hist[col])
        if xf.empty:
            continue
        ax.plot(xf, yf, **_style_for(trial, palette, seen))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{split} {metric_name}")
    ax.legend(loc="best", fontsize=7)
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
        f"Overfitting gap (train − test)",
        figsize=style.figsize(style.WIDTH_FULL, ratio=0.545),
    )
    palette = _palette_for_bases([t.base for t in parsed.values()])
    seen: set = set()
    xlabel = "epoch"
    for name, trial in sorted(parsed.items(), key=lambda kv: (kv[1].base, kv[1].lr, kv[1].lora)):
        hist = histories[name]
        if "train_metric" not in hist.columns or "test_metric" not in hist.columns:
            continue
        gap = pd.to_numeric(hist["train_metric"], errors="coerce") - \
              pd.to_numeric(hist["test_metric"],  errors="coerce")
        x, xlabel = _progress(hist)
        xf, yf = _finite(x, gap)
        if xf.empty:
            continue
        ax.plot(xf, yf, **_style_for(trial, palette, seen))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("train − test")
    ax.legend(loc="best", fontsize=7)
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
        figsize=style.figsize(style.WIDTH_FULL, ratio=(max(4.5, 0.35 * len(df))) / (11)),
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
        f"Learning rate sweep — {metric}",
        figsize=style.figsize(style.WIDTH_FULL, ratio=0.611),
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
        f"LoRA effect on {metric}",
        figsize=style.figsize(style.WIDTH_FULL, ratio=1.000),
    )
    palette = _palette_for_bases(list(pivot["base_short"].unique()))
    # ONE LEGEND ENTRY PER BASE, not a text label per point. The per-point annotations
    # this replaces produced 19 of the 70 overlaps in the 12-08-2026 figure audit, and a
    # paper figure should not carry a label per marker anyway — the learning rate is
    # readable from the marker's position along the diagonal.
    seen = set()
    for _, row in pivot.iterrows():
        base = row["base_short"]
        ax.scatter(row[False], row[True],
                   color=palette.get(base, style.COLORS["annotation"]),
                   s=34, alpha=0.85, edgecolors="none",
                   label=(base if base not in seen else None))
        seen.add(base)
    lo = min(pivot[False].min(), pivot[True].min())
    hi = max(pivot[False].max(), pivot[True].max())
    ax.plot([lo, hi], [lo, hi], color=style.COLORS["reference"],
            linestyle="--", alpha=0.5, linewidth=0.8)
    ax.legend(loc="best", fontsize=7)
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

    # Height from CONTENT, not from a ratio expression. The old
    # `max(4, 0.5 * n_bases) / 12` gave 4/12 for the three bases of run-8, i.e. a 2.1 in
    # figure — and two side-by-side panels with rotated tick labels, a suptitle and a
    # colourbar do not fit in 2.1 in, so constrained_layout gave up with "axes sizes
    # collapsed to zero" and matplotlib drew an unreadable figure.
    n_bases = int(overview["base_short"].nunique())
    height = max(2.9, 0.34 * n_bases + 2.3)
    # NOT sharey. The two panels index different sets of bases — run-8 ran the adapter arm on
    # TabICLv2 only — so the right panel has 1 row where the left has 3. On a SHARED y axis
    # the right panel's `set_yticks(range(1))` overwrote the left panel's ticks and its
    # y limits, cropping two of the three bases out of view entirely and leaving one tick
    # label for three rows. The figure was wrong, not merely ugly.
    fig, axes = plt.subplots(
        1, 2, figsize=style.figsize(style.WIDTH_FULL, ratio=height / style.WIDTH_FULL),
    )
    fig.suptitle(f"{metric} heatmap")
    vmin, vmax = float(overview[metric].min()), float(overview[metric].max())
    images: list = []
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
        # ONE colour scale for BOTH panels. Each `imshow` normalises to its own data by
        # default, so the two panels of this side-by-side comparison were on different
        # scales — the same colour meant a different score left and right, which is the one
        # thing a paired comparison figure must not do.
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        images.append(im)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([f"{c:.0e}" for c in mat.columns], rotation=30, ha="right")
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels([compact_base(i) for i in mat.index], fontsize=8)
        ax.set_title(f"LoRA={lora_flag}")
        cut = (np.nanpercentile(mat.values, 60) if direction == "max"
               else np.nanpercentile(mat.values, 40))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if v < cut else "black")
    # ONE colourbar for the figure. Two of them, at `fraction=0.03` each, plus the 28-character
    # base names on the y axis, left the two panels of a 6.3 in figure with no width at all —
    # constrained_layout reported "axes sizes collapsed to zero" and drew nothing legible.
    if images:
        fig.colorbar(images[0], ax=list(axes), fraction=0.035, pad=0.02, label=metric)
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
        f"Time / accuracy trade-off", figsize=style.figsize(style.WIDTH_FULL, ratio=0.611),
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
        f"{metric} by base checkpoint",
        figsize=style.figsize(style.WIDTH_FULL, ratio=(5.5) / (max(7, 1.1 * len(order)))),
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
        f"When does the best test metric land?",
        figsize=style.figsize(style.WIDTH_FULL, ratio=0.529),
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
        f"Mean seconds / epoch per trial",
        figsize=style.figsize(style.WIDTH_FULL, ratio=(max(4, 0.3 * len(df))) / (11)),
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


# --------------------------------------------------------------------------- #
# The two columns the 08-08-2026 logging added, and nothing plotted until now
# --------------------------------------------------------------------------- #


def plot_weight_drift(track: str, *, only_ok: bool = True, cfg=None):
    """How far each trial's weights moved from the base checkpoint, per model stage.

    ``drift__<stage>`` is ``||w - w0||`` over the parameters of one top-level module,
    recorded on every monitored epoch. It answers the question that reframed run-5:
    *did the model actually train?* A trial whose drift is flat near zero has not moved,
    and its "no effect on the metric" result says nothing about continued pretraining.
    """
    histories = load_all_epoch_histories(track, cfg=cfg)
    parsed = {n: t for n, t in ((n, parse_trial_name(n)) for n in histories) if t is not None}
    if only_ok:
        manifest = load_run_manifest(track, cfg=cfg)
        if not manifest.empty:
            ok = set(manifest.loc[manifest["status"] == "OK", "trial_name"])
            parsed = {n: t for n, t in parsed.items() if n in ok}
    drift_cols = sorted({c for n in parsed for c in histories[n].columns
                         if c.startswith("drift__")})
    if not parsed or not drift_cols:
        return _no_data_fig(
            f"no drift__* columns on track={track} — the per-stage drift logging "
            "was added on 08-08-2026, so runs before it have none"
        )

    fig, ax = _new_fig(f"Weight drift from the base checkpoint")
    palette = _palette_for_bases([t.base for t in parsed.values()])
    seen: set = set()
    xlabel = "epoch"
    for name, trial in sorted(parsed.items(), key=lambda kv: (kv[1].base, kv[1].lr)):
        hist = histories[name]
        cols = [c for c in drift_cols if c in hist.columns]
        if not cols:
            continue
        # Max over stages: the loosest part of the model is what "did it move?" hangs on.
        series = hist[cols].max(axis=1)
        mask = series.notna()
        x, xlabel = _progress(hist)
        xf, yf = _finite(x[mask], series[mask])
        if xf.empty:
            continue
        ax.plot(xf, yf, **_style_for(trial, palette, seen))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\|w - w_0\|$  (max over stages)")
    # Drift spans orders of magnitude across learning rates, so log is the readable
    # scale — but a run where nothing moved is all zeros, and log would raise.
    if any(ln.get_ydata().size and (ln.get_ydata() > 0).all() for ln in ax.lines):
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=7)
    return fig


def plot_per_dataset_loss(trial_name: str, track: str, *, cfg=None):
    """Training loss per dataset over epochs, for one trial.

    ``loss__<dataset_id>`` is that dataset's mean step loss within the epoch. The
    aggregate `train_loss` hides which tables the model is actually learning: a corpus
    loss that falls while one dataset's loss rises is the model trading one table for
    the others, which is what a mixed-domain corpus does when one table dominates.
    """
    histories = load_all_epoch_histories(track, cfg=cfg)
    if trial_name not in histories:
        return _no_data_fig(f"no epoch history for {trial_name}")
    hist = histories[trial_name]
    cols = sorted(c for c in hist.columns if c.startswith("loss__"))
    if not cols:
        return _no_data_fig(
            f"no loss__<dataset> columns for {trial_name} — the per-dataset loss "
            "logging was added on 08-08-2026"
        )

    fig, ax = _new_fig(f"Per-dataset train loss — {compact_base(_base_of(trial_name))}",
                       figsize=style.figsize(style.WIDTH_FULL, ratio=0.60))
    # Order the legend by final loss so the worst-fitting table is easy to find.
    order = sorted(cols, key=lambda c: -hist[c].dropna().iloc[-1] if hist[c].notna().any() else 0)
    ids = [c.removeprefix("loss__") for c in order]
    # DISTINCT colours by position. `style.color` keys on the name and has four fallback
    # slots, so three of the six LGD datasets came out the same yellow.
    palette = style.categorical(ids)
    # The raw series is one point per epoch of a mean step loss, and over 1 200 epochs it is
    # a solid band of noise. Draw the noise faintly and a rolling median on top, so the
    # per-table trend — the thing this figure exists to show — is actually visible.
    win = max(1, len(hist) // 60)
    for col, name in zip(order, ids):
        y = pd.to_numeric(hist[col], errors="coerce")
        ax.plot(hist["epoch"], y, linewidth=0.5, alpha=0.25, color=palette[name])
        if win > 1:
            ax.plot(hist["epoch"], y.rolling(win, min_periods=1, center=True).median(),
                    linewidth=1.4, color=palette[name], label=name)
        else:
            ax.plot(hist["epoch"], y, linewidth=1.4, color=palette[name], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean step loss")
    # OUTSIDE the axes. `loc="best"` put the six dataset ids straight over the curves in the
    # top-left, which is exactly where the interesting early descent happens.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=6,
              frameon=False, handlelength=1.6)
    if win > 1:
        style.note(ax, f"faint = per epoch · bold = rolling median over {win} epochs")
    return fig


def _base_of(trial_name: str) -> str:
    """The base tag of a trial, for a title that fits. Falls back to the full name."""
    t = parse_trial_name(trial_name)
    return t.base_short if t else trial_name
