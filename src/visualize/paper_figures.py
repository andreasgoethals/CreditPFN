"""The figures a paper about this project actually needs.

`training_viz` and `eval_viz` answer "did the run behave?" — a hundred diagnostic views,
most of which belong in an appendix or nowhere. This module holds the small set that
carries the argument, chosen against what the field reports (`tfm-library/SYNTHESIS.md`):

1. `plot_paired_delta`        the headline. Trained minus its OWN untuned base, per
                              dataset. Every continued-pretraining paper reports this and
                              nothing else answers "did it help?".
2. `plot_gain_vs_base`        our own finding: the weaker the base, the larger the gain.
                              The synthesis predicts exactly this ("continued pretraining
                              should help most where the domain is distinctive relative to
                              the prior, while a sufficiently good synthetic backbone may
                              erase the headroom").
3. `plot_mean_rank`           mean rank across datasets, the field's standard aggregate
                              (Hollmann 2025, Garg 2025, Purucker 2026). Immune to one
                              dataset's scale dominating a mean of raw metrics.
4. `plot_reliability`         calibration, the differentiator. TabPFN's selling point is
                              calibrated probabilities; Tanna 2026 shows naive finetuning
                              triples ECE elsewhere, and neither TabICLv2, TabDPT nor Mitra
                              reports ECE at all.
5. `plot_regime_effect`       where the method wins as a function of dataset property —
                              Purucker's analysis (margin vs n rows, ρ=+0.60). The figure
                              that tells a practitioner when to use this.
6. `plot_selection_honesty`   leave-one-dataset-out hyperparameter selection vs the
                              best-on-test number. The winner's-curse correction, computable
                              from results we already have.
7. `plot_forgetting`          rank correlation between trained and base predictions —
                              Kolberg's ρ=0.9935 check, "worth copying".

Every function here degrades gracefully as the corpus grows: nothing draws one bar or one
label per dataset without asking `style.too_many` first.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.visualize import style

#: The column `eval_viz.load_eval_results` uses for the method's results directory. Named
#: here once rather than at eight call sites: it has been `method_dir` and `method_dirname`
#: at different times, and a wrong guess raises a KeyError deep inside a groupby.
_METHOD_COL = "method_dirname"

LOGGER = logging.getLogger(__name__)

#: Higher-is-better metrics. Everything else is treated as lower-is-better, which is what
#: decides the sign of a "gain" and the direction of a rank.
_HIGHER_IS_BETTER = {"roc_auc", "pr_auc", "f1", "accuracy", "r2", "balanced_accuracy",
                     "mcc", "precision", "recall", "specificity", "cohen_kappa",
                     "explained_variance", "pearson_r", "spearman_r", "neg_nll"}


def _ok(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rows that actually produced a number.

    A failed fold is written with `status="FAIL"` and NaN metrics — by design, so a
    partial run is visible rather than silently missing. Averaging over folds WITHOUT
    dropping them turns one failed fold into a NaN for the whole (model, dataset) cell,
    which then disappears from every figure that groups by it. `eval_viz` has always had
    an `_ok_only` filter; this module needs the same one.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df
    if "status" in out.columns:
        out = out[out["status"] == "OK"]
    if metric in out.columns:
        out = out[out[metric].notna()]
    return out.copy()


def higher_is_better(metric: str) -> bool:
    return metric in _HIGHER_IS_BETTER


def _sign(metric: str) -> int:
    return 1 if higher_is_better(metric) else -1


def _new(title: str, *, width: float = style.WIDTH_FULL, ratio: float = 0.55):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=style.figsize(width, ratio=ratio))
    ax.set_title(title)
    return fig, ax


def _empty(reason: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, ratio=0.35))
    ax.text(0.5, 0.5, reason, ha="center", va="center", fontsize=9,
            color=style.COLORS["annotation"], transform=ax.transAxes)
    ax.set_axis_off()
    return fig


# --------------------------------------------------------------------------- #
# Shared: pair every trained model with its own untuned base
# --------------------------------------------------------------------------- #


def paired_deltas(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """One row per (trained model × dataset) with the delta against its OWN base.

    THE comparison this project exists to make, and the only honest one: an unpaired mean
    over models scored on different dataset subsets is dominated by which datasets each
    model happened to get, not by the models. A partially-complete eval — every run so far
    — makes the unpaired version actively misleading.

    Expects the frame `eval_viz.load_eval_results` returns: one row per
    (method × dataset × fold), with `source`, `base_short` and the metric column.
    """
    need = {"source", "base_short", "test_dataset_id", metric}
    if df is None or df.empty or not need <= set(df.columns):
        return pd.DataFrame()
    df = _ok(df, metric)
    if df.empty:
        return pd.DataFrame()

    cell = (df.groupby([_METHOD_COL, "source", "base_short", "test_dataset_id"],
                       dropna=False)[metric]
              .mean().reset_index())
    untuned = cell[cell["source"].str.endswith("-untuned", na=False)]
    trained = cell[cell["source"].str.endswith("-trained", na=False)]
    if untuned.empty or trained.empty:
        return pd.DataFrame()

    base_col = untuned.set_index(["base_short", "test_dataset_id"])[metric]
    rows = []
    for r in trained.itertuples():
        key = (r.base_short, r.test_dataset_id)
        if key not in base_col.index:
            continue
        ref = float(base_col.loc[key])
        rows.append({
            _METHOD_COL: getattr(r, _METHOD_COL), "base_short": r.base_short,
            "test_dataset_id": r.test_dataset_id,
            "untuned": ref, "trained": float(getattr(r, metric)),
            "delta": (float(getattr(r, metric)) - ref) * _sign(metric),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. The headline
# --------------------------------------------------------------------------- #


def plot_paired_delta(df: pd.DataFrame, metric: str = "roc_auc"):
    """Trained minus its own untuned base, one point per (model × dataset).

    Points are grouped by base checkpoint on the x axis and jittered, so the reader sees
    the whole distribution rather than a mean that hides sign changes. The zero line is
    the only reference that matters: above it continued pretraining helped.
    """
    d = paired_deltas(df, metric)
    if d.empty:
        return _empty("no paired trained/untuned cells — the eval needs both arms")

    fig, ax = _new(f"Effect of continued pretraining ({metric}, paired)")
    bases = sorted(d["base_short"].unique())
    rng = np.random.default_rng(0)
    for i, base in enumerate(bases):
        sub = d[d["base_short"] == base]
        x = i + rng.uniform(-0.16, 0.16, len(sub))
        ax.scatter(x, sub["delta"], s=16, alpha=0.75,
                   color=style.color(_base_key(base)), edgecolors="none")
        ax.plot([i - 0.28, i + 0.28], [sub["delta"].mean()] * 2,
                color=style.COLORS["reference"], linewidth=1.4, zorder=3)
    ax.axhline(0, color=style.COLORS["reference"], linewidth=0.8, alpha=0.6)
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels(bases)
    ax.set_ylabel(f"Δ {metric}  (trained − untuned)")
    ax.set_xlabel("")
    style.note(ax, f"n={len(d)} pairs · bar = mean")
    return fig


def _base_key(base_short: str) -> str:
    b = str(base_short).lower()
    if "tabicl" in b:
        return "tabicl"
    if "v2.6" in b:
        return "v2.6"
    if "v3" in b:
        return "v3"
    return base_short


# --------------------------------------------------------------------------- #
# 2. Gain against starting quality
# --------------------------------------------------------------------------- #


def plot_gain_vs_base(df: pd.DataFrame, metric: str = "roc_auc"):
    """Δ from continued pretraining against how good the base already was.

    One point per (trained model × dataset); x is the untuned base's score on that
    dataset. A downward trend is the paper's mechanism: adaptation buys most where the
    pretrained prior fits the domain worst.
    """
    d = paired_deltas(df, metric)
    if d.empty:
        return _empty("no paired trained/untuned cells")

    fig, ax = _new(f"Gain against starting quality ({metric})")
    for base in sorted(d["base_short"].unique()):
        sub = d[d["base_short"] == base]
        ax.scatter(sub["untuned"], sub["delta"], s=18, alpha=0.8,
                   color=style.color(_base_key(base)), label=base, edgecolors="none")
    ax.axhline(0, color=style.COLORS["reference"], linewidth=0.8, alpha=0.6)
    if len(d) >= 3 and d["untuned"].nunique() >= 2:
        b, a = np.polyfit(d["untuned"], d["delta"], 1)
        xs = np.linspace(d["untuned"].min(), d["untuned"].max(), 50)
        ax.plot(xs, a + b * xs, color=style.COLORS["highlight"], linewidth=1.2, zorder=1)
        r = float(np.corrcoef(d["untuned"], d["delta"])[0, 1])
        style.note(ax, f"slope {b:+.3f} · r = {r:+.2f} · n = {len(d)}")
    ax.set_xlabel(f"untuned base {metric} on that dataset")
    ax.set_ylabel(f"Δ {metric}")
    ax.legend(loc="best", fontsize=7)
    return fig


# --------------------------------------------------------------------------- #
# 3. Mean rank
# --------------------------------------------------------------------------- #


def mean_ranks(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Mean rank per method across the datasets where ALL methods were scored.

    Ranking per dataset then averaging is the field's standard aggregate because raw
    metric means are dominated by whichever dataset has the widest spread. Restricting to
    the complete block is what keeps it a fair comparison when an eval is partial.
    """
    need = {_METHOD_COL, "test_dataset_id", metric}
    if df is None or df.empty or not need <= set(df.columns):
        return pd.DataFrame()
    df = _ok(df, metric)
    if df.empty:
        return pd.DataFrame()
    cell = df.groupby([_METHOD_COL, "test_dataset_id"])[metric].mean().reset_index()
    wide = cell.pivot(index="test_dataset_id", columns=_METHOD_COL, values=metric).dropna(axis=0)
    if wide.empty or wide.shape[1] < 2:
        return pd.DataFrame()
    ranks = wide.rank(axis=1, ascending=not higher_is_better(metric))
    out = pd.DataFrame({
        _METHOD_COL: ranks.columns,
        "mean_rank": ranks.mean(axis=0).to_numpy(),
        "sd": ranks.std(axis=0, ddof=1).fillna(0.0).to_numpy(),
        "n_datasets": len(wide),
    })
    return out.sort_values("mean_rank").reset_index(drop=True)


def plot_mean_rank(df: pd.DataFrame, metric: str = "roc_auc", *, label_map=None):
    """Mean rank per method, best at the top, with one standard deviation.

    Scales: above `style.MAX_BARS` methods only the best and worst are drawn, because a
    rank chart with 200 rows communicates nothing a table would not.
    """
    r = mean_ranks(df, metric)
    if r.empty:
        return _empty("no dataset is scored by every method — cannot rank")

    labels = [label_map(m) if label_map else m for m in r[_METHOD_COL]]
    r = r.assign(label=labels)
    shown, hidden = style.head_tail(list(r.itertuples()), style.MAX_BARS)
    h = max(2.0, 0.17 * len(shown) + 0.9)
    fig, ax = _new(f"Mean rank across {int(r['n_datasets'].iloc[0])} datasets ({metric})",
                   ratio=h / style.WIDTH_FULL)
    y = np.arange(len(shown))[::-1]
    ax.barh(y, [s.mean_rank for s in shown], xerr=[s.sd for s in shown],
            color=[style.color(_base_key(s.label)) for s in shown],
            height=0.72, error_kw={"linewidth": 0.7, "ecolor": style.COLORS["annotation"]})
    ax.set_yticks(y)
    ax.set_yticklabels([s.label for s in shown], fontsize=7)
    ax.set_xlabel("mean rank  (1 = best)")
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)
    ax.grid(axis="y", visible=False)
    if hidden:
        style.note(ax, f"{hidden} mid-ranked methods hidden")
    return fig


# --------------------------------------------------------------------------- #
# 4. Calibration
# --------------------------------------------------------------------------- #


def plot_calibration_shift(df: pd.DataFrame):
    """ECE of every trained model against its own untuned base, one point per dataset.

    The diagonal is "calibration unchanged". Below it continued pretraining improved
    calibration; above it, the model became more confident and less right — which is what
    Tanna 2026 reports for naive finetuning, and what a credit-risk regulator would ask
    about first.
    """
    d = paired_deltas(df, "ece")
    if d.empty:
        return _empty("no paired ECE cells")

    fig, ax = _new("Calibration: trained vs its own base (ECE, lower is better)",
                   width=style.WIDTH_HALF, ratio=1.0)
    lo = float(min(d["untuned"].min(), d["trained"].min()))
    hi = float(max(d["untuned"].max(), d["trained"].max()))
    pad = 0.05 * (hi - lo or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=style.COLORS["reference"], linewidth=0.8, alpha=0.6)
    for base in sorted(d["base_short"].unique()):
        sub = d[d["base_short"] == base]
        ax.scatter(sub["untuned"], sub["trained"], s=16, alpha=0.8,
                   color=style.color(_base_key(base)), label=base, edgecolors="none")
    ax.set_xlabel("untuned ECE")
    ax.set_ylabel("trained ECE")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=6)
    worse = int((d["trained"] > d["untuned"]).sum())
    style.note(ax, f"{worse}/{len(d)} worse")
    return fig


# --------------------------------------------------------------------------- #
# 5. Regime analysis
# --------------------------------------------------------------------------- #


def plot_regime_effect(df: pd.DataFrame, manifest: pd.DataFrame,
                       metric: str = "roc_auc", prop: str = "n_rows"):
    """Δ from continued pretraining against a property of the dataset.

    Purucker 2026 does exactly this for the GBDT-over-TFM margin and finds it grows with
    sample size (ρ = +0.60) and with categorical cardinality (+0.47). It is the figure
    that turns "it helps on average" into "it helps here and not there", which is what a
    practitioner needs and what makes a negative result publishable.
    """
    d = paired_deltas(df, metric)
    if d.empty or manifest is None or manifest.empty or prop not in manifest.columns:
        return _empty(f"need paired cells and a manifest with `{prop}`")
    m = manifest[["dataset_id", prop]].copy()
    m[prop] = pd.to_numeric(m[prop], errors="coerce")
    d = (d.merge(m, left_on="test_dataset_id", right_on="dataset_id", how="inner")
           .dropna(subset=[prop, "delta"]))
    if d.empty:
        return _empty(f"no dataset property `{prop}` matched the scored datasets")

    fig, ax = _new(f"Where continued pretraining helps ({metric} vs {prop})")
    for base in sorted(d["base_short"].unique()):
        sub = d[d["base_short"] == base]
        ax.scatter(sub[prop], sub["delta"], s=18, alpha=0.8,
                   color=style.color(_base_key(base)), label=base, edgecolors="none")
    ax.axhline(0, color=style.COLORS["reference"], linewidth=0.8, alpha=0.6)
    # Log only when every value is positive. A manifest column that is missing or zero for
    # the matched datasets makes matplotlib raise "Data has no positive values", which
    # kills the whole notebook rather than degrading one panel.
    if (d[prop] > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel(prop.replace("_", " "))
    ax.set_ylabel(f"Δ {metric}")
    ax.legend(loc="best", fontsize=7)
    # A correlation needs both axes to vary. With one test dataset per property value —
    # or a manifest column that is constant across the scored datasets — `spearmanr`
    # returns NaN and warns; annotating "ρ = nan" is worse than annotating nothing.
    if len(d) >= 4 and d[prop].nunique() > 1 and d["delta"].nunique() > 1:
        from scipy.stats import spearmanr
        rho, p = spearmanr(d[prop], d["delta"])
        if np.isfinite(rho):
            style.note(ax, f"Spearman ρ = {rho:+.2f} (p = {p:.2f}, n = {len(d)})")
    return fig


# --------------------------------------------------------------------------- #
# 6. Honest selection
# --------------------------------------------------------------------------- #


def selection_honesty(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Best-on-test vs leave-one-dataset-out selection, per dataset.

    This project has no validation corpus — too few datasets — so the best trial is
    currently picked on the test set, which is optimistically biased by the winner's
    curse: with 16 trials and a handful of datasets, the maximum is partly noise.

    LODO fixes it without new runs. For each held-out dataset, pick the configuration
    that ranks best on the OTHER datasets, then report that configuration's score on the
    held-out one. The gap between the two columns is the size of the bias.
    """
    need = {_METHOD_COL, "test_dataset_id", "source", metric}
    if df is None or df.empty or not need <= set(df.columns):
        return pd.DataFrame()
    trained = _ok(df, metric)
    trained = trained[trained["source"].str.endswith("-trained", na=False)]
    if trained.empty:
        return pd.DataFrame()
    cell = trained.groupby([_METHOD_COL, "test_dataset_id"])[metric].mean().reset_index()
    wide = cell.pivot(index="test_dataset_id", columns=_METHOD_COL, values=metric).dropna(axis=1)
    if wide.shape[0] < 2 or wide.shape[1] < 2:
        return pd.DataFrame()

    asc = not higher_is_better(metric)
    rows = []
    for ds in wide.index:
        others = wide.drop(index=ds)
        chosen = others.mean(axis=0).sort_values(ascending=asc).index[0]
        best_here = wide.loc[ds].min() if asc else wide.loc[ds].max()
        rows.append({"test_dataset_id": ds, "lodo_choice": chosen,
                     "lodo_score": float(wide.loc[ds, chosen]),
                     "best_on_test": float(best_here)})
    return pd.DataFrame(rows)


def plot_selection_honesty(df: pd.DataFrame, metric: str = "roc_auc"):
    """Per dataset: the score of the configuration chosen without seeing that dataset,
    against the best score achievable on it. The gap is the winner's curse."""
    s = selection_honesty(df, metric)
    if s.empty:
        return _empty("need ≥2 datasets scored by ≥2 trained configurations")

    # At corpus scale one row per dataset is 500 labels on a half-page figure. Above the
    # bar limit the same quantity is shown as the DISTRIBUTION of the gap, which is the
    # number the reader wants anyway: how much the best-on-test figure overstates.
    if style.too_many(len(s), style.MAX_BARS):
        fig, ax = _new(f"Winner's curse ({metric}, {len(s)} datasets)", ratio=0.5)
        gaps = (s["best_on_test"] - s["lodo_score"]).abs()
        ax.hist(gaps, bins=min(40, max(10, len(gaps) // 10)),
                color=style.COLORS["highlight"], alpha=0.85)
        ax.axvline(float(gaps.mean()), color=style.COLORS["reference"], linewidth=1.0)
        ax.set_xlabel(f"optimism of best-on-test  (Δ {metric})")
        ax.set_ylabel("datasets")
        style.note(ax, f"mean {gaps.mean():.4f} · median {gaps.median():.4f}")
        return fig

    fig, ax = _new(f"Honest selection ({metric})", ratio=0.5)
    y = np.arange(len(s))[::-1]
    ax.hlines(y, s["lodo_score"], s["best_on_test"],
              color=style.COLORS["annotation"], linewidth=0.8, alpha=0.7)
    ax.scatter(s["best_on_test"], y, s=22, color=style.COLORS["annotation"],
               label="best on that dataset (optimistic)", zorder=3, edgecolors="none")
    ax.scatter(s["lodo_score"], y, s=22, color=style.COLORS["highlight"],
               label="chosen without seeing it", zorder=3, edgecolors="none")
    ax.set_yticks(y)
    ax.set_yticklabels(s["test_dataset_id"], fontsize=7)
    ax.set_xlabel(metric)
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="best", fontsize=7)
    gap = float((s["best_on_test"] - s["lodo_score"]).abs().mean())
    style.note(ax, f"mean gap {gap:.4f}")
    return fig


# --------------------------------------------------------------------------- #
# 7. Forgetting
# --------------------------------------------------------------------------- #


def plot_forgetting(df: pd.DataFrame, metric: str = "roc_auc"):
    """Trained score against untuned score, per (model × dataset), with the identity line.

    Kolberg 2026 checks continued pretraining for catastrophic forgetting by correlating
    the adapted model against its base on the base's original tasks (ρ = 0.9935) and the
    synthesis calls it "a forgetting check worth copying". Points far below the diagonal
    are datasets where adaptation destroyed what the prior already knew.
    """
    d = paired_deltas(df, metric)
    if d.empty:
        return _empty("no paired trained/untuned cells")

    fig, ax = _new(f"Forgetting check ({metric})", width=style.WIDTH_HALF, ratio=1.0)
    lo = float(min(d["untuned"].min(), d["trained"].min()))
    hi = float(max(d["untuned"].max(), d["trained"].max()))
    pad = 0.03 * (hi - lo or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=style.COLORS["reference"], linewidth=0.8, alpha=0.6)
    for base in sorted(d["base_short"].unique()):
        sub = d[d["base_short"] == base]
        ax.scatter(sub["untuned"], sub["trained"], s=16, alpha=0.8,
                   color=style.color(_base_key(base)), label=base, edgecolors="none")
    ax.set_xlabel(f"untuned {metric}")
    ax.set_ylabel(f"trained {metric}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=6)
    if len(d) >= 3:
        rho = float(pd.Series(d["untuned"]).corr(pd.Series(d["trained"]), method="spearman"))
        style.note(ax, f"Spearman ρ = {rho:.4f}")
    return fig
