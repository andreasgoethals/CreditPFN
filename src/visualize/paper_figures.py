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
    """Map a base tag or a method label onto a registered `style.COLORS` name.

    Delegates to `eval_viz._method_series_name` so a method is the SAME colour in both
    modules. It used to have its own three-way test, which knew nothing about the classical
    baselines: `logreg` fell through to `style.color`'s crc32 fallback and came out orange in
    the mean-rank figure while being grey in every `eval_viz` figure of the same notebook.
    """
    from src.visualize.eval_viz import _method_series_name
    return _method_series_name(str(base_short))


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
        # Report the number of DATASETS, not of points. The points come in one vertical
        # cluster per dataset (every checkpoint shares that dataset's base score), so "n = 75"
        # advertises 75 independent observations where there are 5 — and any reviewer checks
        # this first. `r` is still over all points, which is what the drawn line fits.
        n_ds = d["test_dataset_id"].nunique()
        style.note(ax, f"slope {b:+.3f} · r = {r:+.2f} · {len(d)} pairs on {n_ds} datasets")
    ax.set_xlabel(f"untuned base {metric} on that dataset")
    ax.set_ylabel(f"Δ {metric}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
              borderaxespad=0.0)
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

    # Default to the frame's own display names. `load_eval_results` already computes
    # `method_name` and every other figure uses it, so falling back to the raw directory
    # name printed `tabpfn-trained__v3-default__lr3e-07__fullpass__min5000` down the y axis
    # of the one figure a reader looks at first — and coloured it off-palette, because the
    # colour key is derived from the label.
    if label_map is None and _METHOD_COL in df.columns and "method_name" in df.columns:
        lookup = dict(zip(df[_METHOD_COL], df["method_name"]))
        labels = [lookup.get(m, m) for m in r[_METHOD_COL]]
    else:
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
        # Distinguish "not applicable" from "went wrong". Expected calibration error is a
        # classification quantity, so on the LGD track this panel can never be filled, and a
        # bare "no paired ECE cells" reads as a broken figure rather than as a property of the
        # task. The regression analogue is CRPS, which needs the predicted distribution and is
        # not in the per-fold CSVs.
        # The column EXISTS on both tracks — `EvalRow` carries every metric field — and is
        # entirely NaN on the regression track. So test for a value, not for the column.
        has_ece = (df is not None and "ece" in df.columns and df["ece"].notna().any())
        if not has_ece:
            return _empty("expected calibration error is defined for classification only —\n"
                          "not applicable on a regression track")
        return _empty("no paired trained/untuned cells carry an ECE value")

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
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6,
              borderaxespad=0.0)
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
    # Log only when every value is positive AND the spread actually spans decades. Two
    # dataset sizes 4 637 and 5 627 apart on a log axis produce a page of empty decade ticks
    # around two touching clusters.
    if (d[prop] > 0).all() and d[prop].max() / max(d[prop].min(), 1e-9) >= 10:
        ax.set_xscale("log")
    ax.set_xlabel(prop.replace("_", " "))
    ax.set_ylabel(f"Δ {metric}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
              borderaxespad=0.0)
    # THE CORRELATION IS OVER DATASETS, NOT OVER POINTS. `prop` is a property of the dataset,
    # so every checkpoint on a given dataset shares one x value: correlating the 32 raw pairs
    # of a 2-dataset track yielded "Spearman rho = +0.39 (p = 0.03)" — a significant-looking
    # result that says only "the second dataset scored slightly higher", with n = 2. Collapse
    # to one point per dataset first, and refuse to report a coefficient below 4 of them,
    # which is where a rank correlation stops meaning anything at all.
    per_ds = d.groupby("test_dataset_id").agg(x=(prop, "first"), y=("delta", "mean"))
    n_ds = len(per_ds)
    if n_ds >= 4 and per_ds["x"].nunique() > 1 and per_ds["y"].nunique() > 1:
        from scipy.stats import spearmanr
        rho, p = spearmanr(per_ds["x"], per_ds["y"])
        if np.isfinite(rho):
            style.note(ax, f"Spearman ρ = {rho:+.2f} (p = {p:.2f}) over {n_ds} datasets")
    else:
        style.note(ax, f"{n_ds} datasets — too few for a correlation")
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
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
              borderaxespad=0.0)
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
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6,
              borderaxespad=0.0)
    if len(d) >= 3:
        rho = float(pd.Series(d["untuned"]).corr(pd.Series(d["trained"]), method="spearman"))
        style.note(ax, f"Spearman ρ = {rho:.4f}")
    return fig


# --------------------------------------------------------------------------- #
# 8. Zero-shot foundation model against tuned gradient boosting
# --------------------------------------------------------------------------- #


def zero_shot_vs_baseline(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Per dataset: each untuned base against the BEST tuned classical baseline.

    The comparison a credit-risk reader cares about most, and the one this project can make
    most strongly — and it had no figure. `plot_baselines_vs_tabpfn` pools both groups into
    two boxes, which mixes across datasets and so cannot answer "does it win on this table".
    """
    need = {"source", "base_short", "test_dataset_id", metric}
    if df is None or df.empty or not need <= set(df.columns):
        return pd.DataFrame()
    d = _ok(df, metric)
    if d.empty:
        return pd.DataFrame()
    cell = (d.groupby(["source", "base_short", "test_dataset_id"], dropna=False)[metric]
             .mean().reset_index())
    untuned = cell[cell["source"].str.endswith("-untuned", na=False)]
    base = cell[cell["source"] == "baseline"]
    if untuned.empty or base.empty:
        return pd.DataFrame()
    agg = "max" if higher_is_better(metric) else "min"
    best = base.groupby("test_dataset_id")[metric].agg(agg)
    rows = []
    for r in untuned.itertuples():
        if r.test_dataset_id not in best.index:
            continue
        ref = float(best.loc[r.test_dataset_id])
        val = float(getattr(r, metric))
        rows.append({"base_short": r.base_short, "test_dataset_id": r.test_dataset_id,
                     "model": val, "baseline": ref,
                     "delta": (val - ref) * _sign(metric)})
    return pd.DataFrame(rows)


def plot_zero_shot_vs_baseline(df: pd.DataFrame, metric: str = "roc_auc"):
    """Untuned foundation model minus the best tuned baseline, per (base x dataset).

    Grouped bars, one group per dataset, so the reader sees where the win comes from rather
    than a mean a single easy table could carry. Above zero, a foundation model that was
    never fitted to these data beat three Optuna-tuned baselines.
    """
    d = zero_shot_vs_baseline(df, metric)
    if d.empty:
        return _empty("need untuned foundation models and classical baselines")

    bases = sorted(d["base_short"].unique())
    datasets = sorted(d["test_dataset_id"].unique())
    if style.too_many(len(datasets) * len(bases)):
        # Too wide for a bar per pair: collapse to one distribution per base.
        fig, ax = _new(f"Zero-shot vs best tuned baseline ({metric})")
        ax.boxplot([d.loc[d["base_short"] == b, "delta"].values for b in bases],
                   tick_labels=bases, showmeans=True)
        ax.axhline(0, color=style.COLORS["reference"], linewidth=0.9)
        ax.set_ylabel(f"Δ {metric} vs best baseline")
        style.note(ax, f"{len(datasets)} datasets")
        return fig

    fig, ax = _new(f"Zero-shot vs best tuned baseline ({metric})", ratio=0.5)
    width = 0.8 / len(bases)
    x = np.arange(len(datasets))
    for k, b in enumerate(bases):
        sub = d[d["base_short"] == b].set_index("test_dataset_id").reindex(datasets)
        ax.bar(x + k * width - 0.4 + width / 2, sub["delta"].values, width * 0.92,
               color=style.color(_base_key(b)), label=b)
    ax.axhline(0, color=style.COLORS["reference"], linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([s.split(".", 1)[-1] for s in datasets], rotation=30,
                       ha="right", fontsize=7)
    ax.set_ylabel(f"Δ {metric} vs best baseline")
    # Symmetric about zero with headroom, so the bars read as signed deviations and the
    # legend has somewhere to go. `loc="best"` put it straight on top of the tallest pair.
    span = float(np.nanmax(np.abs(d["delta"]))) or 1.0
    ax.set_ylim(-span * 1.25, span * 1.55)
    ax.legend(loc="upper right", fontsize=7, ncol=len(bases), framealpha=0.9)
    style.note(ax, f"{int((d['delta'] > 0).sum())}/{len(d)} pairs above zero")
    return fig


# --------------------------------------------------------------------------- #
# 9. The corpus arm — run-8's swept axis
# --------------------------------------------------------------------------- #


def plot_corpus_arm(df: pd.DataFrame, metric: str = "roc_auc"):
    """Paired Δ split by the `min_train_rows` corpus filter.

    Garg's ablation reports that continued-pretraining gains scale with the SIZE of the
    tables in the corpus, and that a corpus of tiny tables hurts. `min_train_rows` is this
    project's test of that claim and the only axis of run-8 that moved the result, so it
    deserves a figure rather than a suffix on a leaderboard row.
    """
    d = paired_deltas(df, metric)
    if d.empty or _METHOD_COL not in d.columns:
        return _empty("no paired trained/untuned cells")
    d = d.copy()
    d["arm"] = np.where(d[_METHOD_COL].str.contains("min5000", na=False),
                        "≥ 5 000 rows", "no filter")
    if d["arm"].nunique() < 2:
        return _empty("only one corpus arm present — nothing to compare")

    fig, ax = _new(f"Effect of the corpus size filter ({metric}, paired)", ratio=0.5)
    bases = sorted(d["base_short"].unique())
    arms = ["no filter", "≥ 5 000 rows"]
    width = 0.36
    x = np.arange(len(bases))
    for k, arm in enumerate(arms):
        sel = [d[(d["base_short"] == b) & (d["arm"] == arm)]["delta"] for b in bases]
        ax.bar(x + k * width - width / 2, [s.mean() for s in sel], width * 0.9,
               yerr=[s.sem() for s in sel],
               color=[style.color(_base_key(b)) for b in bases],
               alpha=0.95 if k else 0.45, edgecolor="black", linewidth=0.4,
               error_kw={"linewidth": 0.7, "ecolor": style.COLORS["annotation"]})
    ax.axhline(0, color=style.COLORS["reference"], linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(bases)
    ax.set_ylabel(f"mean Δ {metric}  (trained − untuned)")
    # Two arms of the same colour differ only in alpha, so the legend has to be built by
    # hand rather than from the bar labels.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="grey", alpha=0.45, edgecolor="black", label=arms[0]),
                       Patch(facecolor="grey", alpha=0.95, edgecolor="black", label=arms[1])],
              loc="best", fontsize=7, title="training corpus", title_fontsize=7)
    style.note(ax, "error bars: standard error over (checkpoint, dataset) pairs")
    return fig


# --------------------------------------------------------------------------- #
# 10. The result as an effect size with a confidence interval
# --------------------------------------------------------------------------- #


def plot_effect_ci(df: pd.DataFrame, metric: str = "roc_auc"):
    """Mean paired Δ per base with a 95 % CI over DATASETS, the independent unit.

    When the finding is a null, the confidence interval IS the result: "the effect is
    smaller than X" is a claim, "we measured -0.0013" is not. Aggregating to one value per
    dataset first is what makes the interval honest — run-8's 75 (checkpoint, dataset) pairs
    are 5 independent observations, not 75.
    """
    d = paired_deltas(df, metric)
    if d.empty:
        return _empty("no paired trained/untuned cells")
    from scipy import stats

    rows = []
    for base in sorted(d["base_short"].unique()) + ["all bases"]:
        sub = d if base == "all bases" else d[d["base_short"] == base]
        per_ds = sub.groupby("test_dataset_id")["delta"].mean()
        n = len(per_ds)
        half = (float(stats.t.ppf(0.975, n - 1) * per_ds.std(ddof=1) / np.sqrt(n))
                if n >= 2 else np.nan)
        rows.append({"base": base, "mean": float(per_ds.mean()) if n else np.nan,
                     "half": half, "n": n})
    r = pd.DataFrame(rows)

    fig, ax = _new(f"Effect of continued pretraining, 95 % CI ({metric})", ratio=0.42)
    y = np.arange(len(r))[::-1]
    ax.errorbar(r["mean"], y, xerr=r["half"], fmt="o", markersize=5,
                color=style.COLORS["reference"], ecolor=style.COLORS["annotation"],
                elinewidth=1.1, capsize=3, linestyle="none")
    ax.axvline(0, color=style.COLORS["highlight"], linewidth=1.0, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t.base}  (n={t.n})" for t in r.itertuples()], fontsize=8)
    ax.set_xlabel(f"mean Δ {metric}  (trained − untuned), 95 % CI over datasets")
    ax.grid(axis="y", visible=False)
    # `style.note` writes at the bottom-right INSIDE the axes, which is exactly where the
    # last row's interval is drawn. Reserve a row's worth of space for it.
    ax.set_ylim(-0.85, len(r) - 0.4)
    style.note(ax, "CI crossing zero = no detectable effect")
    return fig


# --------------------------------------------------------------------------- #
# 11. The scheme benchmark: every adaptation scheme against ITS OWN base
# --------------------------------------------------------------------------- #
#
# This project is NOT a benchmark of TabPFN against TabICL. Those differ in pretraining data,
# architecture and parameter count, so a leaderboard across them measures the vendors, not us.
# What it IS a benchmark of is ADAPTATION SCHEMES: learning rate, adapter vs full fine-tune, and
# corpus filter, each scored against the base checkpoint it started from. The figures below hold
# the base fixed and vary only the scheme, which is the comparison that answers "which way of
# continuing to pretrain is best, and is any of them better than not doing it at all".


def _scheme_label(dirname: str) -> str:
    """The adaptation scheme of a result directory, with the base stripped out."""
    import re
    d = str(dirname)
    bits = []
    m = re.search(r"__lr([0-9eE.+\-]+)", d)
    if m:
        bits.append(f"lr {float(m.group(1)):.0e}")
    bits.append("adapter" if ("__lora" in d or "__iclhead" in d) else "full-FT")
    m = re.search(r"__min(\d+)", d)
    bits.append(f"min{int(m.group(1)) // 1000}k" if m else "no filter")
    return " · ".join(bits)


def scheme_table(df, metric: str = "roc_auc"):
    """(base, scheme, dataset) with the delta against that base. Signed so + is better."""
    d = paired_deltas(df, metric)
    if d.empty or _METHOD_COL not in d.columns:
        return pd.DataFrame()
    d = d.copy()
    d["scheme"] = d[_METHOD_COL].map(_scheme_label)
    return d


def plot_scheme_grid(df, metric: str = "roc_auc"):
    """One panel per base: adaptation scheme (rows) x held-out dataset (columns).

    Colour is the change against THAT base's own untuned score on THAT dataset, so zero means
    "continued pretraining changed nothing here", and the panels stay comparable even though the
    bases sit at different absolute scores. This is the per-dataset, per-scheme view the
    aggregates cannot give: a scheme that helps on one table and hurts on another averages to
    nothing and is indistinguishable from a scheme that does nothing anywhere.
    """
    import matplotlib.pyplot as plt
    d = scheme_table(df, metric)
    if d.empty:
        return _empty("no paired trained/untuned cells")

    bases = sorted(d["base_short"].unique())
    datasets = sorted(d["test_dataset_id"].unique())
    schemes = sorted(d["scheme"].unique())
    vmax = float(np.nanmax(np.abs(d["delta"]))) or 1e-6

    h = max(2.8, 0.26 * len(schemes) * len(bases) + 1.5)
    fig, axes = plt.subplots(
        len(bases), 1, squeeze=False,
        figsize=style.figsize(style.WIDTH_FULL, ratio=h / style.WIDTH_FULL),
    )
    im = None
    for ax, base in zip(axes[:, 0], bases):
        sub = d[d["base_short"] == base]
        rows = [s for s in schemes if s in set(sub["scheme"])]
        mat = (sub.pivot_table(index="scheme", columns="test_dataset_id", values="delta",
                               aggfunc="mean")
                  .reindex(index=rows, columns=datasets))
        # A diverging map centred on zero, shared across panels: the sign is the message.
        im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels(mat.index, fontsize=6)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(([s.split(".", 1)[-1] for s in datasets]
                            if base == bases[-1] else []),
                           rotation=30, ha="right", fontsize=6)
        ax.set_title(f"base: {base}", fontsize=8, loc="left")
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=5.5,
                            color="white" if abs(v) > 0.62 * vmax else "black")
    if im is not None:
        cb = fig.colorbar(im, ax=list(axes[:, 0]), fraction=0.03, pad=0.02)
        cb.set_label(f"delta {metric} vs own base", fontsize=7)
    fig.suptitle(f"Adaptation scheme x dataset, against each base ({metric})")
    return fig


def plot_scheme_metrics(df, metrics=("roc_auc", "brier", "ece", "f1")):
    """Mean change per scheme across SEVERAL metrics, one panel per base.

    Discrimination is not the only thing continued pretraining can move, and for credit risk it
    is arguably not the most important one: Brier and ECE are what a validation function reads.
    A scheme that leaves AUC alone while improving Brier is a result, and the AUC-only view
    reports it as a null.
    """
    import matplotlib.pyplot as plt
    present = [m for m in metrics
               if df is not None and m in df.columns and df[m].notna().any()]
    if not present:
        return _empty("none of the requested metrics are present on this track")

    frames = []
    for m in present:
        t = scheme_table(df, m)
        if t.empty:
            continue
        g = t.groupby(["base_short", "scheme"])["delta"].mean().reset_index()
        g["metric"] = m
        frames.append(g)
    if not frames:
        return _empty("no paired cells for these metrics")
    allg = pd.concat(frames, ignore_index=True)

    bases = sorted(allg["base_short"].unique())
    schemes = sorted(allg["scheme"].unique())
    h = max(2.8, 0.24 * len(schemes) + 1.6)
    fig, axes = plt.subplots(
        1, len(bases), squeeze=False, sharey=True,
        figsize=style.figsize(style.WIDTH_FULL, ratio=h / style.WIDTH_FULL),
    )
    y = np.arange(len(schemes))
    width = 0.8 / len(present)
    pal = style.categorical(present)
    for ax, base in zip(axes[0], bases):
        sub = allg[allg["base_short"] == base]
        for k, m in enumerate(present):
            s = sub[sub["metric"] == m].set_index("scheme").reindex(schemes)["delta"]
            ax.barh(y + k * width - 0.4 + width / 2, s.values, width * 0.9,
                    color=pal[m], label=(m if base == bases[0] else None))
        ax.axvline(0, color=style.COLORS["reference"], linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(schemes if base == bases[0] else [], fontsize=6)
        ax.set_title(base, fontsize=8)
        ax.set_xlabel("mean delta (+ better)", fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(axis="y", visible=False)
    axes[0][0].legend(loc="lower left", fontsize=6, title="metric", title_fontsize=6)
    fig.suptitle("Every adaptation scheme against its own base, across metrics")
    return fig
