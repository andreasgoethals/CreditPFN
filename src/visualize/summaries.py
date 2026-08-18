"""Printed text summaries — one per notebook, in that notebook's own section order.

Every notebook's last cell prints a summary (`AGENTS.md` §7), and the runner concatenates
those into `output/All_Results.md`. That document is the only place the run's numbers exist as
text rather than as pixels inside a PDF, so it has to carry **the headline of every figure**,
not just a file count: a reader who cannot open the figures should still be able to state what
the run found, and a figure whose number is not restated here is a figure nobody can quote.

The logic lives here rather than in the notebooks because a notebook in this project contains
no `def` and no `class`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.visualize import eval_viz as ev
from src.visualize import paper_figures as pf
from src.visualize import training_viz as tv


def _fmt(x, nd: int = 4) -> str:
    """Numbers that are missing say so, rather than printing `nan`."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(v) else f"{v:+.{nd}f}" if abs(v) < 1 else f"{v:.{nd}f}"



def _header(title: str, subtitle: str = "") -> list[str]:
    """The banner every printed summary opens with.

    `run_notebooks` concatenates these into `All_Results.md`, where each block used to begin
    with a bare "## PD training" and no indication of what the numbers below were or which
    notebook produced them. A titled, ruled banner makes the document navigable and makes an
    individual summary quotable on its own.
    """
    bar = "=" * 74
    out = [bar, f"  {title.upper()}", bar]
    if subtitle:
        out.append(f"  {subtitle}")
        out.append("")
    return out


def _rule(title: str) -> list[str]:
    return ["", title, "-" * len(title)]


# --------------------------------------------------------------------------- #
# Final-results notebooks (2.0 / 2.1)
# --------------------------------------------------------------------------- #


def eval_summary(track: str) -> str:
    """Text summary of the eval notebook, section by section, in notebook order."""
    metric = ev.primary_metric(track)
    raw = ev.load_eval_results(track)
    out = _header(f"{'2.0' if track == 'pd' else '2.1'}. "
                  f"{track.upper()} final results",
                  f"held-out benchmark · primary metric = {metric}")
    if raw.empty:
        return "\n".join(out + ["", "No eval results on disk."])

    ok = raw[raw["status"] == "OK"] if "status" in raw.columns else raw
    hib = pf.higher_is_better(metric)
    n_ds = ok["test_dataset_id"].nunique()

    out += _rule("1. Coverage")
    out += [
        f"  methods scored     : {ok['method_dirname'].nunique()}",
        f"  held-out datasets  : {n_ds}",
        f"  (method,ds,fold)   : {len(ok)} OK, {len(raw) - len(ok)} FAIL",
        f"  complete grid      : "
        f"{'yes' if len(ok) == ok['method_dirname'].nunique() * n_ds * 5 else 'NO - partial'}",
    ]

    out += _rule("2. Leaderboard (mean over datasets x folds)")
    lb = ev.eval_leaderboard(track)
    for r in lb.head(5).itertuples():
        out.append(f"  {r.method_name:<46} {getattr(r, 'mean', float('nan')):.4f}")
    if len(lb) > 5:
        w = lb.iloc[-1]
        out.append(f"  ... {len(lb) - 6} more ...")
        out.append(f"  {w['method_name']:<46} {w['mean']:.4f}   (last)")

    out += _rule("3. Zero-shot foundation model vs best tuned baseline")
    z = pf.zero_shot_vs_baseline(raw, metric)
    if z.empty:
        out.append("  not computable — needs untuned models and classical baselines")
    else:
        for base, g in z.groupby("base_short"):
            out.append(f"  untuned {base:<16} wins {int((g['delta'] > 0).sum())}/{len(g)}"
                       f"   mean delta {_fmt(g['delta'].mean())}")
        out.append(f"  overall                  {int((z['delta'] > 0).sum())}/{len(z)} pairs"
                   f" above zero")

    out += _rule("4. Effect of continued pretraining (paired vs own base)")
    d = pf.paired_deltas(raw, metric)
    if d.empty:
        out.append("  no paired trained/untuned cells")
    else:
        out.append(f"  overall            {int((d['delta'] > 0).sum())}/{len(d)} wins"
                   f"   mean {_fmt(d['delta'].mean())}   median {_fmt(d['delta'].median())}"
                   f"   best {_fmt(d['delta'].max())}")
        for base, g in d.groupby("base_short"):
            out.append(f"    {base:<16} {int((g['delta'] > 0).sum())}/{len(g)}"
                       f"   mean {_fmt(g['delta'].mean())}")
        # The independent unit is the DATASET; this is the number that can carry a p-value.
        per_ds = d.groupby("test_dataset_id")["delta"].mean()
        out.append(f"  per-dataset mean   n={len(per_ds)}   mean {_fmt(per_ds.mean())}")
        if len(per_ds) >= 2:
            from scipy import stats
            t = stats.ttest_1samp(per_ds, 0)
            half = stats.t.ppf(0.975, len(per_ds) - 1) * per_ds.std(ddof=1) / np.sqrt(len(per_ds))
            out.append(f"  95% CI over datasets: [{per_ds.mean() - half:+.4f},"
                       f" {per_ds.mean() + half:+.4f}]   p = {t.pvalue:.3f}"
                       f"   -> {'no detectable effect' if t.pvalue > 0.05 else 'significant'}")

    out += _rule("5. Corpus-size arm (min_train_rows)")
    if d.empty or "method_dirname" not in d.columns:
        out.append("  not computable")
    else:
        arm = np.where(d["method_dirname"].str.contains("min5000", na=False), "min5000", "none")
        g = d.assign(arm=arm).groupby("arm")["delta"]
        if g.ngroups < 2:
            out.append("  only one arm present")
        for name, s in g:
            out.append(f"  {name:<10} n={len(s):3d}  mean {_fmt(s.mean())}"
                       f"  wins {int((s > 0).sum())}/{len(s)}")

    out += _rule("6. Calibration (PD only)")
    if "ece" not in ok.columns or ok["ece"].isna().all():
        out.append("  no ECE recorded (regression track)")
    else:
        ce = pf.paired_deltas(raw, "ece")
        out.append(f"  mean untuned ECE : {ok[ok['source'].str.endswith('-untuned')]['ece'].mean():.4f}")
        out.append(f"  mean trained ECE : {ok[ok['source'].str.endswith('-trained')]['ece'].mean():.4f}")
        if not ce.empty:
            # `paired_deltas` signs deltas so positive = better; ECE is lower-is-better.
            out.append(f"  paired change    : {_fmt(-ce['delta'].mean())} "
                       f"({int((ce['delta'] > 0).sum())}/{len(ce)} improved)")

    out += _rule("7. Mean rank across datasets scored by every method")
    r = pf.mean_ranks(raw, metric)
    if r.empty:
        out.append("  no complete block of datasets")
    else:
        lookup = dict(zip(raw["method_dirname"], raw["method_name"]))
        for row in r.head(3).itertuples():
            out.append(f"  {lookup.get(getattr(row, 'method_dirname'), '?'):<46}"
                       f" {row.mean_rank:.2f}")

    out += _rule("8. Regime and honesty checks")
    if not d.empty:
        best_ds = d.groupby("test_dataset_id")["delta"].mean()
        out.append(f"  best dataset  : {best_ds.idxmax()}  {_fmt(best_ds.max())}")
        out.append(f"  worst dataset : {best_ds.idxmin()}  {_fmt(best_ds.min())}")
        rho = d["untuned"].corr(d["trained"], method="spearman")
        out.append(f"  forgetting (Spearman trained vs untuned over pairs): {rho:.4f}")
        out.append(f"  gain vs base quality (r over pairs): "
                   f"{d['untuned'].corr(d['delta']):+.2f}"
                   f"   [{len(d)} pairs on {n_ds} datasets]")

    out += _rule("9. Best method on each held-out dataset")
    if not ok.empty:
        cell = ok.groupby(["method_name", "test_dataset_id"])[metric].mean().reset_index()
        for ds, g in cell.groupby("test_dataset_id"):
            row = g.loc[g[metric].idxmax() if hib else g[metric].idxmin()]
            out.append(f"  {ds:<28} {row[metric]:.4f}  {row['method_name'][:40]}")

    out += _rule("10. Fold stability (std across the 5 folds, median over datasets)")
    if not ok.empty and "fold" in ok.columns:
        stds = (ok.groupby(["method_name", "test_dataset_id"])[metric].std()
                  .groupby("method_name").median().sort_values())
        for name, v in list(stds.items())[:3]:
            out.append(f"  most stable   {name[:44]:<44} {v:.4f}")
        for name, v in list(stds.items())[-2:]:
            out.append(f"  least stable  {name[:44]:<44} {v:.4f}")

    out += _rule("11. Failures")
    if len(raw) - len(ok) == 0:
        out.append("  none — every (method, dataset, fold) cell produced a number")
    else:
        bad = raw[~raw.index.isin(ok.index)]
        out.append(f"  {len(bad)} cell(s) failed")
        for (m, ds), g in bad.groupby(["method_name", "test_dataset_id"]):
            out.append(f"    {m[:40]:<40} {ds}  ({len(g)} fold(s))")

    out += _rule("12. Reading it")
    if not d.empty:
        direction = "helped" if d["delta"].mean() > 0 else "did not help"
        out.append(f"  Continued pretraining {direction} on average"
                   f" ({_fmt(d['delta'].mean())} {metric}).")
    if not z.empty:
        out.append(f"  Untuned foundation models beat the best tuned baseline on"
                   f" {int((z['delta'] > 0).sum())}/{len(z)} (base, dataset) pairs.")
    out.append(f"  Every number above rests on {n_ds} held-out datasets"
               f"{' — too few for significance' if n_ds < 10 else ''}.")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Training notebooks (1.0 / 1.1)
# --------------------------------------------------------------------------- #


def training_summary(track: str) -> str:
    """Text summary of the training notebook, section by section."""
    out = _header(f"{'1.0' if track == 'pd' else '1.1'}. {track.upper()} training",
                  "continued pretraining — what the sweep did, not whether it helped")
    man = tv.load_run_manifest(track)
    if man.empty:
        return "\n".join(out + ["", "No training manifest on disk."])

    out += _rule("1. Trials")
    out.append(f"  trials           : {len(man)}")
    if "status" in man.columns:
        for k, v in man["status"].value_counts().items():
            out.append(f"    {k:<14} {v}")
    if "diverge_reason" in man.columns and man["diverge_reason"].notna().any():
        for r in man[man["diverge_reason"].notna()].itertuples():
            out.append(f"    DIVERGED: {getattr(r, 'base_checkpoint', '?')}"
                       f" lr={getattr(r, 'learning_rate', '?')}"
                       f" reason={getattr(r, 'diverge_reason', '?')}")

    out += _rule("2. Budget actually realised")
    for col, label in (("total_optimizer_steps", "optimizer steps"),
                       ("epochs_run", "epochs"), ("steps_per_epoch", "steps/epoch")):
        if col in man.columns and man[col].notna().any():
            out.append(f"  {label:<16} {man[col].min():.0f} - {man[col].max():.0f}")
    if "total_optimizer_steps" in man.columns and "min_train_rows" in man.columns:
        # An epoch cap that binds unevenly across a swept axis confounds that axis with the
        # training budget — the failure mode of run-8's LGD track.
        g = man.groupby("min_train_rows")["total_optimizer_steps"]
        if g.ngroups > 1:
            spread = {k: (int(v.min()), int(v.max())) for k, v in g}
            out.append(f"  steps per corpus arm: {spread}")
            if len({v[0] for v in spread.values()}) > 1:
                out.append("  WARNING: arms received UNEQUAL step budgets — the corpus"
                           " comparison is confounded with training length.")

    out += _rule("3. The corpus this sweep trained on")
    for col, label in (("n_train_datasets", "train datasets"),
                       ("n_test_datasets", "held-out datasets"),
                       ("train_rows_total", "train rows"),
                       ("test_rows_total", "held-out rows")):
        if col in man.columns and man[col].notna().any():
            vals = sorted({int(v) for v in man[col].dropna()})
            out.append(f"  {label:<18} {vals if len(vals) > 1 else vals[0]:}")
    if "min_train_rows" in man.columns and "n_train_datasets" in man.columns:
        # The filter is applied to the TRAIN side only, so it changes the corpus size per arm.
        g = man.groupby("min_train_rows")["n_train_datasets"].agg(["min", "max"])
        if len(g) > 1:
            for arm, r in g.iterrows():
                out.append(f"    min_train_rows={int(arm):<5} -> {int(r['min'])} train datasets")
    if "train_dataset_ids" in man.columns and man["train_dataset_ids"].notna().any():
        ids = str(man["train_dataset_ids"].dropna().iloc[0]).split(";")
        out.append(f"  train ids ({len(ids)}): {', '.join(i.strip() for i in ids[:6])}"
                   + (" ..." if len(ids) > 6 else ""))
    if "test_dataset_ids" in man.columns and man["test_dataset_ids"].notna().any():
        ids = str(man["test_dataset_ids"].dropna().iloc[0]).split(";")
        out.append(f"  held-out ids ({len(ids)}): {', '.join(i.strip() for i in ids)}")

    out += _rule("4. The grid actually run")
    for col, label in (("base_checkpoint", "bases"), ("learning_rate", "learning rates"),
                       ("use_lora", "adapter arm"), ("query_fraction", "query fraction"),
                       ("epoch_pass_mode", "pass mode"), ("min_train_rows", "corpus arms")):
        if col in man.columns and man[col].notna().any():
            vals = man[col].dropna().unique()
            shown = [str(v).split("/")[-1].replace(".ckpt", "") for v in sorted(vals, key=str)]
            out.append(f"  {label:<16} {len(vals)}: {', '.join(shown)}")

    out += _rule("5. Weight drift from the base checkpoint")
    if "final_drift" in man.columns and man["final_drift"].notna().any():
        out.append(f"  final drift      {man['final_drift'].min():.4f}"
                   f" - {man['final_drift'].max():.4f}  (fraction of ||w0||)")
        if "learning_rate" in man.columns:
            for lr, g in man.groupby("learning_rate"):
                out.append(f"    lr={lr:<9.0e} mean {g['final_drift'].mean():.4f}")

    out += _rule("6. Monitor metric (in-loop, 2 000-row eval — NOT the benchmark)")
    ov = tv.training_overview(track)
    if not ov.empty and "best_test_metric" in ov.columns:
        out.append(f"  best_test_metric {ov['best_test_metric'].min():.4f}"
                   f" - {ov['best_test_metric'].max():.4f}")
        if "base_short" in ov.columns:
            for b, g in ov.groupby("base_short"):
                out.append(f"    {tv.compact_base(b):<12} {g['best_test_metric'].max():.4f}")

    out += _rule("7. In-loop movement (baseline -> final on the monitor split)")
    if {"baseline_test_metric", "final_test_metric"} <= set(man.columns):
        ok_rows = man[man["final_test_metric"].notna()
                      & man["baseline_test_metric"].notna()]
        if len(ok_rows):
            delta = ok_rows["final_test_metric"] - ok_rows["baseline_test_metric"]
            name = (str(man["primary_metric_name"].dropna().iloc[0])
                    if "primary_metric_name" in man.columns
                    and man["primary_metric_name"].notna().any() else "metric")
            out.append(f"  monitor {name}: mean change {delta.mean():+.4f}"
                       f"   range {delta.min():+.4f} to {delta.max():+.4f}")
            out.append(f"  improved on the monitor split: {int((delta > 0).sum())}/{len(delta)}"
                       f" trials")
            out.append("  NOTE this is a 2 000-row in-loop probe, not the benchmark. It exists")
            out.append("  to catch a dead run early; direction here has repeatedly disagreed")
            out.append("  with the held-out result, which is the eval notebook's job.")

    out += _rule("8. Recipe and provenance")
    for col, label in (("l2sp_lambda", "L2-SP lambda"), ("warmup_fraction", "warmup fraction"),
                       ("min_lr_fraction", "LR floor fraction"),
                       ("max_rows_per_epoch", "row cap / step"),
                       ("git_commit", "code commit"), ("tfm_library_pin", "tfm-library pin")):
        if col in man.columns and man[col].notna().any():
            vals = man[col].dropna().unique()
            shown = ", ".join(str(v)[:12] for v in sorted(vals, key=str)[:4])
            out.append(f"  {label:<18} {shown}" + (" ..." if len(vals) > 4 else ""))

    out += _rule("9. Cost")
    if "elapsed_sec" in man.columns and man["elapsed_sec"].notna().any():
        out.append(f"  total {man['elapsed_sec'].sum() / 3600:.1f} GPU-h"
                   f"   longest trial {man['elapsed_sec'].max() / 3600:.2f} h")

    out += _rule("10. Reading it")
    out.append("  The numbers here describe TRAINING only. Whether the resulting")
    out.append("  checkpoints are better than their bases is the eval notebook's")
    out.append("  question — the monitor metric above is a 2 000-row proxy.")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Data-exploration notebooks (0.0 / 0.1)
# --------------------------------------------------------------------------- #


def _corpus_table():
    from src.data.exploration import corpus_summary_table
    return corpus_summary_table()


def _anomalies():
    from src.data.exploration import find_anomalous_datasets
    return find_anomalous_datasets()


def _size_bands(rows: pd.Series) -> list[str]:
    """How the corpus splits across the size bands continued pretraining cares about.

    Garg 2025 curated tables of 10k-100k rows and reports that gains scale with table size,
    while a corpus of tiny tables *hurts*. So "how many of ours clear 10k" is the single most
    decision-relevant fact about this corpus, and it belongs in the printed summary rather
    than only in a scatter plot.
    """
    bands = [(0, 2_000, "< 2k"), (2_000, 5_000, "2k-5k"), (5_000, 10_000, "5k-10k"),
             (10_000, 100_000, "10k-100k"), (100_000, 10**12, ">= 100k")]
    out = []
    for lo, hi, label in bands:
        n = int(((rows >= lo) & (rows < hi)).sum())
        if n:
            out.append(f"    {label:<9} {n:2d} datasets ({100 * n / max(len(rows), 1):4.0f} %)")
    return out


def data_summary(stage: str) -> str:
    """Text summary of a data-exploration notebook. `stage` is "raw" or "processed"."""
    raw = stage == "raw"
    # Sections are numbered as they are emitted: the raw and processed branches contain a
    # different number of them, so any literal numbering is wrong for one of the two.
    _n = [0]

    def sec(title: str) -> list[str]:
        _n[0] += 1
        return _rule(f"{_n[0]}. {title}")

    title = f"{'0.0' if raw else '0.1'}. {'Raw' if raw else 'Processed'} data exploration"
    out = _header(title, f"what the {'vendor delivered' if raw else 'sanitiser produced'}")
    try:
        t = _corpus_table()
    except Exception as exc:                                   # no data/ on this machine
        return "\n".join(out + ["", f"Corpus table unavailable: {type(exc).__name__}: {exc}"])
    if t.empty:
        return "\n".join(out + ["", "No datasets on disk."])

    rows_col = "raw_rows" if raw else "post_rows"
    feat_col = "raw_features" if raw else "post_features"

    out += sec("Corpus shape")
    per_track = {k: int(v) for k, v in t["track"].value_counts().sort_index().items()}
    out.append(f"  datasets            : {len(t)}  "
               + "(" + ", ".join(f"{k.upper()} {v}" for k, v in per_track.items()) + ")")
    out.append(f"  total rows          : {int(t[rows_col].sum()):,}")
    out.append(f"  rows per dataset    : min {int(t[rows_col].min()):,}  "
               f"median {int(t[rows_col].median()):,}  max {int(t[rows_col].max()):,}")
    out.append(f"  features per dataset: min {int(t[feat_col].min())}  "
               f"median {int(t[feat_col].median())}  max {int(t[feat_col].max())}")

    out += sec("Size bands (the axis continued-pretraining gains scale on)")
    out += _size_bands(t[rows_col])
    big = int((t[rows_col] >= 10_000).sum())
    out.append(f"  >= 10 000 rows      : {big}/{len(t)} datasets"
               f" — Garg's curated corpus was 71 tables, all 10k-100k")

    out += sec("Per track")
    for track, g in t.groupby("track"):
        out.append(f"  {track.upper()}: {len(g)} datasets, {int(g[rows_col].sum()):,} rows, "
                   f"median {int(g[rows_col].median()):,} rows")
        if track == "pd" and "minority_class_ratio" in g.columns and g["minority_class_ratio"].notna().any():
            m = g["minority_class_ratio"].dropna()
            out.append(f"       minority class ratio: min {m.min():.4f}  median {m.median():.4f}"
                       f"  max {m.max():.4f}")
            severe = int((m < 0.05).sum())
            if severe:
                out.append(f"       {severe} dataset(s) below 5 % minority — severe imbalance")
        if track == "lgd" and "target_mean" in g.columns and g["target_mean"].notna().any():
            out.append(f"       target mean: {g['target_mean'].min():.4f} - "
                       f"{g['target_mean'].max():.4f}"
                       f"   std: {g['target_std'].min():.4f} - {g['target_std'].max():.4f}")

    if raw:
        out += sec("Missingness in the delivered files")
        if "missing_rate_raw" in t.columns and t["missing_rate_raw"].notna().any():
            m = t["missing_rate_raw"].dropna()
            out.append(f"  missing-cell rate  : min {m.min():.4f}  median {m.median():.4f}"
                       f"  max {m.max():.4f}")
            for thresh in (0.10, 0.30, 0.50):
                n = int((m > thresh).sum())
                if n:
                    out.append(f"    above {thresh:.0%}: {n} dataset(s)")
            worst = t.loc[m.idxmax()]
            out.append(f"  worst              : {worst['dataset_id']} "
                       f"({worst['missing_rate_raw']:.1%})")
    else:
        out += sec("What sanitisation changed")
        if {"raw_rows", "post_rows", "raw_features", "post_features"} <= set(t.columns):
            dr = t["post_rows"] - t["raw_rows"]
            df_ = t["post_features"] - t["raw_features"]
            out.append(f"  rows     : {int(t['raw_rows'].sum()):,} -> "
                       f"{int(t['post_rows'].sum()):,}"
                       f"  ({100 * dr.sum() / max(t['raw_rows'].sum(), 1):+.2f} %)")
            out.append(f"  features : {int(t['raw_features'].sum())} -> "
                       f"{int(t['post_features'].sum())}"
                       f"  (median {int(t['raw_features'].median())} -> "
                       f"{int(t['post_features'].median())})")
            lost = t[dr < 0]
            if len(lost):
                out.append(f"  {len(lost)} dataset(s) lost rows; largest loss "
                           f"{int(dr.min()):,} ({t.loc[dr.idxmin(), 'dataset_id']})")
            cap = t[t["post_features"] == t["post_features"].max()]
            out.append(f"  feature cap hit by {len(cap)} dataset(s) at "
                       f"{int(t['post_features'].max())} features")
        out += sec("Feature types after sanitisation")
        if {"n_categorical", "n_numerical"} <= set(t.columns):
            out.append(f"  categorical : total {int(t['n_categorical'].sum())}, "
                       f"median {int(t['n_categorical'].median())} per dataset")
            out.append(f"  numerical   : total {int(t['n_numerical'].sum())}, "
                       f"median {int(t['n_numerical'].median())} per dataset")
            allnum = int((t["n_categorical"] == 0).sum())
            out.append(f"  {allnum}/{len(t)} dataset(s) are purely numerical")

    out += sec("Provenance")
    if "source" in t.columns:
        for src, n in t["source"].value_counts().items():
            out.append(f"  {str(src)[:44]:<44} {n}")

    out += sec("Anomaly screen")
    try:
        a = _anomalies()
    except Exception as exc:
        out.append(f"  screen unavailable: {type(exc).__name__}")
        a = pd.DataFrame()
    if a.empty:
        out.append("  no dataset flagged (missing rate, row floor, minority share all OK)")
    else:
        out.append(f"  {len(a)} dataset(s) flagged:")
        for r in a.itertuples():
            out.append(f"    {getattr(r, 'dataset_id', '?'):<28} "
                       f"{getattr(r, 'reasons', '')}")

    out += sec("Reading it")
    if raw:
        out.append("  This is the corpus as delivered — nothing here has been cleaned, so a")
        out.append("  high missing rate or an odd target count is a fact about the vendor")
        out.append("  file, not a bug. Notebook 0.1 shows what sanitisation made of it.")
    else:
        out.append("  These are the tables the models actually see. The size bands in")
        out.append("  section 2 are the binding constraint on continued pretraining: a")
        out.append("  corpus of small tables is the one condition under which the")
        out.append("  literature reports the method losing to its own starting checkpoint.")
    return "\n".join(out)
