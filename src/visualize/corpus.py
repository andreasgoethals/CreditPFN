"""Corpus geometry, cleaning and exposure views, with bounded dataset pages."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import exploration
from src.data.dataset_names import display_frame, display_name
from src.visualize import style
from src.visualize.campaign import Page


def raw_inventory() -> pd.DataFrame:
    data = exploration.raw_corpus_summary().copy()
    data["features"] = data.raw_cols - data.target_in_raw.astype(int)
    # German Credit's last delivered column is the label even before renaming.
    german = data.dataset_id.eq("0008.german") & ~data.target_in_raw & data.raw_cols.gt(0)
    data.loc[german, "features"] = data.loc[german, "raw_cols"] - 1
    data["rows"] = data.raw_rows.where(data.raw_rows > 0)
    data["features"] = data.features.where(data.features > 0)
    data["missing"] = data.missing_cells_rate
    data["dataset_id"] = data.dataset_id.map(display_name)
    return display_frame(data)


def delivery_checks(data: pd.DataFrame) -> pd.DataFrame:
    """Distinguish expected target construction from genuinely missing inputs."""
    from src.data.preprocessing import DATASET_METADATA
    processed = exploration._resolve_paths()["processed"]
    metadata = {(m["track"], display_name(key)): (key, m) for key, m in DATASET_METADATA.items()}
    notes = {
        "0008.german": "The final raw column is the label; preprocessing renames it to target and maps 1/2 to 0/1.",
        "0008.SBA_loans_case": "Preprocessing selects defaulted loans and derives LGD as charge-off divided by disbursement.",
    }
    rows = []
    for row in data[data.rows.isna() | ~data.target_in_raw].itertuples():
        key, meta = metadata[(row.track, row.dataset_id)]
        path = processed / row.track / (key + ".sanitized.csv")
        present = path.is_file() and meta["target_column"] in pd.read_csv(path, nrows=0).columns
        expected = key in notes and pd.notna(row.rows)
        rows.append(dict(track=row.track, dataset=row.dataset_id, target_in_raw=row.target_in_raw,
                         target_in_processed=present,
                         status="Expected preprocessing" if expected and present else "Inspect delivery",
                         explanation=notes.get(key, "The registered raw file or target column is missing.")))
    return pd.DataFrame(rows, columns=["track", "dataset", "target_in_raw", "target_in_processed", "status", "explanation"])


def processed_inventory() -> pd.DataFrame:
    data = exploration.corpus_summary_table().copy()
    records = []
    for _, row in data.iterrows():
        extra = dict(row)
        try:
            frame = exploration.load_sanitized_dataset(row.track, row.dataset_id)
            predictors = frame.drop(columns=[row.target_column], errors="ignore")
            extra.update(rows=len(frame), features=len(predictors.columns),
                         missing=float(predictors.isna().to_numpy().mean()) if predictors.size else np.nan)
            target = pd.to_numeric(frame[row.target_column], errors="coerce").dropna()
            if row.track == "lgd":
                extra.update(target_mean=target.mean(), target_std=target.std())
            else:
                counts = target.value_counts(normalize=True)
                extra["minority_class_ratio"] = counts.min() if len(counts) > 1 else 0. if len(counts) else np.nan
        except FileNotFoundError:
            extra.update(rows=np.nan, features=np.nan, missing=np.nan,
                         target_mean=np.nan, target_std=np.nan, minority_class_ratio=np.nan)
        records.append(extra)
    result = pd.DataFrame(records).drop(columns="target_column", errors="ignore")
    result["dataset_id"] = result.dataset_id.map(display_name)
    return display_frame(result)


def inventory_summary(data: pd.DataFrame, stage: str) -> str:
    observed = data[data.rows.notna()]
    return (f"{stage}: {len(observed)}/{len(data)} registered datasets available.\n" +
            observed.groupby("track").agg(datasets=("dataset_id", "nunique"), rows=("rows", "sum"),
                min_rows=("rows", "min"), max_rows=("rows", "max"), max_features=("features", "max")).to_string())


def source_table(data: pd.DataFrame) -> pd.DataFrame:
    """Registered datasets per source and task (provenance, not statistical independence)."""
    return (data.groupby(["source", "track"]).size().unstack(fill_value=0)
            .reindex(columns=["pd", "lgd"], fill_value=0).rename(columns=str.upper).reset_index())


def plot_corpus_overview(data: pd.DataFrame, partitions: pd.DataFrame) -> list[Page]:
    """One row per processed table (PD block above LGD, each by decreasing size): rows and
    predictors on log axes, missing predictor cells, and the target summary; the label carries
    the held-out partition."""
    part = partitions.rename(columns={"dataset": "dataset_id", "fold": "partition"})
    d = data.merge(part[["track", "dataset_id", "partition"]], on=["track", "dataset_id"], how="left")
    d["target"] = np.where(d.track.eq("pd"), d.get("minority_class_ratio"), d.get("target_mean"))
    blocks = [d[d.track.eq(t)].sort_values(["rows", "dataset_id"], ascending=[False, True]) for t in ("pd", "lgd")]
    blocks = [b for b in blocks if len(b)]
    if not blocks:
        return []
    ypos, labels, rows = [], [], []
    y = 0
    for block in blocks:
        for row in block.itertuples():
            ypos.append(y)
            part_label = f"P{int(row.partition)}" if pd.notna(row.partition) else "P?"
            labels.append(f"{row.dataset_id} · {part_label}")
            rows.append(row)
            y -= 1
        y -= 0.6
    frame = pd.DataFrame([r._asdict() for r in rows]).assign(y=ypos)
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(style.WIDTH_FULL, min(
        style.MAX_HEIGHT, style.FIGURE_BAND + style.TABLE_ROW_HEIGHT * (len(frame) + 2))),
        gridspec_kw=dict(width_ratios=(1.15, 1, 1, 1)))
    for track in ("pd", "lgd"):
        sub = frame[frame.track.eq(track)]
        if sub.empty:
            continue
        color = style.color(track)
        axes[0].scatter(sub.rows, sub.y, s=style.POINT_SIZE, color=color, label=f"{track.upper()} tables")
        axes[1].scatter(sub.features, sub.y, s=style.POINT_SIZE, color=color)
        axes[2].barh(sub.y, sub.missing * 100, color=color, height=style.BAR_HEIGHT)
        axes[3].barh(sub.y, sub.target, color=color, height=style.BAR_HEIGHT)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("Rows (log scale)")
    axes[1].set_xlabel("Predictors (log scale)")
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
    axes[1].xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50]))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%g"))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[2].set_xlabel("Missing cells (%)")
    axes[2].set_xlim(0, max(5.0, float(np.nanmax(frame.missing * 100)) * 1.1))
    axes[3].set_xlabel("Default rate / mean LGD")
    axes[3].set_xlim(0, 1)
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(ypos, labels)
    axes[0].legend()
    fig.suptitle("Credit corpus: size, predictors, missingness and target")
    tables = frame[["track", "dataset_id", "partition", "rows", "features", "missing", "target"]]
    return [Page("corpus_overview", fig,
                 "Every processed credit table, PD (17) above LGD (8), each block ordered by decreasing row count; "
                 "P1–P4 is the dataset partition in which the table is held out from continued pretraining. Rows and "
                 "predictors use logarithmic axes; missing predictor cells exclude the target; the last panel shows the "
                 "minority-class (default) rate for PD and the mean loss rate for LGD, both on a zero-based axis.",
                 {"Corpus overview": tables})]


def plot_lgd_distributions() -> list[Page]:
    from src.data.preprocessing import DATASET_METADATA
    ids = sorted(k for k, v in DATASET_METADATA.items() if v["track"] == "lgd")
    if not ids:
        return []
    columns = min(4, len(ids))
    nrows = int(np.ceil(len(ids) / columns))
    fig, axes = plt.subplots(nrows, columns, squeeze=False, sharex=True,
                             figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, style.FIGURE_BAND + nrows * 1.25)))
    statistics = []
    for ax, dataset in zip(axes.flat, ids):
        meta = DATASET_METADATA[dataset]
        try:
            frame = exploration.load_sanitized_dataset("lgd", dataset)
            values = pd.to_numeric(frame[meta["target_column"]], errors="coerce").dropna()
        except FileNotFoundError:
            values = pd.Series(dtype=float)
        if not values.empty:
            ax.hist(values, bins=30, weights=np.full(len(values), 1 / len(values)), color=style.color("lgd"))
            for name, value in (("Mean", values.mean()), ("Median", values.median())):
                ax.axvline(value, color=style.color(name.lower()), linestyle=":", label=name)
            statistics.append(dict(dataset=display_name(dataset), rows=len(values), mean=values.mean(),
                                   median=values.median(), share_zero=float((values <= 0).mean()),
                                   share_one=float((values >= 1).mean())))
        ax.set_title(display_name(dataset), fontsize=style.ANNOTATION_SIZE)
    for ax in axes[-1]:
        ax.set_xlabel("LGD")
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of rows")
    for ax in list(axes.flat)[len(ids):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle("LGD target distribution per table")
    return [Page("lgd_distributions", fig,
                 f"Processed LGD target of each of the {len(ids)} LGD tables: 30-bin histograms of the fraction of rows, "
                 "with dotted lines at the table's mean and median. Values follow the registered preprocessing, "
                 "including the [0, 1] clip; tables are not pooled and vertical scales differ between panels.",
                 {"LGD distribution statistics": pd.DataFrame(statistics)})]


def partition_table() -> pd.DataFrame:
    from src.train.config import load_train_config
    from src.train.corpus import split_from_cfg
    from src.utils.experiment import apply_split_index
    rows = []
    for track in ("pd", "lgd"):
        # The second-campaign reference recipe fixes the four dataset partitions every arm shares.
        path = __import__("src.utils.design", fromlist=["config_path"]).config_path("reference", track)
        for fold in range(4):
            cfg = apply_split_index(load_train_config(config_path=str(path)), fold)
            split = split_from_cfg(cfg)
            for ref in split.test:
                rows.append(dict(track=track, dataset=display_name(ref.dataset_id), fold=fold + 1))
    return pd.DataFrame(rows)


def planned_exposure(data: pd.DataFrame, row_cap=10000) -> pd.DataFrame:
    """Illustrate full traversal geometry, explicitly without pretending to measure a run."""
    d = data[data.rows.gt(0)].copy()
    d["chunks"] = np.ceil(d.rows / row_cap).astype(int)
    d["row_share"] = d.rows / d.groupby("track").rows.transform("sum")
    d["full_pass_step_share"] = d.chunks / d.groupby("track").chunks.transform("sum")
    d["one_sample_step_share"] = 1 / d.groupby("track").rows.transform("count")
    d["full_pass_share_ratio"] = d.full_pass_step_share / d.one_sample_step_share
    d["illustrative_row_cap"] = row_cap
    return d


def plot_exposure(data: pd.DataFrame) -> list[Page]:
    """Both tasks in one figure: per table, its share of optimizer updates under one-sample
    visits and under a full pass, with the full-pass/one-sample ratio beside each table."""
    tracks = [t for t in ("pd", "lgd") if t in set(data.track)]
    if not tracks:
        return []
    groups = {t: data[data.track.eq(t)].sort_values("rows", ascending=False) for t in tracks}
    fig, axes = plt.subplots(len(tracks), 1, squeeze=False, figsize=(style.WIDTH_FULL, min(
        style.MAX_HEIGHT, style.FIGURE_BAND + style.TABLE_ROW_HEIGHT * (len(data) + 3))),
        gridspec_kw=dict(height_ratios=[max(len(groups[t]), 2) for t in tracks]))
    for ax, track in zip(axes[:, 0], tracks):
        group = groups[track]
        y = np.arange(len(group))
        for mode, label in (("one_sample", "One sample per table visit"), ("full_pass", "Full pass")):
            ax.scatter(group[f"{mode}_step_share"], y, marker=style.SEED_MARKERS[int(mode == "full_pass")],
                       color=style.SAMPLING_COLORS[mode], label=label, s=style.POINT_SIZE)
        for pos, (_, row) in enumerate(group.iterrows()):
            ratio = row.full_pass_share_ratio
            text = f"{ratio:.3f}×" if round(ratio, 2) == 1 and ratio != 1 else f"{ratio:.2f}×"
            # Right of whichever marker is further right, so a label never covers a marker.
            anchor = max(row.full_pass_step_share, row.one_sample_step_share)
            ax.annotate(text, (anchor, pos), xytext=(style.EXPOSURE_LABEL_PAD, 0),
                        textcoords="offset points", va="center", fontsize=style.ANNOTATION_SIZE,
                        color=style.color("improvement" if ratio > 1 else "deterioration" if ratio < 1
                                          else "reference"))
        ax.set_yticks(y, [f"{r.dataset_id} ({r.rows:,.0f})" for r in group.itertuples()])
        ax.invert_yaxis()
        maximum = group[[f"{mode}_step_share" for mode in style.SAMPLING_LABELS]].max().max()
        ax.set_xlim(0, maximum * style.EXPOSURE_LIMIT_FACTOR)
        ax.set_title(f"{track.upper()} tables")
        ax.grid(axis="x", alpha=0.25)
    cap = int(data.illustrative_row_cap.iloc[0])
    axes[-1, 0].set_xlabel(f"Share of optimizer updates (chunks of {cap:,} rows)")
    axes[0, 0].legend()
    fig.suptitle("Optimizer-update share per table: one sample versus full pass")
    return [Page("planned_exposure", fig,
                 f"Planned share of optimizer updates per credit table if every processed table of the task were in "
                 f"the corpus: one sampled episode per table visit (circles, equal shares) versus a full pass over "
                 f"disjoint chunks of {cap:,} context-plus-query rows (squares, one update per chunk). Labels give the "
                 "table's rows and the full-pass share divided by the one-sample share (blue above one, red below; "
                 "colour does not mean better or worse). Illustration only: runs train on three of four partitions, "
                 "TabPFN v3 and TabICLv2 use 26,000-row episodes, and a run stops at its update budget.",
                 {"Step shares and full-pass share ratios": data[["track", "dataset_id", "rows", "chunks",
                                                                  "one_sample_step_share", "full_pass_step_share",
                                                                  "full_pass_share_ratio"]]})]
