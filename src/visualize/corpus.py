"""Corpus geometry, cleaning and exposure views, with bounded dataset pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import exploration
from src.data.dataset_names import display_frame, display_name
from src.visualize import style
from src.visualize.campaign import Page, _subplots, _heatmap, config_path


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
        except FileNotFoundError:
            extra.update(rows=np.nan, features=np.nan, missing=np.nan)
        records.append(extra)
    result = pd.DataFrame(records).drop(columns="target_column", errors="ignore")
    result["dataset_id"] = result.dataset_id.map(display_name)
    return display_frame(result)


def inventory_summary(data: pd.DataFrame, stage: str) -> str:
    observed = data[data.rows.notna()]
    return (f"{stage}: {len(observed)}/{len(data)} registered datasets available.\n" +
            observed.groupby("track").agg(datasets=("dataset_id", "nunique"), rows=("rows", "sum"),
                min_rows=("rows", "min"), max_rows=("rows", "max"), max_features=("features", "max")).to_string())


def plot_geometry(data: pd.DataFrame, stage: str) -> list[Page]:
    fig, (ax,) = _subplots(f"{stage}: corpus geometry")
    for track in ("pd", "lgd"):
        group = data[data.track.eq(track)].dropna(subset=["rows", "features"])
        ax.scatter(group.rows, group.features, s=style.POINT_SIZE, color=style.color(track),
                   marker=style.SEED_MARKERS[int(track == "lgd")], label=f"{track.upper()} ({len(group)} datasets)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Rows (log scale)"); ax.set_ylabel("Predictors (log scale)")
    ax.legend()
    return [Page("corpus_geometry", fig, f"Predictor count versus row count in the {stage.lower()} corpus, on logarithmic axes. Each point is one available dataset; color and marker distinguish PD and LGD on common axes. "
        + ("Raw columns exclude the registered target or the recognized final label column in German Credit, but can include identifiers later removed during cleaning." if stage == "Raw" else "Predictor counts exclude the target after registered preprocessing."),
        {"Dataset geometry": data[["track", "dataset_id", "rows", "features"]]})]


def plot_profiles(data: pd.DataFrame, stage: str) -> list[Page]:
    pages = []
    for track, group in data.groupby("track"):
        group = group.sort_values(["rows", "dataset_id"], ascending=[False, True], na_position="last")
        for number, (start, stop) in enumerate(style.page_slices(len(group)), 1):
            page = group.iloc[start:stop]
            fig, axes = _subplots(f"{stage} / {track.upper()} / datasets {start+1}–{start+len(page)}", 3, sharey=True)
            y = np.arange(len(page))
            for ax, field, label in zip(axes[:2], ("rows", "features"), ("Rows", "Predictors")):
                bars = ax.barh(y, page[field]-1, left=1, color=style.color(track), height=style.BAR_HEIGHT)
                ax.bar_label(bars, labels=[f"{int(v):,}" if pd.notna(v) else "Unavailable" for v in page[field]],
                             padding=style.BAR_LABEL_PAD, fontsize=style.ANNOTATION_SIZE)
                ax.set_xscale("log")
                ax.set_xlabel(label + " (log)")
                available = group[field].dropna()
                if not available.empty:
                    ax.set_xlim(1, available.max()*style.LOG_BAR_LIMIT_FACTOR)
            axes[0].set_yticks(y, page.dataset_id)
            axes[0].invert_yaxis()
            axes[2].barh(y, page.missing*100, color=style.color(track), height=style.BAR_HEIGHT)
            axes[2].set_xlabel("Missing cells (%)"); axes[2].set_xlim(0,100)
            pages.append(Page(f"dataset_profiles_{track}_{number}", fig,
                f"Row counts, predictor counts and missing-cell percentages for {len(page)} {track.upper()} datasets, sorted by decreasing size and continued across pages. "
                + ("Raw missingness uses all delivered columns, including a target column when present." if stage == "Raw" else "Processed missingness uses predictor cells only, excluding the target.")
                + " Counts use logarithmic axes with a baseline of one; labels give exact counts and bar lengths must not be read as linear ratios. Missingness uses a zero-based linear axis. Missing files remain unknown.",
                {"Dataset sizes and missingness": page[["dataset_id", "rows", "features", "missing"]]}))
    return pages


def plot_concentration(data: pd.DataFrame) -> list[Page]:
    fig, axes = _subplots("How much of the corpus sits in large tables?")
    for track, group in data.groupby("track"):
        sizes = group.rows.dropna().sort_values(ascending=False).to_numpy()
        if len(sizes) and sizes.sum() > 0:
            x = np.r_[0, np.arange(1,len(sizes)+1)/len(sizes)]
            axes[0].plot(x, np.r_[0,sizes.cumsum()/sizes.sum()], marker="o", color=style.color(track), label=track.upper())
    axes[0].plot([0,1],[0,1], color=style.color("reference"), linestyle=":", label="Equal table sizes")
    axes[0].set_xlabel("Fraction of datasets, largest first")
    axes[0].set_ylabel("Fraction of all rows")
    axes[0].legend()
    return [Page("row_concentration", fig, "Cumulative fraction of corpus rows contributed by datasets ordered from largest to smallest, separately for PD and LGD. The diagonal represents equal table sizes. This summarizes row imbalance; it does not by itself give model-specific batch or optimizer-update counts.")]


def plot_sources(data: pd.DataFrame) -> list[Page]:
    counts = data.groupby(["source", "track"]).size().unstack(fill_value=0)
    fig, axes = _subplots("Corpus sources")
    left = np.zeros(len(counts))
    for track in ("pd", "lgd"):
        if track in counts:
            axes[0].barh(counts.index, counts[track], left=left, color=style.color(track), label=track.upper())
            left += counts[track].to_numpy()
    axes[0].set_xlabel("Registered datasets")
    axes[0].legend()
    return [Page("sources", fig, "Registered dataset counts by source and task. Dataset counts describe provenance, not statistical independence between related sources.")]


def plot_cleaning(data: pd.DataFrame) -> list[Page]:
    pages = []
    for field, old, new, label in (("rows", "raw_rows", "post_rows", "Rows"), ("features", "raw_features", "post_features", "Predictors")):
        fig, axes = _subplots(f"Cleaning: {label.lower()} retained", 2)
        for ax, track in zip(axes, ("pd", "lgd")):
            group = data[data.track.eq(track) & data[old].gt(0) & data[new].gt(0)]
            ax.scatter(group[old], group[new], color=style.color(track), s=style.POINT_SIZE)
            if len(group):
                lower, upper = min(group[old].min(), group[new].min()), max(group[old].max(),group[new].max())
                ax.plot([lower,upper],[lower,upper], color=style.color("reference"), linestyle=":")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(f"Registered {label.lower()}"); ax.set_ylabel(f"Processed {label.lower()}")
            ax.set_title(track.upper())
        pages.append(Page(f"cleaning_{field}", fig, f"Processed versus registered {label.lower()} for each dataset, on logarithmic axes; the diagonal is equality. Registration includes dataset-specific corrections, so this comparison is distinct from the unmodified raw-file inventory. Unknown shapes are omitted."))
    return pages


def plot_target_profiles(data: pd.DataFrame) -> list[Page]:
    pages = []
    for track, column, label in (("pd", "minority_class_ratio", "Minority-class fraction"), ("lgd", "target_mean", "Mean LGD")):
        group = data[data.track.eq(track)].dropna(subset=[column]).sort_values(column)
        for number, (start, stop) in enumerate(style.page_slices(len(group), style.SIMPLE_PAGE_ROWS), 1):
            page = group.iloc[start:stop]
            fig, axes = _subplots(f"{track.upper()}: target profiles")
            axes[0].barh(np.arange(len(page)), page[column], color=style.color(track), height=style.BAR_HEIGHT)
            axes[0].set_yticks(range(len(page)), page.dataset_id)
            axes[0].set_xlabel(label)
            axes[0].set_xlim(min(0,float(page[column].min())),max(.5 if track == "pd" else 1,float(page[column].max())))
            pages.append(Page(f"target_profiles_{track}_{number}", fig,
                f"Registered {label.lower()} for each available {track.upper()} dataset, ordered by value. Bars start at zero; each dataset has equal visual weight, regardless of its row count."))
    return pages


def plot_lgd_distributions() -> list[Page]:
    from src.data.preprocessing import DATASET_METADATA
    ids = sorted(k for k,v in DATASET_METADATA.items() if v["track"] == "lgd")
    pages = []
    for start in range(0,len(ids),4):
        statistics = []
        fig, axes = plt.subplots(2,2,figsize=style.figsize(style.WIDTH_FULL,style.PANEL_RATIO), layout="constrained")
        for ax, dataset in zip(axes.flat, ids[start:start+4]):
            meta = DATASET_METADATA[dataset]
            try:
                frame = exploration.load_sanitized_dataset("lgd",dataset)
                values = pd.to_numeric(frame[meta["target_column"]],errors="coerce").dropna()
            except FileNotFoundError:
                values = pd.Series(dtype=float)
            if not values.empty:
                ax.hist(values, bins=30, weights=np.full(len(values),1/len(values)), color=style.color("lgd"))
                for name, value in (("Mean", values.mean()), ("Median", values.median())):
                    ax.axvline(value, color=style.color(name.lower()), linestyle=":", label=name)
                ax.legend()
                statistics.append(dict(dataset=display_name(dataset), rows=len(values), mean=values.mean(), median=values.median()))
            ax.set_title(display_name(dataset), fontsize=style.ANNOTATION_SIZE)
            ax.set_xlabel("LGD"); ax.set_ylabel("Fraction of rows")
        for ax in list(axes.flat)[len(ids[start:start+4]):]:
            ax.set_visible(False)
        pages.append(Page(f"lgd_distributions_{start//4+1}",fig,"Processed LGD target distributions, four datasets per page. Bars show fractions of rows within each dataset; dotted lines locate its mean and median. Values follow the registered preprocessing, including LGD target clipping; this plot applies no additional clipping and does not pool tables.",
                          {"LGD distribution statistics": pd.DataFrame(statistics)}))
    return pages


def partition_table() -> pd.DataFrame:
    from src.train.config import load_train_config
    from src.train.corpus import split_from_cfg
    from src.utils.experiment import apply_split_index
    rows = []
    for track in ("pd", "lgd"):
        for fold in range(4):
            cfg = apply_split_index(load_train_config(config_path=str(config_path(1,track))),fold)
            split = split_from_cfg(cfg)
            for ref in split.test:
                rows.append(dict(track=track,dataset=display_name(ref.dataset_id),fold=fold+1))
    return pd.DataFrame(rows)


def plot_partitions(data: pd.DataFrame) -> list[Page]:
    pages = []
    for track, group in data.groupby("track"):
        matrix = pd.crosstab(group.dataset,group.fold).reindex(columns=range(1,5),fill_value=0)
        matrix.columns = [f"Fold {c}" for c in matrix.columns]
        for number, (start, stop) in enumerate(style.page_slices(len(matrix), style.SIMPLE_PAGE_ROWS), 1):
            page = matrix.iloc[start:stop]
            fig = _heatmap(page,f"{track.upper()}: dataset hold-out assignment",diverging=False,limits=(0,1),label="Held out (1) / training (0)")
            pages.append(Page(f"partitions_{track}_{number}",fig,"Membership of datasets in the four fixed held-out partitions used by experiments 1–3. Each dataset is held out once; the complementary datasets form that partition's adaptation corpus. These are dataset partitions, distinct from outer evaluation folds within a dataset."))
    return pages


def planned_exposure(data: pd.DataFrame, row_cap=10000) -> pd.DataFrame:
    """Illustrate full traversal geometry, explicitly without pretending to measure a run."""
    d = data[data.rows.gt(0)].copy()
    d["chunks"] = np.ceil(d.rows/row_cap).astype(int)
    d["row_share"] = d.rows / d.groupby("track").rows.transform("sum")
    d["full_pass_step_share"] = d.chunks / d.groupby("track").chunks.transform("sum")
    d["one_sample_step_share"] = 1/d.groupby("track").rows.transform("count")
    d["accumulate_step_share"] = d.one_sample_step_share
    d["full_pass_share_ratio"] = d.full_pass_step_share / d.one_sample_step_share
    d["illustrative_row_cap"] = row_cap
    return d


def plot_exposure(data: pd.DataFrame) -> list[Page]:
    pages = []
    for track, group in data.groupby("track"):
        group = group.sort_values("rows",ascending=False)
        for number, (start, stop) in enumerate(style.page_slices(len(group), style.SIMPLE_PAGE_ROWS), 1):
            page = group.iloc[start:stop]
            fig, axes = _subplots(f"{track.upper()}: optimizer-step shares by sampling protocol")
            y = np.arange(len(page))
            for mode, label in (("one_sample", "One sample / accumulate"), ("full_pass", "Full pass")):
                axes[0].scatter(page[f"{mode}_step_share"], y, marker=style.SEED_MARKERS[int(mode == "full_pass")],
                                color=style.SAMPLING_COLORS[mode], label=label, s=style.POINT_SIZE)
            for pos, (_, row) in enumerate(page.iterrows()):
                ratio = row.full_pass_share_ratio
                label = f"{ratio:.3f}×" if round(ratio, 2) == 1 and ratio != 1 else f"{ratio:.2f}×"
                lift = style.EXPOSURE_COLLISION_LIFT if .4 < ratio < .98 else 0
                axes[0].annotate(label,
                    (row.full_pass_step_share, pos),
                    xytext=(style.EXPOSURE_LABEL_PAD, lift), textcoords="offset points",
                    va="center", fontsize=style.ANNOTATION_SIZE,
                    color=style.color("improvement" if row.full_pass_share_ratio > 1 else "deterioration" if row.full_pass_share_ratio < 1 else "reference"))
            axes[0].set_yticks(y, [f"{r.dataset_id} ({r.rows:,.0f})" for r in page.itertuples()])
            axes[0].invert_yaxis()
            axes[0].set_xlabel(f"Optimizer-step share; chunk cap {int(page.illustrative_row_cap.iloc[0]):,} rows")
            maximum = group[[f"{mode}_step_share" for mode in style.SAMPLING_LABELS]].max().max()
            axes[0].set_xlim(0, maximum*style.EXPOSURE_LIMIT_FACTOR)
            axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1), ncol=2)
            pages.append(Page(f"planned_exposure_{track}_{number}",fig,
                f"Illustrative optimizer-step shares using a {int(page.illustrative_row_cap.iloc[0]):,}-row batch cap, all registered processed tables and no skipped updates. "
                "One sample and accumulation share one marker because both assign one update per table visit; full pass assigns one per disjoint chunk. "
                "Dataset labels include row counts; the axis gives the total context-plus-query row cap per chunk. "
                "Multipliers beside full-pass dots give full-pass step share divided by equal-table step share, not the ratio of total update counts. "
                "Green denotes greater and red smaller share; these colors do not imply better or worse performance. "
                "Accumulation consumes all chunks whereas one sample consumes one; equal step shares therefore do not imply equal row exposure. "
                "Actual training uses the partition's training tables, base-specific caps and may stop mid-traversal.",
                {"Step shares and full-pass share ratios": page[["dataset_id", "rows", "chunks", "one_sample_step_share", "full_pass_step_share", "accumulate_step_share", "full_pass_share_ratio"]]}))
    return pages
