"""Second-campaign reports: design tables, training trajectories, endpoint gains, per-table analysis.

Every loader returns an empty frame when evidence is missing; notebooks then report "no evidence
yet" instead of failing, and never treat absence as a zero effect. Paths follow
``src.utils.paths`` (PROJECT for training/results, DATA for manifests); the notebooks run the
same locally after the campaign has been downloaded.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.design import load_design

TRACK_METRIC = {"pd": ("roc_auc", "AUC", 1), "lgd": ("r2", "R²", 1)}
MONITOR_METRIC = {"pd": ("roc_auc", "AUC", 1), "lgd": ("rmse", "RMSE", -1)}
MODEL_LABELS = {"tabpfn-v2": "TabPFN v2", "tabpfn-v2.6": "TabPFN v2.6", "tabpfn-v3": "TabPFN v3",
                "tabicl": "TabICLv2"}
_NAME = re.compile(r"^cpt2_(?P<slug>.+)_a(?P<arm>\d+)_s(?P<part>\d+)_(?P<track>pd|lgd)_"
                   r"(?P<base>tabpfn-v[0-9.]+|tabicl)[^_]*?.*\.trajectory\.csv$")


# ------------------------------------------------------------------------------- design
def experiments(rq: int | None = None) -> dict:
    spec = load_design()["experiments"]
    return {k: v for k, v in spec.items() if rq is None or int(v["rq"]) == int(rq)}


def design_table(rq: int | None = None) -> pd.DataFrame:
    """One row per experiment arm: what changes versus the reference, tasks and run count."""
    from src.utils.design import experiment_arms
    from src.utils.design import load_design as _load
    design = _load()
    rows = []
    for name, spec in experiments(rq).items():
        kind = spec.get("kind", "training")
        if kind in ("inference", "finetune"):
            rows.append(dict(experiment=name, slug=spec["slug"], arm="—", change=spec["question"],
                             tasks="PD, LGD", runs=0))
            continue
        for track in spec["tasks"]:
            for index, arm in enumerate(experiment_arms(name, track, design)):
                models = arm.get("models", design["models"])
                parts = 1 if "single_source" in arm else int(design["partitions"])
                change = ", ".join(f"{k}={v}" for k, v in arm.items() if k != "models") or "reference"
                rows.append(dict(experiment=name, slug=spec["slug"], arm=index, track=track,
                                 change=change, models=len(models), runs=len(models) * parts))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ trajectories
def _training_root(slug: str) -> Path:
    from src.utils.paths import training_dir
    return training_dir(experiment=slug)


def load_trajectories(slugs) -> pd.DataFrame:
    """Long frame of every monitor point of the given experiment slugs (empty if none)."""
    frames = []
    for slug in ([slugs] if isinstance(slugs, str) else slugs):
        try:
            root = _training_root(slug)
        except Exception:  # noqa: BLE001 - unknown or missing layout is missing evidence
            continue
        for path in sorted(root.glob("*/*.trajectory.csv")) if root.is_dir() else []:
            match = _NAME.match(path.name)
            if not match:
                continue
            frame = pd.read_csv(path)
            frame = frame.assign(slug=match["slug"], arm=int(match["arm"]), partition=int(match["part"]),
                                 track=match["track"], model=MODEL_LABELS.get(match["base"], match["base"]),
                                 run=path.name.removesuffix(".trajectory.csv"))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def trajectory_curves(frame: pd.DataFrame, track: str, *, scope: str = "test",
                      x: str = "successful_updates") -> pd.DataFrame:
    """Mean change from update 0 per slug/arm/model, averaged over partitions."""
    if frame.empty:
        return pd.DataFrame()
    frame = frame[(frame.track == track)]
    column = f"{scope}_metric"
    if frame.empty or column not in frame:
        return pd.DataFrame()
    kinds = frame.record_type if "record_type" in frame else pd.Series("trajectory", index=frame.index)
    baseline = frame[(kinds == "trajectory") & (frame.successful_updates == 0)]
    start = baseline.groupby("run")[column].first()
    if x == "successful_updates":
        points = frame[kinds == "trajectory"]
    else:
        # Milestone records sit just past each threshold; label them by the threshold reached.
        from src.utils.design import load_design
        thresholds = sorted(int(v) for v in load_design()["reference"]["monitor"]["processed_rows"])
        points = frame[kinds == "rows_milestone"].copy()
        points[x] = points.processed_rows.map(
            lambda r: max((t for t in thresholds if t <= r), default=np.nan))
        points = points.dropna(subset=[x])
    points = points.assign(change=points[column] - points.run.map(start))
    return points.groupby(["slug", "arm", "model", x])["change"].mean().reset_index()


# --------------------------------------------------------------------- endpoint gains
def _results_frames(slug: str, track: str) -> pd.DataFrame:
    from src.utils.paths import results_dir
    try:
        root = results_dir(experiment=slug) / track.upper()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    # Metric tables are ``*.csv``; ``*.csv.gz`` beside them hold per-row predictions.
    files = sorted(root.rglob("*.csv")) if root.is_dir() else []
    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001 - unreadable shard is reported by the audit, not here
            continue
        if {"model_name", "test_dataset_id", "fold_idx", "status"} <= set(frame.columns):
            frames.append(frame[frame.status == "OK"])
    if not frames:
        return pd.DataFrame()
    # File names carry their timestamp, so the newest score of a model, table and fold is last.
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates(["model_name", "test_dataset_id", "fold_idx"], keep="last"))


#: Outer folds of the five-fold evaluation (config/eval.yaml cv.n_folds).
N_FOLDS = 5


def complete(gains: pd.DataFrame) -> pd.Series:
    """Rows whose comparison covers all five folds (every row when not recorded)."""
    return gains["complete"] if "complete" in gains else pd.Series(True, index=gains.index)


def _paired(slug: str, track: str, metrics) -> pd.DataFrame:
    """Fold-level rows of every trained model of ``slug`` joined with its original model on the
    same table and outer fold (identical context and test rows), for the requested metrics."""
    frames = [_results_frames(slug, track)]
    if slug != "reference":
        frames.append(_results_frames("reference", track))
    frames = [f for f in frames if not f.empty]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    metrics = [m for m in metrics if m in frame]
    if frame.empty or not metrics:
        return pd.DataFrame()
    frame["base"] = frame.model_name.str.extract(r"(tabpfn-v[0-9.]+|tabicl)")[0]
    original = frame[frame.model_source.astype(str).str.endswith("untuned")]
    trained = frame[frame.model_source.astype(str).str.endswith("trained")
                    & frame.model_name.str.contains(f"cpt2_{slug}_a", regex=False)].copy()
    if trained.empty or original.empty:
        return pd.DataFrame()
    trained["arm"] = trained.model_name.str.extract(r"_a(\d+)_")[0].astype(int)
    merged = trained.merge(original[["base", "test_dataset_id", "fold_idx", *metrics]],
                           on=["base", "test_dataset_id", "fold_idx"], suffixes=("", "_original"))
    merged["model"] = merged.base.map(MODEL_LABELS)
    merged["credit"] = ~merged.test_dataset_id.astype(str).str.startswith("openml_")
    return merged


def endpoint_gains(slug: str, track: str) -> pd.DataFrame:
    """Per table, model and arm: CPT minus original, averaged over the paired folds.

    ``folds`` counts the folds scored for both models; ``complete`` marks all five. Summaries
    use complete pairs only, so a failed fold never silently changes the comparison."""
    metric, _, _ = TRACK_METRIC[track]
    merged = _paired(slug, track, [metric])
    if merged.empty:
        return pd.DataFrame()
    merged["gain"] = merged[metric] - merged[metric + "_original"]
    out = (merged.groupby(["arm", "model", "test_dataset_id", "credit"])
           .agg(gain=("gain", "mean"), original=(metric + "_original", "mean"),
                folds=("fold_idx", "nunique")).reset_index())
    out["complete"] = out.folds == N_FOLDS
    return out


#: How a change of each reported metric becomes an improvement (positive = better):
#: +1 higher is better, -1 lower is better, ("target", v) closer to v is better.
METRIC_DIRECTION = {
    "roc_auc": 1, "log_loss": -1, "brier_score": -1, "ece": -1,
    "calibration_slope": ("target", 1.0), "calibration_in_the_large": ("target", 0.0),
    "r2": 1, "nrmse": -1, "rmse": -1, "mae": -1, "crps": -1, "nll": -1,
    "bias": ("target", 0.0), "prediction_std_ratio": ("target", 1.0),
}
METRIC_LABELS = {
    "roc_auc": "AUC", "log_loss": "Log loss", "brier_score": "Brier", "ece": "ECE",
    "calibration_slope": "|slope − 1|", "calibration_in_the_large": "|CITL|",
    "r2": "R²", "nrmse": "nRMSE", "rmse": "RMSE", "mae": "MAE", "crps": "CRPS", "nll": "NLL",
    "bias": "|bias|", "prediction_std_ratio": "|SD ratio − 1|",
}
#: Discrimination first, then probability or distribution quality (RESEARCH_BRIEF metrics).
PROFILE_METRICS = {"pd": ("roc_auc", "log_loss", "brier_score", "ece", "calibration_slope",
                          "calibration_in_the_large"),
                   "lgd": ("r2", "nrmse", "mae", "crps", "bias", "prediction_std_ratio")}


def improvement(metric: str, new, old):
    """Improvement of ``new`` over ``old`` for one metric (positive = better)."""
    direction = METRIC_DIRECTION[metric]
    new, old = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    if isinstance(direction, tuple):
        return np.abs(old - direction[1]) - np.abs(new - direction[1])
    return direction * (new - old)


def endpoint_metrics(slug: str, track: str, metrics=None) -> pd.DataFrame:
    """Long frame per arm, model, table and metric: CPT and original means over the paired folds,
    their difference ``change`` and the oriented ``improvement`` (computed per fold, then
    averaged). ``complete`` marks all five folds finite for both models."""
    metrics = list(metrics or PROFILE_METRICS[track])
    merged = _paired(slug, track, metrics)
    if merged.empty:
        return pd.DataFrame()
    rows = []
    for metric in [m for m in metrics if m in merged]:
        sub = merged[["arm", "model", "test_dataset_id", "credit", "fold_idx", metric,
                      metric + "_original"]].dropna()
        if sub.empty:
            continue
        sub = sub.assign(improvement=improvement(metric, sub[metric], sub[metric + "_original"]),
                         change=sub[metric] - sub[metric + "_original"])
        out = (sub.groupby(["arm", "model", "test_dataset_id", "credit"])
               .agg(cpt=(metric, "mean"), original=(metric + "_original", "mean"),
                    change=("change", "mean"), improvement=("improvement", "mean"),
                    folds=("fold_idx", "nunique")).reset_index())
        rows.append(out.assign(metric=metric))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["complete"] = out.folds == N_FOLDS
    return out


def metric_profile(metrics: pd.DataFrame, *, credit: bool = True) -> pd.DataFrame:
    """Per arm, model and metric over complete tables: mean and median improvement and the number
    of tables that improved."""
    if metrics.empty:
        return pd.DataFrame()
    sub = metrics[metrics.complete & (metrics.credit == credit)]
    if sub.empty:
        return pd.DataFrame()
    return (sub.groupby(["arm", "model", "metric"])
            .agg(tables=("improvement", "size"), mean_improvement=("improvement", "mean"),
                 median_improvement=("improvement", "median"),
                 tables_improved=("improvement", lambda g: int((g > 0).sum())),
                 mean_change=("change", "mean")).reset_index())


BASELINE_NAMES = {"catboost": "CatBoost", "xgboost": "XGBoost", "logreg": "Linear", "linreg": "Linear"}


def baseline_comparison(track: str) -> pd.DataFrame:
    """Per model and table (E1 reference, arm 0): the original and the CPT model's five-fold score
    minus tuned CatBoost's on the same folds, and the linear and (when scored) XGBoost baselines'
    scores minus CatBoost's. Positive: better than CatBoost. Only tables where all are scored on
    all five folds."""
    metric, _, _ = TRACK_METRIC[track]
    frame = _results_frames("reference", track)
    if frame.empty or metric not in frame:
        return pd.DataFrame()
    baseline = frame[frame.model_source.astype(str) == "baseline"].copy()
    baseline["name"] = baseline.model_name.map(BASELINE_NAMES)
    baseline = baseline.dropna(subset=["name", metric])
    if baseline.empty:
        return pd.DataFrame()
    wide = baseline.pivot_table(index=["test_dataset_id", "fold_idx"], columns="name",
                                values=metric).reset_index()
    if not {"CatBoost", "Linear"} <= set(wide.columns):
        return pd.DataFrame()
    paired = _paired("reference", track, [metric])
    if paired.empty:
        return pd.DataFrame()
    merged = paired[paired.arm == 0].merge(wide, on=["test_dataset_id", "fold_idx"]).dropna(
        subset=[metric, metric + "_original", "CatBoost", "Linear"])
    if merged.empty:
        return pd.DataFrame()
    if "XGBoost" not in merged:
        merged["XGBoost"] = np.nan
    merged = merged.assign(original_vs_catboost=merged[metric + "_original"] - merged.CatBoost,
                           cpt_vs_catboost=merged[metric] - merged.CatBoost,
                           linear_vs_catboost=merged.Linear - merged.CatBoost,
                           xgboost_vs_catboost=merged.XGBoost - merged.CatBoost)
    out = (merged.groupby(["model", "test_dataset_id", "credit"])
           .agg(original=(metric + "_original", "mean"), cpt=(metric, "mean"),
                catboost=("CatBoost", "mean"), linear=("Linear", "mean"),
                original_vs_catboost=("original_vs_catboost", "mean"),
                cpt_vs_catboost=("cpt_vs_catboost", "mean"),
                linear_vs_catboost=("linear_vs_catboost", "mean"),
                xgboost=("XGBoost", "mean"), xgboost_vs_catboost=("xgboost_vs_catboost", "mean"),
                folds=("fold_idx", "nunique")).reset_index())
    return out[out.folds == N_FOLDS].reset_index(drop=True)


def gain_summary(gains: pd.DataFrame) -> pd.DataFrame:
    """Per arm, model and table group: tables, mean and median gain over complete tables only;
    ``incomplete_tables`` counts the tables left out."""
    if gains.empty:
        return pd.DataFrame()
    incomplete = (gains[~gains.complete].groupby(["arm", "model", "credit"]).size()
                  .rename("incomplete_tables"))
    gains = gains[gains.complete]
    if gains.empty:
        return incomplete.reset_index()
    return (gains.groupby(["arm", "model", "credit"])
            .agg(tables=("gain", "size"), mean_gain=("gain", "mean"), median_gain=("gain", "median"),
                 tables_improved=("gain", lambda g: int((g > 0).sum())))
            .join(incomplete).fillna({"incomplete_tables": 0}).reset_index())


# ------------------------------------------------------------------- per-table analysis
def table_properties(track: str) -> pd.DataFrame:
    """Dataset properties used to explain per-table gains (credit corpus only)."""
    from src.eval.dataset_loader import load_processed_dataset
    from src.train.corpus import build_dataset_pool
    rows = []
    for ref in sorted(build_dataset_pool(track), key=lambda r: r.dataset_id):
        try:
            data = load_processed_dataset(track, ref.dataset_id)
        except Exception:  # noqa: BLE001
            continue
        X, y = data.X, np.asarray(data.y, dtype=float)
        row = dict(test_dataset_id=ref.dataset_id, rows=len(y), features=X.shape[1],
                   categorical_share=len(data.categorical_columns) / max(1, X.shape[1]),
                   missing_share=float(X.isna().to_numpy().mean()))
        if track == "pd":
            row["default_rate"] = float(np.mean(y))
        else:
            row["zero_share"], row["one_share"] = float(np.mean(y <= 0)), float(np.mean(y >= 1))
        rows.append(row)
    return pd.DataFrame(rows)


def gain_correlations(gains: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of per-table gain with each property, per model (credit tables)."""
    if gains.empty or properties.empty:
        return pd.DataFrame()
    merged = gains[gains.credit & complete(gains)].merge(properties, on="test_dataset_id")
    merged["headroom"] = 1 - merged["original"]
    columns = [c for c in properties.columns if c != "test_dataset_id"] + ["headroom", "original"]
    rows = []
    for (arm, model), group in merged.groupby(["arm", "model"]):
        for column in columns:
            if group[column].nunique() > 1 and len(group) >= 4:
                rho = group[["gain", column]].corr(method="spearman").iloc[0, 1]
                rows.append(dict(arm=arm, model=model, property=column, tables=len(group), spearman=rho))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ experiment 0
def experiment0_summary() -> dict | None:
    from src.utils.pilot_checks import receipts
    path = receipts() / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def experiment0_tables(summary: dict | None) -> dict[str, pd.DataFrame]:
    if not summary:
        return {}
    pilots = pd.DataFrame([dict(pilot=k, passed=bool(v.get("passed")), completed=v.get("completed"),
                                problems=len(v.get("problems", [])))
                           for k, v in summary.get("pilots", {}).items()])
    checks = pd.DataFrame([dict(kind=kind, task=t, passed=bool(v.get("passed")))
                           for kind in ("recovery", "capacity", "evaluation")
                           for t, v in summary.get(kind, {}).items()])
    timing = pd.DataFrame([dict(pilot=k, **t) for k, v in summary.get("pilots", {}).items()
                           for t in v.get("timing", [])])
    return {"pilots": pilots, "checks": checks, "timing": timing}


# ----------------------------------------------------------------------------- display
def pages(fig, name: str, caption: str) -> list:
    """Wrap one figure for ``campaign.show``; an absent figure yields no page."""
    from src.visualize.campaign import Page
    return [] if fig is None else [Page(figure=fig, name=name, caption=caption)]


def corpus_similarity(track: str, properties: pd.DataFrame | None = None) -> pd.DataFrame:
    """Distance of each held-out table to its partition's training tables in property space.

    Properties are log-rows, log-features, categorical/missing shares and the target summary,
    standardised over the track's tables; ``nearest`` is the distance to the closest training
    table, ``mean`` the average distance. Small values mean the CPT corpus contained similar tables.
    """
    from src.train.config import load_train_config
    from src.train.corpus import split_from_cfg
    from src.utils.experiment import apply_split_index
    properties = table_properties(track) if properties is None else properties
    if properties.empty:
        return pd.DataFrame()
    features = properties.set_index("test_dataset_id").copy()
    features["rows"] = np.log10(features["rows"])
    features["features"] = np.log10(features["features"])
    z = (features - features.mean()) / features.std(ddof=0).replace(0, 1)
    from src.utils.design import config_path
    cfg = load_train_config(config_path=str(config_path("reference", track)))
    rows = []
    for partition in range(int(cfg.corpus.n_splits)):
        split = split_from_cfg(apply_split_index(cfg.copy(), partition))
        train = [r.dataset_id for r in split.train if r.dataset_id in z.index]
        for ref in split.test:
            if ref.dataset_id not in z.index or not train:
                continue
            distances = np.sqrt(((z.loc[train] - z.loc[ref.dataset_id]) ** 2).sum(axis=1))
            rows.append(dict(test_dataset_id=ref.dataset_id, partition=partition,
                             nearest=float(distances.min()), mean=float(distances.mean())))
    return pd.DataFrame(rows)


SEEDS = {"reference": {0: 42}, "seed": {0: 43, 1: 44}}   # E1 and the two E2 arms


def seed_noise(track: str) -> pd.DataFrame:
    """Per table and model: the gain at seeds 42 (E1), 43 and 44 (E2), and whether it exceeds seed noise.

    ``seed_range`` is the largest minus the smallest gain over the seeds. ``beyond_noise`` is true
    when all seeds agree in sign and the smallest absolute gain exceeds that range, i.e. the effect
    is larger than what retraining with another seed moves.
    """
    frames = []
    for slug, seeds in SEEDS.items():
        gains = endpoint_gains(slug, track)
        if gains.empty:
            return pd.DataFrame()
        gains = gains[gains.arm.isin(list(seeds)) & complete(gains)].copy()
        gains["seed"] = gains.arm.map(seeds)
        frames.append(gains)
    wide = pd.concat(frames).pivot_table(index=["model", "test_dataset_id", "credit"], columns="seed",
                                         values="gain").dropna()
    if wide.empty:
        return pd.DataFrame()
    wide.columns = [f"gain_seed{c}" for c in wide.columns]
    values = wide.to_numpy()
    wide["mean_gain"] = values.mean(axis=1)
    wide["seed_range"] = values.max(axis=1) - values.min(axis=1)
    same_sign = (np.sign(values) == np.sign(values[:, :1])).all(axis=1) & (values[:, 0] != 0)
    wide["beyond_noise"] = same_sign & (np.abs(values).min(axis=1) > wide.seed_range.to_numpy())
    return wide.reset_index()


def domain_control_gains(track: str) -> pd.DataFrame:
    """E3 versus E1, per held-out credit table and model: gain of credit CPT, of CPT on the matched
    non-credit twins, and their difference. Positive: the credit data helped more than comparable
    non-credit data."""
    credit, control = endpoint_gains("reference", track), endpoint_gains("domain_control", track)
    if credit.empty or control.empty:
        return pd.DataFrame()
    credit, control = credit[credit.credit & complete(credit)], control[control.credit & complete(control)]
    merged = credit.merge(control[["model", "test_dataset_id", "gain"]],
                                         on=["model", "test_dataset_id"], suffixes=("_credit", "_noncredit"))
    merged["credit_minus_noncredit"] = merged.gain_credit - merged.gain_noncredit
    return merged


#: Draws of the hierarchical bootstrap behind every reported interval.
BOOTSTRAP_DRAWS = 2000


def bootstrap_interval(frame: pd.DataFrame, value: str, *, table: str = "test_dataset_id",
                       fold: str = "fold_idx", draws: int = BOOTSTRAP_DRAWS, level: float = 0.95,
                       seed: int = 0) -> dict:
    """Equal-table mean of ``value`` with a hierarchical-bootstrap percentile interval.

    The table is the unit of generalisation and its folds are repeated measurements of it, so a
    draw resamples the tables with replacement and then the folds within every drawn table. Only
    tables with all five folds enter. Few tables give a wide interval, as they should."""
    empty = dict(tables=0, mean=np.nan, low=np.nan, high=np.nan)
    if frame.empty or value not in frame:
        return empty
    matrix = frame.pivot_table(index=table, columns=fold, values=value)
    matrix = matrix[matrix.notna().sum(axis=1) == N_FOLDS].to_numpy(dtype=float)
    if not len(matrix):
        return empty
    rng = np.random.default_rng(seed)
    tables, folds = matrix.shape
    drawn_tables = rng.integers(tables, size=(draws, tables))
    drawn_folds = rng.integers(folds, size=(draws, tables, folds))
    means = matrix[drawn_tables[:, :, None], drawn_folds].mean(axis=(1, 2))
    tail = (1 - level) / 2
    return dict(tables=int(tables), mean=float(matrix.mean()), low=float(np.quantile(means, tail)),
                high=float(np.quantile(means, 1 - tail)))


def gain_intervals(slug: str, track: str) -> pd.DataFrame:
    """Per arm, model and table group (credit, retention): equal-table mean five-fold gain (CPT
    minus original) with its 95% hierarchical-bootstrap interval, and whether it excludes zero."""
    metric, _, _ = TRACK_METRIC[track]
    merged = _paired(slug, track, [metric])
    if merged.empty:
        return pd.DataFrame()
    merged["gain"] = merged[metric] - merged[metric + "_original"]
    rows = [dict(arm=arm, model=model, credit=credit, **bootstrap_interval(group, "gain"))
            for (arm, model, credit), group in merged.groupby(["arm", "model", "credit"])]
    out = pd.DataFrame(rows)
    out = out[out.tables > 0].reset_index(drop=True)
    out["excludes_zero"] = (out.low > 0) | (out.high < 0)
    return out


def domain_control_intervals(track: str) -> pd.DataFrame:
    """Difference in differences per model on the held-out credit tables: gain of credit CPT (E1)
    minus gain of CPT on the matched non-credit twins (E3), fold by fold on identical rows, with
    its 95% hierarchical-bootstrap interval. Positive: credit data helped more."""
    metric, _, _ = TRACK_METRIC[track]
    credit, control = _paired("reference", track, [metric]), _paired("domain_control", track, [metric])
    if credit.empty or control.empty:
        return pd.DataFrame()
    keys = ["model", "test_dataset_id", "fold_idx"]
    for frame in (credit, control):
        frame["gain"] = frame[metric] - frame[metric + "_original"]
    merged = credit[credit.credit & (credit.arm == 0)][[*keys, "gain"]].merge(
        control[control.credit & (control.arm == 0)][[*keys, "gain"]], on=keys,
        suffixes=("_credit", "_noncredit"))
    if merged.empty:
        return pd.DataFrame()
    merged["difference"] = merged.gain_credit - merged.gain_noncredit
    rows = []
    for model, group in merged.groupby("model"):
        interval = bootstrap_interval(group, "difference")
        rows.append(dict(model=model, tables=interval["tables"],
                         gain_credit=bootstrap_interval(group, "gain_credit")["mean"],
                         gain_noncredit=bootstrap_interval(group, "gain_noncredit")["mean"],
                         credit_minus_noncredit=interval["mean"], low=interval["low"], high=interval["high"]))
    out = pd.DataFrame(rows)
    out = out[out.tables > 0].reset_index(drop=True)
    out["excludes_zero"] = (out.low > 0) | (out.high < 0)
    return out


PRECISION_LABELS = {"bf16": "BF16", "fp16": "FP16 with loss scaling"}


def precision_agreement(summary: dict | None) -> pd.DataFrame:
    """Experiment 0 precision check, one row per task, model, training table and candidate
    precision: update-zero loss in FP32 and in the candidate, and the cosine between the two
    gradients on the identical episode. ``table_rank`` orders the training tables by size."""
    rows = []
    for record in ((summary or {}).get("precision") or {}).values():
        if not isinstance(record, dict) or "tables" not in record:
            continue
        model = _model_label(str(record.get("base", "")))
        for table in record["tables"]:
            for key, label in PRECISION_LABELS.items():
                if f"grad_cosine_{key}" not in table:
                    continue
                rows.append(dict(track=record.get("track"), model=model,
                                 table_rank=int(table["table_index"]) + 1, rows=table.get("rows"),
                                 precision=label, loss_fp32=table.get("loss_fp32"),
                                 loss=table.get(f"loss_{key}"), gradient_cosine=table[f"grad_cosine_{key}"],
                                 gradient_norm_ratio=table.get(f"grad_norm_ratio_{key}")))
    return pd.DataFrame(rows)


def cost_comparison(resources: pd.DataFrame, finetune: pd.DataFrame) -> pd.DataFrame:
    """GPU hours per task and model: CPT with the reference recipe (E1, all partitions, paid once
    for every future table) against fine-tuning on a target table (E4, original start, per table:
    all candidate rates, summed over the five folds)."""
    rows = []
    if not resources.empty:
        cpt = resources[(resources.slug == "reference") & (resources.arm == 0)]
        for (track, model), group in cpt.groupby(["track", "model"]):
            rows.append(dict(track=track, model=_model_label(model), cpt_runs=len(group),
                             cpt_hours_per_run=float(group.hours_without_monitor.mean()),
                             cpt_hours_total=float(group.hours_without_monitor.sum())))
    out = pd.DataFrame(rows)
    if finetune.empty or "fit_seconds" not in finetune or out.empty:
        return out
    tuned = finetune[finetune.finetuned.astype(bool) & (finetune.start == "original")]
    per_table = (tuned.groupby(["track", "model", "dataset_id"]).fit_seconds.sum() / 3600).reset_index()
    per_table["model"] = per_table.model.map(_model_label)
    per_model = per_table.groupby(["track", "model"]).fit_seconds.agg(
        finetune_tables="size", finetune_hours_per_table="mean").reset_index()
    out = out.merge(per_model, on=["track", "model"], how="left")
    out["tables_to_break_even"] = out.cpt_hours_total / out.finetune_hours_per_table
    return out


def task_results(kind: str) -> pd.DataFrame:
    """All finished E4 (``finetune``), E9 (``episode_context``) or E10 (``context``) task CSVs."""
    from src.eval.campaign_tasks import SLUGS
    from src.utils.paths import results_dir
    try:
        folder = results_dir(experiment=SLUGS[kind]) / kind
    except Exception:  # noqa: BLE001 - missing layout is missing evidence
        return pd.DataFrame()
    files = sorted(folder.glob(f"{kind}_*.csv.gz")) if folder.is_dir() else []
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _all_folds(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Rows of the groups (``keys``) scored on all five outer folds; partial groups are dropped."""
    if frame.empty:
        return frame
    folds = frame.groupby(keys).fold.transform("nunique")
    return frame[folds == N_FOLDS]


def finetune_summary(results: pd.DataFrame) -> pd.DataFrame:
    """E4 per model and start (original, CPT): the untuned and the fine-tuned score on identical
    rows and their paired difference, averaged within table first (tables with all five folds)."""
    if results.empty or "finetuned" not in results:
        return pd.DataFrame()
    rows = []
    for track, metric in (("pd", "roc_auc"), ("lgd", "r2")):
        sub = results[results.track == track]
        if sub.empty or metric not in sub:
            continue
        wide = sub.pivot_table(index=["model", "start", "dataset_id", "fold"], columns="finetuned",
                               values=metric).dropna().reset_index()
        wide = wide.rename(columns={False: "untuned", True: "finetuned"})
        wide = _all_folds(wide, ["model", "start", "dataset_id"])
        if wide.empty:
            continue
        wide["finetune_gain"] = wide.finetuned - wide.untuned
        per_table = wide.groupby(["model", "start", "dataset_id"])[
            ["untuned", "finetuned", "finetune_gain"]].mean().reset_index()
        summary = per_table.groupby(["model", "start"]).agg(
            tables=("dataset_id", "size"), untuned=("untuned", "mean"), finetuned=("finetuned", "mean"),
            mean_finetune_gain=("finetune_gain", "mean"), median_finetune_gain=("finetune_gain", "median"),
            tables_improved=("finetune_gain", lambda g: int((g > 0).sum()))).reset_index()
        rows.append(summary.assign(track=track, metric=metric))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def context_gain_curve(results: pd.DataFrame, *, common_up_to: int | None = None) -> pd.DataFrame:
    """E10: CPT minus original at each labelled-context size, per model, averaged within table
    first (tables with all five folds at that size).

    Larger sizes exist only for larger tables, so the table population changes along the curve.
    ``common_up_to`` keeps only tables scored at every size up to that value, so the curve below it
    compares the same tables throughout."""
    if results.empty:
        return pd.DataFrame()
    rows = []
    for track, metric in (("pd", "roc_auc"), ("lgd", "r2")):
        sub = results[results.track == track]
        if sub.empty or metric not in sub:
            continue
        wide = sub.pivot_table(index=["model", "dataset_id", "fold", "context_rows"], columns="start",
                               values=metric).dropna().reset_index()
        wide = _all_folds(wide, ["model", "dataset_id", "context_rows"])
        if common_up_to is not None:
            wide = wide[wide.context_rows <= common_up_to]
            sizes = wide.groupby("model").context_rows.nunique()
            have = wide.groupby(["model", "dataset_id"]).context_rows.nunique().reset_index(name="n")
            have = have[have.n == have.model.map(sizes)]
            wide = wide.merge(have[["model", "dataset_id"]], on=["model", "dataset_id"])
        if wide.empty:
            continue
        wide["gain"] = wide["cpt"] - wide["original"]
        per_table = wide.groupby(["model", "dataset_id", "context_rows"]).gain.mean().reset_index()
        curve = (per_table.groupby(["model", "context_rows"]).gain
                 .agg(mean="mean", median="median", tables="size").reset_index())
        rows.append(curve.assign(track=track, metric=metric))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def episode_context_summary(results: pd.DataFrame) -> pd.DataFrame:
    """E9 at matched inference context: mean AUC per episode-size arm, model and context size."""
    if results.empty or "roc_auc" not in results:
        return pd.DataFrame()
    results = _all_folds(results, ["arm", "model", "context_rows", "dataset_id"])
    per_table = results.groupby(["arm", "model", "context_rows", "dataset_id"]).roc_auc.mean().reset_index()
    return per_table.groupby(["arm", "model", "context_rows"]).roc_auc.agg(["mean", "median", "size"]).reset_index()


# ------------------------------------------------------------------- arm settings and labels
_SLUG_EXPERIMENT = {"reference": "E1", "seed": "E2", "domain_control": "E3", "update_size_anchoring": "E5",
                    "scope": "E6", "exposure": "E7", "table_size": "E8",
                    "episode_size": "E9", "transfer": "E11", "precision": "E12"}


def _rate(value: float) -> str:
    mantissa, exponent = f"{float(value):.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def arm_label(slug: str, settings: dict) -> str:
    """Short reader-facing name of one arm: what differs from the reference recipe."""
    scope = settings.get("trainable", "all")
    if slug == "reference":
        return "Reference"
    if slug == "seed":
        return f"Seed {settings['seed']}"
    if slug == "domain_control":
        return "Non-credit twins"
    if slug == "update_size_anchoring":
        return f"lr {_rate(settings['learning_rate'])}, λ {settings['l2sp_lambda']:g}"
    if slug == "scope":
        names = {"frozen_backbone": "Frozen backbone", "diagscale": "DiagScale", "lora_qv_r16": "LoRA"}
        return f"{names.get(scope, scope)}, lr {_rate(settings['learning_rate'])}"
    if slug == "exposure":
        if settings.get("exposure") == "full_pass":
            return "Full pass"
        return f"{int(settings['successful_updates']):,} updates"
    if slug == "precision":
        return {"bf16": "BF16", "fp32": "FP32", "fp16": "FP16"}[settings["training_precision"]]
    if slug == "table_size":
        return {"small_only": "Small tables only", "large_matched": "Large tables, count-matched"}.get(
            settings.get("corpus"), str(settings.get("corpus")))
    if slug == "episode_size":
        if settings.get("episode_rows") is None:   # the reference size: 10,000 or 26,000 by model
            return "Reference episode size"
        return f"Episodes of {int(settings['episode_rows']):,} rows"
    return f"{slug} arm"


def arm_settings(slug: str, track: str) -> pd.DataFrame:
    """One row per arm of ``slug`` for ``track``: the settings that differ from the reference
    (learning rate, L2-SP, scope, corpus, episode rows, sampling, exposure, seed) and a label."""
    from src.utils.design import experiment_arms, experiment_by_slug
    design = load_design()
    ref = design["reference"]
    name, spec = experiment_by_slug(slug, design)
    if track not in spec.get("tasks", []):
        return pd.DataFrame()
    rows = []
    for index, arm in enumerate(experiment_arms(name, track, design)):
        settings = dict(learning_rate=float(arm.get("learning_rate", ref["learning_rate"])),
                        l2sp_lambda=float(arm.get("l2sp_lambda", ref["l2sp_lambda"])),
                        trainable=arm.get("trainable", ref["trainable"]),
                        corpus=arm.get("corpus", ref["corpus"]),
                        episode_rows=arm.get("episode_rows"),
                        pd_class_sampling=arm.get("pd_class_sampling", ref["pd_class_sampling"]),
                        exposure=arm.get("exposure", ref["exposure"]),
                        seed=int(arm.get("seed", ref["seed"])),
                        successful_updates=int(arm.get("successful_updates", ref["successful_updates"])),
                        training_precision=arm.get("training_precision", ref["training_precision"]),
                        models=",".join(arm.get("models", design["models"])))
        rows.append(dict(slug=slug, experiment=name, arm=index, track=track, **settings,
                         label=arm_label(slug, settings)))
    return pd.DataFrame(rows)


def arms_overview(slugs, track: str) -> pd.DataFrame:
    """Per experiment arm and model: equal-table mean gain on held-out credit tables and on the
    non-credit retention panel (complete five-fold tables only), with the arm's settings."""
    rows = []
    for slug in slugs:
        gains = endpoint_gains(slug, track)
        settings = arm_settings(slug, track)
        if gains.empty or settings.empty:
            continue
        gains = gains[gains.complete]
        for (arm, model), group in gains.groupby(["arm", "model"]):
            credit, retention = group[group.credit].gain, group[~group.credit].gain
            rows.append(dict(slug=slug, arm=arm, model=model, track=track,
                             credit_mean=credit.mean() if len(credit) else np.nan,
                             credit_median=credit.median() if len(credit) else np.nan,
                             credit_tables=len(credit), credit_improved=int((credit > 0).sum()),
                             retention_mean=retention.mean() if len(retention) else np.nan,
                             retention_tables=len(retention),
                             retention_improved=int((retention > 0).sum())))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    settings = pd.concat([arm_settings(s, track) for s in slugs], ignore_index=True)
    return out.merge(settings.drop(columns="track"), on=["slug", "arm"], how="left")


def arm_table_gains(slugs, track: str) -> pd.DataFrame:
    """Per-table gains of several experiments in one frame, labelled by arm (complete tables)."""
    frames = []
    for slug in slugs:
        gains = endpoint_gains(slug, track)
        settings = arm_settings(slug, track)
        if gains.empty or settings.empty:
            continue
        frames.append(gains[gains.complete].merge(settings[["arm", "label", "experiment"]], on="arm")
                      .assign(slug=slug))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------- monitor (training) curves
#: Per-table monitor score used for curves: AUC for PD; R² for LGD, so a monitor change reads on
#: the same scale as the five-fold result (the primary LGD monitor metric is RMSE).
MONITOR_SCORE = {"pd": "roc_auc", "lgd": "r2"}
MONITOR_SCOPES = ("test", "train", "ood")


def _monitor_points(frame: pd.DataFrame, x: str) -> pd.DataFrame:
    kinds = frame.record_type if "record_type" in frame else pd.Series("trajectory", index=frame.index)
    if x == "successful_updates":
        return frame[kinds == "trajectory"]
    thresholds = sorted(int(v) for v in load_design()["reference"]["monitor"]["processed_rows"])
    points = frame[kinds == "rows_milestone"].copy()
    points[x] = points.processed_rows.map(lambda r: max((t for t in thresholds if t <= r), default=np.nan))
    return points.dropna(subset=[x])


def monitor_table_changes(frame: pd.DataFrame, track: str, *, x: str = "successful_updates",
                          scopes=MONITOR_SCOPES) -> pd.DataFrame:
    """Long frame: per run, monitored table (scope ``test`` held-out credit, ``train`` the run's own
    training tables, ``ood`` the retention panel) and monitor point, the score and its change
    from the same run's update-0 monitor of that table."""
    if frame.empty or "track" not in frame:
        return pd.DataFrame()
    frame = frame[frame.track == track]
    metric = MONITOR_SCORE[track]
    pattern = re.compile(rf"^score__({'|'.join(scopes)})__(.+)__{re.escape(metric)}$")
    columns = {c: pattern.match(c) for c in frame.columns}
    columns = {c: m for c, m in columns.items() if m}
    if frame.empty or not columns:
        return pd.DataFrame()
    keys = ["run", "slug", "arm", "partition", "track", "model"]
    kinds = frame.record_type if "record_type" in frame else pd.Series("trajectory", index=frame.index)
    start = frame[(kinds == "trajectory") & (frame.successful_updates == 0)]
    points = _monitor_points(frame, x)
    if points.empty or start.empty:
        return pd.DataFrame()
    long = points[keys + [x] + list(columns)].melt(id_vars=keys + [x], var_name="column", value_name="score")
    base = start[["run"] + list(columns)].melt(id_vars="run", var_name="column", value_name="start")
    long = long.merge(base.drop_duplicates(["run", "column"]), on=["run", "column"])
    long["scope"] = long.column.map(lambda c: columns[c].group(1))
    long["table"] = long.column.map(lambda c: columns[c].group(2))
    long["change"] = long.score - long.start
    return long.drop(columns="column").dropna(subset=["change"])


def monitor_curves(changes: pd.DataFrame, *, x: str = "successful_updates") -> pd.DataFrame:
    """Equal-table mean change per slug, arm, model, scope and monitor point: each table is first
    averaged over the partitions that monitored it, so every table counts once."""
    if changes.empty:
        return pd.DataFrame()
    per_table = (changes.groupby(["track", "slug", "arm", "model", "scope", "table", x])
                 .change.mean().reset_index())
    return (per_table.groupby(["track", "slug", "arm", "model", "scope", x])
            .change.agg(change="mean", tables="size").reset_index())


def seed_band(changes: pd.DataFrame, *, scope: str = "test", x: str = "successful_updates") -> pd.DataFrame:
    """Range over the reference seeds (E1 seed 42, E2 seeds 43 and 44) of the equal-table mean
    monitor change, per model and monitor point; only points that all seeds reached."""
    if changes.empty:
        return pd.DataFrame()
    sub = changes[(changes.scope == scope) & changes.slug.isin(list(SEEDS))].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["seed"] = [SEEDS[s].get(a) for s, a in zip(sub.slug, sub.arm)]
    curves = (sub.groupby(["track", "model", "seed", "table", x]).change.mean().reset_index()
              .groupby(["track", "model", "seed", x]).change.mean().reset_index())
    band = curves.groupby(["track", "model", x]).change.agg(low="min", high="max", seeds="size").reset_index()
    return band[band.seeds == 3].reset_index(drop=True)


def drift_curves(frame: pd.DataFrame, track: str) -> pd.DataFrame:
    """Mean relative weight change ‖w − w0‖/‖w0‖ over partitions per slug, arm, model and update.
    Only full-weight runs are comparable (adapters record a different movement measure)."""
    if frame.empty or "weight_drift" not in frame or "track" not in frame:
        return pd.DataFrame()
    points = _monitor_points(frame[frame.track == track], "successful_updates")
    points = points[points.successful_updates > 0].dropna(subset=["weight_drift"])
    if points.empty:
        return pd.DataFrame()
    return (points.groupby(["track", "slug", "arm", "model", "successful_updates"])
            .weight_drift.agg(change="mean", tables="size").reset_index().assign(scope="drift"))


def run_resources(frame: pd.DataFrame) -> pd.DataFrame:
    """Per training run, from its last monitor record: updates, rows processed, wall-clock hours
    (all, and without monitoring), peak reserved GPU memory and final relative weight change."""
    if frame.empty or "elapsed_sec" not in frame:
        return pd.DataFrame()
    rows = []
    for run, group in frame.groupby("run"):
        group = group.sort_values(["successful_updates", "processed_rows"])
        last = group.iloc[-1]
        monitor = group.get("monitor_seconds", pd.Series(0.0, index=group.index)).fillna(0).sum()
        peak = group.get("cuda_peak_reserved_bytes", pd.Series(np.nan, index=group.index)).max()
        rows.append(dict(run=run, slug=last.slug, arm=last.arm, track=last.track, model=last.model,
                         partition=last.partition, updates=int(last.successful_updates),
                         rows_processed=int(last.processed_rows),
                         hours=float(last.elapsed_sec) / 3600,
                         hours_without_monitor=max(float(last.elapsed_sec) - float(monitor), 0) / 3600,
                         peak_reserved_gb=float(peak) / 1e9 if pd.notna(peak) else np.nan,
                         final_weight_change=float(last.get("weight_drift", np.nan))))
    return pd.DataFrame(rows)


def resource_summary(resources: pd.DataFrame) -> pd.DataFrame:
    """Per slug, arm, track and model: runs, mean and total hours, and the largest memory peak."""
    if resources.empty:
        return pd.DataFrame()
    return (resources.groupby(["slug", "arm", "track", "model"])
            .agg(runs=("run", "size"), mean_hours=("hours", "mean"), total_hours=("hours", "sum"),
                 mean_hours_without_monitor=("hours_without_monitor", "mean"),
                 rows_processed=("rows_processed", "mean"),
                 peak_reserved_gb=("peak_reserved_gb", "max"),
                 final_weight_change=("final_weight_change", "mean")).reset_index())


# ------------------------------------------------------------------ evaluation-only tasks
def _model_label(value) -> str:
    from src.visualize.style import model_label
    return MODEL_LABELS.get(value, model_label(str(value)))


def finetune_table(results: pd.DataFrame) -> pd.DataFrame:
    """E4 per track, model and table (all five folds): the original model without fine-tuning,
    CPT without fine-tuning, fine-tuning from the original and fine-tuning from CPT, each as a
    mean over the paired folds, and each condition's change relative to the original model."""
    if results.empty or "finetuned" not in results:
        return pd.DataFrame()
    rows = []
    for track, metric in (("pd", "roc_auc"), ("lgd", "r2")):
        sub = results[results.track == track]
        if sub.empty or metric not in sub:
            continue
        sub = sub.assign(condition=[("cpt" if s == "cpt" else "original") + ("_finetune" if f else "")
                                    for s, f in zip(sub.start, sub.finetuned.astype(bool))])
        wide = sub.pivot_table(index=["model", "dataset_id", "fold"], columns="condition",
                               values=metric).reset_index()
        needed = ["original", "cpt", "original_finetune", "cpt_finetune"]
        if not set(needed) <= set(wide.columns):
            continue
        wide = _all_folds(wide.dropna(subset=needed), ["model", "dataset_id"])
        if wide.empty:
            continue
        table = wide.groupby(["model", "dataset_id"])[needed].mean().reset_index()
        table = table.assign(track=track, metric=metric, model=table.model.map(_model_label),
                             cpt_change=table.cpt - table.original,
                             finetune_change=table.original_finetune - table.original,
                             cpt_finetune_change=table.cpt_finetune - table.original)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def context_table(results: pd.DataFrame) -> pd.DataFrame:
    """E10 per track, model, table and labelled-context size (all five folds): original and CPT
    means and the paired gain. ``full`` marks the size that used the whole available pool
    (capped at the model's context limit) rather than a declared grid size."""
    if results.empty or "context_rows" not in results:
        return pd.DataFrame()
    grid = set(int(v) for v in load_design()["experiments"]["E10"]["context_rows"])
    rows = []
    for track, metric in (("pd", "roc_auc"), ("lgd", "r2")):
        sub = results[results.track == track]
        if sub.empty or metric not in sub:
            continue
        wide = sub.pivot_table(index=["model", "dataset_id", "fold", "context_rows"], columns="start",
                               values=metric).reset_index()
        if not {"original", "cpt"} <= set(wide.columns):
            continue
        wide = _all_folds(wide.dropna(subset=["original", "cpt"]), ["model", "dataset_id", "context_rows"])
        if wide.empty:
            continue
        wide["gain"] = wide.cpt - wide.original
        table = (wide.groupby(["model", "dataset_id", "context_rows"])[["original", "cpt", "gain"]]
                 .mean().reset_index())
        table = table.assign(track=track, metric=metric, model=table.model.map(_model_label),
                             full=~table.context_rows.astype(int).isin(grid))
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def context_cohort_curve(table: pd.DataFrame) -> pd.DataFrame:
    """Equal-table means of the E10 table per track, model and grid size. ``fixed_cohort`` is true
    up to the largest size every table of that model reached, so the mean there compares the same
    tables throughout; beyond it the population shrinks with the size."""
    if table.empty:
        return pd.DataFrame()
    grid = table[~table.full]
    curve = (grid.groupby(["track", "model", "context_rows"])
             .agg(original=("original", "mean"), cpt=("cpt", "mean"), gain=("gain", "mean"),
                  median_gain=("gain", "median"), tables=("gain", "size")).reset_index())
    total = grid.groupby(["track", "model"]).dataset_id.nunique().rename("all_tables").reset_index()
    curve = curve.merge(total, on=["track", "model"])
    curve["fixed_cohort"] = curve.tables == curve.all_tables
    return curve


def episode_context_gains(episode: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """E9 at matched inference context: each episode-size model minus the original model scored
    by E10 with the same context rows of the same fold, per arm, model and table (five folds)."""
    if episode.empty or context.empty or "roc_auc" not in episode or "roc_auc" not in context:
        return pd.DataFrame()
    original = context[(context.track == "pd") & (context.start == "original")][
        ["model", "dataset_id", "fold", "context_rows", "available_context", "roc_auc"]]
    cpt = episode[episode.start == "cpt"]
    merged = cpt.merge(original, on=["model", "dataset_id", "fold", "context_rows"],
                       suffixes=("", "_original"))
    if merged.empty:
        return pd.DataFrame()
    merged["gain"] = merged.roc_auc - merged.roc_auc_original
    merged = _all_folds(merged, ["arm", "model", "dataset_id", "context_rows"])
    if merged.empty:
        return pd.DataFrame()
    grid = set(int(v) for v in load_design()["experiments"]["E9"]["matched_context_evaluation"][:-1])
    merged["context"] = [str(int(c)) if int(c) in grid else "full" for c in merged.context_rows]
    out = (merged.groupby(["arm", "model", "dataset_id", "context"])
           .agg(gain=("gain", "mean"), context_rows=("context_rows", "mean")).reset_index())
    settings = arm_settings("episode_size", "pd")[["arm", "episode_rows"]]
    out = out.merge(settings, on="arm")
    # The reference-size arm has no declared episode size: it is each model's reference size.
    reference = load_design()["reference"]["episode_rows"]
    out["episode_rows"] = out.episode_rows.astype(float).fillna(out.model.map(reference).astype(float))
    out["model"] = out.model.map(_model_label)
    return out


# ------------------------------------------------------------------------ E11 transfer
def transfer_sources(track: str) -> dict[int, str]:
    """Single-source arm index -> source table id (tables of the task sorted by id)."""
    try:
        from src.train.corpus import build_dataset_pool
        return dict(enumerate(sorted(r.dataset_id for r in build_dataset_pool(track))))
    except Exception:  # noqa: BLE001 - the registry may be absent off-cluster
        return {}


def transfer_long(frame: pd.DataFrame, track: str) -> pd.DataFrame:
    """E11: per model, source arm and target table, the monitor change from the first to the last
    monitor point (AUC for PD, R² for LGD; one fixed split per target, not five-fold)."""
    changes = monitor_table_changes(frame[frame.slug == "transfer"] if not frame.empty else frame,
                                    track, scopes=("test", "ood"))
    if changes.empty:
        return pd.DataFrame()
    last = changes.groupby("run").successful_updates.transform("max")
    end = changes[changes.successful_updates == last]
    sources = transfer_sources(track)
    end = end.assign(source=end.arm.map(lambda a: sources.get(int(a), f"source {a}")))
    return end[["model", "arm", "source", "scope", "table", "successful_updates", "change"]].rename(
        columns={"table": "target"}).reset_index(drop=True)


def transfer_matrix(frame: pd.DataFrame, track: str, model: str) -> pd.DataFrame:
    """E11: source (rows) × held-out credit target (columns) monitor change at the end of CPT."""
    long = transfer_long(frame, track)
    if long.empty:
        return pd.DataFrame()
    sub = long[(long.model == model) & (long.scope == "test")]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="source", columns="target", values="change").sort_index()


# ------------------------------------------------------------------------- notebook text
def evidence_note(label: str, frame) -> str:
    """One sentence for the notebook report: what was found, or that nothing exists yet."""
    if frame is None or (hasattr(frame, "empty") and frame.empty):
        return f"No results yet for {label}; the figure is drawn once they exist."
    return f"{label}: {len(frame)} rows of evidence."


def monitor_changes(frame: pd.DataFrame, *, x: str = "successful_updates", scopes=MONITOR_SCOPES) -> pd.DataFrame:
    """``monitor_table_changes`` for both tasks in one frame (empty when nothing was monitored)."""
    parts = [monitor_table_changes(frame, track, x=x, scopes=scopes) for track in ("pd", "lgd")]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def curve_endpoints(curves: pd.DataFrame, *, x: str = "successful_updates") -> pd.DataFrame:
    """The last monitor point of every curve (per track, slug, arm, model and scope)."""
    if curves.empty:
        return pd.DataFrame()
    keys = [k for k in ("track", "slug", "arm", "model", "scope") if k in curves]
    return curves.loc[curves.groupby(keys)[x].idxmax()].reset_index(drop=True)


def select(frame: pd.DataFrame, **conditions) -> pd.DataFrame:
    """Rows matching every ``column=value`` (a list or tuple matches any of its values); an empty
    or column-less frame stays empty, so notebooks can filter evidence that may not exist yet."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        if column not in frame:
            return pd.DataFrame()
        mask &= frame[column].isin(value) if isinstance(value, (list, tuple, set)) else frame[column].eq(value)
    return frame[mask]


def join(left: pd.DataFrame, right: pd.DataFrame, on: list[str], how: str = "left") -> pd.DataFrame:
    """``left.merge(right)`` that tolerates missing evidence: an empty or key-less ``right`` leaves
    ``left`` unchanged, and an empty ``left`` stays empty."""
    if left is None or left.empty:
        return pd.DataFrame()
    if right is None or right.empty or not set(on) <= set(right.columns) or not set(on) <= set(left.columns):
        return left
    return left.merge(right, on=on, how=how)
