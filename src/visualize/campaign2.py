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

from src.utils.design import experiment_by_slug, load_design

TRACK_METRIC = {"pd": ("roc_auc", "AUC", 1), "lgd": ("r2", "R²", 1)}
MONITOR_METRIC = {"pd": ("roc_auc", "AUC", 1), "lgd": ("rmse", "RMSE", -1)}
MODEL_LABELS = {"tabpfn-v2": "TabPFN v2", "tabpfn-v2.6": "TabPFN v2.6", "tabpfn-v3": "TabPFN v3",
                "tabicl": "TabICLv2"}
_NAME = re.compile(r"^cpt2_(?P<slug>.+)_a(?P<arm>\d+)_s(?P<part>\d+)_(?P<track>pd|lgd)_"
                   r"(?P<base>tabpfn-v[0-9.]+|tabicl)[^_]*?.*\.trajectory\.csv$")


# ------------------------------------------------------------------------- table scope
#: Which evaluated tables every loader returns: ``all``, or ``public`` (the proprietary tables
#: removed). The appendix notebooks switch to ``public`` so that every result can be recomputed
#: on data a reader can obtain; training corpora are unchanged (they still contained the
#: proprietary tables), so this restricts evaluation, not training.
TABLE_SCOPES = ("all", "public")
_SCOPE = {"tables": "all"}


def use_tables(scope: str) -> None:
    """Set the table scope of every loader for the rest of the session (``all`` or ``public``)."""
    if scope not in TABLE_SCOPES:
        raise ValueError(f"table scope must be one of {TABLE_SCOPES}, got {scope!r}")
    _SCOPE["tables"] = scope


def table_scope() -> str:
    return _SCOPE["tables"]


def in_scope(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """``frame`` without the rows of proprietary tables when the scope is ``public`` (loaders apply
    it themselves; notebooks use it for frames that must keep every table internally, such as
    table properties, whose distances still refer to all training tables)."""
    if _SCOPE["tables"] == "all" or frame.empty or column not in frame:
        return frame
    from src.data import dataset_names
    from src.data.dataset_names import is_proprietary
    if not dataset_names._PROPRIETARY:
        from src.data.retention import panel_config
        public = dataset_names.PUBLIC_CREDIT_TABLES | {r["id"] for r in panel_config("research")}
        return frame[frame[column].isin(public)]
    return frame[~frame[column].map(is_proprietary).astype(bool)]


def scope_counts(track: str) -> pd.DataFrame:
    """Held-out credit tables of a task in scope and left out by it (for the appendix)."""
    from src.data.dataset_names import PUBLIC_CREDIT_TABLES
    from src.data.preprocessing import DATASET_METADATA
    ids = {i for i, meta in DATASET_METADATA.items() if meta["track"] == track}
    proprietary = len(ids - PUBLIC_CREDIT_TABLES)
    kept = len(ids) - (proprietary if _SCOPE["tables"] == "public" else 0)
    return pd.DataFrame([dict(track=track, scope=_SCOPE["tables"], credit_tables=len(ids),
                              proprietary_tables=proprietary, tables_in_scope=kept)])


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


# --------------------------------------------------------------------- endpoint gains
def _results_frames(slug: str, track: str) -> pd.DataFrame:
    """Five-fold metric rows of one experiment and task, restricted to the table scope."""
    return in_scope(_read_results_frames(slug, track), "test_dataset_id")


def _read_results_frames(slug: str, track: str) -> pd.DataFrame:
    from src.utils.experiment import file_digest
    from src.utils.paths import results_dir
    try:
        root = results_dir(experiment=slug) / track.upper()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    # Metric tables are ``*.csv``; ``*.csv.gz`` beside them hold per-row predictions.
    # The publication timestamp follows the run name, which can differ across retries.
    def published(path):
        match = re.search(r"_(\d{8}_\d{6})(?:__|\.csv$)", path.name)
        return (match.group(1) if match else "", path)
    files = sorted(root.rglob("*.csv"), key=published) if root.is_dir() else []
    frames = []
    for path in files:
        marker = path.with_suffix(".csv.evaluation.json")
        if not marker.is_file():
            continue  # incomplete publication
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        if "__metrics_sha256__" not in receipt:
            continue  # historical receipts predate verified publication
        if receipt["__metrics_sha256__"] != file_digest(path, fresh=True):
            raise ValueError(f"Benchmark metric checksum does not match {marker.name}")
        frame = pd.read_csv(path)
        if not {"model_name", "test_dataset_id", "fold_idx", "status"} <= set(frame.columns):
            raise ValueError(f"Missing benchmark columns in {path.name}")
        frame["_evaluation_key"] = frame.test_dataset_id.map(receipt)
        if not frame["_evaluation_key"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Missing dataset evaluation key in {marker.name}")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    # File names carry their timestamp, so the newest row of a model, table and fold is last. The
    # newest row decides: a fold whose latest scoring failed is missing, not silently replaced by
    # an older success computed by other code.
    frame = pd.concat(frames, ignore_index=True)
    keys = ["model_name", "test_dataset_id"]
    current = frame.groupby(keys)["_evaluation_key"].transform("last")
    newest = (frame[frame["_evaluation_key"] == current].drop(columns="_evaluation_key")
              .drop_duplicates([*keys, "fold_idx"], keep="last"))
    return newest[(newest.status == "OK") & newest.fold_idx.isin(range(N_FOLDS))].reset_index(drop=True)


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
    merged = merged[merged.fold_idx.isin(range(N_FOLDS))].copy()
    merged["gain"] = merged[metric] - merged[metric + "_original"]
    valid = np.isfinite(merged[metric]) & np.isfinite(merged[metric + "_original"])
    merged.loc[~valid, [metric, metric + "_original", "gain", "fold_idx"]] = np.nan
    out = (merged.groupby(["arm", "model", "test_dataset_id", "credit"])
           .agg(gain=("gain", "mean"), original=(metric + "_original", "mean"),
                folds=("fold_idx", "nunique")).reset_index())
    out["complete"] = out.folds == N_FOLDS
    return out


#: How a change of each reported metric becomes an improvement (positive = better):
#: +1 higher is better, -1 lower is better, ("target", v) closer to v is better.
METRIC_DIRECTION = {
    "roc_auc": 1, "log_loss": -1, "brier_score": -1, "ece": -1,
    "ks": 1, "h_measure": 1, "emp": 1, "gini": 1, "pr_auc": 1,
    "spearman_r": 1, "kendall_tau": 1,
    "calibration_slope": ("target", 1.0), "calibration_in_the_large": ("target", 0.0),
    "r2": 1, "nrmse": -1, "rmse": -1, "mae": -1, "crps": -1, "nll": -1,
    "bias": ("target", 0.0), "prediction_std_ratio": ("target", 1.0),
}
METRIC_LABELS = {
    "roc_auc": "AUC", "log_loss": "Log loss", "brier_score": "Brier", "ece": "ECE",
    "ks": "KS", "h_measure": "H-measure", "emp": "EMP", "gini": "Gini", "pr_auc": "PR-AUC",
    "spearman_r": "Spearman ρ", "kendall_tau": "Kendall τ",
    "calibration_slope": "|slope − 1|", "calibration_in_the_large": "|CITL|",
    "r2": "R²", "nrmse": "nRMSE", "rmse": "RMSE", "mae": "MAE", "crps": "CRPS", "nll": "NLL",
    "bias": "|bias|", "prediction_std_ratio": "|SD ratio − 1|",
}
#: Ranking and profit first, then probability or distribution quality (RESEARCH_BRIEF metrics).
#: EMP is the expected maximum profit per applicant, as a fraction of the loan (Verbraken et al.).
PROFILE_METRICS = {"pd": ("roc_auc", "ks", "h_measure", "emp", "log_loss", "brier_score", "ece",
                          "calibration_slope", "calibration_in_the_large"),
                   "lgd": ("r2", "spearman_r", "mae", "crps", "bias", "prediction_std_ratio")}


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
                      metric + "_original"]].replace([np.inf, -np.inf], np.nan).dropna()
        sub = sub[sub.fold_idx.isin(range(N_FOLDS))]
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


def lgd_density_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference TabPFN density scores on held-out credit tables, in natural-log units."""
    frame = _results_frames("reference", "lgd")
    counts = ["nll_rows", "nll_nonfinite_rows", "nll_clipped_rows"]
    losses = ["nll", "nll_capped100"]
    if not set(counts + losses) <= set(frame.columns):
        return pd.DataFrame(), pd.DataFrame()
    sub = frame[(~frame.test_dataset_id.str.startswith("openml_"))
                & (frame.model_source.eq("tabpfn-untuned") |
                   (frame.model_source.eq("tabpfn-trained") & frame.model_name.str.contains("cpt2_reference_a0_")))].copy()
    sub["model"] = sub.model_name.str.extract(r"(tabpfn-v[0-9.]+)", expand=False).map(MODEL_LABELS)
    sub["start"] = np.where(sub.model_source.eq("tabpfn-untuned"), "original", "credit CPT")
    sub = sub.rename(columns={"fold_idx": "fold"})
    keys = ["model", "start", "test_dataset_id"]
    measured = sub.dropna(subset=counts)
    measured = _all_folds(measured, keys)
    if measured.empty:
        return pd.DataFrame(), pd.DataFrame()
    if (measured.duplicated([*keys, "fold"]).any() or (measured.nll_rows <= 0).any()
            or (measured.nll_nonfinite_rows < 0).any()
            or (measured.nll_nonfinite_rows > measured.nll_clipped_rows).any()
            or (measured.nll_clipped_rows > measured.nll_rows).any()):
        raise ValueError("Invalid LGD density counts in benchmark results")
    tables = []
    for group, folds in measured.groupby(keys):
        row = dict(zip(keys, group))
        row.update({c: int(folds[c].sum()) for c in counts})
        row.update({m: float(np.average(folds[m], weights=folds.nll_rows)) for m in losses})
        row.update(nll_nonfinite_rate=row["nll_nonfinite_rows"] / row["nll_rows"],
                   nll_clipped_rate=row["nll_clipped_rows"] / row["nll_rows"])
        tables.append(row)
    per_table = pd.DataFrame(tables)
    summary = []
    for (model, start), group in per_table.groupby(["model", "start"]):
        row = dict(model=model, start=start, tables=len(group), tables_with_finite_nll=int(np.isfinite(group.nll).sum()))
        for col in [*losses, "nll_nonfinite_rate", "nll_clipped_rate"]:
            row["equal_table_" + col] = float(group[col].mean(skipna=False))
        row.update({c: int(group[c].sum()) for c in counts})
        row.update(pooled_nll_nonfinite_rate=row["nll_nonfinite_rows"] / row["nll_rows"],
                   pooled_nll_clipped_rate=row["nll_clipped_rows"] / row["nll_rows"])
        summary.append(row)
    return per_table, pd.DataFrame(summary)


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
    from src.utils import pilot_gate
    from src.utils.pilot_checks import receipts
    root = receipts()
    active = root / "active.json"
    summary = root / "summary.json"
    if active.is_file():
        run = pilot_gate.read(active)["run"]
        dated = pilot_gate.directory(root, run) / "summary.json"
        if dated.is_file():
            return pilot_gate.read(dated)
        record = pilot_gate.read(summary) if summary.is_file() else {}
        return record if record.get("run") == run else None
    return pilot_gate.read(summary) if summary.is_file() else None


def experiment0_tables(summary: dict | None) -> dict[str, pd.DataFrame]:
    if not summary:
        return {}
    pilots = pd.DataFrame([dict(pilot=k, passed=bool(v.get("passed")), completed=v.get("completed"),
                                problems=len(v.get("problems", [])))
                           for k, v in summary.get("pilots", {}).items()])
    checks = pd.DataFrame([dict(kind=kind, task=t, passed=bool(v.get("passed")))
                           for kind in ("recovery", "capacity", "evaluation", "scale", "precision", "benchmark")
                           for t, v in summary.get(kind, {}).items()])
    timing = pd.DataFrame([dict(pilot=k, **t) for k, v in summary.get("pilots", {}).items()
                           for t in v.get("timing", [])])
    updates, own_table = [], []
    for pilot, record in summary.get("pilots", {}).items():
        for trial in record.get("trials", []):
            skips = trial.get("numerical_skips") or {}
            updates.append(dict(pilot=pilot, track=record.get("track"), trial=trial["trial"],
                                status=trial.get("status"), updates=trial.get("successful_updates"),
                                amp_skips=skips.get("amp_skipped_steps"), data_skips=skips.get("data_skipped_steps")))
            check = record.get("pilot_checks", {}).get(trial["trial"], {})
            if "own_table_start" in check:
                own_table.append(dict(track=record["track"], trial=trial["trial"],
                                      metric="AUC" if record["track"] == "pd" else "RMSE",
                                      first=check["own_table_start"], last=check["own_table_end"],
                                      finding=check.get("finding", "")))
    recovery = pd.DataFrame([
        dict(task=int(task), track=r.get("track"), variant=r.get("variant"),
             interrupted_at=r.get("interrupt_after"), bitwise_equal=r.get("weights", {}).get("bitwise_equal"),
             max_weight_difference=r.get("weights", {}).get("max_absolute_difference"),
             monitors_match=r.get("monitors", {}).get("passed"), audit_notes=len(r.get("audit_problems", [])))
        for task, r in summary.get("recovery", {}).items()
    ])
    return {"pilots": pilots, "checks": checks, "timing": timing, "updates": pd.DataFrame(updates),
            "own_table": pd.DataFrame(own_table), "recovery": recovery}


# ----------------------------------------------------------------------------- display
#: Caption prefix of every figure drawn under the public table scope (appendix notebooks).
PUBLIC_CAPTION = ("Public tables only: the proprietary credit tables are removed from the evaluation; "
                  "the CPT corpora still contained them. ")


def pages(fig, name: str, caption: str) -> list:
    """Wrap one figure for ``campaign.show``; an absent figure yields no page. Under the public
    table scope the caption starts with ``PUBLIC_CAPTION``."""
    from src.visualize.campaign import Page
    if _SCOPE["tables"] == "public":
        caption = PUBLIC_CAPTION + caption
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
    # A distance to the original training corpus needs its private members' properties too.
    # Dropping missing members would change both the standardization and the nearest neighbor.
    if len(properties.test_dataset_id.unique()) < int(load_design()["tables"][track]):
        return pd.DataFrame()
    z = _property_space(properties)
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
    return in_scope(pd.DataFrame(rows), "test_dataset_id")


SEEDS = {"reference": {0: 42}, "seed": {0: 43, 1: 44}}   # E1 and the two E2 arms
SEED_LIST = sorted(s for seeds in SEEDS.values() for s in seeds.values())


def seed_noise(track: str) -> pd.DataFrame:
    """Per table and model: the gain at seed 42 (E1) and seeds 43 and 44 (E2), and whether it exceeds seed noise.

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
    # All three seeds are required: with one seed absent the range would be taken over two.
    wide = (pd.concat(frames).pivot_table(index=["model", "test_dataset_id", "credit"], columns="seed",
                                          values="gain").reindex(columns=SEED_LIST).dropna())
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
    """E4 versus E1, per held-out credit table and model: gain of credit CPT, of CPT on the matched
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
    matrix, means = bootstrap_means(frame, value, table=table, fold=fold, draws=draws, seed=seed)
    if matrix is None:
        return dict(tables=0, mean=np.nan, low=np.nan, high=np.nan)
    tail = (1 - level) / 2
    return dict(tables=int(len(matrix)), mean=float(matrix.mean()), low=float(np.quantile(means, tail)),
                high=float(np.quantile(means, 1 - tail)))


def bootstrap_means(frame: pd.DataFrame, value: str, *, table: str = "test_dataset_id",
                    fold: str = "fold_idx", draws: int = BOOTSTRAP_DRAWS, seed: int = 0):
    """(table × fold matrix of complete tables, equal-table means of the hierarchical draws);
    (None, None) without a complete table."""
    if frame.empty or value not in frame:
        return None, None
    valid = frame.loc[frame[fold].isin(range(N_FOLDS)) & np.isfinite(frame[value])]
    matrix = valid.pivot_table(index=table, columns=fold, values=value)
    matrix = matrix.reindex(columns=range(N_FOLDS)).dropna().to_numpy(dtype=float)
    if not len(matrix):
        return None, None
    rng = np.random.default_rng(seed)
    tables, folds = matrix.shape
    drawn_tables = rng.integers(tables, size=(draws, tables))
    drawn_folds = rng.integers(folds, size=(draws, tables, folds))
    return matrix, matrix[drawn_tables[:, :, None], drawn_folds].mean(axis=(1, 2))


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
    minus gain of CPT on the matched non-credit twins (E4), fold by fold on identical rows, with
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



def domain_contrast_intervals(outcomes: dict) -> pd.DataFrame:
    """Per task, outcome and model on the held-out credit tables: the improvement of credit CPT
    (E1) over CPT on the matched non-credit twins (E4), fold by fold on identical rows, with its
    95% hierarchical-bootstrap interval (positive = the credit corpus did better). ``outcomes``
    maps a task to its metrics, as ``analysis.primary``."""
    rows = []
    keys = ["model", "test_dataset_id", "fold_idx"]
    for track, metrics in outcomes.items():
        credit, control = _paired("reference", track, list(metrics)), _paired("domain_control", track, list(metrics))
        if credit.empty or control.empty:
            continue
        for metric in [m for m in metrics if m in credit and m in control]:
            merged = credit[credit.credit & (credit.arm == 0)][[*keys, metric]].merge(
                control[control.credit & (control.arm == 0)][[*keys, metric]], on=keys,
                suffixes=("_credit", "_noncredit")).dropna()
            if merged.empty:
                continue
            merged["improvement"] = improvement(metric, merged[metric + "_credit"], merged[metric + "_noncredit"])
            for model, group in merged.groupby("model"):
                interval = bootstrap_interval(group, "improvement")
                if interval["tables"]:
                    rows.append(dict(track=track, metric=metric, model=model, **interval))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- pre-specified primary endpoint
def primary_spec(design: dict | None = None) -> dict:
    """``analysis.primary`` of config/design.yaml: outcomes and α."""
    return (design or load_design())["analysis"]["primary"]


def _primary_tables(track: str) -> set:
    from src.data.preprocessing import DATASET_METADATA
    registry = pd.DataFrame({"test_dataset_id": [k for k, meta in DATASET_METADATA.items()
                                                 if meta["track"] == track]})
    return set(in_scope(registry, "test_dataset_id").test_dataset_id)


def primary_endpoint(design: dict | None = None, *, draws: int = BOOTSTRAP_DRAWS, seed: int = 0) -> pd.DataFrame:
    """The confirmatory test of RQ1, fixed before the results were opened (RESEARCH_BRIEF).

    Per task, primary outcome and model: the equal-table mean five-fold improvement of the E1
    reference model (seed 42) over the original model on the held-out credit tables, with its
    95% hierarchical-bootstrap interval, a two-sided bootstrap p-value against zero and the Holm
    adjustment over the whole family (tasks × outcomes × models). No equivalence margin is
    pre-specified: a comparison that is not significant is reported with its interval, not called
    equivalent. Improvement is oriented so that positive is better (``improvement``). Everything
    else in the study is secondary.
    """
    spec = primary_spec(design)
    rows = []
    for track, outcomes in spec["outcomes"].items():
        merged = _paired("reference", track, list(outcomes))
        if merged.empty:
            continue
        expected = _primary_tables(track)
        merged = merged[merged.credit & (merged.arm == 0)]
        for metric in [m for m in outcomes if m in merged]:
            sub = merged.dropna(subset=[metric, metric + "_original"]).copy()
            sub["improvement"] = improvement(metric, sub[metric], sub[metric + "_original"])
            for model, group in sub.groupby("model"):
                matrix, means = bootstrap_means(group, "improvement", draws=draws, seed=seed)
                if matrix is None:
                    continue
                per_table = matrix.mean(axis=1)
                tail = min(np.sum(means <= 0), np.sum(means >= 0))
                covered = set(group.test_dataset_id) == expected and len(matrix) == len(expected)
                rows.append(dict(track=track, metric=metric, model=model, tables=len(matrix), expected_tables=len(expected),
                                 tables_improved=int((per_table > 0).sum()), mean=float(matrix.mean()),
                                 low=float(np.quantile(means, 0.025)), high=float(np.quantile(means, 0.975)),
                                 p_value=float(min(1.0, 2 * (tail + 1) / (len(means) + 1))) if covered else np.nan))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # The family is the DECLARED one (tasks x outcomes x models), not the tests that happen to have
    # results: a missing test enters Holm with p = 1, so a partial download cannot shrink the
    # correction and turn a result significant.
    from src.visualize.style import model_label
    design = design or load_design()
    models = [MODEL_LABELS.get(m, model_label(m)) for m in design["models"]]
    family = pd.DataFrame([dict(track=t, metric=m, model=model) for t, ms in spec["outcomes"].items()
                           for m in ms for model in models])
    out = family.merge(out, on=["track", "metric", "model"], how="left")
    out["missing"] = out.p_value.isna()
    out["p_holm"] = holm(out.p_value.fillna(1.0).to_numpy())
    alpha = float(spec["alpha"])
    out["decision"] = np.where(out.missing, "missing", np.where(
        out.p_holm >= alpha, "not significant", np.where(out["mean"] > 0, "better", "worse")))
    out.loc[out.missing & out.tables.notna(), "decision"] = "incomplete"
    return out



def outcome_intervals(slug: str, track: str, metrics, *, arm: int = 0) -> pd.DataFrame:
    """Per metric, model and table group (held-out credit, retention): equal-table mean five-fold
    improvement (positive = better, ``improvement``) with its 95% hierarchical-bootstrap interval."""
    merged = _paired(slug, track, list(metrics))
    if merged.empty:
        return pd.DataFrame()
    merged = merged[merged.arm == arm]
    rows = []
    for metric in [m for m in metrics if m in merged]:
        sub = merged.dropna(subset=[metric, metric + "_original"]).copy()
        sub["improvement"] = improvement(metric, sub[metric], sub[metric + "_original"])
        for (model, credit), group in sub.groupby(["model", "credit"]):
            interval = bootstrap_interval(group, "improvement")
            if interval["tables"]:
                rows.append(dict(track=track, metric=metric, model=model, credit=bool(credit), **interval))
    return pd.DataFrame(rows)

def holm(p_values) -> np.ndarray:
    """Holm's step-down adjustment (family-wise error), monotone and capped at 1."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def positive_control(changes: pd.DataFrame, *, slug: str = "reference", arm: int = 0) -> pd.DataFrame:
    """Does CPT learn what it is trained on? Per task and model, the monitor change at the last
    update on the run's OWN training tables against the held-out credit tables (equal-table
    means; a table monitored by several partitions counts once). A model that does not improve
    its own training tables cannot be expected to transfer; one that does, and still does not
    transfer, isolates the failure to generalisation across tables."""
    if changes.empty:
        return pd.DataFrame()
    sub = changes[(changes.slug == slug) & (changes.arm == arm) & changes.scope.isin(["train", "test"])]
    if sub.empty:
        return pd.DataFrame()
    last = sub.groupby("run").successful_updates.transform("max")
    sub = sub[sub.successful_updates == last]
    per_table = sub.groupby(["track", "model", "scope", "table"]).change.mean().reset_index()
    out = (per_table.groupby(["track", "model", "scope"])
           .change.agg(tables="size", mean_change="mean",
                       tables_improved=lambda g: int((g > 0).sum())).reset_index())
    return out.pivot_table(index=["track", "model"], columns="scope",
                           values=["tables", "mean_change", "tables_improved"]).pipe(
        lambda w: w.set_axis([f"{scope}_{stat}" for stat, scope in w.columns], axis=1)).reset_index()

PRECISION_LABELS = {"bf16": "BF16", "fp16": "FP16", "fp16_norm32": "FP16 + FP32 norms"}
_ALL_PRECISION_LABELS = {**PRECISION_LABELS, "fp32": "FP32", "fp32_bounded": "Bounded strict FP32",
                         "fp32_checkpointed": "Checkpointed FP32"}


def _precision_records(summary):
    """Policy in force at measurement time, including historical receipts with dtype only."""
    for record in ((summary or {}).get("precision") or {}).values():
        if not isinstance(record, dict) or "tables" not in record:
            continue
        base = str(record.get("base", ""))
        match = re.search(r"(tabpfn-v[0-9.]+?)(?=-|$)|(tabicl)", base)
        policy = record.get("training_label") or {
            "float16": "fp16", "float32": "fp32", "bfloat16": "bf16"
        }.get(record.get("training_dtype"))
        yield record, _model_label(match.group(0) if match else base), policy


def precision_agreement(summary: dict | None) -> pd.DataFrame:
    """Experiment 0 precision check, one row per task, model, training table and candidate
    precision: update-zero loss in FP32 and in the candidate, and the cosine between the two
    gradients on the identical episode. ``table_rank`` orders the training tables by size."""
    rows = []
    for record, model, production in _precision_records(summary):
        # Historical receipts measured fewer policies. Include a missing comparison only when
        # that policy was registered as production or recorded somewhere in this probe.
        policies = [key for key in PRECISION_LABELS if key == production or any(
            f"grad_cosine_{key}" in row or f"error_{key}" in row for row in record["tables"])]
        for table in record["tables"]:
            for key in policies:
                cosine = table.get(f"grad_cosine_{key}")
                status = "measured" if pd.notna(cosine) and np.isfinite(cosine) else "unavailable"
                rows.append(dict(track=record.get("track"), model=model,
                                 table_rank=int(table["table_index"]) + 1, rows=table.get("rows"),
                                 precision=PRECISION_LABELS[key],
                                 recorded_training_policy=_ALL_PRECISION_LABELS.get(production, production or "unrecorded"),
                                 role=("unrecorded" if production is None else
                                       "production" if key == production else "comparison"), status=status,
                                 loss_fp32=table.get("loss_fp32"),
                                 loss=table.get(f"loss_{key}"), gradient_cosine=cosine,
                                 gradient_norm_ratio=table.get(f"grad_norm_ratio_{key}")))
    return pd.DataFrame(rows)


def precision_fp32_evidence(summary: dict | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate deployed FP32 measurements from independently computed native reference checks."""
    production_rows, parity_rows = [], []
    for record, model, production in _precision_records(summary):
        identity = dict(track=record.get("track"), model=model,
                        recorded_training_policy=_ALL_PRECISION_LABELS.get(production, production or "unrecorded"))
        parity = record.get("reference_parity")
        if isinstance(parity, dict):
            scope = "all tables: minimum cosine and maximum relative error" if "tables" in parity else "smaller episode"
            parity_rows.append(dict(**identity, comparison=scope, **parity))
        for table in record["tables"]:
            measured = table.get("reference_parity")
            if isinstance(measured, dict):
                parity_rows.append(dict(**identity, comparison="smaller episode", table_rank=int(table["table_index"]) + 1,
                                        **measured))
        if production != "fp32_bounded":
            continue
        for table in record["tables"]:
            production_rows.append(dict(**identity, table_rank=int(table["table_index"]) + 1,
                                        rows=table.get("rows"), loss=table.get("loss_fp32"),
                                        gradient_norm=table.get("grad_norm_fp32"),
                                        gradient_finite=table.get("grad_finite_fp32"), passed=table.get("passed")))
    return pd.DataFrame(production_rows), pd.DataFrame(parity_rows)


def precision_full_episodes(summary: dict | None) -> pd.DataFrame:
    """Full training-size backward probes; older probes may record only feasibility and timing."""
    rows = []
    for record, model, production in _precision_records(summary):
        for table in record.get("full_episode", []):
            for key in ("fp32", "fp32_bounded", "fp32_checkpointed", "production", "fp16", "fp16_norm32", "bf16"):
                trial = table.get(key)
                if not isinstance(trial, dict):
                    continue
                policy = trial.get("policy") or (production if key == "production" else key)
                rows.append(dict(track=record.get("track"), model=model, table_rank=int(table["table_index"]) + 1,
                                 rows=table.get("rows"), precision=_ALL_PRECISION_LABELS.get(policy, policy or "unrecorded"),
                                 role={"production": "production", "fp32": "reference"}.get(key, "comparison"),
                                 status="failed" if "error" in trial else "completed",
                                 gradient_cosine=trial.get("gradient_cosine"),
                                 gradient_agrees=trial.get("gradient_agrees"), loss_agrees=trial.get("loss_agrees"),
                                 loss=trial.get("loss"), gradient_norm=trial.get("gradient_norm"),
                                 seconds=trial.get("seconds"), peak_gb=trial.get("peak_gb")))
    return pd.DataFrame(rows)


def cost_comparison(resources: pd.DataFrame, finetune: pd.DataFrame) -> pd.DataFrame:
    """Compare one reference checkpoint with one target adaptation including its candidate search.

    The four independent CPT runs and five target folds measure these unit costs. Their
    experimental totals are reported separately. A break-even estimate needs every reference
    partition and comparable, recorded GPUs.
    """
    design = load_design()
    partitions = set(range(int(design["partitions"])))
    steps = int(design["reference"]["successful_updates"])
    rows = []
    if not resources.empty:
        cpt = resources[(resources.slug == "reference") & (resources.arm == 0)]
        cpt = cpt[cpt.partition.isin(partitions) & (cpt.updates == steps)
                  & np.isfinite(cpt.hours_without_monitor) & (cpt.hours_without_monitor > 0)]
        for (track, model), group in cpt.groupby(["track", "model"]):
            if group.partition.duplicated().any():
                raise ValueError(f"Duplicate reference partition in {track} / {model} cost records")
            rows.append(dict(track=track, model=_model_label(model), cpt_runs=len(group),
                             cpt_device=_devices(group, "gpu_name"),
                             cpt_hours_per_run=float(group.hours_without_monitor.mean()),
                             cpt_hours_total=float(group.hours_without_monitor.sum())))
    out = pd.DataFrame(rows)
    if finetune.empty or "fit_seconds" not in finetune or out.empty:
        return out
    tuned = finetune[finetune.finetuned.astype(bool) & (finetune.start == "original")]
    keys = ["track", "model", "dataset_id"]
    tuned = _all_folds(tuned.loc[np.isfinite(tuned.fit_seconds) & (tuned.fit_seconds > 0)], keys)
    if tuned.duplicated([*keys, "fold"]).any():
        raise ValueError("Duplicate target adaptation in fine-tuning cost records")
    per_table = (tuned.groupby(keys).fit_seconds.sum() / 3600).reset_index()
    per_table["model"] = per_table.model.map(_model_label)
    per_model = per_table.groupby(["track", "model"]).agg(finetune_tables=("dataset_id", "size"),
        finetune_hours_per_table=("fit_seconds", "mean"), datasets=("dataset_id", set)).reset_index()
    expected = {track: _primary_tables(track) for track in per_model.track.unique()}
    per_model["finetune_complete"] = [datasets == expected[track]
                                      for track, datasets in zip(per_model.track, per_model.pop("datasets"))]
    out = out.merge(per_model, on=["track", "model"], how="left")
    out["finetune_hours_per_fit"] = out.finetune_hours_per_table / N_FOLDS
    hardware = {(track, _model_label(model)): _devices(group, "device_name")
                for (track, model), group in tuned.groupby(["track", "model"])}
    out["finetune_device"] = [hardware.get((t, m), "") for t, m in zip(out.track, out.model)]
    comparable = ((out.cpt_runs == len(partitions)) & out.finetune_complete.eq(True) & out.cpt_device.eq(out.finetune_device)
                  & ~out.cpt_device.isin(["", "cpu", "cuda", "unknown"]) & ~out.cpt_device.str.contains(","))
    out["tables_to_break_even"] = (out.cpt_hours_per_run / out.finetune_hours_per_fit).where(comparable)
    if "predict_seconds" in finetune:
        # Inference cost per 1,000 test rows on the full context, untuned and fine-tuned.
        timed = finetune.dropna(subset=["predict_seconds", "test_rows"]).assign(
            per_1k=lambda f: 1000 * f.predict_seconds / f.test_rows, model=lambda f: f.model.map(_model_label))
        for tuned, column in ((False, "inference_seconds_per_1k_rows"),
                              (True, "finetuned_inference_seconds_per_1k_rows")):
            part = timed[timed.finetuned.astype(bool) == tuned].groupby(["track", "model"]).per_1k.mean()
            out = out.merge(part.rename(column).reset_index(), on=["track", "model"], how="left")
    return out


def _devices(group: pd.DataFrame, column: str) -> str:
    if column not in group:
        return "unknown"
    return ", ".join(sorted(set(group[column].fillna("unknown").astype(str).replace("", "unknown"))))


def task_results(kind: str) -> pd.DataFrame:
    """All finished E5 (``finetune``), E12 (``episode_context``) or E6 (``context``) task CSVs."""
    from src.eval.campaign_tasks import SLUGS
    from src.utils.experiment import file_digest
    from src.utils.paths import results_dir
    try:
        folder = results_dir(experiment=SLUGS[kind]) / kind
    except Exception:  # noqa: BLE001 - missing layout is missing evidence
        return pd.DataFrame()
    frames = []
    code_versions = set()
    for f in sorted(folder.glob(f"{kind}_*.csv.gz")):
        receipt = folder / "completed" / f.name.replace(".csv.gz", ".json")
        if not receipt.is_file():  # still being published
            continue
        done = json.loads(receipt.read_text(encoding="utf-8"))
        identity = done.get("identity", {})
        task = identity.get("task", {})
        predictions = f.with_name(f.name.replace(".csv.gz", ".predictions.npz"))
        if (identity.get("kind") != kind or identity.get("quick") is not False or not identity.get("code")
                or task.get("fold") not in range(N_FOLDS) or task != done.get("task")
                or done.get("predictions") != predictions.name
                or not re.fullmatch(r"[0-9a-f]{64}", str(done.get("predictions_sha256")))
                or done.get("metrics_sha256") != file_digest(f, fresh=True)):
            raise ValueError(f"Invalid {kind} result receipt or changed metrics: {f.name}")
        # A compact download may contain only metrics and receipts. Verify prediction files
        # when present; the notebook consumes the checksummed metric values.
        if predictions.is_file() and done["predictions_sha256"] != file_digest(predictions, fresh=True):
            raise ValueError(f"Changed predictions for {f.name}")
        shard = pd.read_csv(f)
        if (shard.empty or len(shard) != done.get("rows")
                or any(key not in shard or not shard[key].eq(value).all() for key, value in task.items())):
            raise ValueError(f"Rows disagree with the task in {receipt.name}")
        code_versions.add(identity["code"])
        frames.append(shard)
    if len(code_versions) > 1:
        raise ValueError(f"Mixed evaluation revisions in {kind} results")
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return in_scope(frame, "dataset_id")


def _all_folds(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Rows of the groups (``keys``) scored on all five outer folds; partial groups are dropped."""
    if frame.empty:
        return frame
    frame = frame[frame.fold.isin(range(N_FOLDS))]
    folds = frame.groupby(keys).fold.transform("nunique")
    return frame[folds == N_FOLDS]


def finetune_summary(results: pd.DataFrame) -> pd.DataFrame:
    """E5 per model and start (original, CPT): the untuned and the fine-tuned score on identical
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


def episode_context_summary(results: pd.DataFrame) -> pd.DataFrame:
    """E12 at matched inference context: mean AUC per episode-size arm, model and context size."""
    if results.empty or "roc_auc" not in results:
        return pd.DataFrame()
    # The full pool differs by a row or two between folds, so it is one context ("full").
    grid = set(int(v) for v in experiment_by_slug("episode_size")[1]["matched_context_evaluation"][:-1])
    results = results.assign(context=[str(int(c)) if int(c) in grid else "full" for c in results.context_rows])
    results = _all_folds(results, ["arm", "model", "context", "dataset_id"])
    per_table = results.groupby(["arm", "model", "context", "dataset_id"]).roc_auc.mean().reset_index()
    return per_table.groupby(["arm", "model", "context"]).roc_auc.agg(["mean", "median", "size"]).reset_index()


# ------------------------------------------------------------------- arm settings and labels
_SLUG_EXPERIMENT = {"reference": "E1", "seed": "E2", "domain_control": "E4", "update_size_anchoring": "E8",
                    "scope": "E9", "exposure": "E10", "table_size": "E11",
                    "episode_size": "E12", "transfer": "E7", "precision": "E3"}


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
            return "Reference size, large tables"
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
    return in_scope(long.drop(columns="column").dropna(subset=["change"]), "table")


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
    return band[band.seeds == len(SEED_LIST)].reset_index(drop=True)


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
                         gpu_name=_devices(group, "gpu_name"),
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
    """E5 per track, model and table (all five folds): the original model without fine-tuning,
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



#: E5 options in order of per-table cost; a tie goes to the cheaper option (CPT is paid once).
ADAPTATION_OPTIONS = ("original", "cpt", "original_finetune", "cpt_finetune")
#: Labelled training rows of a fold (context after the validation split), for the size bands.
SIZE_BANDS = (0, 1_000, 10_000, np.inf)


def _size_band(rows: float) -> str:
    for low, high in zip(SIZE_BANDS[:-1], SIZE_BANDS[1:]):
        if low <= rows < high:
            return f"{low:,}-{high:,.0f}" if np.isfinite(high) else f">={low:,}"
    return "unknown"


def adaptation_by_size(results: pd.DataFrame) -> pd.DataFrame:
    """E5 per task, model and held-out table: labelled context rows and the three contrasts a
    practitioner chooses between — CPT alone vs the original, fine-tuning vs the original, and
    CPT+FT vs FT (does CPT still add once the table is fine-tuned on) — plus the option with the
    best five-fold score (ties to the cheaper option). The best option is selected on the test
    folds, so its count describes the comparison; it is not a tuned recommendation."""
    table = finetune_table(results)
    if table.empty or "available_context" not in results:
        return table
    rows = (results.assign(model=results.model.map(_model_label))
            .groupby(["track", "model", "dataset_id"]).available_context.mean().rename("context_rows"))
    table = table.merge(rows.reset_index(), on=["track", "model", "dataset_id"], how="left")
    table["cpt_after_finetune"] = table.cpt_finetune - table.original_finetune
    scores = table[list(ADAPTATION_OPTIONS)].to_numpy(dtype=float)
    table["best_option"] = [ADAPTATION_OPTIONS[i] for i in np.argmax(scores, axis=1)]
    table["size_band"] = table.context_rows.map(_size_band)
    return table


def adaptation_summary(table: pd.DataFrame, cost: pd.DataFrame | None = None) -> pd.DataFrame:
    """The E5 recommendation table: per task, labelled-row band and model, the equal-table mean
    change of each option, how often each option scored best, and the GPU hours it costs."""
    if table.empty or "size_band" not in table:
        return pd.DataFrame()
    out = (table.groupby(["track", "size_band", "model"])
           .agg(tables=("dataset_id", "size"), cpt_change=("cpt_change", "mean"),
                finetune_change=("finetune_change", "mean"),
                cpt_finetune_change=("cpt_finetune_change", "mean"),
                cpt_after_finetune=("cpt_after_finetune", "mean")).reset_index())
    wins = (table.groupby(["track", "size_band", "model"]).best_option.value_counts().unstack(fill_value=0)
            .reindex(columns=list(ADAPTATION_OPTIONS), fill_value=0).add_prefix("best_").reset_index())
    out = out.merge(wins, on=["track", "size_band", "model"])
    if cost is not None and not cost.empty and "finetune_hours_per_table" in cost:
        out = out.merge(cost[["track", "model", "cpt_hours_per_run", "finetune_hours_per_fit"]],
                        on=["track", "model"], how="left")
    order = {_size_band(low): i for i, low in enumerate(SIZE_BANDS[:-1])}
    return out.sort_values(["track", "size_band", "model"], key=lambda s: s.map(order) if s.name == "size_band"
                           else s).reset_index(drop=True)

# --------------------------------------------------------------------------- leaderboard
#: TabArena's error per task (lower is better): 1 − AUC for binary tasks, RMSE for regression.
LEADERBOARD_ERROR = {"pd": ("roc_auc", "1 − AUC"), "lgd": ("rmse", "RMSE")}
#: The Elo anchor (= 1000): the linear baseline, the standard reference model of credit scoring.
LEADERBOARD_ANCHOR = {"pd": "Logistic regression", "lgd": "Linear regression"}
LEADERBOARD_BASELINES = {"catboost": "CatBoost", "xgboost": "XGBoost", "logreg": "Logistic regression",
                         "linreg": "Linear regression"}
#: E5 start and fine-tuning → suffix of the method name.
LEADERBOARD_VARIANTS = {("original", False): "", ("cpt", False): " CPT", ("original", True): " FT",
                        ("cpt", True): " CPT+FT"}
#: A method enters a leaderboard only when it is scored on at least this share of the tables.
LEADERBOARD_MIN_COVERAGE = 0.5


def _error(track: str, frame: pd.DataFrame) -> pd.Series:
    metric, _ = LEADERBOARD_ERROR[track]
    values = frame[metric].astype(float)
    return 1.0 - values if track == "pd" else values


def leaderboard_errors(track: str, finetune: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long frame (method, base, variant, table, fold, error) on the held-out credit tables.

    The tuned baselines come from the E1 five-fold evaluation. The foundation models come from E5
    when it exists: the original and the E1 CPT model, each without and with fine-tuning on the
    table, all scored on the same context rows under E5's context cap, so fine-tuning is never
    compared with a model that saw more labelled rows (E1 lets v3 and TabICLv2 use up to 1,000,000
    rows, E5 at most 50,000). Without E5 the untuned models come from E1. E5 uses the same outer
    folds and inner split (``POLICY_SEEDS``), so every method meets on identical test rows."""
    metric, _ = LEADERBOARD_ERROR[track]
    frame = _results_frames("reference", track)
    tuned = (finetune is not None and not finetune.empty and "finetuned" in finetune
             and not finetune[finetune.track == track].empty)
    rows = []
    if not frame.empty and metric in frame:
        frame = frame[~frame.test_dataset_id.astype(str).str.startswith("openml_")].dropna(subset=[metric])
        base = frame.model_name.str.extract(r"(tabpfn-v[0-9.]+|tabicl)")[0]
        source = frame.model_source.astype(str)
        original = source.str.endswith("untuned") & base.notna()
        cpt = (source.str.endswith("trained") & frame.model_name.str.contains("cpt2_reference_a0_", regex=False)
               & base.notna())
        baseline = (source == "baseline") & frame.model_name.isin(LEADERBOARD_BASELINES)
        for mask, variant in (() if tuned else ((original, ""), (cpt, " CPT"))):
            sub = frame[mask]
            label = base[mask].map(MODEL_LABELS)
            rows.append(pd.DataFrame({"method": label + variant, "base": label, "variant": variant.strip() or "original",
                                      "table": sub.test_dataset_id, "fold": sub.fold_idx, "error": _error(track, sub)}))
        sub = frame[baseline]
        rows.append(pd.DataFrame({"method": sub.model_name.map(LEADERBOARD_BASELINES), "base": "baseline",
                                  "variant": "baseline", "table": sub.test_dataset_id, "fold": sub.fold_idx,
                                  "error": _error(track, sub)}))
    if tuned:
        sub = in_scope(finetune[finetune.track == track], "dataset_id")
        if not sub.empty and metric in sub:
            sub = sub.dropna(subset=[metric])
            label = sub.model.map(_model_label)
            suffix = [LEADERBOARD_VARIANTS[(s, bool(f))] for s, f in zip(sub.start, sub.finetuned)]
            rows.append(pd.DataFrame({"method": label + suffix, "base": label,
                                      "variant": [x.strip() or "original" for x in suffix], "table": sub.dataset_id,
                                      "fold": sub.fold, "error": _error(track, sub)}))
    rows = [r for r in rows if not r.empty]
    if not rows:
        return pd.DataFrame(columns=["method", "base", "variant", "table", "fold", "error"])
    errors = pd.concat(rows, ignore_index=True)
    return errors[errors.fold.isin(range(N_FOLDS))].drop_duplicates(["method", "table", "fold"], keep="last")


def leaderboard(track: str, finetune: pd.DataFrame | None = None, *, seed: int = 0) -> dict:
    """TabArena-style leaderboard of one task (``src.visualize.arena``): ``board`` (one row per
    method), ``coverage`` (tables and folds per method, and whether it entered) and ``dropped``
    (tables left out because a method in the board lacked one of their folds). Methods covering
    fewer than half of the tables are left out rather than shrinking every comparison to them."""
    from src.visualize import arena
    errors = leaderboard_errors(track, finetune)
    empty = dict(board=pd.DataFrame(), coverage=pd.DataFrame(), dropped=[])
    if errors.empty:
        return empty
    tables = errors.table.nunique()
    coverage = (errors.groupby(["method", "base", "variant"])
                .agg(tables=("table", "nunique"), units=("fold", "size")).reset_index())
    coverage["entered"] = coverage.tables >= LEADERBOARD_MIN_COVERAGE * tables
    kept = errors[errors.method.isin(coverage.method[coverage.entered])]
    # A table enters with all five outer folds only; a fold missing for every method would
    # otherwise pass the per-method completeness check below.
    folds = kept.groupby("table").fold.nunique()
    short = sorted(folds.index[folds < N_FOLDS])
    kept, dropped = arena.complete_units(kept[~kept.table.isin(short)])
    dropped = sorted(set(dropped) | set(short))
    anchor = LEADERBOARD_ANCHOR[track]
    if kept.empty or anchor not in set(kept.method) or kept.method.nunique() < 2:
        return dict(empty, coverage=coverage, dropped=dropped)
    board = arena.leaderboard(kept[list(arena.COLUMNS)], anchor, seed=seed)
    meta = coverage.set_index("method")[["base", "variant"]]
    board = board.join(meta, on="method")
    board.insert(0, "track", track)
    return dict(board=board, coverage=coverage.assign(track=track), dropped=dropped)


def context_table(results: pd.DataFrame) -> pd.DataFrame:
    """E6 per track, model, table and labelled-context size (all five folds): original and CPT
    means and the paired gain. When the E4 checkpoints were scored (start ``noncredit_cpt``), also
    ``noncredit_gain`` (E4 CPT minus original) and ``credit_minus_noncredit`` (E1 minus E4 CPT) on
    the same rows; NaN where E4 lacks one of the five folds. ``full`` marks the size that used the
    whole available pool (capped at the model's context limit) rather than a declared grid size."""
    if results.empty or "context_rows" not in results:
        return pd.DataFrame()
    grid = set(int(v) for v in experiment_by_slug("context")[1]["context_rows"])
    rows = []
    for track, metric in (("pd", "roc_auc"), ("lgd", "r2")):
        sub = results[results.track == track]
        if sub.empty or metric not in sub:
            continue
        wide = sub.pivot_table(index=["model", "dataset_id", "fold", "context_rows"], columns="start",
                               values=metric).reset_index()
        if not {"original", "cpt"} <= set(wide.columns):
            continue
        # The full pool differs by a row or two between folds (it is what each fold had), so it is
        # grouped as one size (-1); declared grid sizes are identical in every fold.
        wide["size"] = np.where(wide.context_rows.astype(int).isin(grid), wide.context_rows.astype(int), -1)
        wide = _all_folds(wide.dropna(subset=["original", "cpt"]), ["model", "dataset_id", "size"])
        if wide.empty:
            continue
        wide["gain"] = wide.cpt - wide.original
        columns = ["context_rows", "original", "cpt", "gain"]
        if "noncredit_cpt" in wide:
            folds = wide.groupby(["model", "dataset_id", "size"]).noncredit_cpt.transform("count")
            wide.loc[folds < N_FOLDS, "noncredit_cpt"] = np.nan
            wide["noncredit_gain"] = wide.noncredit_cpt - wide.original
            wide["credit_minus_noncredit"] = wide.cpt - wide.noncredit_cpt
            columns += ["noncredit_cpt", "noncredit_gain", "credit_minus_noncredit"]
        table = (wide.groupby(["model", "dataset_id", "size"])[columns]
                 .mean().reset_index())
        table = table.assign(track=track, metric=metric, model=table.model.map(_model_label),
                             full=table["size"] < 0).drop(columns="size")
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def context_cohort_curve(table: pd.DataFrame) -> pd.DataFrame:
    """Equal-table means of the E6 table per track, model and grid size. ``fixed_cohort`` is true
    up to the largest size every table of that model reached, so the mean there compares the same
    tables throughout; beyond it the population shrinks with the size."""
    if table.empty:
        return pd.DataFrame()
    grid = table[~table.full]
    extra = ({"noncredit_gain": ("noncredit_gain", "mean"),
              "credit_minus_noncredit": ("credit_minus_noncredit", "mean"),
              "noncredit_tables": ("noncredit_gain", "count")} if "noncredit_gain" in grid else {})
    curve = (grid.groupby(["track", "model", "context_rows"])
             .agg(original=("original", "mean"), cpt=("cpt", "mean"), gain=("gain", "mean"),
                  median_gain=("gain", "median"), tables=("gain", "size"), **extra).reset_index())
    total = grid.groupby(["track", "model"]).dataset_id.nunique().rename("all_tables").reset_index()
    curve = curve.merge(total, on=["track", "model"])
    curve["fixed_cohort"] = curve.tables == curve.all_tables
    if "noncredit_tables" in curve:
        incomplete = curve.noncredit_tables != curve.tables
        curve.loc[incomplete, ["noncredit_gain", "credit_minus_noncredit"]] = np.nan
    return curve


#: E6 quantities whose trend against context size the slope test reads.
SLOPE_QUANTITIES = {"gain": "Credit CPT (E1) minus original",
                    "noncredit_gain": "Non-credit CPT (E4) minus original",
                    "credit_minus_noncredit": "Credit minus non-credit CPT (E1 minus E4)"}


def context_slopes(table: pd.DataFrame) -> pd.DataFrame:
    """E6 slope test, per track, model, table and quantity: the least-squares slope of the gain on
    log2 of the labelled context rows, i.e. the change of the gain per doubling of the context.

    Every table of a model enters over the same sizes: the grid sizes up to the largest size that
    all its tables reached (the fixed cohort of ``context_cohort_curve``), at least three. A negative
    slope means the gain shrinks as the model sees more labelled target rows: in-context learning
    then supplies what CPT added."""
    if table.empty:
        return pd.DataFrame()
    grid = table[~table.full]
    rows = []
    for (track, model), sub in grid.groupby(["track", "model"]):
        counts = sub.groupby("context_rows").dataset_id.nunique()
        common_sizes = counts.index[counts == sub.dataset_id.nunique()]
        if len(common_sizes) < 3:
            continue
        up_to = common_sizes.max()
        sub = sub[sub.context_rows.isin(common_sizes)]
        for quantity in [q for q in SLOPE_QUANTITIES if q in sub]:
            for table_id, cells in sub.dropna(subset=[quantity]).groupby("dataset_id"):
                if cells.context_rows.nunique() != len(common_sizes):
                    continue
                x = np.log2(cells.context_rows.to_numpy(dtype=float))
                slope = float(np.polyfit(x, cells[quantity].to_numpy(dtype=float), 1)[0])
                rows.append(dict(track=track, model=model, dataset_id=table_id, quantity=quantity,
                                 slope_per_doubling=slope, sizes=int(cells.context_rows.nunique()),
                                 up_to=int(up_to)))
    return pd.DataFrame(rows)


def slope_summary(slopes: pd.DataFrame, *, draws: int = BOOTSTRAP_DRAWS, seed: int = 0) -> pd.DataFrame:
    """Per track, model and quantity: the equal-table mean slope with a 95% bootstrap interval over
    tables (a slope is one value per table) and the number of tables with a negative slope."""
    if slopes.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for (track, model, quantity), group in slopes.groupby(["track", "model", "quantity"]):
        values = group.slope_per_doubling.to_numpy(dtype=float)
        means = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
        rows.append(dict(track=track, model=model, quantity=quantity, tables=len(values),
                         mean=float(values.mean()), low=float(np.quantile(means, 0.025)),
                         high=float(np.quantile(means, 0.975)), tables_negative=int((values < 0).sum())))
    out = pd.DataFrame(rows)
    out["excludes_zero"] = (out.low > 0) | (out.high < 0)
    return out

def episode_context_gains(episode: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """E12 at matched inference context: each episode-size model minus the original model scored
    by E6 with the same context rows of the same fold, per arm, model and table (five folds)."""
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
    # The full pool differs by a row or two between folds, so it is one context ("full").
    grid = set(int(v) for v in experiment_by_slug("episode_size")[1]["matched_context_evaluation"][:-1])
    merged["context"] = [str(int(c)) if int(c) in grid else "full" for c in merged.context_rows]
    merged = _all_folds(merged, ["arm", "model", "dataset_id", "context"])
    if merged.empty:
        return pd.DataFrame()
    out = (merged.groupby(["arm", "model", "dataset_id", "context"])
           .agg(gain=("gain", "mean"), context_rows=("context_rows", "mean")).reset_index())
    settings = arm_settings("episode_size", "pd")[["arm", "episode_rows"]]
    out = out.merge(settings, on="arm")
    # The reference-size arm has no declared episode size: it is each model's reference size.
    reference = load_design()["reference"]["episode_rows"]
    out["episode_rows"] = out.episode_rows.astype(float).fillna(out.model.map(reference).astype(float))
    out["model"] = out.model.map(_model_label)
    return out


# ------------------------------------------------------------------------ E7 transfer
def transfer_sources(track: str) -> dict[int, str]:
    """Single-source arm index -> source table id (tables of the task sorted by id, the rule the
    generated E7 configs use). Refuses a registry whose size differs from the design, since a
    missing table would shift every later label."""
    from src.data.preprocessing import DATASET_METADATA
    ids = sorted(i for i, meta in DATASET_METADATA.items() if meta["track"] == track)
    expected = int(load_design()["tables"][track])
    if len(ids) != expected:
        raise RuntimeError(f"{track}: {len(ids)} registered tables, design has {expected}; E7 source labels would shift")
    return dict(enumerate(ids))


def transfer_long(frame: pd.DataFrame, track: str) -> pd.DataFrame:
    """E7: per model, source arm and target table, the monitor change from the first to the last
    monitor point (AUC for PD, R² for LGD; one fixed split per target, not five-fold)."""
    changes = monitor_table_changes(frame[frame.slug == "transfer"] if not frame.empty else frame,
                                    track, scopes=("test", "ood"))
    if changes.empty:
        return pd.DataFrame()
    # Only runs that reached the designed budget enter: an interrupted run's last monitor point
    # would otherwise stand in for the end of training.
    budget = int(load_design()["experiments"]["E7"]["successful_updates"])
    last = changes.groupby("run").successful_updates.transform("max")
    end = changes[(changes.successful_updates == last) & (last >= budget)]
    sources = transfer_sources(track)
    end = end.assign(source=end.arm.map(lambda a: sources[int(a)]))
    end = in_scope(end, "source")
    end = end[(end.scope != "test") | _distinct_sources(end.source, end.table)]
    return end[["model", "arm", "source", "scope", "table", "successful_updates", "change"]].rename(
        columns={"table": "target"}).reset_index(drop=True)


def transfer_matrix(frame: pd.DataFrame, track: str, model: str) -> pd.DataFrame:
    """E7: source (rows) × held-out credit target (columns) monitor change at the end of CPT."""
    long = transfer_long(frame, track)
    if long.empty:
        return pd.DataFrame()
    sub = long[(long.model == model) & (long.scope == "test")]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="source", columns="target", values="change").sort_index()


def _property_space(properties: pd.DataFrame) -> pd.DataFrame:
    """Table properties standardised over the track's tables (rows and features on a log scale)."""
    features = properties.set_index("test_dataset_id").copy()
    features["rows"] = np.log10(features["rows"])
    features["features"] = np.log10(features["features"])
    return (features - features.mean()) / features.std(ddof=0).replace(0, 1)


def _property_distances(properties: pd.DataFrame) -> pd.DataFrame:
    z = _property_space(in_scope(properties, "test_dataset_id"))
    values = z.to_numpy(dtype=float)
    return pd.DataFrame(np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2)),
                        index=z.index, columns=z.index)


def _distinct_sources(sources: pd.Series, targets: pd.Series) -> pd.Series:
    """Exclude historical self-comparisons and pairs drawn from the same lender."""
    from src.train.corpus import SAME_SOURCE
    eligible = sources != targets
    for group in SAME_SOURCE:
        eligible &= ~(sources.isin(group) & targets.isin(group))
    return eligible


def transfer_pairs(long: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    """E7 change and property distance for each eligible source and held-out target."""
    if long.empty or properties.empty:
        return pd.DataFrame()
    distance = _property_distances(properties)
    sub = long[(long.scope == "test") & long.source.isin(distance.index) & long.target.isin(distance.index)
               & _distinct_sources(long.source, long.target)]
    if sub.empty:
        return pd.DataFrame()
    sub = sub[["model", "source", "target", "change"]].copy()
    sub["distance"] = [float(distance.at[s, t]) for s, t in zip(sub.source, sub.target)]
    return sub.reset_index(drop=True)


def transfer_similarity(pairs: pd.DataFrame, properties: pd.DataFrame | None = None, *,
                        permutations: int = BOOTSTRAP_DRAWS, seed: int = 0) -> pd.DataFrame:
    """E7 similarity test per model: Spearman correlation between the change of a target and its
    property distance to the source, over all source-target pairs.

    Pairs that share a table are not independent, so the p-value is a Mantel permutation test:
    the table labels of the distance matrix are permuted jointly (rows and columns) and the
    correlation recomputed; two-sided. A negative correlation means closer sources transfer better."""
    if pairs.empty:
        return pd.DataFrame()
    distance = _property_distances(properties) if properties is not None else None
    pairs = pairs.loc[_distinct_sources(pairs.source, pairs.target) & np.isfinite(pairs.change)
                      & np.isfinite(pairs.distance)]
    rng = np.random.default_rng(seed)
    rows = []
    for model, group in pairs.groupby("model"):
        tables = sorted(set(group.source) | set(group.target))
        if len(tables) < 4 or len(group) < 6:
            continue
        index = {t: i for i, t in enumerate(tables)}
        s, t = group.source.map(index).to_numpy(), group.target.map(index).to_numpy()
        if distance is None:
            matrix = np.full((len(tables), len(tables)), np.nan)
            matrix[s, t] = matrix[t, s] = group.distance.to_numpy(dtype=float)
        else:
            matrix = distance.loc[tables, tables].to_numpy(dtype=float)
        change = pd.Series(group.change.to_numpy(dtype=float))
        observed = change.corr(pd.Series(group.distance.to_numpy(dtype=float)), method="spearman")
        # Every permutation must score exactly the observed comparison set. Distances for
        # excluded pairs can be computed from table properties without using their outcomes.
        null = np.array([])
        if np.isfinite(matrix[~np.eye(len(tables), dtype=bool)]).all():
            null = np.array([change.corr(pd.Series(matrix[p[s], p[t]]), method="spearman")
                             for p in (rng.permutation(len(tables)) for _ in range(permutations))])
        null = null[np.isfinite(null)]
        rows.append(dict(model=model, pairs=len(group), tables=len(tables), spearman=float(observed),
                         p_value=(float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null)))
                                  if np.isfinite(observed) and len(null) else np.nan),
                         mean_change=float(change.mean())))
    return pd.DataFrame(rows)

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
