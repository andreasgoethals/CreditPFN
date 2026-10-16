"""Post-hoc credit diagnostics from saved, paired out-of-fold predictions.

No inference, fitting or test-selected recipes. Mean error and prediction changes
are observable diagnostics, not an estimator bias/context-variance decomposition.
Only aggregate tables leave this module; individual targets remain private inputs.
"""
from __future__ import annotations

import io
import json
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from src.data.dataset_names import display_name, redact_private_names
from src.utils.paths import group_for_run, results_dir
from src.visualize import campaign as cp, style, training_viz
from src.visualize.inputs import ArchivePath, analysis_location, read_csv

KEYS = ["fold_idx", "row_idx"]
NOTE = (
    "Raw predictions are paired by dataset, training partition, outer fold and row. "
    "All five folds and exactly matching test rows/targets are required. Mean error is "
    "prediction minus observed target: positive means overprediction. Its absolute value "
    "can conceal offsetting subgroup errors. Brier/MSE and prediction movement pool rows "
    "within each table; tables receive equal weight in summaries. These are observed "
    "out-of-fold diagnostics, not squared theoretical bias or context variance."
)


def paired_values(original, adapted, track):
    """Validate exact query pairing and return aggregate, row-weighted diagnostics."""
    if track not in ("pd", "lgd"):
        raise ValueError("Expected pd or lgd predictions")
    for frame in (original, adapted):
        if frame.empty or not {*KEYS, "y_true", "y_pred"} <= set(frame):
            raise ValueError("Missing prediction columns or rows")
        if frame.duplicated(KEYS).any() or frame.row_idx.duplicated().any():
            raise ValueError("Repeated test row in out-of-fold predictions")
        if not np.isfinite(frame[[*KEYS, "y_true", "y_pred"]].to_numpy(float)).all():
            raise ValueError("Nonfinite prediction, target or row identity")
        if not (frame[KEYS].to_numpy() == frame[KEYS].to_numpy().astype(int)).all():
            raise ValueError("Noninteger prediction row identity")
        if set(frame.fold_idx) != set(range(5)) or frame.row_idx.lt(0).any():
            raise ValueError("Expected five outer folds and nonnegative row identities")
        if track == "pd" and (not frame.y_true.isin([0, 1]).all()
                              or not frame.y_pred.between(0, 1).all()):
            raise ValueError("Invalid binary targets or probabilities")
    a, b = [frame.set_index(KEYS).sort_index() for frame in (original, adapted)]
    if not a.index.equals(b.index) or not np.array_equal(a.y_true, b.y_true):
        raise ValueError("Unmatched test row identities or targets")
    y, p, q = a.y_true.to_numpy(), a.y_pred.to_numpy(), b.y_pred.to_numpy()
    error_a, error_b, movement = p-y, q-y, q-p
    return dict(test_rows=len(y), target_mean=y.mean(), original_mean=p.mean(), adapted_mean=q.mean(),
                original_mean_error=error_a.mean(), adapted_mean_error=error_b.mean(),
                absolute_mean_error_improvement=abs(error_a.mean())-abs(error_b.mean()),
                original_mse=np.mean(error_a**2), adapted_mse=np.mean(error_b**2),
                mse_improvement=np.mean(error_a**2-error_b**2),
                prediction_shift=movement.mean(), prediction_rms_change=np.sqrt(np.mean(movement**2)),
                prediction_mean_absolute_change=np.mean(abs(movement)))


class _PredictionReader:
    def __init__(self, campaign):
        group = group_for_run(str(campaign.cfg.run_name))
        self.root = analysis_location(group, "results", results_dir(experiment=group))
        self.track = campaign.track
        self.paths = {}
        for path in (self.root / self.track.upper()).rglob("*.csv"):
            key = redact_private_names(str(path.relative_to(self.root)))
            if key in self.paths:
                raise ValueError("Ambiguous prediction source after dataset aliasing")
            self.paths[key] = path

    def load(self, folds):
        pieces = []
        for source, selected in folds.groupby("source_file"):
            metrics = self.paths.get(source)
            if metrics is None:
                raise FileNotFoundError("Metric source is absent from this download")
            receipt_path = metrics.parent / (metrics.name + ".evaluation.json")
            if not receipt_path.is_file():
                raise FileNotFoundError("Prediction receipt is absent from this download")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8")).get("__predictions__", {})
            name = receipt.get("file", "")
            if not name or PurePosixPath(name).name != name or "\\" in name or name in (".", ".."):
                raise ValueError("Invalid prediction receipt filename")
            path = metrics.parent / name
            if not path.is_file():
                raise FileNotFoundError("Prediction artifact is absent from this download")
            if path.stat().st_size != receipt.get("bytes"):
                raise ValueError("Prediction artifact size differs from its receipt")
            if name.endswith(".parquet"):
                frame = pd.read_parquet(io.BytesIO(path.read_bytes()) if isinstance(path, ArchivePath) else path)
            elif name.endswith(".csv.gz"):
                frame = read_csv(path)
            else:
                raise ValueError("Unsupported prediction artifact format")
            if not {"test_dataset_id", "model_name", *KEYS, "y_true", "y_pred"} <= set(frame):
                raise ValueError("Prediction artifact lacks required columns")
            # A large table repeats the same two labels millions of times.
            for column, alias in (("test_dataset_id", display_name), ("model_name", redact_private_names)):
                frame[column] = frame[column].map({value: alias(value) for value in frame[column].unique()})
            for row in selected.itertuples():
                part = frame[frame.test_dataset_id.eq(row.test_dataset_id)
                             & frame.model_name.eq(row.model_name) & frame.fold_idx.eq(row.fold_idx)]
                if len(part) != row.n_test_rows:
                    raise ValueError("Prediction count differs from the recorded test-fold size")
                measured = float(np.mean((part.y_pred.to_numpy()-part.y_true.to_numpy())**2))
                expected = getattr(row, "brier_score", np.nan) if self.track == "pd" else getattr(row, "rmse", np.nan)**2
                if np.isfinite(expected) and not np.isclose(measured, expected, rtol=1e-6, atol=1e-10):
                    raise ValueError("Saved predictions disagree with the recorded fold score")
                pieces.append(part[[*KEYS, "y_true", "y_pred"]])
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _candidates(campaign, reference):
    data = campaign.evaluation
    if data.empty:
        return data
    data = data[data.domain.fillna("credit").eq("credit")].copy() if "domain" in data else data.copy()
    trained = data.source.str.endswith("-trained", na=False)
    if reference:
        recipe = (np.isclose(data.lr, 3e-7, rtol=1e-10, atol=0)
                  & np.isclose(data.l2sp_lambda, .003) & ~data.use_lora.astype(bool)
                  & data.epoch_pass_mode.eq("one_sample"))
        data = data[~trained | recipe]
    return data


def _arm(row, reference):
    if reference:
        return "Reference CPT"
    if row.adaptation_mode == "diagscale":
        return f"Adapter LR {row.lr:g}"
    return style.SAMPLING_LABELS.get(row.epoch_pass_mode, row.epoch_pass_mode)


def analyze(campaign, *, reference=False, arm=None, counterpart=None):
    """Return (paired dataset diagnostics, coverage), optionally comparing two CPT seeds.

    Missing artifacts produce explicit coverage records; malformed/foreign artifacts
    are rejected. Prediction data are processed one table/recipe at a time.
    """
    data = _candidates(campaign, reference)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame([dict(status="unavailable", reason="No benchmark records")])
    other = _candidates(counterpart, True) if counterpart is not None else data
    if counterpart is not None:
        # Reuse the scientific-configuration check; the test folds must still match below.
        from src.visualize.comparisons import seed_pairs
        seed_pairs(counterpart, campaign, benchmark=True)
    left_reader = _PredictionReader(counterpart or campaign)
    right_reader = left_reader if counterpart is None else _PredictionReader(campaign)
    complete = cp._complete_folds(data, campaign.metric)
    other_complete = cp._complete_folds(other, campaign.metric)
    candidates = data[data.source.str.endswith("-trained", na=False)]
    keys = ["base_short", "split", "test_dataset_id", "method_dirname"]
    results, coverage = [], []
    for key, proposed in candidates.groupby(keys, dropna=False):
        row = proposed.iloc[0]
        meta = dict(base=training_viz.compact_base(row.base_short), dataset=row.test_dataset_id,
                    partition=int(row.split), arm=arm or _arm(row, reference))
        right = complete[complete.method_dirname.eq(row.method_dirname)
                         & complete.split.eq(row.split) & complete.test_dataset_id.eq(row.test_dataset_id)]
        source = "-trained" if counterpart is not None else "-untuned"
        left = other_complete[other_complete.source.str.endswith(source, na=False)
                              & other_complete.base_short.eq(row.base_short)
                              & other_complete.split.eq(row.split)
                              & other_complete.test_dataset_id.eq(row.test_dataset_id)]
        if len(left) != 5 or len(right) != 5:
            coverage.append(dict(**meta, status="unavailable", reason="Need one paired model with all five successful folds"))
            continue
        for column in ("split_seed", "n_train_rows", "n_val_rows", "n_test_rows"):
            if column in left and column in right and not np.array_equal(
                    left.sort_values("fold_idx")[column], right.sort_values("fold_idx")[column]):
                raise ValueError("Paired prediction records use different evaluation splits or context budgets")
        try:
            values = paired_values(left_reader.load(left), right_reader.load(right), campaign.track)
        except FileNotFoundError as exc:
            coverage.append(dict(**meta, status="unavailable", reason=str(exc)))
            continue
        results.append(dict(**meta, **values))
        coverage.append(dict(**meta, status="verified", reason="Five folds; matched row identities, targets and artifact sizes"))
    result = pd.DataFrame(results)
    if not result.empty and result.duplicated(["base", "dataset", "arm"]).any():
        raise ValueError("Credit tables must have one held-out training partition per recipe")
    return result, pd.DataFrame(coverage)


def plot_diagnostics(data, track, *, seeds=False, coverage=None):
    """One compact page; dataset heterogeneity is not a confidence interval."""
    if data.empty:
        return []
    # Use the same table cohort for every arm within a base; coverage stays tabulated.
    selected = []
    for base, group in data.groupby("base"):
        expected = (set(coverage.loc[coverage.base.eq(base), "arm"])
                    if coverage is not None and "base" in coverage else set(group.arm))
        common = group.groupby("dataset").arm.agg(set)
        selected.append(group[group.dataset.isin(common[common.map(lambda x: x == expected)].index)])
    data = pd.concat(selected, ignore_index=True)
    if data.empty:
        return []
    fields = (["prediction_rms_change", "prediction_shift"] if seeds else
              ["absolute_mean_error_improvement", "mse_improvement"])
    labels = (["RMS prediction difference", "Mean prediction shift (43 − 42)"] if seeds else
              ["Reduction in absolute\nmean error", "Brier reduction" if track == "pd" else "MSE reduction"])
    title = "Training-seed prediction sensitivity" if seeds else "Systematic error and predictive accuracy"
    fig, axes = cp._subplots(f"{track.upper()}: {title}", 2, sharey=True)
    summaries, ticks = [], []
    for i, ((base, arm), group) in enumerate(data.groupby(["base", "arm"], sort=True)):
        label = style.model_label(base) if data.arm.nunique() == 1 else f"{style.model_label(base)} / {arm}"
        ticks.append(f"{label} (n={len(group)})")
        for ax, field in zip(axes, fields):
            values = group[field]
            q1, q3 = values.quantile([.25, .75])
            ax.plot([q1, q3], [i, i], color=style.color(base), linewidth=style.THIN_LINE)
            ax.scatter(values.mean(), i, color=style.color(base), marker="D", s=style.POINT_SIZE)
            summaries.append(dict(base=base, arm=arm, metric=field, datasets=len(group),
                                  mean=values.mean(), q25=q1, q75=q3))
    for ax, label in zip(axes, labels):
        ax.axvline(0, color=style.color("reference"), linestyle=":", linewidth=style.THIN_LINE)
        ax.set_xlabel(label + ("" if seeds else " (+ better)"))
        ax.set_yticks(range(len(ticks)), ticks)
    caption = (
        "Matched five-fold out-of-fold predictions on held-out credit tables. Diamonds are equal-table means; "
        "segments are dataset interquartile ranges, not confidence intervals. Each label gives the number of tables. "
        + ("Both checkpoints use the fixed reference recipe; only CPT training seed changes (42 versus 43). "
           "RMS disagreement and signed shifts describe two realizations, not context variance."
           if seeds else "Mean error is predicted minus observed target, pooled over test rows within a table. "
           "The left panel measures reduction of its absolute value; the right measures reduction in squared error. "
           "These are empirical errors, not a theoretical bias–variance decomposition.")
    )
    return [cp.Page("seed_prediction_sensitivity" if seeds else "prediction_diagnostics", fig, caption,
                    {"Paired table diagnostics": data, "Plotted means and quartiles": pd.DataFrame(summaries)})]


def plot_arms(tables, coverage, track):
    """A corpus/episode comparison needs every prescribed arm, including shared controls."""
    if not tables or any(table.empty for table in tables):
        return []
    return plot_diagnostics(pd.concat(tables, ignore_index=True), track,
                            coverage=pd.concat(coverage, ignore_index=True))
