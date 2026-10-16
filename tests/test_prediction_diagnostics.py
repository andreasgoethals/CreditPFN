"""Synthetic per-row evidence checks; never depend on private campaign output."""
import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.visualize import campaign as cp, prediction_diagnostics as pred
from src.visualize.inputs import ArchivePath
from src.visualize.reporting import page_text


def predictions():
    return pd.DataFrame(dict(fold_idx=np.repeat(range(5), 2), row_idx=range(10),
                             y_true=[0, 1]*5, y_pred=[.1, .7]*5))


def test_observed_error_is_not_context_variance():
    a = predictions()
    b = a.assign(y_pred=[.2, .8]*5).sample(frac=1, random_state=42)
    result = pred.paired_values(a, b, "pd")
    assert result["test_rows"] == 10
    assert result["original_mean_error"] == pytest.approx(-.1)
    assert result["adapted_mean_error"] == pytest.approx(0)
    assert result["original_mse"] == pytest.approx(.05)
    assert result["adapted_mse"] == pytest.approx(.04)
    assert result["prediction_rms_change"] == pytest.approx(.1)
    assert result["mse_improvement"] == pytest.approx(.01)
    assert not any("variance" in k or "bias" in k for k in result)
    assert pred.paired_values(a, a, "lgd")["prediction_rms_change"] == 0


@pytest.mark.parametrize("change,reason", [
    (lambda d: pd.concat([d, d.iloc[:1]]), "Repeated"),
    (lambda d: d.assign(row_idx=[0]*10), "Repeated"),
    (lambda d: d.assign(y_pred=[np.nan]*10), "Nonfinite"),
    (lambda d: d.assign(fold_idx=d.fold_idx.replace(4, 5)), "five outer folds"),
    (lambda d: d.assign(row_idx=d.row_idx+.1), "Noninteger"),
    (lambda d: d.assign(y_pred=[1.1]*10), "Invalid binary"),
    (lambda d: d.assign(row_idx=d.row_idx+10), "Unmatched"),
    (lambda d: d.assign(y_true=1-d.y_true), "Unmatched"),
])
def test_pairing_refuses_missing_foreign_or_invalid_rows(change, reason):
    with pytest.raises(ValueError, match=reason):
        pred.paired_values(predictions(), change(predictions()), "pd")


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(pred, "analysis_location", lambda *a, **k: tmp_path)
    rows = []
    for state, values in (("untuned", [.1, .7]*5), ("trained", [.2, .8]*5)):
        method = f"tabpfn-{state}__v3"
        folder = tmp_path / "PD" / method
        folder.mkdir(parents=True)
        metrics = folder / "run_s00_20261016.csv"
        metrics.write_text("metric placeholder", encoding="utf-8")
        p = predictions().assign(test_dataset_id="Table", model_name=state, y_pred=values)
        p.to_csv(metrics.with_suffix(".csv.gz"), index=False, compression="gzip")
        receipt = {"__predictions__": {"file": metrics.with_suffix(".csv.gz").name,
                                     "bytes": metrics.with_suffix(".csv.gz").stat().st_size}}
        (folder / (metrics.name+".evaluation.json")).write_text(json.dumps(receipt), encoding="utf-8")
        for fold in range(5):
            rows.append(dict(source=f"tabpfn-{state}", base_short="v3", split=0,
                             domain="credit", eval_run="run_s00", test_dataset_id="Table", model_name=state,
                             method_dirname=method, fold_idx=fold, status="OK", roc_auc=.8, n_test_rows=2,
                             lr=3e-7, l2sp_lambda=.003, use_lora=False, epoch_pass_mode="one_sample",
                             adaptation_mode="full", source_file=str(metrics.relative_to(tmp_path))))
    return cp.Campaign(SimpleNamespace(run_name="cpt_main_v5", track="pd"), pd.DataFrame(),
                       pd.DataFrame(), pd.DataFrame(), pd.DataFrame(rows))


def test_receipts_and_folds_are_required(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    result, coverage = pred.analyze(run, reference=True)
    assert len(result) == 1 and coverage.status.eq("verified").all()
    run.evaluation.loc[0, "status"] = "FAIL"
    result, coverage = pred.analyze(run, reference=True)
    assert result.empty and coverage.status.eq("unavailable").all()


def test_missing_predictions_are_unavailable_not_zero(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    next(tmp_path.rglob("*.csv.gz")).unlink()
    result, coverage = pred.analyze(run, reference=True)
    assert result.empty and coverage.status.eq("unavailable").all()


def test_receipt_size_mismatch_fails_closed(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    path = next(tmp_path.rglob("*.evaluation.json"))
    meta = json.loads(path.read_text()); meta["__predictions__"]["bytes"] += 1
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact size"):
        pred.analyze(run, reference=True)


def test_incomplete_artifact_cannot_pass_using_its_metrics(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    path = next(tmp_path.rglob("*.csv.gz"))
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False, compression="gzip")
    receipt = path.parent / (path.name.removesuffix(".gz")+".evaluation.json")
    receipt.write_text(json.dumps({"__predictions__": {"file":path.name, "bytes":path.stat().st_size}}))
    with pytest.raises(ValueError, match="test-fold size"):
        pred.analyze(run, reference=True)


def test_changed_evaluation_context_is_not_a_cpt_effect(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    run.evaluation["n_train_rows"] = 100
    run.evaluation.loc[0, "n_train_rows"] = 99
    with pytest.raises(ValueError, match="context budgets"):
        pred.analyze(run, reference=True)


def test_predictions_must_reproduce_recorded_metric(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    run.evaluation["brier_score"] = .99
    with pytest.raises(ValueError, match="recorded fold score"):
        pred.analyze(run, reference=True)


def test_zip_reader_keeps_predictions_in_the_archive(tmp_path, monkeypatch):
    run = fixture(tmp_path, monkeypatch)
    archive = tmp_path / "results.zip"
    files = list((tmp_path / "PD").rglob("*"))
    with zipfile.ZipFile(archive, "w") as out:
        for path in files:
            if path.is_file():
                out.write(path, path.relative_to(tmp_path).as_posix())
    run.evaluation["source_file"] = run.evaluation.source_file.str.replace("\\", "/", regex=False)
    monkeypatch.setattr(pred, "analysis_location", lambda *a, **k: ArchivePath(archive))
    result, coverage = pred.analyze(run, reference=True)
    assert len(result) == 1 and coverage.status.eq("verified").all()


def test_plot_is_compact_complete_and_has_no_unmeasured_variance(tmp_path, monkeypatch):
    result, coverage = pred.analyze(fixture(tmp_path, monkeypatch), reference=True)
    pages = pred.plot_diagnostics(result, "pd", coverage=coverage)
    assert len(pages) == 1 and len(pages[0].figure.axes) == 2
    assert "original_mean_error" in page_text(pages[0])
    assert "not a theoretical bias" in pages[0].caption
    assert pages[0].figure.get_figwidth() == pytest.approx(6.75)
    missing = coverage.assign(arm="Missing arm", status="unavailable")
    assert not pred.plot_diagnostics(result, "pd", coverage=pd.concat([coverage, missing]))
    assert not pred.plot_arms([result, pd.DataFrame()], [coverage], "pd")
