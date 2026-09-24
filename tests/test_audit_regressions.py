"""Research-validity regressions that can be checked without pretrained GPU models."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def test_eval_resume_does_not_borrow_another_split(tmp_path, monkeypatch):
    from src.eval.benchmark import find_existing_results, _method_dirname
    from src.model.base import ModelHandle
    monkeypatch.setenv("CREDITPFN_OUTPUT_ROOT", str(tmp_path))
    h = ModelHandle(name="xgboost", track="pd", source="baseline", task_type="classification")
    folder = tmp_path / "output/general/results/PD" / _method_dirname(h)
    folder.mkdir(parents=True)
    pd.DataFrame([dict(test_dataset_id="demo", fold_idx=0, status="OK")]).to_csv(
        folder / "exp1v2_s00_20260922_120000.csv", index=False)
    assert not find_existing_results(h, "demo", track="pd", results_base_dir="output/general/results",
                                     n_folds_required=1, run_name="exp1v2_s01")
    assert find_existing_results(h, "demo", track="pd", results_base_dir="output/general/results",
                                 n_folds_required=1, run_name="exp1v2_s00")


def test_effects_pair_before_averaging_splits_and_folds():
    from src.visualize.paper_figures import paired_deltas
    common = dict(base_short="v3", test_dataset_id="demo", fold_idx=0)
    frame = pd.DataFrame([
        dict(common, method_dirname="base", source="tabpfn-untuned", split=0, roc_auc=.6),
        dict(common, method_dirname="base", source="tabpfn-untuned", split=1, roc_auc=.9),
        dict(common, method_dirname="trained", source="tabpfn-trained", split=0, roc_auc=.7),
        dict(common, method_dirname="trained", source="tabpfn-trained", split=2, roc_auc=.1),
    ])
    result = paired_deltas(frame)
    assert len(result) == 1
    assert result.iloc[0]["delta"] == pytest.approx(.1)


def test_corpus_guard_refuses_silent_dataset_omission(monkeypatch):
    import src.train.corpus as corpus
    monkeypatch.setattr("src.data.preprocessing.DATASET_METADATA", {
        "one": {"track": "pd"}, "two": {"track": "pd"}})
    monkeypatch.setattr(corpus, "build_dataset_pool", lambda track: [SimpleNamespace(dataset_id="one")])
    with pytest.raises(ValueError, match="Incomplete pd corpus"):
        corpus.split_corpus(track="pd", require_complete_registry=True)


def test_trained_eval_rejects_real_provenance_train_test_overlap(tmp_path):
    from src.eval.benchmark import resolve_test_datasets
    from src.model.base import ModelHandle
    path = tmp_path / "model.ckpt"
    Path(str(path) + ".provenance.json").write_text(json.dumps({
        "training_datasets": ["demo"], "test_datasets": ["demo"]}))
    handle = ModelHandle(name="trained", track="pd", source="tabpfn-trained", task_type="classification", base_path=path)
    with pytest.raises(ValueError, match="overlaps"):
        resolve_test_datasets(handle, cfg_test_dataset_ids=["demo"])


def test_test_monitor_cannot_change_training_divergence_decision():
    from src.train.loop import EpochRecord, _divergence_reason
    rows = [EpochRecord(epoch=i, train_loss=1 / (i + 1), elapsed_sec=i, lr=1e-6) for i in range(5)]
    for test_score in (.5, .9, float("nan")):
        assert _divergence_reason(rows, [(test_score, .75)] * 5, 5, "roc_auc", 1) is None
        assert _divergence_reason(rows, [(test_score, .5)] * 5, 5, "roc_auc", 1) == "auc_random"


def test_structured_logger_preserves_traceback():
    import logging
    import sys
    from src.utils.logging_setup import _StructuredFormatter
    try:
        raise RuntimeError("diagnostic marker")
    except RuntimeError:
        record = logging.LogRecord("test", logging.ERROR, "test.py", 1, "trial failed", (), sys.exc_info())
    text = _StructuredFormatter(use_color=False).format(record)
    assert "Traceback" in text and "RuntimeError: diagnostic marker" in text


def test_exploration_reconstructs_missing_manifest_without_writing(tmp_path, monkeypatch):
    from src.data import exploration, preprocessing, register
    metadata = {"demo": dict(track="pd", task_type="classification", target_column="y",
                             source="test", source_url="", categorical_columns=[])}
    monkeypatch.setattr(preprocessing, "DATASET_METADATA", metadata)
    monkeypatch.setattr(register, "DATASET_METADATA", metadata)
    monkeypatch.setattr(preprocessing, "apply_dataset_specific_fixes", lambda frame, _: frame)
    raw = tmp_path / "raw" / "pd"
    raw.mkdir(parents=True)
    pd.DataFrame({"x": [1, 2, 3], "y": [0, 0, 1]}).to_csv(raw / "demo.csv", index=False)
    cfg = SimpleNamespace(paths=SimpleNamespace(raw=str(raw.parent), processed=str(tmp_path / "processed"),
        manifest_pd=str(tmp_path / "pd.csv"), manifest_lgd=str(tmp_path / "lgd.csv")))
    exploration.clear_summary_cache()
    result = exploration.load_manifests(cfg)
    assert result["pd"].iloc[0]["n_rows"] == 3
    assert result["pd"].iloc[0]["n_cols"] == 1
    assert float(result["pd"].iloc[0]["minority_class_ratio"]) == pytest.approx(1 / 3, abs=1e-6)
    assert result["lgd"].empty
    assert not Path(cfg.paths.manifest_pd).exists()
    exploration.clear_summary_cache()
