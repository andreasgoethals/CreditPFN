"""The matched non-credit control corpus of E3."""
import re
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from src.data import control_matching as cm


def test_committed_panel_is_the_optimal_matching_of_its_recorded_profiles():
    panel = cm.load_panel()
    profiles = cm.recorded_profiles(panel)
    assert cm.render(cm.build(profiles)) == cm.PANEL.read_text(encoding="utf-8")
    assert {t: len(p) for t, p in profiles.items()} == {"pd": 17, "lgd": 8}   # one twin per credit table
    ids = [item["data_id"] for item in panel["domain_control"]]
    assert len(ids) == len(set(ids))


def test_recorded_profiles_equal_the_processed_credit_tables():
    try:
        live = {track: cm.credit_profile(track) for track in ("pd", "lgd")}
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"processed credit tables unavailable: {exc}")
    if not live["pd"]:
        pytest.skip("processed credit tables unavailable")
    assert live == cm.recorded_profiles(cm.load_panel()), "run python -m src.data.control_matching --write"


def test_control_never_uses_credit_finance_or_forgetting_panel_tables():
    from src.data.retention import panel_config
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
    research = {norm(i["name"]) for i in panel_config("research")} | {i["data_id"] for i in panel_config("research")}
    for item in cm.load_panel()["domain_control"]:
        assert norm(item["name"]) not in research and item["data_id"] not in research
        assert not re.search("credit|bank|insur|loan|bankrupt", item["name"], re.I)


def test_cost_is_zero_only_for_a_large_enough_twin():
    credit = {"twin": "0001", "rows": 1000, "features": 10, "categorical": 2}
    big = {"rows": 1000, "features": 10, "categorical": 2}
    assert cm.cost(credit, big) == 0
    assert cm.cost(credit, dict(big, rows=500)) == pytest.approx(np.log(2))
    assert cm.cost(credit, dict(big, categorical=0)) == pytest.approx(0.2)    # two categoricals short
    assert 0 < cm.cost(credit, dict(big, rows=100000)) < 0.3


def test_matching_is_one_to_one_and_optimal():
    profile = [{"twin": "a", "rows": 100, "features": 5, "categorical": 0},
               {"twin": "b", "rows": 10000, "features": 5, "categorical": 0}]
    candidates = [{"data_id": 1, "name": "big", "rows": 20000, "features": 5, "categorical": 0},
                  {"data_id": 2, "name": "small", "rows": 150, "features": 5, "categorical": 0}]
    twins = {t["twin"]: t["data_id"] for t in cm.match(profile, candidates)}
    assert twins == {"a": 2, "b": 1}


def test_apply_match_subsamples_rows_by_class_and_features_by_type():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({f"n{i}": rng.normal(size=2000) for i in range(6)})
    for i in range(3):
        frame[f"c{i}"] = rng.integers(0, 4, 2000)
    frame["target"] = (rng.random(2000) < 0.2).astype(int)
    spec = {"twin": "0001", "rows": 500, "features": 5, "categorical": 2}
    out, cats = cm.apply_match(frame, ["c0", "c1", "c2"], spec, "pd", seed=7)
    assert len(out) == 500 and len(out.columns) == 6 and len(cats) == 2
    assert out.target.mean() == pytest.approx(frame.target.mean(), abs=0.01)
    again, _ = cm.apply_match(frame, ["c0", "c1", "c2"], spec, "pd", seed=7)
    pd.testing.assert_frame_equal(out, again)
    short, cats = cm.apply_match(frame, ["c0"], dict(spec, features=4, categorical=3), "lgd", seed=1)
    assert len(short.columns) == 5 and cats == ["c0"]                 # missing categoricals filled by numerics


def test_partition_trains_on_the_twins_of_its_own_credit_training_tables(monkeypatch):
    from src.data import retention
    from src.train import corpus
    panel = [{"id": f"openml_{n}", "track": "pd", "match": {"twin": f"000{n}"}} for n in range(1, 5)]
    monkeypatch.setattr(retention, "panel_config", lambda name: panel)
    monkeypatch.setattr(retention, "load_refs", lambda name, track: [NS(dataset_id=p["id"]) for p in panel])
    split = corpus.CorpusSplit(train=[NS(dataset_id="0003.x"), NS(dataset_id="0001.y")],
                               test=[NS(dataset_id="0002.z")])
    out = corpus.training_source_split(split, "domain_control", "pd")
    assert [r.dataset_id for r in out.train] == ["openml_3", "openml_1"]
    assert out.test == split.test


def test_openml_downloads_retry_transient_errors(monkeypatch, tmp_path):
    import urllib.error

    from src.utils import prepare_retention as pr
    calls, waits = [], []

    def flaky(**kwargs):
        calls.append(kwargs["data_id"])
        if len(calls) < 3:
            raise urllib.error.HTTPError("u", 504, "Gateway Time-out", {}, None)
        return "bunch"
    monkeypatch.setattr(pr.datasets, "fetch_openml", flaky)
    assert pr.fetch_openml(7, tmp_path, backoff=(1, 2, 3), sleep=waits.append) == "bunch"
    assert calls == [7, 7, 7] and waits == [1, 2]
    monkeypatch.setattr(pr.datasets, "fetch_openml", lambda **k: (_ for _ in ()).throw(ValueError("md5")))
    with pytest.raises(ValueError):
        pr.fetch_openml(7, tmp_path, backoff=(1,), sleep=waits.append)


def test_twins_respect_the_download_bound():
    pool = {c["data_id"]: c for track in ("pd", "lgd") for c in cm.candidates(track)}
    for item in cm.load_panel()["domain_control"]:
        candidate = pool[item["data_id"]]
        assert candidate["rows"] * candidate["features"] <= cm.MAX_CELLS


@pytest.mark.parametrize("config", ["E03_domain_control/pd_a0.yaml", "E03_domain_control/lgd_a0.yaml",
                                    "E08_table_size/pd_a0.yaml", "E08_table_size/pd_a1.yaml"])
def test_modified_training_corpora_still_prepare(config):
    """Folds partition the credit registry even when the training corpus is replaced or filtered."""
    from pathlib import Path

    from src.utils.prepare_experiment import prepare
    try:
        report = prepare(Path("config/experiments") / config)
    except FileNotFoundError as exc:
        pytest.skip(f"inputs unavailable: {exc}")
    assert report["partitions"] == 4 and sum(report["test_counts"]) in (17, 8)
