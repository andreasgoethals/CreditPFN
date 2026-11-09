"""The research retention panel is the agreed TabArena-v0.1 subset, prepared like credit tables."""
import json

import numpy as np
import pandas as pd
import pytest

from src.data.retention import load_refs, panel_config

EXCLUDED = {46918, 46932, 46919, 46929, 46962, 46950, 46938}
BINARY = {46913, 46921, 46952, 46927, 46940, 46930, 46956, 46912, 46915, 46969, 46916,
          46911, 46979, 46924, 46947, 46937, 46935, 46905, 46910}
REGRESSION = {46954, 46917, 46931, 46904, 46907, 46953, 46964, 46942, 46934, 46961, 46928, 46949}


@pytest.fixture(scope="module")
def research():
    return panel_config("research")


def test_panel_is_the_agreed_19_binary_and_12_regression_tables(research):
    pd_ids = {e["data_id"] for e in research if e["track"] == "pd"}
    lgd_ids = {e["data_id"] for e in research if e["track"] == "lgd"}
    assert (len(pd_ids), len(lgd_ids)) == (19, 12)
    assert pd_ids == BINARY and lgd_ids == REGRESSION


def test_no_multiclass_and_task_matches_track(research):
    for e in research:
        assert e["tabarena"]["classes"] == (2 if e["track"] == "pd" else None), e["id"]


def test_every_table_has_fewer_than_50000_rows(research):
    assert all(e["tabarena"]["n"] < 50000 for e in research)


def test_no_credit_overlap_or_credit_like_table(research):
    assert not EXCLUDED & {e["data_id"] for e in research}


def test_ids_are_unique_and_consistent(research):
    assert len({e["id"] for e in research}) == len({e["data_id"] for e in research}) == 31
    assert all(e["id"] == f"openml_{e['data_id']}" and e["source"] == "openml" for e in research)


def test_no_name_looks_like_credit_risk(research):
    words = ("credit", "loan", "default", "bankrupt", "lending")
    assert not [e["name"] for e in research if any(w in e["name"].lower() for w in words)]


def test_research_tables_use_the_credit_sanitize_pipeline(research):
    assert all(e["preprocessing"] == "credit_sanitize" for e in research)


def test_smoke_panel_stays_small_and_paired():
    smoke = panel_config("smoke")
    assert len(smoke) == 2 and {e["track"] for e in smoke} == {"pd", "lgd"}


def test_wide_tables_are_capped_and_regression_targets_are_not_clipped():
    from src.utils.prepare_retention import sanitize_frame
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 1777)), columns=[f"f{i}" for i in range(1777)])
    X["colour"] = pd.Categorical(rng.choice(["a", "b", "c"], size=200))
    y = rng.normal(50, 20, size=200)
    item = {"id": "openml_0", "track": "lgd"}
    frame, cats, info = sanitize_frame(X, y, item, ["colour"])
    assert frame.shape[1] - 1 <= info["max_columns"] == 64
    assert info["source_features"] == 1778
    assert np.allclose(frame["target"], y) and frame["target"].max() > 1
    assert set(cats) <= {"colour"}


def test_prepare_supersedes_changed_definitions_without_deleting(tmp_path, monkeypatch):
    from src.utils import prepare_retention as prep
    item = {"id": "package_diabetes", "track": "lgd", "source": "sklearn", "name": "diabetes"}
    monkeypatch.setattr(prep, "panel_config", lambda panel: [item])
    first = prep.prepare(("smoke",), root=tmp_path)[0]
    assert prep.prepare(("smoke",), root=tmp_path)[0]["sha256"] == first["sha256"]
    item["preprocessing"] = "credit_sanitize"
    with pytest.raises(ValueError, match="supersede"):
        prep.prepare(("smoke",), root=tmp_path)
    again = prep.prepare(("smoke",), root=tmp_path, supersede_changed=True)[0]
    assert again["preprocessing"] == "credit_sanitize" and again["max_columns"] == 64
    moved = list((tmp_path / "superseded").glob("*/package_diabetes.*"))
    assert len(moved) == 2


def test_load_refs_reads_prepared_research_metadata(tmp_path, monkeypatch):
    from src.data import retention
    from src.utils.experiment import file_digest
    item = panel_config("research")[0]
    path = tmp_path / (item["id"] + ".csv.gz")
    pd.DataFrame({"x": [1.0, 2.0], "target": [0, 1]}).to_csv(path, index=False)
    meta = dict(item, file=path.name, rows=2, categorical_columns=[], sha256=file_digest(path))
    (tmp_path / (item["id"] + ".json")).write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(retention, "resolve_data_path", lambda p: tmp_path)
    monkeypatch.setattr(retention, "panel_config", lambda panel: [item])
    [ref] = load_refs("research", "pd")
    assert ref.dataset_id == item["id"] and ref.task_type == "classification"
