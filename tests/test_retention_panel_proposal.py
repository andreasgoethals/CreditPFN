"""The proposed retention panel stays small, non-credit and free of corpus overlap."""
from pathlib import Path

from omegaconf import OmegaConf

PANEL = OmegaConf.to_container(OmegaConf.load(
    Path(__file__).resolve().parents[1] / "config/retention_panel_proposed.yaml"))
GROUPS = ("classification_binary", "classification_multiclass_candidates", "regression")


def entries():
    return [e for group in GROUPS for e in PANEL[group]]


def test_every_panel_table_has_fewer_than_25000_rows():
    assert all(e["rows"] < 25000 for e in entries())


def test_no_panel_table_is_excluded_for_credit_overlap():
    excluded = {e["data_id"] for e in PANEL["excluded_credit_overlap"] + PANEL["excluded_borderline"]}
    assert not excluded & {e["data_id"] for e in entries()}


def test_no_panel_name_looks_like_credit_risk():
    words = ("credit", "loan", "default", "bankrupt", "heloc", "lending")
    assert not [e["name"] for e in entries() if any(w in e["name"].lower() for w in words)]


def test_panel_ids_are_unique():
    ids = [e["data_id"] for e in entries()]
    assert len(ids) == len(set(ids))
