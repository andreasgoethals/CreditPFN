"""Corrections from the 16-11-2026 review: leakage in single tables, same-source partitions,
the missing-class guard, seeded TabPFN views and the single-table smoke test."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocessing import DATASET_METADATA, apply_dataset_specific_fixes, read_raw

REPO = Path(__file__).resolve().parents[1]


def _table(prefix: str, track: str) -> str:
    """Registry id by number and task (some ids carry names that must stay out of tracked files)."""
    return next(k for k, m in DATASET_METADATA.items() if k.startswith(prefix) and m["track"] == track)


def test_hackerearth_drops_post_outcome_and_payment_to_date_columns(tmp_path, monkeypatch):
    import hashlib
    from src.data import lendingclub_outcomes as lc
    df = pd.DataFrame({"member_id": [1, 2], "loan_status": [0, 1], "recoveries": [5.0, 0.0],
                       "collection_recovery_fee": [1.0, 0.0], "total_rec_int": [9.0, 3.0],
                       "total_rec_late_fee": [0.0, 1.0], "last_week_pay": ["9th week", "4th week"],
                       "funded_amnt": [100, 200]})
    raw_csv = tmp_path / "pd" / "0006.hackerearth.csv"
    df.attrs["challenge_path"] = raw_csv
    path = lc.key_path(raw_csv)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"member_id,loan_status\n2,0\n")
    monkeypatch.setattr(lc, "KEY_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    out = apply_dataset_specific_fixes(df, "0006.hackerearth")
    assert set(out.columns) == {"loan_status", "funded_amnt"}
    assert out["loan_status"].tolist() == [0] and out["funded_amnt"].tolist() == [200]
    path.write_bytes(b"member_id,loan_status\n2,1\n")
    with pytest.raises(ValueError, match="invalid verified"):
        apply_dataset_specific_fixes(df, "0006.hackerearth")


def test_pd_0009_keeps_one_row_per_loan_and_drops_the_sentinel_column():
    df = pd.DataFrame({"Loan ID": ["a", "a", "b"], "Customer ID": [1, 1, 2],
                       "Loan Status": ["Fully Paid", "Fully Paid", "Charged Off"],
                       "Current Loan Amount": [99999999, 99999999, 5000],
                       "Credit Score": [np.nan, 700.0, 650.0]})
    out = apply_dataset_specific_fixes(df, _table("0009.", "pd"))
    assert len(out) == 2 and "Current Loan Amount" not in out
    assert out["Credit Score"].tolist() == [700.0, 650.0]          # the most complete copy is kept


def test_lgd_0002_drops_realised_mortgage_insurance_cash_flows():
    df = pd.DataFrame({"_ELGD": [0.2, 0.5], "_Mirecovery": [1.0, 0.0], "_Miclaimbal": [3.0, 2.0],
                       "percent_of_primary_pmi_coverage": [25, 0], "UPB_At_Resolution": [9.0, 8.0]})
    out = apply_dataset_specific_fixes(df, _table("0002.", "lgd"))
    assert {"_Mirecovery", "_Miclaimbal"}.isdisjoint(out.columns)
    assert {"percent_of_primary_pmi_coverage", "UPB_At_Resolution"} <= set(out.columns)


def test_german_keeps_its_first_loan(tmp_path):
    path = tmp_path / "german.csv"
    path.write_text("A11,6,1\nA12,48,2\n", encoding="utf-8")
    frame = apply_dataset_specific_fixes(read_raw(path, "0008.german"), "0008.german")
    assert len(frame) == 2 and frame["target"].tolist() == [0, 1]
    raw = REPO / "data" / "raw" / "pd" / "0008.german.csv"
    if raw.is_file():
        frame = apply_dataset_specific_fixes(read_raw(raw, "0008.german"), "0008.german")
        assert len(frame) == 1000 and frame["target"].value_counts().to_dict() == {0: 700, 1: 300}


def test_same_source_tables_share_a_partition_and_sizes_stay_balanced():
    from src.train.corpus import SAME_SOURCE, _assign_folds
    ids = [f"{i:04d}.t" for i in range(1, 18)]
    ids[3], ids[5] = SAME_SOURCE[0]
    for seed in range(40):
        folds = [_assign_folds(ids, n_folds=4, fold=f, seed=seed) for f in range(4)]
        tests = [{d for d, b in fold.items() if b == "test"} for fold in folds]
        assert sorted(len(t) for t in tests) == [4, 4, 4, 5]
        assert set().union(*tests) == set(ids)
        assert any(set(SAME_SOURCE[0]) <= t for t in tests)


def test_missing_class_guard_compares_each_member_in_its_own_label_space():
    from src.train.loop import _query_missing_context_class
    member = lambda y, perm: NS(y_context=torch.tensor(y), class_permutation=perm)  # noqa: E731
    query = torch.tensor([0, 1])
    # Context has only class 0; the second member's permutation relabels it as 1. A union of the
    # members' labels would be {0, 1} and wrongly look complete.
    batch = NS(task_type="classification", y_query=query,
               members=[member([0, 0], None), member([1, 1], np.array([1, 0]))])
    assert _query_missing_context_class(batch)
    batch.members = [member([0, 1], None), member([1, 0], np.array([1, 0]))]
    assert not _query_missing_context_class(batch)


def test_tabpfn_views_depend_on_the_training_seed():
    import inspect

    from src.train import tabpfn_preprocessing as tp
    assert "view_seed" in inspect.signature(tp.clean_loaded_dataset).parameters
    source = inspect.getsource(tp.clean_loaded_dataset)
    assert "view_seed" in source.split("hashlib.blake2b(")[1].split(")")[0]


@pytest.mark.parametrize("own, gated", [((0.85, 0.86, 0.80), True), ((0.80, 0.79, 0.78), False)])
def test_single_table_smoke_test_fails_only_when_every_model_fails(monkeypatch, own, gated):
    from src.utils import pilot_checks as pc
    start, *ends = own
    rows = [dict(status="OK", trial=f"t{i}") for i in range(len(ends))]
    frames = {f"t{i}": pd.DataFrame(dict(record_type="trajectory", train_metric=[start, end],
                                         monitor_seconds=[1.0, 1.0]))
              for i, end in enumerate(ends)}
    monkeypatch.setattr(pc, "_trajectory", lambda row, track: frames[row["trial"]])
    cfg = NS(tunable=NS(l2sp_lambdas=[0.003]), optimizer=NS(l2sp_lambda=0.003), train={})
    checks = pc.pilot_checks("sanity_single", "pd", {"trials": rows}, cfg)
    assert all(c["passed"] for c in checks.values()) is gated
    assert any("finding" in c for c in checks.values()) is gated
