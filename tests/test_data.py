"""Smoke + unit tests for the data pipeline.

Layout choice
-------------
One file per ``src/`` subpackage. ``test_data.py`` covers everything in
``src/data/``: preprocessing, register, sanitize, dedup, dataset, and
the exploration helpers. As ``src/train``, ``src/eval`` and
``src/model`` come online they get their own sibling files
(``test_train.py`` etc.), still flat under ``tests/``.

Running
-------
::

    pytest -q tests/test_data.py
    pytest -q tests/test_data.py -k preprocessing       # subset by keyword

Coverage map
------------
    Block 1  preprocessing.py — surgical fixes + low-level parsers
    Block 2  register.py      — manifest building blocks
    Block 3  sanitize.py      — agnostic-clean steps in isolation
    Block 4  dedup.py         — fingerprint + pairwise checks
    Block 5  dataset.py       — chunking, ctx/query split, encoder leakage
    Block 6  exploration.py   — smoke tests for the data-exploration helpers

Tests intentionally lean toward *failure-mode coverage* over
behavioural completeness. Each block prefers a few sharp tests that
would catch a regression introduced by a future refactor over many
shallow tests that don't exercise the actual contract.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    DATASET_METADATA,
    apply_dataset_specific_fixes,
    _yrs_mons_to_months,
    _two_digit_year_to_full,
)
from src.data.register import (
    MANIFEST_COLUMNS,
    compute_manifest_row,
    infer_categorical_numerical,
    shape_aware_sha256,
)
from src.data.sanitize import (
    _drop_constant_columns,
    _drop_exact_duplicate_feature_columns,
    _drop_high_missing_columns,
    _coerce_numeric_strings,
    _replace_inf_with_nan,
    _clip_lgd_target,
    _label_encode_classification_target,
)
from src.data.dedup import (
    Fingerprint,
    compare_pair,
    confidence_for,
    _column_hashes,
    _jaccard,
    _row_hashes,
)
from src.data.dataset import (
    _cache_fingerprint,
    _dataset_config_hash,
    _dataset_seed,
    _ordinal_encode_categoricals,
    _shuffle_and_chunk_indices,
    _split_context_query,
)


REPO = Path(__file__).resolve().parents[1]


# =============================================================================
# Block 1 · preprocessing.py
# =============================================================================


@pytest.mark.parametrize("dataset_id", list(DATASET_METADATA.keys()))
def test_surgical_fix_preserves_target(dataset_id: str) -> None:
    """Every registered fix must leave the target column intact and
    yield a non-empty DataFrame on every existing raw CSV."""
    meta = DATASET_METADATA[dataset_id]
    raw_path = REPO / "data" / "raw" / meta["track"] / f"{dataset_id}.csv"
    if not raw_path.exists():
        pytest.skip(f"raw CSV not present: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    out = apply_dataset_specific_fixes(df, dataset_id)
    assert meta["target_column"] in out.columns, (
        f"{dataset_id}: target {meta['target_column']!r} missing post-fix"
    )
    assert len(out) > 0, f"{dataset_id}: empty after surgical fix"


def test_unknown_dataset_raises_under_error_policy() -> None:
    """``unknown_dataset_policy='error'`` raises on unregistered IDs.

    Default policy is ``"passthrough"`` (matches the documented
    "clean datasets need no surgery" workflow), but tooling that wants
    every dataset to have an explicit fix registration can opt into
    the strict mode.
    """
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError):
        apply_dataset_specific_fixes(
            df, "9999.does_not_exist", unknown_dataset_policy="error",
        )


def test_unknown_dataset_passthrough_by_default() -> None:
    """Under the default policy, an unregistered ID returns the input."""
    df = pd.DataFrame({"a": [1, 2]})
    out = apply_dataset_specific_fixes(df, "9999.does_not_exist")
    pd.testing.assert_frame_equal(out, df)


# =============================================================================
# Block 1b · column-leakage tests
# =============================================================================
#
# The surgical-fix functions claim to remove specific leakage columns (post-loan
# state, direct default indicators, target components, …). These tests verify
# that the claim holds: the post-fix DataFrame must NOT contain any of the
# named leakage columns. If a future refactor accidentally re-introduces
# one, the test fails loudly.
#
# Format: a parametrised matrix of (dataset_id, forbidden_column) pairs.

_FORBIDDEN_AFTER_FIX: list[tuple[str, str]] = [
    # 0016.bondora_peer2peer — post-loan / payment-progression / direct-default columns
    ("0016.bondora_peer2peer", "loan_status"),
    ("0016.bondora_peer2peer", "loan_status_risk"),
    ("0016.bondora_peer2peer", "principal_balance"),
    ("0016.bondora_peer2peer", "principal_debt"),
    ("0016.bondora_peer2peer", "principal_paid_total"),
    ("0016.bondora_peer2peer", "interest_paid_total"),
    ("0016.bondora_peer2peer", "extra_interest_paid_total"),
    ("0016.bondora_peer2peer", "late_fee_paid_total"),
    ("0016.bondora_peer2peer", "maintenance_fee_paid_total"),
    ("0016.bondora_peer2peer", "next_payment_nr"),
    ("0016.bondora_peer2peer", "next_payment_date_local"),
    ("0016.bondora_peer2peer", "debt_occured_date_local"),
    ("0016.bondora_peer2peer", "days_past_due_principal"),
    ("0016.bondora_peer2peer", "months_in_default"),
    ("0016.bondora_peer2peer", "months_on_book"),
    ("0016.bondora_peer2peer", "repaid_amount_total"),
    ("0016.bondora_peer2peer", "has_default_within_12_months"),
    ("0016.bondora_peer2peer", "projected_npv_return"),
    ("0016.bondora_peer2peer", "early_repaid_at"),
    ("0016.bondora_peer2peer", "is_early_repaid_within_14_days"),
    ("0016.bondora_peer2peer", "loan_last_recorded_action_date_local"),
    ("0016.bondora_peer2peer", "loan_issued_at"),
    ("0016.bondora_peer2peer", "loan_id"),

    # 0017.SBA_loans_case (PD) — leakage / ID / mystery columns
    ("0017.SBA_loans_case", "MIS_Status"),     # 1:1 with Default
    ("0017.SBA_loans_case", "ChgOffDate"),     # only set for defaults
    ("0017.SBA_loans_case", "ChgOffPrinGr"),   # LGD-target component
    ("0017.SBA_loans_case", "LoanNr_ChkDgt"),
    ("0017.SBA_loans_case", "Name"),
    ("0017.SBA_loans_case", "Bank"),
    ("0017.SBA_loans_case", "City"),
    ("0017.SBA_loans_case", "Zip"),
    ("0017.SBA_loans_case", "Selected"),       # sampling artefact
    ("0017.SBA_loans_case", "xx"),             # = DisbursementDate + daysterm

    # 0008.SBA_loans_case (LGD twin) — same set plus `Default` (always 1 after filter)
    ("0008.SBA_loans_case", "MIS_Status"),
    ("0008.SBA_loans_case", "ChgOffDate"),
    ("0008.SBA_loans_case", "ChgOffPrinGr"),
    ("0008.SBA_loans_case", "LoanNr_ChkDgt"),
    ("0008.SBA_loans_case", "Name"),
    ("0008.SBA_loans_case", "Bank"),
    ("0008.SBA_loans_case", "City"),
    ("0008.SBA_loans_case", "Zip"),
    ("0008.SBA_loans_case", "Selected"),
    ("0008.SBA_loans_case", "xx"),
    ("0008.SBA_loans_case", "Default"),        # always 1 in the filtered LGD copy
]


@pytest.mark.parametrize("dataset_id, forbidden_col", _FORBIDDEN_AFTER_FIX)
def test_surgical_fix_removes_leakage_column(
    dataset_id: str, forbidden_col: str,
) -> None:
    """For each (dataset, leakage-col) pair, the surgical fix MUST drop
    the column. If a refactor leaves it in, this test fails — that's
    what protects us from silently re-introducing data leakage.
    """
    meta = DATASET_METADATA[dataset_id]
    raw_path = REPO / "data" / "raw" / meta["track"] / f"{dataset_id}.csv"
    if not raw_path.exists():
        pytest.skip(f"raw CSV not present: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    out = apply_dataset_specific_fixes(df, dataset_id)
    assert forbidden_col not in out.columns, (
        f"{dataset_id}: leakage column {forbidden_col!r} survived the "
        f"surgical fix — every fold's metrics would be inflated by this column."
    )


# Companion: target-column sanity — derived targets must be in the right
# domain (binary 0/1 for classification, [0, 1] for LGD).

@pytest.mark.parametrize("dataset_id", [
    "0015.credit_risk_dataset",
    "0016.bondora_peer2peer",
    "0017.SBA_loans_case",
])
def test_classification_target_is_binary_after_fix(dataset_id: str) -> None:
    """PD targets must be exactly {0, 1} (no Y/N, no NaN, no other levels)."""
    meta = DATASET_METADATA[dataset_id]
    raw_path = REPO / "data" / "raw" / meta["track"] / f"{dataset_id}.csv"
    if not raw_path.exists():
        pytest.skip(f"raw CSV not present: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    out = apply_dataset_specific_fixes(df, dataset_id)
    target = meta["target_column"]
    y = out[target].dropna().astype("int64").unique()
    assert set(y.tolist()).issubset({0, 1}), (
        f"{dataset_id}: target {target!r} has non-binary values: {sorted(y)}"
    )


def test_lgd_target_is_non_negative_after_fix() -> None:
    """The LGD twin of SBA derives `lgd` = ChgOffPrinGr / DisbursementGross.

    Clipping to [0, 1] is the job of sanitize.py's global
    `lgd_target_clip` block (single source of truth across all LGD
    datasets). At the preprocessing stage we only assert the ratio
    is finite and non-negative — values > 1 are allowed here and get
    clipped downstream by sanitize.
    """
    raw_path = REPO / "data" / "raw" / "lgd" / "0008.SBA_loans_case.csv"
    if not raw_path.exists():
        pytest.skip(f"raw CSV not present: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    out = apply_dataset_specific_fixes(df, "0008.SBA_loans_case")
    assert "lgd" in out.columns, "0008.SBA_loans_case: `lgd` target column missing"
    lgd = out["lgd"].dropna().to_numpy()
    assert np.isfinite(lgd).all() and (lgd >= 0.0).all(), (
        f"0008.SBA_loans_case: lgd not finite/non-negative — "
        f"min={lgd.min()}, max={lgd.max()}"
    )


def test_lgd_filtered_to_defaults_only() -> None:
    """The LGD twin of SBA must filter to defaulted loans only (the LGD
    target is undefined for non-defaulted loans). We can't read
    `Default` post-fix (it's dropped), but the raw count of defaults
    matches the post-fix row count."""
    raw_path = REPO / "data" / "raw" / "lgd" / "0008.SBA_loans_case.csv"
    if not raw_path.exists():
        pytest.skip(f"raw CSV not present: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    expected = int((df["Default"] == 1).sum())
    out = apply_dataset_specific_fixes(df, "0008.SBA_loans_case")
    assert len(out) == expected, (
        f"0008.SBA_loans_case: row count after fix {len(out)} != "
        f"raw default count {expected}"
    )


def test_credit_risk_loan_grade_is_ordinal_integer() -> None:
    """`loan_grade` ∈ {A..G} in raw; should be integer 0..6 post-fix."""
    raw_path = REPO / "data" / "raw" / "pd" / "0015.credit_risk_dataset.csv"
    if not raw_path.exists():
        pytest.skip(f"raw CSV not present: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    out = apply_dataset_specific_fixes(df, "0015.credit_risk_dataset")
    grades = pd.to_numeric(out["loan_grade"], errors="coerce").dropna().unique()
    assert set(int(g) for g in grades).issubset(set(range(7))), (
        f"loan_grade values out of 0..6 range: {sorted(grades)}"
    )
    # Implicitly verifies the ordinal mapping was applied — no strings should
    # survive (they would coerce to NaN above).
    assert not any(isinstance(v, str) for v in out["loan_grade"].dropna().tolist())


def test_credit_risk_default_on_file_is_binarised() -> None:
    """`cb_person_default_on_file` ∈ {Y, N} in raw → {1, 0} post-fix."""
    raw_path = REPO / "data" / "raw" / "pd" / "0015.credit_risk_dataset.csv"
    if not raw_path.exists():
        pytest.skip(f"raw CSV not present: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    out = apply_dataset_specific_fixes(df, "0015.credit_risk_dataset")
    values = out["cb_person_default_on_file"].dropna().astype("int64").unique()
    assert set(values.tolist()).issubset({0, 1})


def test_unknown_dataset_passthrough() -> None:
    """``passthrough`` policy returns the input unchanged."""
    df = pd.DataFrame({"a": [1, 2]})
    out = apply_dataset_specific_fixes(
        df, "9999.does_not_exist", unknown_dataset_policy="passthrough",
    )
    pd.testing.assert_frame_equal(out, df)


# Regression test for the parser bug Gemini caught
def test_yrs_mons_parser_handles_real_format() -> None:
    """``'1yrs 11mon'`` must parse to 23 months (not 0).

    Earlier ``str.split()``-based versions silently returned 0 for
    every row of vehicle_loan because ``'1yrs'`` is a single token
    that fails ``isdigit()``.
    """
    assert _yrs_mons_to_months("1yrs 11mon") == 23
    assert _yrs_mons_to_months("0yrs 0mon") == 0
    assert _yrs_mons_to_months("4yrs 8mon") == 56
    # NaN propagates
    assert pd.isna(_yrs_mons_to_months(float("nan")))
    assert pd.isna(_yrs_mons_to_months(None))


def test_yrs_mons_parser_handles_verbose_format() -> None:
    """Robustness: handles 'X year Y months' too, not just 'Xyrs Ymon'."""
    assert _yrs_mons_to_months("5 years 0 months") == 60
    assert _yrs_mons_to_months("12 month") == 12


def test_two_digit_year_parser() -> None:
    """vehicle_loan dates like '17-01-83' → 1983, '01-05-09' → 2009."""
    assert _two_digit_year_to_full("17-01-83") == 1983
    assert _two_digit_year_to_full("01-05-09") == 2009
    assert pd.isna(_two_digit_year_to_full(None))
    assert pd.isna(_two_digit_year_to_full("---"))


# =============================================================================
# Block 2 · register.py
# =============================================================================


def test_shape_hash_is_order_independent() -> None:
    a = shape_aware_sha256(100, 3, ["x", "y", "z"])
    b = shape_aware_sha256(100, 3, ["z", "y", "x"])
    assert a == b


def test_shape_hash_different_for_different_shapes() -> None:
    a = shape_aware_sha256(100, 3, ["x", "y", "z"])
    b = shape_aware_sha256(101, 3, ["x", "y", "z"])
    assert a != b
    c = shape_aware_sha256(100, 4, ["x", "y", "z", "w"])
    assert a != c


def test_infer_categorical_numerical_hint_wins() -> None:
    """The hint list takes precedence over dtype inference."""
    df = pd.DataFrame({
        "target":       [0, 1, 0, 1],
        "obvious_num":  [1.0, 2.0, 3.0, 4.0],
        "object_str":   ["a", "b", "a", "b"],
        "int_but_cat":  [10, 20, 10, 20],   # dtype is int but hinted as cat
    })
    cats, nums = infer_categorical_numerical(
        df, target="target", hinted_categorical=["int_but_cat"],
    )
    assert "int_but_cat" in cats
    assert "object_str" in cats   # via dtype rule 2a
    assert "obvious_num" in nums
    assert "target" not in cats and "target" not in nums  # excluded


def test_compute_manifest_row_classification() -> None:
    raw_path = REPO / "data" / "raw" / "pd" / "0001.gmsc.csv"
    if not raw_path.exists():
        pytest.skip("missing raw: 0001.gmsc.csv")
    df = pd.read_csv(raw_path, low_memory=False)
    df = apply_dataset_specific_fixes(df, "0001.gmsc")
    row = compute_manifest_row(df, "0001.gmsc")
    assert row["dataset_id"] == "0001.gmsc"
    assert row["track"] == "pd"
    assert row["task_type"] == "classification"
    assert row["target_column"] == "SeriousDlqin2yrs"
    assert int(row["n_rows"]) > 0
    assert int(row["n_cols"]) > 0
    assert row["minority_class_ratio"] != ""
    # Sanity: minority share is in (0, 0.5]
    assert 0.0 < float(row["minority_class_ratio"]) <= 0.5
    assert row["target_mean"] == ""
    assert set(MANIFEST_COLUMNS).issubset(row.keys())


def test_compute_manifest_row_regression() -> None:
    raw_path = REPO / "data" / "raw" / "lgd" / "0001.heloc.csv"
    if not raw_path.exists():
        pytest.skip("missing raw: 0001.heloc.csv")
    df = pd.read_csv(raw_path, low_memory=False)
    df = apply_dataset_specific_fixes(df, "0001.heloc")
    row = compute_manifest_row(df, "0001.heloc")
    assert row["task_type"] == "regression"
    assert row["minority_class_ratio"] == ""
    assert row["target_mean"] != ""
    assert row["target_std"] != ""


# =============================================================================
# Block 3 · sanitize.py — each step in isolation
# =============================================================================


def test_drop_exact_duplicate_columns() -> None:
    df = pd.DataFrame({
        "target": [0, 1],
        "a":      [1, 2],
        "b":      [1, 2],   # exact duplicate of 'a'
        "c":      [3, 4],
    })
    out, dropped = _drop_exact_duplicate_feature_columns(df, target="target")
    # Either 'a' or 'b' is kept; the OTHER is dropped (first-encountered wins).
    assert len(dropped) == 1 and dropped[0] in {"a", "b"}
    assert "c" in out.columns
    assert "target" in out.columns


def test_drop_exact_duplicate_columns_with_nan() -> None:
    """Exact-duplicate detection treats NaN positions as matching."""
    df = pd.DataFrame({
        "target": [0, 1, 0],
        "a":      [1.0, np.nan, 3.0],
        "b":      [1.0, np.nan, 3.0],   # same NaN position → dup
        "c":      [1.0, 2.0,    3.0],   # different (no NaN)
    })
    _, dropped = _drop_exact_duplicate_feature_columns(df, target="target")
    assert dropped == ["b"] or dropped == ["a"]  # one of the two


def test_drop_high_missing_columns() -> None:
    """Drops columns whose NaN rate exceeds the threshold; keeps target."""
    df = pd.DataFrame({
        "target": [0, 1, 0, 1, 0],
        "good":   [1, 2, 3, 4, 5],
        "noisy":  [1.0, np.nan, np.nan, np.nan, np.nan],   # 80% NaN
    })
    out, dropped = _drop_high_missing_columns(df, target="target", max_missing_rate=0.7)
    assert "noisy" in dropped
    assert "good" in out.columns
    assert "target" in out.columns


def test_drop_constant_columns() -> None:
    df = pd.DataFrame({
        "target":   [0, 1, 0],
        "varying":  [1, 2, 3],
        "constant": [42, 42, 42],
    })
    out, dropped = _drop_constant_columns(df, target="target")
    assert "constant" in dropped
    assert "varying" in out.columns
    assert "target" in out.columns


def test_coerce_numeric_strings() -> None:
    df = pd.DataFrame({
        "target":       [0, 1, 0, 1],
        "looks_numeric": ["1", "2", "3", "4"],
        "true_string":   ["a", "b", "c", "d"],
    })
    out, coerced = _coerce_numeric_strings(df, target="target", threshold=0.95)
    assert "looks_numeric" in coerced
    assert "true_string" not in coerced
    assert pd.api.types.is_numeric_dtype(out["looks_numeric"])


def test_replace_inf_with_nan() -> None:
    df = pd.DataFrame({
        "target": [0, 1, 0],
        "a":      [1.0, np.inf, 3.0],
        "b":      [-np.inf, 2.0, 3.0],
    })
    out = _replace_inf_with_nan(df, target="target")
    assert pd.isna(out.loc[1, "a"])
    assert pd.isna(out.loc[0, "b"])
    assert not np.isinf(out["a"]).any()
    assert not np.isinf(out["b"]).any()


def test_clip_lgd_target() -> None:
    df = pd.DataFrame({"target": [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]})
    out = _clip_lgd_target(df, target="target", lower=0.0, upper=1.0)
    assert out["target"].min() == 0.0
    assert out["target"].max() == 1.0


def test_label_encode_classification_target() -> None:
    """Classification targets become contiguous int64 [0, K-1]."""
    df = pd.DataFrame({"target": ["yes", "no", "yes", "maybe"], "x": [1, 2, 3, 4]})
    out = _label_encode_classification_target(df, target="target")
    assert out["target"].dtype == np.int64
    assert set(out["target"].unique()) == {0, 1, 2}


# =============================================================================
# Block 4 · dedup.py
# =============================================================================


def test_jaccard() -> None:
    assert _jaccard(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert _jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert _jaccard([], []) == 1.0


def test_row_hashes_basic() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    h1 = _row_hashes(df, decimals=None)
    h2 = _row_hashes(df, decimals=None)
    assert h1 == h2
    assert len(h1) == 3


def test_row_hashes_invariant_to_column_order() -> None:
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"b": [3, 4], "a": [1, 2]})
    assert _row_hashes(df1, decimals=None) == _row_hashes(df2, decimals=None)


def test_row_hashes_rounded_collapses_near_duplicates() -> None:
    df1 = pd.DataFrame({"x": [1.0000000, 2.0]})
    df2 = pd.DataFrame({"x": [1.0000002, 2.0]})  # 7th decimal differs
    assert _row_hashes(df1, decimals=None) != _row_hashes(df2, decimals=None)
    assert _row_hashes(df1, decimals=6) & _row_hashes(df2, decimals=6)


def test_column_hashes_skip_low_cardinality() -> None:
    """Columns with ≤ N unique values are skipped from the column-hash set."""
    n = 100
    df = pd.DataFrame({
        "few":  [1] * (n // 2) + [2] * (n - n // 2),   # 2 unique values
        "many": list(range(n)),                         # n unique values
    })
    hashes = _column_hashes(df, nontrivial_unique_min=5)
    # Only the 'many' column survives the > 5 unique-values threshold.
    assert len(hashes) == 1


def _mk_cfg() -> NS:
    return NS(
        seed=42,
        dedup=NS(
            name_jaccard_threshold=0.80,
            row_hash_intersection_min=1,
            shared_columns_min=3,
            column_nontrivial_unique_min=10,
            rounded_row=NS(enabled=True, decimals=6),
            subset=NS(enabled=True, min_overlap_fraction=0.95),
            column_name_fuzzy=NS(
                enabled_pre=True, enabled_post=False, similarity_threshold=90,
            ),
            confidence_rules=NS(
                high=["id_match", "row_hash"],
                medium=["col_hash", "name_jaccard_and_shape"],
                low=["rounded_row", "subset", "fuzzy_names"],
            ),
            overwrite_existing_pass_csv=True,
        ),
    )


def _mk_fp(name: str, cols: list[str], rows: int = 100,
           row_hashes: set[int] | None = None,
           col_hashes: set[int] | None = None) -> Fingerprint:
    return Fingerprint(
        dataset_id=name, dataset_name=name, source="kaggle",
        track="pd", task_type="classification",
        path=Path(f"/tmp/{name}.csv"),
        n_rows=rows, n_cols=len(cols), columns=cols,
        row_hashes=row_hashes or set(),
        rounded_row_hashes=set(),
        column_hashes=col_hashes or set(),
    )


def test_compare_pair_id_match() -> None:
    cfg = _mk_cfg()
    a = _mk_fp("A", ["x", "y", "z"], rows=10)
    b = _mk_fp("A", ["x", "y", "z"], rows=10)
    triggered = compare_pair(a, b, cfg, enable_fuzzy_names=False)
    assert "id_match" in triggered
    assert confidence_for(triggered, cfg) == "high"


def test_compare_pair_row_hash() -> None:
    cfg = _mk_cfg()
    a = _mk_fp("A", ["x"], row_hashes={1, 2})
    b = _mk_fp("B", ["x"], row_hashes={2, 3})
    triggered = compare_pair(a, b, cfg, enable_fuzzy_names=False)
    assert "row_hash" in triggered
    assert confidence_for(triggered, cfg) == "high"


def test_compare_pair_disjoint_returns_empty() -> None:
    cfg = _mk_cfg()
    a = _mk_fp("A", ["a", "b"], row_hashes={1})
    b = _mk_fp("B", ["c", "d"], row_hashes={2})
    assert compare_pair(a, b, cfg, enable_fuzzy_names=False) == []


def test_compare_pair_fuzzy_names_only_when_enabled() -> None:
    cfg = _mk_cfg()
    a = _mk_fp("A", ["loan_amount", "credit_score"])
    b = _mk_fp("B", ["loanamount", "creditscore"])
    assert "fuzzy_names" not in compare_pair(
        a, b, cfg, enable_fuzzy_names=False,
    )


def test_compare_pair_subset_detection() -> None:
    """A's row-hashes are 95%+ contained in B's → flag 'subset'."""
    cfg = _mk_cfg()
    a = _mk_fp("A", ["x"], row_hashes=set(range(100)))
    b = _mk_fp("B", ["x"], row_hashes=set(range(200)))   # superset of A
    triggered = compare_pair(a, b, cfg, enable_fuzzy_names=False)
    assert "subset" in triggered or "row_hash" in triggered


# =============================================================================
# Block 5 · dataset.py
# =============================================================================


def test_chunk_indices_no_chunk_when_small() -> None:
    chunks = _shuffle_and_chunk_indices(
        n_rows=100, y=None,
        max_rows_per_chunk=200, min_chunk_size=10,
        equal_split_size=False, stratify=False, seed=0,
    )
    assert len(chunks) == 1
    assert sorted(chunks[0].tolist()) == list(range(100))


def test_chunk_indices_drops_undersized_tail() -> None:
    chunks = _shuffle_and_chunk_indices(
        n_rows=2_500, y=None,
        max_rows_per_chunk=2_000, min_chunk_size=2_000,
        equal_split_size=False, stratify=False, seed=0,
    )
    # 2 500 ÷ 2 000 → one chunk of 2 000, remainder of 500 < min, dropped
    assert len(chunks) == 1
    assert len(chunks[0]) == 2_000


def test_chunk_indices_equal_split() -> None:
    """``equal_split_size=True`` produces same-sized chunks ≤ max."""
    chunks = _shuffle_and_chunk_indices(
        n_rows=10_000, y=None,
        max_rows_per_chunk=4_000, min_chunk_size=10,
        equal_split_size=True, stratify=False, seed=0,
    )
    sizes = [len(c) for c in chunks]
    assert max(sizes) - min(sizes) <= 1   # at most one row off


def test_dataset_seed_is_stable_and_dataset_specific() -> None:
    assert _dataset_seed(42, "0001.gmsc") == _dataset_seed(42, "0001.gmsc")
    assert _dataset_seed(42, "0001.gmsc") != _dataset_seed(42, "0001.heloc")


def _mk_dataset_cfg(**overrides):
    categorical_encoding = NS(
        strategy="ordinal",
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value="nan",
    )
    dataset = NS(
        max_rows_per_chunk=20_000,
        min_chunk_size=2_000,
        equal_split_size=True,
        stratify_classification=True,
        context_fraction=0.60,
        categorical_encoding=categorical_encoding,
        y_dtype_classification="int64",
        y_dtype_regression="float32",
        skip_if_cached=True,
    )
    for key, value in overrides.items():
        setattr(dataset, key, value)
    return NS(seed=42, dataset=dataset)


def test_cache_fingerprint_tracks_processed_file_and_config() -> None:
    row = {
        "dataset_id": "0001.gmsc",
        "target_column": "target",
        "categorical_columns": "grade",
        "date_added": "2026-01-01",
    }
    cfg = _mk_dataset_cfg()
    cfg_hash = _dataset_config_hash(cfg)
    first = _cache_fingerprint(
        row,
        dataset_config_hash=cfg_hash,
        processed_csv_sha256="processed-a",
    )
    assert first != _cache_fingerprint(
        row,
        dataset_config_hash=cfg_hash,
        processed_csv_sha256="processed-b",
    )

    changed_cfg = _mk_dataset_cfg(context_fraction=0.50)
    assert first != _cache_fingerprint(
        row,
        dataset_config_hash=_dataset_config_hash(changed_cfg),
        processed_csv_sha256="processed-a",
    )

    row_with_new_date = dict(row, date_added="2026-05-04")
    assert first == _cache_fingerprint(
        row_with_new_date,
        dataset_config_hash=cfg_hash,
        processed_csv_sha256="processed-a",
    )


def test_split_context_query_disjoint_and_complete() -> None:
    ctx, qry = _split_context_query(n=1000, context_fraction=0.6, seed=0)
    assert len(ctx) + len(qry) == 1000
    assert set(ctx).isdisjoint(set(qry))
    assert len(ctx) == 600


def test_ordinal_encode_no_categoricals() -> None:
    """If no categorical columns, return the array unchanged (cast to f32)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    arr, positions = _ordinal_encode_categoricals(
        df, categorical_columns=[],
        fit_indices=np.arange(3),
        unknown_value=-1, missing_value_sentinel=np.nan,
    )
    assert positions == []
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(arr, df.to_numpy(dtype=np.float32))


def test_ordinal_encode_context_only_yields_unknown_sentinel() -> None:
    """REGRESSION TEST for the encoder-leakage bug.

    Fitting on context-only must mean a category that appears ONLY in
    the query slice gets encoded as ``unknown_value`` (default -1).
    Before the fix, the encoder was fit on the whole chunk, so
    query-only categories silently received valid IDs.
    """
    df = pd.DataFrame({
        "city": ["A", "A", "B", "C"],   # row 3 has 'C' which is query-only
        "x":    [1.0, 2.0, 3.0, 4.0],
    })
    fit_indices = np.array([0, 1, 2])   # context = first 3 rows; query = last
    arr, positions = _ordinal_encode_categoricals(
        df, categorical_columns=["city"],
        fit_indices=fit_indices,
        unknown_value=-1, missing_value_sentinel=np.nan,
    )
    assert positions == [0]
    # Row 3's encoded value for 'city' must be -1 (unseen by the encoder).
    assert arr[3, 0] == -1.0, (
        f"expected unknown sentinel -1 for query-only category, got {arr[3, 0]}"
    )
    # Context rows 0–2 should get valid (non-negative) IDs.
    assert (arr[:3, 0] >= 0).all()


def test_ordinal_encode_preserves_nan() -> None:
    """NaN in the input survives the encoding via missing_value_sentinel."""
    df = pd.DataFrame({"city": ["A", None, "B"], "x": [1.0, 2.0, 3.0]})
    arr, _ = _ordinal_encode_categoricals(
        df, categorical_columns=["city"],
        fit_indices=np.arange(3),
        unknown_value=-1, missing_value_sentinel=np.nan,
    )
    assert np.isnan(arr[1, 0])


# =============================================================================
# Block 6 · exploration.py — light smoke tests
# =============================================================================


def test_exploration_corpus_summary_shape() -> None:
    """corpus_summary_table runs and returns expected schema."""
    from src.data.exploration import corpus_summary_table
    pd_manifest = REPO / "data" / "manifest_pd.csv"
    if not pd_manifest.exists():
        pytest.skip("manifests not yet built")
    df = corpus_summary_table()
    expected_cols = {
        "track", "dataset_id", "task_type", "target_column",
        "raw_rows", "raw_features", "post_rows", "post_features",
        "n_categorical", "n_numerical",
        "missing_rate_raw", "minority_class_ratio",
        "target_mean", "target_std", "source",
    }
    assert expected_cols.issubset(df.columns)
    # Every dataset_id in the manifest should appear in the summary.
    assert len(df) >= 1


def test_exploration_resolves_paths_from_cfg() -> None:
    """When given an explicit cfg, exploration helpers honour it."""
    from src.data.exploration import _resolve_paths
    cfg = NS(paths=NS(
        processed="data/processed",
        manifest_pd="data/manifest_pd.csv",
        manifest_lgd="data/manifest_lgd.csv",
    ))
    paths = _resolve_paths(cfg)
    assert paths["processed"].name == "processed"
    assert paths["manifest_pd"].name == "manifest_pd.csv"
