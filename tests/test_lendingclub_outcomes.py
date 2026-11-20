"""A loan is eligible only when its application and final lender outcome link unambiguously."""
import numpy as np
import pandas as pd
import pytest

from src.data import lendingclub_outcomes as lc


def test_matching_removes_early_incomplete_and_ambiguous_loans_before_using_final_status():
    fields = dict.fromkeys(lc.MATCH_FIELDS, 2)
    fields.update(grade="A", sub_grade="A1", purpose="car", addr_state="CA", home_ownership="RENT",
                  term="36 months", int_rate="9.4%", pub_rec=0, revol_util="31.2%")
    raw = pd.DataFrame([dict(fields, loan_amnt=i, member_id=i, loan_status=0) for i in range(1, 8)])
    sources = pd.DataFrame([dict(fields, loan_amnt=i, issue_year=2012, loan_status="Fully Paid")
                            for i in range(1, 8)])
    sources.loc[0, "issue_year"] = 2011
    sources.loc[1, "loan_status"] = "Charged Off"
    sources.loc[1, "int_rate"] = "9.40"
    sources.loc[1, "grade"] = " a "
    sources["term"] = 36
    # A profile occurring once in each vintage is ambiguous even when its later row is terminal.
    sources = pd.concat([sources, sources.iloc[[2]].assign(issue_year=2010)], ignore_index=True)
    raw = pd.concat([raw, raw.iloc[[3]].assign(member_id=8)], ignore_index=True)
    raw.loc[4, "dti"] = np.nan
    key, stats = lc.match_outcomes(raw, sources)
    assert key.to_dict("list") == {"member_id": [2, 6, 7], "loan_status": [1, 0, 0]}
    assert stats["unique_matches"] == 4 and stats["early_matches"] == 1
    sources.loc[1, "revol_util"] = "28%"
    with pytest.raises(ValueError, match="revolving utilization"):
        lc.match_outcomes(raw, sources)
    sources.loc[1, "revol_util"] = "31.2%"
    sources.loc[1, "loan_status"] = "Current"
    with pytest.raises(ValueError, match="terminal"):
        lc.match_outcomes(raw, sources)
