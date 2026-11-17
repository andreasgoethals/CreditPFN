"""TabArena-style leaderboard (src/visualize/arena.py) on hand-checkable cases.

The module re-implements TabArena's ``bencheval`` (see its docstring); on random leaderboards
with ties and unequal fold counts it reproduced bencheval's Elo, interval, rank, win rate,
improvability and normalized score (one-off check, docs/LITERATURE.md, 17-11-2026). These
tests pin the definitions so a later change cannot drift from them silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.visualize import arena


def _frame(errors: dict, folds: dict | None = None) -> pd.DataFrame:
    """{method: {table: [error per fold]}} -> long frame."""
    rows = [dict(method=m, table=t, fold=f, error=e)
            for m, tables in errors.items() for t, values in tables.items() for f, e in enumerate(values)]
    return pd.DataFrame(rows)


def test_two_methods_elo_gap_is_the_bradley_terry_log_odds():
    # A wins 3 of 4 folds on each of two tables: Elo gap = 400 log10(3).
    frame = _frame({"A": {"t1": [1, 1, 1, 3], "t2": [1, 1, 1, 3]},
                    "B": {"t1": [2, 2, 2, 2], "t2": [2, 2, 2, 2]}})
    elo = arena.elo(frame, "B", rounds=5)
    assert elo.loc["B", "elo"] == pytest.approx(1000.0)
    assert elo.loc["A", "elo"] - elo.loc["B", "elo"] == pytest.approx(400 * np.log10(3), abs=0.05)


def test_ties_count_half_and_tables_weigh_equally():
    # t1 (2 folds): A wins both; t2 (4 folds): all ties. Per table weight 1 -> A: 1 + 0.5*1, B: 0.5.
    frame = _frame({"A": {"t1": [1, 1], "t2": [5, 5, 5, 5]}, "B": {"t1": [2, 2], "t2": [5, 5, 5, 5]}})
    fights = arena.battles(frame)
    assert fights.groupby("table").weight.sum().to_dict() == {"t1": 1.0, "t2": 1.0}
    assert set(fights[fights.table == "t2"].winner) == {"tie"}
    elo = arena.elo(frame, "B", rounds=5)
    assert elo.loc["A", "elo"] - 1000 == pytest.approx(400 * np.log10(1.5 / 0.5), abs=0.05)


def test_anchor_only_shifts_every_rating():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame([dict(method=m, table=f"t{t}", fold=f, error=rng.uniform() + 0.1 * i)
                          for i, m in enumerate("ABCD") for t in range(6) for f in range(3)])
    one, two = arena.elo(frame, "A", rounds=20), arena.elo(frame, "C", rounds=20)
    shift = (two.elo - one.elo).to_numpy()
    assert np.allclose(shift, shift[0]) and two.loc["C", "elo"] == pytest.approx(1000)
    assert np.allclose(two.elo_high - two.elo, one.elo_high - one.elo)     # interval widths unchanged
    assert (one.elo_low <= one.elo).all() and (one.elo <= one.elo_high).all()


def test_rank_winrate_and_improvability_per_fold_then_table():
    frame = _frame({"A": {"t1": [1.0, 1.0], "t2": [1.0]}, "B": {"t1": [2.0, 1.0], "t2": [4.0]}})
    board = arena.leaderboard(frame, "B", rounds=5, improvability_rounds=5).set_index("method")
    # t1: A ranks 1 and 1.5 (tie) -> 1.25; t2: 1. Mean over tables 1.125.
    assert board.loc["A", "rank"] == pytest.approx(1.125)
    assert board.loc["A", "winrate"] == pytest.approx(1 - (1.125 - 1))
    assert board.loc["A", "improvability"] == pytest.approx(0.0)
    # B: t1 folds 1 - 1/2 = 0.5 and 0 -> 0.25; t2: 1 - 1/4 = 0.75; mean 0.5.
    assert board.loc["B", "improvability"] == pytest.approx(0.5)


def test_normalized_score_uses_the_median_so_a_weak_baseline_cannot_stretch_it():
    base = {"A": {"t1": [1.0], "t2": [1.0]}, "B": {"t1": [1.5], "t2": [1.2]},
            "C": {"t1": [2.0], "t2": [2.0]}, "Weak": {"t1": [4.0], "t2": [3.0]}}
    worse = {**base, "Weak": {"t1": [40.0], "t2": [30.0]}}
    one = 1 - arena.normalized_error(_frame(base)).groupby("method").normalized_error.mean()
    two = 1 - arena.normalized_error(_frame(worse)).groupby("method").normalized_error.mean()
    assert one["A"] == 1.0 and one["Weak"] == 0.0
    pd.testing.assert_series_equal(one, two)                                # min-max would change B
    # t1: median of (1, 1.5, 2, 4) = 1.75 -> B = 1 - 0.5/0.75; t2: median 1.6 -> B = 1 - 0.2/0.6.
    assert one["B"] == pytest.approx(1 - (0.5 / 0.75 + 0.2 / 0.6) / 2)


def test_complete_units_and_input_checks():
    frame = _frame({"A": {"t1": [1, 2], "t2": [1, 2]}, "B": {"t1": [2, 1], "t2": [2]}})
    kept, dropped = arena.complete_units(frame)
    assert dropped == ["t2"] and set(kept.table) == {"t1"}
    with pytest.raises(ValueError, match="duplicate"):
        arena.leaderboard(pd.concat([frame, frame]), "A")
    with pytest.raises(ValueError, match="anchor"):
        arena.elo(kept, "missing", rounds=2)
