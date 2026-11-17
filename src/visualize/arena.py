"""TabArena-style leaderboard: Elo, average rank, win rate, normalized score and improvability.

A re-implementation of TabArena's evaluation code (``bencheval`` and ``tabarena`` packages of
github.com/autogluon/tabarena, commit ``cc507b8``, read 17-11-2026), kept dependency-free so the
local environment stays identical to the cluster container. ``tests/test_arena.py`` pins the
behaviour on hand-checkable cases; a one-off comparison against ``bencheval`` itself is recorded
in docs/LITERATURE.md (local). Correspondence:

* **Battles** (``EloHelper.convert_results_to_battles``): every unordered pair of methods meets
  once per (table, fold); the lower error wins, equal errors tie; each battle weighs
  1 / (folds of its table), so every table contributes equally.
* **Elo** (``EloHelper.compute_mle_elo``, pair-compressed path): Bradley–Terry by maximum
  likelihood — a logistic regression without intercept (C = 1e6, tolerance 1e-10) on two rows per
  pair carrying the weighted wins (+ half the ties) of each side; Elo = 400 · coefficient + 1000
  on base 10. ``BenchmarkEvaluator.compute_elo`` (``post_calibrate=True``) then shifts every
  rating so that the anchor method sits at 1000.
* **Interval** (``EloHelper.get_bootstrap_result``): 200 draws (``LeaderboardReporter`` default)
  that resample tables with replacement (seed 0); the 2.5 and 97.5% quantiles of the uncalibrated
  draws, as distances from the uncalibrated point estimate, are attached to the calibrated
  rating and clipped at zero.
* **Rank, win rate, improvability** (``BenchmarkEvaluator.compute_results_per_task`` with the
  fold as seed column, ``average_seeds=False``): computed per (table, fold) — average rank of the
  error; win rate = 1 − (rank − 1) / (methods − 1); improvability = 1 − best error / error —
  averaged over folds within a table, then over tables. Improvability gets a 100-draw table
  bootstrap (seed 0).
* **Normalized score** (``NormalizedScorer`` via ``compute_normalized_error_dynamic``): per
  table, each method's fold-mean error is placed between the best method (0) and the median
  method (1) of that table and clipped to [0, 1] (denominator at least 1e-5); the score is one
  minus its mean over tables. Methods at or below the median get 0, so a weak baseline cannot
  stretch the scale of the others.

The input is a long frame with one row per method, table and fold and an error column (lower is
better): 1 − AUC for PD, RMSE for LGD, as on TabArena. Every method must be scored on every
(table, fold); ``complete_units`` restricts a frame to the tables where all methods are.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SCALE, BASE, INIT_RATING = 400, 10, 1000.0
ANCHOR_ELO = 1000.0
ELO_BOOTSTRAP_ROUNDS = 200
IMPROVABILITY_BOOTSTRAP_ROUNDS = 100
SOLVER_TOL = 1e-10
SOLVER_C = 1e6
SOLVER_MAX_ITER = 1000
NORMALIZER_FLOOR = 1e-5
COLUMNS = ("method", "table", "fold", "error")


def _check(errors: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(errors.columns)
    if missing:
        raise ValueError(f"arena: missing columns {sorted(missing)}")
    frame = errors[list(COLUMNS)].copy()
    if frame.duplicated(["method", "table", "fold"]).any():
        raise ValueError("arena: duplicate (method, table, fold) rows")
    if not np.isfinite(frame.error.to_numpy(dtype=float)).all():
        raise ValueError("arena: non-finite errors")
    return frame


def complete_units(errors: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Rows of the tables on which every method has every fold of that table; the dropped tables."""
    frame = errors.copy()
    methods = frame.method.nunique()
    per_table = frame.groupby("table").agg(folds=("fold", "nunique"), rows=("method", "size"))
    keep = per_table.index[per_table.rows == per_table.folds * methods]
    dropped = sorted(set(per_table.index) - set(keep))
    return frame[frame.table.isin(keep)].reset_index(drop=True), dropped


# ------------------------------------------------------------------------------- battles
def battles(errors: pd.DataFrame) -> pd.DataFrame:
    """One row per (table, fold) and unordered method pair: ``method_1``, ``method_2``, ``winner``
    ('1', '2' or 'tie') and ``weight`` = 1 / folds of the table."""
    frame = _check(errors)
    pairs = frame.merge(frame, on=["table", "fold"], suffixes=("_1", "_2"))
    pairs = pairs[pairs.method_1 < pairs.method_2]
    winner = np.where(pairs.error_1 < pairs.error_2, "1", np.where(pairs.error_1 > pairs.error_2, "2", "tie"))
    folds = frame.groupby("table").fold.nunique()
    out = pd.DataFrame({"method_1": pairs.method_1.to_numpy(), "method_2": pairs.method_2.to_numpy(),
                        "winner": winner, "table": pairs.table.to_numpy(),
                        "weight": 1.0 / pairs.table.map(folds).to_numpy(dtype=float)})
    return out.reset_index(drop=True)


def _pair_statistics(fights: pd.DataFrame):
    """Methods, unordered pairs (u < v), tables, and per table × pair the effective wins of u and v
    (wins plus half the ties, weighted): the sufficient statistics of the Bradley–Terry fit."""
    methods = pd.Index(sorted(set(fights.method_1) | set(fights.method_2)))
    keys = list(zip(fights.method_1, fights.method_2))          # method_1 < method_2 already
    pair_index = pd.Index(sorted(set(keys)))
    tables = pd.Index(sorted(fights.table.unique(), key=str))
    su = np.zeros((len(tables), len(pair_index)))
    sv = np.zeros_like(su)
    t = tables.get_indexer(fights.table)
    p = pair_index.get_indexer(pd.MultiIndex.from_tuples(keys))
    w = fights.weight.to_numpy(dtype=float)
    win = fights.winner.to_numpy()
    np.add.at(su, (t, p), np.where(win == "1", w, np.where(win == "tie", 0.5 * w, 0.0)))
    np.add.at(sv, (t, p), np.where(win == "2", w, np.where(win == "tie", 0.5 * w, 0.0)))
    return methods, list(pair_index), tables, su, sv


def _fit(methods: pd.Index, pairs: list, wins_u: np.ndarray, wins_v: np.ndarray) -> pd.Series:
    """Uncalibrated Bradley–Terry Elo from per-pair effective wins (TabArena's compressed MLE)."""
    from sklearn.linear_model import LogisticRegression
    active = (wins_u + wins_v) > 0
    if not active.any():
        raise ValueError("arena: no pair of methods carries any weight")
    x = np.zeros((2 * int(active.sum()), len(methods)))
    y = np.tile([1.0, 0.0], int(active.sum()))
    weight = np.empty(2 * int(active.sum()))
    log_base = math.log(BASE)
    for row, i in enumerate(np.flatnonzero(active)):
        u, v = methods.get_loc(pairs[i][0]), methods.get_loc(pairs[i][1])
        x[2 * row:2 * row + 2, u], x[2 * row:2 * row + 2, v] = log_base, -log_base
        weight[2 * row], weight[2 * row + 1] = wins_u[i], wins_v[i]
    model = LogisticRegression(fit_intercept=False, C=SOLVER_C, max_iter=SOLVER_MAX_ITER, tol=SOLVER_TOL)
    model.fit(x, y, sample_weight=weight)
    return pd.Series(SCALE * model.coef_[0] + INIT_RATING, index=methods)


def elo(errors: pd.DataFrame, anchor: str, *, rounds: int = ELO_BOOTSTRAP_ROUNDS, seed: int = 0) -> pd.DataFrame:
    """Per method: Elo with the anchor at 1000, and the lower and upper end of its 95% interval."""
    methods, pairs, tables, su, sv = _pair_statistics(battles(errors))
    if anchor not in methods:
        raise ValueError(f"arena: anchor {anchor!r} is not among the methods")
    point = _fit(methods, pairs, su.sum(axis=0), sv.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(rounds):
        counts = np.bincount(rng.choice(len(tables), size=len(tables), replace=True),
                             minlength=len(tables)).astype(float)
        draws.append(_fit(methods, pairs, counts @ su, counts @ sv))
    draws = pd.DataFrame(draws)
    upper = (draws.quantile(0.975) - point).clip(lower=0)
    lower = (point - draws.quantile(0.025)).clip(lower=0)
    rating = point + (ANCHOR_ELO - point[anchor])
    return pd.DataFrame({"elo": rating, "elo_low": rating - lower, "elo_high": rating + upper})


# ------------------------------------------------------------------- per-unit summaries
def per_unit(errors: pd.DataFrame) -> pd.DataFrame:
    """Per method, table and fold: average rank of the error, win rate and improvability."""
    frame = _check(errors)
    unit = frame.groupby(["table", "fold"]).error
    n = frame.groupby(["table", "fold"]).method.transform("size")
    frame["rank"] = unit.rank(method="average", ascending=True)
    frame["winrate"] = np.where(n > 1, 1 - (frame["rank"] - 1) / (n - 1).clip(lower=1), np.nan)
    best = unit.transform("min")
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["improvability"] = (1 - best / frame.error).fillna(0.0)
    return frame


def normalized_error(errors: pd.DataFrame) -> pd.DataFrame:
    """Per method and table: the fold-mean error, scaled between the table's best method (0) and
    its median method (1) and clipped to [0, 1] (TabArena's normalized error)."""
    frame = _check(errors).groupby(["method", "table"]).error.mean().reset_index()
    top = frame.groupby("table").error.transform("min")
    median = frame.groupby("table").error.transform("median")
    frame["normalized_error"] = ((frame.error - top) / np.clip(median - top, NORMALIZER_FLOOR, None)).clip(0, 1)
    return frame


def _table_means(units: pd.DataFrame, column: str) -> pd.DataFrame:
    """Method × table matrix of a per-unit column averaged over folds."""
    return units.groupby(["method", "table"])[column].mean().unstack("table")


def leaderboard(errors: pd.DataFrame, anchor: str, *, rounds: int = ELO_BOOTSTRAP_ROUNDS,
                improvability_rounds: int = IMPROVABILITY_BOOTSTRAP_ROUNDS, seed: int = 0) -> pd.DataFrame:
    """The leaderboard, one row per method sorted by Elo: ``elo`` (anchor = 1000) with its 95%
    interval, ``rank`` (average), ``winrate``, ``normalized_score`` (1 = best on every table),
    ``improvability`` (0 = best on every fold) with its 95% interval, and the counts of tables
    and (table, fold) units."""
    frame = _check(errors)
    units = per_unit(frame)
    out = elo(frame, anchor, rounds=rounds, seed=seed)
    for column in ("rank", "winrate", "improvability"):
        out[column] = _table_means(units, column).mean(axis=1)
    improv = _table_means(units, "improvability")
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.choice(improv.shape[1], size=improv.shape[1], replace=True),
                                   minlength=improv.shape[1]) for _ in range(improvability_rounds)], axis=1)
    boot = pd.DataFrame(improv.to_numpy() @ counts / improv.shape[1], index=improv.index).T
    out["improvability_low"] = boot.quantile(0.025)
    out["improvability_high"] = boot.quantile(0.975)
    out["normalized_score"] = 1 - normalized_error(frame).groupby("method").normalized_error.mean()
    out["tables"] = frame.table.nunique()
    out["units"] = len(frame[["table", "fold"]].drop_duplicates())
    out = out.sort_values("elo", ascending=False)
    out.index.name = "method"
    return out.reset_index()
