"""Cross-experiment synthesis: pairing, outcome counts and recipe selection.

These quantities feed both the overview notebooks and experiment 1, so an error
here would silently change every reported CPT-versus-original effect.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.visualize import synthesis as sy


def _rows(method, source, base, table, split, scores, domain="credit"):
    return [dict(method_dirname=method, source=source, base_short=base, test_dataset_id=table, split=split,
                 fold_idx=fold, roc_auc=score, status="OK", domain=domain) for fold, score in enumerate(scores)]


def _run(rows, folds=4):
    return SimpleNamespace(evaluation=pd.DataFrame(rows), cfg=SimpleNamespace(corpus=SimpleNamespace(n_folds=folds)))


def test_table_effects_pair_each_fold_with_its_own_original_split():
    trained = "tabpfn-trained__v3-default__lr3e-07__l2sp0.003"
    rows = (_rows("tabpfn-untuned__v3-default", "tabpfn-untuned", "v3-default", "A", 0, [.70, .70, .70, .70, .70])
            + _rows(trained, "tabpfn-trained", "v3-default", "A", 0, [.71, .72, .69, .71, .71]))
    data = sy.table_effects(_run(rows), "roc_auc", experiment=1)
    row = data.iloc[0]
    assert row.base == "v3" and row.approach == "Full CPT" and bool(row.reference)
    assert row.effect == pytest.approx(.0080)
    assert (row.folds, row.folds_improved) == (5, 4)
    assert row.config == "E1 all · LR 3e-07 · λ 0.003"


def test_incomplete_folds_drop_the_table_rather_than_averaging_fewer_folds():
    trained = "tabpfn-trained__v3-default__lr3e-07__l2sp0.003"
    rows = (_rows("tabpfn-untuned__v3-default", "tabpfn-untuned", "v3-default", "A", 0, [.7] * 5)
            + _rows(trained, "tabpfn-trained", "v3-default", "A", 0, [.71] * 4))
    assert sy.table_effects(_run(rows), "roc_auc", experiment=1).empty


def test_retention_requires_every_cpt_partition():
    trained = "tabpfn-trained__v3-default__lr3e-07__l2sp0.003"
    rows = []
    for split in range(2):
        rows += _rows("tabpfn-untuned__v3-default", "tabpfn-untuned", "v3-default", "R", split, [.8] * 5, "noncredit")
        rows += _rows(trained, "tabpfn-trained", "v3-default", "R", split, [.81] * 5, "noncredit")
    assert sy.table_effects(_run(rows, folds=2), "roc_auc", experiment=1, domain="ood").partitions.eq(2).all()
    assert sy.table_effects(_run(rows, folds=4), "roc_auc", experiment=1, domain="ood").empty


def test_outcomes_use_the_declared_negligible_band():
    band = sy.style.TIE_BAND["roc_auc"]
    labels = sy.outcome(np.array([band * 2, band / 2, -band / 2, -band * 2]), "roc_auc")
    assert labels.tolist() == ["Improved", "Negligible", "Negligible", "Worse"]


def _grid(effects):
    return pd.DataFrame([dict(base="v3", dataset=t, config=c, effect=e, reference=c == "ref")
                         for c, values in effects.items() for t, e in zip("ABCD", values)])


def test_selection_check_scores_the_choice_on_the_held_out_table():
    # "lucky" wins on average only because of table A; leaving A out selects "ref".
    data = _grid({"ref": [0.0, .01, .01, .01], "lucky": [.10, 0.0, 0.0, 0.0]})
    row = sy.selection_check(data).iloc[0]
    assert row.best_observed_recipe == "lucky"
    assert row.best_observed_mean == pytest.approx(.025)
    # A held out -> ref chosen (0.0 on A); B/C/D held out -> lucky chosen (0.0 each).
    assert row.leave_one_table_out_mean == pytest.approx(0.0)
    assert row.reference_mean == pytest.approx(.0075)
