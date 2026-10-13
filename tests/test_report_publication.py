"""Complete numerical summaries stay compact and selectively publishable."""
from io import StringIO
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.visualize.reporting import table_text


def test_partial_sampling_panels_explain_missing_cohorts(monkeypatch):
    from src.visualize import dashboards as db
    run = SimpleNamespace(track="pd", cfg=SimpleNamespace(train=SimpleNamespace(
        trajectory_steps=[0, 250])))
    observations = pd.DataFrame({"base": ["v2", "v3"],
                                 "sampling": ["one_sample", "one_sample"]})
    monkeypatch.setattr(db, "monitor_observations", lambda *a, **k: observations)
    monkeypatch.setattr(db.cp, "trajectory_curve", lambda run, arm, **k:
        pd.DataFrame({"x": [0, 250], "mean": [0., .01] if arm.base.iloc[0] == "v2"
                      else [np.nan, np.nan]}))
    pages = db.plot_followup_curves(run)
    assert pages
    for page in pages:
        missing = [axis for axis in page.figure.axes
                   if any(text.get_text() == "Incomplete matched cohort" for text in axis.texts)]
        assert missing
        assert all(axis.get_xlim() == (0, 250) for axis in missing)
        assert page.tables["Displayed curves"]["mean"].isna().sum() == 2


def test_compact_tables_preserve_all_rows_labels_and_missing_values():
    frame = pd.DataFrame({"label": ['a,b', 'line\nbreak', 'last'],
                          "value": [1.23456789, np.nan, -3e-7]},
                         index=["first", "middle", "final"])
    text = table_text(frame)
    restored = pd.read_csv(StringIO(text.split("\n", 1)[1]), index_col=0,
                           keep_default_na=False)
    assert list(restored.index) == list(frame.index)
    assert list(restored.label) == list(frame.label)
    assert list(restored.value) == ['1.2345679', 'unavailable', '-3e-07']


def test_compact_tables_do_not_truncate_and_avoid_alignment_padding():
    frame = pd.DataFrame({"label": ["long configuration identifier " * 3] * 100,
                          "value": np.arange(100, dtype=float)})
    text = table_text(frame)
    restored = pd.read_csv(StringIO(text.split("\n", 1)[1]), index_col=0)
    assert len(restored) == 100
    assert restored.value.tolist() == frame.value.tolist()
    aligned = frame.to_string(max_rows=None, max_cols=None, max_colwidth=None)
    assert len(text) < len(aligned)


def test_git_publishes_only_the_shared_reports():
    root = Path(__file__).resolve().parents[1]
    public = ['output CreditPFN/All_Results.md',
              'output CreditPFN/figures/CAPTIONS.md']
    private = ['output CreditPFN/experiment1/manifests/private.csv',
               'output CreditPFN/experiment2/results/predictions.csv',
               'output CreditPFN/figures/experiment1/figure.pdf']
    check = subprocess.run(['git', 'check-ignore', '--no-index', *public, *private],
                           cwd=root, capture_output=True, text=True, check=False)
    assert check.returncode == 0
    assert set(check.stdout.splitlines()) == set(private)
