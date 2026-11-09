"""Complete numerical summaries stay compact and selectively publishable."""
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.visualize.reporting import table_text


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
    private = ['output CreditPFN/reference/manifests/private.csv',
               'output CreditPFN/seed/results/predictions.csv',
               'output CreditPFN/figures/rq1/figure.pdf']
    check = subprocess.run(['git', 'check-ignore', '--no-index', *public, *private],
                           cwd=root, capture_output=True, text=True, check=False)
    assert check.returncode == 0
    assert set(check.stdout.splitlines()) == set(private)
