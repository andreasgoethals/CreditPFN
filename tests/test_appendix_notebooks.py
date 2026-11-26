"""The appendix notebooks are the results notebooks recomputed on public tables only, generated."""
from __future__ import annotations

import json

import pytest

from src.utils import appendix_notebooks as an
from src.utils.paths import notebooks_dir


def test_appendix_notebooks_are_generated_from_the_current_sources():
    assert an.stale() == [], "run: python -m src.utils.appendix_notebooks"


@pytest.mark.parametrize("name", list(an.SOURCES))
def test_each_twin_switches_to_public_tables_before_any_loader(name):
    source = an.SOURCES[name]
    nb = json.loads((notebooks_dir() / "appendix" / f"{name}.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    setup = code[0]
    assert f"FigureSaver('appendix/{name}')" in setup and f"FigureSaver('{source}')" not in setup
    assert setup.index("c2.use_tables('public')") < setup.index("FigureSaver(")
    assert "c2.scope_counts" in code[1]
    assert code[-1].strip() == "print(report.summary(sink))"
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert not any(old in text for old in an.COUNT_TEXT)
    assert "".join(nb["cells"][0]["source"]).startswith("# Appendix, public tables: ")


def test_public_table_counts_match_the_registry():
    from src.visualize import campaign2 as c2
    counts = [c2.scope_counts(t) for t in ("pd", "lgd")]
    if any(c.empty for c in counts):
        pytest.skip("dataset registry not available")
    for track, frame in zip(("pd", "lgd"), counts):
        row = frame.iloc[0]
        assert row.credit_tables == an.ALL_TABLES[track]
        assert row.credit_tables - row.proprietary_tables == an.PUBLIC_TABLES[track]


def test_runner_orders_the_appendix_last():
    from src.utils.run_notebooks import discover
    names = discover()
    first_appendix = min(i for i, n in enumerate(names) if n.startswith("appendix/"))
    assert all(n.startswith("appendix/") for n in names[first_appendix:])
