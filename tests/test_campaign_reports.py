"""Analysis contracts: bounded pages, matched comparisons and no cross-run leakage."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.visualize import style


@pytest.fixture(autouse=True)
def close_figures():
    style.apply()
    yield
    plt.close("all")


def test_nested_notebooks_are_discovered_and_hidden_backups_ignored(tmp_path,monkeypatch):
    from src.utils import run_notebooks as rn
    for path in ("rq1/01_report.ipynb","rq2/01_report.ipynb",
                 "general/01_data.ipynb","rq1/.ipynb_checkpoints/01_report.ipynb"):
        p=tmp_path/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text("{}")
    monkeypatch.setattr(rn,"notebooks_dir",lambda:tmp_path)
    assert rn.discover() == ("general/01_data","rq1/01_report","rq2/01_report")
    assert rn.discover(("rq2",)) == ("rq2/01_report",)


def test_nested_figure_ownership_never_clears_sibling(isolated_output):
    from src.visualize.figures import FigureSaver, read_manifest
    first = FigureSaver("rq1/01_report")
    fig,ax=plt.subplots(figsize=style.figsize())
    ax.plot([0,1],[0,1]); first.save(fig,"test",caption="A line.")
    FigureSaver("rq2/01_report")
    assert first.last_path.exists()
    assert len(read_manifest("rq1/01_report")) == 1
    FigureSaver("rq1/01_report")
    assert not first.last_path.exists()


def test_all_notebooks_are_thin_ordered_and_use_owned_savers():
    notebooks=sorted(Path("notebooks").rglob("*.ipynb"))
    assert {p.parent.name for p in notebooks} == {'experiment0','general','rq1','rq2','rq3','rq4'}
    for path in notebooks:
        nb=json.loads(path.read_text(encoding="utf8"))
        code=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"]
        assert code[-1].strip() == "print(report.summary(sink))"
        assert f"FigureSaver('{path.relative_to('notebooks').with_suffix('').as_posix()}')" in "\n".join(code)
        for cell in code:
            tree=ast.parse("\n".join(line for line in cell.splitlines() if not line.startswith("%")))
            assert not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for n in ast.walk(tree))
            assert "figsize=" not in cell and "color=" not in cell


def test_planned_exposure_distinguishes_chunks_from_table_visits():
    from src.visualize.corpus import planned_exposure
    data=pd.DataFrame(dict(dataset_id=["small","large"],track=["pd","pd"],rows=[10,90]))
    result=planned_exposure(data,row_cap=10)
    assert result.one_sample_step_share.tolist() == [.5,.5]
    assert result.full_pass_step_share.tolist() == [.1,.9]
