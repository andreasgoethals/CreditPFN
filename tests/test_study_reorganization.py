"""Scientific and storage invariants of the ten-study presentation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.utils import study_layout as layout
from src.visualize import research_views as rv, context_curves as cc


def tree(root):
    root.mkdir()
    original = {}
    for group in layout.CAMPAIGNS:
        file = root / group / "results" / "evidence.csv"
        file.parent.mkdir(parents=True)
        file.write_bytes((group + " unique measurement\n").encode())
        original[group] = file.read_bytes()
    return original


def test_two_tier_migration_preserves_every_byte_once_and_is_idempotent(tmp_path, monkeypatch):
    roots = [tmp_path/"DATA", tmp_path/"PROJECT"]
    originals = [tree(root) for root in roots]
    preview = layout.migrate(roots)
    assert all(r["files"] == 8 for r in preview)
    assert not any((r/layout.MARKER).exists() for r in roots)
    layout.migrate(roots, apply=True)
    for root, original in zip(roots, originals):
        for old, content in original.items():
            assert (root/layout.CAMPAIGNS[old]/"results/evidence.csv").read_bytes() == content
        assert len(list(root.rglob("evidence.csv"))) == 8
        assert json.loads((root/layout.MARKER).read_text()) == layout.manifest()
        assert layout.payload("experiment1", root) == "shared/main_grid"
        assert not (root/".study-v2-migration.lock").exists()
        assert not (root/".study-v2-staging").exists()
    def unexpected_write(*args):
        raise AssertionError("A completed migration must not rewrite publication metadata")
    monkeypatch.setattr(layout, "_journal", unexpected_write)
    assert all(r["state"] == "already migrated" for r in layout.migrate(roots, apply=True))


def test_migration_resumes_after_interrupted_destination_move(tmp_path, monkeypatch):
    root = tmp_path/"output"; original = tree(root)
    rename = Path.rename
    calls = 0
    def interrupted(self, target):
        nonlocal calls
        result = rename(self, target)
        calls += 1
        if calls == 10:
            raise OSError("simulated interruption after rename")
        return result
    monkeypatch.setattr(Path, "rename", interrupted)
    with pytest.raises(OSError, match="interruption"):
        layout.migrate([root], apply=True)
    assert not (root/layout.MARKER).exists()
    monkeypatch.setattr(Path, "rename", rename)
    layout.migrate([root], apply=True)
    assert all((root/layout.CAMPAIGNS[g]/"results/evidence.csv").read_bytes()==b for g,b in original.items())


def test_migration_refuses_overlap_existing_lock_and_unknown_marker(tmp_path):
    root=tmp_path/"out"; tree(root)
    with pytest.raises(ValueError, match="Overlapping"):
        layout.migrate([root, root/"experiment1"])
    (root/".study-v2-migration.lock").write_text("owned")
    with pytest.raises(RuntimeError, match="ownership"):
        layout.migrate([root], apply=True)
    assert (root/".study-v2-migration.lock").read_text()=="owned"
    (root/layout.MARKER).write_text('{}')
    with pytest.raises(ValueError, match="Unrecognized"):
        layout.payload("experiment1", root)


def factorial():
    rows=[]
    # Anchoring: +2 full, +5 frozen. Freezing: -1 unanchored, +2 anchored.
    for dataset in ("A", "B"):
        for frozen, lam, effect in ((False,0,10), (False,.003,12), (True,0,9), (True,.003,14)):
            rows.append(dict(domain="credit", metric="roc_auc", base="v2", dataset=dataset,
                             learning_rate=3e-7, sampling="one_sample", frozen=frozen,
                             l2sp_lambda=lam, effect=effect))
    return pd.DataFrame(rows)


def test_factorial_simple_effects_and_interaction_require_all_four_cells():
    frame=factorial()
    e3=rv.factorial_contrasts(frame, 3)
    e4=rv.factorial_contrasts(frame, 4)
    assert e3.groupby("condition").contrast.mean().to_dict()=={"All parameters":2.,"Frozen backbone":5.,"Interaction":3.}
    assert e4.groupby("condition").contrast.mean().to_dict()=={"L2-SP off":-1.,"L2-SP 0.003":2.,"Interaction":3.}
    # Missing frozen+anchored must not become a zero or a partially matched average.
    assert set(rv.factorial_contrasts(frame.iloc[:-1],3).dataset)=={"A"}
    with pytest.raises(ValueError, match="Duplicate"):
        rv.factorial_contrasts(pd.concat([frame,frame.iloc[:1]]),3)


def test_context_all_metrics_keep_absolute_scores_and_paired_differences():
    from tests.test_new_experiment_reports import context_fixture
    study=context_fixture()
    study.cells["rmse"]=1-study.cells.r2
    pages=cc.plot_context_curves(study)
    assert len(pages)==6
    for page in pages:
        data=page.tables["Complete paired table values"]
        assert data.metric.nunique()==2
        assert data.groupby(["metric","family","context_rows"]).dataset.nunique().nunique()==1
        for metric,g in data.groupby("metric"):
            sign=-1 if metric in ("brier_score","rmse") else 1
            assert np.allclose(g.gain, sign*(g.adapted-g.original))
        page.figure.canvas.draw()
    named=cc.plot_table_curves(study)
    assert named and all("Named-table absolute scores and effects" in p.tables for p in named)
    # One unavailable secondary cell removes that table from both metric averages.
    study.cells.loc[(study.cells.dataset=="Table 0") & (study.cells.state=="cpt"),"brier_score"]=np.nan
    page=cc.plot_context_curves(study)[0]
    assert page.provenance["fixed_tables"]==["Table 1","Table 2"]


def test_atomic_checkpoint_replacement_retries_only_transient_windows_errors(tmp_path, monkeypatch):
    from src.train import checkpoint_io as io
    original=io.os.replace
    source,dest=tmp_path/"source",tmp_path/"dest"
    source.write_bytes(b"new");dest.write_bytes(b"old")
    calls=[]
    def blocked(a,b):
        calls.append(1)
        if len(calls)<3:
            e=PermissionError("scanner");e.winerror=32;raise e
        original(a,b)
    monkeypatch.setattr(io,"os",SimpleNamespace(replace=blocked,name="nt"))
    monkeypatch.setattr(io.time,"sleep",lambda _:None)
    io._replace(source,dest)
    assert len(calls)==3 and dest.read_bytes()==b"new"
    source.write_bytes(b"next")
    def denied(a,b):
        e=PermissionError("persistent");e.winerror=5;raise e
    io.os.replace=denied
    with pytest.raises(PermissionError):io._replace(source,dest)
    assert dest.read_bytes()==b"new" and source.read_bytes()==b"next"


def test_shared_factorial_registered_once_with_explicit_views():
    record=layout.manifest()
    assert len(record["studies"])==10 and record["factorial_unique_runs"]==512
    assert {record["studies"][f"experiment{n}"]["sources"][0] for n in (1,3,4)}=={"shared/main_grid"}


def test_learning_rate_panels_do_not_change_table_cohorts():
    frame=factorial()
    frame=pd.concat([frame,frame.assign(learning_rate=1e-5)],ignore_index=True)
    # One missing rate for table B must remove it from the entire mean curve.
    frame=frame[~(frame.dataset.eq("B") & frame.learning_rate.eq(1e-5))]
    pages=rv.plot_learning_rates(frame,"pd")
    assert pages and set(pages[0].tables["Paired table effects"].dataset)=={"A"}
    assert pages[0].tables["Mean and table spread"].tables.eq(1).all()


def test_missing_secondary_metric_does_not_create_inconsistent_context_panel():
    from tests.test_new_experiment_reports import context_fixture
    study=context_fixture()
    study.cells["rmse"]=1-study.cells.r2
    study.cells.loc[study.cells.track.eq("pd"),"brier_score"]=np.nan
    assert all("context_pd_" not in page.name for page in cc.plot_context_curves(study))


def test_approach_rate_comparison_keeps_the_same_tables():
    from src.visualize import synthesis as sy
    rows = [dict(experiment=5, base="v3", approach="DiagScale", l2sp_lambda=0.,
                 dataset=table, learning_rate=rate, effect=value)
            for table, rate, value in (("A", 1e-4, .01), ("A", 1e-3, .02), ("B", 1e-4, .99))]
    page = sy.plot_rate_response(pd.DataFrame(rows), "pd")[0]
    assert set(page.tables["Fixed-cohort table effects"].dataset) == {"A"}
    assert page.tables["Rate response"]["mean"].tolist() == [.01, .02]
    assert page.tables["Rate response"].tables.eq(1).all()


def test_paginated_table_maps_keep_every_value_and_one_colour_scale(monkeypatch):
    from src.visualize import synthesis as sy
    rows = [dict(base="v3", dataset=table, config=f"Arm {arm}", original=.7,
                 effect=(arm + 1) / 1000, approach="Full CPT")
            for arm in range(17) for table in ("A", "B")]
    data = pd.DataFrame(rows)
    monkeypatch.setattr(sy, "study_arms", lambda frame: frame)
    monkeypatch.setattr(sy, "config_order", lambda frame: list(frame.config.unique()))
    pages = sy.plot_table_map(data, "pd", "v3")
    assert len(pages) == 3
    columns = [name for page in pages for name in page.tables["Table × arm effects"] if name != "dataset"]
    assert columns == list(data.config.unique())
    scales = [(page.figure.axes[0].images[0].norm.vmin, page.figure.axes[0].images[0].norm.vmax) for page in pages]
    assert len(set(scales)) == 1
    assert all(len(page.tables["Table × arm effects"].columns) <= sy.style.TABLE_MAP_COLUMNS + 1 for page in pages)
