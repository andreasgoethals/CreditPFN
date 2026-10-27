"""Family-scoped re-evaluation: roster filter and guarded removal of old results.

Re-scoring TabICLv2 in FP32 must delete exactly its own result folders (original and
adapted, both tasks, both tiers), keep every other model, and never be able to leave
old and new folds side by side.
"""
from types import SimpleNamespace

import pytest

from src.utils import reset_family_benchmark as reset
from src.utils.paths import OUTPUT_DIR_NAME, resolve_staging_path


def _folders(root, names):
    for track, name in names:
        folder = root / "experiment1" / "results" / track / name
        folder.mkdir(parents=True)
        (folder / "fold.csv").write_text("x\n1\n", encoding="utf-8")


def test_reset_lists_then_removes_only_the_family(isolated_output, monkeypatch):
    monkeypatch.setattr("src.utils.clean_run.check_cluster_writers", lambda experiment=None: None)
    project = resolve_staging_path(OUTPUT_DIR_NAME)
    _folders(project, [("PD", "tabicl-untuned__tabicl-v2"), ("PD", "tabicl-trained__tabicl-v2__lr3e-07__l2sp0.003"),
                       ("LGD", "tabicl-trained__tabicl-v2__lr1e-06__l2sp0"), ("PD", "tabpfn-trained__v3-default__lr3e-07"),
                       ("PD", "catboost")])
    pointer = project / "experiment1" / "consolidated" / "cpt_main_v5" / "LATEST.json"
    (pointer.parent / "old").mkdir(parents=True)
    (pointer.parent / "old" / "inventory.json").write_text(
        '{"sources": {"eval_pd": [{"path": "PD/tabicl-untuned__tabicl-v2/fold.csv"}]}}', encoding="utf-8")
    pointer.write_text('{"snapshot": "old"}', encoding="utf-8")

    preview = reset.reset("experiment1", "tabicl")
    assert not preview["applied"] and len(preview["folders"]) == 3 and preview["files"] == 3
    assert all(p.exists() for p in map(type(project), preview["folders"]))

    done = reset.reset("experiment1", "tabicl", apply=True)
    assert done["applied"]
    remaining = sorted(p.name for p in (project / "experiment1" / "results" / "PD").iterdir())
    assert remaining == ["catboost", "tabpfn-trained__v3-default__lr3e-07"]
    assert not (project / "experiment1" / "results" / "LGD" / "tabicl-trained__tabicl-v2__lr1e-06__l2sp0").exists()
    # The old snapshot pointer is superseded, not deleted, so re-consolidation can publish.
    assert not pointer.exists()
    assert len(list(pointer.parent.glob("LATEST.superseded-tabicl-*.json"))) == 1


def test_reset_rejects_unbenchmarked_groups_and_unknown_families(isolated_output):
    with pytest.raises(ValueError):
        reset.family_dirs("experiment4", "tabicl")
    with pytest.raises(ValueError):
        reset.family_dirs("experiment1", "xgboost")


def test_family_filter_keeps_original_and_adapted_models_of_one_family():
    from scripts.eval_pipeline import filter_family
    roster = [(SimpleNamespace(source=s), None) for s in
              ("tabicl-untuned", "tabicl-trained", "tabpfn-untuned", "tabpfn-trained", "baseline")]
    assert [h.source for h, _ in filter_family(roster, "tabicl")] == ["tabicl-untuned", "tabicl-trained"]
    assert len(filter_family(roster, "")) == 5
    with pytest.raises(ValueError):
        filter_family(roster, "catboost")


def test_experiment1_and_seed_repeat_score_tabicl_in_fp32():
    from src.eval.config import load_eval_configs
    for config in ("config/experiment1/pd.yaml", "config/experiment1/lgd.yaml",
                   "config/experiment2/pd.yaml", "config/experiment2/lgd.yaml"):
        eval_cfg, _ = load_eval_configs([], [], config_path=config, split_index=0)
        assert dict(eval_cfg.tabicl_inference_options) == {"use_amp": False, "use_fa3": False}


def test_only_results_older_than_the_reset_block_a_family_rerun(isolated_output, monkeypatch):
    monkeypatch.setattr("src.utils.clean_run.check_cluster_writers", lambda experiment=None: None)
    project = resolve_staging_path(OUTPUT_DIR_NAME)
    _folders(project, [("PD", "tabicl-untuned__tabicl-v2")])
    assert len(reset.stale_results("experiment1", "tabicl")) == 1      # never reset: old results block
    reset.reset("experiment1", "tabicl", apply=True)
    assert reset.stale_results("experiment1", "tabicl") == []
    # An interrupted FP32 run leaves new files; they must not block its resubmission.
    _folders(project, [("PD", "tabicl-untuned__tabicl-v2")])
    assert reset.stale_results("experiment1", "tabicl") == []


def test_reset_refuses_when_other_snapshot_sources_are_already_missing(isolated_output, monkeypatch):
    import json
    monkeypatch.setattr("src.utils.clean_run.check_cluster_writers", lambda experiment=None: None)
    project = resolve_staging_path(OUTPUT_DIR_NAME)
    _folders(project, [("PD", "tabicl-untuned__tabicl-v2")])
    snapshot = project / "experiment1" / "consolidated" / "cpt_main_v5" / "s1"
    snapshot.mkdir(parents=True)
    (snapshot / "inventory.json").write_text(json.dumps({"sources": {
        "eval_pd": [{"path": "PD/tabicl-untuned__tabicl-v2/fold.csv"}],
        "training_pd": [{"path": "pd/archived_epochs.csv"}]}}), encoding="utf-8")
    (snapshot.parent / "LATEST.json").write_text('{"snapshot": "s1"}', encoding="utf-8")
    preview = reset.reset("experiment1", "tabicl")
    assert preview["missing_snapshot_sources"] == ["cpt_main_v5: pd/archived_epochs.csv"]
    with pytest.raises(RuntimeError):
        reset.reset("experiment1", "tabicl", apply=True)
    assert (project / "experiment1" / "results" / "PD" / "tabicl-untuned__tabicl-v2").exists()
