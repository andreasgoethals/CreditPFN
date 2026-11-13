"""The second campaign is the reference recipe plus declared deviations, and nothing else."""
import pytest

from src.utils import design as d


def test_every_arm_differs_from_the_reference_only_by_its_declared_deviations():
    spec = d.load_design()
    for (name, track, index), cfg in d.arm_configs(spec).items():
        arm = d.experiment_arms(name, track, spec)[index]
        assert d.undeclared_differences(cfg, arm, track, spec) == [], (name, track, index)


def test_reference_recipe_matches_the_agreed_values():
    ref = d.reference_config("pd")
    assert ref["tunable"]["learning_rates"] == [3e-7]
    assert ref["optimizer"] == {"name": "adamw", "weight_decay": 0.0, "l2sp_lambda": 0.003}
    assert ref["tunable"]["query_fractions"] == [0.40]
    assert ref["train"]["context_sampling"] == "stratified"          # proportional PD sampling
    assert ref["train"]["target_total_steps"] == 2500
    assert ref["train"]["max_rows_per_step"] is None                 # model maxima from data.yaml
    assert ref["tunable"]["epoch_pass_modes"] == ["one_sample"]
    assert ref["train"]["regression_objective"] == "window_nll"
    assert ref["train"]["epoch_eval_n_estimators"] == 2 == ref["train"]["epoch_eval_n_estimators_tabicl"]
    assert ref["train"]["epoch_eval_subsample_samples"] == 10000
    assert ref["train"]["trajectory_rows"][0] == 1_000_000
    assert len(ref["tunable"]["classifier_base_paths"]) == 4


def test_monitoring_is_dense_early_then_every_250_and_500_updates():
    points = d.monitor_updates(10000)
    assert points[:7] == [0, 50, 100, 150, 200, 250, 500]
    assert points[-1] == 10000 and len(points) == 30
    assert all(b - a == 250 for a, b in zip(points[5:14], points[6:15])) and points[14] == 2500
    assert all(b - a == 500 for a, b in zip(points[14:], points[15:]))
    assert d.monitor_updates(2500) == points[:15] and d.monitor_updates(250) == points[:6]


def test_run_counts_match_the_agreed_grid():
    counts = {}
    spec = d.load_design()
    for (name, track, index), cfg in d.arm_configs(spec).items():
        bases = cfg["tunable"]["classifier_base_paths" if track == "pd" else "regressor_base_paths"]
        counts[name] = counts.get(name, 0) + len(bases) * int(cfg["corpus"]["n_splits"])
    assert counts == {"E1": 32, "E2": 64, "E4": 32, "E6": 288, "E7": 192, "E8": 64,
                      "E9": 32, "E10": 72, "E12": 100, "E3": 24}
    assert sum(counts.values()) == 900              # 800 standard runs + 100 single-source runs


def test_numbering_follows_the_research_questions():
    spec = d.load_design()["experiments"]
    numbers = [int(name[1:]) for name in spec]
    assert numbers == list(range(1, 13))
    rqs = [spec[f"E{i}"]["rq"] for i in numbers]
    assert rqs == sorted(rqs) and set(rqs) == {1, 2, 3, 4}
    assert [spec[f"E{i}"]["slug"] for i in numbers] == [
        "reference", "seed", "precision", "domain_control", "target_finetune", "update_size_anchoring", "scope",
        "exposure", "table_size", "episode_size", "context", "transfer"]
    assert spec["E3"]["tasks"] == ["pd", "lgd"]


def test_precision_arms_change_only_the_precision():
    arms = d.load_design()["experiments"]["E3"]["arms"]
    assert [(a["training_precision"], a["models"]) for a in arms] == [("bf16", ["v2.6", "v3"]), ("fp32", ["tabicl-v2"])]
    reference = d.reference_config("pd")["train"]
    assert (reference["amp"], reference["amp_dtype"], reference["tabicl_precision"]) == (True, "fp16", "fp16")
    bf16 = d.apply_arm(d.reference_config("pd"), arms[0], "pd")["train"]
    fp32 = d.apply_arm(d.reference_config("pd"), arms[1], "pd")["train"]
    assert (bf16["amp"], bf16["amp_dtype"]) == (True, "bf16")
    assert (fp32["amp"], fp32["tabicl_precision"]) == (False, "fp32")
    assert d.load_design()["experiments"]["E8"]["arms"] == [{"exposure": "full_pass"}, {"successful_updates": 10000}]


def test_unknown_deviations_are_refused():
    with pytest.raises(ValueError, match="Undeclared deviation"):
        d.apply_arm(d.reference_config("lgd"), {"dropout": 0.1}, "lgd")
    with pytest.raises(ValueError, match="PD-only"):
        d.apply_arm(d.reference_config("lgd"), {"pd_class_sampling": "balanced"}, "lgd")
    with pytest.raises(ValueError, match="corpus policy"):
        d.apply_arm(d.reference_config("pd"), {"corpus": "everything"}, "pd")


def test_lgd_never_uses_class_balancing():
    for (name, track, _), cfg in d.arm_configs().items():
        if track == "lgd":
            assert cfg["train"]["context_sampling"] == "stratified", name


def test_lora_keeps_the_reference_l2sp_on_the_effective_update():
    for (name, track, index), cfg in d.arm_configs().items():
        if cfg["train"]["adaptation_mode"] in ("lora", "diagscale"):
            assert cfg["tunable"]["l2sp_lambdas"] == [0.003], (name, index)


def test_update_size_grid_cells():
    arms = d.load_design()["experiments"]["E6"]["arms"]
    cells = {(a["learning_rate"], a.get("l2sp_lambda", 0.003)) for a in arms if "learning_rate" in a}
    assert cells == {(1e-7, .003), (1e-6, .003), (3e-6, .003), (1e-5, .003),
                     (3e-6, 0.0), (3e-6, .03), (1e-5, 0.0), (1e-5, .03)}
    # ...plus anchoring off at the reference rate itself.
    assert arms[-1] == {"l2sp_lambda": 0.0}


def test_three_reference_seeds_and_the_rubachev_finetuning_rate():
    spec = d.load_design()["experiments"]
    assert [a["seed"] for a in spec["E2"]["arms"]] == [43, 44]
    assert spec["E5"]["recipe"]["learning_rates"] == [1e-6, 1e-5, 6.5e-5]


def test_transfer_runs_index_every_table_once_and_train_on_one_table():
    spec = d.load_design()
    for track, count in spec["tables"].items():
        arms = d.experiment_arms("E12", track, spec)
        assert [a["single_source"] for a in arms] == list(range(count))
    for (name, _, _), cfg in d.arm_configs(spec).items():
        if name == "E12":
            assert cfg["corpus"]["n_folds"] is None and cfg["corpus"]["n_splits"] == 1
            assert cfg["train"]["target_total_steps"] == 2500
            assert cfg["tunable"]["learning_rates"] == [1e-6]


def test_episode_size_doubles_and_16384_only_for_large_context_models():
    arms = d.load_design()["experiments"]["E10"]["arms"]
    assert [a.get("episode_rows") for a in arms] == [2048, 4096, 8192, 16384, None]   # last: reference size
    assert arms[3]["models"] == ["v3", "tabicl-v2"] and "models" not in arms[-1]
    assert all(a["corpus"] == "tables_above_10000_rows" for a in arms)


def test_generated_configs_resolve_to_valid_training_grids(tmp_path):
    from src.train.config import load_train_config, resolve_grid
    paths = d.materialize(tmp_path)
    assert len(paths) == 78
    for path in paths:
        cfg = load_train_config(config_path=str(path))
        assert resolve_grid(cfg, single=False)


def test_router_accepts_exactly_the_design_slugs():
    from src.utils import paths
    spec = d.load_design()
    slugs = {e["slug"] for e in spec["experiments"].values()}
    assert slugs == set(paths.CAMPAIGN2_GROUPS)
    for (name, track, index), cfg in d.arm_configs(spec).items():
        assert paths.group_for_run(cfg["run_name"]) == cfg["experiment"]["output_group"]
        assert paths.experiment_group(cfg["experiment"]["output_group"]) == cfg["experiment"]["output_group"]


def test_committed_campaign_configs_match_the_design(tmp_path):
    from omegaconf import OmegaConf
    d.materialize(tmp_path)
    committed_root = d.CONFIG_ROOT
    fresh = {p.relative_to(tmp_path).as_posix(): OmegaConf.load(p) for p in tmp_path.rglob("*.yaml")}
    committed = {p.relative_to(committed_root).as_posix(): OmegaConf.load(p)
                 for p in committed_root.rglob("*.yaml") if not p.parent.name.startswith("E00_")}
    assert fresh.keys() == committed.keys()
    assert all(fresh[k] == committed[k] for k in fresh), "regenerate with python -m src.utils.design --write"


def test_domain_control_trains_on_matched_twins_for_both_tasks():
    for track in ("pd", "lgd"):
        cfg = d.arm_configs()[("E4", track, 0)]
        assert cfg["corpus"]["train_source"] == "domain_control" and cfg["corpus"]["n_folds"] == 4


def test_config_folders_follow_the_experiment_numbering():
    spec = d.load_design()
    assert d.experiment_folder("E6", spec).name == "E06_update_size_anchoring"
    assert d.config_path("reference", "pd", 0, spec).parent.name == "E01_reference"
    assert d.PILOT_FOLDER.name == "E00_pilots"
