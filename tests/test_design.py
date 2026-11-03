"""The second campaign is the reference recipe plus declared deviations, and nothing else."""
import pytest

from src.utils import design as d


def test_every_arm_differs_from_the_reference_only_by_its_declared_deviations():
    spec = d.load_design()
    for (name, track, index), cfg in d.arm_configs(spec).items():
        arm = spec["experiments"][name]["arms"][index]
        assert d.undeclared_differences(cfg, arm, track, spec) == [], (name, track, index)


def test_reference_recipe_matches_the_agreed_values():
    ref = d.reference_config("pd")
    assert ref["tunable"]["learning_rates"] == [3e-7]
    assert ref["optimizer"] == {"name": "adamw", "weight_decay": 0.0, "l2sp_lambda": 0.003}
    assert ref["tunable"]["query_fractions"] == [0.40]
    assert ref["train"]["context_sampling"] == "stratified"          # proportional PD sampling
    assert ref["train"]["target_total_steps"] == 10000
    assert ref["train"]["max_rows_per_step"] is None                 # model maxima from data.yaml
    assert ref["tunable"]["epoch_pass_modes"] == ["one_sample"]
    assert len(ref["tunable"]["classifier_base_paths"]) == 4


def test_monitoring_is_dense_early_then_every_250_updates():
    points = d.monitor_updates(10000)
    assert points[:7] == [0, 50, 100, 150, 200, 250, 500]
    assert points[-1] == 10000 and len(points) == 45
    assert all(b - a == 250 for a, b in zip(points[5:], points[6:]))
    assert d.monitor_updates(20000)[-1] == 20000


def test_run_counts_match_the_design():
    counts = {}
    spec = d.load_design()
    for (name, track, index), cfg in d.arm_configs(spec).items():
        bases = cfg["tunable"]["classifier_base_paths" if track == "pd" else "regressor_base_paths"]
        counts[name] = counts.get(name, 0) + len(bases) * spec["partitions"]
    assert counts == {"E1": 32, "E2": 32, "E3": 32, "E4": 288, "E5": 288, "E6": 16,
                      "E7": 32, "E8": 32, "E9": 16, "E10": 24}
    assert sum(counts.values()) == 792


def test_unknown_deviations_are_refused():
    with pytest.raises(ValueError, match="Undeclared deviation"):
        d.apply_arm(d.reference_config("lgd"), {"dropout": 0.1}, "lgd")
    with pytest.raises(ValueError, match="PD-only"):
        d.apply_arm(d.reference_config("lgd"), {"pd_class_sampling": "balanced"}, "lgd")


def test_lgd_never_uses_class_balancing():
    for (name, track, _), cfg in d.arm_configs().items():
        if track == "lgd":
            assert cfg["train"]["context_sampling"] == "stratified", name


def test_generated_configs_resolve_to_valid_training_grids(tmp_path):
    from src.train.config import load_train_config, resolve_grid
    paths = d.materialize(tmp_path)
    assert len(paths) == 50
    for path in paths:
        cfg = load_train_config(config_path=str(path))
        assert resolve_grid(cfg, single=False)


def test_router_accepts_exactly_the_design_slugs():
    from src.utils import paths
    spec = d.load_design()
    slugs = {e["slug"] for e in spec["experiments"].values()} | {e["slug"] for e in spec["optional"].values()}
    assert slugs == set(paths.CAMPAIGN2_GROUPS)
    for (name, track, index), cfg in d.arm_configs(spec).items():
        assert paths.group_for_run(cfg["run_name"]) == cfg["experiment"]["output_group"]
        assert paths.experiment_group(cfg["experiment"]["output_group"]) == cfg["experiment"]["output_group"]
