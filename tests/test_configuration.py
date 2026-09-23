"""Configuration precedence and grid edge cases shared by all entry points."""
from omegaconf import OmegaConf
import pytest

from src.eval.config import load_eval_configs
from src.train.config import load_train_config, resolve_grid, training_members


def test_eval_cli_overrides_phase_but_retains_partition_and_other_phase_defaults():
    phase = load_train_config(config_path="config/experiment0_pd.yaml")
    requested = not bool(phase.evaluation.results.save_predictions)
    eval_cfg, train_cfg = load_eval_configs(
        [f"results.save_predictions={str(requested).lower()}"], ["train.grad_clip_norm=2.5"],
        config_path="config/experiment0_pd.yaml", split_index=0)
    assert eval_cfg.results.save_predictions == requested
    assert eval_cfg.cache_controls == phase.evaluation.cache_controls
    assert eval_cfg.cv.n_folds == OmegaConf.load("config/eval.yaml").cv.n_folds
    assert train_cfg.train.grad_clip_norm == 2.5 and train_cfg.corpus.fold == 0
    assert train_cfg.seed == phase.experiment.training_seeds[0]
    assert train_cfg.run_name == phase.run_name + "_s00"


def test_scalar_learning_rate_and_single_respect_family_filter():
    cfg = load_train_config(config_path="config/experiment0_pd.yaml")
    cfg.tunable.learning_rates = 1e-6
    cfg.tunable.frozen_backbone = [True]
    cfg.tunable.adapter_families = ["tabicl"]
    grid = resolve_grid(cfg, single=False)
    assert len(grid) == 1 and "tabicl" in grid[0][0] and grid[0][1] == 1e-6
    assert resolve_grid(cfg, single=True) == grid
    cfg.tunable.adapter_families = ["no-such-family"]
    with pytest.raises(SystemExit, match="no trials remain"):
        resolve_grid(cfg, single=True)


@pytest.mark.parametrize("axis", ["learning_rates", "l2sp_lambdas", "frozen_backbone",
                                "classifier_base_paths", "epoch_pass_modes"])
def test_empty_grid_axis_fails_before_launch(axis):
    cfg = load_train_config(config_path="config/experiment0_pd.yaml")
    cfg.tunable[axis] = []
    with pytest.raises(SystemExit, match="at least one value"):
        resolve_grid(cfg, single=False)


def test_members_resolve_each_family_and_track():
    cfg = OmegaConf.create({"train": {"n_estimators_finetune": {"pd": 2, "lgd": 8},
                                    "n_estimators_finetune_tabicl": 3}})
    assert training_members(cfg, "pd", "tabpfn") == 2
    assert training_members(cfg, "lgd", "tabpfn") == 8
    assert training_members(cfg, "lgd", "tabicl") == 3
    cfg.train.n_estimators_finetune = 4
    assert training_members(cfg, "pd", "tabpfn") == 4
