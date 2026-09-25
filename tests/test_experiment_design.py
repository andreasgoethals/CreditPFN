"""Scientific identities, exact budgets and uninterrupted-versus-recovered training."""
from pathlib import Path
import os
import random
from types import SimpleNamespace as NS

import pytest
import torch
from omegaconf import OmegaConf

from tests.test_train import synthetic_processed, _DummyClassifier


@pytest.fixture
def execution_settings(monkeypatch):
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn
        torch.backends.cudnn.benchmark = benchmark
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def test_execution_policy_sets_workspace_before_cuda_and_keeps_errors_strict(monkeypatch, execution_settings):
    from src.train.recovery import configure_execution
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    torch.use_deterministic_algorithms(True, warn_only=True)
    result = configure_execution(True)
    assert result["deterministic_algorithms"]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    configure_execution(False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic


def test_execution_policy_refuses_to_change_workspace_after_cuda_init(monkeypatch, execution_settings):
    from src.train.recovery import configure_execution
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="before initializing CUDA"):
        configure_execution(True)
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    assert configure_execution(True)["cublas_workspace_config"] == ":16:8"


@pytest.mark.parametrize("value,expected", [(None, 1000), (200, 200), (2000, 1000)])
def test_debug_row_limit_only_lowers_measured_cap(value, expected):
    from src.train.config import limit_training_rows
    assert limit_training_rows(OmegaConf.create({"train": {"max_rows_per_step": value}}), 1000) == expected


@pytest.mark.parametrize("value", [0, 3, True, 12.5, "200"])
def test_debug_row_limit_rejects_invalid_values(value):
    from src.train.config import limit_training_rows
    with pytest.raises(ValueError, match="integer >= 4"):
        limit_training_rows(OmegaConf.create({"train": {"max_rows_per_step": value}}), 1000)


def test_only_recovery_controls_request_strict_numerics_and_small_batches():
    from src.train.config import load_train_config
    from src.utils.experiment import digest_json, scientific_config
    for path in Path("config").glob("experiment*/*.yaml"):
        cfg = load_train_config(config_path=str(path))
        recovery = path.stem.startswith("recovery_")
        assert cfg.train.deterministic is recovery
        assert cfg.train.max_rows_per_step == (2048 if recovery else None)
    cfg = load_train_config(config_path="config/experiment0/recovery_pd.yaml")
    before = digest_json(scientific_config(cfg))
    cfg.train.deterministic = False
    assert digest_json(scientific_config(cfg)) != before
    cfg.train.deterministic = True
    cfg.train.max_rows_per_step = 1024
    assert digest_json(scientific_config(cfg)) != before


def test_main_and_sampling_grid_counts_and_shared_partitions():
    from src.train.config import load_train_config, resolve_grid
    from src.utils.experiment import apply_split_index
    from src.train.corpus import _assign_folds
    for track, size in (("pd", 17), ("lgd", 8)):
        main = load_train_config(config_path=f"config/experiment1/{track}.yaml")
        sampling = load_train_config(config_path=f"config/experiment3/{track}.yaml")
        assert len(resolve_grid(main, single=False)) * main.corpus.n_splits == 256
        assert len(resolve_grid(sampling, single=False)) * sampling.corpus.n_splits == 48
        assert set(sampling.tunable.epoch_pass_modes) == {"one_sample", "full_pass", "accumulate"}
        assert sampling.train.context_sampling == "stratified"
        seeds = load_train_config(config_path=f"config/experiment2/{track}.yaml")
        assert len(resolve_grid(seeds, single=False)) * seeds.corpus.n_splits == 16
        assert list(seeds.experiment.training_seeds) == [43]
        assert seeds.seed == 43
        assert seeds.corpus.split_seed == main.corpus.split_seed
        assert seeds.train.context_sampling == main.train.context_sampling
        assert seeds.train.monitor_seed == main.train.monitor_seed
        from src.eval.config import load_eval_configs
        main_eval, _ = load_eval_configs([], [], config_path=f"config/experiment1/{track}.yaml")
        seed_eval, _ = load_eval_configs([], [], config_path=f"config/experiment2/{track}.yaml")
        assert seed_eval.seed == main_eval.seed == 99
        assert list(seeds.tunable.learning_rates) == [3e-7]
        assert list(seeds.tunable.l2sp_lambdas) == [.003]
        assert list(seeds.tunable.frozen_backbone) == [False]
        assert list(seeds.tunable.epoch_pass_modes) == ['one_sample']
        test_sets = []
        for index in range(4):
            cfg = apply_split_index(OmegaConf.create(OmegaConf.to_container(main)), index)
            repeat = apply_split_index(OmegaConf.create(OmegaConf.to_container(sampling)), index)
            assert cfg.seed == repeat.seed == 42
            assert cfg.corpus.fold == repeat.corpus.fold == index
            assert cfg.corpus.split_seed == repeat.corpus.split_seed == 1729
            assigned = _assign_folds([str(i) for i in range(size)], n_folds=4,
                                           fold=index, seed=1729)
            test_sets.append({k for k, v in assigned.items() if v == "test"})
        assert len(set.union(*test_sets)) == sum(map(len, test_sets)) == size


def test_pass_modes_have_distinct_names_and_evaluation_directories():
    from src.train.loop import descriptive_name
    from src.eval.benchmark import _method_dirname
    from src.model.base import ModelHandle
    names, directories = set(), set()
    for mode in ("one_sample", "full_pass", "accumulate"):
        names.add(descriptive_name(run_name="r", track="pd", base_path="model.ckpt",
                                   learning_rate=1e-6, seed=42, epoch_pass_mode=mode))
        handle = ModelHandle(name="r", source="tabpfn-trained", track="pd", task_type="classification",
                             extra={"base_checkpoint": "model.ckpt", "learning_rate": 1e-6,
                                    "epoch_pass_mode": mode})
        directories.add(_method_dirname(handle))
    assert len(names) == len(directories) == 3


def test_fingerprint_covers_scientific_changes_but_not_workers():
    from src.train.config import load_train_config
    from src.utils.experiment import scientific_config, digest_json
    cfg = load_train_config(config_path="config/experiment1/pd.yaml")
    before = digest_json(scientific_config(cfg))
    cfg.train.dataloader_workers = 8
    assert digest_json(scientific_config(cfg)) == before
    cfg.train.target_total_steps = 6000
    assert digest_json(scientific_config(cfg)) != before


@pytest.mark.parametrize("mode", ["one_sample", "full_pass", "accumulate"])
@pytest.mark.parametrize("terminal_divergence", [False, True])
def test_exact_budget_and_mid_epoch_recovery_match_uninterrupted(
        mode, terminal_divergence, synthetic_processed, tmp_path, monkeypatch, execution_settings):
    import src.train.loop as loop
    import src.train.recovery as recovery
    snapshots = {}
    provenances = {}

    class DropoutModel(_DummyClassifier):
        def forward(self, x, y, **kwargs):
            return super().forward(torch.nn.functional.dropout(x, p=.2, training=self.training), y, **kwargs) * (0.5 + random.random())

    def fake_loader(path, **kwargs):
        return DropoutModel(4), torch.nn.CrossEntropyLoss(), NS(num_features=4, num_classes=2), None

    def fake_save(model, arch, path, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"toy")
        snapshots[path.name] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        provenances[path.name] = kwargs.get("provenance", {})
        return path

    monkeypatch.setattr(loop, "load_tabpfn_for_training", fake_loader)
    monkeypatch.setattr(loop, "save_finetuned", fake_save)
    real_load = OmegaConf.load
    monkeypatch.setattr(OmegaConf, "load", lambda p: OmegaConf.create({"finetuning": {
        "max_rows_per_epoch": 80, "query_fraction": .4, "context_sampling": "balanced"}})
        if "data.yaml" in str(p) else real_load(p))
    cfg = OmegaConf.create({"seed": 42, "run_name": "toy", "track": "pd", "device": "cpu",
        "tunable": {"classifier_base_paths": ["base.ckpt"], "learning_rates": [.001]},
        "corpus": {"train_fraction": .6, "test_fraction": .4},
        "optimizer": {"weight_decay": 0., "l2sp_lambda": .003},
        "scheduler": {"warmup_fraction": .1},
        "train": {"epochs": 2, "target_total_steps": 7, "max_epochs_for_step_budget": 20,
                  "deterministic": True, "max_rows_per_step": 40,
                  "grad_clip_norm": 1., "amp": False, "dataloader_workers": 0,
                  "context_sampling": "stratified",
                  "epoch_eval_subsample_samples": 0, "n_estimators_finetune": 1,
                  "trajectory_steps": [0, 3, 7], "recovery_every_updates": 3,
                  "numerical_stopping_only": True},
        "checkpoint": {"trained_dir": str(tmp_path / "weights")}})
    if terminal_divergence:
        import src.train.optimization as optimization
        original_step = optimization.step_mean_gradient
        cfg.train.divergence_patience = 1
        def fail_after_three(model, optimizer, scaler, **kwargs):
            if optimizer.state and max(float(s.get("step", 0)) for s in optimizer.state.values()) >= 3:
                optimizer.zero_grad(set_to_none=True)
                return float("nan"), False
            return original_step(model, optimizer, scaler, **kwargs)
        monkeypatch.setattr(optimization, "step_mean_gradient", fail_after_three)
    expected_trajectory = []
    expected = loop.train_one_config(cfg, pass_mode=mode, save_path=tmp_path / "continuous.ckpt",
                                     on_trajectory_end=expected_trajectory.append)
    real_save = recovery.save_recovery

    def interrupt_after_three(*args, **kwargs):
        real_save(*args, **kwargs)
        if kwargs["progress"]["successful_updates"] == 3:
            raise recovery.TrainingInterrupted("simulated allocation interruption")

    monkeypatch.setattr(recovery, "save_recovery", interrupt_after_three)
    with pytest.raises(recovery.TrainingInterrupted):
        loop.train_one_config(cfg, pass_mode=mode, save_path=tmp_path / "resumed.ckpt")
    monkeypatch.setattr(recovery, "save_recovery", real_save)
    trajectory = []
    actual = loop.train_one_config(cfg, pass_mode=mode, save_path=tmp_path / "resumed.ckpt",
                                  on_trajectory_end=trajectory.append)
    assert actual.total_optimizer_steps == expected.total_optimizer_steps
    assert actual.diverged == expected.diverged == terminal_divergence
    if not terminal_divergence:
        assert actual.total_optimizer_steps == 7
    assert actual.rows_seen == expected.rows_seen
    assert [r.successful_updates for r in trajectory] == [r.successful_updates for r in expected_trajectory]
    assert all(torch.equal(snapshots["continuous.ckpt"][k], v)
               for k, v in snapshots["resumed.ckpt"].items())
    assert provenances["resumed.ckpt"]["hyperparameters"]["execution"]["deterministic_algorithms"]
    assert provenances["resumed.ckpt"]["hyperparameters"]["max_rows_per_epoch"] == 40
    assert not (tmp_path / "resumed.ckpt.resume.pt").exists()


def test_monitoring_restores_rng_and_module_modes():
    import random
    import numpy as np
    from src.train.recovery import capture_rng, restore_rng, preserve_random_state
    model = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(2, 1)).train()
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(state)
    with preserve_random_state(model):
        model.eval()
        random.seed(1)
        np.random.seed(2)
        torch.manual_seed(3)
        torch.rand(10)
    assert model.training and all(m.training for m in model.modules())
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
