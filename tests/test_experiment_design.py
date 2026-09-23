"""Scientific identities, exact budgets and uninterrupted-versus-recovered training."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from omegaconf import OmegaConf

from tests.test_train import synthetic_processed, _DummyClassifier


def test_main_and_seed_grid_counts_and_shared_partitions():
    from scripts.train_pipeline import _load_cfg, _resolve_grid, _apply_split_index
    from src.train.corpus import _assign_folds
    for track, size in (("pd", 17), ("lgd", 8)):
        main = _load_cfg(config_path=f"config/experiment1_{track}.yaml")
        seeds = _load_cfg(config_path=f"config/seed_check_{track}.yaml")
        assert len(_resolve_grid(main, single=False)) * main.corpus.n_splits == 256
        assert len(_resolve_grid(seeds, single=False)) * seeds.corpus.n_splits == 64
        test_sets = []
        for index in range(4):
            cfg = _apply_split_index(OmegaConf.create(OmegaConf.to_container(main)), index)
            repeat = _apply_split_index(OmegaConf.create(OmegaConf.to_container(seeds)), index + 4)
            assert cfg.seed == 42 and repeat.seed == 44
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
    from scripts.train_pipeline import _load_cfg
    from src.utils.experiment import scientific_config, digest_json
    cfg = _load_cfg(config_path="config/experiment1_pd.yaml")
    before = digest_json(scientific_config(cfg))
    cfg.train.dataloader_workers = 8
    assert digest_json(scientific_config(cfg)) == before
    cfg.train.target_total_steps = 6000
    assert digest_json(scientific_config(cfg)) != before


@pytest.mark.parametrize("mode", ["one_sample", "full_pass", "accumulate"])
def test_exact_budget_and_mid_epoch_recovery_match_uninterrupted(
        mode, synthetic_processed, tmp_path, monkeypatch):
    import src.train.loop as loop
    import src.train.recovery as recovery
    snapshots = {}

    class DropoutModel(_DummyClassifier):
        def forward(self, x, y, **kwargs):
            return super().forward(torch.nn.functional.dropout(x, p=.2, training=self.training), y, **kwargs)

    def fake_loader(path, **kwargs):
        return DropoutModel(4), torch.nn.CrossEntropyLoss(), NS(num_features=4, num_classes=2), None

    def fake_save(model, arch, path, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"toy")
        snapshots[path.name] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return path

    monkeypatch.setattr(loop, "load_tabpfn_for_training", fake_loader)
    monkeypatch.setattr(loop, "save_finetuned", fake_save)
    real_load = OmegaConf.load
    monkeypatch.setattr(OmegaConf, "load", lambda p: OmegaConf.create({"finetuning": {
        "max_rows_per_epoch": 80, "query_fraction": .4}}) if "data.yaml" in str(p) else real_load(p))
    cfg = OmegaConf.create({"seed": 42, "run_name": "toy", "track": "pd", "device": "cpu",
        "tunable": {"classifier_base_paths": ["base.ckpt"], "learning_rates": [.001]},
        "corpus": {"train_fraction": .6, "test_fraction": .4},
        "optimizer": {"weight_decay": 0., "l2sp_lambda": .003},
        "scheduler": {"warmup_fraction": .1},
        "train": {"epochs": 2, "target_total_steps": 7, "max_epochs_for_step_budget": 20,
                  "grad_clip_norm": 1., "amp": False, "dataloader_workers": 0,
                  "epoch_eval_subsample_samples": 0, "n_estimators_finetune": 1,
                  "trajectory_steps": [0, 3, 7], "recovery_every_updates": 3,
                  "numerical_stopping_only": True},
        "checkpoint": {"trained_dir": str(tmp_path / "weights")}})
    expected = loop.train_one_config(cfg, pass_mode=mode, save_path=tmp_path / "continuous.ckpt")
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
    assert actual.total_optimizer_steps == expected.total_optimizer_steps == 7
    assert actual.rows_seen == expected.rows_seen
    assert [r.successful_updates for r in trajectory] == [0, 3, 7]
    assert all(torch.equal(snapshots["continuous.ckpt"][k], v)
               for k, v in snapshots["resumed.ckpt"].items())
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
