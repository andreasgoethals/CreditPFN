"""Fast local safeguards for the seed/sampling research and its GPU canary."""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest
import torch

from src.utils import auxiliary_check as aux
from src.train.config import load_train_config, resolve_grid


def test_auxiliary_configs_keep_roles_and_exact_research_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(aux, "select_tables", lambda *args: (
        [NS(dataset_id="train_a"), NS(dataset_id="train_b")], [NS(dataset_id="held")]))
    paths = aux.make_configs(tmp_path)
    assert len(paths) == 13  # Seven PD cases, six LGD cases, four bases per case.
    configs = {p.stem: load_train_config(config_path=str(p)) for p in paths}
    assert sum(len(resolve_grid(c, single=False)) for c in configs.values()) == 52
    for cfg in configs.values():
        assert cfg.corpus.n_splits == 1 and cfg.corpus.split_seed == 1729
        assert set(cfg.corpus.train_dataset_ids).isdisjoint(cfg.corpus.test_dataset_ids)
        assert cfg.train.max_rows_per_step == 256
        assert cfg.train.target_total_steps == 6 and list(cfg.train.trajectory_steps) == [0, 3, 6]
        assert cfg.train.monitor_seed == 31415 and cfg.train.deterministic
        assert list(cfg.tunable.learning_rates) == [3e-7] and list(cfg.tunable.l2sp_lambdas) == [.003]
        assert list(cfg.tunable.frozen_backbone) == [False]
        assert cfg.checkpoint.trained_dir == "checkpoints/trained/experiment0"
        assert cfg.experiment.output_group == "experiment0"
        assert list(cfg.experiment.training_seeds) == [cfg.seed]
    for track in ("pd", "lgd"):
        before, after = (configs[f"auxiliary_{track}_{c}"] for c in ("seed42", "seed43"))
        assert before.seed == 42 and after.seed == 43
        assert before.corpus == after.corpus and before.train == after.train
        assert before.train.context_sampling == ("balanced" if track == "pd" else "stratified")
        for mode in ("full_pass", "accumulate"):
            cfg = configs[f"auxiliary_{track}_{mode}"]
            assert cfg.train.context_sampling == "stratified"
            assert list(cfg.tunable.epoch_pass_modes) == [mode]
            resumed = configs[f"auxiliary_{track}_{mode}_resumed"]
            assert cfg.train == resumed.train and cfg.tunable == resumed.tunable and cfg.seed == resumed.seed


def test_auxiliary_selection_stays_inside_existing_fold_and_bounds_compute(monkeypatch):
    split = NS(train=[NS(dataset_id=f"tr{i}", n_rows=n) for i, n in enumerate([200, 700, 500, 90000])],
               test=[NS(dataset_id="te", n_rows=300)])
    monkeypatch.setattr(aux, "split_from_cfg", lambda _: split)
    limits = aux.policy()
    train, held = aux.select_tables(None, limits)
    assert [d.n_rows for d in train] == [500, 700] and held == split.test
    limits.max_training_rows = 1000
    with pytest.raises(ValueError, match="size limit"):
        aux.select_tables(None, limits)


def test_auxiliary_seed_check_rejects_unchanged_training_or_changed_baseline(monkeypatch):
    from src.utils import recovery_check as recovery
    monkeypatch.setattr(recovery, "compare", lambda *a: {"max_non_diagnostic_difference": .01})
    curve = pd.DataFrame({"successful_updates": [0,3,6], "metric__test__table": [.7,.71,.72]})
    assert aux.seed_report({"path":"a"}, {"path":"b"}, [curve,curve])["passed"]
    changed = curve.copy(); changed.loc[0,"metric__test__table"] = .9
    assert not aux.seed_report({"path":"a"}, {"path":"b"}, [curve,changed])["passed"]
    monkeypatch.setattr(recovery, "compare", lambda *a: {"max_non_diagnostic_difference": 0.})
    assert not aux.seed_report({"path":"a"}, {"path":"b"}, [curve,curve])["passed"]


def test_accumulated_anchor_is_added_once_and_clipped_after_mean():
    from src.train.optimization import step_mean_gradient
    # Different chunk objectives, same parameter: averaging must keep lambda fixed.
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad(): model.weight.fill_(2.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    for target in (0., 1., 3.):
        loss = (model.weight-target).square().mean() + .003*(model.weight-1.).square().sum()
        loss.backward()
    expected_grad = np.mean([2*(2-target) for target in (0,1,3)]) + .006
    norm, ok = step_mean_gradient(model, optimizer, scaler, microbatches=3, max_norm=None)
    assert ok and norm == pytest.approx(expected_grad)
    assert model.weight.item() == pytest.approx(2-.1*expected_grad)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_outer_test_rows_never_enter_inner_fit_or_validation(task):
    from src.eval.benchmark import _make_outer_folds, _inner_split
    y = np.tile([0,1],100) if task == "classification" else np.linspace(0,1,200)
    observed = []
    for fold, (train, test) in enumerate(_make_outer_folds(y, task_type=task, n_folds=5, seed=99)):
        fit, validation = _inner_split(train, y[train], task_type=task, val_fraction=.2, seed=99+fold)
        assert set(fit).isdisjoint(validation) and set(fit).isdisjoint(test) and set(validation).isdisjoint(test)
        assert set(fit) | set(validation) == set(train)
        observed.extend(test)
    assert sorted(observed) == list(range(200))


def test_auxiliary_cpu_audit_needs_all_eight_matching_reports(tmp_path):
    from src.utils.atomic import write_json
    for track in ("pd","lgd"):
        for trial in range(4):
            write_json(tmp_path/f"auxiliary_{track}_{trial}.json", dict(track=track,trial=trial,passed=True,
                exposure_passed=True,
                arms=[{"case":c} for c in aux.cases(track)], seed={"passed":True}, benchmark={"passed":True},
                recovery={m:{"passed":True} for m in ("full_pass","accumulate")}))
    assert len(aux.audit_reports(tmp_path)) == 8
    write_json(tmp_path/"auxiliary_pd_0.json", {"track":"lgd", "trial":0, "passed":True})
    with pytest.raises(RuntimeError, match="incomplete, mismatched or failed"):
        aux.audit_reports(tmp_path)


def test_auxiliary_task_failure_sends_failed_callback(monkeypatch):
    from contextlib import contextmanager
    from pathlib import Path
    from src.utils import experiment0 as flow
    @contextmanager
    def locked(_): yield Path("unused"), {"phase":"auxiliary"}
    monkeypatch.setattr(flow, "locked", locked)
    monkeypatch.setattr(flow, "root", lambda: Path("unused"))
    def fail(*args): raise RuntimeError("Bad gradients")
    monkeypatch.setattr(aux, "run", fail)
    calls = []
    monkeypatch.setattr(flow, "complete", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="Bad gradients"):
        aux.main(["--id","abc","--task","7"])
    assert calls == [("abc","auxiliary","lgd",3,1)]


def test_auxiliary_workflow_waits_for_all_tasks_then_stops_at_its_own_receipt(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from src.utils import experiment0 as flow, submit_bounded
    folder = tmp_path / "abc"
    folder.mkdir()
    state = {"id": "abc", "part": "auxiliary", "jobs": []}
    @contextmanager
    def locked(_):
        yield folder / "state.json", state
    submitted, audits = [], []
    monkeypatch.setattr(flow, "locked", locked)
    monkeypatch.setattr(flow, "root", lambda: tmp_path)
    monkeypatch.setattr(submit_bounded, "submit", lambda command, **kwargs:
                        submitted.append((command, kwargs)) or "123;mindwell")
    monkeypatch.setattr(flow, "_cpu", lambda *args: audits.append(args) or "456")
    monkeypatch.setattr(aux, "audit_reports", lambda _: [{"passed": True}] * 8)
    flow.launch("abc", "auxiliary")
    command, bounds = submitted[0]
    assert "--array=0-7%4" in command and command[-1] == "scripts/slurm/auxiliary.slurm"
    assert bounds["limit"] == 4 and bounds["cluster"] == "mindwell"
    assert state["jobs"] == [{"phase": "auxiliary", "cluster": "mindwell", "id": "123"}]
    for task in range(7):
        flow.complete("abc", "auxiliary", "pd" if task < 4 else "lgd", task % 4, 0)
        assert not audits
    flow.complete("abc", "auxiliary", "lgd", 3, 0)
    assert audits == [("abc", "audit", "auxiliary")]
    flow.audit("abc", "auxiliary")
    assert state["status"] == "passed" and (tmp_path / "auxiliary_passed.json").is_file()
    assert len(submitted) == len(audits) == 1
