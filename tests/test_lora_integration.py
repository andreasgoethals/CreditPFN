"""LoRA protocol, production export, naming and storage integration boundaries."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn.utils import parametrize

from src.model import lora
from src.train.config import load_train_config, resolve_grid


def adapted_linear():
    model = nn.Module()
    model.projection = nn.Linear(5, 4, bias=False)
    model.requires_grad_(False)
    adapter = lora.LowRankWeight(model.projection.weight, projections=['q'],
                                 slices=[(0, 4)], rank=2, alpha=4)
    parametrize.register_parametrization(model.projection, 'weight', adapter)
    model._creditpfn_lora = dict(schema=lora.SCHEMA, targets=[dict(
        module='projection', weight='weight', slices=[[0, 4]])])
    with torch.no_grad():
        adapter.B_q.fill_(.05)
    return model


@pytest.mark.parametrize('track', ['pd', 'lgd'])
def test_production_grid_and_names_are_separate_and_unambiguous(track):
    from src.visualize.campaign import planned_trials
    from src.visualize.training_viz import parse_trial_name
    from src.utils.paths import group_for_run
    cfg = load_train_config(config_path=f'config/experiment6/{track}.yaml')
    assert len(resolve_grid(cfg, single=False)) == 6
    assert cfg.corpus.n_splits == cfg.corpus.n_folds == 4
    assert cfg.train.target_total_steps == 10000
    assert cfg.lora.r == 16 and cfg.lora.alpha == 32 and cfg.lora.dropout == 0
    assert cfg.experiment.lora_gate and not cfg.lora.enabled
    assert dict(cfg.evaluation.tabicl_inference_options) == dict(use_amp=False, use_fa3=False)
    roster = planned_trials(cfg)
    assert len(roster) == roster.trial_name.nunique() == 24
    for name in roster.trial_name:
        assert name.endswith('_lora')
        assert parse_trial_name(name).adaptation_mode == 'lora'
        assert group_for_run(parse_trial_name(name).run_name) == 'experiment8'
    old = load_train_config(config_path=f'config/experiment1/{track}.yaml')
    assert set(roster.trial_name).isdisjoint(planned_trials(old).trial_name)
    assert cfg.train.context_sampling == old.train.context_sampling
    assert cfg.train.amp == old.train.amp


@pytest.mark.parametrize('override', [
    'lora.enabled=true', 'tunable.frozen_backbone=[true]',
    'tunable.l2sp_lambdas=[0.003]', 'optimizer.weight_decay=0.01',
    'lora.r=0', 'lora.alpha=0', 'lora.dropout=0.1',
    'lora.target_modules=[q_projection]',
    'tunable.classifier_base_paths=[checkpoints/tabpfn-v3.5-classifier.ckpt]',
])
def test_incompatible_lora_designs_fail_before_loading_weights(override):
    cfg = load_train_config([override], config_path='config/experiment6/pd.yaml')
    with pytest.raises(ValueError):
        resolve_grid(cfg, single=False)


def test_workflow_routing_is_scoped_on_both_storage_tiers(tmp_path, monkeypatch):
    from src.utils import paths
    monkeypatch.setattr(paths, 'outputs_dir', lambda: tmp_path / 'data' / 'output CreditPFN')
    monkeypatch.setattr(paths, 'resolve_staging_path', lambda p: tmp_path / 'project' / p)
    monkeypatch.setenv('CREDITPFN_LORA_WORKFLOW', 'caseABC123')
    suffix = 'experiment0/lora/caseABC123'
    assert paths.logs_dir('experiment0') == tmp_path / 'data/output CreditPFN' / suffix / 'logs'
    assert paths.manifests_dir('experiment0').parent.name == 'caseABC123'
    assert paths.training_dir(experiment='experiment0') == tmp_path / 'project/output CreditPFN' / suffix / 'training'
    assert paths.results_dir(experiment='experiment0').parent.name == 'caseABC123'
    assert paths.logs_dir('experiment1') == tmp_path / 'data/output CreditPFN/experiment1/logs'
    assert paths.training_dir(experiment='experiment8') == tmp_path / 'project/output CreditPFN/experiment8/training'
    monkeypatch.setenv('CREDITPFN_LORA_WORKFLOW', '../experiment1')
    with pytest.raises(ValueError):
        paths.logs_dir('experiment0')


def test_optimizer_contains_only_adapter_factors():
    from src.train.loop import _make_optimizer_and_scheduler
    cfg = load_train_config(config_path='config/experiment6/pd.yaml')
    model = adapted_linear()
    optimizer, _ = _make_optimizer_and_scheduler(model, cfg, total_steps=4)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {
        id(p) for p in lora.adapter_parameters(model)}
    assert id(model.projection.parametrizations.weight.original) not in {
        id(p) for g in optimizer.param_groups for p in g['params']}


@pytest.mark.parametrize('kind', ['tabicl', 'tabpfn_final', 'tabpfn_monitor'])
def test_all_native_save_paths_merge_without_mutating_training(tmp_path, monkeypatch, kind):
    from src.train.model import save_finetuned
    from src.train.tabicl_model import save_finetuned_tabicl
    from src.train.loop import _save_eval_snapshot
    import tabpfn.model_loading as loading
    monkeypatch.setattr(loading, '_resolve_architecture_name', lambda c: 'tabpfn_v3', raising=False)
    model = adapted_linear()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ids = [id(p) for p in lora.adapter_parameters(model)]
    expected = model.projection(torch.randn(3, 5))
    path = tmp_path / f'{kind}.ckpt'
    if kind == 'tabicl':
        save_finetuned_tabicl(model, {'recompute': True}, path)
    else:
        saver = save_finetuned if kind == 'tabpfn_final' else _save_eval_snapshot
        saver(model, SimpleNamespace(), path, inference_config={'FIXED': True})
    payload = torch.load(path, weights_only=False)
    state = payload['state_dict']
    assert set(state) == {'projection.weight'}
    assert torch.equal(state['projection.weight'], model.projection.weight)
    assert list(before) == list(model.state_dict())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert ids == [id(p) for p in lora.adapter_parameters(model)]
    if kind.startswith('tabpfn'):
        assert payload['architecture_name'] == 'tabpfn_v3'
        assert payload['inference_config'] == {'FIXED': True}
    # Saving must leave the live graph usable for the next training update.
    expected.sum().backward()
    assert all(p.grad is not None for p in lora.adapter_parameters(model))
