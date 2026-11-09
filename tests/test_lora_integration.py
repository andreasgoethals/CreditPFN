"""LoRA protocol, production export, naming and storage integration boundaries."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from src.model import lora


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


@pytest.mark.parametrize('kind', ['tabicl', 'tabpfn_final', 'tabpfn_monitor'])
def test_all_native_save_paths_merge_without_mutating_training(tmp_path, monkeypatch, kind):
    import tabpfn.model_loading as loading

    from src.train.loop import _save_eval_snapshot
    from src.train.model import save_finetuned
    from src.train.tabicl_model import save_finetuned_tabicl
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
