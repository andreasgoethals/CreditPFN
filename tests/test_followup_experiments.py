"""Cheap invariants for isolated follow-ups. No released weights or GPU required."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from src.eval import context_curve as cc
from src.model import diagscale as ds


def context_policy():
    """The second-campaign E10 context policy (TabPFN v3), as campaign_tasks builds it."""
    from src.eval.campaign_tasks import policy
    from src.utils.design import load_design
    return policy('v3', load_design())


class SoftmaxScalingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.head_dim = 2, 3
        self.weight = nn.Parameter(torch.randn(2, 3))

    def forward(self, q, n):
        return q * (2 + torch.tanh(q * self.weight))


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.head_dim = 2, 3
        self.in_proj_weight = nn.Parameter(torch.randn(6, 6))
        self.ssmax_layer = None

    def forward(self, q):
        return self.ssmax_layer(q, 4) if self.ssmax_layer else q


def small_model(family):
    model = nn.Module()
    model.attention = SoftmaxScalingMLP() if family == 'tabpfn' else MultiheadAttention()
    return model


@pytest.mark.parametrize('family', ['tabpfn', 'tabicl'])
def test_identity_shapes_flag_parity_and_optimizer_isolation(family):
    from src.train.loop import _make_optimizer_and_scheduler
    from src.train.optimization import step_mean_gradient
    model = small_model(family)
    original = [(p, p.detach().clone(), p.requires_grad) for p in model.parameters()]
    shape = (3, 4, 2, 3) if family == 'tabpfn' else (3, 2, 4, 3)
    q = torch.randn(shape)
    before = model.attention(q, 4) if family == 'tabpfn' else model.attention(q)
    params = ds.inject(model, family=family)
    after = model.attention(q, 4) if family == 'tabpfn' else model.attention(q)
    assert torch.equal(before, after)
    assert len(params) == 1 and params[0].numel() == 6 and params[0].dtype == torch.float32
    wrapped = next(m for m in model.modules() if isinstance(m, ds.QueryScale))
    assert wrapped.cast_multiplier == (family == 'tabicl')
    info = model._creditpfn_freeze_info
    assert info['trainable_params'] == 6 and info['original_weights_updated'] is False
    assert info['trainable_parameter_names'] == list(ds.adapter_state(model))
    assert info['backbone_autograd_preserved'] == (family == 'tabicl')
    for p, _, flag in original:
        assert p.requires_grad == (flag if family == 'tabicl' else False)
    cfg = OmegaConf.create({'optimizer': {'lr': .1, 'weight_decay': .01}, 'scheduler': {'warmup_fraction': .1, 'min_lr_fraction': .05}})
    opt, _ = _make_optimizer_and_scheduler(model, cfg, total_steps=6)
    assert [id(p) for g in opt.param_groups for p in g['params']] == [id(p) for p in params]
    # This test isolates a nonzero optimizer step from the schedule's zero-LR warmup start.
    opt.param_groups[0]['lr'] = .1
    after.square().mean().backward()
    for p, _, _ in original:
        if p.requires_grad:
            p.grad = torch.full_like(p, 1e30)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    norm, success = step_mean_gradient(model, opt, scaler, microbatches=2, max_norm=1.)
    assert success and norm < 1e20
    assert torch.count_nonzero(params[0])
    assert all(torch.equal(p, initial) for p, initial, _ in original)
    assert all(p.grad is None for p in model.parameters())


def test_multiplier_is_after_query_dependent_transform():
    original = SoftmaxScalingMLP()
    wrap = ds.QueryScale(original, (2,3), 'cpu')
    with torch.no_grad():
        wrap.delta.fill_(.5)
    q = torch.randn(3,4,2,3)
    assert torch.allclose(wrap(q,4), original(q,4)*1.5)
    assert not torch.allclose(wrap(q,4), original(q*1.5,4))


def test_query_multiplier_matches_family_specific_precision():
    q = torch.ones(1, 2, 3, dtype=torch.bfloat16)
    for cast in (False, True):
        layer = ds.QueryScale(None, (2, 3), 'cpu', cast_multiplier=cast)
        layer.delta.data.fill_(.001)
        actual = layer(q, 4)
        multiplier = 1 + layer.delta
        expected = q * (multiplier.to(q.dtype) if cast else multiplier)
        assert torch.equal(actual, expected)
        assert actual.dtype == (q.dtype if cast else torch.float32)
        if not cast:
            assert torch.all(actual > q.float())


def test_adapter_rejects_unsupported_duplicate_and_invalid_state():
    with pytest.raises(ValueError, match='No compatible'):
        ds.inject(nn.Linear(2,3), family='tabpfn')
    model = small_model('tabpfn')
    ds.inject(model, family='tabpfn')
    with pytest.raises(ValueError, match='already injected'):
        ds.inject(model, family='tabpfn')
    state = ds.adapter_state(model)
    key = next(iter(state))
    with pytest.raises(ValueError, match='names'):
        ds.restore(model, {})
    with pytest.raises(ValueError, match='shape'):
        ds.restore(model, {key: torch.zeros(9)})
    with pytest.raises(ValueError, match='values'):
        ds.restore(model, {key: torch.full_like(state[key], float('nan'))})


def test_bundle_pins_base_and_rejects_native_loader_schema(tmp_path, monkeypatch):
    from src.utils.experiment import file_digest
    base = tmp_path/'tabpfn-v3-classifier-v3_default.ckpt'
    base.write_bytes(b'original')
    model = small_model('tabpfn')
    ds.inject(model, family='tabpfn', base_path=str(base), base_sha256=file_digest(base))
    ds.adapter_parameters(model)[0].data.fill_(.05)
    path = tmp_path/'tiny_diagscale.ckpt'
    assert ds.save_if_adapter(model, path, {'example': True})
    payload, actual = ds.read_bundle(path, family='tabpfn')
    assert actual == base and 'state_dict' not in payload
    assert len(payload['adapter_state']) == 1
    clone = small_model('tabpfn')
    ds.inject(clone, family='tabpfn')
    ds.restore(clone, payload['adapter_state'])
    assert torch.equal(ds.adapter_parameters(model)[0], ds.adapter_parameters(clone)[0])
    base.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='hash differs'):
        ds.read_bundle(path, family='tabpfn')


def test_native_hook_restores_before_cache_and_keeps_explicit_v3(monkeypatch):
    from src.model import tabpfn_models
    model = small_model('tabpfn')
    ds.inject(model, family='tabpfn')
    ds.adapter_parameters(model)[0].data.fill_(.3)
    payload = {'adapter_state': ds.adapter_state(model), 'creditpfn_adapter': {'base_sha256': 'abc'}}
    monkeypatch.setattr(ds, 'read_bundle', lambda *a, **k: (payload, Path('tabpfn-v3-classifier-v3_default.ckpt')))
    called = []
    class Estimator:
        def _initialize_model_variables(self):
            self.models_ = [small_model('tabpfn')]
        def fit(self):
            self._initialize_model_variables()
            self.cache = ds.adapter_parameters(self.models_[0])[0].detach().clone()
    def construct(task, path, **kwargs):
        called.append(str(path))
        return Estimator()
    monkeypatch.setattr(tabpfn_models, '_make_tabpfn', construct)
    est = ds.make_estimator('classification', 'named_diagscale.ckpt', family='tabpfn')
    est.fit()
    assert torch.allclose(est.cache, torch.full((2,3),.3))
    assert called == ['tabpfn-v3-classifier-v3_default.ckpt']


def test_adapter_optimizer_and_rng_recovery(tmp_path):
    from src.train.optimization import step_mean_gradient
    from src.train.recovery import load_recovery, save_recovery
    model = small_model('tabpfn')
    ds.inject(model, family='tabpfn')
    opt = torch.optim.AdamW(ds.adapter_parameters(model), lr=.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1-n/10)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    def step():
        model.attention(torch.randn(2,4,2,3),4).square().mean().backward()
        step_mean_gradient(model, opt, scaler, microbatches=1, max_norm=1.)
        schedule.step()
    for _ in range(3): step()
    path = tmp_path/'recovery.pt'
    save_recovery(path, model=model, optimizer=opt, scheduler=schedule, scaler=scaler,
                  identity={'id':'test'}, progress={'updates':3})
    for _ in range(3): step()
    expected = ds.adapter_state(model)
    assert load_recovery(path, model=model, optimizer=opt, scheduler=schedule,
                         scaler=scaler, identity={'id':'test'}) == {'updates':3}
    for _ in range(3): step()
    assert all(torch.equal(expected[k], v) for k,v in ds.adapter_state(model).items())


@pytest.mark.parametrize('kind', ['classification', 'regression'])
def test_context_folds_keep_all_test_rows_nested_and_disjoint(kind):
    cfg = context_policy()
    y = np.arange(505)%2 if kind == 'classification' else np.random.default_rng(42).normal(size=505)
    observed = []
    for fold in range(5):
        order, val, test = cc.fold_indices(y, kind, fold, cfg)
        assert len(set(order)|set(val)|set(test)) == len(y)
        assert not (set(order)&set(test) or set(order)&set(val) or set(test)&set(val))
        assert len(np.unique(order)) == len(order)
        assert np.array_equal(order, cc.fold_indices(y,kind,fold,cfg)[0])
        for size in cc.budgets(len(order),cfg):
            assert set(order[:min(cfg['context_rows'])]).issubset(order[:size])
            if kind == 'classification': assert len(np.unique(y[order[:size]])) == 2
        observed.extend(test)
    assert sorted(observed) == list(range(len(y)))


def test_context_prefix_never_uses_test_labels():
    indices=np.arange(100)
    labels=np.arange(120)%2
    a=cc.context_order(indices,labels,'classification',42)
    labels[100:]=7
    assert np.array_equal(a, cc.context_order(indices,labels,'classification',42))


def test_predict_batches_do_not_drop_tail_rows():
    model=SimpleNamespace(predict=lambda x: x[:,0])
    x=np.arange(103).reshape(-1,1)
    assert np.array_equal(cc.predict_all(model,x,'regression',16),np.arange(103))


def test_model_cast_preserves_fp32_adapter_values():
    model=small_model('tabpfn')
    ds.inject(model,family='tabpfn')
    delta=ds.adapter_parameters(model)[0]
    delta.data.fill_(.000123456789)
    expected=delta.detach().clone()
    model.bfloat16()
    assert delta.dtype==torch.float32 and torch.equal(delta,expected)
    assert next(model.attention.base.parameters()).dtype==torch.bfloat16


def test_context_encoding_fits_only_the_context(monkeypatch):
    from src.eval.dataset_loader import ProcessedDataset
    from src.model import tabpfn_models
    calls=[]
    class Estimator:
        def fit(self,X,y):
            calls.append(('fit',X.copy(),y.copy()))
        def predict_proba(self,X):
            calls.append(('predict',X.copy()))
            return np.tile([.6,.4],(len(X),1))
    options = []
    def construct(*args, **kwargs):
        options.append(kwargs)
        return Estimator()
    monkeypatch.setattr(tabpfn_models,'_make_tabpfn',construct)
    data=ProcessedDataset(pd.DataFrame({'cat':['a','a','b','b','test-only','test-only']}),
                          np.array([0,1,0,1,0,1]),['cat'],'classification','synthetic','pd')
    metrics,blob=cc.score_cell(data,np.array([0,1]),np.array([2,3]),np.array([4,5]),'v3.ckpt','tabpfn',context_policy(),device='cpu')
    assert calls[0][0]=='fit' and calls[0][1].shape==(2,1)
    assert not options[0].get('ignore_pretraining_limits', False)
    assert all((record[1]==-1).all() for record in calls[1:])
    import io
    with np.load(io.BytesIO(blob)) as arrays:
        assert arrays['test'].tolist()==[4,5]
        assert arrays['context'].tolist()==[0,1]
    limited = dict(context_policy(), max_context_rows=1)
    with pytest.raises(ValueError, match='declared'):
        cc.score_cell(data,np.array([0,1]),np.array([2,3]),np.array([4,5]),'v3.ckpt','tabpfn',limited,device='cpu')


def test_large_filter_changes_training_only(monkeypatch,tmp_path):
    from src.train import corpus
    refs=[corpus.DatasetRef(f'table{i}','pd','classification','y',(),tmp_path/f'{i}.csv',
                           15000 if i%2 else 10000) for i in range(16)]
    monkeypatch.setattr(corpus,'build_dataset_pool',lambda track:refs)
    seen=[]
    for fold in range(4):
        all_tables=corpus.split_corpus(track='pd',n_folds=4,fold=fold,seed=1729)
        large=corpus.split_corpus(track='pd',n_folds=4,fold=fold,seed=1729,min_train_rows=10001)
        assert large.test==all_tables.test
        assert large.train==[r for r in all_tables.train if r.n_rows>10000]
        assert not ({r.dataset_id for r in large.train}&{r.dataset_id for r in large.test})
        seen.extend(r.dataset_id for r in large.test)
    assert len(seen)==len(set(seen))==16


def test_tabicl_precision_is_matched_between_base_and_adapter(monkeypatch):
    from src.model import tabicl_models as models
    calls=[]
    monkeypatch.setattr(models, 'import_tabicl_sklearn', lambda: (lambda **k:calls.append(k),)*2)
    models._make_tabicl('classification', 'tabicl.ckpt')
    assert 'use_amp' not in calls[-1] and 'use_fa3' not in calls[-1]
    models._make_tabicl('classification', 'tabicl.ckpt', use_amp=False, use_fa3=False)
    assert calls[-1]['use_amp'] is False and calls[-1]['use_fa3'] is False
    model=small_model('tabicl'); ds.inject(model, family='tabicl')
    payload={'adapter_state':ds.adapter_state(model),'creditpfn_adapter':{'base_sha256':'hash'}}
    monkeypatch.setattr(ds, 'read_bundle', lambda *a,**k:(payload,Path('tabicl.ckpt')))
    class Estimator:
        def __init__(self, **k): calls.append(k)
        def _load_model(self): self.model_=small_model('tabicl')
    from src.train import tabicl_compat
    monkeypatch.setattr(tabicl_compat, 'import_tabicl_sklearn', lambda:(Estimator,Estimator))
    est=ds.make_estimator('classification','test_diagscale.ckpt',family='tabicl')
    est._load_model()
    assert calls[-1]['use_amp'] is False and calls[-1]['use_fa3'] is False
    assert len(ds.adapter_parameters(est.model_)) == 1
