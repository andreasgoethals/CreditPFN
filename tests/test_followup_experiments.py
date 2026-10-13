"""Cheap invariants for isolated follow-ups. No released weights or GPU required."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from src.model import diagscale as ds
from src.eval import context_curve as cc


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
    from src.train.optimization import step_mean_gradient
    from src.train.loop import _make_optimizer_and_scheduler
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


def test_cache_probe_detects_adapter_omission_on_cached_path():
    from src.utils.followup_checks import attention_cache_probe
    class Attention(MultiheadAttention):
        omit_cached_adapter = False

        def forward(self, query, key=None, value=None, *, need_kv=False, cached_kv=None):
            q = query.reshape(1, 6, 2, 3).transpose(1, 2)
            out = self.ssmax_layer(q, 10)
            if cached_kv is not None and self.omit_cached_adapter:
                out = q
            return (out, key, value) if need_kv else out

    attention = Attention()
    attention.ssmax_layer = ds.QueryScale(None, (2, 1, 3), 'cpu')
    with pytest.raises(AssertionError, match='nonzero'):
        attention_cache_probe(attention, cache_factory=SimpleNamespace)
    attention.ssmax_layer.delta.data.fill_(.2)
    assert attention_cache_probe(attention, cache_factory=SimpleNamespace) == 0
    attention.omit_cached_adapter = True
    with pytest.raises(AssertionError, match='disagree'):
        attention_cache_probe(attention, cache_factory=SimpleNamespace)


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
    from src.utils import paths
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
    from src.train.recovery import save_recovery, load_recovery
    from src.train.optimization import step_mean_gradient
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
    cfg = cc.policy()
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


def test_budget_deduplication_and_capacity_guard():
    cfg=cc.policy()
    assert cc.budgets(32,cfg)==[8,12,16,24,32]
    assert cc.budgets(34,cfg)==[8,12,16,24,32,34]
    assert max(cc.budgets(1000000,cfg))==cfg['max_context_rows']
    assert cfg['max_context_rows'] == 50000 and len(cfg['context_rows']) == 29


def test_new_grid_sizes_and_existing_design_unchanged():
    from src.train.config import load_train_config, resolve_grid
    for exp, total in [(1,512),(2,32),(3,96),(5,48)]:
        assert sum(len(resolve_grid(c:=load_train_config(config_path=f'config/experiment{exp}/{t}.yaml'),single=False))*int(c.corpus.n_splits)
                   for t in ('pd','lgd'))==total
    configs=[*Path('config/experiment6').glob('*.yaml'), *Path('config/experiment7').glob('*.yaml')]
    assert len(configs)==3
    assert sum(len(resolve_grid(c:=load_train_config(config_path=str(p)),single=False))*int(c.corpus.n_splits) for p in configs)==32
    for path in configs:
        cfg=load_train_config(config_path=str(path))
        assert cfg.train.max_rows_per_step<=26000
        assert {t[6] for t in resolve_grid(cfg,single=False)} == ({0,10001} if 'experiment6' in str(path) else {10001})
    for track in ('pd','lgd'):
        cfg=load_train_config(config_path=f'config/experiment5/{track}.yaml')
        grid=resolve_grid(cfg,single=False)
        assert all(('tabpfn-v3-' in t[0] or 'tabicl-' in t[0]) and not t[2] and t[7]==0 for t in grid)
        assert list(cfg.tunable.learning_rates)==[.0001,.001,.03]
        assert cfg.train.amp is False and cfg.train.max_rows_per_step == 8192
        assert dict(cfg.evaluation.tabicl_inference_options) == {'use_amp': False, 'use_fa3': False}


def test_followup_gate_never_accepts_missing_receipts(monkeypatch,tmp_path):
    from src.utils import followup_checks as checks
    monkeypatch.delenv('CREDITPFN_FOLLOWUP_GATE',raising=False)
    with pytest.raises(RuntimeError,match='require'):
        checks.require_gate()
    monkeypatch.setattr(checks,'folder',lambda id:tmp_path)
    with pytest.raises(FileNotFoundError): checks.require_gate('newworkflow')


def test_scoped_cleanup_never_crosses_experiment_boundaries(tmp_path,monkeypatch):
    from src.utils import clean_run as clean
    data,project=tmp_path/'DATA',tmp_path/'PROJECT'
    monkeypatch.setattr(clean,'outputs_dir',lambda:data/'output CreditPFN')
    monkeypatch.setattr(clean,'resolve_staging_path',lambda p:project/p)
    monkeypatch.setattr(clean,'resolve_output_path',lambda p:data/p)
    monkeypatch.setattr(clean,'checkpoints_dir',lambda *p:project/'checkpoints'/Path(*p))
    roots=clean.experiment_roots('experiment5')
    assert len(roots)==6
    for tier in (data,project):
        for exp in (1,5):
            p=tier/'output CreditPFN'/f'experiment{exp}'/'results'/'value.csv'
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text('preserve' if exp==1 else 'remove')
    for root in roots: clean.wipe(root)
    assert all((tier/'output CreditPFN/experiment1/results/value.csv').read_text()=='preserve' for tier in (data,project))
    assert all(not (tier/'output CreditPFN/experiment5/results/value.csv').exists() for tier in (data,project))
    with pytest.raises(ValueError):clean.experiment_roots('../experiment1')
    with pytest.raises(ValueError):clean.experiment_roots('general')


def test_global_delete_requires_explicit_scope():
    from src.utils.clean_run import main
    with pytest.raises(SystemExit):main(['--clean'])


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
    from src.model import tabpfn_models
    from src.eval.dataset_loader import ProcessedDataset
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
    metrics,blob=cc.score_cell(data,np.array([0,1]),np.array([2,3]),np.array([4,5]),'v3.ckpt','tabpfn',cc.policy(),device='cpu')
    assert calls[0][0]=='fit' and calls[0][1].shape==(2,1)
    assert not options[0].get('ignore_pretraining_limits', False)
    assert all((record[1]==-1).all() for record in calls[1:])
    import io
    with np.load(io.BytesIO(blob)) as arrays:
        assert arrays['test'].tolist()==[4,5]
        assert arrays['context'].tolist()==[0,1]
    limited = dict(cc.policy(), max_context_rows=1)
    with pytest.raises(ValueError, match='declared'):
        cc.score_cell(data,np.array([0,1]),np.array([2,3]),np.array([4,5]),'v3.ckpt','tabpfn',limited,device='cpu')


def test_context_resume_does_not_repeat_completed_cells(tmp_path,monkeypatch):
    from src.utils import followup_checks
    from src.eval import dataset_loader
    from src.train import corpus
    base,cpt,datafile=[tmp_path/name for name in ('v3.ckpt','cpt.ckpt','data.csv')]
    for p in (base,cpt,datafile,Path(str(cpt)+'.provenance.json')):p.write_text('fixture')
    cfg=cc.policy();cfg['context_rows']=[16,32];cfg['max_context_rows']=32
    task=dict(track='pd',family='tabpfn',partition=0,dataset_id='synthetic',fold=0,
              base=str(base),cpt=str(cpt),base_sha256=cc.file_digest(base),cpt_sha256=cc.file_digest(cpt),
              data_sha256=cc.file_digest(datafile),provenance_sha256=cc.file_digest(Path(str(cpt)+'.provenance.json')))
    monkeypatch.setattr(followup_checks,'require_gate',lambda:None)
    monkeypatch.setattr(cc,'read_plan',lambda p:dict(tasks=[task],config=cfg,sha256='prepared'))
    monkeypatch.setattr(cc,'results_dir',lambda **k:tmp_path/'project')
    monkeypatch.setattr(cc,'manifests_dir',lambda *a:tmp_path/'data')
    monkeypatch.setattr(corpus,'build_dataset_pool',lambda t:[SimpleNamespace(dataset_id='synthetic',processed_csv=datafile)])
    data=dataset_loader.ProcessedDataset(pd.DataFrame({'x':np.arange(100)}),np.arange(100)%2,[],
                                       'classification','synthetic','pd')
    monkeypatch.setattr(dataset_loader,'load_processed_dataset',lambda *a:data)
    calls=[]
    def fail_second(data,context,val,test,path,*args):
        calls.append((len(context),str(path)))
        if len(calls)==2:raise RuntimeError('interrupted')
        return {'roc_auc':.5},b'predictions'
    monkeypatch.setattr(cc,'score_cell',fail_second)
    with pytest.raises(RuntimeError,match='interrupted'):cc.run_task('plan',0)
    cc.run_task('plan',0)
    assert calls.count((16,str(base)))==1
    assert len(calls)==5
    cc.run_task('plan',0)
    assert len(calls)==5
    metrics=pd.read_csv(tmp_path/'project/context_v1/task_0000.csv.gz')
    assert len(metrics)==4 and metrics.test_indices_sha256.nunique()==1
    assert metrics.test_rows.tolist()==[20]*4


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


def test_gate_rejects_failed_stale_and_changed_evidence(tmp_path, monkeypatch):
    from src.utils import followup_checks as checks
    monkeypatch.setattr(checks, 'folder', lambda key: tmp_path)
    monkeypatch.setattr(checks, 'fingerprint', lambda: 'code')
    monkeypatch.setattr(checks, 'environment_versions', lambda **k: {'torch': 'version'})
    detail = tmp_path/'evidence.json'
    detail.write_text('{}')
    for stage in ('smoke', 'capacity'):
        for task in range(4):
            record = dict(passed=True, fingerprint='code', versions={'torch':'version'},
                          evidence=str(detail), evidence_sha256=checks.file_digest(detail), stage=stage, task=task)
            (tmp_path/f'{stage}_{task}.json').write_text(json.dumps(record))
    assert checks.require_gate('workflow') == 'workflow'
    receipt = tmp_path/'capacity_3.json'
    record=json.loads(receipt.read_text()); record['passed']=False
    receipt.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='failed'): checks.require_gate('workflow')
    record['passed']=True; record['fingerprint']='old-code'; receipt.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='stale'): checks.require_gate('workflow')
    record['fingerprint']='code'; receipt.write_text(json.dumps(record))
    detail.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match='evidence changed'): checks.require_gate('workflow')


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


def test_cleanup_blocks_unknown_or_same_experiment_writers(monkeypatch):
    from src.utils.clean_run import check_cluster_writers
    import subprocess
    monkeypatch.setenv('VSC_DATA','/data/test');monkeypatch.setenv('USER','test')
    def query(command, **kwargs):
        if command[0]=='squeue':
            return SimpleNamespace(stdout='123_[0-2]|creditpfn-pd-train\n')
        return SimpleNamespace(stdout='SubmitLine=sbatch CREDITPFN_CONFIG=/data/CreditPFN/config/experiment1/pd.yaml')
    monkeypatch.setattr(subprocess,'run',query)
    check_cluster_writers('experiment5')
    with pytest.raises(RuntimeError,match='still queued'):check_cluster_writers('experiment1')
    with pytest.raises(RuntimeError,match='still queued'):check_cluster_writers()
    monkeypatch.setattr(subprocess,'run',lambda command,**k:SimpleNamespace(stdout='123|creditpfn-unknown' if command[0]=='squeue' else ''))
    with pytest.raises(RuntimeError,match='still queued'):check_cluster_writers('experiment5')


def test_followup_workflow_cleanup_matches_whole_id_only(tmp_path, monkeypatch):
    from src.utils import clean_run as clean
    root=tmp_path/'experiment0';root.mkdir()
    wanted=root/'cpt_diagscale_debug_abc123_pd.log';wanted.write_text('old')
    other=root/'cpt_diagscale_debug_abc1234_pd.log';other.write_text('keep')
    monkeypatch.setattr(clean,'experiment_roots',lambda *args:[root])
    monkeypatch.setattr(clean,'check_cluster_writers',lambda *args:None)
    monkeypatch.delenv('CREDITPFN_ACTIVE_LOG',raising=False)
    clean.main(['--experiment','experiment0','--workflow','abc123'])
    assert wanted.exists()
    clean.main(['--experiment','experiment0','--workflow','abc123','--clean'])
    assert not wanted.exists() and other.read_text()=='keep'


def test_cleanup_protects_cross_experiment_checkpoint_and_gate_consumers(monkeypatch):
    from src.utils.clean_run import check_cluster_writers
    import subprocess
    monkeypatch.setenv('VSC_DATA', '/data/test')
    monkeypatch.setenv('USER', 'test')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(stdout='123|creditpfn-context-eval\n'))
    for experiment in ('experiment0', 'experiment1', 'experiment4'):
        with pytest.raises(RuntimeError, match='input consumers'):
            check_cluster_writers(experiment)
    check_cluster_writers('experiment5')
    def adapter_job(command, **kwargs):
        return SimpleNamespace(stdout='124|creditpfn-pd-train\n' if command[0]=='squeue' else
                               'SubmitLine=sbatch CREDITPFN_CONFIG=/repo/config/experiment5/pd.yaml')
    monkeypatch.setattr(subprocess, 'run', adapter_job)
    with pytest.raises(RuntimeError, match='input consumers'):
        check_cluster_writers('experiment0')


def test_new_slurm_scripts_parse_and_never_submit_locally():
    import subprocess
    from tests.test_benchmark_submission import bash_path
    names=['run_experiment0_followups.sh','followup_prepare.slurm','followup_gpu.slurm',
           'run_experiment4.sh','context_benchmark.slurm','run_experiment5.sh','run_experiment6.sh','run_experiment7.sh']
    for name in names:
        result=subprocess.run([bash_path(),'-n',str(Path('scripts/slurm')/name)],capture_output=True,text=True)
        assert result.returncode==0, result.stderr
