"""Synthetic design, leakage, shared-anchor and cluster-routing checks. No training."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.train import corpus, size_studies as design
from src.train.config import load_train_config, resolve_grid
from src.utils.experiment import apply_split_index, digest_json


def configs():
    return [load_train_config(config_path=p) for p in
            ['config/experiment6/pd.yaml', 'config/experiment7/pd_rows2048.yaml', 'config/experiment7/pd_rows26000.yaml']]


@pytest.fixture
def pool(monkeypatch, tmp_path):
    refs = [corpus.DatasetRef(f'table{i}', 'pd', 'classification', 'y', (), tmp_path/f'{i}.csv',
                             40000 if i % 3 else 999) for i in range(17)]
    monkeypatch.setattr(corpus, 'build_dataset_pool', lambda track: refs)
    from src.data import preprocessing
    monkeypatch.setattr(preprocessing, 'DATASET_METADATA', {r.dataset_id: {'track': 'pd'} for r in refs})
    return refs


def test_large_pool_and_heldout_partitions_are_identical_at_every_episode_size(pool):
    seen = []
    for fold in range(4):
        all_cfg, *episode_cfgs = [apply_split_index(c, fold) for c in configs()]
        complete = corpus.split_from_cfg(all_cfg, min_train_rows=0)
        large = corpus.split_from_cfg(all_cfg, min_train_rows=10001)
        assert large.train == [r for r in complete.train if r.n_rows > 10000]
        for cfg in episode_cfgs:
            cur = corpus.split_from_cfg(cfg)
            assert cur.train == large.train
            assert cur.test == large.test == complete.test
            assert set(r.dataset_id for r in cur.train).isdisjoint(r.dataset_id for r in cur.test)
        seen.extend(r.dataset_id for r in complete.test)
    assert sorted(seen) == sorted(r.dataset_id for r in pool)


def test_size_guard_rejects_short_table_without_refiltering_or_touching_test(pool):
    cfg = configs()[-1]
    split = corpus.split_from_cfg(cfg)
    from dataclasses import replace
    short = replace(split.train[0], n_rows=15000)
    with pytest.raises(ValueError, match='fixed large-table pool'):
        design.validate_split(cfg, corpus.CorpusSplit([short, *split.train[1:]], split.test), 10001)
    design.validate_split(cfg, corpus.CorpusSplit(split.train, [replace(r, n_rows=12) for r in split.test]), 10001)
    with pytest.raises(ValueError, match='three eligible'):
        design.validate_split(cfg, corpus.CorpusSplit(split.train[:2], split.test), 10001)


@pytest.mark.parametrize('field,value', [('corpus.split_seed', 42), ('corpus.train_dataset_ids',['table1']),
    ('tunable.epoch_pass_modes',['full_pass']), ('tunable.query_fractions',[.2]),
    ('train.context_sampling','balanced'), ('train.n_estimators_finetune_tabicl',4),
    ('train.max_rows_per_step',50000)])
def test_design_rejects_confounding_overrides(field, value):
    cfg = configs()[-1]
    OmegaConf.update(cfg, field, value)
    with pytest.raises(ValueError): design.validate_config(cfg)


def test_runtime_cap_cannot_silently_shorten_size_study():
    cfg = configs()[-1]
    design.validate_runtime_cap(cfg, 26000, None)
    for rows, cells in ((2048,None),(26000,400000)):
        with pytest.raises(ValueError, match='Runtime capacity'):
            design.validate_runtime_cap(cfg, rows, cells)
    old = load_train_config(config_path='config/experiment1/pd.yaml')
    design.validate_runtime_cap(old, 10000, None)


def payload(cfg, pool):
    from src.utils.experiment import scientific_config
    result = {'config':OmegaConf.to_container(cfg), 'trials':{}, 'identities':{},
              'evaluation_protocol': {'config': {'folds': 5}, 'code_sha256': 'same-benchmark'}}
    for fold in range(4):
        cur = apply_split_index(OmegaConf.create(OmegaConf.to_container(cfg)), fold)
        for i, trial in enumerate(resolve_grid(cur,single=False)):
            split = corpus.split_from_cfg(cur, min_train_rows=trial[6])
            spec = {'config':scientific_config(cur), 'trial':list(trial), 'code_sha256':'same-code',
                'versions':{'torch':'same'}, 'base_sha256':trial[0], 'data_config':{},
                'datasets':{kind:[{'id':r.dataset_id,'sha256':r.dataset_id+'-bytes'}
                                  for r in getattr(split,kind)] for kind in ('train','test')}}
            key = digest_json(spec)
            result['trials'][f'{cur.run_name}/{i}'] = key
            result['identities'][key] = spec
    return result


def test_shared_anchor_reuses_eight_trials_once_and_never_all_table_controls(pool):
    ref = payload(configs()[0],pool)
    for cfg in configs()[1:]:
        mapping = design.match_reference(payload(cfg,pool),ref)
        assert len(mapping) == len(set(mapping.values())) == 8
        assert all(ref['identities'][ref['trials'][name]]['trial'][6] == 10001 for name in mapping.values())


def test_real_preparation_binds_and_preserves_shared_reference(pool, tmp_path, monkeypatch):
    import json
    from src.data import retention
    from src.utils import experiment, prepare_experiment as prep
    monkeypatch.setenv('CREDITPFN_EXPERIMENT', 'general')
    monkeypatch.setattr(prep, 'manifests_dir', lambda group: tmp_path / group / 'manifests')
    monkeypatch.setattr(experiment, 'file_digest', lambda path: Path(path).name + '-bytes')
    monkeypatch.setattr(experiment, 'code_identity', lambda **kwargs: 'same-source')
    monkeypatch.setattr(experiment, 'environment_versions', lambda **kwargs: {'test': 'synthetic'})
    monkeypatch.setattr(retention, 'load_refs', lambda *args: [])
    with pytest.raises(FileNotFoundError):
        prep.prepare(Path('config/experiment7/pd_rows2048.yaml'), write=True)
    source = prep.prepare(Path('config/experiment6/pd.yaml'), write=True)
    child = prep.prepare(Path('config/experiment7/pd_rows2048.yaml'), write=True)
    reference, bound = prep.read_plan(Path(source['plan'])), prep.read_plan(Path(child['plan']))
    assert bound['shared_reference']['sha256'] == reference['sha256']
    assert len(bound['shared_reference']['trials']) == 8
    prep.check_shared_reference(bound)
    # A checksum-valid replacement is still the wrong control.
    reference['config']['seed'] = 43
    reference['sha256'] = digest_json({k: v for k, v in reference.items() if k != 'sha256'})
    Path(source['plan']).write_text(json.dumps(reference), encoding='utf-8')
    with pytest.raises(RuntimeError, match='reference changed'):
        prep.check_shared_reference(bound)


@pytest.mark.parametrize('mutation', ['rows','weights','version','seed','budget','schedule','eval','benchmark','missing','duplicate'])
def test_anchor_rejects_noncomparable_or_ambiguous_evidence(pool, mutation):
    ref, cur = payload(configs()[0],pool), payload(configs()[1],pool)
    spec = next(iter(cur['identities'].values()))
    if mutation == 'rows': spec['datasets']['train'][0]['sha256']='changed-rows'
    if mutation == 'weights': spec['base_sha256']='different-base'
    if mutation == 'version': spec['versions']['torch']='different'
    if mutation == 'seed': spec['config']['seed']=43
    if mutation == 'budget': spec['config']['train']['target_total_steps']=5000
    if mutation == 'schedule': spec['config']['scheduler']['warmup_fraction']=.2
    if mutation == 'eval': cur['config']['evaluation']['cache_controls']=False
    if mutation == 'benchmark': cur['evaluation_protocol']['config']['folds']=3
    if mutation == 'missing': cur['trials'].pop(next(iter(cur['trials'])))
    if mutation == 'duplicate':
        key = next(k for k in ref['trials'].values() if ref['identities'][k]['trial'][6]==10001)
        ref['trials']['extra/0']=key
    with pytest.raises(ValueError): design.match_reference(cur,ref)


def test_planned_benchmark_rejects_cli_settings_and_environment_drift(monkeypatch):
    from src.utils import experiment, prepare_experiment as prep
    cfg = configs()[0]
    eval_cfg = OmegaConf.merge(OmegaConf.load('config/eval.yaml'), cfg.evaluation)
    monkeypatch.setattr(experiment, 'code_identity', lambda **kwargs: 'eval-source')
    monkeypatch.setattr(experiment, 'environment_versions', lambda **kwargs: {'torch': 'tested'})
    planned = {'evaluation_protocol': design.evaluation_protocol(cfg)}
    prep.check_size_evaluation(cfg, planned, eval_cfg)
    changed = OmegaConf.merge(eval_cfg, {'results': {'base_dir': 'another-storage-tier'}})
    prep.check_size_evaluation(cfg, planned, changed)
    for change in ({'tabpfn_n_estimators': 8}, {'cv': {'n_folds': 3}}, {'max_rows_per_model': {'v3': 2000}}):
        with pytest.raises(RuntimeError, match='benchmark protocol'):
            prep.check_size_evaluation(cfg, planned, OmegaConf.merge(eval_cfg, change))
    monkeypatch.setattr(experiment, 'environment_versions', lambda **kwargs: {'torch': 'different'})
    with pytest.raises(RuntimeError, match='benchmark protocol'):
        prep.check_size_evaluation(cfg, planned, eval_cfg)


def test_actual_sampler_preserves_requested_rows_and_has_disjoint_context_query(tmp_path):
    from src.train.dataloader import ProcessedDatasetLoader
    from src.train.sampling import EpochSampler
    path=tmp_path/'synthetic.csv'
    pd.DataFrame({'x':np.arange(30000),'y':np.arange(30000)%2}).to_csv(path,index=False)
    ref=corpus.DatasetRef('synthetic','pd','classification','y',(),path,30000)
    for cap in (2048,8192,26000):
        loader=ProcessedDatasetLoader([ref],max_rows_per_epoch=cap,query_fraction=.4,
                                      seed=42,context_sampling='stratified')
        sampler=EpochSampler(loader,seed=42,accumulate=False)
        batch=loader[next(iter(sampler))]
        context=batch.X_context[:,0,0].numpy()
        query=batch.X_query[:,0,0].numpy()
        assert len(context)+len(query)==cap and len(query)==round(.4*cap)
        assert not set(context)&set(query)
        repeated=loader[next(iter(sampler))]
        assert np.array_equal(batch.y_query,repeated.y_query)
        sampler.set_epoch(1)
        assert not np.array_equal(loader[next(iter(sampler))].X_context,batch.X_context)


@pytest.mark.parametrize('experiment,names', [(6,['pd.yaml']), (7,['pd_rows2048.yaml','pd_rows26000.yaml'])])
def test_size_benchmark_array_routes_every_cell_once(experiment,names):
    from tests.test_benchmark_submission import bash_path
    code='''set -euo pipefail
for ((i=0;i<COUNT;i++)); do
 export SLURM_ARRAY_TASK_ID=$i
 source scripts/slurm/_benchmark_dispatch.sh
 printf '%s|%s|%s|%s\n' "$TRACK" "$CREDITPFN_SPLIT_INDEX" "$CREDITPFN_EVAL_TASK_INDEX" "$CREDITPFN_CONFIG"
done
'''
    env=dict(os.environ,CREDITPFN_EXPERIMENT=f'experiment{experiment}',CREDITPFN_BENCHMARK_SPLITS='4',
             EVAL_TASKS='16',VSC_DATA='/data/test',COUNT=str(len(names)*64))
    run=subprocess.run([bash_path(),'--noprofile','--norc','-c',code],env=env,text=True,capture_output=True,timeout=15)
    assert run.returncode==0,run.stderr
    rows=[line.split('|') for line in run.stdout.splitlines()]
    assert rows==[['pd',str(f),str(s),f'/data/test/CreditPFN/config/experiment{experiment}/{name}']
                  for name in names for f in range(4) for s in range(16)]


def test_cleanup_preserves_a_shared_control_even_without_live_jobs(tmp_path, monkeypatch):
    from src.utils import clean_run
    monkeypatch.setattr(clean_run,'outputs_dir',lambda:tmp_path)
    path=tmp_path/'experiment7/manifests/plans/pd.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"shared_reference":{"run":"cpt_corpus_v1"}}')
    clean_run.check_saved_consumers('experiment7','all')
    clean_run.check_saved_consumers('experiment6','figures')
    for scope in ('all','training','benchmark'):
        with pytest.raises(RuntimeError,match='shared control'):
            clean_run.check_saved_consumers('experiment6',scope)


def size_report_fixture(experiment=7, configurations=None):
    from src.visualize import campaign as cp, size_studies as ss
    arms={}
    for cfg in (configurations if configurations is not None else ([configs()[0]] if experiment==6 else configs())):
        if experiment==7 and cfg.experiment.size_study=='corpus_policy':
            minimums=[10001]
        else:
            minimums=sorted({t[6] for t in resolve_grid(cfg,single=False)})
        for minimum in minimums:
            trials=cp.planned_trials(cfg)
            trials=trials[trials.min_train_rows.eq(minimum)].assign(status='OK',n_train_datasets=2)
            label=('All tables' if minimum==0 else 'Large tables') if experiment==6 else f'{cfg.train.max_rows_per_step:,} rows'
            trajectories,histories,evaluation=[],[],[]
            for t in trials.itertuples():
                delta=np.log2(cfg.train.max_rows_per_step)/1000+minimum/1000000
                for u in cfg.train.trajectory_steps:
                    trajectories.append(dict(trial_name=t.trial_name,successful_updates=u,processed_rows=u*cfg.train.max_rows_per_step,
                        weight_drift=u/1e7,**{f'metric__test__Table {t.partition}':.7+delta*u/10000},
                        metric__ood__Panel=.7-delta*u/10000))
                    if u:
                        histories.append(dict(trial_name=t.trial_name,successful_updates=u,train_loss=.4-u/1e5,
                            grad_norm_mean=1+u/1e5,l2sp_penalty=u/1e8,observed_tables=2,full_observed_tables=2,
                            complete_pass=True,partial_epoch=False))
                short=f'{t.base}-default' if 'tabpfn' in t.trial_name else 'tabicl-v2'
                family='tabpfn' if 'tabpfn' in t.trial_name else 'tabicl'
                method=f'{family}-trained__{short}__lr3e-07__l2sp0.003'+(f'__min{minimum}' if minimum else '')
                for domain,dataset in (('credit',f'Table {t.partition}'),('ood','Panel')):
                    for fold in range(5):
                        row=dict(method_dirname=method,source=family+'-trained',base_short=short,test_dataset_id=dataset,
                                 split=t.partition,eval_run=f'test_s{t.partition}',fold_idx=fold,status='OK',domain=domain,
                                 roc_auc=.7+delta,brier_score=.2-delta)
                        evaluation += [row,dict(row,method_dirname=f'{family}-untuned__{short}',source=family+'-untuned',
                                               roc_auc=.7,brier_score=.2)]
            arms[label]=cp.Campaign(cfg,trials,pd.DataFrame(trajectories),pd.DataFrame(histories),pd.DataFrame(evaluation))
    return arms


def test_size_reports_use_shared_anchor_once_and_require_all_five_folds():
    from src.visualize import size_studies as ss
    arms=size_report_fixture()
    data=ss.matched_observations(arms,benchmark=True)
    assert set(data.arm)==set(arms)
    assert len(data)==30  # two bases x (four credit tables + one OOD) x three arms
    pairs=ss.contrasts(data,'8,192 rows')
    assert len(pairs)==20 and set(pairs.arm)=={'2,048 rows','26,000 rows'}
    run=arms['26,000 rows']
    run.evaluation=run.evaluation[~(run.evaluation.domain.eq('ood')&run.evaluation.split.eq(0)&run.evaluation.fold_idx.eq(4))]
    assert set(ss.matched_observations(arms,benchmark=True).domain)=={'credit'}


def test_new_size_figures_have_complete_numerical_equivalents(tmp_path):
    from src.visualize import size_studies as ss, style
    from src.visualize.reporting import page_text
    style.apply()
    for experiment in (6,7):
        arms=size_report_fixture(experiment)
        pages=ss.plot_optimization(arms)+ss.plot_monitors(arms)+ss.plot_benchmark(arms)+ss.plot_benchmark(arms,'brier_score')
        assert len(pages)==5
        for page in pages:
            assert page.tables and page.caption and page_text(page)
            assert page.figure.get_figwidth()==style.WIDTH_FULL
            page.figure.canvas.draw()
            page.figure.savefig(tmp_path/f'exp{experiment}_{page.name}.pdf')


def test_new_size_notebooks_show_pending_without_inventing_evidence(tmp_path, monkeypatch):
    from src.visualize import size_studies as ss
    monkeypatch.setenv('CREDITPFN_ANALYSIS_ROOT',str(tmp_path))
    monkeypatch.delenv('CREDITPFN_ANALYSIS_PROJECT_ROOT',raising=False)
    for exp,planned in ((6,16),(7,24)):
        arms=ss.load_arms(exp)
        assert ss.coverage(arms).planned.sum()==planned
        assert ss.coverage(arms).completed.sum()==0
        assert ss.plot_monitors(arms)==ss.plot_benchmark(arms)==ss.plot_optimization(arms)==[]


def test_partial_size_monitors_do_not_reserve_empty_noncredit_panels():
    from src.visualize import size_studies as ss, style
    style.apply()
    arms = size_report_fixture(6)
    for run in arms.values():
        run.trajectories = run.trajectories.drop(columns='metric__ood__Panel')
    page, = ss.plot_monitors(arms)
    assert len(page.figure.axes) == 2
    assert 'non-credit' not in page.figure._suptitle.get_text()
    for ax in page.figure.axes:
        assert ax.get_subplotspec().get_gridspec().nrows == 1
    assert set(page.tables['Matched table effects'].domain) == {'credit'}
