"""Publication invariants for context-budget and adapter studies; synthetic only."""
import gzip
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.eval.context_curve import policy
from src.utils.experiment import digest_json
from src.visualize import campaign as cp, context_curves as cc, adapter_study as ds, style
from src.visualize.eval_viz import _decode_method_dirname
from src.visualize.training_viz import parse_trial_name
from src.visualize.reporting import page_text


def context_fixture():
    cfg = policy()
    cfg['context_rows'] = [8, 16, 32, 64, 128, 2048, 50000]
    rows = []
    for track in ('pd', 'lgd'):
        for family in ('tabpfn', 'tabicl'):
            for d, maximum in enumerate((128, 2048, 50000)):
                for fold in range(5):
                    for n in cfg['context_rows']:
                        if n > maximum:
                            continue
                        for state in ('base', 'cpt'):
                            score = .65 + .03*np.log10(n) + d*.02 + fold*.001
                            shift = (.012 - .002*np.log2(n)) if state == 'cpt' else 0
                            rows.append(dict(track=track, family=family, partition=d, dataset=f'Table {d}',
                                fold=fold, context_rows=n, state=state, roc_auc=score+shift, r2=score+shift,
                                brier_score=.2-score*.1-shift, elapsed_seconds=n/100*(1.1 if state=='cpt' else 1)))
    return cc.ContextStudy(cfg, pd.DataFrame(rows), pd.DataFrame())


def test_context_curves_keep_a_fixed_cohort_and_require_five_pairs():
    study = context_fixture()
    pairs = cc.paired_tables(study, 'roc_auc')
    assert pairs.folds.eq(5).all()
    cohort = cc.fixed_cohort(study, pairs, 'pd', 2048)
    assert set(cohort.dataset) == {'Table 1', 'Table 2'}
    assert cohort.groupby('context_rows').dataset.nunique().eq(2).all()
    mask = (study.cells.dataset.eq('Table 1') & study.cells.track.eq('pd') & study.cells.family.eq('tabicl')
            & study.cells.fold.eq(4) & study.cells.context_rows.eq(16) & study.cells.state.eq('cpt'))
    study.cells = study.cells[~mask]
    cohort = cc.fixed_cohort(study, cc.paired_tables(study, 'roc_auc'), 'pd', 2048)
    assert set(cohort.dataset) == {'Table 2'}
    # A count of five with the wrong fold identity must not pass.
    study.cells.loc[study.cells.fold.eq(4), 'fold'] = 5
    assert cc.paired_tables(study, 'roc_auc').empty


def test_context_metric_signs_and_no_duplicate_weighting():
    study = context_fixture()
    pairs = cc.paired_tables(study, 'brier_score')
    assert np.allclose(pairs.gain, pairs.original - pairs.adapted)
    study.cells = pd.concat([study.cells, study.cells.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match='Duplicate'):
        cc.paired_tables(study, 'roc_auc')


def context_export(cfg):
    task = dict(track='pd', family='tabpfn', partition=0, fold=0, dataset_id='test_table')
    rows = [dict(**task, state=state, context_rows=n, available_context_rows=16,
        test_rows=5, validation_rows=4, test_indices_sha256='test', context_indices_sha256=f'context-{n}', roc_auc=.75)
        for n in (8, 12, 16) for state in ('base', 'cpt')]
    frame = pd.DataFrame(rows)
    return task, frame


@pytest.mark.parametrize('damage', ['test_rows', 'context_hash', 'missing_cell', 'wrong_task'])
def test_context_export_rejects_changed_pairs_or_incomplete_grids(damage):
    cfg = policy(); task, frame = context_export(cfg)
    receipt = {'cells': len(frame)}
    if damage == 'test_rows': frame.loc[0, 'test_rows'] = 4
    if damage == 'context_hash': frame.loc[0, 'context_indices_sha256'] = 'other'
    if damage == 'missing_cell': frame = frame.iloc[:-1]
    if damage == 'wrong_task': frame.loc[0, 'fold'] = 1
    with pytest.raises(ValueError): cc._check_cells(frame, task, receipt, cfg)


def test_context_zip_exports_require_matching_receipts_and_do_not_modify_inputs(tmp_path, monkeypatch):
    cfg = policy(); task, frame = context_export(cfg)
    plan = dict(config=cfg, tasks=[task], schema=1)
    plan['sha256'] = digest_json(plan)
    receipt = dict(plan_sha256=plan['sha256'], task=0, cells=len(frame), passed=True)
    data, project = tmp_path/'data.zip', tmp_path/'project.zip'
    with zipfile.ZipFile(data, 'w') as z:
        z.writestr('experiment4/manifests/plans/context_v1.json', json.dumps(plan))
        z.writestr('experiment4/manifests/completed/context_v1/task_0000.json', json.dumps(receipt))
    with zipfile.ZipFile(project, 'w') as z:
        z.writestr('output CreditPFN/experiment4/results/context_v1/task_0000.csv.gz', gzip.compress(frame.to_csv(index=False).encode()))
        z.writestr('output CreditPFN/experiment4/results/context_v1/task_0000.sqlite', b'not read by notebook')
    monkeypatch.setenv('CREDITPFN_ANALYSIS_ROOT', str(data))
    monkeypatch.setenv('CREDITPFN_ANALYSIS_PROJECT_ROOT', str(project))
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (data, project)]
    study = cc.load_context_study()
    assert study.inventory.status.tolist() == ['Verified export'] and len(study.cells) == 6
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (data, project)]
    monkeypatch.setenv('CREDITPFN_ANALYSIS_ROOT', str(project))
    study = cc.load_context_study()
    assert study.cells.empty and study.inventory.status.tolist() == ['Plan unavailable']


def test_adapter_filenames_and_planned_roster_are_recognized(tmp_path, monkeypatch):
    monkeypatch.setenv('CREDITPFN_ANALYSIS_ROOT', str(tmp_path))
    monkeypatch.delenv('CREDITPFN_ANALYSIS_PROJECT_ROOT', raising=False)
    for track in ('pd', 'lgd'):
        run = cp.load_campaign(5, track)
        assert len(run.trials) == 24 and run.trials.base.nunique() == 2
        for name in run.trials.trial_name:
            trial = parse_trial_name(name)
            assert trial and trial.adaptation_mode == 'diagscale' and not trial.lora
        assert run.trials.status.eq('PENDING').all()
        assert ds.plot_optimization(run) == [] and ds.plot_monitors(run) == [] and ds.plot_benchmark(run) == []
    meta = _decode_method_dirname('tabpfn-trained__v3-default__lr0.001__l2sp0__diagscale')
    assert meta['base_short'] == 'v3-default' and meta['adaptation_mode'] == 'diagscale'
    assert not meta['use_lora']


def adapter_fixture(track='pd'):
    from src.train.config import load_train_config
    cfg = load_train_config(config_path=f'config/experiment5/{track}.yaml')
    trials = cp.planned_trials(cfg).assign(status='OK', n_train_datasets=2, n_test_datasets=1)
    histories, trajectories, evaluation = [], [], []
    for t in trials.itertuples():
        for u in (0,250,1000,2500,5000,10000):
            drift = t.learning_rate * u / 1000
            trajectories.append(dict(trial_name=t.trial_name, successful_updates=u, weight_drift=drift,
                **{f'score__test__Table {t.partition}__{m}': .7 + t.partition*.01 + drift*.01 for m in ('roc_auc','r2')},
                **{f'score__ood__Panel__{m}': .7 - drift*.01 for m in ('roc_auc','r2')}))
            if u:
                histories.append(dict(trial_name=t.trial_name, successful_updates=u, train_loss=.5/(1+u/10000),
                    grad_norm_mean=drift+1, clipped_frac=.001*u/10000, elapsed_sec=u*.4,
                    observed_tables=2, full_observed_tables=2,
                    complete_pass=True, partial_epoch=False))
        short = 'v3-default' if 'tabpfn' in t.trial_name else 'tabicl-v2'
        family = 'tabpfn' if short == 'v3-default' else 'tabicl'
        for domain, dataset in (('credit',f'Table {t.partition}'),('ood','Panel')):
            for fold in range(5):
                evaluation.append(dict(method_dirname=f'{family}-trained__{short}__lr{t.learning_rate:g}__l2sp0__diagscale',
                    source=family+'-trained', base_short=short, test_dataset_id=dataset, split=t.partition, eval_run=f'test_s{t.partition:02d}',
                    fold_idx=fold, status='OK', domain=domain, roc_auc=.71-t.learning_rate, r2=.4-t.learning_rate,
                    rmse=.5+t.learning_rate, brier_score=.1+t.learning_rate))
                if t.learning_rate == 1e-4:
                    evaluation.append(dict(evaluation[-1], method_dirname=f'{family}-untuned__{short}', source=family+'-untuned',
                        roc_auc=.7, r2=.35, rmse=.6, brier_score=.13))
    run = cp.Campaign(cfg,trials,pd.DataFrame(trajectories),pd.DataFrame(histories),pd.DataFrame(evaluation))
    run._diagnostic_cache = {"resources": pd.DataFrame([
        dict(trial_name=t.trial_name, base=t.base, phase="training", successful_updates=u,
             gpu_utilization_percent=80+u/1000, device_memory_used_mib=20000+u)
        for t in trials.itertuples() for u in (250,1000,2500,5000,10000)])}
    return run


def test_adapter_benchmark_requires_matching_partitions_and_all_rates():
    run = adapter_fixture()
    assert ds.matched_rates(run).dataset.nunique() == 4
    data = ds.matched_rates(run,domain='ood')
    assert data.dataset.nunique() == 1 and data.partitions.eq(4).all()
    mask = (run.evaluation.domain.eq('ood') & run.evaluation.split.eq(0) &
        run.evaluation.method_dirname.str.contains('lr0.001') & run.evaluation.fold_idx.eq(4))
    run.evaluation = run.evaluation[~mask]
    assert ds.matched_rates(run,domain='ood').empty
    # Credit results remain unaffected by missing non-credit data.
    assert ds.matched_rates(run).dataset.nunique() == 4


def test_new_dense_figures_have_text_equivalents_and_publication_geometry(tmp_path):
    style.apply()
    study = context_fixture()
    pages = cc.plot_context_support(study) + cc.plot_context_curves(study)
    for track in ('pd','lgd'):
        run = adapter_fixture(track)
        pages += ds.plot_optimization(run) + ds.plot_monitors(run) + ds.plot_resources(run) + ds.plot_benchmark(run)
        if track == 'pd': pages += ds.plot_benchmark(run,metric='brier_score')
    assert len(pages) >= 14
    for index,page in enumerate(pages):
        assert page.tables and page.caption and page_text(page)
        assert page.figure.get_figwidth() == style.WIDTH_FULL
        assert page.figure.get_figheight() <= style.MAX_HEIGHT
        page.figure.savefig(tmp_path/f'{index:02d}_{page.name}.pdf')
        page.figure.canvas.draw()
        assert not any('Full /' in a.get_title() or 'Frozen /' in a.get_title() for a in page.figure.axes)


def test_adapter_resource_comparison_uses_honest_scales_and_preserves_values():
    run = adapter_fixture()
    page, = ds.plot_resources(run)
    utilization, memory = page.figure.axes
    assert utilization.get_ylim() == (0, 100)
    assert memory.get_ylim()[0] == 0 and 'GiB' in memory.get_ylabel()
    table = page.tables['Displayed counters and trial support'].dropna(subset=['median'])
    measured = table[table.measure.eq('Device memory (GiB)')]
    # The first 2,000-update window contains samples at 250 and 1,000:
    # its within-trial median is 20,625 MiB, not the value at the window edge.
    expected_mib = {2000: 20625, 4000: 22500, 6000: 25000, 10000: 30000}
    assert np.allclose(measured['median'], measured.window_end.map(expected_mib) / 1024)
    assert measured.base.nunique() == 2
