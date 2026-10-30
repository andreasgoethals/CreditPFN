"""A failed extension retries without rebinding or rerunning completed refresh work."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils import study_workflow as workflow


def test_existing_refresh_binds_clean_original_checkout_and_preparation(tmp_path, monkeypatch):
    from src.utils import evaluation_refresh
    checkout = tmp_path / 'source'
    checkout.mkdir()
    record = tmp_path / 'workflow.json'
    record.write_text(json.dumps(dict(id='old', prepared=True, runtime_version='image1')))
    monkeypatch.setattr(evaluation_refresh, 'root', lambda _: tmp_path)
    monkeypatch.setenv('VSC_DATA', str(tmp_path))
    dirty = [False]
    monkeypatch.setattr(workflow.subprocess, 'run', lambda args, **kw: SimpleNamespace(
        stdout=(' M source.py' if dirty[0] else '') if 'status' in args else 'a' * 40))
    binding = workflow.existing_refresh('old', checkout, 'image1')
    source = dict(runtime_version='image1', existing_refresh=binding)
    workflow.verify_refresh_source(source)
    record.write_text(json.dumps(dict(id='old', prepared=True, runtime_version='image1', changed=True)))
    with pytest.raises(RuntimeError, match='preparation changed'):
        workflow.verify_refresh_source(source)
    dirty[0] = True
    with pytest.raises(RuntimeError, match='clean'):
        workflow.existing_refresh('old', checkout, 'image1')


def test_existing_refresh_refuses_non_data_checkout_and_other_runtime(tmp_path, monkeypatch):
    from src.utils import evaluation_refresh
    checkout = tmp_path / 'source'; checkout.mkdir()
    monkeypatch.setenv('VSC_DATA', str(tmp_path / 'different'))
    with pytest.raises(ValueError, match='DATA'):
        workflow.existing_refresh('old', checkout, 'image1')
    monkeypatch.setenv('VSC_DATA', str(tmp_path))
    monkeypatch.setattr(workflow.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=''))
    monkeypatch.setattr(evaluation_refresh, 'root', lambda _: tmp_path)
    (tmp_path / 'workflow.json').write_text(json.dumps(dict(id='old', prepared=True, runtime_version='other')))
    with pytest.raises(RuntimeError, match='runtime/workflow'):
        workflow.existing_refresh('old', checkout, 'image1')


def test_recovery_runs_original_evaluator_and_propagates_failure(tmp_path, monkeypatch):
    import subprocess
    calls = []
    source = dict(runtime_version='image1', existing_refresh=dict(id='old', checkout=str(tmp_path)))
    monkeypatch.setattr(workflow, 'verify_refresh_source', lambda _: calls.append('verified'))
    monkeypatch.setenv('CREDITPFN_LEGACY_ID', 'new')
    monkeypatch.setenv('CREDITPFN_RUNTIME_VERSION', 'image1')
    monkeypatch.setenv('CREDITPFN_RUNTIME_ACCEPTANCE', '1')
    def run(args, **kwargs):
        calls.append((args, kwargs))
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(workflow.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        workflow.run_refresh(source, 'new', 'inspect')
    assert calls[0] == 'verified'
    args, options = calls[1]
    assert args[1:] == ['-m', 'src.utils.evaluation_refresh', 'inspect', 'old']
    assert options['cwd'] == str(tmp_path) and options['check']
    assert 'CREDITPFN_LEGACY_ID' not in options['env']
    assert 'CREDITPFN_RUNTIME_ACCEPTANCE' not in options['env']
    assert options['env']['CREDITPFN_RUNTIME_VERSION'] == 'image1'


def test_recovery_launch_does_not_migrate_prepare_or_submit_another_refresh(tmp_path, monkeypatch):
    from src.utils import evaluation_refresh, study_layout, legacy_submit, clean_run
    monkeypatch.setattr(workflow, 'root', lambda _: tmp_path)
    monkeypatch.setattr(workflow, 'settings', lambda _: dict(existing_refresh={'id':'old'}))
    monkeypatch.setattr(legacy_submit, 'root', lambda _: tmp_path)
    def forbidden(*a, **k):
        raise AssertionError('Recovery must not change the existing refresh')
    monkeypatch.setattr(study_layout, 'migrate', forbidden)
    monkeypatch.setattr(evaluation_refresh, 'prepare', forbidden)
    monkeypatch.setattr(clean_run, 'check_cluster_writers', forbidden)
    calls = []
    def submit(value, phase):
        calls.append((value, phase))
        (tmp_path / 'tests.submission.json').write_text(json.dumps(dict(accepted=[{'key':'inspect','id':'123'}])))
    monkeypatch.setattr(legacy_submit, 'submit', submit)
    monkeypatch.setattr(legacy_submit, 'submit_plan', lambda plan, path: calls.append(plan))
    workflow.launch('new')
    assert calls[0] == ('new','tests')
    assert len(calls[1]) == 1 and calls[1][0]['key'] == 'research'
    assert '--dependency=afterok:123' in calls[1][0]['args']
    assert '--kill-on-invalid-dep=yes' in calls[1][0]['args']


def test_research_omits_purged_refresh_jobs_but_waits_for_live_work(tmp_path, monkeypatch):
    from src.utils import legacy_submit
    def folder(value):
        p = tmp_path / value; p.mkdir(exist_ok=True); return p
    monkeypatch.setattr(workflow, 'root', folder)
    monkeypatch.setattr(legacy_submit, 'root', folder)
    monkeypatch.setattr(workflow, 'settings', lambda _: dict(existing_refresh={'id':'old'}))
    monkeypatch.setattr(legacy_submit, 'submit', lambda *a: None)
    (folder('old') / 'refresh.submission.json').write_text(json.dumps(dict(accepted=[dict(id='101')])))
    (folder('new') / 'research.submission.json').write_text(json.dumps(dict(accepted=[dict(id='102'),dict(id='103')])))
    monkeypatch.setenv('USER','fixture')
    monkeypatch.setattr(workflow.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='102_[0-4]\n999\n'))
    calls = []
    monkeypatch.setattr(legacy_submit, 'submit_plan', lambda plan, path: calls.append(plan))
    workflow.research('new')
    args = calls[0][0]['args']
    assert '--dependency=afterany:102' in args
    assert not any('101' in x or '103' in x or '999' in x for x in args)


def test_failed_extension_audit_prevents_refresh_publication(monkeypatch):
    monkeypatch.setattr(workflow, 'settings', lambda _: {})
    def failed(_):
        raise RuntimeError('incomplete extension')
    monkeypatch.setattr(workflow, 'audit_extension', failed)
    calls = []
    monkeypatch.setattr(workflow, 'run_refresh', lambda *a: calls.append(a))
    with pytest.raises(RuntimeError, match='incomplete extension'):
        workflow.finish('new')
    assert not calls


def test_partial_refresh_publication_resumes_then_verifies_retained_bytes(tmp_path, monkeypatch):
    from src.utils import evaluation_refresh
    from src.utils.experiment import file_digest
    monkeypatch.setattr(evaluation_refresh, 'root', lambda _: tmp_path)
    target = tmp_path / 'active'; target.mkdir()
    evidence = target / 'predictions.csv'; evidence.write_text('synthetic fixture')
    move = dict(target=str(target), done=False, files={'predictions.csv':file_digest(evidence)})
    journal = tmp_path / 'publication.json'
    journal.write_text(json.dumps(dict(complete=False,moves=[move])))
    calls = []
    def run(record, value, action):
        calls.append(action)
        move['done'] = True
        journal.write_text(json.dumps(dict(complete=True,moves=[move])))
    monkeypatch.setattr(workflow, 'run_refresh', run)
    assert workflow.finalize_refresh({'existing_refresh':{'id':'old'}},'new') == 'old'
    assert calls == ['promote']
    evidence.write_text('changed after publication')
    with pytest.raises(RuntimeError, match='evidence changed'):
        workflow.finalize_refresh({'existing_refresh':{'id':'old'}},'new')


def test_fresh_recovery_can_validate_alongside_original_audit(tmp_path, monkeypatch):
    from src.utils import legacy_submit, clean_run
    spec=tmp_path/'environment.json'; spec.write_text('{}')
    monkeypatch.setattr(workflow, 'root', lambda _: tmp_path/'new')
    monkeypatch.setattr(workflow, 'existing_refresh', lambda *a: dict(id='old',commit='a'*40))
    def forbidden(*a):
        raise AssertionError('Fresh isolated recovery does not migrate shared output')
    monkeypatch.setattr(clean_run, 'check_cluster_writers', forbidden)
    submitted=[]
    monkeypatch.setattr(legacy_submit, 'submit_plan', lambda plan, path: submitted.extend(plan))
    workflow.submit('new','image1',spec,refresh_id='old',refresh_checkout=tmp_path)
    assert [p['key'] for p in submitted] == ['runtime_prepare','runtime_gpu','runtime_accept','launch']
    with pytest.raises(ValueError, match='fresh workflow'):
        workflow.submit('old','image1',spec,refresh_id='old',refresh_checkout=tmp_path)
