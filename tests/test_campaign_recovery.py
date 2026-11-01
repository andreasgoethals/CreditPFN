"""Recovery must preserve science, reject stale gates and never duplicate completed work."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from src.utils import campaign_recovery as cr, source_transition as st
from src.utils.experiment import digest_json


def test_reviewed_transition_keeps_training_math_and_rejects_an_optimizer_change(monkeypatch):
    from src.utils.paths import REPO_ROOT
    result = st.verify_delta(REPO_ROOT, '9c55d72fe64ef380d74ad57c7b3274d2f5458111')
    assert result['before'].startswith('9c55d72')
    original = Path.read_text
    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.name == 'loop.py':
            modified = text.replace('optimizer.zero_grad(set_to_none=True)', 'optimizer.zero_grad(set_to_none=False)')
            assert modified != text
            return modified
        return text
    monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(RuntimeError, match='Training mathematics'):
        st.verify_delta(REPO_ROOT, result['before'])


def test_transition_rejects_scientific_file_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(st, '_git', lambda *args: 'config/experiment5/pd.yaml')
    with pytest.raises(RuntimeError, match='Unreviewed source'):
        st.verify_delta(tmp_path, 'original')


def test_acceptance_bypass_never_leaks_to_child_jobs(monkeypatch):
    from src.utils.legacy_submit import submission_environment
    monkeypatch.setenv('CREDITPFN_RUNTIME_ACCEPTANCE','1')
    monkeypatch.setenv('CREDITPFN_LEGACY_RUNTIME_COMPARISON','1')
    monkeypatch.setenv('CREDITPFN_RUNTIME_VERSION','accepted_image')
    child=submission_environment()
    assert 'CREDITPFN_RUNTIME_ACCEPTANCE' not in child
    assert 'CREDITPFN_LEGACY_RUNTIME_COMPARISON' not in child
    assert child['CREDITPFN_RUNTIME_VERSION']=='accepted_image'


def test_native_lustre_submission_checks_the_actual_cluster_response(tmp_path):
    from src.utils.legacy_submit import submit_plan
    calls=[]
    def runner(args, **kwargs):
        calls.append(args)
        return NS(returncode=0,stdout='123;wice',stderr='')
    plan=[dict(key='finish',needs=[],tasks=None,args=['finish.slurm'])]
    state=submit_plan(plan,tmp_path/'receipt.json',runner=runner,max_array_size=1001,cluster='wice')
    assert state['accepted']==[dict(key='finish',tasks=None,id='123')]
    assert '--clusters=wice' in calls[0] and '--clusters=mindwell' not in calls[0]
    with pytest.raises(RuntimeError,match='cluster differs'):
        submit_plan(plan,tmp_path/'receipt.json',runner=runner,max_array_size=1001,cluster='mindwell')
    with pytest.raises(RuntimeError,match='unconfirmed'):
        submit_plan(plan,tmp_path/'wrong.json',runner=lambda *a,**k:NS(returncode=0,stdout='123;mindwell',stderr=''),max_array_size=1001,cluster='wice')


def test_final_handoff_retains_running_arrays_physical_task_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(cr,'root',lambda _:tmp_path)
    (tmp_path/'research.submission.json').write_text(json.dumps(dict(
        in_flight=None,index=1,plan=[{}],accepted=[dict(id='123')])) )
    monkeypatch.setenv('USER','fixture')
    def queue(args,**kwargs):
        assert args[-1]=='%F'
        return '123\n123\n'
    monkeypatch.setattr(cr.subprocess,'check_output',queue)
    with pytest.raises(RuntimeError,match='consumers remain'):
        cr.dispatch('repair')


def test_compatibility_requires_an_explicit_gated_transition(monkeypatch):
    monkeypatch.setattr(st, 'authorization', lambda *a: None)
    assert st.compatible_code('same', 'same')
    assert not st.compatible_code('old', 'new')
    monkeypatch.setattr(st, 'authorization', lambda *a: dict(old_train='old', new_train='new', old_eval='e1', new_eval='e2'))
    assert st.compatible_code('old', 'new')
    assert st.compatible_code('e1', 'e2', stage='eval')
    assert not st.compatible_code('new', 'old')
    assert not st.compatible_code('different', 'new')


@pytest.mark.parametrize('field,value', [('seed', 43), ('rows', 8191), ('rate', .01), ('data', 'other'), ('versions', 'changed'), ('code_sha256', 'unapproved')])
def test_planned_identity_accepts_only_code_continuity(tmp_path, monkeypatch, field, value):
    from src.utils import prepare_experiment as prep
    original = dict(code_sha256='old', seed=42, rows=8192, rate=3e-7, data='sha', versions='fixed')
    path = tmp_path/'plan.json'
    payload = dict(identities={digest_json(original):original})
    payload['sha256'] = digest_json(payload)
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(prep, 'plan_path', lambda *a: path)
    monkeypatch.setattr(st, 'authorization', lambda: dict(id='original', old_train='old', new_train='new'))
    monkeypatch.setenv('CREDITPFN_LEGACY_ID','original')
    cfg = NS(experiment=NS(extension_id='original'), run_name='run', track='pd')
    current = dict(original, code_sha256='new')
    assert st.planned_identity(cfg, current) == dict(sha256=digest_json(original), specification=original)
    current[field] = value
    with pytest.raises(RuntimeError, match='Scientific specification'):
        st.planned_identity(cfg, current)


def test_authorization_checks_fresh_gates_and_binding_tampering(tmp_path, monkeypatch):
    from src.utils import legacy_design, legacy_checks, runtime_validation
    monkeypatch.setattr(legacy_design, 'root', lambda _:tmp_path)
    monkeypatch.setattr(st, 'code_identity', lambda **kw:'eval' if kw else 'train')
    calls=[]
    monkeypatch.setattr(runtime_validation, 'require', lambda v:calls.append(('runtime',v)))
    monkeypatch.setattr(legacy_checks, 'require_gate', lambda v:calls.append(('gate',v)))
    evidence=tmp_path/'evidence';evidence.write_text('passed')
    record=dict(id='original', passed=True, new_train='train', new_eval='eval', gate_id='fresh', runtime='image',
                bindings=[dict(path=str(evidence),sha256=st.file_digest(evidence))])
    path=tmp_path/'source-transition.json';path.write_text(json.dumps(record))
    assert st.authorization('original')==record
    assert calls==[('runtime','image'),('gate','fresh')]
    evidence.write_text('changed')
    with pytest.raises(RuntimeError, match='evidence changed'):st.authorization('original')
    record['new_train']='stale';path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match='identity changed'):st.authorization('original')


def test_bootstrap_requires_runtime_all_eighty_cases_and_native_signal(tmp_path, monkeypatch):
    from src.utils import legacy_submit
    monkeypatch.setattr(cr, 'root', lambda _:tmp_path/'recovery')
    monkeypatch.setattr(cr, 'extension_root', lambda _:tmp_path/'gate')
    monkeypatch.setattr(st, 'verify_delta', lambda *a:dict(after='current'))
    calls=[]
    monkeypatch.setattr(legacy_submit, 'submit_plan', lambda plan,*a:calls.extend(plan))
    # The real flock has separate Windows/native tests; no Linux emulation here.
    from contextlib import nullcontext
    monkeypatch.setattr(cr, 'writer', lambda *a:nullcontext())
    monkeypatch.setattr(cr, 'archive_failures', lambda *a:None)
    cr.bootstrap('repair','original','fresh','before','image','environment.json')
    bykey={p['key']:p for p in calls}
    assert bykey['runtime_gpu']['tasks']==list(range(10))
    assert bykey['smoke']['tasks']==list(range(12))
    assert bykey['capacity']['tasks']==list(range(68))
    assert set(bykey['recover']['needs'])=={'inspect','signal'}
    assert bykey['signal']['needs']==['runtime_accept']
    assert not any('%' in str(x['args']) for x in calls)


def test_missing_roster_uses_saved_context_and_completed_training(tmp_path, monkeypatch):
    from src.utils import legacy_submit, audit_experiment
    original=tmp_path/'original';original.mkdir()
    (original/'workflow.json').write_text(json.dumps(dict(production={'e5_pd_v2':'config.yaml'})))
    monkeypatch.setattr(cr,'record',lambda _:dict(original='old'))
    monkeypatch.setattr(cr,'extension_root',lambda _:original)
    monkeypatch.setattr(cr,'resolve_staging_path',lambda _:tmp_path/'project')
    monkeypatch.setattr(cr,'manifests_dir',lambda _:tmp_path/'context')
    monkeypatch.setattr(audit_experiment,'audit',lambda _:dict(expected=4,problems=[],diverged=0,
        trials=[dict(status=s) for s in ['OK','PENDING','OK','OK']]))
    context=tmp_path/'context.json';context.write_text(json.dumps(dict(sha256='fixed',config={'run_name':'curve'})))
    receipts=tmp_path/'context/completed/curve';receipts.mkdir(parents=True)
    (receipts/'task_0000.json').write_text(json.dumps(dict(plan_sha256='fixed',passed=True,task=0)))
    full=[dict(key=f'e5_pd_v2_s{i}',tasks=[0],needs=[],args=['--time=03:00:00']) for i in range(4)]
    full+=[dict(key='e10_v2',tasks=[0,1],needs=[],args=['context.slurm',str(context)])]
    full+=[dict(key=f'e5_pd_v2_s{i}_benchmark',tasks=list(range(16)),needs=[f'e5_pd_v2_s{i}'],args=[]) for i in range(4)]
    monkeypatch.setattr(legacy_submit,'research_plan',lambda _:deepcopy(full))
    plan=cr.missing_plan('repair')
    bykey={x['key']:x for x in plan}
    assert [x['key'] for x in plan[:2]]==['e5_pd_v2_s1','e10_v2']
    assert bykey['e10_v2']['tasks']==[1]
    assert '--time=08:00:00' in bykey['e10_v2']['args']
    assert bykey['e5_pd_v2_s0_benchmark']['needs']==[]
    assert bykey['e5_pd_v2_s1_benchmark']['needs']==['e5_pd_v2_s1']
    assert len(bykey)==6
    (receipts/'task_0000.json').write_text(json.dumps(dict(plan_sha256='wrong',passed=True,task=0)))
    with pytest.raises(RuntimeError,match='Context completion'):cr.missing_plan('repair')


def test_cleanup_ledger_resumes_without_removing_changed_or_peer_files(tmp_path, monkeypatch):
    out=tmp_path/'output';out.mkdir()
    folder=tmp_path/'recovery';folder.mkdir()
    failed=out/'failed.log';failed.write_text('old failure')
    peer=out/'peer.log';peer.write_text('success')
    archive=tmp_path/'retained.tar';archive.write_bytes(b'archive')
    ledger=dict(archive=str(archive),sha256=cr.file_digest(archive),files={'failed.log':cr.file_digest(failed)},removed=[])
    (folder/'failed-output-retirement.json').write_text(json.dumps(ledger))
    monkeypatch.setattr(cr,'root',lambda _:folder)
    monkeypatch.setattr(cr,'record',lambda _:dict(original='original'))
    monkeypatch.setattr(cr,'outputs_dir',lambda:out)
    cr.archive_failures('repair')
    assert not failed.exists() and peer.read_text()=='success'
    failed.write_text('new unrelated execution')
    with pytest.raises(RuntimeError,match='changed after archival'):cr.archive_failures('repair')
    assert failed.read_text()=='new unrelated execution'


def test_final_coordinator_retains_global_manifest_scope(tmp_path, monkeypatch):
    from src.utils import study_workflow
    monkeypatch.delenv('CREDITPFN_LEGACY_ID',raising=False)
    monkeypatch.setattr(cr,'record',lambda _:dict(original='original'))
    monkeypatch.setattr(cr,'root',lambda _:tmp_path/'repair')
    monkeypatch.setattr(study_workflow,'root',lambda _:tmp_path/'study')
    (tmp_path/'repair').mkdir();(tmp_path/'study').mkdir()
    (tmp_path/'study/complete.json').write_text('verified completion')
    calls=[]
    def check(identifier):
        import os
        assert 'CREDITPFN_LEGACY_ID' not in os.environ
        calls.append(identifier)
    monkeypatch.setattr(st,'authorization',check)
    monkeypatch.setattr(study_workflow,'finish',check)
    cr.finish('repair')
    assert calls==['original','original']
    assert json.loads((tmp_path/'repair/complete.json').read_text())['passed']
