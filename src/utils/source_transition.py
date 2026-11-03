"""Explicit, gated continuity for the optimizer-boundary signal repair.

Original plans, trial IDs and passing evidence remain immutable. A transition
can only be installed after fresh runtime, full adapter/capacity and native
signal checks. It never relaxes a scientific setting or a numerical tolerance.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

from src.utils.experiment import code_identity, digest_json, file_digest


def _git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True, encoding='utf-8').strip()


def verify_delta(repo, before):
    """Reject any model/data/config change; verify the two training edits exactly."""
    allowed = {
        'scripts/slurm/_run_train.sh', 'scripts/slurm/study_cpu.slurm',
        'scripts/slurm/recovery_cpu.slurm', 'scripts/slurm/recovery_signal.slurm',
        'src/train/loop.py', 'src/train/recovery.py', 'src/eval/context_curve.py',
        'src/utils/experiment.py', 'src/utils/prepare_experiment.py',
        'src/utils/legacy_checks.py', 'src/utils/legacy_submit.py',
        'src/utils/source_transition.py', 'src/utils/campaign_recovery.py',
        'src/utils/signal_probe.py',
        'src/utils/admission_recovery.py', 'scripts/slurm/admission_resume.slurm',
        'src/utils/paths.py', 'src/utils/repair_extension_outputs.py', 'scripts/slurm/finalize_outputs.slurm',
        'src/utils/evaluation_evidence.py',
    }
    changed = _git(repo, 'diff', '--name-only', before, 'HEAD').splitlines()
    forbidden = [p for p in changed if p not in allowed and not p.startswith(('docs/', 'tests/'))]
    if forbidden:
        raise RuntimeError(f'Unreviewed source transition: {forbidden}')
    original = _git(repo, 'show', before + ':src/train/loop.py')
    expected = original.replace(
        'load_recovery, save_recovery, TrainingInterrupted',
        'load_recovery, save_recovery, TrainingInterrupted, stop_requested\n    from src.utils.source_transition import execution_record')
    expected = expected.replace('os.environ.get("CREDITPFN_STOP_REQUESTED") == "1"', 'stop_requested()')
    expected = expected.replace('"trial_identity": trial_identity,',
        '"trial_identity": trial_identity,\n        "execution_source": execution_record(),')
    actual = (Path(repo) / 'src/train/loop.py').read_text(encoding='utf-8')
    if ast.dump(ast.parse(expected)) != ast.dump(ast.parse(actual)):
        raise RuntimeError('Training mathematics changed beyond the reviewed signal/provenance repair')
    original = ast.parse(_git(repo, 'show', before + ':src/train/recovery.py'))
    actual = ast.parse((Path(repo) / 'src/train/recovery.py').read_text(encoding='utf-8'))
    actual.body = [n for n in actual.body if not isinstance(n, ast.FunctionDef) or n.name != 'stop_requested']
    if ast.dump(original) != ast.dump(actual):
        raise RuntimeError('Checkpoint serialization or RNG recovery changed')
    for name, excluded in [('src/eval/context_curve.py', {'read_plan'}),
                           ('src/utils/experiment.py', {'trial_identity'}),
                           ('src/utils/prepare_experiment.py', {'check_prepared', 'check_size_evaluation'}),
                           ('src/utils/legacy_checks.py', {'read_workflow'}),
                           ('src/utils/evaluation_evidence.py', {'predictions'})]:
        old = ast.parse(_git(repo, 'show', before + ':' + name))
        new = ast.parse((Path(repo) / name).read_text(encoding='utf-8'))
        old.body = [n for n in old.body if not isinstance(n, ast.FunctionDef) or n.name not in excluded]
        new.body = [n for n in new.body if not isinstance(n, ast.FunctionDef) or n.name not in excluded]
        if ast.dump(old) != ast.dump(new):
            raise RuntimeError(f'Numerical/input logic changed in {name}')
    # The reviewed placement repair is confined to aliases for output paths.
    # Input/weight resolution, scope construction and every other helper stay exact.
    old = ast.parse(_git(repo, 'show', before + ':src/utils/paths.py'))
    new = ast.parse((Path(repo) / 'src/utils/paths.py').read_text(encoding='utf-8'))
    for tree in (old, new):
        tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef) or node.name != '_output_alias']
    if ast.dump(old) != ast.dump(new):
        raise RuntimeError('Input/weight/path logic changed beyond output placement')
    original = _git(repo, 'show', before + ':src/utils/experiment.py')
    expected = original.replace('return {"sha256": digest_json(payload), "specification": payload}',
        'from src.utils.source_transition import planned_identity\n    return planned_identity(cfg, payload)')
    if ast.dump(ast.parse(expected)) != ast.dump(ast.parse((Path(repo)/'src/utils/experiment.py').read_text(encoding='utf-8'))):
        raise RuntimeError('Trial specification construction changed')
    return dict(before=before, after=_git(repo, 'rev-parse', 'HEAD'), changed=changed)


def authorization(identifier=None):
    """No environment flag can waive a gate; all bindings are checked on disk."""
    from src.utils.legacy_design import root, validate_id
    value = identifier or os.environ.get('CREDITPFN_LEGACY_ID')
    if not value:
        return None
    path = root(validate_id(value)) / 'source-transition.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (record.get('id') != value or record.get('passed') is not True
            or record.get('new_train') != code_identity()
            or record.get('new_eval') != code_identity(stage='eval')):
        raise RuntimeError('Source-transition identity changed')
    if record['gate_id'] == value:
        raise RuntimeError('Transition cannot certify its own original gate')
    from src.utils.runtime_validation import require
    from src.utils.legacy_checks import require_gate
    require(record['runtime'])
    require_gate(record['gate_id'])
    for binding in record['bindings']:
        if file_digest(Path(binding['path'])) != binding['sha256']:
            raise RuntimeError('Source-transition validation evidence changed')
    return record


def compatible_code(previous, current, *, stage='train', identifier=None):
    if previous == current:
        return True
    record = authorization(identifier)
    return bool(record and previous == record['old_' + stage] and current == record['new_' + stage])


def planned_identity(cfg, payload):
    """Retain the original planned identity only for a completely equal specification."""
    from src.utils.prepare_experiment import plan_path, read_plan
    current = dict(sha256=digest_json(payload), specification=payload)
    if not os.environ.get('CREDITPFN_LEGACY_ID'):
        return current
    record = authorization()
    if not record:
        return current
    if str(cfg.experiment.extension_id) != record['id']:
        raise RuntimeError('Trial escaped its approved source-transition scope')
    if payload.get('code_sha256') != record['new_train']:
        raise RuntimeError('Scientific specification has an unapproved execution source')
    original = deepcopy(payload)
    original['code_sha256'] = record['old_train']
    key = digest_json(original)
    plan = read_plan(plan_path(str(cfg.run_name), str(cfg.track)))
    if plan['identities'].get(key) != original:
        raise RuntimeError('Scientific specification differs from the original planned trial')
    return dict(sha256=key, specification=original)


def execution_record():
    record = authorization()
    return (dict(planned_train_code=record['old_train'], executed_train_code=record['new_train'],
                 transition_sha256=file_digest(Path(record['transition_path']))) if record else None)


def install(identifier, gate_id, runtime, before, signal_receipt):
    from src.utils.legacy_design import root
    from src.utils.legacy_checks import require_gate, verify_bindings
    from src.utils.runtime_validation import require, root as runtime_root
    from src.utils.paths import REPO_ROOT
    from src.utils.atomic import write_json
    from src.utils.experiment import environment_versions
    require(runtime)
    require_gate(gate_id)
    delta = verify_delta(REPO_ROOT, before)
    signal = json.loads(Path(signal_receipt).read_text())
    if signal != dict(passed=True, code=code_identity(stage='eval'), runtime=runtime,
                      warnings=['USR1', 'TERM'], exact_resume=True):
        raise RuntimeError('Native container signal/restart check is missing or stale')
    old = json.loads((root(identifier) / 'workflow.json').read_text())
    if old['versions'] != environment_versions(stage='eval'):
        raise RuntimeError('Transition must preserve the exact installed scientific environment')
    verify_bindings(old['files'], full_hash=True)
    verify_bindings(old['inputs'], full_hash=True)
    # Verify the original independent inspection and all eighty original receipts
    # against their original workflow, without relabelling any of them.
    from src.utils.legacy_checks import receipts
    checked = receipts(identifier, old, hashes=True)
    expected = dict(id=identifier, passed=True, workflow_sha256=digest_json(old), receipts=checked)
    if json.loads((root(identifier) / 'inspection.json').read_text()) != expected:
        raise RuntimeError('Original eighty-case gate no longer matches its evidence')
    plan_files = [Path(x['path']) for x in old['files'] if '/manifests/plans/' in x['path']]
    train_codes, eval_codes = set(), set()
    for path in plan_files:
        value = json.loads(path.read_text())
        if 'identities' in value:
            train_codes.update(x['code_sha256'] for x in value['identities'].values())
        elif 'code' in value:
            eval_codes.add(value['code'])
    if len(train_codes) != 1 or len(eval_codes) != 1:
        raise RuntimeError('Original source identities are ambiguous')
    path = root(identifier) / 'source-transition.json'
    gate_marker = root(gate_id) / 'workflow.json'
    if gate_marker.exists():
        gate_marker.rename(root(gate_id) / 'validation-workflow.json')
    files = [root(identifier)/'workflow.json', root(identifier)/'inspection.json',
             root(gate_id)/'inspection.json', root(gate_id)/'validation-workflow.json',
             runtime_root(runtime)/'accepted.json', Path(signal_receipt)]
    record = dict(id=identifier, passed=True, gate_id=gate_id, runtime=runtime,
                  old_train=train_codes.pop(), old_eval=eval_codes.pop(),
                  old_fingerprint=old['fingerprint'], new_train=code_identity(),
                  new_eval=code_identity(stage='eval'), transition_path=str(path), source_delta=delta,
                  bindings=[dict(path=str(p), sha256=file_digest(p)) for p in files])
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise RuntimeError('An immutable transition already exists with other contents')
    else:
        write_json(path, record, exclusive=True)
    return authorization(identifier)
