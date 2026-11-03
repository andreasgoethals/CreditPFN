from copy import deepcopy
from pathlib import Path

import pytest

from src.utils.repair_extension_outputs import move, roster, sha, validate_roster


def artifact(root, name='cpt_lora_legacy_case1_v2_s00_20261102_123456.csv'):
    p = root / 'experiment6/results/PD/tabpfn-trained__v2__lora' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b'original immutable result bytes\n')
    return p


def test_relocation_preserves_bytes_and_unrelated_experiments(tmp_path):
    selected = artifact(tmp_path)
    other = artifact(tmp_path, 'cpt_lora_v1_s00_20261001.csv')
    different = artifact(tmp_path, 'cpt_lora_legacy_case2_v2_s00_20261102.csv')
    planned = roster(tmp_path, 'case1', ['experiment6'])
    assert len(planned) == 1
    writes = []
    move(planned, lambda: writes.append(deepcopy(planned)))
    target = Path(planned[0]['target'])
    assert target == tmp_path / 'experiment6/extensions/case1/results/PD' / selected.parent.name / selected.name
    assert target.read_bytes() == other.read_bytes() == different.read_bytes()
    assert not selected.exists() and other.exists() and different.exists()
    assert writes[0][0]['done']


def test_resume_after_publish_before_receipt_and_repeated_completion(tmp_path):
    source = artifact(tmp_path)
    planned = roster(tmp_path, 'case1', ['experiment6'])
    target = Path(planned[0]['target'])
    target.parent.mkdir(parents=True)
    source.rename(target)
    move(planned, lambda: None)
    assert planned[0]['done']
    move(planned, lambda: None)
    target.write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match='changed'):
        move(planned, lambda: None)


def test_collision_and_changed_source_never_overwrite_or_delete(tmp_path):
    source = artifact(tmp_path)
    planned = roster(tmp_path, 'case1', ['experiment6'])
    source.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='Source changed'):
        move(planned, lambda: None)
    assert source.read_bytes() == b'changed'
    source.write_bytes(b'original immutable result bytes\n')
    target = Path(planned[0]['target'])
    target.parent.mkdir(parents=True)
    target.write_bytes(b'other campaign')
    with pytest.raises(RuntimeError, match='collision'):
        move(planned, lambda: None)
    assert source.exists() and target.read_bytes() == b'other campaign'


@pytest.mark.parametrize('identifier,groups', [('bad/path', ['experiment6']), ('case1', ['experiment1'])])
def test_relocation_rejects_unowned_scope(tmp_path, identifier, groups):
    artifact(tmp_path)
    with pytest.raises(ValueError):
        roster(tmp_path, identifier, groups)


def test_journal_cannot_redirect_or_duplicate_owned_files(tmp_path):
    artifact(tmp_path)
    planned = roster(tmp_path, 'case1', ['experiment6'])
    validate_roster(planned, tmp_path, 'case1')
    with pytest.raises(RuntimeError, match='Duplicate'):
        validate_roster(planned + planned, tmp_path, 'case1')
    changed = deepcopy(planned)
    changed[0]['target'] = str(tmp_path / 'experiment1/results/stolen.csv')
    with pytest.raises(RuntimeError, match='target escaped'):
        validate_roster(changed, tmp_path, 'case1')
    changed = deepcopy(planned)
    changed[0]['source'] = str(tmp_path.parent / 'outside.csv')
    with pytest.raises(RuntimeError, match='source escaped'):
        validate_roster(changed, tmp_path, 'case1')
