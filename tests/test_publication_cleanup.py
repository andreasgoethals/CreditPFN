import json

import pytest
from omegaconf import OmegaConf

from src.utils import clean_run, publication_cleanup, study_layout
from src.utils.atomic import write_json


def page(root, group, name, campaigns, ids=()):
    folder = root / 'figures' / group
    entry = dict(stem=name, name=name, caption='Fixture', index=1,
                 provenance=dict(campaign_ids=campaigns, extension_ids=list(ids)))
    write_json(folder / '_metadata' / (name+'.json'), [entry])
    pdf = folder / (name+'.pdf')
    pdf.write_bytes(b'pdf')
    return pdf


def test_publication_cleanup_uses_dependencies_and_preserves_other_rqs(tmp_path, monkeypatch):
    monkeypatch.setenv('CREDITPFN_OUTPUT_ROOT', str(tmp_path))
    monkeypatch.setenv('CREDITPFN_STAGING_ROOT', str(tmp_path))
    monkeypatch.delenv('VSC_DATA', raising=False)
    root = clean_run.outputs_dir()
    lora = page(root, 'rq2', 'lora', ['experiment8'])
    mixed = page(root, 'general', '03_overview_pd', ['experiment1', 'experiment8'])
    unrelated = page(root, 'rq3', 'sampling', ['experiment3'])
    corpus = page(root, 'general', '01_raw_data', [])
    clean_run.main(['--experiment', 'experiment6', '--scope', 'figures', '--clean'])
    assert not lora.exists() and not mixed.exists()
    assert unrelated.exists() and corpus.exists()
    assert json.loads((root/'figures/rq2/_metadata/lora.json').read_text()) == []


def test_extension_cleanup_invalidates_combined_pages_but_preserves_originals(tmp_path):
    changed = page(tmp_path, 'rq2', '02_results', ['experiment5','experiment8'], ['new'])
    original = page(tmp_path, 'rq2', 'original', ['experiment8'])
    other = page(tmp_path, 'rq3', '02_results', ['experiment3'])
    files, edits = publication_cleanup.plan(tmp_path, {'experiment8'}, extension_id='new')
    assert files == [changed]
    assert original.exists() and other.exists()
    assert len(edits) == 1


def test_shared_factorial_requires_explicit_scientific_retirement():
    with pytest.raises(SystemExit):
        clean_run.main(['--experiment', 'experiment3', '--clean'])


def test_yaml_is_the_exact_layout_authority_and_public_numbers_resolve():
    expected = OmegaConf.to_container(OmegaConf.load('config/study.yaml'), resolve=True)
    assert study_layout.manifest() == {key: expected[key] for key in ('schema','campaigns','studies','factorial_unique_runs')}
    assert study_layout.campaign_for_study(6) == 'experiment8'
    assert study_layout.campaign_for_study(10) == 'experiment4'
    assert study_layout.notebook_campaigns('rq2/02_results') == ['experiment1','experiment5','experiment8']


def test_publication_rejects_metadata_path_escape(tmp_path):
    page(tmp_path, 'rq2', 'bad', ['experiment8'], ['new'])
    p = tmp_path / 'figures/rq2/_metadata/bad.json'
    entries = json.loads(p.read_text())
    entries[0]['stem'] = '../outside'
    write_json(p, entries)
    with pytest.raises(ValueError, match='filename'):
        publication_cleanup.plan(tmp_path, {'experiment8'}, extension_id='new')
