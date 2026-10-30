"""Invalidate owned publications by their campaign and extension dependencies."""
import json
from pathlib import Path

from src.utils.atomic import write_json
from src.utils.study_layout import notebook_campaigns


def plan(root, campaigns, *, extension_id=None):
    from src.utils.clean_run import validate_tree
    root = Path(root)
    files, edits = [], {}
    for manifest in sorted((root / 'figures').glob('*/_metadata/*.json')):
        validate_tree(manifest)
        group = manifest.parent.parent.name
        notebook = group + '/' + manifest.stem
        entries = json.loads(manifest.read_text(encoding='utf-8'))
        owned = []
        for entry in entries:
            provenance = entry.get('provenance', {})
            dependencies = provenance.get('campaign_ids', notebook_campaigns(notebook))
            # Old single-experiment figures use public folder numbers; retained
            # solely to retire pre-RQ artifacts, never as the current convention.
            if not dependencies and group.startswith('experiment'):
                from src.utils.study_layout import campaign_for_study
                dependencies = [campaign_for_study(group)]
            if not set(dependencies).intersection(campaigns):
                continue
            if extension_id is not None:
                ids = provenance.get('extension_ids', [])
                if provenance.get('extension_id') != extension_id and extension_id not in ids:
                    continue
            stem = entry['stem']
            if Path(stem).name != stem or '/' in stem or '\\' in stem or stem in ('.', '..'):
                raise ValueError('Invalid publication filename; refusing cleanup')
            path = manifest.parent.parent / (stem + '.pdf')
            validate_tree(path)
            files.append(path)
            owned.append(entry)
        if owned:
            edits[manifest] = [e for e in entries if e not in owned]
    return list(dict.fromkeys(files)), edits


def apply(files, edits):
    for path in files:
        path.unlink(missing_ok=True)
    for path, entries in edits.items():
        write_json(path, entries)
    if edits:
        from src.utils.run_notebooks import discover, write_captions
        write_captions(discover(), migrate=False)
        print('Invalidated dependent figures and captions. Rerun notebooks to refresh inline output and summaries.')
