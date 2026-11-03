"""Recover misplaced extension benchmarks without recomputing or changing their bytes.

Run this helper by absolute path from an isolated DATA Git checkout; scientific
audits/publication import their original, accepted evaluator from --checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def roster(project_output, identifier, groups):
    import re
    if not re.fullmatch(r'[A-Za-z0-9]{1,40}', identifier):
        raise ValueError('Invalid extension ID')
    entries = []
    for group in groups:
        if group not in {'experiment5', 'experiment6', 'experiment8', 'experiment9'}:
            raise ValueError('Unsupported public benchmark group')
        parent = project_output / group
        origin = parent / 'results'
        target = parent / 'extensions' / identifier / 'results'
        for track in ('PD', 'LGD'):
            for file in sorted((origin / track).glob('tabpfn-*/*')):
                if not file.is_file() or f'_legacy_{identifier}_' not in file.name:
                    continue
                if (not file.name.startswith('cpt_') or file.is_symlink()
                        or not file.resolve().is_relative_to(origin.resolve())):
                    raise RuntimeError('Unexpected extension result file')
                if file.suffix not in {'.csv', '.gz', '.json', '.parquet'}:
                    raise RuntimeError('Unknown extension result artifact')
                destination = target / file.relative_to(origin)
                if destination.exists():
                    raise RuntimeError('Destination already exists; reconcile before migration')
                entries.append(dict(source=str(file), target=str(destination),
                                    bytes=file.stat().st_size, sha256=sha(file), done=False))
    if not entries:
        raise RuntimeError('No misplaced extension result artifacts')
    return entries


def move(entries, publish):
    """Resume a rename only after verifying bytes, including the crash-before-receipt case."""
    for index, item in enumerate(entries):
        source, target = Path(item['source']), Path(item['target'])
        if item['done']:
            if source.exists() or not target.is_file() or sha(target) != item['sha256']:
                raise RuntimeError('Completed relocation changed')
            continue
        if source.is_file():
            if (source.stat().st_size != item['bytes'] or sha(source) != item['sha256']
                    or target.exists() and not os.path.samefile(source, target)):
                raise RuntimeError('Source changed or target collision; preserving both')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(source, target)  # Atomic create; never overwrite an existing artifact.
            if sha(target) != item['sha256']:
                raise RuntimeError('Target changed before source retirement')
            source.unlink()
        if not target.is_file() or target.stat().st_size != item['bytes'] or sha(target) != item['sha256']:
            raise RuntimeError('Relocated artifact failed verification')
        item['done'] = True
        if (index + 1) % 32 == 0:
            publish()
    publish()
    # Remove only empty method/track directories touched by these exact files.
    for directory in sorted({Path(i['source']).parent for i in entries}, key=str):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def validate_roster(entries, project_output, identifier):
    import re
    root = project_output.resolve()
    sources, targets = set(), set()
    for item in entries:
        source, target = Path(item['source']), Path(item['target'])
        if not source.is_absolute() or not source.resolve().is_relative_to(root):
            raise RuntimeError('Relocation source escaped PROJECT output')
        parts = source.relative_to(project_output).parts
        if (len(parts) != 5 or parts[0] not in {'experiment5','experiment6','experiment8','experiment9'}
                or parts[1] != 'results' or parts[2] not in {'PD','LGD'} or not parts[3].startswith('tabpfn-')
                or not parts[4].startswith('cpt_') or f'_legacy_{identifier}_' not in parts[4]):
            raise RuntimeError('Relocation does not own this source')
        expected = project_output / parts[0] / 'extensions' / identifier / 'results' / Path(*parts[2:])
        if target != expected or not target.resolve().is_relative_to(root) or source.is_symlink() or target.is_symlink():
            raise RuntimeError('Relocation target escaped its exact extension scope')
        if (source in sources or target in targets or not re.fullmatch('[0-9a-f]{64}', item['sha256'])
                or not isinstance(item['bytes'], int) or item['bytes'] < 0):
            raise RuntimeError('Duplicate or malformed relocation entry')
        sources.add(source); targets.add(target)


def run(checkout, recovery, apply=False):
    import sys
    checkout = checkout.resolve(strict=True)
    os.chdir(checkout)
    sys.path.insert(0, str(checkout))
    from src.utils import campaign_recovery as cr
    from src.utils.source_transition import authorization
    from src.utils.atomic import write_json
    from src.utils.paths import resolve_staging_path, outputs_dir
    cfg = cr.record(recovery)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if actual != cfg['source']:
        raise RuntimeError('Frozen scientific checkout changed')
    folder = cr.root(recovery)
    with cr.writer(folder / 'output-finalization.lease'):
        authorization(cfg['original'])
        queue = subprocess.check_output(['squeue', '-M', 'mindwell,wice', '-u', os.environ['USER'],
            '-h', '-o', '%j'], text=True, timeout=30)
        if any(name.startswith(('creditpfn-pd-', 'creditpfn-lgd-', 'creditpfn-context-',
                'creditpfn-benchmark-', 'creditpfn-admission-', 'creditpfn-recovery-')) for name in queue.splitlines()):
            raise RuntimeError('Scientific or submission consumers remain')
        journal = folder / 'benchmark-relocation.json'
        if journal.exists():
            record = json.loads(journal.read_text())
            if record['extension'] != cfg['original'] or record['source'] != actual:
                raise RuntimeError('Relocation belongs to another source/campaign')
        else:
            files = roster(resolve_staging_path('output CreditPFN'), cfg['original'],
                           ['experiment5', 'experiment6', 'experiment8', 'experiment9'])
            record = dict(extension=cfg['original'], source=actual, files=files, complete=False)
        validate_roster(record['files'], resolve_staging_path('output CreditPFN'), cfg['original'])
        print(f"Verified relocation roster: {len(record['files'])} files / "
              f"{sum(v['bytes'] for v in record['files'])} bytes", flush=True)
        if not apply:
            return record
        if not journal.exists():
            write_json(journal, record, exclusive=True)
        move(record['files'], lambda: write_json(journal, record))
        record['complete'] = True
        write_json(journal, record)
        print('Relocation verified. Starting original-source independent final audits/publication.', flush=True)
        # Keep all accepted scientific code and all numeric/cohort checks. Only
        # replace the legacy CSV decoder with the reviewed lossless dtype reader.
        import ast
        import importlib.util
        from src.utils import evaluation_evidence as evidence
        reader_file = Path(__file__).with_name('evaluation_evidence.py')
        before = ast.parse(Path(evidence.__file__).read_text(encoding='utf-8'))
        after = ast.parse(reader_file.read_text(encoding='utf-8'))
        for tree in (before, after):
            tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef) or node.name != 'predictions']
        if ast.dump(before) != ast.dump(after):
            raise RuntimeError('Independent numeric/cohort checks changed beyond prediction decoding')
        spec = importlib.util.spec_from_file_location('_verified_prediction_reader', reader_file)
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        write_json(folder / 'auditor-io-fidelity.json', dict(scientific_source=actual,
            original_reader_sha256=sha(Path(evidence.__file__)), corrected_reader_sha256=sha(reader_file),
            unchanged_numeric_checks=True, lossless_float32_only=True))
        evidence.predictions = reader.predictions
        cr.finish(recovery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--recovery', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    run(args.checkout, args.recovery, args.apply)
