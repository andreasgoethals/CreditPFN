"""Ten research analyses over immutable historical campaigns; never renumber trial IDs.

The main factorial is stored once, under shared/main_grid. E1/E3/E4 are views of
that evidence, not additional runs. The other original payloads move intact.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path

MARKER = "STUDY_LAYOUT.json"
CAMPAIGNS = {"experiment1": "shared/main_grid", "experiment2": "experiment2",
             "experiment3": "experiment7", "experiment4": "experiment10",
             "experiment5": "experiment5", "experiment6": "experiment8",
             "experiment7": "experiment9", "experiment8": "experiment6"}
NUMBERS = {1: 1, 2: 2, 3: 7, 4: 10, 5: 5, 6: 8, 7: 9, 8: 6}
STUDIES = {
    1: (1, "Learning rate and training progress", ["experiment1"], "Full updates; L2-SP 0.003; all prescribed rates"),
    2: (1, "Training seed sensitivity", ["experiment1", "experiment2"], "Matched reference recipes at seeds 42 and 43"),
    3: (2, "L2-SP anchoring", ["experiment1"], "Both L2-SP levels, matched within rate/base/scope; retain interaction"),
    4: (2, "Backbone freezing", ["experiment1"], "Full/frozen at both L2-SP levels; retain interaction"),
    5: (2, "DiagScale recipe", ["experiment5"], "All prescribed rates; recipe-level evaluation"),
    6: (2, "LoRA recipe", ["experiment8"], "All prescribed rates; recipe-level evaluation"),
    7: (3, "Row exposure and update weighting", ["experiment3"], "One sample, full pass and accumulation at equal updates"),
    8: (3, "Training corpus composition", ["experiment6"], "All versus large PD tables at fixed episode size"),
    9: (3, "Episode size", ["experiment6", "experiment7"], "Fixed large-table corpus; shared 8192-row control"),
    10: (4, "Labeled target context", ["experiment4"], "Original/CPT and paired difference; fixed complete tests; five folds"),
}


def manifest():
    return dict(schema=2, campaigns=CAMPAIGNS, studies={f"experiment{k}":
        dict(rq=v[0], title=v[1], sources=[CAMPAIGNS[s] for s in v[2]], analysis=v[3])
        for k, v in STUDIES.items()}, factorial_unique_runs=512)


def payload(group, root):
    marker = root / MARKER
    if not marker.is_file():
        return group
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record != manifest():
        raise ValueError("Unrecognized study layout; refuse ambiguous result routing")
    return CAMPAIGNS.get(group, group)


def _hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@contextmanager
def _lock(root):
    path = root / ".study-v2-migration.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(dict(pid=os.getpid()), stream)
    except FileExistsError as error:
        raise RuntimeError(f"Migration ownership lock exists; inspect before recovery: {path}") from error
    try:
        yield
    finally:
        path.unlink()


def migrate(roots, *, apply=False, workers=4):
    roots = list(dict.fromkeys(Path(p).resolve(strict=True) for p in roots))
    if not roots or any(not p.is_dir() for p in roots):
        raise ValueError("Migration needs existing output directories")
    if any(a != b and (a in b.parents or b in a.parents) for a in roots for b in roots):
        raise ValueError("Overlapping migration roots")
    with ExitStack() as stack:
        if apply:
            for root in sorted(roots):
                stack.enter_context(_lock(root))
        return _migrate(roots, apply=apply, workers=workers)


def _migrate(roots, *, apply=False, workers=4):
    """Move complete campaign directories, hash-verify, then publish layout markers.

    No scientific file is rewritten or deleted. A persistent journal supports
    restart after interruption; ambiguous existing targets fail closed. The caller
    must stop writers before applying. Checkpoints retain their historical paths.
    """
    roots = list(dict.fromkeys(Path(p).resolve(strict=True) for p in roots))
    if any(a != b and (a in b.parents or b in a.parents) for a in roots for b in roots):
        raise ValueError("Overlapping migration roots")
    reports = []
    for root in roots:
        if (root / MARKER).is_file():
            payload("experiment1", root)
            reports.append(dict(root=str(root), state="already migrated"))
            continue
        journal = root / ".study-v2-migration.json"
        stage = root / ".study-v2-staging"
        if stage.is_symlink():
            raise ValueError("Migration staging directory is linked; inspect its ownership")
        if journal.is_file():
            record = json.loads(journal.read_text())
            if record["mapping"] != CAMPAIGNS:
                raise ValueError("Migration mapping changed")
        else:
            sources = [g for g in CAMPAIGNS if (root / g).is_dir()]
            files = [p for g in sources for p in (root / g).rglob("*") if p.is_file()]
            if any(p.is_symlink() for g in sources for p in [root/g, *(root/g).rglob("*")]):
                raise ValueError("Migration source contains a symlink; inspect its ownership")
            record = dict(mapping=CAMPAIGNS, sources=sources, files={})
            if not apply:
                reports.append(dict(root=str(root), directories=len(sources), files=len(files),
                                    bytes=sum(p.stat().st_size for p in files), state="preview"))
                continue
            with ThreadPoolExecutor(max_workers=workers) as pool:
                record["files"] = dict(zip((p.relative_to(root).as_posix() for p in files), pool.map(_hash, files)))
            _journal(journal, record)
        if not apply:
            reports.append(dict(root=str(root), state="interrupted; apply resumes journal"))
            continue
        stage.mkdir(exist_ok=True)
        # Stage every original first, before any destination can occupy an old name.
        staged = set(record.get("staged", []))
        for group in record["sources"]:
            if group in staged:
                continue
            origin, temporary = root/group, stage/group
            if temporary.exists() and not origin.exists():
                pass  # rename completed before journal publication
            elif origin.is_dir() and not temporary.exists():
                origin.rename(temporary)
            else:
                raise RuntimeError(f"Ambiguous migration stage for {group}")
            staged.add(group); record["staged"] = sorted(staged)
            _journal(journal, record)
        for group in record["sources"]:
            temporary, destination = stage/group, root/CAMPAIGNS[group]
            if temporary.exists() and not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary.rename(destination)
            elif temporary.exists() or not destination.is_dir():
                raise RuntimeError(f"Ambiguous migration destination for {group}")
        pairs = []
        for name, digest in record["files"].items():
            group, relative = name.split("/", 1)
            pairs.append((root/CAMPAIGNS[group]/relative, digest))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for (path, expected), actual in zip(pairs, pool.map(_hash, (p for p, _ in pairs))):
                if actual != expected:
                    raise RuntimeError(f"Migration content mismatch: {path}")
        stage.rmdir()  # Only the now-empty, owned staging directory; never recursive cleanup.
        reports.append(dict(root=str(root), files=len(pairs), state="hash verified"))
    if apply:
        # Readers switch only after all supplied storage tiers verified successfully.
        for root in roots:
            if (root / MARKER).is_file():
                continue  # Already validated above; a completed migration is a true no-op.
            for group, study in manifest()["studies"].items():
                directory = root/group
                directory.mkdir(exist_ok=True)
                _journal(directory/"STUDY.json", study)
            _journal(root/MARKER, manifest())
    return reports


def _journal(path, record):
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.apply:
        from src.utils.clean_run import check_cluster_writers
        check_cluster_writers()
    print(json.dumps(migrate(args.roots, apply=args.apply, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
