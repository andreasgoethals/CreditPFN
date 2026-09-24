"""Merge the old output directory into output CreditPFN on both persistent tiers.

Stop CreditPFN writers first. Preview by default; --apply moves files without
overwriting different contents. Identical duplicates are removed only after their
checksums match. Data and checkpoints are outside the migration scope.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.experiment import file_digest
from src.utils.paths import OUTPUT_DIR_NAME, get_roots


def migrate(roots: list[Path], *, apply: bool = False) -> dict:
    moves, duplicates, old_roots, directories = [], [], [], []
    for root in sorted({p.resolve() for p in roots}):
        old, new = root / "output", root / OUTPUT_DIR_NAME
        if old.is_symlink() or new.is_symlink():
            raise ValueError("Output roots must be real directories, not symlinks")
        if not old.exists():
            continue
        old_roots.append(old)
        for source in sorted(old.rglob("*")):
            target = new / source.relative_to(old)
            if source.is_symlink() or not target.resolve().is_relative_to(new.resolve()):
                raise ValueError(f"Refusing linked/escaping output path: {source}")
            if source.is_dir():
                if target.exists() and not target.is_dir():
                    raise FileExistsError(f"Directory/file collision: {target}")
                directories.append((source, target))
            elif target.exists():
                if not target.is_file() or file_digest(source) != file_digest(target):
                    raise FileExistsError(f"Different files share a destination; nothing moved: {target}")
                duplicates.append((source, target))
            else:
                moves.append((source, target))
    report = {"apply": apply, "files_to_move": len(moves), "identical_duplicates": len(duplicates),
              "old_roots": [str(p) for p in old_roots], "new_directory": OUTPUT_DIR_NAME}
    if not apply:
        return report
    # Validate all roots/collisions before the first mutation. Each rename remains
    # on its own storage tier; interrupted migrations are safe to rerun.
    for source in old_roots:
        source.with_name(OUTPUT_DIR_NAME).mkdir(exist_ok=True)
    for _, target in directories:
        target.mkdir(parents=True, exist_ok=True)
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"A writer created {target}; stop writers before migrating")
        source.rename(target)
    for source, target in duplicates:
        if file_digest(source) != file_digest(target):
            raise RuntimeError("Output changed during migration; stop writers before retrying")
        source.unlink()
    for source, _ in sorted(directories, key=lambda pair: len(pair[0].parts), reverse=True):
        source.rmdir()
    for source in old_roots:
        source.rmdir()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    roots = get_roots()
    print(json.dumps(migrate([roots["output_root"], roots["staging_root"]], apply=args.apply), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
