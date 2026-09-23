"""Copy processed tables and original weights to an immutable site-local scratch cache."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path

from omegaconf import OmegaConf

from src.utils.experiment import digest_json, file_digest
from src.utils.paths import resolve_staging_path


def stage(destination: Path, *, write=False, source: Path | None = None) -> dict:
    from src.data.preprocessing import DATASET_METADATA
    source = source or resolve_staging_path(".")
    cfg = OmegaConf.load("config/train.yaml")
    names = [f"data/processed/{m['track']}/{did}.sanitized.csv" for did, m in DATASET_METADATA.items()]
    names += list(cfg.tunable.classifier_base_paths) + list(cfg.tunable.regressor_base_paths)
    names = sorted(set(names))
    missing = [n for n in names if not (source / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} registered input files are missing; prepare the source corpus/weights first")
    report = {"files": len(names), "bytes": sum((source / n).stat().st_size for n in names),
              "source": str(source), "destination": str(destination), "written": write}
    if not write:
        return report
    inventory = {n: {"sha256": file_digest(source / n), "size": (source / n).stat().st_size} for n in names}
    key = digest_json(inventory)
    root = destination / "inputs" / key[:20]
    for name, metadata in inventory.items():
        target = root / name
        if target.exists():
            if file_digest(target) != metadata["sha256"]:
                raise RuntimeError("An immutable input cache was modified; use a fresh destination")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name("." + target.name + "." + uuid.uuid4().hex)
        try:
            shutil.copyfile(source / name, pending)
            if file_digest(pending) != metadata["sha256"]:
                raise RuntimeError("Input changed during transfer")
            os.replace(pending, target)
        finally:
            pending.unlink(missing_ok=True)
    from src.utils.atomic import write_json
    write_json(root / "inventory.json", {"sha256": key, "files": inventory})
    pointer = destination / "ACTIVE_INPUTS.json"
    write_json(pointer, {"root": str(root.resolve()), "sha256": key})
    report.update(root=str(root), pointer=str(pointer))
    return report


def resolve(pointer: Path) -> Path:
    data = json.loads(pointer.read_text(encoding="utf-8"))
    root = Path(data["root"])
    if not root.resolve().is_relative_to((pointer.parent / "inputs").resolve()):
        raise ValueError("Scratch cache pointer escapes its input-cache directory")
    inventory = json.loads((root / "inventory.json").read_text(encoding="utf-8"))
    if inventory["sha256"] != data["sha256"] or digest_json(inventory["files"]) != data["sha256"]:
        raise RuntimeError("Scratch input inventory failed validation")
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--destination", type=Path)
    mode.add_argument("--resolve", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.resolve:
        print(resolve(args.resolve))
    else:
        print(json.dumps(stage(args.destination, write=args.write, source=args.source), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
