"""Stage a sealed, relocatable site-wrapper runtime on the compute cluster's native scratch."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import uuid

from src.utils.runtime_container import sha256


def stage(runtime, destination):
    import fcntl  # Native Linux job setup only; no third-party dependency/bootstrap install.
    runtime, destination = Path(runtime).resolve(strict=True), Path(destination).resolve()
    if runtime == destination or runtime in destination.parents or destination in runtime.parents:
        raise ValueError("Runtime cache must be separate from persistent source")
    manifest = json.loads((runtime/"runtime.json").read_text())
    if not manifest.get("images") or not manifest.get("symlinks"):
        raise ValueError("Runtime image/symlink inventory is missing")
    digest = sha256(runtime/"runtime.json")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with (destination.parent/(destination.name+".lock")).open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=destination/"CACHE.json"
        if destination.exists():
            saved=json.loads(receipt.read_text())
            if saved["manifest_sha256"]!=digest:raise RuntimeError("Runtime cache belongs to another image")
            for name,record in saved["files"].items():
                stat=(destination/name).stat()
                if [stat.st_size,stat.st_mtime_ns]!=record:raise RuntimeError("Immutable runtime cache changed")
            for name,target in manifest["symlinks"].items():
                if os.readlink(destination/name)!=target:raise RuntimeError("Runtime cache symlink changed")
            return destination
        pending=destination.parent/("."+destination.name+"."+uuid.uuid4().hex)
        pending.mkdir()
        # Keep failed owned work for diagnosis; never erase unknown scratch files.
        for name,expected in manifest["files"].items():
            source,target=runtime/name,pending/name
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Runtime artifact escaped its directory")
            if sha256(source)!=expected:raise RuntimeError("Persistent runtime changed")
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            if sha256(target)!=expected:raise RuntimeError("Runtime staging checksum failed")
        for name,target in manifest["symlinks"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Runtime link escaped its directory")
            if os.readlink(runtime/name)!=target:raise RuntimeError("Persistent runtime symlink changed")
            link=pending/name;link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(target)
        shutil.copy2(runtime/"runtime.json",pending/"runtime.json")
        files={name:[(pending/name).stat().st_size,(pending/name).stat().st_mtime_ns] for name in manifest["files"]}
        (pending/"CACHE.json").write_text(json.dumps(dict(source=str(runtime),manifest_sha256=digest,files=files)))
        pending.rename(destination)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime",type=Path);parser.add_argument("destination",type=Path)
    args=parser.parse_args();print(stage(args.runtime,args.destination))


if __name__=="__main__":main()
