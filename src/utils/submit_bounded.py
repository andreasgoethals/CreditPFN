"""Submit an array into a persistent Slurm concurrency pool shared by all phases/tracks.

Each array reserves lanes and depends on the previous users of those lanes. An
array's afterany dependency completes only when all its tasks finish. One pool
is used per controller; this is not a limit on unrelated projects or direct sbatch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from src.utils.paths import manifests_dir


def reserve(pool: dict, *, slots: int, limit: int, active: set[str]):
    if not 1 <= limit <= slots:
        raise ValueError("Array concurrency must be between one and the pool size")
    if pool and pool["slots"] != slots:
        if any(j in active for j in pool["lanes"]):
            raise ValueError("Cannot resize a concurrency pool with active jobs")
        pool = {}
    lanes = list(pool.get("lanes", [""] * slots))
    cursor = int(pool.get("cursor", 0))
    chosen = [(cursor + i) % slots for i in range(limit)]
    dependencies = sorted({lanes[i] for i in chosen if lanes[i] in active})
    return {"slots": slots, "lanes": lanes, "cursor": (cursor + limit) % slots}, chosen, dependencies


def submit(command: list[str], *, slots: int, limit: int, cluster: str, pool_path: Path | None = None) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cluster) or not command or command[0] != "sbatch":
        raise ValueError("Expected a named Slurm controller and an sbatch command")
    path = pool_path or manifests_dir("general") / "scheduler" / f"pool-{cluster}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Production is Linux. A failed lock must stop submission, never silently overrun the cap.
    import fcntl
    with path.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = subprocess.run(["squeue", "--clusters", cluster, "--user", os.environ["USER"],
                                "--noheader", "--format=%F"], capture_output=True, text=True, check=True)
        active = set(queue.stdout.split())
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if old.get("pending_submission"):
            raise RuntimeError(f"Uncertain earlier submission: reconcile squeue with {path} before retrying")
        pool, chosen, dependencies = reserve(old, slots=slots, limit=limit, active=active)
        cmd = list(command)
        array = next((i for i, x in enumerate(cmd) if x.startswith("--array=")), None)
        if array is None:
            raise ValueError("A bounded submission requires an explicit array")
        cmd[array] = cmd[array].split("%", 1)[0] + f"%{limit}"
        if "--parsable" not in cmd:
            cmd.insert(1, "--parsable")
        if dependencies:
            dep = "afterany:" + ":".join(dependencies)
            existing = next((i for i, x in enumerate(cmd) if x.startswith("--dependency=")), None)
            if existing is None:
                cmd.insert(1, "--dependency=" + dep)
            else:
                cmd[existing] += "," + dep
        from src.utils.atomic import write_json
        # Fail closed if the process dies after sbatch accepted a job but before its id
        # is recorded. Never silently exceed the cap after an uncertain submission.
        pending_pool = dict(pool, pending_submission=cmd)
        write_json(path, pending_pool)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            write_json(path, old)
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        if result.stderr:
            import sys
            print(result.stderr, end="", file=sys.stderr)
        job = result.stdout.strip().split(";", 1)[0]
        if not job.isdigit():
            raise RuntimeError("Submission response is ambiguous; inspect squeue before retrying")
        for i in chosen:
            pool["lanes"][i] = job
        write_json(path, pool)
        return result.stdout.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        print(submit(command, slots=args.slots, limit=args.limit, cluster=args.cluster))
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        import sys
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
