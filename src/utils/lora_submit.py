"""Submit the entire LoRA gate as one same-controller Slurm dependency chain."""
from __future__ import annotations

import argparse
import os
import re
import subprocess

from src.utils.atomic import write_json
from src.utils.lora_checks import folder, validate_id
from src.utils.paths import REPO_ROOT


def submit(identifier):
    """Reserve one output scope and submit tests only; never start production."""
    validate_id(identifier)
    if not os.environ.get("VSC_DATA"):
        raise RuntimeError("Submit LoRA checks from the VSC repository")
    root = folder(identifier)
    root.mkdir(parents=True, exist_ok=False)  # Atomic reservation; no duplicate chain.
    write_json(root / ".scope.json", dict(kind="creditpfn_lora_scope", schema=1, id=identifier), exclusive=True)
    record = dict(schema=1, kind="creditpfn_lora_submission", id=identifier,
                  cluster="mindwell", status="submitting", jobs=[])
    destination = root / "submission.json"
    write_json(destination, record, exclusive=True)
    stages = (
        ("prepare", ["--partition=batch_graniterapids", "--cpus-per-task=8", "--mem=16G",
                     "scripts/slurm/lora_prepare.slurm", identifier]),
        ("smoke", ["--array=0-3%2", "--time=01:00:00", "scripts/slurm/lora_gpu.slurm", "smoke", identifier]),
        ("capacity", ["--array=0-3%2", "--time=03:00:00", "scripts/slurm/lora_gpu.slurm", "capacity", identifier]),
        ("inspect", ["--partition=batch_graniterapids", "--cpus-per-task=8", "--mem=16G",
                     "scripts/slurm/lora_prepare.slurm", identifier, "inspect"]),
    )
    # SBATCH_* variables otherwise override script directives unexpectedly.
    environment = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    try:
        for stage, arguments in stages:
            command = ["sbatch", "--parsable", "--clusters=mindwell", "--account=lp_verbekelab"]
            dependency = record["jobs"][-1]["id"] if record["jobs"] else None
            if dependency:
                command += [f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"]
            command += arguments
            # Record the attempted command before the external side effect. Unknown
            # outcomes are never retried automatically or silently cancelled.
            record["attempting"] = dict(stage=stage, command=command)
            write_json(destination, record)
            result = subprocess.run(command, text=True, capture_output=True, check=True,
                                    timeout=60, env=environment, cwd=REPO_ROOT)
            response = result.stdout.strip()
            match = re.fullmatch(r"([1-9][0-9]*)(?:;mindwell)?", response)
            if not match:
                raise RuntimeError(f"Unrecognized sbatch response for {stage}: {response!r}; inspect queue before retrying")
            job = dict(stage=stage, id=match.group(1), dependency=dependency)
            record["jobs"].append(job)
            record.pop("attempting")
            write_json(destination, record)
            print(f"Submitted {stage}: {job['id']} on mindwell" +
                  (f" (after successful {dependency})" if dependency else ""), flush=True)
    except BaseException as error:
        record.update(status="incomplete_submission", error=type(error).__name__)
        write_json(destination, record)
        print(f"Submission stopped. Preserve {destination}; inspect recorded jobs before retrying.", flush=True)
        raise
    record["status"] = "submitted"
    write_json(destination, record)
    print(f"Entire safety sequence queued for {identifier}. You may close the terminal.", flush=True)
    print("Failed prerequisites prevent downstream tests from running. No production jobs submitted.", flush=True)
    print(f"After successful inspection: CREDITPFN_LORA_GATE={identifier} bash scripts/slurm/run_experiment6.sh train", flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    args = parser.parse_args(argv)
    submit(args.id)


if __name__ == "__main__":
    main()
