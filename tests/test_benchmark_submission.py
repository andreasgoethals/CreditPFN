"""Combined benchmark routing, without submitting jobs or loading checkpoints."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]


def bash_path():
    path = shutil.which("bash") or str(
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Git/bin/bash.exe"
    )
    if not Path(path).is_file():
        pytest.skip("Bash required")
    return path


def test_combined_array_maps_every_track_partition_and_shard_once():
    result = subprocess.run(
        [bash_path(), "--noprofile", "--norc", "-c", '''
set -euo pipefail
export CREDITPFN_BENCHMARK_SPLITS=4 EVAL_TASKS=16
export CREDITPFN_EXPERIMENT=experiment2 VSC_DATA=/data/example
for ((i=0; i<128; i++)); do
    export SLURM_ARRAY_TASK_ID=$i
    source "$1"
    printf '%s %s %s %s\n' "$TRACK" "$CREDITPFN_SPLIT_INDEX" "$CREDITPFN_EVAL_TASK_INDEX" "$CREDITPFN_CONFIG"
done
''', "test", (REPO / "scripts/slurm/_benchmark_dispatch.sh").as_posix()],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    rows = [line.split() for line in result.stdout.splitlines()]
    expected = [(track, str(partition), str(shard))
                for track in ("pd", "lgd") for partition in range(4) for shard in range(16)]
    assert [tuple(row[:3]) for row in rows] == expected
    assert all(row[3] == f"/data/example/CreditPFN/config/experiment2/{row[0]}.yaml" for row in rows)


@pytest.mark.parametrize("index", [-1, 128])
def test_combined_array_rejects_out_of_range_indices(index):
    result = subprocess.run(
        [bash_path(), "--noprofile", "--norc", "-c", 'set -euo pipefail; source "$1"',
         "test", (REPO / "scripts/slurm/_benchmark_dispatch.sh").as_posix()],
        env=dict(os.environ, CREDITPFN_BENCHMARK_SPLITS="4", EVAL_TASKS="16",
                 SLURM_ARRAY_TASK_ID=str(index)),
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 2
    assert "Invalid benchmark array index" in result.stderr


@pytest.mark.parametrize("metadata", ["4", ""])
def test_benchmark_dry_preview_and_failed_readiness_never_submit(metadata):
    # Shell functions replace external commands in this subprocess only. The real
    # Python readiness check is deliberately not run against research storage.
    result = subprocess.run(
        [bash_path(), "--noprofile", "--norc", "-c", '''
set -euo pipefail
python() { if [[ -n "$TEST_META" ]]; then printf '%s\n' "$TEST_META"; else return 1; fi; }
sbatch() { echo 'UNEXPECTED SUBMISSION' >&2; return 99; }
squeue() { echo 'UNEXPECTED QUEUE QUERY' >&2; return 99; }
export -f python sbatch squeue
DRY=1 bash "$1" experiment2
''', "test", (REPO / "scripts/slurm/run_benchmark.sh").as_posix()],
        env=dict(os.environ, TEST_META=metadata, EVAL_TASKS="16", EVAL_CONCURRENCY="4",
                 GLOBAL_CONCURRENCY="16"),
        capture_output=True, text=True, timeout=15,
    )
    assert "UNEXPECTED" not in result.stderr
    if metadata:
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert len(lines) == 2
        assert "--clusters=mindwell" in lines[0]
        assert "--clusters=wice" in lines[1]
        assert all("--array=0-127%4" in line for line in lines)
        assert "benchmark_foundation.slurm" in lines[0]
        assert "benchmark_classical.slurm" in lines[1]
    else:
        assert result.returncode == 1
        assert "no jobs submitted" in result.stderr
