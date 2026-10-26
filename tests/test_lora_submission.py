"""Shell syntax and mock submission checks; never call Slurm or research Python."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]


def bash_path():
    path = shutil.which("bash") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Git/bin/bash.exe")
    if not Path(path).is_file():
        pytest.skip("Bash required")
    return path


@pytest.mark.parametrize("name", ["run_experiment0_lora.sh", "lora_prepare.slurm", "lora_gpu.slurm", "run_experiment8.sh"])
def test_lora_shell_syntax(name):
    result = subprocess.run([bash_path(), "-n", str(REPO / "scripts/slurm" / name)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("stage,expected", [("prepare", "lora_prepare.slurm"), ("smoke", "--array=0-3%2 --time=01:00:00"),
                                           ("capacity", "--array=0-3%2 --time=03:00:00")])
def test_debug_launcher_submits_only_requested_stage(stage, expected):
    result = subprocess.run([bash_path(), "--noprofile", "--norc", "-c", '''
set -euo pipefail
sbatch() { printf 'MOCK %s\n' "$*"; }
export -f sbatch
bash "$1" "$2" testworkflow
''', "test", (REPO / "scripts/slurm/run_experiment0_lora.sh").as_posix(), stage], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("MOCK") == 1 and expected in result.stdout
    assert "train_pd" not in result.stdout and "benchmark" not in result.stdout


def test_debug_launcher_rejects_traversal_before_submitting():
    result = subprocess.run([bash_path(), "--noprofile", "--norc", "-c", '''
sbatch() { echo UNEXPECTED; }
export -f sbatch
bash "$1" smoke ../other
''', "test", (REPO / "scripts/slurm/run_experiment0_lora.sh").as_posix()], capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and "UNEXPECTED" not in result.stdout


def test_failed_gate_stops_production_launcher_before_submitting():
    result = subprocess.run([bash_path(), "--noprofile", "--norc", "-c", '''
python() { return 1; }
sbatch() { echo UNEXPECTED; }
export -f python sbatch
bash "$1" train
''', "test", (REPO / "scripts/slurm/run_experiment8.sh").as_posix()], capture_output=True, text=True, timeout=15)
    assert result.returncode == 1 and "UNEXPECTED" not in result.stdout


def test_gpu_and_prepare_logs_are_scoped_before_log_creation():
    for name in ("lora_prepare.slurm", "lora_gpu.slurm"):
        source = (REPO / "scripts/slurm" / name).read_text()
        assert source.index("export CREDITPFN_LORA_WORKFLOW=") < source.index("start_job_log")
        assert "CREDITPFN_REQUIRE_STAGING=1" in source
