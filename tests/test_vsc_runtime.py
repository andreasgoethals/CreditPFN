"""Operational regressions, using fake Slurm allocations and temporary storage."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS
import shlex

import numpy as np
import pytest


def test_batch_headers_follow_leuven_submission_requirements():
    """Bash syntax checks alone cannot catch missing Slurm/site options."""
    for path in Path("scripts/slurm").glob("*.slurm"):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "#!/bin/bash -l", path
        directives = []
        for line in lines[1:]:
            if line.strip() and not line.startswith("#"):
                break
            if line.startswith("#SBATCH "):
                directives.extend(shlex.split(line[len("#SBATCH "):]))
        opts = dict(item.split("=", 1) for item in directives if "=" in item)
        for key in ("--clusters", "--partition", "--account", "--time", "--mem"):
            assert opts.get(key), (path, key)
        if opts["--partition"].startswith("gpu"):
            assert opts.get("--gpus-per-node") == "1", path
            cores, memory = {
                ("mindwell", "gpu_b200"): (24, 194400),
                ("wice", "gpu_a100"): (18, 126000),
                ("wice", "gpu_h100"): (16, 187200),
            }[opts["--clusters"], opts["--partition"]]
            assert int(opts["--cpus-per-task"]) <= cores, path
            assert int(opts["--mem"].removesuffix("G")) * 1024 <= memory, path


@pytest.mark.parametrize("requested,expected", [(None, 2), (-1, 2), (0, 2), (9, 2), (1, 1)])
def test_boosting_threads_obey_the_task_allocation(monkeypatch, requested, expected):
    from src.model.boosting import XGBoostModel, CatBoostModel
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    for cls, key in ((XGBoostModel, "n_jobs"), (CatBoostModel, "thread_count")):
        params = {} if requested is None else {key: requested}
        model = cls(task_type="classification", params=params)
        assert model._params[key] == expected


def test_catboost_pool_and_prediction_do_not_default_back_to_all_cores(monkeypatch):
    from src.model.boosting import CatBoostModel
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    calls = []
    def record(*args, **kwargs):
        calls.append(kwargs["thread_count"])
        return np.ones((2, 2))
    monkeypatch.setitem(sys.modules, "catboost", NS(Pool=record))
    model = CatBoostModel(task_type="classification")
    model._model = NS(predict=record, predict_proba=record)
    model.predict(np.ones((2, 2)))
    model.predict_proba(np.ones((2, 2)))
    assert calls == [2, 2, 2, 2]


def test_training_native_pools_leave_room_for_workers(monkeypatch):
    from src.utils.cpu import configure_training_threads
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    previous = torch.get_num_threads()
    try:
        with threadpool_limits(limits=None):
            assert configure_training_threads(workers=1) == 1
            assert torch.get_num_threads() == 1
            assert all(p["num_threads"] <= 1 for p in threadpool_info())
    finally:
        torch.set_num_threads(previous)


def test_worker_budget_respects_a_narrower_cpu_affinity(monkeypatch):
    from src.train.loop import _resolve_dataloader_workers
    from src.utils.cpu import allocated_cpus
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "24")
    monkeypatch.delenv("CREDITPFN_DATALOADER_WORKERS", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {2, 3}, raising=False)
    assert allocated_cpus() == 2
    assert _resolve_dataloader_workers(NS(train=NS(dataloader_workers=4))) == 1


@pytest.mark.parametrize("cluster,site", [("mindwell", "gpfs"), ("wice", "lustre")])
def test_job_setup_replaces_inherited_threads_and_uses_cluster_local_inputs(tmp_path, cluster, site):
    bash = shutil.which("bash") or str(Path(os.environ.get("LOCALAPPDATA", "")) /
                                       "Programs/Git/bin/bash.exe")
    if not Path(bash).is_file():
        pytest.skip("Bash required")
    fake = tmp_path / "conda"
    (fake / "bin").mkdir(parents=True)
    hook = fake / "etc/profile.d/conda.sh"
    hook.parent.mkdir(parents=True)
    python = fake / "bin/python"
    python.write_text('''#!/usr/bin/env bash
if [[ "$*" == *--resolve* ]]; then echo "${@: -1}"; fi
''', encoding="utf-8", newline="\n")
    python.chmod(0o755)
    activator = Path("scripts/slurm/_activate_env.sh").resolve()
    # Source our fake Conda hook, so no installed environment is activated.
    script = '''set -euo pipefail
FAKE_PREFIX="$(cd "$FAKE_PREFIX" && pwd)"
source "$2"
source "$1"
printf '%s\\n' "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$CREDITPFN_DATA_ROOT"
'''
    env = dict(os.environ, FAKE_PREFIX=fake.as_posix(), CONDA_EXE="", VIRTUAL_ENV="",
        VSC_DATA=tmp_path.as_posix(), CREDITPFN_CACHE_ROOT=(tmp_path / "cache").as_posix(),
        CREDITPFN_USE_SCRATCH="1", SLURM_CPUS_PER_TASK="2", SLURM_CLUSTER_NAME=cluster,
        OMP_NUM_THREADS="96", MKL_NUM_THREADS="96", OPENBLAS_NUM_THREADS="96",
        VSC_SCRATCH_GPFS1=(tmp_path / "gpfs").as_posix(),
        VSC_SCRATCH_LUSTRE1=(tmp_path / "lustre").as_posix(), CREDITPFN_INPUT_POINTER="")
    # The activator finds conda on PATH and requests its hook. Return the hook
    # from the fake function, while subsequent activation sets the fake prefix.
    hook.write_text('''conda() {
    if [[ "$*" == 'shell.bash hook' ]]; then echo 'conda() { export CONDA_PREFIX="$FAKE_PREFIX"; }';
    else export CONDA_PREFIX="$FAKE_PREFIX"; fi
}
''', encoding="utf-8", newline="\n")
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", script, "test",
        activator.as_posix(), hook.as_posix()], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-4:] == ["2", "2", "2",
        (tmp_path / site / "CreditPFN/ACTIVE_INPUTS.json").as_posix()]


def test_diagnostics_use_gpfs_and_publish_even_on_graceful_interruption(tmp_path, monkeypatch):
    from src.utils.training_files import TrainingFiles
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(tmp_path / "gpfs"))
    destination = tmp_path / "project/output CreditPFN/experiment0/training/pd"
    destination.mkdir(parents=True)
    (destination / "trial.resources.csv").write_text("old sample\n")
    files = TrainingFiles(destination, "trial")
    with pytest.raises(RuntimeError, match="interrupted"):
        with files:
            (files.directory / "trial.csv").write_text("complete epoch\n")
            with (files.directory / "trial.resources.csv").open("a") as stream:
                stream.write("new sample\n")
            assert not (destination / "trial.csv").exists()
            assert files.directory.is_relative_to(tmp_path / "gpfs")
            raise RuntimeError("interrupted")
    assert (destination / "trial.csv").read_text() == "complete epoch\n"
    assert (destination / "trial.resources.csv").read_text() == "old sample\nnew sample\n"
    assert not files.directory.exists()


def test_failed_publication_keeps_working_diagnostics_and_previous_file(tmp_path, monkeypatch):
    import src.utils.training_files as module
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(tmp_path / "gpfs"))
    destination = tmp_path / "project"
    destination.mkdir()
    (destination / "trial.csv").write_text("previous\n")
    files = module.TrainingFiles(destination, "trial")
    def fail(*args):
        raise OSError("synthetic publication failure")
    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="publication failure"):
        with files:
            (files.directory / "trial.csv").write_text("new\n")
    assert (files.directory / "trial.csv").read_text() == "new\n"
    assert (destination / "trial.csv").read_text() == "previous\n"
    assert not list(destination.glob("*.tmp"))


def test_submission_timeout_never_allows_an_automatic_duplicate(tmp_path, monkeypatch):
    import src.utils.submit_bounded as module
    monkeypatch.setitem(sys.modules, "fcntl", NS(LOCK_EX=1, flock=lambda *a: None))
    monkeypatch.setenv("USER", "test-user")
    calls = []
    def run(command, **kwargs):
        assert kwargs.get("timeout") == 45
        calls.append(command[0])
        if command[0] == "sbatch":
            raise subprocess.TimeoutExpired(command, 45)
        return NS(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(module.subprocess, "run", run)
    pool = tmp_path / "pool.json"
    command = ["sbatch", "--array=0-1%2", "job.slurm"]
    with pytest.raises(subprocess.TimeoutExpired):
        module.submit(command, slots=2, limit=2, cluster="mindwell", pool_path=pool)
    assert json.loads(pool.read_text())["pending_submission"]
    with pytest.raises(RuntimeError, match="Uncertain"):
        module.submit(command, slots=2, limit=2, cluster="mindwell", pool_path=pool)
    assert calls.count("sbatch") == 1


def test_queue_timeout_stops_launcher_before_any_submission(tmp_path):
    bash = shutil.which("bash") or str(Path(os.environ.get("LOCALAPPDATA", "")) /
                                       "Programs/Git/bin/bash.exe")
    if not Path(bash).is_file():
        pytest.skip("Bash required")
    config = tmp_path / "phase config.yaml"
    config.write_text("track: pd\n", encoding="utf-8")
    launcher = Path("scripts/slurm/run_experiment.sh").resolve()
    script = '''python() {
    if [[ "$1" != - ]]; then echo 'UNEXPECTED_SUBMISSION' >&2; return 31; fi
    cat >/dev/null
    printf '%s\\n' pd cpt_null_synthetic 1 2 4 False 'base.ckpt one_sample'
}
timeout() {
    [[ "$1 $2 $3" == '--kill-after=5s 45s squeue' ]] || return 32
    return 124
}
export -f python timeout
bash "$1" "$2"
'''
    env = dict(os.environ, DRY="", STAGES="train", TRIALS_PER_TASK="1", SPLITS="1",
               SPLIT_START="0", SEGMENT_MINUTES="0", GLOBAL_CONCURRENCY="16",
               THROTTLE="4", EVAL_CONCURRENCY="4", USER="test-user")
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", script, "test",
        launcher.as_posix(), config.as_posix()], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 124, result.stderr
    assert "UNEXPECTED_SUBMISSION" not in result.stderr
