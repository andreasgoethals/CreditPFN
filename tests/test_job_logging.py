"""Real `_job_log.sh` runs: the generated config owns the log folder; exit cleans only owned caches."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_eval_control_sharing import bash_path

ROOT = Path(__file__).resolve().parents[1]


def _env(tmp_path, **extra):
    env = dict(os.environ, VSC_DATA=str(tmp_path), CREDITPFN_OUTPUT_ROOT=str(tmp_path),
               CREDITPFN_EXPERIMENT="experiment0", SLURM_JOB_ID="9000", SLURM_RESTART_COUNT="0", **extra)
    for name in ("CREDITPFN_CONFIG", "CREDITPFN_OWNED_CACHE"):
        if name not in extra:
            env.pop(name, None)
    return env


@pytest.mark.parametrize("config", ["E00_pilots/reference_pd.yaml", "E07_exposure/pd_a0.yaml"])
def test_generated_config_selects_its_log_folder(tmp_path, config):
    from omegaconf import OmegaConf
    path = ROOT / "config/experiments" / config
    owner = OmegaConf.load(path).experiment.output_group
    subprocess.run([bash_path(), "-c", 'source scripts/slurm/_job_log.sh; start_job_log contract; echo "OWNER=$CREDITPFN_EXPERIMENT"'],
                   cwd=ROOT, env=_env(tmp_path, CREDITPFN_CONFIG=str(path)), check=True)
    log = (tmp_path / "output CreditPFN" / owner / "logs/contract_9000_r0.log").read_text(encoding="utf-8")
    assert f"OWNER={owner}" in log and "END exit_code=0" in log


@pytest.mark.parametrize("exit_code", [0, 7])
def test_job_exit_removes_only_a_successful_owned_cache(tmp_path, exit_code):
    import getpass

    from src.utils.job_cache import MARKER
    cache = tmp_path / "scratch/CreditPFN/job_cache/test/9000-r0"
    cache.mkdir(parents=True)
    (cache / MARKER).write_text(json.dumps(dict(schema=1, job="9000", restart="0", cluster="mindwell",
                                                user=getpass.getuser())), encoding="utf-8")
    (cache / "nv").mkdir()
    (cache / "nv/kernel").write_bytes(b"cache")
    peer = cache.parent / "9001-r0"
    peer.mkdir()
    (peer / "keep").write_bytes(b"peer")
    env = _env(tmp_path, SLURM_CLUSTER_NAME="mindwell", VSC_SCRATCH_NODE=str(tmp_path / "scratch"),
               CREDITPFN_OWNED_CACHE=str(cache), CREDITPFN_SYSTEM_PYTHON=sys.executable)
    result = subprocess.run([bash_path(), "-c", f"source scripts/slurm/_job_log.sh; start_job_log cachetest; exit {exit_code}"],
                            cwd=ROOT, env=env)
    assert result.returncode == exit_code
    assert cache.exists() == bool(exit_code)
    assert (peer / "keep").read_bytes() == b"peer"
    log = tmp_path / "output CreditPFN/experiment0/logs/cachetest_9000_r0.log"
    assert f"END exit_code={exit_code}" in log.read_text(encoding="utf-8")


def test_notebook_sources_have_no_encoding_corruption():
    # Catch accidental Windows default-decoding before it reaches plot titles.
    artifacts = ("Â·", "Â²", "â€", "�")
    for path in (ROOT / "notebooks").rglob("*.ipynb"):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            assert not any(bad in "".join(cell["source"]) for bad in artifacts), path
