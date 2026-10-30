"""Public labels cannot change saved scientific identities or job/output ownership."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from src.utils import study_layout as layout

ROOT = Path(__file__).resolve().parents[1]


def bash():
    binary = shutil.which("bash") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Git/bin/bash.exe")
    if not Path(binary).is_file():
        pytest.skip("Bash is required for real launcher boundary tests")
    return binary


def test_every_public_shell_identity_matches_the_authority():
    script = "source scripts/slurm/_study_map.sh\n" + "\n".join(
        f"study_campaign experiment{n}" for n in range(11))
    result = subprocess.run([bash(), "-c", script], cwd=ROOT, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [layout.campaign_for_study(n) for n in range(11)]
    script = "source scripts/slurm/_study_map.sh\n" + "\n".join(
        f"study_config_group experiment{n}" for n in range(9))
    result = subprocess.run([bash(), "-c", script], cwd=ROOT, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [Path(layout.config_folder(n)).name for n in range(9)]


@pytest.mark.parametrize("public", [1,2,5,6,7,8,9,10])
def test_public_config_and_real_job_logging_preserve_recorded_owner(tmp_path, public):
    record = layout.campaign_for_study(public)
    configs = sorted((ROOT / f"config/experiment{public}").glob("*.yaml"))
    assert configs
    root = tmp_path / "output CreditPFN"
    root.mkdir()
    (root / layout.MARKER).write_text(json.dumps(layout.manifest()), encoding="utf-8")
    for index, path in enumerate(configs):
        cfg = OmegaConf.load(path)
        owner = cfg.experiment if isinstance(cfg.experiment, str) else cfg.experiment.output_group
        assert owner == record
        if "checkpoint" in cfg:
            assert record in cfg.checkpoint.trained_dir
        env = dict(os.environ, VSC_DATA=str(tmp_path), CREDITPFN_OUTPUT_ROOT=str(tmp_path),
                   CREDITPFN_CONFIG=str(path), CREDITPFN_EXPERIMENT="experiment0", SLURM_JOB_ID=str(9000+index))
        for name in ("CREDITPFN_LEGACY_ID", "CREDITPFN_LORA_WORKFLOW", "CREDITPFN_OWNED_CACHE"):
            env.pop(name,None)
        subprocess.run([bash(),"-c",'source scripts/slurm/_job_log.sh; start_job_log contract; echo "OWNER=$CREDITPFN_EXPERIMENT"'],
                       cwd=ROOT,env=env,check=True)
        log=root/layout.CAMPAIGNS[record]/"logs"/f"contract_{9000+index}_r0.log"
        assert f"OWNER={record}" in log.read_text(encoding="utf-8")
        assert "END exit_code=0" in log.read_text(encoding="utf-8")


def test_layout_changes_invalidate_both_numerical_source_identities(tmp_path):
    from src.utils.experiment import code_identity
    config = tmp_path / "config/study.yaml"
    config.parent.mkdir()
    config.write_text("schema: 2\n",encoding="utf-8")
    before={stage:code_identity(tmp_path,stage=stage) for stage in ("train","eval")}
    config.write_text("schema: 3\n",encoding="utf-8")
    assert all(code_identity(tmp_path,stage=stage)!=before[stage] for stage in before)


@pytest.mark.parametrize("exit_code", [0,7])
def test_real_job_exit_removes_only_successful_owned_cache(tmp_path,exit_code):
    from src.utils.job_cache import MARKER
    import getpass
    cache=tmp_path/"scratch/CreditPFN/job_cache/test/9000-r0"
    cache.mkdir(parents=True)
    (cache/MARKER).write_text(json.dumps(dict(schema=1,job="9000",restart="0",cluster="mindwell",user=getpass.getuser())),encoding="utf-8")
    (cache/"nv").mkdir();(cache/"nv/kernel").write_bytes(b"cache")
    peer=cache.parent/"9001-r0";peer.mkdir();(peer/"keep").write_bytes(b"peer")
    env=dict(os.environ,VSC_DATA=str(tmp_path),CREDITPFN_OUTPUT_ROOT=str(tmp_path),CREDITPFN_EXPERIMENT="experiment0",
             SLURM_JOB_ID="9000",SLURM_RESTART_COUNT="0",SLURM_CLUSTER_NAME="mindwell",VSC_SCRATCH_NODE=str(tmp_path/"scratch"),
             CREDITPFN_OWNED_CACHE=str(cache),CREDITPFN_SYSTEM_PYTHON=sys.executable)
    for name in ("CREDITPFN_CONFIG","CREDITPFN_LEGACY_ID","CREDITPFN_LORA_WORKFLOW"):
        env.pop(name,None)
    result=subprocess.run([bash(),"-c",f"source scripts/slurm/_job_log.sh; start_job_log cachetest; exit {exit_code}"],cwd=ROOT,env=env)
    assert result.returncode==exit_code
    assert cache.exists()==bool(exit_code)
    assert (peer/"keep").read_bytes()==b"peer"
    log=tmp_path/"output CreditPFN/experiment0/logs/cachetest_9000_r0.log"
    assert f"END exit_code={exit_code}" in log.read_text(encoding="utf-8")


def test_notebook_sources_have_no_encoding_corruption():
    # Catch accidental Windows default-decoding before it reaches plot titles.
    artifacts = ("\u00c2\u00b7", "\u00c2\u00b2", "\u00e2\u20ac", "\ufffd")
    for path in (ROOT / "notebooks").rglob("*.ipynb"):
        nb=json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            assert not any(bad in "".join(cell["source"]) for bad in artifacts), path
