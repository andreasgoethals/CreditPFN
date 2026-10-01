"""Cross-experiment control reuse leaves numerical identities and donor bytes intact."""
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.eval import cache
from src.eval.config import load_eval_configs


def bash_path():
    path = shutil.which("bash") or str(Path(os.environ.get("LOCALAPPDATA", ""))/"Programs/Git/bin/bash.exe")
    if not Path(path).is_file():
        pytest.skip("Bash required")
    return path


def artifact(path, key, *, fold_count=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(fold_idx=i, status="OK", model_name="untuned", test_dataset_id="test", n_test_rows=2)
            for i in range(fold_count)]
    predictions = [dict(model_name="untuned", test_dataset_id="test", fold_idx=i, row_idx=j)
                   for i in range(fold_count) for j in range(2)]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(dict(key=key, rows=rows, predictions=predictions), stream)


def share(target):
    script = Path(__file__).resolve().parents[1]/"scripts/slurm/_eval_cache.sh"
    return subprocess.run([bash_path(), "--noprofile", "--norc", "-c",
        'set -euo pipefail; source "$1"; share_eval_controls "$2"', "test", script.as_posix(), target.as_posix()],
        capture_output=True, text=True, timeout=20)


def test_published_control_shared_without_copy_or_overwrite(tmp_path):
    root = tmp_path/"output CreditPFN"
    key = "a"*64
    donor = root/"experiment1/evaluation_cache"/(key+".json.gz")
    artifact(donor, key)
    original = donor.read_bytes()
    target = root/"experiment2/evaluation_cache"
    result = share(target)
    assert result.returncode == 0, result.stderr
    linked = target/donor.name
    assert linked.read_bytes() == original
    assert os.path.samefile(donor, linked)
    rows = cache.load(target.parent/"results", key, n_folds=5)
    assert len(rows) == 5
    predictions = cache.load_predictions(target.parent/"results", key)
    assert cache.predictions_complete(rows, predictions)
    assert not cache.predictions_complete(rows, predictions[:-1])
    assert share(target).returncode == 0 and donor.read_bytes() == original
    # Atomic replacement changes the donor's entry, never the imported artifact.
    replacement = donor.with_name(".pending")
    artifact(replacement, key, fold_count=3)
    os.replace(replacement, donor)
    assert linked.read_bytes() == original
    assert share(target).returncode == 0 and linked.read_bytes() == original


def test_incomplete_wrong_key_and_unpublished_controls_are_not_accepted(tmp_path):
    root = tmp_path/"output CreditPFN"
    donor = root/"experiment1/evaluation_cache"
    artifact(donor/("b"*64+".json.gz"), "b"*64, fold_count=4)
    artifact(donor/("c"*64+".json.gz"), "d"*64)
    artifact(donor/".pending.json.gz", "e"*64)
    artifact(root/"experiment0/evaluation_cache"/("f"*64+".json.gz"), "f"*64)
    target = root/"experiment3/evaluation_cache"
    assert share(target).returncode == 0
    for key in ("b"*64, "c"*64, "e"*64, "f"*64):
        assert cache.load(target.parent/"results", key, n_folds=5) is None
    assert sorted(p.name for p in target.iterdir()) == ["b"*64+".json.gz", "c"*64+".json.gz"]


def test_nonresearch_cache_is_untouched(tmp_path):
    target = tmp_path/"output CreditPFN/experiment0/evaluation_cache"
    assert share(target).returncode == 0
    assert not target.exists()


@pytest.mark.parametrize("track", ["pd", "lgd"])
@pytest.mark.parametrize("source", ["tabpfn-untuned", "tabicl-untuned", "baseline"])
def test_research_phase_configs_share_only_identical_evaluation_controls(monkeypatch, track, source):
    monkeypatch.setattr(cache, "file_digest", lambda path: "fixed-input-bytes")
    monkeypatch.setattr(cache, "code_identity", lambda **kwargs: "same-numerical-code")
    monkeypatch.setattr(cache, "environment_versions", lambda **kwargs: {"test": "same-environment"})
    handle = SimpleNamespace(source=source, name=source, base_path=None)
    configs = [load_eval_configs([], [], config_path=f"config/experiment{experiment}/{track}.yaml")[0]
               for experiment in (1, 2, 3)]
    keys = [cache.evaluation_key(handle, "test", track=track, config=OmegaConf.to_container(cfg, resolve=True))
            for cfg in configs]
    assert len(set(keys)) == 1
    configs[-1].seed += 1
    changed = cache.evaluation_key(handle, "test", track=track,
                                   config=OmegaConf.to_container(configs[-1], resolve=True))
    assert changed != keys[0]
