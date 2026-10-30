"""Recovery requests must cross a non-exec container wrapper without killing it."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("warning", ["USR1", "TERM"])
def test_nonexec_wrapper_preserves_native_checkpoint_exit(tmp_path, warning):
    bash = shutil.which("bash") or str(Path(os.environ.get("LOCALAPPDATA", "")) /
                                       "Programs/Git/bin/bash.exe")
    if not Path(bash).is_file():
        pytest.skip("Bash required")
    wrapper = tmp_path / "nonexec wrapper.sh"
    native = tmp_path / "native trainer.sh"
    wrapper.write_text('bash "$1"\n', encoding="utf-8")
    native.write_text('''touch "$READY"
for ((i=0; i<80; i++)); do
    if [[ -n "${CREDITPFN_STOP_REQUEST_FILE:-}" && -f "$CREDITPFN_STOP_REQUEST_FILE" ]]; then
        echo checkpoint_requested
        exit 75
    fi
    sleep 0.025
done
exit 42
''', encoding="utf-8")
    cache = tmp_path / "owned cache"
    cache.mkdir()
    keep = cache / "unrelated"
    keep.write_text("preserve", encoding="utf-8")
    script = '''set -euo pipefail
source "$1"
python() { bash "$WRAPPER" "$NATIVE"; }
sent=0
wait() {
    for ((i=0; i<80 && ! sent; i++)); do
        if [[ -f "$READY" ]]; then
            sent=1
            kill "-$WARNING" "$$"
        else sleep 0.025; fi
    done
    builtin wait "$@"
}
if run_training_python unused; then rc=0; else rc=$?; fi
echo "child_exit=$rc"
[[ "$rc" == 75 ]]
[[ -z "$(trap -p USR1 TERM)" ]]
'''
    result = subprocess.run([bash, "--noprofile", "--norc", "-c", script, "test",
                             Path("scripts/slurm/_run_train.sh").resolve().as_posix()],
        env=dict(os.environ, WRAPPER=wrapper.as_posix(), NATIVE=native.as_posix(),
                 READY=(tmp_path / "ready").as_posix(), WARNING=warning,
                 CREDITPFN_OWNED_CACHE=cache.as_posix()),
        text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checkpoint_requested" in result.stdout and "child_exit=75" in result.stdout
    assert list(cache.iterdir()) == [keep]


def test_native_stop_flag_and_owned_marker_are_both_supported(tmp_path, monkeypatch):
    from src.train.recovery import stop_requested

    monkeypatch.delenv("CREDITPFN_STOP_REQUESTED", raising=False)
    monkeypatch.delenv("CREDITPFN_STOP_REQUEST_FILE", raising=False)
    assert not stop_requested()
    marker = tmp_path / "requested"
    monkeypatch.setenv("CREDITPFN_STOP_REQUEST_FILE", str(marker))
    assert not stop_requested()
    marker.mkdir()
    assert not stop_requested()
    marker.rmdir()
    marker.touch()
    assert stop_requested()
    marker.unlink()
    assert not stop_requested()
    monkeypatch.setenv("CREDITPFN_STOP_REQUESTED", "1")
    assert stop_requested()
