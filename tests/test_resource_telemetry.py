"""Exercise device selection, measurement failures and the standalone GPU check."""
import csv
import json
import subprocess
from types import SimpleNamespace

import pytest

from src.train import telemetry

BARE_UUID = "01234567-89ab-cdef-0123-456789abcdef"


def fake_cuda(monkeypatch, identifier=BARE_UUID):
    monkeypatch.setattr(telemetry.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(telemetry.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(telemetry.torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(uuid=identifier))


@pytest.mark.parametrize("identifier", [BARE_UUID, "GPU-" + BARE_UUID, "MIG-" + BARE_UUID])
def test_monitor_selects_allocated_uuid_and_writes_real_counters(tmp_path, monkeypatch, identifier):
    fake_cuda(monkeypatch, identifier)
    # Logical device 0 need not be physical GPU 0. Never substitute a numeric index.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    expected = identifier if identifier.startswith(("GPU-", "MIG-")) else "GPU-" + identifier
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        if command[command.index("-i") + 1] != expected:
            raise subprocess.CalledProcessError(6, command, output="No devices were found")
        return SimpleNamespace(stdout="73, 42, 12000, 190.5, 45\n")

    monkeypatch.setattr(telemetry.subprocess, "run", query)
    path = tmp_path / "trial.resources.csv"
    with telemetry.ResourceMonitor(path):
        pass
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(calls) == len(rows) == 1
    assert rows[0]["gpu_status"] == "sampled"
    assert rows[0]["gpu_uuid"] == expected
    assert float(rows[0]["gpu_utilization_percent"]) == 73
    assert float(rows[0]["device_memory_used_mib"]) == 12000
    assert float(rows[0]["power_watts"]) == 190.5


@pytest.mark.parametrize("missing", [None, ""])
def test_absent_uuid_stays_unavailable(tmp_path, monkeypatch, missing):
    fake_cuda(monkeypatch, missing)
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: pytest.fail("No GPU selector"))
    with telemetry.ResourceMonitor(tmp_path / "trial.resources.csv") as monitor:
        assert monitor.gpu is None


def test_failed_query_keeps_bounded_details_warns_once_and_can_recover(tmp_path, monkeypatch, caplog):
    monitor = telemetry.ResourceMonitor(tmp_path / "unused.csv")
    monitor.gpu = "GPU-" + BARE_UUID

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(6, command, output="No devices were found\n" + "x" * 1000)

    monkeypatch.setattr(telemetry.subprocess, "run", fail)
    first, second = monitor.sample(), monitor.sample()
    assert first["gpu_status"] == second["gpu_status"] == "CalledProcessError"
    assert first["gpu_exit_code"] == 6
    assert first["gpu_error"].startswith("No devices were found ")
    assert len(first["gpu_error"]) <= 384
    assert first["gpu_utilization_percent"] == ""
    assert len([r for r in caplog.records if "GPU resource sampling" in r.message]) == 1
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="0, 0, 0, N/A, 35"))
    recovered = monitor.sample()
    assert recovered["gpu_status"] == "sampled"
    assert recovered["gpu_error"] == recovered["gpu_exit_code"] == ""
    assert recovered["gpu_utilization_percent"] == 0
    assert recovered["power_watts"] == ""
    assert not (tmp_path / "unused.csv").exists()


@pytest.mark.parametrize("values,status", [
    ("N/A, N/A, 120, N/A, N/A", "unsupported"),
    ("NaN, 0, 120, 60, 40", "unsupported"),
    ("30, 20, N/A, 60, 40", "unsupported"),
    ("30, 20, 120, 60", "ValueError"),
    ("30, 20, 120, 60, 40\n30, 20, 120, 60, 40", "ValueError"),
])
def test_unavailable_or_malformed_counters_cannot_pass(tmp_path, monkeypatch, values, status):
    monitor = telemetry.ResourceMonitor(tmp_path / "unused.csv")
    monitor.gpu = "GPU-" + BARE_UUID
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=values))
    assert monitor.sample()["gpu_status"] == status


@pytest.mark.parametrize("error", [FileNotFoundError("missing nvidia-smi"),
    subprocess.TimeoutExpired("nvidia-smi", 3, output=b"device query stalled")])
def test_missing_tool_and_timeout_remain_explicit(tmp_path, monkeypatch, error):
    monitor = telemetry.ResourceMonitor(tmp_path / "unused.csv")
    monitor.gpu = "GPU-" + BARE_UUID

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(telemetry.subprocess, "run", fail)
    row = monitor.sample()
    assert row["gpu_status"] == type(error).__name__
    assert row["gpu_error"] and row["gpu_utilization_percent"] == ""


@pytest.mark.parametrize("available,values,expected", [
    (True, "12, 1, 200, 80, 40", 0),
    (True, "N/A, N/A, N/A, N/A, N/A", 1),
    (False, "", 1),
])
def test_gpu_check_uses_same_sampler_without_inputs_training_or_files(monkeypatch, capsys, available, values, expected):
    from src.utils import preflight
    fake_cuda(monkeypatch)
    monkeypatch.setattr(telemetry.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=values))
    monkeypatch.setattr(preflight, "_resolved_roots", lambda: pytest.fail("GPU check must not read datasets"))
    monkeypatch.setattr(telemetry.ResourceMonitor, "__enter__", lambda _: pytest.fail("No thread or CSV for this check"))
    assert preflight.main(["--gpu-resources"]) == expected
    row = json.loads(capsys.readouterr().out)
    assert row["gpu_status"] == ("sampled" if expected == 0 else "unsupported" if available else "unavailable")

