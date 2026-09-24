"""Bounded numeric diagnostics. No full-weight snapshots or per-step console dumps."""
from __future__ import annotations

import csv
import logging
import os
import subprocess
import threading
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

import torch

_ACTIVE = ContextVar("creditpfn_resource_monitor", default=None)


def progress(updates: int, rows: int, phase="training"):
    monitor = _ACTIVE.get()
    if monitor is not None:
        monitor.position = (int(updates), int(rows), phase)


class ResourceMonitor:
    """Sample the allocated GPU and process tree, once per interval, in one trial file.

    GPU utilization/power are device-level samples, not exact kernel time or energy.
    Missing counters have an explicit status, never a fabricated zero.
    """
    def __init__(self, path: Path, *, enabled=True, interval=20):
        self.path, self.enabled = Path(path), bool(enabled)
        self.interval = max(5., float(interval))
        self.position = (0, 0, "startup")
        self.stop = threading.Event()
        self.segment = uuid.uuid4().hex[:12]
        self.gpu = None
        self.process = None

    def __enter__(self):
        self.token = _ACTIVE.set(self)
        if not self.enabled:
            return self
        try:
            import psutil
            self.process = psutil.Process()
            self.process.cpu_percent()
        except ImportError:
            pass
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            # A UUID avoids confusing CUDA_VISIBLE_DEVICES with nvidia-smi's physical index.
            self.gpu = str(getattr(props, "uuid", "")) or None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        update, rows, phase = self.position
        record = dict(utc_timestamp=time.time(), segment=self.segment, job_id=os.environ.get("SLURM_JOB_ID", "local"),
                      successful_updates=update, processed_rows=rows, phase=phase,
                      process_tree_rss_bytes="", process_cpu_percent="", gpu_uuid=self.gpu or "",
                      gpu_utilization_percent="", memory_utilization_percent="", device_memory_used_mib="",
                      power_watts="", temperature_c="", gpu_status="unavailable")
        if self.process is not None:
            try:
                record["process_tree_rss_bytes"] = sum(p.memory_info().rss for p in
                    [self.process, *self.process.children(recursive=True)] if p.is_running())
                record["process_cpu_percent"] = self.process.cpu_percent()
            except Exception:
                pass  # Workers may exit between enumeration and sampling.
        if self.gpu:
            try:
                result = subprocess.run(["nvidia-smi", "-i", self.gpu,
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
                values = result.stdout.strip().split(",")
                for key, value in zip(("gpu_utilization_percent", "memory_utilization_percent",
                    "device_memory_used_mib", "power_watts", "temperature_c"), values, strict=True):
                    try:
                        record[key] = float(value)
                    except ValueError:
                        record[key] = ""
                record["gpu_status"] = "sampled"
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                record["gpu_status"] = type(exc).__name__
        return record

    def _run(self):
        try:
            empty = not self.path.exists() or not self.path.stat().st_size
            with self.path.open("a", newline="", encoding="utf-8") as stream:
                writer = None
                while True:
                    row = self.sample()
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        if empty:
                            writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    if self.stop.wait(self.interval):
                        break
        except Exception:
            logging.getLogger(__name__).exception("Resource telemetry failed; numeric training records remain separate")

    def __exit__(self, *exc):
        self.stop.set()
        if self.enabled:
            self.thread.join(timeout=5)
        _ACTIVE.reset(self.token)


@torch.no_grad()
def parameter_statistics(model, anchor: dict, optimizer=None) -> list[dict]:
    """All parameter tensors at fixed milestones; exact norms, bounded sampled quantiles."""
    records = []
    for name, parameter in model.named_parameters():
        value = parameter.detach().float()
        flat = value.reshape(-1)
        if not flat.numel():
            continue
        sampled = flat[::max(1, (flat.numel()+4095)//4096)]
        quantiles = torch.quantile(sampled, torch.tensor([.05,.5,.95],device=value.device)).cpu().tolist()
        record = dict(parameter=name, stage=name.split(".")[0], trainable=parameter.requires_grad,
            elements=parameter.numel(), norm=float(value.norm()), mean=float(value.mean()),
            std=float(value.std(unbiased=False)), abs_max=float(value.abs().max()),
            sampled_q05=quantiles[0], sampled_q50=quantiles[1], sampled_q95=quantiles[2],
            anchored=name in anchor, absolute_change=float("nan"),
            relative_change=float("nan"), cosine_to_initial=float("nan"),
            adam_first_moment_norm=float("nan"), adam_second_moment_norm=float("nan"))
        if name in anchor:
            initial = anchor[name].to(device=value.device, dtype=torch.float32)
            norm0 = float(initial.norm())
            record["absolute_change"] = float((value-initial).norm())
            record["relative_change"] = record["absolute_change"]/norm0 if norm0 else float("nan")
            denom = norm0*record["norm"]
            record["cosine_to_initial"] = float((value*initial).sum())/denom if denom else float("nan")
        if optimizer is not None:
            state = optimizer.state.get(parameter, {})
            for source,target in (("exp_avg","adam_first_moment_norm"),("exp_avg_sq","adam_second_moment_norm")):
                if source in state:
                    record[target] = float(state[source].float().norm())
        records.append(record)
    return records
