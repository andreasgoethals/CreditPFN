"""Persist resolved configurations and storage/job provenance.

Training phase/grid loading lives in src.train.config; src.eval.config composes
that training phase with evaluation settings. Data preprocessing reads config/data.yaml.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.utils.paths import describe, ensure, manifests_dir

def dump_resolved(cfg, task_name: str, *, extra: dict | None = None) -> Path:
    """Write the fully resolved config a run used to `output CreditPFN/<experiment>/manifests/resolved/`.

    Called once per entry point, right after logging is set up. The YAML in `config/` may
    have been edited — or overridden on the command line — since, so this copy is the only
    reliable answer to "what produced this result?". Six months later that question is asked
    about a number, not about a commit.

    Includes the resolved storage roots (`paths.describe()`) and the SLURM identifiers,
    because a result that cannot be located is as good as lost: run-4's checkpoints were in
    two different places depending on whether staging was writable from the node.
    """
    from omegaconf import OmegaConf

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slurm = {k: v for k, v in os.environ.items() if k.startswith("SLURM_JOB")
             or k in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME")}
    payload = {
        "task": task_name,
        "written_utc": stamp,
        "paths": describe(),
        "slurm": slurm,
        "config": OmegaConf.to_container(cfg, resolve=True),
        **(extra or {}),
    }
    suffix = f"_a{slurm['SLURM_ARRAY_TASK_ID']}" if "SLURM_ARRAY_TASK_ID" in slurm else ""
    path = ensure(manifests_dir() / "resolved" / f"{task_name}_{stamp}{suffix}_{uuid.uuid4().hex[:8]}.json")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
