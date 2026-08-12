"""Reading `config/`. EMPTY ON PURPOSE — this project fills it in.

How a project reads its configuration is project-specific: what the knobs are, whether a file
describes one run or a sweep, whether anything is validated. The template does not guess.

What the template does ask: whatever you build here, a run should write the **fully resolved**
configuration it used into `output/manifests/`. The YAML on disk may have been edited since, so
that copy is the only reliable answer to "what produced this result?".
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.utils.paths import describe, ensure, manifests_dir

#: What this project fills in is only the template's one ask. The knobs themselves are read
#: with OmegaConf at each entry point (`scripts/{data,train,eval}_pipeline.py`), because a
#: CreditPFN config describes a whole grid and the CLI can override any leaf — so there is no
#: single "load" worth centralising, but there IS one snapshot worth writing.


def dump_resolved(cfg, task_name: str, *, extra: dict | None = None) -> Path:
    """Write the fully resolved config a run used to `output/manifests/resolved/`.

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
    path = ensure(manifests_dir() / "resolved" / f"{task_name}_{stamp}{suffix}.json")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
