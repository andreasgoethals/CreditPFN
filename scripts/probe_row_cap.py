#!/usr/bin/env python
"""Measure synthetic forward/backward memory for each base and adaptation mode.

Submit through scripts/slurm/probe_row_cap.slurm. This screen excludes optimizer
state, L2-SP and evaluation: use the real pilot to confirm the operating margin.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.train.capacity import probe_configured_bases  # noqa: E402


def _gpu_is_usable() -> bool:
    """Refuse to probe on a GPU that cannot give a meaningful answer.

    WHY (2026-08-05): run bare on a Genius **login** node, this script found
    the node's *display* GPU (Quadro P6000, sm_61), which
    ``torch.cuda.is_available()`` happily reports as available even though
    (a) the installed PyTorch has no sm_61 kernels and (b) the card is tiny
    and shared. The result was a bare ``CUDA error: out of memory`` while
    merely moving the model onto the device — an error that looks like "the
    row cap is too high" but actually means "wrong machine".

    A capacity probe on the wrong GPU is worse than no probe: its numbers
    would be silently transplanted into config/data.yaml. So fail loudly and
    name the submit command instead.
    """
    import socket
    import torch

    try:
        props = torch.cuda.get_device_properties(0)
    except Exception as exc:                                   # pragma: no cover
        print(f"ERROR: could not query GPU 0 ({type(exc).__name__}: {exc}).")
        return False
    cap = f"sm_{props.major}{props.minor}"
    supported = set(torch.cuda.get_arch_list())
    host = socket.gethostname()
    on_login = "login" in host and not os.environ.get("SLURM_JOB_ID")

    if cap not in supported or on_login:
        print("=" * 78)
        print("REFUSING TO PROBE — this is not a usable capacity-probe GPU.")
        print("=" * 78)
        print(f"  host              : {host}")
        print(f"  slurm job         : {os.environ.get('SLURM_JOB_ID', '<none — interactive>')}")
        print(f"  gpu               : {props.name} ({cap}, "
              f"{props.total_memory / 1e9:.1f} GB)")
        print(f"  pytorch supports  : {' '.join(sorted(supported))}")
        if on_login:
            print("\n  A login node's GPU is a shared DISPLAY device, not a compute GPU.")
        if cap not in supported:
            print(f"\n  This PyTorch build has no {cap} kernels, so anything it did"
                  "\n  report would be meaningless (and it usually fails with a"
                  "\n  misleading 'CUDA error: out of memory' during model .to(device)).")
        print("\n  Submit it as a batch job on a real compute GPU instead:")
        print("      sbatch scripts/slurm/probe_row_cap.slurm")
        print("  (Mindwell gpu_b200, 1 GPU, ~1 h; log lands in "
              "$VSC_DATA/CreditPFN/output/logs/.)")
        print("=" * 78)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", choices=["pd", "lgd", "both"], default="both")
    ap.add_argument("--rows", type=int, nargs="+", help="override the per-base row grids")
    ap.add_argument("--adaptation", choices=["full", "frozen", "both"], default="both")
    args = ap.parse_args()

    import torch
    from omegaconf import OmegaConf
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not _gpu_is_usable():
        return 1
    if device != "cuda":
        print("WARNING: no CUDA device: timing only, no capacity measurement.")
    modes = (False, True) if args.adaptation == "both" else (args.adaptation == "frozen",)
    tracks = ["pd", "lgd"] if args.track == "both" else [args.track]
    probe_configured_bases(OmegaConf.load("config/train.yaml"), tracks, args.rows,
                           device=device, frozen_modes=modes)
    print("Confirm the operating margin with a real pilot before changing any row cap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
