"""One report answering everything we need to know before committing a 3 000-trial campaign.

    python scripts/cluster_report.py                 # everything that needs no GPU
    python scripts/cluster_report.py --gpu           # + the GPU sections (run under sbatch)
    python scripts/cluster_report.py --gpu --probe   # + the row-cap sweep (slow, ~15 min)

WHY THIS EXISTS. Six of the last ten problems in this project were environment or accounting
facts that no test could catch: a probe measuring one ensemble member while training used two,
a login node handing out an unsupported GPU, calibration columns dropped between a dict and a
dataclass, a queue limit nobody had written down. Each cost a run. This collects the facts in
one place so the next campaign starts from measurements instead of assumptions.

Sections, and the decision each one informs:

  1  environment     which python/torch/tabpfn is actually live, and whether a stray venv is
                     shadowing the conda env (that exact bug sent pip installs to the wrong
                     prefix for a week)
  2  slurm limits    MaxSubmitJobsPerUser, MaxWall, partition walltimes — the numbers that
                     decide whether a 12-split submission is accepted or silently truncated
  3  accounting      credits left, and the charge rate of every partition we might use
  4  gpu             which card, how much memory, and whether this torch build has kernels
                     for it (sm_61 login GPUs do not, and fail with a misleading OOM)
  5  precision       BF16/TF32 support and measured matmul throughput, per card, so routing
                     decisions rest on this cluster's numbers
  6  attention       whether the cuDNN fused kernel that caps TabICLv2 at 26k rows can be
                     switched off, and what the alternatives cost
  7  checkpoints     every base the configs reference, present or missing, with parameter
                     counts and the trainable fraction under a frozen backbone
  8  data            processed datasets on disk, rows and features, per track
  9  row caps        measured peak memory per (base, mode, rows) at the REAL member count
 10  io              read throughput of the staging filesystem, which is what a per-step
                     CSV loader is bounded by

Everything is printed as plain text with one fact per line, so the whole report can be pasted
back into a conversation.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BAR = "=" * 78


def head(n: int, title: str) -> None:
    print(f"\n{BAR}\n{n:2d}. {title.upper()}\n{BAR}")


def run(cmd: list[str], timeout: int = 60) -> str:
    """Run a command, return stdout+stderr, never raise."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return f"(command not found: {cmd[0]})"
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"
    except Exception as exc:                                   # pragma: no cover
        return f"({type(exc).__name__}: {exc})"


# --------------------------------------------------------------------------- 1
def section_environment() -> None:
    head(1, "environment")
    print(f"  host            : {socket.gethostname()}")
    print(f"  platform        : {platform.platform()}")
    print(f"  python          : {sys.version.split()[0]}  at {sys.executable}")
    print(f"  cwd             : {Path.cwd()}")
    print(f"  repo            : {REPO}")

    # A stray active virtualenv silently wins over `conda activate`, which sent a week of pip
    # installs to the wrong prefix. Worth one line forever.
    venv = os.environ.get("VIRTUAL_ENV")
    conda = os.environ.get("CONDA_DEFAULT_ENV")
    print(f"  VIRTUAL_ENV     : {venv or '(none)'}")
    print(f"  CONDA_DEFAULT_ENV: {conda or '(none)'}")
    if venv and conda:
        print("  ** WARNING: both are set. The virtualenv shadows conda on PATH. **")

    for mod in ("torch", "tabpfn", "tabicl", "numpy", "pandas", "sklearn",
                "omegaconf", "huggingface_hub", "pyarrow"):
        try:
            import importlib.metadata as md
            print(f"  {mod:16}: {md.version(mod)}")
        except Exception:
            try:
                m = __import__(mod)
                print(f"  {mod:16}: {getattr(m, '__version__', 'present')}")
            except Exception:
                print(f"  {mod:16}: NOT INSTALLED")

    print("\n  git:")
    print(f"    commit        : {run(['git', 'rev-parse', '--short', 'HEAD']).strip()}")
    print(f"    branch        : {run(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()}")
    dirty = run(["git", "status", "--porcelain"]).strip()
    print(f"    uncommitted   : {len(dirty.splitlines())} path(s)")
    print(f"    tfm-library   : {run(['git', 'submodule', 'status', 'tfm-library']).strip()[:60]}")

    print("\n  environment variables that change where output lands:")
    for k in ("CREDITPFN_DATA_ROOT", "CREDITPFN_OUTPUT_ROOT", "CREDITPFN_STAGING_ROOT",
              "VSC_DATA", "VSC_SCRATCH", "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID",
              "SLURM_CLUSTER_NAME", "SLURM_JOB_PARTITION", "CUDA_VISIBLE_DEVICES"):
        print(f"    {k:24}= {os.environ.get(k, '(unset)')}")


# --------------------------------------------------------------------------- 2
def section_slurm_limits() -> None:
    head(2, "slurm limits — the numbers that decide if a submission is accepted")
    if not shutil.which("sacctmgr") and not shutil.which("scontrol"):
        print("  no slurm client here (run this on a login node for these sections)")
        return

    user = os.environ.get("USER", "")
    print("  QOS limits (MaxSubmitJobsPerUser is the '500 jobs' ceiling):")
    out = run(["sacctmgr", "-nP", "show", "qos",
               "format=Name,MaxSubmitJobsPerUser,MaxJobsPerUser,MaxWall,MaxTRESPerUser"])
    for line in out.splitlines():
        if line.strip():
            print(f"    {line}")

    print("\n  associations for this user (which QOS and account apply):")
    out = run(["sacctmgr", "-nP", "show", "assoc", f"user={user}",
               "format=Cluster,Account,QOS,MaxSubmitJobs,MaxJobs"])
    for line in out.splitlines()[:20]:
        if line.strip():
            print(f"    {line}")

    for cluster in ("genius", "wice", "mindwell"):
        print(f"\n  {cluster} partitions (name, walltime limit, nodes, state):")
        out = run(["sinfo", "-M", cluster, "-h", "-o", "%P|%l|%D|%a|%G"])
        for line in out.splitlines():
            if line.strip() and not line.startswith("CLUSTER"):
                print(f"    {line}")

    print("\n  currently queued/running for this user:")
    for cluster in ("genius", "wice", "mindwell"):
        n = len([x for x in run(["squeue", "-M", cluster, "-u", user, "-h", "-o", "%i"]).splitlines()
                 if x.strip() and not x.startswith("CLUSTER")])
        print(f"    {cluster:10}: {n} task(s)")


# --------------------------------------------------------------------------- 3
def section_accounting() -> None:
    head(3, "accounting — credits and charge rates")
    for tool in ("sam-balance", "sam-list-usagerecords", "sam-statement"):
        if shutil.which(tool):
            print(f"  $ {tool}")
            for line in run([tool]).splitlines()[:25]:
                print(f"    {line}")
            break
    else:
        print("  sam-* tools not on PATH (login node only)")

    print("\n  charge rates from the VSC docs, credits per GPU-minute:")
    for name, rate in (("Genius P100", 41.67), ("Genius V100", 59.58),
                       ("wICE A100", 141.67), ("Mindwell B200", 437.50),
                       ("wICE H100", 569.44)):
        print(f"    {name:16} {rate:8.2f}   = {rate * 60:9,.0f} credits per GPU-hour")
    print("  NOTE P100 is sm_60 and V100 is sm_70; check section 4 for kernel support.")


# --------------------------------------------------------------------------- 4
def section_gpu() -> None:
    head(4, "gpu — card, memory, and whether this torch has kernels for it")
    print(run(["nvidia-smi"], timeout=60)[:2000] or "  nvidia-smi unavailable")
    try:
        import torch
    except ImportError:
        print("  torch not importable")
        return
    print(f"  torch            : {torch.__version__}")
    print(f"  cuda available   : {torch.cuda.is_available()}")
    print(f"  cuda build       : {torch.version.cuda}")
    print(f"  arch list        : {torch.cuda.get_arch_list()}")
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cap = f"sm_{p.major}{p.minor}"
        ok = cap in [a.replace("sm_", "sm_") for a in torch.cuda.get_arch_list()]
        print(f"  device {i}: {p.name}  {p.total_memory / 1e9:.0f} GB  {cap}  "
              f"SMs={p.multi_processor_count}  kernels={'YES' if ok else 'NO — will fail'}")


# --------------------------------------------------------------------------- 5
def section_precision() -> None:
    head(5, "precision and throughput — measured on THIS card")
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        print("  no cuda")
        return
    print(f"  bf16 supported   : {torch.cuda.is_bf16_supported()}")
    print(f"  tf32 matmul      : {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  tf32 cudnn       : {torch.backends.cudnn.allow_tf32}")

    n = 8192
    for dtype in (torch.float32, torch.bfloat16):
        try:
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            b = torch.randn(n, n, device="cuda", dtype=dtype)
            for _ in range(3):
                a @ b
            torch.cuda.synchronize()
            t0 = time.monotonic()
            reps = 10
            for _ in range(reps):
                a @ b
            torch.cuda.synchronize()
            dt = (time.monotonic() - t0) / reps
            tflops = 2 * n ** 3 / dt / 1e12
            print(f"  {str(dtype):22} {n}x{n} matmul: {dt * 1e3:7.2f} ms  {tflops:7.1f} TFLOP/s")
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  {str(dtype):22} FAILED: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- 6
def section_attention() -> None:
    head(6, "attention backends — the TabICLv2 26k ceiling")
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return
    if not torch.cuda.is_available():
        print("  no cuda")
        return
    b = torch.backends.cuda
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled",
                 "cudnn_sdp_enabled"):
        fn = getattr(b, name, None)
        print(f"  {name:28}: {fn() if callable(fn) else '(not in this torch)'}")

    # Does a long-sequence attention actually run with each backend? This is the direct test of
    # whether disabling cuDNN lifts the TabICLv2 row ceiling.
    print("\n  scaled_dot_product_attention at TabICLv2-like shapes (batch 2, heads 4, dim 64):")
    for seq in (8_192, 26_000, 40_000, 60_000):
        for label, toggles in (("cudnn on ", dict(cudnn=True)),
                               ("cudnn off", dict(cudnn=False))):
            try:
                if hasattr(b, "enable_cudnn_sdp"):
                    b.enable_cudnn_sdp(toggles["cudnn"])
                q = torch.randn(2, 4, seq, 64, device="cuda", dtype=torch.bfloat16,
                                requires_grad=True)
                torch.cuda.reset_peak_memory_stats()
                out = F.scaled_dot_product_attention(q, q, q)
                out.sum().backward()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"    seq={seq:>6}  {label}: OK    peak {peak:6.2f} GB")
                del q, out
                torch.cuda.empty_cache()
            except Exception as exc:
                msg = str(exc).split("\n")[0][:90]
                print(f"    seq={seq:>6}  {label}: FAIL  {type(exc).__name__}: {msg}")
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------- 7
def section_checkpoints() -> None:
    head(7, "checkpoints — present, size, and the frozen trainable fraction")
    try:
        sys.path.insert(0, str(REPO))
        from src.utils.stage_checkpoints import checkpoints_root, wanted_checkpoints
        import torch
    except Exception as exc:
        print(f"  cannot import: {type(exc).__name__}: {exc}")
        return
    dest = checkpoints_root()
    print(f"  dir: {dest}")
    for name in wanted_checkpoints():
        f = dest / name
        if not f.is_file():
            print(f"  MISSING  {name}")
            continue
        line = f"  ok  {name:44} {f.stat().st_size / 1e6:6.0f} MB"
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck) if isinstance(ck, dict) else {}
            if isinstance(sd, dict) and sd:
                sizes: dict[str, int] = {}
                for k, v in sd.items():
                    if hasattr(v, "numel"):
                        sizes[str(k).split(".")[0]] = sizes.get(str(k).split(".")[0], 0) + v.numel()
                tot = sum(sizes.values())
                big = max(sizes, key=sizes.get) if sizes else "?"
                frac = 100 * (tot - sizes.get(big, 0)) / max(1, tot)
                line += (f"  {tot / 1e6:6.1f}M params   backbone={big}"
                         f"  trainable-if-frozen={frac:.2f}%")
        except Exception as exc:
            line += f"  (unreadable: {type(exc).__name__})"
        print(line)


# --------------------------------------------------------------------------- 8
def section_data() -> None:
    head(8, "data on disk")
    try:
        sys.path.insert(0, str(REPO))
        from omegaconf import OmegaConf
        from src.utils.paths import apply_data_source_from_cfg, processed_dir
        apply_data_source_from_cfg(OmegaConf.load(REPO / "config" / "data.yaml"))
    except Exception as exc:
        print(f"  cannot resolve paths: {type(exc).__name__}: {exc}")
        return
    import csv
    for track in ("pd", "lgd"):
        d = Path(processed_dir(track))
        files = sorted(d.glob("*.csv")) if d.is_dir() else []
        print(f"  {track.upper()}: {len(files)} processed CSV(s) under {d}")
        total = 0
        for f in files:
            try:
                with f.open(newline="", encoding="utf-8") as fh:
                    r = csv.reader(fh)
                    ncol = len(next(r))
                    nrow = sum(1 for _ in r)
                total += nrow
                print(f"    {f.stem:30} {nrow:>9,} rows  {ncol:>4} cols")
            except Exception as exc:
                print(f"    {f.stem:30} unreadable ({type(exc).__name__})")
        print(f"    {'TOTAL':30} {total:>9,} rows")


# --------------------------------------------------------------------------- 9
def section_row_caps(rows_grid: list[int]) -> None:
    head(9, "row caps — measured at the REAL member count, both adaptation modes")
    try:
        sys.path.insert(0, str(REPO))
        from omegaconf import OmegaConf
        from scripts.probe_row_cap import probe_base, probe_tabicl_base
        from src.train.tabicl_compat import model_family
    except Exception as exc:
        print(f"  cannot import the probe: {type(exc).__name__}: {exc}")
        return
    cfg = OmegaConf.load(REPO / "config" / "train.yaml")
    members = int(OmegaConf.select(cfg, "train.n_estimators_finetune.pd") or 2)
    print(f"  n_estimators_finetune = {members} (this is what training uses)")
    for track, key in (("pd", "classifier_base_paths"), ("lgd", "regressor_base_paths")):
        for base in (OmegaConf.select(cfg, f"tunable.{key}") or []):
            for frozen in (False, True):
                tag = "frozen" if frozen else "full  "
                print(f"\n  --- {Path(str(base)).name}  track={track}  mode={tag}")
                fn = (probe_tabicl_base if model_family(str(base)) == "tabicl"
                      else probe_base)
                try:
                    fn(str(base), track, rows_grid, "cuda",
                       n_estimators=members, freeze_backbone=frozen)
                except TypeError:
                    fn(str(base), track, rows_grid, "cuda")
                except Exception as exc:
                    print(f"    FAILED: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- 10
def section_io() -> None:
    head(10, "filesystem throughput — what the per-step CSV loader is bounded by")
    try:
        sys.path.insert(0, str(REPO))
        from omegaconf import OmegaConf
        from src.utils.paths import apply_data_source_from_cfg, processed_dir
        apply_data_source_from_cfg(OmegaConf.load(REPO / "config" / "data.yaml"))
        cand = sorted(Path(processed_dir("pd")).glob("*.csv"), key=lambda f: -f.stat().st_size)
    except Exception as exc:
        print(f"  cannot resolve: {type(exc).__name__}: {exc}")
        return
    if not cand:
        print("  no processed CSVs to read")
        return
    f = cand[0]
    mb = f.stat().st_size / 1e6
    for label in ("cold-ish", "warm"):
        t0 = time.monotonic()
        with f.open("rb") as fh:
            while fh.read(8 << 20):
                pass
        dt = time.monotonic() - t0
        print(f"  {label:9} read of {f.name} ({mb:.0f} MB): {dt:6.2f} s  = {mb / max(dt, 1e-9):7.1f} MB/s")
    try:
        import pandas as pd
        t0 = time.monotonic()
        df = pd.read_csv(f)
        print(f"  pandas parse: {time.monotonic() - t0:6.2f} s for {len(df):,} rows "
              f"x {df.shape[1]} cols")
    except Exception as exc:
        print(f"  pandas parse failed: {type(exc).__name__}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", action="store_true", help="include the GPU sections")
    ap.add_argument("--probe", action="store_true", help="include the slow row-cap sweep")
    ap.add_argument("--rows", default="10000,26000,50000,90000",
                    help="comma-separated row grid for --probe")
    args = ap.parse_args(argv)

    print(BAR)
    print("  CreditPFN CLUSTER REPORT")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(BAR)

    section_environment()
    section_slurm_limits()
    section_accounting()
    section_checkpoints()
    section_data()
    if args.gpu:
        section_gpu()
        section_precision()
        section_attention()
        section_io()
        if args.probe:
            section_row_caps([int(x) for x in args.rows.split(",") if x.strip()])
    else:
        print("\n(GPU sections skipped — re-run with --gpu under sbatch for those.)")

    print(f"\n{BAR}\n  END OF REPORT\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
