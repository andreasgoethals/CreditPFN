#!/bin/bash -l
# =============================================================================
#  CreditPFN — slurm array: continued pretraining (PD / classifier)
# =============================================================================
#
#  ONE TRIAL PER ARRAY TASK. Each trial produces ONE log file:
#      ${OUTPUT_ROOT}/logs/train_pd_<YYYYMMDD>_<HHMMSS>_j<JOBID>_a<TASKID>.log
#  with the slurm boilerplate, the python loop's per-step training
#  log, and the final summary line all in one place. Slurm's own
#  --output goes to /dev/null so we don't get a competing stub file.
#
#  Trial = one (base_checkpoint × learning_rate) tuple from
#  `cfg.tunable` in `config/train.yaml`. The multi-chunk policy is
#  fixed (one chunk per parent dataset). Compute the array size with:
#
#      python scripts/train_pipeline.py --list-trials track=pd
#
#  Submit with:
#
#      N=$(python scripts/train_pipeline.py --list-trials track=pd)
#      sbatch --array=0-$((N - 1))%4 scripts/slurm/train_pd.slurm
#
#  (`%4` caps in-flight tasks at 4. Drop or raise as the queue
#   tolerates.)
#
#  wICE `gpu_h100` partition is a MIX of two H100 generations:
#      * NVIDIA H100 80 GB HBM3   (most nodes — `s16gNN`)
#      * NVIDIA H100 NVL 94 GB    (subset)
#  Slurm does not guarantee which one you get on a given task. The
#  `n_finetune_ctx_plus_query_samples = 10_000` default in train.yaml
#  is sized for the 80 GB card so every task fits. The comment in
#  train.yaml has the full memory budget table.
# =============================================================================

#SBATCH --job-name=creditpfn-pd-train
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00   # full_pass on PD's big datasets @ 50 epochs is slow;
                          # 12h killed 9 full_pass PD trials on 2026-06-01.
                          # eval now uses afterany so killed trials don't block
                          # eval, but 24h lets more full_pass trials finish.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-23%8   # 24 trials (2 bases × 3 LRs × 2 LoRA × 1 qf × 1 acc
                         #            × 2 pass-modes). Recompute via
                         #   `python scripts/train_pipeline.py --list-trials track=pd`.
                         # Over-sizing is safe: surplus tasks exit 0 cleanly
                         # (see scripts/train_pipeline.py soft no-op).
                         #
                         # **Resource-asks tightened 2026-05-28** based on
                         # VSC docs (`repositories/VSC Documentation.txt`):
                         #   "Short jobs may benefit from back fill, i.e.,
                         #    if the scheduler finds a gap … long enough to
                         #    run that job before it has enough resources
                         #    to start a large higher-priority parallel job."
                         # AND:
                         #   "On some clusters, a short walltime will give
                         #    you a higher priority."
                         # AND:
                         #   "Many of our clusters have a stricter limit on
                         #    the number of long-running jobs than on the
                         #    number of jobs with a shorter walltime."
                         #
                         # So shorter `--time` → better queue position via
                         # backfill + potentially higher concurrent-job limit.
                         # Observed for ONE_SAMPLE @ 50 epochs: ~37-39 min/trial
                         # (8 v3 lr=3e-7 trials, 2026-05-28/29) at the old 10k
                         # row cap — LoRA/qf/accumulate barely moved it.
                         # BUT this array now also runs the FULL_PASS arm at
                         # the 10k row cap (v3; v2.6 = 6k): full_pass takes
                         # ~Σ ceil(rows/cap) steps/epoch (≈ 8-12× more steps for
                         # PD's big datasets), so a full_pass trial can run
                         # several hours. --time=24:00:00 is sized for that
                         # worst case (12h killed 9 full_pass PD trials on
                         # 2026-06-01); the fast one_sample trials simply finish
                         # early and free the slot. **Check the first full_pass
                         # trial's `epoch_dt` in the log and extrapolate
                         # (50×epoch_dt + setup); lower --time to ~02:00:00 for
                         # one_sample-only runs to regain backfill priority.**
                         #
                         # `%8` (up from `%4`) — keeps the total sweep wall-time
                         # bounded (fast one_sample trials cycle through the 8
                         # slots quickly) while staying well under VSC's
                         # user-level job count limits.
                         #
                         # `cpus-per-task=16` matches the documented gpu_h100
                         # per-GPU limit (16 cores per GPU). `mem=64G` is
                         # well within the 187 200 MiB per-GPU budget.

# Stagger array starts to avoid I/O storms on shared scratch.
sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1

# wICE's gpu_h100 partition mixes H100 80 GB and H100 NVL 94 GB cards.
# `expandable_segments` lets the allocator reclaim fragmented blocks,
# which the OOM error message explicitly recommends. Cheap insurance —
# no measurable slowdown.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

# OUTPUT_ROOT is always durable storage; DATA_ROOT is resolved from
# `config/data.yaml`'s `paths.data_source` knob inside _activate_env.sh.
# See `scripts/slurm/data.slurm` for the full precedence ladder.
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"
# Large durable artefacts (checkpoints, eval results) go to project staging.
# Set this before submitting:
#   export CREDITPFN_STAGING_ROOT="/staging/leuven/stg_XXXXX/CreditPFN"
# Falls back to CREDITPFN_OUTPUT_ROOT if unset (no staging available).

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/manifests" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/epochs/pd"
if [[ -n "${CREDITPFN_STAGING_ROOT:-}" ]]; then
    mkdir -p "${CREDITPFN_STAGING_ROOT}/checkpoints/trained/pd"
else
    mkdir -p "${CREDITPFN_OUTPUT_ROOT}/checkpoints/trained/pd"
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/train_pd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"

# Single log file for the whole task (slurm + python).
exec > "$LOG" 2>&1

cd "${VSC_DATA}/CreditPFN"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditPFN — TRAINING (PD)"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Array task ID  : ${SLURM_ARRAY_TASK_ID}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Date           : $(date)"
echo "Log file       : ${LOG}"
echo "DATA_ROOT      : ${CREDITPFN_DATA_ROOT}"
echo "OUTPUT_ROOT    : ${CREDITPFN_OUTPUT_ROOT}"
echo "STAGING_ROOT   : ${CREDITPFN_STAGING_ROOT:-<not set — using OUTPUT_ROOT>}"
nvidia-smi || true
echo "=========================================="

python -u scripts/train_pipeline.py \
    --trial-index "${SLURM_ARRAY_TASK_ID}" \
    --log-path "$LOG" \
    track=pd

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
