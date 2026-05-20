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
#  wICE `gpu_h100` partition gives us KU Leuven's newest GPU
#  (NVIDIA H100 NVL, 96GB) — the best fit for TabPFN finetuning.
# =============================================================================

#SBATCH --job-name=creditpfn-pd-train
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-31%4   # 32 trials (4 bases x 4 LRs x 2 LoRA). Recompute
                         # via `python scripts/train_pipeline.py --list-trials track=pd`.
                         # Over-sizing is safe: surplus tasks exit 0 cleanly
                         # (see scripts/train_pipeline.py soft no-op).

# Stagger array starts to avoid I/O storms on shared scratch.
sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1

# wICE's gpu_h100 partition mixes H100 80 GB and H100 NVL 94 GB cards.
# `expandable_segments` lets the allocator reclaim fragmented blocks,
# which the OOM error message explicitly recommends. Cheap insurance —
# no measurable slowdown.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Honour user-set CREDITPFN_DATA_ROOT (propagated via submit_full_pipeline.sh
# --export=ALL,...). Default = scratch; override to $VSC_DATA/CreditPFN
# if you want to keep data on home (e.g. as a scratch-purge workaround).
export CREDITPFN_DATA_ROOT="${CREDITPFN_DATA_ROOT:-${VSC_SCRATCH}/CreditPFN}"
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/manifests" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/epochs/pd" \
         "${CREDITPFN_OUTPUT_ROOT}/checkpoints/trained/pd"

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
nvidia-smi || true
echo "=========================================="

python -u scripts/train_pipeline.py \
    --trial-index "${SLURM_ARRAY_TASK_ID}" \
    --log-path "$LOG" \
    track=pd

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
