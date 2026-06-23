#!/bin/bash -l
# =============================================================================
#  CreditPFN — slurm array: continued pretraining (LGD / regressor)
#  Target: Mindwell cluster — NVIDIA B200 SXM6 Blackwell (192 GiB VRAM)
# =============================================================================
#
#  Same structure as train_pd_b200.slurm — see that file for full notes.
#  Intended for running LGD training on Mindwell while PD runs on wICE
#  (or both on Mindwell simultaneously for maximum parallelism).
# =============================================================================

#SBATCH --job-name=creditpfn-lgd-b200
#SBATCH --cluster=mindwell
#SBATCH --account=lp_mindwell_pilot
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00   # LGD corpus is smaller; full_pass fits in ~3h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-23%12   # 24 trials. See train_pd_b200.slurm for rationale.

sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"
# export CREDITPFN_STAGING_ROOT="/staging/leuven/stg_XXXXX/CreditPFN"

: "${CREDITPFN_DATA_ROOT:=${VSC_SCRATCH_LUSTRE1:-${VSC_DATA}}/CreditPFN}"
export CREDITPFN_DATA_ROOT

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/manifests" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/epochs/lgd"
if [[ -n "${CREDITPFN_STAGING_ROOT:-}" ]]; then
    mkdir -p "${CREDITPFN_STAGING_ROOT}/checkpoints/trained/lgd"
else
    mkdir -p "${CREDITPFN_OUTPUT_ROOT}/checkpoints/trained/lgd"
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/train_lgd_b200_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
exec > "$LOG" 2>&1

cd "${VSC_DATA}/CreditPFN"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditPFN — TRAINING (LGD) — Mindwell B200"
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
    track=lgd

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
