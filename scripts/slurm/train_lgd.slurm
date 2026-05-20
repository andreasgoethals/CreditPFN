#!/bin/bash -l
# =============================================================================
#  CreditPFN — slurm array: continued pretraining (LGD / regressor)
#
#  Same structure as train_pd.slurm — see that file's header for
#  the array-size + submission flow.
# =============================================================================

#SBATCH --job-name=creditpfn-lgd-train
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
                         # via `python scripts/train_pipeline.py --list-trials track=lgd`.
                         # Over-sizing is safe: surplus tasks exit 0 cleanly.

sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1

# See train_pd.slurm for the rationale.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# See data.slurm for the rationale on these defaults + overrides.
export CREDITPFN_DATA_ROOT="${CREDITPFN_DATA_ROOT:-${VSC_SCRATCH}/CreditPFN}"
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/manifests" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/epochs/lgd" \
         "${CREDITPFN_OUTPUT_ROOT}/checkpoints/trained/lgd"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/train_lgd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
exec > "$LOG" 2>&1

cd "${VSC_DATA}/CreditPFN"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditPFN — TRAINING (LGD)"
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
    track=lgd

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
