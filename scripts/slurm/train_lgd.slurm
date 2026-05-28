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
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-47%4   # 48 trials (2 bases × 3 LRs × 2 LoRA × 2 qf × 2 acc).
                         # Recompute via
                         #   `python scripts/train_pipeline.py --list-trials track=lgd`.
                         # Over-sizing is safe: surplus tasks exit 0 cleanly.
                         #
                         # `cpus-per-task=16` matches the documented gpu_h100
                         # per-GPU limit (VSC docs, gpu_h100 row of the
                         # cpu_resource_limits_in_gpu_jobs table).

sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1

# See train_pd.slurm for the rationale.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

# See data.slurm for the rationale: OUTPUT_ROOT always durable;
# DATA_ROOT resolved from config/data.yaml inside _activate_env.sh.
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
