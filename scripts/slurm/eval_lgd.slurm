#!/bin/bash
# =============================================================================
#  CreditPFN — slurm array: cross-model benchmark (LGD / regressor)
#
#  Same logic as eval_pd.slurm — see that file's header.
# =============================================================================

#SBATCH --job-name=creditpfn-lgd-eval
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-99%32

sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1

export CREDITPFN_DATA_ROOT="${VSC_SCRATCH}/CreditPFN"
export CREDITPFN_OUTPUT_ROOT="${VSC_DATA}/CreditPFN"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs" \
         "${CREDITPFN_OUTPUT_ROOT}/results"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/eval_lgd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
exec > "$LOG" 2>&1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate CreditPFN
cd "${VSC_DATA}/CreditPFN"

echo "=========================================="
echo "CreditPFN — EVAL (LGD)"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Array task ID  : ${SLURM_ARRAY_TASK_ID}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Date           : $(date)"
echo "Log file       : ${LOG}"
nvidia-smi || true
echo "=========================================="

python -u scripts/eval_pipeline.py \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --log-path "$LOG" \
    track=lgd

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
