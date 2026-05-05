#!/bin/bash
# =============================================================================
#  CreditPFN — slurm array: continued pretraining (LGD / regressor)
# =============================================================================
#
#  Same logic as `train_pd.slurm`; see that file's header. Submit with:
#
#      N=$(python scripts/train_pipeline.py --list-trials track=lgd)
#      sbatch --array=0-$((N - 1))%4 scripts/slurm/train_lgd.slurm
# =============================================================================

#SBATCH --job-name=creditpfn-lgd-train
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/logs/slurm/train_lgd_%A_%a.out
#SBATCH --error=results/logs/slurm/train_lgd_%A_%a.err
#SBATCH --array=0-17%4

sleep $((RANDOM % 60 + 1))

set -euo pipefail

export PYTHONUNBUFFERED=1
export CREDITPFN_DATA_ROOT="${VSC_SCRATCH}/CreditPFN"
export CREDITPFN_OUTPUT_ROOT="${VSC_DATA}/CreditPFN"

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate CreditPFN

cd "${VSC_DATA}/CreditPFN"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/results/logs/slurm" \
         "${CREDITPFN_OUTPUT_ROOT}/logs/train" \
         "${CREDITPFN_OUTPUT_ROOT}/logs/runs" \
         "${CREDITPFN_OUTPUT_ROOT}/checkpoints/trained/lgd"

echo "=========================================="
echo "CreditPFN — TRAINING (LGD)"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Array task ID  : ${SLURM_ARRAY_TASK_ID}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Date           : $(date)"
echo "DATA_ROOT      : ${CREDITPFN_DATA_ROOT}"
echo "OUTPUT_ROOT    : ${CREDITPFN_OUTPUT_ROOT}"
nvidia-smi || true
echo "=========================================="

python -u scripts/train_pipeline.py \
    --trial-index "${SLURM_ARRAY_TASK_ID}" \
    --log-path "${CREDITPFN_OUTPUT_ROOT}/logs/train_lgd_${SLURM_ARRAY_JOB_ID}.log" \
    track=lgd

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
