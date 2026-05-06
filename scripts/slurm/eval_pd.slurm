#!/bin/bash
# =============================================================================
#  CreditPFN — slurm array: cross-model benchmark (PD / classifier)
# =============================================================================
#
#  ONE (model × test_dataset) PER ARRAY TASK. With ~3 000 test
#  datasets × ~25 models, the cartesian product is ~75 000 tasks; we
#  parallelise the heavy ones (XGBoost / CatBoost with per-fold
#  Optuna HPO) by running each in its own slurm task. Each task
#  writes its own per-method file under
#  ``results/PD/<method>/<run_name>_<timestamp>_<dataset>.csv``,
#  so concurrent tasks NEVER touch the same file (no locking, no
#  races). Aggregation downstream is `pd.read_csv` over a glob.
#
#  Compute the array size with:
#
#      python scripts/eval_pipeline.py --list-tasks track=pd
#
#  Submit with:
#
#      N=$(python scripts/eval_pipeline.py --list-tasks track=pd)
#      sbatch --array=0-$((N - 1))%32 scripts/slurm/eval_pd.slurm
#
#  TabPFN tasks need GPU (wICE gpu_h100). Boosting / linear baselines
#  could in principle run on cheaper CPU partitions, but mixing
#  partitions per array task is awkward; we use one GPU partition for
#  everything and trust slurm's scheduling. CPU-only baselines just
#  ignore the GPU.
# =============================================================================

#SBATCH --job-name=creditpfn-pd-eval
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
#SBATCH --array=0-99%32   # placeholder; recompute via --list-tasks

sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1

export CREDITPFN_DATA_ROOT="${VSC_SCRATCH}/CreditPFN"
export CREDITPFN_OUTPUT_ROOT="${VSC_DATA}/CreditPFN"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs" \
         "${CREDITPFN_OUTPUT_ROOT}/results"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/eval_pd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
exec > "$LOG" 2>&1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate CreditPFN
cd "${VSC_DATA}/CreditPFN"

echo "=========================================="
echo "CreditPFN — EVAL (PD)"
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
    track=pd

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
