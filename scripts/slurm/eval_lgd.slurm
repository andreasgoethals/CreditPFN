#!/bin/bash -l
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
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-199%32   # generous upper bound; surplus tasks exit 0
                          # cleanly via the soft no-op in eval_pipeline.py.
                          # Recompute the tight bound with:
                          #   python scripts/eval_pipeline.py --list-tasks track=lgd
                          # …or just submit via scripts/slurm/submit_full_pipeline.sh.
                          #
                          # `cpus-per-task=16` matches the documented gpu_h100
                          # per-GPU limit (VSC docs gpu_h100 row of the
                          # cpu_resource_limits_in_gpu_jobs table).

sleep $((RANDOM % 15 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1

# See data.slurm for the rationale: OUTPUT_ROOT always durable;
# DATA_ROOT resolved from config/data.yaml inside _activate_env.sh.
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"
# Large durable artefacts (eval results) go to project staging.
# Set this before submitting:
#   export CREDITPFN_STAGING_ROOT="/staging/leuven/stg_XXXXX/CreditPFN"
# Falls back to CREDITPFN_OUTPUT_ROOT if unset.

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs"
if [[ -n "${CREDITPFN_STAGING_ROOT:-}" ]]; then
    mkdir -p "${CREDITPFN_STAGING_ROOT}/output/results"
else
    mkdir -p "${CREDITPFN_OUTPUT_ROOT}/output/results"
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/eval_lgd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
exec > "$LOG" 2>&1

cd "${VSC_DATA}/CreditPFN"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditPFN — EVAL (LGD)"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Array task ID  : ${SLURM_ARRAY_TASK_ID}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Date           : $(date)"
echo "Log file       : ${LOG}"
echo "OUTPUT_ROOT    : ${CREDITPFN_OUTPUT_ROOT}"
echo "STAGING_ROOT   : ${CREDITPFN_STAGING_ROOT:-<not set — using OUTPUT_ROOT>}"
nvidia-smi || true
echo "=========================================="

python -u scripts/eval_pipeline.py \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --log-path "$LOG" \
    track=lgd

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
