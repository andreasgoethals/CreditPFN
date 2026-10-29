#!/bin/bash -l
#SBATCH --job-name=creditpfn-refresh-gpu
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=1
export PYTHONUNBUFFERED=1 CREDITPFN_EVAL_REFRESH="${1:?refresh ID required}"
unset CREDITPFN_LEGACY_ID CREDITPFN_LORA_WORKFLOW CREDITPFN_RUNTIME_ACCEPTANCE
source scripts/slurm/_job_log.sh
start_job_log "refresh_${CREDITPFN_EVAL_REFRESH}_${SLURM_ARRAY_TASK_ID:?}"
source scripts/slurm/_activate_env.sh
python -m src.utils.evaluation_refresh run "$CREDITPFN_EVAL_REFRESH" --task "$SLURM_ARRAY_TASK_ID"
