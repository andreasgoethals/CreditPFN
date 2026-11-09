#!/bin/bash -l
# Experiment 0 GPU checks: pilot_gpu.slurm <recovery|capacity|evaluation|precision>, one case per array element.
#SBATCH --job-name=creditpfn-pilot-gpu
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${VSC_DATA:?}/CreditPFN"
action="${1:?recovery, capacity, evaluation or precision}"
: "${CREDITPFN_RUNTIME_VERSION:?accepted runtime required}"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=1 CREDITPFN_REQUIRE_STAGING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CREDITPFN_CONFIG CREDITPFN_STOP_AFTER_UPDATES CREDITPFN_STOP_REQUESTED
source scripts/slurm/_job_log.sh
start_job_log "campaign2_${action}_${SLURM_ARRAY_TASK_ID:?}"
source scripts/slurm/_activate_env.sh
nvidia-smi || true
python -m src.utils.pilot_checks "$action" --task "$SLURM_ARRAY_TASK_ID"
