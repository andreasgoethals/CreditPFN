#!/bin/bash -l
# Experiment 0 GPU checks: pilot_gpu.slurm <recovery|capacity|evaluation|precision|scale>.
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
# The checkout that submitted the job (a second DATA checkout runs its own code).
cd "${SLURM_SUBMIT_DIR:?}"
test -f scripts/slurm/run_experiment.sh || { echo "Submit from the CreditPFN repository root, not $PWD" >&2; exit 2; }
action="${1:?recovery, capacity, evaluation, precision or scale}"
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
