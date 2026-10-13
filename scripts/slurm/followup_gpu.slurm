#!/bin/bash -l
# One modern base and task per GPU. Capacity checks require all smoke receipts.
#SBATCH --job-name=creditpfn-followup-gpu
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
unset CREDITPFN_CONFIG CREDITPFN_STOP_AFTER_UPDATES CREDITPFN_FLOW_ID CREDITPFN_FLOW_PHASE
export CREDITPFN_SEGMENT_SECONDS=0
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=1 CREDITPFN_REQUIRE_STAGING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/slurm/_job_log.sh
start_job_log "followup_${2:?}_${1:?}"
source scripts/slurm/_activate_env.sh
python -u -m src.utils.followup_checks "${1:?stage required}" --id "${2:?workflow ID required}" --task "${SLURM_ARRAY_TASK_ID:?}"
