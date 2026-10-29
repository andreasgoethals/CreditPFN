#!/bin/bash -l
#SBATCH --job-name=creditpfn-legacy-gpu
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
unset CREDITPFN_CONFIG CREDITPFN_LORA_WORKFLOW CREDITPFN_LORA_GATE CREDITPFN_FLOW_ID CREDITPFN_FLOW_PHASE CREDITPFN_STOP_AFTER_UPDATES
export CREDITPFN_LEGACY_ID="${2:?ID required}" CREDITPFN_EXPERIMENT=experiment0
export CREDITPFN_USE_SCRATCH=1 CREDITPFN_REQUIRE_STAGING=1 CREDITPFN_SEGMENT_SECONDS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 PYTHONUNBUFFERED=1
source scripts/slurm/_job_log.sh
start_job_log "legacy_${1:?stage required}"
source scripts/slurm/_activate_env.sh
python -u -m src.utils.legacy_checks "$1" --id "$CREDITPFN_LEGACY_ID" --task "${SLURM_ARRAY_TASK_ID:?}"
