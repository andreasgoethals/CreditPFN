#!/bin/bash -l
#SBATCH --job-name=creditpfn-lora-gpu
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
unset CREDITPFN_CONFIG CREDITPFN_STOP_AFTER_UPDATES CREDITPFN_FLOW_ID CREDITPFN_FLOW_PHASE CREDITPFN_LORA_GATE
export CREDITPFN_LORA_WORKFLOW="${2:?workflow ID required}"
[[ "$CREDITPFN_LORA_WORKFLOW" =~ ^[a-zA-Z0-9]{1,80}$ ]] || { echo 'Invalid LoRA workflow ID'; exit 2; }
export CREDITPFN_SEGMENT_SECONDS=0 CREDITPFN_EXPERIMENT=experiment0
export CREDITPFN_USE_SCRATCH=1 CREDITPFN_REQUIRE_STAGING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42
source scripts/slurm/_job_log.sh
start_job_log "lora_${1:?stage required}"
source scripts/slurm/_activate_env.sh
python -u -m src.utils.lora_checks "${1:?}" --id "$CREDITPFN_LORA_WORKFLOW" --task "${SLURM_ARRAY_TASK_ID:?}"
