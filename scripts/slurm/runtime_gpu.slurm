#!/bin/bash -l
#SBATCH --job-name=creditpfn-runtime-gpu
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
cd "${SLURM_SUBMIT_DIR:?}"
version="${1:?}"; task="${SLURM_ARRAY_TASK_ID:?}"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 PYTHONUNBUFFERED=1
unset CREDITPFN_RUNTIME_VERSION CREDITPFN_LEGACY_ID CREDITPFN_LORA_WORKFLOW
source scripts/slurm/_job_log.sh
start_job_log "runtime_${version}_${task}"
if (( task < 8 )); then
    source scripts/slurm/_activate_env.sh
    python -m src.utils.runtime_validation sample "$version" --task "$task" --arm original
fi
export CREDITPFN_RUNTIME_VERSION="$version" CREDITPFN_RUNTIME_ACCEPTANCE=1
source scripts/slurm/_activate_env.sh
if (( task < 8 )); then
    python -m src.utils.runtime_validation sample "$version" --task "$task" --arm container
    python -m src.utils.runtime_validation compare "$version" --task "$task"
elif (( task == 8 )); then
    python -m src.utils.runtime_validation capacity "$version" --track pd
elif (( task == 9 )); then
    python -m src.utils.runtime_validation capacity "$version" --track lgd
else
    exit 2
fi
