#!/bin/bash -l
#SBATCH --job-name=creditpfn-study-controller
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=1 PYTHONUNBUFFERED=1
unset CREDITPFN_LEGACY_ID CREDITPFN_LORA_WORKFLOW CREDITPFN_RUNTIME_ACCEPTANCE CREDITPFN_EVAL_REFRESH
source scripts/slurm/_job_log.sh
start_job_log "study_${1:?}_${2:?}"
source scripts/slurm/_activate_env.sh
python -m src.utils.study_workflow "$1" "$2"
