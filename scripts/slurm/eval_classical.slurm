#!/bin/bash -l
# CPU baseline HPO is cached across training recipes, folds and seeds.
#SBATCH --job-name=creditpfn-classical
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${VSC_DATA:?}/CreditPFN"
source scripts/slurm/_job_log.sh
start_job_log eval_classical
export CREDITPFN_USE_SCRATCH=0 CREDITPFN_EVAL_KIND=classical
source scripts/slurm/_activate_env.sh
python -u scripts/eval_pipeline.py --config "${CREDITPFN_CONFIG:?}" \
    --split-index "${CREDITPFN_SPLIT_INDEX:?}" --task-index "${SLURM_ARRAY_TASK_ID:?}" \
    --tasks "${EVAL_TASKS:-16}" --log-path "$LOG"
