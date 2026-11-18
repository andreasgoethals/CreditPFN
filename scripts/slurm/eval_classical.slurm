#!/bin/bash -l
# CPU baseline HPO is cached across training recipes, folds and seeds.
#SBATCH --job-name=creditpfn-classical
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
# The checkout that submitted the job (a second DATA checkout runs its own code).
cd "${SLURM_SUBMIT_DIR:?}"
test -f scripts/slurm/run_experiment.sh || { echo "Submit from the CreditPFN repository root, not $PWD" >&2; exit 2; }
source scripts/slurm/_job_log.sh
start_job_log eval_classical
export CREDITPFN_USE_SCRATCH=0 CREDITPFN_EVAL_KIND=classical
source scripts/slurm/_activate_env.sh
source scripts/slurm/_eval_cache.sh
share_eval_controls
python -u scripts/eval_pipeline.py --config "${CREDITPFN_CONFIG:?}" \
    --split-index "${CREDITPFN_SPLIT_INDEX:?}" --task-index "${SLURM_ARRAY_TASK_ID:?}" \
    --tasks "${EVAL_TASKS:-16}" --log-path "$LOG"
