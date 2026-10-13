#!/bin/bash -l
# Resumes completed context cells; prediction arrays live in project SQLite files.
#SBATCH --job-name=creditpfn-context-eval
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
unset CREDITPFN_CONFIG
export CREDITPFN_EXPERIMENT=experiment4 CREDITPFN_USE_SCRATCH=1 CREDITPFN_REQUIRE_STAGING=1
source scripts/slurm/_job_log.sh
start_job_log context_eval
source scripts/slurm/_activate_env.sh
python -u scripts/context_pipeline.py run --plan "${1:?plan path required}" --task "${SLURM_ARRAY_TASK_ID:?}"
