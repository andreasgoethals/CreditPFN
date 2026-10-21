#!/bin/bash -l
#SBATCH --job-name=creditpfn-review-gpu
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --array=0-14%2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
unset CREDITPFN_CONFIG CREDITPFN_FLOW_ID CREDITPFN_FLOW_PHASE
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=1 CREDITPFN_REQUIRE_STAGING=1
source scripts/slurm/_job_log.sh
start_job_log "inference_review_${1:?review ID required}_${SLURM_ARRAY_TASK_ID:?}"
source scripts/slurm/_activate_env.sh
python -u -m src.utils.inference_review run --id "$1" --task "$SLURM_ARRAY_TASK_ID"
