#!/bin/bash -l
#SBATCH --job-name=creditpfn-review-cpu
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
[[ "${1:-}" == prepare || "${1:-}" == inspect ]] || { echo "Expected prepare or inspect" >&2; exit 2; }
unset CREDITPFN_CONFIG CREDITPFN_FLOW_ID CREDITPFN_FLOW_PHASE
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=0 CREDITPFN_REQUIRE_STAGING=1
source scripts/slurm/_job_log.sh
start_job_log "inference_review_${1}_${2:?review ID required}"
source scripts/slurm/_activate_env.sh
python -u -m src.utils.inference_review "$1" --id "$2"
