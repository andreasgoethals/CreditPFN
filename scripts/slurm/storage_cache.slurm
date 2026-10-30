#!/bin/bash -l
#SBATCH --job-name=creditpfn-cache-retire
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CREDITPFN_EXPERIMENT=experiment0
unset CREDITPFN_LEGACY_ID CREDITPFN_LORA_WORKFLOW CREDITPFN_OWNED_CACHE
source scripts/slurm/_job_log.sh
start_job_log storage_cache
# Standard-library metadata work only. No environment installation or activation.
plan="$CREDITPFN_OUTPUT_ROOT/output CreditPFN/experiment0/manifests/storage_review/cache_${SLURM_CLUSTER_NAME}_${SLURM_JOB_ID}.json"
/usr/bin/python3 -m src.utils.retire_job_caches "$plan"
if [[ "${1:-preview}" == apply ]]; then
    /usr/bin/python3 -m src.utils.retire_job_caches "$plan" --apply
elif [[ "${1:-preview}" != preview ]]; then
    echo 'Usage: storage_cache.slurm [preview|apply]' >&2
    exit 2
fi
