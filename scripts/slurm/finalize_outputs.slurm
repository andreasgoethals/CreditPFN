#!/bin/bash -l
# Byte-preserving output repair; use original-source audits before publication.
#SBATCH --job-name=creditpfn-output-finalize
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_verbekelab
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
helper="${SLURM_SUBMIT_DIR:?}/src/utils/repair_extension_outputs.py"
science="${1:?frozen scientific checkout required}"
recovery="${2:?recovery ID required}"
export CREDITPFN_RUNTIME_VERSION="${3:?accepted PROJECT runtime version required}"
cd "$science"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=0 PYTHONUNBUFFERED=1
unset CREDITPFN_CONFIG CREDITPFN_LEGACY_ID CREDITPFN_EVAL_REFRESH CREDITPFN_RUNTIME_ACCEPTANCE
unset CREDITPFN_DATA_ROOT CREDITPFN_BASE_CACHE_ROOT CREDITPFN_INPUT_POINTER
source scripts/slurm/_job_log.sh
start_job_log "output_finalize_${recovery}"
source scripts/slurm/_activate_env.sh
python -u "$helper" --checkout "$science" --recovery "$recovery" --apply
