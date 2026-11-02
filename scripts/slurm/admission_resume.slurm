#!/bin/bash -l
# Submission-only helper from a separate DATA checkout; all science stays frozen.
#SBATCH --job-name=creditpfn-admission-resume
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
helper="${SLURM_SUBMIT_DIR:?}/src/utils/admission_recovery.py"
science="${1:?frozen scientific DATA checkout required}"
recovery="${2:?recovery ID required}"
cd "$science"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=0 PYTHONUNBUFFERED=1
unset CREDITPFN_CONFIG CREDITPFN_LEGACY_ID CREDITPFN_EVAL_REFRESH CREDITPFN_RUNTIME_ACCEPTANCE
unset CREDITPFN_DATA_ROOT CREDITPFN_BASE_CACHE_ROOT CREDITPFN_INPUT_POINTER
source scripts/slurm/_job_log.sh
start_job_log "admission_resume_${recovery}"
source scripts/slurm/_activate_env.sh
source scripts/slurm/_run_train.sh
if run_training_python -u "$helper" --checkout "$science" --recovery "$recovery"; then rc=0; else rc=$?; fi
if [[ "$rc" == 75 ]]; then
    scontrol requeue "${SLURM_JOB_ID:?}"
    exit 0
fi
exit "$rc"
