#!/bin/bash -l
#SBATCH --job-name=creditpfn-recovery-controller
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=0 PYTHONUNBUFFERED=1
unset CREDITPFN_CONFIG CREDITPFN_LEGACY_ID CREDITPFN_EVAL_REFRESH CREDITPFN_RUNTIME_ACCEPTANCE
# Roots belong to the child cluster, never its submitting CPU/GPU allocation.
unset CREDITPFN_DATA_ROOT CREDITPFN_BASE_CACHE_ROOT CREDITPFN_INPUT_POINTER
[[ "${1:?}" != bootstrap ]] || export CREDITPFN_RUNTIME_ACCEPTANCE=1
[[ "$1" != recover ]] || export CREDITPFN_USE_SCRATCH=1
source scripts/slurm/_job_log.sh
start_job_log "recovery_${1}_${2:?}"
source scripts/slurm/_activate_env.sh
source scripts/slurm/_run_train.sh
if run_training_python -u -m src.utils.campaign_recovery "$@"; then rc=0; else rc=$?; fi
if [[ "$rc" == 75 && "$1" == recover ]]; then
    scontrol requeue "${SLURM_JOB_ID:?}"
    exit 0
fi
exit "$rc"
