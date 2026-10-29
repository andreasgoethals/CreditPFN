#!/bin/bash -l
#SBATCH --job-name=creditpfn-runtime-cpu
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
action="${1:?}"; export CREDITPFN_RUNTIME_VERSION="${2:?}"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_RUNTIME_ACCEPTANCE=1
export CREDITPFN_USE_SCRATCH=0 PYTHONUNBUFFERED=1
source scripts/slurm/_job_log.sh
start_job_log "runtime_${action}_${CREDITPFN_RUNTIME_VERSION}"
if [[ "$action" == prepare ]]; then
    _project="${CREDITPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-/lustre1/project/stg_00211}}"
    [[ "$(basename "$_project")" == CreditPFN ]] || _project="$_project/CreditPFN"
    /usr/bin/python3 -m src.utils.runtime_container seal "$_project/runtimes/$CREDITPFN_RUNTIME_VERSION" --specification "${3:?}"
    unset _project
fi
source scripts/slurm/_activate_env.sh
case "$action" in
    prepare)
        python -m src.utils.runtime_validation cpu "$CREDITPFN_RUNTIME_VERSION" --specification "${3:?recorded environment specification required}"
        python -m src.utils.stage_inputs --destination "$VSC_SCRATCH_GPFS1/CreditPFN" --write ;;
    accept)
        python -m src.utils.runtime_validation accept "$CREDITPFN_RUNTIME_VERSION" --runtime "$CREDITPFN_RUNTIME_ROOT" ;;
    *) echo 'Unknown runtime CPU action'; exit 2 ;;
esac
