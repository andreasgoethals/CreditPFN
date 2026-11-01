#!/bin/bash -l
#SBATCH --job-name=creditpfn-study-controller
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=0 PYTHONUNBUFFERED=1
unset CREDITPFN_LEGACY_ID CREDITPFN_LORA_WORKFLOW CREDITPFN_RUNTIME_ACCEPTANCE CREDITPFN_EVAL_REFRESH
# The bootstrap only queues the acceptance graph; it cannot release research.
# This removes the need for an installed DATA environment on the login node.
if [[ "${1:?}" == submit ]]; then
    export CREDITPFN_RUNTIME_ACCEPTANCE=1 CREDITPFN_USE_SCRATCH=0
fi
source scripts/slurm/_job_log.sh
start_job_log "study_${1:?}_${2:?}"
source scripts/slurm/_activate_env.sh
if [[ "$1" != submit ]]; then
    case "${SLURM_CLUSTER_NAME:?}" in
        mindwell) _native_inputs="${VSC_SCRATCH_GPFS1:?}/CreditPFN" ;;
        wice) _native_inputs="${VSC_SCRATCH_LUSTRE1:?}/CreditPFN" ;;
        *) echo 'Unsupported controller cluster'; exit 2 ;;
    esac
    if [[ ! -f "$_native_inputs/ACTIVE_INPUTS.json" ]]; then
        python -m src.utils.stage_inputs --destination "$_native_inputs" --write
    fi
    _resolved_native=$(python -m src.utils.stage_inputs --resolve "$_native_inputs/ACTIVE_INPUTS.json")
    export CREDITPFN_DATA_ROOT="$_resolved_native"
    export CREDITPFN_BASE_CACHE_ROOT="$CREDITPFN_DATA_ROOT"
fi
python -m src.utils.study_workflow "$@"
