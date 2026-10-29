#!/bin/bash -l
#SBATCH --job-name=creditpfn-runtime-build
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export CREDITPFN_EXPERIMENT=experiment0
unset CREDITPFN_CONFIG CREDITPFN_LEGACY_ID CREDITPFN_LORA_WORKFLOW
source scripts/slurm/_job_log.sh
start_job_log "runtime_build_${1:?version ID required}"
bash scripts/slurm/_build_runtime.sh "${1:?version ID required}" "${2:?recorded interpreter required}" "${3:?Conda executable required}"
