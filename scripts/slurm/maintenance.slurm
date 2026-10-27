#!/bin/bash -l
# Maintenance and diagnostics; GPU probes require an explicit Mindwell GPU override.
# Submit from the repository; cleanup preserves this job's active log.
#SBATCH --job-name=creditpfn-maintenance
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${VSC_DATA:?}/CreditPFN"
for arg in "$@"; do
 case "$arg" in *config/experiment[0-8]/*) export CREDITPFN_EXPERIMENT="$(basename "$(dirname "$arg")")" ;; esac
done
source scripts/slurm/_job_log.sh
start_job_log maintenance
export CREDITPFN_USE_SCRATCH=0
source scripts/slurm/_activate_env.sh
action="${1:?Choose stage, prepare, preflight, consolidate, experiment0, inspect-recovery, probe-repeatability, clean, reset-family or audit}"; shift
case "$action" in
    experiment0) python -u -m src.utils.experiment0 "$@" ;;
    inspect-recovery) python -u -m src.utils.recovery_check --inspect "$@" ;;
    probe-repeatability) python -u -m src.utils.gpu_repeatability "$@" ;;
    stage) python -m src.utils.stage_inputs --destination "${VSC_SCRATCH_GPFS1:?}/CreditPFN" "$@" ;;
    prepare) python -m src.utils.prepare_experiment "$@" ;;
    prepare-context) python -u scripts/context_pipeline.py prepare "$@" ;;
    preflight) python -m src.utils.preflight "$@" ;;
    consolidate) python -m src.utils.consolidate_output "$@" ;;
    clean) python -m src.utils.clean_run "$@" ;;
    reset-family) python -m src.utils.reset_family_benchmark "$@" ;;
    audit) python -m src.utils.audit_experiment "$@" ;;
    *) echo "Unknown maintenance action: $action" >&2; exit 2 ;;
esac
