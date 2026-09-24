#!/bin/bash -l
# CPU work: stage inputs, fingerprint plans, consolidate, clean and audit outcomes.
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
source scripts/slurm/_job_log.sh
start_job_log maintenance
export CREDITPFN_USE_SCRATCH=0
source scripts/slurm/_activate_env.sh
action="${1:?Choose stage, prepare, preflight, consolidate, migrate-output, clean or audit}"; shift
case "$action" in
    stage) python -m src.utils.stage_inputs --destination "${VSC_SCRATCH_GPFS1:?}/CreditPFN" "$@" ;;
    prepare) python -m src.utils.prepare_experiment "$@" ;;
    preflight) python -m src.utils.preflight "$@" ;;
    consolidate) python -m src.utils.consolidate_output "$@" ;;
    migrate-output) python -m src.utils.migrate_output "$@" ;;
    clean) python -m src.utils.clean_run "$@" ;;
    audit) python -m src.utils.audit_experiment "$@" ;;
    *) echo "Unknown maintenance action: $action" >&2; exit 2 ;;
esac
