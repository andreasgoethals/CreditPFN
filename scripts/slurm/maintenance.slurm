#!/bin/bash -l
# CPU work: stage inputs, fingerprint plans, archive/consolidate and audit outcomes.
# Submit from the repository; its own log stays outside the directories being archived.
#SBATCH --job-name=creditpfn-maintenance
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=maintenance_%j.log
#SBATCH --export=ALL
set -euo pipefail
cd "${VSC_DATA:?}/CreditPFN"
export CREDITPFN_USE_SCRATCH=0
source scripts/slurm/_activate_env.sh
action="${1:?Choose stage, prepare, archive, consolidate or audit}"; shift
case "$action" in
    stage) python -m src.utils.stage_inputs --destination "${VSC_SCRATCH_GPFS1:?}/CreditPFN" "$@" ;;
    prepare) python -m src.utils.prepare_experiment "$@" ;;
    archive) python -m src.utils.archive_experiment "$@" ;;
    consolidate) python -m src.utils.consolidate_output "$@" ;;
    audit) python -m src.utils.audit_experiment "$@" ;;
    *) echo "Unknown maintenance action: $action" >&2; exit 2 ;;
esac
