#!/bin/bash -l
# CPU data registration/sanitization. Prepare inputs before staging/plans.
# 64 GB retains the measured margin for wide-table preparation.
#SBATCH --job-name=creditpfn-data
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch_graniterapids
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL

set -euo pipefail
# The checkout that submitted the job (a second DATA checkout runs its own code).
cd "${SLURM_SUBMIT_DIR:?}"
test -f scripts/slurm/run_experiment.sh || { echo "Submit from the CreditPFN repository root, not $PWD" >&2; exit 2; }
source scripts/slurm/_job_log.sh
start_job_log data
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export CREDITPFN_USE_SCRATCH=0
source scripts/slurm/_activate_env.sh
if [[ "${1:-}" == "--refresh-domain-control" ]]; then
    shift
    python -m src.utils.prepare_retention --panel domain_control --supersede-changed
fi
python -u scripts/data_pipeline.py --log-path "$LOG" "$@"
# Training jobs read an immutable, content-addressed scratch snapshot of the inputs. Refresh it here,
# so a rebuilt table can never be trained from the snapshot taken before the rebuild.
if [[ -n "${CREDITPFN_RUNTIME_VERSION:-}" && -n "${VSC_SCRATCH_GPFS1:-}" ]]; then
    python -m src.utils.stage_inputs --destination "$VSC_SCRATCH_GPFS1/CreditPFN" --write
fi
