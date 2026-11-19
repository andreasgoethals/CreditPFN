#!/bin/bash -l
# Campaign controller in the accepted runtime (CPU only; GPU work is submitted, never run here).
#   experiment0:   container gate tests, LGD density audit, pilot plans, then every Experiment 0 job
#   submit E<n>:   plans and submission of one experiment within the queue budget (resumable)
#   summary:       audit of every pilot and GPU-check receipt (queued by experiment0)
#SBATCH --job-name=creditpfn-campaign-cpu
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
# The checkout that submitted the job (a second DATA checkout runs its own code).
cd "${SLURM_SUBMIT_DIR:?}"
test -f scripts/slurm/run_experiment.sh || { echo "Submit from the CreditPFN repository root, not $PWD" >&2; exit 2; }
action="${1:?experiment0, submit or summary}"
: "${CREDITPFN_RUNTIME_VERSION:?accepted runtime required}"
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=0 PYTHONUNBUFFERED=1
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA:?}/CreditPFN}"
export CREDITPFN_STAGING_ROOT="${CREDITPFN_STAGING_ROOT:-/gpfs1/project/prj_00013/CreditPFN}"
export CREDITPFN_REQUIRE_STAGING=1
unset CREDITPFN_CONFIG CREDITPFN_RUNTIME_ACCEPTANCE
source scripts/slurm/_job_log.sh
start_job_log "campaign_${action}${2:+_$2}"
echo "commit=$(git rev-parse HEAD)"
source scripts/slurm/_activate_env.sh
# Keep review and submission consistent even when two controllers start together. Independent
# campaign submitters share the lock. Slurm releases the descriptor if its controller dies.
gate_dir="$CREDITPFN_OUTPUT_ROOT/output CreditPFN/experiment0/manifests/pilots"
mkdir -p "$gate_dir"
exec 9>>"$gate_dir/controller.lock"
case "$action" in
    experiment0|rerun-experiment0)
        flock -x 9
        if [[ "$action" == experiment0 ]]; then
            ids=$(python -m src.utils.campaign_launch begin)
        else
            ids=$(python -m src.utils.campaign_launch begin --only "${2:?selection JSON}")
        fi
        read -r CREDITPFN_PILOT_RUN CREDITPFN_PILOT_SET <<< "$ids"
        export CREDITPFN_PILOT_RUN CREDITPFN_PILOT_SET ;;
    submit|summary) flock -s 9 ;;
esac
if [[ "$action" == experiment0 || "$action" == rerun-experiment0 ]]; then
        # A selective rerun can be the first run on changed code. Publish a passing CPU
        # preflight only after all applicable project tests, including upstream parity, pass.
        CREDITPFN_REQUIRE_PARITY=1 python -m pytest -q -p no:cacheprovider tests
        python -m src.utils.campaign_launch preflight
fi
case "$action" in
    experiment0)
        python -m src.utils.density_audit --out "$gate_dir/runs/$CREDITPFN_PILOT_SET/density_audit.json" \
            || echo "DENSITY AUDIT INCOMPLETE (recorded; not a gate)"
        python -m src.utils.campaign_launch prepare config/experiments/E00_pilots
        python -m src.utils.campaign_launch submit-experiment0 config/experiments/E00_pilots ;;
    rerun-experiment0)
        # Resubmit a part of Experiment 0 after a fix: ${2} is the JSON selection (see campaign_launch).
        python -m src.utils.campaign_launch prepare config/experiments/E00_pilots --only "${2:?selection JSON}"
        python -m src.utils.campaign_launch rerun-experiment0 config/experiments/E00_pilots --only "$2" ;;
    submit)
        python -m src.utils.campaign_launch submit "${2:?experiment number, e.g. E8}" ;;
    summary)
        python -m src.utils.pilot_checks summary ;;
    *) echo 'Unknown campaign action'; exit 2 ;;
esac
