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
cd "${VSC_DATA:?}/CreditPFN"
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
case "$action" in
    experiment0)
        # Gate: every test of second-campaign mechanics, with upstream-parity inputs required.
        CREDITPFN_REQUIRE_PARITY=1 python -m pytest -q -p no:cacheprovider \
            tests/test_upstream_parity.py tests/test_adapter_l2sp.py tests/test_design.py \
            tests/test_pilot_checks.py tests/test_regression_objectives.py \
            tests/test_diagscale_precision.py tests/test_tabpfn_precision_policy.py \
            tests/test_environment_parity.py tests/test_retention_panel.py tests/test_corpus_policies.py \
            tests/test_target_finetune.py tests/test_campaign_tasks.py tests/test_control_matching.py \
            tests/test_archive_regressions.py tests/test_campaign_launch.py
        # Whole suite for the record.
        python -m pytest -q -p no:cacheprovider tests || echo "FULL SUITE HAD FAILURES (recorded above; not a gate)"
        python -m src.utils.density_audit --out "$CREDITPFN_OUTPUT_ROOT/output CreditPFN/experiment0/manifests/pilots/density_audit.json" \
            || echo "DENSITY AUDIT INCOMPLETE (recorded; not a gate)"
        python -m src.utils.campaign_launch prepare config/experiments/E00_pilots
        python -m src.utils.campaign_launch submit-experiment0 config/experiments/E00_pilots ;;
    submit)
        python -m src.utils.campaign_launch submit "${2:?experiment number, e.g. E6}" ;;
    summary)
        python -m src.utils.pilot_checks summary ;;
    *) echo 'Unknown campaign action'; exit 2 ;;
esac
