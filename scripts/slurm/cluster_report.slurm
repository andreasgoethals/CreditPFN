#!/bin/bash -l
# =============================================================================
#  ONE report per GPU type, before committing the campaign.
#
#      # login node only (sections 1-3, 7-8: env, slurm limits, credits, files)
#      python -m src.utils.cluster_report
#
#      # then one job per card, so every GPU's characteristics are measured
#      sbatch --clusters=mindwell --partition=gpu_b200 scripts/slurm/cluster_report.slurm
#      sbatch --clusters=wice     --partition=gpu_a100 scripts/slurm/cluster_report.slurm
#      sbatch --clusters=wice     --partition=gpu_h100 scripts/slurm/cluster_report.slurm
#
#      # add the row-cap sweep (slow, ~15-25 min, and the reason to give it 2 h).
#      # Trailing args go to the report, not to sbatch:
#      sbatch --clusters=mindwell --partition=gpu_b200 \
#             scripts/slurm/cluster_report.slurm --probe
#
#  Run the login-node invocation FIRST: sections 2 and 3 (slurm limits, credits)
#  need the sacctmgr/sam-* clients, which compute nodes do not have.
# =============================================================================
#SBATCH --job-name=creditpfn-report
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL

set -euo pipefail
cd "${VSC_DATA:?}/CreditPFN"
source scripts/slurm/_job_log.sh
start_job_log cluster_report

# shellcheck source=/dev/null
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo " CreditPFN cluster report"
echo " job=${SLURM_JOB_ID:-?}  cluster=${SLURM_CLUSTER_NAME:-?}"
echo " partition=${SLURM_JOB_PARTITION:-?}  node=$(hostname)"
echo " date=$(date)"
echo "=========================================="

# Flags arrive as sbatch's trailing SCRIPT arguments, not as env vars. `PROBE=1 sbatch`
# looked like it worked and silently did not - jobs 11524571/2 both skipped section 9 -
# because what sbatch exports is set by the site's SBATCH_EXPORT, not by our shell.
ARGS=(--gpu "$@")
[[ -n "${PROBE:-}" ]] && ARGS+=(--probe)      # fallback, no longer relied on

echo "+ python -m src.utils.cluster_report ${ARGS[*]}"
python -u -m src.utils.cluster_report "${ARGS[@]}"
