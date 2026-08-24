#!/usr/bin/env bash
# =============================================================================
#  ONE report per GPU type, before committing the campaign.
#
#      # login node only (sections 1-3, 7-8: env, slurm limits, credits, files)
#      python scripts/cluster_report.py
#
#      # then one job per card, so every GPU's characteristics are measured
#      sbatch --clusters=mindwell --partition=gpu_b200 scripts/slurm/cluster_report.slurm
#      sbatch --clusters=wice     --partition=gpu_a100 scripts/slurm/cluster_report.slurm
#      sbatch --clusters=wice     --partition=gpu_h100 scripts/slurm/cluster_report.slurm
#
#      # add the row-cap sweep (slow, ~15-25 min, and the reason to give it 2 h).
#      # Trailing args go to cluster_report.py, not to sbatch:
#      sbatch --clusters=mindwell --partition=gpu_b200 \
#             scripts/slurm/cluster_report.slurm --probe
#
#  Run the login-node invocation FIRST: sections 2 and 3 (slurm limits, credits)
#  need the sacctmgr/sam-* clients, which compute nodes do not have.
# =============================================================================
#SBATCH --job-name=creditpfn-report
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=output/logs/cluster_report_%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# shellcheck source=/dev/null
source scripts/slurm/_activate_env.sh

mkdir -p output/logs

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

echo "+ cluster_report.py ${ARGS[*]}"
python -u scripts/cluster_report.py "${ARGS[@]}"

echo "=========================================="
echo " END status=$? - $(date)"
echo "=========================================="
