#!/bin/bash -l
# CPU preparation, followed by four bounded adapter smoke tasks.
#SBATCH --job-name=creditpfn-followup-prepare
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
unset CREDITPFN_CONFIG CREDITPFN_FLOW_ID CREDITPFN_FLOW_PHASE
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=0
source scripts/slurm/_job_log.sh
start_job_log followup_prepare
source scripts/slurm/_activate_env.sh
workflow=$(python -u -m src.utils.followup_checks prepare)
[[ "$workflow" =~ ^[a-zA-Z0-9]+$ ]] || { echo "Preparation failed: $workflow"; exit 1; }
echo "Follow-up workflow: $workflow"
bash scripts/slurm/run_experiment0_followups.sh smoke "$workflow"
