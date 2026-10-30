#!/usr/bin/env bash
# Isolated controls for public experiments 5 and 8–10. No production training is submitted.
set -euo pipefail
cd "$(dirname "$0")/../.."
export CREDITPFN_EXPERIMENT=experiment0
case "${1:-prepare}" in
 prepare)
   sbatch scripts/slurm/followup_prepare.slurm ;;
 smoke|capacity)
   [[ "${2:-}" =~ ^[a-zA-Z0-9]+$ ]] || { echo 'Provide the workflow ID from preparation'; exit 2; }
   wall=01:00:00
   [[ "$1" == capacity ]] && wall=03:00:00
   sbatch --array=0-3%2 --time="${WALLTIME:-$wall}" scripts/slurm/followup_gpu.slurm "$1" "$2" ;;
 inspect)
   python -m src.utils.followup_checks inspect --id "${2:?workflow ID required}" ;;
 *) echo 'Usage: run_experiment0_followups.sh prepare | smoke ID | capacity ID | inspect ID'; exit 2 ;;
esac
