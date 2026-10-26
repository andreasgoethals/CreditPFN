#!/usr/bin/env bash
# Explicit stages: no production experiment is submitted by this workflow.
set -euo pipefail
cd "$(dirname "$0")/../.."
action="${1:-prepare}"
workflow="${2:-}"
if [[ ( "$action" == prepare || "$action" == all ) && -z "$workflow" ]]; then
  workflow="$(date +%Y%m%d%H%M%S)${RANDOM}${RANDOM}"
fi
[[ "$workflow" =~ ^[a-zA-Z0-9]{1,80}$ ]] || { echo 'Provide a 1–80 character alphanumeric LoRA workflow ID'; exit 2; }
case "$action" in
 all)
   export CREDITPFN_USE_SCRATCH=0
   source scripts/slurm/_activate_env.sh
   python -u -m src.utils.lora_submit --id "$workflow" ;;
 prepare)
   echo "LoRA workflow: $workflow"
   sbatch scripts/slurm/lora_prepare.slurm "$workflow" ;;
 smoke|capacity)
   wall=01:00:00
   [[ "$action" == capacity ]] && wall=03:00:00
   sbatch --array=0-3%2 --time="${WALLTIME:-$wall}" scripts/slurm/lora_gpu.slurm "$action" "$workflow" ;;
 diagnose)
   task="${LORA_TASK:-2}"
   [[ "$task" =~ ^[0-3]$ ]] || { echo 'LORA_TASK must be 0, 1, 2 or 3'; exit 2; }
   sbatch --array="$task%1" --time="${WALLTIME:-00:15:00}" scripts/slurm/lora_gpu.slurm diagnose "$workflow" ;;
 inspect)
   sbatch scripts/slurm/lora_prepare.slurm "$workflow" inspect ;;
 *) echo 'Usage: run_experiment0_lora.sh all [ID] | prepare [ID] | smoke ID | capacity ID | inspect ID | diagnose ID'; exit 2 ;;
esac
