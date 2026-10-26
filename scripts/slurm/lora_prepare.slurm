#!/bin/bash -l
#SBATCH --job-name=creditpfn-lora-prepare
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
unset CREDITPFN_CONFIG CREDITPFN_FLOW_ID CREDITPFN_FLOW_PHASE CREDITPFN_LORA_GATE
export CREDITPFN_LORA_WORKFLOW="${1:?workflow ID required}"
[[ "$CREDITPFN_LORA_WORKFLOW" =~ ^[a-zA-Z0-9]{1,80}$ ]] || { echo 'Invalid LoRA workflow ID'; exit 2; }
export CREDITPFN_EXPERIMENT=experiment0 CREDITPFN_USE_SCRATCH=0 CREDITPFN_REQUIRE_STAGING=1
# Ownership survives a failed conda activation, so that even setup-only logs can
# be removed by the narrowly scoped cleaner. The ID is validated before joining.
scope="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA:?}/CreditPFN}/output CreditPFN/experiment0/lora/$CREDITPFN_LORA_WORKFLOW"
mkdir -p "$scope"
if [[ ! -e "$scope/.scope.json" ]]; then
  (set -o noclobber; printf '{"kind":"creditpfn_lora_scope","id":"%s","schema":1}\n' "$CREDITPFN_LORA_WORKFLOW" > "$scope/.scope.json")
fi
source scripts/slurm/_job_log.sh
start_job_log "lora_${2:-prepare}"
source scripts/slurm/_activate_env.sh
action="${2:-prepare}"
[[ "$action" == prepare || "$action" == inspect ]] || { echo 'Invalid LoRA CPU action'; exit 2; }
python -u -m src.utils.lora_checks "$action" --id "$CREDITPFN_LORA_WORKFLOW"
if [[ "$action" == prepare ]]; then
  echo "Preparation complete. In an 'all' submission, the smoke array is already queued."
else
  echo "Gate ready: export CREDITPFN_LORA_GATE=$CREDITPFN_LORA_WORKFLOW"
fi
