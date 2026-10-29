#!/usr/bin/env bash
# One invocation queues the complete gate or research sequence. Site admission
# can continue after this terminal closes; inspect the persistent submission log.
set -euo pipefail
cd "$(dirname "$0")/../.."
phase="${1:?usage: run_legacy_extensions.sh tests|research ID}"
identifier="${2:?extension ID required}"
[[ "$identifier" =~ ^[a-zA-Z0-9]{1,40}$ ]] || { echo 'Invalid extension ID'; exit 2; }
case "$phase" in tests|research) ;; *) echo 'Invalid phase'; exit 2 ;; esac
unset CREDITPFN_CONFIG CREDITPFN_LORA_WORKFLOW CREDITPFN_LORA_GATE CREDITPFN_FLOW_ID CREDITPFN_FLOW_PHASE
export CREDITPFN_USE_SCRATCH=0 PYTHONUNBUFFERED=1
source scripts/slurm/_activate_env.sh
folder="$(python -c 'import sys; from src.utils.legacy_design import root; print(root(sys.argv[1]))' "$identifier")"
mkdir -p "$folder"
log="$folder/submit_${phase}_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup python -u -m src.utils.legacy_submit "$phase" --id "$identifier" > "$log" 2>&1 < /dev/null &
echo "Submission process: $!; log: $log"
echo 'The submitter fills available site queue capacity without a running-task throttle.'
echo 'Keep this checkout and environment unchanged until all jobs finish.'
