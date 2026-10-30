#!/usr/bin/env bash
# Submit all context-curve tasks in one bounded GPU array after CPU preparation.
set -euo pipefail
cd "$(dirname "$0")/../.."
export CREDITPFN_EXPERIMENT=experiment4
plan="${1:?Pass the prepared context plan path}"
count=$(python - "$plan" <<'PY'
import sys
from src.eval.context_curve import read_plan
from src.utils.followup_checks import require_gate
require_gate()
print(len(read_plan(sys.argv[1])['tasks']))
PY
)
[[ "$count" =~ ^[0-9]+$ ]] && (( count > 0 )) || exit 2
python -m src.utils.submit_bounded --slots 16 --limit 4 --cluster mindwell -- \
  sbatch --parsable --array="0-$((count-1))%4" --time="${WALLTIME:-04:00:00}" \
  scripts/slurm/context_benchmark.slurm "$plan"
