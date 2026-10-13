#!/usr/bin/env bash
# Episode size on one fixed large-table pool; the 8192-row anchor stays in experiment 6.
set -euo pipefail
cd "$(dirname "$0")/../.."
python -c 'from src.utils.followup_checks import require_gate; require_gate()'
case "${1:?Choose train or benchmark}" in
 train) export STAGES=train SEGMENT_MINUTES="${SEGMENT_MINUTES:-90}" CREDITPFN_AUTO_REQUEUE=1 ;;
 benchmark) exec bash scripts/slurm/run_benchmark.sh experiment7 ;;
 *) exit 2 ;;
esac
for rows in 2048 26000; do
  bash scripts/slurm/run_experiment.sh "config/experiment7/pd_rows${rows}.yaml"
done
