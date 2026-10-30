#!/usr/bin/env bash
# Corpus eligibility at one fixed episode cap; benchmarking is a separate phase.
set -euo pipefail
cd "$(dirname "$0")/../.."
python -c 'from src.utils.followup_checks import require_gate; require_gate()'
case "${1:?Choose train or benchmark}" in
 train) export STAGES=train SEGMENT_MINUTES="${SEGMENT_MINUTES:-90}" CREDITPFN_AUTO_REQUEUE=1 ;;
 benchmark) exec bash scripts/slurm/run_benchmark.sh experiment8 ;;
 *) exit 2 ;;
esac
bash scripts/slurm/run_experiment.sh config/experiment8/pd.yaml
