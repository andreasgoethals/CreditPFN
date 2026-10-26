#!/usr/bin/env bash
# LoRA is isolated from the completed experiment-1 grid.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset CREDITPFN_LORA_WORKFLOW
python -c 'from src.utils.lora_checks import require_gate; require_gate()'
case "${1:?Choose train or benchmark}" in
 train)
   for track in pd lgd; do
     STAGES=train SEGMENT_MINUTES="${SEGMENT_MINUTES:-90}" CREDITPFN_AUTO_REQUEUE=1 \
       bash scripts/slurm/run_experiment.sh "config/experiment8/$track.yaml"
   done ;;
 benchmark) bash scripts/slurm/run_benchmark.sh experiment8 ;;
 *) echo 'Usage: run_experiment8.sh train | benchmark'; exit 2 ;;
esac
