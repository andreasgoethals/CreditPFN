#!/usr/bin/env bash
# DiagScale has a training phase, then one benchmark phase for all checkpoints.
set -euo pipefail
cd "$(dirname "$0")/../.."
python -c 'from src.utils.followup_checks import require_gate; require_gate()'
case "${1:?Choose train or benchmark}" in
 train)
   for track in pd lgd; do
     STAGES=train SEGMENT_MINUTES="${SEGMENT_MINUTES:-90}" CREDITPFN_AUTO_REQUEUE=1 \
       bash scripts/slurm/run_experiment.sh "config/experiment5/$track.yaml"
   done ;;
 benchmark) bash scripts/slurm/run_benchmark.sh experiment5 ;;
 *) exit 2 ;;
esac
