#!/bin/bash
# =============================================================================
#  CreditPFN — full pipeline submitter (data → train → eval) on VSC
# =============================================================================
#
#  Submits THREE jobs with `--dependency=afterok:` chaining so each
#  stage starts only after the previous one succeeds:
#
#    1. data.slurm                                    (genius batch CPU)
#    2. train_pd.slurm + train_lgd.slurm  (arrays)    (wice gpu_h100)
#    3. eval.slurm  (one per track)                   (wice gpu_h100)
#
#  Each stage writes per-job logs under
#  $VSC_DATA/CreditPFN/results/logs/slurm/ so a crash mid-stage is
#  recoverable: re-run a specific .slurm by hand and the chain picks
#  up where it left off (skip-if-cached, append-not-overwrite).
#
#  Usage (from the repo root, on a VSC login node):
#
#      bash scripts/slurm/submit_full_pipeline.sh
#
#  Optional knobs (override via env vars before invoking):
#
#      CONCURRENCY=4   # max in-flight train trials per array
#      TRACKS="pd lgd" # train + eval just one track if you want
# =============================================================================

set -euo pipefail

CONCURRENCY="${CONCURRENCY:-4}"
TRACKS="${TRACKS:-pd lgd}"

cd "$(dirname "$0")/../.."

echo "Submitting CreditPFN full pipeline …"
echo "  TRACKS      : ${TRACKS}"
echo "  CONCURRENCY : ${CONCURRENCY}"

# 1) Data preprocessing.
DATA_JID=$(sbatch --parsable scripts/slurm/data.slurm)
echo "  data        : ${DATA_JID}"

# 2) Training (one array job per track), each waiting on data.
TRAIN_JIDS=()
for TR in ${TRACKS}; do
    SCRIPT="scripts/slurm/train_${TR}.slurm"
    N=$(python scripts/train_pipeline.py --list-trials track="${TR}")
    JID=$(sbatch --parsable \
        --dependency="afterok:${DATA_JID}" \
        --array=0-$((N - 1))%"${CONCURRENCY}" \
        "${SCRIPT}")
    echo "  train ${TR}   : ${JID}  (array 0..$((N - 1)))"
    TRAIN_JIDS+=("${JID}")
done

# 3) Eval — one job per track, each waiting on the matching training array.
for i in "${!TRAIN_JIDS[@]}"; do
    TR=$(echo "${TRACKS}" | awk -v i="$((i + 1))" '{print $i}')
    JID="${TRAIN_JIDS[$i]}"
    EVAL_JID=$(sbatch --parsable \
        --dependency="afterok:${JID}" \
        --export="ALL,TRACK=${TR}" \
        scripts/slurm/eval.slurm)
    echo "  eval  ${TR}   : ${EVAL_JID}  (waits on ${JID})"
done

echo
echo "All jobs submitted. Watch progress with:"
echo "    squeue --me --clusters=genius,wice"
echo
echo "Per-stage logs land under \$VSC_DATA/CreditPFN/results/logs/slurm/"
