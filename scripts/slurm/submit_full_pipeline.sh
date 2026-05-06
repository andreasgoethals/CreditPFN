#!/bin/bash
# =============================================================================
#  CreditPFN — full pipeline submitter (data → train → eval) on VSC
# =============================================================================
#
#  Submits FOUR stages with `--dependency=afterok:` chaining so each
#  stage starts only after the previous one succeeds:
#
#    1. data.slurm                       (1 job ; genius batch CPU)
#    2. train_pd.slurm + train_lgd.slurm (arrays; wice gpu_h100)
#    3. eval_pd.slurm  + eval_lgd.slurm  (arrays; wice gpu_h100,
#                                         one (model × dataset) per
#                                         array task — heavy HPO
#                                         tasks parallelise cleanly)
#
#  Each stage writes ONE log file per slurm task to
#  `$VSC_DATA/CreditPFN/logs/<task>_<YYYYMMDD>_<HHMMSS>_j<JOBID>_a<TASKID>.log`.
#  Slurm's own `--output` is /dev/null in every .slurm — the bash
#  `exec >` redirection inside each script is the source of truth.
#
#  Usage (from a VSC login node):
#
#      bash scripts/slurm/submit_full_pipeline.sh
#
#  Optional knobs (override via env vars before invoking):
#
#      TRAIN_CONCURRENCY=4   # max in-flight train trials per array
#      EVAL_CONCURRENCY=32   # max in-flight eval (model × dataset) tasks
#      TRACKS="pd lgd"       # train + eval just one track if you want
# =============================================================================

set -euo pipefail

TRAIN_CONCURRENCY="${TRAIN_CONCURRENCY:-4}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-32}"
TRACKS="${TRACKS:-pd lgd}"

cd "$(dirname "$0")/../.."

echo "Submitting CreditPFN full pipeline …"
echo "  TRACKS             : ${TRACKS}"
echo "  TRAIN_CONCURRENCY  : ${TRAIN_CONCURRENCY}"
echo "  EVAL_CONCURRENCY   : ${EVAL_CONCURRENCY}"

# 1) Data preprocessing.
DATA_JID=$(sbatch --parsable scripts/slurm/data.slurm)
echo "  data               : ${DATA_JID}"

# 2) Training (one array job per track), each waiting on data.
declare -A TRAIN_JIDS=()
for TR in ${TRACKS}; do
    SCRIPT="scripts/slurm/train_${TR}.slurm"
    N=$(python scripts/train_pipeline.py --list-trials track="${TR}")
    JID=$(sbatch --parsable \
        --dependency="afterok:${DATA_JID}" \
        --array=0-$((N - 1))%"${TRAIN_CONCURRENCY}" \
        "${SCRIPT}")
    TRAIN_JIDS["$TR"]="${JID}"
    echo "  train ${TR}        : ${JID}  (array 0..$((N - 1)))"
done

# 3) Eval — one array job per track, gated on the matching training array.
for TR in ${TRACKS}; do
    SCRIPT="scripts/slurm/eval_${TR}.slurm"
    DEP="${TRAIN_JIDS[$TR]}"
    # Number of (model × dataset) tasks for this track. Note: this
    # depends on the manifest produced by training, but we only need
    # an upper bound at submission time — the script uses --task-index
    # to pick its assigned pair. We compute N from the *current* state
    # of the manifest (may grow as training finishes), then submit the
    # array gated on `afterok` so eval doesn't actually run before
    # training is done.
    N=$(python scripts/eval_pipeline.py --list-tasks track="${TR}" || echo 1)
    EVAL_JID=$(sbatch --parsable \
        --dependency="afterok:${DEP}" \
        --array=0-$((N - 1))%"${EVAL_CONCURRENCY}" \
        "${SCRIPT}")
    echo "  eval  ${TR}        : ${EVAL_JID}  (array 0..$((N - 1)), waits on ${DEP})"
done

echo
echo "All jobs submitted. Watch progress with:"
echo "    squeue --me --clusters=genius,wice"
echo
echo "Per-task logs:    \$VSC_DATA/CreditPFN/logs/<task>_<ts>_j<jid>_a<tid>.log"
echo "Manifests:        \$VSC_DATA/CreditPFN/manifests/<run_name>_<track>.csv"
echo "Trained ckpts:    \$VSC_DATA/CreditPFN/checkpoints/trained/<track>/"
echo "Benchmark CSVs:   \$VSC_DATA/CreditPFN/results/<TRACK>/<method>/"
