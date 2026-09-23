#!/usr/bin/env bash
# Shared body for the PD/LGD resource wrappers. Source from the repository root.
TRACK="${1:?training track required}"
source scripts/slurm/_job_log.sh
start_job_log "train_${TRACK}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
source scripts/slurm/_activate_env.sh
trap 'exit 143' TERM INT

EXTRA_ARGS=(--config "${CREDITPFN_CONFIG:?use run_experiment.sh with a named config}"
            --split-index "${CREDITPFN_SPLIT_INDEX:?dataset partition required}")
TPT="${CREDITPFN_TRIALS_PER_TASK:-1}"
[[ "$TPT" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid trials per task' >&2; exit 1; }
FIRST=$(( ${SLURM_ARRAY_TASK_ID:?} * TPT ))
LAST=$(( FIRST + TPT - 1 ))
# Let errors reach the log and stop the job; never turn a failed lookup into zero trials.
N_ALL="$(python scripts/train_pipeline.py --list-trials "${EXTRA_ARGS[@]}" "track=$TRACK")"
N_ALL="${N_ALL%$'\r'}"
[[ "$N_ALL" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid grid size: $N_ALL" >&2; exit 1; }
(( LAST < N_ALL )) || LAST=$(( N_ALL - 1 ))
echo "Task ${SLURM_ARRAY_TASK_ID}: trials ${FIRST}..${LAST} of ${N_ALL} (${TPT}/task)"
if (( FIRST > LAST )); then
    echo 'Nothing to do: task starts beyond the grid.'
    exit 0
fi

TRIAL_FAMILY="$(python scripts/train_pipeline.py --trial-family "$FIRST" "${EXTRA_ARGS[@]}" "track=$TRACK")"
TRIAL_FAMILY="${TRIAL_FAMILY%$'\r'}"
case "$TRIAL_FAMILY" in
    tabpfn|tabicl) python -c "from src.train.${TRIAL_FAMILY}_compat import smoke_test; smoke_test('$TRACK')" ;;
    *) echo "Unexpected model family: $TRIAL_FAMILY" >&2; exit 1 ;;
esac
nvidia-smi || true
source scripts/slurm/_run_train.sh
RC=0
for (( TRIAL=FIRST; TRIAL<=LAST; TRIAL++ )); do
    echo "TRIAL ${TRIAL} - $(date)"
    if run_training_python -u scripts/train_pipeline.py --trial-index "$TRIAL" \
        --log-path "$LOG" "${EXTRA_ARGS[@]}" "track=$TRACK"
    then
        echo "TRIAL ${TRIAL} completed."
    else
        RC=$?
        if [[ "$RC" == 75 ]]; then
            if [[ "${CREDITPFN_AUTO_REQUEUE:-0}" == 1 && "${SLURM_RESTART_COUNT:-0}" -lt "${CREDITPFN_MAX_REQUEUES:-20}" ]]; then
                scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" && exit 0
            fi
            echo 'Recovery saved. Resubmit this task with the same configuration.' >&2
            exit 75
        fi
        echo "TRIAL ${TRIAL} FAILED (rc=${RC}); continuing this task's remaining trials." >&2
    fi
done
exit "$RC"
