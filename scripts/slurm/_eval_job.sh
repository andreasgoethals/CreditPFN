#!/usr/bin/env bash
# Shared body for the PD/LGD foundation-model evaluation resource wrappers.
TRACK="${1:?evaluation track required}"
source scripts/slurm/_job_log.sh
start_job_log "eval_${TRACK}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
source scripts/slurm/_activate_env.sh
trap 'exit 143' TERM INT

# Python resolves project storage and creates result directories at the point of use.
python -u scripts/eval_pipeline.py \
    --config "${CREDITPFN_CONFIG:?use run_experiment.sh with a named config}" \
    --split-index "${CREDITPFN_SPLIT_INDEX:?dataset partition required}" \
    --task-index "${SLURM_ARRAY_TASK_ID:?}" --tasks "${EVAL_TASKS:-16}" \
    --log-path "$LOG" "track=$TRACK"
