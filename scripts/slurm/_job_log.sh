#!/usr/bin/env bash
# Source before environment activation so import/setup failures reach the same log.
start_job_log() {
    local task="${1:?log task name required}"
    export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA:?}/CreditPFN}"
    local group="${CREDITPFN_EXPERIMENT:-general}"
    case "${CREDITPFN_CONFIG:-}" in
        */config/experiment[0-3]/*|config/experiment[0-3]/*)
            group="$(basename "$(dirname "$CREDITPFN_CONFIG")")" ;;
    esac
    export CREDITPFN_EXPERIMENT="$group"
    case "$group" in general|experiment[0-3]) ;; *) echo "Invalid output group: $group" >&2; return 2 ;; esac
    mkdir -p "${CREDITPFN_OUTPUT_ROOT}/output/${group}/logs"
    LOG="${CREDITPFN_OUTPUT_ROOT}/output/${group}/logs/${task}_${SLURM_JOB_ID:?}_r${SLURM_RESTART_COUNT:-0}.log"
    export CREDITPFN_ACTIVE_LOG="$LOG"
    exec >> "$LOG" 2>&1
    trap 'rc=$?; echo "END exit_code=${rc} - $(date)"' EXIT
    echo "START task=${task} job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} - $(date)"
    echo "cluster=${SLURM_CLUSTER_NAME:-?} partition=${SLURM_JOB_PARTITION:-?} node=${SLURMD_NODENAME:-?}"
    echo "log=${LOG}"
}
