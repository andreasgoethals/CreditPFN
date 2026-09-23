#!/usr/bin/env bash
# Source before environment activation so import/setup failures reach the same log.
start_job_log() {
    local task="${1:?log task name required}"
    export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA:?}/CreditPFN}"
    mkdir -p "${CREDITPFN_OUTPUT_ROOT}/output/logs"
    LOG="${CREDITPFN_OUTPUT_ROOT}/output/logs/${task}_${SLURM_JOB_ID:?}_r${SLURM_RESTART_COUNT:-0}.log"
    export CREDITPFN_ACTIVE_LOG="$LOG"
    exec >> "$LOG" 2>&1
    trap 'rc=$?; echo "END exit_code=${rc} - $(date)"' EXIT
    echo "START task=${task} job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} - $(date)"
    echo "cluster=${SLURM_CLUSTER_NAME:-?} partition=${SLURM_JOB_PARTITION:-?} node=${SLURMD_NODENAME:-?}"
    echo "log=${LOG}"
}
