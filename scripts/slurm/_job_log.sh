#!/usr/bin/env bash
# Source before environment activation so import/setup failures reach the same log.
start_job_log() {
    local task="${1:?log task name required}"
    export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA:?}/CreditPFN}"
    local group="${CREDITPFN_EXPERIMENT:-general}"
    # The generated config declares its output group; read it before Python is activated.
    if [[ -f "${CREDITPFN_CONFIG:-}" ]]; then
        local declared
        declared=$(sed -n -E 's/^  output_group: ([a-z0-9_]+)\r?$/\1/p' "$CREDITPFN_CONFIG")
        [[ -z "$declared" ]] || group="$declared"
    fi
    export CREDITPFN_EXPERIMENT="$group"
    # Second-campaign slugs mirror src/utils/paths.py CAMPAIGN2_GROUPS (tested).
    case "$group" in
        general|experiment0|reference|seed|domain_control|target_finetune|update_size_anchoring|scope|exposure|table_size|episode_size|context|transfer) ;;
        *) echo "Invalid output group: $group" >&2; return 2 ;;
    esac
    local scope="$group"
    mkdir -p "${CREDITPFN_OUTPUT_ROOT}/output CreditPFN/${scope}/logs"
    LOG="${CREDITPFN_OUTPUT_ROOT}/output CreditPFN/${scope}/logs/${task}_${SLURM_JOB_ID:?}_r${SLURM_RESTART_COUNT:-0}.log"
    export CREDITPFN_ACTIVE_LOG="$LOG"
    exec >> "$LOG" 2>&1
    trap 'rc=$?; if [[ "$rc" == 0 && -n "${CREDITPFN_OWNED_CACHE:-}" ]]; then "${CREDITPFN_SYSTEM_PYTHON:-/usr/bin/python3}" -m src.utils.job_cache clean "$CREDITPFN_OWNED_CACHE" || echo "CACHE CLEANUP PENDING: preserve ownership record for inspection"; fi; echo "END exit_code=${rc} - $(date)"' EXIT
    # Install before setup. Without explicit handlers, Bash's EXIT trap can
    # print the preceding command's zero status after an unhandled signal.
    # Training temporarily replaces USR1/TERM with checkpoint forwarding.
    trap 'echo "SIGNAL USR1 outside training - $(date)"; exit 138' USR1
    trap 'echo "SIGNAL TERM outside training - $(date)"; exit 143' TERM
    trap 'echo "SIGNAL INT - $(date)"; exit 130' INT
    echo "START task=${task} job=${SLURM_JOB_ID} restart=${SLURM_RESTART_COUNT:-0} - $(date)"
    echo "cluster=${SLURM_CLUSTER_NAME:-?} partition=${SLURM_JOB_PARTITION:-?} node=${SLURMD_NODENAME:-?}"
    echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-?} cpus_per_node=${SLURM_JOB_CPUS_PER_NODE:-?} mem_per_node_mib=${SLURM_MEM_PER_NODE:-?} gpus=${SLURM_JOB_GPUS:-none}"
    echo "log=${LOG}"
}
