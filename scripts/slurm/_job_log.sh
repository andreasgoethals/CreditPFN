#!/usr/bin/env bash
# Source before environment activation so import/setup failures reach the same log.
start_job_log() {
    local task="${1:?log task name required}"
    export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA:?}/CreditPFN}"
    local group="${CREDITPFN_EXPERIMENT:-general}"
    # Public config folder numbers differ from immutable checkpoint campaign IDs.
    # Read the explicit scalar instead of inferring ownership from a directory.
    if [[ -f "${CREDITPFN_CONFIG:-}" ]]; then
        local declared
        declared=$(sed -n -E 's/^  output_group: (experiment[0-9]+)\r?$/\1/p; s/^experiment: (experiment[0-9]+)\r?$/\1/p' "$CREDITPFN_CONFIG")
        [[ -z "$declared" ]] || group="$declared"
    fi
    export CREDITPFN_EXPERIMENT="$group"
    case "$group" in general|experiment[0-8]) ;; *) echo "Invalid output group: $group" >&2; return 2 ;; esac
    local scope="$group"
    if [[ -f "$CREDITPFN_OUTPUT_ROOT/output CreditPFN/STUDY_LAYOUT.json" ]]; then
        # Bootstrap logging precedes Python activation; mapping matches study_layout.CAMPAIGNS.
        case "$group" in
            experiment1) scope=shared/main_grid ;;
            experiment3) scope=experiment7 ;;
            experiment4) scope=experiment10 ;;
            experiment6) scope=experiment8 ;;
            experiment7) scope=experiment9 ;;
            experiment8) scope=experiment6 ;;
        esac
    fi
    if [[ -n "${CREDITPFN_LEGACY_ID:-}" ]]; then
        [[ "$CREDITPFN_LEGACY_ID" =~ ^[a-zA-Z0-9]{1,40}$ ]] || { echo 'Invalid extension ID' >&2; return 2; }
        case "$group" in experiment0|experiment[4-8]) ;; *) echo 'Invalid extension experiment' >&2; return 2 ;; esac
        scope="$scope/extensions/$CREDITPFN_LEGACY_ID"
    elif [[ "$group" == experiment0 && -n "${CREDITPFN_LORA_WORKFLOW:-}" ]]; then
        [[ "$CREDITPFN_LORA_WORKFLOW" =~ ^[a-zA-Z0-9]+$ ]] || { echo 'Invalid LoRA workflow ID' >&2; return 2; }
        scope="experiment0/lora/$CREDITPFN_LORA_WORKFLOW"
    fi
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
