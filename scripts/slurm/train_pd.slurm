#!/bin/bash -l
# CreditPFN train: use the named-config run_experiment.sh launcher.
# GPFS inputs; project weights/results; live DATA shards, consolidated post-run.
# Array sizing, concurrency and walltime are supplied by the launcher.
#SBATCH --job-name=creditpfn-pd-train
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --array=0-0%1

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Blackwell: expandable_segments lets the CUDA allocator reclaim fragmented
# blocks — cheap insurance for the larger in-context tensors.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Logs + manifests + per-epoch CSVs -> $VSC_DATA (NFS, backed up, visible
# from every cluster). Trained checkpoints -> project staging (resolved in
# src/utils/paths.py from $CREDITPFN_STAGING_ROOT / $TABPFN_STAGING_ROOT /
# the built-in /lustre1/project/stg_00211 default).
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/output/logs" \
         "${CREDITPFN_OUTPUT_ROOT}/output/manifests" \
         "${CREDITPFN_OUTPUT_ROOT}/output/manifests/epochs/pd"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/output/logs/train_pd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
exec > "$LOG" 2>&1

_log_exit() {
    rc=$?
    echo "=========================================="
    if [[ "$rc" -eq 0 ]]; then
        echo "END status=OK - $(date)"
    else
        echo "END status=FAILED exit_code=${rc} - $(date)"
    fi
    echo "=========================================="
}
trap _log_exit EXIT
trap 'exit 143' TERM INT

cd "${VSC_DATA}/CreditPFN"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditPFN - TRAINING (PD) - Mindwell B200"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Array job/task : ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Cluster        : ${SLURM_CLUSTER_NAME:-?}  partition=${SLURM_JOB_PARTITION:-?}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Account        : ${SLURM_JOB_ACCOUNT:-?}"
echo "Date           : $(date)"
echo "Log file       : ${LOG}"
echo "DATA_ROOT      : ${CREDITPFN_DATA_ROOT:-<auto: staging>}"
echo "OUTPUT_ROOT    : ${CREDITPFN_OUTPUT_ROOT}"
echo "STAGING_ROOT   : ${CREDITPFN_STAGING_ROOT:-${TABPFN_STAGING_ROOT:-<built-in /lustre1/project/stg_00211>}}"
echo "CUDA_VISIBLE   : ${CUDA_VISIBLE_DEVICES:-?}"
echo "PYTORCH_ALLOC  : ${PYTORCH_CUDA_ALLOC_CONF}"
nvidia-smi || true
echo "=========================================="

# Cross-cluster self-sequencing: when launched by run_full_pipeline.sh, wait for
# the data stage (running on wICE) to signal completion via a sentinel file on
# $VSC_DATA (NFS, visible from every cluster) before reading the processed CSVs.
# This lets the whole pipeline be fired at once without cross-cluster afterok.
# No-op unless CREDITPFN_WAIT_DATA is set.
if [[ -n "${CREDITPFN_WAIT_DATA:-}" ]]; then
    SENT="${CREDITPFN_OUTPUT_ROOT}/.sentinels/data_done"
    echo "Waiting up to 90 min for data sentinel: ${SENT}"
    for _i in $(seq 1 180); do
        [[ -f "${SENT}" ]] && { echo "data sentinel found after $((_i*30))s."; break; }
        sleep 30
    done
    if [[ ! -f "${SENT}" ]]; then
        echo "ERROR: data sentinel not found after 90 min — aborting (data stage failed?)." >&2
        exit 1
    fi
fi

# PRE-FLIGHT SMOKE TEST (seconds, no GPU compute) — verifies every import the
# trial will need + staging writability BEFORE burning B200 time. Added after
# the 2026-07-03 run lost 3 h to failures a 10-second check would have caught.
python -c "from src.train.tabpfn_compat import smoke_test; smoke_test('pd')" || {
    echo "PREFLIGHT FAILED: tabpfn import smoke test — aborting before training." >&2
    exit 1
}
# ---- this task's chunk of the trial grid -------------------------------------
# CREDITPFN_TRIALS_PER_TASK (default 1) trials run SEQUENTIALLY in this one job, so a 768-cell
# campaign fits under the 500 submitted-job ceiling and pays startup once per chunk instead of
# once per trial. run_experiment.sh sizes --time to match and asserts that a chunk never
# straddles two model families (the grid is base-major, so it does not when C divides the
# per-base block).
TPT="${CREDITPFN_TRIALS_PER_TASK:-1}"
FIRST=$(( SLURM_ARRAY_TASK_ID * TPT ))
LAST=$(( FIRST + TPT - 1 ))
N_ALL="$(python scripts/train_pipeline.py --list-trials \
         ${CREDITPFN_CONFIG:+--config "${CREDITPFN_CONFIG}"} track=pd 2>/dev/null \
         | tail -1 || echo 0)"
if [[ "$N_ALL" =~ ^[0-9]+$ ]] && (( N_ALL > 0 )) && (( LAST > N_ALL - 1 )); then
    LAST=$(( N_ALL - 1 ))
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: trials ${FIRST}..${LAST} of ${N_ALL} (${TPT}/task)"
if (( FIRST > LAST )); then
    echo "Nothing to do for this task (chunk starts past the end of the grid)."
    exit 0
fi

# TabICL preflight — ONLY for trials that actually use a tabicl base (grid
# indices are family-ordered, but ask rather than assume). Scoping it this way
# means a missing/broken `tabicl` install fails that family's 16 trials
# instead of killing all 48, which matters because the tabicl bases are an
# addition to an otherwise-working TabPFN sweep. (2026-08-05)
TRIAL_FAMILY="$(python scripts/train_pipeline.py --trial-family "${FIRST}" ${CREDITPFN_CONFIG:+--config "${CREDITPFN_CONFIG}"} track=pd 2>/dev/null || true)"
if [[ "${TRIAL_FAMILY}" == "tabicl" ]]; then
    python -c "from src.train.tabicl_compat import smoke_test; smoke_test('pd')" || {
        echo "PREFLIGHT FAILED: tabicl import smoke test (chunk ${FIRST}..${LAST} uses a tabicl base)." >&2
        echo "  Fix: pip install 'tabicl[finetune]>=2.1.1,<3' INTO THE ENV THIS JOB ACTIVATES" >&2
        echo "       (the [finetune] extra carries 'transformers'; without it" >&2
        echo "        inference works and only TRAINING fails.)" >&2
        echo "       (see 'Active conda env' above — an interactive venv is NOT it)." >&2
        exit 1
    }
fi
# (Staging WRITABILITY is probed in python — resolve_writable_staging_path —
#  at save-path resolution, before any training compute; unwritable staging
#  now falls back to $VSC_DATA with a loud warning instead of Errno 13.)

# CREDITPFN_CONFIG / CREDITPFN_SPLIT_INDEX are exported by
# scripts/slurm/run_experiment.sh. Unset means "the default sweep in
# config/train.yaml", so every submission made before run-9 behaves as it did.
EXTRA_ARGS=()
[[ -n "${CREDITPFN_CONFIG:-}" ]] && EXTRA_ARGS+=(--config "${CREDITPFN_CONFIG}")
[[ -n "${CREDITPFN_SPLIT_INDEX:-}" ]] && EXTRA_ARGS+=(--split-index "${CREDITPFN_SPLIT_INDEX}")

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/.sentinels"

source scripts/slurm/_run_train.sh

RC=0
for TRIAL in $(seq "$FIRST" "$LAST"); do
    echo "=============================================================="
    echo " TRIAL ${TRIAL}   $(date)"
    echo "=============================================================="
    # `if` suspends set -e for this command on purpose: one bad trial must not discard the
    # other trials packed into this task. The chunk still exits non-zero so the failure is
    # visible, and eval is gated on the per-TRIAL sentinels below, never on the task.
    if run_training_python -u scripts/train_pipeline.py \
        --trial-index "${TRIAL}" \
        --log-path "$LOG" \
        "${EXTRA_ARGS[@]}" \
        track=pd
    then
        touch "${CREDITPFN_OUTPUT_ROOT}/.sentinels/train_ok_pd_${TRIAL}"
    else
        RC=$?
        if [[ "$RC" == 75 ]]; then
            if [[ "${CREDITPFN_AUTO_REQUEUE:-0}" == 1 && "${SLURM_RESTART_COUNT:-0}" -lt "${CREDITPFN_MAX_REQUEUES:-20}" ]]; then
                scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" && exit 0
            fi
            echo "Recovery saved. Resubmit this task with the same configuration." >&2
            exit 75
        fi
        echo "TRIAL ${TRIAL} FAILED (rc=${RC}) — continuing with the rest of this chunk." >&2
    fi
done
exit "$RC"
