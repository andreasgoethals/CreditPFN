#!/bin/bash -l
# =============================================================================
#  CreditPFN — slurm array: continued pretraining (LGD / regressor)
#  TARGET: Mindwell cluster — NVIDIA B200 SXM6 (Blackwell), 192 GiB VRAM
#
#  Same structure as train_pd.slurm — see that file's header for the full
#  rationale (Mindwell-only training, storage tiers, pilot account, backfill).
#  LGD has a smaller corpus (6 train + 2 test datasets), so trials finish
#  faster; --time is set lower accordingly.
# =============================================================================

#SBATCH --job-name=creditpfn-lgd-train
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00   # RAISED from 4 h for run-8, and this one is not margin.
                          # LGD used to finish in minutes because it was silently
                          # undertrained: `target_total_steps` only trimmed epochs,
                          # never extended them, so LGD ran 800-3 200 steps against
                          # a 9 100 target. Now it reaches the full 20 000, which at
                          # LGD's measured epoch costs is v2.6 ~4.0 h, v3 ~3.7 h,
                          # tabicl ~0.6 h. 10 h covers the worst case with margin
                          # for the smaller `min_train_rows=5000` corpus, where
                          # fewer datasets mean more epochs for the same steps.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Propagate submission env so a bare cross-cluster `sbatch` doesn't hit
# Slurm's login-shell env retrieval ("user env retrieval failed requeued
# held"). run_full_pipeline.sh's CLI --export overrides this. (2026-07-08)
#SBATCH --export=ALL
#SBATCH --array=0-15%16   # 16 trials/track for run-8 — see train_pd.slurm for
                          # which sweep axes were dropped and why.

sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/output/logs" \
         "${CREDITPFN_OUTPUT_ROOT}/output/manifests" \
         "${CREDITPFN_OUTPUT_ROOT}/output/manifests/epochs/lgd"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/output/logs/train_lgd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
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
echo "CreditPFN - TRAINING (LGD) - Mindwell B200"
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

# Cross-cluster self-sequencing (see train_pd.slurm): wait for the wICE data
# stage's completion sentinel before reading processed CSVs. No-op unless
# CREDITPFN_WAIT_DATA is set (run_full_pipeline.sh sets it).
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

# PRE-FLIGHT SMOKE TEST (see train_pd.slurm) — the 'lgd' variant also checks
# the bar-distribution import that killed all 32 LGD trials on 2026-07-03.
python -c "from src.train.tabpfn_compat import smoke_test; smoke_test('lgd')" || {
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
         ${CREDITPFN_CONFIG:+--config "${CREDITPFN_CONFIG}"} track=lgd 2>/dev/null \
         | tail -1 || echo 0)"
if [[ "$N_ALL" =~ ^[0-9]+$ ]] && (( N_ALL > 0 )) && (( LAST > N_ALL - 1 )); then
    LAST=$(( N_ALL - 1 ))
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: trials ${FIRST}..${LAST} of ${N_ALL} (${TPT}/task)"
if (( FIRST > LAST )); then
    echo "Nothing to do for this task (chunk starts past the end of the grid)."
    exit 0
fi

# TabICL preflight — ONLY for trials using a tabicl base, so a missing
# `tabicl` install costs that family's trials rather than the whole grid.
# See train_pd.slurm for the full rationale. (2026-08-05)
TRIAL_FAMILY="$(python scripts/train_pipeline.py --trial-family "${FIRST}" ${CREDITPFN_CONFIG:+--config "${CREDITPFN_CONFIG}"} track=lgd 2>/dev/null || true)"
if [[ "${TRIAL_FAMILY}" == "tabicl" ]]; then
    python -c "from src.train.tabicl_compat import smoke_test; smoke_test('lgd')" || {
        echo "PREFLIGHT FAILED: tabicl import smoke test (chunk ${FIRST}..${LAST} uses a tabicl base)." >&2
        echo "  Fix: pip install 'tabicl[finetune]>=2.1.1,<3' INTO THE ENV THIS JOB ACTIVATES" >&2
        echo "       (the [finetune] extra carries 'transformers'; without it" >&2
        echo "        inference works and only TRAINING fails.)" >&2
        echo "       (see 'Active conda env' above — an interactive venv is NOT it)." >&2
        exit 1
    }
fi
# (Staging writability is probed in python — resolve_writable_staging_path —
#  before any training compute; falls back to $VSC_DATA if unwritable.)

# CREDITPFN_CONFIG / CREDITPFN_SPLIT_INDEX are exported by
# scripts/slurm/run_experiment.sh. Unset means "the default sweep in
# config/train.yaml", so every submission made before run-9 behaves as it did.
EXTRA_ARGS=()
[[ -n "${CREDITPFN_CONFIG:-}" ]] && EXTRA_ARGS+=(--config "${CREDITPFN_CONFIG}")
[[ -n "${CREDITPFN_SPLIT_INDEX:-}" ]] && EXTRA_ARGS+=(--split-index "${CREDITPFN_SPLIT_INDEX}")

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/.sentinels"

RC=0
for TRIAL in $(seq "$FIRST" "$LAST"); do
    echo "=============================================================="
    echo " TRIAL ${TRIAL}   $(date)"
    echo "=============================================================="
    # `if` suspends set -e for this command on purpose: one bad trial must not discard the
    # other trials packed into this task. The chunk still exits non-zero so the failure is
    # visible, and eval is gated on the per-TRIAL sentinels below, never on the task.
    if python -u scripts/train_pipeline.py \
        --trial-index "${TRIAL}" \
        --log-path "$LOG" \
        "${EXTRA_ARGS[@]}" \
        track=lgd
    then
        touch "${CREDITPFN_OUTPUT_ROOT}/.sentinels/train_ok_lgd_${TRIAL}"
    else
        RC=$?
        echo "TRIAL ${TRIAL} FAILED (rc=${RC}) — continuing with the rest of this chunk." >&2
    fi
done
exit "$RC"
