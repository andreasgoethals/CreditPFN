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
#SBATCH --time=04:00:00   # Measured run-5: every LGD trial finished in <25 min
                          # at 50 epochs, so <1 h at 100. Short requests backfill
                          # far better — see train_pd.slurm for the rationale.
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
#SBATCH --array=0-17%24   # 18 trials (3 bases incl. tabicl-v2). See
                          # train_pd.slurm for the full rationale.

sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/manifests" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/epochs/lgd"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/train_lgd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
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
# TabICL preflight — ONLY for trials using a tabicl base, so a missing
# `tabicl` install costs that family's 16 trials rather than all 48.
# See train_pd.slurm for the full rationale. (2026-08-05)
TRIAL_FAMILY="$(python scripts/train_pipeline.py --trial-family "${SLURM_ARRAY_TASK_ID}" track=lgd 2>/dev/null || true)"
if [[ "${TRIAL_FAMILY}" == "tabicl" ]]; then
    python -c "from src.train.tabicl_compat import smoke_test; smoke_test('lgd')" || {
        echo "PREFLIGHT FAILED: tabicl import smoke test (trial ${SLURM_ARRAY_TASK_ID} uses a tabicl base)." >&2
        echo "  Fix: pip install 'tabicl[finetune]>=2.1.1,<3' INTO THE ENV THIS JOB ACTIVATES" >&2
        echo "       (the [finetune] extra carries 'transformers'; without it" >&2
        echo "        inference works and only TRAINING fails.)" >&2
        echo "       (see 'Active conda env' above — an interactive venv is NOT it)." >&2
        exit 1
    }
fi
# (Staging writability is probed in python — resolve_writable_staging_path —
#  before any training compute; falls back to $VSC_DATA if unwritable.)

python -u scripts/train_pipeline.py \
    --trial-index "${SLURM_ARRAY_TASK_ID}" \
    --log-path "$LOG" \
    track=lgd

# Reached ONLY on success (set -e) - per-task sentinel for the eval gate.
mkdir -p "${CREDITPFN_OUTPUT_ROOT}/.sentinels"
touch "${CREDITPFN_OUTPUT_ROOT}/.sentinels/train_ok_lgd_${SLURM_ARRAY_TASK_ID}"
