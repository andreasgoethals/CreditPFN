#!/bin/bash -l
# =============================================================================
#  CreditPFN — slurm array: cross-model benchmark (PD / classifier)
# =============================================================================
#
#  ONE (model × test_dataset) PER ARRAY TASK. At the planned full corpus
#  scale (~3 000 test datasets × ~25 models) the cartesian product is
#  ~75 000 tasks; we
#  parallelise the heavy ones (XGBoost / CatBoost with per-fold
#  Optuna HPO) by running each in its own slurm task. Each task
#  writes its own per-method file under
#  ``results/PD/<method>/<run_name>_<timestamp>_<dataset>.csv``,
#  so concurrent tasks NEVER touch the same file (no locking, no
#  races). Aggregation downstream is `pd.read_csv` over a glob.
#
#  Compute the array size with:
#
#      python scripts/eval_pipeline.py --list-tasks track=pd
#
#  Submit with:
#
#      N=$(python scripts/eval_pipeline.py --list-tasks --tasks 16 track=pd)
#      sbatch --array=0-$((N - 1))%16 scripts/slurm/eval_pd.slurm
#
#  …or via scripts/slurm/run_full_pipeline.sh, which sizes the array and picks the
#  cluster from EVAL_CLUSTER / EVAL_PARTITIONS.
#
#  WHY MINDWELL AND NOT wICE (measured, run-8). The eval ran on wICE gpu_h100 +
#  gpu_a100 and reached an average concurrency of 0.20 — 3.6 GPU-h took 17.9 h of
#  wall-clock, and the gpu_a100 half never started at all. The VSC scheduler weights
#  fairshare on the user's walltime over the last SEVEN DAYS, so the ~120 GPU-h of
#  training submitted days earlier is exactly what sinks the eval's priority; and
#  wICE's 36 GPUs serve the whole university, with the cheapest of them (A100) the
#  most contended. Mindwell's 24 B200s gave us 15-21 concurrent on the same days.
#  Per GPU-minute: A100 141.7, B200 437.5, H100 569.4 — the partition that looked
#  cheap was the most expensive one.
#
#  CPU-only baselines ignore the GPU; one partition for everything is simpler than
#  routing per task, and the whole eval is under 10 GPU-h.
# =============================================================================

#SBATCH --job-name=creditpfn-pd-eval
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00  # MEASURED run-8: packed tasks took 28-40 min.
                         # A short request backfills into gaps between big
                         # jobs, which is the only lever we have on a
                         # partition where fairshare is against us.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16   # <= 24, the gpu_b200 per-GPU limit
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Propagate submission env so a bare cross-cluster `sbatch` doesn't hit
# Slurm's login-shell env retrieval ("user env retrieval failed requeued
# held"). run_full_pipeline.sh's CLI --export overrides this. (2026-07-08)
#SBATCH --export=ALL
#SBATCH --array=0-15%16   # 16 PACKED tasks (see EVAL_TASKS in
                          # run_full_pipeline.sh). Since 11-08-2026 a task is a
                          # BATCH of (model x dataset) cells of roughly equal
                          # estimated cost, not a single cell: the 209-task
                          # arrays this replaced averaged 0.73 concurrent jobs
                          # because each task computed for ~94 s and had to be
                          # scheduled separately. Surplus indices still exit 0
                          # cleanly via the soft no-op in eval_pipeline.py.
                          # Recompute with:
                          #   python scripts/eval_pipeline.py --list-tasks --tasks 16 track=pd
                          # …or submit via scripts/slurm/run_full_pipeline.sh,
                          # which sizes the array dynamically.

sleep $((RANDOM % 15 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Logs -> $VSC_DATA (NFS, backed up). Eval result CSVs are large/durable and
# go to project STAGING — resolved in src/utils/paths.py from
# $CREDITPFN_STAGING_ROOT / $TABPFN_STAGING_ROOT / the built-in
# /lustre1/project/stg_00211 default. wICE uses Lustre, where staging lives,
# so eval reads checkpoints + writes results directly (no stage-in needed).
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/output/logs"
mkdir -p "${CREDITPFN_STAGING_ROOT:-${CREDITPFN_OUTPUT_ROOT}}/output/results"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/output/logs/eval_pd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
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
echo "CreditPFN - EVAL (PD)"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Array job/task : ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Cluster        : ${SLURM_CLUSTER_NAME:-?}  partition=${SLURM_JOB_PARTITION:-?}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Account        : ${SLURM_JOB_ACCOUNT:-?}"
echo "Date           : $(date)"
echo "Log file       : ${LOG}"
echo "OUTPUT_ROOT    : ${CREDITPFN_OUTPUT_ROOT}"
echo "STAGING_ROOT   : ${CREDITPFN_STAGING_ROOT:-${TABPFN_STAGING_ROOT:-<built-in /lustre1/project/stg_00211>}}"
echo "CUDA_VISIBLE   : ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi || true
echo "=========================================="

# --tasks must MATCH the value --list-tasks was called with at submit time, or
# the two disagree about which cells belong to task i. run_full_pipeline.sh
# exports EVAL_TASKS so both sides read the same number.
# CREDITPFN_CONFIG / CREDITPFN_SPLIT_INDEX are exported by run_experiment.sh, exactly as for
# the training jobs. Eval NEEDS both: it rebuilds the held-out dataset draw from the training
# corpus block, so the wrong split index evaluates a checkpoint on tables it was trained on —
# silently, and in the direction that inflates every score. Unset means "the default sweep in
# config/train.yaml", so older submissions behave as they did.
EVAL_ARGS=()
[[ -n "${CREDITPFN_CONFIG:-}" ]] && EVAL_ARGS+=(--config "${CREDITPFN_CONFIG}")
[[ -n "${CREDITPFN_SPLIT_INDEX:-}" ]] && EVAL_ARGS+=(--split-index "${CREDITPFN_SPLIT_INDEX}")
echo "eval args: ${EVAL_ARGS[*]:-<none: default sweep>}"

python -u scripts/eval_pipeline.py \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --tasks "${EVAL_TASKS:-16}" \
    --log-path "$LOG" \
    "${EVAL_ARGS[@]}" \
    track=pd
