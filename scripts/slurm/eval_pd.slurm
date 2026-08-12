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
#      N=$(python scripts/eval_pipeline.py --list-tasks track=pd)
#      sbatch --array=0-$((N - 1))%32 scripts/slurm/eval_pd.slurm
#
#  TabPFN tasks need GPU (wICE gpu_h100). Boosting / linear baselines
#  could in principle run on cheaper CPU partitions, but mixing
#  partitions per array task is awkward; we use one GPU partition for
#  everything and trust slurm's scheduling. CPU-only baselines just
#  ignore the GPU.
# =============================================================================

#SBATCH --job-name=creditpfn-pd-eval
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_h100   # run_full_pipeline.sh launches a SECOND eval
                               # array on gpu_a100 (via --partition override) so
                               # eval drains across BOTH wICE GPU pools (20 H100
                               # + 16 A100 = 36 GPUs). Both partitions cooperate
                               # safely: each task skips already-scored pairs.
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00  # raised from 2h (2026-07-13): v2.6 x 100k-row
                          # datasets on the A100 pool blew 2h (8 walltime
                          # kills in the Jul-12 run); 5h + the 50k v2.6 row
                          # cap gives ample margin.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
python -u scripts/eval_pipeline.py \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --tasks "${EVAL_TASKS:-16}" \
    --log-path "$LOG" \
    track=pd
