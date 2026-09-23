#!/bin/bash -l
# CreditPFN eval: use the named-config run_experiment.sh launcher.
# GPFS inputs; project weights/results; live DATA shards, consolidated post-run.
# Array sizing, concurrency and walltime are supplied by the launcher.
#SBATCH --job-name=creditpfn-lgd-eval
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --export=ALL
#SBATCH --array=0-0%1

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Logs -> $VSC_DATA (NFS, backed up). Eval result CSVs are large/durable and
# go to project STAGING — resolved in src/utils/paths.py from
# $CREDITPFN_STAGING_ROOT / $TABPFN_STAGING_ROOT / the built-in
# /lustre1/project/stg_00211 default. wICE uses Lustre (where staging lives).
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/output/logs"
mkdir -p "${CREDITPFN_STAGING_ROOT:-${CREDITPFN_OUTPUT_ROOT}}/output/results"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/output/logs/eval_lgd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
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
echo "CreditPFN - EVAL (LGD)"
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
# the two disagree about which cells belong to task i. run_experiment.sh
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
    track=lgd
