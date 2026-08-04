#!/bin/bash -l
# =============================================================================
#  CreditPFN — slurm array: cross-model benchmark (LGD / regressor)
#
#  Same logic as eval_pd.slurm — see that file's header.
# =============================================================================

#SBATCH --job-name=creditpfn-lgd-eval
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_h100   # run_full_pipeline.sh launches a SECOND eval
                               # array on gpu_a100 (via --partition override) so
                               # eval drains across BOTH wICE GPU pools.
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
#SBATCH --array=0-199%32   # generous upper bound; surplus tasks exit 0
                          # cleanly via the soft no-op in eval_pipeline.py.
                          # Recompute the tight bound with:
                          #   python scripts/eval_pipeline.py --list-tasks track=lgd
                          # …or just submit via scripts/slurm/run_full_pipeline.sh.
                          #
                          # `cpus-per-task=16` matches the documented gpu_h100
                          # per-GPU limit (VSC docs gpu_h100 row of the
                          # cpu_resource_limits_in_gpu_jobs table).

sleep $((RANDOM % 15 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Logs -> $VSC_DATA (NFS, backed up). Eval result CSVs are large/durable and
# go to project STAGING — resolved in src/utils/paths.py from
# $CREDITPFN_STAGING_ROOT / $TABPFN_STAGING_ROOT / the built-in
# /lustre1/project/stg_00211 default. wICE uses Lustre (where staging lives).
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs"
mkdir -p "${CREDITPFN_STAGING_ROOT:-${CREDITPFN_OUTPUT_ROOT}}/output/results"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/eval_lgd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
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

python -u scripts/eval_pipeline.py \
    --task-index "${SLURM_ARRAY_TASK_ID}" \
    --log-path "$LOG" \
    track=lgd
