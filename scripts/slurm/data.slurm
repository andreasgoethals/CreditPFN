#!/bin/bash
# =============================================================================
#  CreditPFN — slurm: data preprocessing
# =============================================================================
#
#  Stage 1 of the on-VSC pipeline. Reads `data/raw/{pd,lgd}/<id>.csv`
#  from $CREDITPFN_DATA_ROOT (= $VSC_SCRATCH/CreditPFN), runs the
#  five-stage pipeline, and writes:
#    * data/cached/{pd,lgd}/<id>/chunk_*.npz  → DATA_ROOT (scratch)
#    * dedup/doubles_{track}_{pre,post}.csv   → OUTPUT_ROOT (durable)
#    * manifest_{pd,lgd}.csv                  → OUTPUT_ROOT (durable)
#
#  CPU-only — uses the wICE `batch` partition (IceLake).
#
#  Why wICE, not Genius: the train and eval stages run on wICE
#  (`gpu_h100`), and SLURM on VSC does NOT support cross-cluster
#  dependencies (Genius and wICE have separate controllers). Putting
#  every stage on wICE means the `afterok:` chain in
#  `submit_full_pipeline.sh` just works. If you ever need to put data
#  back on Genius, also break the dependency chain — submit data
#  first, wait for it to finish, then submit train/eval manually.
#
#  Submit with:
#      sbatch scripts/slurm/data.slurm
#  Or chained from `submit_full_pipeline.sh` (recommended).
#
#  All slurm-level output goes to /dev/null; the *real* log is a
#  single timestamped file `${OUTPUT_ROOT}/logs/data_<YYYYMMDD>_<HHMMSS>.log`
#  (slurm boilerplate + python output + final summary line all in
#  one place).
# =============================================================================

#SBATCH --job-name=creditpfn-data
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
export PYTHONUNBUFFERED=1

# Default: raw datasets + cache live on scratch (big, parallel, no
# backup — scratch policy is auto-cleanup of files unaccessed for 1
# month). If the user exports CREDITPFN_DATA_ROOT before submitting
# (e.g. to keep data on $VSC_DATA as a temporary workaround for a
# scratch purge), the value propagates through the submitter via
# `sbatch --export=ALL,...` and the `:-` default below honours it.
export CREDITPFN_DATA_ROOT="${CREDITPFN_DATA_ROOT:-${VSC_SCRATCH}/CreditPFN}"
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/data_${TS}_j${SLURM_JOB_ID}.log"

# Redirect everything from this point on (echo, nvidia-smi, python
# stdout/stderr) to a single log file. Slurm `--output=/dev/null`
# means slurm itself doesn't create a competing file.
exec > "$LOG" 2>&1

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate CreditPFN

cd "${VSC_DATA}/CreditPFN"

echo "=========================================="
echo "CreditPFN — DATA PREPROCESSING"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Date           : $(date)"
echo "Log file       : ${LOG}"
echo "DATA_ROOT      : ${CREDITPFN_DATA_ROOT}"
echo "OUTPUT_ROOT    : ${CREDITPFN_OUTPUT_ROOT}"
echo "=========================================="

python -u scripts/data_pipeline.py --log-path "$LOG"

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
