#!/bin/bash -l
# =============================================================================
#  CreditPFN — slurm: data preprocessing
# =============================================================================
#
#  Stage 1 of the on-VSC pipeline. Reads `data/raw/{pd,lgd}/<id>.csv`
#  from $CREDITPFN_DATA_ROOT (auto-detected by src.utils.paths when
#  unset — probes $VSC_SCRATCH/CreditPFN, $VSC_SCRATCH, $VSC_DATA/CreditPFN
#  in that order; whichever has CSVs wins), runs the
#  five-stage pipeline, and writes:
#    * data/processed/{pd,lgd}/<id>.sanitized.csv → DATA_ROOT (scratch)
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
# 64 GB: 0014.algorithmwatch is 159k×2986 cells (~3.8 GB as float64 in
# memory). Ward-linkage FeatureAgglomeration on that wide a numerical
# block needs ~4× the matrix size in scratch buffers — peak resident
# size during sanitize was ~25 GB in the 2026-05-19 run that 16 GB
# killed (job 66727474). 64 GB leaves headroom for the next-widest
# dataset (0011.loan_default, 105k×769) and any future addition.
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
export PYTHONUNBUFFERED=1

# OUTPUT_ROOT is always durable storage ($VSC_DATA on VSC) — it holds
# the log file this script is about to open, manifests, training
# checkpoints, eval CSVs, and notebook figures. Never on scratch.
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

# DATA_ROOT is NOT defaulted here. It's resolved inside
# `scripts/slurm/_activate_env.sh` from `config/data.yaml`'s
# `paths.data_source` knob (after conda activation). Precedence:
#   1. Explicit `CREDITPFN_DATA_ROOT` exported by the user (highest)
#   2. `cfg.paths.data_source` in config/data.yaml
#   3. VSC default ($VSC_SCRATCH/CreditPFN)
# Setting a default here would silently shadow the yaml (bug fix 2026-05-21).

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/data_${TS}_j${SLURM_JOB_ID}.log"

# Redirect everything from this point on (echo, nvidia-smi, python
# stdout/stderr) to a single log file. Slurm `--output=/dev/null`
# means slurm itself doesn't create a competing file.
exec > "$LOG" 2>&1

cd "${VSC_DATA}/CreditPFN"
source scripts/slurm/_activate_env.sh

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
