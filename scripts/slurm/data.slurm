#!/bin/bash
# =============================================================================
#  CreditPFN — slurm: data preprocessing
# =============================================================================
#
#  Stage 1 of the on-VSC pipeline. Reads `data/raw/{pd,lgd}/<id>.csv`
#  from $CREDITPFN_DATA_ROOT (= $VSC_SCRATCH/CreditPFN), runs the
#  five-stage pipeline, and writes `data/cached/{pd,lgd}/<id>/chunk_*.npz`
#  back to scratch.
#
#  CPU-only — uses the Genius `batch` partition. The data pipeline is
#  largely pandas + scikit-learn so multiple cores help.
#
#  Submit with:
#      sbatch scripts/slurm/data.slurm
#
#  Or chained from `full_pipeline_chain.slurm` (recommended).
# =============================================================================

#SBATCH --job-name=creditpfn-data
#SBATCH --cluster=genius
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=results/logs/slurm/data_%A.out
#SBATCH --error=results/logs/slurm/data_%A.err

set -euo pipefail

# Force unbuffered Python so logs stream live to slurm-out.
export PYTHONUNBUFFERED=1

# ---- Path roots (the only env-difference vs. local) -------------------- #
# Datasets live on scratch; durable outputs (logs/checkpoints/results) on data.
export CREDITPFN_DATA_ROOT="${VSC_SCRATCH}/CreditPFN"
export CREDITPFN_OUTPUT_ROOT="${VSC_DATA}/CreditPFN"

# ---- Conda env --------------------------------------------------------- #
export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate CreditPFN

# ---- Project location -------------------------------------------------- #
cd "${VSC_DATA}/CreditPFN"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/results/logs/slurm" \
         "${CREDITPFN_OUTPUT_ROOT}/logs"

echo "=========================================="
echo "CreditPFN — DATA PREPROCESSING"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Date           : $(date)"
echo "DATA_ROOT      : ${CREDITPFN_DATA_ROOT}"
echo "OUTPUT_ROOT    : ${CREDITPFN_OUTPUT_ROOT}"
echo "=========================================="

python -u scripts/data_pipeline.py \
    --log-path "${CREDITPFN_OUTPUT_ROOT}/logs/data_${SLURM_JOB_ID}.log"

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
