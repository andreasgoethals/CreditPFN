#!/bin/bash -l
# =============================================================================
#  CreditPFN — row-cap capacity probe (separate experiment, one B200, ~20 min)
# =============================================================================
#  Measures TRAINING peak GPU memory + step time at escalating in-context row
#  counts for every base checkpoint (see scripts/probe_row_cap.py). Run this
#  BEFORE a full sweep whenever the row caps in config/data.yaml change, to
#  verify the projected scaling (v3 ~linear in cells; v2.6 ~quadratic in rows)
#  on the actual hardware.
#
#  Submit:   sbatch scripts/slurm/probe_row_cap.slurm
#  Log:      $VSC_DATA/CreditPFN/logs/probe_row_cap_<ts>_j<jid>.log
# =============================================================================

#SBATCH --job-name=creditpfn-rowprobe
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Propagate the submission environment directly. WITHOUT this, a bare
# `sbatch scripts/slurm/probe_row_cap.slurm` submitted cross-cluster
# (Genius → Mindwell) makes Slurm reconstruct the login env by spawning the
# user's login shell on the target node — which fails on this cluster with
# "user env retrieval failed requeued held" (observed 2026-07-08, job
# 10285206). --export=ALL skips that path. run_full_pipeline.sh passes its
# own --export on the CLI, which overrides this line.
#SBATCH --export=ALL

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/probe_row_cap_${TS}_j${SLURM_JOB_ID}.log"
exec > "$LOG" 2>&1

cd "${VSC_DATA}/CreditPFN"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditPFN — ROW-CAP PROBE — Mindwell B200"
echo "Job ${SLURM_JOB_ID}  node=${SLURMD_NODENAME:-?}  $(date)"
nvidia-smi || true
echo "=========================================="

python -u scripts/probe_row_cap.py

echo "DONE — $(date)"
