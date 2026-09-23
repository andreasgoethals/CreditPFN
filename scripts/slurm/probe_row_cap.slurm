#!/bin/bash -l
# Synthetic forward/backward capacity probe; confirm margins with real pilots.
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
#SBATCH --export=ALL

set -euo pipefail
cd "${VSC_DATA:?}/CreditPFN"
source scripts/slurm/_job_log.sh
start_job_log probe_row_cap
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
source scripts/slurm/_activate_env.sh
nvidia-smi || true
python -u scripts/probe_row_cap.py "$@"
