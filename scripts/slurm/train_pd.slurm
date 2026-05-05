#!/bin/bash
# =============================================================================
#  CreditPFN — slurm array: continued pretraining (PD / classifier)
# =============================================================================
#
#  ONE TRIAL PER ARRAY TASK. Trial = one (base_checkpoint × learning_rate
#  × multi_chunk_policy) tuple from `cfg.tunable` in `config/train.yaml`.
#  The full grid size is 3 × 3 × 2 = 18 with the default cfg; compute it
#  fresh with:
#
#      python scripts/train_pipeline.py --list-trials track=pd
#
#  Submit with:
#
#      N=$(python scripts/train_pipeline.py --list-trials track=pd)
#      sbatch --array=0-$((N - 1))%4 scripts/slurm/train_pd.slurm
#
#  (`%4` caps in-flight tasks at 4. Tune for your queue. Drop the `%4`
#   to let slurm fan out as far as it can.)
#
#  The wICE `gpu_h100` partition gives us the newest GPU on KU Leuven
#  (NVIDIA H100 NVL, 96GB) — the best fit for TabPFN finetuning per
#  the user's "use the best GPUs" instruction. Fall back to `gpu_a100`
#  by changing the `--partition` line below if `gpu_h100` is busy.
# =============================================================================

#SBATCH --job-name=creditpfn-pd-train
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/logs/slurm/train_pd_%A_%a.out
#SBATCH --error=results/logs/slurm/train_pd_%A_%a.err
#SBATCH --array=0-17%4   # 18 trials, max 4 concurrent — recompute with --list-trials

# Stagger array starts to avoid I/O storms on shared scratch.
sleep $((RANDOM % 60 + 1))

set -euo pipefail

export PYTHONUNBUFFERED=1
export CREDITPFN_DATA_ROOT="${VSC_SCRATCH}/CreditPFN"
export CREDITPFN_OUTPUT_ROOT="${VSC_DATA}/CreditPFN"

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate CreditPFN

cd "${VSC_DATA}/CreditPFN"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/results/logs/slurm" \
         "${CREDITPFN_OUTPUT_ROOT}/logs/train" \
         "${CREDITPFN_OUTPUT_ROOT}/logs/runs" \
         "${CREDITPFN_OUTPUT_ROOT}/checkpoints/trained/pd"

echo "=========================================="
echo "CreditPFN — TRAINING (PD)"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Array task ID  : ${SLURM_ARRAY_TASK_ID}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Date           : $(date)"
echo "DATA_ROOT      : ${CREDITPFN_DATA_ROOT}"
echo "OUTPUT_ROOT    : ${CREDITPFN_OUTPUT_ROOT}"
nvidia-smi || true
echo "=========================================="

# Each trial appends one row to logs/runs/<run_name>_pd.csv.
python -u scripts/train_pipeline.py \
    --trial-index "${SLURM_ARRAY_TASK_ID}" \
    --log-path "${CREDITPFN_OUTPUT_ROOT}/logs/train_pd_${SLURM_ARRAY_JOB_ID}.log" \
    track=pd

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
