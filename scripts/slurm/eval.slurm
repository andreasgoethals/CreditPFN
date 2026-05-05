#!/bin/bash
# =============================================================================
#  CreditPFN — slurm: cross-model benchmark on the held-out test split
# =============================================================================
#
#  Stage 3 of the on-VSC pipeline. Reads:
#    * the trained-checkpoint manifest at logs/runs/<run_name>_<track>.csv
#    * each trained .ckpt under checkpoints/trained/<track>/
#    * each base .ckpt under checkpoints/
#    * the test chunks recomputed from `cfg.corpus`
#
#  Per (model × test_chunk × fold), fits and scores; writes long-format
#  CSVs to:
#
#      results/<TRACK>/<method>/<run_name>_<timestamp>.csv
#
#  GPU recommended (TabPFN inference is much faster on GPU). Uses
#  the same wICE gpu_h100 partition as training. If you only need
#  the classical baselines, set `cfg.baselines.enabled` accordingly
#  and switch to `--cluster=genius --partition=batch` for cheaper CPU
#  resources.
#
#  Submit with:
#
#      sbatch --export=TRACK=pd  scripts/slurm/eval.slurm
#      sbatch --export=TRACK=lgd scripts/slurm/eval.slurm
# =============================================================================

#SBATCH --job-name=creditpfn-eval
#SBATCH --cluster=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/logs/slurm/eval_%A.out
#SBATCH --error=results/logs/slurm/eval_%A.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export CREDITPFN_DATA_ROOT="${VSC_SCRATCH}/CreditPFN"
export CREDITPFN_OUTPUT_ROOT="${VSC_DATA}/CreditPFN"

export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
source activate CreditPFN

cd "${VSC_DATA}/CreditPFN"

TRACK="${TRACK:-pd}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/results/logs/slurm" \
         "${CREDITPFN_OUTPUT_ROOT}/results"

echo "=========================================="
echo "CreditPFN — EVAL (track=${TRACK})"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Date           : $(date)"
echo "DATA_ROOT      : ${CREDITPFN_DATA_ROOT}"
echo "OUTPUT_ROOT    : ${CREDITPFN_OUTPUT_ROOT}"
nvidia-smi || true
echo "=========================================="

python -u scripts/eval_pipeline.py \
    --log-path "${CREDITPFN_OUTPUT_ROOT}/logs/eval_${TRACK}_${SLURM_JOB_ID}.log" \
    track="${TRACK}"

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
