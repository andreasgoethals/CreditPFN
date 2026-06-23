#!/bin/bash -l
# =============================================================================
#  CreditPFN — slurm array: continued pretraining (PD / classifier)
#  Target: Mindwell cluster — NVIDIA B200 SXM6 Blackwell (192 GiB VRAM)
# =============================================================================
#
#  SAME PIPELINE AS train_pd.slurm, different cluster / partition / account.
#
#  Why B200 over H100:
#    * 192 GiB VRAM (vs 80 GiB on H100) → can run larger n_finetune_ctx
#      without OOM.  Current default (10 000 rows) was sized for H100;
#      on B200 you can safely push to 20 000–25 000.  Edit
#      `config/train.yaml` → `train.n_finetune_ctx_plus_query_samples`.
#    * Less queue pressure (fewer users; pilot programme).
#    * 3 nodes × 8 GPUs = 24 GPUs; with %12 concurrency you use half
#      the cluster for PD while LGD runs on wICE in parallel.
#
#  Data access:
#    $VSC_SCRATCH_LUSTRE1 (auto-set on Mindwell) mounts wICE's Lustre
#    scratch, so existing datasets don't need to be copied.  If you
#    prefer GPFS-local copies, upload to $VSC_SCRATCH on Mindwell and
#    set CREDITPFN_DATA_ROOT explicitly before submitting.
#
#  Account note:
#    `lp_mindwell_pilot` is the pilot account — check remaining credits:
#        sam-balance -A lp_mindwell_pilot
#    If exhausted, switch to a regular account and remove this note.
#
#  Submit from a Genius login node:
#
#      N=$(python scripts/train_pipeline.py --list-trials track=pd)
#      export CREDITPFN_STAGING_ROOT="/staging/leuven/stg_XXXXX/CreditPFN"
#      sbatch --array=0-$((N - 1))%12 scripts/slurm/train_pd_b200.slurm
# =============================================================================

#SBATCH --job-name=creditpfn-pd-b200
#SBATCH --cluster=mindwell
#SBATCH --account=lp_mindwell_pilot
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00   # use gpu_b200_long + 7d for very long full_pass runs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-23%12   # 24 trials. %12 = half of Mindwell's 24 B200 GPUs;
                          # raise to %24 to saturate the full cluster when idle.

# Stagger starts to avoid I/O storms.
sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1

# B200 / Blackwell: expandable_segments is recommended for large model workloads.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Logs + manifests → VSC_DATA (NFS-backed).
# Large artefacts (checkpoints) → staging (no purge, low inode budget).
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"
# Set CREDITPFN_STAGING_ROOT before submitting, or uncomment + fill in:
# export CREDITPFN_STAGING_ROOT="/staging/leuven/stg_XXXXX/CreditPFN"

# Cross-cluster data: $VSC_SCRATCH_LUSTRE1 mounts wICE Lustre on Mindwell.
# Falls back to VSC_DATA if Lustre mount is unavailable.
: "${CREDITPFN_DATA_ROOT:=${VSC_SCRATCH_LUSTRE1:-${VSC_DATA}}/CreditPFN}"
export CREDITPFN_DATA_ROOT

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/manifests" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/epochs/pd"
if [[ -n "${CREDITPFN_STAGING_ROOT:-}" ]]; then
    mkdir -p "${CREDITPFN_STAGING_ROOT}/checkpoints/trained/pd"
else
    mkdir -p "${CREDITPFN_OUTPUT_ROOT}/checkpoints/trained/pd"
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/train_pd_b200_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
exec > "$LOG" 2>&1

cd "${VSC_DATA}/CreditPFN"
source scripts/slurm/_activate_env.sh

echo "=========================================="
echo "CreditPFN — TRAINING (PD) — Mindwell B200"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Array task ID  : ${SLURM_ARRAY_TASK_ID}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Date           : $(date)"
echo "Log file       : ${LOG}"
echo "DATA_ROOT      : ${CREDITPFN_DATA_ROOT}"
echo "OUTPUT_ROOT    : ${CREDITPFN_OUTPUT_ROOT}"
echo "STAGING_ROOT   : ${CREDITPFN_STAGING_ROOT:-<not set — using OUTPUT_ROOT>}"
nvidia-smi || true
echo "=========================================="

python -u scripts/train_pipeline.py \
    --trial-index "${SLURM_ARRAY_TASK_ID}" \
    --log-path "$LOG" \
    track=pd

echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
