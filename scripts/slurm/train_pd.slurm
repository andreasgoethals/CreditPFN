#!/bin/bash -l
# =============================================================================
#  CreditPFN — slurm array: continued pretraining (PD / classifier)
#  TARGET: Mindwell cluster — NVIDIA B200 SXM6 (Blackwell), 192 GiB VRAM
# =============================================================================
#
#  *** Continued pretraining of the TabPFN foundation models runs ONLY on
#      Mindwell B200. *** The B200's 192 GiB VRAM (2.4x the 80 GiB H100) lets
#      us train at a larger in-context size (config/data.yaml:
#      finetuning.max_rows_per_epoch — v3: 26 000, v2.6: 11 000 for PD's
#      two ensemble members; LGD scales these caps for eight members) to chase
#      Real-TabPFN's "more context -> bigger gains". Data prep and evaluation
#      stay on wICE (see data.slurm, eval_*.slurm).
#
#  ONE TRIAL PER ARRAY TASK. Trial = one
#    (base_checkpoint x learning_rate x use_lora x query_fraction
#     x accumulate x epoch_pass_mode) tuple from `cfg.tunable`.
#  Compute the array size with:
#      python scripts/train_pipeline.py --list-trials track=pd
#
#  Submit (from a Genius login node — Mindwell has no own login node):
#      N=$(python scripts/train_pipeline.py --list-trials track=pd)
#      sbatch --clusters=mindwell --array=0-$((N - 1))%24 scripts/slurm/train_pd.slurm
#  …or just `bash scripts/slurm/run_full_pipeline.sh`, which sizes the
#  array and targets the right cluster automatically.
#
#  STORAGE (see docs/VSC_GUIDE.md "Storage tiers"):
#    * Datasets + base ckpts + trained ckpts -> project staging
#      ($CREDITPFN_STAGING_ROOT, default /lustre1/project/stg_00211/CreditPFN).
#      Staging is on Lustre. Mindwell prefers GPFS for SUSTAINED I/O, but our
#      training only does a one-time startup read of the processed CSVs (then
#      caches in RAM) + a single checkpoint write at the end — not sustained
#      I/O — so direct Lustre access is fine. If a Mindwell job is ever killed
#      for I/O, switch to the stage-in/out pattern documented in VSC_GUIDE.
#    * Logs + manifests + per-epoch CSVs -> $VSC_DATA/CreditPFN (NFS, backed up).
#
#  ACCOUNT: lp_verbekelab (has Mindwell access — verified via
#  `sacctmgr -s show user $USER cluster=mindwell`). The free `lp_mindwell_pilot`
#  is gone now that Mindwell is in production. B200 credit weight is
#  437.5/GPU-min. Override at submit time with `sbatch --account=<acct> …`.
# =============================================================================

#SBATCH --job-name=creditpfn-pd-train
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00   # GPU partitions cap at 72h (no gpu_*_long on VSC).
                          # full_pass @ 50 epochs on the big PD datasets at the
                          # 26k-row B200 cap is the worst case; one_sample
                          # trials finish far sooner and free the slot. Tune
                          # down (e.g. 04:00:00 for one_sample-only) to gain
                          # backfill priority — shorter walltime => better queue
                          # position on VSC.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24   # max CPU cores per B200 GPU on gpu_b200 (192 cores / 8 GPUs)
#SBATCH --mem=120G           # B200 nodes have 1536 GiB / 8 GPUs ~= 190 GiB per GPU;
                             # 120G covers the larger row cap + the in-RAM clean cache.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Propagate submission env so a bare cross-cluster `sbatch` doesn't hit
# Slurm's login-shell env retrieval ("user env retrieval failed requeued
# held"). run_full_pipeline.sh's CLI --export overrides this. (2026-07-08)
#SBATCH --export=ALL
#SBATCH --array=0-31%24   # 32 trials (2 bases x 4 LRs x 2 LoRA x 1 qf x 1 acc
                          #            x 2 pass-modes). %24 saturates all 24
                          #            Mindwell B200 GPUs. Recompute via
                          #   python scripts/train_pipeline.py --list-trials track=pd
                          # Over-sizing is safe (surplus tasks exit 0 cleanly).

# Stagger array starts to avoid an I/O storm on the shared filesystem.
sleep $((RANDOM % 30 + 1))

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Blackwell: expandable_segments lets the CUDA allocator reclaim fragmented
# blocks — cheap insurance for the larger in-context tensors.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Logs + manifests + per-epoch CSVs -> $VSC_DATA (NFS, backed up, visible
# from every cluster). Trained checkpoints -> project staging (resolved in
# src/utils/paths.py from $CREDITPFN_STAGING_ROOT / $TABPFN_STAGING_ROOT /
# the built-in /lustre1/project/stg_00211 default).
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/logs" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/manifests" \
         "${CREDITPFN_OUTPUT_ROOT}/output/training/epochs/pd"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${CREDITPFN_OUTPUT_ROOT}/logs/train_pd_${TS}_j${SLURM_ARRAY_JOB_ID}_a${SLURM_ARRAY_TASK_ID}.log"
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
echo "CreditPFN - TRAINING (PD) - Mindwell B200"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Array job/task : ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Cluster        : ${SLURM_CLUSTER_NAME:-?}  partition=${SLURM_JOB_PARTITION:-?}"
echo "Node           : ${SLURMD_NODENAME:-?}"
echo "Account        : ${SLURM_JOB_ACCOUNT:-?}"
echo "Date           : $(date)"
echo "Log file       : ${LOG}"
echo "DATA_ROOT      : ${CREDITPFN_DATA_ROOT:-<auto: staging>}"
echo "OUTPUT_ROOT    : ${CREDITPFN_OUTPUT_ROOT}"
echo "STAGING_ROOT   : ${CREDITPFN_STAGING_ROOT:-${TABPFN_STAGING_ROOT:-<built-in /lustre1/project/stg_00211>}}"
echo "CUDA_VISIBLE   : ${CUDA_VISIBLE_DEVICES:-?}"
echo "PYTORCH_ALLOC  : ${PYTORCH_CUDA_ALLOC_CONF}"
nvidia-smi || true
echo "=========================================="

# Cross-cluster self-sequencing: when launched by run_full_pipeline.sh, wait for
# the data stage (running on wICE) to signal completion via a sentinel file on
# $VSC_DATA (NFS, visible from every cluster) before reading the processed CSVs.
# This lets the whole pipeline be fired at once without cross-cluster afterok.
# No-op unless CREDITPFN_WAIT_DATA is set.
if [[ -n "${CREDITPFN_WAIT_DATA:-}" ]]; then
    SENT="${CREDITPFN_OUTPUT_ROOT}/.sentinels/data_done"
    echo "Waiting up to 90 min for data sentinel: ${SENT}"
    for _i in $(seq 1 180); do
        [[ -f "${SENT}" ]] && { echo "data sentinel found after $((_i*30))s."; break; }
        sleep 30
    done
    if [[ ! -f "${SENT}" ]]; then
        echo "ERROR: data sentinel not found after 90 min — aborting (data stage failed?)." >&2
        exit 1
    fi
fi

# PRE-FLIGHT SMOKE TEST (seconds, no GPU compute) — verifies every import the
# trial will need + staging writability BEFORE burning B200 time. Added after
# the 2026-07-03 run lost 3 h to failures a 10-second check would have caught.
python -c "from src.train.tabpfn_compat import smoke_test; smoke_test('pd')" || {
    echo "PREFLIGHT FAILED: tabpfn import smoke test — aborting before training." >&2
    exit 1
}
# (Staging WRITABILITY is probed in python — resolve_writable_staging_path —
#  at save-path resolution, before any training compute; unwritable staging
#  now falls back to $VSC_DATA with a loud warning instead of Errno 13.)

python -u scripts/train_pipeline.py \
    --trial-index "${SLURM_ARRAY_TASK_ID}" \
    --log-path "$LOG" \
    track=pd

# Reached ONLY on success (set -e aborts on a python failure/exit 1) —
# the per-task success sentinel the eval gate requires before releasing eval.
mkdir -p "${CREDITPFN_OUTPUT_ROOT}/.sentinels"
touch "${CREDITPFN_OUTPUT_ROOT}/.sentinels/train_ok_pd_${SLURM_ARRAY_TASK_ID}"
