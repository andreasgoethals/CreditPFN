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
#SBATCH --time=10:00:00   # SIZED FROM MEASUREMENT (2026-08-06), not guessed.
                          # Run-5 per-trial wall-clock at 50 epochs, full_pass:
                          #   v2.6 ~2.4 h | v3 ~1.4 h | tabicl ~0.5 h
                          # At the new 100 epochs the worst case (v2.6) is ~5 h,
                          # so 10 h is 2x margin.
                          # WHY THIS MATTERS: the old 48 h request was the likely
                          # cause of run-5's terrible concurrency — Slurm only
                          # ever ran 1-2 of the 24 requested array tasks, so 43.7
                          # GPU-h took 47 h of wall-clock. A short walltime is
                          # far easier to BACKFILL into gaps between big jobs, so
                          # it buys queue position. Re-measure if epochs, the row
                          # caps, or the corpus change.
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
#SBATCH --array=0-35%24   # 36 trials (3 bases [v3, v2.6, tabicl-v2] x 3 LRs
                          #            x 2 adapt-modes x 2 qf x 1 acc
                          #            x 1 pass-mode). %24 saturates all 24
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
# TabICL preflight — ONLY for trials that actually use a tabicl base (grid
# indices are family-ordered, but ask rather than assume). Scoping it this way
# means a missing/broken `tabicl` install fails that family's 16 trials
# instead of killing all 48, which matters because the tabicl bases are an
# addition to an otherwise-working TabPFN sweep. (2026-08-05)
TRIAL_FAMILY="$(python scripts/train_pipeline.py --trial-family "${SLURM_ARRAY_TASK_ID}" track=pd 2>/dev/null || true)"
if [[ "${TRIAL_FAMILY}" == "tabicl" ]]; then
    python -c "from src.train.tabicl_compat import smoke_test; smoke_test('pd')" || {
        echo "PREFLIGHT FAILED: tabicl import smoke test (trial ${SLURM_ARRAY_TASK_ID} uses a tabicl base)." >&2
        echo "  Fix: pip install 'tabicl[finetune]>=2.1.1,<3' INTO THE ENV THIS JOB ACTIVATES" >&2
        echo "       (the [finetune] extra carries 'transformers'; without it" >&2
        echo "        inference works and only TRAINING fails.)" >&2
        echo "       (see 'Active conda env' above — an interactive venv is NOT it)." >&2
        exit 1
    }
fi
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
