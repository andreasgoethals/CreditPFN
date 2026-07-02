#!/bin/bash
# =============================================================================
#  CreditPFN — multi-cluster pipeline submitter (data -> train -> eval)
# =============================================================================
#
#  THREE stages, deliberately split across TWO clusters:
#
#    1. data.slurm                        CPU   — wICE `batch`        (writes
#                                                 processed CSVs to staging)
#    2. train_pd.slurm + train_lgd.slurm  GPU   — Mindwell `gpu_b200` (continued
#                                                 pretraining — Mindwell ONLY)
#    3. eval_pd.slurm  + eval_lgd.slurm   GPU   — wICE `gpu_h100`     (benchmark)
#
#  *** Why no single afterok chain? ***  VSC runs Genius, wICE and Mindwell as
#  separate Slurm clusters selected by `--clusters=`. Slurm `--dependency`
#  (afterok/afterany) does NOT work across clusters. Since training is on
#  Mindwell and data/eval are on wICE, the stages CANNOT be auto-chained.
#  This submitter therefore submits each stage independently and relies on:
#    * data -> train : datasets persist in project staging (visible from both
#      clusters); training reads them. Run data FIRST and let it finish before
#      training reads (the submitter submits data, then training — if you run
#      the whole thing cold, give data a head start or submit STAGES=data first).
#    * train -> eval : the eval pipeline is ROBUST — it scores whatever trained
#      checkpoints exist in staging and skips the rest, so a re-run picks up
#      late-finishing trials. Submitting eval now is safe; re-submit after
#      training completes to score the stragglers.
#
#  Datasets, base checkpoints, trained checkpoints and benchmark results all
#  live in project STAGING (default /lustre1/project/stg_00211/CreditPFN,
#  overridable via $CREDITPFN_STAGING_ROOT or $TABPFN_STAGING_ROOT). Logs,
#  manifests and per-epoch CSVs live on $VSC_DATA (NFS, backed up). See
#  docs/VSC_GUIDE.md.
#
#  Usage (from a Genius login node):
#      bash scripts/slurm/submit_full_pipeline.sh
#
#  Knobs (env vars):
#      STAGES="data train eval"          # which stages to submit (any subset/order)
#      TRACKS="pd lgd"                   # which tracks to train/eval
#      TRAIN_CONCURRENCY=24              # max in-flight Mindwell train tasks (24 B200)
#      EVAL_CONCURRENCY=32               # max in-flight eval tasks PER partition
#      EVAL_PARTITIONS="gpu_h100 gpu_a100"  # eval runs a SEPARATE array per
#                                           # partition; the task range is split
#                                           # across them so wICE's 20 H100 + 16
#                                           # A100 drain eval in parallel. Set to
#                                           # "gpu_h100" for a single array.
#      CONDA_ENV=CreditPFN
# =============================================================================

set -euo pipefail

STAGES="${STAGES:-data train eval}"
TRACKS="${TRACKS:-pd lgd}"
TRAIN_CONCURRENCY="${TRAIN_CONCURRENCY:-24}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-32}"
EVAL_PARTITIONS="${EVAL_PARTITIONS:-gpu_h100 gpu_a100}"
CONDA_ENV="${CONDA_ENV:-CreditPFN}"
# Override the Mindwell training account when the built-in `lp_mindwell_pilot`
# (the free pilot) is no longer valid for you — the pilot closed when Mindwell
# went to production. Find yours with `sam-balance`, then run e.g.
#   TRAIN_ACCOUNT=lp_yourproject STAGES=train bash scripts/slurm/submit_full_pipeline.sh
# Empty → use the account hard-coded in train_{pd,lgd}.slurm.
TRAIN_ACCOUNT="${TRAIN_ACCOUNT:-}"

cd "$(dirname "$0")/../.."

# ---------------------------------------------------------------------------
# Activate the project conda env (the login-node python lacks omegaconf etc.).
# ---------------------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    if ! source scripts/slurm/_activate_env.sh; then
        echo "ERROR: could not activate conda env '${CONDA_ENV}'." >&2
        echo "       Run 'source activate ${CONDA_ENV}' and re-invoke." >&2
        exit 1
    fi
fi
if ! python -c "import omegaconf, src.train.corpus" 2>/dev/null; then
    echo "ERROR: the '${CONDA_ENV}' env is missing project dependencies." >&2
    echo "       Re-install with: pip install -e \".[dev]\"" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve the storage roots ONCE and propagate them to every spawned job.
# Slurm does not inherit env vars across clusters by default, so we pass them
# on each `sbatch --export`. data_source -> staging by default (datasets are
# the largest files; project storage is their canonical home).
# ---------------------------------------------------------------------------
read -r CREDITPFN_DATA_ROOT CREDITPFN_OUTPUT_ROOT CREDITPFN_STAGING_ROOT < <(
    python -c "
from omegaconf import OmegaConf
from src.utils.paths import apply_data_source_from_cfg, get_roots
apply_data_source_from_cfg(OmegaConf.load('config/data.yaml'))
r = get_roots()
print(r['data_root'], r['output_root'], r['staging_root'])
"
)
SBATCH_EXPORT="ALL,CREDITPFN_DATA_ROOT=${CREDITPFN_DATA_ROOT},CREDITPFN_OUTPUT_ROOT=${CREDITPFN_OUTPUT_ROOT},CREDITPFN_STAGING_ROOT=${CREDITPFN_STAGING_ROOT}"

echo "Submitting CreditPFN pipeline …"
echo "  STAGES                : ${STAGES}"
echo "  TRACKS                : ${TRACKS}"
echo "  TRAIN_CONCURRENCY     : ${TRAIN_CONCURRENCY}  (Mindwell gpu_b200)"
echo "  EVAL_CONCURRENCY      : ${EVAL_CONCURRENCY}  per partition"
echo "  EVAL_PARTITIONS       : ${EVAL_PARTITIONS}  (wICE)"
echo "  CREDITPFN_DATA_ROOT   : ${CREDITPFN_DATA_ROOT}"
echo "  CREDITPFN_OUTPUT_ROOT : ${CREDITPFN_OUTPUT_ROOT}"
echo "  CREDITPFN_STAGING_ROOT: ${CREDITPFN_STAGING_ROOT}"
echo

strip_cluster_suffix() { echo "${1%%;*}"; }   # `<jid>;<cluster>` -> `<jid>`

DATA_JID=""

# ---------------------------------------------------------------------------
# Stage 1 — DATA preprocessing (wICE CPU). Writes processed CSVs to staging.
# ---------------------------------------------------------------------------
if [[ " ${STAGES} " == *" data "* ]]; then
    if [[ -d "${CREDITPFN_DATA_ROOT}/data/raw" ]]; then
        n_pd=$(find "${CREDITPFN_DATA_ROOT}/data/raw/pd"  -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l || echo 0)
        n_lgd=$(find "${CREDITPFN_DATA_ROOT}/data/raw/lgd" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l || echo 0)
        if [[ "${n_pd}" -eq 0 && "${n_lgd}" -eq 0 ]]; then
            echo "ERROR: no raw CSVs under ${CREDITPFN_DATA_ROOT}/data/raw/." >&2
            echo "       Upload them to project staging and re-submit." >&2
            exit 1
        fi
        echo "  raw datasets found    : pd=${n_pd}  lgd=${n_lgd}"
    fi
    DATA_JID=$(strip_cluster_suffix "$(sbatch --parsable --export="${SBATCH_EXPORT}" scripts/slurm/data.slurm)")
    echo "  data (wICE)           : ${DATA_JID}"
fi

# ---------------------------------------------------------------------------
# Stage 2 — TRAINING (Mindwell B200). Continued pretraining runs ONLY here.
# Cross-cluster dep on data is impossible; data must be complete first.
# ---------------------------------------------------------------------------
if [[ " ${STAGES} " == *" train "* ]]; then
    if [[ -n "${DATA_JID}" ]]; then
        echo "  NOTE: training cannot afterok-depend on the wICE data job"
        echo "        (cross-cluster). Ensure data job ${DATA_JID} FINISHES"
        echo "        before the Mindwell training tasks start reading."
    fi
    ACCT_FLAG=()
    [[ -n "${TRAIN_ACCOUNT}" ]] && ACCT_FLAG=(--account="${TRAIN_ACCOUNT}")
    for TR in ${TRACKS}; do
        N=$(python scripts/train_pipeline.py --list-trials track="${TR}")
        JID=$(strip_cluster_suffix "$(sbatch --parsable \
            --export="${SBATCH_EXPORT}" \
            "${ACCT_FLAG[@]}" \
            --array=0-$((N - 1))%"${TRAIN_CONCURRENCY}" \
            "scripts/slurm/train_${TR}.slurm")")
        echo "  train ${TR} (Mindwell) : ${JID:-<FAILED>}  (array 0..$((N - 1)); account=${TRAIN_ACCOUNT:-<slurm default>})"
    done
fi

# ---------------------------------------------------------------------------
# Stage 3 — EVAL (wICE H100). Robust: scores whatever checkpoints exist.
# ---------------------------------------------------------------------------
if [[ " ${STAGES} " == *" eval "* ]]; then
    for TR in ${TRACKS}; do
        if [[ "$TR" == "pd" ]]; then PLANNED=$(python scripts/train_pipeline.py --list-trials track=pd 2>/dev/null || echo 0)
        else PLANNED=$(python scripts/train_pipeline.py --list-trials track=lgd 2>/dev/null || echo 0); fi
        UPPER_N=$(python -c "
import sys; sys.path.insert(0, '.')
from omegaconf import OmegaConf
eval_cfg = OmegaConf.load('config/eval.yaml')
train_cfg = OmegaConf.load(eval_cfg.train_cfg_path)
n_baselines = sum(1 for b in eval_cfg.baselines.enabled if b != 'tabpfn-untuned')
if '${TR}' == 'pd' and 'linreg' in eval_cfg.baselines.enabled: n_baselines -= 1
if '${TR}' == 'lgd' and 'logreg' in eval_cfg.baselines.enabled: n_baselines -= 1
n_untuned = len(train_cfg.tunable.classifier_base_paths if '${TR}' == 'pd' else train_cfg.tunable.regressor_base_paths)
n_planned = int('${PLANNED}' or 0)
from src.train.corpus import split_from_cfg
split = split_from_cfg(train_cfg, track='${TR}')
n_test = len({c.dataset_id for c in split.test})
print((n_baselines + n_untuned + n_planned) * max(1, n_test))
" 2>/dev/null || echo 1)
        UPPER_N=${UPPER_N:-1}; [[ "$UPPER_N" -lt 1 ]] && UPPER_N=1
        # Split the [0, UPPER_N) task range into contiguous chunks, one per
        # eval partition, so wICE's H100 + A100 pools drain eval in parallel
        # with NO overlap (each pair is owned by exactly one partition).
        # shellcheck disable=SC2206
        PARTS=(${EVAL_PARTITIONS}); K=${#PARTS[@]}
        CHUNK=$(( (UPPER_N + K - 1) / K ))   # ceil(UPPER_N / K)
        for i in "${!PARTS[@]}"; do
            P="${PARTS[$i]}"
            LO=$(( i * CHUNK ))
            HI=$(( (i + 1) * CHUNK - 1 ))
            [[ "$HI" -ge "$UPPER_N" ]] && HI=$(( UPPER_N - 1 ))
            [[ "$LO" -gt "$HI" ]] && continue   # more partitions than tasks
            JID=$(strip_cluster_suffix "$(sbatch --parsable \
                --export="${SBATCH_EXPORT}" \
                --partition="${P}" \
                --array="${LO}-${HI}%${EVAL_CONCURRENCY}" \
                "scripts/slurm/eval_${TR}.slurm")")
            echo "  eval  ${TR} (wICE ${P}) : ${JID}  (array ${LO}..${HI})"
        done
    done
    echo
    echo "  NOTE: eval scores the checkpoints present in staging NOW. Re-run"
    echo "        this with STAGES=eval after training finishes to pick up"
    echo "        late-finishing trials (already-scored pairs are skipped)."
fi

echo
echo "Watch progress across both clusters with:"
echo "    squeue --me --clusters=wice,mindwell"
echo
echo "Datasets:         ${CREDITPFN_STAGING_ROOT}/data/processed/<track>/"
echo "Trained ckpts:    ${CREDITPFN_STAGING_ROOT}/checkpoints/trained/<track>/"
echo "Benchmark CSVs:   ${CREDITPFN_STAGING_ROOT}/output/results/<TRACK>/<method>/"
echo "Per-task logs:    ${CREDITPFN_OUTPUT_ROOT}/logs/<task>_<ts>_j<jid>_a<tid>.log"
echo "Training manif.:  ${CREDITPFN_OUTPUT_ROOT}/output/training/manifests/<run_name>_<track>.csv"
echo "Per-epoch CSVs:   ${CREDITPFN_OUTPUT_ROOT}/output/training/epochs/<track>/<descriptive>.csv"
echo "Notebook figures: ${CREDITPFN_OUTPUT_ROOT}/output/figures/<notebook>/*.pdf"
