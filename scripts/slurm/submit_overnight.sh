#!/bin/bash
# =============================================================================
#  CreditPFN — FIRE-ONCE overnight pipeline (data → train → eval), one command.
# =============================================================================
#  Submits ALL stages at once and self-sequences them across the two clusters
#  WITHOUT any cross-cluster afterok dependency (VSC doesn't support those):
#
#    [1] data  (wICE `batch`, CPU)            → writes a "data_done" sentinel on
#                                               $VSC_DATA (NFS, seen everywhere)
#         ⇣  train waits for the sentinel at startup (CREDITPFN_WAIT_DATA=1)
#    [2] train pd + lgd (Mindwell `gpu_b200`, 32 trials each)
#         ⇣  an eval "gate" (wICE, 1 CPU) watches the train jobs via squeue
#    [3] eval gate (wICE `batch`)             → finishes when training finishes
#         ⇣  eval arrays `afterok` the gate → PENDING (no GPU held) until then
#    [4] eval pd + lgd (wICE `gpu_h100` + `gpu_a100`)
#
#  Fire it once and walk away; all results are ready in the morning. Datasets,
#  checkpoints and results live in project staging; logs on $VSC_DATA.
#
#  Usage (Genius login node):
#      bash scripts/slurm/submit_overnight.sh
#  Knobs (env): TRAIN_ACCOUNT (default lp_verbekelab), EVAL_ACCOUNT, TRACKS,
#      TRAIN_CONCURRENCY, EVAL_CONCURRENCY, EVAL_PARTITIONS, CONDA_ENV.
# =============================================================================
set -euo pipefail

TRACKS="${TRACKS:-pd lgd}"
TRAIN_ACCOUNT="${TRAIN_ACCOUNT:-lp_verbekelab}"      # has Mindwell access
EVAL_ACCOUNT="${EVAL_ACCOUNT:-lp_verbekelab}"
TRAIN_CONCURRENCY="${TRAIN_CONCURRENCY:-24}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-32}"
EVAL_PARTITIONS="${EVAL_PARTITIONS:-gpu_h100 gpu_a100}"
CONDA_ENV="${CONDA_ENV:-CreditPFN}"

cd "$(dirname "$0")/../.."
REPO="$PWD"

# --- activate env (login-node python lacks omegaconf / src.*) ----------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    if ! source scripts/slurm/_activate_env.sh; then
        echo "ERROR: could not activate conda env '${CONDA_ENV}'." >&2
        echo "       Run 'conda activate ${CONDA_ENV}' first." >&2
        exit 1
    fi
fi

# --- resolve storage roots once and propagate to every job -------------------
read -r CREDITPFN_DATA_ROOT CREDITPFN_OUTPUT_ROOT CREDITPFN_STAGING_ROOT < <(
    python -c "
from omegaconf import OmegaConf
from src.utils.paths import apply_data_source_from_cfg, get_roots
apply_data_source_from_cfg(OmegaConf.load('config/data.yaml'))
r = get_roots(); print(r['data_root'], r['output_root'], r['staging_root'])
"
)
SBATCH_EXPORT="ALL,CREDITPFN_DATA_ROOT=${CREDITPFN_DATA_ROOT},CREDITPFN_OUTPUT_ROOT=${CREDITPFN_OUTPUT_ROOT},CREDITPFN_STAGING_ROOT=${CREDITPFN_STAGING_ROOT}"

strip() { echo "${1%%;*}"; }            # "<jid>;<cluster>" → "<jid>"

echo "=================================================================="
echo "CreditPFN — one-shot overnight pipeline"
echo "  TRAIN_ACCOUNT : ${TRAIN_ACCOUNT}   (Mindwell gpu_b200)"
echo "  EVAL_ACCOUNT  : ${EVAL_ACCOUNT}    (wICE ${EVAL_PARTITIONS})"
echo "  TRACKS        : ${TRACKS}"
echo "  DATA_ROOT     : ${CREDITPFN_DATA_ROOT}"
echo "  OUTPUT_ROOT   : ${CREDITPFN_OUTPUT_ROOT}"
echo "  STAGING_ROOT  : ${CREDITPFN_STAGING_ROOT}"

# --- raw-data presence check (fail fast before burning queue time) -----------
if [[ -d "${CREDITPFN_DATA_ROOT}/data/raw" ]]; then
    n_pd=$(find "${CREDITPFN_DATA_ROOT}/data/raw/pd"  -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l || echo 0)
    n_lgd=$(find "${CREDITPFN_DATA_ROOT}/data/raw/lgd" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l || echo 0)
    echo "  raw datasets  : pd=${n_pd}  lgd=${n_lgd}"
    if [[ "${n_pd}" -eq 0 && "${n_lgd}" -eq 0 ]]; then
        echo "ERROR: no raw CSVs under ${CREDITPFN_DATA_ROOT}/data/raw/ — upload them first." >&2
        exit 1
    fi
fi
echo "=================================================================="

# --- clear a stale data sentinel from a previous run -------------------------
mkdir -p "${CREDITPFN_OUTPUT_ROOT}/.sentinels" "${CREDITPFN_OUTPUT_ROOT}/logs"
rm -f "${CREDITPFN_OUTPUT_ROOT}/.sentinels/data_done"

# --- [1] DATA (wICE) ---------------------------------------------------------
DATA_JID=$(strip "$(sbatch --parsable --export="${SBATCH_EXPORT}" scripts/slurm/data.slurm)")
echo "  [1] data  (wICE batch)        : ${DATA_JID}"

# --- [2] TRAIN (Mindwell); each task waits for the data_done sentinel --------
TRAIN_CSV=""
for TR in ${TRACKS}; do
    N=$(python scripts/train_pipeline.py --list-trials track="${TR}")
    JID=$(strip "$(sbatch --parsable \
        --export="${SBATCH_EXPORT},CREDITPFN_WAIT_DATA=1" \
        --account="${TRAIN_ACCOUNT}" \
        --array=0-$((N - 1))%"${TRAIN_CONCURRENCY}" \
        "scripts/slurm/train_${TR}.slurm")")
    TRAIN_CSV="${TRAIN_CSV:+${TRAIN_CSV},}${JID}"
    echo "  [2] train ${TR} (Mindwell b200): ${JID:-<FAILED>}  (array 0..$((N - 1)))"
done

# --- [3] EVAL GATE (wICE, 1 CPU) — releases eval when training finishes -------
GATE_JID=$(strip "$(sbatch --parsable \
    --clusters=wice --account="${EVAL_ACCOUNT}" --partition=batch \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=21:00:00 \
    --job-name=creditpfn-eval-gate --chdir="${REPO}" \
    --output="${CREDITPFN_OUTPUT_ROOT}/logs/eval_gate_%j.log" \
    --wrap="bash scripts/slurm/_wait_for_jobs.sh '${TRAIN_CSV}' 2400")")
echo "  [3] eval gate (wICE batch)    : ${GATE_JID}  (watches Mindwell ${TRAIN_CSV})"

# --- [4] EVAL (wICE gpu; afterok the gate; split across both GPU pools) ------
for TR in ${TRACKS}; do
    PLANNED=$(python scripts/train_pipeline.py --list-trials track="${TR}" 2>/dev/null || echo 0)
    UPPER_N=$(python -c "
import sys; sys.path.insert(0, '.')
from omegaconf import OmegaConf
eval_cfg = OmegaConf.load('config/eval.yaml')
train_cfg = OmegaConf.load(eval_cfg.train_cfg_path)
n_baselines = sum(1 for b in eval_cfg.baselines.enabled if b != 'tabpfn-untuned')
if '${TR}' == 'pd'  and 'linreg' in eval_cfg.baselines.enabled: n_baselines -= 1
if '${TR}' == 'lgd' and 'logreg' in eval_cfg.baselines.enabled: n_baselines -= 1
n_untuned = len(train_cfg.tunable.classifier_base_paths if '${TR}' == 'pd' else train_cfg.tunable.regressor_base_paths)
n_planned = int('${PLANNED}' or 0)
from src.train.corpus import split_from_cfg
split = split_from_cfg(train_cfg, track='${TR}')
n_test = len({c.dataset_id for c in split.test})
print((n_baselines + n_untuned + n_planned) * max(1, n_test))
" 2>/dev/null || echo 1)
    UPPER_N=${UPPER_N:-1}; [[ "$UPPER_N" -lt 1 ]] && UPPER_N=1
    # shellcheck disable=SC2206
    PARTS=(${EVAL_PARTITIONS}); K=${#PARTS[@]}
    CHUNK=$(( (UPPER_N + K - 1) / K ))
    for i in "${!PARTS[@]}"; do
        P="${PARTS[$i]}"; LO=$(( i * CHUNK )); HI=$(( (i + 1) * CHUNK - 1 ))
        [[ "$HI" -ge "$UPPER_N" ]] && HI=$(( UPPER_N - 1 ))
        [[ "$LO" -gt "$HI" ]] && continue
        JID=$(strip "$(sbatch --parsable \
            --clusters=wice --account="${EVAL_ACCOUNT}" --partition="${P}" \
            --dependency=afterok:"${GATE_JID}" \
            --export="${SBATCH_EXPORT}" \
            --array="${LO}-${HI}%${EVAL_CONCURRENCY}" \
            "scripts/slurm/eval_${TR}.slurm")")
        echo "  [4] eval ${TR} (wICE ${P}) : ${JID}  (array ${LO}..${HI}, afterok gate)"
    done
done

echo "=================================================================="
echo "Fired everything. Nothing else to do — check back tomorrow."
echo "  Watch:   squeue --me --clusters=wice,mindwell"
echo "  Logs:    ${CREDITPFN_OUTPUT_ROOT}/logs/"
echo "  Results: ${CREDITPFN_STAGING_ROOT}/output/results/  +  checkpoints/trained/"
echo "=================================================================="
