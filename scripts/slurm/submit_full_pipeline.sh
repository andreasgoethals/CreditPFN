#!/bin/bash
# =============================================================================
#  CreditPFN — full pipeline submitter (data → train → eval) on VSC
# =============================================================================
#
#  Submits FOUR stages with `--dependency=afterok:` chaining so each
#  stage starts only after the previous one succeeds:
#
#    1. data.slurm                       (1 job ; genius batch CPU)
#    2. train_pd.slurm + train_lgd.slurm (arrays; wice gpu_h100)
#    3. eval_pd.slurm  + eval_lgd.slurm  (arrays; wice gpu_h100,
#                                         one (model × dataset) per
#                                         array task — heavy HPO
#                                         tasks parallelise cleanly)
#
#  Each stage writes ONE log file per slurm task to
#  `$VSC_DATA/CreditPFN/logs/<task>_<YYYYMMDD>_<HHMMSS>_j<JOBID>_a<TASKID>.log`.
#  Slurm's own `--output` is /dev/null in every .slurm — the bash
#  `exec >` redirection inside each script is the source of truth.
#
#  Usage (from a VSC login node):
#
#      bash scripts/slurm/submit_full_pipeline.sh
#
#  Optional knobs (override via env vars before invoking):
#
#      TRAIN_CONCURRENCY=4   # max in-flight train trials per array
#      EVAL_CONCURRENCY=32   # max in-flight eval (model × dataset) tasks
#      TRACKS="pd lgd"       # train + eval just one track if you want
# =============================================================================

set -euo pipefail

TRAIN_CONCURRENCY="${TRAIN_CONCURRENCY:-4}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-32}"
TRACKS="${TRACKS:-pd lgd}"
CONDA_ENV="${CONDA_ENV:-CreditPFN}"

cd "$(dirname "$0")/../.."

# ---------------------------------------------------------------------------
# Activate the project conda env if it isn't already. The login-node `python`
# does NOT have omegaconf / src.train / etc.; those live in the env created
# during one-time setup. This block is a no-op when the user has already run
# `source activate CreditPFN`.
# ---------------------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    # Try the standard conda hook first; fall back to a hardcoded shim
    # that mirrors what the .slurm scripts do.
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}" 2>/dev/null || true
    fi
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]] \
       && [[ -d "${VSC_DATA:-}/miniconda3" ]]; then
        export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"
        # `source activate` is the legacy shim; quieter than `conda activate`
        # under set -u.
        # shellcheck disable=SC1091
        source activate "${CONDA_ENV}" 2>/dev/null || true
    fi
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
        echo "ERROR: could not activate conda env '${CONDA_ENV}'." >&2
        echo "       Run 'source activate ${CONDA_ENV}' before this script," >&2
        echo "       or set CONDA_ENV=<name> if you use a different env name." >&2
        exit 1
    fi
fi

# Sanity-check the env has the project deps the submitter needs to run on
# the login node (omegaconf for cfg load, src.train.corpus for the eval
# upper-bound calculation). A clear error here saves debugging an opaque
# ModuleNotFoundError stack trace 40 lines down.
if ! python -c "import omegaconf, src.train.corpus" 2>/dev/null; then
    echo "ERROR: the '${CONDA_ENV}' env is missing project dependencies." >&2
    echo "       Re-install with: pip install -r requirements.txt" >&2
    exit 1
fi

echo "Submitting CreditPFN full pipeline …"
echo "  CONDA_ENV          : ${CONDA_ENV}"
echo "  TRACKS             : ${TRACKS}"
echo "  TRAIN_CONCURRENCY  : ${TRAIN_CONCURRENCY}"
echo "  EVAL_CONCURRENCY   : ${EVAL_CONCURRENCY}"

# 1) Data preprocessing.
DATA_JID=$(sbatch --parsable scripts/slurm/data.slurm)
echo "  data               : ${DATA_JID}"

# 2) Training (one array job per track), each waiting on data.
declare -A TRAIN_JIDS=()
for TR in ${TRACKS}; do
    SCRIPT="scripts/slurm/train_${TR}.slurm"
    N=$(python scripts/train_pipeline.py --list-trials track="${TR}")
    JID=$(sbatch --parsable \
        --dependency="afterok:${DATA_JID}" \
        --array=0-$((N - 1))%"${TRAIN_CONCURRENCY}" \
        "${SCRIPT}")
    TRAIN_JIDS["$TR"]="${JID}"
    echo "  train ${TR}        : ${JID}  (array 0..$((N - 1)))"
done

# 3) Eval — one array job per track, gated on the matching training array.
#
# Array size: we cannot run `eval_pipeline.py --list-tasks` here because
# the trained-model manifest is still empty at submission time. Instead
# we compute the EXPECTED upper bound from the cfg cardinality:
#
#     N_eval = (n_baselines + n_untuned + n_planned_trials) * n_test_datasets
#
# where n_planned_trials is what `train_pipeline.py --list-trials` returns
# (i.e. base_paths × learning_rates). Some training trials may fail; those
# slots become no-op tasks (the eval pipeline's skip-existing guard exits
# early). That's much better than under-sizing the array and silently
# skipping newly-trained checkpoints.
N_PD_PLANNED=$(python scripts/train_pipeline.py --list-trials track=pd 2>/dev/null || echo 0)
N_LGD_PLANNED=$(python scripts/train_pipeline.py --list-trials track=lgd 2>/dev/null || echo 0)
for TR in ${TRACKS}; do
    SCRIPT="scripts/slurm/eval_${TR}.slurm"
    DEP="${TRAIN_JIDS[$TR]}"
    if [[ "$TR" == "pd" ]]; then PLANNED_TRIALS=${N_PD_PLANNED}; else PLANNED_TRIALS=${N_LGD_PLANNED}; fi
    # Upper bound: the eval roster at run time = baselines + untuned + trained.
    # Each gets paired with every test dataset. We approximate the count
    # below; the eval script's --task-index N then runs exactly the Nth
    # pair from the freshly-built roster (which by then includes every
    # OK-trained checkpoint).
    UPPER_N=$(python -c "
import sys
sys.path.insert(0, '.')
from omegaconf import OmegaConf
eval_cfg = OmegaConf.load('config/eval.yaml')
train_cfg = OmegaConf.load(eval_cfg.train_cfg_path)
n_baselines = sum(1 for b in eval_cfg.baselines.enabled
                  if b != 'tabpfn-untuned')
# logreg only counts for pd; linreg only for lgd
if '${TR}' == 'pd' and 'linreg' in eval_cfg.baselines.enabled:
    n_baselines -= 1
if '${TR}' == 'lgd' and 'logreg' in eval_cfg.baselines.enabled:
    n_baselines -= 1
n_untuned = len(train_cfg.tunable.classifier_base_paths
                if '${TR}' == 'pd' else train_cfg.tunable.regressor_base_paths)
n_planned = int('${PLANNED_TRIALS}' or 0)
# Test dataset count: from corpus split (Mode A fractions or Mode B explicit).
from src.train.corpus import split_from_cfg
split = split_from_cfg(train_cfg, track='${TR}')
n_test = len({c.dataset_id for c in split.test})
print((n_baselines + n_untuned + n_planned) * max(1, n_test))
" 2>/dev/null || echo 1)
    UPPER_N=${UPPER_N:-1}
    if [[ "$UPPER_N" -lt 1 ]]; then UPPER_N=1; fi
    EVAL_JID=$(sbatch --parsable \
        --dependency="afterok:${DEP}" \
        --array=0-$((UPPER_N - 1))%"${EVAL_CONCURRENCY}" \
        "${SCRIPT}")
    echo "  eval  ${TR}        : ${EVAL_JID}  (array 0..$((UPPER_N - 1)), waits on ${DEP})"
done

echo
echo "All jobs submitted. Watch progress with:"
echo "    squeue --me --clusters=genius,wice"
echo
echo "Per-task logs:    \$VSC_DATA/CreditPFN/logs/<task>_<ts>_j<jid>_a<tid>.log"
echo "Manifests:        \$VSC_DATA/CreditPFN/manifests/<run_name>_<track>.csv"
echo "Trained ckpts:    \$VSC_DATA/CreditPFN/checkpoints/trained/<track>/"
echo "Benchmark CSVs:   \$VSC_DATA/CreditPFN/results/<TRACK>/<method>/"
