#!/usr/bin/env bash
# Submit one experiment: every trial of its grid, for every dataset split.
#
#   bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml
#   SPLITS=4 bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml   # a subset
#   DRY=1   bash scripts/slurm/run_experiment.sh config/experiment2_pd.yaml    # print only
#
# WHY A LOOP OVER SPLITS AND AN ARRAY OVER TRIALS. The dataset split is deliberately NOT a
# sweep axis (it would multiply the trial grid and put a split tag inside every trial name).
# So each split is its own submission, with its own manifest under run_name "<name>_s<NN>",
# and the array inside it covers the hyperparameter grid. The analysis then averages over
# splits within each hyperparameter cell, which is the whole point of Experiment 1.
set -euo pipefail

CONFIG="${1:?usage: run_experiment.sh <config.yaml>}"
[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 1; }

cd "$(dirname "$0")/../.."

# n_splits comes from the config unless overridden. `null`/absent means a pinned split.
N_SPLITS="${SPLITS:-$(python - "$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.merge(OmegaConf.load("config/train.yaml"), OmegaConf.load(sys.argv[1]))
print(int(OmegaConf.select(cfg, "corpus.n_splits") or 1))
PY
)}"

N_TRIALS="$(python scripts/train_pipeline.py --config "$CONFIG" --list-trials | tail -1)"
TRACK="$(python - "$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.merge(OmegaConf.load("config/train.yaml"), OmegaConf.load(sys.argv[1]))
print(str(cfg.track))
PY
)"

# Concurrency cap per array. Training is one GPU per trial; 16 is what this account has
# actually been granted concurrently on Mindwell (RESULTS.md, runs 6-8).
THROTTLE="${THROTTLE:-16}"
JOB="scripts/slurm/train_${TRACK}.slurm"

echo "=============================================================="
echo " config   : $CONFIG"
echo " track    : $TRACK"
echo " trials   : $N_TRIALS   (per split)"
echo " splits   : $N_SPLITS"
echo " total    : $((N_TRIALS * N_SPLITS)) trials"
echo " job      : $JOB   array 0-$((N_TRIALS - 1))%${THROTTLE}"
echo "=============================================================="

for (( k=0; k<N_SPLITS; k++ )); do
    CMD=(sbatch --array="0-$((N_TRIALS - 1))%${THROTTLE}"
         --export=ALL,CREDITPFN_CONFIG="$CONFIG",CREDITPFN_SPLIT_INDEX="$k"
         "$JOB")
    if [[ -n "${DRY:-}" ]]; then
        echo "DRY: ${CMD[*]}"
    else
        "${CMD[@]}"
    fi
done

echo "submitted ${N_SPLITS} array job(s)."
echo "the job script must pass \$CREDITPFN_CONFIG and \$CREDITPFN_SPLIT_INDEX through to"
echo "train_pipeline.py as --config and --split-index."
