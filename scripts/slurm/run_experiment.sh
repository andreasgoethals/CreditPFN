#!/usr/bin/env bash
# Submit one experiment: every trial of its grid, for every dataset split, routed to whichever
# cluster suits the model.
#
#   bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml
#   SPLITS=4 bash ...            # only the first 4 splits
#   DRY=1 bash ...               # print the sbatch lines, submit nothing
#   ROUTE=0 bash ...             # everything on Mindwell, no per-model routing
#
# WHY SPLITS ARE A LOOP AND TRIALS ARE AN ARRAY. The dataset split is not a sweep axis — making
# it one would multiply the grid and bury a split tag inside every trial name. Each split is its
# own submission writing its own manifest (run_name "<name>_sNN"); the array covers the grid.
#
# WHY THREE CLUSTERS. They have SEPARATE queues and separate fairshare, so submitting to all
# three multiplies the throughput of a 1 792-trial experiment — and the cheap partition is
# genuinely cheap: A100 141.7 credits/GPU-min against B200 437.5 and H100 569.4.
# What fits where, at TWO ensemble members (24-08 probe, slopes GB per 1k rows per member):
#   v3      26k x 2 x 2.51 = 131 GB  ] need the 183 GB B200
#   v2.6    11k x 2 x 5.44 = 120 GB  ]
#   v2      14k x 2 x 3.96 = 111 GB  ]  (does NOT fit an 80 GB A100)
#   tabicl  26k x 2 x 0.52 =  27 GB  ] fits an 80 GB A100 with room to spare
# Only TabICLv2 goes to wICE. An earlier version of this file also sent v2 there on the
# strength of a single-member probe reading of ~30 GB; at two members it is 111 GB and every
# such trial would have died on allocation.
set -euo pipefail

CONFIG="${1:?usage: run_experiment.sh <config.yaml>}"
[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 1; }
cd "$(dirname "$0")/../.."

read_cfg() {   # read one dotted key from the merged config
    python - "$CONFIG" "$1" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.merge(OmegaConf.load("config/train.yaml"), OmegaConf.load(sys.argv[1]))
v = OmegaConf.select(cfg, sys.argv[2])
print("" if v is None else v)
PY
}

TRACK="$(read_cfg track)"
N_SPLITS="${SPLITS:-$(read_cfg corpus.n_splits)}"
N_SPLITS="${N_SPLITS:-1}"
N_TRIALS="$(python scripts/train_pipeline.py --config "$CONFIG" --list-trials | tail -1)"
THROTTLE="${THROTTLE:-16}"
# VSC rejects submissions past 500 QUEUED JOBS per user, and an array counts its tasks,
# not itself. 48 trials x 12 splits = 576 tasks, so a naive loop loses the tail. Submit
# in waves of at most this many tasks and wait for room between waves.
MAX_QUEUED="${MAX_QUEUED:-450}"
ACCOUNT="${CREDITPFN_ACCOUNT:-lp_verbekelab}"
JOB="scripts/slurm/train_${TRACK}.slurm"

# Which cluster each trial belongs on. `--trial-family` already tells us tabpfn vs tabicl, but
# the split we need is by MEMORY, so ask for the base checkpoint of each trial instead.
mapfile -t TRIAL_BASE < <(python - "$CONFIG" "$N_TRIALS" <<'PY'
import sys
from omegaconf import OmegaConf
sys.path.insert(0, ".")
import scripts.train_pipeline as tp
cfg = OmegaConf.merge(OmegaConf.load("config/train.yaml"), OmegaConf.load(sys.argv[1]))
for t in tp._resolve_grid(cfg, single=False):
    # tuple is (base, lr, frozen, qf, accum, pass_mode, min_train_rows, l2sp)
    print(t[0].rsplit("/", 1)[-1], "frozen" if t[2] else "full")
PY
)

# WHERE EACH TRIAL GOES, and why the complexity earns its keep: mindwell and wice run
# SEPARATE schedulers, so a trial queued on one does not wait behind a trial queued on the
# other. Queue time dominates wall-clock here, so spreading over both buys more than any
# per-GPU speed difference does. Limits confirmed by the 24-08-2026 report: gpu_b200,
# gpu_h100 and gpu_a100 all cap at 3-00:00:00, and the `normal` QOS allows 500 submitted jobs.
#
# Memory at TWO training members (measured GB per 1k rows per member x cap x 2 members):
#   v3      26k x 2 x 2.51 = 131 GB ]
#   v2.6    11k x 2 x 5.44 = 120 GB ] B200 (183 GB) only
#   v2      14k x 2 x 3.96 = 111 GB ]
#   tabicl  26k x 2 x 0.52 =  27 GB   fits an 80 GB A100 with room to spare
#
# A frozen backbone builds no autograd graph through the transformer stack, so those
# activations are never retained and the same row cap fits a much smaller card. That is why
# the frozen arm goes to wice: it is the half of the grid that does not need 183 GB. That
# claim is still UNMEASURED until section 9 of cluster_report.py runs, so until then override
# with FROZEN_DEST="mindwell gpu_b200".
route_for() {
    local base="$1" mode="${2:-full}"
    if [[ "${ROUTE:-1}" == "0" ]]; then echo "mindwell gpu_b200"; return; fi
    case "$base" in
        *tabicl-*) echo "${TABICL_DEST:-wice gpu_a100}"; return ;;
    esac
    if [[ "$mode" == "frozen" ]]; then
        echo "${FROZEN_DEST:-wice gpu_h100}"
    else
        echo "${FULL_DEST:-mindwell gpu_b200}"
    fi
}

# Group trial indices by destination so each destination gets ONE array with an explicit list,
# rather than one job per trial.
declare -A BUCKET
for (( t=0; t<N_TRIALS; t++ )); do
    key="$(route_for ${TRIAL_BASE[$t]:-unknown full})"
    BUCKET["$key"]="${BUCKET[$key]:+${BUCKET[$key]},}$t"
done

echo "=============================================================="
echo " config : $CONFIG        track: $TRACK"
echo " trials : $N_TRIALS per split     splits: $N_SPLITS     total: $((N_TRIALS * N_SPLITS))"
for key in "${!BUCKET[@]}"; do
    n=$(awk -F, '{print NF}' <<< "${BUCKET[$key]}")
    echo " route  : ${key}  <- ${n} trials/split"
done
echo "=============================================================="

queued_tasks() {   # tasks this user currently has across both controllers
    local n=0 c
    for c in mindwell wice genius; do
        n=$(( n + $(squeue -M "$c" -u "$USER" -h -t PD,R -o "%i" 2>/dev/null | wc -l) ))
    done
    echo "$n"
}

for (( k=0; k<N_SPLITS; k++ )); do
    # WAIT FOR ROOM. Without this the 501st task is rejected and that split silently never
    # runs — the failure mode is a gap in the results, not an error at the end of the sweep.
    if [[ -z "${DRY:-}" ]]; then
        while :; do
            q=$(queued_tasks)
            (( q + N_TRIALS <= MAX_QUEUED )) && break
            echo "  queue at ${q} tasks; waiting for room for split ${k} (${N_TRIALS} tasks)..."
            sleep 300
        done
    fi
    for key in "${!BUCKET[@]}"; do
        read -r cluster partition <<< "$key"
        CMD=(sbatch --clusters="$cluster" --partition="$partition" --account="$ACCOUNT"
             --array="${BUCKET[$key]}%${THROTTLE}"
             --export=ALL,CREDITPFN_CONFIG="$CONFIG",CREDITPFN_SPLIT_INDEX="$k"
             "$JOB")
        if [[ -n "${DRY:-}" ]]; then echo "DRY: ${CMD[*]}"; else "${CMD[@]}"; fi
    done
done
echo "submitted $((N_SPLITS * ${#BUCKET[@]})) array job(s) across ${#BUCKET[@]} destination(s)."
