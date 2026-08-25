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
# VSC rejects submissions past 500 QUEUED JOBS per user, and an array counts its TASKS,
# not itself. Submit in waves of at most this many and wait for room between them.
MAX_QUEUED="${MAX_QUEUED:-450}"
# Trials packed into ONE array task, run sequentially. 48 trials x 8 splits x 2 tracks =
# 768 tasks at 1/task, which blows the 500 ceiling; at 4/task it is 192, and each task
# pays the 2-4 min startup once instead of four times. Bigger is fewer jobs but longer
# jobs, and a long request backfills later — 4 is the balance point.
TRIALS_PER_TASK="${TRIALS_PER_TASK:-4}"
# Worst measured trial is v3 PD full-FT: 5 000 steps x 0.93 s = 77 min. 90 min/trial plus
# 30 min of startup and monitor evals, capped at the 72 h partition limit.
MINUTES=$(( TRIALS_PER_TASK * 90 + 30 ))
(( MINUTES > 4320 )) && MINUTES=4320
WALLTIME="${WALLTIME:-$(printf '%d:%02d:00' $(( MINUTES / 60 )) $(( MINUTES % 60 )))}"
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

# WHERE EACH TRIAL GOES. mindwell and wice run SEPARATE schedulers, so a trial queued on one
# does not wait behind a trial queued on the other, and queue time dominates wall-clock here.
# Limits confirmed 24-08-2026: gpu_b200 / gpu_h100 / gpu_a100 all cap at 3-00:00:00, and the
# `normal` QOS allows 500 submitted jobs.
#
# MEASURED peak allocation at 2 training members (cluster job 11524668, section 9), which is
# what decides where a trial CAN run:
#
#   v2      10k =  79 GB   (26k OOMs; configured cap 14k ~ 111 GB)  ]
#   v2.6    10k = 109 GB   (26k OOMs; configured cap 11k ~ 120 GB)  ]  B200 (183 GB) ONLY
#   v3      26k = 131 GB   (50k OOMs)                               ]
#   tabicl  26k =  27 GB   (50k dies on a kernel shape, not OOM)    <- fits an 80 GB A100
#
# The frozen arm is NOT cheaper in memory — measured identical to full mode to two decimals,
# because peak sits in the forward pass and both families already recompute activations during
# backward. So freeze mode does NOT change where a trial can run, and routing on it (as this
# script did between 24-08 morning and afternoon) sent 100+ GB trials to an 80 GB card.
#
# Value per unit of work also came out opposite to the earlier estimate. Measured bf16:
# B200 1586 TFLOP/s, A100 289 TFLOP/s — the B200 is 5.5x faster, not 2.2x. At 26 250 vs
# 8 500 credits/GPU-hour that makes the B200 1.8x CHEAPER per unit of work. TabICLv2 goes to
# wice anyway, to run its queue in parallel with mindwell's; that premium buys wall-clock and
# applies only to the cheapest quarter of the grid.
route_for() {
    local base="$1" mode="${2:-full}"     # mode kept for logging; it must NOT affect placement
    if [[ "${ROUTE:-1}" == "0" ]]; then echo "mindwell gpu_b200"; return; fi
    case "$base" in
        # DEFAULT: everything on the B200. Measured, the wice split does not pay — putting
        # TabICLv2 on the A100 costs 1237 GPU-h / 20.6M credits against 702 / 18.4M all on
        # mindwell, and buys 0.3 days (7.0 vs 7.3) because the A100 is 5.5x slower. Set
        # TABICL_DEST="wice gpu_a100" to spill onto wice when the mindwell queue is backed up;
        # that is a congestion valve, not the cheaper plan.
        *tabicl-*) echo "${TABICL_DEST:-mindwell gpu_b200}" ;;
        *)         echo "${TABPFN_DEST:-mindwell gpu_b200}" ;;
    esac
}

# Group trial indices by destination so each destination gets ONE array with an explicit list,
# rather than one job per trial.
# Map each TASK to a destination, and refuse to submit if a task would straddle two of
# them. The grid is base-major (48 trials = 4 bases x 12), so any TRIALS_PER_TASK dividing the
# per-base block keeps a task inside one base — but assert it rather than trust it, because
# both the routing and the per-family tabicl preflight are keyed on the base.
declare -A TASK_DEST
for (( t=0; t<N_TRIALS; t++ )); do
    d="$(route_for ${TRIAL_BASE[$t]:-unknown full})"
    task=$(( t / TRIALS_PER_TASK ))
    if [[ -n "${TASK_DEST[$task]:-}" && "${TASK_DEST[$task]}" != "$d" ]]; then
        echo "ERROR: TRIALS_PER_TASK=${TRIALS_PER_TASK} makes task ${task} span two" >&2
        echo "       destinations ('${TASK_DEST[$task]}' and '${d}'). Choose a value that" >&2
        echo "       divides the per-base block size (${N_TRIALS}/n_bases), e.g. 1, 2, 3, 4, 6." >&2
        exit 1
    fi
    TASK_DEST[$task]="$d"
done
N_TASKS=${#TASK_DEST[@]}
echo "${N_TRIALS} trials -> ${N_TASKS} array task(s) at ${TRIALS_PER_TASK}/task, walltime ${WALLTIME}"

declare -A BUCKET
for task in $(printf '%s\n' "${!TASK_DEST[@]}" | sort -n); do
    key="${TASK_DEST[$task]}"
    BUCKET["$key"]="${BUCKET[$key]:+${BUCKET[$key]},}$task"
done

echo "=============================================================="
echo " config : $CONFIG        track: $TRACK"
echo " trials : $N_TRIALS per split     splits: $N_SPLITS     total: $((N_TRIALS * N_SPLITS))"
for key in "${!BUCKET[@]}"; do
    n=$(awk -F, '{print NF}' <<< "${BUCKET[$key]}")
    echo " route  : ${key}  <- ${n} tasks/split"
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
            (( q + N_TASKS <= MAX_QUEUED )) && break
            echo "  queue at ${q} tasks; waiting for room for split ${k} (${N_TASKS} tasks)..."
            sleep 300
        done
    fi
    for key in "${!BUCKET[@]}"; do
        read -r cluster partition <<< "$key"
        CMD=(sbatch --clusters="$cluster" --partition="$partition" --account="$ACCOUNT"
             --array="${BUCKET[$key]}%${THROTTLE}" --time="$WALLTIME"
             --export=ALL,CREDITPFN_CONFIG="$CONFIG",CREDITPFN_SPLIT_INDEX="$k",CREDITPFN_TRIALS_PER_TASK="$TRIALS_PER_TASK"
             "$JOB")
        if [[ -n "${DRY:-}" ]]; then echo "DRY: ${CMD[*]}"; else "${CMD[@]}"; fi
    done
done
echo "submitted $((N_SPLITS * ${#BUCKET[@]})) array job(s) across ${#BUCKET[@]} destination(s)."
