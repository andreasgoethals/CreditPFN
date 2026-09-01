#!/usr/bin/env bash
# Submit one experiment: every trial of its grid, for every dataset split, routed to whichever
# cluster suits the model.
#
#   bash scripts/slurm/run_experiment.sh config/experiment1_pd.yaml
#   STAGES=eval bash ...         # score an already-trained experiment
#   STAGES="train eval" bash ... # both (eval is NOT chained — see below)
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
# Requested trials per array task. 2 keeps jobs ~3.5 h, which backfills far better than the old
# 6.5 h at 4/task — VSC priority favours short walltime (official docs + measured: 48h -> 1-2
# GPUs, 10h -> 15-21). It is AUTO-CLAMPED below to a divisor of the per-base block so a task
# never straddles two model families; exp0 has 1 trial/base, so it falls back to 1 automatically.
TRIALS_PER_TASK="${TRIALS_PER_TASK:-2}"
# Per-base block: the grid is base-major, so this many consecutive trials share one base.
N_BASES="$(python - "$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
sys.path.insert(0, ".")
import scripts.train_pipeline as tp
cfg = OmegaConf.merge(OmegaConf.load("config/train.yaml"), OmegaConf.load(sys.argv[1]))
print(len({t[0] for t in tp._resolve_grid(cfg, single=False)}))
PY
)"
BLOCK=$(( N_TRIALS / N_BASES ))
_req="$TRIALS_PER_TASK"
for (( d=_req; d>=1; d-- )); do (( BLOCK % d == 0 )) && { TRIALS_PER_TASK=$d; break; }; done
if (( TRIALS_PER_TASK != _req )); then
    echo "note: TRIALS_PER_TASK=${_req} does not divide the ${BLOCK}-trial per-base block; using ${TRIALS_PER_TASK}."
fi
# Walltime is sized PER PASS MODE, because the two differ ~6x in wall-clock for the SAME 5000-step
# budget. full_pass reaches 5000 steps in ~25-57 epochs (v3 full-FT: 5000 x 0.93 s = 77 min), but
# accumulate makes only ONE update per dataset, so it needs ~385 epochs -- each a full pass over
# all datasets -- ~= 7 h (TabICLv2 measured, exp1_pd). A single flat walltime cannot serve both:
# the old 3:30 sizing (90 min/trial) killed every accumulate trial at ~50% (exp1_pd 29-08 -> 0/384
# accumulate ever completed), and sizing UP for accumulate would needlessly lower full_pass's
# scheduling priority (shorter walltime backfills better: 48h -> 1-2 GPUs, 10h -> 15-21). So:
# full_pass keeps the short, high-priority walltime; accumulate gets its own long one.
hms() { printf '%d:%02d:00' $(( $1 / 60 )) $(( $1 % 60 )); }
FULL_MIN=$(( TRIALS_PER_TASK * 90 + 30 ))                          # 90 min/trial + 30 startup
ACC_MIN=$(( TRIALS_PER_TASK * ${ACC_MIN_PER_TRIAL:-600} + 30 ))    # ~10 h/trial (TabICL 7 h + margin)
(( FULL_MIN > 4320 )) && FULL_MIN=4320                             # 72 h partition cap
(( ACC_MIN  > 4320 )) && ACC_MIN=4320
FULL_WALLTIME="${WALLTIME:-$(hms "$FULL_MIN")}"                    # WALLTIME= overrides full_pass
ACC_WALLTIME="${ACC_WALLTIME:-$(hms "$ACC_MIN")}"                 # ACC_WALLTIME= overrides accumulate
ACCOUNT="${CREDITPFN_ACCOUNT:-lp_verbekelab}"
STAGES="${STAGES:-train}"
JOB="scripts/slurm/train_${TRACK}.slurm"
EVAL_JOB="scripts/slurm/eval_${TRACK}.slurm"
# Eval packs its (model x dataset x fold) cells into this many array tasks. Exported
# because eval_*.slurm reads it too, and both sides must agree on the number or they
# disagree about which cells belong to task i.
export EVAL_TASKS="${EVAL_TASKS:-16}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"

# Per trial: the routing base + frozen flag (route_for keys placement on MEMORY, i.e. the base
# checkpoint) AND the pass mode (walltime keys on it -- see above). `--trial-family` only tells us
# tabpfn vs tabicl, so read the grid directly. One line/trial: "<base> <full|frozen> <pass_mode>".
mapfile -t TRIAL_INFO < <(python - "$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
sys.path.insert(0, ".")
import scripts.train_pipeline as tp
cfg = OmegaConf.merge(OmegaConf.load("config/train.yaml"), OmegaConf.load(sys.argv[1]))
for t in tp._resolve_grid(cfg, single=False):
    # tuple is (base, lr, frozen, qf, accum, pass_mode, min_train_rows, l2sp)
    print(t[0].rsplit("/", 1)[-1], "frozen" if t[2] else "full", t[5])
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
declare -A TASK_DEST TASK_PASS
for (( t=0; t<N_TRIALS; t++ )); do
    read -r _b _fr _pm <<< "${TRIAL_INFO[$t]}"
    _pm="${_pm%$'\r'}"                         # defend against a \r if python emits CRLF
    d="$(route_for "${_b:-unknown}" "${_fr:-full}")"
    task=$(( t / TRIALS_PER_TASK ))
    if [[ -n "${TASK_DEST[$task]:-}" && "${TASK_DEST[$task]}" != "$d" ]]; then
        echo "ERROR: TRIALS_PER_TASK=${TRIALS_PER_TASK} makes task ${task} span two" >&2
        echo "       destinations ('${TASK_DEST[$task]}' and '${d}'). Choose a value that" >&2
        echo "       divides the per-base block size (${N_TRIALS}/n_bases), e.g. 1, 2, 3, 4, 6." >&2
        exit 1
    fi
    # A task must also not straddle pass modes, or one walltime cannot fit it. The grid runs
    # ...FFAA... in pairs (pass mode is the axis just inside frozen), so any TRIALS_PER_TASK
    # dividing 2 keeps a task single-mode; assert it rather than trust it.
    if [[ -n "${TASK_PASS[$task]:-}" && "${TASK_PASS[$task]}" != "$_pm" ]]; then
        echo "ERROR: TRIALS_PER_TASK=${TRIALS_PER_TASK} makes task ${task} span two pass modes" >&2
        echo "       ('${TASK_PASS[$task]}' and '${_pm}'). Use a value that divides 2 (1 or 2)." >&2
        exit 1
    fi
    TASK_DEST[$task]="$d"
    TASK_PASS[$task]="$_pm"
done
N_TASKS=${#TASK_DEST[@]}
echo "${N_TRIALS} trials -> ${N_TASKS} array task(s) at ${TRIALS_PER_TASK}/task; walltime full_pass=${FULL_WALLTIME} accumulate=${ACC_WALLTIME}"

declare -A BUCKET
for task in $(printf '%s\n' "${!TASK_DEST[@]}" | sort -n); do
    key="${TASK_DEST[$task]}|${TASK_PASS[$task]}"   # destination AND pass mode -> each its own walltime
    BUCKET["$key"]="${BUCKET[$key]:+${BUCKET[$key]},}$task"
done

echo "=============================================================="
echo " config : $CONFIG        track: $TRACK"
echo " trials : $N_TRIALS per split     splits: $N_SPLITS     total: $((N_TRIALS * N_SPLITS))"
for key in "${!BUCKET[@]}"; do
    n=$(awk -F, '{print NF}' <<< "${BUCKET[$key]}")
    IFS='|' read -r _d _pm <<< "$key"
    [[ "$_pm" == "accumulate" ]] && _wt="$ACC_WALLTIME" || _wt="$FULL_WALLTIME"
    echo " route  : ${_d}  [${_pm}]  <- ${n} tasks/split, walltime ${_wt}"
done
echo "=============================================================="

queued_tasks() {   # array TASKS this user has across all controllers. -r EXPANDS arrays, so a
                   # pending [0-47] counts as 48 (how the QOS MaxSubmitJobsPerUser limit actually
                   # counts it) rather than as 1 -- without -r the throttle under-counts by ~48x,
                   # sails past the 500 cap, and every submit then bounces off submit_retry for
                   # hours instead of waiting cleanly here (exp1_pd, 01-09-2026).
    local n=0 c
    for c in mindwell wice genius; do
        n=$(( n + $(squeue -M "$c" -u "$USER" -h -r -t PD,R -o "%i" 2>/dev/null | wc -l) ))
    done
    echo "$n"
}

# Submit one job, RETRYING when the QOS submit-limit rejects it. `queued_tasks` throttles up
# front, but it counts PENDING+RUNNING while the QOS `MaxSubmitJobsPerUser` counts ALL submitted
# states — and that quota is shared with any OTHER project on this account (e.g. CreditICL) — so
# the two disagree and a raw sbatch failure would abort the whole loop (set -e) and silently lose
# every later split AND the eval stage. Echoes sbatch's stdout ("<jobid>;<cluster>") on success.
submit_retry() {
    local out rc tries=0
    while :; do
        if out=$("$@" 2>&1); then echo "$out"; return 0; fi
        rc=$?
        if [[ "$out" == *QOSMaxSubmitJobPerUserLimit* || "$out" == *"job submit limit"* ]]; then
            (( ++tries ))
            echo "  submit-limit hit (attempt ${tries}); the 500-task quota (shared with your" >&2
            echo "  other jobs) is full — waiting 300s for room..." >&2
            sleep 300
            continue
        fi
        echo "$out" >&2                       # a DIFFERENT error: surface it, do not mask
        return "$rc"
    done
}

EVAL_DEST="${EVAL_DEST:-mindwell gpu_b200}"
read -r ECLUSTER EPARTITION <<< "$EVAL_DEST"
# Training job ids per split that are ON THE EVAL CLUSTER, for the dependency below. A
# dependency can only reference jobs on the same controller, so cross-cluster training
# (e.g. TABICL_DEST=wice) simply is not chained — eval for that split then runs without a
# dependency and relies on skip-existing, exactly as STAGES=eval does.
declare -A DEP_JOBS

SPLIT_START="${SPLIT_START:-0}"   # resume from split k (recovery after a partial submission)
if [[ " ${STAGES} " == *" train "* ]]; then
for (( k=SPLIT_START; k<N_SPLITS; k++ )); do
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
        IFS='|' read -r dest pm <<< "$key"
        read -r cluster partition <<< "$dest"
        [[ "$pm" == "accumulate" ]] && wt="$ACC_WALLTIME" || wt="$FULL_WALLTIME"
        CMD=(sbatch --parsable --clusters="$cluster" --partition="$partition" --account="$ACCOUNT"
             --array="${BUCKET[$key]}%${THROTTLE}" --time="$wt"
             --export=ALL,CREDITPFN_CONFIG="$CONFIG",CREDITPFN_SPLIT_INDEX="$k",CREDITPFN_TRIALS_PER_TASK="$TRIALS_PER_TASK"
             "$JOB")
        if [[ -n "${DRY:-}" ]]; then
            echo "DRY: ${CMD[*]}"
        else
            # --parsable prints "<jobid>;<cluster>"; keep the id only when it is on the
            # eval cluster, so the eval dependency below can reference it.
            out=$(submit_retry "${CMD[@]}")
            jid="${out%%;*}"
            if [[ "$cluster" == "$ECLUSTER" && -n "$jid" ]]; then
                DEP_JOBS[$k]="${DEP_JOBS[$k]:+${DEP_JOBS[$k]}:}$jid"
            fi
        fi
    done
done
echo "submitted $(( (N_SPLITS - SPLIT_START) * ${#BUCKET[@]} )) training array job(s) across ${#BUCKET[@]} destination(s)."
fi

# ----------------------------------------------------------------- eval ----
# One eval array per split, carrying the SAME $CONFIG and split index the training half
# used. That is the point of doing it here: eval_pipeline rebuilds the held-out dataset
# draw from the training corpus block, so a mismatch scores each checkpoint against
# datasets it may have TRAINED on — silently, and in the direction that inflates scores.
#
# No afterok dependency, deliberately: eval skips cells that already have results, so
# re-running is how a partly-finished campaign gets completed. Chaining would let one
# failed training task block the scoring of every sibling that succeeded.
if [[ " ${STAGES} " == *" eval "* ]]; then
    for (( k=SPLIT_START; k<N_SPLITS; k++ )); do
        # Same queue-room guard as the training stage. 8 splits x 16 tasks x 2 tracks = 256
        # eval tasks, and training may still be queued, so the 500 ceiling is reachable here
        # too — and going over it silently drops a split's scoring.
        if [[ -z "${DRY:-}" ]]; then
            while :; do
                q=$(queued_tasks)
                (( q + EVAL_TASKS <= MAX_QUEUED )) && break
                echo "  queue at ${q} tasks; waiting for room for eval split ${k}..."
                sleep 300
            done
        fi
        DEP=()
        # Chain onto this split's training only when we submitted it THIS run and it is on
        # the eval cluster. afterany: wait for training to finish, success or not, so a
        # partial failure still gets its good checkpoints scored (eval skips the rest).
        if [[ " ${STAGES} " == *" train "* && -n "${DEP_JOBS[$k]:-}" ]]; then
            DEP=(--dependency="afterany:${DEP_JOBS[$k]}")
        fi
        CMD=(sbatch --clusters="$ECLUSTER" --partition="$EPARTITION" --account="$ACCOUNT"
             --array="0-$((EVAL_TASKS - 1))%${EVAL_CONCURRENCY}" "${DEP[@]}"
             --export=ALL,CREDITPFN_CONFIG="$CONFIG",CREDITPFN_SPLIT_INDEX="$k",EVAL_TASKS="$EVAL_TASKS"
             "$EVAL_JOB")
        if [[ -n "${DRY:-}" ]]; then echo "DRY: ${CMD[*]}"; else submit_retry "${CMD[@]}" >/dev/null; fi
    done
    echo "submitted $(( N_SPLITS - SPLIT_START )) eval array job(s) of ${EVAL_TASKS} tasks to ${EVAL_DEST}."
fi
