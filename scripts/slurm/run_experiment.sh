#!/usr/bin/env bash
# Submit a prepared experiment. Preview with DRY=1. See docs/VSC.md.
# A shared per-controller pool bounds all arrays submitted through this launcher.
# Short, realistic requests improve backfill opportunities; they do not buy priority.
set -euo pipefail
CONFIG="${1:?usage: run_experiment.sh <config.yaml>}"
[[ -f "$CONFIG" ]] || { echo "No such config: $CONFIG" >&2; exit 1; }
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
cd "$(dirname "$0")/../.."

# One lightweight configuration read; --list-trials is not needed repeatedly.
mapfile -t META < <(python - "$CONFIG" <<'PYCFG'
import sys
from omegaconf import OmegaConf
from scripts.train_pipeline import _load_cfg, _resolve_grid
cfg = _load_cfg(config_path=sys.argv[1])
for key in ('track', 'run_name', 'corpus.n_splits', 'train.target_total_steps',
            'train.dataloader_workers', 'experiment.require_plan'):
    print(OmegaConf.select(cfg, key, default=''))
for t in _resolve_grid(cfg, single=False):
    print(t[0].rsplit('/', 1)[-1], t[5])
PYCFG
)
(( ${#META[@]} > 6 )) || { echo 'Could not resolve experiment configuration' >&2; exit 1; }
# Python on Windows emits CRLF, including during local dry checks.
for i in "${!META[@]}"; do META[$i]="${META[$i]%$'\r'}"; done
TRACK="${META[0]}"; RUN="${META[1]}"; BUDGET="${META[3]}"
[[ "$TRACK" == pd || "$TRACK" == lgd ]] || exit 1
N_SPLITS="${SPLITS:-${META[2]}}"; N_SPLITS="${N_SPLITS:-1}"
N_TRIALS=$(( ${#META[@]} - 6 ))
SPLIT_START="${SPLIT_START:-0}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-1}"
GLOBAL_CONCURRENCY="${GLOBAL_CONCURRENCY:-16}"
THROTTLE="${THROTTLE:-4}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-4}"
MAX_QUEUED="${MAX_QUEUED:-450}"
ACCOUNT="${CREDITPFN_ACCOUNT:-lp_verbekelab}"
STAGES="${STAGES:-train}"
export EVAL_TASKS="${EVAL_TASKS:-16}"
export CREDITPFN_DATALOADER_WORKERS="${CREDITPFN_DATALOADER_WORKERS:-${META[4]}}"
export CREDITPFN_USE_SCRATCH="${CREDITPFN_USE_SCRATCH:-1}"
export CREDITPFN_REQUIRE_STAGING="${CREDITPFN_REQUIRE_STAGING:-1}"
# Opt in to short resumable segments after the null/pilot gate. The schedule still
# spans the complete update budget. Ten minutes cover monitoring/save/exit work.
SEGMENT_MINUTES="${SEGMENT_MINUTES:-0}"
export CREDITPFN_SEGMENT_SECONDS=$(( SEGMENT_MINUTES * 60 ))
export CREDITPFN_AUTO_REQUEUE="${CREDITPFN_AUTO_REQUEUE:-0}"
(( TRIALS_PER_TASK > 0 && GLOBAL_CONCURRENCY > 0 && THROTTLE > 0 && EVAL_CONCURRENCY > 0 )) || exit 1
(( THROTTLE <= GLOBAL_CONCURRENCY && EVAL_CONCURRENCY <= GLOBAL_CONCURRENCY )) || {
    echo 'Array concurrency exceeds GLOBAL_CONCURRENCY' >&2; exit 1;
}
(( SPLIT_START >= 0 && SPLIT_START < N_SPLITS && N_SPLITS <= ${META[2]:-1} )) || {
    echo 'Requested splits fall outside the prepared experiment' >&2; exit 1;
}
if (( SEGMENT_MINUTES > 0 && TRIALS_PER_TASK != 1 )); then
    echo 'Segmented training requires TRIALS_PER_TASK=1' >&2; exit 1
fi
if [[ "$STAGES" != train && "$STAGES" != eval && "$STAGES" != 'train eval' ]]; then
    echo 'STAGES must be train, eval, or "train eval"' >&2; exit 1
fi
if [[ -z "${DRY:-}" && "${META[5]}" == True ]]; then
    python - "$CONFIG" <<'PYPLAN'
import sys
from pathlib import Path
from src.utils.prepare_experiment import check_prepared
check_prepared(Path(sys.argv[1]))
PYPLAN
fi

hms() { printf '%d:%02d:00' $(( $1 / 60 )) $(( $1 % 60 )); }
walltime_for() {
    local base="$1" mode="$2" rate
    if (( SEGMENT_MINUTES > 0 )); then hms $(( SEGMENT_MINUTES + 10 )); return; fi
    if [[ "$mode" == accumulate && -n "${ACC_WALLTIME:-}" ]]; then echo "$ACC_WALLTIME"; return; fi
    if [[ -n "${WALLTIME:-}" ]]; then echo "$WALLTIME"; return; fi
    if (( BUDGET <= 250 )); then hms 60; return; fi
    # Historical full_pass/accumulate measurements are provisional for protocol 4.
    # Override WALLTIME with the pilot report; never shorten measured context caps.
    rate=90
    if [[ "$mode" == accumulate ]]; then
        case "$base" in
            *tabicl*) rate=600 ;; *v2.6*) rate=810 ;; *v3*) rate=720 ;; *) rate=1170 ;;
        esac
    fi
    hms $(( TRIALS_PER_TASK * rate + 30 ))
}
route_for() {
    if [[ "${ROUTE:-1}" == 0 ]]; then echo 'mindwell gpu_b200'; return; fi
    case "$1" in
        *tabicl-*) echo "${TABICL_DEST:-mindwell gpu_b200}" ;;
        *) echo "${TABPFN_DEST:-mindwell gpu_b200}" ;;
    esac
}
export CREDITPFN_EVAL_KIND="${EVAL_KIND:-foundation}"
EVAL_JOB="scripts/slurm/eval_${TRACK}.slurm"
if [[ "$CREDITPFN_EVAL_KIND" == classical ]]; then
    [[ "$STAGES" == eval ]] || { echo 'Classical CPU scoring is a separate STAGES=eval submission' >&2; exit 1; }
    EVAL_JOB=scripts/slurm/eval_classical.slurm
    read -r ECLUSTER EPARTITION <<< "${EVAL_DEST:-wice batch}"
    EVAL_WALLTIME="${EVAL_WALLTIME:-04:00:00}"
else
    read -r ECLUSTER EPARTITION <<< "${EVAL_DEST:-mindwell gpu_b200}"
fi
declare -A TASK_KEY BUCKET DEP_JOBS
for (( t=0; t<N_TRIALS; t++ )); do
    read -r base mode <<< "${META[$((t+6))]}"
    destination="$(route_for "$base")"
    if [[ "$STAGES" == 'train eval' && "${destination%% *}" != "$ECLUSTER" ]]; then
        echo 'Combined train/eval requires one controller. Submit STAGES=eval after cross-cluster training finishes.' >&2
        exit 1
    fi
    task=$(( t / TRIALS_PER_TASK ))
    key="$destination|$mode|$base"
    if [[ -n "${TASK_KEY[$task]:-}" && "${TASK_KEY[$task]}" != "$key" ]]; then
        echo "Task $task mixes bases or pass modes; use TRIALS_PER_TASK=1" >&2; exit 1
    fi
    TASK_KEY[$task]="$key"
done
for task in $(printf '%s\n' "${!TASK_KEY[@]}" | sort -n); do
    key="${TASK_KEY[$task]}"
    BUCKET["$key"]="${BUCKET[$key]:+${BUCKET[$key]},}$task"
done
mapfile -t KEYS < <(printf '%s\n' "${!BUCKET[@]}" | sort)
echo "$RUN $TRACK: $N_TRIALS trials/partition x $N_SPLITS partitions; shared cap=$GLOBAL_CONCURRENCY/controller"
echo "workers=$CREDITPFN_DATALOADER_WORKERS scratch=$CREDITPFN_USE_SCRATCH segment=${SEGMENT_MINUTES}min"

queued_tasks() {
    # Array elements, not array headers. Count all states against the submit quota.
    squeue -M "$1" -u "$USER" -h -r -o '%i' | wc -l
}
wait_for_room() {
    [[ -n "${DRY:-}" ]] && return
    local q
    while :; do
        q="$(queued_tasks "$1")"
        (( q + $2 <= MAX_QUEUED )) && return
        echo "Controller $1 has $q submitted tasks; waiting for quota room." >&2
        sleep 60
    done
}
submit_retry() {
    local out rc
    while :; do
        if out=$("$@" 2>&1); then echo "$out"; return 0; else rc=$?; fi
        if [[ "$out" == *QOSMaxSubmitJobPerUserLimit* || "$out" == *'job submit limit'* ]]; then
            echo 'Submission quota reached; waiting for room.' >&2; sleep 60; continue
        fi
        echo "$out" >&2; return "$rc"
    done
}
submit_array() {
    local cluster="$1" limit="$2"; shift 2
    if [[ -n "${DRY:-}" ]]; then
        echo "DRY: pool=$cluster cap=$GLOBAL_CONCURRENCY array_limit=$limit $*"
    else
        submit_retry python -m src.utils.submit_bounded --slots "$GLOBAL_CONCURRENCY" --limit "$limit" --cluster "$cluster" -- "$@"
    fi
}
if [[ " $STAGES " == *' train '* ]]; then
    for (( k=SPLIT_START; k<N_SPLITS; k++ )); do
        for key in "${KEYS[@]}"; do
            IFS='|' read -r destination mode base <<< "$key"
            read -r cluster partition <<< "$destination"
            wt="$(walltime_for "$base" "$mode")"
            [[ "$cluster" == wice ]] && cpus=16 || cpus=24
            n=$(awk -F, '{print NF}' <<< "${BUCKET[$key]}")
            limit=$THROTTLE; (( n < limit )) && limit=$n
            wait_for_room "$cluster" "$n"
            CMD=(sbatch --parsable --clusters="$cluster" --partition="$partition" --account="$ACCOUNT"
                 --array="${BUCKET[$key]}%${limit}" --time="$wt" --cpus-per-task="$cpus"
                 --signal=B:USR1@600 --requeue
                 --export=ALL,CREDITPFN_CONFIG="$CONFIG",CREDITPFN_SPLIT_INDEX="$k",CREDITPFN_TRIALS_PER_TASK="$TRIALS_PER_TASK"
                 "scripts/slurm/train_${TRACK}.slurm")
            out="$(submit_array "$cluster" "$limit" "${CMD[@]}")"
            if [[ -n "${DRY:-}" ]]; then echo "$out"; else
                jid="${out%%;*}"; DEP_JOBS[$k]="${DEP_JOBS[$k]:+${DEP_JOBS[$k]}:}$jid"
            fi
        done
    done
fi
if [[ " $STAGES " == *' eval '* ]]; then
    for (( k=SPLIT_START; k<N_SPLITS; k++ )); do
        wait_for_room "$ECLUSTER" "$EVAL_TASKS"
        DEP=()
        [[ -n "${DEP_JOBS[$k]:-}" ]] && DEP=(--dependency="afterany:${DEP_JOBS[$k]}")
        CMD=(sbatch --parsable --clusters="$ECLUSTER" --partition="$EPARTITION" --account="$ACCOUNT"
             --array="0-$((EVAL_TASKS-1))%${EVAL_CONCURRENCY}" --time="${EVAL_WALLTIME:-02:00:00}" "${DEP[@]}"
             --export=ALL,CREDITPFN_CONFIG="$CONFIG",CREDITPFN_SPLIT_INDEX="$k",EVAL_TASKS="$EVAL_TASKS"
             "$EVAL_JOB")
        submit_array "$ECLUSTER" "$EVAL_CONCURRENCY" "${CMD[@]}"
    done
fi
[[ -n "${DRY:-}" ]] && echo 'Preview complete; no jobs submitted.' || echo 'Submission complete; inspect queue and per-trial outcomes.'
