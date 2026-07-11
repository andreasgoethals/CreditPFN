#!/bin/bash
# =============================================================================
#  CreditPFN — FIRE-ONCE full pipeline (data → train → eval), one command.
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
#      bash scripts/slurm/run_full_pipeline.sh                 # everything
#      STAGES="eval"       bash scripts/slurm/run_full_pipeline.sh   # re-score only
#      STAGES="data train" bash scripts/slurm/run_full_pipeline.sh   # skip eval
#  Knobs (env): STAGES (any subset of "data train eval"), TRACKS,
#      TRAIN_ACCOUNT (default lp_verbekelab), EVAL_ACCOUNT,
#      TRAIN_CONCURRENCY, EVAL_CONCURRENCY, EVAL_PARTITIONS, CONDA_ENV.
#  Stage coupling is handled automatically: train only waits for the data
#  sentinel when data was submitted in the same invocation, and eval only
#  goes through the training gate when train was submitted too (a bare
#  STAGES=eval submits the eval arrays immediately — the skip-existing
#  logic makes re-scoring idempotent).
# =============================================================================
set -euo pipefail

STAGES="${STAGES:-data train eval}"
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

# NOTE: if-form (not `[[ ]] && var=1`) — under `set -e` a false test at the
# end of a && list aborts the whole script.
run_data=""; run_train=""; run_eval=""
if [[ " ${STAGES} " == *" data "*  ]]; then run_data=1;  fi
if [[ " ${STAGES} " == *" train "* ]]; then run_train=1; fi
if [[ " ${STAGES} " == *" eval "*  ]]; then run_eval=1;  fi

echo "=================================================================="
echo "CreditPFN — one-shot full pipeline"
echo "  STAGES        : ${STAGES}"
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

mkdir -p "${CREDITPFN_OUTPUT_ROOT}/.sentinels" "${CREDITPFN_OUTPUT_ROOT}/logs"

# --- [1] DATA (wICE) ---------------------------------------------------------
DATA_JID=""
if [[ -n "${run_data}" ]]; then
    # Clear the stale completion sentinel so this run's train tasks wait for
    # THIS run's data job, not a previous run's leftover marker.
    rm -f "${CREDITPFN_OUTPUT_ROOT}/.sentinels/data_done"
    DATA_JID=$(strip "$(sbatch --parsable --export="${SBATCH_EXPORT}" scripts/slurm/data.slurm)")
    echo "  [1] data  (wICE batch)        : ${DATA_JID}"
else
    echo "  [1] data  : skipped (not in STAGES) — training will use the processed CSVs already in staging."
fi

# --- [2] TRAIN (Mindwell); waits for the data sentinel only if data runs too --
TRAIN_CSV=""
if [[ -n "${run_train}" ]]; then
    # Clear stale PER-TASK success sentinels. One file per array index lets the
    # gate distinguish a complete grid from "at least one trial succeeded".
    rm -f "${CREDITPFN_OUTPUT_ROOT}/.sentinels/train_ok_pd" \
          "${CREDITPFN_OUTPUT_ROOT}/.sentinels/train_ok_lgd" \
          "${CREDITPFN_OUTPUT_ROOT}/.sentinels/"train_ok_pd_* \
          "${CREDITPFN_OUTPUT_ROOT}/.sentinels/"train_ok_lgd_*
    WAIT_FLAG=""
    [[ -n "${run_data}" ]] && WAIT_FLAG=",CREDITPFN_WAIT_DATA=1"
    for TR in ${TRACKS}; do
        N=$(python scripts/train_pipeline.py --list-trials track="${TR}")
        JID=$(strip "$(sbatch --parsable \
            --export="${SBATCH_EXPORT}${WAIT_FLAG}" \
            --account="${TRAIN_ACCOUNT}" \
            --array=0-$((N - 1))%"${TRAIN_CONCURRENCY}" \
            "scripts/slurm/train_${TR}.slurm")")
        TRAIN_CSV="${TRAIN_CSV:+${TRAIN_CSV},}${JID}"
        echo "  [2] train ${TR} (Mindwell b200): ${JID:-<FAILED>}  (array 0..$((N - 1)))"
    done
else
    echo "  [2] train : skipped (not in STAGES)."
fi

# --- [3+4] EVAL — submitted only AFTER training finishes ---------------------
# CHANGED 2026-07-08 (user request): eval is NO LONGER pre-queued while training
# runs. When training is in this batch, the gate job (a cheap 1-CPU wICE job)
# waits for training to finish and THEN submits the eval arrays - as long as at
# least one trial saved a checkpoint. A partial grid is scored and LOUDLY
# flagged as partial (rather than skipped entirely), so a single diverged trial
# doesn't waste the whole sweep, while incomplete results can't be mistaken for
# a complete experiment. No eval GPU job sits queued during training.
# Eval-only invocations submit immediately.
#
# The eval commands are generated here but compute the ACTUAL post-training
# roster size when the generated script executes. The old planned-size estimate
# launched surplus tasks whenever training was incomplete. Stride-interleaved
# pool splitting keeps every GPU pool busy without overlap.
GATE_JID=""
if [[ -n "${run_eval}" ]]; then
    EVAL_SCRIPT="${CREDITPFN_OUTPUT_ROOT}/.sentinels/eval_submit_$(date +%Y%m%d_%H%M%S).sh"
    gated=0; [[ -n "${run_train}" ]] && gated=1
    {
        echo "#!/bin/bash"
        echo "# Auto-generated by run_full_pipeline.sh - submits exact eval arrays."
        echo "set -euo pipefail; cd '${REPO}'"
    } > "${EVAL_SCRIPT}"
    # shellcheck disable=SC2206
    PARTS=(${EVAL_PARTITIONS}); K=${#PARTS[@]}
    for TR in ${TRACKS}; do
        PLANNED=$(python scripts/train_pipeline.py --list-trials track="${TR}" 2>/dev/null || echo 0)
        {
            echo ""
            echo "# ---- eval ${TR} (exact roster computed at execution) ----"
            if [[ "${gated}" == "1" ]]; then
                echo "EXPECTED=${PLANNED}"
                echo "SUCCESS=\$(find '${CREDITPFN_OUTPUT_ROOT}/.sentinels' -maxdepth 1 -type f -name 'train_ok_${TR}_*' | wc -l)"
                # Robust gate (Claude, 2026-07-10): skip eval ONLY when NOTHING
                # trained. A single diverged/failed/preempted trial (an EXPECTED
                # outcome in a 4-LR sweep) must not skip the whole track's eval —
                # the trained checkpoints that DO exist are still worth scoring.
                # Partial grids are loudly flagged so results are never mistaken
                # for a complete homogeneous experiment (Codex's valid concern).
                echo 'if [[ "$SUCCESS" -lt 1 ]]; then'
                echo "  echo \"eval ${TR}: SKIPPED - no training task succeeded (0/\${EXPECTED}).\""
                echo "else"
                echo '  if [[ "$SUCCESS" != "$EXPECTED" ]]; then'
                echo "    echo \"eval ${TR}: WARNING - PARTIAL training grid (\${SUCCESS}/\${EXPECTED} tasks succeeded). Scoring the checkpoints that exist; treat results as PARTIAL and re-run eval after the grid completes.\""
                echo "  fi"
            fi
            indent=""; [[ "${gated}" == "1" ]] && indent="  "
            echo "${indent}N=\$(python scripts/eval_pipeline.py --list-tasks track='${TR}')"
            echo "${indent}if [[ ! \"\${N}\" =~ ^[0-9]+$ ]] || [[ \"\${N}\" -lt 1 ]]; then"
            echo "${indent}  echo \"eval ${TR}: ERROR - invalid/empty task count: \${N}\" >&2"
            echo "${indent}  exit 1"
            echo "${indent}else"
            # MODEL-parity pool split (fix 2026-07-11): each pool gets the
            # explicit comma-list of task indices for model_idx % K == i, so
            # every pool covers ALL datasets. The old stride split (i-N:K)
            # aligned with the dataset count whenever they shared a factor —
            # LGD's 2 datasets × 2 pools sent ALL of lgd_lendingclub to the
            # slow A100 pool, leaving a whole dataset unscored for hours.
            for i in "${!PARTS[@]}"; do
                echo "${indent}  IDX=\$(python scripts/eval_pipeline.py --list-tasks --pools ${K} --pool ${i} track='${TR}')"
                echo "${indent}  if [[ -n \"\${IDX}\" ]]; then"
                echo "${indent}    sbatch --clusters=wice --account='${EVAL_ACCOUNT}' --partition='${PARTS[$i]}' --export='${SBATCH_EXPORT}' --array=\"\${IDX}%${EVAL_CONCURRENCY}\" scripts/slurm/eval_${TR}.slurm"
                echo "${indent}    echo \"submitted eval ${TR} on ${PARTS[$i]} (pool ${i}/${K} of \${N} tasks)\""
                echo "${indent}  fi"
            done
            echo "${indent}fi"
            if [[ "${gated}" == "1" ]]; then
                echo "fi"
            fi
        } >> "${EVAL_SCRIPT}"
    done

    if [[ -n "${run_train}" && -n "${TRAIN_CSV}" ]]; then
        GATE_JID=$(strip "$(sbatch --parsable \
            --clusters=wice --account="${EVAL_ACCOUNT}" --partition=batch \
            --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=21:00:00 \
            --job-name=creditpfn-eval-gate --chdir="${REPO}" \
            --export="${SBATCH_EXPORT}" \
            --output="${CREDITPFN_OUTPUT_ROOT}/logs/eval_gate_%j.log" \
            --wrap="bash scripts/slurm/_wait_for_jobs.sh '${TRAIN_CSV}' 2400 '${EVAL_SCRIPT}'")")
        echo "  [3] eval gate (wICE batch)    : ${GATE_JID}  — submits eval AFTER training finishes"
        echo "      (per-track, if >=1 trial trained; partial grids scored + flagged; no GPU eval queued meanwhile)"
    else
        echo "  [3+4] eval : no training in this batch — submitting eval now (re-score existing checkpoints)"
        bash "${EVAL_SCRIPT}"
    fi
else
    echo "  [3+4] eval : skipped (not in STAGES)."
fi

echo "=================================================================="
echo "Fired everything. Nothing else to do — check back tomorrow."
echo "  Watch:   squeue --me --clusters=wice,mindwell"
echo "  Logs:    ${CREDITPFN_OUTPUT_ROOT}/logs/"
echo "  Results: ${CREDITPFN_STAGING_ROOT}/output/results/  +  checkpoints/trained/"
echo "=================================================================="
