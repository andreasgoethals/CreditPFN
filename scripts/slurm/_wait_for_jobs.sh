#!/bin/bash
# =============================================================================
#  eval-gate: block until the Mindwell training jobs finish, then verify SUCCESS.
# =============================================================================
#  Used by run_full_pipeline.sh to bridge the train→eval cross-cluster gap
#  WITHOUT a cross-cluster afterok (VSC doesn't support those) and WITHOUT
#  holding a GPU while waiting: this runs as a cheap 1-CPU job on wICE, and the
#  eval GPU arrays `afterok`-depend on it, so they stay PENDING until release.
#
#  Post-mortem hardening (2026-07-04 run): the previous gate released on queue
#  COMPLETION alone — all 64 training tasks had FAILED, yet eval was released
#  against an empty trained-model manifest, and the gate log contained just two
#  lines for a 4.7-hour watch. This version:
#    * logs EVERY poll (timestamp + raw remaining-count) so any misfire is
#      diagnosable after the fact;
#    * requires 2 CONSECUTIVE empty-queue polls (guards transient empty reads);
#    * after the queue drains, checks the per-track SUCCESS sentinels that
#      train_{pd,lgd}.slurm write only when a trial actually SAVED — and
#      says loudly whether eval is being released onto real checkpoints or
#      onto a baselines-only roster.
#  It still always exits 0 (eval's skip-existing logic is safe either way and
#  the timeout is the backstop) — but the log now tells the whole story.
#
#  Usage:  _wait_for_jobs.sh <comma-separated-mindwell-jobids> [max_polls]
#  Env:    CREDITPFN_OUTPUT_ROOT (for the .sentinels dir; default $VSC_DATA/CreditPFN)
# =============================================================================
set -uo pipefail

JIDS="${1:?usage: _wait_for_jobs.sh <comma-separated-jobids> [max_polls]}"
MAX="${2:-2400}"                       # polls × 30s  (2400 = 20 h)
OUT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"
SENT_DIR="${OUT}/.sentinels"

echo "$(date '+%F %T') eval-gate: watching Mindwell train jobs [${JIDS}] (max $((MAX * 30 / 3600)) h, poll=30s)"
echo "$(date '+%F %T') eval-gate: success sentinels expected under ${SENT_DIR}/train_ok_{pd,lgd}"

consecutive_empty=0
for i in $(seq 1 "${MAX}"); do
    out=$(squeue -M mindwell -h -o "%i %T" -j "${JIDS}" 2>&1)
    rc=$?
    if [ ${rc} -eq 0 ]; then
        remaining=$(printf '%s\n' "${out}" | grep -cE '^[0-9]' || true)
        echo "$(date '+%F %T') eval-gate: poll ${i}  remaining=${remaining}  [$(printf '%s' "${out}" | grep -E '^[0-9]' | tr '\n' ';' | cut -c1-160)]"
        if [ "${remaining}" -eq 0 ]; then
            consecutive_empty=$((consecutive_empty + 1))
            if [ "${consecutive_empty}" -ge 2 ]; then
                echo "$(date '+%F %T') eval-gate: queue drained (2 consecutive empty polls) after $((i * 30)) s."
                break
            fi
        else
            consecutive_empty=0
        fi
    else
        # A FAILED query must never count as completion — keep waiting.
        echo "$(date '+%F %T') eval-gate: poll ${i}  squeue FAILED (rc=${rc}): $(printf '%s' "${out}" | head -c 200)"
        consecutive_empty=0
    fi
    sleep 30
done

# --- success verification ----------------------------------------------------
ok_pd=""; ok_lgd=""
[ -f "${SENT_DIR}/train_ok_pd"  ] && ok_pd="yes"
[ -f "${SENT_DIR}/train_ok_lgd" ] && ok_lgd="yes"
echo "$(date '+%F %T') eval-gate: success sentinels — pd=${ok_pd:-NO} lgd=${ok_lgd:-NO}"

if [ -z "${ok_pd}" ] && [ -z "${ok_lgd}" ]; then
    echo "=================================================================="
    echo "WARNING: NO training task on EITHER track reported success."
    echo "         Eval will run against a baselines-only roster (no trained"
    echo "         checkpoints). Check the train_*.log files on \$VSC_DATA"
    echo "         before trusting any downstream comparison."
    echo "=================================================================="
elif [ -z "${ok_pd}" ] || [ -z "${ok_lgd}" ]; then
    echo "WARNING: only ONE track has successful trials (pd=${ok_pd:-NO} lgd=${ok_lgd:-NO})."
fi

echo "$(date '+%F %T') eval-gate: releasing eval."
exit 0
