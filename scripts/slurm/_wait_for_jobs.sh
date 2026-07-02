#!/bin/bash
# =============================================================================
#  eval-gate: block until a set of Mindwell training jobs have all finished.
# =============================================================================
#  Used by submit_overnight.sh to bridge the train→eval cross-cluster gap
#  WITHOUT a cross-cluster afterok (which VSC doesn't support) and WITHOUT
#  holding a GPU while waiting: this runs as a cheap 1-CPU job on wICE, and the
#  eval GPU arrays `afterok`-depend on it, so they stay PENDING (no GPU held)
#  until training on Mindwell is done. It watches the train jobs via `squeue
#  -M mindwell`; always exits 0 so the eval arrays are always released
#  (on timeout they score whatever checkpoints exist so far).
#
#  Usage:  bash _wait_for_jobs.sh <comma-separated-jobids> [max_polls]
# =============================================================================
set -uo pipefail

JIDS="${1:?usage: _wait_for_jobs.sh <comma-separated-jobids> [max_polls]}"
MAX="${2:-2400}"                       # polls × 30s  (2400 = 20 h)

echo "eval-gate: watching Mindwell train jobs [${JIDS}] (up to $((MAX * 30 / 3600)) h) …"
for i in $(seq 1 "${MAX}"); do
    # Query the Mindwell queue for the train jobs. Guard on the exit code: only
    # a SUCCESSFUL query returning zero job-id lines means "training finished".
    # A failed query (transient connectivity, or ids not yet visible) must NOT
    # release eval early — keep waiting; the timeout is the backstop.
    out=$(squeue -M mindwell -h -o "%i" -j "${JIDS}" 2>/dev/null)
    if [ $? -eq 0 ]; then
        # Count lines starting with a digit (job ids) — ignores the
        # "CLUSTER: mindwell" banner. 0 ⇒ all train jobs have finished.
        remaining=$(printf '%s\n' "${out}" | grep -cE '^[0-9]' || true)
        if [ "${remaining}" -eq 0 ]; then
            echo "eval-gate: training complete after $((i * 30)) s — releasing eval."
            exit 0
        fi
    fi
    sleep 30
done
echo "eval-gate: TIMEOUT after $((MAX * 30 / 3600)) h — releasing eval anyway "
echo "           (it will score whatever checkpoints exist; re-run eval later if needed)."
exit 0
