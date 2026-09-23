#!/usr/bin/env bash
# Forward Slurm's batch-shell warning to the Python parent, leaving workers alive.
run_training_python() {
    local child rc interrupted
    python "$@" &
    child=$!
    trap 'interrupted=1; kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
    while :; do
        interrupted=0
        if wait "$child"; then rc=0; else rc=$?; fi
        # wait returns early when a shell trap runs; wait again for the child's true exit code.
        [[ "$interrupted" == 1 ]] || break
    done
    trap - USR1 TERM
    return "$rc"
}
