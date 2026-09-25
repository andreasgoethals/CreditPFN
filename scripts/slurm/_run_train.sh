#!/usr/bin/env bash
# Forward Slurm's batch-shell warning to the Python parent, leaving workers alive.
run_training_python() {
    local child rc interrupted previous_usr1 previous_term
    previous_usr1="$(trap -p USR1)"
    previous_term="$(trap -p TERM)"
    python "$@" &
    child=$!
    trap 'interrupted=1; kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
    while :; do
        interrupted=0
        if wait "$child"; then rc=0; else rc=$?; fi
        # wait returns early when a shell trap runs; wait again for the child's true exit code.
        [[ "$interrupted" == 1 ]] || break
    done
    # Restore setup/callback handlers, including the caller's nonzero exit
    # behavior, rather than leaving the batch shell unprotected between trials.
    if [[ -n "$previous_usr1" ]]; then eval "$previous_usr1"; else trap - USR1; fi
    if [[ -n "$previous_term" ]]; then eval "$previous_term"; else trap - TERM; fi
    return "$rc"
}
