#!/usr/bin/env bash
# Request optimizer-boundary recovery without signalling a non-exec runtime wrapper.
run_training_python() {
    local child rc interrupted previous_usr1 previous_term request_scope stop_request
    previous_usr1="$(trap -p USR1)"
    previous_term="$(trap -p TERM)"
    request_scope=$(mktemp -d "${CREDITPFN_OWNED_CACHE:-${TMPDIR:-/tmp}}/train-stop.XXXXXXXX") || return
    stop_request="$request_scope/requested"
    # The site python wrapper is a Bash process rather than the native trainer.
    # A signal to $! can kill that wrapper without reaching Python. This owned
    # file crosses the container bind boundary and never signals data workers.
    trap 'interrupted=1; : > "$stop_request"' USR1 TERM
    CREDITPFN_STOP_REQUEST_FILE="$stop_request" python "$@" &
    child=$!
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
    rm -f -- "$stop_request"
    rmdir -- "$request_scope"
    return "$rc"
}
