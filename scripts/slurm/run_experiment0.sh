#!/usr/bin/env bash
# One login-node command; GPU work only starts after the CPU checks pass.
set -euo pipefail
cd "$(dirname "$0")/../.."
PART="${1:-part1}"
[[ "$PART" == part1 || "$PART" == part2 || "$PART" == recovery ]] || { echo 'Choose part1, part2 or recovery' >&2; exit 2; }
# The workflow chooses its own task configs; an inherited one can reroute job logs.
unset CREDITPFN_CONFIG
export CREDITPFN_EXPERIMENT=experiment0
export CREDITPFN_OUTPUT_ROOT="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA:?}/CreditPFN}"
export CREDITPFN_STAGING_ROOT="${CREDITPFN_STAGING_ROOT:-/lustre1/project/stg_00211/CreditPFN}"
# Network access belongs on the login node. This reuses checksum-verified downloads.
unset CREDITPFN_DATA_ROOT CREDITPFN_BASE_CACHE_ROOT
python -m src.utils.prepare_retention
python -m src.utils.experiment0 start --part "$PART"
