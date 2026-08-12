#!/bin/bash -l
# =============================================================================
#  CreditPFN — relocate big files from $VSC_DATA into PROJECT STAGING
# =============================================================================
#
#  WHEN YOU NEED THIS
#  ------------------
#  The big files (datasets + base/trained checkpoints) must live in project
#  ("staging") storage so they persist and don't eat the tiny 75 GiB $VSC_DATA
#  quota. If you can upload from your laptop STRAIGHT to staging (scp/WinSCP to
#  the absolute path /lustre1/project/stg_00211/CreditPFN/...), you do NOT need
#  this script — see docs/VSC.md §0.
#
#  Use this ONLY if you can reach $VSC_DATA from your transfer tool but NOT the
#  project storage. Then: (1) upload from your laptop into
#  $VSC_DATA/CreditPFN/{data,checkpoints}; (2) submit THIS job. It runs on a
#  wICE compute node — which mounts the Lustre filesystem that holds staging —
#  and copies those trees into the project CreditPFN folder. After it succeeds
#  you can delete the $VSC_DATA copies to reclaim quota.
#
#  Submit:
#      sbatch scripts/slurm/stage_to_project.slurm
#  (Optionally pre-set CREDITPFN_STAGING_ROOT / TABPFN_STAGING_ROOT to override
#   the built-in /lustre1/project/stg_00211 default.)
# =============================================================================

#SBATCH --job-name=creditpfn-stage
#SBATCH --clusters=wice
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Propagate submission env so a bare cross-cluster `sbatch` doesn't hit
# Slurm's login-shell env retrieval ("user env retrieval failed requeued
# held"). run_full_pipeline.sh's CLI --export overrides this. (2026-07-08)
#SBATCH --export=ALL

set -euo pipefail
export PYTHONUNBUFFERED=1

SRC="${CREDITPFN_OUTPUT_ROOT:-${VSC_DATA}/CreditPFN}"   # where you uploaded

# Resolve the project CreditPFN folder (append /CreditPFN unless already there).
STAGING_BASE="${CREDITPFN_STAGING_ROOT:-${TABPFN_STAGING_ROOT:-/lustre1/project/stg_00211}}"
case "${STAGING_BASE}" in
    */CreditPFN) DEST="${STAGING_BASE}" ;;
    *)           DEST="${STAGING_BASE}/CreditPFN" ;;
esac

mkdir -p "${VSC_DATA}/CreditPFN/output/logs"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${VSC_DATA}/CreditPFN/output/logs/stage_to_project_${TS}_j${SLURM_JOB_ID}.log"
exec > "$LOG" 2>&1

echo "=========================================="
echo "CreditPFN — STAGE TO PROJECT STORAGE"
echo "=========================================="
echo "Job ID    : ${SLURM_JOB_ID}   node=${SLURMD_NODENAME:-?}   date=$(date)"
echo "SOURCE    : ${SRC}"
echo "DEST      : ${DEST}"
echo "Log file  : ${LOG}"
echo "=========================================="

mkdir -p "${DEST}/data" "${DEST}/checkpoints"

# rsync if available (resumable + verifies); else cp -rn (no-clobber).
copy_tree() {  # $1=subdir
    local sub="$1"
    if [[ ! -d "${SRC}/${sub}" ]]; then
        echo "  - ${SRC}/${sub} not present — skipping."
        return 0
    fi
    echo "  -> copying ${sub}/ ..."
    if command -v rsync >/dev/null 2>&1; then
        rsync -ah --info=progress2 "${SRC}/${sub}/" "${DEST}/${sub}/"
    else
        cp -rn "${SRC}/${sub}/." "${DEST}/${sub}/"
    fi
}

copy_tree data
copy_tree checkpoints

echo "------------------------------------------"
echo "Sizes after copy:"
du -sh "${DEST}/data" "${DEST}/checkpoints" 2>/dev/null || true
echo "------------------------------------------"
echo "DONE — $(date)"
echo
echo "Verify, then reclaim \$VSC_DATA quota with (CAREFUL — deletes the copies):"
echo "    rm -rf \"${SRC}/data\" \"${SRC}/checkpoints\""
echo "=========================================="
