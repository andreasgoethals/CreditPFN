#!/bin/bash
# =============================================================================
#  Shared conda-env activator for every CreditPFN slurm script.
# =============================================================================
#
#  Why this file exists
#  --------------------
#  The previous `export PATH="${VSC_DATA}/miniconda3/bin:${PATH}"; source
#  activate CreditPFN` pattern was fragile: it assumed conda lives at exactly
#  ${VSC_DATA}/miniconda3 and relied on the slurm shell having sourced
#  ~/.bashrc. If either assumption broke the job would happily continue with
#  the system Python and then explode at the first `import numpy`.
#
#  This script tries every install location we've seen in the wild and uses
#  the real conda shell hook (which respects whatever the user actually
#  installed). If none of them work it fails loud BEFORE the python invocation.
#
#  Usage
#  -----
#  Inside any `.slurm` script (with `#!/bin/bash -l` shebang and after `cd`
#  into the repo root):
#
#      source scripts/slurm/_activate_env.sh
#
#  Optional: set `CONDA_ENV=<name>` before sourcing to use a different env name
#  (default: CreditPFN).
# =============================================================================

CONDA_ENV="${CONDA_ENV:-CreditPFN}"

# --------------------------------------------------------------------------
# An active virtualenv SHADOWS conda and wins silently (2026-08-05).
# `#!/bin/bash -l` sources ~/.bashrc; if that auto-activates a venv, its
# bin/ sits ahead of the conda env's on PATH. `conda activate` then reports
# success while `python` and `pip` still resolve to the VENV — so a package
# installed "into CreditPFN" lands somewhere the job never looks, and vice
# versa. Observed interactively: `conda activate CreditPFN; pip install ...`
# wrote to a venv belonging to an entirely different project.
# Neutralise it before touching conda: drop the venv's bin/ from PATH and
# unset the marker.
# --------------------------------------------------------------------------
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "WARNING: a virtualenv is active (${VIRTUAL_ENV}) and would shadow" >&2
    echo "         the conda env. Removing it from PATH for this job." >&2
    PATH="$(echo "${PATH}" | tr ':' '\n' | grep -v "^${VIRTUAL_ENV}/bin$" | paste -sd: -)"
    export PATH
    unset VIRTUAL_ENV
    unset VIRTUAL_ENV_PROMPT 2>/dev/null || true
fi

_try_source_conda() {
    # Args: a candidate `conda.sh` path. Returns 0 on success.
    if [[ -f "$1" ]]; then
        # shellcheck disable=SC1090
        source "$1"
        return 0
    fi
    return 1
}

# 1) Already-initialised conda from the user's ~/.bashrc (the most common case
#    when the shebang above is `#!/bin/bash -l`). $CONDA_EXE is set when
#    `conda init` has run successfully.
if [[ -n "${CONDA_EXE:-}" ]] && [[ -x "${CONDA_EXE}" ]]; then
    eval "$(${CONDA_EXE} shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    # 2) Search the common manual-install locations on VSC. Order: data root
    #    first (because $VSC_DATA survives scratch purges), then $HOME.
    _try_source_conda "${VSC_DATA:-}/miniconda3/etc/profile.d/conda.sh"  || \
    _try_source_conda "${VSC_DATA:-}/miniforge3/etc/profile.d/conda.sh"  || \
    _try_source_conda "${VSC_DATA:-}/mambaforge/etc/profile.d/conda.sh"  || \
    _try_source_conda "${HOME}/miniconda3/etc/profile.d/conda.sh"        || \
    _try_source_conda "${HOME}/miniforge3/etc/profile.d/conda.sh"        || \
    _try_source_conda "${HOME}/mambaforge/etc/profile.d/conda.sh"        || \
    _try_source_conda "/apps/leuven/rocky9/sapphirerapids/2024a/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh"  || {
        echo "ERROR: could not locate a conda/mamba installation." >&2
        echo "       Searched \$CONDA_EXE, \$PATH, then:" >&2
        echo "         \$VSC_DATA / \$HOME under {miniconda3, miniforge3, mambaforge}" >&2
        echo "       Either run 'conda init bash' in your ~/.bashrc, or" >&2
        echo "       install miniforge at \$VSC_DATA/miniforge3 and re-submit." >&2
        exit 1
    }
fi

# --------------------------------------------------------------------------
# Activate + VERIFY. `conda activate` can "succeed" on a broken or EMPTY env
# (created without python, or whose install was interrupted) — the env's bin/
# then contributes nothing to PATH and `python` silently falls through to
# /bin/python (observed on the 2026-07-07 probe_row_cap job: "Active python:
# /bin/python"). So after activating we verify the interpreter really belongs
# to the env AND carries the project deps; on failure we FALL BACK to the
# base env (which may hold the full stack when the named env was never built)
# before giving up — a running job with a loud warning beats a dead one.
# --------------------------------------------------------------------------

_prepend_conda_bin() {
    # `conda activate` sets CONDA_PREFIX but does not always WIN the PATH race.
    # On a Genius login node with a Python Lmod module loaded, `python` still
    # resolved to /apps/leuven/.../Python/3.12.3/bin/python while CONDA_PREFIX
    # correctly pointed at our env — so the health check below rejected a
    # perfectly good env and the launcher aborted (observed 2026-08-05 from
    # run_full_pipeline.sh; the same activation worked on a compute node, which
    # has no such module loaded). Forcing the env's bin to the front makes the
    # interpreter match CONDA_PREFIX. A duplicate PATH entry is harmless;
    # `hash -r` clears bash's cached location for an already-resolved `python`.
    [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]] || return 0
    PATH="${CONDA_PREFIX}/bin:${PATH}"
    export PATH
    hash -r 2>/dev/null || true
    return 0
}

_env_is_healthy() {
    # The active python must live under $CONDA_PREFIX and import the deps.
    local py
    py="$(command -v python || true)"
    if [[ -z "${CONDA_PREFIX:-}" || "${py}" != "${CONDA_PREFIX}"* ]]; then
        echo "  [activate] python (${py:-none}) is NOT inside CONDA_PREFIX (${CONDA_PREFIX:-unset}) — env is broken/empty." >&2
        return 1
    fi
    local err
    if ! err=$(python -c "import numpy, torch, omegaconf, tabpfn" 2>&1); then
        echo "  [activate] dependency import failed in env '${CONDA_DEFAULT_ENV:-?}':" >&2
        echo "${err}" | head -3 | sed 's/^/      /' >&2
        return 1
    fi
    return 0
}

_activated=""
if conda activate "${CONDA_ENV}" 2>/dev/null && _prepend_conda_bin && _env_is_healthy; then
    _activated="${CONDA_ENV}"
else
    echo "WARNING: env '${CONDA_ENV}' is unusable — trying the 'base' env as a fallback." >&2
    if conda activate base 2>/dev/null && _prepend_conda_bin && _env_is_healthy; then
        _activated="base"
        echo "WARNING: running in the BASE env. Repair the named env when convenient:" >&2
        echo "         conda create -y -n ${CONDA_ENV} --clone base" >&2
        echo "         conda activate ${CONDA_ENV} && pip install -e \".[dev]\"" >&2
    fi
fi

if [[ -z "${_activated}" ]]; then
    echo "ERROR: no usable conda env (tried '${CONDA_ENV}' and 'base')." >&2
    echo "       Available envs:" >&2
    conda env list >&2
    echo "       Repair once from a login node:" >&2
    echo "         conda create -y -n ${CONDA_ENV} --clone base      # reuses base's torch/CUDA stack" >&2
    echo "         conda activate ${CONDA_ENV}" >&2
    echo "         pip install -e \".[dev]\"" >&2
    echo "         pip install --upgrade 'tabpfn @ git+https://github.com/PriorLabs/tabPFN.git@main'" >&2
    exit 1
fi

echo "Active conda env: ${CONDA_DEFAULT_ENV:-?} ($(command -v python))"


# =============================================================================
#  Resolve CREDITPFN_DATA_ROOT from config/data.yaml (one source of truth)
# =============================================================================
#
#  The slurm boilerplate above set `CREDITPFN_DATA_ROOT` to either the user's
#  explicit export or `$VSC_SCRATCH/CreditPFN`. Now that conda is active we
#  can finally consult `config/data.yaml`'s `paths.data_source` knob and
#  re-resolve. Precedence (mirroring src/utils/paths.apply_data_source_from_cfg):
#
#    1. Explicit user export (CREDITPFN_DATA_ROOT set on submission)
#    2. `cfg.paths.data_source = "data"`    → $VSC_DATA/CreditPFN
#    3. `cfg.paths.data_source = "scratch"` → $VSC_SCRATCH/CreditPFN  (VSC default)
#
#  We honour an explicit user export by checking whether the env var differs
#  from the standard slurm default (the value the .slurm script just set).
#  If the user wants a one-off override they can set
#  `CREDITPFN_DATA_ROOT=/some/path bash scripts/slurm/run_full_pipeline.sh`
#  and the value will pass through unchanged.

_resolved_data_root=$(python -c "
from omegaconf import OmegaConf
from src.utils.paths import apply_data_source_from_cfg
print(apply_data_source_from_cfg(OmegaConf.load('config/data.yaml')))
" 2>/dev/null)

if [[ -n "${_resolved_data_root}" ]]; then
    export CREDITPFN_DATA_ROOT="${_resolved_data_root}"
fi
unset _resolved_data_root
