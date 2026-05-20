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

# Actually activate the project env.
if ! conda activate "${CONDA_ENV}" 2>/dev/null; then
    echo "ERROR: 'conda activate ${CONDA_ENV}' failed." >&2
    echo "       Available envs:" >&2
    conda env list >&2
    echo "       Create the env once with:" >&2
    echo "         mamba create -y -n ${CONDA_ENV} python=3.12" >&2
    echo "         source activate ${CONDA_ENV}" >&2
    echo "         pip install -r requirements.txt" >&2
    exit 1
fi

# Sanity-check the env has the project deps the slurm job will need. Fail
# loud and early — better than a 40-line traceback halfway through the data
# pipeline.
if ! python -c "import numpy, torch, omegaconf, tabpfn" 2>/dev/null; then
    echo "ERROR: conda env '${CONDA_ENV}' is missing project dependencies." >&2
    echo "       Active python: $(command -v python)" >&2
    echo "       Reinstall with:" >&2
    echo "         conda activate ${CONDA_ENV}" >&2
    echo "         pip install -r requirements.txt" >&2
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
#  `CREDITPFN_DATA_ROOT=/some/path bash scripts/slurm/submit_full_pipeline.sh`
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
