#!/bin/bash
# Sourced by _activate_env after CPU thread/cache setup. The persistent runtime is immutable.
[[ "${CREDITPFN_RUNTIME_VERSION:-}" =~ ^[a-zA-Z0-9]+$ ]] || { echo 'Invalid runtime version'; return 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Container activation requires a Slurm allocation'; return 2; }
_runtime_project="${CREDITPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-/lustre1/project/stg_00211}}"
[[ "$(basename "$_runtime_project")" == CreditPFN ]] || _runtime_project="$_runtime_project/CreditPFN"
_persistent_runtime="$_runtime_project/runtimes/$CREDITPFN_RUNTIME_VERSION"
case "${SLURM_CLUSTER_NAME:-}" in
    mindwell) _runtime_native="${VSC_SCRATCH_GPFS1:?}" ;;
    wice) _runtime_native="${VSC_SCRATCH_LUSTRE1:?}" ;;
    *) echo 'Unknown native runtime filesystem'; return 2 ;;
esac
unset PYTHONPATH VIRTUAL_ENV VIRTUAL_ENV_PROMPT
export PYTHONNOUSERSITE=1 CW_FORCE_CONDA_ACTIVATE=1
_native_runtime=$(/usr/bin/python3 -m src.utils.runtime_cache "$_persistent_runtime" "$_runtime_native/CreditPFN/runtimes/$CREDITPFN_RUNTIME_VERSION") || return
export PATH="$_native_runtime/bin:$PATH"
hash -r
# Site wrappers support NVIDIA passthrough through their extra options.
export APPTAINER_NV=1 SINGULARITY_NV=1
export CREDITPFN_RUNTIME_ROOT="$_persistent_runtime"
if [[ "${CREDITPFN_RUNTIME_ACCEPTANCE:-0}" != 1 ]]; then
    python -m src.utils.runtime_validation require "$CREDITPFN_RUNTIME_VERSION" >/dev/null || return
fi
python -c 'import os,torch,numpy,tabpfn; assert os.getenv("APPTAINER_CONTAINER") or os.getenv("SINGULARITY_CONTAINER"); print("Active sealed runtime:",os.environ["CREDITPFN_RUNTIME_VERSION"],"torch",torch.__version__)' || return
unset _runtime_project _runtime_native _persistent_runtime _native_runtime
