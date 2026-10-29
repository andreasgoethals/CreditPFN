#!/bin/bash
set -euo pipefail
identifier="${1:?}"; original_python="${2:?}"; conda_executable="${3:?}"
[[ "$identifier" =~ ^[a-zA-Z0-9]+$ && -n "${SLURM_JOB_ID:-}" ]] || exit 2
repo="$PWD"
project="${CREDITPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-/lustre1/project/stg_00211}}"
[[ "$(basename "$project")" == CreditPFN ]] || project="$project/CreditPFN"
destination="$project/runtimes/$identifier"
spec="$repo/output CreditPFN/experiment0/runtime/$identifier"
[[ ! -e "$destination" && ! -e "$spec" ]] || { echo 'Version already exists; inspect before retrying'; exit 2; }
module load hpc-container-wrapper/0.4.2
scratch=$(realpath -e "${VSC_SCRATCH_LUSTRE1:?}")
work=$(mktemp -d "$scratch/creditpfn-runtime-$identifier-XXXXXX")
echo "Owned build workspace: $work"
export APPTAINER_TMPDIR="$work/apptainer-tmp" APPTAINER_CACHEDIR="$work/apptainer-cache"
export PIP_CACHE_DIR="$work/pip-cache" CONDA_PKGS_DIRS="$work/conda-packages"
if [[ -n "${CREDITPFN_BUILD_CACHE_FROM:-}" ]]; then
    _cache_from=$(realpath -e "$CREDITPFN_BUILD_CACHE_FROM")
    [[ "$_cache_from" == "$scratch"/creditpfn-runtime-* && -d "$_cache_from/pip-cache" && -d "$_cache_from/conda-packages" ]] || {
        echo 'Build-cache reuse escaped a verified owned runtime workspace'; exit 2;
    }
    export PIP_CACHE_DIR="$_cache_from/pip-cache" CONDA_PKGS_DIRS="$_cache_from/conda-packages"
    unset _cache_from
fi
export PYTHONNOUSERSITE=1
unset PYTHONPATH VIRTUAL_ENV
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$(dirname "$destination")"
"$original_python" -m src.utils.runtime_container record "$spec" --conda "$conda_executable"
printf '%s\n' "$work" > "$spec/build-workspace.txt"
if ! command -v mksquashfs >/dev/null; then
    libexec=$(apptainer buildcfg | sed -n 's/^LIBEXECDIR=//p')
    [[ -x "$libexec/apptainer/bin/mksquashfs" ]] || exit 2
    export PATH="$libexec/apptainer/bin:$PATH"
fi
wrapper=$(dirname "$(dirname "$(readlink -f "$(command -v conda-containerize)")")")
"$original_python" - "$wrapper" "$work" "$spec" "$repo" <<'PY'
import os,sys,shutil,yaml,shlex
from pathlib import Path
from src.utils.runtime_container import patch_template
wrapper,work,spec,repo=map(Path,sys.argv[1:])
private=work/'wrapper'
shutil.copytree(wrapper,private,symlinks=True)
template=private/'templates/conda.sh'
template.write_text(patch_template(template.read_text()))
alias=private/'frontends/conda-containerize'
alias.unlink(missing_ok=True);alias.symlink_to('containerize')
cfg=yaml.safe_load((wrapper/'default_config/config.yaml').read_text())
cfg.setdefault('force',{}).update(build_tmpdir_base=str(work),workdir=str(work),share_container=False,
 max_num_cpus=8,num_cpus=8,sqfs_options='-noappend -mem 4G -processors 8',singularity_executable_path=shutil.which('apptainer'))
(work/'site.yaml').write_text(yaml.safe_dump(cfg))
shutil.copyfile(spec/'conda-explicit.txt',work/'conda-explicit.txt')
q=shlex.quote
post=['set -euo pipefail', 'export PYTHONNOUSERSITE=1',
 'export PATH="$env_root/bin:$PATH"', 'hash -r',
 '"$env_root/bin/python" -I -c '+q('import sys,os; from pathlib import Path; actual=Path(sys.prefix).resolve(); expected=Path(os.environ["env_root"]).resolve(); print("Build interpreter:", actual, "expected:", expected, flush=True); assert actual==expected'),
 '"$env_root/bin/python" -m pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu130 -r '+q(str(spec/'pip-lock.txt')),
 'cd '+q(str(repo)), '"$env_root/bin/python" -m src.utils.runtime_container compare '+q(str(spec/'environment.json'))]
(work/'post_install.sh').write_text('\n'.join(post)+'\n')
PY
export CW_GLOBAL_YAML="$work/site.yaml" CW_DEBUG_KEEP_FILES=1
cd "$work"
bash "$work/wrapper/frontends/conda-containerize" new --prefix "$destination" --post-install "$work/post_install.sh" "$work/conda-explicit.txt"
cd "$repo"
"$destination/bin/python" -m src.utils.runtime_container compare "$spec/environment.json"
"$original_python" -m src.utils.runtime_container seal "$destination" --specification "$spec/environment.json"
echo 'BUILD COMPLETE: original environment retained; CPU/GPU acceptance still required.'
