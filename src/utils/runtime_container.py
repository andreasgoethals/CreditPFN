"""Record and recreate the existing VSC environment in an immutable site container.

The original environment is read only. Unpacked installation/build trees belong
on native scratch; only the sealed image, wrappers and manifest reach PROJECT.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit


MINIFORGE_VERSION = "25.3.1-0"
MINIFORGE_SHA256 = "376b160ed8130820db0ab0f3826ac1fc85923647f75c1b8231166e3d559ab768"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed():
    return {normalized(d.metadata["Name"]): d.version for d in metadata.distributions()
            if d.metadata.get("Name") and normalized(d.metadata["Name"]) != "creditpfn"}


def source_packages():
    """Retain immutable Git provenance and installed source bytes, not just versions."""
    result = {}
    for distribution in metadata.distributions():
        raw = distribution.read_text("direct_url.json")
        if not raw:
            continue
        direct = json.loads(raw)
        vcs = direct.get("vcs_info", {})
        if vcs.get("vcs") != "git":
            continue  # Conda build-machine file URLs are covered by the explicit lock.
        url = urlsplit(direct["url"])
        commit = vcs.get("commit_id", "")
        if (url.scheme != "https" or url.username or url.password or url.query
                or url.fragment or not re.fullmatch(r"[a-f0-9]{40}", commit)):
            raise RuntimeError("Cannot safely reproduce non-public or unpinned Git package")
        name = normalized(distribution.metadata["Name"])
        files = {str(p): sha256(distribution.locate_file(p)) for p in distribution.files or []
                 if str(p).endswith(".py") and ".dist-info/" not in str(p)}
        if not files:
            raise RuntimeError(f"Cannot verify installed source files for {name}")
        result[name] = dict(url=direct["url"], commit=commit, files=files)
    return result


def record(destination, conda):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    command = [str(conda), "list", "--prefix", sys.prefix]
    explicit = subprocess.check_output(command + ["--explicit", "--md5"], text=True)
    packages = json.loads(subprocess.check_output(command + ["--json"], text=True))
    versions = installed()
    pip_packages = {normalized(p["name"]) for p in packages if p.get("channel") == "pypi"}
    # Conda metadata may still describe a distribution subsequently replaced by pip.
    for p in packages:
        name = normalized(p["name"])
        if name in versions and versions[name] != p["version"]:
            pip_packages.add(name)
    pip_packages.discard("creditpfn")
    sources = source_packages()
    pip_packages.update(sources)
    requirement = [f"{name} @ git+{sources[name]['url']}@{sources[name]['commit']}"
                   if name in sources else f"{name}==={versions[name]}"
                   for name in sorted(pip_packages) if name in versions]
    (destination / "conda-explicit.txt").write_text(explicit, encoding="utf-8")
    (destination / "pip-lock.txt").write_text("\n".join(requirement) + "\n", encoding="utf-8")
    evidence = dict(python=sys.version, packages=versions, conda=packages,
                    source_prefix=sys.prefix, platform=sys.platform, sources=sources)
    (destination / "environment.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"Recorded {len(packages)} Conda entries and {len(requirement)} pip pins", flush=True)


def patch_template(source):
    replacements = {
        "mamba $_EC create --name $CW_ENV_NAME $_FF": 'mamba $_EC create --prefix "$env_root" $_FF',
        "conda $_EC create --name $CW_ENV_NAME $_FF": '"$CW_INSTALLATION_PATH/miniforge/bin/conda" $_EC create --prefix "$env_root" $_FF',
        "conda activate $CW_ENV_NAME": 'conda activate "$env_root"',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("Site wrapper changed: review its Conda prefix handling")
        source = source.replace(old, new)
    lines = source.splitlines()
    lookup = [i for i, line in enumerate(lines) if "CW_CONDA_VERSION=" in line and "curl" in line]
    download = [i for i, line in enumerate(lines) if "curl" in line and "--output Miniforge_inst.sh" in line]
    if len(lookup) != 1 or len(download) != 1:
        raise ValueError("Site wrapper changed: review Miniforge download")
    lines[lookup[0]] = f"CW_CONDA_VERSION={MINIFORGE_VERSION}"
    lines[download[0]] = (
        '[[ "$CW_CONDA_ARCH" == Linux-x86_64 ]] || exit 2\n'
        'if [[ -n "${CREDITPFN_MINIFORGE_INSTALLER:-}" ]]; then\n'
        'cp -- "$CREDITPFN_MINIFORGE_INSTALLER" Miniforge_inst.sh\nelse\n'
        'curl --fail --location --show-error --retry 2 '
        '"https://github.com/conda-forge/miniforge/releases/download/$CW_CONDA_VERSION/Miniforge3-$CW_CONDA_VERSION-$CW_CONDA_ARCH.sh" '
        '--output Miniforge_inst.sh\nfi\n'
        f"printf '%s  Miniforge_inst.sh\\n' {MINIFORGE_SHA256} | sha256sum --check -")
    for i, line in enumerate(lines):
        if "follow_log" in line and "_inst_miniforge.log" in line:
            lines[i] += '\nwait "$inst_pid"\n[[ -x "$CW_INSTALLATION_PATH/miniforge/bin/conda" ]]'
    return "\n".join(lines) + "\n"


def compare(specification):
    expected = json.loads(Path(specification).read_text(encoding="utf-8"))
    actual = installed()
    differences = {k: (v, actual.get(k)) for k, v in expected["packages"].items() if actual.get(k) != v}
    if sys.version.split()[0] != expected["python"].split()[0] or differences:
        raise RuntimeError(f"Recreated runtime differs from recorded environment: {differences}")
    actual_sources = source_packages()
    if actual_sources != expected.get("sources", {}):
        raise RuntimeError("Recreated Git package commit or installed source bytes differ")
    if not os.environ.get("APPTAINER_CONTAINER") and not os.environ.get("SINGULARITY_CONTAINER"):
        raise RuntimeError("Runtime validation escaped the container")
    print(f"PASS: Python and all {len(expected['packages'])} recorded distributions match inside container", flush=True)


def seal(directory, specification):
    root = Path(directory).resolve()
    images = sorted([*root.rglob("*.sif"), *root.rglob("*.sqfs")])
    if not images or not (root / "bin/python").is_file():
        raise RuntimeError("Incomplete container build")
    if list(root.rglob("site-packages")):
        raise RuntimeError("Unpacked dependencies must not remain on PROJECT")
    files = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
             if p.is_file() and not p.is_symlink() and p != root/"runtime.json"}
    links = {str(p.relative_to(root)): os.readlink(p) for p in sorted(root.rglob("*")) if p.is_symlink()}
    receipt = dict(schema=1, files=files, environment_sha256=sha256(specification),
                   images=[str(p.relative_to(root)) for p in images], symlinks=links, validated=False)
    (root / "runtime.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(dict(images=receipt["images"], files=len(files), directory=str(root))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "compare", "seal"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--conda", type=Path)
    parser.add_argument("--specification", type=Path)
    args = parser.parse_args()
    if args.action == "record":
        record(args.path, args.conda)
    elif args.action == "compare":
        compare(args.path)
    else:
        seal(args.path, args.specification)


if __name__ == "__main__":
    main()
