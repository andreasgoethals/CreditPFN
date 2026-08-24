"""Stage every base checkpoint the config's ladder needs, from a login node.

    python -m src.utils.stage_checkpoints              # what is present, what is missing
    python -m src.utils.stage_checkpoints --download   # fetch the missing ones

RUN THIS ON A LOGIN NODE. Compute nodes have no outbound internet, and every loader passes
``allow_auto_download=False`` so a missing checkpoint fails loudly at trial start rather than
silently downloading mid-job on a GPU we are paying for.

Replaces the hand-edited snippet that used to live in ``docs/VSC.md``: the ladder is read from
the config, so adding a base to ``tunable.classifier_base_paths`` is all it takes for this to
know about it — one place to change instead of two.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

#: Where each base checkpoint comes from. Keyed by the FILENAME our configs use, because that is
#: what `_infer_version` parses and what the loaders look for on disk — the upstream filename is
#: an implementation detail of the repo we pull from.
#:
#: TabICLv2 ships both heads in one repo; TabPFN ships one repo per generation. v2 is the Nature
#: model (7.2 M parameters — an order of magnitude smaller than v2.6/v3, which is expected, not
#: a sign of a wrong file).
SOURCES: dict[str, tuple[str, str]] = {
    # our filename                                    (hf repo,                upstream file)
    "tabpfn-v2-classifier-v2_default.ckpt":     ("Prior-Labs/TabPFN-v2", "tabpfn-v2-classifier.ckpt"),
    "tabpfn-v2-regressor-v2_default.ckpt":      ("Prior-Labs/TabPFN-v2", "tabpfn-v2-regressor.ckpt"),
    "tabicl-classifier-v2-20260212.ckpt":       ("jingang/TabICL",       "tabicl-classifier-v2-20260212.ckpt"),
    "tabicl-regressor-v2-20260212.ckpt":        ("jingang/TabICL",       "tabicl-regressor-v2-20260212.ckpt"),
}

#: Checkpoints that have no public download and must be copied in by hand. Listed so the report
#: says "copy this" rather than "not found", which is a different action.
MANUAL = {
    "tabpfn-v2.6-classifier-v2.6_default.ckpt",
    "tabpfn-v2.6-regressor-v2.6_default.ckpt",
    "tabpfn-v3-classifier-v3_default.ckpt",
    "tabpfn-v3-regressor-v3_default.ckpt",
}


def wanted_checkpoints(config: str | None = None) -> list[str]:
    """Filenames the config's base ladder refers to, both tracks."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/train.yaml")
    if config and config != "config/train.yaml":
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config))
    out: list[str] = []
    for key in ("classifier_base_paths", "regressor_base_paths"):
        for p in (OmegaConf.select(cfg, f"tunable.{key}") or []):
            out.append(Path(str(p)).name)
    return sorted(dict.fromkeys(out))


def checkpoints_root() -> Path:
    from src.utils.paths import checkpoints_dir, resolve_staging_path

    return Path(resolve_staging_path(str(checkpoints_dir())))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="experiment config whose ladder to stage")
    ap.add_argument("--download", action="store_true", help="fetch the missing ones")
    args = ap.parse_args(argv)

    dest = checkpoints_root()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints dir: {dest}")

    present, missing, manual = [], [], []
    for name in wanted_checkpoints(args.config):
        if (dest / name).is_file():
            present.append(name)
        elif name in SOURCES:
            missing.append(name)
        else:
            manual.append(name)

    for n in present:
        print(f"  ok       {n}  ({(dest / n).stat().st_size / 1e6:.0f} MB)")
    for n in missing:
        print(f"  MISSING  {n}  <- {SOURCES[n][0]}")
    for n in manual:
        note = "copy by hand (no public download)" if n in MANUAL else "unknown source"
        print(f"  MISSING  {n}  <- {note}")

    if missing and args.download:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("\nhuggingface_hub is not installed in this environment.", file=sys.stderr)
            return 1
        print()
        for name in missing:
            repo, upstream = SOURCES[name]
            print(f"  downloading {name} from {repo} ...", flush=True)
            local = hf_hub_download(repo, upstream)
            shutil.copy2(local, dest / name)
            print(f"    staged -> {dest / name}")
    elif missing:
        print("\nre-run with --download to fetch the ones marked MISSING above.")

    if manual:
        print("\nThese have no public download. Copy them into the directory above, e.g.")
        print("  scp checkpoints/<file> vsc38338@login.hpc.kuleuven.be:"
              f"{dest}/")
    return 1 if (missing or manual) else 0


if __name__ == "__main__":
    raise SystemExit(main())
