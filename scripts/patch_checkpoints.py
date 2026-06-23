"""Patch trained checkpoints that are missing ``architecture_name`` and/or
``inference_config`` keys.

Root cause: ``save_finetuned`` (src/train/model.py) previously saved only
``{state_dict, config}`` (and optional ``provenance``), omitting the two keys
that ``load_model`` requires to instantiate the correct architecture class.
Without them, ``load_model`` falls back to V2 architecture inference and
produces "Missing key(s) in state_dict" when loading V3 or V2.6 weights.

This script patches every affected checkpoint in-place by:

  1. Loading the trained checkpoint.
  2. Reading ``provenance["hyperparameters"]["base_checkpoint"]`` to locate
     the base checkpoint from which the trained model was initialised.
  3. Loading the base checkpoint and copying its ``architecture_name`` and
     ``inference_config`` into the trained checkpoint's dict.
  4. Resaving the patched dict to the same path (atomic via a temp file).

Run on the supercomputer after ``git pull`` picks up the model.py fix:

    cd ${VSC_DATA}/CreditPFN
    python scripts/patch_checkpoints.py

Or with explicit paths:

    python scripts/patch_checkpoints.py \\
        --output-root /path/to/output \\
        --base-ckpt-dir /path/to/project/checkpoints
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch


def _find_trained_checkpoints(output_root: Path) -> list[Path]:
    trained_dir = output_root / "checkpoints" / "trained"
    if not trained_dir.exists():
        return []
    return sorted(trained_dir.rglob("*.ckpt"))


def _load_safe(path: Path) -> dict:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"torch.load failed for {path}: {exc}") from exc


def _resolve_base_ckpt(ckpt: dict, base_ckpt_dir: Path) -> Path | None:
    """Find the base checkpoint path from provenance or by filename inference."""
    prov = ckpt.get("provenance") or {}
    hp = prov.get("hyperparameters") or {}
    base_str = hp.get("base_checkpoint", "")
    if base_str:
        p = Path(base_str)
        if p.exists():
            return p
        # Try relative to base_ckpt_dir (provenance stores the full path from
        # the supercomputer, which differs locally).
        local = base_ckpt_dir / p.name
        if local.exists():
            return local
    return None


def patch_checkpoint(path: Path, base_ckpt_dir: Path, *, dry_run: bool = False) -> str:
    """Patch one checkpoint.  Returns a status string."""
    ckpt = _load_safe(path)

    missing = [
        k for k in ("architecture_name", "inference_config")
        if k not in ckpt
    ]
    if not missing:
        return "ok (already has all keys)"

    base_path = _resolve_base_ckpt(ckpt, base_ckpt_dir)
    if base_path is None:
        return f"SKIP — could not locate base checkpoint (provenance={ckpt.get('provenance', {}).get('hyperparameters', {}).get('base_checkpoint', '<missing>')})"

    base_ckpt = _load_safe(base_path)

    patched = False
    if "architecture_name" in missing:
        if "architecture_name" not in base_ckpt:
            return f"SKIP — base checkpoint {base_path.name} also lacks architecture_name"
        ckpt["architecture_name"] = base_ckpt["architecture_name"]
        patched = True

    if "inference_config" in missing:
        if "inference_config" not in base_ckpt:
            return f"SKIP — base checkpoint {base_path.name} also lacks inference_config"
        ckpt["inference_config"] = base_ckpt["inference_config"]
        patched = True

    if not patched:
        return "SKIP — nothing to patch"

    if dry_run:
        arch = ckpt.get("architecture_name", "?")
        return f"DRY-RUN — would add {missing} (arch={arch!r}) from {base_path.name}"

    # Atomic save: write to a sibling temp file, then replace.
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(fd)
        torch.save(ckpt, tmp)
        shutil.move(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    arch = ckpt.get("architecture_name", "?")
    return f"PATCHED — added {missing} (arch={arch!r}) from {base_path.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=os.environ.get("CREDITPFN_OUTPUT_ROOT", ""),
        help="Path to the CreditPFN output root (default: $CREDITPFN_OUTPUT_ROOT)",
    )
    parser.add_argument(
        "--base-ckpt-dir",
        default="",
        help=(
            "Directory that contains base .ckpt files "
            "(default: <project_root>/checkpoints). "
            "Falls back to the path stored in provenance."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be patched without writing anything.",
    )
    args = parser.parse_args()

    if not args.output_root:
        sys.exit(
            "ERROR: --output-root not set and $CREDITPFN_OUTPUT_ROOT is empty.\n"
            "Run: python scripts/patch_checkpoints.py --output-root /path/to/output"
        )

    output_root = Path(args.output_root)
    # Default base-ckpt-dir: <project root>/checkpoints
    # (the script lives in scripts/, so parent.parent = project root)
    default_base = Path(__file__).parent.parent / "checkpoints"
    base_ckpt_dir = Path(args.base_ckpt_dir) if args.base_ckpt_dir else default_base

    checkpoints = _find_trained_checkpoints(output_root)
    if not checkpoints:
        print(f"No trained checkpoints found under {output_root}/checkpoints/trained/")
        return

    print(f"Found {len(checkpoints)} trained checkpoint(s) under {output_root}")
    print(f"Base checkpoint dir: {base_ckpt_dir}")
    if args.dry_run:
        print("DRY-RUN mode — no files will be modified.\n")

    n_patched = n_ok = n_skipped = 0
    for path in checkpoints:
        status = patch_checkpoint(path, base_ckpt_dir, dry_run=args.dry_run)
        rel = path.relative_to(output_root)
        print(f"  {rel}: {status}")
        if status.startswith("PATCHED"):
            n_patched += 1
        elif status.startswith("ok"):
            n_ok += 1
        else:
            n_skipped += 1

    print(f"\nSummary: {n_patched} patched, {n_ok} already OK, {n_skipped} skipped.")
    if n_patched > 0 and not args.dry_run:
        print("Re-run the eval pipeline to score the patched checkpoints.")


if __name__ == "__main__":
    main()
