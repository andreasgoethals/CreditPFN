"""One-off migration: retag L2-SP checkpoints saved before the 22-09-2026 fix.

Before the fix, ``train_one_config`` dropped the ``_l2sp<λ>`` tag from the checkpoint name (see
``src/train/loop.py``), so every trial's checkpoint was written WITHOUT it — and because the swept
λ also never reached training (``scripts/train_pipeline.py``), each of those checkpoints was in fact
trained at the config default λ=0.003. Both "arms" of a cell therefore wrote the same untagged file
and one overwrote the other.

This inserts the correct ``_l2sp<λ>`` tag — read from each checkpoint's own provenance sidecar,
which recorded the *effective* λ it trained at — into the filename, so the fixed pipeline's
resume-skip check finds these survivors and re-runs only the missing λ=0 arm instead of the whole
grid. It renames the ``.ckpt`` and its ``.provenance.json`` / ``.epoch_eval.ckpt`` sidecars together.

Idempotent: a name that already carries ``_l2sp`` is left alone. Dry-run by default.

    python -m src.utils.migrate_l2sp_checkpoints --track both            # preview
    python -m src.utils.migrate_l2sp_checkpoints --track both --apply    # rename
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

LOGGER = logging.getLogger("migrate_l2sp")

#: The l2sp tag sits BEFORE the optional adapter tag (``_lora`` / ``_iclhead``) and the extension,
#: matching ``descriptive_name()``'s order ``...{pass}{rows}{l2sp}{lora}.ckpt``.
_TAIL = re.compile(r"(?P<adapter>_lora|_iclhead)?\.ckpt$")

#: What every pre-fix checkpoint actually trained at (the config default), used only when a
#: checkpoint's provenance is missing or unreadable.
_DEFAULT_LAMBDA = 0.003


def retag(name: str, l2sp_lambda: float) -> str:
    """Insert ``_l2sp<λ>`` into a tagless checkpoint filename; idempotent."""
    if "_l2sp" in name:
        return name
    tag = f"_l2sp{float(l2sp_lambda):g}"
    return _TAIL.sub(lambda m: f"{tag}{m.group('adapter') or ''}.ckpt", name)


def _lambda_from_provenance(prov_path: Path, default: float = _DEFAULT_LAMBDA) -> float:
    """The effective λ this checkpoint trained at, from its provenance sidecar."""
    try:
        blob = json.loads(prov_path.read_text(encoding="utf-8"))
        hp = (blob.get("provenance") or blob).get("hyperparameters", {})
        return float(hp.get("l2sp_lambda", default))
    except Exception as exc:                                           # noqa: BLE001
        LOGGER.warning("provenance unreadable (%s); assuming λ=%g for %s",
                       exc, default, prov_path.name)
        return default


def plan_renames(root: Path) -> list[tuple[Path, Path, float]]:
    """(old, new, λ) for every tagless final checkpoint under ``root``."""
    out: list[tuple[Path, Path, float]] = []
    for ckpt in sorted(root.rglob("*.ckpt")):
        if ckpt.name.endswith(".epoch_eval.ckpt"):
            continue                                    # transient eval snapshot, not a final ckpt
        if "_l2sp" in ckpt.name:
            continue                                    # already tagged (fixed-code run, or re-run)
        lam = _lambda_from_provenance(ckpt.with_suffix(ckpt.suffix + ".provenance.json"))
        new = ckpt.with_name(retag(ckpt.name, lam))
        if new != ckpt:
            out.append((ckpt, new, lam))
    return out


def apply_renames(renames: list[tuple[Path, Path, float]], *, apply: bool) -> None:
    for ckpt, new, lam in renames:
        LOGGER.info("%s  %s -> %s  (λ=%g)",
                    "RENAME" if apply else "DRY-RUN", ckpt.name, new.name, lam)
        if not apply:
            continue
        for suffix in ("", ".provenance.json", ".epoch_eval.ckpt"):
            src = Path(str(ckpt) + suffix)
            if src.exists():
                dst = Path(str(new) + suffix)
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Retag pre-22-09-2026 L2-SP checkpoints.")
    ap.add_argument("--track", choices=("pd", "lgd", "both"), default="both")
    ap.add_argument("--root", type=Path, default=None,
                    help="checkpoints/trained dir (default: resolved staging checkpoints/trained)")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: dry-run)")
    args = ap.parse_args(argv)

    from src.utils.paths import checkpoints_dir
    base = args.root or (checkpoints_dir() / "trained")
    tracks = ("pd", "lgd") if args.track == "both" else (args.track,)

    total: list[tuple[Path, Path, float]] = []
    for tr in tracks:
        root = base / tr
        if not root.is_dir():
            LOGGER.info("skip %s: %s does not exist", tr, root)
            continue
        renames = plan_renames(root)
        LOGGER.info("%s: %d checkpoint(s) to retag under %s", tr, len(renames), root)
        apply_renames(renames, apply=args.apply)
        total.extend(renames)

    LOGGER.info("%s %d checkpoint(s).", "Renamed" if args.apply else "Would rename", len(total))
    if total and not args.apply:
        LOGGER.info("Re-run with --apply to perform the renames.")
    return 0


if __name__ == "__main__":                                            # pragma: no cover
    sys.exit(main())
