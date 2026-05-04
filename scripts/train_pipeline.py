"""End-to-end orchestrator for continued pretraining.

Mirrors ``scripts/data_pipeline.py``. The actual training math lives in
:mod:`src.train.loop`; this script's job is to:

  1. Decide which (base, lr, multi-chunk-policy) tuples to train. By
     default it iterates over the cartesian product of every list
     under ``cfg.tunable``. With ``--single`` it uses only the FIRST
     value of each list (one run, one checkpoint).
  2. Call :func:`src.train.loop.train_one_config` for each tuple.
  3. Save each finished checkpoint to
     ``cfg.checkpoint.trained_dir/<descriptive_name>.ckpt`` (the
     descriptive name is built by
     :func:`src.train.loop.descriptive_name`).
  4. Persist a ``logs/runs/<run_name>_<track>.csv`` row per trained
     config: HP-tuple, test metric, checkpoint path, walltime. This
     CSV is the manifest the future ``src/eval/`` reads to compare
     TabPFN variants against XGBoost / CatBoost / TabICL — same
     train/test split, same chunks, same metric.
  5. Append one summary line to the run log (whether ``--single`` or
     a full grid).

CLI usage::

    # default = cartesian product over `cfg.tunable.*`
    python scripts/train_pipeline.py

    # only one run (the first value of every tunable list)
    python scripts/train_pipeline.py --single

    # restrict to one track at a time
    python scripts/train_pipeline.py track=lgd

    # any Hydra-style override is honoured
    python scripts/train_pipeline.py train.epochs=10
    python scripts/train_pipeline.py tunable.learning_rates=[1e-5]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import sys as _sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Allow `python scripts/train_pipeline.py` (vs `-m scripts.train_pipeline`).
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from src.utils.run_log import resolve_run_log  # noqa: E402

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CSV row
# --------------------------------------------------------------------------- #


@dataclass
class RunRow:
    """One row of the per-track training manifest."""
    track: str
    base_checkpoint: str
    learning_rate: float
    multi_chunk_policy: str
    seed: int
    test_metric_name: str
    test_metric_raw: float | None
    n_train_datasets: int
    n_test_datasets: int
    n_train_chunks: int
    n_test_chunks: int
    final_ckpt_path: str | None
    elapsed_sec: float
    status: str                       # "OK" | "FAIL"
    error: str | None


# --------------------------------------------------------------------------- #
# Cfg loading + Hydra-style overrides
# --------------------------------------------------------------------------- #


def _load_cfg(overrides: list[str] | None = None):
    """Load ``config/training.yaml`` and apply ``key=value`` overrides."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("config/training.yaml")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def _resolve_grid(cfg, *, single: bool) -> list[tuple[str, float, str]]:
    """Materialise the (base, lr, policy) tuples to train.

    With ``single=True``: take the first value of every tunable list.
    Otherwise: full cartesian product.
    """
    track = str(cfg.track)
    bases = (
        list(cfg.tunable.classifier_base_paths) if track == "pd"
        else list(cfg.tunable.regressor_base_paths)
    )
    lrs = [float(x) for x in cfg.tunable.learning_rates]
    policies = list(cfg.tunable.multi_chunk_policies)

    if single:
        return [(str(bases[0]), float(lrs[0]), str(policies[0]))]
    return [
        (str(b), float(lr), str(p))
        for b, lr, p in itertools.product(bases, lrs, policies)
    ]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def run(
    single: bool = False,
    overrides: list[str] | None = None,
    log_path: Path | str | None = None,
    cfg=None,
) -> int:
    """Train one config (``--single``) or every (base × lr × policy) tuple.

    Returns
    -------
    ``0`` on full success, ``1`` if any trial raised.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    if cfg is None:
        cfg = _load_cfg(overrides)
    log, _ = resolve_run_log(log_path)

    track = str(cfg.track)
    if track not in ("pd", "lgd"):
        raise ValueError(f"track must be 'pd' or 'lgd'; got {track!r}")
    grid = _resolve_grid(cfg, single=single)
    LOGGER.info(
        "Training plan: %d run(s) on track=%s (%s)",
        len(grid), track, "single" if single else "cartesian grid",
    )

    # CSV manifest path — the future eval module reads this.
    csv_path = Path("logs/runs") / f"{cfg.run_name}_{track}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy import; avoids loading torch when --help'ing.
    from src.train.loop import train_one_config

    rows: list[RunRow] = []
    failures = 0
    t_outer = time.monotonic()

    for trial_idx, (base, lr, policy) in enumerate(grid, start=1):
        LOGGER.info(
            "\n=== Trial %d/%d  base=%s  lr=%g  policy=%s ===",
            trial_idx, len(grid), Path(base).name, lr, policy,
        )
        t_trial = time.monotonic()
        try:
            result = train_one_config(
                cfg, track=track,
                base_checkpoint=base,
                learning_rate=lr,
                multi_chunk_policy=policy,
            )
            rows.append(RunRow(
                track=track, base_checkpoint=base, learning_rate=lr,
                multi_chunk_policy=policy, seed=int(cfg.seed),
                test_metric_name=result.test_metric_name,
                test_metric_raw=result.test_metric_raw,
                n_train_datasets=result.n_train_datasets,
                n_test_datasets=result.n_test_datasets,
                n_train_chunks=result.n_train_chunks,
                n_test_chunks=result.n_test_chunks,
                final_ckpt_path=str(result.final_ckpt_path),
                elapsed_sec=result.elapsed_sec,
                status="OK", error=None,
            ))
        except Exception as exc:                           # noqa: BLE001
            failures += 1
            LOGGER.error("Trial %d failed: %s", trial_idx, exc, exc_info=True)
            rows.append(RunRow(
                track=track, base_checkpoint=base, learning_rate=lr,
                multi_chunk_policy=policy, seed=int(cfg.seed),
                test_metric_name="",
                test_metric_raw=None,
                n_train_datasets=0, n_test_datasets=0,
                n_train_chunks=0, n_test_chunks=0,
                final_ckpt_path=None,
                elapsed_sec=time.monotonic() - t_trial,
                status="FAIL", error=f"{type(exc).__name__}: {exc}",
            ))

        _write_csv(rows, csv_path)        # persist after every trial

    elapsed = time.monotonic() - t_outer
    summary = (
        f"train_pipeline: status={'OK' if failures == 0 else f'FAIL[{failures}/{len(grid)}]'}  "
        f"track={track}  mode={'single' if single else 'grid'}  "
        f"trials={len(grid)}  csv={csv_path}  elapsed={elapsed:.1f}s"
    )
    log.write(summary)
    print(summary)

    return 0 if failures == 0 else 1


# --------------------------------------------------------------------------- #
# CSV writer
# --------------------------------------------------------------------------- #


def _write_csv(rows: list[RunRow], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Continued pretraining for TabPFN-2.6 on the credit corpus.",
    )
    p.add_argument(
        "--single", action="store_true",
        help="Train only ONE config (the first value of every list under "
             "cfg.tunable). Default: cartesian product of all tunable lists.",
    )
    p.add_argument(
        "--log-path", default=None,
        help="Append the run summary to this log file instead of creating "
             "a fresh logs/<timestamp>.log file.",
    )
    args, unknown = p.parse_known_args(argv)
    overrides = [a for a in unknown if "=" in a and not a.startswith("-")]
    leftover = [a for a in unknown if a not in overrides]
    if leftover:
        p.error(f"unrecognised arguments: {leftover}")
    return args, overrides


if __name__ == "__main__":
    args, overrides = _parse_args()
    raise SystemExit(run(
        single=args.single,
        overrides=overrides,
        log_path=args.log_path,
    ))
