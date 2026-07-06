"""Version-tolerant imports for TabPFN internals that moved between releases.

WHY THIS EXISTS (run post-mortem, 2026-07-04)
---------------------------------------------
The Jul-3 full run lost all 32 LGD training trials in 2–5 seconds each to::

    No module named 'tabpfn.architectures.base'

Root cause: our regressor code path imported
``tabpfn.architectures.base.bar_distribution`` — the module path of the 2.x
PyPI line — but the VSC env runs tabpfn **8.0.8** (installed from git main),
where the module moved to ``tabpfn.architectures.shared.bar_distribution``
(verified against the refreshed source dump, ``repositories/TabPFN .txt``:
``FILE: src/tabpfn/architectures/shared/bar_distribution.py``). Classifiers
never import the bar distribution, which is why every PD trial loaded fine on
the exact same env while every LGD trial died before the debug banner.

This module is the single place that knows the candidate paths. It also
registers ``sys.modules`` aliases so that *pickled* objects inside older
checkpoints whose class path references a legacy module location still
unpickle under the new layout (and vice versa).

Usage::

    from src.train.tabpfn_compat import import_bar_distribution
    FullSupportBarDistribution = import_bar_distribution().FullSupportBarDistribution
"""

from __future__ import annotations

import importlib
import logging
import sys

LOGGER = logging.getLogger(__name__)

# Candidate module paths, newest layout first. 8.x = architectures.shared;
# 2.x PyPI = architectures.base (with model.bar_distribution as a re-export).
_BAR_DIST_CANDIDATES = (
    "tabpfn.architectures.shared.bar_distribution",   # tabpfn >= 8.x (git main)
    "tabpfn.architectures.base.bar_distribution",     # tabpfn 2.x PyPI line
    "tabpfn.model.bar_distribution",                  # tabpfn 2.x legacy alias
)

_cached_module = None


def import_bar_distribution():
    """Return the bar-distribution module under whichever path this tabpfn has.

    Tries the known locations newest-first, caches the winner, and registers
    ALL candidate paths as ``sys.modules`` aliases of the found module so that
    unpickling a checkpoint written under a different tabpfn version resolves
    its class references too. Raises a single informative ImportError listing
    every attempted path if none import.
    """
    global _cached_module
    if _cached_module is not None:
        return _cached_module

    errors: list[str] = []
    for path in _BAR_DIST_CANDIDATES:
        try:
            mod = importlib.import_module(path)
        except Exception as exc:                       # noqa: BLE001
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if not hasattr(mod, "FullSupportBarDistribution"):
            errors.append(f"{path}: imported but has no FullSupportBarDistribution")
            continue
        _cached_module = mod
        # Alias the other locations so legacy pickles resolve.
        for alias in _BAR_DIST_CANDIDATES:
            if alias not in sys.modules:
                sys.modules[alias] = mod
        if path != _BAR_DIST_CANDIDATES[0]:
            LOGGER.info("tabpfn_compat: bar_distribution found at legacy path %s", path)
        return mod

    raise ImportError(
        "Could not locate TabPFN's bar_distribution module under any known "
        "path — the installed tabpfn's layout is newer than this code knows. "
        "Attempted:\n  " + "\n  ".join(errors)
    )


def smoke_test(track: str) -> None:
    """Fail-fast preflight for a SLURM prolog: verify every import the given
    track's training will need, in seconds, BEFORE any GPU time is spent.

    Exits with a clear one-line error on failure (caller: ``python -c
    "from src.train.tabpfn_compat import smoke_test; smoke_test('lgd')"``).
    """
    import tabpfn                                                   # noqa: F401
    from tabpfn import TabPFNClassifier, TabPFNRegressor            # noqa: F401
    from tabpfn.model_loading import load_model_criterion_config    # noqa: F401
    # Ensemble preprocessor moved too: 8.x = preprocessing.ensemble (a
    # package); 2.x = flat tabpfn.preprocessing module. Accept either.
    try:
        from tabpfn.preprocessing.ensemble import TabPFNEnsemblePreprocessor  # noqa: F401
    except ModuleNotFoundError:
        from tabpfn.preprocessing import EnsembleConfig              # noqa: F401
    if track == "lgd":
        mod = import_bar_distribution()
        assert hasattr(mod, "FullSupportBarDistribution")
    print(f"tabpfn_compat smoke_test OK (track={track}, "
          f"tabpfn={getattr(tabpfn, '__version__', '?')})")
