"""Privacy guard — no proprietary dataset identifier may appear in a TRACKED file.

The proprietary stems are read from the GITIGNORED ``src/data/_private_names.py``, so this
test never contains them itself. It SKIPS when that mapping is absent (e.g. a public CI
checkout); run it LOCALLY — where the mapping and data live — before every push.

Covers the leak this project actually had: private slugs in docs, configs, source,
notebook JSON, and result dumps. Figure *pixels* need the separate PDF-text check noted in
the remediation plan; this test guards the source tree that lands on GitHub.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _proprietary_stems() -> list[str] | None:
    try:
        from src.data._private_names import PROPRIETARY
    except Exception:
        return None
    # longest first so 'base_modelisation' is reported before its substring 'base_model'
    return sorted(PROPRIETARY, key=len, reverse=True)


def test_accessor_anonymises_proprietary_and_degrades():
    """The tracked accessor maps a private slug to Prop* (mapping present) and never crashes."""
    from src.data.dataset_names import display_name, is_proprietary

    stems = _proprietary_stems()
    if stems is None:
        pytest.skip("private mapping absent — accessor degrades to raw slug by design")
    # a proprietary slug renders as an anonymised name, not its stem
    for stem in stems:
        shown = display_name(f"0001.{stem}")
        assert stem not in shown.lower(), f"{stem!r} leaked through display_name -> {shown!r}"
        assert is_proprietary(stem)
    # a public slug is unchanged in spirit (never flagged proprietary)
    assert not is_proprietary("0001.gmsc")


def test_no_proprietary_slug_in_tracked_files():
    """`git ls-files` (== what is on GitHub) must contain no proprietary slug/stem."""
    stems = _proprietary_stems()
    if stems is None:
        pytest.skip("private mapping absent (public checkout) — run locally before pushing")
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
    ).stdout.splitlines()
    # Slug boundary that treats '_' and digits as part of the token (word-boundary \b would
    # miss `corr_heloc` and `..._0009_bank_status_auc`, both of which reached a public repo).
    pats = [(s, re.compile(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])")) for s in stems]
    offenders: dict[str, list[str]] = {}
    for rel in tracked:
        if not rel or rel.startswith("tfm-library/"):
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        hit = sorted({s for s, pat in pats if pat.search(text)})
        if hit:
            offenders[rel] = hit
    assert not offenders, (
        f"proprietary dataset slugs found in {len(offenders)} tracked file(s) — scrub them "
        f"(and rewrite history, since the repo was public):\n"
        + "\n".join(f"  {f}: {h}" for f, h in sorted(offenders.items()))
    )
