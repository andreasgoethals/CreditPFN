"""Privacy guard — no proprietary dataset identifier may appear in a TRACKED file.

The proprietary stems are read from the GITIGNORED ``src/data/_private_names.py``, so this
test never contains them itself. It SKIPS when that mapping is absent (e.g. a public CI
checkout); run it LOCALLY — where the mapping and data live — before every push.

Covers private slugs in docs, configs, source, readable notebook content, and result dumps.
Encoded image bytes are excluded; rendered figures and exported PDF text require a separate
inspection. This test guards the source tree that lands on GitHub.
"""
from __future__ import annotations

import base64
import binascii
import json
import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_BINARY_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"}


def _notebook_text_for_scan(text: str) -> str:
    """Scan readable content; encoded pixels can match short names by chance."""
    notebook = json.loads(text)
    for cell in notebook.get("cells", []):
        bundles = [out.get("data", {}) for out in cell.get("outputs", [])]
        bundles.extend(cell.get("attachments", {}).values())
        for bundle in bundles:
            for mime in _BINARY_MIMES.intersection(bundle):
                payload = bundle[mime]
                if isinstance(payload, list):
                    payload = "".join(payload)
                try:
                    base64.b64decode("".join(payload.split()), validate=True)
                except (AttributeError, ValueError, binascii.Error):
                    continue  # Unexpected text still needs the privacy scan.
                bundle[mime] = "[encoded binary content; inspect the rendered figure separately]"
    return json.dumps(notebook, ensure_ascii=False)


def test_notebook_privacy_scan_preserves_text_and_svg_but_omits_encoded_bytes():
    encoded = base64.b64encode(b"arbitrary pixels").decode("ascii")
    notebook = {"cells": [{
        "source": ["private_table_source"],
        "metadata": {"description": "private_table_metadata"},
        "outputs": [{"data": {
            "image/png": encoded,
            "image/svg+xml": "<svg>private_table_svg</svg>",
            "text/plain": "private_table_output",
        }}],
        "attachments": {"figure": {"image/png": encoded, "text/plain": "private_table_attachment"}},
    }]}
    scanned = _notebook_text_for_scan(json.dumps(notebook))
    assert encoded not in scanned
    for location in ("source", "metadata", "svg", "output", "attachment"):
        assert "private_table_" + location in scanned


def test_notebook_privacy_scan_keeps_unexpected_text_in_a_binary_field():
    notebook = {"cells": [{"outputs": [{"data": {"image/png": "private_table"}}]}]}
    assert "private_table" in _notebook_text_for_scan(json.dumps(notebook))


def _proprietary_stems() -> list[str] | None:
    try:
        from src.data._private_names import PROPRIETARY
    except Exception:
        return None
    # Longest first so a longer identifier is reported before its prefix.
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
    # Underscores are separators; word-boundary \b would miss names inside filenames.
    pats = [(s, re.compile(r"(?<![A-Za-z0-9])" + re.escape(s) + r"(?![A-Za-z0-9])")) for s in stems]
    offenders: dict[str, list[str]] = {}
    for rel in tracked:
        # Temporary, explicit project exception: preprocessing remains tracked
        # with routing slugs until the code is frozen (user handover 22-09-2026).
        if not rel or rel.startswith("tfm-library/") or rel == "src/data/preprocessing.py":
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if rel.endswith(".ipynb"):
            text = _notebook_text_for_scan(text)
        hit = sorted({s for s, pat in pats if pat.search(text)})
        if hit:
            offenders[rel] = hit
    assert not offenders, (
        f"proprietary dataset slugs found in {len(offenders)} tracked file(s) — scrub them "
        f"(and rewrite history, since the repo was public):\n"
        + "\n".join(f"  {f}: {h}" for f, h in sorted(offenders.items()))
    )
