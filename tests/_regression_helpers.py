"""Snapshot regression helpers for pipeline output fingerprints.

The maintainer-in-the-loop pattern: pipeline outputs are fingerprinted
into a JSON ``tests/references/<variant>.json`` file. On each test run
the current output is compared against the reference; if any metric
diverges beyond its tolerance the test fails with a structured diff
plus an explicit instruction for regenerating the reference if the
change is intentional.

Public surface:
  - :func:`assert_matches_reference`: the assertion helper.
  - :data:`REFERENCE_DIR`: path to ``tests/references/``.

Environment variable:
  ``TRACER_UPDATE_REFERENCES=1`` causes the helper to (re)write the
  reference file rather than compare. Use this locally after confirming
  a pipeline change is an intentional improvement.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest


REFERENCE_DIR = Path(__file__).parent / "references"


def _format_diff(name: str, ref: dict, cur: dict, tolerances: dict) -> str:
    """Build a human-readable diff message for a failed reference match."""
    lines = [f"Pipeline output diverged from reference {REFERENCE_DIR / (name + '.json')}:", ""]
    lines.append(f"  {'metric':<28} | {'reference':>12} | {'current':>12} | {'delta':>10} | tolerance")
    lines.append(f"  {'-' * 28}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}-+----------")

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    diverged = []
    for key in sorted(set(ref.keys()) | set(cur.keys())):
        rv = ref.get(key)
        cv = cur.get(key)
        tol = tolerances.get(key, 0)
        if isinstance(rv, list) or isinstance(cv, list):
            # Skip nested structures here; reported separately below
            continue
        if isinstance(rv, dict) or isinstance(cv, dict):
            continue
        if rv is None or cv is None:
            lines.append(f"  {key:<28} | {fmt(rv):>12} | {fmt(cv):>12} | {'(missing)':>10} | n/a")
            diverged.append(key)
            continue
        if isinstance(rv, (int, float)) and isinstance(cv, (int, float)):
            # NaN-on-both-sides counts as match (a metric that's
            # legitimately undefined under the test conditions, e.g.
            # ARI vs ground truth when cell_id is all "-1").
            if (isinstance(rv, float) and math.isnan(rv)
                    and isinstance(cv, float) and math.isnan(cv)):
                continue
            delta = cv - rv
            if math.isfinite(rv) and math.isfinite(cv) and abs(delta) <= tol:
                continue
            try:
                delta_s = f"{delta:>+10.4f}"
            except (ValueError, TypeError):
                delta_s = f"{'(undef)':>10}"
            lines.append(
                f"  {key:<28} | {fmt(rv):>12} | {fmt(cv):>12} | {delta_s} | ±{tol}"
            )
            diverged.append(key)
        else:
            if rv != cv:
                lines.append(f"  {key:<28} | {fmt(rv):>12} | {fmt(cv):>12} | (changed)  | exact")
                diverged.append(key)

    # Stage progression diff (first divergence only)
    if "stage_progression" in ref and "stage_progression" in cur:
        ref_p = ref["stage_progression"]
        cur_p = cur["stage_progression"]
        for i, (rs, cs) in enumerate(zip(ref_p, cur_p)):
            if rs != cs:
                lines.append("")
                lines.append(f"Stage progression first divergence at stage \"{rs.get('stage', '?')}\":")
                lines.append(f"  reference: {rs}")
                lines.append(f"  current:   {cs}")
                diverged.append("stage_progression")
                break
        if len(ref_p) != len(cur_p):
            diverged.append("stage_progression_length")
            lines.append("")
            lines.append(f"Stage count changed: ref={len(ref_p)} cur={len(cur_p)}")

    if not diverged:
        return ""

    lines.append("")
    lines.append("If this change is intentional (algorithm improvement / new behavior),")
    lines.append("regenerate the reference:")
    lines.append("")
    lines.append(f"  TRACER_UPDATE_REFERENCES=1 pytest tests/test_pipeline_regression.py")
    lines.append("")
    lines.append(f"then commit tests/references/{name}.json.")
    return "\n".join(lines)


def assert_matches_reference(name: str, current: dict[str, Any],
                             tolerances: dict[str, float] | None = None) -> None:
    """Compare ``current`` against ``tests/references/<name>.json``.

    On first run (no reference file), or when ``TRACER_UPDATE_REFERENCES``
    is set in the environment, write the reference and skip the test.

    Otherwise compare each numeric metric within its tolerance; emit a
    structured diff and ``pytest.fail`` if anything diverges.

    Parameters
    ----------
    name : str
        Stem of the reference file, e.g. ``"segmented"``.
    current : dict
        The fingerprint dict to compare. Should contain only JSON-
        serialisable values.
    tolerances : dict[str, float] or None
        Per-key absolute tolerance for numeric metrics. Keys not listed
        require exact equality.
    """
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = REFERENCE_DIR / f"{name}.json"
    tolerances = tolerances or {}

    if os.environ.get("TRACER_UPDATE_REFERENCES") or not ref_path.exists():
        ref_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        if os.environ.get("TRACER_UPDATE_REFERENCES"):
            pytest.skip(f"Updated reference: {ref_path}")
        else:
            pytest.skip(f"Wrote initial reference: {ref_path}")

    reference = json.loads(ref_path.read_text())
    diff = _format_diff(name, reference, current, tolerances)
    if diff:
        pytest.fail(diff, pytrace=False)
