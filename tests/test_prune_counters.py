"""Diagnostic counters for the Phase-1 prune kernel's early-exit paths.

``continue`` statements in ``_prune_cells_nuclear_seed_impl`` drop a cell
before Phase 1b/1c ever run, and nothing counted them — so "how often does
this fire?" could not be answered without an A/B. Counting them is what
showed ``seed_coherence_floor`` fired 0-1 times per ROI (it was then
removed) and that ``fallback`` was being double-counted in cell scope.

Counters are module-level in the kernel (same pattern as
``set_admit_independent``) and read via ``get_prune_counters()``.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tracer import _cy_prune  # noqa: E402
from tracer.pruning import prune_transcripts_nuclear_seed  # noqa: E402

COUNTER_KEYS = {"cells", "seed_empty", "thin_nucleus", "fallback"}


def _case():
    """One cell with an incoherent nucleus + coherent cytoplasm.

    Under a NUCLEAR seed the primary seed is the anti-correlated nuclear
    set, which must trip ``seed_coherence_floor``; under a WHOLE-CELL seed
    it must not.
    """
    good, bad = ["A", "B", "C"], ["X", "Y", "Z"]
    panel = pd.DataFrame([
        {"gene_i": a, "gene_j": b,
         "PMI": 0.9 if (a in good and b in good) else -0.5}
        for a, b in itertools.combinations(good + bad, 2)
    ])
    recs, tid = [], 0
    for g in bad:
        for _ in range(2):
            recs.append(dict(transcript_id=tid, cell_id="c1", feature_name=g,
                             x=float(tid % 5), y=float(tid % 3), z=0.0,
                             overlaps_nucleus=True)); tid += 1
    for g in good:
        for _ in range(8):
            recs.append(dict(transcript_id=tid, cell_id="c1", feature_name=g,
                             x=float(tid % 5), y=float(tid % 3), z=0.0,
                             overlaps_nucleus=False)); tid += 1
    return pd.DataFrame(recs), panel


def _run(nuclear_seed_only):
    df, panel = _case()
    _cy_prune.reset_prune_counters()
    prune_transcripts_nuclear_seed(
        df, panel, cell_id_col="cell_id", gene_col="feature_name",
        nuclear_col="overlaps_nucleus", threshold=0.2, unassigned_id="-1",
        metric_col="PMI", nan_fill=0.0, min_nuclear_genes=3,
        nuclear_only_admit=False,
        nuclear_seed_only=nuclear_seed_only, fallback_whole_cell_admit=True,
        tx_weighted=True, n_jobs=1,
    )
    return _cy_prune.get_prune_counters()


def test_counter_api_exists():
    assert hasattr(_cy_prune, "get_prune_counters")
    assert hasattr(_cy_prune, "reset_prune_counters")
    _cy_prune.reset_prune_counters()
    c = _cy_prune.get_prune_counters()
    assert COUNTER_KEYS <= set(c), f"missing counters: {COUNTER_KEYS - set(c)}"
    assert all(c[k] == 0 for k in COUNTER_KEYS), f"reset did not zero: {c}"


def test_thin_nucleus_is_scope_invariant_but_fallback_is_not():
    """``thin_nucleus`` is a property of the DATA; ``fallback`` is a path.

    Under prune_scope="cell" the whole-cell seed is the PRIMARY path, so
    nothing falls back even though the thin-nucleus cells still exist.
    Conflating the two made a cell-scope run look like it performed 173
    fallbacks when it performed none.
    """
    c_cell = _run(nuclear_seed_only=False)
    c_nuc = _run(nuclear_seed_only=True)
    assert c_cell["thin_nucleus"] == c_nuc["thin_nucleus"], (
        "thin_nucleus must not depend on the seed scope; "
        f"cell={c_cell}, nuclear={c_nuc}")
    assert c_cell["fallback"] == 0, (
        f"no fallback is possible under a whole-cell seed; got {c_cell}")
