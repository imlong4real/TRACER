"""Behavioural lock for the Phase-1a SEED source.

``prune_scope="cell"`` seeds Phase 1a on the WHOLE CELL; ``"nuclear"``
seeds on nuclear tx only (legacy). The config-wiring assertions live in
``tests/test_prune_scope.py``; what is locked here is that the seed
source actually changes the partition — a nucleus whose genes are
mutually anti-correlated must not be able to veto a cell whose cytoplasm
is coherent.

The kernel keeps ``nuclear_seed_only`` / ``nuclear_only_admit`` as
FUNCTION arguments (``prune_transcripts_nuclear_seed``); only the
user-facing config collapses them into ``prune_scope``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _incoherent_nucleus_case():
    """One cell whose NUCLEUS carries mutually anti-correlated genes while
    its CYTOPLASM carries a clean coherent program.

    This is the failure the whole-cell seed exists to fix: the nuclear seed
    collapses (greedy prune drops all but one anti-correlated gene), so the
    cell's real program — visible only in cytoplasm — is discarded. The
    legacy whole-cell FALLBACK cannot save it, because the fallback is gated
    on a COUNT test (``n_unique_nuc < min_nuclear_genes``) and this nucleus
    has exactly ``min_nuclear_genes`` genes.
    """
    import itertools
    import pandas as pd

    good, bad = ["A", "B", "C"], ["X", "Y", "Z"]
    panel = pd.DataFrame([
        {"gene_i": g1, "gene_j": g2,
         "PMI": 0.9 if (g1 in good and g2 in good) else -0.5}
        for g1, g2 in itertools.combinations(good + bad, 2)
    ])

    recs, tid = [], 0
    for g in bad:                      # nuclear, mutually anti-correlated
        for _ in range(2):
            recs.append(dict(transcript_id=tid, cell_id="c1", feature_name=g,
                             x=float(tid % 5), y=float(tid % 3), z=0.0,
                             overlaps_nucleus=True))
            tid += 1
    for g in good:                     # cytoplasmic, coherent
        for _ in range(8):
            recs.append(dict(transcript_id=tid, cell_id="c1", feature_name=g,
                             x=float(tid % 5), y=float(tid % 3), z=0.0,
                             overlaps_nucleus=False))
            tid += 1
    return pd.DataFrame(recs), panel


def _prune(df, panel, nuclear_seed_only):
    from tracer.pruning import prune_transcripts_nuclear_seed
    out, _ = prune_transcripts_nuclear_seed(
        df.copy(), panel, cell_id_col="cell_id", gene_col="feature_name",
        nuclear_col="overlaps_nucleus", threshold=0.2, unassigned_id="-1",
        metric_col="PMI", nan_fill=0.0, min_nuclear_genes=3,
        nuclear_only_admit=False,
        nuclear_seed_only=nuclear_seed_only, fallback_whole_cell_admit=True,
        tx_weighted=True, n_jobs=1,
    )
    lab = out["tracer_id"].astype(str)
    kept = out[lab != "-1"]
    return set(kept["feature_name"]), len(kept)


def test_nuclear_seed_only_true_loses_coherent_cytoplasm():
    """Legacy behaviour: the collapsed nuclear seed discards the real program."""
    df, panel = _incoherent_nucleus_case()
    genes, n_kept = _prune(df, panel, nuclear_seed_only=True)
    assert not {"A", "B", "C"} & genes, (
        f"nuclear seed unexpectedly retained cytoplasmic program: {sorted(genes)}"
    )
    assert n_kept < len(df) // 2, (
        f"expected most tx discarded under the nuclear seed; kept {n_kept}/{len(df)}"
    )


def test_whole_cell_seed_rescues_coherent_cytoplasm():
    """nuclear_seed_only=False seeds on the whole cell, so the coherent
    cytoplasmic program survives instead of being vetoed by the nucleus."""
    df, panel = _incoherent_nucleus_case()
    genes, n_kept = _prune(df, panel, nuclear_seed_only=False)
    assert {"A", "B", "C"} <= genes, (
        f"whole-cell seed failed to retain the coherent program; got {sorted(genes)}"
    )
    assert n_kept > len(df) * 0.7, (
        f"expected most tx retained under the whole-cell seed; kept {n_kept}/{len(df)}"
    )
