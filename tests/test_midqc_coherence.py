"""fix/midqc-coherence: coherence (QC + reporting) uses the informative-edges
denominator (|w|>tau), so purity + conflict == 1. Merge-decision ΔC (Stitch /
Maha-remerge via stitching.coherence) is intentionally NOT touched here.
See project_coherence_metric_midqc memory."""
import numpy as np
import pandas as pd
import pytest
from tracer.metrics import compute_cell_coherence
from tracer.pipeline import _qc_demote_low_coherence


def _W(n, pairs):
    W = np.zeros((n, n), dtype=np.float32)
    for (i, j), v in pairs.items():
        W[i, j] = W[j, i] = v
    return W


def test_reporting_coherence_purity_plus_conflict_is_one():
    # one cell, 4 genes present; 6 pairs: 3 purity, 1 conflict, 2 dead.
    W = _W(4, {
        (0, 1): 0.5, (1, 2): 0.3, (2, 3): 0.25,   # purity (>0.2)
        (0, 2): -0.5,                              # conflict (<-0.2)
        (0, 3): 0.1, (1, 3): -0.05,                # dead (|.|<=0.2)
    })
    M = np.array([[1, 1, 1, 1]], dtype=np.int32)
    col_idx = np.array([0, 1, 2, 3], dtype=np.int32)
    # default real_signal_threshold must give informative-edges denominator (=tau)
    coh, purity, conflict, _ = compute_cell_coherence(
        M=M, col_idx=col_idx, npmi_mat=W, threshold=0.2,
    )
    assert purity[0] == pytest.approx(0.75)     # 3 / 4 informative
    assert conflict[0] == pytest.approx(0.25)   # 1 / 4 informative
    assert purity[0] + conflict[0] == pytest.approx(1.0)
    assert coh[0] == pytest.approx(0.5)


def _limbo_diluted_aux(n=10):
    """10 genes: 3 purity (0.5) + 1 conflict (-0.3) + 41 limbo (0.1) pairs.
    C(all-pairs) = 2/45 = 0.044 (<= floor 0.05) but C(informative) = 2/4 = 0.5."""
    W = np.full((n, n), 0.1, dtype=np.float32)
    np.fill_diagonal(W, 0.0)
    for i, j in [(0, 1), (0, 2), (0, 3)]:
        W[i, j] = W[j, i] = 0.5
    W[0, 4] = W[4, 0] = -0.3
    aux = {"gene_to_idx": {f"g{k}": k for k in range(n)}, "W": W}
    df = pd.DataFrame({
        "tracer_id": ["E1"] * n,
        "feature_name": [f"g{k}" for k in range(n)],
        "_etype": ["cell"] * n,
    })
    return df, aux


def test_midqc_demote_uses_informative_denominator_by_default():
    # default rst -> informative (tau=0.2): C=0.5 > floor -> entity KEPT
    df, aux = _limbo_diluted_aux()
    out, stats = _qc_demote_low_coherence(
        df, entity_col="tracer_id", aux=aux, min_C=0.05, min_n_genes=2,
        threshold=0.2, metric="pmi", unassigned_id="-1",
    )
    assert (out["tracer_id"] == "E1").all()
    assert stats["entities_demoted_low_C"] == 0


def test_midqc_demote_legacy_all_pairs_still_demotes():
    # explicit legacy rst=0.05 -> C=0.044 <= floor -> entity DEMOTED (discriminates)
    df, aux = _limbo_diluted_aux()
    out, stats = _qc_demote_low_coherence(
        df, entity_col="tracer_id", aux=aux, min_C=0.05, min_n_genes=2,
        threshold=0.2, metric="pmi", unassigned_id="-1",
        real_signal_threshold=0.05,
    )
    assert (out["tracer_id"] == "-1").all()
    assert stats["entities_demoted_low_C"] == 1
