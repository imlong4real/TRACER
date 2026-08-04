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


# ---- P3: coherence-triggered sibling promotion (option A) -------------------

def _main_fails_sibling_passes_aux():
    """Main C1 (genes 0,1,2) has C<0 (conflict-heavy) -> fails floor.
    Sibling C1-1 (genes 3,4,5) is all-purity -> passes. Same parent cell_id."""
    W = np.zeros((6, 6), dtype=np.float32)
    for (i, j), v in {(0, 1): -0.5, (0, 2): -0.5, (1, 2): 0.3,   # main: 2 conflict, 1 purity
                      (3, 4): 0.5, (3, 5): 0.5, (4, 5): 0.5}.items():  # sibling: all purity
        W[i, j] = W[j, i] = v
    aux = {"gene_to_idx": {f"g{k}": k for k in range(6)}, "W": W}
    df = pd.DataFrame({
        "tracer_id":   ["C1", "C1", "C1", "C1-1", "C1-1", "C1-1"],
        "cell_id":     ["C1", "C1", "C1", "C1",   "C1",   "C1"],
        "feature_name": ["g0", "g1", "g2", "g3",  "g4",   "g5"],
        "_etype":      ["cell", "cell", "cell", "partial", "partial", "partial"],
    })
    return df, aux


def test_midqc_promotes_coherent_sibling_when_main_fails():
    df, aux = _main_fails_sibling_passes_aux()
    out, stats = _qc_demote_low_coherence(
        df, entity_col="tracer_id", aux=aux, min_C=0.05, min_n_genes=2,
        threshold=0.2, metric="pmi", unassigned_id="-1",
    )
    # sibling C1-1 promoted to the main "C1" label with _etype cell
    promoted = out[out["feature_name"].isin(["g3", "g4", "g5"])]
    assert (promoted["tracer_id"] == "C1").all()
    assert (promoted["_etype"] == "cell").all()
    # the old failing main's tx are released to unassigned
    old_main = out[out["feature_name"].isin(["g0", "g1", "g2"])]
    assert (old_main["tracer_id"] == "-1").all()
    # no orphan "C1-1" label remains
    assert (out["tracer_id"] != "C1-1").all()


def test_midqc_promotion_matches_siblings_by_label_not_drifted_cellid():
    # Regression: a failing main's tx cell_id has DRIFTED (reassignment) to a
    # different nucleus, and an entity from THAT nucleus must NOT be promoted
    # into the main's label. Siblings are defined by tracer_id label structure
    # ({C} main / {C}-{k} partial), not the cell_id column.
    W = np.zeros((6, 6), dtype=np.float32)
    for (i, j), v in {(0, 1): -0.5, (0, 2): -0.5, (1, 2): 0.3,     # main "A-1": conflict-heavy -> fails
                      (3, 4): 0.5, (3, 5): 0.5, (4, 5): 0.5}.items():  # "B-1-1": all purity, passes
        W[i, j] = W[j, i] = v
    aux = {"gene_to_idx": {f"g{k}": k for k in range(6)}, "W": W}
    df = pd.DataFrame({
        # main A-1's tx have DRIFTED cell_id "B" (not its own "A")
        "tracer_id":    ["A-1", "A-1", "A-1", "B-1-1", "B-1-1", "B-1-1"],
        "cell_id":      ["B",   "B",   "B",   "B",     "B",     "B"],
        "feature_name": ["g0",  "g1",  "g2",  "g3",    "g4",    "g5"],
        "_etype":       ["cell","cell","cell","partial","partial","partial"],
    })
    out, stats = _qc_demote_low_coherence(
        df, entity_col="tracer_id", aux=aux, min_C=0.05, min_n_genes=2,
        threshold=0.2, metric="pmi", unassigned_id="-1",
    )
    # B-1-1 is NOT a label-sibling of A-1 (prefix "A" vs "B"), so it must NOT be
    # promoted into "A-1". A-1 fails -> released; B-1-1 untouched.
    assert (out[out.feature_name.isin(["g0","g1","g2"])]["tracer_id"] == "-1").all()
    assert (out[out.feature_name.isin(["g3","g4","g5"])]["tracer_id"] == "B-1-1").all()
    assert stats.get("entities_promoted", 0) == 0


def test_midqc_no_sibling_still_drops_main():
    # main fails, no sibling -> released, nothing promoted (existing behavior)
    df, aux = _main_fails_sibling_passes_aux()
    df = df[df["tracer_id"] == "C1"].copy()   # drop the sibling
    out, stats = _qc_demote_low_coherence(
        df, entity_col="tracer_id", aux=aux, min_C=0.05, min_n_genes=2,
        threshold=0.2, metric="pmi", unassigned_id="-1",
    )
    assert (out["tracer_id"] == "-1").all()
