"""fallback_whole_cell_admit: a nucleus with <min_nuclear_genes nuclear genes
takes the whole-cell fallback seed. Under nuclear_only_admit, its cytoplasmic tx
are skipped in Phase 1b, so if the (thin, non-fitting) nuclear genes don't match
the whole-cell seed the cell DIES. With fallback_whole_cell_admit=True the
fallback cell admits its whole-cell tx, forming a main cell on the dominant local
program. (seed_coherence_floor was removed 2026-08-23; Mid-QC gates coherence.)"""
import numpy as np
from tracer import _cy_prune

# genes: 0=n0, 1=n1 (nuclear), 2,3,4 = c0,c1,c2 (coherent cytoplasmic program)
def _W():
    W = np.full((5, 5), np.nan, dtype=np.float32)
    for a, b in [(2, 3), (2, 4), (3, 4)]:      # coherent cyto trio
        W[a, b] = W[b, a] = 0.5
    for a in (0, 1):                            # nuclear genes anti-fit everything
        for b in (1, 2, 3, 4):
            if a != b:
                W[a, b] = W[b, a] = -0.3
    return W

def _run(fallback_admit):
    cell = [np.array([0, 1, 2, 3, 4], dtype=np.int32)]      # 1 cell, 5 tx
    gene = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    nuc  = np.array([1, 1, 0, 0, 0], dtype=np.uint8)         # n0,n1 nuclear; c* cyto
    return np.asarray(_cy_prune.prune_cells_nuclear_seed(
        cell, gene, nuc, _W(),
        0.2, 3, 0,                    # threshold, min_nuclear_genes, skip_1c
        1, 1, 1,                      # nuclear_only_admit, tx_weighted, veto=mean
        0.0, 0.2, 25.0, 0.0, -0.2,
        int(fallback_admit),
    ))

def test_fallback_off_cell_dies():
    # only 2 nuclear genes (<3) -> fallback seed = coherent {c0,c1,c2}; nuclear
    # tx don't fit; cyto skipped -> nothing admitted -> all unassigned.
    codes = _run(fallback_admit=False)
    assert list(codes) == [2, 2, 2, 2, 2]

def test_fallback_on_forms_cell_from_cytoplasm():
    codes = _run(fallback_admit=True)
    # cytoplasmic trio admitted to MAIN (code 0); non-fitting nuclear tx unassigned
    assert list(codes[2:]) == [0, 0, 0]     # c0,c1,c2 -> main
    assert list(codes[:2]) == [2, 2]        # n0,n1 -> unassigned
