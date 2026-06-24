"""Phase-1 Reassign-1c gene-fit: dense vs sparse parity.

Reassign-1c (`_reassign_nuclear_post_1c_etype`) moves a nuclear transcript
from a cell's main entity to a sibling 1c partial when the partial's mean
PMI gene-fit beats the main's by a margin. The mean must SKIP unobserved
(structurally-absent) gene pairs, not count them as 0 — otherwise a
gene-sparse partial is unfairly penalized.

The synthetic fixture is the PDAC "jikammne-1" EMT-doublet analog: an
epithelial main entity carrying a nuclear CD68 read that belongs to the
co-segmented macrophage partial. CD68's PMI gene-fit is strong against the
macrophage seed and weak/absent against the epithelial seed, so it should
repatriate to the partial.

Two backends are exercised:
  * dense  (G,G) float32 with NaN for unobserved pairs (legacy path),
  * sparse symmetric CSR built from the bootstrap upper-triangle (scales to
    whole-transcriptome panels).

Tests assert (a) the two agree bit-for-bit on a fully-dense panel and
(b) on a panel with genuinely-absent pairs the sparse path matches the
dense NaN-skip path and DIFFERS from the buggy absent-as-0 fill.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline_runner import _reassign_nuclear_post_1c_etype  # noqa: E402

pytest.importorskip("tracer._cy_reassign")

PARENT = "jikammne-1"          # dash-containing cell_id (Xenium FFPE style)
PARTIAL = "jikammne-1-1"       # the co-segmented macrophage partial
MAIN_GENES = ["EPCAM", "KRT8", "VIM", "CD68", "STAT1"]
PARTIAL_GENES = ["CD163", "LYZ"]
GENES = MAIN_GENES + PARTIAL_GENES
GENE_TO_IDX = {g: i for i, g in enumerate(GENES)}
G = len(GENES)


def _gi(name: str) -> int:
    return GENE_TO_IDX[name]


# Symmetric PMI panel. Keys are unordered gene-name pairs. CD68 fits the
# macrophage seed and not the epithelial one; STAT1 is gene-sparse against
# the main seed (only one stored partner) and is the discriminator for the
# absent-skip-vs-fill behavior. Note the explicit 0.0 (EPCAM,CD68): an
# observed PMI of exactly zero must stay a stored, counted entry.
_PMI = {
    ("EPCAM", "KRT8"): 0.9,
    ("EPCAM", "VIM"): 0.85,
    ("KRT8", "VIM"): 0.88,
    ("EPCAM", "CD68"): 0.0,     # observed zero — must be preserved
    ("KRT8", "CD68"): -0.1,
    ("VIM", "CD68"): 0.05,
    ("CD68", "CD163"): 0.9,
    ("CD68", "LYZ"): 0.8,
    ("CD163", "LYZ"): 0.9,
    ("EPCAM", "CD163"): -0.2,
    ("EPCAM", "LYZ"): -0.2,
    ("KRT8", "CD163"): -0.2,
    ("KRT8", "LYZ"): -0.2,
    ("VIM", "CD163"): 0.1,
    ("VIM", "LYZ"): 0.1,
    ("STAT1", "EPCAM"): 0.8,
    ("STAT1", "CD163"): 0.5,
    ("STAT1", "LYZ"): 0.5,
    # Deliberately ABSENT in the sparse panel:
    #   STAT1-KRT8, STAT1-VIM, STAT1-CD68
}
# Pairs we leave unobserved (indeterminate) on the sparse panel.
_ABSENT = {("STAT1", "KRT8"), ("STAT1", "VIM"), ("STAT1", "CD68")}


def _norm(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


# Normalize pair keys so lookups are order-insensitive.
_PMI = {_norm(*k): v for k, v in _PMI.items()}
_ABSENT = {_norm(*k) for k in _ABSENT}


def _all_pairs() -> list[tuple[str, str]]:
    return [
        (GENES[i], GENES[j]) for i in range(G) for j in range(i + 1, G)
    ]


def _build_dense(*, fill_absent_with) -> np.ndarray:
    """Dense (G,G) float32. Stored pairs carry their PMI; absent pairs get
    ``fill_absent_with`` (np.nan for the skip path, 0.0 for the buggy
    fill path). Diagonal is NaN (self-pairs are never gene-fit)."""
    W = np.full((G, G), np.nan, dtype=np.float32)
    for a, b in _all_pairs():
        key = _norm(a, b)
        if key in _PMI:
            v = _PMI[key]
        elif key in _ABSENT:
            v = fill_absent_with
        else:  # pragma: no cover - every pair is classified above
            raise AssertionError(f"unclassified pair {key}")
        if v is None:
            continue
        i, j = _gi(a), _gi(b)
        W[i, j] = v
        W[j, i] = v
    return W


def _build_sparse(*, include_absent: bool) -> sp.csr_matrix:
    """Upper-triangle CSR like the bootstrap's ``W_sparse``. Absent pairs
    are structurally omitted (the whole point); when ``include_absent`` we
    additionally store them (used to make a *fully-dense* sparse panel for
    the bit-for-bit parity test)."""
    rows, cols, data = [], [], []
    for a, b in _all_pairs():
        key = _norm(a, b)
        if key in _PMI:
            v = _PMI[key]
        elif key in _ABSENT:
            if not include_absent:
                continue
            v = 0.0
        else:  # pragma: no cover
            raise AssertionError(f"unclassified pair {key}")
        i, j = _gi(a), _gi(b)
        lo, hi = (i, j) if i < j else (j, i)
        rows.append(lo)
        cols.append(hi)
        data.append(v)
    return sp.coo_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(G, G),
        dtype=np.float32,
    ).tocsr()


def _make_df() -> pd.DataFrame:
    """One transcript per gene. Main-entity genes are nuclear `cell` tx
    labeled with the cell_id; partial genes are nuclear `partial` tx."""
    rows = []
    for g in MAIN_GENES:
        rows.append((PARENT, g, True, PARENT, "cell"))
    for g in PARTIAL_GENES:
        rows.append((PARENT, g, True, PARTIAL, "partial"))
    return pd.DataFrame(
        rows,
        columns=["cell_id", "feature_name", "overlaps_nucleus",
                 "tracer_id", "_etype"],
    )


def _run(W) -> dict[str, str]:
    """Run reassign-1c with the given W backend; return gene -> final
    tracer_id label."""
    df = _make_df()
    aux = {"gene_to_idx": GENE_TO_IDX, "W": W}
    out, _stats = _reassign_nuclear_post_1c_etype(
        df, entity_col="tracer_id", aux=aux, margin=0.05,
    )
    return dict(zip(out["feature_name"], out["tracer_id"]))


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_dense_repatriates_cd68_to_macrophage_partial():
    """Baseline: under the correct NaN-skip convention CD68 moves to the
    macrophage partial; epithelial genes and the gene-sparse STAT1 stay."""
    labels = _run(_build_dense(fill_absent_with=np.nan))
    assert labels["CD68"] == PARTIAL
    for g in ["EPCAM", "KRT8", "VIM", "STAT1"]:
        assert labels[g] == PARENT, g
    for g in PARTIAL_GENES:
        assert labels[g] == PARTIAL, g


def test_dense_sparse_parity_fully_dense_panel():
    """Bit-for-bit parity: when every off-diagonal pair is stored, the
    dense and sparse kernels must produce identical moves."""
    dense_labels = _run(_build_dense(fill_absent_with=0.0))
    sparse_labels = _run(_build_sparse(include_absent=True))
    assert dense_labels == sparse_labels


def test_sparse_skips_absent_matches_nan_not_zero_fill():
    """On a panel with genuinely-absent pairs:
      * the sparse path must equal the dense NaN-skip path, and
      * it must DIFFER from the buggy absent-as-0 fill path.

    The discriminator is STAT1: with absent pairs skipped its single
    stored main partner (PMI 0.8) keeps it on the main entity; filling the
    three absent main pairs with 0 drags its main mean down to 0.2, below
    the macrophage mean (0.5), so the buggy path wrongly repatriates it."""
    sparse_labels = _run(_build_sparse(include_absent=False))
    nan_skip_labels = _run(_build_dense(fill_absent_with=np.nan))
    zero_fill_labels = _run(_build_dense(fill_absent_with=0.0))

    # Sparse skip == dense NaN-skip (the correct convention).
    assert sparse_labels == nan_skip_labels

    # CD68 still repatriates under the correct convention.
    assert sparse_labels["CD68"] == PARTIAL
    # STAT1 stays on main under skip, but the zero-fill bug moves it.
    assert sparse_labels["STAT1"] == PARENT
    assert zero_fill_labels["STAT1"] == PARTIAL
    assert sparse_labels != zero_fill_labels


def test_coo_stack_symmetrization_preserves_stored_zero():
    """Guard the wiring's choice of COO-stack over ``W + W.T``: scipy's
    sparse add eliminates explicit zeros, which would silently drop an
    observed PMI of exactly 0.0 (here EPCAM-CD68). Verify the stored zero
    survives symmetrization in both off-diagonal positions, while the
    naive ``W + W.T`` drops it (documents WHY we don't use it)."""
    W = _build_sparse(include_absent=False)
    ei, ci = _gi("EPCAM"), _gi("CD68")

    # COO-stack symmetrization (mirrors _reassign_nuclear_post_1c_etype).
    Wu = W.tocoo()
    Wsym = sp.csr_matrix(
        (
            np.concatenate([Wu.data, Wu.data]).astype(np.float32),
            (np.concatenate([Wu.row, Wu.col]),
             np.concatenate([Wu.col, Wu.row])),
        ),
        shape=W.shape, dtype=np.float32,
    )
    Wsym.sort_indices()

    def _stored(M, r, c):
        row = M.indices[M.indptr[r]:M.indptr[r + 1]]
        return c in row

    assert _stored(Wsym, ei, ci) and _stored(Wsym, ci, ei)
    assert Wsym[ei, ci] == 0.0 and Wsym[ci, ei] == 0.0

    # The unsafe alternative drops it.
    bad = (W + W.T).tocsr()
    assert not _stored(bad, ei, ci)


def test_neg_one_sentinel_counts_as_anti_evidence():
    """A stored negative (`neg_one`-style mutual-exclusion sentinel) must
    count as real anti-evidence, not be skipped. Make CD68's macrophage
    fit barely positive, then add a stored strong-negative CD68-CD163: the
    extra anti-evidence pulls the partial mean below threshold and CD68
    stays on main. Skipping the stored negative would (wrongly) move it."""
    pmi = dict(_PMI)
    pmi[_norm("CD68", "CD163")] = -5.0   # stored strong-negative sentinel
    pmi[_norm("CD68", "LYZ")] = 0.1
    # Build a sparse panel from this perturbed dict (all pairs except the
    # designated absent ones).
    rows, cols, data = [], [], []
    for a, b in _all_pairs():
        key = _norm(a, b)
        if key in pmi:
            v = pmi[key]
        elif key in _ABSENT:
            continue
        else:  # pragma: no cover
            raise AssertionError(key)
        i, j = _gi(a), _gi(b)
        lo, hi = (i, j) if i < j else (j, i)
        rows.append(lo); cols.append(hi); data.append(v)
    W = sp.coo_matrix(
        (np.asarray(data, np.float32), (rows, cols)), shape=(G, G),
        dtype=np.float32,
    ).tocsr()
    labels = _run(W)
    # macrophage mean = (-5.0 + 0.1)/2 = -2.45 < main mean → stays on main.
    assert labels["CD68"] == PARENT
