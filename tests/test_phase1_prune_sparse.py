"""Phase-1 nuclear-seed prune: dense vs sparse PMI backend.

`prune_cells_nuclear_seed` (dense (G,G) float32) and its sparse sibling
`prune_cells_nuclear_seed_sparse` (symmetric CSR built from the bootstrap's
upper-triangle `W_sparse`) must produce identical per-tx codes when the
panel is fully observed (bit-for-bit), and the sparse path must SKIP
structurally-absent pairs — matching the dense NaN-skip path and DIFFERING
from the legacy absent-as-0 fill (`nan_fill=0.0`).

The sparse backend is what unblocks whole-transcriptome panels: the dense
float32 (G,G) is ~1.6 GB at 20k genes and OOMs the prune around 15k.

These tests drive the Cython kernels directly so the comparison is exact
and independent of the Python wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_cy = pytest.importorskip("tracer._cy_prune")
prune_dense = _cy.prune_cells_nuclear_seed
prune_sparse = _cy.prune_cells_nuclear_seed_sparse

# Gene vocabulary: epithelial / macrophage modules + a gene-sparse STAT1.
# Kept alphabetically sorted so this fixture's gene→index ordering matches
# the one `build_dense_pmi_matrix_small_panel` derives (sorted unique genes) — the
# greedy bad-edge prune's tie-break is index-dependent, so the dense and
# sparse wrapper paths must share a gene ordering to be comparable.
GENES = sorted(["EPCAM", "KRT8", "VIM", "CD163", "LYZ", "CD68", "STAT1"])
IDX = {g: i for i, g in enumerate(GENES)}
G = len(GENES)

# Observed PMI pairs (unordered keys, normalized at load). STAT1 is observed
# only against EPCAM and CD163; its pairs with KRT8/VIM/CD68/LYZ are ABSENT
# (the skip-vs-fill discriminator).
_PMI = {
    ("EPCAM", "KRT8"): 0.9,
    ("EPCAM", "VIM"): 0.85,
    ("KRT8", "VIM"): 0.88,
    ("CD163", "LYZ"): 0.9,
    ("CD163", "CD68"): 0.8,
    ("LYZ", "CD68"): 0.82,
    ("EPCAM", "CD163"): -0.3,
    ("EPCAM", "LYZ"): -0.3,
    ("EPCAM", "CD68"): -0.3,
    ("KRT8", "CD163"): -0.3,
    ("KRT8", "LYZ"): -0.3,
    ("KRT8", "CD68"): -0.3,
    ("VIM", "CD163"): -0.2,
    ("VIM", "LYZ"): -0.2,
    ("VIM", "CD68"): 0.0,      # observed zero — must be preserved/counted
    ("STAT1", "EPCAM"): 0.8,
    ("STAT1", "CD163"): 0.5,
}
# Pairs left unobserved on the sparse panel.
_ABSENT_NAMES = {
    ("STAT1", "KRT8"), ("STAT1", "VIM"),
    ("STAT1", "CD68"), ("STAT1", "LYZ"),
}


def _norm(a, b):
    return (a, b) if a <= b else (b, a)


_PMI = {_norm(*k): v for k, v in _PMI.items()}
_ABSENT = {_norm(*k) for k in _ABSENT_NAMES}


def _all_pairs():
    return [(GENES[i], GENES[j]) for i in range(G) for j in range(i + 1, G)]


def _build_dense(*, fill_absent):
    W = np.full((G, G), np.nan, dtype=np.float32)
    for a, b in _all_pairs():
        key = _norm(a, b)
        if key in _PMI:
            v = _PMI[key]
        elif key in _ABSENT:
            v = fill_absent
        else:  # pragma: no cover
            raise AssertionError(f"unclassified {key}")
        if v is None:
            continue
        i, j = IDX[a], IDX[b]
        W[i, j] = v
        W[j, i] = v
    return W


def _build_sym_csr(*, include_absent):
    """Symmetric CSR (both off-diagonal positions stored) like what the
    wrapper feeds the sparse kernel after COO-stack symmetrization."""
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
            raise AssertionError(key)
        i, j = IDX[a], IDX[b]
        rows += [i, j]
        cols += [j, i]
        data += [v, v]
    M = sp.csr_matrix(
        (np.asarray(data, np.float32), (rows, cols)), shape=(G, G),
        dtype=np.float32,
    )
    M.sort_indices()
    return M


def _csr_args(M):
    return (M.indptr.astype(np.int32),
            M.indices.astype(np.int32),
            M.data.astype(np.float32))


# Two-cell tx layout:
#   Cell A — epithelial nuclear seed {EPCAM,KRT8,VIM} + a cytoplasmic STAT1
#            (the absent-pair discriminator).
#   Cell B — an EMT/macrophage doublet: nuclear {EPCAM,KRT8,VIM,CD163,LYZ,
#            CD68} exercising 1a stripping and 1c partial formation.
def _make_inputs():
    rows = []  # (gene, is_nuclear, cell)
    for g in ["EPCAM", "KRT8", "VIM"]:
        rows.append((g, 1, "A"))
    rows.append(("STAT1", 0, "A"))      # cytoplasmic STAT1
    for g in ["EPCAM", "KRT8", "VIM", "CD163", "LYZ", "CD68"]:
        rows.append((g, 1, "B"))

    tx_gene = np.array([IDX[g] for g, _, _ in rows], dtype=np.int32)
    tx_nuc = np.array([n for _, n, _ in rows], dtype=np.uint8)
    cells = [c for _, _, c in rows]
    cell_lists = []
    for cid in ["A", "B"]:
        cell_lists.append(
            np.array([i for i, c in enumerate(cells) if c == cid],
                     dtype=np.int32)
        )
    return cell_lists, tx_gene, tx_nuc, rows


KW = dict(min_nuclear_genes=3, skip_phase_1c=0,
          seed_coherence_floor=-1e30, nuclear_only_admit=0, tx_weighted=1)


def _run_dense(W, threshold):
    cell_lists, tx_gene, tx_nuc, _ = _make_inputs()
    return np.asarray(prune_dense(cell_lists, tx_gene, tx_nuc, W,
                                  float(threshold), **KW))


def _run_sparse(M, threshold):
    cell_lists, tx_gene, tx_nuc, _ = _make_inputs()
    ip, ix, dt = _csr_args(M)
    return np.asarray(prune_sparse(cell_lists, tx_gene, tx_nuc, ip, ix, dt,
                                   float(threshold), **KW))


# --------------------------------------------------------------------------

def test_dense_sparse_parity_fully_observed_bitforbit():
    """Every off-diagonal pair stored → dense and sparse kernels must
    return identical per-tx codes."""
    W = _build_dense(fill_absent=0.0)         # absent filled → fully dense
    M = _build_sym_csr(include_absent=True)   # absent stored → fully dense
    for thr in (0.0, 0.2, 0.5):
        d = _run_dense(W, thr)
        s = _run_sparse(M, thr)
        np.testing.assert_array_equal(d, s, err_msg=f"threshold={thr}")


def test_sparse_matches_dense_nan_skip_not_zero_fill():
    """With genuinely-absent pairs, the sparse path must equal the dense
    NaN-skip path and DIFFER from the buggy absent-as-0 fill.

    Discriminator: the cytoplasmic STAT1 tx in cell A, tested against the
    epithelial seed at threshold 0.5. Observed STAT1-EPCAM = 0.8; STAT1's
    pairs with KRT8/VIM are absent. Skip → mean 0.8 ≥ 0.5 → admitted to
    main (code 0). Zero-fill → mean (0.8+0+0)/3 = 0.267 < 0.5 → rejected;
    STAT1 is cytoplasmic so 1c can't rescue it → unassigned (code 2)."""
    thr = 0.5
    sparse = _run_sparse(_build_sym_csr(include_absent=False), thr)
    nan_skip = _run_dense(_build_dense(fill_absent=np.nan), thr)
    zero_fill = _run_dense(_build_dense(fill_absent=0.0), thr)

    np.testing.assert_array_equal(sparse, nan_skip)

    _, _, _, rows = _make_inputs()
    stat1_pos = next(i for i, (g, n, c) in enumerate(rows)
                     if g == "STAT1" and c == "A")
    assert sparse[stat1_pos] == 0          # admitted to main under skip
    assert zero_fill[stat1_pos] == 2       # wrongly rejected under fill
    assert not np.array_equal(sparse, zero_fill)


def test_sparse_preserves_observed_zero():
    """An observed PMI of exactly 0.0 (VIM-CD68 here) must survive the
    symmetric-CSR build as a stored, counted entry — `eliminate_zeros()`
    or `W + W.T` would drop it. Verify it is present in both off-diagonal
    positions of the symmetric CSR the kernel consumes."""
    M = _build_sym_csr(include_absent=False)
    vi, ci = IDX["VIM"], IDX["CD68"]

    def _stored(r, c):
        return c in M.indices[M.indptr[r]:M.indptr[r + 1]]

    assert _stored(vi, ci) and _stored(ci, vi)
    assert M[vi, ci] == 0.0 and M[ci, vi] == 0.0


def test_neg_one_sentinel_counts_in_greedy_strip():
    """A stored strong-negative (`neg_one`-style mutual-exclusion
    sentinel) must count as anti-evidence in the greedy bad-edge prune; a
    structurally-absent pair must not. Inject a stored strong-negative
    between two otherwise-coherent epithelial seed genes and confirm the
    sparse kernel strips one of them (vs. leaving the seed intact when the
    same pair is absent)."""
    thr = 0.2

    # Baseline: clean epithelial seed, all positive → no strip; the three
    # epithelial nuclear tx in cell B admit to main (code 0).
    base = _run_sparse(_build_sym_csr(include_absent=False), thr)

    # Perturb: store a strong-negative EPCAM-KRT8 sentinel.
    pmi = dict(_PMI)
    pmi[_norm("EPCAM", "KRT8")] = -5.0
    rows, cols, data = [], [], []
    for a, b in _all_pairs():
        key = _norm(a, b)
        if key in pmi:
            v = pmi[key]
        elif key in _ABSENT:
            continue
        else:  # pragma: no cover
            raise AssertionError(key)
        i, j = IDX[a], IDX[b]
        rows += [i, j]; cols += [j, i]; data += [v, v]
    M = sp.csr_matrix((np.asarray(data, np.float32), (rows, cols)),
                      shape=(G, G), dtype=np.float32)
    M.sort_indices()
    perturbed = _run_sparse(M, thr)

    # The stored sentinel must change the outcome (a strip / demotion),
    # proving it counted; an absent pair would have left `base` unchanged.
    assert not np.array_equal(base, perturbed)


# --------------------------------------------------------------------------
# Wrapper-level integration: prune_transcripts_nuclear_seed
# --------------------------------------------------------------------------
import pandas as pd  # noqa: E402

from tracer.metrics import PmiBootstrapResult  # noqa: E402
from tracer.pruning import prune_transcripts_nuclear_seed  # noqa: E402


def _npmi_df(*, fill_absent):
    recs = []
    for a, b in _all_pairs():
        key = _norm(a, b)
        if key in _PMI:
            v = _PMI[key]
        elif key in _ABSENT:
            if fill_absent is None:
                continue
            v = fill_absent
        recs.append((a, b, v))
    return pd.DataFrame(recs, columns=["gene_i", "gene_j", "PMI"])


def _bootstrap_result(*, include_absent):
    rows, cols, data = [], [], []
    for a, b in _all_pairs():
        key = _norm(a, b)
        if key in _PMI:
            v = _PMI[key]
        elif key in _ABSENT:
            if not include_absent:
                continue
            v = 0.0
        i, j = IDX[a], IDX[b]
        lo, hi = (i, j) if i < j else (j, i)
        rows.append(lo)
        cols.append(hi)
        data.append(v)
    W = sp.coo_matrix(
        (np.asarray(data, np.float32), (rows, cols)), shape=(G, G),
        dtype=np.float32,
    ).tocsr()
    return PmiBootstrapResult(W_sparse=W, genes=np.asarray(GENES))


def _df():
    _, _, _, rows = _make_inputs()
    recs = [(c, g, bool(n)) for (g, n, c) in rows]
    return pd.DataFrame(
        recs, columns=["cell_id", "feature_name", "overlaps_nucleus"])


def _prune(npmi, **kw):
    out, _aux = prune_transcripts_nuclear_seed(
        _df(), npmi,
        cell_id_col="cell_id", gene_col="feature_name",
        nuclear_col="overlaps_nucleus", metric_col="PMI",
        min_nuclear_genes=3, **kw,
    )
    return out["tracer_id"].to_numpy()


def test_wrapper_dense_sparse_parity_fully_observed():
    """End-to-end through prune_transcripts_nuclear_seed: on a fully-
    observed panel the dense NaN-skip path (nan_fill=None) and the sparse
    PmiBootstrapResult path produce identical tracer_id labels."""
    for thr in (0.0, 0.2, 0.5):
        dense = _prune(_npmi_df(fill_absent=0.0), nan_fill=None, threshold=thr)
        sparse = _prune(_bootstrap_result(include_absent=True), threshold=thr)
        np.testing.assert_array_equal(dense, sparse, err_msg=f"thr={thr}")


def test_wrapper_nan_fill_is_no_op_on_sparse_only_path():
    """Under the sparse-only PMI panel ingest (the dense
    `build_dense_pmi_matrix_small_panel` path retired in favor of
    `build_sparse_pmi_matrix_from_long`), `nan_fill` becomes a no-op
    because the CSR encoding has no NaN entries to fill — structurally-
    absent pairs are absent (skipped by _wget), period.

    The three pathways below all now reduce to the same sparse CSR:
      - bootstrap result with absent rows omitted,
      - long DF with absent rows omitted, nan_fill=None,
      - long DF with absent rows omitted, nan_fill=0.0 (legacy knob).

    Prior to the dense-PMI retirement the third path differed (it
    populated absent pairs as explicit 0.0 in the dense (G,G) ndarray
    and counted them in the mean-admission denominator). That behavior
    is intentionally retired: see commit log for the sparse-only PMI
    ingest migration."""
    thr = 0.5
    sparse = _prune(_bootstrap_result(include_absent=False), threshold=thr)
    nan_skip = _prune(_npmi_df(fill_absent=None), nan_fill=None, threshold=thr)
    zero_fill = _prune(_npmi_df(fill_absent=None), nan_fill=0.0, threshold=thr)

    np.testing.assert_array_equal(sparse, nan_skip)
    np.testing.assert_array_equal(sparse, zero_fill)  # nan_fill no-op


def test_wrapper_aux_W_is_sparse_for_bootstrap_input():
    """The sparse path keeps aux['W'] as the (upper-triangle) CSR so
    downstream reassign-1c stays on the sparse backend instead of
    densifying."""
    _out, aux = prune_transcripts_nuclear_seed(
        _df(), _bootstrap_result(include_absent=False),
        cell_id_col="cell_id", gene_col="feature_name",
        nuclear_col="overlaps_nucleus", metric_col="PMI",
        min_nuclear_genes=3, threshold=0.2,
    )
    assert sp.issparse(aux["W"])
