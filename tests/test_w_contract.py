"""The ``aux["W"]`` contract: one symmetric gene x gene PMI matrix.

Panel builders emit the upper triangle only; ``prune_transcripts_fast``
passes a symmetrized W on to later stages while (before unified-prune step
2b) ``prune_transcripts_nuclear_seed`` passed the upper triangle on. Every
consumer therefore met both forms, and each one handled only one of them:

* consumers that symmetrized on their own (Stitch ``W + W.T``, the
  coherence slice ``sub + sub.T``, the housekeeping row+column count)
  DOUBLED every value or count when W was already symmetric;
* consumers that read W as given (cascade, Rescue veto, Mid-QC) saw half
  the pairs when W was triangular.

These tests pin the first half of the fix: the self-symmetrizing consumers
give the same answer for a triangular and a symmetric W.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from tracer._utils import fold_symmetric_dense, symmetric_csr
from tracer.stitching import (
    _slice_pmi_submatrix,
    coherence,
    compute_housekeeping_mask,
)

G = 60


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(0)
    A = sp.random(G, G, density=0.35, random_state=3, format="coo", dtype=np.float32)
    A.data = (A.data * 1.6 - 0.8).astype(np.float32)
    U = sp.triu(A, 1).tocsr()
    U.data[::11] = 0.0                    # observed PMI of exactly 0 must survive
    S = symmetric_csr(U)
    gene_sets = [np.sort(rng.choice(G, size=k, replace=False)) for k in (2, 5, 12, 30)]
    gene_sets += [rng.permutation(gs) for gs in gene_sets]   # unsorted order too
    return U, S, gene_sets


def _same_csr(a, b):
    return (np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
            and np.array_equal(a.data, b.data))


def test_symmetric_csr_properties(panels):
    U, S, _ = panels
    assert (S != S.T).nnz == 0
    assert S.nnz == 2 * U.nnz, "explicit zeros dropped or entries duplicated"
    assert _same_csr(symmetric_csr(S), S), "not idempotent"
    assert _same_csr(symmetric_csr(U.T.tocsr()), S), "lower triangle not handled"
    assert np.array_equal(S.toarray(), (U + U.T).toarray())


def test_fold_symmetric_dense(panels):
    U, S, sets = panels
    for gs in sets:
        bu = np.asarray(U[gs, :][:, gs].todense())
        bs = np.asarray(S[gs, :][:, gs].todense())
        assert np.array_equal(fold_symmetric_dense(bu), bu + bu.T)
        assert np.array_equal(fold_symmetric_dense(bs), bs)


def test_slice_does_not_double_symmetric_w(panels):
    U, S, sets = panels
    for gs in sets:
        assert np.array_equal(_slice_pmi_submatrix(U, gs), _slice_pmi_submatrix(S, gs))


@pytest.mark.parametrize("mode", ["count", "magnitude"])
def test_coherence_same_for_triangular_and_symmetric(panels, mode):
    U, S, sets = panels
    for gs in sets:
        assert coherence(gs, U, mode=mode) == coherence(gs, S, mode=mode)


def test_housekeeping_mask_same_for_triangular_and_symmetric(panels):
    U, S, _ = panels
    for k in (2, 5, 8):
        mu = compute_housekeeping_mask(U, min_strong_count=k)
        ms = compute_housekeeping_mask(S, min_strong_count=k)
        assert np.array_equal(mu, ms)
        # and both agree with the dense, unambiguous definition
        D = S.toarray()
        np.fill_diagonal(D, 0)
        ref = ((D > 0.05).sum(1) >= k) | ((D < -0.05).sum(1) >= k)
        assert np.array_equal(ms, ref)
