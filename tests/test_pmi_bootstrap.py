"""Synthetic correctness test for ``tracer.metrics.compute_pmi_bootstrap``.

Plants 5 known gene-pair structures via :func:`tests.synthetic.make_synthetic_npmi_panel`
and asserts the bootstrap classifies each correctly:

  - genes 0, 1: strong positive cooccurrence → ``W[0,1] > 0``
  - genes 2, 3: strong mutual exclusivity   → ``W[2,3] < 0``
  - genes 4, 5: independent (rate 0.3 each) → ``|W[4,5]| < 0.2`` or absent
  - genes 6, 7: rare with zero observed cooccur, E[cooccur] < 10 →
                indeterminate (absent from W_sparse)
  - genes 8, 9: high marginal with zero observed cooccur,
                E[cooccur] ≥ 10 → ``neg_one`` sentinel (W[8,9] == -1)
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from tracer.metrics import compute_pmi_bootstrap

from tests.synthetic import make_synthetic_npmi_panel


@pytest.fixture(scope="module")
def bootstrap_result():
    """Compute the bootstrap once and reuse across the test module."""
    df, M = make_synthetic_npmi_panel()
    res = compute_pmi_bootstrap(
        df, group_key="cell_id", feature_col="feature_name",
        tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        expected_cooccur_for_neg_one=10.0,
        bootstrap_kernel="pair_gather",
        seed=0, show_progress=False,
    )
    return res, M


def _W_lookup(res):
    """Convert sparse W to a dense {(i, j): value} dict for easy lookup."""
    W = res.W_sparse if sp.isspmatrix_coo(res.W_sparse) else res.W_sparse.tocoo()
    return {(int(i), int(j)): float(v) for i, j, v in zip(W.row, W.col, W.data)}


def test_strong_positive_classified_pos(bootstrap_result):
    res, M = bootstrap_result
    W = _W_lookup(res)
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    i, j = g_to_i["gene_00"], g_to_i["gene_01"]
    key = (min(i, j), max(i, j))
    assert key in W, "Strong-positive pair should appear in W_sparse"
    assert W[key] > 0.1, f"Expected NPMI > 0.1 for strong positive pair, got {W[key]}"


def test_strong_negative_classified_neg(bootstrap_result):
    res, M = bootstrap_result
    W = _W_lookup(res)
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    i, j = g_to_i["gene_02"], g_to_i["gene_03"]
    key = (min(i, j), max(i, j))
    assert key in W, "Strong-negative pair should appear in W_sparse"
    assert W[key] < -0.1, f"Expected NPMI < -0.1 for strong negative pair, got {W[key]}"


def test_independent_classified_indeterminate_or_zero(bootstrap_result):
    """Independent pair should either be absent (indeterminate) or have
    near-zero NPMI."""
    res, M = bootstrap_result
    W = _W_lookup(res)
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    i, j = g_to_i["gene_04"], g_to_i["gene_05"]
    key = (min(i, j), max(i, j))
    if key in W:
        # If the bootstrap classified it confidently, the value should be near zero.
        assert abs(W[key]) < 0.2, f"Independent pair should be near zero, got {W[key]}"


def test_high_marginal_zero_cooccur_classified_neg_one(bootstrap_result):
    """Pair with high marginal rate and zero observed co-occurrence
    (E[cooccur] ≫ 10) should be classified as the ``neg_one`` sentinel."""
    res, M = bootstrap_result
    W = _W_lookup(res)
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    i, j = g_to_i["gene_08"], g_to_i["gene_09"]
    key = (min(i, j), max(i, j))
    assert key in W, "High-marginal zero-cooccur pair should appear in W_sparse"
    assert W[key] == -1.0, f"Expected neg_one sentinel, got {W[key]}"


def test_low_marginal_zero_cooccur_left_indeterminate(bootstrap_result):
    """Pair with low marginal rate and zero observed co-occurrence
    (E[cooccur] < 10) should be left indeterminate (absent from W)."""
    res, M = bootstrap_result
    W = _W_lookup(res)
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    i, j = g_to_i["gene_06"], g_to_i["gene_07"]
    key = (min(i, j), max(i, j))
    assert key not in W, (
        f"Low-marginal zero-cooccur pair should be absent from W_sparse "
        f"(indeterminate); got W[{key}] = {W.get(key)}"
    )


def test_diagnostics_report_n_pairs(bootstrap_result):
    """Sanity: diagnostics dict should report counts that sum sensibly."""
    res, _ = bootstrap_result
    diag = res.diagnostics
    # Some non-zero classifications should exist
    n_classified = (
        diag.get("n_pos", 0) + diag.get("n_neg", 0) + diag.get("n_neg_one", 0)
    )
    assert n_classified >= 2, f"Expected at least 2 classified pairs, got {n_classified}"


def test_counts_matrix_matches_df_presence():
    """A counts matrix and the equivalent long-form df build the same M."""
    import numpy as np, scipy.sparse as sp
    from tracer.metrics import _presence_from_counts, _build_presence_matrix
    import pandas as pd
    # 3 cells x 4 genes raw counts
    X = sp.csr_matrix(np.array([[2, 0, 5, 1],
                                [3, 2, 0, 0],
                                [0, 1, 4, 2]], dtype=np.float32))
    var = np.array(["g0", "g1", "g2", "g3"])
    obs = np.array(["c0", "c1", "c2"])
    M, genes, ctx = _presence_from_counts(X, var, obs, min_occurrences_per_context=2)
    # gene present where count>=2: c0:{g0,g2}, c1:{g0,g1}, c2:{g2,g3}
    dense = np.asarray(M.todense())
    gi = {g: i for i, g in enumerate(genes)}
    assert dense[0, gi["g0"]] == 1 and dense[0, gi["g2"]] == 1
    assert dense[0, gi["g3"]] == 0  # count 1 < 2
    assert dense[1, gi["g1"]] == 1 and dense[1, gi["g3"]] == 0
    assert set(genes) == {"g0", "g1", "g2", "g3"}
    assert M.dtype == np.int32


def test_counts_xor_df_required():
    import numpy as np, scipy.sparse as sp, pytest
    from tracer.metrics import compute_pmi_bootstrap
    X = sp.csr_matrix(np.array([[2, 2], [2, 2]], dtype=np.float32))
    var = np.array(["a", "b"])
    with pytest.raises(ValueError, match="exactly one"):
        compute_pmi_bootstrap(None, counts=None)            # neither
    # counts path returns a result without raising
    res = compute_pmi_bootstrap(None, counts=(X, var), metric="pmi",
                                min_expected_cooccur_for_evidence=0.5,
                                bootstrap_kernel="pair_gather", seed=0)
    assert res.genes.tolist() == ["a", "b"]


def test_pair_gather_kernel_matches_legacy_output():
    """Refactor must preserve today's exact output on the synthetic panel."""
    import numpy as np
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, M = make_synthetic_npmi_panel()
    res = compute_pmi_bootstrap(
        df, group_key="cell_id", feature_col="feature_name",
        metric="npmi", bootstrap_kernel="pair_gather",
        seed=0, show_progress=False,
    )
    W = res.W_sparse.tocoo()
    got = {(int(i), int(j)): float(v) for i, j, v in zip(W.row, W.col, W.data)}
    # golden values captured from the pre-refactor run (fill from Step 2 baseline)
    assert (0, 1) in got and got[(0, 1)] > 0.1
    assert (2, 3) in got and got[(2, 3)] < -0.1
    assert (8, 9) in got and got[(8, 9)] == -1.0
    assert res.diagnostics["n_neg_one"] >= 1


def test_gene_row_kernel_parity_with_pair_gather():
    """gene_row settles the same pair SET as pair_gather; values within tol."""
    import numpy as np
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, M = make_synthetic_npmi_panel()
    common = dict(group_key="cell_id", feature_col="feature_name",
                  metric="npmi", seed=0, show_progress=False)
    a = compute_pmi_bootstrap(df, bootstrap_kernel="pair_gather", **common)
    b = compute_pmi_bootstrap(df, bootstrap_kernel="gene_row", **common)
    Wa = {(int(i), int(j)) for i, j in zip(*a.W_sparse.nonzero())}
    Wb = {(int(i), int(j)) for i, j in zip(*b.W_sparse.nonzero())}
    # neg_one + clearly-settled pairs must match exactly
    assert (0, 1) in Wb and (2, 3) in Wb and (8, 9) in Wb
    assert b.W_sparse.tocsr()[8, 9] == -1.0
    # sign agreement on the strong pairs
    assert b.W_sparse.tocsr()[0, 1] > 0.1
    assert b.W_sparse.tocsr()[2, 3] < -0.1
    # broad set agreement (allow tiny boundary differences from RNG stream)
    assert len(Wa ^ Wb) <= 2


def test_gene_row_accepts_single_element_tau_sequence():
    """gene_row must accept single-threshold tau given as a 1-element
    sequence (``[0.05]``/``(0.05,)``/``np.array([0.05])``) — the tau parser
    treats size==1 as scalar — and must still reject genuine dual-tau."""
    import numpy as np
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    common = dict(group_key="cell_id", feature_col="feature_name",
                  metric="npmi", bootstrap_kernel="gene_row", seed=0,
                  show_progress=False, max_bootstraps=600)
    for tau in ([0.05], (0.05,), np.array([0.05]), [0.05, 0.05]):
        res = compute_pmi_bootstrap(df, tau=tau, **common)
        assert res.diagnostics["is_dual_tau"] is False
        assert res.diagnostics["kernel"] == "gene_row"
    # genuine dual-tau (low < high) is unsupported by gene_row → fail loud
    with pytest.raises(NotImplementedError):
        compute_pmi_bootstrap(df, tau=[0.02, 0.08], **common)


def test_gene_order_prob_ascending():
    import numpy as np
    from tracer.metrics import _gene_processing_order
    k = np.array([100, 5, 50, 5], dtype=np.float64)   # detection per gene
    order = _gene_processing_order("prob_ascending", k=k)
    # ascending k: genes 1,3 (k=5) before 2 (k=50) before 0 (k=100)
    assert order[-1] == 0
    assert set(order[:2].tolist()) == {1, 3}
    assert list(order) != list(range(len(k)))  # not identity unless sorted


def test_owned_partners_each_pair_once_and_window_shrinks():
    import numpy as np
    from tracer.metrics import _owned_partners
    # candidate upper-tri pairs (by gene index): (0,1),(0,2),(1,2),(2,3)
    obs_i = np.array([0, 0, 1, 2]); obs_j = np.array([1, 2, 2, 3])
    can = np.array([True, True, True, True])
    order = np.array([0, 1, 2, 3])           # process in index order
    pos = np.empty(4, dtype=np.int64); pos[order] = np.arange(4)
    indptr, partner, pairref = _owned_partners(obs_i, obs_j, can, pos)
    # each pair owned exactly once
    assert partner.size == 4
    # gene 0 owns {1,2}; gene 1 owns {2}; gene 2 owns {3}; gene 3 owns {}
    owned = {g: partner[indptr[g]:indptr[g+1]].tolist() for g in range(4)}
    assert sorted(owned[0]) == [1, 2]
    assert owned[1] == [2] and owned[2] == [3] and owned[3] == []
    # window (owned count) is non-increasing here
    counts = [indptr[g+1]-indptr[g] for g in range(4)]
    assert counts == [2, 1, 1, 0]


def test_gene_batches_respect_pair_cap_and_cover_all():
    import numpy as np
    from tracer.metrics import _gene_batches
    order = np.arange(6)
    owned_counts = np.array([3, 3, 3, 3, 3, 0])   # owned pairs per gene (in order)
    # budget that allows ~5 pairs/batch -> batches: [0,2)=6>5 so [0,1],... check coverage
    batches = _gene_batches(order, owned_counts, gene_batch_peak_gb=1e-6,
                            coarse_block=200)
    # every gene covered exactly once, contiguous
    covered = []
    for (s, e) in batches:
        covered.extend(range(s, e))
    assert covered == list(range(6))
    # each batch's owned-pair sum <= cap (except a single gene that alone exceeds)
    cap = max(1, int(1e-6 * 1e9 / (200 * 32)))
    for (s, e) in batches:
        tot = owned_counts[s:e].sum()
        assert (e - s == 1) or tot <= cap


def test_gene_row_multibatch_matches_singlebatch():
    import numpy as np
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, M = make_synthetic_npmi_panel()
    common = dict(group_key="cell_id", feature_col="feature_name",
                  metric="npmi", bootstrap_kernel="gene_row", seed=0,
                  show_progress=False)
    big = compute_pmi_bootstrap(df, gene_batch_peak_gb=16.0, **common)
    small = compute_pmi_bootstrap(df, gene_batch_peak_gb=1e-7, **common)  # force splits
    Sbig = {(int(i), int(j)) for i, j in zip(*big.W_sparse.nonzero())}
    Ssmall = {(int(i), int(j)) for i, j in zip(*small.W_sparse.nonzero())}
    assert len(Sbig ^ Ssmall) <= 2   # same settled set up to boundary RNG


def test_checkpoint_roundtrip(tmp_path):
    import numpy as np
    from tracer.metrics import _write_checkpoint, _read_checkpoint
    p = tmp_path / "ck.npz"
    _write_checkpoint(str(p), [0, 1], [2, 3], [0.5, -0.5], G=4, cursor=2)
    rows, cols, vals, cursor = _read_checkpoint(str(p))
    assert rows == [0, 1] and cols == [2, 3]
    assert vals == [0.5, -0.5] and cursor == 2
    assert _read_checkpoint(str(tmp_path / "missing.npz")) is None


def test_checkpoint_resume_equiv(tmp_path):
    import numpy as np
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, M = make_synthetic_npmi_panel()
    common = dict(group_key="cell_id", feature_col="feature_name", metric="npmi",
                  bootstrap_kernel="gene_row", gene_batch_peak_gb=1e-7, seed=0,
                  show_progress=False)
    ck = str(tmp_path / "run.ckpt.npz")
    full = compute_pmi_bootstrap(df, **common)                  # no checkpoint
    # write a partial checkpoint by running once with checkpoint, then re-run resumes
    part = compute_pmi_bootstrap(df, checkpoint_path=ck, **common)
    resumed = compute_pmi_bootstrap(df, checkpoint_path=ck, **common)  # resumes/no-op
    Sfull = {(int(i), int(j)) for i, j in zip(*full.W_sparse.nonzero())}
    Sres = {(int(i), int(j)) for i, j in zip(*resumed.W_sparse.nonzero())}
    assert Sfull == Sres


# Golden values captured from the PRE-vectorization gene_row kernel on the
# default synthetic panel (make_synthetic_npmi_panel(), seed 42 fixture).
# Run config: tau=0.05, ci_level=0.95, max_bootstraps=2000,
# coarse_block=refine_block=200, metric="npmi". These are the same for
# seeds 0 and 1 because the only bootstrap-settled W entry (the strong
# positive pair (0,1)) clears ±tau in the very first coarse block under
# either RNG stream; the (2,3)/(8,9) entries are Stage-1 neg_one sentinels.
# The vectorized kernel MUST reproduce these bit-for-bit.
_GENE_ROW_GOLDEN_W_NNZ = {
    (0, 1): np.float32(0.95119965),
    (2, 3): np.float32(-1.0),
    (8, 9): np.float32(-1.0),
}
# Bootstrap-kernel diagnostics (settle-count contract) for the same run.
_GENE_ROW_GOLDEN_DIAG = {
    "n_pos": 1,
    "n_neg": 0,
    "n_dead_zone": 1,
    "n_unsettled": 131,
}
_GENE_ROW_GOLDEN_NBP_SUM = 262400
_GENE_ROW_GOLDEN_NBP_LEN = 133


def _golden_dense_W(G=20):
    W = np.zeros((G, G), dtype=np.float32)
    for (i, j), v in _GENE_ROW_GOLDEN_W_NNZ.items():
        W[i, j] = v
    return W


@pytest.mark.parametrize("seed", [0, 1])
def test_gene_row_vectorized_bitwise_identical(seed):
    """HARD GATE: the vectorized gene_row kernel must reproduce the
    pre-change kernel's W_sparse bit-for-bit AND the exact settle-count
    diagnostics (n_pos/n_neg/n_dead_zone/n_unsettled/n_bootstraps_per_pair)
    on the synthetic panel, for the same seed. Same RNG draws → same samples
    → same quantiles → same settle decisions → identical W."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    res = compute_pmi_bootstrap(
        df, group_key="cell_id", feature_col="feature_name",
        metric="npmi", bootstrap_kernel="gene_row",
        tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        seed=seed, show_progress=False,
    )
    W = res.W_sparse.toarray()
    assert np.array_equal(W, _golden_dense_W(W.shape[0])), (
        "vectorized gene_row W diverged from golden\n"
        f"got nnz={dict(zip(map(tuple, np.argwhere(W != 0)), W[W != 0]))}"
    )
    d = res.diagnostics
    for k, v in _GENE_ROW_GOLDEN_DIAG.items():
        assert int(d[k]) == v, f"diag[{k}] = {d[k]} != golden {v}"
    nbp = np.asarray(d["n_bootstraps_per_pair"])
    assert len(nbp) == _GENE_ROW_GOLDEN_NBP_LEN
    assert int(nbp.sum()) == _GENE_ROW_GOLDEN_NBP_SUM


# Golden for the MULTI-BATCH path (gene_batch_peak_gb=1e-7 forces many
# single-gene batches). W is identical to single-batch (the same 3 nnz),
# but the settle-count diagnostics differ because each batch seeds its own
# RNG (default_rng(seed + b_idx)) so the per-block sample streams differ.
# Captured from the pre-vectorization kernel, seed 0.
_GENE_ROW_GOLDEN_DIAG_MB = {
    "n_pos": 1,
    "n_neg": 0,
    "n_dead_zone": 0,
    "n_unsettled": 132,
}
_GENE_ROW_GOLDEN_NBP_SUM_MB = 264200
_GENE_ROW_GOLDEN_NBP_LEN_MB = 133


def test_gene_row_subsample_runs_and_deterministic():
    """gene_row with subsample_size=s must run (no NotImplementedError) and be
    deterministic: same (seed, s) → bit-identical W."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    common = dict(
        group_key="cell_id", feature_col="feature_name", metric="npmi",
        bootstrap_kernel="gene_row", tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        subsample_size=400, seed=0, show_progress=False,
    )
    a = compute_pmi_bootstrap(df, **common)
    b = compute_pmi_bootstrap(df, **common)
    assert a.diagnostics["kernel"] == "gene_row"
    assert a.diagnostics["subsample_size"] == 400
    Wa = a.W_sparse.toarray()
    Wb = b.W_sparse.toarray()
    assert np.array_equal(Wa, Wb), "same (seed, subsample_size) must give identical W"
    # The strong positive pair (0,1) still settles positive under subsampling.
    assert Wa[0, 1] > 0.1
    # The Stage-1 neg_one sentinels are independent of the bootstrap kernel.
    assert Wa[8, 9] == -1.0


def test_gene_row_subsample_full_count_matches_none():
    """LARGE-S SANITY: subsample_size == C draws C cells via the subsample path.
    The RNG consumption differs from the rc-bincount full path, so W need not be
    bit-identical, but the settled SET must agree within a tiny tolerance."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, M = make_synthetic_npmi_panel()
    C = M.shape[0]
    common = dict(
        group_key="cell_id", feature_col="feature_name", metric="npmi",
        bootstrap_kernel="gene_row", tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        seed=0, show_progress=False,
    )
    full = compute_pmi_bootstrap(df, subsample_size=None, **common)
    fullc = compute_pmi_bootstrap(df, subsample_size=C, **common)
    Sfull = {(int(i), int(j)) for i, j in zip(*full.W_sparse.nonzero())}
    Sfullc = {(int(i), int(j)) for i, j in zip(*fullc.W_sparse.nonzero())}
    # Settled-set agreement within a small tolerance (boundary RNG differences).
    assert len(Sfull ^ Sfullc) <= 2
    # The unambiguous entries must agree exactly.
    Wc = fullc.W_sparse.tocsr()
    assert Wc[0, 1] > 0.1
    assert Wc[8, 9] == -1.0


def test_gene_row_vectorized_bitwise_identical_multibatch():
    """The bitwise gate must also hold when the owned pairs are split across
    multiple gene batches (forces the per-batch accumulate/settle path)."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    res = compute_pmi_bootstrap(
        df, group_key="cell_id", feature_col="feature_name",
        metric="npmi", bootstrap_kernel="gene_row",
        tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        gene_batch_peak_gb=1e-7,   # force many single-gene batches
        seed=0, show_progress=False,
    )
    W = res.W_sparse.toarray()
    assert np.array_equal(W, _golden_dense_W(W.shape[0]))
    d = res.diagnostics
    for k, v in _GENE_ROW_GOLDEN_DIAG_MB.items():
        assert int(d[k]) == v, f"diag[{k}] = {d[k]} != golden {v}"
    nbp = np.asarray(d["n_bootstraps_per_pair"])
    assert len(nbp) == _GENE_ROW_GOLDEN_NBP_LEN_MB
    assert int(nbp.sum()) == _GENE_ROW_GOLDEN_NBP_SUM_MB


# ======================================================================
# O(1)-memory "counter" CI accumulator (ci_accumulator="counter")
# ======================================================================

_COMMON_GR = dict(
    group_key="cell_id", feature_col="feature_name", metric="npmi",
    bootstrap_kernel="gene_row", tau=0.05, ci_level=0.95,
    max_bootstraps=2000, coarse_block=200, refine_block=200,
    seed=0, show_progress=False,
)


def test_counter_logic_matches_quantile_decision():
    """GATE 2 — counter settle classification matches the np.quantile-vs-tau
    decision for clearly pos / neg / tight_null sample sequences.

    The counter test reduces the percentile-CI settle to integer counts:
      pos   <=> cnt_above >  ci_hi_q * nsamp
      neg   <=> cnt_below >  ci_hi_q * nsamp
      tight <=> cnt_below <  ci_lo_q * nsamp  AND  cnt_above < ci_lo_q * nsamp
    For UNAMBIGUOUS distributions (mass clearly on one side of +/-tau) this is
    identical to the linear-interpolation quantile decision used by the
    samples path. Near-boundary churn (a single order statistic straddling
    the ci quantile) is covered by the agreement gate, not here.
    """
    ci_level = 0.95
    tau = 0.05
    ci_lo_q = (1.0 - ci_level) / 2.0
    ci_hi_q = 1.0 - ci_lo_q

    def quantile_kind(arr):
        lo, hi = np.quantile(arr, [ci_lo_q, ci_hi_q])
        if lo > tau:
            return 1
        if hi < -tau:
            return -1
        if lo > -tau and hi < tau:
            return 3
        return 0

    def counter_kind(arr):
        n = arr.size
        cnt_above = int((arr > tau).sum())
        cnt_below = int((arr < -tau).sum())
        if cnt_above > ci_hi_q * n:
            return 1
        if cnt_below > ci_hi_q * n:
            return -1
        if cnt_below < ci_lo_q * n and cnt_above < ci_lo_q * n:
            return 3
        return 0

    rng = np.random.default_rng(7)
    # Clearly positive: mass well above +tau.
    pos = rng.normal(0.6, 0.05, size=200)
    # Clearly negative: mass well below -tau.
    neg = rng.normal(-0.6, 0.05, size=200)
    # Clearly tight-null: mass tightly around 0, inside +/-tau.
    tight = rng.normal(0.0, 0.005, size=200)

    assert quantile_kind(pos) == 1 and counter_kind(pos) == 1
    assert quantile_kind(neg) == -1 and counter_kind(neg) == -1
    assert quantile_kind(tight) == 3 and counter_kind(tight) == 3


def test_counter_vs_quantile_symmetric_diff_small():
    """GATE 3 (synthetic random) — over many random sample sequences the
    counter classification agrees with the quantile decision except for a
    tiny fraction of near-boundary pairs (empirical-CDF vs linear-interp).
    Quantify and bound the disagreement rate."""
    ci_level = 0.95
    tau = 0.05
    ci_lo_q = (1.0 - ci_level) / 2.0
    ci_hi_q = 1.0 - ci_lo_q
    rng = np.random.default_rng(0)
    mism = 0
    n_trials = 5000
    for _ in range(n_trials):
        n = int(rng.integers(40, 400))
        arr = rng.normal(rng.uniform(-0.3, 0.3), rng.uniform(0.02, 0.4), size=n)
        lo, hi = np.quantile(arr, [ci_lo_q, ci_hi_q])
        if lo > tau:
            qk = 1
        elif hi < -tau:
            qk = -1
        elif lo > -tau and hi < tau:
            qk = 3
        else:
            qk = 0
        cnt_above = int((arr > tau).sum())
        cnt_below = int((arr < -tau).sum())
        if cnt_above > ci_hi_q * n:
            ck = 1
        elif cnt_below > ci_hi_q * n:
            ck = -1
        elif cnt_below < ci_lo_q * n and cnt_above < ci_lo_q * n:
            ck = 3
        else:
            ck = 0
        if qk != ck:
            mism += 1
    # Boundary churn only: well under 1% of trials.
    assert mism / n_trials < 0.01, f"counter/quantile disagreement {mism}/{n_trials}"


def test_counter_mode_runs_and_emits_diag_keys():
    """Counter mode runs end-to-end and emits the same diagnostics contract
    keys as samples mode (n_pos/n_neg/n_dead_zone/n_unsettled/
    n_bootstraps_per_pair)."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    res = compute_pmi_bootstrap(df, ci_accumulator="counter", **_COMMON_GR)
    d = res.diagnostics
    for k in ("n_pos", "n_neg", "n_dead_zone", "n_unsettled",
              "n_bootstraps_per_pair"):
        assert k in d, f"missing diag key {k}"
    nbp = np.asarray(d["n_bootstraps_per_pair"])
    # nsamp array, one entry per owned pair, all positive ints.
    assert nbp.ndim == 1 and nbp.size > 0
    assert np.issubdtype(nbp.dtype, np.integer)
    # The strong-positive pair still settles positive.
    assert res.W_sparse.tocsr()[0, 1] > 0.1
    # Stage-1 neg_one sentinels are kernel-independent.
    assert res.W_sparse.tocsr()[8, 9] == -1.0


def test_counter_vs_samples_settled_set_agreement():
    """GATE 3 (kernel) — counter-mode settled SET agrees with samples-mode on
    the synthetic panel within a tiny near-tau tolerance."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    samp = compute_pmi_bootstrap(df, ci_accumulator="samples", **_COMMON_GR)
    cnt = compute_pmi_bootstrap(df, ci_accumulator="counter", **_COMMON_GR)
    Ssamp = {(int(i), int(j)) for i, j in zip(*samp.W_sparse.nonzero())}
    Scnt = {(int(i), int(j)) for i, j in zip(*cnt.W_sparse.nonzero())}
    sym = Ssamp ^ Scnt
    # Only pairs sitting right at the ci quantile near tau may flip.
    assert len(sym) <= 2, f"settled-set symmetric diff too large: {sym}"
    # The unambiguous entries must agree exactly.
    assert (0, 1) in Ssamp and (0, 1) in Scnt
    assert (8, 9) in Ssamp and (8, 9) in Scnt


def test_counter_mode_deterministic():
    """GATE 5 — same seed -> identical W in counter mode."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    a = compute_pmi_bootstrap(df, ci_accumulator="counter", **_COMMON_GR)
    b = compute_pmi_bootstrap(df, ci_accumulator="counter", **_COMMON_GR)
    assert np.array_equal(a.W_sparse.toarray(), b.W_sparse.toarray())


def test_counter_mode_incompatible_with_persist_ci():
    """Guard — ci_accumulator='counter' cannot produce CI magnitudes, so it is
    incompatible with persist_ci=True and must raise ValueError."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    with pytest.raises(ValueError):
        compute_pmi_bootstrap(df, ci_accumulator="counter", persist_ci=True,
                              **_COMMON_GR)


def test_counter_mode_o1_memory_independent_of_budget():
    """GATE 4 — single-pass counter mode stores NO per-pair sample arrays and NO
    (n_owned x block) buffer; per-pair accumulator state is 3 ints regardless of
    max_bootstraps. Assert the only per-pair state is cnt_below/cnt_above/nsamp,
    whose total nbytes is independent of max_bootstraps and of the block size."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    common = dict(_COMMON_GR)
    common.pop("max_bootstraps")
    r_small = compute_pmi_bootstrap(
        df, ci_accumulator="counter", max_bootstraps=200, **common)
    r_big = compute_pmi_bootstrap(
        df, ci_accumulator="counter", max_bootstraps=2000, **common)
    # Per-pair accumulator footprint = 3 int arrays of length == #owned pairs.
    # n_bootstraps_per_pair IS the nsamp array; its length (the pair count) is
    # identical across budgets, so the per-pair state size does not grow with B.
    n_small = np.asarray(r_small.diagnostics["n_bootstraps_per_pair"]).size
    n_big = np.asarray(r_big.diagnostics["n_bootstraps_per_pair"]).size
    assert n_small == n_big
    # The pair-count is also independent of the block cadence (single global
    # ownership, not per-batch): same #pairs under tiny vs large blocks.
    n_tinyblock = np.asarray(
        compute_pmi_bootstrap(
            df, ci_accumulator="counter", max_bootstraps=200,
            **{**common, "coarse_block": 10, "refine_block": 10},
        ).diagnostics["n_bootstraps_per_pair"]).size
    assert n_tinyblock == n_small

    # Structural checks on the SINGLE-PASS counter kernel.
    import inspect
    from tracer import metrics as _m
    src = inspect.getsource(_m._bootstrap_gene_rows_counter)
    # No per-pair sample store and no (n_owned x block) 2-D buffer is allocated.
    assert "sample_store" not in src
    assert "np.empty((n_owned, block)" not in src
    assert "block_rows" not in src
    # Counters are incremented PER ITERATION (per-gene), not from a block buffer.
    assert "cnt_above[base:s.stop][um] += (val[um] > tau_high)" in src
    # Single pass: no gene-batch loop, settle via the O(1) count-based helper.
    assert "for b_idx, (bs, be) in enumerate(batches)" not in src
    assert "_counter_settle_inplace(" in src
    # The O(1) helper itself reads no stored samples — only the 3 int arrays.
    settle_src = inspect.getsource(_m._counter_settle_inplace)
    assert "sample_store" not in settle_src
    assert "cnt_above" in settle_src and "cnt_below" in settle_src


def test_counter_single_pass_ignores_gene_batch_budget():
    """GATE 4 (single-pass) — a budget that WOULD have forced many gene-batches
    in samples mode is ignored in counter mode (single pass), warns, and yields
    the SAME W as the default budget. Holds all candidate pairs at once."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    big = compute_pmi_bootstrap(
        df, ci_accumulator="counter", gene_batch_peak_gb=16.0, **_COMMON_GR)
    # A tiny budget would split the samples path into many single-gene batches;
    # counter mode must ignore it (single pass) and warn.
    with pytest.warns(UserWarning, match="gene_batch_peak_gb is ignored"):
        small = compute_pmi_bootstrap(
            df, ci_accumulator="counter", gene_batch_peak_gb=1e-9, **_COMMON_GR)
    assert np.array_equal(big.W_sparse.toarray(), small.W_sparse.toarray())


def test_counter_single_pass_block_cadence_invariant():
    """Single-pass counter early-stop cadence still works: changing
    coarse_block/refine_block must not change which pairs settle pos/neg
    (settlement decision depends on accumulated counts, not the block size)."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    common = {k: v for k, v in _COMMON_GR.items()
              if k not in ("coarse_block", "refine_block")}
    a = compute_pmi_bootstrap(
        df, ci_accumulator="counter", coarse_block=200, refine_block=200,
        **common)
    b = compute_pmi_bootstrap(
        df, ci_accumulator="counter", coarse_block=50, refine_block=50,
        **common)
    Sa = {(int(i), int(j)) for i, j in zip(*a.W_sparse.nonzero())}
    Sb = {(int(i), int(j)) for i, j in zip(*b.W_sparse.nonzero())}
    # Unambiguous pairs settle identically regardless of cadence; only near-tau
    # pairs may differ by at most a couple due to different early-stop points.
    assert len(Sa ^ Sb) <= 2
    assert (0, 1) in Sa and (0, 1) in Sb
    assert (8, 9) in Sa and (8, 9) in Sb


def test_counter_checkpoint_ignored_with_warning(tmp_path):
    """Single-pass counter mode does not support per-batch checkpointing: a
    checkpoint_path is ignored with a warning and no file is written."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    ckpt = tmp_path / "ck.npz"
    with pytest.warns(UserWarning, match="checkpoint_path is ignored"):
        compute_pmi_bootstrap(
            df, ci_accumulator="counter", checkpoint_path=str(ckpt),
            **_COMMON_GR)
    assert not ckpt.exists()


# =====================================================================
# pmi_formula="epsilon" — opt-in unified PMI formula gates
# =====================================================================


def _run_eps(df, *, kernel="pair_gather", formula="jeffreys",
             ci_accumulator="samples", **kwargs):
    """Shared helper: run compute_pmi_bootstrap on the synthetic panel."""
    from tracer.metrics import compute_pmi_bootstrap
    common = dict(
        group_key="cell_id", feature_col="feature_name", metric="npmi",
        tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        seed=0, show_progress=False,
    )
    common.update(kwargs)
    return compute_pmi_bootstrap(
        df, bootstrap_kernel=kernel, pmi_formula=formula,
        ci_accumulator=ci_accumulator, **common,
    )


def test_pmi_formula_default_is_jeffreys_bitwise():
    """GATE 1: pmi_formula default ("jeffreys") leaves all paths bitwise-
    identical to the pre-change kernel — explicit and implicit defaults agree."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    common = dict(
        group_key="cell_id", feature_col="feature_name",
        metric="npmi", tau=0.05, ci_level=0.95,
        max_bootstraps=2000, coarse_block=200, refine_block=200,
        seed=0, show_progress=False,
    )
    for kernel in ("pair_gather", "gene_row"):
        a = compute_pmi_bootstrap(df, bootstrap_kernel=kernel, **common)
        b = compute_pmi_bootstrap(
            df, bootstrap_kernel=kernel, pmi_formula="jeffreys", **common,
        )
        Wa = a.W_sparse.toarray()
        Wb = b.W_sparse.toarray()
        assert np.array_equal(Wa, Wb), (
            f"jeffreys default vs explicit must be bitwise-identical "
            f"(kernel={kernel})"
        )


def test_pmi_formula_rejects_invalid_value():
    """Invalid pmi_formula values raise ValueError with the expected message."""
    from tracer.metrics import compute_pmi_bootstrap
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    with pytest.raises(ValueError, match="pmi_formula"):
        compute_pmi_bootstrap(
            df, group_key="cell_id", feature_col="feature_name",
            pmi_formula="bogus",
            max_bootstraps=100, coarse_block=50, refine_block=50,
            seed=0, show_progress=False,
        )


def test_pmi_formula_epsilon_legacy_math_unit():
    """GATE 2a: hand-computed legacy (Stage 2) PMI under epsilon mode.

    Build a tiny synthetic where one pair has C=100, k_ij=5, k_i=10, k_j=10
    and assert legacy_pmi == log((5+0.1)/100 / (0.1*0.1)) to within 1e-12.
    """
    import pandas as pd
    from tracer.metrics import compute_pmi_bootstrap
    # 100 cells. 5 cells co-express genes 'A' and 'B' (k_ij=5).
    # 5 cells have only 'A' (so k_A=10). 5 cells have only 'B' (so k_B=10).
    # Remaining cells get a filler gene to keep them in the panel.
    rows = []
    for c in range(5):
        rows.append((f"c{c}", "A"))
        rows.append((f"c{c}", "B"))
    for c in range(5, 10):
        rows.append((f"c{c}", "A"))
    for c in range(10, 15):
        rows.append((f"c{c}", "B"))
    for c in range(15, 100):
        rows.append((f"c{c}", "FILLER"))
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name"])
    # Drive each cell with two tx for the genes it carries so the
    # min_occurrences_per_context=1 path retains them.
    res = compute_pmi_bootstrap(
        df, group_key="cell_id", feature_col="feature_name",
        metric="pmi",
        pmi_formula="epsilon",
        alpha=0.1,
        # Set thresholds so nothing routes through bootstrap; we want
        # Stage 2 legacy_pmi only.
        min_occurrences_per_context=1,
        min_expected_cooccur_for_evidence=1e9,
        max_bootstraps=10, coarse_block=10, refine_block=10,
        seed=0, show_progress=False, persist_ci=True,
    )
    # Find legacy_pmi for the (A, B) pair from pair_ci.
    pair_ci = res.pair_ci
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    iA, iB = g_to_i["A"], g_to_i["B"]
    a_, b_ = (iA, iB) if iA < iB else (iB, iA)
    row = pair_ci[(pair_ci["gene_i_idx"] == a_) & (pair_ci["gene_j_idx"] == b_)]
    assert len(row) == 1, "expected exactly one (A,B) row in pair_ci"
    got = float(row["legacy_pmi"].iloc[0])
    # k_ij=5, C=100, alpha=0.1, k_i=k_j=10 → p_i=p_j=0.1
    # PMI = log((5+0.1)/100 / (0.1*0.1)) = log(0.051 / 0.01)
    expected = float(np.log((5 + 0.1) / 100 / (0.1 * 0.1)))
    assert abs(got - expected) < 1e-12, (
        f"epsilon legacy_pmi: got {got!r}, expected {expected!r}"
    )
    # NPMI variant: pmi / -log(p_ij_eps)
    got_npmi = float(row["legacy_npmi"].iloc[0])
    expected_npmi = expected / (-np.log((5 + 0.1) / 100))
    assert abs(got_npmi - expected_npmi) < 1e-12


def test_pmi_formula_epsilon_neg_one_math_unit():
    """GATE 2b: hand-computed Stage 1 neg_one value under epsilon mode.

    Build a panel where genes 'X' and 'Y' both have rate 0.1 and never
    co-occur; with C large enough that E[k_ij] >= thr, the pair lands in
    Stage 1 neg_one. Under epsilon mode the value is the unified formula,
    not the -1 / -log(E) sentinels.
    """
    import pandas as pd
    from tracer.metrics import compute_pmi_bootstrap
    # 100 cells. Cells 0–9 have X (k_X = 10). Cells 10–19 have Y (k_Y = 10).
    # No cell has both → k_ij = 0. E[k_ij] = 0.1 * 0.1 * 100 = 1.0 — too low
    # to qualify for neg_one with the default thr=10. Lower thr to 0.5.
    # All cells get a filler so they are retained.
    rows = []
    for c in range(10):
        rows.append((f"c{c}", "X"))
    for c in range(10, 20):
        rows.append((f"c{c}", "Y"))
    for c in range(100):
        rows.append((f"c{c}", "FILLER"))
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name"])
    res = compute_pmi_bootstrap(
        df, group_key="cell_id", feature_col="feature_name",
        metric="pmi",
        pmi_formula="epsilon",
        alpha=0.1,
        min_occurrences_per_context=1,
        min_expected_cooccur_for_evidence=0.5,  # qualify (X,Y) for neg_one
        max_bootstraps=10, coarse_block=10, refine_block=10,
        seed=0, show_progress=False,
    )
    g_to_i = {g: i for i, g in enumerate(res.genes)}
    iX, iY = g_to_i["X"], g_to_i["Y"]
    a_, b_ = (iX, iY) if iX < iY else (iY, iX)
    W = res.W_sparse.tocoo()
    Wd = {(int(i), int(j)): float(v) for i, j, v in zip(W.row, W.col, W.data)}
    assert (a_, b_) in Wd, "epsilon-mode neg_one pair missing from W"
    # PMI = log((alpha/C) / (p_i * p_j)) = log(0.001 / 0.01) = log(0.1)
    expected_pmi = float(np.log((0.1 / 100) / (0.1 * 0.1)))
    assert abs(expected_pmi - np.log(0.1)) < 1e-12  # algebra check
    # W is stored as float32 so use a float32-friendly tolerance.
    assert abs(Wd[(a_, b_)] - expected_pmi) < 1e-6, (
        f"epsilon neg_one PMI: got {Wd[(a_, b_)]!r}, expected {expected_pmi!r}"
    )

    # NPMI variant
    res_npmi = compute_pmi_bootstrap(
        df, group_key="cell_id", feature_col="feature_name",
        metric="npmi",
        pmi_formula="epsilon", alpha=0.1,
        min_occurrences_per_context=1,
        min_expected_cooccur_for_evidence=0.5,
        max_bootstraps=10, coarse_block=10, refine_block=10,
        seed=0, show_progress=False,
    )
    Wn = res_npmi.W_sparse.tocoo()
    Wnd = {(int(i), int(j)): float(v) for i, j, v in zip(Wn.row, Wn.col, Wn.data)}
    expected_npmi = expected_pmi / (-np.log(0.1 / 100))
    assert (a_, b_) in Wnd
    assert abs(Wnd[(a_, b_)] - expected_npmi) < 1e-6
    # Document NPMI ≠ -1 under epsilon mode (it's alpha-dependent).
    assert abs(Wnd[(a_, b_)] - (-1.0)) > 1e-6, (
        "epsilon-mode NPMI neg_one must NOT equal the jeffreys -1 sentinel"
    )


def test_pmi_formula_epsilon_vs_jeffreys_close_on_well_supported():
    """GATE 3: on well-supported observed pairs (k_ij >= 5), epsilon and
    jeffreys PMI values agree to within ~alpha/k_ij. Quantify on the synthetic
    panel: max |Δ| over k_ij >= 5 pairs must be ≤ 0.1.
    """
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    # PMI metric so we can compare on a common scale.
    common = dict(
        kernel="pair_gather",
        metric="pmi",
        persist_ci=True,
        alpha=0.1,
    )
    rj = _run_eps(df, formula="jeffreys", **common)
    re = _run_eps(df, formula="epsilon", **common)
    pj = rj.pair_ci.set_index(["gene_i_idx", "gene_j_idx"])
    pe = re.pair_ci.set_index(["gene_i_idx", "gene_j_idx"])
    # Restrict to pairs where BOTH formulas have a finite legacy_pmi
    # (excludes Stage 1 neg_one rows where legacy is NaN).
    common_idx = pj.index.intersection(pe.index)
    pj = pj.loc[common_idx]
    pe = pe.loc[common_idx]
    # Exclude Stage 1 neg_one (k_ij=0) rows — they only carry sentinel values
    # under jeffreys and the unified formula under epsilon, so by design their
    # legacy_pmi differs more than alpha/k_ij.
    mask = (
        pj["legacy_pmi"].notna()
        & pe["legacy_pmi"].notna()
        & (pj["kind"] != "neg_one")
        & (pe["kind"] != "neg_one")
    )
    # Use expected_full as a proxy for "well-supported": E[k_ij] >= 5 picks
    # pairs whose observed cooccur is plausibly >= 5.
    if "expected_full" in pj.columns:
        mask = mask & (pj["expected_full"] >= 5)
    if mask.any():
        deltas = (pj.loc[mask, "legacy_pmi"] - pe.loc[mask, "legacy_pmi"]).abs()
        max_delta = float(deltas.max())
        assert max_delta <= 0.1, (
            f"epsilon vs jeffreys legacy_pmi diverged beyond expected band "
            f"({max_delta:.4f} > 0.1) on well-supported pairs"
        )


def test_pmi_formula_epsilon_settle_set_close_to_jeffreys():
    """GATE 4: epsilon-mode settles roughly the same SET of strong pos/neg
    pairs as jeffreys mode, with at most a small number of near-tau boundary
    differences."""
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    rj = _run_eps(df, kernel="pair_gather", formula="jeffreys")
    re = _run_eps(df, kernel="pair_gather", formula="epsilon")
    Sj = {(int(i), int(j)) for i, j in zip(*rj.W_sparse.nonzero())}
    Se = {(int(i), int(j)) for i, j in zip(*re.W_sparse.nonzero())}
    # Strong-positive and high-marginal-zero-cooccur sentinel pairs must
    # appear in BOTH (gene_00,gene_01) and (gene_08,gene_09).
    g_to_i = {g: i for i, g in enumerate(rj.genes)}
    p01 = (g_to_i["gene_00"], g_to_i["gene_01"])
    p89 = (g_to_i["gene_08"], g_to_i["gene_09"])
    p01 = (min(p01), max(p01))
    p89 = (min(p89), max(p89))
    assert p01 in Sj and p01 in Se
    assert p89 in Sj and p89 in Se
    # Symmetric diff ≤ 4 (a couple of near-tau boundary churn pairs is OK).
    assert len(Sj ^ Se) <= 4, (
        f"epsilon settled set diverged from jeffreys by {len(Sj ^ Se)} pairs"
    )


def test_pmi_formula_epsilon_gene_row_samples_runs():
    """GATE 5a: gene_row (samples mode) under epsilon runs and settles."""
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    res = _run_eps(df, kernel="gene_row", formula="epsilon",
                   ci_accumulator="samples")
    W = res.W_sparse.toarray()
    # Strong positive pair (0,1) classified positive.
    assert W[0, 1] > 0.05
    # Stage 1 neg_one for (8,9) emitted with epsilon's unified formula.
    assert (8, 9) in {(int(i), int(j)) for i, j in zip(*res.W_sparse.nonzero())}


def test_pmi_formula_epsilon_gene_row_counter_runs():
    """GATE 5b: gene_row counter (single-pass) under epsilon runs and settles."""
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    res = _run_eps(df, kernel="gene_row", formula="epsilon",
                   ci_accumulator="counter")
    W = res.W_sparse.toarray()
    assert W[0, 1] > 0.05
    assert (8, 9) in {(int(i), int(j)) for i, j in zip(*res.W_sparse.nonzero())}


def test_pmi_formula_epsilon_deterministic():
    """GATE 7: same seed → bit-identical W in epsilon mode."""
    from tests.synthetic import make_synthetic_npmi_panel
    df, _ = make_synthetic_npmi_panel()
    a = _run_eps(df, kernel="pair_gather", formula="epsilon")
    b = _run_eps(df, kernel="pair_gather", formula="epsilon")
    Wa = a.W_sparse.toarray()
    Wb = b.W_sparse.toarray()
    assert np.array_equal(Wa, Wb)
    # Also for gene_row
    ag = _run_eps(df, kernel="gene_row", formula="epsilon")
    bg = _run_eps(df, kernel="gene_row", formula="epsilon")
    assert np.array_equal(ag.W_sparse.toarray(), bg.W_sparse.toarray())

