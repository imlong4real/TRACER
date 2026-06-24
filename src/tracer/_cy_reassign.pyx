"""Cython kernel for Phase 1 Reassign-1c.

Replaces the per-parent Python loop in
``tests._pipeline_runner._reassign_nuclear_post_1c_etype`` with a flat
CSR-driven kernel that runs in parallel across parents via OpenMP
``prange``. The inner per-candidate-tx work — computing the
self-exclusion-aware mean PMI against the main seed and each partial
seed — is a tight C loop with no Python overhead.

Public entry: ``reassign_nuclear_post_1c_kernel``.
"""
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport isnan, NAN
from cython.parallel cimport prange

cnp.import_array()


cdef inline double _mean_pmi_excl_self(
    const float[:, ::1] W,
    int g,
    const int *seed,
    int n_seed,
) nogil:
    """Mean of W[g, s] over s in `seed`, excluding s==g and NaN entries.
    Returns NaN if no finite entry exists after self-exclusion.
    """
    cdef int i, s
    cdef int count = 0
    cdef double total = 0.0
    cdef float v
    for i in range(n_seed):
        s = seed[i]
        if s == g:
            continue
        v = W[g, s]
        # libc isnan; finite check via not-NaN (W is dense float32 PMI;
        # +/-inf shouldn't occur but treating them as non-finite is fine
        # — we just skip).
        if isnan(v):
            continue
        total += <double>v
        count += 1
    if count == 0:
        return NAN
    return total / count


cdef inline double _mean_pmi_sparse_excl_self(
    const int[::1] indptr,
    const int[::1] indices,
    const float[::1] data,
    int g,
    const int *seed,
    int n_seed,
) nogil:
    """Mean of *stored* Wsym[g, s] over s in `seed`, excluding s==g and
    structurally-absent (unstored) entries. Returns NaN if no stored
    entry survives self-exclusion.

    This is the sparse analog of :func:`_mean_pmi_excl_self`: a pair the
    bootstrap never stored (it fell below the evidence threshold) is
    *skipped*, not counted as 0 — the opposite of the coherence/stitching
    convention. ``neg_one`` mutual-exclusion sentinels are stored negative
    values, so they are found by the lookup and count as real anti-evidence.

    ``Wsym`` must be the symmetric CSR with column indices sorted within
    each row (``csr.sort_indices()``); the lookup is a per-seed binary
    search over row ``g``'s stored columns. Summing in seed order (rather
    than stored-column order) reproduces the dense kernel's floating-point
    accumulation exactly, so a fully-dense panel matches bit-for-bit.
    """
    cdef int i, s, lo, hi, mid, row_start, row_end
    cdef int count = 0
    cdef double total = 0.0
    row_start = indptr[g]
    row_end = indptr[g + 1]
    if row_end == row_start:
        return NAN
    for i in range(n_seed):
        s = seed[i]
        if s == g:
            # Self-exclusion is also automatic (the diagonal is never
            # stored), but skip explicitly to mirror the dense kernel.
            continue
        # Binary search for column `s` in indices[row_start:row_end].
        lo = row_start
        hi = row_end
        while lo < hi:
            mid = (lo + hi) >> 1
            if indices[mid] < s:
                lo = mid + 1
            else:
                hi = mid
        if lo < row_end and indices[lo] == s:
            total += <double>data[lo]
            count += 1
    if count == 0:
        return NAN
    return total / count


def reassign_nuclear_post_1c_kernel(
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    cnp.ndarray[cnp.int32_t, ndim=1] parent_cand_offsets,
    cnp.ndarray[cnp.int32_t, ndim=1] cand_gene_idx,
    cnp.ndarray[cnp.int32_t, ndim=1] parent_main_offsets,
    cnp.ndarray[cnp.int32_t, ndim=1] main_genes,
    cnp.ndarray[cnp.int32_t, ndim=1] parent_partial_offsets,
    cnp.ndarray[cnp.int32_t, ndim=1] partial_gene_offsets,
    cnp.ndarray[cnp.int32_t, ndim=1] partial_genes,
    double margin,
):
    """Per-candidate-tx best-partial decision.

    Parameters
    ----------
    W : (G, G) float32 contiguous PMI matrix.
    parent_cand_offsets : (n_parents+1,) int32
        CSR offsets into ``cand_gene_idx``. Candidate tx for parent ``p``
        live at positions ``cand_gene_idx[parent_cand_offsets[p] : parent_cand_offsets[p+1]]``.
    cand_gene_idx : (n_cands_total,) int32
        Gene index per candidate tx. ``-1`` marks "no valid gene"
        (candidate gets ``-1`` in the output).
    parent_main_offsets : (n_parents+1,) int32
        CSR offsets into ``main_genes``. Main entity's gene set for
        parent ``p`` is ``main_genes[parent_main_offsets[p] : parent_main_offsets[p+1]]``.
    main_genes : (n_main_genes_total,) int32
        Flat gene indices for each parent's main entity seed.
    parent_partial_offsets : (n_parents+1,) int32
        CSR offsets into the per-parent partial-enumeration arrays.
        Partials for parent ``p`` are indexed ``part_start = parent_partial_offsets[p]``
        through ``parent_partial_offsets[p+1]``.
    partial_gene_offsets : (n_partials_total+1,) int32
        CSR offsets into ``partial_genes``. Gene set for the partial
        at flat index ``pi`` is ``partial_genes[partial_gene_offsets[pi] : partial_gene_offsets[pi+1]]``.
    partial_genes : (n_partial_genes_total,) int32
        Flat gene indices for each partial's seed gene set.
    margin : double
        Admission margin: a candidate moves to partial ``k`` only if
        ``mean_pmi_partial_k > best_mean_so_far + margin``.

    Returns
    -------
    out_partial_local_idx : (n_cands_total,) int32
        For each candidate tx: the LOCAL partial index (0-based relative
        to the parent's partial range) of the best-fit partial, or -1
        if the candidate stays on the main entity. The caller maps
        local indices back to actual partial-entity labels.
    """
    cdef Py_ssize_t n_parents = parent_cand_offsets.shape[0] - 1
    cdef Py_ssize_t n_cands_total = cand_gene_idx.shape[0]

    # Output buffer (initialized to -1 = "no move").
    cdef cnp.ndarray[cnp.int32_t, ndim=1] out = np.full(
        n_cands_total, -1, dtype=np.int32
    )

    # Memoryviews for fast inner-loop access.
    cdef const float[:, ::1] W_view = W
    cdef const int[::1] cand_off = parent_cand_offsets
    cdef const int[::1] cand_g   = cand_gene_idx
    cdef const int[::1] main_off = parent_main_offsets
    cdef const int[::1] main_g   = main_genes
    cdef const int[::1] part_off = parent_partial_offsets
    cdef const int[::1] pg_off   = partial_gene_offsets
    cdef const int[::1] pg_idx   = partial_genes
    cdef int[::1] out_view = out

    cdef Py_ssize_t p, ci, pi, n_partials
    cdef int cand_start, cand_end, main_start, main_end
    cdef int part_start, part_end, pg_start, pg_end
    cdef int g
    cdef double mean_main, best_mean, mp
    cdef int best_p_local, p_local

    # prange over parents — each parent's candidates and partials are
    # independent, so this is embarrassingly parallel.
    for p in prange(n_parents, nogil=True, schedule="dynamic"):
        cand_start = cand_off[p]
        cand_end = cand_off[p + 1]
        main_start = main_off[p]
        main_end = main_off[p + 1]
        part_start = part_off[p]
        part_end = part_off[p + 1]
        n_partials = part_end - part_start

        if n_partials == 0:
            # No partials → no possible move.
            for ci in range(cand_start, cand_end):
                out_view[ci] = -1
            continue

        for ci in range(cand_start, cand_end):
            g = cand_g[ci]
            if g < 0:
                out_view[ci] = -1
                continue

            mean_main = _mean_pmi_excl_self(
                W_view, g,
                &main_g[main_start], main_end - main_start,
            )
            if isnan(mean_main):
                out_view[ci] = -1
                continue

            best_mean = mean_main
            best_p_local = -1

            for p_local in range(<int>n_partials):
                pi = part_start + p_local
                pg_start = pg_off[pi]
                pg_end = pg_off[pi + 1]
                if pg_end == pg_start:
                    continue
                mp = _mean_pmi_excl_self(
                    W_view, g,
                    &pg_idx[pg_start], pg_end - pg_start,
                )
                if (not isnan(mp)) and mp > best_mean + margin:
                    best_mean = mp
                    best_p_local = p_local

            out_view[ci] = best_p_local

    return out


def reassign_nuclear_post_1c_kernel_sparse(
    cnp.ndarray[cnp.int32_t, ndim=1] W_indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] W_indices,
    cnp.ndarray[cnp.float32_t, ndim=1] W_data,
    cnp.ndarray[cnp.int32_t, ndim=1] parent_cand_offsets,
    cnp.ndarray[cnp.int32_t, ndim=1] cand_gene_idx,
    cnp.ndarray[cnp.int32_t, ndim=1] parent_main_offsets,
    cnp.ndarray[cnp.int32_t, ndim=1] main_genes,
    cnp.ndarray[cnp.int32_t, ndim=1] parent_partial_offsets,
    cnp.ndarray[cnp.int32_t, ndim=1] partial_gene_offsets,
    cnp.ndarray[cnp.int32_t, ndim=1] partial_genes,
    double margin,
):
    """Sparse-PMI variant of :func:`reassign_nuclear_post_1c_kernel`.

    Identical per-candidate best-partial decision logic, but the gene-fit
    means are computed against a *symmetric sparse* CSR PMI panel instead
    of a dense ``(G, G)`` matrix. This is what unlocks whole-transcriptome
    (~18-20k gene) panels, where a dense float32 ``(G, G)`` would be ~1.6 GB.

    Structurally-absent pairs are skipped (treated as indeterminate / NaN),
    matching the dense path's NaN-skip — NOT counted as 0. See
    :func:`_mean_pmi_sparse_excl_self`.

    Parameters
    ----------
    W_indptr, W_indices, W_data : CSR arrays of the *symmetric* PMI panel.
        Build via COO-stack symmetrization of the bootstrap's
        upper-triangle ``W_sparse`` (``W + W.T`` is unsafe — scipy's add
        eliminates explicit zeros, dropping observed PMI == 0.0 entries).
        ``W_indices`` must be column-sorted within each row
        (``csr.sort_indices()``). Never call ``eliminate_zeros()``: an
        observed PMI of exactly 0.0 must remain a stored, counted entry.
    All other parameters and the return value match
    :func:`reassign_nuclear_post_1c_kernel`.
    """
    cdef Py_ssize_t n_parents = parent_cand_offsets.shape[0] - 1
    cdef Py_ssize_t n_cands_total = cand_gene_idx.shape[0]

    cdef cnp.ndarray[cnp.int32_t, ndim=1] out = np.full(
        n_cands_total, -1, dtype=np.int32
    )

    cdef const int[::1] W_ip   = W_indptr
    cdef const int[::1] W_ind  = W_indices
    cdef const float[::1] W_dat = W_data
    cdef const int[::1] cand_off = parent_cand_offsets
    cdef const int[::1] cand_g   = cand_gene_idx
    cdef const int[::1] main_off = parent_main_offsets
    cdef const int[::1] main_g   = main_genes
    cdef const int[::1] part_off = parent_partial_offsets
    cdef const int[::1] pg_off   = partial_gene_offsets
    cdef const int[::1] pg_idx   = partial_genes
    cdef int[::1] out_view = out

    cdef Py_ssize_t p, ci, pi, n_partials
    cdef int cand_start, cand_end, main_start, main_end
    cdef int part_start, part_end, pg_start, pg_end
    cdef int g
    cdef double mean_main, best_mean, mp
    cdef int best_p_local, p_local

    for p in prange(n_parents, nogil=True, schedule="dynamic"):
        cand_start = cand_off[p]
        cand_end = cand_off[p + 1]
        main_start = main_off[p]
        main_end = main_off[p + 1]
        part_start = part_off[p]
        part_end = part_off[p + 1]
        n_partials = part_end - part_start

        if n_partials == 0:
            for ci in range(cand_start, cand_end):
                out_view[ci] = -1
            continue

        for ci in range(cand_start, cand_end):
            g = cand_g[ci]
            if g < 0:
                out_view[ci] = -1
                continue

            mean_main = _mean_pmi_sparse_excl_self(
                W_ip, W_ind, W_dat, g,
                &main_g[main_start], main_end - main_start,
            )
            if isnan(mean_main):
                out_view[ci] = -1
                continue

            best_mean = mean_main
            best_p_local = -1

            for p_local in range(<int>n_partials):
                pi = part_start + p_local
                pg_start = pg_off[pi]
                pg_end = pg_off[pi + 1]
                if pg_end == pg_start:
                    continue
                mp = _mean_pmi_sparse_excl_self(
                    W_ip, W_ind, W_dat, g,
                    &pg_idx[pg_start], pg_end - pg_start,
                )
                if (not isnan(mp)) and mp > best_mean + margin:
                    best_mean = mp
                    best_p_local = p_local

            out_view[ci] = best_p_local

    return out
