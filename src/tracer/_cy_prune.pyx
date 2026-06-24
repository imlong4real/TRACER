# cython: boundscheck=False, wraparound=False, nonecheck=False
# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as cnp

from libc.stdlib cimport malloc, free
from cython.parallel cimport prange
cimport openmp


def prune_cells(list g_lists, cnp.ndarray[cnp.float32_t, ndim=2] W, double threshold):
    """
    Bulk prune helper callable from Python.

    Parameters
    ----------
    g_lists : list
        List of 1D integer numpy arrays (gene indices) or None entries.
    W : ndarray[float32, 2D]
        Full NPMI matrix.
    threshold : float
        NPMI threshold.

    Returns
    -------
    list
        List of removed gene index lists (python lists) or []/None.
    """
    cdef Py_ssize_t n
    cdef list out
    cdef Py_ssize_t idx
    cdef object arr

    n = len(g_lists)
    out = [None] * n
    for idx in range(n):
        g = g_lists[idx]
        if g is None:
            out[idx] = None
            continue
        arr = np.asarray(g, dtype=np.int32)
        if arr.size <= 1:
            out[idx] = []
            continue
        out[idx] = prune_single(arr, W, threshold)

    return out


def prune_single(cnp.ndarray[cnp.int32_t, ndim=1] g_local, cnp.ndarray[cnp.float32_t, ndim=2] W, double threshold):
    """Prune a single gene list. Returns removed gene indices as Python list."""
    cdef int k
    cdef int i, j
    cdef int gi, gj
    cdef int active_count
    cdef int maxc, argmax
    cdef float val

    cdef object active
    cdef object bad
    cdef object bad_counts

    cdef cnp.int32_t[:] gids
    cdef cnp.float32_t[:, :] Wv
    cdef cnp.uint8_t[:, :] bad_mv
    cdef cnp.uint8_t[:] active_mv
    cdef cnp.int32_t[:] badc_mv

    k = g_local.shape[0]
    # create local numpy arrays for masks/counts (fast with memoryviews)
    active = np.ones(k, dtype=np.uint8)
    bad = np.zeros((k, k), dtype=np.uint8)
    bad_counts = np.zeros(k, dtype=np.int32)

    gids = g_local
    Wv = W
    bad_mv = bad
    active_mv = active
    badc_mv = bad_counts

    # compute bad matrix and counts
    for i in range(k):
        gi = int(gids[i])
        for j in range(k):
            if i == j:
                continue
            gj = int(gids[j])
            val = Wv[gi, gj]
            # NaN check
            if val != val:
                continue
            if val < threshold:
                bad_mv[i, j] = 1
                badc_mv[i] += 1

    active_count = k
    while active_count > 1:
        # find active index with max bad_counts
        maxc = -1
        argmax = -1
        for i in range(k):
            if active_mv[i]:
                if badc_mv[i] > maxc:
                    maxc = badc_mv[i]
                    argmax = i

        if maxc <= 0:
            break

        # remove argmax
        active_mv[argmax] = 0
        active_count -= 1

        # decrement neighbors' counts
        for j in range(k):
            if active_mv[j] and bad_mv[argmax, j]:
                badc_mv[j] -= 1
        badc_mv[argmax] = 0

    # collect removed genes (those inactive)
    cdef list removed = []
    for i in range(k):
        if not bool(active_mv[i]):
            removed.append(int(gids[i]))

    return removed


cdef inline bint _wget(
    cnp.float32_t[:, :] W,
    const int[::1] W_indptr,
    const int[::1] W_indices,
    const float[::1] W_data,
    int use_sparse,
    int row,
    int col,
    float *out,
) nogil:
    """Unified PMI accessor for the nuclear-seed prune. Returns True and
    writes ``W[row, col]`` to ``out`` when the pair is OBSERVED; returns
    False when it should be SKIPPED.

    Two backends, identical "skip the unobserved" semantics:
      * dense  (``use_sparse == 0``): observed == not-NaN. ``W`` is the
        dense float32 PMI matrix with NaN for unobserved pairs.
      * sparse (``use_sparse == 1``): observed == structurally stored in
        the symmetric CSR. A pair the bootstrap never stored (below the
        evidence threshold) is absent and skipped — NOT treated as 0
        (that is the coherence convention and the wrong one for seed
        coherence / gene-fit). ``neg_one`` mutual-exclusion sentinels are
        stored negatives, so they are found and count as anti-evidence.
        ``W_indices`` must be column-sorted within each row.

    On a fully-observed (dense) panel the two backends return the same
    value for every pair, so the kernel's results match bit-for-bit.
    """
    cdef int lo, hi, row_end, mid
    cdef float v
    if use_sparse:
        lo = W_indptr[row]
        row_end = W_indptr[row + 1]
        hi = row_end
        while lo < hi:
            mid = (lo + hi) >> 1
            if W_indices[mid] < col:
                lo = mid + 1
            else:
                hi = mid
        if lo < row_end and W_indices[lo] == col:
            out[0] = W_data[lo]
            return True
        return False
    else:
        v = W[row, col]
        if v == v:  # not NaN
            out[0] = v
            return True
        return False


cdef inline double _mean_internal_pmi(
    cnp.int32_t[:] genes,
    int n_genes,
    cnp.float32_t[:, :] W,
    const int[::1] W_indptr,
    const int[::1] W_indices,
    const float[::1] W_data,
    int use_sparse,
) nogil:
    """Mean PMI over off-diagonal pairs within a gene set. Returns
    +inf when the set has <2 genes (caller treats as "trivially
    coherent": no pairs means no internal incoherence). Unobserved
    pairs are skipped (see :func:`_wget`)."""
    cdef int i, j
    cdef double total = 0.0
    cdef int count = 0
    cdef float v
    if n_genes < 2:
        return 1e30
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            if _wget(W, W_indptr, W_indices, W_data, use_sparse,
                     genes[i], genes[j], &v):
                total += v
                count += 1
    if count == 0:
        return 0.0
    return total / count


cdef inline int _mean_pmi_test(
    int gene_idx,
    cnp.int32_t[:] seed,
    int seed_len,
    cnp.float32_t[:, :] W,
    const int[::1] W_indptr,
    const int[::1] W_indices,
    const float[::1] W_data,
    int use_sparse,
    double threshold,
) nogil:
    """Returns 1 if mean PMI(gene_idx, seed) >= threshold (unobserved
    pairs skipped), else 0. Returns 0 if no observed pair or empty seed."""
    cdef int j
    cdef double total = 0.0
    cdef int count = 0
    cdef float v
    if seed_len == 0:
        return 0
    for j in range(seed_len):
        if _wget(W, W_indptr, W_indices, W_data, use_sparse,
                 gene_idx, seed[j], &v):
            total += v
            count += 1
    if count == 0:
        return 0
    if (total / count) >= threshold:
        return 1
    return 0


cdef cnp.ndarray _greedy_prune_to_retained(
    cnp.int32_t[:] g_local,
    cnp.int32_t[:] tx_counts,
    cnp.float32_t[:, :] W,
    const int[::1] W_indptr,
    const int[::1] W_indices,
    const float[::1] W_data,
    int use_sparse,
    double threshold,
):
    """Greedy bad-edge prune with TX-WEIGHTED conflict scoring.

    For each gene i, compute score[i] = sum_j (bad[i,j] * tx_counts[j])
    — i.e., total tx of conflicting other-genes. Strip the gene with
    max score. Safeguard: a gene with own tx count > score is
    PROTECTED from being stripped (its own evidence majority-outweighs
    its conflicts). Returns retained gene indices as int32 ndarray.

    Unobserved gene pairs are skipped (never marked bad), via
    :func:`_wget` — same semantics under the dense and sparse backends.
    """
    cdef int k = g_local.shape[0]
    if k <= 1:
        return np.asarray(g_local).copy()
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] active = np.ones(k, dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] bad = np.zeros((k, k), dtype=np.uint8)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] bad_score = np.zeros(k, dtype=np.int32)
    cdef cnp.uint8_t[:] active_mv = active
    cdef cnp.uint8_t[:, :] bad_mv = bad
    cdef cnp.int32_t[:] badscore_mv = bad_score
    cdef int i, j, gi, gj, active_count, maxc, argmax
    cdef float val
    # Build bad-edge matrix and tx-weighted score per gene.
    for i in range(k):
        gi = int(g_local[i])
        for j in range(k):
            if i == j:
                continue
            gj = int(g_local[j])
            if not _wget(W, W_indptr, W_indices, W_data, use_sparse,
                         gi, gj, &val):
                continue
            if val < threshold:
                bad_mv[i, j] = 1
                badscore_mv[i] += tx_counts[j]
    active_count = k
    while active_count > 1:
        # Find gene with max tx-weighted bad-score among ACTIVE genes
        # NOT protected by the own-tx safeguard.
        maxc = -1
        argmax = -1
        for i in range(k):
            if not active_mv[i]:
                continue
            # Safeguard: own evidence > conflict-tx → protected, skip.
            if tx_counts[i] > badscore_mv[i]:
                continue
            if badscore_mv[i] > maxc:
                maxc = badscore_mv[i]
                argmax = i
        if maxc <= 0 or argmax < 0:
            break
        active_mv[argmax] = 0
        active_count -= 1
        # On strip: every neighbour j with bad[argmax,j]=1 loses
        # tx_counts[argmax] worth of conflict from their score.
        for j in range(k):
            if active_mv[j] and bad_mv[argmax, j]:
                badscore_mv[j] -= tx_counts[argmax]
        badscore_mv[argmax] = 0
    # collect retained (active) gene indices
    cdef int n_kept = 0
    for i in range(k):
        if active_mv[i]:
            n_kept += 1
    cdef cnp.ndarray[cnp.int32_t, ndim=1] kept = np.empty(n_kept, dtype=np.int32)
    cdef cnp.int32_t[:] kept_mv = kept
    cdef int idx = 0
    for i in range(k):
        if active_mv[i]:
            kept_mv[idx] = g_local[i]
            idx += 1
    return kept


cdef cnp.ndarray _prune_cells_nuclear_seed_core(
    list cell_tx_idx_lists,
    cnp.int32_t[:] tx_gene_mv,
    cnp.uint8_t[:] tx_nuc_mv,
    cnp.float32_t[:, :] W_mv,
    const int[::1] W_indptr,
    const int[::1] W_indices,
    const float[::1] W_data,
    int use_sparse,
    int n_tx,
    double threshold,
    int min_nuclear_genes,
    int skip_phase_1c,
    double seed_coherence_floor,
    int nuclear_only_admit,
    int tx_weighted,
    int veto_mode,
    double min_admit_threshold,
    double mean_admit_threshold,
    double aggregator_percentile,
    double real_signal_threshold,
    double neg_npmi_threshold,
):
    """Shared orchestration for the dense and sparse nuclear-seed prune.

    All PMI access goes through :func:`_wget`, which encodes the same
    "skip the unobserved pair" semantics for both backends, so the dense
    and sparse paths produce identical per-tx codes on a fully-observed
    panel (bit-for-bit) and differ only when the sparse panel genuinely
    omits pairs (then sparse SKIPS, matching dense-with-NaN). See the
    public :func:`prune_cells_nuclear_seed` /
    :func:`prune_cells_nuclear_seed_sparse` wrappers.
    """
    cdef int n_cells = len(cell_tx_idx_lists)
    cdef cnp.ndarray[cnp.int8_t, ndim=1] out = np.full(n_tx, 2, dtype=np.int8)
    cdef cnp.int8_t[:] out_mv = out

    cdef Py_ssize_t cidx
    cdef cnp.ndarray[cnp.int32_t, ndim=1] tx_inds_arr
    cdef cnp.int32_t[:] tx_inds_mv
    cdef int n_cell_tx, ti, tx_row, g, n_unique_nuc, fitted, n_unique_all

    # Reusable scratch buffers
    cdef cnp.ndarray[cnp.int32_t, ndim=1] uniq_nuc
    cdef cnp.ndarray[cnp.int32_t, ndim=1] uniq_all
    cdef cnp.ndarray[cnp.int32_t, ndim=1] seed
    cdef cnp.int32_t[:] seed_mv
    cdef int seed_len
    cdef cnp.ndarray[cnp.int32_t, ndim=1] uniq_rej_nuc
    cdef cnp.ndarray[cnp.int32_t, ndim=1] sub_seed
    cdef cnp.int32_t[:] sub_seed_mv
    cdef int sub_seed_len

    # Scratch float buffer for the hybrid admission gate's percentile
    # aggregator. Sized at the PMI panel's row dimension. Allocated once,
    # reused across all cell loops. Only meaningful when veto_mode != 1
    # (legacy mean); harmless otherwise.
    cdef int panel_n_genes
    if use_sparse:
        panel_n_genes = W_indptr.shape[0] - 1
    else:
        panel_n_genes = W_mv.shape[0]
    cdef cnp.ndarray[cnp.float32_t, ndim=1] pmi_buf_arr = np.zeros(
        panel_n_genes if panel_n_genes > 0 else 1, dtype=np.float32
    )
    cdef cnp.float32_t[:] pmi_buf_mv = pmi_buf_arr

    # Seed-membership lookup via a compact "in" check using sorted seed.
    # For typical seed sizes (<50 genes), linear scan is faster than a
    # set; use a small inline helper.

    for cidx in range(n_cells):
        tx_inds_arr = cell_tx_idx_lists[cidx]
        tx_inds_mv = tx_inds_arr
        n_cell_tx = tx_inds_arr.shape[0]
        if n_cell_tx == 0:
            continue

        # Collect unique nuclear gene indices for this cell.
        # (Python-level numpy ops here are fine: per-cell, not per-tx.)
        nuc_genes = []
        for ti in range(n_cell_tx):
            tx_row = tx_inds_mv[ti]
            if tx_nuc_mv[tx_row]:
                g = tx_gene_mv[tx_row]
                if g >= 0:
                    nuc_genes.append(g)
        uniq_nuc = np.unique(np.asarray(nuc_genes, dtype=np.int32))
        n_unique_nuc = uniq_nuc.shape[0]

        if n_unique_nuc < min_nuclear_genes:
            # Fallback: prune on whole-cell gene set (matches the Python
            # reference impl). Phase 1b/1c still run on the resulting
            # seed.
            all_genes = []
            for ti in range(n_cell_tx):
                tx_row = tx_inds_mv[ti]
                g = tx_gene_mv[tx_row]
                if g >= 0:
                    all_genes.append(g)
            uniq_all_arr = np.asarray(all_genes, dtype=np.int32)
            uniq_all = np.unique(uniq_all_arr)
            n_unique_all = uniq_all.shape[0]
            if n_unique_all == 0:
                continue
            # Whole-cell tx counts: all tx contribute (nuc + cyto).
            tx_counts_all = np.zeros(n_unique_all, dtype=np.int32)
            for ti in range(n_unique_all):
                tx_counts_all[ti] = int(np.sum(uniq_all_arr == uniq_all[ti]))
            if not tx_weighted:
                tx_counts_all = np.ones(n_unique_all, dtype=np.int32)
            seed = _greedy_prune_to_retained(
                uniq_all, tx_counts_all, W_mv,
                W_indptr, W_indices, W_data, use_sparse, threshold)
        else:
            # ---- Phase 1a: nuclear seed (tx-weighted) ----
            # Per-gene nuclear tx counts for the tx-weighted greedy.
            nuc_arr = np.asarray(nuc_genes, dtype=np.int32)
            tx_counts_nuc = np.zeros(n_unique_nuc, dtype=np.int32)
            for ti in range(n_unique_nuc):
                tx_counts_nuc[ti] = int(np.sum(nuc_arr == uniq_nuc[ti]))
            if not tx_weighted:
                tx_counts_nuc = np.ones(n_unique_nuc, dtype=np.int32)
            seed = _greedy_prune_to_retained(
                uniq_nuc, tx_counts_nuc, W_mv,
                W_indptr, W_indices, W_data, use_sparse, threshold)
        seed_mv = seed
        seed_len = seed.shape[0]
        if seed_len == 0:
            # No seed found; all tx → unassigned (code 2 already default)
            continue

        # Coherence floor on primary seed: reject seeds whose internal
        # mean PMI is below the floor (avoids accepting a clique that
        # is merely "non-conflicting" but biologically degenerate).
        if _mean_internal_pmi(
                seed_mv, seed_len, W_mv,
                W_indptr, W_indices, W_data, use_sparse) < seed_coherence_floor:
            continue

        # ---- Phase 1b: per-tx admit by mean PMI to seed ----
        # rejected accumulator (Python list, per-cell, OK overhead).
        # When nuclear_only_admit is set, cytoplasmic tx are not eligible
        # for admission to main here — they stay unassigned (code 2)
        # for downstream Group/Rescue/Stitch to route by spatial+gene
        # proximity. Identity is established exclusively from nuclear tx.
        rejected_rows = []
        for ti in range(n_cell_tx):
            tx_row = tx_inds_mv[ti]
            g = tx_gene_mv[tx_row]
            if g < 0:
                # No gene index — leave as unassigned (already default 2)
                rejected_rows.append(tx_row)
                continue
            if nuclear_only_admit and not tx_nuc_mv[tx_row]:
                # Cytoplasmic tx skipped — stays unassigned for Rescue.
                continue
            # Check if g is in seed (linear scan; seed_len typically tiny)
            fitted = 0
            for j in range(seed_len):
                if seed_mv[j] == g:
                    fitted = 1
                    break
            if fitted == 0:
                fitted = _admission_test(
                    g,
                    &seed_mv[0],
                    seed_len,
                    W_mv,
                    W_indptr, W_indices, W_data, use_sparse,
                    &pmi_buf_mv[0],
                    veto_mode,
                    mean_admit_threshold if veto_mode != 1 else threshold,
                    min_admit_threshold,
                    aggregator_percentile,
                    real_signal_threshold,
                    neg_npmi_threshold,
                    0,  # small_entity_guard_n unused at Phase 1b
                    1,  # legacy_mean_test = 1 (Phase 1b mean uses `>=`)
                )
            if fitted:
                out_mv[tx_row] = 0  # main
            else:
                rejected_rows.append(tx_row)

        if skip_phase_1c or len(rejected_rows) == 0:
            continue

        # ---- Phase 1c: recursive prune on rejected tx's nuclear genes ----
        rej_nuc = []
        for tx_row in rejected_rows:
            if tx_nuc_mv[tx_row]:
                g = tx_gene_mv[tx_row]
                if g >= 0:
                    rej_nuc.append(g)
        if len(rej_nuc) == 0:
            continue
        rej_nuc_arr = np.asarray(rej_nuc, dtype=np.int32)
        uniq_rej_nuc = np.unique(rej_nuc_arr)
        if uniq_rej_nuc.shape[0] < min_nuclear_genes:
            continue

        # Tx-weighted prune for sub-seed too.
        n_uniq_rej = uniq_rej_nuc.shape[0]
        tx_counts_rej = np.zeros(n_uniq_rej, dtype=np.int32)
        for ti in range(n_uniq_rej):
            tx_counts_rej[ti] = int(np.sum(rej_nuc_arr == uniq_rej_nuc[ti]))
        if not tx_weighted:
            tx_counts_rej = np.ones(n_uniq_rej, dtype=np.int32)
        sub_seed = _greedy_prune_to_retained(
            uniq_rej_nuc, tx_counts_rej, W_mv,
            W_indptr, W_indices, W_data, use_sparse, threshold)
        sub_seed_mv = sub_seed
        sub_seed_len = sub_seed.shape[0]
        if sub_seed_len == 0:
            continue

        # Coherence floor on sub-seed: reject if internal mean PMI is
        # below the floor — prevents degenerate "any non-conflicting
        # clique" sub-seeds from forming spurious partials. Rest-pile
        # tx stay unassigned (code 2, already default) for downstream
        # Rescue to route based on neighbour gene composition.
        if _mean_internal_pmi(
                sub_seed_mv, sub_seed_len, W_mv,
                W_indptr, W_indices, W_data, use_sparse) < seed_coherence_floor:
            continue

        # Re-test rejected tx against sub-seed. When nuclear_only_admit
        # is set, cytoplasmic rejected tx are not eligible for partial
        # admission either — they stay unassigned for Rescue.
        for tx_row in rejected_rows:
            g = tx_gene_mv[tx_row]
            if g < 0:
                continue
            if nuclear_only_admit and not tx_nuc_mv[tx_row]:
                continue
            fitted = 0
            for j in range(sub_seed_len):
                if sub_seed_mv[j] == g:
                    fitted = 1
                    break
            if fitted == 0:
                fitted = _admission_test(
                    g,
                    &sub_seed_mv[0],
                    sub_seed_len,
                    W_mv,
                    W_indptr, W_indices, W_data, use_sparse,
                    &pmi_buf_mv[0],
                    veto_mode,
                    mean_admit_threshold if veto_mode != 1 else threshold,
                    min_admit_threshold,
                    aggregator_percentile,
                    real_signal_threshold,
                    neg_npmi_threshold,
                    0,
                    1,
                )
            if fitted:
                out_mv[tx_row] = 1  # partial
            # else: stays 2 (unassigned) — already default

    return out


def prune_cells_nuclear_seed(
    list cell_tx_idx_lists,
    cnp.ndarray[cnp.int32_t, ndim=1] tx_gene_idx,
    cnp.ndarray[cnp.uint8_t, ndim=1] tx_is_nuclear,
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    double threshold,
    int min_nuclear_genes,
    int skip_phase_1c,
    double seed_coherence_floor=-1e30,
    int nuclear_only_admit=0,
    int tx_weighted=1,
    int veto_mode=1,
    double min_admit_threshold=0.0,
    double mean_admit_threshold=0.2,
    double aggregator_percentile=25.0,
    double real_signal_threshold=0.0,
    double neg_npmi_threshold=-0.2,
):
    """Batch nuclear-seed prune over many cells — DENSE PMI backend.

    Parameters
    ----------
    cell_tx_idx_lists : list of np.int32 1D arrays
        Per-cell row-index arrays (positions in tx_gene_idx / tx_is_nuclear).
    tx_gene_idx : int32 ndarray
        Per-tx gene-vocabulary index. -1 indicates "no gene index" (skip).
    tx_is_nuclear : uint8 ndarray
        Per-tx nuclear flag (0/1).
    W : float32 2D ndarray
        Dense PMI matrix. Unobserved pairs are NaN and skipped (note: the
        ``prune_transcripts_nuclear_seed`` wrapper historically fills NaN
        with 0 via ``nan_fill``; pass ``nan_fill=None`` for NaN-skip
        parity with the sparse backend).
    threshold : float
        Admission threshold for mean PMI to seed (also bad-edge threshold
        for the greedy prune).
    min_nuclear_genes : int
        Skip Phase 1a / 1c if a cell has fewer than this many unique nuclear genes.
    skip_phase_1c : int
        If non-zero, skip Phase 1c (rejected tx → unassigned directly).
    seed_coherence_floor : float
        Minimum mean internal PMI required for a seed (1a) or sub-seed
        (1c) to be accepted. Seeds below this floor are rejected:
        for the primary seed, all that cell's tx → unassigned; for the
        sub-seed, all rest-pile tx → unassigned (no partial formed).
        Default -1e30 disables the check (back-compat).
    nuclear_only_admit : int
        If non-zero, restrict 1b admission and 1c re-test to NUCLEAR
        tx only — cytoplasmic tx leave Phase 1 as unassigned (code 2)
        regardless of gene-fit. Group/Rescue/Stitch then route them
        downstream by spatial+gene proximity. Establishes per-cell
        identity exclusively from spatially-trusted nuclear tx.
        Default 0 disables (back-compat: cyto admitted by gene-fit).

    Returns
    -------
    np.int8 ndarray
        Per-tx assignment code: 0 = main (parent cell), 1 = partial,
        2 = unassigned, 3 = fallback-needed (cell had insufficient
        nuclear genes; caller should fall back to whole-cell prune).
    """
    # Dummy CSR arrays (never indexed when use_sparse == 0).
    cdef cnp.ndarray[cnp.int32_t, ndim=1] _ip = np.zeros(1, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] _ix = np.zeros(1, dtype=np.int32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] _dt = np.zeros(1, dtype=np.float32)
    return _prune_cells_nuclear_seed_core(
        cell_tx_idx_lists, tx_gene_idx, tx_is_nuclear, W,
        _ip, _ix, _dt, 0,
        tx_gene_idx.shape[0],
        threshold, min_nuclear_genes, skip_phase_1c,
        seed_coherence_floor, nuclear_only_admit, tx_weighted,
        veto_mode, min_admit_threshold, mean_admit_threshold,
        aggregator_percentile, real_signal_threshold, neg_npmi_threshold,
    )


def prune_cells_nuclear_seed_sparse(
    list cell_tx_idx_lists,
    cnp.ndarray[cnp.int32_t, ndim=1] tx_gene_idx,
    cnp.ndarray[cnp.uint8_t, ndim=1] tx_is_nuclear,
    cnp.ndarray[cnp.int32_t, ndim=1] W_indptr,
    cnp.ndarray[cnp.int32_t, ndim=1] W_indices,
    cnp.ndarray[cnp.float32_t, ndim=1] W_data,
    double threshold,
    int min_nuclear_genes,
    int skip_phase_1c,
    double seed_coherence_floor=-1e30,
    int nuclear_only_admit=0,
    int tx_weighted=1,
    int veto_mode=1,
    double min_admit_threshold=0.0,
    double mean_admit_threshold=0.2,
    double aggregator_percentile=25.0,
    double real_signal_threshold=0.0,
    double neg_npmi_threshold=-0.2,
):
    """Batch nuclear-seed prune over many cells — SPARSE CSR PMI backend.

    Same algorithm and per-tx output codes as
    :func:`prune_cells_nuclear_seed`, but the PMI panel is a symmetric
    sparse CSR (built from the bootstrap's upper-triangle ``W_sparse``)
    instead of a dense ``(G, G)`` matrix — the float32 dense matrix is
    ~1.6 GB at 20k genes and OOMs the prune around 15k. This unblocks
    whole-transcriptome panels.

    Structurally-absent pairs are SKIPPED (not counted as 0). Build the
    CSR by COO-stacking the upper triangle (``W + W.T`` is unsafe: scipy's
    sparse add eliminates explicit zeros, dropping observed PMI == 0.0),
    ``sort_indices()``, and never ``eliminate_zeros()``. ``W_indices``
    must be column-sorted within each row.
    """
    # Dummy dense matrix (never indexed when use_sparse == 1).
    cdef cnp.ndarray[cnp.float32_t, ndim=2] _W = np.zeros((1, 1), dtype=np.float32)
    return _prune_cells_nuclear_seed_core(
        cell_tx_idx_lists, tx_gene_idx, tx_is_nuclear, _W,
        W_indptr, W_indices, W_data, 1,
        tx_gene_idx.shape[0],
        threshold, min_nuclear_genes, skip_phase_1c,
        seed_coherence_floor, nuclear_only_admit, tx_weighted,
        veto_mode, min_admit_threshold, mean_admit_threshold,
        aggregator_percentile, real_signal_threshold, neg_npmi_threshold,
    )


def top_k_positive_clique_per_entity(
    list gene_id_lists,
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    int K,
    double pos_threshold,
):
    """For each entity, identify its top-K signature genes — the K genes
    with highest sum of positive PMI to other genes in the same entity.

    Returns int32 ndarray of shape (n_entities, K). Unused slots are -1
    (e.g., when an entity has fewer than K genes).

    Used as a cheap pre-filter signature: two cells whose top-clique
    cross-PMI block contains a strong-negative entry are very unlikely
    to have positive ΔC at the full-pair level.
    """
    cdef int n_ent = len(gene_id_lists)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] out = np.full((n_ent, K), -1, dtype=np.int32)
    cdef cnp.int32_t[:, :] out_mv = out
    cdef cnp.float32_t[:, :] W_mv = W
    cdef int e, n_genes, i, j, gi, gj, k
    cdef float v
    cdef cnp.ndarray[cnp.int32_t, ndim=1] g_arr
    cdef cnp.int32_t[:] g_mv
    cdef cnp.ndarray[cnp.float64_t, ndim=1] scores
    cdef cnp.float64_t[:] scores_mv
    cdef cnp.ndarray[cnp.int64_t, ndim=1] order

    for e in range(n_ent):
        obj = gene_id_lists[e]
        if obj is None:
            continue
        g_arr = np.asarray(obj, dtype=np.int32)
        n_genes = g_arr.shape[0]
        if n_genes == 0:
            continue
        if n_genes <= K:
            g_mv = g_arr
            for i in range(n_genes):
                out_mv[e, i] = g_mv[i]
            continue
        scores = np.zeros(n_genes, dtype=np.float64)
        scores_mv = scores
        g_mv = g_arr
        for i in range(n_genes):
            gi = g_mv[i]
            for j in range(n_genes):
                if i == j:
                    continue
                gj = g_mv[j]
                v = W_mv[gi, gj]
                if v != v:
                    continue
                if v > pos_threshold:
                    scores_mv[i] += v
        # Top K by score (np.argsort returns ascending; take last K)
        order = np.argsort(scores)
        for k in range(K):
            out_mv[e, k] = g_mv[order[n_genes - 1 - k]]
    return out


def fast_gate_pairs(
    cnp.ndarray[cnp.int32_t, ndim=2] top_cliques,        # [n_ent, K]
    cnp.ndarray[cnp.int32_t, ndim=2] pair_indices,        # [n_pairs, 2]
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    double mean_threshold,
):
    """Vectorised batch gate using MEAN cross-PMI of the top-clique
    block. Returns uint8 array of length n_pairs: 1 if the pair
    SURVIVES (mean cross-PMI in K×K block is ≥ mean_threshold).

    For each pair (i, j): compute mean(W[top_cliques[i, *],
    top_cliques[j, *]]) over finite entries (skipping -1 placeholders
    and self-pairs g_a == g_b). Reject if mean < mean_threshold.

    A single strongly-negative entry no longer kills the pair (which
    over-rejected legitimate Phase-1c partial reattachments where
    one or two markers diverge but the broader signature is still
    coherent). Default `mean_threshold = 0.0` rejects only when the
    cross block is net-negative on average.
    """
    cdef int n_pairs = pair_indices.shape[0]
    cdef int K = top_cliques.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] keep = np.ones(n_pairs, dtype=np.uint8)
    cdef cnp.uint8_t[:] keep_mv = keep
    cdef cnp.int32_t[:, :] tc_mv = top_cliques
    cdef cnp.int32_t[:, :] pi_mv = pair_indices
    cdef cnp.float32_t[:, :] W_mv = W
    cdef int p, i, j, a, b, gi, gj
    cdef int n_finite
    cdef double total
    cdef float v
    for p in range(n_pairs):
        i = pi_mv[p, 0]
        j = pi_mv[p, 1]
        total = 0.0
        n_finite = 0
        for a in range(K):
            gi = tc_mv[i, a]
            if gi < 0:
                continue
            for b in range(K):
                gj = tc_mv[j, b]
                if gj < 0:
                    continue
                if gi == gj:
                    continue
                v = W_mv[gi, gj]
                if v != v:
                    continue
                total += v
                n_finite += 1
        # No observations → keep (no evidence of incompatibility)
        if n_finite == 0:
            continue
        if (total / n_finite) < mean_threshold:
            keep_mv[p] = 0
    return keep


def coherence_count_kernel(
    cnp.ndarray[cnp.int32_t, ndim=1] gene_ids,
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    double threshold,
    double real_signal_threshold = 0.0,
):
    """Count-mode coherence in pure C. Returns (C, purity, conflict).

    Equivalent to coherence(gene_ids, W, mode='count', threshold=...)
    in stitching.py — single C loop over the upper-triangular gene-pair
    submatrix, no numpy intermediates. ~5-10× faster per call.

    When ``real_signal_threshold > 0``, pairs with
    ``|W[i,j]| <= real_signal_threshold`` are excluded from BOTH
    numerator and denominator (the "real players" gate; see
    ``coherence_count_per_entity_batch`` for full semantics).
    Default 0.0 preserves legacy ``n_finite``-denominator behaviour.
    """
    cdef int k = gene_ids.shape[0]
    if k < 2:
        return 0.0, 0.0, 0.0
    cdef cnp.int32_t[:] g_mv = gene_ids
    cdef cnp.float32_t[:, :] W_mv = W
    cdef int i, j, gi, gj
    cdef int n_above = 0
    cdef int n_below = 0
    cdef int n_finite = 0
    cdef int n_real_signal = 0
    cdef int denom
    cdef float v, av
    cdef double neg_thr = -threshold
    cdef double rs_thr = real_signal_threshold
    cdef int rs_active = 1 if rs_thr > 0.0 else 0
    cdef double purity, conflict
    for i in range(k):
        gi = g_mv[i]
        for j in range(i + 1, k):
            gj = g_mv[j]
            v = W_mv[gi, gj]
            if v != v:  # NaN
                continue
            n_finite += 1
            if rs_active:
                av = v if v >= 0 else -v
                if av <= rs_thr:
                    continue
                n_real_signal += 1
            if v > threshold:
                n_above += 1
            elif v < neg_thr:
                n_below += 1
    denom = n_real_signal if rs_active else n_finite
    if denom == 0:
        return 0.0, 0.0, 0.0
    purity = <double> n_above / denom
    conflict = <double> n_below / denom
    return (purity - conflict), purity, conflict


def coherence_count_primitives(
    cnp.ndarray[cnp.int32_t, ndim=1] gene_ids,
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    double threshold,
    double real_signal_threshold = 0.0,
):
    """Like `coherence_count_kernel` but returns the raw counts
    (n_above, n_below, n_denom) instead of (C, purity, conflict).
    Used by the decomposable-coherence Stitch path for primitive-sum
    arithmetic across merges.

    When ``real_signal_threshold > 0``, the third element is
    ``n_real_signal`` (count of pairs above the noise floor) rather
    than ``n_finite``. The sum-of-primitives identity used in
    Stitch's union-coherence composition still holds: a pair excluded
    from numerator AND denominator stays excluded after summing
    self/cross primitives.
    """
    cdef int k = gene_ids.shape[0]
    if k < 2:
        return 0, 0, 0
    cdef cnp.int32_t[:] g_mv = gene_ids
    cdef cnp.float32_t[:, :] W_mv = W
    cdef int i, j, gi, gj
    cdef int n_above = 0
    cdef int n_below = 0
    cdef int n_finite = 0
    cdef int n_real_signal = 0
    cdef float v, av
    cdef double neg_thr = -threshold
    cdef double rs_thr = real_signal_threshold
    cdef int rs_active = 1 if rs_thr > 0.0 else 0
    for i in range(k):
        gi = g_mv[i]
        for j in range(i + 1, k):
            gj = g_mv[j]
            v = W_mv[gi, gj]
            if v != v:  # NaN
                continue
            n_finite += 1
            if rs_active:
                av = v if v >= 0 else -v
                if av <= rs_thr:
                    continue
                n_real_signal += 1
            if v > threshold:
                n_above += 1
            elif v < neg_thr:
                n_below += 1
    return n_above, n_below, (n_real_signal if rs_active else n_finite)


def coherence_count_per_entity_batch(
    cnp.ndarray[cnp.int32_t, ndim=1] ent_offsets,
    cnp.ndarray[cnp.int32_t, ndim=1] ent_genes,
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    double threshold,
    double real_signal_threshold = 0.0,
):
    """Count-mode coherence for many entities in one C-level batch.

    Parameters
    ----------
    ent_offsets : int32 [n_ents + 1]
        CSR offsets. Entity ``e`` owns gene indices in
        ``ent_genes[ent_offsets[e]:ent_offsets[e+1]]``.
        Each entity's gene-index slice MUST be deduplicated (no
        repeated genes); the kernel does not check this.
    ent_genes : int32 [total_genes_across_entities]
        CSR data — flat array of gene indices into W.
    W : float32 [n_genes, n_genes]
        Pairwise PMI/NPMI matrix (NaN for missing).
    threshold : double
        Count cutoff. Pairs with |W[i,j]| > threshold count as
        purity (positive) or conflict (negative).
    real_signal_threshold : double, default 0.0
        Noise floor for the "real players" gate. Pairs with
        ``|W[i,j]| <= real_signal_threshold`` are excluded from BOTH
        the numerator AND the denominator — they are treated as
        "not a real player" (NaN, sparse-implicit zero, tight_null,
        and dead_zone all collapse to the same bucket). When
        ``real_signal_threshold == 0.0`` (legacy default), the
        denominator is ``n_finite`` (all non-NaN pairs), preserving
        backward-compatible behaviour. When > 0, the denominator
        is ``n_real_signal`` and ``C`` reflects the (signed)
        majority direction among pairs that actually carry
        information — making coherence panel-shape-agnostic
        across dense (legacy) and sparse (bootstrap, Visium HD)
        W matrices.

    Returns
    -------
    C : float32 [n_ents]   — purity − conflict (the coherence)
    P : float32 [n_ents]   — purity (fraction of pairs above +threshold)
    N : float32 [n_ents]   — conflict (fraction below −threshold)

    Equivalent to looping over entities and calling
    ``coherence_count_kernel`` per entity, but with no Python
    overhead per call. ~50-100x faster on the typical 60-100k
    entity full-tissue Mid-QC pass.
    """
    cdef int n_ents = ent_offsets.shape[0] - 1
    cdef cnp.ndarray[cnp.float32_t, ndim=1] C_out = np.zeros(n_ents, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] P_out = np.zeros(n_ents, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] N_out = np.zeros(n_ents, dtype=np.float32)

    cdef cnp.int32_t[:] off_mv = ent_offsets
    cdef cnp.int32_t[:] g_mv = ent_genes
    cdef cnp.float32_t[:, :] W_mv = W
    cdef cnp.float32_t[:] C_mv = C_out
    cdef cnp.float32_t[:] P_mv = P_out
    cdef cnp.float32_t[:] N_mv = N_out

    cdef int e, lo, hi, i, j, gi, gj, n_genes
    cdef int n_above, n_below, n_finite, n_real_signal, denom
    cdef float v, av
    cdef double neg_thr = -threshold
    cdef double rs_thr = real_signal_threshold
    cdef int rs_active = 1 if rs_thr > 0.0 else 0

    for e in range(n_ents):
        lo = off_mv[e]
        hi = off_mv[e + 1]
        n_genes = hi - lo
        if n_genes < 2:
            # C, P, N stay 0.0
            continue
        n_above = 0
        n_below = 0
        n_finite = 0
        n_real_signal = 0
        for i in range(lo, hi):
            gi = g_mv[i]
            for j in range(i + 1, hi):
                gj = g_mv[j]
                v = W_mv[gi, gj]
                if v != v:  # NaN
                    continue
                n_finite += 1
                if rs_active:
                    av = v if v >= 0 else -v
                    if av <= rs_thr:
                        # Not a real player — skip from both num and denom.
                        continue
                    n_real_signal += 1
                if v > threshold:
                    n_above += 1
                elif v < neg_thr:
                    n_below += 1
        denom = n_real_signal if rs_active else n_finite
        if denom == 0:
            continue
        P_mv[e] = (<float> n_above) / denom
        N_mv[e] = (<float> n_below) / denom
        C_mv[e] = P_mv[e] - N_mv[e]

    return C_out, P_out, N_out


def coherence_cross_primitives(
    cnp.ndarray[cnp.int32_t, ndim=1] gene_ids_a,
    cnp.ndarray[cnp.int32_t, ndim=1] gene_ids_b,
    cnp.ndarray[cnp.float32_t, ndim=2] W,
    double threshold,
    double real_signal_threshold = 0.0,
):
    """Cross-set primitives: count of (g_a, g_b) pairs with g_a in
    gene_ids_a, g_b in gene_ids_b, g_a != g_b, where W[g_a, g_b] is
    above/below threshold. Returns (n_above, n_below, n_denom).

    Used to compute coh(union) from primitives:
      coh(P ∪ Q) = (a + b + a×b - common_internal_double_count) / ...
    where common_internal subtracts overlap to avoid double-count.

    Caller must pass gene_ids_a and gene_ids_b as DISJOINT arrays for
    the simple-sum semantics. For overlap-aware union, decompose
    P ∪ Q = (P\Q) ∪ (Q\P) ∪ (P∩Q) into 3 disjoint segments and call
    this kernel pairwise.

    When ``real_signal_threshold > 0``, the third element is
    ``n_real_signal`` (count of cross pairs with ``|W|`` above the
    noise floor) rather than ``n_finite``. Pass the SAME
    ``real_signal_threshold`` to all self/cross primitive calls in
    a Stitch union to keep the primitive-sum arithmetic consistent.
    """
    cdef int ka = gene_ids_a.shape[0]
    cdef int kb = gene_ids_b.shape[0]
    if ka == 0 or kb == 0:
        return 0, 0, 0
    cdef cnp.int32_t[:] ga_mv = gene_ids_a
    cdef cnp.int32_t[:] gb_mv = gene_ids_b
    cdef cnp.float32_t[:, :] W_mv = W
    cdef int i, j, gi, gj
    cdef int n_above = 0
    cdef int n_below = 0
    cdef int n_finite = 0
    cdef int n_real_signal = 0
    cdef float v, av
    cdef double neg_thr = -threshold
    cdef double rs_thr = real_signal_threshold
    cdef int rs_active = 1 if rs_thr > 0.0 else 0
    for i in range(ka):
        gi = ga_mv[i]
        for j in range(kb):
            gj = gb_mv[j]
            if gi == gj:
                continue
            v = W_mv[gi, gj]
            if v != v:
                continue
            n_finite += 1
            if rs_active:
                av = v if v >= 0 else -v
                if av <= rs_thr:
                    continue
                n_real_signal += 1
            if v > threshold:
                n_above += 1
            elif v < neg_thr:
                n_below += 1
    return n_above, n_below, (n_real_signal if rs_active else n_finite)


cdef inline void _insertion_sort_floats(float *buf, int n) noexcept nogil:
    """In-place ascending insertion sort. O(n^2), fine for n ≤ ~300
    (panel-bounded). No GIL — safe to call from any context."""
    cdef int i, j
    cdef float tmp
    for i in range(1, n):
        tmp = buf[i]
        j = i - 1
        while j >= 0 and buf[j] > tmp:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = tmp


cdef inline double _compute_gene_fit(
    int g_idx,
    int e_off_lo, int e_off_hi,
    cnp.int32_t[:] ent_g_mv,
    cnp.float32_t[:, :] W_mv,
) noexcept nogil:
    """Mean PMI of orphan gene `g_idx` against entity's seed gene set,
    excluding the self-pair if present. NaN entries are skipped.

    Returns -1e9 when no finite PMI pairs exist — used as a sentinel
    by witness-mode tiebreak (caller treats lower-than-anything as
    "no info"). Matches the Python branch's gene_fit calculation in
    spatial.py:reassign_unassigned_grid_pool.
    """
    cdef double pmi_sum = 0.0
    cdef int n_finite = 0
    cdef int ig, eg
    cdef double v
    for ig in range(e_off_lo, e_off_hi):
        eg = ent_g_mv[ig]
        if eg == g_idx:
            continue
        v = W_mv[g_idx, eg]
        if v == v:  # not NaN
            pmi_sum += v
            n_finite += 1
    if n_finite == 0:
        return -1e9
    return pmi_sum / n_finite


cdef inline double _percentile_sorted(float *sorted_buf, int n, double p) noexcept nogil:
    """Linear-interpolated percentile of an ascending-sorted buffer.
    Matches numpy.percentile(..., interpolation='linear') semantics.
    p in [0, 100]."""
    cdef double idx, frac
    cdef int lo, hi
    if n <= 0:
        return 0.0
    if n == 1:
        return sorted_buf[0]
    idx = (p / 100.0) * (n - 1)
    lo = <int> idx
    if lo < 0:
        lo = 0
    hi = lo + 1
    if hi >= n:
        return sorted_buf[lo]
    frac = idx - lo
    return sorted_buf[lo] * (1.0 - frac) + sorted_buf[hi] * frac


cdef inline int _admission_test(
    int gene_idx,
    const cnp.int32_t *seed,
    int seed_len,
    cnp.float32_t[:, :] W,
    const int[::1] W_indptr,       # sparse CSR row offsets (dummy when use_sparse=0)
    const int[::1] W_indices,      # sparse CSR column indices
    const float[::1] W_data,       # sparse CSR data
    int use_sparse,                # 0 = dense W (NaN-skip); 1 = sparse CSR (absent-skip)
    float *pmi_buf,                # caller-owned scratch >= seed_len
    int veto_mode,                 # 0=min, 1=mean, 2=hybrid
    double mean_threshold,
    double min_admit_threshold,
    double aggregator_percentile,
    double real_signal_threshold,
    double neg_npmi_threshold,
    int small_entity_guard_n,
    int legacy_mean_test,          # 1 = `>=` (Phase 1b legacy), 0 = `>` (Rescue legacy)
) nogil:
    """Unified admission gate for a candidate gene against a seed/entity
    gene set. Returns 1 = admit, 0 = veto.

    Modes
    -----
    0 (min)    : Veto if any seed gene has PMI < ``neg_npmi_threshold``.
                 NaN entries are skipped (not treated as < threshold).
    1 (mean)   : Legacy mean-PMI test. Admit if mean(finite PMIs) passes
                 ``mean_threshold`` (>= when ``legacy_mean_test==1``;
                 > when 0). NaN entries skipped. Empty / no-finite → veto.
                 The ``real_signal_threshold > 0`` path adds the "real
                 players" filter (Rescue's rs_active), and
                 ``small_entity_guard_n > 0`` adds the found-neg fallback.
    2 (hybrid) : Real-signal filter + unanimous-min fast-pass + percentile
                 gate. Mirrors the Rescue hybrid branch (see
                 `spatial.py:reassign_unassigned_grid_pool` for the
                 Python reference impl). When
                 ``real_signal_threshold <= 0`` it degenerates to a
                 simple min-fast-pass + mean fallback over finite PMIs.

    Note on NaN semantics: callers that nan-fill W to 0 will see those
    cells as "real-signal" only if ``|0| > real_signal_threshold`` (i.e.
    never for any positive threshold). For the hybrid gate to work as
    intended, callers should pass an unfilled W (NaN-preserving) or
    accept that nan-filled pairs are treated as in-band-zero (= dropped
    by the real-signal filter when ``real_signal_threshold > 0``).
    """
    cdef int j, eg
    cdef int n_finite = 0
    cdef int n_signal = 0
    cdef int found_neg = 0
    cdef double pmi_sum = 0.0
    cdef double mean_p, p_aggregate
    cdef float v, av, min_signal_f, min_v_f
    cdef int rs_active = 1 if real_signal_threshold > 0.0 else 0
    cdef double rs_thr = real_signal_threshold

    if seed_len <= 0:
        # Empty seed: legacy `_mean_pmi_test` returned 0 (no admit);
        # mirror that across all modes for safety.
        return 0

    if veto_mode == 0:
        # min: veto on any PMI < neg_threshold. Unobserved pairs are
        # skipped (NaN under dense, absent under sparse) — see _wget.
        for j in range(seed_len):
            eg = seed[j]
            if eg == gene_idx:
                continue
            if not _wget(W, W_indptr, W_indices, W_data, use_sparse,
                         gene_idx, eg, &v):
                continue
            if v < neg_npmi_threshold:
                return 0
        return 1

    if veto_mode == 1:
        # mean (legacy).
        if rs_active:
            # Real-players filter; aggregator-percentile gate.
            n_signal = 0
            for j in range(seed_len):
                eg = seed[j]
                if eg == gene_idx:
                    continue
                if not _wget(W, W_indptr, W_indices, W_data, use_sparse,
                             gene_idx, eg, &v):
                    continue
                av = v if v >= 0.0 else -v
                if av <= rs_thr:
                    continue
                pmi_buf[n_signal] = v
                n_signal += 1
            if n_signal == 0:
                return 1  # no real signal → defer (admit)
            _insertion_sort_floats(pmi_buf, n_signal)
            p_aggregate = _percentile_sorted(pmi_buf, n_signal, aggregator_percentile)
            if legacy_mean_test:
                return 1 if p_aggregate >= mean_threshold else 0
            return 1 if p_aggregate > mean_threshold else 0
        else:
            pmi_sum = 0.0
            n_finite = 0
            found_neg = 0
            for j in range(seed_len):
                eg = seed[j]
                if eg == gene_idx:
                    continue
                if not _wget(W, W_indptr, W_indices, W_data, use_sparse,
                             gene_idx, eg, &v):
                    continue
                pmi_sum += v
                n_finite += 1
                if v < neg_npmi_threshold:
                    found_neg = 1
            if small_entity_guard_n > 0 and n_finite < small_entity_guard_n:
                # Rescue's small-entity fallback (min-veto).
                return 0 if found_neg else 1
            if n_finite == 0:
                # Legacy `_mean_pmi_test` returns 0 (no admit) on no-finite.
                return 0
            mean_p = pmi_sum / n_finite
            if legacy_mean_test:
                return 1 if mean_p >= mean_threshold else 0
            return 1 if mean_p > mean_threshold else 0

    # veto_mode == 2: hybrid.
    if rs_active:
        n_signal = 0
        min_signal_f = 1e30
        for j in range(seed_len):
            eg = seed[j]
            if eg == gene_idx:
                continue
            if not _wget(W, W_indptr, W_indices, W_data, use_sparse,
                         gene_idx, eg, &v):
                continue
            av = v if v >= 0.0 else -v
            if av <= rs_thr:
                continue
            pmi_buf[n_signal] = v
            if v < min_signal_f:
                min_signal_f = v
            n_signal += 1
        if n_signal == 0:
            return 1  # defer
        if min_signal_f > min_admit_threshold:
            return 1  # unanimous-positive fast-pass
        _insertion_sort_floats(pmi_buf, n_signal)
        p_aggregate = _percentile_sorted(pmi_buf, n_signal, aggregator_percentile)
        # Rescue's hybrid uses `> mean_threshold` (i.e. veto when <=); preserve.
        return 1 if p_aggregate > mean_threshold else 0
    else:
        # No real-signal filter active: min-fast-pass + mean-fallback
        # over finite pairs. Matches Rescue's hybrid non-rs branch.
        pmi_sum = 0.0
        n_finite = 0
        min_v_f = 1e30
        for j in range(seed_len):
            eg = seed[j]
            if eg == gene_idx:
                continue
            if not _wget(W, W_indptr, W_indices, W_data, use_sparse,
                         gene_idx, eg, &v):
                continue
            pmi_sum += v
            n_finite += 1
            if v < min_v_f:
                min_v_f = v
        if n_finite == 0:
            return 0  # veto (Rescue mirrors this with vetoed=1)
        if min_v_f > min_admit_threshold:
            return 1
        mean_p = pmi_sum / n_finite
        return 1 if mean_p > mean_threshold else 0


cdef void _rescue_one_tx(
    int i,
    int tid,
    int current_gen,
    float *pmi_buf,                # per-thread slice (caller already offsets)
    cnp.int8_t[:, :] cache_mv,     # (n_threads, n_ent)
    cnp.int32_t[:, :] gen_mv,      # (n_threads, n_ent)
    cnp.int32_t[:, :] witness_count_mv,  # (n_threads, n_ent), valid iff gen==current_gen
    cnp.float32_t[:, :] min_dist_mv,     # (n_threads, n_ent), valid iff gen==current_gen
    cnp.int32_t[:, :] touched_ents_mv,   # (n_threads, max_touched) — witness-touched ents
    int max_touched,
    int n_ent,
    int max_bin_key_plus_one,
    int has_z,
    double z_bound,
    int veto_mode,
    int rs_active,
    double rs_thr,
    double agg_p,
    double mean_threshold,
    int small_entity_guard_n,
    double neg_npmi_threshold,
    double min_admit_threshold,
    int rank_policy,                # 0 = distance, 1 = witness
    int witness_min_admit,
    int witness_cap,
    int witness_div,                 # small-component cap divisor
    int witness_tiebreak,            # 0 = distance, 1 = gene_fit
    cnp.int32_t[:] ent_size_mv,      # n_tx per entity (caller pre-computed)
    cnp.float32_t[:, :] una_c_mv,
    cnp.int64_t[:] una_g_mv,
    cnp.int64_t[:, :] nb_bins_mv,
    cnp.float32_t[:, :] ass_c_mv,
    cnp.int32_t[:] ass_ent_mv,
    cnp.int64_t[:] bin_off_mv,
    cnp.int64_t[:] bin_data_mv,
    cnp.int32_t[:] ent_off_mv,
    cnp.int32_t[:] ent_g_mv,
    cnp.float32_t[:, :] W_mv,
    const int[::1] dummy_W_indptr,    # for the dense-only _admission_test call
    const int[::1] dummy_W_indices,
    const float[::1] dummy_W_data,
    cnp.int32_t[:] best_ent_mv,
    cnp.float32_t[:] best_dist_mv,
    cnp.int32_t[:] reason_mv,
    cnp.int32_t[:] sef_mv,
) nogil:
    """Per-tx Rescue body. Extracted from `rescue_per_tx_batch`'s inner
    loop so it can be called from prange without tripping Cython's
    reduction-variable detection. All scalar work is function-local.

    Per-thread mutable state (`cache_row`, `gen_row`, `pmi_buf` slice)
    is passed in by the caller; the helper writes to them but they
    never alias across threads.
    """
    cdef int j, k, b, off_lo, off_hi, ass_li, ent
    cdef int e_off_lo, e_off_hi, n_ent_genes, ig, eg
    cdef int g_idx, vetoed, any_vetoed, found_neg, n_finite, used_fallback
    cdef int g_in_E
    cdef int n_signal
    cdef float v_f, av_f, min_signal_f
    cdef double dx, dy, dz, d, best_d, pmi_sum, pmi_val, mean_p, min_pmi
    cdef double p_aggregate

    # Witness-mode scratch
    cdef int n_touched = 0
    cdef int t_idx, best_w_ent, raw_w, w_eff, small_cap, ent_size
    cdef int best_w_eff
    cdef double best_tb, tb, best_w_d2, this_d2
    cdef double ent_min_d_sq

    g_idx = <int> una_g_mv[i]
    if g_idx < 0:
        reason_mv[i] = 1
        return

    best_d = 1e300
    best_ent_mv[i] = -1
    any_vetoed = 0
    used_fallback = 0

    for j in range(9):
        b = <int> nb_bins_mv[i, j]
        if b < 0 or b >= max_bin_key_plus_one:
            continue
        off_lo = <int> bin_off_mv[b]
        off_hi = <int> bin_off_mv[b + 1]
        for k in range(off_lo, off_hi):
            ass_li = <int> bin_data_mv[k]
            if has_z:
                dz = ass_c_mv[ass_li, 2] - una_c_mv[i, 2]
                if dz < 0: dz = -dz
                if dz > z_bound:
                    continue

            ent = ass_ent_mv[ass_li]
            if ent < 0 or ent >= n_ent:
                continue

            if gen_mv[tid, ent] == current_gen:
                if cache_mv[tid, ent] == 1:
                    continue
                # else cached as OK; fall through to distance update
            else:
                e_off_lo = ent_off_mv[ent]
                e_off_hi = ent_off_mv[ent + 1]
                n_ent_genes = e_off_hi - e_off_lo
                vetoed = 0

                if n_ent_genes == 0:
                    vetoed = 0
                else:
                    # Hybrid: short-circuit when candidate gene already
                    # belongs to E's seed (rest of E has already been
                    # vetted compatible with g by an earlier stage).
                    # Mean/min modes never had this fast-path; preserve.
                    g_in_E = 0
                    if veto_mode == 2:
                        for ig in range(e_off_lo, e_off_hi):
                            if ent_g_mv[ig] == g_idx:
                                g_in_E = 1
                                break
                    if g_in_E:
                        vetoed = 0
                    else:
                        # Rescue's mean+non-rs_active branch tracks
                        # `used_fallback` when small_entity_guard kicks
                        # in with a positive (no-neg) outcome. The
                        # shared helper doesn't surface that counter, so
                        # we detect the fallback condition here and
                        # bookkeep before the helper call. Behavior on
                        # the admit/veto decision is identical to the
                        # helper's mean+small_entity_guard branch.
                        if (veto_mode == 1
                                and not rs_active
                                and small_entity_guard_n > 0):
                            pmi_sum = 0.0
                            n_finite = 0
                            found_neg = 0
                            for ig in range(e_off_lo, e_off_hi):
                                eg = ent_g_mv[ig]
                                if eg == g_idx:
                                    continue
                                pmi_val = W_mv[g_idx, eg]
                                if pmi_val == pmi_val:
                                    pmi_sum += pmi_val
                                    n_finite += 1
                                    if pmi_val < neg_npmi_threshold:
                                        found_neg = 1
                            if n_finite < small_entity_guard_n and not found_neg and n_finite > 0:
                                used_fallback += 1
                        # Dispatch to the shared admission gate. Rescue
                        # uses `> mean_threshold` (legacy_mean_test=0).
                        # Rescue is dense-only — pass dummy sparse arrays
                        # and use_sparse=0. (Sparse-W support for Rescue
                        # would need _wget-backed plumbing here too.)
                        vetoed = 0 if _admission_test(
                            g_idx,
                            &ent_g_mv[e_off_lo],
                            e_off_hi - e_off_lo,
                            W_mv,
                            dummy_W_indptr,
                            dummy_W_indices,
                            dummy_W_data,
                            0,  # use_sparse = 0 (Rescue is dense-only)
                            pmi_buf,
                            veto_mode,
                            mean_threshold,
                            min_admit_threshold,
                            agg_p,
                            rs_thr if rs_active else 0.0,
                            neg_npmi_threshold,
                            small_entity_guard_n,
                            0,  # legacy_mean_test = 0 (Rescue: `>`)
                        ) else 1

                cache_mv[tid, ent] = 1 if vetoed else 2
                gen_mv[tid, ent] = current_gen

            if cache_mv[tid, ent] == 1:
                any_vetoed = 1
                continue

            dx = ass_c_mv[ass_li, 0] - una_c_mv[i, 0]
            dy = ass_c_mv[ass_li, 1] - una_c_mv[i, 1]
            d = dx * dx + dy * dy
            if has_z:
                dz = ass_c_mv[ass_li, 2] - una_c_mv[i, 2]
                d += dz * dz

            # Witness-mode accumulation. cache_mv == 2 means "OK and
            # not yet witness-initialized"; we promote to 3 on first
            # contribution. Subsequent visits of the same entity
            # increment the count and update min-distance.
            if rank_policy == 1:
                if cache_mv[tid, ent] == 2:
                    cache_mv[tid, ent] = 3
                    witness_count_mv[tid, ent] = 0
                    min_dist_mv[tid, ent] = <cnp.float32_t> 1e30
                    if n_touched < max_touched:
                        touched_ents_mv[tid, n_touched] = ent
                        n_touched += 1
                witness_count_mv[tid, ent] += 1
                if <cnp.float32_t> d < min_dist_mv[tid, ent]:
                    min_dist_mv[tid, ent] = <cnp.float32_t> d

            # Distance branch: greedy nearest tracker (unchanged).
            if d < best_d:
                best_d = d
                best_ent_mv[i] = ent

    # Post-loop rank-policy dispatch.
    if rank_policy == 1:
        # Witness mode: re-rank using the per-entity capped witness
        # count, breaking ties by gene-fit or distance. Overrides the
        # greedy `best_ent_mv[i]` selected during the inner loop.
        best_w_eff = 0
        best_tb = -1e300
        best_w_ent = -1
        best_w_d2 = 1e300
        for t_idx in range(n_touched):
            ent = touched_ents_mv[tid, t_idx]
            raw_w = witness_count_mv[tid, ent]
            ent_size = ent_size_mv[ent]
            # ceil(ent_size / witness_div)
            small_cap = (ent_size + witness_div - 1) // witness_div
            w_eff = raw_w
            if w_eff > witness_cap:
                w_eff = witness_cap
            if w_eff > small_cap:
                w_eff = small_cap
            if w_eff < witness_min_admit:
                continue
            if witness_tiebreak == 1:  # gene_fit
                e_off_lo = ent_off_mv[ent]
                e_off_hi = ent_off_mv[ent + 1]
                if e_off_hi == e_off_lo:
                    tb = -1e9
                else:
                    tb = _compute_gene_fit(
                        g_idx, e_off_lo, e_off_hi, ent_g_mv, W_mv,
                    )
            else:                       # distance (negated so higher = nearer)
                tb = -min_dist_mv[tid, ent]
            # Lex pick: (w_eff desc, tb desc, entity_id asc).
            this_d2 = min_dist_mv[tid, ent] * min_dist_mv[tid, ent]
            if (best_w_ent < 0
                    or w_eff > best_w_eff
                    or (w_eff == best_w_eff and tb > best_tb)
                    or (w_eff == best_w_eff and tb == best_tb
                        and ent < best_w_ent)):
                best_w_eff = w_eff
                best_tb = tb
                best_w_ent = ent
                best_w_d2 = this_d2
        # Overwrite distance-greedy pick with witness pick.
        best_ent_mv[i] = best_w_ent
        if best_w_ent >= 0:
            best_d = best_w_d2

    if best_ent_mv[i] == -1:
        reason_mv[i] = 2 if any_vetoed else 1
    else:
        best_dist_mv[i] = <cnp.float32_t> (best_d ** 0.5)
    sef_mv[i] = used_fallback


def rescue_per_tx_batch(
    cnp.ndarray[cnp.float32_t, ndim=2] una_coords,    # [n_una, 3] (x,y,z)
    cnp.ndarray[cnp.int64_t, ndim=1] una_g_idx,       # [n_una], gene-vocab idx, -1 = skip
    cnp.ndarray[cnp.int64_t, ndim=2] nb_bins,         # [n_una, 9] bin-keys (-1 = skip)
    cnp.ndarray[cnp.float32_t, ndim=2] assigned_coords,  # [n_assigned, 3]
    cnp.ndarray[cnp.int32_t, ndim=1] assigned_ent_id,    # [n_assigned] entity codes
    cnp.ndarray[cnp.int64_t, ndim=1] bin_offsets,        # [max_bin_key+2] CSR offsets
    cnp.ndarray[cnp.int64_t, ndim=1] bin_data,           # CSR data (assigned local idx)
    cnp.ndarray[cnp.int32_t, ndim=1] ent_gene_offsets,   # [n_entities+1] CSR
    cnp.ndarray[cnp.int32_t, ndim=1] ent_gene_idx,       # CSR data (sorted gene idxs)
    cnp.ndarray[cnp.float32_t, ndim=2] W,                # PMI matrix (NaN-filled = 0 ok)
    double z_bound,                                       # 0.0 ⇒ no z-bound
    int veto_mode,                                        # 0=min, 1=mean, 2=hybrid
    double mean_threshold,
    int small_entity_guard_n,
    double neg_npmi_threshold,
    double min_admit_threshold = 0.0,                     # hybrid: min-PMI fast-pass cutoff
    double real_signal_threshold = 0.0,                   # noise floor; >0 enables real-players gate
    double aggregator_percentile = 50.0,                  # percentile of real-signal pmis (real-players gate)
    # Rank-policy parameters (witness-mode opt-in).
    int rank_policy = 0,                                  # 0=distance, 1=witness
    int witness_min_admit = 3,
    int witness_cap = 3,
    int witness_small_component_cap_divisor = 2,
    int witness_tiebreak = 1,                             # 0=distance, 1=gene_fit
    cnp.ndarray[cnp.int32_t, ndim=1] ent_size = None,     # entity tx counts; required when rank_policy=1
):
    """Per-unassigned-tx Rescue batch.

    Mirrors the Python loop in `reassign_unassigned_grid_pool` (in
    `spatial.py`) — one C-level pass over all unassigned tx, bin-gated
    candidate gathering, veto check (min OR mean), nearest-candidate-tx
    distance pick.

    The "real players" gate (``real_signal_threshold > 0``) replaces
    the legacy mean-of-finite veto with a percentile of pairs whose
    ``|PMI| > real_signal_threshold``. Pairs in the noise band collapse
    with NaNs and explicit-zeros into a single "not informative"
    bucket. Active in BOTH ``mean`` and ``hybrid`` modes when
    ``real_signal_threshold > 0``; falls back to legacy logic when 0.
    ``aggregator_percentile`` (default 50 = median) tunes
    strict↔liberal — lower demands more pairs above ``mean_threshold``,
    higher tolerates a longer left tail.

    Returns
    -------
    best_ent_id : int32[n_una]
        Entity-id code of the best matching entity, or -1 if no rescue.
    matched_dist : float32[n_una]
        Distance to the best candidate (sqrt). NaN if no rescue.
    reason : int32[n_una]
        0 = rescued, 1 = no_candidates, 2 = blocked_by_veto.
    n_small_entity_fallback : int32[n_una]
        Per-tx count of entities that fell back to min-veto due to
        small-entity-guard. Sum across tx for the stat.
    """
    cdef int n_una = una_coords.shape[0]
    cdef int n_ent = ent_gene_offsets.shape[0] - 1
    cdef int max_bin_key_plus_one = bin_offsets.shape[0] - 1
    cdef int has_z = (una_coords.shape[1] >= 3) and (z_bound > 0.0)

    cdef cnp.ndarray[cnp.int32_t, ndim=1] best_ent_arr = np.full(n_una, -1, dtype=np.int32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] best_dist_arr = np.full(n_una, np.nan, dtype=np.float32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] reason_arr = np.zeros(n_una, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] sef_arr = np.zeros(n_una, dtype=np.int32)

    cdef cnp.float32_t[:, :] una_c_mv = una_coords
    cdef cnp.int64_t[:]     una_g_mv = una_g_idx
    cdef cnp.int64_t[:, :]  nb_bins_mv = nb_bins
    cdef cnp.float32_t[:, :] ass_c_mv = assigned_coords
    cdef cnp.int32_t[:]     ass_ent_mv = assigned_ent_id
    cdef cnp.int64_t[:]     bin_off_mv = bin_offsets
    cdef cnp.int64_t[:]     bin_data_mv = bin_data
    cdef cnp.int32_t[:]     ent_off_mv = ent_gene_offsets
    cdef cnp.int32_t[:]     ent_g_mv = ent_gene_idx
    cdef cnp.float32_t[:, :] W_mv = W
    cdef cnp.int32_t[:]     best_ent_mv = best_ent_arr
    cdef cnp.float32_t[:]   best_dist_mv = best_dist_arr
    cdef cnp.int32_t[:]     reason_mv = reason_arr
    cdef cnp.int32_t[:]     sef_mv = sef_arr

    # Per-entity decision cache: per-thread storage so prange iters
    # don't collide. Shape (n_threads × n_ent). gen counter is unique
    # per tx (current_gen = i+1) so different iters never see stale hits.
    cdef int n_threads = openmp.omp_get_max_threads()
    if n_threads < 1:
        n_threads = 1
    cdef cnp.ndarray[cnp.int8_t, ndim=2] ent_cache = np.zeros(
        (n_threads, n_ent), dtype=np.int8,
    )
    cdef cnp.int8_t[:, :] cache_mv = ent_cache
    cdef cnp.ndarray[cnp.int32_t, ndim=2] cache_gen = np.zeros(
        (n_threads, n_ent), dtype=np.int32,
    )
    cdef cnp.int32_t[:, :] gen_mv = cache_gen
    cdef int tid
    cdef int current_gen = 0

    # Real-players gate state. Buffer sized to max entity gene count
    # (bounded by panel size). Heap-allocated once; reused across all
    # entity decisions in this batch.
    cdef int rs_active = 1 if real_signal_threshold > 0.0 else 0
    cdef double rs_thr = real_signal_threshold
    cdef double agg_p = aggregator_percentile
    cdef int max_ent_size = 0
    cdef int _e
    cdef int _ent_size
    if rs_active:
        for _e in range(n_ent):
            _ent_size = ent_gene_offsets[_e + 1] - ent_gene_offsets[_e]
            if _ent_size > max_ent_size:
                max_ent_size = _ent_size
        if max_ent_size < 1:
            max_ent_size = 1
    cdef int pmi_buf_per_thread = max_ent_size if rs_active else 1
    cdef float *pmi_buf = <float*> malloc(
        <size_t>(n_threads * pmi_buf_per_thread) * sizeof(float)
    )
    if pmi_buf == NULL:
        raise MemoryError("rescue_per_tx_batch: failed to allocate pmi_buf")

    # Witness-mode per-thread buffers. Allocated unconditionally so the
    # nogil body can index them without branching on rank_policy at the
    # outer level; the body itself gates witness work on rank_policy.
    # touched_ents bound: typical 9-bin × z-bound neighborhood has
    # at most a few hundred unique entities — 256 is a safe ceiling.
    cdef int max_touched = 256
    cdef cnp.ndarray[cnp.int32_t, ndim=2] witness_count_arr = np.zeros(
        (n_threads, n_ent), dtype=np.int32,
    )
    cdef cnp.int32_t[:, :] witness_count_mv = witness_count_arr
    cdef cnp.ndarray[cnp.float32_t, ndim=2] min_dist_arr = np.zeros(
        (n_threads, n_ent), dtype=np.float32,
    )
    cdef cnp.float32_t[:, :] min_dist_mv = min_dist_arr
    cdef cnp.ndarray[cnp.int32_t, ndim=2] touched_ents_arr = np.zeros(
        (n_threads, max_touched), dtype=np.int32,
    )
    cdef cnp.int32_t[:, :] touched_ents_mv = touched_ents_arr

    # Entity-size view. Witness mode requires it; distance mode
    # ignores it but we still bind a memview so the call signature
    # is uniform.
    cdef cnp.ndarray[cnp.int32_t, ndim=1] ent_size_local
    if ent_size is None:
        ent_size_local = np.zeros(n_ent, dtype=np.int32)
    else:
        if ent_size.shape[0] != n_ent:
            raise ValueError(
                f"ent_size length {ent_size.shape[0]} != n_ent {n_ent}"
            )
        ent_size_local = ent_size
    cdef cnp.int32_t[:] ent_size_mv = ent_size_local

    cdef int i
    cdef int tid_loop

    # Dense-only dummy CSR buffers for the shared _admission_test
    # helper. Rescue never indexes sparse W; these satisfy the typed
    # memoryview contract.
    cdef cnp.ndarray[cnp.int32_t, ndim=1] _dummy_ip = np.zeros(1, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] _dummy_ix = np.zeros(1, dtype=np.int32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] _dummy_dt = np.zeros(1, dtype=np.float32)
    cdef const int[::1]   _dummy_ip_mv = _dummy_ip
    cdef const int[::1]   _dummy_ix_mv = _dummy_ix
    cdef const float[::1] _dummy_dt_mv = _dummy_dt

    try:
        # Per-tx work runs in parallel via prange. Per-thread state
        # (pmi_buf slice + cache row + gen row + witness buffers) is
        # passed into the helper through tid indexing; helper is
        # nogil-safe.
        for i in prange(n_una, nogil=True, schedule="dynamic"):
            tid_loop = openmp.omp_get_thread_num()
            _rescue_one_tx(
                i, tid_loop, i + 1,
                pmi_buf + tid_loop * pmi_buf_per_thread,
                cache_mv, gen_mv,
                witness_count_mv, min_dist_mv, touched_ents_mv,
                max_touched,
                n_ent, max_bin_key_plus_one,
                has_z, z_bound, veto_mode,
                rs_active, rs_thr, agg_p,
                mean_threshold, small_entity_guard_n,
                neg_npmi_threshold, min_admit_threshold,
                rank_policy, witness_min_admit, witness_cap,
                witness_small_component_cap_divisor, witness_tiebreak,
                ent_size_mv,
                una_c_mv, una_g_mv, nb_bins_mv,
                ass_c_mv, ass_ent_mv, bin_off_mv, bin_data_mv,
                ent_off_mv, ent_g_mv, W_mv,
                _dummy_ip_mv, _dummy_ix_mv, _dummy_dt_mv,
                best_ent_mv, best_dist_mv, reason_mv, sef_mv,
            )
    finally:
        free(pmi_buf)

    return best_ent_arr, best_dist_arr, reason_arr, sef_arr

