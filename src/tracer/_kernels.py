"""Numba-compiled kernels shared across per-row NPMI aggregation.

Every purity / conflict / relu variant in `metrics.py` and `core.py`
reduces to the same primitive: for each row of a binary cell-by-gene
presence matrix, enumerate unordered pairs of present genes and
accumulate a handful of summary statistics over the corresponding
entries of the NPMI matrix.

This module exposes a single parallel kernel that returns every stat
the four pre-refactor loops were computing. All downstream metrics are
derived arithmetically from those stats — eliminating four
near-duplicate Python loops that used to cost minutes on 200K-cell
runs.

Per-row outputs (vectors of length `n_rows`):

    k            : number of present genes in the row
    n_pos_above  : count of finite NPMI pairs strictly > `threshold`
    sum_neg      : |sum of finite NPMI pairs that are < 0|
    pos_relu_sum : sum over pairs of max( ReLU_tau(v), 0 )
    neg_relu_sum : sum over pairs of max(-ReLU_tau(v), 0 )

From those, the four pre-refactor metrics collapse to:

    K = k * (k - 1) / 2                         # total pair count
    original_purity   = n_pos_above / K         # matches compute_cell_purity
    original_conflict = sum_neg     / K         # matches compute_cell_conflict
    relu_purity       = pos_relu_sum / K        # matches _relu variants
    relu_conflict     = neg_relu_sum / K
    signal_strength   = pos_relu_sum + neg_relu_sum
    relative_purity   = pos_relu_sum / signal_strength   (when > eps)
    relative_conflict = neg_relu_sum / signal_strength   (when > eps)

NaN entries in `npmi_mat` are treated as "unobserved" and contribute
zero to every accumulator except `k` (this matches the pre-kernel
behaviour: original `np.mean(vals > threshold)` counted NaN as
"not above", ReLU of NaN was zero, etc.).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = ["pair_aggregate_dense"]


@njit(parallel=True, cache=True, fastmath=False)
def _kernel(
    M_u8, col_idx, npmi_mat, threshold, tau,
    k_out, n_pos_above_out, sum_neg_out,
    pos_relu_out, neg_relu_out,
):
    n_rows = M_u8.shape[0]
    n_cols = M_u8.shape[1]

    for r in prange(n_rows):
        # Collect present-gene global indices into a scratch buffer.
        present = np.empty(n_cols, dtype=np.int64)
        k = 0
        for c in range(n_cols):
            if M_u8[r, c]:
                present[k] = col_idx[c]
                k += 1
        k_out[r] = k

        if k < 2:
            n_pos_above_out[r] = 0
            sum_neg_out[r] = 0.0
            pos_relu_out[r] = 0.0
            neg_relu_out[r] = 0.0
            continue

        n_pos_above = 0
        sum_neg = 0.0
        pos_relu = 0.0
        neg_relu = 0.0
        for i in range(k):
            gi = present[i]
            for j in range(i + 1, k):
                gj = present[j]
                v = npmi_mat[gi, gj]
                # Skip NaN: contributes 0 to every accumulator below.
                if v != v:
                    continue
                if v > threshold:
                    n_pos_above += 1
                if v < 0.0:
                    sum_neg += -v
                # Symmetric ReLU branch (inline to avoid extra temporaries)
                if v > tau:
                    pos_relu += v - tau
                elif v < -tau:
                    neg_relu += -v - tau
                # else within dead-zone → ReLU = 0 → no contribution

        n_pos_above_out[r] = n_pos_above
        sum_neg_out[r] = sum_neg
        pos_relu_out[r] = pos_relu
        neg_relu_out[r] = neg_relu


def pair_aggregate_dense(
    M: np.ndarray,
    col_idx: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    threshold: float = 0.05,
    tau: float = 0.05,
):
    """Per-row NPMI-pair summary statistics.

    Parameters
    ----------
    M : np.ndarray[uint8|int8|bool], shape (n_rows, n_cols)
        Binary presence matrix. Row r has a 1 in column c iff that row
        contains the gene at global index `col_idx[c]`.
    col_idx : np.ndarray[int], shape (n_cols,)
        Mapping from local column index → row/column in `npmi_mat`.
    npmi_mat : np.ndarray[float32|float64], shape (G, G)
        Symmetric NPMI matrix. NaN ⇒ unobserved.
    threshold : float
        Threshold used by `compute_cell_purity` (count vals > threshold).
    tau : float
        Symmetric ReLU dead-zone half-width used by the `_relu` variants.

    Returns
    -------
    k : np.ndarray[int64] (n_rows,)
        Number of present genes per row.
    n_pos_above : np.ndarray[int64] (n_rows,)
        Per-row count of upper-triangle pairs with NPMI > `threshold`.
    sum_neg : np.ndarray[float64] (n_rows,)
        Per-row |sum| of upper-triangle pairs with NPMI < 0.
    pos_relu_sum : np.ndarray[float64] (n_rows,)
        Per-row sum of positive symmetric-ReLU(NPMI, tau) contributions.
    neg_relu_sum : np.ndarray[float64] (n_rows,)
        Per-row sum of negative symmetric-ReLU(NPMI, tau) contributions.
    """
    M_u8 = np.ascontiguousarray(M, dtype=np.uint8)
    col_idx_i64 = np.ascontiguousarray(col_idx, dtype=np.int64)
    # Kernel reads npmi_mat as a 2-D buffer; any float dtype is fine but
    # contiguity + no byte-swap is not. Make a safe view.
    if npmi_mat.dtype != np.float64 and npmi_mat.dtype != np.float32:
        npmi_mat = np.ascontiguousarray(npmi_mat, dtype=np.float32)
    else:
        npmi_mat = np.ascontiguousarray(npmi_mat)

    n_rows = M_u8.shape[0]
    k_arr = np.empty(n_rows, dtype=np.int64)
    n_pos = np.empty(n_rows, dtype=np.int64)
    sum_neg = np.empty(n_rows, dtype=np.float64)
    pos_relu = np.empty(n_rows, dtype=np.float64)
    neg_relu = np.empty(n_rows, dtype=np.float64)

    _kernel(
        M_u8, col_idx_i64, npmi_mat, float(threshold), float(tau),
        k_arr, n_pos, sum_neg, pos_relu, neg_relu,
    )
    return k_arr, n_pos, sum_neg, pos_relu, neg_relu


# ---------------------------------------------------------------------------
# top-k present-in-profile ReLU aggregation
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True, fastmath=False)
def _kernel_topk(M_u8, col_idx, npmi_mat, tau, top_k,
                 k_out, pos_relu_out, neg_relu_out, nsig_out):
    """Per-row ReLU pos/neg sums restricted to the top-k strongest
    interactions AMONG genes present in the row.

    For each row we (a) collect present-gene global indices, (b) accumulate the
    signed symmetric-ReLU(NPMI, tau) contributions of every present-gene pair
    whose |NPMI| clears the ``tau`` dead-zone, then (c) keep only the ``top_k``
    contributions with the largest magnitude before summing the positive and
    negative parts.  When ``top_k <= 0`` or a row has <= ``top_k`` signal pairs
    every signal pair is kept (identical to the full ReLU aggregation).
    """
    n_rows = M_u8.shape[0]
    n_cols = M_u8.shape[1]
    for r in prange(n_rows):
        present = np.empty(n_cols, dtype=np.int64)
        k = 0
        for c in range(n_cols):
            if M_u8[r, c]:
                present[k] = col_idx[c]
                k += 1
        k_out[r] = k
        if k < 2:
            pos_relu_out[r] = 0.0
            neg_relu_out[r] = 0.0
            nsig_out[r] = 0
            continue
        maxp = k * (k - 1) // 2
        contrib = np.empty(maxp, dtype=np.float64)  # signed relu-adjusted value
        m = 0
        for i in range(k):
            gi = present[i]
            for j in range(i + 1, k):
                gj = present[j]
                v = npmi_mat[gi, gj]
                if v != v:            # NaN -> unobserved
                    continue
                if v > tau:
                    contrib[m] = v - tau
                    m += 1
                elif v < -tau:
                    contrib[m] = v + tau   # negative; magnitude = -v - tau
                    m += 1
        nsig_out[r] = m
        pos = 0.0
        neg = 0.0
        if m == 0:
            pos_relu_out[r] = 0.0
            neg_relu_out[r] = 0.0
            continue
        if top_k <= 0 or m <= top_k:
            for t in range(m):
                cval = contrib[t]
                if cval > 0.0:
                    pos += cval
                else:
                    neg += -cval
        else:
            absv = np.empty(m, dtype=np.float64)
            for t in range(m):
                a = contrib[t]
                absv[t] = a if a >= 0.0 else -a
            abss = np.sort(absv)             # ascending
            cutoff = abss[m - top_k]         # k-th largest magnitude
            taken = 0
            for t in range(m):
                a = contrib[t]
                mag = a if a >= 0.0 else -a
                if mag > cutoff:
                    if a > 0.0:
                        pos += a
                    else:
                        neg += -a
                    taken += 1
            need = top_k - taken
            if need > 0:                     # break ties at the cutoff value
                for t in range(m):
                    if need == 0:
                        break
                    a = contrib[t]
                    mag = a if a >= 0.0 else -a
                    if mag == cutoff:
                        if a > 0.0:
                            pos += a
                        else:
                            neg += -a
                        need -= 1
        pos_relu_out[r] = pos
        neg_relu_out[r] = neg


def pair_aggregate_topk(M, col_idx, npmi_mat, *, tau=0.05, top_k=500):
    """Top-k present-in-profile ReLU aggregation (wrapper around ``_kernel_topk``).

    Returns ``(k, pos_relu, neg_relu, n_signal)`` where ``k`` is the number of
    present genes per row, ``pos_relu``/``neg_relu`` are the summed positive /
    negative symmetric-ReLU contributions over the top-k strongest present-gene
    interactions, and ``n_signal`` is how many present-gene pairs cleared the
    ``tau`` dead-zone (before top-k truncation).  ``top_k <= 0`` disables the
    truncation (full aggregation).
    """
    M_u8 = np.ascontiguousarray(M, dtype=np.uint8)
    col_idx_i64 = np.ascontiguousarray(col_idx, dtype=np.int64)
    if npmi_mat.dtype != np.float64 and npmi_mat.dtype != np.float32:
        npmi_mat = np.ascontiguousarray(npmi_mat, dtype=np.float32)
    else:
        npmi_mat = np.ascontiguousarray(npmi_mat)
    n_rows = M_u8.shape[0]
    k_arr = np.empty(n_rows, dtype=np.int64)
    pos_relu = np.empty(n_rows, dtype=np.float64)
    neg_relu = np.empty(n_rows, dtype=np.float64)
    nsig = np.empty(n_rows, dtype=np.int64)
    _kernel_topk(M_u8, col_idx_i64, npmi_mat, float(tau), int(top_k),
                 k_arr, pos_relu, neg_relu, nsig)
    return k_arr, pos_relu, neg_relu, nsig
