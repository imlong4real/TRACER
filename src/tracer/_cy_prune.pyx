# cython: boundscheck=False, wraparound=False, nonecheck=False
# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as cnp


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

