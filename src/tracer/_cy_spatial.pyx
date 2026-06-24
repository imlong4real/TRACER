# cython: language_level=3
# Fast label-constrained connected components using DSU

cimport numpy as cnp
import numpy as np

cdef inline int dsu_find(int x, int[:] parent):
    cdef int px
    while parent[x] != x:
        px = parent[x]
        parent[x] = parent[px]
        x = parent[x]
    return x

cdef inline void dsu_union(int a, int b, int[:] parent, cnp.int8_t[:] rank):
    cdef int ra = dsu_find(a, parent)
    cdef int rb = dsu_find(b, parent)
    cdef cnp.int8_t ra_rank
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        ra_rank = rank[ra] + 1
        rank[ra] = ra_rank

def label_constrained_components(int n,
                                 cnp.ndarray src,
                                 cnp.ndarray dst,
                                 cnp.ndarray codes,
                                 int invalid_code=-1):
    """
    Compute connected components under the constraint that only edges (u,v)
    with codes[u] == codes[v] != invalid_code are considered.

    Parameters
    ----------
    n : int
        Number of nodes.
    src, dst : 1D arrays of edge endpoints (same length E).
    codes : 1D array of length n with integer label codes per node; invalid_code marks excluded nodes.
    invalid_code : int
        Code value to treat as invalid (skip all unions involving it).

    Returns
    -------
    roots : np.ndarray[int32]
        Root representative for each node after unions, length n.
    """
    cdef cnp.ndarray[cnp.int32_t, ndim=1] s32 = np.asarray(src, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] d32 = np.asarray(dst, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] c32 = np.asarray(codes, dtype=np.int32)

    cdef int E = s32.shape[0]
    if d32.shape[0] != E:
        raise ValueError("src and dst must have same length")

    cdef cnp.ndarray[cnp.int32_t, ndim=1] parent = np.arange(n, dtype=np.int32)
    cdef cnp.ndarray[cnp.int8_t, ndim=1] rank = np.zeros(n, dtype=np.int8)

    cdef int i, u, v, cu, cv
    for i in range(E):
        u = s32[i]
        v = d32[i]
        if u < 0 or u >= n or v < 0 or v >= n:
            continue
        cu = c32[u]
        cv = c32[v]
        if cu == cv and cu != invalid_code:
            dsu_union(u, v, parent, rank)

    # compress and return roots
    cdef cnp.ndarray[cnp.int32_t, ndim=1] roots = np.empty(n, dtype=np.int32)
    for i in range(n):
        roots[i] = dsu_find(i, parent)
    return roots
