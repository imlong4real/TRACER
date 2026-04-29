"""Graph-building utilities (kNN + Delaunay + bin-grid + PyG↔NetworkX conversion)."""

import numpy as np
import torch
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from torch_geometric.data import Data

from ._repro import _ensure_reproducibility_seed


# ----------------------------
# 2D xy bin-grid primitives
# ----------------------------
# Bin keys pack (bx, by) signed int32s into a single int64 for use as
# dict keys. A bias of 2**31 is added before packing so that all in-range
# bins map to non-negative int64 values, which keeps the bit layout simple
# and avoids sign-extension surprises.
#
# Layout: key = ((bx + BIAS) << 32) | (by + BIAS)
#
# Constraints: bx + BIAS and by + BIAS must each fit in a uint32, i.e.
# bx, by ∈ [-2**31, 2**31 - 1]. For coordinates in µm and G ≥ 1, this
# allows tissue extents up to ~2 billion µm — far beyond any realistic
# microscopy field of view.

_BIN_BIAS = 1 << 31
_BIN_LO_MASK = (1 << 32) - 1


def bin_xy(coords: np.ndarray, G: float) -> np.ndarray:
    """Hash xy coordinates to packed int64 bin keys at bin size G.

    z is ignored intentionally (per design — phenotype embeds z when
    transcripts are stacked vertically).

    Parameters
    ----------
    coords : (N, 2|3) ndarray
        Spatial coordinates. Only columns 0 and 1 (x, y) are used.
    G : float
        Bin size in µm. Must be > 0.

    Returns
    -------
    bin_keys : (N,) int64 ndarray
        Packed bin keys, unique per (floor(x/G), floor(y/G)).
    """
    if G <= 0:
        raise ValueError(f"G must be positive (got {G})")
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"coords must be a 2D array with at least 2 columns (got shape {coords.shape})"
        )
    bx = np.floor(coords[:, 0] / G).astype(np.int64) + _BIN_BIAS
    by = np.floor(coords[:, 1] / G).astype(np.int64) + _BIN_BIAS
    if (bx < 0).any() or (bx > _BIN_LO_MASK).any():
        raise ValueError(
            "bx out of representable range; reduce coord magnitude or increase G"
        )
    if (by < 0).any() or (by > _BIN_LO_MASK).any():
        raise ValueError(
            "by out of representable range; reduce coord magnitude or increase G"
        )
    return (bx << 32) | by


_UINT64_MASK = (1 << 64) - 1


def unpack_bin(bin_key: int) -> tuple[int, int]:
    """Inverse of `bin_xy` for a single key. Returns (bx, by) as Python ints.

    The mask-to-unsigned step is required because numpy int64 keys may be
    negative (when the upper biased bx has bit 31 set), and Python's `>>`
    is arithmetic for negative ints which would corrupt the upper 32 bits.
    """
    k = int(bin_key) & _UINT64_MASK
    bx = (k >> 32) - _BIN_BIAS
    by = (k & _BIN_LO_MASK) - _BIN_BIAS
    return bx, by


_NEIGHBOR_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBOR_OFFSETS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
# Half-neighborhood for symmetric edge enumeration without double-counting.
# Combined with within-bin pairs, these cover all unordered (bin_a, bin_b)
# pairs in their respective grid topologies exactly once.
_NEIGHBOR_OFFSETS_HALF_8 = ((0, 1), (1, -1), (1, 0), (1, 1))
_NEIGHBOR_OFFSETS_HALF_4 = ((0, 1), (1, 0))
# Backward-compat alias: "half" defaulted to the 8-connected variant.
_NEIGHBOR_OFFSETS_HALF = _NEIGHBOR_OFFSETS_HALF_8


def neighbor_bins(bin_key: int, *, topology: str = "8") -> list[int]:
    """Return list of neighbor bin keys around `bin_key`.

    Output keys use the same numpy int64 signed representation as
    `bin_xy`, so they are interchangeable as dict keys.

    Parameters
    ----------
    bin_key : int
        A packed int64 bin key (as produced by `bin_xy`).
    topology : {"4", "8", "half", "half-4", "half-8"}
        - "4": 4-connected (axial only).
        - "8": 8-connected (axial + diagonal).
        - "half" / "half-8": 4 directions {(0,1),(1,-1),(1,0),(1,1)} —
          half of 8-connected, for symmetric pair enumeration.
        - "half-4": 2 directions {(0,1),(1,0)} — half of 4-connected.
    """
    if topology == "8":
        offsets = _NEIGHBOR_OFFSETS_8
    elif topology == "4":
        offsets = _NEIGHBOR_OFFSETS_4
    elif topology in ("half", "half-8"):
        offsets = _NEIGHBOR_OFFSETS_HALF_8
    elif topology == "half-4":
        offsets = _NEIGHBOR_OFFSETS_HALF_4
    else:
        raise ValueError(
            f"topology must be '4', '8', 'half', 'half-8', or 'half-4' (got {topology!r})"
        )
    bx, by = unpack_bin(bin_key)
    # Build via numpy int64 so the signed bit pattern matches bin_xy().
    bx_arr = np.fromiter((bx + dx for dx, _ in offsets), dtype=np.int64, count=len(offsets)) + _BIN_BIAS
    by_arr = np.fromiter((by + dy for _, dy in offsets), dtype=np.int64, count=len(offsets)) + _BIN_BIAS
    packed = (bx_arr << 32) | by_arr
    return packed.tolist()


# ---------- Phase 3: Create Connected Components on Unassigned Transcripts ----------
def build_graph(
    df,
    *,
    k=40,
    dist_threshold=1.5,
    coord_cols=("x", "y", "z"),
):
    """
    Build a 3D kNN transcript graph with distance thresholding.

    Nodes = transcripts
    Edges = kNN neighbors within dist_threshold

    Returns
    -------
    data : torch_geometric.data.Data
        Attributes:
          - pos        : (N,3) float32
          - edge_index : (2,E) long
          - gene_name  : (N,) np.ndarray[str]
    """
    _ensure_reproducibility_seed()

    coords = df[list(coord_cols)].to_numpy(dtype=np.float32)
    N = coords.shape[0]

    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="auto",
        n_jobs=-1,
    ).fit(coords)

    distances, indices = nbrs.kneighbors(coords, return_distance=True)

    # Deterministic neighbor ordering: round distances to stabilize near-ties
    # and tie-break by neighbor index (lexicographic order).
    dist_key = np.round(distances, decimals=6)
    order = np.lexsort((indices, dist_key))
    distances = np.take_along_axis(distances, order, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)

    # mask: within distance AND not self
    mask = (distances <= dist_threshold)
    mask[:, 0] = False  # remove self

    # vectorized edge extraction
    src = np.repeat(np.arange(N), mask.sum(axis=1))
    tgt = indices[mask]

    edge_index = torch.from_numpy(
        np.vstack([src, tgt]).astype(np.int64)
    )

    data = Data(
        pos=torch.from_numpy(coords),
        edge_index=edge_index,
    )
    data.gene_name = df["feature_name"].to_numpy().astype(str)
    data.id = df['transcript_id'].to_numpy().astype(str)

    print(
        f"Constructed {edge_index.shape[1]:,} edges among {N:,} transcripts "
        f"(k≤{k}, d≤{dist_threshold} µm)"
    )

    return data


# ----------------------------
# Bin-grid edge builder (stage 2 drop-in for build_graph)
# ----------------------------
def build_grid_graph_xy(
    df,
    *,
    k=None,
    dist_threshold,
    coord_cols=("x", "y", "z"),
    G=None,
    neighborhood="8",
    exact_distance_filter=True,
):
    """Build a transcript graph by xy bin-hashing instead of kNN.

    Drop-in replacement for `build_graph` matching the
    `(df, k, dist_threshold, coord_cols) -> Data` contract used by
    `annotate_unassigned_components_fast` and
    `enforce_spatial_coherence_fast`.

    Edges are emitted between transcripts that share an xy bin, plus
    transcripts whose bins are xy-adjacent (4- or 8-connected). When
    `exact_distance_filter=True`, edges are post-filtered to those
    within `dist_threshold` Euclidean distance (using the full coord_cols,
    so z is honored in the filter even though z is ignored for binning).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `coord_cols`, `feature_name`, `transcript_id`.
    k : ignored
        Kept for signature compatibility with `build_graph`.
    dist_threshold : float
        Distance threshold in µm. Default G is derived from this.
    coord_cols : tuple
        Spatial coord column names. First two are x, y (used for binning).
    G : float, optional
        Bin size. Defaults to `8.0` µm. When `exact_distance_filter=True`
        (default), bin size only affects which candidates are *considered*
        before the exact-distance filter; the final edge set still
        matches the kNN result up to `dist_threshold`. Pass
        `dist_threshold / sqrt(2)` to minimize candidate-pair work
        for small `dist_threshold`.
    neighborhood : {"4", "8", "self"}
        Bin-adjacency topology. Default "8" (axial + diagonal).
        ``"self"`` emits same-bin edges only — each bin becomes a
        disconnected clique. Useful for stages whose semantic is "one
        bin = one candidate entity" (e.g. Stage 2 component formation
        when bin size matches expected cell size).
    exact_distance_filter : bool
        If True, post-filter emitted edges by Euclidean distance ≤
        `dist_threshold`. If False, all bin/neighbor-bin pairs are kept
        (faster, fuzzier — effective range up to √2·G at corners with
        8-connectivity).

    Returns
    -------
    data : torch_geometric.data.Data
        Same fields as `build_graph`: pos, edge_index, gene_name, id.
        edge_index is (2, E) with each undirected edge stored once as
        (i, j) with i < j; downstream callers symmetrize as needed.
    """
    _ensure_reproducibility_seed()

    if G is None:
        G = 8.0
    if neighborhood not in ("4", "8", "self"):
        raise ValueError(f"neighborhood must be '4', '8', or 'self' (got {neighborhood!r})")

    coords = df[list(coord_cols)].to_numpy(dtype=np.float32)
    N = coords.shape[0]
    if N == 0:
        edge_index = torch.empty((2, 0), dtype=torch.int64)
        data = Data(pos=torch.from_numpy(coords), edge_index=edge_index)
        data.gene_name = df["feature_name"].to_numpy().astype(str)
        data.id = df["transcript_id"].to_numpy().astype(str)
        return data

    # Bin transcripts by xy at scale G.
    bin_keys = bin_xy(coords[:, :2], G)

    # Group transcripts by bin via argsort.
    order = np.argsort(bin_keys, kind="stable")
    sorted_keys = bin_keys[order]
    unique_bins, group_starts = np.unique(sorted_keys, return_index=True)
    group_ends = np.concatenate([group_starts[1:], [N]])

    bin_to_idxs: dict[int, np.ndarray] = {
        int(bk): order[s:e] for bk, s, e in zip(unique_bins.tolist(), group_starts, group_ends)
    }

    src_chunks: list[np.ndarray] = []
    tgt_chunks: list[np.ndarray] = []

    # Within-bin: all unordered pairs (i < j).
    for idxs in bin_to_idxs.values():
        n = idxs.size
        if n < 2:
            continue
        ii, jj = np.triu_indices(n, k=1)
        src_chunks.append(idxs[ii])
        tgt_chunks.append(idxs[jj])

    # Cross-bin: half-neighborhood ensures every unordered (bin_a, bin_b) pair
    # is enumerated exactly once when neighborhood="8" with full topology.
    # neighborhood="self" emits ONLY same-bin edges — each bin becomes its own
    # disconnected clique (the "1 bin = 1 candidate cell" semantic).
    if neighborhood == "self":
        half = ()
    else:
        half_offsets_8 = _NEIGHBOR_OFFSETS_HALF
        half_offsets_4 = ((0, 1), (1, 0))  # right and down only
        half = half_offsets_8 if neighborhood == "8" else half_offsets_4

    for bk, idxs_a in bin_to_idxs.items():
        bx, by = unpack_bin(bk)
        for dx, dy in half:
            nb_key = int(
                np.int64(
                    ((np.int64(bx + dx) + _BIN_BIAS) << 32)
                    | (np.int64(by + dy) + _BIN_BIAS)
                )
            )
            idxs_b = bin_to_idxs.get(nb_key)
            if idxs_b is None or idxs_b.size == 0:
                continue
            # Cartesian product (idxs_a × idxs_b) — emit i < j on the fly.
            a_grid, b_grid = np.meshgrid(idxs_a, idxs_b, indexing="ij")
            a_flat = a_grid.ravel()
            b_flat = b_grid.ravel()
            lo = np.minimum(a_flat, b_flat)
            hi = np.maximum(a_flat, b_flat)
            src_chunks.append(lo)
            tgt_chunks.append(hi)

    if src_chunks:
        src = np.concatenate(src_chunks).astype(np.int64)
        tgt = np.concatenate(tgt_chunks).astype(np.int64)
    else:
        src = np.empty(0, dtype=np.int64)
        tgt = np.empty(0, dtype=np.int64)

    # Optional exact-distance post-filter (uses full coord_cols incl. z).
    if exact_distance_filter and src.size > 0:
        diff = coords[src] - coords[tgt]
        d = np.sqrt((diff * diff).sum(axis=1))
        mask = d <= float(dist_threshold)
        src = src[mask]
        tgt = tgt[mask]

    edge_index = torch.from_numpy(np.vstack([src, tgt]))

    data = Data(
        pos=torch.from_numpy(coords),
        edge_index=edge_index,
    )
    data.gene_name = df["feature_name"].to_numpy().astype(str)
    data.id = df["transcript_id"].to_numpy().astype(str)

    print(
        f"Constructed {edge_index.shape[1]:,} edges among {N:,} transcripts "
        f"(grid G={G:.3f}, neighborhood={neighborhood}, "
        f"exact_filter={exact_distance_filter}, d≤{dist_threshold} µm)"
    )

    return data


# ----------------------------
# 3D bin-grid graph (xy + z bin neighborhood)
# ----------------------------
# Layout: key = ((bx + BIAS_3D) << 42) | ((by + BIAS_3D) << 21) | (bz + BIAS_3D)
# 21 bits per axis = 2M values per axis. With G=2µm, that's ±2km range.
_BIN_BIAS_3D = 1 << 20  # 1,048,576
_BIN_MASK_3D = (1 << 21) - 1


def _pack3(bx, by, bz):
    """Pack 3D bin coords into int64 keys (vectorized)."""
    bx = np.asarray(bx, dtype=np.int64) + _BIN_BIAS_3D
    by = np.asarray(by, dtype=np.int64) + _BIN_BIAS_3D
    bz = np.asarray(bz, dtype=np.int64) + _BIN_BIAS_3D
    return (bx << 42) | (by << 21) | bz


def build_grid_graph_xyz(
    df,
    *,
    k=None,
    dist_threshold=None,
    coord_cols=("x", "y", "z"),
    G_xy=2.0,
    G_z=2.0,
    xy_neighborhood="8",
    z_neighborhood_depth=1,
    exact_distance_filter=False,
):
    """3D bin-neighborhood transcript graph.

    Bins tx by (bx, by, bz) at scales G_xy (xy) and G_z (z). Edges
    connect tx in the same bin plus a 3D bin-neighborhood: 8-connected
    in xy × ±z_neighborhood_depth in z.

    For Visium HD compatibility: when z is constant across the dataset
    (single section), z_neighborhood_depth is effectively a no-op (all
    tx land in z-bin 0).

    Parameters
    ----------
    G_xy, G_z : float
        Bin sizes (µm) in xy and z respectively.
    xy_neighborhood : {"4", "8"}
        xy bin-adjacency topology.
    z_neighborhood_depth : int
        How many z-bin steps above and below to include. 1 → 3 z-bins
        total (z-1, z, z+1). 0 → same z-bin only.
    exact_distance_filter : bool
        Optional Euclidean d ≤ dist_threshold post-filter on the bin
        candidates. False (default) means pure bin-neighborhood — the
        VHD-compatible call.

    Returns
    -------
    Data : same fields as build_grid_graph_xy.
    """
    _ensure_reproducibility_seed()
    if G_xy <= 0 or G_z <= 0:
        raise ValueError(f"G_xy and G_z must be positive (got {G_xy}, {G_z})")
    if xy_neighborhood not in ("4", "8"):
        raise ValueError(f"xy_neighborhood must be '4' or '8' (got {xy_neighborhood!r})")

    coords = df[list(coord_cols)].to_numpy(dtype=np.float32)
    N = coords.shape[0]
    if N == 0:
        edge_index = torch.empty((2, 0), dtype=torch.int64)
        data = Data(pos=torch.from_numpy(coords), edge_index=edge_index)
        data.gene_name = df["feature_name"].to_numpy().astype(str)
        data.id = df["transcript_id"].to_numpy().astype(str)
        return data

    bx = np.floor(coords[:, 0] / G_xy).astype(np.int64)
    by = np.floor(coords[:, 1] / G_xy).astype(np.int64)
    if coords.shape[1] >= 3:
        bz = np.floor(coords[:, 2] / G_z).astype(np.int64)
    else:
        bz = np.zeros(N, dtype=np.int64)

    bin_keys = _pack3(bx, by, bz)
    order = np.argsort(bin_keys, kind="stable")
    sorted_keys = bin_keys[order]
    unique_bins, group_starts = np.unique(sorted_keys, return_index=True)
    group_ends = np.concatenate([group_starts[1:], [N]])
    bin_to_idxs: dict[int, np.ndarray] = {
        int(bk): order[s:e] for bk, s, e in zip(unique_bins.tolist(), group_starts, group_ends)
    }

    # Half-neighborhood offsets in 3D for unique unordered pair enumeration.
    # Strategy: for any two distinct bins {a, b} with offset (dx, dy, dz) from a,
    # we want exactly one of {(a, b) via offset, (b, a) via -offset} enumerated.
    # Half rule: offset (dx, dy, dz) is "forward" iff (dz > 0) OR
    # (dz == 0 AND ((dx > 0) OR (dx == 0 AND dy > 0))).
    xy_half_8 = [(0, 1), (1, -1), (1, 0), (1, 1)]
    xy_half_4 = [(0, 1), (1, 0)]
    xy_half = xy_half_8 if xy_neighborhood == "8" else xy_half_4
    xy_full_8 = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    xy_full_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    xy_full = xy_full_8 if xy_neighborhood == "8" else xy_full_4

    cross_offsets: list[tuple[int, int, int]] = []
    # dz > 0: full xy + same xy
    for dz in range(1, z_neighborhood_depth + 1):
        cross_offsets.append((0, 0, dz))  # same xy, z+
        for dx, dy in xy_full:
            cross_offsets.append((dx, dy, dz))
    # dz == 0: half xy
    for dx, dy in xy_half:
        cross_offsets.append((dx, dy, 0))

    src_chunks: list[np.ndarray] = []
    tgt_chunks: list[np.ndarray] = []

    # Within-bin pairs.
    for idxs in bin_to_idxs.values():
        n = idxs.size
        if n < 2:
            continue
        ii, jj = np.triu_indices(n, k=1)
        src_chunks.append(idxs[ii])
        tgt_chunks.append(idxs[jj])

    # Cross-bin pairs.
    for bk, idxs_a in bin_to_idxs.items():
        # Unpack to (bx, by, bz) for offset arithmetic.
        k = int(bk)
        bzv = (k & _BIN_MASK_3D) - _BIN_BIAS_3D
        byv = ((k >> 21) & _BIN_MASK_3D) - _BIN_BIAS_3D
        bxv = ((k >> 42) & _BIN_MASK_3D) - _BIN_BIAS_3D
        for dx, dy, dz in cross_offsets:
            nb_key = int(_pack3(bxv + dx, byv + dy, bzv + dz))
            idxs_b = bin_to_idxs.get(nb_key)
            if idxs_b is None or idxs_b.size == 0:
                continue
            a_grid, b_grid = np.meshgrid(idxs_a, idxs_b, indexing="ij")
            a_flat = a_grid.ravel()
            b_flat = b_grid.ravel()
            lo = np.minimum(a_flat, b_flat)
            hi = np.maximum(a_flat, b_flat)
            src_chunks.append(lo)
            tgt_chunks.append(hi)

    if src_chunks:
        src = np.concatenate(src_chunks).astype(np.int64)
        tgt = np.concatenate(tgt_chunks).astype(np.int64)
    else:
        src = np.empty(0, dtype=np.int64)
        tgt = np.empty(0, dtype=np.int64)

    if exact_distance_filter and src.size > 0:
        diff = coords[src] - coords[tgt]
        d = np.sqrt((diff * diff).sum(axis=1))
        mask = d <= float(dist_threshold)
        src = src[mask]
        tgt = tgt[mask]

    edge_index = torch.from_numpy(np.vstack([src, tgt]))
    data = Data(pos=torch.from_numpy(coords), edge_index=edge_index)
    data.gene_name = df["feature_name"].to_numpy().astype(str)
    data.id = df["transcript_id"].to_numpy().astype(str)
    print(
        f"Constructed {edge_index.shape[1]:,} edges among {N:,} transcripts "
        f"(grid 3D G_xy={G_xy:.2f}, G_z={G_z:.2f}, xy_nbhd={xy_neighborhood}, "
        f"z_depth=±{z_neighborhood_depth})"
    )
    return data


# ----------------------------
# Build Delaunay edges (3D/2D)
# ----------------------------
def _edges_from_simplices(simplices: np.ndarray):
    """
    Convert simplices (indices) to undirected edge list (i, j) with i<j.
    """
    s = np.asarray(simplices)
    if s.ndim != 2:
        raise ValueError("simplices must be a 2D array of indices")
    if not np.issubdtype(s.dtype, np.integer):
        raise ValueError("simplices must contain integer indices")

    edges = set()
    for simp in s:
        for a in range(len(simp)):
            for b in range(a + 1, len(simp)):
                i, j = int(simp[a]), int(simp[b])
                if i > j:
                    i, j = j, i
                edges.add((i, j))
    # Deterministic edge ordering regardless of simplex order
    return sorted(edges, key=lambda e: (e[0], e[1]))


def delaunay_edges(points: np.ndarray):
    """
    points: (N, D) with D=2 or 3
    Returns a list of undirected edges (i, j) with i<j
    """
    tri = Delaunay(points)
    simplices = tri.simplices  # (n_simp, D+1)
    return _edges_from_simplices(simplices)


#
def to_networkx(
    data,
    *,
    directed=False,
    remove_isolated=True,
    to_undirected=None,
):
    """
    Convert PyG Data → NetworkX graph.

    Node attributes
    ---------------
    feature_name : str
    pos          : np.ndarray shape (3,)
    connectivity : int

    Edge attributes
    ---------------
    length
    npmi
    cond_prob
    """

    # Backwards-compatibility: some callers pass `to_undirected=True`.
    if to_undirected is not None:
        directed = not bool(to_undirected)

    src, tgt = data.edge_index.numpy()

    if not directed:
        mask = src < tgt
        src, tgt = src[mask], tgt[mask]

    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes
    pos = data.pos.numpy()

    for i, g in enumerate(data.gene_name):
        G.add_node(
            i,
            feature_name=str(g),
            pos=pos[i],
        )

    # Edge lengths
    lengths = np.linalg.norm(pos[src] - pos[tgt], axis=1)

    # Add edges
    for i, (s, t) in enumerate(zip(src, tgt)):
        G.add_edge(
            int(s),
            int(t),
            length=float(lengths[i]),
            npmi=float(data.npmi[i]) if hasattr(data, "npmi") else 0.0,
            cond_prob=float(data.cond_prob[i]) if hasattr(data, "cond_prob") else 0.0,
        )

    # Connectivity (degree)
    connectivity = dict(G.degree())
    nx.set_node_attributes(G, connectivity, "connectivity")

    if remove_isolated:
        iso = [n for n, d in connectivity.items() if d == 0]
        G.remove_nodes_from(iso)

    return G
