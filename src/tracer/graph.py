"""Graph-building utilities (kNN + Delaunay + PyG↔NetworkX conversion)."""

import numpy as np
import torch
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from torch_geometric.data import Data

from ._repro import _ensure_reproducibility_seed


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
