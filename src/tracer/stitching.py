"""Phase 4: Hierarchical entity stitching."""

import heapq

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ._repro import _ensure_reproducibility_seed
from ._utils import relu_symmetric
from .graph import delaunay_edges


# ---------- Phase 4: Hierarchical Stitching ----------
# ----------------------------
# Helpers: entity type / parse
# ----------------------------
def infer_entity_type(entity_id: str) -> str:
    """
    Returns one of: 'cell', 'partial', 'component', 'drop', 'unknown'
    """
    if entity_id is None or (isinstance(entity_id, float) and np.isnan(entity_id)):
        return "unknown"
    s = str(entity_id)
    if s == "DROP":
        return "drop"
    if s.startswith("UNASSIGNED_"):
        return "component"
    if "-" in s:
        return "partial"
    # otherwise treat as cell (original)
    return "cell"


def build_entity_table(
    df_final: pd.DataFrame,
    *,
    entity_col: str,
    gene_col: str = "feature_name",
    coord_cols=("x", "y", "z"),
):
    """
    Build per-entity summary:
      - centroid (x,y,z)
      - unique genes list
      - type: cell/partial/component
    """
    df = df_final.copy()
    df[gene_col] = df[gene_col].astype(str).str.strip()

    # filter out drops / missing
    ent = df[entity_col].astype(str)
    keep = ent.notna() & (ent != "DROP") & (ent != "nan")
    df = df.loc[keep].copy()

    # entity type
    df["_etype"] = df[entity_col].map(infer_entity_type)
    df = df[df["_etype"].isin(["cell", "partial", "component"])].copy()

    # centroid
    cent = df.groupby(entity_col, sort=True)[list(coord_cols)].mean()

    # unique genes per entity (sorted for deterministic downstream mapping)
    genes = df.groupby(entity_col, sort=True)[gene_col].unique()
    genes = genes.apply(lambda arr: np.sort(arr.astype(str)))

    etype = df.groupby(entity_col)["_etype"].first()

    summary = cent.join(genes.rename("genes")).join(etype.rename("etype"))
    summary = summary.reset_index().rename(columns={entity_col: "entity_id"})
    return summary


# -------------------------------------------
# Coherence C(gene-set) using NPMI
# -------------------------------------------
def coherence_C_from_genes(
    gene_ids: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    purity_threshold=0.05,
):
    """
    C = purity - conflict
    purity = mean(vals > purity_threshold) over observed (finite) vals in upper triangle
    conflict = mean(-vals[vals < 0]) over observed vals < 0
    If <2 genes or no observed pairs -> return 0.0
    """
    k = int(gene_ids.size)
    if k < 2:
        return 0.0, 0.0, 0.0  # C, purity, conflict

    sub = npmi_mat[np.ix_(gene_ids, gene_ids)]
    iu = np.triu_indices(k, k=1)
    vals = sub[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.0, 0.0

    purity = float(np.mean(vals > purity_threshold))
    neg = -vals[vals < 0]
    conflict = float(neg.mean()) if neg.size > 0 else 0.0
    C = purity - conflict
    return float(C), purity, conflict

def coherence_C_from_genes_relu(
    gene_ids: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    tau=0.05,
    use_relative=False,
):
    """
    ReLU-based coherence score: C = purity - conflict

    Uses symmetric ReLU to suppress weak associations and weight
    stronger evidence more heavily.

    purity = sum(positive ReLU) / total_pairs (absolute)
    conflict = sum(negative ReLU) / total_pairs (absolute)
    C = purity - conflict

    If <2 genes or no observed pairs -> return 0.0

    Parameters
    ----------
    gene_ids : np.ndarray
        Array of gene indices
    npmi_mat : np.ndarray
        NPMI matrix
    tau : float
        Dead-zone threshold for symmetric ReLU
    use_relative : bool
        If True, compute coherence using relative purity/conflict;
        otherwise use absolute purity/conflict (default behavior).

    Returns
    -------
    C : float
        Coherence score (purity - conflict)
    purity : float
        Absolute purity score
    conflict : float
        Absolute conflict score
    """
    k = int(gene_ids.size)
    if k < 2:
        return 0.0, 0.0, 0.0

    sub = npmi_mat[np.ix_(gene_ids, gene_ids)]
    iu = np.triu_indices(k, k=1)
    vals = sub[iu]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return 0.0, 0.0, 0.0

    # Apply symmetric ReLU
    rvals = relu_symmetric(vals, tau)

    K = vals.size
    pos_sum = np.sum(np.maximum(rvals, 0.0))
    neg_sum = np.sum(np.maximum(-rvals, 0.0))

    if use_relative:
        total_abs = pos_sum + neg_sum
        if total_abs > 0:
            purity = float(pos_sum / total_abs)
            conflict = float(neg_sum / total_abs)
        else:
            purity = 0.0
            conflict = 0.0
    else:
        purity = float(pos_sum / K)
        conflict = float(neg_sum / K)

    C = purity - conflict

    return float(C), purity, conflict


def deltaC_between_clusters(
    genes_u: np.ndarray,
    genes_v: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    purity_threshold=0.05,
    penalize_simplicity=True,
):
    """
    ΔC = C_union - max(C_u, C_v) (if not penalize_simplicity)
    """
    # individual
    C_u, _, _ = coherence_C_from_genes(genes_u, npmi_mat, purity_threshold=purity_threshold)
    C_v, _, _ = coherence_C_from_genes(genes_v, npmi_mat, purity_threshold=purity_threshold)

    # union
    union = np.unique(np.concatenate([genes_u, genes_v]))
    C_union, _, _ = coherence_C_from_genes(union, npmi_mat, purity_threshold=purity_threshold)

    if not penalize_simplicity:
        return C_union - max(C_u, C_v)

    nu = max(int(genes_u.size), 1)
    nv = max(int(genes_v.size), 1)
    n_union = nu + nv

    C_u_adj = C_u - 1.0 / nu
    C_v_adj = C_v - 1.0 / nv
    C_sep = max(C_u_adj, C_v_adj)

    deltaC = C_union - (1.0 / n_union) - C_sep
    return float(deltaC)

def deltaC_between_clusters_relu(
    genes_u: np.ndarray,
    genes_v: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    tau=0.05,
    use_relative=False,
    penalize_simplicity=True,
):
    """
    ReLU-based ΔC = C_union - max(C_u, C_v) (if not penalize_simplicity)

    Uses ReLU-based coherence scoring for more robust cluster merging.
    """
    # individual
    C_u, _, _ = coherence_C_from_genes_relu(genes_u, npmi_mat, tau=tau, use_relative=use_relative)
    C_v, _, _ = coherence_C_from_genes_relu(genes_v, npmi_mat, tau=tau, use_relative=use_relative)

    # union
    union = np.unique(np.concatenate([genes_u, genes_v]))
    C_union, _, _ = coherence_C_from_genes_relu(union, npmi_mat, tau=tau, use_relative=use_relative)

    if not penalize_simplicity:
        return C_union - max(C_u, C_v)

    nu = max(int(genes_u.size), 1)
    nv = max(int(genes_v.size), 1)
    n_union = nu + nv

    C_u_adj = C_u - 1.0 / nu
    C_v_adj = C_v - 1.0 / nv
    C_sep = max(C_u_adj, C_v_adj)

    deltaC = C_union - (1.0 / n_union) - C_sep
    return float(deltaC)


# ----------------------------
# Union-Find (Disjoint Set Union)
# ----------------------------
class DSU:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


# --------------------------------------
# Constrained hierarchical ΔC stitching
# --------------------------------------
def stitch_entities_hierarchical(
    summary_df: pd.DataFrame,
    aux: dict,
    *,
    purity_threshold=0.05,
    tau=0.05,
    use_relu=True,
    use_relative=False,
    penalize_simplicity=True,
    deltaC_min=0.0,
    use_3d=True,
    dist_threshold: float | None = None,
):
    """
    summary_df columns required:
      - entity_id
      - x,y,z (or x,y if use_3d=False)
      - genes (np.ndarray[str])
      - etype in {'cell','partial','component'}

    Parameters
    ----------
    summary_df : pd.DataFrame
        Entity summary with required columns
    aux : dict
        Contains NPMI matrix ("W") and gene mapping ("gene_to_idx")
    purity_threshold : float
        Threshold for original scoring (used if use_relu=False)
    tau : float
        Dead-zone threshold for ReLU (used if use_relu=True)
    use_relu : bool
        If True, use ReLU-based coherence scoring (default)
    use_relative : bool
        If True (and use_relu=True), use relative_purity and
        relative_conflict for the stitching criterion.
    penalize_simplicity : bool
        If True, penalize smaller gene sets in deltaC
    deltaC_min : float
        Minimum deltaC threshold for merging
    use_3d : bool
        Use 3D or 2D coordinates
    delaunay_backend : str
        Delaunay backend: "scipy" (default), "fade2d", "fade3d", "gdel3d"

    Returns:
      - mapping entity_id -> stitched_entity_id (string)
      - clusters info (optional)
    """
    npmi_mat = aux["W"]
    gene_to_idx = aux["gene_to_idx"]

    # map entity -> gene indices
    entity_ids = summary_df["entity_id"].astype(str).to_numpy()
    etypes = summary_df["etype"].astype(str).to_numpy()

    gene_id_lists = []
    for genes in summary_df["genes"].values:
        g = pd.Index(np.asarray(genes, dtype=str)).map(gene_to_idx)
        g = np.sort(g[~pd.isna(g)].astype(int).unique())
        gene_id_lists.append(np.asarray(g, dtype=np.int32))

    # points
    if use_3d:
        pts = summary_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    else:
        pts = summary_df[["x", "y"]].to_numpy(dtype=np.float64)

    N = len(entity_ids)
    if N <= 1:
        return {entity_ids[0]: entity_ids[0]}, {}

    # Delaunay edges (use SciPy by default)
    edges = delaunay_edges(pts)

    # Optionally filter edges by geometric length to reduce candidate merges
    if dist_threshold is not None:
        if len(edges) > 0:
            ei = np.asarray(edges, dtype=np.int64)
            p0 = pts[ei[:, 0]]
            p1 = pts[ei[:, 1]]
            dists = np.linalg.norm(p0 - p1, axis=1)
            keep = dists <= float(dist_threshold)
            edges = [tuple(x) for x in ei[keep]]

    # adjacency on original nodes
    adj = [[] for _ in range(N)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    # cluster metadata tracked at DSU roots
    dsu = DSU(N)

    # track whether a cluster contains a real cell (constraint)
    has_cell = np.array([t == "cell" for t in etypes], dtype=bool)

    # For label preference
    # store lists of member entity_ids by type at roots (kept as python sets for simplicity)
    cell_ids = [set([entity_ids[i]]) if etypes[i] == "cell" else set() for i in range(N)]
    partial_ids = [set([entity_ids[i]]) if etypes[i] == "partial" else set() for i in range(N)]
    comp_ids = [set([entity_ids[i]]) if etypes[i] == "component" else set() for i in range(N)]

    # store gene_id union at roots (as sorted unique arrays)
    root_genes = gene_id_lists[:]  # list of np arrays

    # constraint: can we merge clusters A and B?
    def can_merge(ra, rb):
        # never merge two clusters that both contain a cell
        if has_cell[ra] and has_cell[rb]:
            return False
        return True

    # compute deltaC between current roots
    if use_relu:
        def compute_deltaC_roots(ra, rb):
            return deltaC_between_clusters_relu(
                root_genes[ra],
                root_genes[rb],
                npmi_mat,
                tau=tau,
                use_relative=use_relative,
                penalize_simplicity=penalize_simplicity,
            )
    else:
        def compute_deltaC_roots(ra, rb):
            return deltaC_between_clusters(
                root_genes[ra],
                root_genes[rb],
                npmi_mat,
                purity_threshold=purity_threshold,
                penalize_simplicity=penalize_simplicity,
            )

    # max-heap of candidate edges by deltaC (lazy updates)
    def _heap_item(dc, a, b):
        # Deterministic tie-breaking: enforce ordered endpoints
        if a > b:
            a, b = b, a
        return (-dc, a, b)

    heap = []
    for i, j in edges:
        di = compute_deltaC_roots(i, j)
        if np.isfinite(di) and di >= deltaC_min:
            heapq.heappush(heap, _heap_item(di, i, j))

    # greedy merging
    while heap:
        neg_dc, a, b = heapq.heappop(heap)
        dc = -neg_dc

        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            continue
        if not can_merge(ra, rb):
            continue

        # recompute deltaC for current clusters (because a,b may have merged)
        dc_now = compute_deltaC_roots(ra, rb)
        if not (np.isfinite(dc_now) and dc_now >= deltaC_min):
            continue

        # merge (choose new root)
        rnew = dsu.union(ra, rb)
        rold = rb if rnew == ra else ra

        # update cluster metadata onto rnew
        has_cell[rnew] = has_cell[rnew] or has_cell[rold]
        cell_ids[rnew] |= cell_ids[rold]
        partial_ids[rnew] |= partial_ids[rold]
        comp_ids[rnew] |= comp_ids[rold]

        # union genes
        if root_genes[rnew].size == 0:
            root_genes[rnew] = root_genes[rold]
        elif root_genes[rold].size == 0:
            pass
        else:
            root_genes[rnew] = np.unique(np.concatenate([root_genes[rnew], root_genes[rold]])).astype(np.int32)

        # clear old to save memory
        cell_ids[rold].clear()
        partial_ids[rold].clear()
        comp_ids[rold].clear()
        root_genes[rold] = np.empty((0,), dtype=np.int32)

        # push new candidate edges across boundary: use neighbors of members a,b (cheap)
        # We approximate by reusing original node adjacency via a and b endpoints
        # and letting lazy recompute handle staleness.
        # (This is usually good enough for Delaunay graphs.)
        for nbr in (adj[a] + adj[b]):
            rn = dsu.find(nbr)
            rr = dsu.find(rnew)
            if rn == rr:
                continue
            if not can_merge(rr, rn):
                continue
            dtry = compute_deltaC_roots(rr, rn)
            if np.isfinite(dtry) and dtry >= deltaC_min:
                heapq.heappush(heap, _heap_item(dtry, rr, rn))

    # choose stitched label per final root with priority: cell > partial > component
    root_to_label = {}
    for i in range(N):
        r = dsu.find(i)
        if r in root_to_label:
            continue
        if cell_ids[r]:
            label = sorted(cell_ids[r])[0]          # deterministic
        elif partial_ids[r]:
            label = sorted(partial_ids[r])[0]
        else:
            label = sorted(comp_ids[r])[0]
        root_to_label[r] = label

    entity_to_stitched = {entity_ids[i]: root_to_label[dsu.find(i)] for i in range(N)}
    return entity_to_stitched, {"root_to_label": root_to_label}

def apply_stitching_to_transcripts(
    df_final: pd.DataFrame,
    aux: dict,
    *,
    entity_col="cell_id_final",   # final id column
    gene_col="feature_name",
    coord_cols=("x", "y", "z"),
    purity_threshold=0.05,
    tau=0.05,
    use_relu=True,
    penalize_simplicity=True,
    deltaC_min=0.0,
    use_3d=True,
    out_col="cell_id_stitched",
):
    _ensure_reproducibility_seed()
    # build entity table (centroids + genes)
    summary = build_entity_table(
        df_final,
        entity_col=entity_col,
        gene_col=gene_col,
        coord_cols=coord_cols,
    )

    # rename centroid cols to x,y,z expected by stitching function
    # (build_entity_table keeps original names)
    if tuple(coord_cols) == ("x", "y", "z"):
        summary = summary.rename(columns={"x": "x", "y": "y", "z": "z"})
    else:
        # if different coordinate column names used, map them:
        summary = summary.rename(columns={coord_cols[0]: "x", coord_cols[1]: "y", coord_cols[2]: "z"})

    # stitch entities
    entity_to_stitched, info = stitch_entities_hierarchical(
        summary_df=summary.rename(columns={"entity_id": "entity_id"}),
        aux=aux,
        purity_threshold=purity_threshold,
        tau=tau,
        use_relu=use_relu,
        penalize_simplicity=penalize_simplicity,
        deltaC_min=deltaC_min,
        use_3d=use_3d,
        dist_threshold=None,
    )

    # map back to transcripts
    df_out = df_final.copy()
    ent = df_out[entity_col].astype(str)

    # default: keep original entity label (DROP stays DROP)
    df_out[out_col] = ent

    # apply stitched labels to non-drop entities
    mask = ent.notna() & (ent != "DROP") & (ent != "nan")
    df_out.loc[mask, out_col] = ent[mask].map(entity_to_stitched).fillna(ent[mask])

    return df_out, entity_to_stitched


def apply_stitching_to_transcripts_fast(
    df_final: pd.DataFrame,
    aux: dict,
    *,
    entity_col="cell_id_final",
    gene_col="feature_name",
    coord_cols=("x", "y", "z"),
    purity_threshold=0.05,
    tau=0.05,
    use_relu=True,
    penalize_simplicity=True,
    deltaC_min=0.0,
    use_3d=True,
    out_col="cell_id_stitched",
    show_progress: bool = True,
):
    """
    Fast wrapper around `apply_stitching_to_transcripts`.
    - Builds entity table and runs hierarchical stitching, with optional progress bars.
    - Uses ReLU-based coherence scoring by default for robust cluster merging.
    - Returns same outputs as original function.

    Parameters
    ----------
    df_final : pd.DataFrame
        Transcript-level data with entity assignments
    aux : dict
        Contains NPMI matrix ("W") and gene mapping ("gene_to_idx")
    entity_col : str
        Column with current entity labels
    gene_col : str
        Column with gene names
    coord_cols : tuple
        Coordinate column names
    purity_threshold : float
        Threshold for original scoring (used if use_relu=False)
    tau : float
        Dead-zone threshold for ReLU (used if use_relu=True, default)
    use_relu : bool
        If True, use ReLU-based coherence (default, faster and more robust)
    penalize_simplicity : bool
        Penalize smaller gene sets in deltaC
    deltaC_min : float
        Minimum deltaC for merging
    use_3d : bool
        Use 3D coordinates
    out_col : str
        Output column name
    show_progress : bool
        Show progress bar

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with stitched labels
    entity_to_stitched : dict
        Mapping from original to stitched entity IDs
    """
    _ensure_reproducibility_seed()
    # build entity table (centroids + genes)
    if show_progress:
        # small progress step for entity build
        pbar = tqdm(total=2, desc="stitching")
    else:
        pbar = None

    summary = build_entity_table(
        df_final,
        entity_col=entity_col,
        gene_col=gene_col,
        coord_cols=coord_cols,
    )
    if pbar is not None:
        pbar.update(1)

    # rename centroid cols if necessary
    if tuple(coord_cols) == ("x", "y", "z"):
        summary = summary.rename(columns={"x": "x", "y": "y", "z": "z"})
    else:
        summary = summary.rename(columns={coord_cols[0]: "x", coord_cols[1]: "y", coord_cols[2]: "z"})

    # stitch entities (this is the heavy op - uses ReLU by default)
    entity_to_stitched, info = stitch_entities_hierarchical(
        summary_df=summary.rename(columns={"entity_id": "entity_id"}),
        aux=aux,
        purity_threshold=purity_threshold,
        tau=tau,
        use_relu=use_relu,
        penalize_simplicity=penalize_simplicity,
        deltaC_min=deltaC_min,
        use_3d=use_3d,
        dist_threshold=None,
    )

    if pbar is not None:
        pbar.update(1)
        pbar.close()

    # map back to transcripts using vectorized numpy lookup (much faster than pandas.map())
    df_out = df_final.copy()
    ent = df_out[entity_col].astype(str)
    df_out[out_col] = ent

    mask = ent.notna() & (ent != "DROP") & (ent != "nan")

    if mask.sum() > 0:
        # Fully vectorized mapping using pandas.Series.map() (much faster than loop)
        ent_values = ent[mask]

        # Convert dict to pandas Series for vectorized .map()
        mapping_series = pd.Series(entity_to_stitched)

        # Vectorized map with fillna for unmapped values (keeps original)
        stitched_values = ent_values.map(mapping_series).fillna(ent_values)

        # Single assignment
        df_out.loc[mask, out_col] = stitched_values

    return df_out, entity_to_stitched


def apply_stitching_to_transcripts_memory_efficient(
    df_final: pd.DataFrame,
    aux: dict,
    *,
    entity_col="cell_id_final",
    gene_col="feature_name",
    coord_cols=("x", "y", "z"),
    purity_threshold=0.05,
    tau=0.05,
    use_relu=True,
    use_relative=False,
    penalize_simplicity=True,
    deltaC_min=0.0,
    use_3d=True,
    dist_threshold: float | None = 15.0,
    out_col="cell_id_stitched",
    show_progress: bool = True,
    in_place: bool = False,
    map_mode: str = "categorical",
    chunk_size: int | None = 2_000_000,
):
    """
    Memory-efficient stitching wrapper optimized for very large datasets (10M+ rows).

    This function mirrors `apply_stitching_to_transcripts_fast` but minimizes
    temporary allocations when mapping stitched labels back to transcripts.

    Parameters
    ----------
    df_final : pd.DataFrame
        Transcript-level data with entity assignments
    aux : dict
        Contains NPMI matrix ("W") and gene mapping ("gene_to_idx")
    entity_col : str
        Column with current entity labels
    gene_col : str
        Column with gene names
    coord_cols : tuple
        Coordinate column names
    purity_threshold : float
        Threshold for original scoring (used if use_relu=False)
    tau : float
        Dead-zone threshold for ReLU (used if use_relu=True, default)
    use_relu : bool
        If True, use ReLU-based coherence (default, faster and more robust)
    use_relative : bool
        If True (and use_relu=True), use relative_purity and
        relative_conflict for stitching.
    penalize_simplicity : bool
        Penalize smaller gene sets in deltaC
    deltaC_min : float
        Minimum deltaC for merging
    use_3d : bool
        Use 3D coordinates
    out_col : str
        Output column name
    show_progress : bool
        Show progress bar
    in_place : bool
        If True, write output to the input DataFrame without copying
    map_mode : {"categorical", "chunked"}
        Mapping strategy to minimize memory use.
        - "categorical": map category codes (fast, low memory)
        - "chunked": map in chunks using pandas Series.map()
    chunk_size : int or None
        Chunk size for "chunked" mapping. None maps all at once.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with stitched labels
    entity_to_stitched : dict
        Mapping from original to stitched entity IDs
    """
    _ensure_reproducibility_seed()
    if show_progress:
        pbar = tqdm(total=2, desc="stitching")
    else:
        pbar = None

    summary = build_entity_table(
        df_final,
        entity_col=entity_col,
        gene_col=gene_col,
        coord_cols=coord_cols,
    )
    if pbar is not None:
        pbar.update(1)

    if tuple(coord_cols) == ("x", "y", "z"):
        summary = summary.rename(columns={"x": "x", "y": "y", "z": "z"})
    else:
        summary = summary.rename(columns={coord_cols[0]: "x", coord_cols[1]: "y", coord_cols[2]: "z"})

    entity_to_stitched, info = stitch_entities_hierarchical(
        summary_df=summary.rename(columns={"entity_id": "entity_id"}),
        aux=aux,
        purity_threshold=purity_threshold,
        tau=tau,
        use_relu=use_relu,
        use_relative=use_relative,
        penalize_simplicity=penalize_simplicity,
        deltaC_min=deltaC_min,
        use_3d=use_3d,
        dist_threshold=dist_threshold,
    )

    if pbar is not None:
        pbar.update(1)
        pbar.close()

    df_out = df_final if in_place else df_final.copy()
    ent = df_out[entity_col]

    if map_mode == "categorical":
        ent_cat = ent.astype("category")
        categories = ent_cat.cat.categories.astype(str)
        mapped_categories = pd.Index(categories).map(lambda x: entity_to_stitched.get(x, x))

        # Fast path: one-to-one mapping (no merges) -> just rename categories
        if mapped_categories.is_unique:
            df_out[out_col] = ent_cat.cat.rename_categories(mapped_categories)
        else:
            # Slow path: merges exist, recode via factorization
            new_cat_codes, new_categories = pd.factorize(mapped_categories, sort=False)
            ent_codes = ent_cat.cat.codes.to_numpy(copy=False)

            out_codes = np.full_like(ent_codes, -1)
            valid = ent_codes >= 0
            if valid.any():
                out_codes[valid] = new_cat_codes[ent_codes[valid]]

            df_out[out_col] = pd.Categorical.from_codes(out_codes, categories=new_categories)
    elif map_mode == "chunked":
        ent_str = ent.astype(str)
        df_out[out_col] = ent_str

        mask = ent_str.notna() & (ent_str != "DROP") & (ent_str != "nan")
        if mask.any():
            idx = np.flatnonzero(mask.to_numpy())
            mapping_series = pd.Series(entity_to_stitched)

            if chunk_size is None:
                vals = ent_str.iloc[idx]
                mapped = vals.map(mapping_series).fillna(vals)
                df_out.iloc[idx, df_out.columns.get_loc(out_col)] = mapped.to_numpy()
            else:
                for start in range(0, len(idx), chunk_size):
                    end = start + chunk_size
                    sel = idx[start:end]
                    vals = ent_str.iloc[sel]
                    mapped = vals.map(mapping_series).fillna(vals)
                    df_out.iloc[sel, df_out.columns.get_loc(out_col)] = mapped.to_numpy()
    else:
        raise ValueError("map_mode must be 'categorical' or 'chunked'")

    return df_out, entity_to_stitched
