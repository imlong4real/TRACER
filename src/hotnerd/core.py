import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
import networkx as nx
from collections import Counter
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
from torch_geometric.utils import to_networkx
import heapq
from collections import defaultdict

# ---------- Phase 1/2: Conservative NPMI pruning ----------
# Denoise cell and create partial cell IDs based on NPMI gene coherence (Phase 1)
# Then further denoise partial cells (Phase 2)
def build_dense_npmi_matrix(
    npmi_df,
    gene_i_col="gene_i",
    gene_j_col="gene_j",
    npmi_col="NPMI",
):
    """
    Build dense symmetric NPMI matrix.
    Missing pairs remain NaN (conservative).
    """
    npmi_df = npmi_df.copy()
    npmi_df[gene_i_col] = npmi_df[gene_i_col].astype(str).str.strip()
    npmi_df[gene_j_col] = npmi_df[gene_j_col].astype(str).str.strip()
    npmi_df[npmi_col] = pd.to_numeric(npmi_df[npmi_col], errors="coerce")

    genes = pd.Index(
        np.unique(
            np.concatenate(
                [npmi_df[gene_i_col].values, npmi_df[gene_j_col].values]
            )
        )
    ).astype(str)

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    G = len(genes)

    W = np.full((G, G), np.nan, dtype=np.float32)
    np.fill_diagonal(W, np.nan)

    ai = npmi_df[gene_i_col].map(gene_to_idx).to_numpy()
    bi = npmi_df[gene_j_col].map(gene_to_idx).to_numpy()
    vv = npmi_df[npmi_col].to_numpy(dtype=np.float32)

    W[ai, bi] = vv
    W[bi, ai] = vv

    return np.asarray(genes), gene_to_idx, W

def prune_genes_by_npmi_greedy(
    gene_ids: np.ndarray,
    W: np.ndarray,
    threshold: float = -0.1,
):
    """
    Iteratively remove gene with the largest number of
    observed NPMI < threshold edges.
    Missing (NaN) pairs are ignored.
    """
    k = gene_ids.size
    if k <= 1:
        return np.ones(k, dtype=bool)

    subW = W[np.ix_(gene_ids, gene_ids)]
    bad = (subW < threshold)
    bad &= np.isfinite(subW)  # only penalize observed pairs
    np.fill_diagonal(bad, False)

    active = np.ones(k, dtype=bool)
    bad_counts = bad.sum(axis=1).astype(int)

    while active.sum() > 1:
        act = np.flatnonzero(active)
        if bad_counts[act].max() == 0:
            break

        rm = act[np.argmax(bad_counts[act])]
        active[rm] = False

        neighbors = np.flatnonzero(active & bad[rm])
        bad_counts[neighbors] -= 1
        bad_counts[rm] = 0

    return active

#
def prune_transcripts(
    df,
    npmi_df,
    cell_id_col="cell_id",
    gene_col="feature_name",
    threshold=-0.1,
    unassigned_id="-1",
):
    """
    Two-pass conservative NPMI pruning.
    Partial cell IDs are string-based: cellID-1
    """
    df = df.copy()
    df["_cell_str"] = df[cell_id_col].astype(str)
    df[gene_col] = df[gene_col].astype(str).str.strip()

    genes, gene_to_idx, W = build_dense_npmi_matrix(npmi_df)
    df["_gene_idx"] = df[gene_col].map(gene_to_idx)

    # ---------- PASS 1 ----------
    df["cell_id_npmi_cons_p1"] = df["_cell_str"]
    df["npmi_cons_p1_status"] = np.where(
        df["_cell_str"] == unassigned_id,
        "unassigned_input",
        "core",
    )

    partial_map = {}

    for cid, sub in df[df["_cell_str"] != unassigned_id].groupby("_cell_str", sort=False):
        g_local = sub["_gene_idx"].dropna().astype(int).unique()
        if g_local.size <= 1:
            continue

        keep_mask = prune_genes_by_npmi_greedy(g_local, W, threshold)
        removed = g_local[~keep_mask]
        if removed.size == 0:
            continue

        pid = f"{cid}-1"
        partial_map[cid] = pid
        rem_set = set(removed.tolist())

        mask = (df["_cell_str"] == cid) & (df["_gene_idx"].isin(rem_set))
        df.loc[mask, "cell_id_npmi_cons_p1"] = pid
        df.loc[mask, "npmi_cons_p1_status"] = "partial_p1"

    # ---------- PASS 2 ----------
    df["cell_id_npmi_cons_p2"] = df["cell_id_npmi_cons_p1"]
    df["npmi_cons_p2_status"] = "unchanged"

    for pid in set(partial_map.values()):
        sub = df[df["cell_id_npmi_cons_p1"] == pid]
        g_local = sub["_gene_idx"].dropna().astype(int).unique()
        if g_local.size <= 1:
            df.loc[sub.index, "npmi_cons_p2_status"] = "partial_p2"
            continue

        keep_mask = prune_genes_by_npmi_greedy(g_local, W, threshold)
        removed = g_local[~keep_mask]

        if removed.size == 0:
            df.loc[sub.index, "npmi_cons_p2_status"] = "partial_p2"
            continue

        rem_set = set(removed.tolist())
        mask = (df["cell_id_npmi_cons_p1"] == pid) & (df["_gene_idx"].isin(rem_set))

        df.loc[~mask & (df["cell_id_npmi_cons_p1"] == pid), "npmi_cons_p2_status"] = "partial_p2"
        df.loc[mask, "cell_id_npmi_cons_p2"] = unassigned_id
        df.loc[mask, "npmi_cons_p2_status"] = "unassigned_from_partial"

    df.drop(columns=["_cell_str", "_gene_idx"], inplace=True)

    aux = {
        "genes": genes,
        "gene_to_idx": gene_to_idx,
        "W": W,
        "partial_map": partial_map,
        "threshold": threshold,
    }
    return df, aux

def pairwise_npmi_stats(gene_ids, W):
    if gene_ids.size <= 1:
        return dict(
            sum_npmi=np.nan,
            min_npmi=np.nan,
            p25=np.nan,
            p50=np.nan,
            p75=np.nan,
            n_pairs=0,
        )

    subW = W[np.ix_(gene_ids, gene_ids)]
    iu = np.triu_indices(len(gene_ids), k=1)
    vals = subW[iu]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return dict(
            sum_npmi=np.nan,
            min_npmi=np.nan,
            p25=np.nan,
            p50=np.nan,
            p75=np.nan,
            n_pairs=0,
        )

    return dict(
        sum_npmi=float(vals.sum()),
        min_npmi=float(vals.min()),
        p25=float(np.percentile(vals, 25)),
        p50=float(np.percentile(vals, 50)),
        p75=float(np.percentile(vals, 75)),
        n_pairs=int(vals.size),
    )

def diagnostic_npmi_report(df, aux, cell_id):
    W = aux["W"]
    gene_to_idx = aux["gene_to_idx"]
    cid = str(cell_id)
    pid = aux["partial_map"].get(cid)

    rows = []

    def summarize(name, sub):
        genes = sub["feature_name"].astype(str).unique()
        gids = pd.Index(genes).map(gene_to_idx).dropna().astype(int).unique()
        stats = pairwise_npmi_stats(gids, W)
        return {
            "stage": name,
            "n_transcripts": len(sub),
            "n_unique_genes": len(genes),
            **stats,
        }

    rows.append(summarize("original", df[df["cell_id"] == cell_id]))
    rows.append(summarize("core_pass1", df[(df["cell_id"] == cell_id) & (df["npmi_cons_p1_status"] == "core")]))

    if pid:
        rows.append(summarize("partial_pass1", df[df["cell_id_npmi_cons_p1"] == pid]))
        rows.append(summarize("partial_pass2", df[(df["cell_id_npmi_cons_p2"] == pid) &
                                                  (df["npmi_cons_p2_status"] == "partial_p2")]))
        rows.append(summarize("unassigned_from_partial",
                              df[(df["cell_id_npmi_cons_p1"] == pid) &
                                 (df["npmi_cons_p2_status"] == "unassigned_from_partial")]))

    return pd.DataFrame(rows)

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
    coords = df[list(coord_cols)].to_numpy(dtype=np.float32)
    N = coords.shape[0]

    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="auto",
        n_jobs=-1,
    ).fit(coords)

    distances, indices = nbrs.kneighbors(coords, return_distance=True)

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

#
def annotate_unassigned_components(
    df_pruned: pd.DataFrame,
    aux: dict,
    *,
    build_graph_fn,                 # pass build_graph here
    prune_fn,                       # pass prune_genes_by_npmi_greedy here
    coord_cols=("x", "y", "z"),
    k=8,
    dist_threshold=1.5,
    min_comp_size=50,
    npmi_threshold=-0.1,
    unassigned_final_col="cell_id_npmi_cons_p2",  # conservative pruning output column
    cell_id_col="cell_id",
    gene_col="feature_name",
    transcript_id_col="transcript_id",
):
    """
    1) Take all unassigned transcripts (cell_id_npmi_cons_p2 == "-1" by default)
    2) Build kNN graph + connected components
    3) Drop small components (< min_comp_size)
    4) For remaining comps, greedy NPMI prune genes once (iterative greedy until coherent)
       and drop transcripts belonging to removed genes
    5) Write:
       - unassigned_comp_id (NaN for non-unassigned)
       - unassigned_qc_status
       - cell_id_final
    """
    df = df_pruned.copy()

    # Ensure transcript_id exists
    if transcript_id_col not in df.columns:
        df[transcript_id_col] = df.index.astype(str)

    # Normalize gene names
    df[gene_col] = df[gene_col].astype(str).str.strip()

    # Use final-unassigned definition: anything with cell_id_npmi_cons_p2 == "-1"
    if unassigned_final_col in df.columns:
        is_unassigned = df[unassigned_final_col].astype(str) == "-1"
        assigned_id_series = df[unassigned_final_col].astype(str)  # already includes original + partial ids
    else:
        # fallback: original unassigned only
        is_unassigned = df[cell_id_col].astype(str) == "-1"
        assigned_id_series = df[cell_id_col].astype(str)

    df["unassigned_comp_id"] = pd.Series(index=df.index, dtype="object")
    df["unassigned_qc_status"] = pd.Series(index=df.index, dtype="object")
    df.loc[is_unassigned, "unassigned_qc_status"] = "unassigned_raw"
    
    # If nothing unassigned, just define cell_id_final and return
    if is_unassigned.sum() == 0:
        df["cell_id_final"] = assigned_id_series
        return df

    # Subset unassigned transcripts and build graph
    df_u = df.loc[is_unassigned].copy()

    # Build graph (expects transcript_id in df_u; your build_graph uses df['transcript_id'])
    data_u = build_graph_fn(
        df_u,
        k=k,
        dist_threshold=dist_threshold,
        coord_cols=coord_cols,
    )

    # Keep isolated nodes so component mapping covers all nodes
    G_nx = to_networkx(data_u, directed=False, remove_isolated=False)
    components = list(nx.connected_components(G_nx))

    # Map node -> component index
    num_nodes = df_u.shape[0]
    comp_idx = np.full(num_nodes, -1, dtype=np.int32)
    for ci, comp in enumerate(components):
        comp_idx[list(comp)] = ci
    assert (comp_idx >= 0).all(), "Some nodes did not get assigned to a component (unexpected)."

    # Build mapping back to transcript_id
    # Assign component IDs back by index alignment (NO merge)
    comp_ids_str = np.array([f"UNASSIGNED_{i}" for i in comp_idx], dtype=object)

    df.loc[df_u.index, "unassigned_comp_id"] = comp_ids_str


    # Component sizes
    comp_sizes = pd.Series(comp_idx).value_counts().sort_index()
    comp_size_map = {f"UNASSIGNED_{i}": int(sz) for i, sz in comp_sizes.items()}
    df["unassigned_comp_size"] = df["unassigned_comp_id"].map(comp_size_map)

    # Drop small components
    drop_small = is_unassigned & df["unassigned_comp_size"].notna() & (df["unassigned_comp_size"] < min_comp_size)
    df.loc[drop_small, "unassigned_qc_status"] = "drop_small_comp"

    # For large comps: NPMI prune genes and drop pruned genes
    W = aux["W"]
    gene_to_idx = aux["gene_to_idx"]

    # only operate on comps that are large and not already dropped
    keep_candidate = is_unassigned & (~drop_small) & df["unassigned_comp_id"].notna()
    large_comp_ids = df.loc[keep_candidate, "unassigned_comp_id"].unique()

    # precompute gene idx per transcript (NaN if gene missing from NPMI table)
    gene_idx_all = df[gene_col].map(gene_to_idx)

    for comp_id in large_comp_ids:
        comp_mask = (df["unassigned_comp_id"] == comp_id) & keep_candidate
        if comp_mask.sum() == 0:
            continue

        # unique genes in this component (only those present in NPMI gene_to_idx)
        g_local = gene_idx_all.loc[comp_mask].dropna().astype(int).unique()
        if g_local.size <= 1:
            # Nothing to prune; keep as an unassigned component
            df.loc[comp_mask, "unassigned_qc_status"] = "keep_unassigned_comp"
            continue

        kept_mask = prune_fn(g_local, W, threshold=npmi_threshold)
        removed_gene_ids = g_local[~kept_mask]

        if removed_gene_ids.size == 0:
            df.loc[comp_mask, "unassigned_qc_status"] = "keep_unassigned_comp"
            continue

        removed_set = set(map(int, removed_gene_ids.tolist()))
        # drop transcripts whose gene is in removed_set (only among unassigned comp)
        drop_gene_mask = comp_mask & gene_idx_all.isin(removed_set)

        df.loc[comp_mask & (~drop_gene_mask), "unassigned_qc_status"] = "keep_unassigned_comp"
        df.loc[drop_gene_mask, "unassigned_qc_status"] = "drop_npmi_pruned_gene"

    # Build final cell id:
    # - assigned: keep original/partial ID (from unassigned_final_col if exists)
    # - unassigned and kept comp: comp id
    # - dropped: "DROP"
    cell_id_final = assigned_id_series.copy()

    # for unassigned kept
    kept_unassigned = is_unassigned & (df["unassigned_qc_status"] == "keep_unassigned_comp")
    cell_id_final.loc[kept_unassigned] = df.loc[kept_unassigned, "unassigned_comp_id"].astype(str)

    # for dropped
    dropped = is_unassigned & df["unassigned_qc_status"].isin(["drop_small_comp", "drop_npmi_pruned_gene"])
    cell_id_final.loc[dropped] = "DROP"

    df["cell_id_final"] = cell_id_final

    return df

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
    cent = df.groupby(entity_col)[list(coord_cols)].mean()

    # unique genes per entity
    genes = df.groupby(entity_col)[gene_col].unique()

    etype = df.groupby(entity_col)["_etype"].first()

    summary = cent.join(genes.rename("genes")).join(etype.rename("etype"))
    summary = summary.reset_index().rename(columns={entity_col: "entity_id"})
    return summary


# ----------------------------
# Build Delaunay edges (3D/2D)
# ----------------------------
def delaunay_edges(points: np.ndarray):
    """
    points: (N, D) with D=2 or 3
    Returns a list of undirected edges (i, j) with i<j
    """
    tri = Delaunay(points)
    simplices = tri.simplices  # (n_simp, D+1)
    edges = set()
    for simp in simplices:
        simp = list(simp)
        for a in range(len(simp)):
            for b in range(a + 1, len(simp)):
                i, j = simp[a], simp[b]
                if i > j:
                    i, j = j, i
                edges.add((i, j))
    return sorted(edges)


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
    penalize_simplicity=True,
    deltaC_min=0.0,
    use_3d=True,
):
    """
    summary_df columns required:
      - entity_id
      - x,y,z (or x,y if use_3d=False)
      - genes (np.ndarray[str])
      - etype in {'cell','partial','component'}

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
        g = g[~pd.isna(g)].astype(int).unique()
        gene_id_lists.append(np.asarray(g, dtype=np.int32))

    # points
    if use_3d:
        pts = summary_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    else:
        pts = summary_df[["x", "y"]].to_numpy(dtype=np.float64)

    N = len(entity_ids)
    if N <= 1:
        return {entity_ids[0]: entity_ids[0]}, {}

    # Delaunay edges
    edges = delaunay_edges(pts)

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
    def compute_deltaC_roots(ra, rb):
        return deltaC_between_clusters(
            root_genes[ra],
            root_genes[rb],
            npmi_mat,
            purity_threshold=purity_threshold,
            penalize_simplicity=penalize_simplicity,
        )

    # max-heap of candidate edges by deltaC (lazy updates)
    heap = []
    for i, j in edges:
        di = compute_deltaC_roots(i, j)
        if np.isfinite(di) and di >= deltaC_min:
            heapq.heappush(heap, (-di, i, j))

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
                heapq.heappush(heap, (-dtry, rr, rn))

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
    penalize_simplicity=True,
    deltaC_min=0.0,
    use_3d=True,
    out_col="cell_id_stitched",
):
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
        penalize_simplicity=penalize_simplicity,
        deltaC_min=deltaC_min,
        use_3d=use_3d,
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

#
def calculate_rankings(
    df,
    *,
    min_support=5,
    npmi_thresh=-0.1,
    score_power_npmi=3,
    score_power_cond=2,
):
    """
    Efficiently compute ranking scores for gene–gene pairs based on NPMI
    and conditional probabilities.

    Returns a sorted DataFrame with ranking scores and QC flags.
    """

    # Select only required columns 
    cols = [
        "gene_i", "gene_j",
        "P_i", "P_j", "P_ij",
        "NPMI", "P_j_given_i", "P_i_given_j",
        "count_ij",
    ]
    df = df[cols].copy()

    # Replace NaN NPMI safely (vectorized)
    npmi = df["NPMI"].to_numpy(copy=False)
    npmi = np.nan_to_num(npmi, nan=-1.0)
    df["NPMI"] = npmi

    # Drop mask (vectorized boolean logic)
    drop = (
        (df["count_ij"].values < min_support) |
        (npmi <= npmi_thresh)
    )
    df["drop"] = drop

    # Compute conditional probability once
    cond_prob = np.minimum(
        df["P_j_given_i"].values,
        df["P_i_given_j"].values,
    )
    df["cond_prob"] = cond_prob

    # Compute score ONLY for non-dropped rows
    # (huge speedup if many rows are filtered)
    score = np.zeros(len(df), dtype=np.float64)

    valid = ~drop
    score[valid] = (
        (npmi[valid] ** score_power_npmi) *
        (cond_prob[valid] ** score_power_cond)
    )

    df["score"] = score

    # Sort once (descending)
    df.sort_values("score", ascending=False, inplace=True)

    return df

#
def calculate_thresholds(
    ranked_df,
    *,
    min_support=5,
    top_ind=4,
    min_npmi_thresh=0.01,
    max_npmi_thresh=0.1,
    min_cond_prob_thresh=0.001,
    max_cond_prob_thresh=0.01,
):
    """
    Efficiently compute gene-adaptive thresholds for NPMI and conditional probability
    and map them back to ranked_df.

    Returns
    -------
    ranked_df : DataFrame
        Original DataFrame with two new columns:
          - NPMI_thresh
          - cond_prob_thresh
    """

    # Filter once by support
    df = ranked_df[ranked_df["count_ij"] >= min_support]

    # NPMI thresholds (one global sort)
    df_npmi = df.sort_values(
        ["gene_i", "NPMI"],
        ascending=[True, False],
        kind="mergesort",   # stable, faster for groupby
    )

    npmi_thresh = (
        df_npmi
        .groupby("gene_i", sort=False)
        .apply(
            lambda g: g["NPMI"].iloc[top_ind]
            if len(g) > top_ind
            else g["NPMI"].iloc[-1]
        )
        .clip(lower=min_npmi_thresh, upper=max_npmi_thresh)
    )


    # Conditional probability thresholds
    df_cp = df.sort_values(
        ["gene_i", "cond_prob"],
        ascending=[True, False],
        kind="mergesort",
    )

    cond_prob_thresh = (
        df_cp
        .groupby("gene_i", sort=False)
        .apply(
            lambda g: g["cond_prob"].iloc[top_ind]
            if len(g) > top_ind
            else g["cond_prob"].iloc[-1]
        )
        .clip(lower=min_cond_prob_thresh, upper=max_cond_prob_thresh)
    )
    
    # Map thresholds back (vectorized)
    ranked_df = ranked_df.copy()
    
    ranked_df["NPMI_thresh"] = ranked_df["gene_i"].map(npmi_thresh)
    ranked_df["cond_prob_thresh"] = ranked_df["gene_i"].map(cond_prob_thresh)

    return ranked_df

#
def add_edge_prob_stats(
    data,
    ranked_df,
):
    """
    Attach NPMI and conditional probability to edges in a PyG Data object.
    
    Semantics:
    - Supported gene pairs → real NPMI / cond_prob
    - drop == True         → neutral (0.0)
    - missing pairs        → neutral (0.0)
    """
    src, tgt = data.edge_index.numpy()
    genes = data.gene_name

    # build lookup once
    key = ranked_df["gene_i"].astype(str) + "||" + ranked_df["gene_j"].astype(str)

    npmi_vals = ranked_df["NPMI"].to_numpy()
    cond_vals = np.minimum(
        ranked_df["P_j_given_i"].to_numpy(),
        ranked_df["P_i_given_j"].to_numpy(),
    )
    drop_vals = ranked_df["drop"].to_numpy()

    # For dropped pairs → neutral (0)
    npmi_vals = np.where(drop_vals, 0.0, npmi_vals)
    cond_vals = np.where(drop_vals, 0.0, cond_vals)

    npmi_map = dict(zip(key, npmi_vals))
    cond_map = dict(zip(key, cond_vals))

    edge_keys = genes[src] + "||" + genes[tgt]

    npmis = np.fromiter(
        (npmi_map.get(k, 0.0) for k in edge_keys),
        dtype=np.float32,
        count=len(edge_keys),
    )
    cond_probs = np.fromiter(
        (cond_map.get(k, 0.0) for k in edge_keys),
        dtype=np.float32,
        count=len(edge_keys),
    )

    data.npmi = torch.from_numpy(npmis)
    data.cond_prob = torch.from_numpy(cond_probs)

    return data

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

#
def build_gene_threshold_maps_from_ranked_df(
    ranked_df,
    *,
    min_npmi_thresh=0.01,
    max_npmi_thresh=0.2,
    min_cond_prob_thresh=0.001,
    max_cond_prob_thresh=0.2,
):
    """
    Build per-gene threshold lookup tables from ranked_df.
    """

    # One row per gene_i is enough (thresholds already mapped)
    df_unique = ranked_df[["gene_i", "NPMI_thresh", "cond_prob_thresh"]].drop_duplicates("gene_i")

    npmi_thr = dict(
        zip(
            df_unique["gene_i"],
            df_unique["NPMI_thresh"]
            .clip(lower=min_npmi_thresh, upper=max_npmi_thresh),
        )
    )

    cond_thr = dict(
        zip(
            df_unique["gene_i"],
            df_unique["cond_prob_thresh"]
            .clip(lower=min_cond_prob_thresh, upper=max_cond_prob_thresh),
        )
    )

    return npmi_thr, cond_thr

#
def prune_graph(
    G,
    ranked_df,
    *,
    distance_thresholds=(0.5, 1.5),
    directed=False,
    min_npmi_thresh=0.01,
    max_npmi_thresh=0.1,
    min_cond_prob_thresh=0.001,
    max_cond_prob_thresh=0.01,
    prune_npmi_thresh=-0.1,
):
    """
    Prune edges from graph G using gene-adaptive thresholds stored in ranked_df.

    Mutates G in place.
    """

    # Cache node → gene mapping
    node_gene = {
        n: data.get("feature_name")
        for n, data in G.nodes(data=True)
    }

    # Build gene → threshold lookup once
    npmi_thr, cond_thr = build_gene_threshold_maps_from_ranked_df(
        ranked_df,
        min_npmi_thresh=min_npmi_thresh,
        max_npmi_thresh=max_npmi_thresh,
        min_cond_prob_thresh=min_cond_prob_thresh,
        max_cond_prob_thresh=max_cond_prob_thresh,
    )

    default_npmi_high = max_npmi_thresh
    default_cond_high = max_cond_prob_thresh

    d_lo, d_hi = distance_thresholds

    # Single tight edge loop
    edges_to_remove = []

    for u, v, attr in G.edges(data=True):

        gi = node_gene[u]

        npmi = attr.get("npmi", 0.0)
        cond = attr.get("cond_prob", 0.0)
        dist = attr.get("length", float("inf"))

        npmi_hi = npmi_thr.get(gi, default_npmi_high)
        cond_hi = cond_thr.get(gi, default_cond_high)

        # Strong positive edge → keep + mark
        if (
            npmi > npmi_hi and
            cond > cond_hi and
            dist < d_lo
        ):
            attr["connect"] = 1
            continue

        # Prune edge
        if (
            npmi <= prune_npmi_thresh or
            cond <= min_cond_prob_thresh or
            dist >= d_hi
        ):
            edges_to_remove.append((u, v))

    # Bulk removal
    G.remove_edges_from(edges_to_remove)

    return G

#
def build_npmi_matrix_from_long(npmi_long):
    genes = np.union1d(npmi_long["gene_i"].unique(), npmi_long["gene_j"].unique())
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    G = len(genes)

    npmi_mat = np.zeros((G, G), dtype=np.float32)
    for row in npmi_long.itertuples(index=False):
        i = gene_to_idx[row.gene_i]
        j = gene_to_idx[row.gene_j]
        npmi_mat[i, j] = row.NPMI
        npmi_mat[j, i] = row.NPMI

    col_idx = np.arange(G, dtype=np.int32)
    return genes, gene_to_idx, npmi_mat, col_idx

#
def compute_purity_conflict_per_cc(M, npmi_mat, col_idx, purity_threshold=0.05):
    """
    Compute purity & conflict per row of M (each row = connected component).

    For k = 1:
        purity = 0
        conflict = 0
    For k >= 2:
        purity = frac(NPMI > threshold)
        conflict = mean(|negative NPMI|)
    """
    n_cells = M.shape[0]
    purity = np.zeros(n_cells, dtype=np.float32)
    conflict = np.zeros(n_cells, dtype=np.float32)

    for i in range(n_cells):
        present = np.where(M[i] == 1)[0]
        k = len(present)

        if k < 2:
            purity[i] = 0.0
            conflict[i] = 0.0
            continue

        gi = col_idx[present]
        sub = npmi_mat[np.ix_(gi, gi)]
        vals = sub[np.triu_indices_from(sub, k=1)]

        if vals.size == 0:
            purity[i] = 0.0
            conflict[i] = 0.0
            continue

        purity[i] = float(np.mean(vals > purity_threshold))

        neg = -vals[vals < 0]
        K = k * (k - 1) / 2
        conflict[i] = float(neg.sum() / K) if K > 0 else 0.0

    return purity, conflict

#
def purity_conflict_from_cc(
    G_pruned,
    npmi_long,
    df_local,
    *,
    purity_threshold=0.05,
    return_matrix=False,
    min_cc_size=1,
    return_node_mapping=True,   
):
    """
    Given a pruned transcript graph (NetworkX) where each node has attribute
    'feature_name' (gene), compute purity & conflict per connected component (CC).

    Additionally computes:
      - centroid (x, y, z) per CC
      - assigned original cell_id (majority vote)
      - node_to_component mapping (optional)

    Parameters
    ----------
    G_pruned : networkx.Graph
        Pruned transcript graph. Node IDs must align with df_local index.
    npmi_long : pd.DataFrame
        Long-form NPMI table with columns: gene_i, gene_j, NPMI.
    df_local : pd.DataFrame
        Transcript-level DataFrame with columns:
        ['x', 'y', 'z', 'cell_id', 'feature_name'].
    purity_threshold : float
        Threshold used for purity scoring.
    return_matrix : bool
        If True, also return (M_cc, genes).
    min_cc_size : int
        Skip connected components smaller than this.
    return_node_mapping : bool
        If True, return node_to_component mapping.

    Returns
    -------
    summary_df : pd.DataFrame
        Per-CC metrics including centroid and assigned cell_id.
    M_cc : np.ndarray (optional)
        CC × gene presence matrix.
    genes : np.ndarray (optional)
        Gene ordering for M_cc.
    node_to_component : dict (optional)
        Mapping: transcript_node_id → component_id
    """

    # Build NPMI matrix
    genes = sorted(set(npmi_long["gene_i"]).union(npmi_long["gene_j"]))
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    Gsize = len(genes)

    npmi_mat = np.zeros((Gsize, Gsize), dtype=float)
    for row in npmi_long.itertuples(index=False):
        i = gene_to_idx[row.gene_i]
        j = gene_to_idx[row.gene_j]
        npmi_mat[i, j] = row.NPMI
        npmi_mat[j, i] = row.NPMI

    col_idx = np.arange(Gsize)

    # Extract connected components
    if G_pruned.is_directed():
        comps_raw = list(nx.weakly_connected_components(G_pruned))
    else:
        comps_raw = list(nx.connected_components(G_pruned))

    # filter by size
    comps = [c for c in comps_raw if len(c) >= min_cc_size]
    n_cc = len(comps)

    # Node → component mapping  
    node_to_component = {}
    for comp_id, nodes in enumerate(comps):
        for n in nodes:
            node_to_component[n] = comp_id

    # Allocate outputs
    M_cc = np.zeros((n_cc, Gsize), dtype=np.int8)
    n_nodes = np.zeros(n_cc, dtype=np.int32)
    n_unique_genes = np.zeros(n_cc, dtype=np.int32)

    centroid = np.zeros((n_cc, 3), dtype=np.float32)
    assigned_cell = np.empty(n_cc, dtype=object)
    cell_id_purity = np.zeros(n_cc, dtype=np.float32)


    # Populate CC statistics
    for i, nodes in enumerate(comps):
        nodes = np.fromiter(nodes, dtype=np.int64)
        n_nodes[i] = len(nodes)

        # centroid
        coords = df_local.loc[nodes, ["x", "y", "z"]].to_numpy()
        centroid[i] = coords.mean(axis=0)

        # cell_id majority vote
        cell_ids = df_local.loc[nodes, "cell_id"].to_numpy()
        counter = Counter(cell_ids)
        major_cell, major_count = counter.most_common(1)[0]
        assigned_cell[i] = major_cell
        cell_id_purity[i] = major_count / len(nodes)

        # gene presence
        gset = set()
        for n in nodes:
            g = G_pruned.nodes[n].get("feature_name")
            if g in gene_to_idx:
                gset.add(g)

        n_unique_genes[i] = len(gset)
        for g in gset:
            M_cc[i, gene_to_idx[g]] = 1


    # Purity / conflict scoring
    purity, conflict = compute_purity_conflict_per_cc(
        M=M_cc,
        npmi_mat=npmi_mat,
        col_idx=col_idx,
        purity_threshold=purity_threshold,
    )

    # Build summary table
    summary_df = pd.DataFrame({
        "component_id": np.arange(n_cc, dtype=int),
        "n_nodes": n_nodes,
        "n_unique_genes": n_unique_genes,
        "purity": purity,
        "conflict": conflict,
        "centroid_x": centroid[:, 0],
        "centroid_y": centroid[:, 1],
        "centroid_z": centroid[:, 2],
        "assigned_cell_id": assigned_cell,
        "cell_id_purity": cell_id_purity,
    }).sort_values(
        ["n_nodes", "purity"],
        ascending=[False, False],
    ).reset_index(drop=True)


    # Return logic
    outputs = [summary_df]

    if return_matrix:
        outputs.extend([M_cc, np.array(genes)])

    if return_node_mapping:
        outputs.append(node_to_component)

    return tuple(outputs) if len(outputs) > 1 else summary_df

#
def build_cc_delaunay_graph(
    summary_df,
    *,
    use_3d=True,
):
    """
    Build a CC-level Delaunay graph from centroid coordinates.

    Nodes
    -----
    component_id

    Edges
    -----
    Delaunay adjacency edges with:
      - length (µm) [metadata only]

    NOTE:
    - No distance thresholding.
    - Geometry defines adjacency only, not scoring.
    """

    # Extract centroids
    if use_3d:
        pts = summary_df[["centroid_x", "centroid_y", "centroid_z"]].to_numpy()
    else:
        pts = summary_df[["centroid_x", "centroid_y"]].to_numpy()

    comp_ids = summary_df["component_id"].to_numpy()

    # Delaunay triangulation
    if len(pts) < (4 if use_3d else 3):
        G_cc = nx.Graph()
        G_cc.add_nodes_from(comp_ids.astype(int))
        return G_cc

    delaunay = Delaunay(pts)

    # Build graph
    G_cc = nx.Graph()
    G_cc.add_nodes_from(comp_ids.astype(int))

    for simplex in delaunay.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                u_idx = simplex[i]
                v_idx = simplex[j]

                u = int(comp_ids[u_idx])
                v = int(comp_ids[v_idx])

                if u == v:
                    continue

                d = float(euclidean(pts[u_idx], pts[v_idx]))
                G_cc.add_edge(u, v, length=d)

    return G_cc

#
def compute_deltaC_stitch(
    G_frag,
    M,
    npmi_mat,
    col_idx,
    *,
    purity_threshold=0.05,
    penalize_simplicity=True,
):
    """
    Computes ΔC for stitching using gene coherence only.

    ΔC = C_union - max(C_u, C_v)          (no simplicity)
    ΔC = C_union - 1/n_union - max(C_u - 1/n_u, C_v - 1/n_v)  (with simplicity)

    IMPORTANT:
    - Singlet–singlet stitching is forbidden.
    """

    purity, conflict = compute_purity_conflict_per_cc(
        M, npmi_mat, col_idx, purity_threshold
    )
    C_cell = purity - conflict

    n_genes = np.sum(M == 1, axis=1)
    dC = {}

    for u, v, data in G_frag.edges(data=True):


        # Missing guard
        if np.isnan(C_cell[u]) or np.isnan(C_cell[v]):
            data["deltaC"] = np.nan
            dC[(u, v)] = np.nan
            continue


        # Prevent singlet–singlet
        if n_genes[u] == 1 and n_genes[v] == 1:
            data["deltaC"] = -np.inf
            dC[(u, v)] = -np.inf
            continue

        # Union coherence
        present_union = np.where((M[u] == 1) | (M[v] == 1))[0]
        k = len(present_union)

        if k < 2:
            C_union = 0.0
        else:
            gi = col_idx[present_union]
            sub = npmi_mat[np.ix_(gi, gi)]
            vals = sub[np.triu_indices_from(sub, k=1)]

            if len(vals) == 0:
                C_union = 0.0
            else:
                purity_union = np.mean(vals > purity_threshold)
                neg = -vals[vals < 0]
                conflict_union = neg.mean() if len(neg) > 0 else 0.0
                C_union = purity_union - conflict_union

        # Separation / penalty
        if penalize_simplicity:
            nu = max(n_genes[u], 1)
            nv = max(n_genes[v], 1)
            n_union = nu + nv

            C_u_adj = C_cell[u] - 1.0 / nu
            C_v_adj = C_cell[v] - 1.0 / nv
            C_sep = max(C_u_adj, C_v_adj)

            deltaC = C_union - (1.0 / n_union) - C_sep
        else:
            C_sep = max(C_cell[u], C_cell[v])
            deltaC = C_union - C_sep

        data["deltaC"] = float(deltaC)
        dC[(u, v)] = float(deltaC)

    return dC, C_cell, purity, conflict

#
def stitch_connected_components(
    summary_df,
    M_cc,
    npmi_mat,
    col_idx,
    *,
    purity_threshold=0.05,
    penalize_simplicity=True,
    use_3d=True,
):
    """
    Perform ΔC-based greedy stitching on CC-level graph
    using gene coherence only.
    """

    # Build CC-level Delaunay graph
    G_cc = build_cc_delaunay_graph(summary_df, use_3d=use_3d)

    # Compute ΔC on CC graph
    dC, C_cell, purity, conflict = compute_deltaC_stitch(
        G_frag=G_cc,
        M=M_cc,
        npmi_mat=npmi_mat,
        col_idx=col_idx,
        purity_threshold=purity_threshold,
        penalize_simplicity=penalize_simplicity,
    )

    # Keep only stitchable edges
    G_pos = nx.Graph()
    for u, v, data in G_cc.edges(data=True):
        if data.get("deltaC", -np.inf) >= 0:
            G_pos.add_edge(u, v, deltaC=data["deltaC"])

    # Assign stitched CC IDs
    stitched_cc_id = {}
    deltaC_max = {}
    next_id = 0

    for comp in nx.connected_components(G_pos):
        comp = list(comp)
        for u in comp:
            stitched_cc_id[u] = next_id
            vals = [G_pos[u][v]["deltaC"] for v in G_pos.neighbors(u)]
            deltaC_max[u] = max(vals) if vals else np.nan
        next_id += 1

    # isolated CCs
    for u in G_cc.nodes():
        if u not in stitched_cc_id:
            stitched_cc_id[u] = next_id
            deltaC_max[u] = np.nan
            next_id += 1

    # Attach back
    summary_df_out = summary_df.copy()
    summary_df_out["stitched_cc_id"] = summary_df_out["component_id"].map(stitched_cc_id)
    summary_df_out["deltaC_max"] = summary_df_out["component_id"].map(deltaC_max)

    return summary_df_out, G_cc
