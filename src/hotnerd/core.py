import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
import networkx as nx
from collections import Counter

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

    print(
        f"Constructed {edge_index.shape[1]:,} edges among {N:,} transcripts "
        f"(k≤{k}, d≤{dist_threshold} µm)"
    )

    return data

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
):
    """
    Convert PyG Data → NetworkX graph and compute node connectivity.
    """
    src, tgt = data.edge_index.numpy()

    if not directed:
        mask = src < tgt
        src, tgt = src[mask], tgt[mask]

    G = nx.DiGraph() if directed else nx.Graph()

    # add nodes
    for i, g in enumerate(data.gene_name):
        G.add_node(i, feature_name=g)

    # edge lengths
    pos = data.pos.numpy()
    lengths = np.linalg.norm(pos[src] - pos[tgt], axis=1)

    # edge attributes
    for i, (s, t) in enumerate(zip(src, tgt)):
        G.add_edge(
            int(s),
            int(t),
            length=float(lengths[i]),
            npmi=float(data.npmi[i]) if hasattr(data, "npmi") else 0.0,
            cond_prob=float(data.cond_prob[i]) if hasattr(data, "cond_prob") else 0.0,
        )

    # connectivity = degree
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


