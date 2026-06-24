"""CC-level scoring: rankings, thresholds, purity/conflict, CC stitching."""

from collections import Counter

import numpy as np
import pandas as pd
import torch
import networkx as nx
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean

from ._kernels import pair_aggregate_dense


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


def compute_purity_conflict_per_cc(M, npmi_mat, col_idx, purity_threshold=0.05):
    """
    Compute purity & conflict per row of M (each row = connected component).

    For k = 1:
        purity = 0
        conflict = 0
    For k >= 2:
        purity = frac(NPMI > threshold)
        conflict = mean(|negative NPMI|)

    Implementation note: the per-row work is one parallel kernel pass
    (see `tracer._kernels.pair_aggregate_dense`); the Python double loop
    that used to live here was the bottleneck of the per-CC metrics at
    200K+ components.
    """
    k_arr, n_pos, sum_neg, _pos_relu, _neg_relu = pair_aggregate_dense(
        M, col_idx, npmi_mat, threshold=purity_threshold, tau=0.0,
    )
    n_pairs_total = k_arr * (k_arr - 1) // 2
    has_pairs = n_pairs_total > 0

    purity = np.zeros(M.shape[0], dtype=np.float32)
    conflict = np.zeros(M.shape[0], dtype=np.float32)
    # Original compute_cell_purity used `np.mean(vals > threshold)` which
    # counts ALL pairs (including NaN pairs that are False) in the denom.
    # Matching that exactly: n_pos / n_pairs_total.
    purity[has_pairs] = (n_pos[has_pairs] / n_pairs_total[has_pairs]).astype(np.float32)
    conflict[has_pairs] = (sum_neg[has_pairs] / n_pairs_total[has_pairs]).astype(np.float32)

    return purity, conflict

def compute_purity_conflict_per_cc_relu(M, npmi_mat, col_idx, tau=0.05, eps=1e-8):
    """
    ReLU-based purity & conflict computation per connected component.

    Uses symmetric ReLU to suppress weak associations and weight
    stronger evidence more heavily.

    For k = 1:
        All metrics = 0
    For k >= 2:
        - absolute_purity = sum(positive ReLU) / total_pairs
        - absolute_conflict = sum(negative ReLU) / total_pairs
        - relative_purity = positive_signal / total_signal
        - relative_conflict = negative_signal / total_signal
        - signal_strength = total magnitude of non-zero ReLU values

    Parameters
    ----------
    M : np.ndarray, shape (n_cc, n_genes)
        Binary presence/absence matrix per connected component
    npmi_mat : np.ndarray
        Full NPMI matrix
    col_idx : np.ndarray
        Gene indices mapping to NPMI matrix columns
    tau : float
        Dead-zone threshold for symmetric ReLU
    eps : float
        Minimum signal strength for computing relative metrics

    Returns
    -------
    purity : np.ndarray
        Absolute purity scores per CC
    conflict : np.ndarray
        Absolute conflict scores per CC
    relative_purity : np.ndarray
        Relative purity (fraction of total signal) per CC
    relative_conflict : np.ndarray
        Relative conflict (fraction of total signal) per CC
    signal_strength : np.ndarray
        Total signal magnitude per CC
    """
    k_arr, _n_pos, _sum_neg, pos_relu, neg_relu = pair_aggregate_dense(
        M, col_idx, npmi_mat, threshold=0.0, tau=tau,
    )
    n_pairs_total = k_arr * (k_arr - 1) // 2
    has_pairs = n_pairs_total > 0
    total_abs = pos_relu + neg_relu

    n_cells = M.shape[0]
    purity = np.zeros(n_cells, dtype=np.float32)
    conflict = np.zeros(n_cells, dtype=np.float32)
    relative_purity = np.zeros(n_cells, dtype=np.float32)
    relative_conflict = np.zeros(n_cells, dtype=np.float32)
    signal_strength = np.zeros(n_cells, dtype=np.float32)

    purity[has_pairs] = (pos_relu[has_pairs] / n_pairs_total[has_pairs]).astype(np.float32)
    conflict[has_pairs] = (neg_relu[has_pairs] / n_pairs_total[has_pairs]).astype(np.float32)
    signal_strength[has_pairs] = total_abs[has_pairs].astype(np.float32)

    has_signal = has_pairs & (total_abs > eps)
    relative_purity[has_signal] = (pos_relu[has_signal] / total_abs[has_signal]).astype(np.float32)
    relative_conflict[has_signal] = (neg_relu[has_signal] / total_abs[has_signal]).astype(np.float32)

    return purity, conflict, relative_purity, relative_conflict, signal_strength

#
def purity_conflict_from_cc(
    G_pruned,
    npmi_long,
    df_local,
    *,
    purity_threshold=0.05,
    tau=0.05,
    use_relu=True,
    eps=1e-8,
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
        Threshold used for purity scoring (only used if use_relu=False).
    tau : float
        Dead-zone threshold for symmetric ReLU (only used if use_relu=True).
    use_relu : bool
        If True, use ReLU-based scoring with relative metrics.
        If False, use original threshold-based scoring.
    eps : float
        Minimum signal strength for computing relative metrics (only used if use_relu=True).
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
        If use_relu=True, includes: purity, conflict, relative_purity,
        relative_conflict, signal_strength.
        If use_relu=False, includes: purity, conflict (original metrics).
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
    if use_relu:
        purity, conflict, relative_purity, relative_conflict, signal_strength = compute_purity_conflict_per_cc_relu(
            M=M_cc,
            npmi_mat=npmi_mat,
            col_idx=col_idx,
            tau=tau,
            eps=eps,
        )

        # Build summary table with ReLU metrics
        summary_df = pd.DataFrame({
            "component_id": np.arange(n_cc, dtype=int),
            "n_nodes": n_nodes,
            "n_unique_genes": n_unique_genes,
            "purity": purity,
            "conflict": conflict,
            "relative_purity": relative_purity,
            "relative_conflict": relative_conflict,
            "signal_strength": signal_strength,
            "centroid_x": centroid[:, 0],
            "centroid_y": centroid[:, 1],
            "centroid_z": centroid[:, 2],
            "assigned_cell_id": assigned_cell,
            "cell_id_purity": cell_id_purity,
        }).sort_values(
            ["n_nodes", "purity"],
            ascending=[False, False],
        ).reset_index(drop=True)
    else:
        purity, conflict = compute_purity_conflict_per_cc(
            M=M_cc,
            npmi_mat=npmi_mat,
            col_idx=col_idx,
            purity_threshold=purity_threshold,
        )

        # Build summary table with original metrics
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

    for comp in sorted(nx.connected_components(G_pos), key=lambda c: min(c)):
        comp = sorted(comp)
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
