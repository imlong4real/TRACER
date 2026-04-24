"""Phase 3/5/6: spatial analysis — components, coherence enforcement, reassignment."""

import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as scipy_cc
from sklearn.neighbors import NearestNeighbors

from . import _cy_prune, _cy_spatial
from ._repro import _ensure_reproducibility_seed
from ._utils import prepare_transcript_df
from .graph import build_graph, to_networkx  # noqa: F401 — used internally/callers


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
    _ensure_reproducibility_seed()
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
    # Deterministic component ordering across Python hash seeds
    components = sorted(components, key=lambda comp: min(comp))

    # Map node -> component index
    num_nodes = df_u.shape[0]
    comp_idx = np.full(num_nodes, -1, dtype=np.int32)
    for ci, comp in enumerate(components):
        comp_idx[sorted(comp)] = ci
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
    large_comp_ids = np.sort(df.loc[keep_candidate, "unassigned_comp_id"].unique())

    # precompute gene idx per transcript (NaN if gene missing from NPMI table)
    gene_idx_all = df[gene_col].map(gene_to_idx)

    for comp_id in large_comp_ids:
        comp_mask = (df["unassigned_comp_id"] == comp_id) & keep_candidate
        if comp_mask.sum() == 0:
            continue

        # unique genes in this component (only those present in NPMI gene_to_idx)
        g_local = np.sort(gene_idx_all.loc[comp_mask].dropna().astype(int).unique())
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


def annotate_unassigned_components_fast(
    df_pruned: pd.DataFrame,
    aux: dict,
    *,
    build_graph_fn,
    prune_fn,
    coord_cols=("x", "y", "z"),
    k=8,
    dist_threshold=1.5,
    min_comp_size=50,
    npmi_threshold=-0.1,
    unassigned_final_col="cell_id_npmi_cons_p2",
    cell_id_col="cell_id",
    gene_col="feature_name",
    transcript_id_col="transcript_id",
    show_progress: bool = True,
    in_place: bool = False,
):
    """
    Faster variant of `annotate_unassigned_components`.
    - Uses `_cy_prune.prune_cells` when available to prune per-component gene lists in bulk.
    - Shows a progress bar if `show_progress` is True.
    - `in_place`: if True, mutate the input DataFrame instead of copying
      it. Saves ~N bytes × row-count on the transient copy.
    """
    _ensure_reproducibility_seed()
    if not in_place:
        df = df_pruned.copy()
    else:
        df = df_pruned

    if transcript_id_col not in df.columns:
        df[transcript_id_col] = df.index.astype(str)

    # feature_name → Categorical (no-op if already); dropped the redundant
    # `.astype(str).str.strip()` that scanned all 100M+ rows twice.
    prepare_transcript_df(df, gene_col=gene_col)

    if unassigned_final_col in df.columns:
        is_unassigned = df[unassigned_final_col].astype(str) == "-1"
        assigned_id_series = df[unassigned_final_col].astype(str)
    else:
        is_unassigned = df[cell_id_col].astype(str) == "-1"
        assigned_id_series = df[cell_id_col].astype(str)

    # `unassigned_qc_status` only ever takes one of 4 sentinel values —
    # declare the full vocabulary up front so the column stays categorical
    # through every `.loc[..., col] = "…"` assignment below. Drops memory
    # from ~35 MiB to ~1.5 MiB at 1.4M rows (and ~3 GB at 100M).
    _QC_STATUS_CATS = [
        "unassigned_raw", "drop_small_comp",
        "drop_npmi_pruned_gene", "keep_unassigned_comp",
    ]
    df["unassigned_comp_id"] = pd.Series(index=df.index, dtype="object")
    df["unassigned_qc_status"] = pd.Categorical(
        [None] * len(df), categories=_QC_STATUS_CATS,
    )
    df.loc[is_unassigned, "unassigned_qc_status"] = "unassigned_raw"

    if is_unassigned.sum() == 0:
        df["cell_id_final"] = assigned_id_series
        return df

    df_u = df.loc[is_unassigned].copy()

    data_u = build_graph_fn(
        df_u,
        k=k,
        dist_threshold=dist_threshold,
        coord_cols=coord_cols,
    )

    # Use scipy's faster connected components detection
    if show_progress:
        pbar_cc = tqdm(total=3, desc="unassigned_analysis")

    num_nodes = data_u.num_nodes
    edge_index = data_u.edge_index.numpy()

    if show_progress:
        pbar_cc.update(1)
        pbar_cc.set_description("building_cc_matrix")

    # Build sparse adjacency matrix (undirected, so add both directions)
    rows = np.concatenate([edge_index[0], edge_index[1]])
    cols = np.concatenate([edge_index[1], edge_index[0]])
    data_sp = np.ones(len(rows), dtype=np.float32)
    adj_matrix = csr_matrix((data_sp, (rows, cols)), shape=(num_nodes, num_nodes))

    if show_progress:
        pbar_cc.update(1)
        pbar_cc.set_description("computing_cc")

    n_comps, comp_labels = scipy_cc(adj_matrix, directed=False, return_labels=True)

    if show_progress:
        pbar_cc.update(1)
        pbar_cc.set_description("post_cc_mapping")

    # Efficiently convert comp_labels directly to comp_idx (skip intermediate set creation)
    num_nodes = df_u.shape[0]
    comp_idx = comp_labels.astype(np.int32)  # comp_labels already assigns each node to a component

    if show_progress:
        pbar_cc.update(1)
        pbar_cc.close()

    comp_ids_str = np.array([f"UNASSIGNED_{i}" for i in comp_idx], dtype=object)
    df.loc[df_u.index, "unassigned_comp_id"] = comp_ids_str

    comp_sizes = pd.Series(comp_idx).value_counts().sort_index()
    comp_size_map = {f"UNASSIGNED_{i}": int(sz) for i, sz in comp_sizes.items()}
    df["unassigned_comp_size"] = df["unassigned_comp_id"].map(comp_size_map)

    drop_small = is_unassigned & df["unassigned_comp_size"].notna() & (df["unassigned_comp_size"] < min_comp_size)
    df.loc[drop_small, "unassigned_qc_status"] = "drop_small_comp"

    W = aux["W"]
    gene_to_idx = aux["gene_to_idx"]
    gene_idx_all = df[gene_col].map(gene_to_idx)

    keep_candidate = is_unassigned & (~drop_small) & df["unassigned_comp_id"].notna()
    large_comp_ids = np.sort(df.loc[keep_candidate, "unassigned_comp_id"].unique())

    # Prepare per-component gene lists using vectorized groupby (much faster than per-component filtering)
    if show_progress:
        pbar_groupby = tqdm(total=2, desc="grouping_genes")

    comp_gene_map = {}
    df_candidate = df.loc[keep_candidate].copy()
    df_candidate["_gene_idx_local"] = gene_idx_all.loc[keep_candidate]

    if show_progress:
        pbar_groupby.update(1)
        pbar_groupby.set_description("grouping")

    # Group by component and get unique gene indices per component
    for comp_id, group in df_candidate.groupby("unassigned_comp_id", sort=True):
        g_local = np.sort(group["_gene_idx_local"].dropna().astype(int).unique())
        if g_local.size > 0:
            comp_gene_map[comp_id] = np.asarray(g_local, dtype=np.int32)

    if show_progress:
        pbar_groupby.update(1)
        pbar_groupby.close()

    # Bulk-prune all components through the Cython kernel. Python fallback
    # removed — `_cy_prune` is a hard import now, so the only reason this
    # would fail is a real bug worth surfacing.
    if len(comp_gene_map) > 0:
        comp_keys = sorted(comp_gene_map.keys())
        g_arrays = [comp_gene_map[k] if comp_gene_map[k].size > 0 else None for k in comp_keys]
        removed_lists = _cy_prune.prune_cells(g_arrays, W, float(npmi_threshold))

        iterator = zip(comp_keys, removed_lists)
        if show_progress:
            iterator = tqdm(list(iterator), desc="prune_comps")

        # Vectorized approach: build a mask for genes to drop, then apply once
        drop_gene_mask_all = np.zeros(len(df), dtype=bool)

        for comp_id, removed in iterator:
            if removed is None or len(removed) == 0:
                # Mark all transcripts in this component as kept
                comp_mask = (df["unassigned_comp_id"] == comp_id) & keep_candidate
                df.loc[comp_mask, "unassigned_qc_status"] = "keep_unassigned_comp"
                continue

            # Get indices of transcripts in this component
            comp_indices = np.where((df["unassigned_comp_id"] == comp_id) & keep_candidate)[0]

            if len(comp_indices) == 0:
                continue

            removed_set = set(map(int, removed))
            comp_gene_mask = gene_idx_all.iloc[comp_indices].isin(removed_set).to_numpy()
            drop_gene_mask_all[comp_indices[comp_gene_mask]] = True

            df.loc[comp_indices[~comp_gene_mask], "unassigned_qc_status"] = "keep_unassigned_comp"

        df.loc[drop_gene_mask_all, "unassigned_qc_status"] = "drop_npmi_pruned_gene"

    cell_id_final = assigned_id_series.copy()
    kept_unassigned = is_unassigned & (df["unassigned_qc_status"] == "keep_unassigned_comp")
    cell_id_final.loc[kept_unassigned] = df.loc[kept_unassigned, "unassigned_comp_id"].astype(str)

    dropped = is_unassigned & df["unassigned_qc_status"].isin(["drop_small_comp", "drop_npmi_pruned_gene"])
    cell_id_final.loc[dropped] = "DROP"

    df["cell_id_final"] = cell_id_final

    return df


# ---------- Phase 5: Finetune assignment based on spatial coherence ----------
def enforce_spatial_coherence(
    df_stitched: pd.DataFrame,
    build_graph_fn,
    *,
    entity_col: str = "cell_id_stitched",
    coord_cols=("x", "y", "z"),
    k: int = 5,
    dist_threshold: float = 3.0,
    out_col: str = "cell_id_spatial",
):
    """
    For each entity in `entity_col` (cell / partial / pseudo-cell),
    check if its transcripts are spatially connected in a kNN graph.

    - Build ONE global kNN graph over all transcripts with k=5, dist_threshold=3.
    - For each entity label (excluding DROP / NaN), restrict to transcripts
      with that label and compute connected components on the induced subgraph.
    - If >1 component:
         * largest component keeps original label
         * others get new labels: f"{label}-2", f"{label}-3", ...
    - Returns a new df with an added column `out_col` containing
      the split-aware labels.
    """
    _ensure_reproducibility_seed()
    df = df_stitched.copy()

    # base labels we are checking
    base_labels = df[entity_col].astype(str)

    # Initialize spatial label as the stitched label
    df[out_col] = base_labels

    # Build graph once on ALL transcripts
    # (we assume df has x,y,z and transcript_id as required by build_graph_fn)
    df = df.reset_index(drop=True)
    df["__node_idx"] = np.arange(len(df), dtype=int)

    data_all = build_graph_fn(
        df,
        k=k,
        dist_threshold=dist_threshold,
        coord_cols=coord_cols,
    )
    G = to_networkx(data_all, to_undirected=True)

    # For each label, check connectivity in induced subgraph
    labels = np.sort(base_labels.unique())
    for label in labels:
        if label == "DROP" or label == "nan":
            continue

        mask = (base_labels == label)
        node_idx = df.loc[mask, "__node_idx"].to_numpy()

        if node_idx.size <= 1:
            continue

        # induced subgraph on these nodes only
        subG = G.subgraph(node_idx)
        comps = list(nx.connected_components(subG))

        if len(comps) <= 1:
            continue  # spatially coherent

        # sort by size descending, tie-break by min node id for determinism
        comps_sorted = sorted(comps, key=lambda c: (-len(c), min(c)))

        # largest keeps original label; others get label-2, label-3, ...
        for i, comp_nodes in enumerate(comps_sorted):
            if i == 0:
                new_label = label
            else:
                new_label = f"{label}-{i+1}"

            comp_nodes = np.array(sorted(comp_nodes), dtype=int)
            # mark these transcripts with new_label
            df.loc[df["__node_idx"].isin(comp_nodes), out_col] = new_label

    # cleanup
    df = df.drop(columns=["__node_idx"])
    return df


def enforce_spatial_coherence_fast(
    df_stitched: pd.DataFrame,
    build_graph_fn,
    *,
    entity_col: str = "cell_id_stitched",
    coord_cols=("x", "y", "z"),
    k: int = 5,
    dist_threshold: float = 3.0,
    out_col: str = "cell_id_spatial",
    show_progress: bool = True,
    in_place: bool = False,
):
    """
    Fast variant of `enforce_spatial_coherence` with a progress bar.

    `in_place=True` skips the defensive DataFrame copy. Safe for our
    pipeline where upstream stages hand their freshly-built df to the
    next stage — at 100M rows the copy is ~10 GB of transient overhead.
    Note: we still need to `reset_index` below, so a partial mutation
    happens even with `in_place=True`.
    """
    _ensure_reproducibility_seed()
    df = df_stitched if in_place else df_stitched.copy()
    base_labels = df[entity_col].astype(str)
    df[out_col] = base_labels
    df = df.reset_index(drop=True)
    n = len(df)
    df["__node_idx"] = np.arange(n, dtype=np.int32)

    # Build transcript graph once
    data_all = build_graph_fn(
        df,
        k=k,
        dist_threshold=dist_threshold,
        coord_cols=coord_cols,
    )

    # Compute label-constrained components in a single Cython DSU pass.
    # The previous NetworkX fallback materialised the whole graph in
    # Python and blew up at 100M-node scale; it's been removed.
    if not hasattr(data_all, "edge_index") or data_all.edge_index.numel() == 0:
        # No edges → every label is trivially one component; out_col already
        # equals base_labels from the initial assignment above.
        df = df.drop(columns=["__node_idx"])
        return df

    edge_index = data_all.edge_index.numpy()
    src = edge_index[0].astype(np.int32)
    dst = edge_index[1].astype(np.int32)

    labels_arr = base_labels.to_numpy()
    lab_codes = pd.Categorical(labels_arr)
    assert len(lab_codes.categories) < 2**31, (
        "enforce_spatial_coherence_fast: label vocabulary exceeds int32 range"
    )
    codes = lab_codes.codes.astype(np.int64)
    invalid = (labels_arr == "DROP") | (labels_arr == "nan")
    if invalid.any():
        codes = codes.copy()
        codes[invalid] = -1

    roots = _cy_spatial.label_constrained_components(int(n), src, dst, codes, -1)

    # For each label code, split by root and assign suffixes to non-largest comps
    out = df[out_col].to_numpy(dtype=object)
    uniq_codes = np.unique(codes)
    uniq_codes = uniq_codes[uniq_codes >= 0]
    iterator = uniq_codes
    if show_progress:
        iterator = tqdm(uniq_codes, desc="spatial_labels")
    for c in iterator:
        idx = np.where(codes == c)[0]
        if idx.size <= 1:
            continue
        roots_c = roots[idx]
        uniq_r, counts = np.unique(roots_c, return_counts=True)
        if uniq_r.size <= 1:
            continue
        # Deterministic tie-break by root id when counts tie
        order = np.lexsort((uniq_r, -counts))
        lab = str(lab_codes.categories[c])
        for i, oi in enumerate(order):
            r = uniq_r[oi]
            if i == 0:
                # Largest component keeps the original label (already set)
                continue
            new_lab = f"{lab}-{i+1}"
            sel = (roots_c == r)
            out[idx[sel]] = new_lab
    df[out_col] = out
    df = df.drop(columns=["__node_idx"])
    return df

# ---------- Phase 6: Reassign unassigned transcripts to nearby partials/components ----------
def reassign_unassigned_to_nearby_entities(
    df_spatial: pd.DataFrame,
    entity_summary: pd.DataFrame,
    *,
    entity_col: str = "cell_id_spatial",
    out_col: str = "cell_id_finetuned",
    coord_cols=("x", "y", "z"),
    dist_threshold: float = 20.0,
    unassigned_labels=None,
    only_partial_component: bool = True,
    show_progress: bool = True,
    in_place: bool = False,
):
    """
    Phase 6: Reassign unassigned transcripts to the nearest "partial" or "component" entity
    if they are within a distance threshold.

    This function takes transcripts with unassigned labels (DROP, -1, UNASSIGNED, etc.) and
    finds the nearest spatial entity of type "partial" or "component", assigning them if
    within the distance threshold. This helps capture transcripts that are spatially close
    to partial/component clusters but were not assigned during earlier phases.

    Parameters
    ----------
    df_spatial : pd.DataFrame
        Transcript-level DataFrame with entity assignments (from Phase 5).
        Must contain `entity_col` and `coord_cols`.
    entity_summary : pd.DataFrame
        Entity summary table (output from build_entity_table).
        Must contain columns: entity_id, x, y, z, etype.
        If not provided, will be built from df_spatial.
    entity_col : str
        Input column with current entity labels (e.g., "cell_id_spatial").
    out_col : str
        Output column name for reassigned labels (e.g., "cell_id_finetuned").
    coord_cols : tuple
        Coordinate column names.
    dist_threshold : float
        Maximum Euclidean distance for reassignment (default 20).
    unassigned_labels : set or None
        Labels to consider as "unassigned" (e.g., {"DROP", "-1", "UNASSIGNED"}).
        If None, defaults to {"DROP", "-1", "UNASSIGNED", "nan"}.
    only_partial_component : bool
        If True, only assign to "partial" or "component" entities (not "cell").
        If False, assign to any entity type.
    show_progress : bool
        Show progress bar during processing.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with new `out_col` containing reassigned labels.
        Unassigned transcripts within threshold are reassigned.
        Unassigned transcripts beyond threshold keep original labels.
        Assigned transcripts keep their original labels.
    n_reassigned : int
        Number of transcripts that were reassigned.
    reassignment_stats : dict
        Statistics about the reassignment process:
        - total_unassigned: total unassigned transcripts found
        - total_reassigned: transcripts successfully reassigned
        - mean_distance: mean distance of reassigned transcripts
        - max_distance: maximum distance of reassigned transcripts

    Notes
    -----
    - Uses KNN with k=1 for fast nearest-entity lookup
    - Euclidean distance in coordinate space
    - Original assigned transcripts are never modified
    """
    _ensure_reproducibility_seed()
    if unassigned_labels is None:
        unassigned_labels = {"DROP", "-1", "UNASSIGNED", "nan"}

    df = df_spatial if in_place else df_spatial.copy()
    df[out_col] = df[entity_col].astype(str)

    # Identify unassigned transcripts
    labels = df[entity_col].astype(str)
    unassigned_mask = labels.isin(unassigned_labels)

    n_unassigned = unassigned_mask.sum()
    if n_unassigned == 0:
        return df, 0, {
            "total_unassigned": 0,
            "total_reassigned": 0,
            "mean_distance": np.nan,
            "max_distance": np.nan,
        }

    # Extract unassigned transcript coordinates
    unassigned_idx = np.where(unassigned_mask)[0]
    unassigned_coords = df.loc[unassigned_idx, list(coord_cols)].to_numpy(dtype=np.float32)

    # Filter entity summary
    if only_partial_component:
        entity_mask = entity_summary["etype"].isin(["partial", "component"])
        entities = entity_summary[entity_mask].copy()
    else:
        entities = entity_summary.copy()

    if len(entities) == 0:
        return df, 0, {
            "total_unassigned": n_unassigned,
            "total_reassigned": 0,
            "mean_distance": np.nan,
            "max_distance": np.nan,
        }

    # Entity coordinates and IDs
    # Drop entities with missing coordinates (centroid could be NaN if all members lacked coords)
    before_entities = len(entities)
    entities = entities.dropna(subset=list(coord_cols))
    dropped_entities = before_entities - len(entities)
    if dropped_entities > 0 and show_progress:
        print(f"Warning: dropped {dropped_entities} entities with NaN centroids before KNN build")

    entity_coords = entities[list(coord_cols)].to_numpy(dtype=np.float32)
    entity_ids = entities["entity_id"].to_numpy(dtype=object)

    # Also drop unassigned transcripts that lack valid coordinates.
    # `unassigned_idx` must be kept aligned with `unassigned_coords` — previously
    # a NaN-coords branch silently desynchronised them, causing either an
    # IndexError below or incorrect label assignment.
    if np.isnan(unassigned_coords).any():
        valid_mask = ~np.isnan(unassigned_coords).any(axis=1)
        n_invalid = int((~valid_mask).sum())
        if n_invalid > 0 and show_progress:
            print(f"Warning: {n_invalid} unassigned transcripts have NaN coordinates and will be skipped")
        unassigned_idx = unassigned_idx[valid_mask]
        unassigned_coords = unassigned_coords[valid_mask]

    # Build KNN index for fast nearest-neighbor lookup
    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", metric="euclidean")
    knn.fit(entity_coords)

    # Find nearest entity for each unassigned transcript
    distances, indices = knn.kneighbors(unassigned_coords)
    distances = distances.ravel()
    indices = indices.ravel()

    # Determine which unassigned transcripts should be reassigned
    within_threshold = distances <= dist_threshold
    n_reassigned = int(within_threshold.sum())

    # Vectorized reassignment: `unassigned_idx` is positional
    # (from np.where(...)[0]), so use .iloc with a column position to avoid
    # the O(n) per-row alignment cost of .loc[scalar, col].
    if n_reassigned:
        sel_rows = unassigned_idx[within_threshold]
        new_labels = entity_ids[indices[within_threshold]]
        col_pos = df.columns.get_loc(out_col)
        df.iloc[sel_rows, col_pos] = new_labels

    # Compute statistics
    if n_reassigned > 0:
        reassigned_distances = distances[within_threshold]
        mean_distance = float(np.mean(reassigned_distances))
        max_distance = float(np.max(reassigned_distances))
    else:
        mean_distance = np.nan
        max_distance = np.nan

    stats = {
        "total_unassigned": int(n_unassigned),
        "total_reassigned": int(n_reassigned),
        "mean_distance": mean_distance,
        "max_distance": max_distance,
    }

    if show_progress:
        print(f"Phase 6: Reassigned {n_reassigned}/{n_unassigned} unassigned transcripts "
              f"(threshold={dist_threshold})")
        if n_reassigned > 0:
            print(f"  Mean distance: {mean_distance:.2f}, Max distance: {max_distance:.2f}")

    return df, n_reassigned, stats

def reassign_unassigned_to_nearby_entities_fast(
    df_spatial: pd.DataFrame,
    entity_summary: pd.DataFrame = None,
    *,
    entity_col: str = "cell_id_spatial",
    gene_col: str = "feature_name",
    coord_cols=("x", "y", "z"),
    out_col: str = "cell_id_finetuned",
    dist_threshold: float = 20.0,
    unassigned_labels=None,
    only_partial_component: bool = True,
    show_progress: bool = True,
    in_place: bool = False,
):
    """
    Fast wrapper around reassign_unassigned_to_nearby_entities.

    Builds entity summary if not provided, then performs reassignment with progress bar.

    Parameters
    ----------
    df_spatial : pd.DataFrame
        Transcript-level DataFrame from Phase 5.
    entity_summary : pd.DataFrame, optional
        Pre-computed entity summary (from build_entity_table).
        If None, will be built from df_spatial automatically.
    entity_col : str
        Input entity column.
    gene_col : str
        Gene/feature column name.
    coord_cols : tuple
        Coordinate columns.
    out_col : str
        Output column name.
    dist_threshold : float
        Maximum distance for reassignment.
    unassigned_labels : set or None
        Unassigned labels to identify.
    only_partial_component : bool
        Only assign to partial/component.
    show_progress : bool
        Show progress.

    Returns
    -------
    df_out : pd.DataFrame
        Reassigned transcript data.
    n_reassigned : int
        Number of reassigned transcripts.
    stats : dict
        Reassignment statistics.
    """
    _ensure_reproducibility_seed()
    # Lazy import to avoid circular dependency: stitching imports graph,
    # and spatial is imported by _repro's smoke test path.
    from .stitching import build_entity_table

    if entity_summary is None:
        if show_progress:
            print("Building entity summary...")
        entity_summary = build_entity_table(
            df_spatial,
            entity_col=entity_col,
            gene_col=gene_col,
            coord_cols=coord_cols,
        )

    return reassign_unassigned_to_nearby_entities(
        df_spatial,
        entity_summary,
        entity_col=entity_col,
        out_col=out_col,
        coord_cols=coord_cols,
        dist_threshold=dist_threshold,
        unassigned_labels=unassigned_labels,
        only_partial_component=only_partial_component,
        show_progress=show_progress,
        in_place=in_place,
    )
