"""Phase 3/5/6: spatial analysis — components, coherence enforcement, reassignment."""

import math

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
    entity_col: str = "tracer_id",
    out_col: str = "tracer_id",
    cell_id_col: str = "cell_id",
    gene_col: str = "feature_name",
    transcript_id_col: str = "transcript_id",
    debug_stages: bool = False,
    show_progress: bool = True,
    in_place: bool = False,
):
    """
    Annotate unassigned transcripts with connected-component IDs and
    NPMI-prune their genes.

    Parameters
    ----------
    entity_col : str
        Column to read for "is this transcript currently unassigned"
        ("-1") — defaults to `"tracer_id"`, the canonical pipeline
        column written by `prune_transcripts_fast`.
    out_col : str
        Where to write the updated assignment (in place by default).
        Rows that started as "-1" become either `"UNASSIGNED_<i>"`
        (kept component) or `"DROP"` (dropped).
    debug_stages : bool
        When True, additionally writes legacy snapshot columns:
        `cell_id_final` (mirrors `out_col` post-stage), plus the
        diagnostic columns `unassigned_comp_id`,
        `unassigned_qc_status`, `unassigned_comp_size`. Default False
        keeps output minimal.
    in_place : bool
        Skip the defensive `df.copy()`.
    """
    _ensure_reproducibility_seed()
    if not in_place:
        df = df_pruned.copy()
    else:
        df = df_pruned

    if transcript_id_col not in df.columns:
        df[transcript_id_col] = df.index.astype(str)

    prepare_transcript_df(df, gene_col=gene_col)

    # Read the canonical pipeline column to find unassigned transcripts.
    # Fallback to `cell_id_col` only if the pipeline column doesn't exist
    # yet (e.g., user is running stage 2 standalone on raw data).
    if entity_col in df.columns:
        is_unassigned = df[entity_col].astype(str) == "-1"
        assigned_id_series = df[entity_col].astype(str)
    elif cell_id_col in df.columns:
        is_unassigned = df[cell_id_col].astype(str) == "-1"
        assigned_id_series = df[cell_id_col].astype(str)
    else:
        raise ValueError(
            f"Neither `{entity_col}` nor `{cell_id_col}` found in df. "
            "Run `prune_transcripts_fast` first or pass an explicit "
            "`entity_col`."
        )

    # Diagnostic columns are kept locally — only attached to df when
    # `debug_stages=True`. Categorical with full vocabulary so the
    # later `.loc[..., col] = "…"` assignments don't error out.
    _QC_STATUS_CATS = [
        "unassigned_raw", "drop_small_comp",
        "drop_npmi_pruned_gene", "keep_unassigned_comp",
    ]
    qc_status = pd.Categorical([None] * len(df), categories=_QC_STATUS_CATS)
    qc_status[is_unassigned.to_numpy()] = "unassigned_raw"
    comp_id_series = pd.Series(index=df.index, dtype="object")

    if is_unassigned.sum() == 0:
        df[out_col] = assigned_id_series
        if debug_stages:
            df["cell_id_final"] = assigned_id_series
            df["unassigned_qc_status"] = qc_status
            df["unassigned_comp_id"] = comp_id_series
            df["unassigned_comp_size"] = pd.Series(np.nan, index=df.index, dtype="float32")
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

    # Components → strings, kept locally (only attached to df under
    # debug_stages). Component IDs are written to comp_id_series here.
    comp_ids_str = np.array([f"UNASSIGNED_{i}" for i in comp_idx], dtype=object)
    comp_id_series.loc[df_u.index] = comp_ids_str

    comp_sizes = pd.Series(comp_idx).value_counts().sort_index()
    comp_size_map = {f"UNASSIGNED_{i}": int(sz) for i, sz in comp_sizes.items()}
    comp_size_series = comp_id_series.map(comp_size_map)

    drop_small = is_unassigned & comp_size_series.notna() & (comp_size_series < min_comp_size)
    qc_status[drop_small.to_numpy()] = "drop_small_comp"

    W = aux["W"]
    gene_to_idx = aux["gene_to_idx"]
    gene_idx_all = df[gene_col].map(gene_to_idx)

    keep_candidate = is_unassigned & (~drop_small) & comp_id_series.notna()

    if show_progress:
        pbar_groupby = tqdm(total=2, desc="grouping_genes")

    comp_gene_map = {}
    df_candidate = df.loc[keep_candidate, [gene_col]].copy()
    df_candidate["_comp_id_local"] = comp_id_series.loc[keep_candidate].astype(str)
    df_candidate["_gene_idx_local"] = gene_idx_all.loc[keep_candidate]

    if show_progress:
        pbar_groupby.update(1)
        pbar_groupby.set_description("grouping")

    for comp_id, group in df_candidate.groupby("_comp_id_local", sort=True):
        g_local = np.sort(group["_gene_idx_local"].dropna().astype(int).unique())
        if g_local.size > 0:
            comp_gene_map[comp_id] = np.asarray(g_local, dtype=np.int32)

    if show_progress:
        pbar_groupby.update(1)
        pbar_groupby.close()

    # Bulk-prune all components through the Cython kernel.
    if len(comp_gene_map) > 0:
        comp_keys = sorted(comp_gene_map.keys())
        g_arrays = [comp_gene_map[k] if comp_gene_map[k].size > 0 else None for k in comp_keys]
        removed_lists = _cy_prune.prune_cells(g_arrays, W, float(npmi_threshold))

        iterator = zip(comp_keys, removed_lists)
        if show_progress:
            iterator = tqdm(list(iterator), desc="prune_comps")

        drop_gene_mask_all = np.zeros(len(df), dtype=bool)
        comp_id_series_str = comp_id_series.astype(str)
        keep_candidate_arr = keep_candidate.to_numpy()

        for comp_id, removed in iterator:
            comp_mask_arr = (comp_id_series_str.to_numpy() == comp_id) & keep_candidate_arr
            comp_indices = np.where(comp_mask_arr)[0]

            if comp_indices.size == 0:
                continue

            if removed is None or len(removed) == 0:
                qc_status[comp_indices] = "keep_unassigned_comp"
                continue

            removed_set = set(map(int, removed))
            comp_gene_mask = gene_idx_all.iloc[comp_indices].isin(removed_set).to_numpy()
            drop_gene_mask_all[comp_indices[comp_gene_mask]] = True
            qc_status[comp_indices[~comp_gene_mask]] = "keep_unassigned_comp"

        qc_status[drop_gene_mask_all] = "drop_npmi_pruned_gene"

    # Build the canonical assignment column.
    final_assignment = assigned_id_series.copy()
    qc_arr = np.asarray(qc_status)
    kept_unassigned_arr = is_unassigned.to_numpy() & (qc_arr == "keep_unassigned_comp")
    if kept_unassigned_arr.any():
        kept_idx = np.where(kept_unassigned_arr)[0]
        final_assignment.iloc[kept_idx] = comp_id_series.iloc[kept_idx].astype(str).values

    dropped_arr = is_unassigned.to_numpy() & (
        (qc_arr == "drop_small_comp") | (qc_arr == "drop_npmi_pruned_gene")
    )
    if dropped_arr.any():
        final_assignment.iloc[np.where(dropped_arr)[0]] = "DROP"

    # Canonical output: in-place write to `out_col`. With debug_stages,
    # also expose the legacy snapshot columns + diagnostic columns.
    df[out_col] = final_assignment.values
    if debug_stages:
        df["cell_id_final"] = final_assignment.values
        df["unassigned_comp_id"] = comp_id_series
        df["unassigned_comp_size"] = comp_size_series.astype("float32")
        df["unassigned_qc_status"] = qc_status

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
    entity_col: str = "tracer_id",
    coord_cols=("x", "y", "z"),
    k: int = 5,
    dist_threshold: float = 3.0,
    out_col: str = "tracer_id",
    debug_stages: bool = False,
    show_progress: bool = True,
    in_place: bool = False,
    min_fragment_size: int = 0,
    fragment_demote_label: str = "-1",
    skip_clean_cells: bool = False,
):
    """
    Fast variant of `enforce_spatial_coherence` with a progress bar.

    Reads `entity_col` (default `tracer_id`), splits any spatially
    disconnected label into `lab`, `lab-2`, `lab-3`, …, and writes back
    to `out_col` (defaults to `tracer_id` — in-place update).

    `debug_stages=True` additionally writes the same result to a
    legacy-named `cell_id_spatial` snapshot column.

    `in_place=True` skips the defensive DataFrame copy. Note: we still
    need to `reset_index` below, so a partial mutation happens even with
    `in_place=True`.
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
        if debug_stages:
            df["cell_id_spatial"] = df[out_col].copy()
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
    # Skip unassigned-equivalent labels: their tx haven't been placed by
    # any pipeline stage, so running label-constrained CCs on them and
    # creating "-1-2", "-1-3" sub-labels just produces ghost partials that
    # bypass Stage 2's annotate quality checks. Stage 2 is the proper
    # place for forming components from the unassigned pool.
    invalid = (
        (labels_arr == "DROP")
        | (labels_arr == "nan")
        | (labels_arr == "-1")
        | (labels_arr == "UNASSIGNED")
    )
    if skip_clean_cells:
        # A "clean cell" label has no '-' (e.g., '11765'); a partial / component
        # has '-' (e.g., '11765-1') or starts with 'UNASSIGNED_'. The Mickey-
        # Mouse over-segmentation case is mostly in partials (tx pruned from
        # different ears of the cell), not in cell cores. Splitting only
        # partials avoids over-fragmenting clean cells in dense regions where
        # kNN is unreliable.
        is_clean_cell = np.array([
            ('-' not in s) and (not s.startswith('UNASSIGNED_'))
            and s not in ('DROP', 'nan', 'UNASSIGNED', '-1')
            for s in labels_arr
        ])
        invalid = invalid | is_clean_cell
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
        # Suffix counter for sub-fragment labels. Increments only when we
        # actually emit a new sub-label, so a series of <min_fragment_size
        # tiny fragments doesn't waste suffixes (and the original label's
        # main component is always at suffix-level "1" implicitly).
        suffix_idx = 1
        for i, oi in enumerate(order):
            r = uniq_r[oi]
            if i == 0:
                # Largest component keeps the original label (already set)
                continue
            sel = (roots_c == r)
            frag_size = int(sel.sum())
            if min_fragment_size > 0 and frag_size < min_fragment_size:
                # Tiny disconnected fragment: demote to unassigned. Calling
                # them part of the cell is wrong (they're spatially
                # separated); making them their own partial label creates
                # ghost partials. Demoting lets rescue/annotate place them
                # properly.
                out[idx[sel]] = fragment_demote_label
                continue
            suffix_idx += 1
            new_lab = f"{lab}-{suffix_idx}"
            out[idx[sel]] = new_lab
    df[out_col] = out
    if debug_stages:
        df["cell_id_spatial"] = out.copy()
    df = df.drop(columns=["__node_idx"])
    return df

def enforce_spatial_coherence_per_label(
    df: pd.DataFrame,
    *,
    entity_col: str = "tracer_id",
    coord_cols=("x", "y", "z"),
    k: int = 8,
    edge_trim_pct: float = 10.0,
    min_size_for_split: int = 8,
    min_fragment_size: int = 0,
    fragment_demote_label: str = "-1",
    out_col: str = "tracer_id",
    in_place: bool = False,
    show_progress: bool = True,
):
    """Stage-4 alternative: per-label kNN + percentile-trim edge filter + CC.

    Unlike `enforce_spatial_coherence_fast`, this function builds a
    SEPARATE kNN graph PER LABEL (not a single global kNN that's then
    label-restricted). Per-label kNN avoids the dense-region failure
    mode where global kNN's k=5 nearest neighbors are mostly from
    *other* cells, leaving same-cell tx without enough within-label
    edges and causing spurious splits.

    Algorithm per label L (with size n ≥ ``min_size_for_split``):
      1. Build kNN graph on L's tx with k = min(``k``, n-1) and
         **no global distance threshold** (per-cell trim is below).
      2. Compute all edge lengths.
      3. Trim the top ``edge_trim_pct``% by length (these are the
         "bridge" edges in Mickey-Mouse cells).
      4. Find connected components of the trimmed graph.
      5. Largest CC keeps L; smaller CCs become ``L-2``, ``L-3``, ...
         (subject to ``min_fragment_size`` filter).

    Labels with size < ``min_size_for_split`` are left untouched
    (CC analysis on tiny labels is noise-dominated).

    Parameters
    ----------
    edge_trim_pct : float
        Percentage of longest edges to remove per label. Default 10.0
        means top-10% by length get cut. The natural per-cell outlier
        threshold for typical Mickey-Mouse vs. compact cells.
    min_size_for_split : int
        Skip CC analysis for labels with fewer than this many tx.
        Default 8 matches the kNN k.
    """
    df_out = df if in_place else df.copy()
    if out_col != entity_col:
        df_out[out_col] = df_out[entity_col]
    labels = df_out[out_col].astype(str)
    coords_all = df_out[list(coord_cols)].to_numpy(dtype=np.float32)

    out = df_out[out_col].to_numpy(dtype=object).copy()
    iter_groups = df_out.groupby(labels).indices.items()
    if show_progress:
        iter_groups = tqdm(list(iter_groups), desc="per-label-CC")

    invalid_set = {"-1", "DROP", "nan", "UNASSIGNED"}
    n_split = 0
    n_skipped_small = 0
    for lab, idxs in iter_groups:
        if str(lab) in invalid_set or str(lab).startswith("UNASSIGNED_"):
            continue
        n = len(idxs)
        if n < min_size_for_split:
            n_skipped_small += 1
            continue
        coords = coords_all[idxs]
        k_eff = min(k, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords)
        d, ni = nbrs.kneighbors(coords, return_distance=True)
        # Edges: (src, dst, length); exclude the self-edge (column 0)
        src = np.repeat(np.arange(n, dtype=np.int32), k_eff)
        dst = ni[:, 1:].astype(np.int32).ravel()
        lengths = d[:, 1:].astype(np.float32).ravel()
        # Trim top edge_trim_pct% by length
        if edge_trim_pct > 0.0 and lengths.size > 0:
            cutoff = float(np.percentile(lengths, 100.0 - edge_trim_pct))
            keep = lengths <= cutoff
            src = src[keep]; dst = dst[keep]
        # CC
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        graph = csr_matrix(
            (np.ones(src.size, dtype=np.int8), (src, dst)),
            shape=(n, n),
        )
        n_ccs, comp_lbl = connected_components(graph, directed=False)
        if n_ccs <= 1:
            continue
        # Sort CCs by size descending; largest keeps original label.
        cc_sizes = np.bincount(comp_lbl, minlength=n_ccs)
        order = np.argsort(-cc_sizes, kind="stable")  # descending
        n_split += 1
        suffix_idx = 1
        for rank, c in enumerate(order):
            if rank == 0:
                continue  # largest keeps the original label
            sel = comp_lbl == c
            frag_size = int(sel.sum())
            if min_fragment_size > 0 and frag_size < min_fragment_size:
                # Tiny disconnected fragment: keep it with parent (don't
                # split, don't demote). Mickey-Mouse ear of <N tx is
                # almost always benign; relabeling them creates routing
                # work for stitch that's likely to misroute.
                continue
            suffix_idx += 1
            out[idxs[sel]] = f"{lab}-{suffix_idx}"
    df_out[out_col] = out
    if show_progress:
        print(f"per-label CC: {n_split:,} labels split, "
              f"{n_skipped_small:,} skipped (size < {min_size_for_split})")
    return df_out


# ---------- Phase 6: Reassign unassigned transcripts to nearby partials/components ----------
def reassign_unassigned_to_nearby_entities(
    df_spatial: pd.DataFrame,
    entity_summary: pd.DataFrame | None = None,  # deprecated; ignored under new algo
    *,
    aux: dict | None = None,
    entity_col: str = "tracer_id",
    gene_col: str = "feature_name",
    out_col: str = "tracer_id",
    coord_cols=("x", "y", "z"),
    dist_threshold: float = 20.0,
    neg_npmi_threshold: float = -0.05,
    unassigned_labels=None,
    only_partial_component: bool = True,
    debug_stages: bool = False,
    debug_legacy_col: str = "cell_id_finetuned_2",
    show_progress: bool = True,
    in_place: bool = False,
):
    """
    Phase 6: Reassign unassigned transcripts to the nearest assigned-transcript's
    entity within a distance threshold, with an optional NPMI negative veto.

    Algorithm
    ---------
    1. Build a KD-tree over assigned transcript coordinates (filtered to
       partial/component entities if `only_partial_component=True`).
    2. For each unassigned transcript at (x, y, z) with gene g:
         a. Find all assigned transcripts within `dist_threshold` (radius
            search), sorted by increasing 3D distance.
         b. Walk neighbours in distance order. Skip if `aux` is provided and
            the neighbour's entity has any g' with NPMI(g, g') ≤
            `neg_npmi_threshold` (strong avoidance vetoes the rescue).
         c. Reassign the unassigned transcript to the first non-vetoed
            neighbour's entity. Stop walking.
    3. If no acceptable neighbour exists, leave the transcript as -1.

    This algorithm replaces the previous entity-centroid KNN approach
    because the old centroid-based distance was unreliable for
    spatially-elongated entities: an unassigned tx near one stray
    transcript of an entity stretched across a large z-range would see
    the entity as 10+ µm away (distance to global centroid), even
    though there was a member transcript right next to it. The
    nearest-tx distance avoids that miscalculation.

    Parameters
    ----------
    df_spatial : pd.DataFrame
        Transcript-level DataFrame with entity assignments (from Phase 5).
        Must contain `entity_col`, `gene_col`, and `coord_cols`.
    entity_summary : pd.DataFrame, deprecated
        Ignored. Retained as a positional argument for back-compat with
        the old centroid-based signature.
    aux : dict, optional
        If provided, must contain "W" (NPMI matrix) and "gene_to_idx".
        Enables the negative-NPMI veto: candidates whose entity has any
        gene g' with NPMI(g, g') ≤ neg_npmi_threshold are skipped.
        If None, no NPMI gating — pure nearest-tx reassignment.
    gene_col : str
        Gene/feature column name. Required when `aux` is provided.
    dist_threshold : float
        Maximum Euclidean (3D) distance for reassignment (default 20).
    neg_npmi_threshold : float
        Negative NPMI veto threshold (default -0.05). Only consulted
        when `aux` is provided.
    only_partial_component : bool
        If True, only consider partial/component entities as targets
        (skip whole cells).

    Returns
    -------
    df_out, n_reassigned, stats
        stats keys: total_unassigned, total_reassigned, n_blocked_by_neg_veto,
                    mean_distance, max_distance.
    """
    _ensure_reproducibility_seed()
    from .stitching import infer_entity_type

    if unassigned_labels is None:
        unassigned_labels = {"DROP", "-1", "UNASSIGNED", "nan"}

    df = df_spatial if in_place else df_spatial.copy()
    df[out_col] = df[entity_col].astype(str)

    # Identify unassigned transcripts
    labels = df[entity_col].astype(str)
    unassigned_mask = labels.isin(unassigned_labels)

    n_unassigned = int(unassigned_mask.sum())
    null_stats = {
        "total_unassigned": n_unassigned,
        "total_reassigned": 0,
        "n_blocked_by_neg_veto": 0,
        "mean_distance": np.nan,
        "max_distance": np.nan,
    }
    if n_unassigned == 0:
        return df, 0, null_stats

    # Extract unassigned transcript coordinates and genes
    unassigned_idx = np.where(unassigned_mask)[0]
    unassigned_coords = df.loc[unassigned_idx, list(coord_cols)].to_numpy(dtype=np.float32)
    if aux is not None:
        gene_to_idx = aux["gene_to_idx"]
        unassigned_genes = df.loc[unassigned_idx, gene_col].astype(str).to_numpy()
        unassigned_g_idx = np.array(
            [gene_to_idx.get(g, -1) for g in unassigned_genes], dtype=np.int64,
        )
    else:
        unassigned_g_idx = None

    # Drop unassigned transcripts that lack valid coords (or, when NPMI gating is on,
    # whose gene isn't in the panel).
    if unassigned_g_idx is not None:
        valid_u = ~np.isnan(unassigned_coords).any(axis=1) & (unassigned_g_idx >= 0)
    else:
        valid_u = ~np.isnan(unassigned_coords).any(axis=1)
    if not valid_u.all():
        unassigned_idx = unassigned_idx[valid_u]
        unassigned_coords = unassigned_coords[valid_u]
        if unassigned_g_idx is not None:
            unassigned_g_idx = unassigned_g_idx[valid_u]
    if len(unassigned_idx) == 0:
        return df, 0, null_stats

    # Build the assigned-tx pool. Filter by entity type if requested.
    assigned_mask = (~unassigned_mask).to_numpy()
    if only_partial_component:
        # Keep only assigned tx whose entity_id is partial / component.
        all_labels = df[entity_col].astype(str).to_numpy()
        partial_or_component = np.array([
            infer_entity_type(lab) in ("partial", "component")
            for lab in all_labels
        ], dtype=bool)
        assigned_mask = assigned_mask & partial_or_component

    if assigned_mask.sum() == 0:
        return df, 0, null_stats

    assigned_idx = np.where(assigned_mask)[0]
    assigned_coords = df.iloc[assigned_idx][list(coord_cols)].to_numpy(dtype=np.float32)
    assigned_entities = df.iloc[assigned_idx][entity_col].astype(str).to_numpy()
    valid_a = ~np.isnan(assigned_coords).any(axis=1)
    assigned_idx = assigned_idx[valid_a]
    assigned_coords = assigned_coords[valid_a]
    assigned_entities = assigned_entities[valid_a]
    if len(assigned_idx) == 0:
        return df, 0, null_stats

    # If aux is provided, build per-entity gene set for veto.
    entity_genes_lookup: dict[str, frozenset] = {}
    if aux is not None:
        gene_to_idx = aux["gene_to_idx"]
        W = aux["W"]
        # Group assigned-tx by entity to derive gene-set per entity.
        all_entity_labels = df[entity_col].astype(str).to_numpy()
        all_genes = df[gene_col].astype(str).to_numpy()
        for ent, gene_str in zip(all_entity_labels, all_genes):
            if ent in unassigned_labels:
                continue
            gi = gene_to_idx.get(gene_str)
            if gi is None:
                continue
            entity_genes_lookup.setdefault(ent, set()).add(int(gi))
        # Convert to frozenset for fast intersection.
        entity_genes_lookup = {k: frozenset(v) for k, v in entity_genes_lookup.items()}

    # KD-tree over assigned-tx coords (3D euclidean).
    knn = NearestNeighbors(radius=dist_threshold, algorithm="kd_tree", metric="euclidean")
    knn.fit(assigned_coords)
    radius_dists, radius_idxs = knn.radius_neighbors(unassigned_coords, return_distance=True)

    # Per-gene veto cache (only used if aux is provided).
    neg_cache: dict[int, frozenset] = {}

    def negative_set(g_idx: int) -> frozenset:
        cached = neg_cache.get(g_idx)
        if cached is not None:
            return cached
        row = np.asarray(W[g_idx])
        s = frozenset(int(x) for x in np.where(row <= neg_npmi_threshold)[0].tolist())
        neg_cache[g_idx] = s
        return s

    new_labels = np.empty(len(unassigned_idx), dtype=object)
    matched = np.zeros(len(unassigned_idx), dtype=bool)
    matched_dist = np.zeros(len(unassigned_idx), dtype=np.float32)
    n_blocked_by_neg_veto = 0

    for i in range(len(unassigned_idx)):
        dists = radius_dists[i]
        idxs = radius_idxs[i]
        if len(idxs) == 0:
            continue
        # Sort by distance.
        order = np.argsort(dists)
        dists = dists[order]
        idxs = idxs[order]

        if aux is not None and unassigned_g_idx is not None:
            g_idx = int(unassigned_g_idx[i])
            neg = negative_set(g_idx)
        else:
            neg = None

        # Walk neighbours in distance order; accept first non-vetoed.
        accepted = False
        any_vetoed = False
        for d, neighbor_pos in zip(dists.tolist(), idxs.tolist()):
            cand_entity = assigned_entities[int(neighbor_pos)]
            if neg is not None:
                ent_set = entity_genes_lookup.get(cand_entity, frozenset())
                if ent_set & neg:
                    any_vetoed = True
                    continue
            new_labels[i] = cand_entity
            matched[i] = True
            matched_dist[i] = float(d)
            accepted = True
            break
        if not accepted and any_vetoed:
            n_blocked_by_neg_veto += 1

    n_reassigned = int(matched.sum())
    if n_reassigned > 0:
        sel_rows = unassigned_idx[matched]
        col_pos = df.columns.get_loc(out_col)
        col_data = df[out_col]
        if isinstance(col_data.dtype, pd.CategoricalDtype):
            new_cats = set(map(str, new_labels[matched])) - set(col_data.cat.categories)
            if new_cats:
                df[out_col] = col_data.cat.add_categories(sorted(new_cats))
        df.iloc[sel_rows, col_pos] = new_labels[matched]

    if n_reassigned > 0:
        d_arr = matched_dist[matched]
        mean_distance = float(d_arr.mean())
        max_distance = float(d_arr.max())
    else:
        mean_distance = np.nan
        max_distance = np.nan

    stats = {
        "total_unassigned": int(n_unassigned),
        "total_reassigned": n_reassigned,
        "n_blocked_by_neg_veto": n_blocked_by_neg_veto,
        "mean_distance": mean_distance,
        "max_distance": max_distance,
    }

    if show_progress:
        print(f"Phase 6: Reassigned {n_reassigned}/{n_unassigned} unassigned transcripts "
              f"(threshold={dist_threshold}, veto={'on' if aux is not None else 'off'})")
        if n_reassigned > 0:
            print(f"  Mean distance: {mean_distance:.2f}, Max distance: {max_distance:.2f}")
        if n_blocked_by_neg_veto > 0:
            print(f"  Blocked by neg-NPMI veto: {n_blocked_by_neg_veto}")

    if debug_stages and debug_legacy_col != out_col:
        df[debug_legacy_col] = df[out_col].copy()

    return df, n_reassigned, stats

def demote_small_entities(
    df: pd.DataFrame,
    *,
    entity_col: str = "tracer_id",
    out_col: str | None = None,
    min_size: int = 5,
    unassigned_label: str = "-1",
    keep_labels: tuple[str, ...] = ("DROP", "-1", "nan"),
    exempt_types: tuple[str, ...] = ("cell",),
    in_place: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Demote transcripts of small entities (< `min_size`) to `unassigned_label`.

    Pairs cleanly with `reassign_unassigned_to_nearby_entities_fast`: this
    function flags small fragments as unassigned, then Phase 6 redistributes
    them to nearby large entities by spatial proximity.

    Parameters
    ----------
    df : pd.DataFrame
        Transcript-level data with one row per transcript.
    entity_col : str
        Column with current entity labels.
    out_col : str or None
        Where to write the new labels. If None, writes in-place on `entity_col`.
    min_size : int
        Entities with fewer than this many transcripts get demoted. Default 5.
        Use 1 to disable (no entity has fewer than 1 tx).
    unassigned_label : str
        Label assigned to demoted transcripts. Default `"-1"`.
    keep_labels : tuple of str
        Labels that are pre-existing unassigned/dropped — never demoted further.
        Default `("DROP", "-1", "nan")`.
    exempt_types : tuple of str
        Entity types (per `infer_entity_type`) that are protected from
        demotion regardless of size. Default `("cell",)` — whole cells
        are locked segmentation outputs and must never be removed even
        if they fall below `min_size` after upstream pruning. Pass `()`
        to disable the type protection.
    in_place : bool
        Skip the defensive `df.copy()`.

    Returns
    -------
    df_out : pd.DataFrame
    n_demoted : int
        Number of transcripts whose label was changed to `unassigned_label`.
    """
    if min_size <= 1:
        if out_col is not None and out_col != entity_col:
            df_out = df if in_place else df.copy()
            df_out[out_col] = df_out[entity_col]
            return df_out, 0
        return (df if in_place else df.copy()), 0

    df_out = df if in_place else df.copy()
    if out_col is None:
        out_col = entity_col
    if out_col != entity_col:
        df_out[out_col] = df_out[entity_col]

    labels = df_out[out_col].astype(str)
    counts = labels.value_counts()
    small_labels = set(counts.index[counts < min_size]) - set(keep_labels)
    if exempt_types:
        # Lazy import to avoid circular dep at module load.
        from .stitching import infer_entity_type
        exempt_set = set(exempt_types)
        small_labels = {
            lab for lab in small_labels
            if infer_entity_type(lab) not in exempt_set
        }
    if not small_labels:
        return df_out, 0

    demote_mask = labels.isin(small_labels).to_numpy()
    n_demoted = int(demote_mask.sum())
    if n_demoted > 0:
        col_data = df_out[out_col]
        # If the column is a pandas Categorical, ensure unassigned_label is a
        # known category before assignment; otherwise pandas raises TypeError.
        if isinstance(col_data.dtype, pd.CategoricalDtype):
            if unassigned_label not in col_data.cat.categories:
                df_out[out_col] = col_data.cat.add_categories([unassigned_label])
        col_pos = df_out.columns.get_loc(out_col)
        df_out.iloc[np.where(demote_mask)[0], col_pos] = unassigned_label

    return df_out, n_demoted


def reassign_unassigned_by_gene_compat(
    df: pd.DataFrame,
    aux: dict,
    entity_summary: pd.DataFrame | None = None,
    *,
    entity_col: str = "tracer_id",
    gene_col: str = "feature_name",
    coord_cols=("x", "y", "z"),
    out_col: str = "tracer_id",
    dist_threshold: float = 5.0,
    npmi_threshold: float = 0.05,
    unassigned_labels: set | None = None,
    only_partial_component: bool = True,
    in_place: bool = False,
) -> tuple[pd.DataFrame, int, dict]:
    """Reassign unassigned transcripts to the nearest *gene-compatible* entity.

    For each unassigned transcript with gene g, find the nearest candidate
    entity within `dist_threshold` whose gene set either:
      (a) already contains g, OR
      (b) contains some gene g' with NPMI(g, g') >= `npmi_threshold`.

    Preserves Stage 1's NPMI-based phenotype separation: a transcript that
    was pruned away from cell C (because its gene had no positive NPMI
    relationship with C's consensus gene set) will not be reassigned back
    to C — Phase 6's spatial-only logic would have done so.

    Cheaper than `deltaC` for single transcripts: per-tx work is O(number
    of nearby candidates × |entity gene set|), no submatrix extraction.

    Parameters
    ----------
    df : pd.DataFrame
        Transcript-level data with `entity_col`, `gene_col`, `coord_cols`.
    aux : dict
        Must contain "W" (NPMI matrix, G x G float) and "gene_to_idx"
        (gene name -> int index).
    entity_summary : pd.DataFrame, optional
        Pre-built entity summary. If None, will be built from `df`.
    npmi_threshold : float
        Minimum NPMI value for a gene to be considered "compatible" with g.
        Default 0.05 (matches the conservative-pruning consistency bar).

    Returns
    -------
    df_out, n_reassigned, stats
    """
    _ensure_reproducibility_seed()
    from .stitching import build_entity_table

    if unassigned_labels is None:
        unassigned_labels = {"DROP", "-1", "UNASSIGNED", "nan"}

    df_out = df if in_place else df.copy()
    df_out[out_col] = df_out[entity_col].astype(str)

    # Identify unassigned transcripts.
    labels = df_out[entity_col].astype(str)
    unassigned_mask = labels.isin(unassigned_labels)
    n_unassigned = int(unassigned_mask.sum())
    if n_unassigned == 0:
        return df_out, 0, {"total_unassigned": 0, "total_reassigned": 0,
                           "mean_distance": np.nan, "max_distance": np.nan}

    # Build entity summary if missing.
    if entity_summary is None:
        entity_summary = build_entity_table(
            df_out, entity_col=entity_col, gene_col=gene_col,
            coord_cols=coord_cols,
        )

    if only_partial_component:
        entities = entity_summary[entity_summary["etype"].isin(["partial", "component"])].copy()
    else:
        entities = entity_summary.copy()
    entities = entities.dropna(subset=list(coord_cols))
    if len(entities) == 0:
        return df_out, 0, {"total_unassigned": n_unassigned, "total_reassigned": 0,
                           "mean_distance": np.nan, "max_distance": np.nan}

    # Per-entity gene-index sets (np int32 arrays converted to frozenset).
    W = aux["W"]
    gene_to_idx = aux["gene_to_idx"]
    n_genes = W.shape[0]

    entity_genes: list[frozenset] = []
    for genes in entities["genes"].values:
        gi = pd.Index(np.asarray(genes, dtype=str)).map(gene_to_idx)
        gi = gi[~pd.isna(gi)].astype(int).unique()
        entity_genes.append(frozenset(int(x) for x in gi))

    entity_coords = entities[list(coord_cols)].to_numpy(dtype=np.float32)
    entity_ids = entities["entity_id"].to_numpy(dtype=object)

    # Unassigned tx: positional indices, coords, and gene indices.
    unassigned_idx = np.where(unassigned_mask)[0]
    unassigned_coords = df_out.iloc[unassigned_idx][list(coord_cols)].to_numpy(dtype=np.float32)
    unassigned_genes = df_out.iloc[unassigned_idx][gene_col].astype(str).to_numpy()
    unassigned_g_idx = np.array(
        [gene_to_idx.get(g, -1) for g in unassigned_genes],
        dtype=np.int64,
    )

    # Drop tx with NaN coords or unknown gene.
    valid = ~np.isnan(unassigned_coords).any(axis=1) & (unassigned_g_idx >= 0)
    if not valid.all():
        unassigned_idx = unassigned_idx[valid]
        unassigned_coords = unassigned_coords[valid]
        unassigned_g_idx = unassigned_g_idx[valid]
        unassigned_genes = unassigned_genes[valid]

    if len(unassigned_idx) == 0:
        return df_out, 0, {"total_unassigned": n_unassigned, "total_reassigned": 0,
                           "mean_distance": np.nan, "max_distance": np.nan}

    # KD-tree over entity centroids; query *all* entities within radius.
    knn = NearestNeighbors(radius=dist_threshold, algorithm="kd_tree", metric="euclidean")
    knn.fit(entity_coords)
    radius_dists, radius_idxs = knn.radius_neighbors(unassigned_coords, return_distance=True)

    # Cache "compatible gene set" per query gene index, since many tx share genes.
    compat_cache: dict[int, frozenset] = {}

    def compatible_genes(g_idx: int) -> frozenset:
        cached = compat_cache.get(g_idx)
        if cached is not None:
            return cached
        row = W[g_idx] if hasattr(W, "shape") else np.asarray(W[g_idx])
        compat = frozenset(
            int(x) for x in np.where(np.asarray(row) >= npmi_threshold)[0].tolist()
        )
        compat_cache[g_idx] = compat
        return compat

    new_labels = np.empty(len(unassigned_idx), dtype=object)
    matched = np.zeros(len(unassigned_idx), dtype=bool)
    matched_dist = np.zeros(len(unassigned_idx), dtype=np.float32)

    for i, (dists, idxs, g_idx) in enumerate(
        zip(radius_dists, radius_idxs, unassigned_g_idx)
    ):
        if len(idxs) == 0:
            continue
        order = np.argsort(dists)
        dists = dists[order]
        idxs = idxs[order]
        compat: frozenset | None = None
        for d, ent_i in zip(dists, idxs):
            ent_i = int(ent_i)
            ent_set = entity_genes[ent_i]
            if int(g_idx) in ent_set:
                new_labels[i] = entity_ids[ent_i]
                matched[i] = True
                matched_dist[i] = float(d)
                break
            if compat is None:
                compat = compatible_genes(int(g_idx))
            if ent_set & compat:
                new_labels[i] = entity_ids[ent_i]
                matched[i] = True
                matched_dist[i] = float(d)
                break

    if matched.any():
        sel_rows = unassigned_idx[matched]
        col_pos = df_out.columns.get_loc(out_col)
        col_data = df_out[out_col]
        if isinstance(col_data.dtype, pd.CategoricalDtype):
            new_cats = set(map(str, new_labels[matched])) - set(col_data.cat.categories)
            if new_cats:
                df_out[out_col] = col_data.cat.add_categories(sorted(new_cats))
        df_out.iloc[sel_rows, col_pos] = new_labels[matched]

    n_reassigned = int(matched.sum())
    if n_reassigned > 0:
        d_arr = matched_dist[matched]
        mean_d = float(d_arr.mean())
        max_d = float(d_arr.max())
    else:
        mean_d = float("nan")
        max_d = float("nan")

    return df_out, n_reassigned, {
        "total_unassigned": n_unassigned,
        "total_reassigned": n_reassigned,
        "mean_distance": mean_d,
        "max_distance": max_d,
    }


def reassign_unassigned_grid_pool(
    df: pd.DataFrame,
    aux: dict,
    *,
    entity_col: str = "tracer_id",
    gene_col: str = "feature_name",
    coord_cols=("x", "y", "z"),
    out_col: str = "tracer_id",
    G: float = 5.0,
    neg_npmi_threshold: float = -0.05,
    z_bound: float | None = None,
    unassigned_labels: set | None = None,
    only_partial_component: bool = False,
    in_place: bool = False,
    veto_mode: str = "min",
    mean_threshold: float = 0.0,
    small_entity_guard_n: int = 3,
    pos_npmi_threshold=None,  # deprecated; ignored. Kept for back-compat.
) -> tuple[pd.DataFrame, int, dict]:
    """Grid-bin rescue: distance-priority with NPMI as a negative veto.

    For each unassigned transcript with gene `g` at ``(x, y, z)``:

    1. **Spatial gating.** Collect assigned transcripts whose xy bin is
       in the 9-cell Moore neighborhood (own bin + 8 neighbors at scale
       ``G``) AND whose ``|Δz| ≤ z_bound``. The z-bound matches the
       implicit xy reach of ~``G·√2``, so the spatial gate is
       effectively a 3D ball of radius ~``G·√2`` (within bin tolerance
       in xy and exactly bounded in z).

    2. **NPMI veto.** Group surviving transcripts by their entity. Drop
       any entity whose gene set contains a ``g'`` with
       ``NPMI(g, g') ≤ neg_npmi_threshold`` — strong avoidance vetoes
       the rescue.

    3. **Distance ranking.** Among survivors, compute each entity's
       3D distance to the unassigned transcript as the **minimum over
       its in-bin-neighborhood transcripts** (nearest-tx, not
       centroid). Reassign to the entity with the smallest such
       distance. Ties broken by lexicographic entity_id.

    This logic differs from the previous PREFERRED/NEUTRAL pool
    selection: distance is now the primary criterion; NPMI is a yes/no
    filter, not a categorical preference. Earlier behavior could pull
    transcripts across z-gaps larger than the bin size when a
    PREFERRED entity was farther than a NEUTRAL one — under the new
    rules, the nearest non-vetoed entity wins.

    Parameters
    ----------
    G : float
        Bin size in µm. The 8-neighbor reach is ~``G·√2`` µm in xy.
    neg_npmi_threshold : float
        Veto threshold for ``veto_mode='min'``. Any pair ``(g, g')`` in
        E.genes with ``NPMI(g, g') ≤ neg_npmi_threshold`` rejects E.
    z_bound : float or None
        Maximum |Δz| from unassigned transcript to candidate
        transcripts. ``None`` defaults to ``G·√2`` (matching the xy
        reach of the bin neighborhood). Programmable per-dataset for
        anisotropic z-resolution.
    veto_mode : {'min', 'mean'}
        How to score the absorbed gene against the candidate entity's
        gene set.

        - ``'min'`` (default, legacy): reject E if the most-negative
          observed PMI between g and any g' in E.genes is below
          ``neg_npmi_threshold``. Sensitive to a single avoiding pair.
        - ``'mean'``: reject E if the *mean* PMI over observed pairs
          (g, g' for g' in E.genes) is at or below ``mean_threshold``.
          Tolerates a single complementary state-marker outlier as
          long as the aggregate signal is positive.

    mean_threshold : float
        Used when ``veto_mode='mean'``. Reject E if mean PMI ≤
        ``mean_threshold``. Default 0.0 (above-independence on average).
    small_entity_guard_n : int
        Used when ``veto_mode='mean'``. If E has fewer than this many
        observed (finite-PMI) gene pairs against g, fall back to the
        ``'min'`` rule with ``neg_npmi_threshold`` for that entity —
        the mean estimator is too noisy on tiny entities. Set to 0 to
        disable the fallback.
    pos_npmi_threshold : ignored
        Accepted for backward compatibility with the previous
        PREFERRED/NEUTRAL API; unused under the distance-priority
        algorithm.
    """
    if veto_mode not in ("min", "mean"):
        raise ValueError(f"veto_mode must be 'min' or 'mean', got {veto_mode!r}")
    _ensure_reproducibility_seed()
    from .stitching import build_entity_table, infer_entity_type
    from .graph import bin_xy, neighbor_bins

    if unassigned_labels is None:
        unassigned_labels = {"DROP", "-1", "UNASSIGNED", "nan"}

    df_out = df if in_place else df.copy()
    df_out[out_col] = df_out[entity_col].astype(str)

    labels_str = df_out[entity_col].astype(str)
    unassigned_mask = labels_str.isin(unassigned_labels).to_numpy()
    n_unassigned = int(unassigned_mask.sum())
    z_bound_eff = float(z_bound) if z_bound is not None else float(G) * math.sqrt(2.0)
    null_stats = {
        "total_unassigned": n_unassigned, "total_reassigned": 0,
        "n_blocked_by_neg_veto": 0, "n_no_candidates": 0,
        "z_bound": z_bound_eff,
        "mean_distance": float("nan"), "max_distance": float("nan"),
    }
    if n_unassigned == 0:
        return df_out, 0, null_stats

    entity_summary = build_entity_table(
        df_out, entity_col=entity_col, gene_col=gene_col,
        coord_cols=coord_cols,
    )
    gene_to_idx = aux["gene_to_idx"]
    W = aux["W"]

    entity_genes_lookup: dict[str, frozenset] = {}
    skip_entity_set: set[str] = set()
    z_col_idx = list(coord_cols).index("z") if "z" in coord_cols else None
    for ent_row in entity_summary.itertuples():
        eid = str(ent_row.entity_id)
        gi = pd.Index(np.asarray(ent_row.genes, dtype=str)).map(gene_to_idx)
        gi = gi[~pd.isna(gi)].astype(int).unique()
        entity_genes_lookup[eid] = frozenset(int(x) for x in gi)
        if only_partial_component and infer_entity_type(eid) == "cell":
            skip_entity_set.add(eid)

    # Assigned-tx pool for the bin index (excluding cells if requested).
    assigned_mask = ~unassigned_mask
    if only_partial_component:
        assigned_mask = assigned_mask & ~labels_str.isin(skip_entity_set).to_numpy()
    if assigned_mask.sum() == 0:
        return df_out, 0, null_stats

    assigned_idx = np.where(assigned_mask)[0]
    assigned_coords = df_out.iloc[assigned_idx][list(coord_cols)].to_numpy(dtype=np.float32)
    assigned_entities = df_out.iloc[assigned_idx][entity_col].astype(str).to_numpy()
    valid_a = ~np.isnan(assigned_coords).any(axis=1)
    assigned_idx = assigned_idx[valid_a]
    assigned_coords = assigned_coords[valid_a]
    assigned_entities = assigned_entities[valid_a]
    if len(assigned_idx) == 0:
        return df_out, 0, null_stats

    bin_keys_a = bin_xy(assigned_coords[:, :2], G)
    bin_to_local_idxs: dict[int, list[int]] = {}
    for local_i, bk in enumerate(bin_keys_a.tolist()):
        bin_to_local_idxs.setdefault(int(bk), []).append(local_i)

    # Unassigned tx candidates.
    una_idx = np.where(unassigned_mask)[0]
    una_coords = df_out.iloc[una_idx][list(coord_cols)].to_numpy(dtype=np.float32)
    una_genes = df_out.iloc[una_idx][gene_col].astype(str).to_numpy()
    una_g_idx = np.array(
        [gene_to_idx.get(g, -1) for g in una_genes], dtype=np.int64,
    )
    valid_u = ~np.isnan(una_coords).any(axis=1) & (una_g_idx >= 0)
    una_idx = una_idx[valid_u]
    una_coords = una_coords[valid_u]
    una_g_idx = una_g_idx[valid_u]
    if len(una_idx) == 0:
        return df_out, 0, null_stats

    una_bin_keys = bin_xy(una_coords[:, :2], G)

    # Per-gene cache for the negative-veto set.
    neg_cache: dict[int, frozenset] = {}

    def negative_set(g_idx: int) -> frozenset:
        cached = neg_cache.get(g_idx)
        if cached is not None:
            return cached
        row = np.asarray(W[g_idx])
        s = frozenset(int(x) for x in np.where(row <= neg_npmi_threshold)[0].tolist())
        neg_cache[g_idx] = s
        return s

    new_labels = np.empty(len(una_idx), dtype=object)
    matched = np.zeros(len(una_idx), dtype=bool)
    matched_dist = np.zeros(len(una_idx), dtype=np.float32)
    n_blocked_by_neg_veto = 0
    n_no_candidates = 0
    n_small_entity_fallback = 0  # mean-mode only

    # Detect whether W is sparse; mean-mode needs row vectors.
    import scipy.sparse as sp_mod
    W_is_sparse = sp_mod.issparse(W)

    n_coord_cols = len(coord_cols)
    for i, (bk, g_idx) in enumerate(zip(una_bin_keys.tolist(), una_g_idx.tolist())):
        bk = int(bk)
        # Collect assigned-tx local indices in bin + 8-neighbors, then
        # apply the z-bound to keep only spatially close candidates.
        local_idxs_raw: list[int] = []
        for nb in [bk] + neighbor_bins(bk, topology="8"):
            chunk = bin_to_local_idxs.get(int(nb))
            if chunk:
                local_idxs_raw.extend(chunk)
        if not local_idxs_raw:
            n_no_candidates += 1
            continue

        # z-bound filter (when z_col_idx is defined).
        if z_col_idx is not None:
            z_tx = float(una_coords[i, z_col_idx])
            local_li = np.fromiter(local_idxs_raw, dtype=np.int64)
            cand_z = assigned_coords[local_li, z_col_idx]
            in_bound = np.abs(cand_z - z_tx) <= z_bound_eff
            local_li = local_li[in_bound]
            if local_li.size == 0:
                n_no_candidates += 1
                continue
        else:
            local_li = np.fromiter(local_idxs_raw, dtype=np.int64)

        # PMI veto setup. Both modes need the negative_set for the small-
        # entity fallback (mean) or the primary check (min).
        neg = negative_set(int(g_idx))
        # In mean mode, also fetch the full PMI row for vectorised entity-
        # mean computation.
        row_g: np.ndarray | None = None
        if veto_mode == "mean":
            if W_is_sparse:
                row_g = np.asarray(W.getrow(int(g_idx)).todense()).ravel()
            else:
                row_g = np.asarray(W[int(g_idx)])
        # Compute 3D distance from unassigned tx to each surviving
        # candidate transcript (vectorised).
        dxyz = assigned_coords[local_li, :n_coord_cols] - una_coords[i, :n_coord_cols]
        dists = np.sqrt(np.sum(dxyz * dxyz, axis=1))

        # Per-entity nearest-tx distance, with veto. Cache per-entity
        # mean-PMI decisions within this tx (one-shot per entity).
        best_ent: str | None = None
        best_dist: float = float("inf")
        any_vetoed = False
        ent_decision_cache: dict[str, bool] = {}  # True = veto, False = OK
        for li_local, d in zip(local_li.tolist(), dists.tolist()):
            ent = assigned_entities[li_local]
            ent_set = entity_genes_lookup.get(ent, frozenset())

            if veto_mode == "min":
                vetoed = bool(ent_set and (ent_set & neg))
            else:
                cached = ent_decision_cache.get(ent)
                if cached is not None:
                    vetoed = cached
                else:
                    if not ent_set:
                        vetoed = False
                    else:
                        # Drop self-pair if present.
                        other = [gi for gi in ent_set if gi != int(g_idx)]
                        if not other:
                            vetoed = False
                        else:
                            pmis = row_g[np.asarray(other, dtype=np.int64)]
                            finite = np.isfinite(pmis)
                            n_valid = int(finite.sum())
                            if n_valid < small_entity_guard_n:
                                # Fall back to min-veto for this entity.
                                vetoed = bool(ent_set & neg)
                                if not (ent_set & neg) and n_valid > 0:
                                    n_small_entity_fallback += 1
                            elif n_valid == 0:
                                # No observed-PMI evidence at all — block by
                                # default. (Absence of evidence ≠ evidence of
                                # presence: don't grow E into g without any
                                # positive signal.)
                                vetoed = True
                            else:
                                mean_p = float(pmis[finite].mean())
                                vetoed = mean_p <= mean_threshold
                    ent_decision_cache[ent] = vetoed

            if vetoed:
                any_vetoed = True
                continue
            if d < best_dist:
                best_dist = float(d)
                best_ent = ent
        if best_ent is None:
            if any_vetoed:
                n_blocked_by_neg_veto += 1
            else:
                n_no_candidates += 1
            continue

        new_labels[i] = best_ent
        matched[i] = True
        matched_dist[i] = best_dist

    if matched.any():
        sel_rows = una_idx[matched]
        col_pos = df_out.columns.get_loc(out_col)
        col_data = df_out[out_col]
        if isinstance(col_data.dtype, pd.CategoricalDtype):
            new_cats = set(map(str, new_labels[matched])) - set(col_data.cat.categories)
            if new_cats:
                df_out[out_col] = col_data.cat.add_categories(sorted(new_cats))
        df_out.iloc[sel_rows, col_pos] = new_labels[matched]

    n_reassigned = int(matched.sum())
    if n_reassigned > 0:
        d_arr = matched_dist[matched]
        mean_d = float(d_arr.mean())
        max_d = float(d_arr.max())
    else:
        mean_d = float("nan")
        max_d = float("nan")

    return df_out, n_reassigned, {
        "total_unassigned": n_unassigned,
        "total_reassigned": n_reassigned,
        "n_blocked_by_neg_veto": n_blocked_by_neg_veto,
        "n_no_candidates": n_no_candidates,
        "n_small_entity_fallback": n_small_entity_fallback,
        "veto_mode": veto_mode,
        "z_bound": z_bound_eff,
        "mean_distance": mean_d,
        "max_distance": max_d,
    }


def pre_stage2_rescue(
    df: pd.DataFrame,
    aux: dict,
    *,
    entity_col: str = "tracer_id",
    gene_col: str = "feature_name",
    coord_cols=("x", "y", "z"),
    out_col: str = "tracer_id",
    G: float = 2.0,
    neg_npmi_threshold: float = -0.05,
    z_bound: float | None = None,
    cluster_guard_n: int = 3,
    in_place: bool = False,
    veto_mode: str = "min",
    mean_threshold: float = 0.0,
    small_entity_guard_n: int = 3,
    pos_npmi_threshold=None,  # deprecated; ignored. Kept for back-compat.
) -> tuple[pd.DataFrame, int, int, dict]:
    """Pre-Stage-2 rescue: tight-scale NPMI-categorical reassignment of
    Stage-1-pruned transcripts, guarded by a same-bin same-gene cluster
    check that preserves potential novel UNASSIGNED_* components.

    Algorithm
    ---------
    1. Identify unassigned tx (entity_col matches "-1" / "DROP" / etc).
    2. Cluster guard (default ON): bin unassigned tx at G, count
       same-bin same-gene unassigned peers, and EXCLUDE tx with peer
       count >= `cluster_guard_n` from rescue. Excluded tx remain "-1"
       so Stage 2 can later cluster them into UNASSIGNED_* components.
    3. Run `reassign_unassigned_grid_pool` with G=2 on the remaining
       (non-excluded) unassigned tx.

    Returns
    -------
    df_out, n_rescued, n_skipped_by_guard, stats
    """
    _ensure_reproducibility_seed()
    from .graph import bin_xy

    df_out = df if in_place else df.copy()
    df_out[out_col] = df_out[entity_col].astype(str)

    unassigned_set = {"DROP", "-1", "UNASSIGNED", "nan"}
    labels_str = df_out[entity_col].astype(str)
    unassigned_mask = labels_str.isin(unassigned_set).to_numpy()
    if unassigned_mask.sum() == 0:
        return df_out, 0, 0, {"total_unassigned": 0, "total_reassigned": 0,
                              "skipped_by_guard": 0,
                              "n_blocked_by_neg_veto": 0, "n_no_candidates": 0,
                              "mean_distance": float("nan"), "max_distance": float("nan")}

    n_skipped = 0
    excluded_mask = np.zeros(len(df_out), dtype=bool)

    if cluster_guard_n > 0:
        # Bin unassigned tx; count same-bin same-gene unassigned peers.
        gene_to_idx = aux["gene_to_idx"]
        una_idx = np.where(unassigned_mask)[0]
        una_coords = df_out.iloc[una_idx][list(coord_cols)].to_numpy(dtype=np.float32)
        una_genes = df_out.iloc[una_idx][gene_col].astype(str).to_numpy()
        una_g_idx = np.array(
            [gene_to_idx.get(g, -1) for g in una_genes], dtype=np.int64,
        )
        valid = ~np.isnan(una_coords).any(axis=1) & (una_g_idx >= 0)
        valid_local_idx = np.where(valid)[0]
        una_bin = bin_xy(una_coords[valid][:, :2], G)
        # Count by (bin, gene)
        from collections import Counter
        bg_counts = Counter(zip(una_bin.tolist(), una_g_idx[valid].tolist()))
        # For each unassigned tx, check its (bin, gene) count
        bin_keys_full = np.full(len(una_idx), -1, dtype=np.int64)
        bin_keys_full[valid_local_idx] = una_bin
        gene_idx_full = una_g_idx
        for k_local, (bk, g) in enumerate(zip(bin_keys_full.tolist(), gene_idx_full.tolist())):
            if g < 0:
                continue
            if bg_counts.get((int(bk), int(g)), 0) - 1 >= cluster_guard_n:
                # peer count (excluding self) >= cluster_guard_n
                excluded_mask[una_idx[k_local]] = True
                n_skipped += 1

    # Run rescue on a "shielded" copy where excluded tx aren't visible
    # to the unassigned set. Simplest: temporarily relabel them so the
    # rescue's `unassigned_labels` filter skips them.
    SHIELD_LABEL = "__GUARD_SKIP__"
    if excluded_mask.any():
        col_data = df_out[out_col]
        col_pos = df_out.columns.get_loc(out_col)
        if isinstance(col_data.dtype, pd.CategoricalDtype) and SHIELD_LABEL not in col_data.cat.categories:
            df_out[out_col] = col_data.cat.add_categories([SHIELD_LABEL])
        df_out.iloc[np.where(excluded_mask)[0], col_pos] = SHIELD_LABEL

    df_out, n_reassigned, stats = reassign_unassigned_grid_pool(
        df_out, aux=aux,
        entity_col=out_col, gene_col=gene_col, coord_cols=coord_cols,
        out_col=out_col,
        G=G,
        neg_npmi_threshold=neg_npmi_threshold,
        z_bound=z_bound,
        only_partial_component=False,
        in_place=True,
        veto_mode=veto_mode,
        mean_threshold=mean_threshold,
        small_entity_guard_n=small_entity_guard_n,
    )

    # Restore: any tx still holding SHIELD_LABEL after rescue → reset to
    # "-1". This covers two cases:
    #   (a) Originally-shielded tx (excluded_mask): label survived the
    #       rescue unchanged. Restore so Stage 2 sees them as unassigned.
    #   (b) Tx that the rescue ABSORBED into the SHIELD_LABEL "entity"
    #       (because it appeared as just another assigned label to
    #       reassign_unassigned_grid_pool). Without this restore the
    #       shield becomes a fake cell — visible only when shielded
    #       tx form an isolated cluster, but the same misrouting leaks
    #       silently into normal segmented runs too.
    if excluded_mask.any() or True:  # always check (b) even when nothing was shielded
        labels_now = df_out[out_col].astype(str).to_numpy()
        sl_after = (labels_now == SHIELD_LABEL)
        n_absorbed_into_shield = int((sl_after & ~excluded_mask).sum())
        if sl_after.any():
            col_pos = df_out.columns.get_loc(out_col)
            df_out.iloc[np.where(sl_after)[0], col_pos] = "-1"
        # Bogus absorptions shouldn't count as real rescues.
        n_reassigned -= n_absorbed_into_shield
        stats["n_absorbed_into_shield_reverted"] = n_absorbed_into_shield

    stats["skipped_by_guard"] = n_skipped
    return df_out, n_reassigned, n_skipped, stats


def reassign_unassigned_to_nearest_tx_no_neg(
    df: pd.DataFrame,
    aux: dict,
    *,
    entity_col: str = "tracer_id",
    gene_col: str = "feature_name",
    coord_cols=("x", "y", "z"),
    out_col: str = "tracer_id",
    dist_threshold: float = 5.0,
    neg_npmi_threshold: float = -0.1,
    unassigned_labels: set | None = None,
    only_partial_component: bool = False,
    in_place: bool = False,
) -> tuple[pd.DataFrame, int, dict]:
    """Reassign unassigned tx by nearest *transcript* (not centroid), gated by
    "no strongly-negative NPMI" against the candidate's entity gene set.

    Algorithm
    ---------
    For each unassigned transcript t with gene g:
      1. Find the transcript-level neighbors of t within `dist_threshold`
         (KD-tree over all *assigned* transcripts).
      2. Walk neighbors in order of increasing distance.
      3. For each neighbor's entity E, check whether any gene g' in E's
         gene set has `NPMI(g, g') <= neg_npmi_threshold`. If yes, REJECT
         (strong incompatibility) and continue to the next neighbor.
      4. Otherwise ACCEPT — assign t to E. Stop walking.
      5. If no acceptable neighbor exists within `dist_threshold`, leave as -1.

    Differences vs `reassign_unassigned_to_nearby_entities_fast`:
      - Distance is to the nearest transcript, not the entity centroid.
        Better for irregular or large entities (a tx near the boundary of
        a long cell is closer to a member tx than to the centroid).
      - Phenotype gating uses *negative* NPMI evidence to reject, rather
        than requiring positive NPMI to accept. Aligns with how Stage 1's
        conservative pruning works at positive threshold settings: we only
        veto if there's strong evidence of incompatibility.

    Parameters
    ----------
    df : pd.DataFrame
        Transcript-level data.
    aux : dict
        Must contain "W" (NPMI matrix) and "gene_to_idx".
    neg_npmi_threshold : float
        Cutoff for "strong negative evidence". Default -0.1, matching the
        conservative-pruning bar.
    only_partial_component : bool
        If True, neighbor transcripts assigned to "cell" entities are
        skipped during the walk. Default False (cells eligible — the
        whole point of nearest-tx is to handle proximity faithfully).

    Returns
    -------
    df_out, n_reassigned, stats
    """
    _ensure_reproducibility_seed()
    from .stitching import build_entity_table, infer_entity_type

    if unassigned_labels is None:
        unassigned_labels = {"DROP", "-1", "UNASSIGNED", "nan"}

    df_out = df if in_place else df.copy()
    df_out[out_col] = df_out[entity_col].astype(str)

    labels_str = df_out[entity_col].astype(str)
    unassigned_mask = labels_str.isin(unassigned_labels).to_numpy()
    n_unassigned = int(unassigned_mask.sum())
    if n_unassigned == 0:
        return df_out, 0, {
            "total_unassigned": 0, "total_reassigned": 0,
            "mean_distance": float("nan"), "max_distance": float("nan"),
        }

    # Build entity_id -> gene-index frozenset, with optional skip for cells.
    entity_summary = build_entity_table(
        df_out, entity_col=entity_col, gene_col=gene_col,
        coord_cols=coord_cols,
    )
    gene_to_idx = aux["gene_to_idx"]
    W = aux["W"]

    entity_genes_lookup: dict[str, frozenset] = {}
    skip_entity_set: set[str] = set()
    for ent_row in entity_summary.itertuples():
        eid = str(ent_row.entity_id)
        gi = pd.Index(np.asarray(ent_row.genes, dtype=str)).map(gene_to_idx)
        gi = gi[~pd.isna(gi)].astype(int).unique()
        entity_genes_lookup[eid] = frozenset(int(x) for x in gi)
        if only_partial_component and infer_entity_type(eid) == "cell":
            skip_entity_set.add(eid)

    # Assigned transcripts -- the searchable index.
    assigned_mask = ~unassigned_mask
    if only_partial_component:
        # Drop assigned tx whose entity is a cell.
        cell_mask = labels_str.isin(skip_entity_set).to_numpy()
        assigned_mask = assigned_mask & ~cell_mask
    if assigned_mask.sum() == 0:
        return df_out, 0, {
            "total_unassigned": n_unassigned, "total_reassigned": 0,
            "mean_distance": float("nan"), "max_distance": float("nan"),
        }

    assigned_idx = np.where(assigned_mask)[0]
    assigned_coords = df_out.iloc[assigned_idx][list(coord_cols)].to_numpy(dtype=np.float32)
    assigned_entities = df_out.iloc[assigned_idx][entity_col].astype(str).to_numpy()
    # Drop NaN-coord assigned tx.
    valid_a = ~np.isnan(assigned_coords).any(axis=1)
    assigned_coords = assigned_coords[valid_a]
    assigned_entities = assigned_entities[valid_a]
    assigned_idx = assigned_idx[valid_a]
    if len(assigned_idx) == 0:
        return df_out, 0, {
            "total_unassigned": n_unassigned, "total_reassigned": 0,
            "mean_distance": float("nan"), "max_distance": float("nan"),
        }

    # Unassigned tx with valid coords + known gene index.
    una_idx = np.where(unassigned_mask)[0]
    una_coords = df_out.iloc[una_idx][list(coord_cols)].to_numpy(dtype=np.float32)
    una_genes = df_out.iloc[una_idx][gene_col].astype(str).to_numpy()
    una_g_idx = np.array(
        [gene_to_idx.get(g, -1) for g in una_genes], dtype=np.int64,
    )
    valid_u = ~np.isnan(una_coords).any(axis=1) & (una_g_idx >= 0)
    una_idx = una_idx[valid_u]
    una_coords = una_coords[valid_u]
    una_g_idx = una_g_idx[valid_u]
    if len(una_idx) == 0:
        return df_out, 0, {
            "total_unassigned": n_unassigned, "total_reassigned": 0,
            "mean_distance": float("nan"), "max_distance": float("nan"),
        }

    knn = NearestNeighbors(radius=dist_threshold, algorithm="kd_tree", metric="euclidean")
    knn.fit(assigned_coords)
    radius_dists, radius_idxs = knn.radius_neighbors(una_coords, return_distance=True)

    # Per-gene cache: set of gene indices that are "strongly incompatible" with g.
    neg_cache: dict[int, frozenset] = {}

    def negative_set(g_idx: int) -> frozenset:
        cached = neg_cache.get(g_idx)
        if cached is not None:
            return cached
        row = np.asarray(W[g_idx])
        neg = frozenset(
            int(x) for x in np.where(row <= neg_npmi_threshold)[0].tolist()
        )
        neg_cache[g_idx] = neg
        return neg

    new_labels = np.empty(len(una_idx), dtype=object)
    matched = np.zeros(len(una_idx), dtype=bool)
    matched_dist = np.zeros(len(una_idx), dtype=np.float32)

    for i, (dists, idxs, g_idx) in enumerate(zip(radius_dists, radius_idxs, una_g_idx)):
        if len(idxs) == 0:
            continue
        order = np.argsort(dists)
        dists = dists[order]
        idxs = idxs[order]
        neg: frozenset | None = None
        for d, neighbor_pos in zip(dists, idxs):
            cand_entity = assigned_entities[int(neighbor_pos)]
            ent_genes = entity_genes_lookup.get(cand_entity, frozenset())
            if not ent_genes:
                continue
            if neg is None:
                neg = negative_set(int(g_idx))
            if ent_genes & neg:
                continue  # strong negative evidence -> reject
            new_labels[i] = cand_entity
            matched[i] = True
            matched_dist[i] = float(d)
            break

    if matched.any():
        sel_rows = una_idx[matched]
        col_pos = df_out.columns.get_loc(out_col)
        col_data = df_out[out_col]
        if isinstance(col_data.dtype, pd.CategoricalDtype):
            new_cats = set(map(str, new_labels[matched])) - set(col_data.cat.categories)
            if new_cats:
                df_out[out_col] = col_data.cat.add_categories(sorted(new_cats))
        df_out.iloc[sel_rows, col_pos] = new_labels[matched]

    n_reassigned = int(matched.sum())
    if n_reassigned > 0:
        d_arr = matched_dist[matched]
        mean_d = float(d_arr.mean())
        max_d = float(d_arr.max())
    else:
        mean_d = float("nan")
        max_d = float("nan")

    return df_out, n_reassigned, {
        "total_unassigned": n_unassigned,
        "total_reassigned": n_reassigned,
        "mean_distance": mean_d,
        "max_distance": max_d,
    }


def reassign_unassigned_to_nearby_entities_fast(
    df_spatial: pd.DataFrame,
    entity_summary: pd.DataFrame = None,
    *,
    aux: dict | None = None,
    entity_col: str = "tracer_id",
    gene_col: str = "feature_name",
    coord_cols=("x", "y", "z"),
    out_col: str = "tracer_id",
    dist_threshold: float = 20.0,
    neg_npmi_threshold: float = -0.05,
    unassigned_labels=None,
    only_partial_component: bool = True,
    debug_stages: bool = False,
    debug_legacy_col: str = "cell_id_finetuned_2",
    show_progress: bool = True,
    in_place: bool = False,
):
    """Thin wrapper around :func:`reassign_unassigned_to_nearby_entities`.

    Note: as of the distance-priority refactor, ``entity_summary`` is no
    longer required (the inner function uses df_spatial directly). The
    parameter is retained as a no-op for back-compat with existing
    callers; pass it freely or omit it.

    Parameters
    ----------
    aux : dict, optional
        If provided, must contain "W" and "gene_to_idx". Enables the
        negative-NPMI veto on candidate entities. If None, no veto.
    neg_npmi_threshold : float
        Veto threshold for the optional NPMI gate.

    See :func:`reassign_unassigned_to_nearby_entities` for the full
    parameter and algorithm documentation.
    """
    _ensure_reproducibility_seed()
    # entity_summary is now redundant under the nearest-tx algorithm —
    # we only need df_spatial. Silently ignore it if passed.
    return reassign_unassigned_to_nearby_entities(
        df_spatial,
        entity_summary,
        aux=aux,
        entity_col=entity_col,
        gene_col=gene_col,
        out_col=out_col,
        coord_cols=coord_cols,
        dist_threshold=dist_threshold,
        neg_npmi_threshold=neg_npmi_threshold,
        unassigned_labels=unassigned_labels,
        only_partial_component=only_partial_component,
        debug_stages=debug_stages,
        debug_legacy_col=debug_legacy_col,
        show_progress=show_progress,
        in_place=in_place,
    )
