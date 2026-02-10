import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
import networkx as nx
from collections import Counter
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as scipy_cc
from torch_geometric.utils import to_networkx
import heapq
from collections import defaultdict
from tqdm.auto import tqdm
import concurrent.futures

# Cython accelerator for pruning
try:
    from . import _cy_prune as _cy_prune
except Exception:
    _cy_prune = None
    try:
        import pyximport
        pyximport.install(setup_args={"include_dirs": [np.get_include()]}, language_level=3)
        from . import _cy_prune as _cy_prune
    except Exception:
        _cy_prune = None

# Cython accelerator for spatial label-constrained components
try:
    from . import _cy_spatial as _cy_spatial
except Exception:
    _cy_spatial = None
    try:
        import pyximport
        pyximport.install(setup_args={"include_dirs": [np.get_include()]}, language_level=3)
        from . import _cy_spatial as _cy_spatial
    except Exception:
        _cy_spatial = None

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


# Attempt to replace the Python implementation with a Cython-compiled one
# If Cython and a compiler are available, this will import a compiled
# _cy_prune module; otherwise the pure-Python function above remains in use.
try:
    import importlib
    _cy = importlib.import_module("hotnerd._cy_prune")
    prune_genes_by_npmi_greedy = _cy.prune_genes_by_npmi_greedy
except Exception:
    try:
        import pyximport
        pyximport.install(language_level=3)
        import importlib
        _cy = importlib.import_module("hotnerd._cy_prune")
        prune_genes_by_npmi_greedy = _cy.prune_genes_by_npmi_greedy
    except Exception:
        # Leave the pure-Python implementation as a fallback
        pass

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


def prune_transcripts_fast(
    df,
    npmi_df,
    cell_id_col="cell_id",
    gene_col="feature_name",
    threshold=-0.1,
    unassigned_id="-1",
    n_jobs: int = 1,
    show_progress: bool = True,
):
    """
    Parallelized version of `prune_transcripts` with progress bars.

    - `n_jobs` controls number of worker threads (use -1 for all cores).
    - Uses thread-based parallelism to avoid copying large `W` matrix between processes.

    Behavior and returned columns match `prune_transcripts`.
    """
    df = df.copy()
    # Work directly with cell_id_col to avoid expensive conversion on 28M rows
    # Only convert cell IDs if they're not already strings
    if df[cell_id_col].dtype != 'object':
        df["_cell_str"] = df[cell_id_col].astype(str)
    else:
        df["_cell_str"] = df[cell_id_col]
    
    # Optimize gene conversion: avoid double .astype(str).str.strip() on 28M rows
    if df[gene_col].dtype != 'object':
        df[gene_col] = df[gene_col].astype(str)
    # Skip .str.strip() for performance; assume input is already clean or build_dense_npmi_matrix handles it
    # If needed, only strip during npmi_df processing, not on df

    genes, gene_to_idx, W = build_dense_npmi_matrix(npmi_df)
    df["_gene_idx"] = df[gene_col].map(gene_to_idx)

    # ---------- PASS 1 (parallelizable) ----------
    df["cell_id_npmi_cons_p1"] = df["_cell_str"]
    df["npmi_cons_p1_status"] = np.where(
        df["_cell_str"] == unassigned_id,
        "unassigned_input",
        "core",
    )

    partial_map = {}

    # Prepare per-cell unique gene lists (only cells that are not unassigned)
    grp = df[df["_cell_str"] != unassigned_id].groupby("_cell_str")["_gene_idx"].apply(
        lambda s: np.asarray(pd.Index(s.dropna().astype(int)).unique(), dtype=np.int32)
    )

    cell_items = list(grp.items())
    total_cells = len(cell_items)

    # normalize n_jobs
    if n_jobs is None or n_jobs == 0:
        n_jobs = 1
    if n_jobs < 0:
        try:
            import os

            n_jobs = max(1, os.cpu_count() or 1)
        except Exception:
            n_jobs = 1

    results = []

    # If compiled Cython accelerator is available, use it to process all cells
    if _cy_prune is not None:
        cell_ids = [cid for cid, _ in cell_items]
        g_arrays = [gl if (gl is not None and gl.size > 0) else None for _, gl in cell_items]
        try:
            removed_lists = _cy_prune.prune_cells(g_arrays, W, float(threshold))
        except Exception:
            removed_lists = None

        if removed_lists is not None:
            for cid, removed in zip(cell_ids, removed_lists):
                if removed:
                    partial_map[cid] = f"{cid}-1"
                    results.append((cid, removed))

        if show_progress:
            pbar = tqdm(total=total_cells, desc="prune_pass1")
            pbar.update(total_cells)
            pbar.close()
        else:
            pbar = None
    else:
        def _process_cell(item):
            cid, g_local = item
            if g_local.size <= 1:
                return cid, None
            keep_mask = prune_genes_by_npmi_greedy(g_local, W, threshold)
            removed = g_local[~keep_mask]
            if removed.size == 0:
                return cid, None
            return cid, removed.tolist()

        if show_progress:
            pbar = tqdm(total=total_cells, desc="prune_pass1")
        else:
            pbar = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_process_cell, it): it[0] for it in cell_items}
            for fut in concurrent.futures.as_completed(futures):
                cid, removed = fut.result()
                if removed is not None:
                    partial_map[cid] = f"{cid}-1"
                    results.append((cid, removed))
                if pbar is not None:
                    pbar.update(1)
        if pbar is not None:
            pbar.close()

    # Apply pass1 removals to df in a vectorized batch
    if results:
        # build mapping table of (cell_str, gene_idx) -> pid
        rows = []
        for cid, removed in results:
            pid = partial_map[cid]
            for g in removed:
                rows.append((cid, int(g), pid))

        if rows:
            map_df = pd.DataFrame(rows, columns=["_cell_str_map", "_gene_idx_map", "_pid"])

            # prepare indexed view of original df for efficient merging
            df_idx = df.reset_index().rename(columns={"index": "_orig_index"})[["_orig_index", "_cell_str", "_gene_idx"]]
            # coerce gene idx to pandas nullable Int to allow exact matching
            df_idx["_gene_idx"] = df_idx["_gene_idx"].astype("Int64")
            map_df["_gene_idx_map"] = map_df["_gene_idx_map"].astype("Int64")

            merged = pd.merge(
                df_idx,
                map_df,
                left_on=["_cell_str", "_gene_idx"],
                right_on=["_cell_str_map", "_gene_idx_map"],
                how="inner",
            )

            if not merged.empty:
                # assign in one vectorized operation
                df.loc[merged["_orig_index"], "cell_id_npmi_cons_p1"] = merged["_pid"].values
                df.loc[merged["_orig_index"], "npmi_cons_p1_status"] = "partial_p1"

    # show that pass1 application is complete
    if show_progress:
        tqdm(desc="apply_pass1", total=1).update(1)

    # ---------- PASS 2 (parallelizable over partials) ----------
    df["cell_id_npmi_cons_p2"] = df["cell_id_npmi_cons_p1"]
    df["npmi_cons_p2_status"] = "unchanged"

    pids = list(set(partial_map.values()))
    if pids:
        # prepare per-partial unique gene lists
        grp_p = df[df["cell_id_npmi_cons_p1"].isin(pids)].groupby("cell_id_npmi_cons_p1")["_gene_idx"].apply(
            lambda s: np.asarray(pd.Index(s.dropna().astype(int)).unique(), dtype=np.int32)
        )

        partial_items = list(grp_p.items())
        total_partials = len(partial_items)

        if show_progress:
            pbar2 = tqdm(total=total_partials, desc="prune_pass2")
        else:
            pbar2 = None

        results2 = []

        # If cython accelerator is available, use it for partials
        if _cy_prune is not None:
            pids = [pid for pid, _ in partial_items]
            g_arrays = [gl if (gl is not None and gl.size > 0) else None for _, gl in partial_items]
            try:
                removed_lists = _cy_prune.prune_cells(g_arrays, W, float(threshold))
            except Exception:
                removed_lists = None

            if removed_lists is not None:
                # update progress incrementally so the bar reflects work being applied
                for pid, removed in zip(pids, removed_lists):
                    if removed:
                        results2.append((pid, removed))
                    if pbar2 is not None:
                        pbar2.update(1)
                if pbar2 is not None:
                    pbar2.close()
        else:
            def _process_partial(item):
                pid, g_local = item
                if g_local.size <= 1:
                    return pid, None
                keep_mask = prune_genes_by_npmi_greedy(g_local, W, threshold)
                removed = g_local[~keep_mask]
                if removed.size == 0:
                    return pid, None
                return pid, removed.tolist()

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as ex:
                futures = {ex.submit(_process_partial, it): it[0] for it in partial_items}
                for fut in concurrent.futures.as_completed(futures):
                    pid, removed = fut.result()
                    if removed is not None:
                        results2.append((pid, removed))
                    if pbar2 is not None:
                        pbar2.update(1)
            if pbar2 is not None:
                pbar2.close()

        # Apply pass2 changes in a vectorized fashion
        # results2: list of (pid, removed)
        rows2 = []
        removed_pids = set()
        for pid, removed in results2:
            removed_pids.add(pid)
            for g in removed:
                rows2.append((pid, int(g)))

        # prepare df index view
        df_idx2 = df.reset_index().rename(columns={"index": "_orig_index"})[["_orig_index", "cell_id_npmi_cons_p1", "_gene_idx"]]
        df_idx2["_gene_idx"] = df_idx2["_gene_idx"].astype("Int64")

        if rows2:
            map2 = pd.DataFrame(rows2, columns=["_pid_map", "_gene_idx_map"]).astype({"_gene_idx_map": "Int64"})
            merged2 = pd.merge(
                df_idx2,
                map2,
                left_on=["cell_id_npmi_cons_p1", "_gene_idx"],
                right_on=["_pid_map", "_gene_idx_map"],
                how="inner",
            )

            if not merged2.empty:
                # these rows should be unassigned_from_partial
                df.loc[merged2["_orig_index"], "cell_id_npmi_cons_p2"] = unassigned_id
                df.loc[merged2["_orig_index"], "npmi_cons_p2_status"] = "unassigned_from_partial"

        # Mark remaining rows for all partial pids as partial_p2 in one vectorized operation
        if pids:
            pids_set = set(pids)
            mask_pid_any = df["cell_id_npmi_cons_p1"].isin(pids_set)
            mask_unassigned = df["npmi_cons_p2_status"] == "unassigned_from_partial"
            mask_keep_any = mask_pid_any & (~mask_unassigned)
            if mask_keep_any.any():
                df.loc[mask_keep_any, "npmi_cons_p2_status"] = "partial_p2"

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


def _prune_one_component(comp_id, g_local, prune_fn, W, npmi_threshold):
    """
    Prune genes for a single component.
    Returns: (comp_id, removed_gene_ids)
    """
    if g_local.size <= 1:
        return (comp_id, np.array([], dtype=np.int32))
    
    kept_mask = prune_fn(g_local, W, threshold=npmi_threshold)
    removed_gene_ids = g_local[~kept_mask]
    return (comp_id, removed_gene_ids)


def _prune_components_parallel(comp_gene_map, df, keep_candidate, comp_mask_template, 
                                prune_fn, W, npmi_threshold, gene_idx_all, show_progress=True,
                                n_workers=4):
    """
    Prune components in parallel using ThreadPoolExecutor.
    Updates df in-place with pruning results.
    """
    comp_items = list(comp_gene_map.items())
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_prune_one_component, comp_id, g_local, prune_fn, W, npmi_threshold): comp_id
            for comp_id, g_local in comp_items
        }
        
        iterator = concurrent.futures.as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="prune_comps")
        
        for future in iterator:
            try:
                comp_id, removed_gene_ids = future.result()
                
                comp_mask = (df["unassigned_comp_id"] == comp_id) & keep_candidate
                
                if removed_gene_ids.size == 0:
                    df.loc[comp_mask, "unassigned_qc_status"] = "keep_unassigned_comp"
                else:
                    removed_set = set(map(int, removed_gene_ids.tolist()))
                    drop_gene_mask = comp_mask & gene_idx_all.isin(removed_set)
                    df.loc[comp_mask & (~drop_gene_mask), "unassigned_qc_status"] = "keep_unassigned_comp"
                    df.loc[drop_gene_mask, "unassigned_qc_status"] = "drop_npmi_pruned_gene"
            except Exception as e:
                if show_progress:
                    print(f"[ERROR] Component pruning failed: {e}")
                continue


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
):
    """
    Faster variant of `annotate_unassigned_components`.
    - Uses `_cy_prune.prune_cells` when available to prune per-component gene lists in bulk.
    - Shows a progress bar if `show_progress` is True.
    """
    df = df_pruned.copy()

    if transcript_id_col not in df.columns:
        df[transcript_id_col] = df.index.astype(str)

    df[gene_col] = df[gene_col].astype(str).str.strip()

    if unassigned_final_col in df.columns:
        is_unassigned = df[unassigned_final_col].astype(str) == "-1"
        assigned_id_series = df[unassigned_final_col].astype(str)
    else:
        is_unassigned = df[cell_id_col].astype(str) == "-1"
        assigned_id_series = df[cell_id_col].astype(str)

    df["unassigned_comp_id"] = pd.Series(index=df.index, dtype="object")
    df["unassigned_qc_status"] = pd.Series(index=df.index, dtype="object")
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
    large_comp_ids = df.loc[keep_candidate, "unassigned_comp_id"].unique()

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
    for comp_id, group in df_candidate.groupby("unassigned_comp_id"):
        g_local = group["_gene_idx_local"].dropna().astype(int).unique()
        if g_local.size > 0:
            comp_gene_map[comp_id] = np.asarray(g_local, dtype=np.int32)
    
    if show_progress:
        pbar_groupby.update(1)
        pbar_groupby.close()

    # If cython prune available, do bulk pruning
    if _cy_prune is not None and len(comp_gene_map) > 0:
        if show_progress:
            print(f"[INFO] Using Cython-accelerated pruning ({len(comp_gene_map)} components)")
        comp_keys = list(comp_gene_map.keys())
        g_arrays = [comp_gene_map[k] if comp_gene_map[k].size > 0 else None for k in comp_keys]
        removed_lists = None
        try:
            removed_lists = _cy_prune.prune_cells(g_arrays, W, float(npmi_threshold))
        except Exception as e:
            if show_progress:
                print(f"[WARNING] Cython pruning failed: {e}, falling back to Python")
            removed_lists = None

        if removed_lists is not None:
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
                
                # Check which of these transcripts have genes in the removed set (faster)
                removed_set = set(map(int, removed))
                comp_gene_mask = gene_idx_all.iloc[comp_indices].isin(removed_set).to_numpy()
                drop_gene_mask_all[comp_indices[comp_gene_mask]] = True
                
                # Mark kept ones
                df.loc[comp_indices[~comp_gene_mask], "unassigned_qc_status"] = "keep_unassigned_comp"
            
            # Apply all drops at once
            df.loc[drop_gene_mask_all, "unassigned_qc_status"] = "drop_npmi_pruned_gene"
        else:
            # fallback to Python pruning per component with progress (parallelized)
            if show_progress:
                print(f"[WARNING] Cython not available, using Python pruning ({len(comp_gene_map)} components)")
            _prune_components_parallel(
                comp_gene_map, df, keep_candidate, comp_mask_template=(df["unassigned_comp_id"], df.index),
                prune_fn=prune_fn, W=W, npmi_threshold=npmi_threshold, 
                gene_idx_all=gene_idx_all, show_progress=show_progress
            )
    else:
        # No cython or no comps: fallback to Python loop with optional progress
        if len(comp_gene_map) > 0:
            if show_progress:
                print(f"[WARNING] Cython not available, using Python pruning ({len(comp_gene_map)} components)")
            _prune_components_parallel(
                comp_gene_map, df, keep_candidate, comp_mask_template=(df["unassigned_comp_id"], df.index),
                prune_fn=prune_fn, W=W, npmi_threshold=npmi_threshold,
                gene_idx_all=gene_idx_all, show_progress=show_progress
            )

    cell_id_final = assigned_id_series.copy()
    kept_unassigned = is_unassigned & (df["unassigned_qc_status"] == "keep_unassigned_comp")
    cell_id_final.loc[kept_unassigned] = df.loc[kept_unassigned, "unassigned_comp_id"].astype(str)

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
    return sorted(edges)


def delaunay_edges(points: np.ndarray):
    """
    points: (N, D) with D=2 or 3
    Returns a list of undirected edges (i, j) with i<j
    """
    tri = Delaunay(points)
    simplices = tri.simplices  # (n_simp, D+1)
    return _edges_from_simplices(simplices)





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
    penalize_simplicity=True,
):
    """
    ReLU-based ΔC = C_union - max(C_u, C_v) (if not penalize_simplicity)
    
    Uses ReLU-based coherence scoring for more robust cluster merging.
    """
    # individual
    C_u, _, _ = coherence_C_from_genes_relu(genes_u, npmi_mat, tau=tau)
    C_v, _, _ = coherence_C_from_genes_relu(genes_v, npmi_mat, tau=tau)

    # union
    union = np.unique(np.concatenate([genes_u, genes_v]))
    C_union, _, _ = coherence_C_from_genes_relu(union, npmi_mat, tau=tau)

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
    tau=0.05,
    use_relu=True,
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
    labels = base_labels.unique()
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

        # sort by size descending
        comps_sorted = sorted(comps, key=len, reverse=True)

        # largest keeps original label; others get label-2, label-3, ...
        for i, comp_nodes in enumerate(comps_sorted):
            if i == 0:
                new_label = label
            else:
                new_label = f"{label}-{i+1}"

            comp_nodes = np.array(list(comp_nodes), dtype=int)
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
):
    """
    Fast variant of `enforce_spatial_coherence` with a progress bar.
    """
    df = df_stitched.copy()
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

    # Try the Cython path: compute label-constrained components in one pass
    if _cy_spatial is not None and hasattr(data_all, "edge_index") and data_all.edge_index.numel() > 0:
        edge_index = data_all.edge_index.numpy()
        src = edge_index[0].astype(np.int32)
        dst = edge_index[1].astype(np.int32)

        labels_arr = base_labels.to_numpy()
        # map labels to integer codes; treat DROP/nan as invalid (-1)
        lab_codes = pd.Categorical(labels_arr)
        codes = lab_codes.codes.astype(np.int64)
        # mark invalids
        invalid = (labels_arr == "DROP") | (labels_arr == "nan")
        if invalid.any():
            codes = codes.copy()
            codes[invalid] = -1

        # compute components constrained by labels
        try:
            roots = _cy_spatial.label_constrained_components(int(n), src, dst, codes, -1)
        except Exception:
            roots = None

        if roots is not None:
            # For each label code, split by root and assign suffixes for non-largest comps
            out = df[out_col].to_numpy(dtype=object)
            # iterate unique valid codes
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
                order = np.argsort(counts)[::-1]
                lab = str(lab_codes.categories[c])
                for i, oi in enumerate(order):
                    r = uniq_r[oi]
                    if i == 0:
                        new_lab = lab
                    else:
                        new_lab = f"{lab}-{i+1}"
                    sel = (roots_c == r)
                    if i == 0:
                        # ensure main stays as original (optional, already default)
                        continue
                    out[idx[sel]] = new_lab
            df[out_col] = out
            df = df.drop(columns=["__node_idx"])  # not used in this path
            return df

    # Fallback: original per-label subgraph approach with progress bar (optimized)
    G = to_networkx(data_all, to_undirected=True)

    labels = base_labels.unique()
    iterable = labels
    if show_progress:
        iterable = tqdm(labels, desc="spatial_labels")
    
    # Pre-compute for faster indexing
    node_idx_arr = df["__node_idx"].to_numpy()
    base_labels_arr = base_labels.to_numpy()
    out_arr = df[out_col].to_numpy(dtype=object)

    for label in iterable:
        if label == "DROP" or label == "nan":
            continue

        # Use numpy boolean indexing (faster than pandas)
        label_mask = (base_labels_arr == label)
        node_idx = node_idx_arr[label_mask]
        label_positions = np.where(label_mask)[0]

        if node_idx.size <= 1:
            continue

        subG = G.subgraph(node_idx)
        comps = list(nx.connected_components(subG))

        if len(comps) <= 1:
            continue

        comps_sorted = sorted(comps, key=len, reverse=True)
        
        # Build node -> position mapping for fast lookup
        node_to_pos = {node: label_positions[i] for i, node in enumerate(node_idx)}
        
        # Vectorized assignment: collect all updates then apply once
        for i, comp_nodes in enumerate(comps_sorted):
            if i == 0:
                new_label = label
            else:
                new_label = f"{label}-{i+1}"
            
            # Get positions for all nodes in this component
            positions = [node_to_pos[node] for node in comp_nodes if node in node_to_pos]
            if positions:
                out_arr[positions] = new_label
    
    df[out_col] = out_arr

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
    if unassigned_labels is None:
        unassigned_labels = {"DROP", "-1", "UNASSIGNED", "nan"}
    
    df = df_spatial.copy()
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
    entity_coords = entities[list(coord_cols)].to_numpy(dtype=np.float32)
    entity_ids = entities["entity_id"].to_numpy(dtype=object)
    
    # Build KNN index for fast nearest-neighbor lookup
    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", metric="euclidean")
    knn.fit(entity_coords)
    
    # Find nearest entity for each unassigned transcript
    distances, indices = knn.kneighbors(unassigned_coords)
    distances = distances.ravel()
    indices = indices.ravel()
    
    # Determine which unassigned transcripts should be reassigned
    within_threshold = distances <= dist_threshold
    n_reassigned = within_threshold.sum()
    
    # Reassign transcripts within threshold
    for i, unassigned_pos in enumerate(unassigned_idx):
        if within_threshold[i]:
            entity_idx = indices[i]
            new_label = entity_ids[entity_idx]
            df.loc[unassigned_pos, out_col] = new_label
    
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
    )

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
def relu_symmetric(x, tau):
    """
    Two-sided ReLU with dead zone [-tau, tau].
    
    Values in [-tau, tau] are zeroed out.
    Values above tau are shifted down by tau.
    Values below -tau are shifted up by tau.
    
    Parameters
    ----------
    x : array_like
        Input values
    tau : float
        Dead zone threshold
        
    Returns
    -------
    out : np.ndarray
        ReLU-transformed values
    """
    out = np.zeros_like(x)
    out[x > tau] = x[x > tau] - tau
    out[x < -tau] = x[x < -tau] + tau
    return out

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
    n_cells = M.shape[0]
    purity = np.zeros(n_cells, dtype=np.float32)
    conflict = np.zeros(n_cells, dtype=np.float32)
    relative_purity = np.zeros(n_cells, dtype=np.float32)
    relative_conflict = np.zeros(n_cells, dtype=np.float32)
    signal_strength = np.zeros(n_cells, dtype=np.float32)

    for i in range(n_cells):
        present = np.where(M[i] == 1)[0]
        k = len(present)

        if k < 2:
            # All metrics remain 0
            continue

        gi = col_idx[present]
        sub = npmi_mat[np.ix_(gi, gi)]
        vals = sub[np.triu_indices_from(sub, k=1)]

        if vals.size == 0:
            continue

        # Apply symmetric ReLU
        rvals = relu_symmetric(vals, tau)

        K = k * (k - 1) / 2
        pos_sum = np.sum(np.maximum(rvals, 0.0))
        neg_sum = np.sum(np.maximum(-rvals, 0.0))
        total_abs = pos_sum + neg_sum

        # Absolute metrics (normalized by number of pairs)
        purity[i] = float(pos_sum / K)
        conflict[i] = float(neg_sum / K)
        signal_strength[i] = float(total_abs)

        # Relative metrics (normalized by total signal)
        if total_abs > eps:
            relative_purity[i] = float(pos_sum / total_abs)
            relative_conflict[i] = float(neg_sum / total_abs)

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
