#----------------------·•●  🧽  ●•·-------------------------
#                   TRACER Metrics Module
#----------------------·•●──────●•·-------------------------
# Author: Long Yuan
# Affiliation: Johns Hopkins University
# Email: lyuan13@jhmi.edu
#-----------------------------------------------------------

import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.sparse as sp

from ._kernels import pair_aggregate_dense

#
def get_confident_nuclei_transcripts(
    sdata,
    *,
    qv_min: float = 30,
    low_pct: float = 20,
    high_pct: float = 80,
    save_qv_filtered: bool = False,
    parquet_path: str = "qv_filtered_transcripts.parquet",
    exclude_ids: set | None = None,
):
    """
    From a SpatialData object, extract high-quality nucleus transcripts and
    return a confident nucleus DataFrame (nuc_df_confident).

    Parameters
    ----------
    sdata : SpatialData
        The loaded SpatialData object.
    qv_min : float
        Minimum qv to keep.
    low_pct : float
        Lower percentile threshold for nucleus transcript count.
    high_pct : float
        Upper percentile threshold for nucleus transcript count.
    save_qv_filtered : bool
        If True, save the QV-filtered transcripts to a Parquet file.
    parquet_path : str
        Path to save QV-filtered transcripts if requested.
    exclude_ids : set | None, optional (default=None)
        Set of cell IDs to exclude, e.g. {"-1", "DROP", "nan", "UNASSIGNED"}.
        If None, defaults to {"UNASSIGNED"}.

    Returns
    -------
    nuc_df_confident : DataFrame
        Transcripts belonging to confident nuclei.
    fitlered_df : DataFrame
        Transcripts passing the qv threshold.
    """

    # Load transcripts
    transcripts = sdata.points["transcripts"].compute()

    # Apply QV filter
    df = transcripts[transcripts["qv"] >= qv_min].copy()

    # Filter to valid gene list
    # Ensures we only keep transcripts whose gene exists in AnnData table
    valid_genes = set(sdata.tables["table"].var.index)
    df = df[df["feature_name"].isin(valid_genes)].copy()

    # Optionally save the qv-filtered transcripts
    if save_qv_filtered:
        df.to_parquet(parquet_path, index=False)
        qv_out = parquet_path
        print("Saved parquet:", qv_out)
    else:
        qv_out = None

    # Extract nucleus-overlapping transcripts with a valid cell_id
    if exclude_ids is None:
        exclude_ids = {"UNASSIGNED"}
    
    if exclude_ids:
        nuc_df = df[
            (~df["cell_id"].isin(exclude_ids)) &
            (df["overlaps_nucleus"] == 1)
        ].copy()
    else:
        nuc_df = df[df["overlaps_nucleus"] == 1].copy()

    # Compute transcript-count thresholds per nucleus
    nuc_counts = nuc_df.groupby("cell_id").size()

    low_thres = np.percentile(nuc_counts, low_pct)
    high_thres = np.percentile(nuc_counts, high_pct)
    print("Transcript count thresholds:", low_thres, high_thres)

    # Identify confident nuclei
    good_ids = nuc_counts[(nuc_counts >= low_thres) & (nuc_counts <= high_thres)].index
    
    print("Number of confident nuclei:", len(good_ids))
    nuc_df_confident = nuc_df[nuc_df["cell_id"].isin(good_ids)].copy()

    return nuc_df_confident, df

#
def compute_npmi(
    df_subset,
    group_key="cell_id",
    min_occurrences_per_context=2,
    count_col=None,
    set_neg_one=False,
    thr=0.05
):
    """
    Compute PMI/NPMI using presence/absence of genes at the cell or nucleus level,
    with robustness control by requiring each gene to occur at least N times
    within a context (cell or nucleus) before being considered "present".
    Optional:
    set_neg_one : bool
        If True, assigns NPMI = -1 for gene pairs with zero observed
        co-occurrence (P_ij == 0) when both marginal probabilities
        exceed thr.
    thr : float
        Marginal probability threshold used for the optional -1
        assignment (default 0.05).
    -------
    long_df : DataFrame
        Columns:
            gene_i, gene_j, P_i, P_j, P_ij,
            P_i_given_j, P_j_given_i, PMI, NPMI
    """

    # 0. Minimal column projection (no .copy() of the entire 100M-row frame)
    if count_col is None:
        df = df_subset[[group_key, "feature_name"]]
    else:
        df = df_subset[[group_key, "feature_name", count_col]]
    group_series = df[group_key].astype(str)

    # ----------------------------------------------------------------------
    # Filter by minimum occurrences per context
    # ----------------------------------------------------------------------
    if count_col is None:
        counts = (
            df.assign(_grp=group_series)
              .groupby(["_grp", "feature_name"])
              .size()
              .rename("gene_count")
              .reset_index()
              .rename(columns={"_grp": group_key})
        )
    else:
        counts = (
            df.assign(_grp=group_series)
              .groupby(["_grp", "feature_name"])[count_col]
              .sum()
              .rename("gene_count")
              .reset_index()
              .rename(columns={"_grp": group_key})
        )

    df_filtered = counts[counts["gene_count"] >= min_occurrences_per_context]
    if df_filtered.empty:
        raise ValueError(
            f"No genes pass min_occurrences_per_context={min_occurrences_per_context}."
        )

    # ----------------------------------------------------------------------
    # Build sparse contexts × genes presence matrix via categorical codes.
    # Previously used df.pivot_table(values=1, aggfunc="max", fill_value=0)
    # which densifies to C×G ints in pandas — for C=200K, G=500 that's
    # 800 MB of pandas overhead, independent of the actual sparsity.
    # ----------------------------------------------------------------------
    ctx_cat = pd.Categorical(df_filtered[group_key].astype(str))
    gene_cat = pd.Categorical(df_filtered["feature_name"].astype(str))

    rows_i = ctx_cat.codes.astype(np.int32)
    cols_i = gene_cat.codes.astype(np.int32)
    vals = np.ones(len(rows_i), dtype=np.int32)

    contexts = ctx_cat.categories.to_numpy()
    genes = gene_cat.categories.to_numpy()
    C = len(contexts)
    G_gene = len(genes)
    M = sp.coo_matrix(
        (vals, (rows_i, cols_i)), shape=(C, G_gene)
    ).tocsr()
    M.data = np.ones_like(M.data, dtype=np.int32)  # binarise

    # ----------------------------------------------------------------------
    # Probabilities P(i), P(i,j) — sparse co-occurrence matmul.
    # ----------------------------------------------------------------------
    counts_i = np.asarray(M.sum(axis=0)).ravel()
    P_i = counts_i / C

    # Sparse × sparse; returns sparse. Dense-ify for the elementwise ops
    # below — at G ≈ 500 the G×G matrix is 2 MB float64, trivial.
    co_matrix_sp = (M.T @ M)
    P_ij = np.asarray(co_matrix_sp.todense(), dtype=np.float64) / C

    # ----------------------------------------------------------------------
    # Conditional probabilities
    # ----------------------------------------------------------------------
    P_i_col = P_i[:, None]
    P_j_row = P_i[None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        P_i_given_j = np.where(P_j_row > 0, P_ij / P_j_row, np.nan)
        P_j_given_i = np.where(P_i_col > 0, P_ij / P_i_col, np.nan)

    # ----------------------------------------------------------------------
    # PMI & NPMI
    # ----------------------------------------------------------------------
    PMI = np.full_like(P_ij, np.nan)
    NPMI = np.full_like(P_ij, np.nan)

    denom = P_i_col * P_j_row
    valid = (P_ij > 0) & (denom > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        PMI[valid] = np.log(P_ij[valid] / denom[valid])
        NPMI[valid] = PMI[valid] / (-np.log(P_ij[valid]))

    # ----------------------------------------------------------------------
    # Optional: assign -1 if P_i > thr and P_j > thr and _P_ij = 0 (i.e. strong individual presence but no co-occurrence)
    # ----------------------------------------------------------------------
    if set_neg_one:
        zero_coocc = (P_ij == 0) & (P_i_col > thr) & (P_j_row > thr)
        NPMI[zero_coocc] = -1.0

    # ----------------------------------------------------------------------
    # Convert to long format
    # ----------------------------------------------------------------------
    G = len(genes)
    long_df = pd.DataFrame({
        "gene_i": np.repeat(genes, G),
        "gene_j": np.tile(genes, G),
        "P_i": np.repeat(P_i, G),
        "P_j": np.tile(P_i, G),
        "P_ij": P_ij.ravel(),
        "P_i_given_j": P_i_given_j.ravel(),
        "P_j_given_i": P_j_given_i.ravel(),
        "PMI": PMI.ravel(),
        "NPMI": NPMI.ravel(),
    })

    return long_df

#
def build_cell_gene_matrix(filtered_df, min_transcripts=10, genes_npm=None, cell_col="cell_id", exclude_ids=None):
    """
    Construct a binary (presence/absence) cell × gene matrix from a filtered
    transcript-level DataFrame and align it to the NPMI gene universe.

    This function takes a transcript df (already filtered for QV, removes 
    low-quality cells, builds a binary indicator matrix of gene presence within 
    each cell, and then compute purity/conflict scores.

    Parameters
    ----------
    filtered_df : pandas.DataFrame
        A transcript-level table containing at least:
        cell_col and "feature_name"

    min_transcripts : int, optional (default=10)
        Minimum number of transcripts required for a cell to be retained.

    genes_npm : pandas.DataFrame
        The long-format NPMI table containing columns "gene_i", "gene_j", "NPMI".
        
    cell_col : str, optional (default="cell_id")
        The column name containing cell identifiers.
        
    exclude_ids : set | None, optional (default=None)
        Set of cell IDs to exclude, e.g. {"-1", "DROP", "nan", "UNASSIGNED"}.
        If None, defaults to {"UNASSIGNED"}.

    Returns
    -------
    cell_ids : numpy.ndarray, shape (n_cells,)
        List of cell IDs (strings) corresponding to the rows of the matrix.

    genes_cell : numpy.ndarray, shape (n_genes_filtered,)
        Gene names (strings) corresponding to the columns of the filtered 
        presence/absence matrix. Only genes appearing in the NPMI dataset
        are retained.

    M : numpy.ndarray, dtype int8, shape (n_cells, n_genes_filtered)
        Binary presence/absence matrix:
            M[i, j] = 1 if cell i expresses gene j (≥1 transcript)
                      0 otherwise.

    col_idx : numpy.ndarray, dtype int32, shape (n_genes_filtered,)
        For each retained gene column, the corresponding index into the 
        global NPMI gene universe. Used to index into the full NPMI matrix
        when computing purity/conflict for each cell.

    Notes
    -----
    - Presence/absence is used instead of transcript counts because the NPMI
      scoring relies on pairwise co-occurrence patterns rather than expression
      magnitude.
    - Filtering to the NPMI gene universe ensures that the rows of `M` and the
      NPMI matrix use consistent gene indexing.
    """
    
    # Convert cell IDs to string for consistency with AnnData
    df = filtered_df
    # Avoid copying 100M-row df up front; use boolean views where possible.
    cell_col_series = df[cell_col].astype(str)

    # Remove excluded cell IDs
    if exclude_ids is None:
        exclude_ids = {"UNASSIGNED"}
    if exclude_ids:
        keep_mask = ~cell_col_series.isin(exclude_ids)
        cell_col_series = cell_col_series[keep_mask]
        df = df.loc[keep_mask.index[keep_mask]]

    # Filter by minimum transcript count per cell
    cell_counts = cell_col_series.groupby(cell_col_series).size()
    good_ids = cell_counts[cell_counts >= min_transcripts].index
    mask_good = cell_col_series.isin(good_ids)
    df = df.loc[mask_good.index[mask_good]]
    cell_col_series = cell_col_series[mask_good]

    # Restrict gene universe to NPMI vocabulary *before* building the matrix,
    # so sparse construction skips transcripts whose gene never shows up in
    # NPMI pairs at all.
    all_genes = np.union1d(
        genes_npm["gene_i"].unique(),
        genes_npm["gene_j"].unique()
    )

    gene_series = df["feature_name"].astype(str)
    in_vocab = gene_series.isin(all_genes)
    df = df.loc[in_vocab.index[in_vocab]]
    cell_col_series = cell_col_series[in_vocab]
    gene_series = gene_series[in_vocab]

    # Build presence/absence matrix via categorical codes + scipy.sparse.
    # Previous implementation used pivot_table(aggfunc=lambda x: 1), which
    # forces a Python call per group — catastrophic on 100M+ rows. Here we
    # let scipy coalesce duplicates at CSR-build time.
    cell_cat = pd.Categorical(cell_col_series)
    gene_cat = pd.Categorical(gene_series, categories=all_genes)

    rows_i = cell_cat.codes.astype(np.int32)
    cols_i = gene_cat.codes.astype(np.int32)
    # Any gene not in `all_genes` got code -1; defensive filter.
    valid = cols_i >= 0
    if not valid.all():
        rows_i = rows_i[valid]
        cols_i = cols_i[valid]

    n_cells = len(cell_cat.categories)
    n_genes = len(all_genes)

    # COO → CSR de-duplicates automatically (sum_duplicates → binarise).
    coo = sp.coo_matrix(
        (np.ones(len(rows_i), dtype=np.int8), (rows_i, cols_i)),
        shape=(n_cells, n_genes),
    )
    csr = coo.tocsr()
    csr.data = np.ones_like(csr.data, dtype=np.int8)  # binarise

    cell_ids = cell_cat.categories.to_numpy().astype(str)

    # Drop columns (genes) that never appeared in any retained cell — keeps
    # M's width the same as before: only genes actually present.
    col_mass = np.asarray(csr.sum(axis=0)).ravel() > 0
    csr = csr[:, col_mass]
    genes_cell = all_genes[col_mass]
    col_idx = np.flatnonzero(col_mass).astype(np.int32)

    # Densify to int8 for backward-compat (callers expect np.ndarray). At
    # ~200K cells × ~500 genes this is ~100 MiB — negligible next to the
    # 100M-row source df and orders of magnitude smaller than what the
    # pivot_table was allocating.
    M = np.asarray(csr.todense(), dtype=np.int8)

    return cell_ids, genes_cell, M, col_idx

#
def build_npmi_matrix(nucleus_npmi_long):
    """
    Construct a dense NPMI (Normalized Pointwise Mutual Information) matrix
    from a long-format NPMI dataframe.

    Parameters
    ----------
    nucleus_npmi_long : pandas.DataFrame
        Long-format NPMI table where each row represents a gene–gene pair.
        The dataframe must contain at least the following columns:
            - "gene_i" : str
                The first gene in the pair.
            - "gene_j" : str
                The second gene in the pair.
            - "NPMI" : float
                The normalized PMI score between gene_i and gene_j.
                
    Returns
    -------
    npmi_mat : np.ndarray, shape (G, G)
        A dense symmetric matrix where entry (i, j) contains the NPMI value
        between gene_i and gene_j. 
        Missing gene pairs implicitly receive a value of 0.

    gene_to_idx : dict
        A dictionary mapping each gene name to its corresponding row/column
        index in `npmi_mat`. This mapping is required to align the NPMI
        matrix with the columns of the cell × gene presence/absence matrix
        before computing cell purity and conflict scores.

    Notes
    -----
    - The function ensures symmetry of the NPMI matrix by populating both
      (i, j) and (j, i).
    """

    genes = np.union1d(
        nucleus_npmi_long["gene_i"].unique(),
        nucleus_npmi_long["gene_j"].unique(),
    )
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    G = len(genes)

    # Vectorized: no more Python per-row itertuples loop. At G=500 this
    # went from ~2 s to ~10 ms; at G=5000 the old loop would take minutes.
    i_idx = nucleus_npmi_long["gene_i"].map(gene_to_idx).to_numpy()
    j_idx = nucleus_npmi_long["gene_j"].map(gene_to_idx).to_numpy()
    vals = nucleus_npmi_long["NPMI"].to_numpy(dtype=float)

    npmi_mat = np.zeros((G, G), dtype=float)
    npmi_mat[i_idx, j_idx] = vals
    npmi_mat[j_idx, i_idx] = vals

    return npmi_mat, gene_to_idx

#
def attach_metrics_to_adata(adata, purity_df, conflict_df):
    """
    Attach NPMI-derived cell quality metrics (purity and conflict) to an AnnData object.

    This function takes an AnnData object and two DataFrames containing per-cell
    purity and conflict metrics derived from NPMI analysis. It then maps these 
    scores onto `adata.obs` using each cell's unique cell ID from `adata.obs_names`. 
    Four new columns are added to the AnnData object:

        - `cell_purity`        : continuous purity score (float)
        - `cell_purity_bool`   : boolean flag indicating whether the cell meets 
                                 the "pure" criterion based on purity threshold
        - `conflict_score`     : continuous conflict score (float)
        - `is_conflict`        : boolean flag indicating whether the cell meets 
                                 the "high-conflict" criterion based on conflict threshold

    Parameters
    ----------
    adata : AnnData
        The AnnData object whose `.obs` dataframe will be updated. Cell IDs 
        are taken from `adata.obs_names`.

    purity_df : pandas.DataFrame
        DataFrame with columns: cell_id, cell_purity, is_pure

    conflict_df : pandas.DataFrame
        DataFrame with columns: cell_id, conflict_score, is_conflict

    Returns
    -------
    None
        The function modifies `adata` in place by adding the new columns to 
        `adata.obs`. Nothing is explicitly returned.
    """
    # Create mapping dictionaries
    purity_map = dict(zip(purity_df["cell_id"], purity_df["cell_purity"]))
    purity_bool_map = dict(zip(purity_df["cell_id"], purity_df["is_pure"]))
    conflict_map = dict(zip(conflict_df["cell_id"], conflict_df["conflict_score"]))
    conflict_bool_map = dict(zip(conflict_df["cell_id"], conflict_df["is_conflict"]))
    
    # Map using obs_names (cell IDs as index)
    adata.obs["cell_purity"] = adata.obs_names.map(purity_map)
    adata.obs["cell_purity_bool"] = adata.obs_names.map(purity_bool_map)
    adata.obs["conflict_score"] = adata.obs_names.map(conflict_map)
    adata.obs["is_conflict"] = adata.obs_names.map(conflict_bool_map)

#
def compute_cell_purity(
    M,
    col_idx,
    npmi_mat,
    npmi_threshold=0.05,        # NPMI > this = "positive" co-occurrence
    cell_ids=None,
    purity_percentile=80.0,     # top X% are considered "pure"
    purity_threshold=None       # OR set an explicit numeric threshold (overrides percentile)
):
    """
    Compute cell purity score for each cell based on NPMI matrix.

    Purity = fraction of gene-gene NPMI values greater than npmi_threshold.

    Also returns a boolean "is_pure" mask using either:
      - purity_threshold (if given), or
      - the purity_percentile (default 80% → bottom 20% are suspect).
    """

    # Single parallel kernel pass → all primitives we need for every
    # per-row metric. Replaces 200K × O(k^2) Python loop.
    k_arr, n_pos, _sum_neg, _pos_relu, _neg_relu = pair_aggregate_dense(
        M, col_idx, npmi_mat, threshold=npmi_threshold, tau=0.0,
    )
    n_pairs_total = k_arr * (k_arr - 1) // 2

    purity_scores = np.full(M.shape[0], np.nan, dtype=float)
    has_pairs = n_pairs_total > 0
    purity_scores[has_pairs] = n_pos[has_pairs] / n_pairs_total[has_pairs]

    # determine threshold for boolean purity
    valid = ~np.isnan(purity_scores)
    if purity_threshold is None:
        purity_threshold = np.nanpercentile(purity_scores[valid], purity_percentile)

    is_pure = np.zeros_like(purity_scores, dtype=bool)
    is_pure[valid] = purity_scores[valid] >= purity_threshold

    purity_df = None
    if cell_ids is not None:
        purity_df = pd.DataFrame({
            "cell_id": cell_ids,
            "cell_purity": purity_scores,
            "is_pure": is_pure
        })

    return purity_scores, is_pure, purity_threshold, purity_df

#
def compute_cell_conflict(
    M,
    col_idx,
    npmi_mat,
    cell_ids=None,
    conflict_percentile=80.0,   # top X% most conflicting
    conflict_threshold=None     # optional explicit threshold for conflict_score
):
    """
    Conflict score = normalized weighted magnitude of negative NPMI pairs.
    Higher = more contaminated / merged.
    """

    # Kernel returns `sum_neg` per row; conflict = sum_neg / total_pairs.
    # `threshold` arg below doesn't affect sum_neg, only n_pos_above.
    k_arr, _n_pos, sum_neg, _pos_relu, _neg_relu = pair_aggregate_dense(
        M, col_idx, npmi_mat, threshold=0.0, tau=0.0,
    )
    n_pairs_total = k_arr * (k_arr - 1) // 2

    conflict_scores = np.full(M.shape[0], np.nan, dtype=float)
    has_pairs = n_pairs_total > 0
    conflict_scores[has_pairs] = sum_neg[has_pairs] / n_pairs_total[has_pairs]

    valid = ~np.isnan(conflict_scores)
    if conflict_threshold is None:
        conflict_threshold = np.nanpercentile(
            conflict_scores[valid],
            conflict_percentile
        )

    is_conflict = np.zeros_like(conflict_scores, dtype=bool)
    is_conflict[valid] = conflict_scores[valid] >= conflict_threshold

    # optional DF
    if cell_ids is not None:
        conflict_df = pd.DataFrame({
            "cell_id": cell_ids,
            "conflict_score": conflict_scores,
            "is_conflict": is_conflict
        })
    else:
        conflict_df = pd.DataFrame({
            "conflict_score": conflict_scores,
            "is_conflict": is_conflict
        })

    return conflict_scores, is_conflict, conflict_threshold, conflict_df
#
def compute_purity_and_conflict(
    filtered_df,
    nucleus_npmi_long,
    adata,
    *,
    cell_col="cell_id",
    min_transcripts_per_cell=10,
    exclude_ids=None,
    npmi_threshold=0.05,
    purity_percentile=80.0,
    conflict_percentile=80.0,
):
    """
    Starting from filtered_df (already QV- and gene-filtered),
    compute:
      - cell purity score
      - cell conflict score
    and attach them to adata.obs

    Parameters
    ----------
    filtered_df : DataFrame
        Transcript-level data
    nucleus_npmi_long : DataFrame
        Pre-computed NPMI matrix in long format
    adata : AnnData
        AnnData object to attach metrics to
    cell_col : str
        Column name containing cell IDs in filtered_df
    min_transcripts_per_cell : int
        Minimum transcripts required per cell
    npmi_threshold : float
        NPMI threshold for purity calculation
    purity_percentile : float
        Percentile for purity threshold
    conflict_percentile : float
        Percentile for conflict threshold
    exclude_ids : set | None
        Set of cell IDs to exclude, e.g. {"-1", "DROP", "nan", "UNASSIGNED"}

    Returns:
        purity_df, conflict_df
    """
    # -------- Build cell × gene matrix --------
    cell_ids, genes_cell, M, col_idx = build_cell_gene_matrix(
        filtered_df,
        min_transcripts=min_transcripts_per_cell,
        genes_npm=nucleus_npmi_long,
        cell_col=cell_col,
        exclude_ids=exclude_ids,
    )

    # -------- Build NPMI matrix --------
    npmi_mat, gene_to_idx_all = build_npmi_matrix(nucleus_npmi_long)

    # -------- Purity --------
    purity_scores, is_pure, purity_thr, purity_df = compute_cell_purity(
        M=M,
        col_idx=col_idx,
        npmi_mat=npmi_mat,
        npmi_threshold=npmi_threshold,
        cell_ids=cell_ids,
        purity_percentile=purity_percentile,
    )

    print("Purity threshold used:", purity_thr)

    # -------- Conflict --------
    conflict_scores, is_conflict, conflict_thr, conflict_df = compute_cell_conflict(
        M=M,
        col_idx=col_idx,
        npmi_mat=npmi_mat,
        cell_ids=cell_ids,
        conflict_percentile=conflict_percentile,
    )

    print("Conflict threshold used:", conflict_thr)

    # -------- Attach results to adata.obs --------
    attach_metrics_to_adata(adata, purity_df, conflict_df)

    return purity_df, conflict_df

#
from ._utils import relu_symmetric  # noqa: E402 — re-exported for back-compat

#
def compute_cell_purity_relu(
    M,
    col_idx,
    npmi_mat,
    tau=0.05,                  # dead-zone threshold
    cell_ids=None,
    purity_percentile=80.0,
    purity_threshold=None,
    eps=1e-8                   # minimum signal for normalization
):
    """
    ReLU-based cell purity score with relative metrics.

    Uses a symmetric ReLU on NPMI to:
      - zero out weak associations within [-tau, tau]
      - weight stronger positive/negative evidence more
      
    Computes:
      - Absolute purity: sum of positive ReLU values normalized by number of pairs
      - Relative purity: fraction of total signal that is positive
      - Relative conflict: fraction of total signal that is negative
      - Signal strength: total magnitude of non-zero ReLU values

    Parameters
    ----------
    M : np.ndarray, shape (n_cells, n_genes)
        Binary presence/absence matrix
    col_idx : np.ndarray
        Gene indices mapping to NPMI matrix columns
    npmi_mat : np.ndarray
        Full NPMI matrix
    tau : float
        Dead-zone threshold for symmetric ReLU
    cell_ids : array-like, optional
        Cell identifiers for output DataFrame
    purity_percentile : float
        Percentile for purity threshold (if threshold not provided)
    purity_threshold : float, optional
        Explicit threshold for binary purity classification
    eps : float
        Minimum signal strength for computing relative metrics

    Returns
    -------
    purity_scores : np.ndarray
        Absolute purity scores per cell
    is_pure : np.ndarray
        Boolean array indicating pure cells
    purity_threshold : float
        Threshold used for classification
    purity_df : pd.DataFrame or None
        DataFrame with all purity metrics if cell_ids provided
    """
    k_arr, _n_pos, _sum_neg, pos_relu, neg_relu = pair_aggregate_dense(
        M, col_idx, npmi_mat, threshold=0.0, tau=tau,
    )
    n_pairs_total = k_arr * (k_arr - 1) // 2
    has_pairs = n_pairs_total > 0
    total_abs = pos_relu + neg_relu

    n_cells = M.shape[0]
    purity_scores = np.full(n_cells, np.nan, dtype=float)
    signal_strength = np.full(n_cells, np.nan, dtype=float)
    relative_purity = np.full(n_cells, np.nan, dtype=float)
    relative_conflict = np.full(n_cells, np.nan, dtype=float)

    purity_scores[has_pairs] = pos_relu[has_pairs] / n_pairs_total[has_pairs]
    signal_strength[has_pairs] = total_abs[has_pairs]

    has_signal = has_pairs & (total_abs > eps)
    relative_purity[has_signal] = pos_relu[has_signal] / total_abs[has_signal]
    relative_conflict[has_signal] = neg_relu[has_signal] / total_abs[has_signal]

    valid = ~np.isnan(purity_scores)

    if purity_threshold is None:
        purity_threshold = np.nanpercentile(
            purity_scores[valid], purity_percentile
        )

    is_pure = np.zeros_like(purity_scores, dtype=bool)
    is_pure[valid] = purity_scores[valid] >= purity_threshold

    purity_df = None
    if cell_ids is not None:
        purity_df = pd.DataFrame({
            "cell_id": cell_ids,
            "cell_purity_relu": purity_scores,
            "signal_strength": signal_strength,
            "relative_purity": relative_purity,
            "relative_conflict": relative_conflict,
            "is_pure": is_pure
        })

    return purity_scores, is_pure, purity_threshold, purity_df

#
def compute_cell_conflict_relu(
    M,
    col_idx,
    npmi_mat,
    tau=0.05,
    cell_ids=None,
    conflict_percentile=80.0,
    conflict_threshold=None,
    eps=1e-8
):
    """
    ReLU-based conflict score with relative metrics.

    Measures magnitude-weighted negative evidence
    after suppressing weak NPMI values within [-tau, tau].
    
    Computes:
      - Absolute conflict: sum of negative ReLU values normalized by number of pairs
      - Relative conflict: fraction of total signal that is negative
      - Relative purity: fraction of total signal that is positive
      - Signal strength: total magnitude of non-zero ReLU values

    Parameters
    ----------
    M : np.ndarray, shape (n_cells, n_genes)
        Binary presence/absence matrix
    col_idx : np.ndarray
        Gene indices mapping to NPMI matrix columns
    npmi_mat : np.ndarray
        Full NPMI matrix
    tau : float
        Dead-zone threshold for symmetric ReLU
    cell_ids : array-like, optional
        Cell identifiers for output DataFrame
    conflict_percentile : float
        Percentile for conflict threshold (if threshold not provided)
    conflict_threshold : float, optional
        Explicit threshold for binary conflict classification
    eps : float
        Minimum signal strength for computing relative metrics

    Returns
    -------
    conflict_scores : np.ndarray
        Absolute conflict scores per cell
    is_conflict : np.ndarray
        Boolean array indicating high-conflict cells
    conflict_threshold : float
        Threshold used for classification
    conflict_df : pd.DataFrame or None
        DataFrame with all conflict metrics if cell_ids provided
    """
    k_arr, _n_pos, _sum_neg, pos_relu, neg_relu = pair_aggregate_dense(
        M, col_idx, npmi_mat, threshold=0.0, tau=tau,
    )
    n_pairs_total = k_arr * (k_arr - 1) // 2
    has_pairs = n_pairs_total > 0
    total_abs = pos_relu + neg_relu

    n_cells = M.shape[0]
    conflict_scores = np.full(n_cells, np.nan, dtype=float)
    signal_strength = np.full(n_cells, np.nan, dtype=float)
    relative_purity = np.full(n_cells, np.nan, dtype=float)
    relative_conflict = np.full(n_cells, np.nan, dtype=float)

    conflict_scores[has_pairs] = neg_relu[has_pairs] / n_pairs_total[has_pairs]
    signal_strength[has_pairs] = total_abs[has_pairs]

    has_signal = has_pairs & (total_abs > eps)
    relative_purity[has_signal] = pos_relu[has_signal] / total_abs[has_signal]
    relative_conflict[has_signal] = neg_relu[has_signal] / total_abs[has_signal]

    valid = ~np.isnan(conflict_scores)

    if conflict_threshold is None:
        conflict_threshold = np.nanpercentile(
            conflict_scores[valid], conflict_percentile
        )

    is_conflict = np.zeros_like(conflict_scores, dtype=bool)
    is_conflict[valid] = conflict_scores[valid] >= conflict_threshold

    if cell_ids is not None:
        conflict_df = pd.DataFrame({
            "cell_id": cell_ids,
            "cell_conflict_relu": conflict_scores,
            "signal_strength": signal_strength,
            "relative_purity": relative_purity,
            "relative_conflict": relative_conflict,
            "is_conflict": is_conflict
        })
    else:
        conflict_df = pd.DataFrame({
            "cell_conflict_relu": conflict_scores,
            "signal_strength": signal_strength,
            "relative_purity": relative_purity,
            "relative_conflict": relative_conflict,
            "is_conflict": is_conflict
        })

    return conflict_scores, is_conflict, conflict_threshold, conflict_df

#
def attach_metrics_to_adata_relu(adata, purity_df, conflict_df):
    """
    Attach ReLU-based NPMI metrics to AnnData object.
    
    This function adds the following columns to adata.obs:
        - cell_purity_relu: absolute purity score
        - relative_purity: fraction of signal that is positive
        - relative_conflict: fraction of signal that is negative  
        - signal_strength: total magnitude of non-zero ReLU values
        - is_pure: boolean flag for pure cells
        - cell_conflict_relu: absolute conflict score
        - is_conflict: boolean flag for high-conflict cells
        
    Parameters
    ----------
    adata : AnnData
        The AnnData object to update
    purity_df : pd.DataFrame
        DataFrame with purity metrics from compute_cell_purity_relu
    conflict_df : pd.DataFrame
        DataFrame with conflict metrics from compute_cell_conflict_relu
        
    Returns
    -------
    None
        Modifies adata.obs in place
    """
    # Map purity metrics
    purity_map = dict(zip(purity_df["cell_id"], purity_df["cell_purity_relu"]))
    rel_purity_map = dict(zip(purity_df["cell_id"], purity_df["relative_purity"]))
    signal_map_p = dict(zip(purity_df["cell_id"], purity_df["signal_strength"]))
    purity_bool_map = dict(zip(purity_df["cell_id"], purity_df["is_pure"]))
    
    # Map conflict metrics
    conflict_map = dict(zip(conflict_df["cell_id"], conflict_df["cell_conflict_relu"]))
    rel_conflict_map = dict(zip(conflict_df["cell_id"], conflict_df["relative_conflict"]))
    conflict_bool_map = dict(zip(conflict_df["cell_id"], conflict_df["is_conflict"]))
    
    # Attach to adata.obs
    adata.obs["cell_purity_relu"] = adata.obs_names.map(purity_map)
    adata.obs["relative_purity"] = adata.obs_names.map(rel_purity_map)
    adata.obs["relative_conflict"] = adata.obs_names.map(rel_conflict_map)
    adata.obs["signal_strength"] = adata.obs_names.map(signal_map_p)
    adata.obs["is_pure_relu"] = adata.obs_names.map(purity_bool_map)
    adata.obs["cell_conflict_relu"] = adata.obs_names.map(conflict_map)
    adata.obs["is_conflict_relu"] = adata.obs_names.map(conflict_bool_map)

#
def compute_purity_and_conflict_relu(
    filtered_df,
    nucleus_npmi_long,
    adata,
    *,
    cell_col="cell_id",
    min_transcripts_per_cell=10,
    exclude_ids=None,
    tau=0.05,
    purity_percentile=80.0,
    conflict_percentile=80.0,
    eps=1e-8
):
    """
    Compute ReLU-based cell purity and conflict scores and attach to adata.
    
    This function uses a symmetric ReLU transformation to:
      - Suppress weak NPMI associations (within [-tau, tau])
      - Weight stronger positive and negative evidence more heavily
      - Compute both absolute and relative metrics
    
    The following metrics are computed and attached to adata.obs:
      - cell_purity_relu: absolute purity (positive evidence / total pairs)
      - cell_conflict_relu: absolute conflict (negative evidence / total pairs)
      - relative_purity: positive signal / total signal
      - relative_conflict: negative signal / total signal
      - signal_strength: total magnitude of non-zero ReLU values
      - is_pure_relu: boolean flag for pure cells
      - is_conflict_relu: boolean flag for high-conflict cells

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Transcript-level data (already QV- and gene-filtered)
    nucleus_npmi_long : pd.DataFrame
        Pre-computed NPMI matrix in long format
    adata : AnnData
        AnnData object to attach metrics to
    cell_col : str
        Column name containing cell IDs in filtered_df
    min_transcripts_per_cell : int
        Minimum transcripts required per cell
    exclude_ids : set | None
        Set of cell IDs to exclude (e.g., {"UNASSIGNED", "DROP"})
    tau : float
        Dead-zone threshold for symmetric ReLU
    purity_percentile : float
        Percentile for purity threshold
    conflict_percentile : float
        Percentile for conflict threshold
    eps : float
        Minimum signal strength for computing relative metrics

    Returns
    -------
    purity_df : pd.DataFrame
        DataFrame with purity metrics per cell
    conflict_df : pd.DataFrame
        DataFrame with conflict metrics per cell
    """
    # -------- Build cell × gene matrix --------
    cell_ids, genes_cell, M, col_idx = build_cell_gene_matrix(
        filtered_df,
        min_transcripts=min_transcripts_per_cell,
        genes_npm=nucleus_npmi_long,
        cell_col=cell_col,
        exclude_ids=exclude_ids,
    )

    # -------- Build NPMI matrix --------
    npmi_mat, gene_to_idx_all = build_npmi_matrix(nucleus_npmi_long)

    # -------- ReLU-based Purity --------
    purity_scores, is_pure, purity_thr, purity_df = compute_cell_purity_relu(
        M=M,
        col_idx=col_idx,
        npmi_mat=npmi_mat,
        tau=tau,
        cell_ids=cell_ids,
        purity_percentile=purity_percentile,
        eps=eps,
    )

    print("ReLU Purity threshold used:", purity_thr)

    # -------- ReLU-based Conflict --------
    conflict_scores, is_conflict, conflict_thr, conflict_df = compute_cell_conflict_relu(
        M=M,
        col_idx=col_idx,
        npmi_mat=npmi_mat,
        tau=tau,
        cell_ids=cell_ids,
        conflict_percentile=conflict_percentile,
        eps=eps,
    )

    print("ReLU Conflict threshold used:", conflict_thr)

    # -------- Attach results to adata.obs --------
    attach_metrics_to_adata_relu(adata, purity_df, conflict_df)

    return purity_df, conflict_df
