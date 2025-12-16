#----------------------·•●  🧽  ●•·-------------------------
#                   HotNerd Metrics Module
#----------------------·•●──────●•·-------------------------
# Author: Long Yuan
# Affiliation: Johns Hopkins University
# Email: lyuan13@jhmi.edu
#-----------------------------------------------------------

import numpy as np
import pandas as pd
import geopandas as gpd

#
def get_confident_nuclei_transcripts(
    sdata,
    *,
    qv_min: float = 30,
    low_pct: float = 20,
    high_pct: float = 80,
    save_qv_filtered: bool = False,
    parquet_path: str = "qv_filtered_transcripts.parquet",
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
    nuc_df = df[
        (df["cell_id"] != "UNASSIGNED") &
        (df["overlaps_nucleus"] == 1)
    ].copy()

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
    min_occurrences_per_context=2
):
    """
    Compute PMI/NPMI using presence/absence of genes at the cell or nucleus level,
    with robustness control by requiring each gene to occur at least N times
    within a context (cell or nucleus) before being considered "present".

    Returns
    -------
    long_df : DataFrame
        Columns:
            gene_i, gene_j, P_i, P_j, P_ij,
            P_i_given_j, P_j_given_i, PMI, NPMI
    """

    # 0. Subset to necessary columns
    df = df_subset[[group_key, "feature_name"]].copy()

    # ----------------------------------------------------------------------
    # Filter by minimum occurrences per context
    # ----------------------------------------------------------------------
    # Count gene occurrences within each cell/nucleus
    counts = (
        df.groupby([group_key, "feature_name"])
        .size()
        .rename("gene_count")
        .reset_index()
    )

    # Keep only those gene occurrences with enough counts
    df_filtered = counts[counts["gene_count"] >= min_occurrences_per_context].copy()

    if df_filtered.empty:
        raise ValueError(
            f"No genes pass min_occurrences_per_context={min_occurrences_per_context}."
        )

    # For presence/absence, set value = 1 for all retained (context, gene) pairs
    df_filtered["value"] = 1

    # ----------------------------------------------------------------------
    # Pivot to contexts × genes matrix (presence/absence)
    # ----------------------------------------------------------------------
    M = df_filtered.pivot_table(
        index=group_key,
        columns="feature_name",
        values="value",
        aggfunc="max",
        fill_value=0
    )

    contexts = M.index.to_numpy()
    genes = M.columns.to_numpy()
    C = M.shape[0]

    # ----------------------------------------------------------------------
    # Probabilities P(i), P(i,j)
    # ----------------------------------------------------------------------
    counts_i = M.sum(axis=0).to_numpy()
    P_i = counts_i / C

    co_matrix = (M.T @ M).to_numpy()
    P_ij = co_matrix / C

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
def build_cell_gene_matrix(filtered_df, min_transcripts=10, genes_npm=None):
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
        "cell_id" and "feature_name"

    min_transcripts : int, optional (default=10)
        Minimum number of transcripts required for a cell to be retained.

    genes_npm : pandas.DataFrame
        The long-format NPMI table containing columns "gene_i", "gene_j", "NPMI".

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
    
    # Remove unassigned cells
    df = filtered_df[filtered_df["cell_id"] != "UNASSIGNED"].copy()

    # Filter by minimum transcript count per cell
    cell_counts = df.groupby("cell_id").size()
    good_ids = cell_counts[cell_counts >= min_transcripts].index
    df = df[df["cell_id"].isin(good_ids)].copy()

    df["value"] = 1

    # Pivot to cell × gene
    cell_gene = df.pivot_table(
        index="cell_id",
        columns="feature_name",
        values="value",
        aggfunc=lambda x: 1,
        fill_value=0,
    )

    cell_ids = cell_gene.index.to_numpy()
    genes_cell = cell_gene.columns.to_numpy()
    M = cell_gene.to_numpy().astype(np.int8)

    # Restrict to genes that appear in NPMI dataset
    all_genes = np.union1d(
        genes_npm["gene_i"].unique(),
        genes_npm["gene_j"].unique()
    )
    gene_to_idx_all = {g: i for i, g in enumerate(all_genes)}

    mask = np.array([g in gene_to_idx_all for g in genes_cell])
    M = M[:, mask]
    genes_cell = genes_cell[mask]
    col_idx = np.array([gene_to_idx_all[g] for g in genes_cell], dtype=np.int32)

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
    gene_to_idx = {g:i for i,g in enumerate(genes)}
    G = len(genes)

    npmi_mat = np.zeros((G, G), dtype=float)

    for row in nucleus_npmi_long.itertuples(index=False):
        i = gene_to_idx[row.gene_i]
        j = gene_to_idx[row.gene_j]
        npmi_mat[i,j] = row.NPMI
        npmi_mat[j,i] = row.NPMI

    return npmi_mat, gene_to_idx

#
def attach_metrics_to_adata(adata, purity_df, conflict_df):
    """
    Attach NPMI-derived cell quality metrics (purity and conflict) to an AnnData object.

    This function takes an AnnData object and two DataFrames containing per-cell
    purity and conflict metrics derived from NPMI analysis. It then maps these 
    scores onto `adata.obs` using each cell's unique `cell_id`. Four new columns 
    are added to the AnnData object:

        - `cell_purity`        : continuous purity score (float)
        - `cell_purity_bool`   : boolean flag indicating whether the cell meets 
                                 the "pure" criterion based on purity threshold
        - `conflict_score`     : continuous conflict score (float)
        - `is_conflict`        : boolean flag indicating whether the cell meets 
                                 the "high-conflict" criterion based on conflict threshold

    Parameters
    ----------
    adata : AnnData
        The AnnData object whose `.obs` dataframe will be updated. It must contain 
        a column `cell_id` uniquely identifying each cell.

    purity_df : pandas.DataFrame

    conflict_df : pandas.DataFrame

    Returns
    -------
    None
        The function modifies `adata` in place by adding the new columns to 
        `adata.obs`. Nothing is explicitly returned.
    """
    # Purity
    adata.obs["cell_purity"] = adata.obs["cell_id"].map(
        dict(zip(purity_df["cell_id"], purity_df["cell_purity"]))
    )
    adata.obs["cell_purity_bool"] = adata.obs["cell_id"].map(
        dict(zip(purity_df["cell_id"], purity_df["is_pure"]))
    )

    # Conflict
    adata.obs["conflict_score"] = adata.obs["cell_id"].map(
        dict(zip(conflict_df["cell_id"], conflict_df["conflict_score"]))
    )
    adata.obs["is_conflict"] = adata.obs["cell_id"].map(
        dict(zip(conflict_df["cell_id"], conflict_df["is_conflict"]))
    )

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

    n_cells = M.shape[0]
    purity_scores = np.full(n_cells, np.nan, dtype=float)

    for row in range(n_cells):
        present_idx = np.where(M[row] == 1)[0]
        gi = col_idx[present_idx]
        k = len(gi)
        if k < 2:
            continue  # leave as NaN

        sub = npmi_mat[np.ix_(gi, gi)]
        vals = sub[np.triu_indices_from(sub, k=1)]

        # purity fraction: proportion of NPMI > npmi_threshold
        purity = np.mean(vals > npmi_threshold)
        purity_scores[row] = purity

    # determine threshold for boolean purity
    valid = ~np.isnan(purity_scores)
    if purity_threshold is None:
        # use percentile of *valid* scores
        purity_threshold = np.nanpercentile(purity_scores[valid], purity_percentile)

    is_pure = np.zeros_like(purity_scores, dtype=bool)
    is_pure[valid] = purity_scores[valid] >= purity_threshold

    # package as DataFrame if cell_ids provided
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

    n_cells = M.shape[0]
    conflict_scores = np.full(n_cells, np.nan, dtype=float)

    for row in range(n_cells):
        present_idx = np.where(M[row] == 1)[0]
        gi = col_idx[present_idx]
        k = len(gi)

        if k < 2:
            continue  # leave as NaN

        sub = npmi_mat[np.ix_(gi, gi)]
        vals = sub[np.triu_indices_from(sub, k=1)]

        # contributions from negative NPMI only (flip sign → positive penalty)
        neg_contribs = -vals[vals < 0]

        K = k * (k - 1) / 2  # total number of pairs
        if K > 0:
            conflict_scores[row] = neg_contribs.sum() / K

    # determine conflict threshold
    valid = ~np.isnan(conflict_scores)
    if conflict_threshold is None:
        conflict_threshold = np.nanpercentile(
            conflict_scores[valid],
            conflict_percentile
        )

    # high conflict = score >= threshold
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
    min_transcripts_per_cell=10,
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

    Returns:
        purity_df, conflict_df
    """
    # -------- Build cell × gene matrix --------
    cell_ids, genes_cell, M, col_idx = build_cell_gene_matrix(
        filtered_df,
        min_transcripts=min_transcripts_per_cell,
        genes_npm=nucleus_npmi_long,
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
