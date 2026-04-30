"""Phase 1/2: Conservative NPMI pruning.

Denoise cell and create partial cell IDs based on NPMI gene coherence (Phase 1),
then further denoise partial cells (Phase 2).
"""

import concurrent.futures  # noqa: F401 — retained for API compatibility

import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # noqa: F401 — used by prune_transcripts_fast

from . import _cy_prune
from ._repro import _ensure_reproducibility_seed
from ._utils import prepare_transcript_df


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


# NOTE: There was previously an attempt here to rebind
# `prune_genes_by_npmi_greedy` to `_cy_prune.prune_genes_by_npmi_greedy`.
# That attribute does not exist on the compiled extension — _cy_prune
# exposes `prune_cells` (batch over many cells) and `prune_single`, which
# the `_fast` entry points call directly. The rebind was silently failing
# under a try/except, so the pure-Python `prune_genes_by_npmi_greedy`
# above has always been the one running here. Keeping it as the single
# reference implementation is intentional; the hot path in production
# uses `_cy_prune.prune_cells` via `prune_transcripts_fast` etc.

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
    _ensure_reproducibility_seed()
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
        g_local = np.sort(sub["_gene_idx"].dropna().astype(int).unique())
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

    for pid in sorted(set(partial_map.values())):
        sub = df[df["cell_id_npmi_cons_p1"] == pid]
        g_local = np.sort(sub["_gene_idx"].dropna().astype(int).unique())
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
    *,
    cell_id_col: str = "cell_id",
    out_col: str = "tracer_id",
    gene_col: str = "feature_name",
    threshold: float = -0.1,
    unassigned_id: str = "-1",
    debug_stages: bool = False,
    n_jobs: int = 1,
    show_progress: bool = True,
    in_place: bool = False,
):
    """
    Two-pass NPMI pruning. Writes pass-2 result to `out_col` in place;
    pass 1's intermediate state is internal-only unless `debug_stages` is
    True. Status columns are also gated behind `debug_stages`.

    Parameters
    ----------
    out_col : str
        Canonical output column name (default `"tracer_id"`). Each
        pipeline stage writes its current best assignment here, in place,
        so the column always reflects the latest stage's output.
    debug_stages : bool
        When True, additionally writes legacy snapshot columns for
        inspection: `cell_id_npmi_cons_p1` (post pass 1), and
        `cell_id_npmi_cons_p2` (post pass 2 = mirrors `out_col`), plus
        the status columns `npmi_cons_p1_status` / `npmi_cons_p2_status`.
        Default False keeps the output minimal for production use.
    n_jobs : int
        Accepted for back-compat; no-op (Cython batch).
    in_place : bool
        Skip the defensive `df.copy()` when caller hands ownership of
        the df to this stage.
    """
    _ensure_reproducibility_seed()
    if not in_place:
        df = df.copy()

    prepare_transcript_df(df, gene_col=gene_col)

    # `_cell_str`: a string view of cell_id used for the pid partial label
    # concatenation (`f"{cid}-1"`) below. If cell_id is already
    # string/object, reuse the reference (no copy); otherwise cast once.
    if df[cell_id_col].dtype != "object":
        df["_cell_str"] = df[cell_id_col].astype(str)
    else:
        df["_cell_str"] = df[cell_id_col]

    genes, gene_to_idx, W = build_dense_npmi_matrix(npmi_df)
    df["_gene_idx"] = df[gene_col].map(gene_to_idx)

    # ---------- PASS 1 ----------
    # Initialise the canonical output column with the raw cell_id labels;
    # pass 1 will overwrite some entries with "{cid}-1" partials, and
    # pass 2 may further demote some partials to `unassigned_id` ("-1").
    # Working column = `out_col` itself (no separate intermediate).
    df[out_col] = df["_cell_str"]
    p1_status = None
    if debug_stages:
        # Pre-declare the full category vocabulary so `.loc[…, col] = "partial_p1"`
        # below doesn't trigger a "not in categories" error. Categorical
        # storage drops this column from ~75 MiB to ~1.5 MiB at 1.4M rows.
        _STATUS_P1_CATS = ["unassigned_input", "core", "partial_p1"]
        p1_status = pd.Categorical(
            np.where(df["_cell_str"] == unassigned_id, "unassigned_input", "core"),
            categories=_STATUS_P1_CATS,
        )

    partial_map = {}

    # Prepare per-cell unique gene lists (only cells that are not unassigned)
    grp = df[df["_cell_str"] != unassigned_id].groupby("_cell_str")["_gene_idx"].apply(
        lambda s: np.asarray(np.sort(pd.Index(s.dropna().astype(int)).unique()), dtype=np.int32)
    )

    cell_items = list(grp.items())
    total_cells = len(cell_items)

    # `n_jobs` is accepted for API compatibility but no longer used: the
    # per-cell pruning now runs as one C-level batch inside the compiled
    # Cython kernel (_cy_prune.prune_cells), so there is no Python
    # ThreadPoolExecutor to parallelize.
    _ = n_jobs

    results = []

    # Batch prune all cells through the compiled Cython kernel. The Python
    # fallback was removed — it was 100–1000× slower and silently ran for
    # hours when the .so wasn't built. If _cy_prune.prune_cells raises
    # (e.g. corrupted gene lists), surface it instead of papering over.
    cell_ids = [cid for cid, _ in cell_items]
    g_arrays = [gl if (gl is not None and gl.size > 0) else None for _, gl in cell_items]
    removed_lists = _cy_prune.prune_cells(g_arrays, W, float(threshold))
    for cid, removed in zip(cell_ids, removed_lists):
        if removed:
            results.append((cid, removed))

    if show_progress:
        pbar = tqdm(total=total_cells, desc="prune_pass1")
        pbar.update(total_cells)
        pbar.close()

    # Deterministic application order (stable across thread completion)
    if results:
        results.sort(key=lambda x: str(x[0]))
        for cid, _ in results:
            partial_map[cid] = f"{cid}-1"

    # Apply pass1 removals — write `pid` partial labels into `out_col`
    # for matching (cell, gene) rows. Status (when debug) marked as
    # `partial_p1` for those rows.
    if results:
        rows = []
        for cid, removed in results:
            pid = partial_map[cid]
            for g in removed:
                rows.append((cid, int(g), pid))

        if rows:
            map_df = pd.DataFrame(rows, columns=["_cell_str_map", "_gene_idx_map", "_pid"])

            # prepare indexed view of original df for efficient merging
            df_idx = df.reset_index().rename(columns={"index": "_orig_index"})[["_orig_index", "_cell_str", "_gene_idx"]]
            df_idx["_gene_idx"] = df_idx["_gene_idx"].astype("Int64")
            map_df["_gene_idx_map"] = map_df["_gene_idx_map"].astype("Int64")

            merged = pd.merge(
                df_idx, map_df,
                left_on=["_cell_str", "_gene_idx"],
                right_on=["_cell_str_map", "_gene_idx_map"],
                how="inner",
            )

            if not merged.empty:
                df.loc[merged["_orig_index"], out_col] = merged["_pid"].values
                if debug_stages and p1_status is not None:
                    p1_status[merged["_orig_index"].to_numpy()] = "partial_p1"

    # Snapshot pass-1 state if requested. Has to happen *before* pass 2
    # mutates `out_col`.
    if debug_stages:
        df["cell_id_npmi_cons_p1"] = df[out_col].copy()
        df["npmi_cons_p1_status"] = p1_status

    if show_progress:
        tqdm(desc="apply_pass1", total=1).update(1)

    # ---------- PASS 2 ----------
    # Pass 2 reads the partial labels currently in `out_col`, prunes their
    # gene sets, and demotes failing rows to `unassigned_id` — in place
    # on `out_col`. No separate `cell_id_npmi_cons_p2` column needed
    # outside debug.
    p2_status = None
    if debug_stages:
        _STATUS_P2_CATS = ["unchanged", "partial_p2", "unassigned_from_partial"]
        p2_status = pd.Categorical(
            ["unchanged"] * len(df), categories=_STATUS_P2_CATS,
        )

    pids = sorted(set(partial_map.values()))
    if pids:
        grp_p = df[df[out_col].isin(pids)].groupby(out_col)["_gene_idx"].apply(
            lambda s: np.asarray(np.sort(pd.Index(s.dropna().astype(int)).unique()), dtype=np.int32)
        )

        partial_items = list(grp_p.items())
        total_partials = len(partial_items)

        if show_progress:
            pbar2 = tqdm(total=total_partials, desc="prune_pass2")
        else:
            pbar2 = None

        results2 = []

        pids = [pid for pid, _ in partial_items]
        g_arrays = [gl if (gl is not None and gl.size > 0) else None for _, gl in partial_items]
        removed_lists = _cy_prune.prune_cells(g_arrays, W, float(threshold))
        for pid, removed in zip(pids, removed_lists):
            if removed:
                results2.append((pid, removed))
            if pbar2 is not None:
                pbar2.update(1)
        if pbar2 is not None:
            pbar2.close()

        if results2:
            results2.sort(key=lambda x: str(x[0]))

        rows2 = []
        removed_pids = set()
        for pid, removed in results2:
            removed_pids.add(pid)
            for g in removed:
                rows2.append((pid, int(g)))

        # df index view: keys = (current partial label in out_col, _gene_idx).
        df_idx2 = df.reset_index().rename(columns={"index": "_orig_index"})[["_orig_index", out_col, "_gene_idx"]]
        df_idx2["_gene_idx"] = df_idx2["_gene_idx"].astype("Int64")

        if rows2:
            map2 = pd.DataFrame(rows2, columns=["_pid_map", "_gene_idx_map"]).astype({"_gene_idx_map": "Int64"})
            merged2 = pd.merge(
                df_idx2, map2,
                left_on=[out_col, "_gene_idx"],
                right_on=["_pid_map", "_gene_idx_map"],
                how="inner",
            )

            if not merged2.empty:
                # demote: out_col → unassigned sentinel for these rows
                df.loc[merged2["_orig_index"], out_col] = unassigned_id
                if debug_stages and p2_status is not None:
                    p2_status[merged2["_orig_index"].to_numpy()] = "unassigned_from_partial"

        # Mark "still-partial" rows in pass 2 status (debug only). A row
        # has pass-2 status = "partial_p2" iff its post-pass-1 label was
        # a partial pid AND pass 2 did not demote it. The pre-pass-2
        # label snapshot is `df["cell_id_npmi_cons_p1"]` (set above
        # when debug_stages was True).
        if debug_stages and p2_status is not None and pids:
            pids_set = set(pids)
            pre_p2 = df["cell_id_npmi_cons_p1"].astype(str)
            still_partial = pre_p2.isin(pids_set) & (df[out_col].astype(str) != unassigned_id)
            kept_partial_idx = np.where(still_partial.to_numpy())[0]
            if kept_partial_idx.size:
                p2_status[kept_partial_idx] = "partial_p2"

    if debug_stages:
        # Snapshot pass-2 state under the legacy name. Mirrors `out_col`.
        df["cell_id_npmi_cons_p2"] = df[out_col].copy()
        if p2_status is not None:
            df["npmi_cons_p2_status"] = p2_status

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
        genes = np.sort(sub["feature_name"].astype(str).unique())
        gids = np.sort(pd.Index(genes).map(gene_to_idx).dropna().astype(int).unique())
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
