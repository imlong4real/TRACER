"""Density-cascade Phase 1 — synthetic-seed cell finder.

Replaces the Group stage in NOSEG pipelines (no input cell_id) and
serves as an opt-in residual handler in SEG pipelines (cascade run on
post-Rescue '-1' tx instead of `annotate_unassigned_components_fast`).

The cascade walks descending bin-density thresholds. At each pass it:
  1. Finds bins with current pool-tx count >= threshold.
  2. Sorts hot bins by density (highest wins contests within a pass).
  3. For each hot bin, greedy-prunes its R=1 Moore neighborhood's gene
     set against the panel PMI matrix; commits tx whose gene survived.
  4. Mismatched tx return to the pool for later passes.

The threshold floor is selected adaptively per pool by
`auto_floor_from_coverage` — see its docstring for the rule.

References:
  /tmp/coverage_by_threshold.py        — auto-floor logic prototype
  /tmp/seg_cascade_6_2.py              — SEG-residual replacement bench
  /tmp/full_homogeneity_seg_vs_cascades.py  — full-tissue homogeneity bench
  density-cascade-handoff.md           — design rationale + empirical results
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

# NOTE: `build_dense_pmi_matrix_small_panel` is intentionally NOT imported at module
# level — that path is a fallback used only when callers invoke
# `density_cascade_phase1` with `panel=` but no `aux={"W": ...}` (i.e.
# standalone, outside the pipeline). The lazy import lives inside that
# else-branch (see line ~268) so cascade can load and run cleanly on
# the sparse-only path the rest of the package is migrating toward.


def _wget(W, i: int, j: int) -> float:
    """Single-element PMI lookup that works for dense ndarray and sparse CSR.

    Dense: returns ``W[i, j]`` directly (may be NaN for structurally-absent
    pairs depending on how W was built).

    Sparse CSR: random-access via the indptr/indices/data triple. Pairs
    encoded as absent return NaN to match dense-NaN semantics (matches
    the existing `_cy_prune._wget` contract used by the Cython kernels).
    """
    if isinstance(W, np.ndarray):
        return float(W[i, j])
    # CSR random access; W.indices for row i is sorted (CSR invariant).
    row_start = int(W.indptr[i])
    row_end = int(W.indptr[i + 1])
    if row_end <= row_start:
        return float("nan")
    indices = W.indices[row_start:row_end]
    pos = int(np.searchsorted(indices, j))
    if pos < (row_end - row_start) and indices[pos] == j:
        return float(W.data[row_start + pos])
    return float("nan")


# ============================================================================
# Auto-floor selection (tx-coverage rule)
# ============================================================================
def _build_grid(df: pd.DataFrame, G: float = 2.0,
                bbox: Optional[tuple] = None) -> tuple[np.ndarray, tuple]:
    """Build a 2D bin-count grid (n_y, n_x) over xy coords.

    If bbox is None, uses the data's min/max as extent.
    Returns (grid, (x_min, y_min, x_max, y_max)).
    """
    xy = df[["x", "y"]].to_numpy(dtype=np.float32)
    if bbox is None:
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
    else:
        x_min, y_min, x_max, y_max = bbox

    bx = np.floor((xy[:, 0] - x_min) / G).astype(np.int64)
    by = np.floor((xy[:, 1] - y_min) / G).astype(np.int64)
    n_x = int(np.ceil((x_max - x_min) / G)) + 1
    n_y = int(np.ceil((y_max - y_min) / G)) + 1
    grid = np.zeros((n_y, n_x), dtype=np.int32)
    np.add.at(grid, (by, bx), 1)
    return grid, (float(x_min), float(y_min), float(x_max), float(y_max))


def _moore_dilate(mask: np.ndarray, R: int = 1) -> np.ndarray:
    """R-radius Moore (square) dilation via scipy.ndimage.

    Falls back to a 4-pass shift+OR if scipy is unavailable.
    """
    try:
        from scipy.ndimage import binary_dilation
        structure = np.ones((2 * R + 1, 2 * R + 1), dtype=bool)
        return binary_dilation(mask, structure=structure)
    except ImportError:  # pragma: no cover
        out = mask.copy()
        for _ in range(R):
            shifted = out.copy()
            shifted[1:, :] |= out[:-1, :]
            shifted[:-1, :] |= out[1:, :]
            shifted[:, 1:] |= out[:, :-1]
            shifted[:, :-1] |= out[:, 1:]
            shifted[1:, 1:] |= out[:-1, :-1]
            shifted[:-1, :-1] |= out[1:, 1:]
            shifted[1:, :-1] |= out[:-1, 1:]
            shifted[:-1, 1:] |= out[1:, :-1]
            out = shifted
        return out


def auto_floor_from_coverage(
    grid: np.ndarray,
    *,
    target_cov: float = 0.65,
    R: int = 1,
    hard_min: int = 2,
    ceiling: Optional[int] = None,
) -> tuple[int, list[tuple[int, float, int]]]:
    """Pick the cascade floor adaptively from runtime tx-coverage.

    Walks thresholds [max(grid)..hard_min] and for each, computes the
    R-Moore-dilated mask of bins with count >= threshold. Returns the
    LARGEST threshold n such that:

        sum(grid[dilated_mask(n)]) / total_tx  >=  target_cov

    If no such n exists (sparse pool), returns hard_min.

    Parameters
    ----------
    grid : 2D int array
        Bin-count grid built by `_build_grid`.
    target_cov : float
        Fraction of tx mass that anchor + R-Moore neighborhoods must
        capture (default 0.65). Empirically lands on the same floors
        as hand-tuned [8..4] (Xenium), [6..2] (Xenium residual), and
        ~[~..12-13] (Visium HD 2µm).
    R : int
        Moore radius. Default 1 (3x3 territory).
    hard_min : int
        Never go below this floor; default 2 (single-tx bins are not
        anchor candidates).
    ceiling : int or None
        If set, only consider thresholds <= ceiling. Default = grid.max().

    Returns
    -------
    (floor, coverage_curve)
        floor: int, the chosen threshold.
        coverage_curve: list of (thr, tx_cov_fraction, n_anchors)
                       in descending threshold order, for diagnostics.
    """
    total_tx = int(grid.sum())
    if total_tx == 0:
        return hard_min, []

    max_thr = int(ceiling if ceiling is not None else grid.max())
    if max_thr < hard_min:
        return hard_min, []

    coverage_curve: list[tuple[int, float, int]] = []
    floor = hard_min

    for thr in range(max_thr, hard_min - 1, -1):
        anchor_mask = grid >= thr
        n_anchors = int(anchor_mask.sum())
        if n_anchors == 0:
            coverage_curve.append((thr, 0.0, 0))
            continue
        dilated = _moore_dilate(anchor_mask, R=R)
        tx_cov = float(grid[dilated].sum()) / total_tx
        coverage_curve.append((thr, tx_cov, n_anchors))
        if tx_cov >= target_cov:
            # Largest n satisfying the target is the FIRST one we hit
            # walking down (since coverage is monotone increasing in
            # decreasing threshold). Stop here.
            floor = thr
            break
    else:
        # No threshold satisfied target_cov; fall back to hard_min.
        floor = hard_min

    return floor, coverage_curve


def auto_thresholds(
    df: pd.DataFrame,
    *,
    G: float = 2.0,
    target_cov: float = 0.65,
    R: int = 1,
    hard_min: int = 2,
    ceiling: Optional[int] = None,
) -> list[int]:
    """Convenience: build grid + auto-floor + return descending threshold list.

    The cascade walks thresholds [max..floor], inclusive, descending.
    """
    grid, _ = _build_grid(df, G=G)
    if ceiling is None:
        ceiling = int(grid.max())
    floor, _ = auto_floor_from_coverage(
        grid, target_cov=target_cov, R=R, hard_min=hard_min, ceiling=ceiling,
    )
    return list(range(ceiling, floor - 1, -1))


# ============================================================================
# Phase 1a-style greedy prune
# ============================================================================
def greedy_prune(unique_gene_idx: list[int], W,
                  threshold: float = 0.05) -> list[int]:
    """Iteratively remove the gene with most bad edges (PMI < threshold).

    Accepts ``W`` as either a dense ``np.ndarray`` or a sparse CSR matrix —
    per-pair lookups go through :func:`_wget`, which has the same NaN
    semantics on both backends (structurally-absent CSR pairs return NaN
    and are treated as "unknown" / not-bad, matching dense-NaN behavior).
    """
    g = list(unique_gene_idx)
    while len(g) >= 2:
        bad_count = [0] * len(g)
        for i in range(len(g)):
            gi = g[i]
            for j in range(len(g)):
                if i == j:
                    continue
                v = _wget(W, gi, g[j])
                if v == v and v < threshold:  # NaN-safe
                    bad_count[i] += 1
        worst_idx = int(np.argmax(bad_count))
        if bad_count[worst_idx] == 0:
            break
        g.pop(worst_idx)
    return g


# ============================================================================
# Density-cascade Phase 1
# ============================================================================
def density_cascade_phase1(
    df: pd.DataFrame,
    panel: Optional[pd.DataFrame] = None,
    *,
    aux: Optional[dict] = None,
    G: float = 2.0,
    thresholds: Sequence[int] | str = "auto",
    territory_radius_bins: int = 1,
    pmi_threshold: float = 0.05,
    min_anchor_tx: int = 3,
    auto_target_cov: float = 0.65,
    auto_hard_min: int = 2,
    auto_ceiling: Optional[int] = None,
) -> dict:
    """Run density-cascade Phase 1 on `df`.

    Parameters
    ----------
    df : DataFrame
        Must have columns x, y, feature_name. May have a `tracer_id`
        col if running on residual; the cascade ignores that — it
        operates on every row.
    panel : DataFrame or None
        Long-form panel with gene_i, gene_j, PMI columns. Used to build
        the dense PMI matrix W if `aux` doesn't provide one.
    aux : dict or None
        If provided and contains 'gene_to_idx' + 'W', skips PMI matrix
        construction. Pass the aux dict from `prune_transcripts_*`.
    thresholds : list of int or "auto"
        Descending bin-density thresholds for the cascade. Use "auto"
        to derive from `auto_floor_from_coverage` at runtime.
    territory_radius_bins : int
        Moore radius around each anchor bin (R=1 = 3x3, R=2 = 5x5).
        Default R=1 per design.
    pmi_threshold : float
        PMI cutoff for greedy prune.
    min_anchor_tx : int
        Min committed tx to accept an anchor.
    auto_* : forwarded to auto_floor_from_coverage when thresholds="auto".

    Returns
    -------
    dict with keys:
        anchors : list of dicts (centroid, threshold, n_tx, ...)
        tx_to_anchor : dict[int, int] mapping df row index -> anchor index
        n_anchors, n_tx_assigned, n_tx_unassigned
        thresholds_used : list of int
        stats_per_pass : list of per-threshold stats
        coverage_curve : list (when thresholds="auto") for diagnostics
    """
    # Resolve W matrix
    if aux is not None and "W" in aux and "gene_to_idx" in aux:
        gene_to_idx = aux["gene_to_idx"]
        W = aux["W"]
        if isinstance(W, np.ndarray):
            W = W.astype(np.float32)
        # Sparse CSR: keep as-is; _wget handles random access without densifying.
    else:
        if panel is None:
            raise ValueError(
                "density_cascade_phase1 needs either panel= or aux= with W"
            )
        # Standalone fallback (no pipeline aux). The pipeline path is
        # sparse end-to-end via the prune stage; this fallback only
        # fires when a caller invokes density_cascade_phase1 directly
        # with a long-DataFrame panel. Use the sparse builder so even
        # the standalone path doesn't materialize a (G,G) dense matrix.
        from tracer.pruning import build_sparse_pmi_matrix_from_long
        _, gene_to_idx, W = build_sparse_pmi_matrix_from_long(
            panel, metric_col="PMI")
    # NaN→0 fill only for dense W (sparse CSR has no NaNs by construction —
    # structurally-absent pairs are skipped by _wget).
    if isinstance(W, np.ndarray):
        np.nan_to_num(W, copy=False, nan=0.0)

    coords = df[["x", "y"]].to_numpy(dtype=np.float32)
    gene_strs = df["feature_name"].to_numpy()
    n_tx = len(df)

    # Map each tx to a gene-vocab index (-1 if not in vocab)
    gene_idx = np.array(
        [gene_to_idx.get(g, -1) for g in gene_strs], dtype=np.int64,
    )
    valid_mask = gene_idx >= 0

    # Resolve thresholds (auto if requested)
    coverage_curve_diag: list[tuple[int, float, int]] = []
    if isinstance(thresholds, str) and thresholds == "auto":
        if not valid_mask.any():
            thresholds_list: list[int] = []
        else:
            df_valid = df.loc[valid_mask]
            grid_v, _ = _build_grid(df_valid, G=G)
            ceiling = (auto_ceiling if auto_ceiling is not None
                          else int(grid_v.max()))
            floor, coverage_curve_diag = auto_floor_from_coverage(
                grid_v, target_cov=auto_target_cov,
                R=territory_radius_bins, hard_min=auto_hard_min,
                ceiling=ceiling,
            )
            thresholds_list = list(range(ceiling, floor - 1, -1))
    else:
        thresholds_list = list(thresholds)

    # Bin in xy
    bin_x = np.floor(coords[:, 0] / G).astype(np.int64)
    bin_y = np.floor(coords[:, 1] / G).astype(np.int64)
    bin_keys = list(zip(bin_x.tolist(), bin_y.tolist()))

    bin_to_tx: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, k in enumerate(bin_keys):
        if valid_mask[i]:
            bin_to_tx[k].append(i)

    # Pool = currently uncommitted, valid-gene tx
    pool: set[int] = set(np.where(valid_mask)[0].tolist())
    bin_pool_count: Counter = Counter()
    for i in pool:
        bin_pool_count[bin_keys[i]] += 1

    anchors: list[dict] = []
    tx_to_anchor: dict[int, int] = {}

    R = territory_radius_bins
    territory_offsets = [(dx, dy) for dx in range(-R, R + 1)
                                  for dy in range(-R, R + 1)]

    stats_per_pass = []

    for t in thresholds_list:
        hot = [(b, bin_pool_count[b]) for b in bin_pool_count
               if bin_pool_count[b] >= t]
        # Sort by density desc — higher density wins contests
        hot.sort(key=lambda x: -x[1])

        n_seeded_pass = 0
        n_committed_pass = 0
        n_returned_pass = 0

        for hot_bin, _density_at_check in hot:
            if bin_pool_count[hot_bin] < t:
                continue

            density_at_seed = bin_pool_count[hot_bin]
            territory = [(hot_bin[0] + dx, hot_bin[1] + dy)
                         for dx, dy in territory_offsets]
            tentative = []
            for b in territory:
                for i in bin_to_tx.get(b, []):
                    if i in pool:
                        tentative.append(i)

            if len(tentative) < min_anchor_tx:
                continue

            ten_genes = gene_idx[tentative]
            unique = np.unique(ten_genes).tolist()
            seed_genes = greedy_prune(unique, W, threshold=pmi_threshold)

            if len(seed_genes) < 2:
                continue
            seed_set = set(seed_genes)

            committed = [i for i in tentative if int(gene_idx[i]) in seed_set]
            returned = [i for i in tentative if int(gene_idx[i]) not in seed_set]

            if len(committed) < min_anchor_tx:
                continue

            cmt_xy = coords[committed]
            centroid = (float(cmt_xy[:, 0].mean()), float(cmt_xy[:, 1].mean()))

            anchor_idx = len(anchors)
            anchors.append({
                "anchor_idx": anchor_idx,
                "hot_bin": hot_bin,
                "centroid": centroid,
                "threshold": t,
                "density_at_seed": density_at_seed,
                "n_tx": len(committed),
                "n_genes": len(seed_genes),
                "seed_genes_idx": seed_genes,
            })
            for i in committed:
                pool.discard(i)
                bin_pool_count[bin_keys[i]] -= 1
                tx_to_anchor[i] = anchor_idx

            n_seeded_pass += 1
            n_committed_pass += len(committed)
            n_returned_pass += len(returned)

        stats_per_pass.append({
            "threshold": t,
            "n_anchors_seeded": n_seeded_pass,
            "n_tx_committed": n_committed_pass,
            "n_tx_returned_to_pool": n_returned_pass,
            "n_pool_remaining": len(pool),
        })

    return {
        "anchors": anchors,
        "tx_to_anchor": tx_to_anchor,
        "n_anchors": len(anchors),
        "n_tx_total": n_tx,
        "n_tx_valid": int(valid_mask.sum()),
        "n_tx_assigned": len(tx_to_anchor),
        "n_tx_unassigned": int(valid_mask.sum()) - len(tx_to_anchor),
        "thresholds_used": thresholds_list,
        "stats_per_pass": stats_per_pass,
        "gene_to_idx": gene_to_idx,
        "coverage_curve": coverage_curve_diag,
    }


# ============================================================================
# Drop-in residual handler (Group replacement for SEG path)
# ============================================================================
def cascade_as_residual_handler(
    df_pruned: pd.DataFrame,
    aux: dict,
    *,
    panel: Optional[pd.DataFrame] = None,
    entity_col: str = "tracer_id",
    G: float = 2.0,
    thresholds: Sequence[int] | str = "auto",
    territory_radius_bins: int = 1,
    pmi_threshold: float = 0.05,
    min_anchor_tx: int = 3,
    auto_target_cov: float = 0.65,
    auto_hard_min: int = 2,
    auto_ceiling: Optional[int] = None,
    label_prefix: str = "cascade_",
) -> pd.DataFrame:
    """Drop-in replacement for `annotate_unassigned_components_fast`.

    Operates only on tx with `entity_col == "-1"` (post-Rescue residual).
    Runs density-cascade on this subset and writes labels of the form
    `{label_prefix}{n}-1` (depth-1 partial) back to `entity_col` for
    committed tx. Tx not anchored remain "-1".

    The `-1` suffix puts cascade output in the partial namespace
    (``_etype = "partial"``), and downstream Stitch's two-dash partial-merger
    can glue fragments of the same biological cell into one entity (output
    e.g. `cascade_5-1-1`). On full Xenium tissue this merges ~10 % of
    cascade fragments and gains +0.003 completeness with stable homogeneity.

    Returns a copy of df_pruned with `entity_col` updated.
    """
    df_out = df_pruned.copy()
    df_out[entity_col] = df_out[entity_col].astype(str)

    is_residual = df_out[entity_col] == "-1"
    if not is_residual.any():
        return df_out

    df_res = df_out.loc[is_residual].reset_index(drop=False).rename(
        columns={"index": "_orig_idx"}
    )

    casc = density_cascade_phase1(
        df_res, panel=panel, aux=aux,
        G=G, thresholds=thresholds,
        territory_radius_bins=territory_radius_bins,
        pmi_threshold=pmi_threshold,
        min_anchor_tx=min_anchor_tx,
        auto_target_cov=auto_target_cov,
        auto_hard_min=auto_hard_min,
        auto_ceiling=auto_ceiling,
    )

    new_labels = df_out[entity_col].to_numpy(dtype=object).copy()
    relabeled_orig_indices: list[int] = []
    for res_idx, anchor_idx in casc["tx_to_anchor"].items():
        orig_idx = df_res["_orig_idx"].iloc[res_idx]
        new_labels[orig_idx] = f"{label_prefix}{anchor_idx}-1"
        relabeled_orig_indices.append(int(orig_idx))
    df_out[entity_col] = new_labels

    # Mirror the cascade label-emission in `_etype`. Cascade entities
    # are emitted as `cascade_<n>-1` and classified as ``partial`` —
    # downstream rerank/reassign machinery treats them symmetrically
    # with Phase-1c partials.
    if "_etype" in df_out.columns and relabeled_orig_indices:
        mask = np.zeros(len(df_out), dtype=bool)
        mask[relabeled_orig_indices] = True
        df_out.loc[mask, "_etype"] = "partial"
    return df_out
