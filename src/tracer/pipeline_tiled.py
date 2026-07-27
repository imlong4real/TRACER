"""Tile-parallel orchestrator for the SEG pipeline.

Partitions a large transcript frame into NxM spatially-disjoint tiles
by cell_id centroid, then runs the full SEG pipeline on each tile in
a separate process. Concatenates results.

Cell-centroid assignment guarantees that every transcript of a given
cell_id ends up in the same tile. Since Stitch/Rescue's spatial reach
is small (~5-10 um) relative to a typical tile (~1-2 mm on a side),
the loss from skipping cross-tile resolution is small: only entities
within ~10 um of a tile edge can fail to stitch with a neighbor in an
adjacent tile.

This module deliberately keeps the boundary-resolution pass as a
documented gap rather than implementing it inline. Callers that need
perfect boundary handling should either (a) use larger tiles with a
margin pass, or (b) post-process the concatenated output with a
boundary-only Stitch.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd


# Canonical unassigned cell_id tokens. Transcripts carrying these ids share no
# real cell, so they must be tiled by their own xy rather than collapsed onto
# the single centroid tile of the "-1" pseudo-cell (which otherwise dumps the
# entire residual/Rescue workload into one tile).
_UNASSIGNED_TOKENS = frozenset({"-1", "DROP", "UNASSIGNED", "nan"})


def _bin_coords_to_tiles(x, y, x_edges, y_edges, n_x, n_y):
    """Vectorized (x, y) -> flat row-major tile index, matching the binning in
    ``_assign_cells_to_tiles`` (searchsorted on the same edges, clipped)."""
    xb = np.clip(np.searchsorted(x_edges, np.asarray(x, float), side="right") - 1, 0, n_x - 1)
    yb = np.clip(np.searchsorted(y_edges, np.asarray(y, float), side="right") - 1, 0, n_y - 1)
    return (xb * n_y + yb).astype(np.int32)


def _tile_edges_from_info(tile_info, n_x, n_y):
    """Reconstruct (x_edges, y_edges) from the per-tile bbox dict produced by
    ``_assign_cells_to_tiles`` so unassigned tx bin onto the identical grid."""
    x_edges = np.array([tile_info[ix * n_y]["x_min"] for ix in range(n_x)]
                       + [tile_info[(n_x - 1) * n_y]["x_max"]], float)
    y_edges = np.array([tile_info[iy]["y_min"] for iy in range(n_y)]
                       + [tile_info[n_y - 1]["y_max"]], float)
    return x_edges, y_edges


# ---------------------------------------------------------------------------
# Worker function (must be module-level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------

def _tile_worker(args: dict) -> dict:
    """Run run_segmented_pipeline on a single tile.

    args dict (passed as a single dict so ProcessPoolExecutor.submit
    can serialize cleanly):
      - tile_idx: identifying integer
      - df:       pd.DataFrame (cell-complete tile slice)
      - panel:    pd.DataFrame (PMI panel; shared across tiles)
      - rerank:   bool (override PHASE1_RERANK_ENABLED)
      - reassign: bool (override PHASE1_REASSIGN_AFTER_1C)

    Returns dict with `tile_idx`, `df_out`, `progression`, `wall_seconds`.
    """
    import os as _os
    import sys as _sys
    import resource as _res
    import time as _t  # repeat-import for worker process
    import tracer.pipeline as runner
    from tracer.pipeline import run_segmented_pipeline

    tile_idx = int(args["tile_idx"])
    df = args["df"]
    panel = args["panel"]
    rerank = bool(args.get("rerank", True))
    reassign = bool(args.get("reassign", True))

    # Tag stage-verbose output with the tile index so workers' interleaved
    # output is parseable.
    if _os.environ.get("TRACER_STAGE_VERBOSE"):
        _os.environ["TRACER_STAGE_TAG"] = f"tile{tile_idx}"

    orig_rerank = runner.PHASE1_RERANK_ENABLED
    orig_reassign = runner.PHASE1_REASSIGN_AFTER_1C
    try:
        runner.PHASE1_RERANK_ENABLED = rerank
        runner.PHASE1_REASSIGN_AFTER_1C = reassign
        t0 = _t.time()
        df_out, prog = run_segmented_pipeline(df, panel)
        wall = _t.time() - t0
    finally:
        runner.PHASE1_RERANK_ENABLED = orig_rerank
        runner.PHASE1_REASSIGN_AFTER_1C = orig_reassign

    # Peak resident set across this worker process's lifetime.
    # macOS: ru_maxrss is in bytes. Linux: ru_maxrss is in KB.
    rusage = _res.getrusage(_res.RUSAGE_SELF)
    peak_rss = int(rusage.ru_maxrss)
    if _sys.platform != "darwin":
        peak_rss *= 1024  # KB → bytes

    return {
        "tile_idx": tile_idx,
        "df_out": df_out,
        "progression": prog,
        "wall_seconds": round(wall, 2),
        "n_input_tx": int(len(df)),
        "n_input_cell_ids": int(df["cell_id"].nunique()),
        "peak_rss_bytes": peak_rss,
    }


# ---------------------------------------------------------------------------
# Tiler
# ---------------------------------------------------------------------------

def _assign_cells_to_tiles(
    df: pd.DataFrame,
    *,
    n_tiles_xy: tuple[int, int],
    cell_id_col: str = "cell_id",
    coord_cols: tuple[str, str] = ("x", "y"),
) -> tuple[pd.Series, dict]:
    """Assign each cell_id to a tile by its centroid.

    Returns (cell_to_tile, tile_info) where:
      - cell_to_tile: pd.Series indexed by cell_id with integer tile_idx
        (flattened row-major index over the n_x * n_y grid)
      - tile_info: dict with bbox per tile_idx
    """
    n_x, n_y = n_tiles_xy
    x_col, y_col = coord_cols

    # Compute per-cell centroid
    cent = df.groupby(cell_id_col)[[x_col, y_col]].mean()

    # Bbox of the sample
    x_min, x_max = float(cent[x_col].min()), float(cent[x_col].max())
    y_min, y_max = float(cent[y_col].min()), float(cent[y_col].max())

    # Tile edges (clip the upper bound so a centroid AT x_max falls in the last bin)
    x_edges = np.linspace(x_min, x_max + 1e-9, n_x + 1)
    y_edges = np.linspace(y_min, y_max + 1e-9, n_y + 1)

    # Tile index per cell
    x_bin = np.clip(np.searchsorted(x_edges, cent[x_col].to_numpy(), side="right") - 1,
                    0, n_x - 1)
    y_bin = np.clip(np.searchsorted(y_edges, cent[y_col].to_numpy(), side="right") - 1,
                    0, n_y - 1)
    tile_idx = (x_bin * n_y + y_bin).astype(np.int32)

    # Build tile_info with bbox
    tile_info: dict[int, dict] = {}
    for ix in range(n_x):
        for iy in range(n_y):
            ti = ix * n_y + iy
            tile_info[ti] = {
                "x_min": float(x_edges[ix]), "x_max": float(x_edges[ix + 1]),
                "y_min": float(y_edges[iy]), "y_max": float(y_edges[iy + 1]),
                "ix": ix, "iy": iy,
            }

    cell_to_tile = pd.Series(tile_idx, index=cent.index, name="tile_idx")
    return cell_to_tile, tile_info


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def _build_aux_from_panel(npmi_panel: pd.DataFrame, metric_col: str = "NPMI") -> dict:
    """Build the gene_to_idx + W dict that Stitch/Rescue expect from a panel."""
    panel = npmi_panel.copy()
    panel["gene_i"] = panel["gene_i"].astype(str)
    panel["gene_j"] = panel["gene_j"].astype(str)
    all_genes = pd.unique(
        pd.concat([panel["gene_i"], panel["gene_j"]], ignore_index=True)
    )
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    G = len(all_genes)
    W = np.full((G, G), np.nan, dtype=np.float32)
    gi = panel["gene_i"].map(gene_to_idx).to_numpy()
    gj = panel["gene_j"].map(gene_to_idx).to_numpy()
    val = panel[metric_col].to_numpy(dtype=np.float32)
    W[gi, gj] = val
    W[gj, gi] = val
    return {"gene_to_idx": gene_to_idx, "W": W}


def _disambiguate_tile_labels(
    df_out: pd.DataFrame,
    cell_to_tile: dict,
    label_col: str,
    cell_id_col: str = "cell_id",
) -> pd.Series:
    """Prefix tile-local generic labels (cascade_*, UNASSIGNED_*) with
    ``tile<idx>_`` so cross-tile concat does not merge unrelated entities.

    Returns a new label Series (does not modify df_out)."""
    labels = df_out[label_col].astype(str)
    needs_prefix = (
        labels.str.startswith("cascade_") | labels.str.startswith("UNASSIGNED_")
    )
    # Key on the tile a transcript was PROCESSED in when available; fall back to
    # the cell_id centroid map (valid only for assigned tx).
    if "_proc_tile" in df_out.columns:
        tile_per_tx = df_out["_proc_tile"].astype("Int64")
    else:
        tile_per_tx = df_out[cell_id_col].map(cell_to_tile).astype("Int64")
    tile_prefix = "tile" + tile_per_tx.astype(str) + "_"
    out = labels.copy()
    out.loc[needs_prefix] = tile_prefix.loc[needs_prefix] + labels.loc[needs_prefix]
    return out


def _apply_global_post_tile_stitch(
    df_out: pd.DataFrame,
    npmi_panel: pd.DataFrame,
    cell_to_tile: dict,
    *,
    label_col: str = "stitched",
    cell_id_col: str = "cell_id",
    stitch_dist_threshold: float = 5.0,   # matches per-tile Stitch default; only ~13 cross-tile pairs would qualify anyway
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Global post-tile Stitch + Final Rescue pass.

    Pipeline (after per-tile concat):
      1. Disambiguate cascade_*/UNASSIGNED_* labels with tile prefix so
         the global frame has unique entity IDs.
      2. Build aux (gene_to_idx + W) from the npmi_panel.
      3. Run apply_stitching_to_transcripts_memory_efficient over the
         merged frame — re-merges cross-tile fragments of the same
         underlying entity.
      4. Run reassign_unassigned_grid_pool — rescues tx that were left
         unassigned by their tile-local QC but find a global home.

    Reads runner-module constants (PMI_THR, RESCUE_*, STITCH_*) so the
    parameter overrides set by the caller propagate through.

    Returns ``(df_out, stats)`` where stats has wall_seconds + n_rescued.
    """
    import tracer.pipeline as runner
    from tracer.stitching import apply_stitching_to_transcripts_memory_efficient
    from tracer.spatial import reassign_unassigned_grid_pool

    t0 = time.time()
    # 1. Disambiguate
    df_out = df_out.copy()
    df_out[label_col] = _disambiguate_tile_labels(
        df_out, cell_to_tile, label_col, cell_id_col
    ).to_numpy()
    t_disambig = time.time() - t0

    # 2. aux
    t = time.time()
    aux = _build_aux_from_panel(npmi_panel, metric_col="NPMI")
    t_aux = time.time() - t

    # 3. Global Stitch
    t = time.time()
    df_out, _ = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_out, aux=aux,
        entity_col=label_col, gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        mode="count", threshold=runner.PMI_THR, metric="pmi",
        penalize_simplicity=True, deltaC_min=0.03,
        dist_threshold=stitch_dist_threshold, out_col=label_col, show_progress=False,
        candidate_source="grid", G=2.0, stitch_neighborhood="8",
        G_z=(runner.STITCH_GZ_UM if runner.STITCH_GZ_UM is not None else 1.0),
        z_neighbor_depth=1,
        min_local_tx_per_entity=runner.STITCH_MIN_LOCAL_TX,
    )
    t_stitch = time.time() - t

    # 4. Final Rescue
    t = time.time()
    df_out, n_rescued, _ = reassign_unassigned_grid_pool(
        df_out, aux=aux,
        entity_col=label_col, gene_col="feature_name",
        coord_cols=("x", "y", "z"), out_col=label_col,
        G=2.0, pos_npmi_threshold=runner.PMI_THR,
        neg_npmi_threshold=runner.RESCUE_NEG_THR,
        only_partial_component=False,
        veto_mode=runner.RESCUE_VETO_MODE,
        mean_threshold=runner.RESCUE_MEAN_ADMIT,
        min_admit_threshold=runner.RESCUE_MIN_ADMIT,
        real_signal_threshold=0.0,
        aggregator_percentile=runner.RESCUE_AGGREGATOR_PERCENTILE,
    )
    t_rescue = time.time() - t

    if show_progress:
        print(f"[global-stitch] disambig {t_disambig:.1f}s + aux {t_aux:.1f}s "
              f"+ stitch {t_stitch:.1f}s + rescue {t_rescue:.1f}s "
              f"(rescued {n_rescued:,} tx)", flush=True)

    return df_out, {
        "wall_disambig": round(t_disambig, 2),
        "wall_aux": round(t_aux, 2),
        "wall_stitch": round(t_stitch, 2),
        "wall_rescue": round(t_rescue, 2),
        "n_rescued": int(n_rescued),
    }


def run_segmented_pipeline_tiled(
    df: pd.DataFrame,
    npmi_panel: pd.DataFrame,
    *,
    n_tiles_xy: tuple[int, int] = (2, 2),
    n_workers: int | None = None,
    cell_id_col: str = "cell_id",
    coord_cols: tuple[str, str] = ("x", "y"),
    rerank: bool = True,
    reassign: bool = True,
    global_stitch: bool = False,
    show_progress: bool = False,
) -> dict:
    """Run run_segmented_pipeline across a spatial tile grid in parallel.

    Parameters
    ----------
    df : DataFrame
        Long-format transcript table with cell_id, x, y, z, feature_name,
        overlaps_nucleus, etc. — the same input run_segmented_pipeline
        expects.
    npmi_panel : DataFrame
        PMI panel (gene_i, gene_j, NPMI). Shared across all tiles.
    n_tiles_xy : (int, int)
        Tile grid: n_x * n_y tiles.
    n_workers : int or None
        Process pool size. Defaults to n_tiles_xy[0] * n_tiles_xy[1].
    cell_id_col, coord_cols : str, tuple
        Column names for cell_id and the xy coordinates used for tile
        assignment.
    rerank, reassign : bool
        Whether to enable Phase1-Rerank / Phase1-Reassign-1c in each
        tile's run (override the runner module's defaults).
    show_progress : bool
        Print tile-start/tile-finish lines.

    Returns
    -------
    result : dict with keys:
      - df_out: concatenated per-tx output (sorted by original index)
      - per_tile: dict[tile_idx, {wall_seconds, progression, n_input_tx,
                                  n_input_cell_ids, n_out_cells, n_out_partials}]
      - tile_info: bbox per tile
      - wall_total_seconds: end-to-end wall time
      - wall_max_tile_seconds: longest single-tile wall time
    """
    n_x, n_y = n_tiles_xy
    n_tiles = n_x * n_y
    if n_workers is None:
        n_workers = n_tiles

    # 1. Tile-assign each cell.
    cell_to_tile, tile_info = _assign_cells_to_tiles(
        df, n_tiles_xy=n_tiles_xy,
        cell_id_col=cell_id_col, coord_cols=coord_cols,
    )

    # 2. Slice df per tile. Assigned tx follow their cell centroid (whole cell
    #    stays together); unassigned tx (shared cell_id "-1") are tiled by their
    #    own xy so the residual/Rescue workload spreads evenly across tiles
    #    instead of collapsing onto the single centroid tile of "-1".
    x_col, y_col = coord_cols
    tile_of_tx = df[cell_id_col].map(cell_to_tile).to_numpy()
    un = df[cell_id_col].astype(str).isin(_UNASSIGNED_TOKENS).to_numpy()
    if un.any():
        x_edges, y_edges = _tile_edges_from_info(tile_info, n_x, n_y)
        tile_of_tx[un] = _bin_coords_to_tiles(
            df.loc[un, x_col], df.loc[un, y_col], x_edges, y_edges, n_x, n_y)
    tile_dfs: dict[int, pd.DataFrame] = {}
    df_with_tile = df.assign(_tile_idx=np.asarray(tile_of_tx, np.int32))
    for ti, sub in df_with_tile.groupby("_tile_idx", sort=False):
        tile_dfs[int(ti)] = sub.drop(columns=["_tile_idx"]).reset_index(drop=True)

    if show_progress:
        print(f"[tiled] tile sizes: " + ", ".join(
            f"tile{ti}={len(td):,}tx/{td['cell_id'].nunique():,}cells"
            for ti, td in sorted(tile_dfs.items())
        ), flush=True)

    # 3. Dispatch worker jobs.
    args_list = [
        {"tile_idx": ti, "df": tile_dfs[ti], "panel": npmi_panel,
         "rerank": rerank, "reassign": reassign}
        for ti in sorted(tile_dfs)
    ]

    t_total = time.time()
    per_tile_results: dict[int, dict] = {}
    if n_workers <= 1:
        # Serial fallback: useful for tests / debugging.
        for a in args_list:
            r = _tile_worker(a)
            per_tile_results[r["tile_idx"]] = r
            if show_progress:
                print(f"[tiled] tile {r['tile_idx']} done in "
                      f"{r['wall_seconds']:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_tile_worker, a): a["tile_idx"] for a in args_list}
            for fut in as_completed(futs):
                r = fut.result()
                per_tile_results[r["tile_idx"]] = r
                if show_progress:
                    print(f"[tiled] tile {r['tile_idx']} done in "
                          f"{r['wall_seconds']:.1f}s "
                          f"({r['n_input_cell_ids']:,} input cells)", flush=True)
    wall_total = time.time() - t_total

    # 4. Concatenate outputs. Tag each row with the tile it was PROCESSED in so
    #    label disambiguation keys on the processing tile — unassigned tx share
    #    cell_id "-1" and cannot be mapped back to a tile via cell_to_tile.
    parts = []
    for ti in sorted(per_tile_results):
        part = per_tile_results[ti]["df_out"]
        part["_proc_tile"] = np.int32(ti)
        parts.append(part)
    df_out = pd.concat(parts, axis=0, ignore_index=True)

    # 4a. Disambiguate tile-local synthetic labels (cascade_*, UNASSIGNED_*) by
    #     prefixing the processing tile. Multi-tile only: a single tile needs no
    #     prefixing and must stay byte-identical to the sequential pipeline.
    if n_tiles > 1:
        _col = "stitched" if "stitched" in df_out.columns else "tracer_id"
        df_out[_col] = _disambiguate_tile_labels(
            df_out, cell_to_tile, _col, cell_id_col=cell_id_col)

    # 4b. Optional global post-tile Stitch + Final Rescue.
    #     Re-merges cross-tile fragments of the same underlying entity and
    #     rescues tx whose best fit lives in a neighboring tile.
    global_stitch_stats: dict | None = None
    if global_stitch:
        if show_progress:
            print(f"[tiled] running global post-tile Stitch + Rescue ...", flush=True)
        col = "stitched" if "stitched" in df_out.columns else "tracer_id"
        df_out, global_stitch_stats = _apply_global_post_tile_stitch(
            df_out, npmi_panel, cell_to_tile,
            label_col=col, cell_id_col=cell_id_col,
            show_progress=show_progress,
        )
        # Refresh _etype after the global pass since labels may have changed
        from tracer._etype import infer_etype_from_label
        df_out["_etype"] = np.asarray(
            infer_etype_from_label(df_out[col])
        ).astype(str)

    # 5. Compute per-tile entity stats for the result summary.
    summary_per_tile: dict[int, dict] = {}
    for ti, r in per_tile_results.items():
        df_t = r["df_out"]
        col = "stitched" if "stitched" in df_t.columns else "tracer_id"
        s = df_t[col].astype(str)
        if "_etype" in df_t.columns:
            etype = df_t["_etype"].astype(str)
        else:
            from tracer._etype import infer_etype_from_label
            etype = pd.Series(np.asarray(infer_etype_from_label(s)).astype(str))
        unassigned_tokens = {"-1", "DROP", "UNASSIGNED", "nan"}
        is_un = s.isin(unassigned_tokens) | s.str.endswith("_rejected", na=False)
        pairs = pd.DataFrame({"lab": s, "etype": etype}).loc[~is_un.to_numpy()]
        per = pairs.drop_duplicates("lab")["etype"].value_counts().to_dict()
        summary_per_tile[ti] = {
            "wall_seconds": r["wall_seconds"],
            "progression": r["progression"],
            "n_input_tx": r["n_input_tx"],
            "n_input_cell_ids": r["n_input_cell_ids"],
            "n_out_cells": int(per.get("cell", 0)),
            "n_out_partials": int(per.get("partial", 0)),
            "n_out_components": int(per.get("component", 0)),
            "n_out_unassigned_tx": int(is_un.sum()),
            "peak_rss_bytes": int(r.get("peak_rss_bytes", 0)),
        }

    wall_max_tile = max(r["wall_seconds"] for r in per_tile_results.values())
    speedup_vs_serial = (
        sum(r["wall_seconds"] for r in per_tile_results.values()) / wall_total
        if wall_total > 0 else float("nan")
    )

    # Aggregate per-worker peak RSS into max + sum (concurrent peak upper-bound).
    rss_values = [
        int(r.get("peak_rss_bytes", 0))
        for r in per_tile_results.values()
    ]
    peak_rss_max_tile_bytes = max(rss_values) if rss_values else 0
    peak_rss_sum_bytes = sum(rss_values)

    # Internal bookkeeping column; never surfaced to callers.
    df_out = df_out.drop(columns=["_proc_tile"], errors="ignore")

    return {
        "df_out": df_out,
        "per_tile": summary_per_tile,
        "tile_info": tile_info,
        "n_tiles_xy": n_tiles_xy,
        "n_workers": n_workers,
        "wall_total_seconds": round(wall_total, 2),
        "wall_max_tile_seconds": round(wall_max_tile, 2),
        "speedup_vs_serial_estimate": round(speedup_vs_serial, 2),
        "peak_rss_max_tile_bytes": peak_rss_max_tile_bytes,
        "peak_rss_sum_bytes": peak_rss_sum_bytes,
        "global_stitch_stats": global_stitch_stats,
    }
