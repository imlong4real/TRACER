"""Canonical ROI inset renderer for ovrlpy × TRACER comparisons.

This module is the **single source of truth** for ROI inset figures.
Both ``run_ovrlpy_tracer_overlap.py`` (live pipeline) and CLI re-generation
import from here so the layout and scaling stay consistent.

Layout (6 panels, one row, dark background)

  morphology z=4 + gene overlay |
  morphology z=5 + gene overlay |
  morphology z=6 + gene overlay |
  morphology z=7 + gene overlay |
  ovrlpy problem score (1 - VSI) |
  TRACER relative conflict (cell-boundary polygons)

Gene overlays
  * dominant-theme genes  -> orange triangles (▲)
  * conflicting-lineage genes -> cyan circles (●)
  Markers are small (s=4-6) and semi-transparent (alpha 0.5-0.7).

Ovrlpy panel
  * displays ``1 - mean_vsi`` per cell (== problem score) so that
    deep purple/black = good (low problem), white/yellow = problematic.
  * uses a GLOBAL vmin/vmax derived from the population, not per-ROI.
  * colorbar is labeled ``"ovrlpy problem score (1 - VSI)"``.

Coordinate scaling
  * accepts a ``coord_transform`` dict with key
    ``physical_size_x_um_per_px`` (and y).  Falls back to 0.2125 µm/px
    if not provided (the cervical Xenium5k default).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import polars as pl
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize


# finalize_atera_figures.py lives under repo_root/scripts/, two parents up.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from finalize_atera_figures import _read_tiff_page_crop  # tile-aware OME-TIFF reader


# ---------------------------------------------------------------------------
# Constants / styling
# ---------------------------------------------------------------------------

DEFAULT_MICRONS_PER_PIXEL = 0.2125  # cervical Xenium5k default
MORPH_Z_PLANES_DEFAULT = (4, 5, 6, 7)
PROBLEM_CMAP = "magma"            # deep purple/black -> yellow/white
TRACER_CMAP = "magma"
DPI = 180

CONFLICT_MARKER = dict(marker="o", color="#00E5FF", s=6.0, alpha=0.65,
                       edgecolors="white", linewidths=0.15)
DOMINANT_MARKER = dict(marker="^", color="#FFA500", s=6.0, alpha=0.65,
                       edgecolors="white", linewidths=0.15)


@dataclass
class Roi:
    name: str
    category: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _ct_um_per_px(coord_transform: Mapping[str, Any] | None) -> tuple[float, float]:
    if coord_transform is None:
        return DEFAULT_MICRONS_PER_PIXEL, DEFAULT_MICRONS_PER_PIXEL
    ux = float(coord_transform.get("physical_size_x_um_per_px", DEFAULT_MICRONS_PER_PIXEL))
    uy = float(coord_transform.get("physical_size_y_um_per_px", ux))
    return ux, uy


def um_to_px(value_um: float | np.ndarray | pd.Series,
             um_per_px: float = DEFAULT_MICRONS_PER_PIXEL) -> np.ndarray:
    return np.asarray(value_um, dtype=float) / um_per_px


def roi_px_bounds(roi: Roi,
                  coord_transform: Mapping[str, Any] | None = None,
                  ) -> tuple[int, int, int, int]:
    ux, uy = _ct_um_per_px(coord_transform)
    return (
        int(roi.xmin / ux), int(roi.xmax / ux),
        int(roi.ymin / uy), int(roi.ymax / uy),
    )


# ---------------------------------------------------------------------------
# Morphology crop reader
# ---------------------------------------------------------------------------


def read_morph_z(tif: tifffile.TiffFile,
                 roi: Roi,
                 z: int,
                 coord_transform: Mapping[str, Any] | None = None,
                 ) -> np.ndarray:
    series0 = tif.series[0]
    if z < 0 or z >= len(series0.pages):
        return np.zeros((1, 1), dtype=np.uint16)
    page = series0.pages[z]
    kf = page.keyframe if hasattr(page, "keyframe") else page
    x0, x1, y0, y1 = roi_px_bounds(roi, coord_transform)
    return _read_tiff_page_crop(page, (x0, x1, y0, y1), kf)


# ---------------------------------------------------------------------------
# Cell polygon loading
# ---------------------------------------------------------------------------


def load_cell_polys_in_roi(
    boundaries: pd.DataFrame | dict[str, np.ndarray],
    cell_ids: list[str],
    coord_transform: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Return ``cell_id -> Nx2 polygon array`` in morphology pixel coords.

    Accepts EITHER a long-format ``pd.DataFrame`` with columns
    ``cell_id, vertex_x, vertex_y`` OR a dict ``cell_id -> Nx2 µm array``
    (the in-memory format produced by ``_load_cell_boundaries`` in the
    main pipeline).  The dict form avoids re-materialising millions of
    rows on the huge cervical-Atera dataset.
    """
    ux, uy = _ct_um_per_px(coord_transform)
    if isinstance(boundaries, dict):
        out: dict[str, np.ndarray] = {}
        for cid in cell_ids:
            v = boundaries.get(str(cid))
            if v is None or len(v) == 0:
                continue
            out[str(cid)] = np.stack([v[:, 0] / ux, v[:, 1] / uy], axis=1)
        return out
    # DataFrame fallback (used by regen_roi_insets.main).
    sub = boundaries[boundaries["cell_id"].isin(cell_ids)].copy()
    sub["vx_px"] = sub["vertex_x"] / ux
    sub["vy_px"] = sub["vertex_y"] / uy
    return {
        cid: g[["vx_px", "vy_px"]].to_numpy()
        for cid, g in sub.groupby("cell_id", sort=False)
    }


# ---------------------------------------------------------------------------
# Transcript scan
# ---------------------------------------------------------------------------


def scan_transcripts_in_roi(
    transcripts_parquet: Path,
    roi: Roi,
    genes: Iterable[str],
) -> pd.DataFrame:
    """Lazy-scan the spatial transcripts parquet for the genes inside ROI."""
    genes = list(genes)
    lf = pl.scan_parquet(transcripts_parquet)
    # Try standardized column names first; fall back to Xenium native ones.
    schema = lf.schema
    x_col = "x_location" if "x_location" in schema else "x"
    y_col = "y_location" if "y_location" in schema else "y"
    g_col = "feature_name" if "feature_name" in schema else "gene"
    expr = (
        (pl.col(x_col) >= roi.xmin) & (pl.col(x_col) <= roi.xmax)
        & (pl.col(y_col) >= roi.ymin) & (pl.col(y_col) <= roi.ymax)
    )
    if genes:
        expr = expr & pl.col(g_col).is_in(genes)
    df = lf.filter(expr).select(
        pl.col(x_col).alias("x"),
        pl.col(y_col).alias("y"),
        pl.col(g_col).alias("gene"),
    ).collect()
    return df.to_pandas()


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _show_morph(ax,
                crop: np.ndarray,
                roi: Roi,
                title: str,
                coord_transform: Mapping[str, Any] | None = None,
                scalebar: bool = False) -> None:
    x0, x1, y0, y1 = roi_px_bounds(roi, coord_transform)
    img = np.log1p(crop.astype(np.float32))
    lo = float(np.nanpercentile(img, 1)) if img.size else 0.0
    hi = float(np.nanpercentile(img, 99)) if img.size else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi,
              extent=(x0, x1, y1, y0), origin="upper")
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    if scalebar:
        try:
            from matplotlib_scalebar.scalebar import ScaleBar
            ux, _ = _ct_um_per_px(coord_transform)
            ax.add_artist(ScaleBar(
                ux, units="um", loc="lower right",
                color="white", frameon=False, length_fraction=0.2,
            ))
        except Exception:
            pass


def _overlay_gene_groups(
    ax,
    roi: Roi,
    transcripts: pd.DataFrame,
    dominant_genes: list[str],
    conflicting_genes: list[str],
    coord_transform: Mapping[str, Any] | None = None,
    show_legend: bool = False,
    dominant_theme: str | None = None,
    conflicting_theme: str | None = None,
) -> tuple[int, int]:
    """Overlay dominant (orange ▲) and conflicting (cyan ●) gene transcripts."""
    ux, uy = _ct_um_per_px(coord_transform)
    n_dom = n_con = 0
    if conflicting_genes:
        sub = transcripts[transcripts["gene"].astype(str).isin(conflicting_genes)]
        if not sub.empty:
            xp = sub["x"].to_numpy() / ux
            yp = sub["y"].to_numpy() / uy
            ax.scatter(xp, yp, **CONFLICT_MARKER,
                       label=(f"Conflicting: {conflicting_theme} "
                              f"({', '.join(conflicting_genes[:3])})"
                              if show_legend and conflicting_theme else None))
            n_con = len(sub)
    if dominant_genes:
        sub = transcripts[transcripts["gene"].astype(str).isin(dominant_genes)]
        if not sub.empty:
            xp = sub["x"].to_numpy() / ux
            yp = sub["y"].to_numpy() / uy
            ax.scatter(xp, yp, **DOMINANT_MARKER,
                       label=(f"Dominant: {dominant_theme} "
                              f"({', '.join(dominant_genes[:3])})"
                              if show_legend and dominant_theme else None))
            n_dom = len(sub)
    if show_legend and (n_dom or n_con):
        leg = ax.legend(fontsize=6, loc="upper right", framealpha=0.6,
                        markerscale=1.5)
        for txt in leg.get_texts():
            txt.set_color("white")
    return n_dom, n_con


def _show_score_polys(
    ax,
    polys: dict[str, np.ndarray],
    val_by_cell: dict[str, float],
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    label: str,
    title: str,
    roi: Roi,
    coord_transform: Mapping[str, Any] | None = None,
) -> None:
    """Cell polygons filled by score (dark background, no morph overlay)."""
    x0, x1, y0, y1 = roi_px_bounds(roi, coord_transform)
    keys = list(polys.keys())
    arr = [polys[k] for k in keys]
    vals = np.asarray([val_by_cell.get(k, np.nan) for k in keys], dtype=float)
    if arr:
        coll = PolyCollection(
            arr, array=vals, cmap=cmap,
            norm=Normalize(vmin=float(vmin), vmax=float(vmax)),
            edgecolor=(0.85, 0.85, 0.85, 0.30), linewidths=0.25, alpha=0.95,
        )
        ax.add_collection(coll)
        cb = plt.colorbar(coll, ax=ax, fraction=0.046)
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=7)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Dominant / conflicting gene picker
# ---------------------------------------------------------------------------


def pick_dominant_conflicting_genes(
    cells_in_roi: pd.DataFrame,
    conflict_gene_df: pd.DataFrame,
    transcripts_in_roi: pd.DataFrame,
    lineage_markers: Mapping[str, list[str]],
    *,
    high_conflict_col: str = "tracer_high_conflict",
    cell_id_col: str = "cell_id",
    n_per_group: int = 5,
) -> dict:
    """Identify the dominant and conflicting lineage for a single ROI.

    Strategy
    --------
    1. Select the ROI's high-TRACER-conflict cells.
    2. Pull their top conflict-driver genes from ``conflict_gene_df``
       (long-format ``cell_id, gene, neg_evidence``); aggregate
       ``neg_evidence`` per gene -> score conflicting lineages.
    3. From transcripts inside the ROI, count each lineage-marker gene's
       detections (excluding the conflict-driver genes) -> score dominant
       lineages.
    4. The dominant lineage is the lineage with the highest total
       dominant-marker transcript count; the conflicting lineage is the
       lineage with the highest negative-evidence sum among conflict
       genes, restricted to lineages != dominant.
    5. For each chosen lineage, return up to ``n_per_group`` marker genes
       that (a) belong to that lineage in ``lineage_markers`` and (b) are
       actually detected in the ROI transcripts.

    Returns a dict with keys: dominant_theme, dominant_genes,
    conflicting_theme, conflicting_genes, selection_basis,
    n_dominant_transcripts, n_conflicting_transcripts, n_high_conflict_cells.
    """
    if cells_in_roi is None or cells_in_roi.empty:
        return _empty_pick("no_cells_in_roi")

    hc = cells_in_roi
    if high_conflict_col in cells_in_roi.columns:
        hc = cells_in_roi[cells_in_roi[high_conflict_col].fillna(False).astype(bool)]
    if hc.empty:
        # fall back to all ROI cells if there are no high-conflict ones.
        hc = cells_in_roi

    hc_ids = set(hc[cell_id_col].astype(str))

    # ----- conflicting lineage votes (sum neg_evidence over the cells) -----
    conf_lineage_score: dict[str, float] = {}
    conf_gene_score: dict[str, float] = {}
    if not conflict_gene_df.empty and "cell_id" in conflict_gene_df.columns:
        sub = conflict_gene_df[conflict_gene_df["cell_id"].astype(str).isin(hc_ids)]
        if not sub.empty:
            agg = sub.groupby("gene")["neg_evidence"].sum()
            for g, s in agg.items():
                conf_gene_score[str(g)] = float(s)
            for lineage, genes in lineage_markers.items():
                gset = set(genes)
                hits = sub[sub["gene"].astype(str).isin(gset)]
                if not hits.empty:
                    conf_lineage_score[lineage] = float(hits["neg_evidence"].sum())

    # ----- dominant lineage votes (count transcripts of lineage markers) ---
    dom_lineage_score: dict[str, int] = {}
    if transcripts_in_roi is not None and not transcripts_in_roi.empty:
        # Exclude transcripts whose gene is among the top conflict drivers,
        # so dominant-theme votes do not double-count conflicting evidence.
        excl = set(conf_gene_score.keys())
        dom_tx = transcripts_in_roi[~transcripts_in_roi["gene"].astype(str).isin(excl)]
        gene_counts = dom_tx["gene"].astype(str).value_counts().to_dict()
        for lineage, genes in lineage_markers.items():
            total = sum(int(gene_counts.get(g, 0)) for g in genes)
            if total > 0:
                dom_lineage_score[lineage] = total

    if not dom_lineage_score and not conf_lineage_score:
        return _empty_pick("no_lineage_evidence", n_hc=len(hc))

    dominant_theme = (
        max(dom_lineage_score, key=dom_lineage_score.get)
        if dom_lineage_score else None
    )
    # conflicting theme = top NPMI-conflict lineage that is != dominant
    conf_sorted = sorted(conf_lineage_score.items(), key=lambda kv: -kv[1])
    conflicting_theme = None
    for theme, _ in conf_sorted:
        if theme != dominant_theme:
            conflicting_theme = theme
            break
    # If no NPMI conflict lineage available, fall back to the 2nd dominant lineage.
    if conflicting_theme is None and len(dom_lineage_score) >= 2:
        dom_sorted = sorted(dom_lineage_score.items(), key=lambda kv: -kv[1])
        conflicting_theme = next((t for t, _ in dom_sorted if t != dominant_theme), None)

    # Pick representative genes from each lineage, restricted to genes actually
    # detected inside the ROI transcripts.
    detected = set(transcripts_in_roi["gene"].astype(str)) if transcripts_in_roi is not None else set()

    def _genes_for(theme: str | None,
                   prefer_scoring: Mapping[str, float] | None = None) -> list[str]:
        if theme is None or theme not in lineage_markers:
            return []
        candidates = [g for g in lineage_markers[theme] if g in detected]
        if not candidates:
            return []
        if prefer_scoring:
            candidates.sort(key=lambda g: -float(prefer_scoring.get(g, 0.0)))
        return candidates[:n_per_group]

    dominant_genes = _genes_for(dominant_theme)
    conflicting_genes = _genes_for(conflicting_theme, prefer_scoring=conf_gene_score)

    # Transcript counts for the picked genes inside the ROI.
    if transcripts_in_roi is not None and not transcripts_in_roi.empty:
        tg = transcripts_in_roi["gene"].astype(str)
        n_dom_tx = int(tg.isin(dominant_genes).sum())
        n_con_tx = int(tg.isin(conflicting_genes).sum())
    else:
        n_dom_tx = n_con_tx = 0

    return {
        "dominant_theme": dominant_theme or "",
        "dominant_genes": dominant_genes,
        "conflicting_theme": conflicting_theme or "",
        "conflicting_genes": conflicting_genes,
        "selection_basis": "transcript_count_for_dominant; npmi_neg_evidence_for_conflicting",
        "n_dominant_transcripts": n_dom_tx,
        "n_conflicting_transcripts": n_con_tx,
        "n_high_conflict_cells": int(len(hc)),
        "dominant_lineage_score": float(dom_lineage_score.get(dominant_theme, 0.0)) if dominant_theme else 0.0,
        "conflicting_lineage_score": float(conf_lineage_score.get(conflicting_theme, 0.0)) if conflicting_theme else 0.0,
    }


def _empty_pick(reason: str, n_hc: int = 0) -> dict:
    return {
        "dominant_theme": "",
        "dominant_genes": [],
        "conflicting_theme": "",
        "conflicting_genes": [],
        "selection_basis": reason,
        "n_dominant_transcripts": 0,
        "n_conflicting_transcripts": 0,
        "n_high_conflict_cells": n_hc,
        "dominant_lineage_score": 0.0,
        "conflicting_lineage_score": 0.0,
    }


# ---------------------------------------------------------------------------
# Canonical renderer
# ---------------------------------------------------------------------------


def render_canonical_roi_inset(
    roi: Roi,
    *,
    morph_path: Path,
    coord_transform: Mapping[str, Any] | None,
    boundaries_df: pd.DataFrame | dict[str, np.ndarray],
    joined: pd.DataFrame,
    transcripts_path: Path,
    dominant_genes: list[str],
    conflicting_genes: list[str],
    dominant_theme: str,
    conflicting_theme: str,
    out_base: Path,
    pop_problem_score_vmin: float,
    pop_problem_score_vmax: float,
    pop_tracer_vmin: float,
    pop_tracer_vmax: float,
    problem_score_threshold: float,
    z_planes: tuple[int, ...] = MORPH_Z_PLANES_DEFAULT,
) -> dict:
    """Render the canonical 6-panel inset for a single ROI.

    Returns per-ROI debug stats (used downstream for the ovrlpy scale-debug
    TSV).
    """
    out_base.parent.mkdir(parents=True, exist_ok=True)

    cells_in = joined[
        joined["cx"].between(roi.xmin, roi.xmax)
        & joined["cy"].between(roi.ymin, roi.ymax)
    ].copy()
    cell_ids = cells_in["cell_id"].astype(str).tolist()
    polys = load_cell_polys_in_roi(boundaries_df, cell_ids, coord_transform)

    # Score dicts (cell-level)
    problem_by_cell = {
        cid: float(1.0 - v)
        for cid, v in zip(cells_in["cell_id"].astype(str), cells_in["mean_vsi"])
        if np.isfinite(v)
    }
    tracer_by_cell = {
        cid: float(v) if np.isfinite(v) else np.nan
        for cid, v in zip(cells_in["cell_id"].astype(str), cells_in["relative_conflict"])
    }

    # Transcript overlays: pull both gene groups in one scan.
    genes_to_pull = list(dict.fromkeys(list(dominant_genes) + list(conflicting_genes)))
    tx = (scan_transcripts_in_roi(transcripts_path, roi, genes_to_pull)
          if genes_to_pull else
          pd.DataFrame(columns=["x", "y", "gene"]))

    # Morphology z-stack reads (single TiffFile open).
    morph_z: dict[int, np.ndarray] = {}
    with tifffile.TiffFile(morph_path) as tif:
        for z in z_planes:
            try:
                morph_z[z] = read_morph_z(tif, roi, z, coord_transform)
            except Exception:
                morph_z[z] = np.zeros((1, 1), dtype=np.uint16)

    with plt.style.context("dark_background"):
        n_panels = len(z_planes) + 2
        fig, axes = plt.subplots(
            1, n_panels, figsize=(3.6 * n_panels, 4.0), dpi=DPI,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        # Morphology + gene overlay panels.
        for i, z in enumerate(z_planes):
            _show_morph(
                axes[i], morph_z[z], roi,
                title=f"morphology z={z}",
                coord_transform=coord_transform,
                scalebar=(i == 0),
            )
            _overlay_gene_groups(
                axes[i], roi, tx, dominant_genes, conflicting_genes,
                coord_transform=coord_transform,
                show_legend=(i == 0),
                dominant_theme=dominant_theme,
                conflicting_theme=conflicting_theme,
            )

        # ovrlpy problem score panel.
        _show_score_polys(
            axes[len(z_planes)], polys, problem_by_cell,
            vmin=pop_problem_score_vmin, vmax=pop_problem_score_vmax,
            cmap=PROBLEM_CMAP,
            label="ovrlpy problem score (1 - VSI)",
            title=f"ovrlpy problem (n={len(polys)})",
            roi=roi, coord_transform=coord_transform,
        )

        # TRACER conflict panel.
        _show_score_polys(
            axes[len(z_planes) + 1], polys, tracer_by_cell,
            vmin=pop_tracer_vmin, vmax=pop_tracer_vmax,
            cmap=TRACER_CMAP,
            label="TRACER relative conflict",
            title=f"TRACER conflict (n={len(polys)})",
            roi=roi, coord_transform=coord_transform,
        )

        fig.suptitle(
            f"[{roi.category}] {roi.name} | "
            f"x=[{roi.xmin:.0f},{roi.xmax:.0f}] um, y=[{roi.ymin:.0f},{roi.ymax:.0f}] um",
            fontsize=10,
        )

        fig.savefig(out_base.with_suffix(".png"), dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        try:
            fig.savefig(out_base.with_suffix(".svg"), dpi=DPI, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        except Exception:
            pass
        plt.close(fig)

    # Per-ROI debug stats for ovrlpy_score_scale_debug.tsv
    vsi_arr = cells_in["mean_vsi"].to_numpy(dtype=float) if not cells_in.empty else np.array([])
    finite = vsi_arr[np.isfinite(vsi_arr)]
    if finite.size:
        min_vsi = float(np.min(finite));  max_vsi = float(np.max(finite))
        med_vsi = float(np.median(finite))
        problem = 1.0 - finite
        min_ps = float(np.min(problem));  max_ps = float(np.max(problem))
        med_ps = float(np.median(problem))
        frac_low = float(np.mean(finite < (1.0 - problem_score_threshold)))
    else:
        min_vsi = med_vsi = max_vsi = float("nan")
        min_ps = med_ps = max_ps = float("nan")
        frac_low = float("nan")

    return {
        "roi_id": roi.name,
        "roi_category": roi.category,
        "n_cells": int(len(polys)),
        "min_vsi": min_vsi,
        "median_vsi": med_vsi,
        "max_vsi": max_vsi,
        "min_problem_score": min_ps,
        "median_problem_score": med_ps,
        "max_problem_score": max_ps,
        "fraction_low_vsi": frac_low,
        "displayed_vmin": float(pop_problem_score_vmin),
        "displayed_vmax": float(pop_problem_score_vmax),
        "threshold_used": float(problem_score_threshold),
        "out_png": str(out_base.with_suffix(".png")),
    }


# ---------------------------------------------------------------------------
# CLI re-generation entry-point
# ---------------------------------------------------------------------------


def load_representative_rois(path: Path) -> list[Roi]:
    raw = json.loads(path.read_text())
    out: list[Roi] = []
    for cat, rois in raw.items():
        for r in rois:
            out.append(Roi(name=r["name"], category=cat,
                           xmin=float(r["xmin"]), xmax=float(r["xmax"]),
                           ymin=float(r["ymin"]), ymax=float(r["ymax"])))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", required=True, type=Path,
                   help="Directory containing tables/, representative_rois.json, qc/coordinate_transform.json.")
    p.add_argument("--transcripts", required=True, type=Path)
    p.add_argument("--morphology", required=True, type=Path)
    p.add_argument("--cell-boundaries", required=True, type=Path)
    p.add_argument("--only", type=str, default=None,
                   help="Only render ROIs whose name contains this substring.")
    p.add_argument("--vsi-low-threshold", type=float, default=0.5,
                   help="VSI value below which a cell is considered ovrlpy-low "
                        "(used only as the displayed threshold metadata).")
    args = p.parse_args(argv)

    outdir = args.outdir.resolve()
    insets_dir = outdir / "final_figures_fixed/roi_insets"
    insets_dir.mkdir(parents=True, exist_ok=True)

    print("Loading representative ROIs...")
    rois = load_representative_rois(outdir / "representative_rois.json")
    if args.only:
        rois = [r for r in rois if args.only in r.name]
    print(f"  {len(rois)} ROIs to render")

    print("Loading cell-level comparison TSV...")
    joined = pd.read_csv(
        outdir / "tables/ovrlpy_tracer_cell_level_comparison.tsv",
        sep="\t", dtype={"cell_id": str},
    )

    print("Loading cell_boundaries.parquet (full)...")
    boundaries_df = pd.read_parquet(
        args.cell_boundaries, columns=["cell_id", "vertex_x", "vertex_y"],
    )
    boundaries_df["cell_id"] = boundaries_df["cell_id"].astype(str)

    ct_path = outdir / "qc" / "coordinate_transform.json"
    coord_transform = (
        json.loads(ct_path.read_text()) if ct_path.exists() else None
    )

    print("Loading conflict-gene attribution table...")
    cg_path = outdir / "tables" / "tracer_cell_top_conflict_genes.tsv"
    conflict_gene_df = (
        pd.read_csv(cg_path, sep="\t", dtype={"cell_id": str})
        if cg_path.exists() else pd.DataFrame()
    )

    # Re-import LINEAGE_MARKERS from the live pipeline so the lineage map
    # stays in sync.
    from run_ovrlpy_tracer_overlap import LINEAGE_MARKERS

    pop_problem = np.nanpercentile(1.0 - joined["mean_vsi"], [1, 99])
    pop_tracer = np.nanpercentile(joined["relative_conflict"], [1, 99])
    pop_ps_clip = (max(0.0, float(pop_problem[0])), min(1.0, float(pop_problem[1])))
    pop_tr_clip = (max(0.0, float(pop_tracer[0])), float(pop_tracer[1]))

    debug_rows: list[dict] = []
    pick_rows: list[dict] = []
    for roi in rois:
        print(f"  rendering {roi.category} :: {roi.name}", flush=True)
        cells_in = joined[
            joined["cx"].between(roi.xmin, roi.xmax)
            & joined["cy"].between(roi.ymin, roi.ymax)
        ]
        tx_for_pick = scan_transcripts_in_roi(args.transcripts, roi, [])
        pick = pick_dominant_conflicting_genes(
            cells_in_roi=cells_in,
            conflict_gene_df=conflict_gene_df,
            transcripts_in_roi=tx_for_pick,
            lineage_markers=LINEAGE_MARKERS,
        )
        pick_rows.append({
            "roi_id": roi.name,
            "roi_category": roi.category,
            **{k: (",".join(v) if isinstance(v, list) else v)
               for k, v in pick.items()},
        })

        debug = render_canonical_roi_inset(
            roi=roi,
            morph_path=args.morphology,
            coord_transform=coord_transform,
            boundaries_df=boundaries_df,
            joined=joined,
            transcripts_path=args.transcripts,
            dominant_genes=pick["dominant_genes"],
            conflicting_genes=pick["conflicting_genes"],
            dominant_theme=pick["dominant_theme"],
            conflicting_theme=pick["conflicting_theme"],
            out_base=insets_dir / f"{roi.category}_{roi.name}_inset",
            pop_problem_score_vmin=pop_ps_clip[0],
            pop_problem_score_vmax=pop_ps_clip[1],
            pop_tracer_vmin=pop_tr_clip[0],
            pop_tracer_vmax=pop_tr_clip[1],
            problem_score_threshold=1.0 - float(args.vsi_low_threshold),
        )
        debug_rows.append(debug)

    qc_dir = outdir / "qc"; qc_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = outdir / "tables"; tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(debug_rows).to_csv(
        qc_dir / "ovrlpy_score_scale_debug.tsv", sep="\t", index=False,
    )
    pd.DataFrame(pick_rows).to_csv(
        tables_dir / "roi_dominant_conflicting_genes.tsv",
        sep="\t", index=False,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
