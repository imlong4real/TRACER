"""Regenerate manuscript-grade ovrlpy x TRACER figures from cached TSVs.

This script reuses the existing per-cell scores in
``ovrlpy_tracer_cell_level_comparison.tsv`` and the morphology OME-TIFF; it
does not rerun ovrlpy or TRACER. It addresses the visualization issues raised
for the Nature Methods resubmission:

  * unified colormap for ovrlpy (1 - VSI) and TRACER conflict
  * dark-background whole-tissue point maps with rasterised small points
  * high-contrast bounding-box overlays (cyan / magenta / lime)
  * ROI insets with robust morphology validation, transformed transcript
    overlays for conflict genes, and segmented-polygon TRACER panel
  * coordinate transform reused from qc/coordinate_transform.json so every
    overlay sits in the same morphology pixel space
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
#  Constants and shared colour conventions
# ---------------------------------------------------------------------------

# Single problem-score colormap: deep blue/purple = good, white/yellow = suspicious.
PROBLEM_CMAP = "magma"

# High-contrast bounding-box colors on a dark tissue.
BOX_COLOR = {
    "A_ovrlpy+_tracer+": "#00E5FF",   # bright cyan
    "B_ovrlpy-_tracer+": "#FF2D9C",   # magenta / hot pink
    "C_ovrlpy+_tracer-": "#7CFC00",   # lime green
}

CATEGORICAL_COLOR = {
    "A_ovrlpy+_tracer+": "#00E5FF",
    "B_ovrlpy-_tracer+": "#FF2D9C",
    "C_ovrlpy+_tracer-": "#7CFC00",
    "D_concordant_clean": "#2A2A55",   # very dark blue/purple
}

DARK_BG = "#0a0a14"
PLOT_DPI = 200


@dataclass
class Roi:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


# ---------------------------------------------------------------------------
#  Centralised coordinate transform (matches run_ovrlpy_tracer_overlap.py)
# ---------------------------------------------------------------------------


def load_coordinate_transform(qc_dir: Path) -> dict:
    path = qc_dir / "coordinate_transform.json"
    if not path.exists():
        raise FileNotFoundError(f"missing coordinate transform JSON: {path}")
    return json.loads(path.read_text())


def transform_to_morphology_space(
    x_um: np.ndarray | pd.Series | list[float],
    y_um: np.ndarray | pd.Series | list[float],
    transform: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the canonical um -> morphology pixel transform.

    Used by every overlay: cell polygons, centroids, transcript points,
    ROI bounding boxes, whole-tissue boxes.
    """
    x = np.asarray(x_um, dtype=float) - float(transform.get("offset_x", 0.0))
    y = np.asarray(y_um, dtype=float) - float(transform.get("offset_y", 0.0))
    x_px = x * float(transform["scale_x"])
    y_px = y * float(transform["scale_y"])
    if bool(transform.get("invert_y", False)):
        h = transform.get("image_height_px")
        if h is None:
            raise ValueError("invert_y requires image_height_px in coordinate transform")
        y_px = float(h) - y_px
    return x_px, y_px


def roi_pixel_bounds(roi: Roi, transform: dict) -> tuple[int, int, int, int]:
    xs, ys = transform_to_morphology_space(
        [roi.xmin, roi.xmax], [roi.ymin, roi.ymax], transform,
    )
    x0 = int(np.floor(np.nanmin(xs)))
    x1 = int(np.ceil(np.nanmax(xs)))
    y0 = int(np.floor(np.nanmin(ys)))
    y1 = int(np.ceil(np.nanmax(ys)))
    h = int(transform.get("image_height_px", 0)) or None
    w = int(transform.get("image_width_px", 0)) or None
    if w is not None:
        x0 = max(0, x0); x1 = min(w, x1)
    if h is not None:
        y0 = max(0, y0); y1 = min(h, y1)
    return x0, x1, y0, y1


# ---------------------------------------------------------------------------
#  Morphology crop with validation
# ---------------------------------------------------------------------------


def read_morph_crop(
    morphology_path: Path,
    roi: Roi,
    transform: dict,
    z_index: int | None = None,
) -> dict | None:
    """Read one z-plane crop. Returns metadata + crop, or None if read fails."""
    import tifffile

    try:
        with tifffile.TiffFile(morphology_path) as tif:
            series = tif.series[0]
            n_z = len(series.pages)
            if z_index is None:
                z_index = n_z // 2
            page = series.pages[z_index]
            keyframe = page.keyframe if hasattr(page, "keyframe") else page
            bounds = roi_pixel_bounds(roi, transform)
            x0, x1, y0, y1 = bounds
            if x1 <= x0 or y1 <= y0:
                return None
            crop = _read_tiff_page_crop(page, bounds, keyframe)
    except Exception as exc:
        return {"crop": None, "bounds": None, "z_index": z_index, "error": str(exc)}
    return {"crop": crop, "bounds": bounds, "z_index": z_index, "n_z_planes": n_z}


def _read_tiff_page_crop(page, bounds: tuple[int, int, int, int], keyframe) -> np.ndarray:
    """Tile-aware rectangular crop, copied from run_ovrlpy_tracer_overlap.py."""
    x0, x1, y0, y1 = bounds
    if not getattr(keyframe, "is_tiled", False):
        arr = page.asarray(out="memmap")
        return np.asarray(arr[y0:y1, x0:x1])
    tw = int(keyframe.tilewidth)
    th = int(keyframe.tilelength)
    width = int(keyframe.imagewidth)
    height = int(keyframe.imagelength)
    ncols = int(math.ceil(width / tw))
    first_col = max(0, x0 // tw)
    last_col = min(ncols - 1, (x1 - 1) // tw)
    first_row = max(0, y0 // th)
    last_row = min(int(math.ceil(height / th)) - 1, (y1 - 1) // th)
    crop = np.zeros((y1 - y0, x1 - x0), dtype=keyframe.dtype)
    fh = page.parent.filehandle
    jpegtables = getattr(keyframe, "jpegtables", None)
    for tr in range(first_row, last_row + 1):
        for tc in range(first_col, last_col + 1):
            idx = tr * ncols + tc
            fh.seek(page.dataoffsets[idx])
            data = fh.read(page.databytecounts[idx])
            tile, _i, _s = keyframe.decode(data, idx, jpegtables=jpegtables)
            if tile is None:
                continue
            tile2 = np.squeeze(tile)
            tile_y0 = tr * th
            tile_x0 = tc * tw
            sy0 = max(y0, tile_y0); sy1 = min(y1, tile_y0 + tile2.shape[0], height)
            sx0 = max(x0, tile_x0); sx1 = min(x1, tile_x0 + tile2.shape[1], width)
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = tile2[
                sy0 - tile_y0:sy1 - tile_y0, sx0 - tile_x0:sx1 - tile_x0,
            ]
    return crop


def validate_morph_crop(crop_info: dict | None, roi_name: str) -> dict:
    """Compute crop QC stats and decide whether the crop is usable."""
    if crop_info is None or crop_info.get("crop") is None:
        return {
            "roi": roi_name,
            "ok": False,
            "reason": crop_info.get("error", "no crop returned") if crop_info else "no crop",
        }
    crop = crop_info["crop"]
    if crop.size == 0:
        return {"roi": roi_name, "ok": False, "reason": "empty crop"}
    qc = {
        "roi": roi_name,
        "z_index": int(crop_info.get("z_index", -1)),
        "shape": list(crop.shape),
        "min": float(np.nanmin(crop)),
        "max": float(np.nanmax(crop)),
        "p1": float(np.nanpercentile(crop, 1)),
        "p50": float(np.nanpercentile(crop, 50)),
        "p99_5": float(np.nanpercentile(crop, 99.5)),
        "frac_zero": float(np.mean(crop == 0)),
        "bounds_px": list(crop_info["bounds"]),
    }
    blank = (qc["frac_zero"] > 0.95) or (qc["p99_5"] - qc["p1"] < 1)
    qc["ok"] = not blank
    if blank:
        qc["reason"] = (
            f"near-blank crop: frac_zero={qc['frac_zero']:.3f}, "
            f"dynamic_range={qc['p99_5'] - qc['p1']:.1f}"
        )
    return qc


def read_morph_with_fallback(
    morphology_path: Path,
    roi: Roi,
    transform: dict,
) -> tuple[dict | None, list[dict]]:
    """Try z_mid, then ramp through z planes if blank. Returns (chosen, attempts)."""
    import tifffile

    with tifffile.TiffFile(morphology_path) as tif:
        n_z = len(tif.series[0].pages)
    z_mid = n_z // 2
    order = [z_mid] + [z for z in range(n_z) if z != z_mid]
    attempts: list[dict] = []
    for zi in order[:6]:  # cap at 6 to bound IO
        ci = read_morph_crop(morphology_path, roi, transform, z_index=zi)
        qc = validate_morph_crop(ci, roi_name=roi.name)
        attempts.append(qc)
        if qc.get("ok"):
            return ci, attempts
    return None, attempts


# ---------------------------------------------------------------------------
#  Cell-polygon loading and filtering
# ---------------------------------------------------------------------------


def load_cell_boundaries(parquet_path: Path) -> dict[str, np.ndarray]:
    cols = ["cell_id", "vertex_x", "vertex_y"]
    df = pd.read_parquet(parquet_path, columns=cols)
    df["cell_id"] = df["cell_id"].astype(str)
    out: dict[str, np.ndarray] = {}
    for cid, sub in df.groupby("cell_id", sort=False):
        out[cid] = sub[["vertex_x", "vertex_y"]].to_numpy(dtype=float, copy=False)
    return out


def polygons_in_roi_px(
    boundaries: dict[str, np.ndarray],
    roi: Roi,
    transform: dict,
    cell_ids: list[str] | None = None,
) -> dict[str, np.ndarray]:
    keys = cell_ids if cell_ids is not None else list(boundaries.keys())
    out: dict[str, np.ndarray] = {}
    for cid in keys:
        verts = boundaries.get(cid)
        if verts is None or len(verts) < 3:
            continue
        x, y = verts[:, 0], verts[:, 1]
        if (x.max() < roi.xmin or x.min() > roi.xmax or
            y.max() < roi.ymin or y.min() > roi.ymax):
            continue
        x_px, y_px = transform_to_morphology_space(x, y, transform)
        out[cid] = np.column_stack([x_px, y_px])
    return out


# ---------------------------------------------------------------------------
#  Whole-tissue figures
# ---------------------------------------------------------------------------


def _style_dark_ax(ax, transform: dict) -> None:
    ax.set_facecolor(DARK_BG)
    for sp in ax.spines.values():
        sp.set_color("#888")
    ax.tick_params(colors="#cccccc", labelsize=8)
    ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    ax.title.set_color("#ffffff")
    ax.set_aspect("equal", adjustable="box")
    if transform.get("display_y_axis_inverted") or transform.get("invert_y"):
        # Plot in µm but visually invert so y=0 is at the top.
        ax.invert_yaxis()


def plot_whole_tissue_scores(
    joined: pd.DataFrame,
    out_base: Path,
    transform: dict,
) -> None:
    """Two-panel dark-bg whole-tissue maps: ovrlpy and TRACER problem scores."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=PLOT_DPI, facecolor=DARK_BG)
    cx = joined["cx"].to_numpy()
    cy = joined["cy"].to_numpy()
    vsi = joined["mean_vsi"].to_numpy(dtype=float)
    conf = joined["relative_conflict"].to_numpy(dtype=float)
    ovrlpy_problem = 1.0 - vsi
    # Use percentile rank for each modality so colormap dynamic range is
    # dataset-independent and visually comparable. With vmin=0.5 in the
    # colormap, only cells above the median light up.
    ovrlpy_rank = pd.Series(ovrlpy_problem).rank(pct=True).to_numpy()
    conf_rank = pd.Series(conf).rank(pct=True).to_numpy()

    ax = axes[0]
    sc = ax.scatter(
        cx, cy, c=ovrlpy_rank, cmap=PROBLEM_CMAP, vmin=0.5, vmax=1.0,
        s=1.2, alpha=0.65, rasterized=True, linewidths=0,
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("ovrlpy problem score (percentile rank of 1 - VSI)",
                 color="#cccccc")
    cb.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#cccccc")
    ax.set_title("ovrlpy problem score (1 - VSI)")
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    _style_dark_ax(ax, transform)

    ax = axes[1]
    sc = ax.scatter(
        cx, cy, c=conf_rank, cmap=PROBLEM_CMAP, vmin=0.5, vmax=1.0,
        s=1.2, alpha=0.65, rasterized=True, linewidths=0,
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("TRACER conflict score (percentile rank)", color="#cccccc")
    cb.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#cccccc")
    ax.set_title("TRACER conflict score")
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    _style_dark_ax(ax, transform)

    fig.tight_layout()
    save_fig(fig, out_base)


def plot_whole_tissue_categorical(
    joined: pd.DataFrame,
    out_base: Path,
    transform: dict,
) -> None:
    """Categorical overlap map: ovrlpy+/TRACER+, TRACER-only, ovrlpy-only, neither."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 8.5), dpi=PLOT_DPI, facecolor=DARK_BG)
    cats = joined["category"].astype(str).to_numpy()

    # Plot D first (background), then A/B/C on top.
    order = ["D_concordant_clean", "C_ovrlpy+_tracer-", "B_ovrlpy-_tracer+", "A_ovrlpy+_tracer+"]
    legend_handles = []
    for cat in order:
        sel = cats == cat
        if not sel.any():
            continue
        color = CATEGORICAL_COLOR.get(cat, "#888888")
        s = 0.08 if cat == "D_concordant_clean" else 0.45
        alpha = 0.25 if cat == "D_concordant_clean" else 0.85
        ax.scatter(
            joined.loc[sel, "cx"], joined.loc[sel, "cy"],
            c=color, s=s, alpha=alpha, rasterized=True, linewidths=0,
        )
        legend_handles.append(plt.Line2D(
            [0], [0], marker="o", linestyle="", color=color, label=cat, markersize=6,
        ))
    leg = ax.legend(
        handles=legend_handles, loc="lower left", framealpha=0.7,
        facecolor="#1a1a2e", edgecolor="#888", fontsize=8,
    )
    for txt in leg.get_texts():
        txt.set_color("#eaeaea")
    ax.set_title("Categorical overlap class")
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    _style_dark_ax(ax, transform)
    fig.tight_layout()
    save_fig(fig, out_base)


def _add_boxes(ax, rois_by_cat: dict[str, list[Roi]], categories: list[str]) -> None:
    for cat in categories:
        c = BOX_COLOR.get(cat, "#ffffff")
        for r in rois_by_cat.get(cat, []):
            ax.add_patch(Rectangle(
                (r.xmin, r.ymin), r.width, r.height,
                fill=False, edgecolor=c, linewidth=3.2, alpha=0.95,
            ))


def plot_whole_tissue_with_boxes(
    joined: pd.DataFrame,
    rois_by_cat: dict[str, list[Roi]],
    out_base: Path,
    transform: dict,
    categories: list[str],
    panel_label: str,
) -> None:
    """Dark background scatter + bounding boxes for the chosen categories.

    Good cells render in deep blue/black; suspicious cells in white/yellow.
    Achieved by ranking each modality's score to a percentile in [0, 1],
    taking the max across modalities, then keeping the colormap heavily
    squashed (vmin=0.5) so only top-quintile cells light up.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 8.5), dpi=PLOT_DPI, facecolor=DARK_BG)
    cx = joined["cx"].to_numpy()
    cy = joined["cy"].to_numpy()
    conf = joined["relative_conflict"].to_numpy(dtype=float)
    ovrlpy_problem = 1.0 - joined["mean_vsi"].to_numpy(dtype=float)
    conf_rank = pd.Series(conf).rank(pct=True, method="average").to_numpy()
    ovr_rank = pd.Series(ovrlpy_problem).rank(pct=True, method="average").to_numpy()
    score = np.maximum(conf_rank, ovr_rank)
    sc = ax.scatter(
        cx, cy, c=score, cmap=PROBLEM_CMAP, vmin=0.5, vmax=1.0,
        s=1.0, alpha=0.55, rasterized=True, linewidths=0,
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("max(percentile rank: ovrlpy problem, TRACER conflict)",
                 color="#cccccc")
    cb.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#cccccc")
    _add_boxes(ax, rois_by_cat, categories)

    # Legend describing box colors.
    legend_handles = []
    for cat in categories:
        if cat not in BOX_COLOR:
            continue
        legend_handles.append(plt.Line2D(
            [0], [0], color=BOX_COLOR[cat], linewidth=3, label=cat,
        ))
    if legend_handles:
        leg = ax.legend(
            handles=legend_handles, loc="lower left", framealpha=0.7,
            facecolor="#1a1a2e", edgecolor="#888", fontsize=8,
        )
        for txt in leg.get_texts():
            txt.set_color("#eaeaea")
    ax.set_title(panel_label)
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    _style_dark_ax(ax, transform)
    fig.tight_layout()
    save_fig(fig, out_base)


# ---------------------------------------------------------------------------
#  ROI inset figure
# ---------------------------------------------------------------------------


def _show_morph(
    ax,
    crop_info: dict,
    transform: dict,
    label_unavailable: str | None = None,
    add_scalebar: bool = False,
) -> None:
    """Display a morphology crop with robust percentile clipping."""
    from matplotlib_scalebar.scalebar import ScaleBar

    if crop_info is None or crop_info.get("crop") is None or crop_info["crop"].size == 0:
        ax.set_facecolor("#181826")
        ax.text(0.5, 0.5, label_unavailable or "morphology unavailable",
                transform=ax.transAxes, ha="center", va="center",
                color="#ffb0b0", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        return
    crop = crop_info["crop"]
    bounds = crop_info["bounds"]
    # Robust contrast for fluorescent morphology with extreme dynamic range:
    # most pixels are dim background, with a sparse very-bright nuclei tail
    # spanning several orders of magnitude. log1p compresses the tail while
    # preserving cellular detail; clip at p1/p99 of the log image.
    img = np.log1p(crop.astype(np.float32))
    lo = float(np.nanpercentile(img, 1))
    hi = float(np.nanpercentile(img, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = None, None
    x0, x1, y0, y1 = bounds
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi,
              extent=(x0, x1, y1, y0), origin="upper")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    if add_scalebar:
        px_um = float(transform["scale_x"])
        ax.add_artist(ScaleBar(
            1.0 / px_um, units="um", loc="lower right",
            color="white", frameon=False, length_fraction=0.2,
        ))
    ax.set_xticks([]); ax.set_yticks([])


def _set_morph_axes_lim(ax, roi: Roi, transform: dict) -> None:
    bounds = roi_pixel_bounds(roi, transform)
    x0, x1, y0, y1 = bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # y inverted so row 0 is at top
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")


def plot_roi_inset_fixed(
    roi: Roi,
    category_label: str,
    morph_crop: dict | None,
    boundaries: dict[str, np.ndarray],
    joined: pd.DataFrame,
    transcripts_in_roi: pd.DataFrame,
    conflict_genes: list[str],
    transform: dict,
    out_base: Path,
    morph_attempts: list[dict],
) -> dict:
    """Manuscript-grade inset: morphology | ovrlpy problem | TRACER conflict | conflict tx."""
    cells_in = joined[
        (joined["cx"] >= roi.xmin) & (joined["cx"] <= roi.xmax)
        & (joined["cy"] >= roi.ymin) & (joined["cy"] <= roi.ymax)
    ].copy()
    cell_ids = cells_in["cell_id"].astype(str).tolist()
    polys = polygons_in_roi_px(boundaries, roi, transform, cell_ids=cell_ids)
    poly_keys = list(polys.keys())
    poly_arr = [polys[k] for k in poly_keys]

    ovrlpy_problem_by_cell = (
        cells_in.assign(p=1.0 - cells_in["mean_vsi"])
        .set_index(cells_in["cell_id"].astype(str))["p"].to_dict()
    )
    conflict_by_cell = (
        cells_in.set_index(cells_in["cell_id"].astype(str))["relative_conflict"].to_dict()
    )
    ovrlpy_vals = np.asarray([ovrlpy_problem_by_cell.get(c, np.nan) for c in poly_keys], dtype=float)
    conflict_vals = np.asarray([conflict_by_cell.get(c, np.nan) for c in poly_keys], dtype=float)

    # Color norms with population-level reference (1-99th percentile of full table).
    pop_ovrlpy_p = np.nanpercentile(1.0 - joined["mean_vsi"], [1, 99])
    pop_conflict_p = np.nanpercentile(joined["relative_conflict"], [1, 99])

    fig, axes = plt.subplots(
        1, 4, figsize=(15.2, 4.0), dpi=PLOT_DPI, constrained_layout=True,
        facecolor="white",
    )

    # Panel 1: morphology
    ax = axes[0]
    label_unavailable = None
    if morph_crop is None:
        attempt_msg = "\n".join(
            f"z={a.get('z_index')}: {a.get('reason', 'ok')}" for a in morph_attempts[:3]
        )
        label_unavailable = f"morphology unavailable\n{attempt_msg}"
    _show_morph(ax, morph_crop, transform,
                label_unavailable=label_unavailable, add_scalebar=True)
    ax.set_title("morphology (mid-z, validated)", fontsize=9)
    _set_morph_axes_lim(ax, roi, transform)

    # Panel 2: ovrlpy problem score = 1 - VSI on cell polygons
    ax = axes[1]
    _show_morph(ax, morph_crop, transform)  # faint background
    if poly_arr:
        coll = PolyCollection(
            poly_arr, array=ovrlpy_vals, cmap=PROBLEM_CMAP,
            norm=Normalize(vmin=max(0.0, float(pop_ovrlpy_p[0])),
                           vmax=min(1.0, float(pop_ovrlpy_p[1]))),
            edgecolor=(0.2, 0.2, 0.2, 0.35), linewidths=0.2, alpha=0.85,
        )
        ax.add_collection(coll)
        cb = plt.colorbar(coll, ax=ax, fraction=0.046)
        cb.set_label("ovrlpy problem (1 - VSI)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    ax.set_title(f"ovrlpy problem score (n={len(poly_arr)})", fontsize=9)
    _set_morph_axes_lim(ax, roi, transform)

    # Panel 3: TRACER conflict score on cell polygons
    ax = axes[2]
    _show_morph(ax, morph_crop, transform)
    if poly_arr:
        coll = PolyCollection(
            poly_arr, array=conflict_vals, cmap=PROBLEM_CMAP,
            norm=Normalize(vmin=max(0.0, float(pop_conflict_p[0])),
                           vmax=float(pop_conflict_p[1])),
            edgecolor=(0.2, 0.2, 0.2, 0.35), linewidths=0.2, alpha=0.85,
        )
        ax.add_collection(coll)
        cb = plt.colorbar(coll, ax=ax, fraction=0.046)
        cb.set_label("TRACER conflict", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    ax.set_title(f"TRACER conflict (n={len(poly_arr)})", fontsize=9)
    _set_morph_axes_lim(ax, roi, transform)

    # Panel 4: conflict-gene transcripts on morphology, y-corrected via same transform
    ax = axes[3]
    _show_morph(ax, morph_crop, transform)
    n_plotted = 0
    if conflict_genes:
        markers = ["o", "s", "^", "v", "D", "P", "*"]
        for gi, gene in enumerate(conflict_genes[:7]):
            sub = transcripts_in_roi[transcripts_in_roi["gene"].astype(str) == gene]
            if sub.empty:
                continue
            gx, gy = transform_to_morphology_space(
                sub["x"].to_numpy(), sub["y"].to_numpy(), transform,
            )
            ax.scatter(
                gx, gy, s=9, marker=markers[gi % len(markers)],
                label=gene, alpha=0.85, edgecolors="white", linewidths=0.2,
            )
            n_plotted += len(sub)
        if n_plotted:
            leg = ax.legend(fontsize=6, loc="best", framealpha=0.7)
            for txt in leg.get_texts():
                txt.set_color("#222")
    ax.set_title(f"conflict-gene transcripts (n_tx={n_plotted})", fontsize=9)
    _set_morph_axes_lim(ax, roi, transform)

    title = f"[{category_label}] {roi.name}"
    fig.suptitle(title, fontsize=11)
    save_fig(fig, out_base)
    return {
        "roi": roi.name,
        "category": category_label,
        "n_polygons": len(poly_arr),
        "n_cells_in_roi": int(len(cells_in)),
        "n_conflict_transcripts_plotted": int(n_plotted),
        "morph_crop_used": bool(morph_crop is not None),
        "morph_z_index": int(morph_crop["z_index"]) if morph_crop else -1,
        "morph_attempts": morph_attempts,
    }


# ---------------------------------------------------------------------------
#  IO helpers
# ---------------------------------------------------------------------------


def save_fig(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suf in (".png", ".pdf", ".svg"):
        fig.savefig(out_base.with_suffix(suf), dpi=PLOT_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)


def load_representative_rois(path: Path) -> dict[str, list[Roi]]:
    raw = json.loads(path.read_text())
    out: dict[str, list[Roi]] = {}
    for cat, rois in raw.items():
        out[cat] = [
            Roi(name=r["name"], xmin=float(r["xmin"]), xmax=float(r["xmax"]),
                ymin=float(r["ymin"]), ymax=float(r["ymax"]))
            for r in rois
        ]
    return out


def scan_conflict_transcripts_in_roi(
    transcripts_parquet: Path, roi: Roi, genes: list[str],
) -> pd.DataFrame:
    lf = pl.scan_parquet(transcripts_parquet)
    expr = (
        (pl.col("x_location") >= roi.xmin)
        & (pl.col("x_location") <= roi.xmax)
        & (pl.col("y_location") >= roi.ymin)
        & (pl.col("y_location") <= roi.ymax)
    )
    if genes:
        expr = expr & pl.col("feature_name").is_in(genes)
    df = lf.filter(expr).select(
        pl.col("x_location").alias("x"),
        pl.col("y_location").alias("y"),
        pl.col("z_location").alias("z"),
        pl.col("feature_name").alias("gene"),
        pl.col("cell_id"),
    ).collect()
    return df.to_pandas()


# ---------------------------------------------------------------------------
#  Conflict-gene picker per ROI
# ---------------------------------------------------------------------------


def pick_conflict_genes_for_roi(
    top_conflict_path: Path,
    cell_ids_in_roi: list[str],
    n: int = 5,
) -> list[str]:
    """Return the most enriched conflict genes in the ROI.

    The conflict-gene table is long-format ``(cell_id, gene, neg_evidence)``
    with one row per (cell, conflict gene). Rank genes by total neg_evidence
    summed across cells in the ROI.
    """
    if not top_conflict_path.exists() or not cell_ids_in_roi:
        return []
    df = pd.read_csv(top_conflict_path, sep="\t", dtype={"cell_id": str})
    df = df[df["cell_id"].astype(str).isin(cell_ids_in_roi)]
    if df.empty or "gene" not in df.columns:
        return []
    score_col = "neg_evidence" if "neg_evidence" in df.columns else None
    if score_col:
        agg = df.groupby("gene")[score_col].sum().sort_values(ascending=False)
    else:
        agg = df["gene"].value_counts()
    return agg.head(n).index.astype(str).tolist()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--transcripts", required=True, type=Path)
    p.add_argument("--morphology", required=True, type=Path)
    p.add_argument("--cell-boundaries", required=True, type=Path)
    args = p.parse_args(argv)

    outdir = args.outdir.resolve()
    final_dir = outdir / "final_figures_fixed"
    final_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = outdir / "qc"
    figs_subdir = final_dir / "roi_insets"
    figs_subdir.mkdir(exist_ok=True)

    transform = load_coordinate_transform(qc_dir)
    rois_by_cat = load_representative_rois(outdir / "representative_rois.json")
    joined = pd.read_csv(
        outdir / "tables/ovrlpy_tracer_cell_level_comparison.tsv",
        sep="\t", dtype={"cell_id": str},
    )

    # 1. Whole-tissue problem score maps (ovrlpy 1-VSI, TRACER conflict).
    plot_whole_tissue_scores(joined, final_dir / "whole_tissue_problem_score_maps", transform)
    # 2. Categorical overlap map.
    plot_whole_tissue_categorical(joined, final_dir / "whole_tissue_categorical_overlap", transform)
    # 3. Bounding-box overlays.
    plot_whole_tissue_with_boxes(
        joined, rois_by_cat, final_dir / "whole_tissue_bounding_boxes_all",
        transform,
        categories=["A_ovrlpy+_tracer+", "B_ovrlpy-_tracer+", "C_ovrlpy+_tracer-"],
        panel_label="All candidate ROIs (cyan = A, magenta = B, lime = C)",
    )
    plot_whole_tissue_with_boxes(
        joined, rois_by_cat,
        final_dir / "whole_tissue_bounding_boxes_by_category_A_ovrlpy_tracer",
        transform, categories=["A_ovrlpy+_tracer+"],
        panel_label="ovrlpy+ / TRACER+ candidate ROIs",
    )
    plot_whole_tissue_with_boxes(
        joined, rois_by_cat,
        final_dir / "whole_tissue_bounding_boxes_by_category_B_tracer_only",
        transform, categories=["B_ovrlpy-_tracer+"],
        panel_label="TRACER-only candidate ROIs",
    )
    plot_whole_tissue_with_boxes(
        joined, rois_by_cat,
        final_dir / "whole_tissue_bounding_boxes_by_category_C_ovrlpy_only",
        transform, categories=["C_ovrlpy+_tracer-"],
        panel_label="ovrlpy-only candidate ROIs",
    )

    # 4. ROI insets — delegated to regen_roi_insets.py (the canonical
    #    implementation).  finalize_atera_figures no longer renders insets
    #    so we don't keep two divergent code paths.
    print("Skipping ROI inset rendering — use regen_roi_insets.py instead.",
          flush=True)

    # 5. Persist (and verify) the coordinate transform copy.
    (final_dir / "coordinate_transform.json").write_text(json.dumps(transform, indent=2))

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
