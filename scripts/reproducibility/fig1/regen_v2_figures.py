#!/usr/bin/env python3
"""v2 replots of Atera and VisiumHD overlap figures.

Cache-only — no scoring or pipeline reruns. Inputs are the already-written
joined-cell tables, ROI selections, and (for ROI insets) the VisiumHD H&E +
cell segmentation polygons. Outputs are written next to the originals with a
``_v2`` suffix or under a new ``roi_insets_v2/`` sub-folder.

What this script regenerates
----------------------------
1. Atera ``whole_tissue_categorical_overlap_v2.svg``
   * 4-way legend labels relabelled to ``ovrlpy ✓ | TRACER ✓`` etc.
   * Legend entries annotated with ``n=xxx,xxx`` counts.
   * Colours preserved.

2. VisiumHD ``whole_tissue_categorical_overlap_v2.svg``
   * Legend relabelled to ``RCTD ✓ | TRACER ✓`` etc. with ``n=xxx,xxx``.
   * Larger circular legend handles to match Atera.
   * Colours preserved.

3. VisiumHD ``whole_tissue_bounding_boxes_all_v2.svg``
   * Per-ROI text labels removed from the canvas.
   * Legend handles switched from outlined boxes to filled coloured boxes.

4. VisiumHD ROI insets v2 (15 ROIs, dark background)
   * Title:  ``{roi_id}   [x: a-b µm, y: c-d µm]   dom={lineage}, n={N}``.
   * Panel order: H&E, 2/4/8/16 µm bins, RCTD problem polygons, TRACER
     conflict polygons. Bin panels are coloured cell-by-cell on a grid by
     ``program state``:
        - dominant-only bin -> ``DOM_COLOR``
        - conflicting-only bin -> ``CONF_COLOR``
        - both (mixed) -> ``MIXED_COLOR``
        - neither (background) -> transparent (H&E shows through)
     The underlying H&E is rendered at low alpha so the program-state grid
     stays the readable layer.
   * Marker overlay on bin panels: triangle (dominant gene) / circle
     (conflicting gene), only the 2-3 canonical genes per ROI.
   * Compact legend:
        ``Dominant: {lineage} (G1, G2, G3)``
        ``Conflicting: {lineage} (G1, G2, G3)``
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle


# ---------------------------------------------------------------------------
# Constants — colour palettes locked to existing pipeline values
# ---------------------------------------------------------------------------
CAT_PALETTE = {
    # Atera
    "A_ovrlpy+_tracer+":  "#00E5FF",   # cyan
    "B_ovrlpy-_tracer+":  "#FF1493",   # magenta
    "C_ovrlpy+_tracer-":  "#39FF14",   # lime
    "D_concordant_clean": "#1a1a3a",   # dark navy
    # VisiumHD
    "A_RCTD+_TRACER+":    "#00E5FF",
    "B_RCTD+_TRACER-":    "#FF1493",
    "C_RCTD-_TRACER+":    "#39FF14",
    "D_RCTD-_TRACER-":    "#1a1a3a",
}

# Atera (ovrlpy / TRACER) — ovrlpy+ = bad, tracer+ = bad.
ATERA_LABEL = {
    "D_concordant_clean": "ovrlpy ✓ | TRACER ✓",
    "C_ovrlpy+_tracer-":  "ovrlpy ✕ | TRACER ✓",
    "B_ovrlpy-_tracer+":  "ovrlpy ✓ | TRACER ✕",
    "A_ovrlpy+_tracer+":  "ovrlpy ✕ | TRACER ✕",
}
ATERA_ORDER = [
    "D_concordant_clean",
    "C_ovrlpy+_tracer-",
    "B_ovrlpy-_tracer+",
    "A_ovrlpy+_tracer+",
]
# VisiumHD (RCTD / TRACER)
VHD_LABEL = {
    "D_RCTD-_TRACER-": "RCTD ✓ | TRACER ✓",
    "C_RCTD-_TRACER+": "RCTD ✓ | TRACER ✕",
    "B_RCTD+_TRACER-": "RCTD ✕ | TRACER ✓",
    "A_RCTD+_TRACER+": "RCTD ✕ | TRACER ✕",
}
VHD_ORDER = [
    "D_RCTD-_TRACER-",
    "C_RCTD-_TRACER+",
    "B_RCTD+_TRACER-",
    "A_RCTD+_TRACER+",
]

# ROI-inset program-state colours (dark-bg friendly, high contrast)
DOM_COLOR = "#00E5FF"     # cyan = dominant program
CONF_COLOR = "#FF1493"    # magenta = conflicting program
MIXED_COLOR = "#FFD700"   # gold = both present in the bin

# ROI-inset marker overlay style
DOM_MARKER = dict(marker="^", color=DOM_COLOR, s=22, alpha=0.85,
                  edgecolors="white", linewidths=0.25)
CONF_MARKER = dict(marker="o", color=CONF_COLOR, s=22, alpha=0.85,
                   edgecolors="white", linewidths=0.25)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1.  Atera categorical overlap v2
# ---------------------------------------------------------------------------
def replot_atera_categorical(atera_dir: Path) -> Path:
    joined_path = atera_dir / "tables" / "ovrlpy_tracer_cell_level_comparison.tsv"
    log(f"Loading Atera joined table {joined_path}")
    # Only the columns we need
    df = pd.read_csv(joined_path, sep="\t", usecols=["category", "cx", "cy"])
    counts = df["category"].value_counts()
    log(f"  category counts: {counts.to_dict()}")

    out = atera_dir / "final_figures_fixed" / "whole_tissue_categorical_overlap_v2.svg"
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
        # Layer D first (background) -> C/B -> A (most-problematic on top)
        for cat in ATERA_ORDER:
            sub = df[df["category"] == cat]
            if sub.empty:
                continue
            ax.scatter(sub["cx"], sub["cy"], s=0.7,
                       c=CAT_PALETTE[cat], alpha=0.75, linewidths=0,
                       rasterized=True)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_title("Atera — whole-tissue ovrlpy × TRACER overlap", color="white")
        ax.set_xlabel("x (µm)", color="white"); ax.set_ylabel("y (µm)", color="white")
        # Custom legend with larger filled circle handles + counts
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=CAT_PALETTE[cat],
                   markeredgecolor="white", markersize=12,
                   label=f"{ATERA_LABEL[cat]}  n={int(counts.get(cat, 0)):,}")
            for cat in ATERA_ORDER
        ]
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=10, facecolor="black", edgecolor="white",
                  labelcolor="white", handletextpad=0.6, framealpha=0.85)
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(out.with_suffix(f".{ext}"), dpi=160, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        plt.close(fig)
    log(f"Wrote {out}")
    return out


# ---------------------------------------------------------------------------
# 2.  VisiumHD categorical overlap v2
# ---------------------------------------------------------------------------
def replot_visiumhd_categorical(vhd_dir: Path, joined_df: pd.DataFrame) -> Path:
    out = vhd_dir / "figures" / "whole_tissue_categorical_overlap_v2.svg"
    counts = joined_df["overlap_category"].value_counts()
    log(f"VisiumHD category counts: {counts.to_dict()}")
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
        for cat in VHD_ORDER:
            sub = joined_df[joined_df["overlap_category"] == cat]
            if sub.empty:
                continue
            ax.scatter(sub["cx_um"], sub["cy_um"], s=0.7,
                       c=CAT_PALETTE[cat], alpha=0.75, linewidths=0,
                       rasterized=True)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_title("VisiumHD kidney — whole-tissue RCTD × TRACER overlap",
                     color="white")
        ax.set_xlabel("x (µm)", color="white"); ax.set_ylabel("y (µm)", color="white")
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=CAT_PALETTE[cat],
                   markeredgecolor="white", markersize=12,
                   label=f"{VHD_LABEL[cat]}  n={int(counts.get(cat, 0)):,}")
            for cat in VHD_ORDER
        ]
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=10, facecolor="black", edgecolor="white",
                  labelcolor="white", handletextpad=0.6, framealpha=0.85)
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(out.with_suffix(f".{ext}"), dpi=160, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        plt.close(fig)
    log(f"Wrote {out}")
    return out


# ---------------------------------------------------------------------------
# 3.  VisiumHD bounding boxes v2
# ---------------------------------------------------------------------------
def replot_visiumhd_bboxes(vhd_dir: Path, joined_df: pd.DataFrame,
                           rois: list[dict]) -> Path:
    out = vhd_dir / "figures" / "whole_tissue_bounding_boxes_all_v2.svg"
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
        # Background: all cells light grey for visible tissue silhouette.
        # Previous values (s=0.5, c="#444", alpha=0.4) rendered the tissue
        # nearly invisible against the black canvas; bumping marker size,
        # raising the brightness, and the alpha makes the tissue legible
        # while still letting the ROI bounding boxes pop.
        ax.scatter(joined_df["cx_um"], joined_df["cy_um"], s=1.0,
                   c="#bdbdbd", alpha=0.65, linewidths=0, rasterized=True)
        for roi in rois:
            color = CAT_PALETTE.get(roi["category"], "white")
            ax.add_patch(Rectangle(
                (roi["x_min_um"], roi["y_min_um"]),
                roi["x_max_um"] - roi["x_min_um"],
                roi["y_max_um"] - roi["y_min_um"],
                edgecolor=color, facecolor="none", lw=2.5,
            ))
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_title("VisiumHD kidney — representative ROI bounding boxes",
                     color="white")
        ax.set_xlabel("x (µm)", color="white"); ax.set_ylabel("y (µm)", color="white")
        # Filled coloured-box legend handles
        legend_handles = [
            Patch(facecolor=CAT_PALETTE[c], edgecolor="white", linewidth=1.0,
                  label=VHD_LABEL[c])
            for c in ["A_RCTD+_TRACER+", "B_RCTD+_TRACER-", "C_RCTD-_TRACER+"]
        ]
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=10, facecolor="black", edgecolor="white",
                  labelcolor="white", handletextpad=0.6, framealpha=0.85,
                  handlelength=2.0, handleheight=1.4)
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(out.with_suffix(f".{ext}"), dpi=160, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        plt.close(fig)
    log(f"Wrote {out}")
    return out


# ---------------------------------------------------------------------------
# 4.  VisiumHD ROI insets v2
# ---------------------------------------------------------------------------
def _aggregate_top_genes(series: pd.Series, k: int) -> list[str]:
    counter: Counter = Counter()
    for entry in series.dropna():
        for g in str(entry).split(";"):
            if g:
                counter[g] += 1
    return [g for g, _ in counter.most_common(k)]


def _block_mean(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img.copy()
    H, W = img.shape[:2]
    H2 = (H // factor) * factor
    W2 = (W // factor) * factor
    crop = img[:H2, :W2]
    shape = (H2 // factor, factor, W2 // factor, factor) + (img.shape[2:] if img.ndim > 2 else ())
    return crop.reshape(shape).mean(axis=(1, 3)).astype(img.dtype)


def _ax_clean(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)


def _load_polys(geojson_path: Path) -> dict[int, np.ndarray]:
    log(f"Loading polygons {geojson_path}")
    with open(geojson_path) as f:
        gj = json.load(f)
    out = {}
    for feat in gj.get("features", []):
        cid = int(feat["properties"]["cell_id"])
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            coords = np.asarray(geom["coordinates"][0], dtype=np.float32)
        elif geom["type"] == "MultiPolygon":
            best = max(geom["coordinates"], key=lambda r: len(r[0]))
            coords = np.asarray(best[0], dtype=np.float32)
        else:
            continue
        out[cid] = coords
    log(f"  {len(out)} polygons")
    return out


def _render_program_state_panel(
    ax,
    *,
    he_crop: np.ndarray,
    in_roi: pd.DataFrame,
    roi: dict,
    bin_um: int,
    dom_genes: list[str],
    conf_genes: list[str],
    he_alpha: float = 0.35,
) -> None:
    """Render H&E (low alpha) + program-state coloured bin grid + markers."""
    ax.set_facecolor("black")
    ax.imshow(he_crop, extent=(
        roi["x_min_um"], roi["x_max_um"],
        roi["y_max_um"], roi["y_min_um"]),
        alpha=he_alpha, interpolation="nearest")

    # Build the program-state grid over the ROI bounds.
    x0, x1 = roi["x_min_um"], roi["x_max_um"]
    y0, y1 = roi["y_min_um"], roi["y_max_um"]
    nx = max(1, int(np.ceil((x1 - x0) / bin_um)))
    ny = max(1, int(np.ceil((y1 - y0) / bin_um)))

    # Per-cell flags: does the cell express any of dom_genes / conf_genes?
    if dom_genes:
        dom_mask = in_roi["top_dominant_genes"].fillna("").str.contains(
            r"(?:^|;)(?:" + "|".join(map(_re_escape, dom_genes)) + r")(?:;|$)",
            regex=True).to_numpy()
    else:
        dom_mask = np.zeros(len(in_roi), dtype=bool)
    if conf_genes:
        conf_mask = in_roi["top_conflicting_genes"].fillna("").str.contains(
            r"(?:^|;)(?:" + "|".join(map(_re_escape, conf_genes)) + r")(?:;|$)",
            regex=True).to_numpy()
    else:
        conf_mask = np.zeros(len(in_roi), dtype=bool)

    cx_um = in_roi["cx_um"].to_numpy()
    cy_um = in_roi["cy_um"].to_numpy()
    ix = np.clip(((cx_um - x0) / bin_um).astype(int), 0, nx - 1)
    iy = np.clip(((cy_um - y0) / bin_um).astype(int), 0, ny - 1)

    dom_grid = np.zeros((ny, nx), dtype=bool)
    conf_grid = np.zeros((ny, nx), dtype=bool)
    np.add.at(dom_grid, (iy[dom_mask], ix[dom_mask]), True)
    np.add.at(conf_grid, (iy[conf_mask], ix[conf_mask]), True)
    # ``np.add.at`` with bool dest sets True where written; bools accumulate.
    dom_grid = dom_grid.astype(bool)
    conf_grid = conf_grid.astype(bool)

    # Paint per-bin coloured rectangles.
    bin_alpha = 0.55
    for j in range(ny):
        for i in range(nx):
            d = dom_grid[j, i]; c = conf_grid[j, i]
            if not (d or c):
                continue
            color = MIXED_COLOR if (d and c) else (DOM_COLOR if d else CONF_COLOR)
            ax.add_patch(Rectangle(
                (x0 + i * bin_um, y0 + j * bin_um), bin_um, bin_um,
                facecolor=color, edgecolor="none", alpha=bin_alpha,
            ))

    # Marker overlay (just dominant + conflicting gene presence per cell;
    # one symbol per gene per cell, capped so panels stay readable).
    if dom_mask.any():
        ax.scatter(cx_um[dom_mask], cy_um[dom_mask], **DOM_MARKER)
    if conf_mask.any():
        ax.scatter(cx_um[conf_mask], cy_um[conf_mask], **CONF_MARKER)

    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    _ax_clean(ax)


def _re_escape(s: str) -> str:
    import re
    return re.escape(s)


def _panel_score_polygons(ax, *, in_roi, polys, roi, score_col, title, cmap):
    polylist = []
    scores = []
    for _, row in in_roi.iterrows():
        cid = int(row["cell_id_int"])
        poly = polys.get(cid)
        if poly is None:
            continue
        microns_per_pixel = roi["_um_per_px"]
        polylist.append(poly.astype(np.float64) * microns_per_pixel)
        scores.append(float(row[score_col]) if pd.notna(row[score_col]) else np.nan)
    if not polylist:
        ax.set_title(title + "\n(no polygons)", color="white", fontsize=8)
        _ax_clean(ax); return
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if finite.size:
        vmin = float(np.nanpercentile(finite, 5))
        vmax = float(np.nanpercentile(finite, 95))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = mpl.colormaps[cmap]
    colors = cmap_obj(norm(np.nan_to_num(scores, nan=vmin)))
    pc = PolyCollection(polylist, facecolors=colors, edgecolors="white",
                        linewidths=0.2)
    ax.add_collection(pc)
    ax.set_xlim(roi["x_min_um"], roi["x_max_um"])
    ax.set_ylim(roi["y_max_um"], roi["y_min_um"])
    ax.set_facecolor("black")
    mean_score = float(np.nanmean(scores))
    ax.set_title(f"{title} (mean={mean_score:.3f})", color="white", fontsize=9)
    _ax_clean(ax)


def render_roi_inset_v2(
    roi: dict, *,
    joined: pd.DataFrame,
    polys: dict[int, np.ndarray],
    hires_img: np.ndarray,
    spatial: dict,
    out_dir: Path,
    bin_sizes_um: list[int],
) -> Path | None:
    """Render the v2 inset for one ROI."""
    # Cells inside this ROI
    in_roi = joined[
        (joined["cx_um"] >= roi["x_min_um"]) & (joined["cx_um"] < roi["x_max_um"])
        & (joined["cy_um"] >= roi["y_min_um"]) & (joined["cy_um"] < roi["y_max_um"])
    ].copy()
    if len(in_roi) == 0:
        log(f"  ROI {roi['roi_id']}: no cells in joined table; skipping")
        return None

    # Canonical 2-3 dominant + 2-3 conflict genes (most-common in this ROI)
    dom_genes = _aggregate_top_genes(in_roi["top_dominant_genes"], k=3)
    conf_genes = _aggregate_top_genes(in_roi["top_conflicting_genes"], k=3)

    # Dominant lineage = mode of predicted lineage among the *category-flagged*
    # cells inside the ROI (falls back to all cells if too few flagged).
    flagged = in_roi[in_roi["overlap_category"] == roi["category"]]
    src = flagged if len(flagged) >= 5 else in_roi
    dom_lineage = src["predicted_dominant_lineage"].mode().iat[0] \
        if not src["predicted_dominant_lineage"].mode().empty else "?"

    # Conflicting lineage = mode of predicted lineage among cells in the ROI
    # that express any of the conflicting genes (likely a different lineage
    # whose program is leaking in).
    if conf_genes:
        conf_cells_mask = in_roi["top_conflicting_genes"].fillna("").str.contains(
            r"(?:^|;)(?:" + "|".join(map(_re_escape, conf_genes)) + r")(?:;|$)",
            regex=True)
        conf_src = in_roi[conf_cells_mask]
        conf_lineage = conf_src["predicted_dominant_lineage"].mode().iat[0] \
            if not conf_src["predicted_dominant_lineage"].mode().empty else "?"
    else:
        conf_lineage = "?"

    # ---- H&E crop ---------------------------------------------------------
    um_per_px = spatial["microns_per_pixel"]
    hires_scalef = spatial["hires_scalef"]
    x0_h = roi["x_min_px"] * hires_scalef
    x1_h = roi["x_max_px"] * hires_scalef
    y0_h = roi["y_min_px"] * hires_scalef
    y1_h = roi["y_max_px"] * hires_scalef
    H_img, W_img = hires_img.shape[:2]
    x0 = max(0, int(np.floor(x0_h))); x1 = min(W_img, int(np.ceil(x1_h)))
    y0 = max(0, int(np.floor(y0_h))); y1 = min(H_img, int(np.ceil(y1_h)))
    if x1 <= x0 or y1 <= y0:
        log(f"  ROI {roi['roi_id']} outside hires image; skipping")
        return None
    he_crop = hires_img[y0:y1, x0:x1]

    # Attach scaling for the polygon panel
    roi["_um_per_px"] = um_per_px

    # ---- Figure -----------------------------------------------------------
    n_panels = 2 + len(bin_sizes_um)  # H&E + N bins + RCTD + TRACER -- actually H&E + bins + RCTD + TRACER
    n_panels = 1 + len(bin_sizes_um) + 2
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.6),
                                 dpi=170)
        axes = np.atleast_1d(axes)

        # Panel 0: H&E
        axes[0].imshow(he_crop, extent=(
            roi["x_min_um"], roi["x_max_um"],
            roi["y_max_um"], roi["y_min_um"]),
            interpolation="nearest")
        axes[0].set_xlim(roi["x_min_um"], roi["x_max_um"])
        axes[0].set_ylim(roi["y_max_um"], roi["y_min_um"])
        axes[0].set_title("H&E", color="white", fontsize=10)
        _ax_clean(axes[0])

        # Panels 1..N: program-state bin panels
        for i, bin_um in enumerate(bin_sizes_um):
            _render_program_state_panel(
                axes[1 + i], he_crop=he_crop, in_roi=in_roi, roi=roi,
                bin_um=bin_um, dom_genes=dom_genes, conf_genes=conf_genes,
            )
            axes[1 + i].set_title(f"{bin_um}×{bin_um} µm bin",
                                  color="white", fontsize=10)

        # Polygon panels
        _panel_score_polygons(
            axes[1 + len(bin_sizes_um)],
            in_roi=in_roi, polys=polys, roi=roi,
            score_col="RCTD_problem_score", title="RCTD problem", cmap="magma",
        )
        _panel_score_polygons(
            axes[1 + len(bin_sizes_um) + 1],
            in_roi=in_roi, polys=polys, roi=roi,
            score_col="TRACER_problem_score", title="TRACER conflict",
            cmap="magma",
        )

        # Title with x/y range — Atera-style
        ttl = (f"{roi['roi_id']}    "
               f"[x: {roi['x_min_um']:.0f}–{roi['x_max_um']:.0f} µm, "
               f"y: {roi['y_min_um']:.0f}–{roi['y_max_um']:.0f} µm]    "
               f"dom={dom_lineage}, n={len(in_roi)}")
        fig.suptitle(ttl, color="white", fontsize=11, y=1.02)

        # Compact gene legend below title
        dom_str = f"Dominant: {dom_lineage} ({', '.join(dom_genes) if dom_genes else 'n/a'})"
        conf_str = f"Conflicting: {conf_lineage} ({', '.join(conf_genes) if conf_genes else 'n/a'})"
        fig.text(0.5, -0.04,
                 f"{dom_str}     {conf_str}",
                 ha="center", color="white", fontsize=9)

        # Bin-state legend
        from matplotlib.lines import Line2D
        bin_handles = [
            Patch(facecolor=DOM_COLOR, edgecolor="white", linewidth=0.5,
                  label="dominant only", alpha=0.7),
            Patch(facecolor=CONF_COLOR, edgecolor="white", linewidth=0.5,
                  label="conflicting only", alpha=0.7),
            Patch(facecolor=MIXED_COLOR, edgecolor="white", linewidth=0.5,
                  label="dominant + conflicting", alpha=0.7),
            Line2D([0], [0], marker="^", linestyle="None",
                   markerfacecolor=DOM_COLOR, markeredgecolor="white",
                   markersize=8, label="dominant gene"),
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=CONF_COLOR, markeredgecolor="white",
                   markersize=8, label="conflicting gene"),
        ]
        fig.legend(handles=bin_handles, loc="lower center",
                   ncol=5, fontsize=8, facecolor="black",
                   edgecolor="white", labelcolor="white",
                   bbox_to_anchor=(0.5, -0.13),
                   handletextpad=0.5, framealpha=0.85)

        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{roi['roi_id']}_v2.png"
        out_svg = out_dir / f"{roi['roi_id']}_v2.svg"
        fig.savefig(out_png, dpi=170, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        fig.savefig(out_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atera-dir", type=Path,
                   default=Path("results/ovrlpy_tracer/cervical_atera_full_memoryaware"))
    p.add_argument("--vhd-dir", type=Path,
                   default=Path("results/kidney_visiumhd_rctd_tracer"))
    p.add_argument("--vhd-spatial-dir", type=Path,
                   default=Path("datasets/kidney_visiumhd_10x/segmented_outputs/spatial"))
    p.add_argument("--vhd-geojson", type=Path,
                   default=Path("datasets/kidney_visiumhd_10x/segmented_outputs/cell_segmentations.geojson"))
    p.add_argument("--bin-sizes-um", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--skip-atera", action="store_true")
    p.add_argument("--skip-vhd", action="store_true")
    p.add_argument("--skip-roi-insets", action="store_true")
    args = p.parse_args()

    written: list[str] = []

    if not args.skip_atera:
        try:
            f = replot_atera_categorical(args.atera_dir)
            written.append(str(f))
        except Exception as e:
            log(f"Atera categorical replot FAILED: {e}")

    if not args.skip_vhd:
        # Load VisiumHD joined (we need it for both categorical and bbox)
        joined_path = args.vhd_dir / "overlap" / "joined_rctd_tracer_scores.tsv.gz"
        log(f"Loading VisiumHD joined table {joined_path}")
        joined = pd.read_csv(
            joined_path, sep="\t",
            usecols=[
                "cell_id_int", "barcode", "cx_um", "cy_um", "cx_px", "cy_px",
                "overlap_category",
                "RCTD_problem_score", "TRACER_problem_score",
                "predicted_dominant_lineage",
                "top_dominant_genes", "top_conflicting_genes",
            ],
        )
        log(f"  joined rows = {len(joined):,}")
        rois_path = args.vhd_dir / "overlap" / "representative_rois.json"
        with open(rois_path) as f:
            rois = json.load(f)
        log(f"Loaded {len(rois)} ROIs from {rois_path}")

        try:
            f = replot_visiumhd_categorical(args.vhd_dir, joined)
            written.append(str(f))
        except Exception as e:
            log(f"VisiumHD categorical replot FAILED: {e}")
        try:
            f = replot_visiumhd_bboxes(args.vhd_dir, joined, rois)
            written.append(str(f))
        except Exception as e:
            log(f"VisiumHD bbox replot FAILED: {e}")

        if not args.skip_roi_insets:
            polys = _load_polys(args.vhd_geojson)
            # Spatial scalefactors
            with open(args.vhd_spatial_dir / "scalefactors_json.json") as f:
                sf = json.load(f)
            spatial = {
                "microns_per_pixel": float(sf["microns_per_pixel"]),
                "hires_scalef": float(sf.get("tissue_hires_scalef", 1.0)),
            }
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            hires_path = args.vhd_spatial_dir / "tissue_hires_image.png"
            hires_img = np.asarray(Image.open(hires_path).convert("RGB"))
            log(f"H&E image {hires_img.shape}; um/px={spatial['microns_per_pixel']:.4f}")

            inset_dir = args.vhd_dir / "figures" / "roi_insets_v2"
            for roi in rois:
                try:
                    f = render_roi_inset_v2(
                        roi, joined=joined, polys=polys, hires_img=hires_img,
                        spatial=spatial, out_dir=inset_dir,
                        bin_sizes_um=args.bin_sizes_um,
                    )
                    if f:
                        written.append(str(f))
                        log(f"  wrote {f}")
                except Exception as e:
                    log(f"ROI {roi['roi_id']} v2 inset FAILED: {e}")

    log("=== summary ===")
    for f in written:
        log(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
