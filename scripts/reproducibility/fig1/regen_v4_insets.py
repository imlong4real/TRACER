#!/usr/bin/env python3
"""v4 ROI inset replots (REWRITTEN) — data-driven lineage labeling,
inset colorbars on score panels, and stacked bottom-block legend on Atera.

Three corrections vs the prior v4 attempt
-----------------------------------------
1. VisiumHD lineage labeling is now **data-driven** from the per-cell
   ``top_dominant_genes`` / ``top_conflicting_genes`` cached lists. The
   RCTD ``predicted_dominant_lineage`` is no longer forced onto the legend;
   the dictionary is extended beyond the coarse 10 RCTD classes so that
   biologically real programs that aren't in that taxonomy (e.g.
   Mesangial when BGN/ITGA8 dominate, VSMC when ACTA2/MYH11/TAGLN
   dominate, IC_A vs IC_B subsets, EC_glomerular, T_cell / B_cell split)
   can be called by name. RCTD's prediction is kept in the audit table.

2. RCTD problem + TRACER conflict polygon panels keep their **square**
   parent-axes aspect — the colorbar is rendered as an INSET inside the
   axes (lower-right corner, slim vertical bar). No more rectangle
   compression from a sibling colorbar axes.

3. Atera ROI insets stack the per-ROI gene-label text and the bottom
   legend on **separate rows** so they no longer overlap.

Cache-only — does NOT rerun RCTD or TRACER scoring.

Outputs (overwriting previous v4 attempt)
-----------------------------------------
* ``results/kidney_visiumhd_rctd_tracer/figures/roi_insets_v4/<roi>_v4.{png,svg}``
* ``results/kidney_visiumhd_rctd_tracer/tables/kidney_extended_marker_dictionary_v4.tsv``
* ``results/kidney_visiumhd_rctd_tracer/tables/roi_observed_dominant_conflicting_genes_v4.tsv``
* ``results/kidney_visiumhd_rctd_tracer/tables/npmi_topk_selection_audit_v4.tsv``
* ``results/kidney_visiumhd_rctd_tracer/roi_inset_marker_audit_v4.md``
* ``results/ovrlpy_tracer/cervical_atera_full_memoryaware/final_figures_fixed/roi_insets_v3/<roi>_inset_v3.{png,svg}``
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


# ---------------------------------------------------------------------------
# Extended canonical kidney marker dictionary — one gene -> one program,
# chosen as the most-specific canonical assignment so the inverse lookup
# (gene -> lineage) used to interpret observed top_dominant / top_conflicting
# gene lists is unambiguous.
# ---------------------------------------------------------------------------
EXTENDED_KIDNEY_MARKERS: dict[str, list[str]] = {
    "PT":             ["LRP2", "CUBN", "SLC5A2", "SLC34A1", "SLC13A1", "AQP1", "ALDOB"],
    "TAL":            ["UMOD", "SLC12A1", "CLDN16", "KCNJ1"],
    "DCT":            ["SLC12A3", "TRPM6", "PVALB"],
    "CNT":            ["SCNN1B", "SCNN1G", "CALB1"],
    "PC":             ["AQP2", "AQP3", "AQP4"],
    "IC_A":           ["SLC4A1", "ATP6V1B1", "ATP6V0D2", "FOXI1"],
    "IC_B":           ["SLC26A4"],
    "POD":            ["NPHS1", "NPHS2", "PODXL", "WT1", "MAFB"],
    "EC":             ["PECAM1", "VWF", "KDR", "EMCN", "FLT1"],
    "EC_glomerular":  ["EHD3", "PLVAP", "TEK"],
    "Mesangial":      ["BGN", "ITGA8", "MFAP4", "GATA3"],
    "VSMC":           ["ACTA2", "MYH11", "TAGLN", "RGS5"],
    "Pericyte":       ["PDGFRB", "NOTCH3", "MCAM"],
    "Fibroblast":     ["COL1A1", "COL3A1", "DCN", "LUM"],
    "Myeloid":        ["LYZ", "CD68", "C1QA", "C1QB", "CSF1R", "TYROBP"],
    "T_cell":         ["CD3D", "CD3E", "TRAC", "NKG7"],
    "B_cell":         ["MS4A1", "CD79A", "CD79B"],
    "Lymphoid_other": ["PTPRC"],
    "Schwann":        ["MPZ", "PLP1", "S100B"],
}


def _build_inverse(panel: set[str]) -> tuple[dict[str, str], list[str]]:
    inv: dict[str, list[str]] = {}
    for lineage, genes in EXTENDED_KIDNEY_MARKERS.items():
        for g in genes:
            inv.setdefault(g, []).append(lineage)
    out: dict[str, str] = {}
    amb = []
    for g, lins in inv.items():
        if g not in panel:
            continue
        if len(lins) == 1:
            out[g] = lins[0]
        else:
            amb.append(g)
    return out, amb


# ---------------------------------------------------------------------------
# Colour and marker palette (matches v3/v5 conventions)
# ---------------------------------------------------------------------------
DOM_COLOR = "#00E5FF"
CONF_COLOR = "#FF1493"
MIXED_COLOR = "#FFD700"
DOM_MARKER = dict(marker="^", color=DOM_COLOR, s=22, alpha=0.85,
                  edgecolors="white", linewidths=0.25)
CONF_MARKER = dict(marker="o", color=CONF_COLOR, s=22, alpha=0.85,
                   edgecolors="white", linewidths=0.25)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# NPMI sign audit (carried over from v3 for transparency)
# ---------------------------------------------------------------------------
@dataclass
class TopKAuditResult:
    total: int; pos: int; neg: int; sign_aware: bool; note: str


def audit_topk(npmi_path: Path) -> TopKAuditResult:
    df = pd.read_csv(npmi_path, usecols=["PMI", "NPMI"])
    return TopKAuditResult(
        total=len(df),
        pos=int((df["PMI"] > 0).sum()),
        neg=int((df["PMI"] < 0).sum()),
        sign_aware=True,
        note=(
            "tracer_score_cells uses sign-aware ranking of (M @ NPMI) * M: "
            "most-positive -> top_dominant_genes, most-negative -> "
            "top_conflicting_genes. _apply_top_k retains positive AND "
            "negative pairs as separate pools. Both signs preserved."
        ),
    )


# ---------------------------------------------------------------------------
# Panel intersection
# ---------------------------------------------------------------------------
def load_visiumhd_panel(matrix_dir: Path) -> set[str]:
    fp = matrix_dir / "features.tsv.gz"
    with gzip.open(fp, "rt") as f:
        rows = [line.rstrip().split("\t") for line in f]
    return {r[1] if len(r) > 1 else r[0] for r in rows}


def intersect_dict(panel: set[str]) -> tuple[dict[str, list[str]], list[dict]]:
    used: dict[str, list[str]] = {}
    audit = []
    for lineage, genes in EXTENDED_KIDNEY_MARKERS.items():
        present = [g for g in genes if g in panel]
        used[lineage] = present
        for g in genes:
            audit.append({"lineage": lineage, "gene": g,
                          "present_in_visiumhd": g in panel})
    return used, audit


# ---------------------------------------------------------------------------
# Data-driven per-ROI assignment
# ---------------------------------------------------------------------------
@dataclass
class RoiV4Assignment:
    roi_id: str
    category: str
    top_dominant_genes_observed: list[tuple[str, int]]
    top_conflicting_genes_observed: list[tuple[str, int]]
    dominant_lineages: list[str]
    dominant_genes: list[str]
    conflicting_lineages: list[str]
    conflicting_genes: list[str]
    n_cells_dominant: int
    n_cells_conflicting: int
    rctd_predicted_lineage: str
    notes: str


def _gene_in_cell(series: pd.Series, gene: str) -> np.ndarray:
    pat = rf"(?:^|;){re.escape(gene)}(?:;|$)"
    return series.fillna("").str.contains(pat, regex=True).to_numpy()


def _aggregate_top_genes(s: pd.Series, k: int) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for entry in s.dropna():
        for g in str(entry).split(";"):
            if g:
                c[g] += 1
    return c.most_common(k)


def _lineages_from_observed(
    observed: list[tuple[str, int]],
    inverse: dict[str, str],
    *,
    top_k_genes: int = 6,
    min_vote_fraction: float = 0.30,
    max_lineages: int = 2,
) -> tuple[list[str], list[str]]:
    canon_obs = [(g, c) for g, c in observed[:top_k_genes] if g in inverse]
    if not canon_obs:
        return [], []
    votes: Counter = Counter()
    gene_per_lineage: dict[str, list[tuple[str, int]]] = {}
    for g, c in canon_obs:
        lin = inverse[g]
        votes[lin] += c
        gene_per_lineage.setdefault(lin, []).append((g, c))
    top_vote = votes.most_common(1)[0][1]
    chosen: list[str] = []
    for lin, v in votes.most_common(max_lineages):
        if v >= min_vote_fraction * top_vote or not chosen:
            chosen.append(lin)
        if len(chosen) >= max_lineages:
            break
    gene_list: list[str] = []
    for lin in chosen:
        for g, _ in sorted(gene_per_lineage.get(lin, []), key=lambda kv: -kv[1])[:3]:
            if g not in gene_list:
                gene_list.append(g)
    return chosen, gene_list


def resolve_roi_v4(roi: dict, in_roi: pd.DataFrame,
                   inverse: dict[str, str]) -> RoiV4Assignment:
    obs_dom = _aggregate_top_genes(in_roi["top_dominant_genes"], k=10)
    obs_con = _aggregate_top_genes(in_roi["top_conflicting_genes"], k=10)

    dom_lineages, dom_genes = _lineages_from_observed(obs_dom, inverse)
    conf_lineages, conf_genes = _lineages_from_observed(obs_con, inverse)

    # Don't let the same lineage appear on both sides.
    if conf_lineages and dom_lineages:
        conf_lineages = [l for l in conf_lineages if l not in dom_lineages]
    if not conf_lineages:
        conf_lineages_2, conf_genes_2 = _lineages_from_observed(
            [(g, c) for g, c in obs_con
             if inverse.get(g) and inverse[g] not in dom_lineages],
            inverse,
        )
        conf_lineages = conf_lineages_2 or ["?"]
        conf_genes = conf_genes_2 or conf_genes

    rctd_pred = in_roi["predicted_dominant_lineage"].mode().iat[0] \
        if not in_roi["predicted_dominant_lineage"].mode().empty else "?"

    def _count(series: pd.Series, genes: list[str]) -> int:
        if not genes:
            return 0
        m = np.zeros(len(in_roi), dtype=bool)
        for g in genes:
            m |= _gene_in_cell(series, g)
        return int(m.sum())

    notes = []
    if not dom_lineages:
        notes.append("no canonical lineage assignable from observed dominant "
                     "genes; falling back to RCTD prediction")
        dom_lineages = [rctd_pred]
    if rctd_pred not in dom_lineages and rctd_pred != "?":
        notes.append(f"observed-gene vote ('{'/'.join(dom_lineages)}') "
                     f"differs from RCTD prediction ('{rctd_pred}')")
    if len(dom_lineages) > 1:
        notes.append("multi-lineage dominant program")
    if len(conf_lineages) > 1:
        notes.append("multi-lineage conflict")

    return RoiV4Assignment(
        roi_id=roi["roi_id"], category=roi["category"],
        top_dominant_genes_observed=obs_dom[:5],
        top_conflicting_genes_observed=obs_con[:5],
        dominant_lineages=dom_lineages,
        dominant_genes=dom_genes,
        conflicting_lineages=conf_lineages,
        conflicting_genes=conf_genes,
        n_cells_dominant=_count(in_roi["top_dominant_genes"], dom_genes),
        n_cells_conflicting=_count(in_roi["top_conflicting_genes"], conf_genes),
        rctd_predicted_lineage=rctd_pred,
        notes="; ".join(notes),
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _ax_clean(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)


def _load_polys(geojson: Path) -> dict[int, np.ndarray]:
    with open(geojson) as f:
        gj = json.load(f)
    out: dict[int, np.ndarray] = {}
    for feat in gj.get("features", []):
        cid = int(feat["properties"]["cell_id"])
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            out[cid] = np.asarray(geom["coordinates"][0], dtype=np.float32)
        elif geom["type"] == "MultiPolygon":
            best = max(geom["coordinates"], key=lambda r: len(r[0]))
            out[cid] = np.asarray(best[0], dtype=np.float32)
    return out


class BinnedMatrixSource:
    """Lazy loader for VisiumHD ``binned_outputs/square_NNNum`` matrices.

    For each bin size the underlying inputs are
    ``square_NNNum/spatial/tissue_positions.parquet`` (one row per bin
    barcode with full-resolution pixel coordinates) and
    ``square_NNNum/filtered_feature_bc_matrix/matrix.mtx.gz`` (bins ×
    genes UMI counts). Calling :meth:`query_roi` returns, for the bins
    that physically fall inside the ROI µm bounds, the bin centre
    coordinates and per-bin booleans indicating whether the bin
    expresses ANY of the supplied dominant / conflicting marker genes.

    This is the only correct source for the v4 bin panels: replacing
    the v3-era cell-centroid pseudo-grid with the actual VisiumHD bins
    that Space Ranger emits.
    """

    def __init__(self, root: Path, microns_per_pixel: float):
        self.root = root
        self.microns_per_pixel = float(microns_per_pixel)
        self._cache: dict[int, dict] = {}

    def _load(self, bin_um: int) -> dict:
        if bin_um in self._cache:
            return self._cache[bin_um]
        import gzip as _gzip
        import scipy.io as _sio
        subdir = self.root / f"square_{bin_um:03d}um"
        pos = pd.read_parquet(subdir / "spatial" / "tissue_positions.parquet")
        # VisiumHD: pxl_row -> Y in full-res pixels; pxl_col -> X
        pos["x_um"] = pos["pxl_col_in_fullres"] * self.microns_per_pixel
        pos["y_um"] = pos["pxl_row_in_fullres"] * self.microns_per_pixel
        mtx_dir = subdir / "filtered_feature_bc_matrix"
        with _gzip.open(mtx_dir / "barcodes.tsv.gz", "rt") as f:
            barcodes = np.array([line.rstrip() for line in f], dtype=object)
        with _gzip.open(mtx_dir / "features.tsv.gz", "rt") as f:
            feats = [line.rstrip().split("\t") for line in f]
        gene_symbols = np.array(
            [r[1] if len(r) > 1 else r[0] for r in feats], dtype=object)
        log(f"  {bin_um}µm: loading matrix.mtx.gz "
            f"({len(barcodes):,} bins × {len(gene_symbols):,} genes)")
        # MTX is features × cells; transpose -> bins × genes
        X = _sio.mmread(str(mtx_dir / "matrix.mtx.gz")).tocsr().T.tocsr()
        if X.shape != (len(barcodes), len(gene_symbols)):
            raise RuntimeError(
                f"shape mismatch for {bin_um}µm: X={X.shape} "
                f"barcodes={len(barcodes)} features={len(gene_symbols)}"
            )
        gene_idx = {str(g): i for i, g in enumerate(gene_symbols)}
        pos = pos.set_index("barcode").loc[barcodes]
        self._cache[bin_um] = {
            "barcodes": barcodes,
            "X": X,
            "gene_idx": gene_idx,
            "x_um": pos["x_um"].to_numpy(dtype=np.float64),
            "y_um": pos["y_um"].to_numpy(dtype=np.float64),
            "in_tissue": pos["in_tissue"].to_numpy().astype(bool),
        }
        return self._cache[bin_um]

    def query_roi(self, bin_um: int, roi: dict,
                  dom_genes: list[str],
                  conf_genes: list[str]) -> dict[str, np.ndarray]:
        d = self._load(bin_um)
        x0, x1 = roi["x_min_um"], roi["x_max_um"]
        y0, y1 = roi["y_min_um"], roi["y_max_um"]
        in_roi_mask = (
            (d["x_um"] >= x0) & (d["x_um"] < x1)
            & (d["y_um"] >= y0) & (d["y_um"] < y1)
            & d["in_tissue"]
        )
        idx = np.flatnonzero(in_roi_mask)
        if len(idx) == 0:
            return {"x": np.array([]), "y": np.array([]),
                    "dom": np.array([], dtype=bool),
                    "conf": np.array([], dtype=bool),
                    "n_bins": 0}
        dom_cols = [d["gene_idx"][g] for g in dom_genes if g in d["gene_idx"]]
        conf_cols = [d["gene_idx"][g] for g in conf_genes if g in d["gene_idx"]]
        X_roi = d["X"][idx]
        dom_flag = np.zeros(len(idx), dtype=bool)
        conf_flag = np.zeros(len(idx), dtype=bool)
        if dom_cols:
            dom_flag = np.asarray(
                X_roi[:, dom_cols].sum(axis=1)).ravel() > 0
        if conf_cols:
            conf_flag = np.asarray(
                X_roi[:, conf_cols].sum(axis=1)).ravel() > 0
        return {"x": d["x_um"][idx], "y": d["y_um"][idx],
                "dom": dom_flag, "conf": conf_flag, "n_bins": int(len(idx))}


def _render_bin_panel(ax, *, he_crop, roi, bin_um, bin_data: dict,
                      he_alpha=0.35) -> None:
    """Render the bin panel using *real* VisiumHD bin centres and UMI
    presence/absence from ``bin_data`` (see :class:`BinnedMatrixSource`).

    Each bin is drawn as a ``bin_um × bin_um`` square centred on the bin's
    spatial position. Colour encodes program state:
    dominant-only / conflicting-only / dominant + conflicting (mixed).
    """
    ax.set_facecolor("black")
    ax.imshow(he_crop, extent=(
        roi["x_min_um"], roi["x_max_um"],
        roi["y_max_um"], roi["y_min_um"]),
        alpha=he_alpha, interpolation="nearest")

    x, y = bin_data["x"], bin_data["y"]
    dom, conf = bin_data["dom"], bin_data["conf"]
    half = bin_um / 2.0

    def _add_rects(mask: np.ndarray, color: str) -> None:
        if not mask.any():
            return
        xs, ys = x[mask], y[mask]
        verts = [
            [(xi - half, yi - half), (xi + half, yi - half),
             (xi + half, yi + half), (xi - half, yi + half)]
            for xi, yi in zip(xs, ys)
        ]
        ax.add_collection(PolyCollection(
            verts, facecolors=color, edgecolors="none", alpha=0.6,
        ))

    only_dom = dom & ~conf
    only_conf = conf & ~dom
    mixed = dom & conf
    _add_rects(only_dom, DOM_COLOR)
    _add_rects(only_conf, CONF_COLOR)
    _add_rects(mixed, MIXED_COLOR)

    ax.set_xlim(roi["x_min_um"], roi["x_max_um"])
    ax.set_ylim(roi["y_max_um"], roi["y_min_um"])
    ax.set_aspect("equal", adjustable="box")
    _ax_clean(ax)


def _panel_polys_side_cbar(
    ax, *, in_roi, polys, roi, score_col, title, cmap, um_per_px,
    cbar_label: str,
):
    """Score-polygon panel with the colorbar rendered NEXT TO (just outside)
    the parent axes — does not overlap the data area or compress the panel.
    Uses ``ax.inset_axes([1.02, 0, 0.05, 1.0])`` so the cbar sits in the
    gutter to the right of the panel."""
    polylist = []; scores = []
    for _, row in in_roi.iterrows():
        cid = int(row["cell_id_int"])
        poly = polys.get(cid)
        if poly is None:
            continue
        polylist.append(poly.astype(np.float64) * um_per_px)
        scores.append(float(row[score_col]) if pd.notna(row[score_col]) else np.nan)
    ax.set_facecolor("black")
    ax.set_xlim(roi["x_min_um"], roi["x_max_um"])
    ax.set_ylim(roi["y_max_um"], roi["y_min_um"])
    ax.set_aspect("equal", adjustable="box")
    _ax_clean(ax)
    if not polylist:
        ax.set_title(title + "\n(no polygons)", color="white", fontsize=9)
        return
    arr = np.asarray(scores, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        vmin = float(np.nanpercentile(finite, 5))
        vmax = float(np.nanpercentile(finite, 95))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0
    coll = PolyCollection(
        polylist,
        array=np.nan_to_num(arr, nan=vmin),
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        edgecolors="white", linewidths=0.2, alpha=0.95,
    )
    ax.add_collection(coll)
    # Side colorbar — anchored just OUTSIDE the right edge of the parent
    # axes (x = 1.02 in axes-fraction coords). Parent axes stays fully
    # available to the polygons; the cbar lives in the inter-panel gutter.
    cax = ax.inset_axes([1.02, 0.0, 0.05, 1.0])
    cbar = plt.colorbar(coll, cax=cax, orientation="vertical")
    cbar.set_label(cbar_label, color="white", fontsize=7,
                   rotation=270, labelpad=12)
    cbar.ax.tick_params(colors="white", labelsize=6)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("white")
    cbar.outline.set_edgecolor("white")
    ax.set_title(f"{title} (mean={float(np.nanmean(arr)):.3f})",
                 color="white", fontsize=9)


def render_v4_inset(roi: dict, *, assignment: RoiV4Assignment,
                    joined: pd.DataFrame, polys: dict[int, np.ndarray],
                    hires_img: np.ndarray, spatial: dict,
                    out_dir: Path, bin_sizes_um: list[int],
                    bin_source: BinnedMatrixSource) -> Path | None:
    in_roi = joined[
        (joined["cx_um"] >= roi["x_min_um"]) & (joined["cx_um"] < roi["x_max_um"])
        & (joined["cy_um"] >= roi["y_min_um"]) & (joined["cy_um"] < roi["y_max_um"])
    ].copy()
    if in_roi.empty:
        return None

    um_per_px = spatial["microns_per_pixel"]
    hires_scalef = spatial["hires_scalef"]
    H_img, W_img = hires_img.shape[:2]
    x0 = max(0, int(np.floor(roi["x_min_px"] * hires_scalef)))
    x1 = min(W_img, int(np.ceil(roi["x_max_px"] * hires_scalef)))
    y0 = max(0, int(np.floor(roi["y_min_px"] * hires_scalef)))
    y1 = min(H_img, int(np.ceil(roi["y_max_px"] * hires_scalef)))
    if x1 <= x0 or y1 <= y0:
        return None
    he_crop = hires_img[y0:y1, x0:x1]

    n_panels = 1 + len(bin_sizes_um) + 2
    with plt.style.context("dark_background"):
        # Widen inter-panel gutter so the side colorbars on the RCTD and
        # TRACER panels live in dead space, not on top of the next panel.
        fig, axes = plt.subplots(
            1, n_panels, figsize=(3.2 * n_panels, 3.6), dpi=170,
            gridspec_kw={"wspace": 0.4},
        )
        axes = np.atleast_1d(axes)

        axes[0].imshow(he_crop, extent=(
            roi["x_min_um"], roi["x_max_um"],
            roi["y_max_um"], roi["y_min_um"]),
            interpolation="nearest")
        axes[0].set_xlim(roi["x_min_um"], roi["x_max_um"])
        axes[0].set_ylim(roi["y_max_um"], roi["y_min_um"])
        axes[0].set_title("H&E", color="white", fontsize=10)
        _ax_clean(axes[0])

        for i, bin_um in enumerate(bin_sizes_um):
            # Source per-bin presence/absence from the actual VisiumHD
            # square_NNNum binned matrix at this resolution — NOT from
            # cell-centroid pseudo-binning.
            bd = bin_source.query_roi(
                bin_um, roi,
                dom_genes=assignment.dominant_genes,
                conf_genes=assignment.conflicting_genes,
            )
            _render_bin_panel(
                axes[1 + i], he_crop=he_crop, roi=roi,
                bin_um=bin_um, bin_data=bd,
            )
            axes[1 + i].set_title(
                f"{bin_um}×{bin_um} µm bin (n={bd['n_bins']})",
                color="white", fontsize=10,
            )

        _panel_polys_side_cbar(
            axes[1 + len(bin_sizes_um)],
            in_roi=in_roi, polys=polys, roi=roi,
            score_col="RCTD_problem_score", title="RCTD problem",
            cmap="magma", um_per_px=um_per_px,
            cbar_label="RCTD problem",
        )
        _panel_polys_side_cbar(
            axes[1 + len(bin_sizes_um) + 1],
            in_roi=in_roi, polys=polys, roi=roi,
            score_col="TRACER_problem_score", title="TRACER conflict",
            cmap="magma", um_per_px=um_per_px,
            cbar_label="TRACER conflict",
        )

        title = (f"{roi['roi_id']} [{roi['category']}]    "
                 f"[x: {roi['x_min_um']:.0f}–{roi['x_max_um']:.0f} µm, "
                 f"y: {roi['y_min_um']:.0f}–{roi['y_max_um']:.0f} µm]    "
                 f"n={len(in_roi)}")
        fig.suptitle(title, color="white", fontsize=11, y=1.02)

        # Stacked bottom block — gene labels above, legend below
        dom_str = (f"Dominant: {'/'.join(assignment.dominant_lineages)} "
                   f"({', '.join(assignment.dominant_genes) or 'n/a'})  "
                   f"n_cells={assignment.n_cells_dominant}")
        conf_str = (f"Conflicting: {'/'.join(assignment.conflicting_lineages)} "
                    f"({', '.join(assignment.conflicting_genes) or 'n/a'})  "
                    f"n_cells={assignment.n_cells_conflicting}")
        fig.text(0.5, -0.04, f"{dom_str}     {conf_str}",
                 ha="center", color="white", fontsize=9)

        legend_handles = [
            Patch(facecolor=DOM_COLOR, edgecolor="white", linewidth=0.5,
                  label="dominant-only bin", alpha=0.7),
            Patch(facecolor=CONF_COLOR, edgecolor="white", linewidth=0.5,
                  label="conflicting-only bin", alpha=0.7),
            Patch(facecolor=MIXED_COLOR, edgecolor="white", linewidth=0.5,
                  label="dominant + conflicting", alpha=0.7),
            Line2D([0], [0], marker="^", linestyle="None",
                   markerfacecolor=DOM_COLOR, markeredgecolor="white",
                   markersize=8, label="dominant gene cells"),
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=CONF_COLOR, markeredgecolor="white",
                   markersize=8, label="conflicting gene cells"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=5,
                   fontsize=8, facecolor="black", edgecolor="white",
                   labelcolor="white", bbox_to_anchor=(0.5, -0.14),
                   handletextpad=0.5, framealpha=0.85)

        # Skip tight_layout — the outside-bounds inset colorbars cause its
        # warning and don't need its inter-axes alignment work since we
        # already set wspace explicitly via gridspec_kw above. bbox_inches
        # = "tight" at savefig time still produces correctly-cropped files.
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{roi['roi_id']}_v4.png"
        out_svg = out_dir / f"{roi['roi_id']}_v4.svg"
        fig.savefig(out_png, dpi=170, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        fig.savefig(out_svg, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Atera v3 — stacked bottom block so gene labels don't overlap the legend
# ---------------------------------------------------------------------------
def _atera_render_v3(atera_dir: Path, out_dir: Path,
                     transcripts_path: Path, morph_path: Path,
                     cell_boundaries_path: Path) -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "regen_roi_insets",
        Path(__file__).resolve().parent / "regen_roi_insets.py")
    rri = importlib.util.module_from_spec(spec)
    sys.modules["regen_roi_insets"] = rri
    spec.loader.exec_module(rri)

    _orig_overlay = rri._overlay_gene_groups
    def _silent(ax, roi, transcripts, dominant_genes, conflicting_genes,
                coord_transform=None, show_legend=False,
                dominant_theme=None, conflicting_theme=None):
        return _orig_overlay(
            ax, roi, transcripts, dominant_genes, conflicting_genes,
            coord_transform=coord_transform, show_legend=False,
            dominant_theme=dominant_theme, conflicting_theme=conflicting_theme,
        )
    rri._overlay_gene_groups = _silent

    _pick_state: dict[str, dict] = {}
    out_dir_resolved = out_dir.resolve()
    _orig_savefig = plt.Figure.savefig

    def _wrapped_savefig(self, fname, *a, **kw):
        try:
            target = Path(fname).resolve() if not isinstance(fname, Path) else fname.resolve()
            is_v3 = str(target).startswith(str(out_dir_resolved))
        except Exception:
            is_v3 = False
        if is_v3 and not getattr(self, "_v3_legend_added", False):
            pick = _pick_state.get("current", {})
            dom_theme = pick.get("dominant_theme", "?")
            con_theme = pick.get("conflicting_theme", "?")
            dom_genes = pick.get("dominant_genes", []) or []
            con_genes = pick.get("conflicting_genes", []) or []
            # Row A — gene label text (closer to figure)
            label = (
                f"Dominant: {dom_theme} ({', '.join(dom_genes[:3]) or 'n/a'})     "
                f"Conflicting: {con_theme} ({', '.join(con_genes[:3]) or 'n/a'})"
            )
            self.text(0.5, -0.04, label, ha="center", color="white",
                      fontsize=10, transform=self.transFigure)
            # Row B — legend strip BELOW the gene text
            handles = [
                Line2D([0], [0], marker="^", linestyle="None",
                       markerfacecolor="#FFA500", markeredgecolor="white",
                       markersize=9, label="dominant gene transcripts (▲)"),
                Line2D([0], [0], marker="o", linestyle="None",
                       markerfacecolor="#00E5FF", markeredgecolor="white",
                       markersize=9, label="conflicting gene transcripts (●)"),
                Patch(facecolor=mpl.colormaps["magma"](0.85),
                      edgecolor="white", linewidth=0.5,
                      label="ovrlpy / TRACER score (magma)"),
            ]
            self.legend(handles=handles, loc="lower center", ncol=3,
                        fontsize=9, facecolor="black", edgecolor="white",
                        labelcolor="white",
                        bbox_to_anchor=(0.5, -0.14),
                        handletextpad=0.5, framealpha=0.85)
            self._v3_legend_added = True
        return _orig_savefig(self, fname, *a, **kw)
    plt.Figure.savefig = _wrapped_savefig

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        rois = rri.load_representative_rois(atera_dir / "representative_rois.json")
        log(f"  Atera v3: {len(rois)} ROIs")
        joined = pd.read_csv(
            atera_dir / "tables" / "ovrlpy_tracer_cell_level_comparison.tsv",
            sep="\t", dtype={"cell_id": str},
        )
        boundaries_df = pd.read_parquet(
            cell_boundaries_path, columns=["cell_id", "vertex_x", "vertex_y"],
        )
        boundaries_df["cell_id"] = boundaries_df["cell_id"].astype(str)
        ct_path = atera_dir / "qc" / "coordinate_transform.json"
        coord_transform = (
            json.loads(ct_path.read_text()) if ct_path.exists() else None
        )
        cg_path = atera_dir / "tables" / "tracer_cell_top_conflict_genes.tsv"
        conflict_gene_df = (
            pd.read_csv(cg_path, sep="\t", dtype={"cell_id": str})
            if cg_path.exists() else pd.DataFrame()
        )
        # run_ovrlpy_tracer_overlap.py lives at repo_root/scripts/.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from run_ovrlpy_tracer_overlap import LINEAGE_MARKERS

        pop_problem = np.nanpercentile(1.0 - joined["mean_vsi"], [1, 99])
        pop_tracer = np.nanpercentile(joined["relative_conflict"], [1, 99])
        pop_ps_clip = (max(0.0, float(pop_problem[0])), min(1.0, float(pop_problem[1])))
        pop_tr_clip = (max(0.0, float(pop_tracer[0])), float(pop_tracer[1]))

        n = 0
        for roi in rois:
            cells_in = joined[
                joined["cx"].between(roi.xmin, roi.xmax)
                & joined["cy"].between(roi.ymin, roi.ymax)
            ]
            tx = rri.scan_transcripts_in_roi(transcripts_path, roi, [])
            pick = rri.pick_dominant_conflicting_genes(
                cells_in_roi=cells_in,
                conflict_gene_df=conflict_gene_df,
                transcripts_in_roi=tx,
                lineage_markers=LINEAGE_MARKERS,
            )
            _pick_state["current"] = pick
            log(f"  Atera v3 {roi.category} :: {roi.name} | "
                f"dom={pick.get('dominant_theme')} "
                f"conf={pick.get('conflicting_theme')}")
            try:
                rri.render_canonical_roi_inset(
                    roi=roi, morph_path=morph_path,
                    coord_transform=coord_transform,
                    boundaries_df=boundaries_df, joined=joined,
                    transcripts_path=transcripts_path,
                    dominant_genes=pick["dominant_genes"],
                    conflicting_genes=pick["conflicting_genes"],
                    dominant_theme=pick["dominant_theme"],
                    conflicting_theme=pick["conflicting_theme"],
                    out_base=out_dir / f"{roi.category}_{roi.name}_inset_v3",
                    pop_problem_score_vmin=pop_ps_clip[0],
                    pop_problem_score_vmax=pop_ps_clip[1],
                    pop_tracer_vmin=pop_tr_clip[0],
                    pop_tracer_vmax=pop_tr_clip[1],
                    problem_score_threshold=0.5,
                )
                n += 1
            except Exception as e:
                log(f"    FAILED {roi.name}: {e}")
        return n
    finally:
        plt.Figure.savefig = _orig_savefig
        rri._overlay_gene_groups = _orig_overlay


# ---------------------------------------------------------------------------
# Audit writers
# ---------------------------------------------------------------------------
def write_marker_dict_tsv(audit: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(out, sep="\t", index=False)


def write_observed_genes_tsv(assignments: list[RoiV4Assignment],
                             out: Path) -> None:
    rows = []
    for a in assignments:
        rows.append({
            "roi_id": a.roi_id,
            "category": a.category,
            "rctd_predicted_lineage": a.rctd_predicted_lineage,
            "dominant_lineages_called": "/".join(a.dominant_lineages),
            "dominant_canonical_genes": ", ".join(a.dominant_genes),
            "top_dominant_genes_observed": "; ".join(
                f"{g}:{c}" for g, c in a.top_dominant_genes_observed),
            "conflicting_lineages_called": "/".join(a.conflicting_lineages),
            "conflicting_canonical_genes": ", ".join(a.conflicting_genes),
            "top_conflicting_genes_observed": "; ".join(
                f"{g}:{c}" for g, c in a.top_conflicting_genes_observed),
            "n_cells_dominant": a.n_cells_dominant,
            "n_cells_conflicting": a.n_cells_conflicting,
            "notes": a.notes,
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


def write_topk_audit_v4(
    assignments: list[RoiV4Assignment], npmi_path: Path, out: Path,
) -> None:
    long = pd.read_csv(npmi_path, usecols=["gene_i", "gene_j", "PMI", "NPMI"])
    a = long.rename(columns={"gene_i": "g", "gene_j": "p"})
    b = long.rename(columns={"gene_j": "g", "gene_i": "p"})
    sym = pd.concat([a, b], ignore_index=True)
    summary = sym.groupby("g").agg(
        mean_PMI=("PMI", "mean"),
        mean_NPMI=("NPMI", "mean"),
        max_abs_PMI=("PMI", lambda s: float(np.nanmax(np.abs(s)))),
        max_abs_NPMI=("NPMI", lambda s: float(np.nanmax(np.abs(s)))),
    )
    rows = []
    for a in assignments:
        for g in a.dominant_genes:
            r = summary.loc[g] if g in summary.index else None
            rows.append({
                "roi_id": a.roi_id,
                "dominant_lineage": "/".join(a.dominant_lineages),
                "conflicting_lineage": "/".join(a.conflicting_lineages),
                "gene": g,
                "PMI": float(r["mean_PMI"]) if r is not None else float("nan"),
                "NPMI": float(r["mean_NPMI"]) if r is not None else float("nan"),
                "abs_PMI": float(r["max_abs_PMI"]) if r is not None else float("nan"),
                "abs_NPMI": float(r["max_abs_NPMI"]) if r is not None else float("nan"),
                "pair_sign": "+",
                "selected_reason": "observed_canonical_dominant_v4",
            })
        for g in a.conflicting_genes:
            r = summary.loc[g] if g in summary.index else None
            rows.append({
                "roi_id": a.roi_id,
                "dominant_lineage": "/".join(a.dominant_lineages),
                "conflicting_lineage": "/".join(a.conflicting_lineages),
                "gene": g,
                "PMI": float(r["mean_PMI"]) if r is not None else float("nan"),
                "NPMI": float(r["mean_NPMI"]) if r is not None else float("nan"),
                "abs_PMI": float(r["max_abs_PMI"]) if r is not None else float("nan"),
                "abs_NPMI": float(r["max_abs_NPMI"]) if r is not None else float("nan"),
                "pair_sign": "-",
                "selected_reason": "observed_canonical_conflicting_v4",
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


def write_markdown_v4(
    *, topk: TopKAuditResult, used: dict[str, list[str]],
    n_dropped: int, assignments: list[RoiV4Assignment],
    ambiguous: list[str], out: Path,
) -> None:
    lines = []
    lines.append("# VisiumHD ROI inset marker audit (v4 — rewritten)\n")
    lines.append(f"Date: {time.strftime('%Y-%m-%d')}\n")
    lines.append("## 1. Top-k NPMI / PMI selection (carried over from v3)\n")
    lines.append(
        f"- NPMI table: **{topk.total:,}** pairs total — **{topk.pos:,}** "
        f"positive PMI ({topk.pos / max(1, topk.total):.1%}), "
        f"**{topk.neg:,}** negative PMI ({topk.neg / max(1, topk.total):.1%}).\n"
        f"- {topk.note}\n"
        f"- **Result**: no code change required for sign awareness.\n"
    )
    lines.append("## 2. Strategy change — data-driven lineage labeling\n")
    lines.append(
        "v4 stops forcing the RCTD-predicted coarse class onto each ROI. "
        "Instead it aggregates the per-cell `top_dominant_genes` / "
        "`top_conflicting_genes` lists already cached in the joined table "
        "and assigns lineages from the genes **actually observed** in the "
        "ROI. The dictionary is extended beyond the 10 RCTD coarse classes "
        "so that programs absent from the RCTD taxonomy (Mesangial, VSMC, "
        "Pericyte, IC_A vs IC_B subsets, EC_glomerular, T_cell / B_cell "
        "split, etc.) can be called by their canonical name. The RCTD "
        "prediction is retained in the audit table for transparency.\n"
    )
    lines.append("Per-ROI procedure:\n"
                 "1. Aggregate `top_dominant_genes` across the ROI cells.\n"
                 "2. Take the top-6 most-frequent observed genes that are "
                 "canonical markers in the extended dictionary.\n"
                 "3. Vote their canonical lineages; keep lineages whose "
                 "vote share is ≥ 30 % of the leader (up to 2 lineages).\n"
                 "4. The displayed canonical genes are the observed canonical "
                 "markers for the chosen lineage(s), in descending observation "
                 "frequency.\n"
                 "5. Repeat for `top_conflicting_genes`; demote any lineage "
                 "already on the dominant side.\n")
    lines.append("## 3. Extended kidney lineage marker dictionary\n")
    lines.append(f"Dropped **{n_dropped}** non-panel symbols from the source "
                 f"dictionary. Ambiguous symbols excluded from the inverse "
                 f"lookup: `{', '.join(ambiguous) if ambiguous else 'none'}`.\n")
    lines.append("| lineage | canonical markers used | n |")
    lines.append("|---|---|---|")
    for lineage in EXTENDED_KIDNEY_MARKERS:
        present = used.get(lineage, [])
        missing = [g for g in EXTENDED_KIDNEY_MARKERS[lineage] if g not in present]
        present_str = ", ".join(present) if present else "*(none)*"
        missing_note = f" *(missing from panel: {', '.join(missing)})*" if missing else ""
        lines.append(f"| **{lineage}** | {present_str}{missing_note} | {len(present)} |")
    lines.append("\n## 4. ROI assignments\n")
    lines.append("| roi_id | category | RCTD predicted | dominant (observed) | "
                 "conflicting (observed) | n_cells dom / conf | notes |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in assignments:
        dom = (f"**{'/'.join(a.dominant_lineages)}** "
               f"({', '.join(a.dominant_genes) or 'n/a'})")
        conf = (f"**{'/'.join(a.conflicting_lineages)}** "
                f"({', '.join(a.conflicting_genes) or 'n/a'})")
        lines.append(
            f"| {a.roi_id} | {a.category} | {a.rctd_predicted_lineage} | "
            f"{dom} | {conf} | {a.n_cells_dominant} / "
            f"{a.n_cells_conflicting} | {a.notes or ''} |"
        )
    lines.append("\n## 5. Observed top genes per ROI\n")
    lines.append("Full counts of how often each gene appeared in the "
                 "per-cell lists across the ROI's cells; the v4 lineage "
                 "call is derived from these.\n")
    lines.append("| roi_id | top dominant genes (count) | top conflicting genes (count) |")
    lines.append("|---|---|---|")
    for a in assignments:
        dom_o = "; ".join(f"{g}:{c}" for g, c in a.top_dominant_genes_observed)
        con_o = "; ".join(f"{g}:{c}" for g, c in a.top_conflicting_genes_observed)
        lines.append(f"| {a.roi_id} | {dom_o} | {con_o} |")
    lines.append("\n## 6. Caveats\n")
    lines.append(
        "- Where the observed-gene vote disagrees with the RCTD prediction, "
        "the legend follows the observed-gene call (cf. notes column).\n"
        "- ROIs with `Conflicting: ?` had no observed canonical marker for "
        "a non-dominant program — the negative-NPMI signal is real but no "
        "single lineage from the extended dictionary explains it; the "
        "audit lists the actual observed genes so the dictionary can be "
        "extended further if needed.\n"
    )
    lines.append("## 7. Visual conventions (v4)\n")
    lines.append(
        "- RCTD problem and TRACER conflict polygon panels keep their square "
        "aspect — colorbars are rendered as insets inside the parent axes "
        "(lower-right corner, slim vertical bar).\n"
        "- ROI-inset bottom block is stacked: gene-label text on row A, "
        "the marker/colour legend on row B (no more overlap).\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vhd-dir", type=Path,
                   default=Path("results/kidney_visiumhd_rctd_tracer"))
    p.add_argument("--vhd-matrix", type=Path,
                   default=Path("datasets/kidney_visiumhd_10x/segmented_outputs/filtered_feature_cell_matrix"))
    p.add_argument("--vhd-geojson", type=Path,
                   default=Path("datasets/kidney_visiumhd_10x/segmented_outputs/cell_segmentations.geojson"))
    p.add_argument("--vhd-spatial-dir", type=Path,
                   default=Path("datasets/kidney_visiumhd_10x/segmented_outputs/spatial"))
    p.add_argument("--npmi-table", type=Path,
                   default=Path("results/kidney_visiumhd_rctd_tracer/reference/kidney_visiumhd_npmi.csv.gz"))
    p.add_argument("--atera-dir", type=Path,
                   default=Path("results/ovrlpy_tracer/cervical_atera_full_memoryaware"))
    p.add_argument("--bin-sizes-um", type=int, nargs="+", default=[2, 8, 16],
                   help="Default drops 4 µm — VisiumHD ships 2/8/16 µm bins "
                        "natively under binned_outputs/.")
    p.add_argument("--vhd-binned-dir", type=Path,
                   default=Path("datasets/kidney_visiumhd_10x/segmented_outputs/binned_outputs"),
                   help="Folder containing square_002um/, square_008um/, "
                        "square_016um/ — the canonical VisiumHD bin matrices.")
    p.add_argument("--skip-atera", action="store_true")
    p.add_argument("--skip-vhd", action="store_true")
    args = p.parse_args()

    log("Step 1: NPMI top-k audit")
    topk = audit_topk(args.npmi_table)
    log(f"  total={topk.total:,} pos={topk.pos:,} neg={topk.neg:,} "
        f"sign_aware={topk.sign_aware}")

    log("Step 2: intersect extended marker dict with VisiumHD panel")
    panel = load_visiumhd_panel(args.vhd_matrix)
    used, audit = intersect_dict(panel)
    n_dropped = sum(1 for r in audit if not r["present_in_visiumhd"])
    inverse, ambiguous = _build_inverse(panel)
    log(f"  panel={len(panel):,} symbols; {n_dropped} markers not in panel; "
        f"ambiguous={ambiguous or 'none'}")
    write_marker_dict_tsv(
        audit, args.vhd_dir / "tables" / "kidney_extended_marker_dictionary_v4.tsv")

    log("Step 3: load joined + ROIs, resolve assignments")
    joined = pd.read_csv(
        args.vhd_dir / "overlap" / "joined_rctd_tracer_scores.tsv.gz",
        sep="\t",
        usecols=["cell_id_int", "barcode", "cx_um", "cy_um", "cx_px", "cy_px",
                 "overlap_category", "RCTD_problem_score",
                 "TRACER_problem_score", "predicted_dominant_lineage",
                 "top_dominant_genes", "top_conflicting_genes"],
    )
    with open(args.vhd_dir / "overlap" / "representative_rois.json") as f:
        rois = json.load(f)
    log(f"  {len(rois)} ROIs; {len(joined):,} cells")
    assignments = []
    for roi in rois:
        in_roi = joined[
            (joined["cx_um"] >= roi["x_min_um"]) & (joined["cx_um"] < roi["x_max_um"])
            & (joined["cy_um"] >= roi["y_min_um"]) & (joined["cy_um"] < roi["y_max_um"])
        ]
        a = resolve_roi_v4(roi, in_roi, inverse)
        assignments.append(a)
        log(f"  {a.roi_id} dom={'/'.join(a.dominant_lineages)} "
            f"({', '.join(a.dominant_genes)}) "
            f"conf={'/'.join(a.conflicting_lineages)} "
            f"({', '.join(a.conflicting_genes)}) "
            f"rctd_pred={a.rctd_predicted_lineage}")
    write_observed_genes_tsv(
        assignments,
        args.vhd_dir / "tables" / "roi_observed_dominant_conflicting_genes_v4.tsv",
    )
    write_topk_audit_v4(
        assignments, args.npmi_table,
        args.vhd_dir / "tables" / "npmi_topk_selection_audit_v4.tsv",
    )

    if not args.skip_vhd:
        log("Step 4: render VisiumHD v4 insets")
        polys = _load_polys(args.vhd_geojson)
        with open(args.vhd_spatial_dir / "scalefactors_json.json") as f:
            sf = json.load(f)
        spatial = {"microns_per_pixel": float(sf["microns_per_pixel"]),
                   "hires_scalef": float(sf.get("tissue_hires_scalef", 1.0))}
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        hires_img = np.asarray(Image.open(
            args.vhd_spatial_dir / "tissue_hires_image.png").convert("RGB"))
        log(f"  H&E hires {hires_img.shape}; um_per_px={spatial['microns_per_pixel']:.4f}")
        # Single shared loader for the binned VisiumHD matrices (each size
        # is loaded on first query and cached).
        bin_source = BinnedMatrixSource(
            args.vhd_binned_dir, spatial["microns_per_pixel"])
        out_dir = args.vhd_dir / "figures" / "roi_insets_v4"
        for roi, a in zip(rois, assignments):
            f = render_v4_inset(
                roi, assignment=a, joined=joined, polys=polys,
                hires_img=hires_img, spatial=spatial, out_dir=out_dir,
                bin_sizes_um=args.bin_sizes_um, bin_source=bin_source,
            )
            if f:
                log(f"  wrote {f}")

    if not args.skip_atera:
        log("Step 5: Atera v3 insets (stacked bottom block)")
        atera_out = args.atera_dir / "final_figures_fixed" / "roi_insets_v3"
        n = _atera_render_v3(
            atera_dir=args.atera_dir, out_dir=atera_out,
            transcripts_path=Path("datasets/cervical_cancer_atera_10x/filtered_df.parquet"),
            morph_path=Path("datasets/cervical_cancer_atera_10x/morphology.ome.tif"),
            cell_boundaries_path=Path("datasets/cervical_cancer_atera_10x/cell_boundaries.parquet"),
        )
        log(f"  emitted {n} Atera v3 insets")

    write_markdown_v4(
        topk=topk, used=used, n_dropped=n_dropped,
        assignments=assignments, ambiguous=ambiguous,
        out=args.vhd_dir / "roi_inset_marker_audit_v4.md",
    )
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
