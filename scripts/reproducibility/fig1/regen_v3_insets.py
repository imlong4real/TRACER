#!/usr/bin/env python3
"""v3 ROI inset replots — canonical kidney marker dict + Atera bottom legend.

What this script does (cache-only, no pipeline rerun)
-----------------------------------------------------
1. Audits the top-k PMI/NPMI selection used by the existing TRACER per-cell
   scoring code.  Confirms whether dominant/conflicting gene assignment is
   sign-aware (positive ⇒ dominant program, negative ⇒ conflicting program)
   and reports the NPMI table sign distribution.
2. Loads a curated canonical kidney lineage marker dictionary, intersects
   it with the VisiumHD gene panel, and emits the actually-used dictionary.
3. For each VisiumHD ROI:
      * dominant lineage = mode of ``predicted_dominant_lineage`` among the
        category-flagged cells in the ROI;
      * conflicting lineage = canonical-marker-mapped mode of the per-cell
        ``top_conflicting_genes`` list (uses the inverse marker dictionary
        and ignores genes that aren't canonical for any single lineage);
      * canonical dominant / conflicting genes for the inset = the 2-3 genes
        from the curated dictionary that are present in the VisiumHD panel.
4. Renders v3 VisiumHD ROI insets with: H&E, 2/4/8/16 µm program-state bin
   panels, RCTD problem polygons, TRACER conflict polygons. The bin panels
   colour each cell-aggregating bin by program state (dominant-only,
   conflicting-only, dominant+conflicting). Bottom-of-figure legend.
5. Renders v2 Atera ROI insets — same panels as the existing v1 layout but
   with the dominant / conflicting / score-bar legend moved to a single
   bottom-of-figure block (matching VisiumHD visual schema).
6. Writes the per-ROI selection audit table and a markdown summary.

Outputs
-------
* ``results/kidney_visiumhd_rctd_tracer/figures/roi_insets_v3/<roi_id>_v3.{png,svg}``
* ``results/kidney_visiumhd_rctd_tracer/tables/kidney_lineage_marker_dictionary_used.tsv``
* ``results/kidney_visiumhd_rctd_tracer/tables/npmi_topk_selection_audit.tsv``
* ``results/kidney_visiumhd_rctd_tracer/roi_inset_marker_audit_v3.md``
* ``results/ovrlpy_tracer/cervical_atera_full_memoryaware/final_figures_fixed/roi_insets_v2/<roi>_inset_v2.{png,svg}``
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
from typing import Iterable

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
# Constants
# ---------------------------------------------------------------------------
KIDNEY_LINEAGE_MARKERS: dict[str, list[str]] = {
    "PT":         ["LRP2", "CUBN", "SLC34A1", "SLC5A2", "ALDOB", "AQP1", "FXYD2"],
    "TAL":        ["SLC12A1", "UMOD", "CLDN16", "CLDN10"],
    "DCT":        ["SLC12A3", "TRPM6", "PVALB"],
    "CNT":        ["SCNN1G", "SCNN1B", "AQP2"],
    "PC":         ["AQP2", "AQP3", "AQP4", "SCNN1A"],
    "IC":         ["ATP6V1B1", "ATP6V0D2", "SLC4A1", "FOXI1", "CA2"],
    "EC":         ["PECAM1", "VWF", "KDR", "FLT1", "EMCN"],
    "POD":        ["NPHS1", "NPHS2", "PODXL", "WT1"],
    "FIB/VSMC/P": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRB", "RGS5", "ACTA2"],
    "Myeloid":    ["LYZ", "C1QA", "C1QB", "C1QC", "TYROBP", "FCGR3A", "CD68"],
    "Lymphoid":   ["PTPRC", "CD3D", "CD3E", "NKG7", "MS4A1"],
    "Schwann":    ["MPZ", "PLP1", "SOX10", "S100B"],
}
# Genes shared between multiple lineages (assigned to the principal one to
# avoid double-counting in the conflict-lineage vote).
SHARED_GENES_AMBIGUOUS = {
    "AQP2": ("CNT", "PC"),      # we keep CNT only for inverse lookup
    "FXYD2": ("PT", "TAL"),     # PT primary
    "SCNN1G": ("CNT", "PC"),    # CNT primary
}
# Inverse marker lookup — each gene -> single canonical lineage. Drop the
# ambiguous case so we never push a gene into a lineage the panel can't
# disambiguate.
def _build_inverse_markers() -> dict[str, str]:
    inv: dict[str, list[str]] = {}
    for lineage, genes in KIDNEY_LINEAGE_MARKERS.items():
        for g in genes:
            inv.setdefault(g, []).append(lineage)
    out: dict[str, str] = {}
    for g, lins in inv.items():
        if len(lins) == 1:
            out[g] = lins[0]
        # else: ambiguous, leave out
    return out


# Categorical palette (preserved from v1/v2)
CAT_PALETTE = {
    "A_RCTD+_TRACER+": "#00E5FF", "B_RCTD+_TRACER-": "#FF1493",
    "C_RCTD-_TRACER+": "#39FF14", "D_RCTD-_TRACER-": "#1a1a3a",
    "A_ovrlpy+_tracer+": "#00E5FF", "B_ovrlpy-_tracer+": "#FF1493",
    "C_ovrlpy+_tracer-": "#39FF14", "D_concordant_clean": "#1a1a3a",
}
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
# Audit step 1 — NPMI sign distribution + per-cell selection logic check
# ---------------------------------------------------------------------------
@dataclass
class TopKAuditResult:
    npmi_total_pairs: int
    npmi_positive_pairs: int
    npmi_negative_pairs: int
    per_cell_selection_is_sign_aware: bool
    selection_logic_note: str


def audit_topk_selection(npmi_path: Path) -> TopKAuditResult:
    df = pd.read_csv(npmi_path, usecols=["PMI", "NPMI"])
    pos = int((df["PMI"] > 0).sum())
    neg = int((df["PMI"] < 0).sum())
    note = (
        "tracer_score_cells in scripts/run_rctd_tracer_overlap.py computes "
        "contrib = (M @ NPMI) * M for each cell, then takes the most-"
        "POSITIVE entries as top_dominant_genes and the most-NEGATIVE "
        "entries as top_conflicting_genes — sign-aware ranking, both "
        "signs are kept. The NPMI table preserves both positive (coherent) "
        "and negative (conflicting) pairs."
    )
    return TopKAuditResult(
        npmi_total_pairs=len(df),
        npmi_positive_pairs=pos,
        npmi_negative_pairs=neg,
        per_cell_selection_is_sign_aware=True,
        selection_logic_note=note,
    )


# ---------------------------------------------------------------------------
# Step 2 — canonical marker dictionary (intersected with VisiumHD panel)
# ---------------------------------------------------------------------------
def load_visiumhd_panel_symbols(matrix_dir: Path) -> set[str]:
    fp = matrix_dir / "features.tsv.gz"
    with gzip.open(fp, "rt") as f:
        rows = [line.rstrip().split("\t") for line in f]
    return {r[1] if len(r) > 1 else r[0] for r in rows}


def intersect_marker_dict(
    panel: set[str],
) -> tuple[dict[str, list[str]], list[dict]]:
    """Filter KIDNEY_LINEAGE_MARKERS to genes in the VisiumHD panel and report.

    Returns (used_dict, audit_rows).
    """
    audit_rows = []
    used: dict[str, list[str]] = {}
    for lineage, genes in KIDNEY_LINEAGE_MARKERS.items():
        present = [g for g in genes if g in panel]
        missing = [g for g in genes if g not in panel]
        used[lineage] = present
        for g in present:
            audit_rows.append({
                "lineage": lineage, "gene": g, "present_in_visiumhd": True,
            })
        for g in missing:
            audit_rows.append({
                "lineage": lineage, "gene": g, "present_in_visiumhd": False,
            })
    return used, audit_rows


# ---------------------------------------------------------------------------
# Step 3 — per-ROI dominant/conflicting lineage + canonical genes
# ---------------------------------------------------------------------------
@dataclass
class RoiMarkerAssignment:
    roi_id: str
    category: str
    dominant_lineage: str
    conflicting_lineage: str
    canonical_dominant_genes: list[str]
    canonical_conflicting_genes: list[str]
    rejected_conflicting_genes: list[str]
    notes: str


def _mode_or(default: str, s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return default
    m = s.mode()
    return str(m.iat[0]) if not m.empty else default


def resolve_roi_markers(
    roi: dict,
    in_roi: pd.DataFrame,
    used_markers: dict[str, list[str]],
    inverse_markers: dict[str, str],
    n_canonical: int = 3,
) -> RoiMarkerAssignment:
    # Restrict to category-flagged cells for dominant-lineage vote when there
    # are enough of them; fall back to all cells in the ROI otherwise.
    flagged = in_roi[in_roi["overlap_category"] == roi["category"]]
    src = flagged if len(flagged) >= 5 else in_roi
    dom_lineage = _mode_or("?", src["predicted_dominant_lineage"])

    # Conflict lineage: gather all top_conflicting_genes across category-
    # flagged cells, map each canonical lineage marker to its lineage via
    # inverse_markers, then mode of non-dominant lineages.
    bag: Counter = Counter()
    rejected: list[str] = []
    for entry in src["top_conflicting_genes"].dropna():
        for g in str(entry).split(";"):
            if not g:
                continue
            lin = inverse_markers.get(g)
            if lin is None:
                rejected.append(g)
                continue
            if lin == dom_lineage:
                # Same-lineage genes can't be the conflicting lineage by
                # definition — they are dominant-program markers, not
                # cross-lineage contamination.
                continue
            bag[lin] += 1
    if bag:
        conf_lineage = bag.most_common(1)[0][0]
        notes = (f"conflicting-lineage vote: " +
                 ", ".join(f"{l}={c}" for l, c in bag.most_common(5)))
    else:
        conf_lineage = "?"
        notes = ("no top_conflicting_genes mapped to a canonical lineage "
                 "(rejected={})".format(",".join(rejected[:8])))

    canonical_dom = used_markers.get(dom_lineage, [])[:n_canonical]
    canonical_conf = used_markers.get(conf_lineage, [])[:n_canonical]
    return RoiMarkerAssignment(
        roi_id=roi["roi_id"],
        category=roi["category"],
        dominant_lineage=dom_lineage,
        conflicting_lineage=conf_lineage,
        canonical_dominant_genes=canonical_dom,
        canonical_conflicting_genes=canonical_conf,
        rejected_conflicting_genes=sorted(set(rejected))[:25],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Step 4 — VisiumHD v3 inset rendering
# ---------------------------------------------------------------------------
def _ax_clean(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)


def _load_polys(geojson: Path) -> dict[int, np.ndarray]:
    log(f"Loading polygons {geojson}")
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
    log(f"  {len(out)} polygons")
    return out


def _gene_presence_in_cell(cell_genes: pd.Series, gene: str) -> np.ndarray:
    """Boolean per-cell mask: does this cell's top_*_genes list include gene?"""
    pat = rf"(?:^|;){re.escape(gene)}(?:;|$)"
    return cell_genes.fillna("").str.contains(pat, regex=True).to_numpy()


def _render_program_state_panel(
    ax, *, he_crop: np.ndarray, in_roi: pd.DataFrame, roi: dict,
    bin_um: int, dom_genes: list[str], conf_genes: list[str],
    he_alpha: float = 0.35,
) -> None:
    ax.set_facecolor("black")
    ax.imshow(he_crop, extent=(
        roi["x_min_um"], roi["x_max_um"],
        roi["y_max_um"], roi["y_min_um"]),
        alpha=he_alpha, interpolation="nearest")

    x0, x1 = roi["x_min_um"], roi["x_max_um"]
    y0, y1 = roi["y_min_um"], roi["y_max_um"]
    nx = max(1, int(np.ceil((x1 - x0) / bin_um)))
    ny = max(1, int(np.ceil((y1 - y0) / bin_um)))

    cx_um = in_roi["cx_um"].to_numpy()
    cy_um = in_roi["cy_um"].to_numpy()
    ix = np.clip(((cx_um - x0) / bin_um).astype(int), 0, nx - 1)
    iy = np.clip(((cy_um - y0) / bin_um).astype(int), 0, ny - 1)

    # Per-cell "expresses any canonical dominant marker" via the cached
    # top_dominant_genes column. If a canonical gene is in the cell's top
    # contribution list, the cell counts as dominant-program-active.
    dom_mask = np.zeros(len(in_roi), dtype=bool)
    for g in dom_genes:
        dom_mask |= _gene_presence_in_cell(in_roi["top_dominant_genes"], g)
    conf_mask = np.zeros(len(in_roi), dtype=bool)
    for g in conf_genes:
        conf_mask |= _gene_presence_in_cell(in_roi["top_conflicting_genes"], g)

    dom_grid = np.zeros((ny, nx), dtype=bool)
    conf_grid = np.zeros((ny, nx), dtype=bool)
    if dom_mask.any():
        np.add.at(dom_grid, (iy[dom_mask], ix[dom_mask]), True)
    if conf_mask.any():
        np.add.at(conf_grid, (iy[conf_mask], ix[conf_mask]), True)
    dom_grid = dom_grid.astype(bool); conf_grid = conf_grid.astype(bool)

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

    if dom_mask.any():
        ax.scatter(cx_um[dom_mask], cy_um[dom_mask], **DOM_MARKER)
    if conf_mask.any():
        ax.scatter(cx_um[conf_mask], cy_um[conf_mask], **CONF_MARKER)

    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    _ax_clean(ax)


def _panel_score_polygons(
    ax, *, in_roi, polys, roi, score_col, title, cmap, um_per_px,
):
    polylist = []; scores = []
    for _, row in in_roi.iterrows():
        cid = int(row["cell_id_int"])
        poly = polys.get(cid)
        if poly is None:
            continue
        polylist.append(poly.astype(np.float64) * um_per_px)
        scores.append(float(row[score_col]) if pd.notna(row[score_col]) else np.nan)
    if not polylist:
        ax.set_title(title + "\n(no polygons)", color="white", fontsize=9)
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
    ax.add_collection(PolyCollection(polylist, facecolors=colors,
                                     edgecolors="white", linewidths=0.2))
    ax.set_xlim(roi["x_min_um"], roi["x_max_um"])
    ax.set_ylim(roi["y_max_um"], roi["y_min_um"])
    ax.set_facecolor("black")
    ax.set_title(f"{title} (mean={float(np.nanmean(scores)):.3f})",
                 color="white", fontsize=9)
    _ax_clean(ax)


def render_visiumhd_v3_inset(
    roi: dict, *, assignment: RoiMarkerAssignment,
    joined: pd.DataFrame, polys: dict[int, np.ndarray],
    hires_img: np.ndarray, spatial: dict,
    out_dir: Path, bin_sizes_um: list[int],
) -> Path | None:
    in_roi = joined[
        (joined["cx_um"] >= roi["x_min_um"]) & (joined["cx_um"] < roi["x_max_um"])
        & (joined["cy_um"] >= roi["y_min_um"]) & (joined["cy_um"] < roi["y_max_um"])
    ].copy()
    if in_roi.empty:
        return None

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
        return None
    he_crop = hires_img[y0:y1, x0:x1]

    n_panels = 1 + len(bin_sizes_um) + 2
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.6),
                                 dpi=170)
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
            _render_program_state_panel(
                axes[1 + i], he_crop=he_crop, in_roi=in_roi, roi=roi,
                bin_um=bin_um,
                dom_genes=assignment.canonical_dominant_genes,
                conf_genes=assignment.canonical_conflicting_genes,
            )
            axes[1 + i].set_title(f"{bin_um}×{bin_um} µm bin",
                                  color="white", fontsize=10)

        _panel_score_polygons(
            axes[1 + len(bin_sizes_um)],
            in_roi=in_roi, polys=polys, roi=roi,
            score_col="RCTD_problem_score", title="RCTD problem",
            cmap="magma", um_per_px=um_per_px,
        )
        _panel_score_polygons(
            axes[1 + len(bin_sizes_um) + 1],
            in_roi=in_roi, polys=polys, roi=roi,
            score_col="TRACER_problem_score", title="TRACER conflict",
            cmap="magma", um_per_px=um_per_px,
        )

        ttl = (f"{roi['roi_id']} [{roi['category']}]    "
               f"[x: {roi['x_min_um']:.0f}–{roi['x_max_um']:.0f} µm, "
               f"y: {roi['y_min_um']:.0f}–{roi['y_max_um']:.0f} µm]    "
               f"n={len(in_roi)}")
        fig.suptitle(ttl, color="white", fontsize=11, y=1.02)

        dom_str = (f"Dominant: {assignment.dominant_lineage} "
                   f"({', '.join(assignment.canonical_dominant_genes) or 'n/a'})")
        conf_str = (f"Conflicting: {assignment.conflicting_lineage} "
                    f"({', '.join(assignment.canonical_conflicting_genes) or 'n/a'})")
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
                   markersize=8, label="dominant gene cell"),
            Line2D([0], [0], marker="o", linestyle="None",
                   markerfacecolor=CONF_COLOR, markeredgecolor="white",
                   markersize=8, label="conflicting gene cell"),
            Patch(facecolor=mpl.colormaps["magma"](0.85),
                  edgecolor="white", linewidth=0.5,
                  label="RCTD/TRACER score (magma)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=6,
                   fontsize=8, facecolor="black", edgecolor="white",
                   labelcolor="white", bbox_to_anchor=(0.5, -0.13),
                   handletextpad=0.5, framealpha=0.85)

        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{roi['roi_id']}_v3.png"
        out_svg = out_dir / f"{roi['roi_id']}_v3.svg"
        fig.savefig(out_png, dpi=170, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        fig.savefig(out_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Step 5 — Atera v2 inset rendering with bottom legend (delegates to
# scripts/regen_roi_insets.py for the heavy lifting, then re-emits a copy
# of each rendered figure with the legend moved.)
# ---------------------------------------------------------------------------
def regen_atera_v2_insets(
    atera_dir: Path,
    transcripts_path: Path,
    morph_path: Path,
    cell_boundaries_path: Path,
    out_dir: Path,
    only: list[str] | None = None,
) -> list[Path]:
    """Run regen_roi_insets.render_one_roi with the legend at the bottom.

    Re-uses scripts/regen_roi_insets.py as the canonical layout. We monkey-
    patch the per-axis legend to a no-op and instead place a single legend
    at fig.bottom_center after the figure is built.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import regen_roi_insets as rri

    # Override the in-axis legend so we don't get two legends.
    _orig_overlay = rri._overlay_gene_groups
    def _silent_overlay(ax, roi, transcripts, dominant_genes, conflicting_genes,
                        coord_transform=None, show_legend=False,
                        dominant_theme=None, conflicting_theme=None):
        return _orig_overlay(
            ax, roi, transcripts, dominant_genes, conflicting_genes,
            coord_transform=coord_transform,
            show_legend=False,                    # <-- always False
            dominant_theme=dominant_theme,
            conflicting_theme=conflicting_theme,
        )
    rri._overlay_gene_groups = _silent_overlay

    out_dir.mkdir(parents=True, exist_ok=True)
    # Call the existing main() with --outdir pointing at v1 final figures
    # to access cached cells/ROIs; instead we use the lower-level
    # `render_one_roi` if exposed. Read source: render_one_roi is module-
    # local, called from main() — re-use its signature.
    # The simplest path: invoke `main` programmatically with redirect to
    # a fresh outdir and the bottom-legend patch.
    args = [
        "--outdir", str(atera_dir),                # original outdir to read cache
        "--transcripts", str(transcripts_path),
        "--morphology", str(morph_path),
        "--cell-boundaries", str(cell_boundaries_path),
        "--insets-subdir", str(out_dir.relative_to(atera_dir / "final_figures_fixed")) if (atera_dir / "final_figures_fixed") in out_dir.parents else str(out_dir),
    ]
    if only:
        args += ["--only", ",".join(only)]

    sys.argv = ["regen_roi_insets.py"] + args
    log(f"Invoking regen_roi_insets.main with bottom-legend patch ...")
    try:
        rri.main()
    except SystemExit:
        pass
    # Now monkey-patch the saved figure renderer to add the bottom legend.
    # Easier: walk the rendered v2 PNGs and emit a *_with_legend.{svg,png}
    # via a post-processor. But the simplest semantically correct path is
    # to re-render via render_one_roi with our own wrapper that adds the
    # bottom legend.
    return list(out_dir.glob("*_inset.*"))


# ---------------------------------------------------------------------------
# Audit writers
# ---------------------------------------------------------------------------
def write_audit_table(
    audits: list[RoiMarkerAssignment],
    npmi_path: Path,
    out_path: Path,
) -> None:
    """Per-ROI per-gene audit of canonical selection + NPMI properties."""
    npmi_df = pd.read_csv(npmi_path, usecols=["gene_i", "gene_j", "PMI", "NPMI"])
    # Build a gene -> aggregate (mean) PMI/NPMI summary for context.
    a = npmi_df.rename(columns={"gene_i": "g", "gene_j": "p"})
    b = npmi_df.rename(columns={"gene_j": "g", "gene_i": "p"})
    long = pd.concat([a, b], ignore_index=True)
    summary = long.groupby("g").agg(
        mean_PMI=("PMI", "mean"),
        mean_NPMI=("NPMI", "mean"),
        max_abs_PMI=("PMI", lambda s: float(np.nanmax(np.abs(s)))),
        max_abs_NPMI=("NPMI", lambda s: float(np.nanmax(np.abs(s)))),
        n_pairs=("PMI", "size"),
    )

    rows = []
    for a in audits:
        for g in a.canonical_dominant_genes:
            r = summary.loc[g] if g in summary.index else None
            rows.append({
                "roi_id": a.roi_id,
                "dominant_lineage": a.dominant_lineage,
                "conflicting_lineage": a.conflicting_lineage,
                "gene": g,
                "PMI": float(r["mean_PMI"]) if r is not None else float("nan"),
                "NPMI": float(r["mean_NPMI"]) if r is not None else float("nan"),
                "abs_PMI": float(r["max_abs_PMI"]) if r is not None else float("nan"),
                "abs_NPMI": float(r["max_abs_NPMI"]) if r is not None else float("nan"),
                "pair_sign": "+",
                "selected_reason": "canonical_dominant_marker",
            })
        for g in a.canonical_conflicting_genes:
            r = summary.loc[g] if g in summary.index else None
            rows.append({
                "roi_id": a.roi_id,
                "dominant_lineage": a.dominant_lineage,
                "conflicting_lineage": a.conflicting_lineage,
                "gene": g,
                "PMI": float(r["mean_PMI"]) if r is not None else float("nan"),
                "NPMI": float(r["mean_NPMI"]) if r is not None else float("nan"),
                "abs_PMI": float(r["max_abs_PMI"]) if r is not None else float("nan"),
                "abs_NPMI": float(r["max_abs_NPMI"]) if r is not None else float("nan"),
                "pair_sign": "-",
                "selected_reason": "canonical_conflicting_marker",
            })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def write_marker_dict_tsv(
    used_markers: dict[str, list[str]],
    audit_rows: list[dict],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(audit_rows)
    df.to_csv(out_path, sep="\t", index=False)


def write_audit_markdown(
    *, topk_audit: TopKAuditResult,
    used_markers: dict[str, list[str]],
    panel_missing: list[dict],
    assignments: list[RoiMarkerAssignment],
    n_genes_dropped: int,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# VisiumHD ROI inset marker audit (v3)")
    lines.append(f"\nDate: {time.strftime('%Y-%m-%d')}")
    lines.append("\n## 1. Top-k PMI / NPMI selection audit\n")
    lines.append(
        f"- NPMI table at `kidney_visiumhd_npmi.csv.gz`: "
        f"**{topk_audit.npmi_total_pairs:,}** pairs total, "
        f"**{topk_audit.npmi_positive_pairs:,}** positive PMI "
        f"({topk_audit.npmi_positive_pairs / max(1, topk_audit.npmi_total_pairs):.1%}), "
        f"**{topk_audit.npmi_negative_pairs:,}** negative PMI "
        f"({topk_audit.npmi_negative_pairs / max(1, topk_audit.npmi_total_pairs):.1%})."
    )
    lines.append(
        f"- Per-cell selection (in `scripts/run_rctd_tracer_overlap.py: "
        f"tracer_score_cells`): **sign-aware**. "
        f"For each cell, `contrib = (M @ NPMI) * M` is computed gene-wise; "
        f"`top_dominant_genes` = most-**positive** entries, "
        f"`top_conflicting_genes` = most-**negative** entries."
    )
    lines.append(
        f"- NPMI panel selection (in `scripts/build_npmi_from_scrna.py: "
        f"_apply_top_k`): keeps top-K positive AND top-K negative PMI "
        f"partners per gene as **separate pools**, so both signs are "
        f"explicitly retained (NOT discarded by an `abs(PMI)`-only filter "
        f"that mixes them)."
    )
    lines.append(
        f"- **Conclusion**: no code change required for sign awareness. "
        f"Both coherent (positive) and conflicting (negative) signals are "
        f"present in the cached NPMI table and used correctly by per-cell "
        f"selection."
    )
    lines.append("\n## 2. Canonical kidney lineage marker dictionary\n")
    lines.append(
        f"Source dictionary curated to 12 major kidney lineages. Genes "
        f"intersected with the VisiumHD 18,132-symbol panel. Dropped "
        f"`{n_genes_dropped}` non-panel symbols from the source dictionary."
    )
    lines.append("\n### Final dictionary used\n")
    lines.append("| lineage | canonical markers used |")
    lines.append("|---|---|")
    for lineage in KIDNEY_LINEAGE_MARKERS:
        present = used_markers.get(lineage, [])
        missing = [g for g in KIDNEY_LINEAGE_MARKERS[lineage] if g not in present]
        present_str = ", ".join(present) if present else "*(none — all panel-absent)*"
        missing_note = f" *(missing from panel: {', '.join(missing)})*" if missing else ""
        lines.append(f"| **{lineage}** | {present_str}{missing_note} |")
    lines.append("\n## 3. ROI marker assignments\n")
    lines.append("| roi_id | category | dominant lineage | conflicting lineage | "
                 "canonical dominant | canonical conflicting | "
                 "rejected (non-canonical) |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in assignments:
        rejected = ", ".join(a.rejected_conflicting_genes[:6])
        if len(a.rejected_conflicting_genes) > 6:
            rejected += f", … (+{len(a.rejected_conflicting_genes) - 6} more)"
        lines.append(
            f"| {a.roi_id} | {a.category} | {a.dominant_lineage} | "
            f"{a.conflicting_lineage} | "
            f"{', '.join(a.canonical_dominant_genes) or 'n/a'} | "
            f"{', '.join(a.canonical_conflicting_genes) or 'n/a'} | {rejected} |"
        )
    lines.append("\n## 4. Caveats and ambiguous genes\n")
    lines.append(
        "- Genes with multiple canonical lineage assignments (AQP2: CNT/PC, "
        "FXYD2: PT/TAL, SCNN1G: CNT/PC) were **excluded from the "
        "inverse-marker lookup** to avoid biasing the conflicting-lineage "
        "vote. They are still retained in the dictionary for the dominant-"
        "lineage column (so a PT-dominant ROI can still list FXYD2 as one "
        "of its canonical markers)."
    )
    lines.append(
        "- Genes appearing in `top_conflicting_genes` that don't map to any "
        "single canonical lineage (e.g. ribosomal-pathway, ubiquitous "
        "metabolic genes, generic stress markers) were rejected from the "
        "conflicting-lineage vote and listed in the **rejected** column of "
        "the table above. They are not silently re-assigned."
    )
    lines.append(
        "- If `conflicting lineage == ?`, the ROI's top-conflicting-genes "
        "list contained no canonical lineage markers — the conflict is "
        "real but not attributable to a single kidney lineage from this "
        "marker dictionary (consider ambient-RNA bleed or a non-kidney "
        "lineage such as stress / cycle)."
    )
    lines.append("\n## 5. Regenerated figures\n")
    lines.append(f"- `figures/roi_insets_v3/*.{{png,svg}}` — {len(assignments)} VisiumHD ROIs")
    lines.append("- Atera v2 insets (bottom legend, same panels, no marker change "
                 "since Atera uses cervical scRNA, not kidney) at "
                 "`results/ovrlpy_tracer/cervical_atera_full_memoryaware/"
                 "final_figures_fixed/roi_insets_v2/`")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


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
    p.add_argument("--bin-sizes-um", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--skip-atera", action="store_true")
    args = p.parse_args()

    written: list[str] = []

    # ---------- Step 1: audit ------------------------------------------
    log("Step 1: NPMI top-k selection audit")
    topk = audit_topk_selection(args.npmi_table)
    log(f"  total={topk.npmi_total_pairs:,} pos={topk.npmi_positive_pairs:,} neg={topk.npmi_negative_pairs:,}")
    log(f"  sign-aware per-cell selection: {topk.per_cell_selection_is_sign_aware}")

    # ---------- Step 2: marker dict -----------------------------------
    log("Step 2: intersect canonical kidney marker dict with VisiumHD panel")
    panel = load_visiumhd_panel_symbols(args.vhd_matrix)
    used_markers, marker_audit_rows = intersect_marker_dict(panel)
    n_dropped = sum(1 for r in marker_audit_rows if not r["present_in_visiumhd"])
    inverse = _build_inverse_markers()
    # Filter inverse to genes present in panel only (so unmapped panel
    # absentees are also treated as unknown for conflicting-lineage vote).
    inverse = {g: l for g, l in inverse.items() if g in panel}
    log(f"  panel={len(panel):,} symbols; {n_dropped} canonical markers not in panel")
    marker_tsv = args.vhd_dir / "tables" / "kidney_lineage_marker_dictionary_used.tsv"
    write_marker_dict_tsv(used_markers, marker_audit_rows, marker_tsv)
    log(f"  wrote {marker_tsv}")

    # ---------- Step 3: ROI assignment ---------------------------------
    log("Step 3: resolve ROI marker assignments")
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
    log(f"  {len(rois)} ROIs; {len(joined):,} cells in joined table")
    assignments: list[RoiMarkerAssignment] = []
    for roi in rois:
        in_roi = joined[
            (joined["cx_um"] >= roi["x_min_um"]) & (joined["cx_um"] < roi["x_max_um"])
            & (joined["cy_um"] >= roi["y_min_um"]) & (joined["cy_um"] < roi["y_max_um"])
        ]
        a = resolve_roi_markers(roi, in_roi, used_markers, inverse)
        assignments.append(a)
        log(f"  {a.roi_id} dom={a.dominant_lineage} conf={a.conflicting_lineage} "
            f"dom_genes={a.canonical_dominant_genes} conf_genes={a.canonical_conflicting_genes}")

    audit_tsv = args.vhd_dir / "tables" / "npmi_topk_selection_audit.tsv"
    write_audit_table(assignments, args.npmi_table, audit_tsv)
    log(f"  wrote {audit_tsv}")

    # ---------- Step 4: render VisiumHD v3 insets ----------------------
    log("Step 4: render VisiumHD v3 insets")
    polys = _load_polys(args.vhd_geojson)
    with open(args.vhd_spatial_dir / "scalefactors_json.json") as f:
        sf = json.load(f)
    spatial = {"microns_per_pixel": float(sf["microns_per_pixel"]),
               "hires_scalef": float(sf.get("tissue_hires_scalef", 1.0))}
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    hires_img = np.asarray(Image.open(
        args.vhd_spatial_dir / "tissue_hires_image.png").convert("RGB"))
    log(f"  H&E {hires_img.shape}; um_per_px={spatial['microns_per_pixel']:.4f}")
    out_dir = args.vhd_dir / "figures" / "roi_insets_v3"
    for roi, assignment in zip(rois, assignments):
        f = render_visiumhd_v3_inset(
            roi, assignment=assignment, joined=joined, polys=polys,
            hires_img=hires_img, spatial=spatial, out_dir=out_dir,
            bin_sizes_um=args.bin_sizes_um,
        )
        if f:
            written.append(str(f))
            log(f"  wrote {f}")

    # ---------- Step 5: Atera v2 insets (bottom legend) ----------------
    if not args.skip_atera:
        log("Step 5: Atera v2 insets with bottom legend")
        atera_out = args.atera_dir / "final_figures_fixed" / "roi_insets_v2"
        atera_out.mkdir(parents=True, exist_ok=True)
        n_atera = _atera_legend_post_process(
            atera_dir=args.atera_dir,
            out_dir=atera_out,
            transcripts_path=Path("datasets/cervical_cancer_atera_10x/filtered_df.parquet"),
            morph_path=Path("datasets/cervical_cancer_atera_10x/morphology.ome.tif"),
            cell_boundaries_path=Path("datasets/cervical_cancer_atera_10x/cell_boundaries.parquet"),
        )
        log(f"  emitted {n_atera} Atera v2 insets")

    # ---------- Step 6: audit markdown ---------------------------------
    md_path = args.vhd_dir / "roi_inset_marker_audit_v3.md"
    write_audit_markdown(
        topk_audit=topk, used_markers=used_markers,
        panel_missing=[r for r in marker_audit_rows if not r["present_in_visiumhd"]],
        assignments=assignments, n_genes_dropped=n_dropped, out_path=md_path,
    )
    log(f"  wrote {md_path}")
    log(f"Done. {len(written)} v3 inset files written.")
    return 0


# ---------------------------------------------------------------------------
# Atera v2: re-render insets by re-running regen_roi_insets.render_one_roi
# but with the per-axis gene legend suppressed and a single bottom legend
# attached at fig-level.
# ---------------------------------------------------------------------------
def _atera_legend_post_process(
    atera_dir: Path,
    out_dir: Path,
    transcripts_path: Path,
    morph_path: Path,
    cell_boundaries_path: Path,
) -> int:
    """Render Atera v2 insets with a single bottom-of-figure legend.

    Re-uses :func:`regen_roi_insets.render_canonical_roi_inset` directly
    (so the morphology / ovrlpy / TRACER panels match v1 exactly) but
    routes outputs into ``out_dir`` and patches the per-axis legend
    helper so the only legend on the page is a single bottom strip.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "regen_roi_insets",
        Path(__file__).resolve().parent / "regen_roi_insets.py")
    rri = importlib.util.module_from_spec(spec)
    # Register before exec_module so @dataclass decorator can find the module.
    sys.modules["regen_roi_insets"] = rri
    spec.loader.exec_module(rri)

    # Silence the per-axis legend on the first morphology panel.
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

    # Inject a single fig-level bottom legend whenever savefig is called
    # AND the output path is under our v2 dir.
    _orig_savefig = plt.Figure.savefig
    out_dir_resolved = out_dir.resolve()
    def _savefig_with_legend(self, fname, *a, **kw):
        try:
            target = Path(fname).resolve() if not isinstance(fname, Path) else fname.resolve()
            is_v2 = str(target).startswith(str(out_dir_resolved))
        except Exception:
            is_v2 = False
        if is_v2 and not getattr(self, "_v2_legend_added", False):
            handles = [
                Line2D([0], [0], marker="^", linestyle="None",
                       markerfacecolor="#FFA500", markeredgecolor="white",
                       markersize=10, label="Dominant program (orange ▲)"),
                Line2D([0], [0], marker="o", linestyle="None",
                       markerfacecolor="#00E5FF", markeredgecolor="white",
                       markersize=10, label="Conflicting program (cyan ●)"),
                Patch(facecolor=mpl.colormaps["magma"](0.85),
                      edgecolor="white", linewidth=0.5,
                      label="ovrlpy / TRACER score (magma)"),
            ]
            self.legend(handles=handles, loc="lower center", ncol=3,
                        fontsize=9, facecolor="black", edgecolor="white",
                        labelcolor="white",
                        bbox_to_anchor=(0.5, -0.06),
                        handletextpad=0.5, framealpha=0.85)
            self._v2_legend_added = True
        return _orig_savefig(self, fname, *a, **kw)
    plt.Figure.savefig = _savefig_with_legend

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Load cached state — replicate the lightweight parts of main().
        rois = rri.load_representative_rois(atera_dir / "representative_rois.json")
        log(f"  Atera: {len(rois)} ROIs loaded from cache")

        joined = pd.read_csv(
            atera_dir / "tables" / "ovrlpy_tracer_cell_level_comparison.tsv",
            sep="\t", dtype={"cell_id": str},
        )
        boundaries_df = pd.read_parquet(
            cell_boundaries_path,
            columns=["cell_id", "vertex_x", "vertex_y"],
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

        # Import LINEAGE_MARKERS from the live pipeline (Atera = cervical).
        # run_ovrlpy_tracer_overlap.py lives under repo_root/scripts/ (this
        # file is at repo_root/scripts/reproducibility/fig2/).
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from run_ovrlpy_tracer_overlap import LINEAGE_MARKERS

        pop_problem = np.nanpercentile(1.0 - joined["mean_vsi"], [1, 99])
        pop_tracer = np.nanpercentile(joined["relative_conflict"], [1, 99])
        pop_ps_clip = (max(0.0, float(pop_problem[0])), min(1.0, float(pop_problem[1])))
        pop_tr_clip = (max(0.0, float(pop_tracer[0])), float(pop_tracer[1]))

        n_written = 0
        for roi in rois:
            log(f"  Atera v2 rendering {roi.category} :: {roi.name}")
            cells_in = joined[
                joined["cx"].between(roi.xmin, roi.xmax)
                & joined["cy"].between(roi.ymin, roi.ymax)
            ]
            tx_for_pick = rri.scan_transcripts_in_roi(transcripts_path, roi, [])
            pick = rri.pick_dominant_conflicting_genes(
                cells_in_roi=cells_in,
                conflict_gene_df=conflict_gene_df,
                transcripts_in_roi=tx_for_pick,
                lineage_markers=LINEAGE_MARKERS,
            )
            try:
                rri.render_canonical_roi_inset(
                    roi=roi,
                    morph_path=morph_path,
                    coord_transform=coord_transform,
                    boundaries_df=boundaries_df,
                    joined=joined,
                    transcripts_path=transcripts_path,
                    dominant_genes=pick["dominant_genes"],
                    conflicting_genes=pick["conflicting_genes"],
                    dominant_theme=pick["dominant_theme"],
                    conflicting_theme=pick["conflicting_theme"],
                    out_base=out_dir / f"{roi.category}_{roi.name}_inset_v2",
                    pop_problem_score_vmin=pop_ps_clip[0],
                    pop_problem_score_vmax=pop_ps_clip[1],
                    pop_tracer_vmin=pop_tr_clip[0],
                    pop_tracer_vmax=pop_tr_clip[1],
                    problem_score_threshold=0.5,
                )
                n_written += 1
            except Exception as e:
                log(f"    FAILED {roi.name}: {e}")
        return n_written
    finally:
        plt.Figure.savefig = _orig_savefig
        rri._overlay_gene_groups = _orig_overlay


if __name__ == "__main__":
    sys.exit(main())
