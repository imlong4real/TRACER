#!/usr/bin/env python
"""Whole-section spatial cell-type map of the TRACER-refined PDAC sample.

One scatter point per TRACER cell at its native centroid, coloured by the
label-transfer cell type, on a black background.  Bounding boxes mark the ROIs
used in the z-stack reconstruction figure and the 3D Open3D ROI figure.

Outputs: outputs/pdac_spatial_celltype_map.{png,svg}

Run:
    /Users/lyuan13/anaconda3/envs/spatial/bin/python \
        scripts/reproducibility/fig2/fig_spatial_celltype_map.py
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, Circle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig2_style as S

FIG2 = Path(__file__).resolve().parents[3] / "datasets/pancreas_cancer_xenium_10x/processed/fig2"
OUT = Path(__file__).resolve().parent / "outputs"

# ROIs from the two referenced figures (global Xenium µm)
ROIS = [
    dict(name="z-stack ROI", x0=4600, y0=3500, w=300, h=300, color="#ffffff",
         src="pdac_zstack_reconstruction.png"),
    dict(name="3D ROI", x0=7097, y0=1569, w=80, h=80, color="#ffe14d",
         src="fig2_3d_roi.png"),
]
ORDER = ["Ductal cell type 2", "Ductal cell type 1", "Acinar cell", "Endocrine cell",
         "Fibroblast cell", "Stellate cell", "Endothelial cell",
         "Macrophage cell", "T cell", "B cell"]


def main():
    at = sc.read_h5ad(FIG2 / "tracer_annotated.h5ad")
    o = at.obs.dropna(subset=["cell_type"]).copy()
    o = o[o.cell_type.isin(ORDER)]
    colors = o["cell_type"].map(S.CELLTYPE_COLORS).values
    print(f"plotting {len(o):,} cells")

    BG = "#000000"
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(16.5, 8.6), facecolor=BG)
    ax.set_facecolor(BG)

    ax.scatter(o.centroid_x.values, o.centroid_y.values, s=1.6, c=colors,
               linewidths=0, alpha=0.85, rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()                      # image-style orientation
    xmin, xmax = o.centroid_x.min(), o.centroid_x.max()
    ymin, ymax = o.centroid_y.min(), o.centroid_y.max()
    pad = 250
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymax + pad, ymin - pad)    # inverted

    # ---- ROI boxes + leader-line labels ----
    label_anchors = [(xmin + 200, ymin + 250), (xmax - 200, ymin + 250)]
    for roi, (lx, ly) in zip(ROIS, label_anchors):
        cx, cy = roi["x0"] + roi["w"] / 2, roi["y0"] + roi["h"] / 2
        # dashed locator circle ("you-are-here"), distinct from the true-size box
        ax.add_patch(Circle((cx, cy), radius=360, fill=False, edgecolor=roi["color"],
                            linewidth=1.3, linestyle=(0, (4, 3)), alpha=0.85, zorder=6))
        # true-size solid ROI box
        rect = Rectangle((roi["x0"], roi["y0"]), roi["w"], roi["h"],
                         fill=False, edgecolor=roi["color"], linewidth=2.4, zorder=7)
        ax.add_patch(rect)
        ax.annotate(
            f"{roi['name']}\n{roi['w']:.0f}×{roi['h']:.0f} µm  ·  x {roi['x0']:.0f}–{roi['x0']+roi['w']:.0f}, "
            f"y {roi['y0']:.0f}–{roi['y0']+roi['h']:.0f}",
            xy=(cx, cy), xytext=(lx, ly),
            ha="left" if lx < (xmin+xmax)/2 else "right", va="top",
            color=roi["color"], fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a0a0a",
                      edgecolor=roi["color"], linewidth=1.2),
            arrowprops=dict(arrowstyle="->", color=roi["color"], lw=1.8,
                            shrinkA=4, shrinkB=2))

    # ---- scale bar (1 mm) ----
    sb = 1000.0
    sx, sy = xmax - sb - 200, ymax - 60
    ax.plot([sx, sx + sb], [sy, sy], color="#ffffff", lw=3, solid_capstyle="butt")
    ax.text(sx + sb / 2, sy - 70, "1 mm", color="#ffffff", ha="center", va="bottom", fontsize=11)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("PDAC Xenium — TRACER-refined spatial cell-type map",
                 color="#f2f5f8", fontsize=19, fontweight="bold", pad=16, loc="left")
    ax.text(0.0, 1.005, f"{len(o):,} cells · label transfer onto TRACER output · "
            "ROIs of the z-stack & 3D figures boxed",
            transform=ax.transAxes, color="#9aa6b2", fontsize=11.5, va="bottom")

    # ---- legend ----
    handles = [Patch(facecolor=S.CELLTYPE_COLORS[t], edgecolor="none",
                     label=t.replace(" cell", "")) for t in ORDER]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
                    frameon=False, fontsize=11, labelcolor="#e8edf2",
                    title="Cell type", title_fontsize=12.5, handlelength=1.2,
                    labelspacing=0.7)
    leg.get_title().set_color("#e8edf2"); leg.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.02, right=0.88, top=0.92, bottom=0.03)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"pdac_spatial_celltype_map.{ext}", dpi=400,
                    facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote pdac_spatial_celltype_map.png/.svg")


if __name__ == "__main__":
    main()
