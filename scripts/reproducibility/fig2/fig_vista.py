#!/usr/bin/env python
"""Figure 2 — TRACER recovers immunoregulatory VSIG4+ TAMs near T cells (dark).

a  VSIG4+ cell-type attribution, original vs TRACER (macrophage specificity jump)
b  spatial ROI: VSIG4+ TAMs co-localising with T cells (immunosuppressive niche)
c  VSIG4+ macrophage -> nearest T-cell distance (enrichment vs tissue baseline)
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.neighbors import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig2_style as S

FIG2 = Path(__file__).resolve().parents[3] / "datasets/pancreas_cancer_xenium_10x/processed/fig2"
OUT = Path(__file__).resolve().parent / "outputs"
CONF = 0.5


def load(name):
    d = pd.read_parquet(FIG2 / f"vista_hypoxia_{name}.parquet")
    return d[d.lt_conf >= CONF].copy()


def attribution(d):
    pos = d[d.VSIG4_pos]
    return pos.cell_type.value_counts(normalize=True)


def main():
    S.use_dark()
    do, dt = load("original"), load("tracer")
    order = ["Macrophage cell", "Fibroblast cell", "Ductal cell type 2", "Stellate cell",
             "B cell", "T cell", "Endothelial cell", "Acinar cell",
             "Ductal cell type 1", "Endocrine cell"]
    ao, at = attribution(do), attribution(dt)

    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.35, 0.95], wspace=0.3)

    # ---- A: VSIG4+ attribution stacked bars ----
    axA = fig.add_subplot(gs[0]); S.style_ax(axA, spines=("bottom",))
    for yi, (lab, ser) in enumerate([("Original", ao), ("TRACER", at)]):
        left = 0
        for t in order:
            v = ser.get(t, 0.0)
            if v <= 0:
                continue
            axA.barh(yi, v * 100, left=left * 100, color=S.CELLTYPE_COLORS[t],
                     edgecolor=S.BG, linewidth=0.7)
            if t == "Macrophage cell" and v > 0.04:
                axA.text((left + v / 2) * 100, yi, f"{v*100:.0f}%", ha="center",
                         va="center", color="#10131a", fontsize=10, fontweight="bold")
            left += v
    axA.set_yticks([0, 1]); axA.set_yticklabels(["Original", "TRACER"], fontsize=11)
    axA.set_xlabel("% of VSIG4+ cells", fontsize=10.5)
    axA.set_xlim(0, 100); axA.set_ylim(-0.6, 1.6)
    axA.set_title("a   VSIG4+ attribution", loc="left", color=S.INK, fontsize=13, fontweight="bold")
    axA.annotate("", xy=(at.get("Macrophage cell", 0) * 100, 1.42),
                 xytext=(ao.get("Macrophage cell", 0) * 100, 0.58),
                 arrowprops=dict(arrowstyle="->", color="#ff8ad8", lw=1.6))
    axA.text(0.5, -0.22, "macrophage specificity "
             f"{ao.get('Macrophage cell',0)*100:.0f}% → {at.get('Macrophage cell',0)*100:.0f}%",
             transform=axA.transAxes, ha="center", color=S.INK_SOFT, fontsize=10)
    # legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=S.CELLTYPE_COLORS[t], label=t.replace(" cell", "")) for t in
           ["Macrophage cell", "Fibroblast cell", "Ductal cell type 2", "T cell"]]
    axA.legend(handles=leg, frameon=False, fontsize=8.5, labelcolor=S.INK,
               loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, handlelength=1.1)

    # ---- find ROI: dense in VSIG4+ Mac and T cells (TRACER) ----
    dt["vmac"] = dt.VSIG4_pos & dt.cell_type.eq("Macrophage cell")
    xy = dt[["centroid_x", "centroid_y"]].values
    best, W = None, 450.0
    xs = np.arange(xy[:, 0].min(), xy[:, 0].max() - W, 150)
    ys = np.arange(xy[:, 1].min(), xy[:, 1].max() - W, 150)
    vmac_xy = xy[dt.vmac.values]; t_xy = xy[dt.cell_type.eq("T cell").values]
    for x0 in xs:
        for y0 in ys:
            nm = ((vmac_xy[:, 0] >= x0) & (vmac_xy[:, 0] < x0 + W) &
                  (vmac_xy[:, 1] >= y0) & (vmac_xy[:, 1] < y0 + W)).sum()
            nt = ((t_xy[:, 0] >= x0) & (t_xy[:, 0] < x0 + W) &
                  (t_xy[:, 1] >= y0) & (t_xy[:, 1] < y0 + W)).sum()
            score = min(nm, nt / 3)
            if best is None or score > best[0]:
                best = (score, x0, y0, nm, nt)
    _, x0, y0, nm, nt = best

    # ---- B: ROI spatial niche ----
    axB = fig.add_subplot(gs[1]); S.style_ax(axB, spines=())
    m = ((dt.centroid_x >= x0) & (dt.centroid_x < x0 + W) &
         (dt.centroid_y >= y0) & (dt.centroid_y < y0 + W))
    r = dt[m]
    axB.scatter(r.centroid_x, r.centroid_y, s=6, c="#2b3340", linewidths=0,
                rasterized=True, label="other cells")
    tc = r[r.cell_type.eq("T cell")]
    axB.scatter(tc.centroid_x, tc.centroid_y, s=18, c=S.CELLTYPE_COLORS["T cell"],
                linewidths=0, label=f"T cell (n={len(tc)})", rasterized=True)
    mac = r[r.cell_type.eq("Macrophage cell")]
    axB.scatter(mac.centroid_x, mac.centroid_y, s=16, c="#6b4a63", linewidths=0,
                label=f"Macrophage (n={len(mac)})", rasterized=True)
    vm = r[r.vmac]
    axB.scatter(vm.centroid_x, vm.centroid_y, s=52, c=S.CELLTYPE_COLORS["Macrophage cell"],
                edgecolors="white", linewidths=0.7, label=f"VSIG4+ TAM (n={len(vm)})", zorder=5)
    axB.set_xticks([]); axB.set_yticks([]); axB.set_aspect("equal")
    axB.set_title("b   VSIG4+ TAM–T cell niche (TRACER ROI)", loc="left",
                  color=S.INK, fontsize=13, fontweight="bold")
    leg = axB.legend(fontsize=8.8, labelcolor=S.INK, loc="upper right", markerscale=1.25,
                     frameon=True, framealpha=0.9, handletextpad=0.5, borderpad=0.6)
    leg.get_frame().set_facecolor(S.PANEL); leg.get_frame().set_edgecolor(S.RULE)
    # ROI coordinate bounds (global Xenium µm)
    axB.text(0.015, 0.02,
             f"ROI  x: {x0:.0f}–{x0+W:.0f} µm   y: {y0:.0f}–{y0+W:.0f} µm",
             transform=axB.transAxes, ha="left", va="bottom", color=S.INK_SOFT,
             fontsize=8.6)
    # scale bar 100 µm
    axB.plot([x0 + 30, x0 + 130], [y0 + W - 25, y0 + W - 25], color=S.INK, lw=2.5)
    axB.text(x0 + 80, y0 + W - 38, "100 µm", color=S.INK, fontsize=8.5, ha="center", va="top")

    # ---- C: distance to nearest T (VSIG4+ Mac vs all cells), TRACER ----
    axC = fig.add_subplot(gs[2]); S.style_ax(axC)
    for name, d, col in [("Original", do, "#6b7a8d"), ("TRACER", dt, "#ff8ad8")]:
        d = d.copy()
        d["vmac"] = d.VSIG4_pos & d.cell_type.eq("Macrophage cell")
        xyc = d[["centroid_x", "centroid_y"]].values
        tmask = d.cell_type.eq("T cell").values
        if tmask.sum() < 10:
            continue
        tree = KDTree(xyc[tmask])
        dist, _ = tree.query(xyc[d.vmac.values], k=1)
        dist = dist.ravel()
        xs_ = np.sort(dist); ys_ = np.arange(1, len(xs_) + 1) / len(xs_)
        axC.plot(xs_, ys_, color=col, lw=2.2, label=f"{name} (med {np.median(dist):.0f}µm)")
    axC.set_xlim(0, 250)
    axC.set_xlabel("VSIG4+ TAM → nearest T cell (µm)", fontsize=10.5)
    axC.set_ylabel("cumulative fraction", fontsize=10.5)
    axC.set_title("c   Immunosuppressive proximity", loc="left", color=S.INK,
                  fontsize=13, fontweight="bold")
    axC.legend(frameon=False, fontsize=9, labelcolor=S.INK, loc="lower right")

    fig.suptitle("TRACER recovers immunoregulatory VSIG4+ macrophages and their T-cell niches",
                 color=S.INK, fontsize=14.5, fontweight="bold", y=1.03)
    S.save(fig, str(OUT / "fig2_vista_vsig4"))
    print(f"ROI x0={x0:.0f} y0={y0:.0f} W={W:.0f} VSIG4mac={nm} T={nt}")
    print("wrote fig2_vista_vsig4.png/.svg")


if __name__ == "__main__":
    main()
