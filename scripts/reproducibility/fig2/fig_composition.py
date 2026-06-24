#!/usr/bin/env python
"""Figure 2 — composition shift & transcriptional cleanup after TRACER (dark)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig2_style as S

FIG2 = Path(__file__).resolve().parents[3] / "datasets/pancreas_cancer_xenium_10x/processed/fig2"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

SHORT = {
    "Ductal cell type 1": "Ductal-1", "Ductal cell type 2": "Ductal-2",
    "Acinar cell": "Acinar", "Endocrine cell": "Endocrine", "T cell": "T",
    "B cell": "B", "Macrophage cell": "Macrophage", "Endothelial cell": "Endothelial",
    "Fibroblast cell": "Fibroblast", "Stellate cell": "Stellate",
}


def main():
    S.use_dark()
    comp = pd.read_csv(FIG2 / "composition_comparison.csv")
    contam = pd.read_csv(FIG2 / "contamination_by_celltype.csv")
    clean = pd.read_csv(FIG2 / "cleanliness_summary.csv").set_index("group")

    order = ["Ductal cell type 2", "Fibroblast cell", "Stellate cell", "Macrophage cell",
             "B cell", "Endothelial cell", "Acinar cell", "T cell",
             "Ductal cell type 1", "Endocrine cell"]
    piv = comp.pivot(index="cell_type", columns="group", values="frac").reindex(order)

    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.32)

    # ---- A: grouped composition bars ----
    axA = fig.add_subplot(gs[0]); S.style_ax(axA)
    groups = ["original", "tracer_complete", "tracer_partial"]
    gcol = {"original": "#6b7a8d", "tracer_complete": "#4c9bff", "tracer_partial": "#3ddc84"}
    glab = {"original": "Original seg.", "tracer_complete": "TRACER complete",
            "tracer_partial": "TRACER partial"}
    yp = np.arange(len(order)); h = 0.26
    for k, g in enumerate(groups):
        axA.barh(yp + (1 - k) * h, piv[g].values * 100, height=h,
                 color=gcol[g], label=glab[g], edgecolor="none")
    axA.set_yticks(yp); axA.set_yticklabels([SHORT[t] for t in order], fontsize=10)
    axA.invert_yaxis()
    axA.set_xlabel("% of confident cells", fontsize=10.5)
    axA.set_title("a   Cell-type composition", loc="left", color=S.INK,
                  fontsize=13, fontweight="bold")
    axA.legend(frameon=False, fontsize=9, labelcolor=S.INK, loc="lower right")

    # ---- B: contamination dumbbell orig -> complete ----
    axB = fig.add_subplot(gs[1]); S.style_ax(axB)
    cc = contam.set_index("cell_type").reindex(order).dropna(subset=["orig", "complete"])
    yb = np.arange(len(cc))
    for i, (t, r) in enumerate(cc.iterrows()):
        axB.plot([r.orig * 100, r.complete * 100], [i, i], color=S.RULE, lw=2, zorder=1)
        axB.scatter(r.orig * 100, i, color="#6b7a8d", s=46, zorder=2)
        axB.scatter(r.complete * 100, i, color="#ff8ad8", s=46, zorder=3)
    axB.set_yticks(yb); axB.set_yticklabels([SHORT[t] for t in cc.index], fontsize=10)
    axB.invert_yaxis()
    axB.set_xlabel("cross-lineage admixture (%)", fontsize=10.5)
    axB.set_title("b   Transcriptional cleanup", loc="left", color=S.INK,
                  fontsize=13, fontweight="bold")
    axB.scatter([], [], color="#6b7a8d", s=46, label="Original")
    axB.scatter([], [], color="#ff8ad8", s=46, label="TRACER complete")
    axB.legend(frameon=False, fontsize=9, labelcolor=S.INK, loc="lower right")
    om, cm = clean.loc["original", "median_contam"], clean.loc["tracer_complete", "median_contam"]
    axB.text(0.97, 0.30, f"median admixture\n{om*100:.0f}% → {cm*100:.0f}%",
             transform=axB.transAxes, ha="right", va="top", color=S.INK_SOFT,
             fontsize=9.5, linespacing=1.3)

    # ---- C: newly recovered confident partial cells ----
    axC = fig.add_subplot(gs[2]); S.style_ax(axC)
    p = comp[comp.group == "tracer_partial"].set_index("cell_type").reindex(order)
    nvals = p["n"].values
    cols = [S.CELLTYPE_COLORS[t] for t in order]
    axC.barh(np.arange(len(order)), nvals / 1000, color=cols, edgecolor="none")
    axC.set_yticks(np.arange(len(order))); axC.set_yticklabels([SHORT[t] for t in order], fontsize=10)
    axC.invert_yaxis()
    axC.set_xlabel("confident partial cells (×10³)", fontsize=10.5)
    axC.set_title("c   New cells recovered", loc="left", color=S.INK,
                  fontsize=13, fontweight="bold")
    axC.text(0.97, 0.04, f"{int(nvals.sum()):,} total", transform=axC.transAxes,
             ha="right", color=S.INK_SOFT, fontsize=9.5)

    fig.suptitle("TRACER recovers new cells and removes cross-lineage transcript admixture",
                 color=S.INK, fontsize=14.5, fontweight="bold", x=0.5, y=1.04)
    S.save(fig, str(OUT / "fig2_composition_cleanliness"))
    print("wrote fig2_composition_cleanliness.png/.svg")


if __name__ == "__main__":
    main()
