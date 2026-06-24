#!/usr/bin/env python
"""Figure 2 supplement (dark): honest-negative hypoxia & EMT panels.

a  EMT is largely a segmentation/admixture artifact on this panel — TRACER
   reduces the 'EMT-high' ductal fraction and EMT-high cells stay the most
   cross-lineage-contaminated (no epithelial EMT genes; FN1/SPARC/TGFB1 are
   shared with CAFs).
b  Hypoxia surrogate (VEGFA+HIF1A only) shows only weak spatial association
   with VISTA(VSIR)/VSIG4 — panel lacks core hypoxia genes.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig2_style as S

FIG2 = Path(__file__).resolve().parents[3] / "datasets/pancreas_cancer_xenium_10x/processed/fig2"
OUT = Path(__file__).resolve().parent / "outputs"
EMT = ["FN1", "SPARC", "TGFB1"]


def emt_stats(name):
    a = sc.read_h5ad(FIG2 / f"{name}_annotated.h5ad")
    a = a[a.obs.n_counts >= 10].copy()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.tl.score_genes(a, EMT, score_name="emt", ctrl_size=50)
    thr = np.quantile(a.obs.emt, 0.75)
    d = a.obs[(a.obs.cell_type == "Ductal cell type 2") & (a.obs.lt_conf >= 0.5)]
    hi = d[d.emt > thr]; lo = d[d.emt <= thr]
    return dict(frac=(d.emt > thr).mean(),
                contam_hi=np.nanmedian(hi.contamination),
                contam_lo=np.nanmedian(lo.contamination))


def hyp_rho(name):
    d = pd.read_parquet(FIG2 / f"vista_hypoxia_{name}.parquet")
    xy = d[["centroid_x", "centroid_y"]].values
    bx = (xy[:, 0] // 100).astype(int); by = (xy[:, 1] // 100).astype(int)
    df = pd.DataFrame({"bx": bx, "by": by, "hyp": d.hypoxia_local.values,
                       "vsir": d.VSIR_pos.values, "vsig4": d.VSIG4_pos.values})
    gb = df.groupby(["bx", "by"]).agg(hyp=("hyp", "mean"), vsir=("vsir", "mean"),
                                      vsig4=("vsig4", "mean"), n=("hyp", "size"))
    gb = gb[gb.n >= 10]
    return {g: spearmanr(gb.hyp, gb[g])[0] for g in ["vsir", "vsig4"]}


def main():
    S.use_dark()
    eo, et = emt_stats("original"), emt_stats("tracer")
    ho, ht = hyp_rho("original"), hyp_rho("tracer")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.subplots_adjust(wspace=0.34)

    # ---- A: EMT — EMT-high ductal cells are the most contaminated ----
    S.style_ax(axA)
    axA.bar([0, 1], [et["contam_hi"]*100, et["contam_lo"]*100], width=0.5,
            color=["#ff5d5d", "#3ddc84"])
    axA.set_xticks([0, 1]); axA.set_xticklabels(["EMT-high\nductal", "EMT-low\nductal"], fontsize=10)
    axA.set_ylabel("cross-lineage contamination (%)", fontsize=10)
    axA.set_title("a   'EMT' is mostly admixture artifact", loc="left", color=S.INK,
                  fontsize=12.5, fontweight="bold")
    axA.text(0.5, 0.95, f"EMT-high ductal fraction {eo['frac']*100:.1f}% → {et['frac']*100:.1f}% after TRACER;\n"
             f"EMT-high cells are {et['contam_hi']/max(et['contam_lo'],1e-6):.0f}× more contaminated → CAF spillover,\n"
             f"not true partial-EMT (no VIM/CDH/ZEB/TWIST on panel)",
             transform=axA.transAxes, color=S.INK_SOFT, fontsize=8.6, va="top", ha="center")

    # ---- B: hypoxia weak ----
    S.style_ax(axB)
    labels = ["VSIR vs hypoxia", "VSIG4 vs hypoxia"]
    xp = np.arange(2)
    axB.bar(xp - 0.18, [ho["vsir"], ho["vsig4"]], width=0.36, color="#6b7a8d", label="Original")
    axB.bar(xp + 0.18, [ht["vsir"], ht["vsig4"]], width=0.36, color="#29e0e0", label="TRACER")
    axB.axhline(0, color=S.RULE, lw=1)
    axB.set_xticks(xp); axB.set_xticklabels(labels, fontsize=10)
    axB.set_ylabel("grid Spearman ρ (100 µm bins)", fontsize=10)
    axB.set_ylim(-0.02, 0.12)
    axB.set_title("b   Hypoxia association is weak", loc="left", color=S.INK,
                  fontsize=12.5, fontweight="bold")
    axB.legend(frameon=False, fontsize=9, labelcolor=S.INK, loc="upper right")
    axB.text(0.02, 0.93, "surrogate score = VEGFA + HIF1A only\n(panel lacks CA9/SLC2A1/LDHA/…)",
             transform=axB.transAxes, color=S.INK_SOFT, fontsize=9, va="top")

    fig.suptitle("Supplement — analyses limited by panel gene coverage (honest negatives)",
                 color=S.INK, fontsize=13.5, fontweight="bold", y=1.02)
    S.save(fig, str(OUT / "fig2_supp_hypoxia_emt"))
    print("EMT:", eo, et)
    print("HYP:", ho, ht)
    print("wrote fig2_supp_hypoxia_emt.png/.svg")


if __name__ == "__main__":
    main()
