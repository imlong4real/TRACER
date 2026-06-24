#!/usr/bin/env python
"""Figure 2 — TRACER sharpens plausible local immune interactions (dark)."""
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


def main():
    S.use_dark()
    r = pd.read_csv(FIG2 / "cci_lr_comparison.csv")
    fold = r.pivot(index="pair", columns="dataset", values="fold")
    pur = r.pivot(index="pair", columns="dataset", values="sender_purity")
    obs = r.pivot(index="pair", columns="dataset", values="obs")
    # well-powered pairs (>=20 contacts in both) vs sparse
    powered = (obs["original"] >= 20) & (obs["tracer"] >= 20)
    fold = fold.reindex(fold["tracer"].sort_values().index)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(wspace=0.42)

    # ---- A: fold enrichment original vs TRACER ----
    S.style_ax(axA)
    yp = np.arange(len(fold)); h = 0.36
    axA.barh(yp + h/2, fold["original"], height=h, color="#6b7a8d", label="Original")
    axA.barh(yp - h/2, fold["tracer"], height=h, color="#3ddc84", label="TRACER")
    axA.axvline(1.0, color=S.RULE, lw=1, ls="--")
    for i, pr in enumerate(fold.index):
        if not powered.get(pr, False):
            axA.text(0.05, i, "sparse", va="center", ha="left", color="#ff5d5d", fontsize=8)
    axA.set_yticks(yp); axA.set_yticklabels(fold.index, fontsize=10)
    axA.set_xlabel("spatial co-occurrence fold (obs / permuted)", fontsize=10.5)
    axA.set_title("a   Local LR interaction enrichment", loc="left", color=S.INK,
                  fontsize=13, fontweight="bold")
    axA.legend(frameon=False, fontsize=9.5, labelcolor=S.INK, loc="lower right")

    # ---- B: sender-lineage purity dumbbell ----
    S.style_ax(axB)
    pp = pur.dropna().reindex([p for p in fold.index if p in pur.dropna().index])
    yb = np.arange(len(pp))
    for i, (pr, row) in enumerate(pp.iterrows()):
        axB.plot([row["original"]*100, row["tracer"]*100], [i, i], color=S.RULE, lw=2, zorder=1)
        axB.scatter(row["original"]*100, i, color="#6b7a8d", s=52, zorder=2)
        axB.scatter(row["tracer"]*100, i, color="#3ddc84", s=52, zorder=3)
    axB.set_yticks(yb); axB.set_yticklabels(pp.index, fontsize=10)
    axB.set_xlabel("sender-lineage purity (%)", fontsize=10.5)
    axB.set_title("b   Implausible senders removed", loc="left", color=S.INK,
                  fontsize=13, fontweight="bold")
    axB.scatter([], [], color="#6b7a8d", s=52, label="Original")
    axB.scatter([], [], color="#3ddc84", s=52, label="TRACER")
    axB.legend(frameon=False, fontsize=9.5, labelcolor=S.INK, loc="lower right")
    mo, mt = pp["original"].mean()*100, pp["tracer"].mean()*100
    axB.text(0.03, 0.04, f"mean {mo:.0f}% → {mt:.0f}%", transform=axB.transAxes,
             color=S.INK_SOFT, fontsize=10)

    fig.suptitle("TRACER strengthens plausible local immune interactions and purifies ligand senders",
                 color=S.INK, fontsize=14, fontweight="bold", y=1.02)
    S.save(fig, str(OUT / "fig2_cci"))
    print("wrote fig2_cci.png/.svg")


if __name__ == "__main__":
    main()
