"""Worked examples: individual Xenium epithelial cells that TRACER splits.

One panel per cell. Every dot is a transcript, at its real x/y. Colour = which
TRACER entity it was assigned to. Shape = whether it is an epithelial marker,
a fibroblast marker, or neither.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns

OUT = C.Path(__file__).parent
ms = yaml.safe_load(open(C.BENCH / "config/panels.yaml"))["marker_sets"]
EPI, FIB = set(ms["epithelial"]), set(ms["fibroblast"])


def main():
    geo = pd.read_csv(OUT / "converted_split_geometry.csv", dtype={"tracer_id": str, "cell_id": str})
    # pick 6 well-populated examples
    ex = geo.sort_values("n_piece", ascending=False).head(60).sample(6, random_state=0)

    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "x", "y", "overlaps_nucleus", "feature_name", "is_gene", "qv"]))
    t = t.loc[t["is_gene"].astype(bool) & (t["qv"] >= 20)]
    t["cell_id"] = t["cell_id"].astype(str); t["tracer_id"] = t["tracer_id"].astype(str)

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    for ax, (_, rw) in zip(axes.ravel(), ex.iterrows()):
        cid, tid = rw["cell_id"], rw["tracer_id"]
        g = t.loc[t["cell_id"] == cid].copy()
        g["piece"] = g["tracer_id"] == tid
        cx, cy = g["x"].mean(), g["y"].mean()
        g["dx"] = g["x"] - cx; g["dy"] = g["y"] - cy
        g["cls"] = np.where(g["feature_name"].isin(EPI), "epi",
                            np.where(g["feature_name"].isin(FIB), "fib", "other"))
        # background: all other transcripts, coloured by piece membership
        o = g.loc[g.cls == "other"]
        ax.scatter(o.loc[~o.piece, "dx"], o.loc[~o.piece, "dy"], s=7, color="#2e7d32",
                   alpha=0.30, linewidths=0)
        ax.scatter(o.loc[o.piece, "dx"], o.loc[o.piece, "dy"], s=7, color="#d73027",
                   alpha=0.30, linewidths=0)
        # markers on top
        for cls, mk, sz in (("epi", "o", 62), ("fib", "^", 62)):
            s = g.loc[g.cls == cls]
            ax.scatter(s.loc[~s.piece, "dx"], s.loc[~s.piece, "dy"], s=sz, marker=mk,
                       facecolor="#2e7d32", edgecolor="k", linewidths=0.6, zorder=4)
            ax.scatter(s.loc[s.piece, "dx"], s.loc[s.piece, "dy"], s=sz, marker=mk,
                       facecolor="#d73027", edgecolor="k", linewidths=0.6, zorder=4)
        ne, nf = int((g.piece & (g.cls == "epi")).sum()), int((g.piece & (g.cls == "fib")).sum())
        re_, rf = int((~g.piece & (g.cls == "epi")).sum()), int((~g.piece & (g.cls == "fib")).sum())
        ax.set_title(f"Xenium cell {cid}\n"
                     f"stays epithelial: {int((~g.piece).sum())} tx  "
                     f"({re_} epi markers, {rf} fib markers)\n"
                     f"carved 'fibroblast' entity: {int(g.piece.sum())} tx  "
                     f"({ne} epi markers, {nf} fib markers)", fontsize=8.5)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        lim = max(abs(g[["dx", "dy"]].to_numpy()).max() * 1.1, 3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        sc = 2.0
        ax.plot([-lim * 0.92, -lim * 0.92 + sc], [-lim * 0.92] * 2, color="k", lw=2)
        ax.text(-lim * 0.92 + sc / 2, -lim * 0.86, f"{sc:.0f} um", ha="center", fontsize=7)

    handles = [
        Line2D([], [], marker="o", ls="", color="#2e7d32", label="stays in the epithelial entity"),
        Line2D([], [], marker="o", ls="", color="#d73027", label="moved to the 'fibroblast' entity"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="epithelial marker (EPCAM/MSLN/PAX8/CDH1)"),
        Line2D([], [], marker="^", ls="", mfc="none", mec="k", label="fibroblast marker (COL5A1/FAP/PDGFRB/THY1/POSTN)"),
        Line2D([], [], marker=".", ls="", color="0.5", label="all other genes (faded)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Six individual Xenium epithelial cells that TRACER splits in two.\n"
                 "Each dot is one transcript at its measured position. The two groups occupy the same space, "
                 "but the markers separate perfectly.", fontsize=12)
    fig.tight_layout(rect=[0, 0.045, 1, 0.93])
    p = OUT / "fig_split_examples.png"; fig.savefig(p, dpi=150); print("wrote", p)

    tot_e = ex["epi_piece"].sum() + ex["epi_rest"].sum()
    print(f"\nthese 6 examples: epithelial markers in carved pieces = {int(ex['epi_piece'].sum())}, "
          f"in remainders = {int(ex['epi_rest'].sum())} (total {int(tot_e)})")
    print(f"fibroblast markers in carved pieces = {int(ex['fib_piece'].sum())}, "
          f"in remainders = {int(ex['fib_rest'].sum())}")


if __name__ == "__main__":
    main()
