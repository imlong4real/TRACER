"""3D transcript clouds with convex hulls for individual switcher cells.

For a chosen Xenium cell: the same-ID remnant entity (annotated fibroblast,
CNV-flat) vs the epithelial PARTIAL entity that received its epithelial
material (CNV-tumour). If TRACER had separated two physically distinct cells
the hulls would be disjoint.
"""
from __future__ import annotations
import sys, warnings, pickle
import numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, Delaunay
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns

OUT = C.Path(__file__).parent
ms = yaml.safe_load(open(C.BENCH / "config/panels.yaml"))["marker_sets"]
EPI, FIB = set(map(str, ms["epithelial"])), set(map(str, ms["fibroblast"]))
CR, CP = "#d73027", "#1b7837"     # remnant ('fibroblast'), partial (epithelial)


def hull_poly(ax, pts, color, alpha=0.16):
    try:
        h = ConvexHull(pts)
    except Exception:
        return np.nan
    tri = [pts[s] for s in h.simplices]
    pc = Poly3DCollection(tri, alpha=alpha, facecolor=color, edgecolor=color, linewidths=0.25)
    ax.add_collection3d(pc)
    return h.volume


def main():
    pairs = pd.read_csv(OUT / "switcher_pairs_3d.csv")
    picks = ["hohngglj-1", "dpbfcefb-1", "hobkehon-1"]
    sel = pairs.set_index("cell_id").loc[picks]

    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "_etype", "feature_name", "x", "y", "z"]))
    t["cell_id"] = t["cell_id"].astype(str); t["tracer_id"] = t["tracer_id"].astype(str)

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 0.75], hspace=0.22, wspace=0.12,
                          left=0.03, right=0.98, top=0.88, bottom=0.08)

    for k, cid in enumerate(picks):
        pid = sel.loc[cid, "partial"]
        ax = fig.add_subplot(gs[0, k], projection="3d")
        g = t.loc[t["cell_id"] == cid]
        rem = g.loc[g["tracer_id"] == cid][["x", "y", "z"]].to_numpy()
        pin = g.loc[g["tracer_id"] == pid][["x", "y", "z"]].to_numpy()
        pout = t.loc[(t["tracer_id"] == pid) & (t["cell_id"] != cid)][["x", "y", "z"]].to_numpy()
        c = np.vstack([rem, pin]).mean(0)
        R, I, O = rem - c, pin - c, (pout - c if len(pout) else pout)

        if len(O):
            ax.scatter(O[:, 0], O[:, 1], O[:, 2], s=2, color=CP, alpha=0.16, linewidths=0,
                       depthshade=False)
        ax.scatter(R[:, 0], R[:, 1], R[:, 2], s=5, color=CR, alpha=0.55, linewidths=0, depthshade=False)
        ax.scatter(I[:, 0], I[:, 1], I[:, 2], s=7, color=CP, alpha=0.75, linewidths=0, depthshade=False)
        vR = hull_poly(ax, R, CR); vI = hull_poly(ax, I, CP)

        mk = g["feature_name"].astype(str)
        em = g.loc[mk.isin(EPI)][["x", "y", "z"]].to_numpy() - c
        fm = g.loc[mk.isin(FIB)][["x", "y", "z"]].to_numpy() - c
        if len(em):
            ax.scatter(em[:, 0], em[:, 1], em[:, 2], s=60, marker="o", facecolor="none",
                       edgecolor="k", linewidths=0.9, depthshade=False)
        if len(fm):
            ax.scatter(fm[:, 0], fm[:, 1], fm[:, 2], s=60, marker="^", facecolor="none",
                       edgecolor="k", linewidths=0.9, depthshade=False)

        rw = sel.loc[cid]
        ax.set_title(f"Xenium cell {cid}\n"
                     f"remnant '{'fibroblast'}' {int(rw.n_rem)} tx  (epi mk {int(rw.epi_rem)}, "
                     f"fib mk {int(rw.fib_rem)})  CNV r={rw.r_rem:+.2f}\n"
                     f"partial {pid} {int(rw.n_part)} tx in-cell  (epi mk {int(rw.epi_part)})  "
                     f"CNV r={rw.r_part:+.2f}\n"
                     f"hull sep={rw.sep:.2f}; {100*rw.frac_part_in_rem_hull:.0f}% of partial pts "
                     f"inside the remnant hull", fontsize=8)
        ax.set_xlabel("x (um)", fontsize=7); ax.set_ylabel("y (um)", fontsize=7)
        ax.set_zlabel("z (um)", fontsize=7)
        ax.tick_params(labelsize=6)
        lim = np.abs(np.vstack([R, I])).max() * 1.05
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.view_init(elev=18, azim=35)

    # summary panels across all 40 pairs
    a = fig.add_subplot(gs[1, 0])
    a.hist(pairs["sep"], bins=np.arange(0, 2.0, 0.1), color="#4575b4", alpha=0.9)
    a.axvline(1.0, color="crimson", ls="--", lw=1.3, label="1.0 = spatially distinct")
    a.set_xlabel("3D centroid distance / mean spread"); a.set_ylabel("switcher pairs")
    a.legend(fontsize=8, frameon=False)
    a.set_title(f"(d) 3D separation, all {len(pairs)} pairs (median {pairs.sep.median():.2f})",
                fontsize=9.5, loc="left")

    a = fig.add_subplot(gs[1, 1])
    a.hist(pairs["frac_part_in_rem_hull"] * 100, bins=np.arange(0, 105, 5), color="#1b7837", alpha=0.85,
           label="partial pts inside remnant hull")
    a.hist(pairs["frac_rem_in_part_hull"] * 100, bins=np.arange(0, 105, 5), color="#d73027", alpha=0.55,
           label="remnant pts inside partial hull")
    a.set_xlabel("% of points inside the other component's hull"); a.set_ylabel("switcher pairs")
    a.legend(fontsize=8, frameon=False)
    a.set_title("(e) the hulls are nested, not disjoint", fontsize=9.5, loc="left")

    a = fig.add_subplot(gs[1, 2])
    a.scatter(pairs["r_rem"], pairs["r_part"], s=26, color="#984ea3", alpha=0.8, linewidths=0)
    a.axhline(0, color="0.7", lw=0.6); a.axvline(0, color="0.7", lw=0.6)
    a.plot([-0.4, 1.05], [-0.4, 1.05], ls=":", color="0.6", lw=0.8)
    a.set_xlabel("remnant: CNV r vs tumour profile")
    a.set_ylabel("partial: CNV r vs tumour profile")
    a.set_xlim(-0.4, 1.05); a.set_ylim(-0.4, 1.05)
    a.set_title(f"(f) same space, different genome\nmedians {pairs.r_rem.median():+.2f} vs "
                f"{pairs.r_part.median():+.2f}", fontsize=9.5, loc="left")

    handles = [Line2D([], [], marker="o", ls="", color=CR, label="remnant entity (-> 'fibroblast')"),
               Line2D([], [], marker="o", ls="", color=CP, label="epithelial PARTIAL entity, in this cell"),
               Line2D([], [], marker="o", ls="", color=CP, alpha=0.3,
                      label="same partial, transcripts in NEIGHBOURING cells"),
               Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="epithelial marker"),
               Line2D([], [], marker="^", ls="", mfc="none", mec="k", label="fibroblast marker")]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Do the two components occupy separate space? 3D transcript clouds with convex hulls",
                 fontsize=13, y=0.985)
    p = OUT / "fig_3d_hulls.png"; fig.savefig(p, dpi=150); print("wrote", p)


if __name__ == "__main__":
    main()
