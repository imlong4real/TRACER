"""One representative switcher cell, in 3D and in orthogonal projection.

fig_3d_hulls.py led with hohngglj-1, which has sep=1.60 -- the most separated
of all 40 pairs, i.e. the least representative. This uses the cell closest to
the median sep instead, shows it from several angles, and adds the three
orthogonal projections, which are honest about interpenetration in a way a
single 3D perspective is not.

--cell picks another switcher (default dplafofh-1, the fibroblast example). Fibroblast
switchers from the original 40 read switcher_pairs_3d.csv / split_gene_vs_space.csv; any
other switcher (myeloid, T cell) reads switcher_class_separation.csv + immune_switchers.csv,
and its own-type markers replace the fibroblast ones. Output for a non-default cell:
fig_one_switcher_3d_<cell>.png.

Only transcripts Xenium assigned to THIS cell are drawn. The partial's mass in
neighbouring cells is deliberately excluded here -- this figure is about what
happened inside one cell boundary.
"""
from __future__ import annotations
import argparse, sys, warnings
import numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns
from fig_axis_separation import auc1d, lda_auc

OUT = C.Path(__file__).parent
DEFAULT_CELL = "dplafofh-1"
CR, CP = "#d73027", "#1b7837"
ms = yaml.safe_load(open(C.BENCH / "config/panels.yaml"))["marker_sets"]
EPI = set(map(str, ms["epithelial"]))
CLAB = {"fibroblast": "FIBROBLAST", "myeloid": "MYELOID", "t_cell": "T CELL"}


def hull(ax3d, pts, color):
    try:
        h = ConvexHull(pts)
    except Exception:
        return
    ax3d.add_collection3d(Poly3DCollection(
        [pts[s] for s in h.simplices], alpha=0.13, facecolor=color,
        edgecolor=color, linewidths=0.25))


def load_row(cell):
    """Unified per-switcher numbers: partial id, in-cell counts, sep, nesting, rules, CNV r."""
    sw = pd.read_csv(OUT / "switcher_pairs_3d.csv").set_index("cell_id")
    if cell in sw.index:
        r = sw.loc[cell]
        a = pd.read_csv(OUT / "split_gene_vs_space.csv").set_index("cell_id").loc[cell]
        return dict(cls="fibroblast", partial=r.partial, n_rem=r.n_rem, n_part=r.n_part,
                    sep=r.sep, nest=r.frac_part_in_rem_hull, acc_pos=a.acc_pos,
                    acc_gene=a.acc_gene, r_rem=r.r_rem, r_part=r.r_part)
    cs = pd.read_csv(OUT / "switcher_class_separation.csv").set_index("cell_id").loc[cell]
    im = pd.read_csv(OUT / "immune_switchers.csv").set_index("cell_id").loc[cell]
    # switcher_class_separation: A = top epithelial partial, B = remnant (in-cell transcripts)
    return dict(cls=cs.cls, partial=cs.partial, n_rem=cs.nB, n_part=cs.nA, sep=cs.sep,
                nest=cs.frac_A_in_B_hull, acc_pos=cs.acc_pos, acc_gene=np.nan,
                r_rem=im.r_remnant, r_part=im.r_partial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=DEFAULT_CELL)
    CELL = ap.parse_args().cell
    r = pd.Series(load_row(CELL))
    pid = r["partial"]
    OWN = set(map(str, ms[r.cls]))
    own = CLAB[r.cls].lower()

    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "_etype", "feature_name", "x", "y", "z"]))
    g = t.loc[t["cell_id"].astype(str) == CELL].copy()
    g["tracer_id"] = g["tracer_id"].astype(str)
    R = g.loc[g["tracer_id"] == CELL][["x", "y", "z"]].to_numpy(float)
    P = g.loc[g["tracer_id"] == pid][["x", "y", "z"]].to_numpy(float)
    c = np.vstack([R, P]).mean(0)
    R, P = R - c, P - c
    # per-axis separation and the best separating direction (held-out AUC)
    aucs = [auc1d(P[:, k], R[:, k]) for k in range(3)]
    best, wv = lda_auc(P, R, np.random.default_rng(1))
    if wv @ (P.mean(0) - R.mean(0)) < 0:
        wv = -wv
    print(f"per-axis AUC x/y/z = {aucs[0]:.3f}/{aucs[1]:.3f}/{aucs[2]:.3f}; "
          f"best direction {np.round(wv,3)} AUC {best:.3f}")

    gm = g["feature_name"].astype(str)
    EM = g.loc[gm.isin(EPI)][["x", "y", "z"]].to_numpy(float) - c
    FM = g.loc[gm.isin(OWN)][["x", "y", "z"]].to_numpy(float) - c

    fig = plt.figure(figsize=(17.5, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.08, 0.92], hspace=0.16, wspace=0.16,
                          left=0.04, right=0.97, top=0.825, bottom=0.07)
    lim = np.abs(np.vstack([R, P])).max() * 1.05

    # --- row 1: three rotations in 3D ---
    for k, az in enumerate([20, 80, 140]):
        ax = fig.add_subplot(gs[0, k], projection="3d")
        ax.scatter(*R.T, s=6, color=CR, alpha=0.55, linewidths=0, depthshade=False)
        ax.scatter(*P.T, s=6, color=CP, alpha=0.70, linewidths=0, depthshade=False)
        hull(ax, R, CR); hull(ax, P, CP)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xlabel("x (um)", fontsize=7); ax.set_ylabel("y (um)", fontsize=7)
        ax.set_zlabel("z (um)", fontsize=7); ax.tick_params(labelsize=6)
        arr = wv * lim * 0.8
        ax.plot([-arr[0], arr[0]], [-arr[1], arr[1]], [-arr[2], arr[2]],
                color="k", lw=2.0, ls="-")
        ax.scatter(*arr, s=40, color="k", marker="^", depthshade=False)
        ax.view_init(elev=18, azim=az)
        ax.set_title(f"3D, azimuth {az}", fontsize=9)

    # --- row 2: orthogonal projections, with markers ---
    for k, (i, j, nm) in enumerate([(0, 1, "x-y"), (0, 2, "x-z"), (1, 2, "y-z")]):
        ax = fig.add_subplot(gs[1, k])
        ax.scatter(R[:, i], R[:, j], s=9, color=CR, alpha=0.55, linewidths=0)
        ax.scatter(P[:, i], P[:, j], s=9, color=CP, alpha=0.75, linewidths=0)
        if len(EM):
            ax.scatter(EM[:, i], EM[:, j], s=95, marker="o", facecolor="none",
                       edgecolor="k", linewidths=1.1)
        if len(FM):
            ax.scatter(FM[:, i], FM[:, j], s=95, marker="^", facecolor="none",
                       edgecolor="k", linewidths=1.1)
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.annotate("", xy=(wv[i] * lim * 0.8, wv[j] * lim * 0.8),
                    xytext=(-wv[i] * lim * 0.8, -wv[j] * lim * 0.8),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.8))
        ax.set_xlabel(f"{nm[0]} (um)   AUC {aucs[i]:.2f}", fontsize=8)
        ax.set_ylabel(f"{nm[2]} (um)   AUC {aucs[j]:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"projection {nm}", fontsize=9)

    handles = [Line2D([], [], marker="o", ls="", color=CR,
                      label=f"-> remnant, keeps ID {CELL}, annotated {CLAB[r.cls]} "
                            f"({int(r.n_rem)} tx)"),
               Line2D([], [], marker="o", ls="", color=CP,
                      label=f"-> epithelial partial {pid} ({int(r.n_part)} tx in this cell)"),
               Line2D([], [], marker="o", ls="", mfc="none", mec="k",
                      label=f"epithelial marker ({len(EM)})"),
               Line2D([], [], marker="^", ls="", mfc="none", mec="k",
                      label=f"{own} marker ({len(FM)})"),
               Line2D([], [], color="k", lw=2.0,
                      label="best separating direction (Fisher LDA)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.905))
    fig.suptitle(
        f"One Xenium cell, {CELL}, after TRACER   |   spatially inseparable, "
        f"transcriptionally distinct\n"
        f"sep {r.sep:.2f}  .  {100*r.nest:.0f}% of the "
        f"partial sits inside the remnant hull  .  predict the split from position "
        f"{r.acc_pos:.2f}" + (f", from gene {r.acc_gene:.2f}" if pd.notna(r.acc_gene) else "")
        + f"\n"
        f"separation AUC  x {aucs[0]:.2f}  y {aucs[1]:.2f}  z {aucs[2]:.2f}  |  best plane of "
        f"ANY orientation {best:.2f}  (0.5 = indistinguishable)\n"
        f"CNV vs tumour profile:  remnant r={r.r_rem:+.2f}   partial r={r.r_part:+.2f}",
        fontsize=11.5, y=0.99)
    name = "fig_one_switcher_3d.png" if CELL == DEFAULT_CELL else f"fig_one_switcher_3d_{CELL}.png"
    p = OUT / name; fig.savefig(p, dpi=150); print("wrote", p)
    print(f"{CELL}: remnant {int(r.n_rem)} tx, partial {int(r.n_part)} tx in-cell; "
          f"epi markers {len(EM)}, {own} markers {len(FM)}")


if __name__ == "__main__":
    main()
