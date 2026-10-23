"""One switcher, viewed for separation: three 3D angles chosen relative to its own
best separating direction, plus orthogonal projections.

Angles: the first view looks PERPENDICULAR to the Fisher-LDA direction (the
direction that best separates remnant from partial, fit on held-out halves), so any
separation that exists from some viewpoint is shown at full width there; the other
two are rotated +60 and +120 degrees about z.

Original run (-0.2). Parts as in fig_switcher_class_separation.py: remnant = same-ID
entity, partial = top_epi_partial (immune_switchers.csv), transcripts INSIDE the
Xenium cell. The cell's transcripts that TRACER sent to OTHER entities are drawn as
faint grey in the 2D panels, so marker symbols landing there are not misread.
"""
from __future__ import annotations
import argparse, sys, warnings
import numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns
from fig_axis_separation import auc1d, lda_auc
from fig_one_switcher_3d import hull

OUT = C.Path(__file__).parent
CR, CP, CO = "#d73027", "#1b7837", "#bdbdbd"
CLAB = {"fibroblast": "fibroblast", "myeloid": "myeloid", "t_cell": "T cell"}
ms = yaml.safe_load(open(C.BENCH / "config/panels.yaml"))["marker_sets"]
MKS = {k: set(map(str, v)) for k, v in ms.items()}


def main(cell):
    im = pd.read_csv(OUT / "immune_switchers.csv").set_index("cell_id").loc[cell]
    sep = pd.read_csv(OUT / "switcher_class_separation.csv").set_index("cell_id").loc[cell]
    cls, pid = im["cls"], im["top_epi_partial"]

    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "feature_name", "x", "y", "z"]))
    g = t.loc[t["cell_id"].astype(str) == cell].copy()
    g["tracer_id"] = g["tracer_id"].astype(str); fn = g["feature_name"].astype(str)
    side = np.where(g["tracer_id"] == cell, "remnant",
                    np.where(g["tracer_id"] == pid, "partial", "other"))
    xyz = g[["x", "y", "z"]].to_numpy(float)
    c = xyz[side != "other"].mean(0); xyz = xyz - c
    R_, P, O = xyz[side == "remnant"], xyz[side == "partial"], xyz[side == "other"]
    epi, own = fn.isin(MKS["epithelial"]).to_numpy(), fn.isin(MKS[cls]).to_numpy()

    best, w = lda_auc(P, R_, np.random.default_rng(1))
    if w @ (P.mean(0) - R_.mean(0)) < 0:
        w = -w
    aucs = [auc1d(P[:, k], R_[:, k]) for k in range(3)]
    # u: main axis of the pooled cloud after removing the w component (cell's long axis)
    Q = np.vstack([P, R_]); Qp = Q - np.outer(Q @ w, w)
    u = np.linalg.svd(Qp - Qp.mean(0), full_matrices=False)[2][0]
    u = u - (u @ w) * w; u /= np.linalg.norm(u)          # exactly orthogonal to w
    lim = np.abs(Q).max() * 1.05

    def count(mask):
        return {s: int((mask & (side == s)).sum()) for s in ("remnant", "partial", "other")}
    ce, co = count(epi), count(own)

    fig = plt.figure(figsize=(23, 11.6))
    gs = fig.add_gridspec(2, 4, hspace=0.26, wspace=0.22, left=0.035, right=0.985,
                          top=0.83, bottom=0.06)

    # ---------------- row 1: 3D from the angle of best separation ----------------
    # Rotate into the frame (w, u, v): w = best separating direction (horizontal on
    # screen), u = cell's long axis orthogonal to w, v = w x u. Then turn the cloud
    # ABOUT w by 0/60/120 deg and view along the third axis: every panel looks
    # perpendicular to w, so the separation along w is always shown at full width,
    # from three different sides. Orthographic projection: perspective can make a
    # near cloud look displaced from a far one.
    v = np.cross(w, u)
    B = np.column_stack([w, u, v])
    Rw, Pw = R_ @ B, P @ B
    for k, th in enumerate([0, 60, 120]):
        rad = np.radians(th)
        Rot = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
        r3, p3 = Rw @ Rot.T, Pw @ Rot.T
        a = fig.add_subplot(gs[0, k], projection="3d")
        a.set_proj_type("ortho")
        a.scatter(*r3.T, s=6, color=CR, alpha=0.5, linewidths=0, depthshade=False)
        a.scatter(*p3.T, s=6, color=CP, alpha=0.7, linewidths=0, depthshade=False)
        hull(a, r3, CR); hull(a, p3, CP)
        a.plot([-lim * .8, lim * .8], [0, 0], [0, 0], color="k", lw=2)
        a.scatter([lim * .8], [0], [0], s=40, color="k", marker=">", depthshade=False)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_zlim(-lim, lim)
        a.set_xlabel("best separating direction (um)", fontsize=7)
        a.set_ylabel("orthogonal 1 (um)", fontsize=7); a.set_zlabel("orthogonal 2 (um)", fontsize=7)
        a.tick_params(labelsize=6)
        a.set_box_aspect((1, 1, 1))
        # azim -90 puts the line of sight in the (orthogonal 1, orthogonal 2) plane, so it is
        # perpendicular to w at ANY elevation; elevation only adds depth cues
        a.view_init(elev=22, azim=-90)
        a.set_title(f"angle {k + 1}: perpendicular to the best direction,\nturned {th}° about it",
                    fontsize=9.5)

    a = fig.add_subplot(gs[0, 3])
    pa, pb = P @ w, R_ @ w
    bins = np.linspace(min(pa.min(), pb.min()), max(pa.max(), pb.max()), 40)
    a.hist(pb, bins=bins, color=CR, alpha=0.6, label=f"remnant ({len(R_)} tx)")
    a.hist(pa, bins=bins, color=CP, alpha=0.6, label=f"partial ({len(P)} tx)")
    a.set_xlabel("position along the best separating direction (um)"); a.set_ylabel("transcripts")
    a.legend(fontsize=8.5, frameon=False)
    # what two real whole cells reach at this separation (control curve, binned medians)
    dm = pd.read_csv(OUT / "distance_matched.csv")
    ct = dm.loc[dm["group"] == "control"]
    bb = pd.cut(ct["sep"], [0, .6, .9, 1.2, 1.6, 2.2, 3, 4.5, 99])
    mc = ct.groupby(bb, observed=True).agg(s=("sep", "median"), v=("auc_best", "median")).dropna()
    ref = float(np.interp(np.log10(sep["sep"]), np.log10(mc["s"]), mc["v"]))
    extrap = " (extrapolated)" if sep["sep"] < mc["s"].min() else ""
    a.set_title(f"along the best direction: held-out AUC {best:.2f}\n0.5 = indistinguishable; "
                f"two real cells at sep {sep['sep']:.2f} reach {ref:.2f}{extrap}", fontsize=9.5)

    # ---------------- row 2: projections ----------------
    def proj(a, X, Y, xl, yl, title):
        xo, yo = X(O), Y(O)
        a.scatter(xo, yo, s=4, color=CO, alpha=0.35, linewidths=0)
        a.scatter(X(R_), Y(R_), s=9, color=CR, alpha=0.5, linewidths=0)
        a.scatter(X(P), Y(P), s=9, color=CP, alpha=0.7, linewidths=0)
        for msk, mk in [(epi, "o"), (own, "^")]:
            M = xyz[msk]
            if len(M):
                a.scatter(X(M), Y(M), s=85, marker=mk, facecolor="none", edgecolor="k", lw=1.1)
        a.set_xlabel(xl, fontsize=8); a.set_ylabel(yl, fontsize=8); a.tick_params(labelsize=7)
        a.set_title(title, fontsize=9.5)

    ax_ = [lambda M: M[:, 0], lambda M: M[:, 1], lambda M: M[:, 2]]
    for k, (i, j, nm) in enumerate([(0, 1, "x-y"), (0, 2, "x-z"), (1, 2, "y-z")]):
        a = fig.add_subplot(gs[1, k])
        proj(a, ax_[i], ax_[j], f"{nm[0]} (um)   AUC {aucs[i]:.2f}",
             f"{nm[2]} (um)   AUC {aucs[j]:.2f}", f"projection {nm}")
        a.annotate("", xy=(w[i] * lim * .8, w[j] * lim * .8),
                   xytext=(-w[i] * lim * .8, -w[j] * lim * .8),
                   arrowprops=dict(arrowstyle="-|>", color="k", lw=1.8))
        a.set_aspect("equal")
        oo = np.abs(np.vstack([Q, O])).max() * 1.03
        a.set_xlim(-oo, oo); a.set_ylim(-oo, oo)
    a = fig.add_subplot(gs[1, 3])
    proj(a, lambda M: M @ w, lambda M: M @ u, "best separating direction (um)",
         "cell's long axis, orthogonal to it (um)", "projection onto the best-separation plane")
    a.set_aspect("equal")

    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=CR,
                               label=f"remnant {cell} -> {CLAB[cls]}"),
                        Line2D([], [], marker="o", ls="", color=CP, label=f"epithelial partial {pid}"),
                        Line2D([], [], marker="o", ls="", color=CO,
                               label="same Xenium cell, sent to OTHER entities (2D panels)"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="epithelial marker"),
                        Line2D([], [], marker="^", ls="", mfc="none", mec="k",
                               label=f"{CLAB[cls]} marker"),
                        Line2D([], [], color="k", lw=2, label="best separating direction")],
               loc="upper center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.885))
    n_all = len(g)
    fig.suptitle(
        f"{CLAB[cls]} switcher {cell}: is there any angle at which the two parts separate?\n"
        f"Xenium cell {n_all} tx -> remnant {len(R_)} ({100*len(R_)/n_all:.0f}%), partial {len(P)} "
        f"({100*len(P)/n_all:.0f}%), other entities {len(O)} ({100*len(O)/n_all:.0f}%)   |   "
        f"epithelial markers remnant/partial/other {ce['remnant']}/{ce['partial']}/{ce['other']}, "
        f"{CLAB[cls]} markers {co['remnant']}/{co['partial']}/{co['other']}\n"
        f"sep {sep['sep']:.2f}   position rule {sep['acc_pos']:.2f}   hull nesting "
        f"{100*sep['nest']:.0f}%   |   CNV r vs tumour: Xenium cell {im['r_xenium']:+.2f}, "
        f"remnant {im['r_remnant']:+.2f}, partial {im['r_partial']:+.2f}",
        fontsize=11.5, y=0.985)
    p = OUT / f"fig_one_switcher_views_{cell}.png"; fig.savefig(p, dpi=140); print("wrote", p)
    print(f"best direction w = {np.round(w, 3)}  |w_z| = {abs(w[2]):.2f}; held-out AUC {best:.3f}; "
          f"per-axis AUC x/y/z {aucs[0]:.2f}/{aucs[1]:.2f}/{aucs[2]:.2f}")
    print(f"markers  epi {ce}  {cls} {co}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="hobkdpkn-1")
    main(ap.parse_args().cell)
