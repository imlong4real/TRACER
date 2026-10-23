"""Are myeloid switchers spatially separable where fibroblast switchers are not?

Hypothesis (user): fibroblast switchers may be genuine EMT (one tumour cell, mixed
programme -> remnant and partial interpenetrate), whereas myeloid switchers may be
two physically distinct cells that Xenium merged (-> remnant and partial separable).

Same definitions as the earlier fibroblast-only analysis, now for every switcher
class (original run, -0.2):
  A = top epithelial partial's transcripts INSIDE the Xenium cell
  B = remnant (same-ID entity) transcripts inside the Xenium cell
  sep        centroid distance / mean spread            (fig_control_adjacent_cells)
  hull nest  % of one side inside the other's hull      (fig_control_adjacent_cells)
  position   LOO 10-NN balanced accuracy                (fig_split_gene_vs_space)
  best plane Fisher-LDA AUC on held-out transcripts     (fig_axis_separation)
Partial = top_epi_partial from immune_switchers.csv (most of the cell's
epithelial markers). Pairs with <20 in-cell transcripts on either side are skipped.

Reference: two-whole-cell pairs across the full separation range
(distance_matched.csv), because separability depends on how far apart the two
clouds are -- class differences are read against that curve, not in isolation.

Caveat for interpretation: a macrophage that has ENGULFED tumour material would
also look spatially inseparable, so "inseparable myeloid" is not proof of one cell.
"""
from __future__ import annotations
import argparse, sys, warnings, zlib
import numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns
from fig_split_gene_vs_space import bal_acc, pos_loo
from fig_axis_separation import auc1d, lda_auc
from fig_control_adjacent_cells import pair_metrics

OUT = C.Path(__file__).parent
RNG = np.random.default_rng(0)
CLASSES = ["fibroblast", "myeloid", "t_cell"]
CLAB = {"fibroblast": "fibroblast", "myeloid": "myeloid", "t_cell": "T cell"}
CC = {"fibroblast": "#984ea3", "myeloid": "#ff7f00", "t_cell": "#e31a1c"}
CR, CP = "#d73027", "#1b7837"
MIN_TX = 20
ms = yaml.safe_load(open(C.BENCH / "config/panels.yaml"))["marker_sets"]
MKS = {k: set(map(str, v)) for k, v in ms.items()}
EPI = MKS["epithelial"]
CSV = OUT / "switcher_class_separation.csv"


def hull(ax3d, pts, color):
    try:
        h = ConvexHull(pts)
    except Exception:
        return
    ax3d.add_collection3d(Poly3DCollection([pts[s] for s in h.simplices], alpha=0.12,
                                           facecolor=color, edgecolor=color, linewidths=0.25))


def compute(sw, by_cell):
    rows = []
    for r in sw.itertuples(index=False):
        g = by_cell[r.cell_id]
        A = g.loc[g["tracer_id"] == r.top_epi_partial, ["x", "y", "z"]].to_numpy(float)
        B = g.loc[g["tracer_id"] == r.cell_id, ["x", "y", "z"]].to_numpy(float)
        if len(A) < MIN_TX or len(B) < MIN_TX:
            continue
        m = pair_metrics(A, B)
        y = np.r_[np.ones(len(A), int), np.zeros(len(B), int)]
        # stable per-pair seed (builtin hash() of str is salted per process)
        best, _ = lda_auc(A, B, np.random.default_rng(zlib.crc32(r.cell_id.encode())))
        m.update(cls=r.cls, cell_id=r.cell_id, partial=r.top_epi_partial, nA=len(A), nB=len(B),
                 acc_pos=bal_acc(y, pos_loo(np.vstack([A, B]), y)), auc_best=best)
        rows.append(m)
    res = pd.DataFrame(rows)
    res["nest"] = res[["frac_A_in_B_hull", "frac_B_in_A_hull"]].mean(1)
    res.to_csv(CSV, index=False)
    return res


def main(ex_cls="myeloid", recompute=False):
    sw = pd.read_csv(OUT / "immune_switchers.csv").dropna(subset=["top_epi_partial"])
    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "feature_name", "x", "y", "z"]))
    t["cell_id"] = t["cell_id"].astype(str); t["tracer_id"] = t["tracer_id"].astype(str)
    by_cell = dict(tuple(t.loc[t["cell_id"].isin(set(sw["cell_id"]))].groupby("cell_id")))

    # the per-switcher table is reused so every example figure shares one bottom row
    res = compute(sw, by_cell) if (recompute or not CSV.exists()) else pd.read_csv(CSV)
    print("pairs analysed:", res["cls"].value_counts().reindex(CLASSES).to_dict(),
          "of", sw["cls"].value_counts().reindex(CLASSES).to_dict(), "with a partial",
          "(recomputed)" if recompute else f"(reused {CSV.name})")

    # sanity: the 40 fibroblast pairs analysed before, where the same partial was used
    old = pd.read_csv(OUT / "switcher_pairs_3d.csv")
    mm = res.merge(old, on="cell_id", suffixes=("", "_old"))
    mm = mm.loc[mm["partial"] == mm["partial_old"]]
    print(f"sanity vs switcher_pairs_3d ({len(mm)} same pairs): max |sep diff| "
          f"{(mm['sep'] - mm['sep_old']).abs().max():.2e}")

    # example: the switcher of ex_cls closest to its class-median best-plane AUC, among
    # REPRESENTATIVE two-part splits -- remnant + partial hold >=50% of the Xenium cell's
    # transcripts AND the partial holds >=50% of its epithelial markers -- with enough
    # transcripts per side to draw (that threshold relaxed only if none qualify)
    im = pd.read_csv(OUT / "immune_switchers.csv")[["cell_id", "n_xen", "top_epi_share"]]
    rx = res.merge(im, on="cell_id", how="left")
    rx["coverage"] = (rx["nA"] + rx["nB"]) / rx["n_xen"]
    rep = (rx["coverage"] >= 0.5) & (rx["top_epi_share"] >= 0.5)
    med = res.loc[res["cls"] == ex_cls, "auc_best"].median()
    for min_side in (250, 150, MIN_TX):
        cand = rx.loc[rep & (rx["cls"] == ex_cls) & (rx["nA"] >= min_side) & (rx["nB"] >= min_side)]
        if len(cand):
            break
    ex = cand.iloc[(cand["auc_best"] - med).abs().argsort().iloc[0]]
    cid, pid = ex["cell_id"], ex["partial"]
    print(f"example {cid}: auc_best {ex.auc_best:.3f} ({ex_cls} median {med:.3f}), sep {ex.sep:.2f}, "
          f"acc_pos {ex.acc_pos:.3f}, nest {ex.nest:.2f}, tx {int(ex.nA)}/{int(ex.nB)}, "
          f"{len(cand)} candidates at >={min_side} tx/side")

    g = by_cell[cid]
    P = g.loc[g["tracer_id"] == pid, ["x", "y", "z"]].to_numpy(float)
    R_ = g.loc[g["tracer_id"] == cid, ["x", "y", "z"]].to_numpy(float)
    c = np.vstack([P, R_]).mean(0); P, R_ = P - c, R_ - c
    fn = g["feature_name"].astype(str)
    EM = g.loc[fn.isin(EPI), ["x", "y", "z"]].to_numpy(float) - c
    MM = g.loc[fn.isin(MKS[ex_cls]), ["x", "y", "z"]].to_numpy(float) - c
    best, wv = lda_auc(P, R_, np.random.default_rng(1))
    if wv @ (P.mean(0) - R_.mean(0)) < 0:
        wv = -wv
    aucs = [auc1d(P[:, k], R_[:, k]) for k in range(3)]
    lim = np.abs(np.vstack([P, R_])).max() * 1.05

    fig = plt.figure(figsize=(22, 11.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9], hspace=0.32, wspace=0.26,
                          left=0.04, right=0.985, top=0.86, bottom=0.07)

    # ---------------- row 1: the example ----------------
    for k, az in enumerate([25, 115]):
        a = fig.add_subplot(gs[0, k], projection="3d")
        a.scatter(*R_.T, s=6, color=CR, alpha=0.5, linewidths=0, depthshade=False)
        a.scatter(*P.T, s=6, color=CP, alpha=0.7, linewidths=0, depthshade=False)
        hull(a, R_, CR); hull(a, P, CP)
        arr = wv * lim * 0.8
        a.plot([-arr[0], arr[0]], [-arr[1], arr[1]], [-arr[2], arr[2]], color="k", lw=2)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_zlim(-lim, lim)
        a.set_xlabel("x (um)", fontsize=7); a.set_ylabel("y (um)", fontsize=7)
        a.set_zlabel("z (um)", fontsize=7); a.tick_params(labelsize=6)
        a.view_init(elev=18, azim=az)
        a.set_title(f"3D, azimuth {az}", fontsize=9)

    a = fig.add_subplot(gs[0, 2])
    a.scatter(R_[:, 0], R_[:, 1], s=9, color=CR, alpha=0.5, linewidths=0)
    a.scatter(P[:, 0], P[:, 1], s=9, color=CP, alpha=0.7, linewidths=0)
    if len(EM):
        a.scatter(EM[:, 0], EM[:, 1], s=90, marker="o", facecolor="none", edgecolor="k", lw=1.1)
    if len(MM):
        a.scatter(MM[:, 0], MM[:, 1], s=90, marker="^", facecolor="none", edgecolor="k", lw=1.1)
    a.annotate("", xy=(wv[0] * lim * .8, wv[1] * lim * .8),
               xytext=(-wv[0] * lim * .8, -wv[1] * lim * .8),
               arrowprops=dict(arrowstyle="-|>", color="k", lw=1.8))
    a.set_aspect("equal"); a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel(f"x (um)   AUC {aucs[0]:.2f}", fontsize=8)
    a.set_ylabel(f"y (um)   AUC {aucs[1]:.2f}", fontsize=8)
    a.set_title(f"x-y projection (z AUC {aucs[2]:.2f})", fontsize=9)

    a = fig.add_subplot(gs[0, 3])
    pa, pb = P @ wv, R_ @ wv
    bins = np.linspace(min(pa.min(), pb.min()), max(pa.max(), pb.max()), 40)
    a.hist(pb, bins=bins, color=CR, alpha=0.6, label=f"remnant -> {CLAB[ex_cls]} ({len(R_)} tx)")
    a.hist(pa, bins=bins, color=CP, alpha=0.6, label=f"epithelial partial ({len(P)} tx)")
    a.set_xlabel("position along the best separating direction (um)")
    a.set_ylabel("transcripts"); a.legend(fontsize=8, frameon=False)
    a.set_title(f"best plane, held-out AUC {best:.2f}", fontsize=9)

    fig.text(0.04, 0.905,
             f"EXAMPLE  {CLAB[ex_cls]} switcher {cid}  |  sep {ex.sep:.2f}   position rule "
             f"{ex.acc_pos:.2f}   hull nesting {100*ex.nest:.0f}%   |   the two parts hold "
             f"{100*ex.coverage:.0f}% of the Xenium cell; the partial holds "
             f"{100*ex.top_epi_share:.0f}% of its epithelial markers",
             fontsize=10.5, fontweight="bold")
    fig.text(0.04, 0.89,
             f"chosen as the {CLAB[ex_cls]} switcher closest to the class median best-plane AUC "
             f"({med:.2f}) among representative two-part splits (parts >=50% of the cell, partial "
             f">=50% of its epithelial markers; {int((rep & (rx['cls'] == ex_cls)).sum())} of "
             f"{int((rx['cls'] == ex_cls).sum())} qualify) with >={min_side} tx per side",
             fontsize=9, style="italic")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=CR,
                               label=f"remnant (same ID) -> {CLAB[ex_cls]}"),
                        Line2D([], [], marker="o", ls="", color=CP, label=f"epithelial partial {pid}"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec="k",
                               label=f"epithelial marker ({len(EM)})"),
                        Line2D([], [], marker="^", ls="", mfc="none", mec="k",
                               label=f"{CLAB[ex_cls]} marker ({len(MM)})"),
                        Line2D([], [], color="k", lw=2, label="best separating direction")],
               loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.955))

    # ---------------- row 2: all switchers by class ----------------
    def by_class(ax, col, ylab, title, ref=None, reflab=None, logy=False, pct=False):
        vals = [res.loc[res["cls"] == c, col].values * (100 if pct else 1) for c in CLASSES]
        bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1.5))
        for p_, c in zip(bp["boxes"], CLASSES):
            p_.set_facecolor(CC[c]); p_.set_alpha(0.35); p_.set_edgecolor(CC[c])
        for k, (v, c) in enumerate(zip(vals, CLASSES), start=1):
            ax.scatter(np.full(len(v), k) + RNG.normal(0, 0.06, len(v)), v, s=16, color=CC[c],
                       alpha=0.8, linewidths=0)
        if ref is not None:
            ax.axhline(ref, color="0.5", ls="--", lw=1, label=reflab)
            ax.legend(fontsize=8, frameon=False, loc="best")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f"{CLAB[c]}\n(n={len(v)})" for c, v in zip(CLASSES, vals)], fontsize=9)
        ax.set_ylabel(ylab)
        p_fm = mannwhitneyu(vals[0], vals[1]).pvalue
        p_ft = mannwhitneyu(vals[0], vals[2]).pvalue
        med_s = " | ".join(f"{np.median(v):.2f}" for v in vals)
        ax.set_title(f"{title}\nmedians {med_s}   fib vs mye p={p_fm:.2g}, fib vs T p={p_ft:.2g}",
                     fontsize=9.5, loc="left")

    by_class(fig.add_subplot(gs[1, 0]), "sep", "sep = centroid distance / mean spread",
             "(a) are the two clouds displaced?", ref=1.0, reflab="1.0 = spatially distinct", logy=True)
    by_class(fig.add_subplot(gs[1, 1]), "auc_best", "held-out AUC, best plane",
             "(b) is there ANY separating plane?", ref=0.5, reflab="0.5 = indistinguishable")
    by_class(fig.add_subplot(gs[1, 2]), "acc_pos", "position rule (balanced acc.)",
             "(c) can position predict the side?", ref=0.5, reflab="chance")

    a = fig.add_subplot(gs[1, 3])
    dm = pd.read_csv(OUT / "distance_matched.csv")
    ct = dm.loc[dm["group"] == "control"]
    a.scatter(ct["sep"], ct["auc_best"], s=14, color="0.6", alpha=0.45, linewidths=0,
              label="two whole cells (control)")
    bb = pd.cut(ct["sep"], [0, .6, .9, 1.2, 1.6, 2.2, 3, 4.5, 99])
    mc = ct.groupby(bb, observed=True).agg(s=("sep", "median"), v=("auc_best", "median")).dropna()
    a.plot(mc["s"], mc["v"], color="0.3", lw=2, marker="o", ms=4, label="control median")
    for c in CLASSES:
        d = res.loc[res["cls"] == c]
        a.scatter(d["sep"], d["auc_best"], s=26, color=CC[c], alpha=0.85, linewidths=0,
                  marker="D", label=f"{CLAB[c]} switchers")
    a.set_xscale("log"); a.axhline(0.5, color="0.7", ls=":", lw=1)
    a.set_xlabel("sep = centroid distance / mean spread"); a.set_ylabel("held-out AUC, best plane")
    a.legend(fontsize=7.5, frameon=False, loc="lower right")
    a.set_title("(d) read against two real cells at the SAME separation", fontsize=9.5, loc="left")

    fig.text(0.04, 0.475, "ALL SWITCHERS  remnant vs top epithelial partial, transcripts inside the "
             "Xenium cell", fontsize=10.5, fontweight="bold")
    fig.suptitle(f"Are {CLAB[ex_cls]} switchers spatially separable where fibroblast switchers "
                 f"are not?", fontsize=14)
    stem = "fig_switcher_class_separation" + ("" if ex_cls == "myeloid" else f"_{ex_cls}")
    p = OUT / f"{stem}.png"; fig.savefig(p, dpi=140); print("wrote", p)

    print("\nmedians by class:")
    print(res.groupby("cls")[["sep", "d3", "nest", "acc_pos", "auc_best", "nA", "nB"]]
          .median().reindex(CLASSES).round(3).to_string())
    print("\nfraction of pairs with sep > 1 / auc_best > 0.85:")
    print(res.groupby("cls").agg(sep_gt1=("sep", lambda s: (s > 1).mean()),
                                 auc_gt085=("auc_best", lambda s: (s > 0.85).mean()))
          .reindex(CLASSES).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--example-class", default="myeloid", choices=CLASSES)
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild switcher_class_separation.csv instead of reusing it")
    a_ = ap.parse_args()
    main(a_.example_class, a_.recompute)
