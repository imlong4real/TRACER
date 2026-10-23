"""Is the switchers' spatial inseparability just a consequence of proximity?

The earlier controls sat at 6-8 um (crowded) and 26 um (typical) centroid
separation; the switcher remnant/partial pairs sit at 1.6 um. Position
separates the controls better partly BECAUSE they are further apart, so that
comparison is confounded.

Fix: sample adjacent whole-cell pairs across the full range of separation,
measure position-separability as a function of separation, and see whether the
switchers fall on that curve (proximity explains everything) or below it (there
is extra mixing beyond proximity).

x-axis is sep = centroid distance / mean spread, not raw um, because a 3 um gap
between two small cells separates more than the same gap between two large ones.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns
from fig_split_gene_vs_space import bal_acc, gene_loo, pos_loo, K
from fig_axis_separation import auc1d, lda_auc

OUT = C.Path(__file__).parent
RNG = np.random.default_rng(0)
MIN_TX, TYPES = 200, ["epithelial", "fibroblast", "myeloid"]
BINS = [0.0, 0.6, 0.9, 1.2, 1.6, 2.2, 3.0, 4.5, 99.0]
PER_BIN = 30


def main():
    sw = pd.read_csv(OUT / "switcher_pairs_3d.csv")
    swacc = pd.read_csv(OUT / "split_gene_vs_space.csv").set_index("cell_id")
    swaxis = pd.read_csv(OUT / "axis_separation.csv")
    swaxis = swaxis.loc[swaxis["group"] == "switcher"].set_index("pair")
    ann = pd.read_csv(C.RUN / "annotations_tracer_whole.csv").set_index("entity_id")["annotation"]

    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "_etype", "feature_name", "x", "y", "z"]))
    t["cell_id"] = t["cell_id"].astype(str); t["tracer_id"] = t["tracer_id"].astype(str)
    w = t.loc[t["_etype"].astype(str) == "cell"].copy()
    w["ann"] = w["tracer_id"].map(ann)
    w = w.loc[w["ann"].isin(TYPES)]

    def spread(g):
        a = g[["x", "y", "z"]].to_numpy(float)
        return np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean())

    gb = w.groupby("tracer_id")
    cen = gb.agg(n=("x", "size"), x=("x", "mean"), y=("y", "mean"), z=("z", "mean"),
                 ann=("ann", "first"))
    cen["spread"] = gb.apply(spread)
    cen = cen.loc[cen["n"] >= MIN_TX]
    ids = cen.index.to_numpy()
    xyz = cen[["x", "y", "z"]].to_numpy()
    sp = cen["spread"].to_numpy()

    # every whole-cell pair within 40 um, scored by sep
    prs = np.array(sorted(cKDTree(xyz).query_pairs(r=40.0)))
    d = np.linalg.norm(xyz[prs[:, 0]] - xyz[prs[:, 1]], axis=1)
    sep = d / ((sp[prs[:, 0]] + sp[prs[:, 1]]) / 2)
    pool = pd.DataFrame({"i": prs[:, 0], "j": prs[:, 1], "d3": d, "sep": sep})
    print(f"{len(pool)} whole-cell pairs within 40 um; sep range "
          f"{pool.sep.min():.2f}-{pool.sep.max():.2f}")
    print("closest whole-cell pair anywhere: sep %.3f, d3 %.2f um"
          % (pool.sep.min(), pool.loc[pool.sep.idxmin(), "d3"]))
    print("switcher pairs: sep median %.3f, d3 median %.2f um"
          % (sw.sep.median(), sw.d3.median()))
    print("control pairs with sep below the switcher median (%.2f): %d"
          % (sw.sep.median(), (pool.sep < sw.sep.median()).sum()))

    by = dict(tuple(w.groupby("tracer_id")))
    sizes = sw[["n_part", "n_rem"]].to_numpy()
    rows, used = [], set()
    pool["bin"] = pd.cut(pool["sep"], BINS, labels=False)
    for b, grp in pool.groupby("bin"):
        grp = grp.sample(frac=1.0, random_state=1)
        took = 0
        for r in grp.itertuples(index=False):
            a, bb = ids[r.i], ids[r.j]
            if a in used or bb in used:
                continue
            used.update([a, bb])
            ga, gbb = by[a], by[bb]
            nA, nB = sizes[took % len(sizes)]
            if len(ga) > nA:
                ga = ga.iloc[RNG.choice(len(ga), nA, replace=False)]
            if len(gbb) > nB:
                gbb = gbb.iloc[RNG.choice(len(gbb), nB, replace=False)]
            A = ga[["x", "y", "z"]].to_numpy(float); B = gbb[["x", "y", "z"]].to_numpy(float)
            y = np.r_[np.ones(len(A), int), np.zeros(len(B), int)]
            genes = np.r_[ga["feature_name"].astype(str).to_numpy(),
                          gbb["feature_name"].astype(str).to_numpy()]
            best, _ = lda_auc(A, B, np.random.default_rng(abs(hash(a)) % 2**31))
            rows.append(dict(group="control", sep=r.sep, d3=r.d3,
                             same_type=cen.loc[a, "ann"] == cen.loc[bb, "ann"],
                             acc_pos=bal_acc(y, pos_loo(np.vstack([A, B]), y)),
                             acc_gene=bal_acc(y, gene_loo(genes, y)), auc_best=best))
            took += 1
            if took == PER_BIN:
                break
    ct = pd.DataFrame(rows)

    sd = sw.set_index("cell_id")
    swr = pd.DataFrame(dict(group="switcher", sep=sd["sep"], d3=sd["d3"], same_type=False,
                            acc_pos=swacc["acc_pos"], acc_gene=swacc["acc_gene"],
                            auc_best=swaxis["auc_best"])).dropna().reset_index(drop=True)
    allr = pd.concat([ct, swr], ignore_index=True)
    allr.to_csv(OUT / "distance_matched.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    GC, GS = "#4575b4", "#984ea3"

    def curve(ax, col, ylab, title):
        ax.scatter(ct["sep"], ct[col], s=26, color=GC, alpha=0.65, linewidths=0,
                   label=f"control: two whole cells (n={len(ct)})")
        ax.scatter(swr["sep"], swr[col], s=34, color=GS, alpha=0.85, linewidths=0,
                   marker="D", label=f"switchers (n={len(swr)})")
        # binned median of the control curve
        b = pd.cut(ct["sep"], BINS)
        m = ct.groupby(b, observed=True).agg(sep=("sep", "median"), v=(col, "median")).dropna()
        ax.plot(m["sep"], m["v"], color=GC, lw=2.0, marker="o", ms=5,
                label="control median")
        # extrapolate the control trend down to the switcher range
        lo = ct.loc[ct["sep"] < 1.2]
        if len(lo) > 5:
            z = np.polyfit(lo["sep"], lo[col], 1)
            xs = np.linspace(swr["sep"].min(), 1.2, 20)
            ax.plot(xs, np.polyval(z, xs), color=GC, lw=1.4, ls="--",
                    label="control trend, extrapolated")
        ax.axhline(0.5, color="0.6", ls=":", lw=1.0)
        ax.axvspan(swr["sep"].min(), swr["sep"].max(), color=GS, alpha=0.07)
        ax.set_xscale("log"); ax.set_xlabel("sep = centroid distance / mean spread")
        ax.set_ylabel(ylab); ax.set_ylim(0.45, 1.03)
        ax.legend(fontsize=7.8, frameon=False, loc="lower right")
        ax.set_title(title, fontsize=10, loc="left")

    curve(axes[0], "acc_pos", f"position rule: LOO {K}-NN (bal. acc.)",
          "(a) position rule vs how far apart the pair is")
    curve(axes[1], "auc_best", "held-out AUC, best plane any orientation",
          "(b) best separating plane vs separation")

    # (c) distance-matched comparison, closest available controls
    ax = axes[2]
    cut = swr["sep"].quantile(0.9)
    lowc = ct.loc[ct["sep"] <= cut]
    vals = [lowc["acc_pos"].values, swr["acc_pos"].values,
            lowc["auc_best"].values, swr["auc_best"].values]
    cols = [GC, GS, GC, GS]
    bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.5))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    for k_, (v, c) in enumerate(zip(vals, cols), start=1):
        ax.scatter(np.full(len(v), k_) + RNG.normal(0, 0.06, len(v)), v, s=16, color=c,
                   alpha=0.8, linewidths=0)
    ax.axhline(0.5, color="0.6", ls=":", lw=1.0)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([f"control\nsep<={cut:.2f}\n(n={len(lowc)})", "switcher",
                        f"control\nsep<={cut:.2f}", "switcher"], fontsize=8)
    ax.set_ylim(0.45, 1.03); ax.set_ylabel("balanced acc. / AUC")
    ax.text(1.5, 1.0, "position rule", ha="center", fontsize=9, color="0.3")
    ax.text(3.5, 1.0, "best plane", ha="center", fontsize=9, color="0.3")
    p1 = mannwhitneyu(lowc["acc_pos"], swr["acc_pos"]).pvalue
    p2 = mannwhitneyu(lowc["auc_best"], swr["auc_best"]).pvalue
    ax.set_title(f"(c) closest available controls vs switchers\n"
                 f"position p={p1:.1e}, best plane p={p2:.1e}", fontsize=10, loc="left")

    fig.suptitle("Controlling for how far apart the two components are", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p = OUT / "fig_distance_matched.png"; fig.savefig(p, dpi=150); print("wrote", p)

    print("\ncontrol binned medians:")
    b = pd.cut(ct["sep"], BINS)
    print(ct.groupby(b, observed=True).agg(n=("sep", "size"), sep=("sep", "median"),
                                           acc_pos=("acc_pos", "median"),
                                           auc_best=("auc_best", "median")).round(3).to_string())
    print(f"\nswitchers: sep {swr.sep.median():.3f}, acc_pos {swr.acc_pos.median():.3f}, "
          f"auc_best {swr.auc_best.median():.3f}")
    print(f"controls with sep<={cut:.2f}: n={len(lowc)}, acc_pos {lowc.acc_pos.median():.3f}, "
          f"auc_best {lowc.auc_best.median():.3f}")


if __name__ == "__main__":
    main()
