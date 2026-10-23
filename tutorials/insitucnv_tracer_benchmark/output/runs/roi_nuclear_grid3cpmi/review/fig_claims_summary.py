"""One panel per numbered claim, for point-by-point review.

Rebuilt from the saved CSVs of each test plus a single pass over the transcript
table. Panel letters match the claim numbers used in the review:

  A5  partials are multi-nucleus aggregates      (recomputed here, not inherited)
  B1a gene split is all-or-nothing
  B1b gene beats position within a cell
  B3  the gene rule is reading cell type
  B4  separability is a function of separation   (this is what killed W1)
  B5  residual below the control curve at matched separation
  B6  switchers sit closer than any two whole cells
  B8  worked example
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom, mannwhitneyu, wilcoxon
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns

OUT = C.Path(__file__).parent
RNG = np.random.default_rng(0)
CR, CP, GC, GS = "#d73027", "#1b7837", "#4575b4", "#984ea3"
EXAMPLE = "dplafofh-1"
MIN_TX, TYPES = 200, ["epithelial", "fibroblast", "myeloid"]


def title(ax, tag, text, sub=""):
    ax.set_title(f"{tag}  {text}" + (f"\n{sub}" if sub else ""), fontsize=9.5, loc="left")


def main():
    sw = pd.read_csv(OUT / "switcher_pairs_3d.csv")
    swacc = pd.read_csv(OUT / "split_gene_vs_space.csv")
    combos = pd.read_csv(OUT / "control_type_combos.csv")
    dm = pd.read_csv(OUT / "distance_matched.csv")
    ann = pd.read_csv(C.RUN / "annotations_tracer_whole.csv").set_index("entity_id")["annotation"]

    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "_etype", "feature_name", "x", "y", "z",
                 "overlaps_nucleus"]))
    t["cell_id"] = t["cell_id"].astype(str); t["tracer_id"] = t["tracer_id"].astype(str)

    # ---- A5: distinct nuclei per switcher-recipient partial (RECOMPUTED) ----
    recips = sorted(set(sw["partial"]))
    nuc = t.loc[(t["overlaps_nucleus"] == 1) & (t["cell_id"] != "UNASSIGNED")]
    rn = nuc.loc[nuc["tracer_id"].isin(recips)]
    per = rn.groupby(["tracer_id", "cell_id"]).size().rename("n").reset_index()
    n_nuclei = per.groupby("tracer_id").size()
    n_nuc50 = per.loc[per["n"] >= 50].groupby("tracer_id").size().reindex(n_nuclei.index).fillna(0)
    print(f"A5 recomputed: {len(n_nuclei)} recipient partials, median distinct nuclei "
          f"{n_nuclei.median():.0f}; median with >=50 tx {n_nuc50.median():.0f}; "
          f">=2 nuclei at >=50: {(n_nuc50 >= 2).sum()}/{len(n_nuc50)}")

    # ---- B1a: per-gene partial fraction ----
    keep = t.loc[t["cell_id"].isin(set(sw["cell_id"]))]
    by_cell = dict(tuple(keep.groupby("cell_id")))
    gf = []
    for r in sw.itertuples(index=False):
        g = by_cell[r.cell_id]
        lab = np.where(g["tracer_id"].to_numpy() == r.partial, 1,
                       np.where(g["tracer_id"].to_numpy() == r.cell_id, 0, -1))
        m = lab >= 0
        d = pd.DataFrame({"g": g["feature_name"].astype(str).to_numpy()[m], "y": lab[m]})
        p_cell = d["y"].mean()
        agg = d.groupby("g")["y"].agg(["size", "sum"])
        agg = agg.loc[agg["size"] >= 10]
        for n, s in agg.itertuples(index=False):
            gf.append((s / n, n, p_cell))
    gf = pd.DataFrame(gf, columns=["frac", "n", "p_cell"])

    # ---- B6: sep of every whole-cell pair ----
    w = t.loc[t["_etype"].astype(str) == "cell"].copy()
    w["ann"] = w["tracer_id"].map(ann)
    w = w.loc[w["ann"].isin(TYPES)]
    gb = w.groupby("tracer_id")
    cen = gb.agg(n=("x", "size"), x=("x", "mean"), y=("y", "mean"), z=("z", "mean"))
    cen["spread"] = gb.apply(lambda g: np.sqrt(
        ((g[["x", "y", "z"]].to_numpy(float) - g[["x", "y", "z"]].to_numpy(float).mean(0)) ** 2)
        .sum(1).mean()))
    cen = cen.loc[cen["n"] >= MIN_TX]
    xyz = cen[["x", "y", "z"]].to_numpy(); spv = cen["spread"].to_numpy()
    prs = np.array(sorted(cKDTree(xyz).query_pairs(r=40.0)))
    dd = np.linalg.norm(xyz[prs[:, 0]] - xyz[prs[:, 1]], axis=1)
    pool_sep = dd / ((spv[prs[:, 0]] + spv[prs[:, 1]]) / 2)

    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    ax = axes.ravel()

    # A5
    a = ax[0]
    a.hist(n_nuclei.values, bins=np.arange(0.5, max(n_nuclei) + 1.5), color=CP, alpha=0.85)
    a.axvline(1, color="crimson", ls="--", lw=1.5, label="1 = it is one cell")
    a.set_xlabel("distinct Xenium nuclei feeding one epithelial partial")
    a.set_ylabel("recipient partials"); a.legend(fontsize=8, frameon=False)
    title(a, "A5", "the partial is a multi-nucleus aggregate",
          f"median {n_nuclei.median():.0f} nuclei; {(n_nuc50>=2).sum()}/{len(n_nuc50)} draw "
          f">=50 tx from 2+ nuclei   [LOAD-BEARING]")

    # B1a
    a = ax[1]
    bins = np.linspace(0, 1, 21)
    a.hist(gf["frac"], bins=bins, weights=np.ones(len(gf)) / len(gf), color=GC, alpha=0.9,
           label=f"observed (n={len(gf)})")
    null = np.concatenate([binom.rvs(int(n), p, size=20, random_state=1) / n
                           for n, p in zip(gf["n"], gf["p_cell"])])
    a.hist(null, bins=bins, weights=np.ones(len(null)) / len(null), histtype="step",
           color="crimson", lw=1.8, label="shuffled within cell")
    ex = ((gf["frac"] < .02) | (gf["frac"] > .98)).mean()
    exn = ((null < .02) | (null > .98)).mean()
    a.set_xlabel("fraction of a gene's transcripts sent to the partial")
    a.set_ylabel("fraction of gene x cell pairs"); a.legend(fontsize=8, frameon=False)
    title(a, "B1a", "the split is all-or-nothing per gene",
          f"{100*ex:.0f}% at the extremes vs {100*exn:.0f}% null   [circular: C1]")

    # B1b
    a = ax[2]
    a.scatter(swacc["acc_pos"], swacc["acc_gene"], s=40, color=GS, alpha=0.85, linewidths=0)
    a.scatter(swacc["null_pos"], swacc["null_gene"], s=24, color="0.6", marker="x", alpha=0.8)
    a.plot([.45, 1.02], [.45, 1.02], ls=":", color="0.6", lw=.9)
    a.axhline(.5, color="0.85", lw=.7); a.axvline(.5, color="0.85", lw=.7)
    a.set_xlim(.45, 1.02); a.set_ylim(.45, 1.02)
    a.set_xlabel("predict split from POSITION (bal. acc.)")
    a.set_ylabel("predict split from GENE (bal. acc.)")
    title(a, "B1b", "gene beats position within a cell",
          f"medians {swacc.acc_gene.median():.2f} vs {swacc.acc_pos.median():.2f}; "
          f"grey x = shuffled   [level is confounded: C3]")

    # B3
    a = ax[3]
    st = combos.loc[combos["same_type"], "acc_gene"].values
    dt = combos.loc[~combos["same_type"], "acc_gene"].values
    vals = [st, dt, swacc["acc_gene"].values]
    cols = ["#92c5de", "#d6604d", GS]
    bp = a.boxplot(vals, widths=.6, patch_artist=True, showfliers=False,
                   medianprops=dict(color="k", lw=1.5))
    for p_, c in zip(bp["boxes"], cols):
        p_.set_facecolor(c); p_.set_alpha(.4); p_.set_edgecolor(c)
    for k, (v, c) in enumerate(zip(vals, cols), start=1):
        a.scatter(np.full(len(v), k) + RNG.normal(0, .06, len(v)), v, s=12, color=c,
                  alpha=.75, linewidths=0)
    a.axhline(.5, color="0.5", ls="--", lw=1)
    a.set_xticks([1, 2, 3]); a.set_xticklabels(
        [f"same type\n(n={len(st)})", f"different types\n(n={len(dt)})", "SWITCHERS"],
        fontsize=8.5)
    a.set_ylabel("gene rule (bal. acc.)"); a.set_ylim(.45, 1.0)
    title(a, "B3", "the gene rule is reading CELL TYPE",
          f"{np.median(st):.2f} vs {np.median(dt):.2f}, p={mannwhitneyu(st,dt).pvalue:.0e}"
          f"   [not distance-confounded]")

    # B4
    a = ax[4]
    ct, sr = dm[dm.group == "control"], dm[dm.group == "switcher"]
    a.scatter(ct["sep"], ct["acc_pos"], s=22, color=GC, alpha=.6, linewidths=0,
              label="two whole cells")
    a.scatter(sr["sep"], sr["acc_pos"], s=30, color=GS, alpha=.85, marker="D", linewidths=0,
              label="switchers")
    b = pd.cut(ct["sep"], [0, .6, .9, 1.2, 1.6, 2.2, 3, 4.5, 99])
    m = ct.groupby(b, observed=True).agg(s=("sep", "median"), v=("acc_pos", "median")).dropna()
    a.plot(m["s"], m["v"], color=GC, lw=2, marker="o", ms=4)
    a.set_xscale("log"); a.axhline(.5, color="0.6", ls=":", lw=1)
    a.set_xlabel("sep = centroid distance / mean spread")
    a.set_ylabel("position rule (bal. acc.)"); a.legend(fontsize=8, frameon=False, loc="lower right")
    title(a, "B4", "separability is a FUNCTION of separation",
          "this is what invalidated W1/W2   [distance confound]")

    # B5
    a = ax[5]
    lo = ct.sep.min()
    ov = sr[(sr.sep >= lo) & (sr.sep <= 2.0)]
    x = np.log10(ct.sep.values); o = np.argsort(x)
    res = {}
    for col, c in [("acc_pos", GS), ("auc_best", "#c994c7")]:
        s_ = UnivariateSpline(x[o], ct[col].values[o], s=len(x) * .004)
        res[col] = ov[col].values - s_(np.log10(ov.sep.values))
    bp = a.boxplot([res["acc_pos"], res["auc_best"]], widths=.5, patch_artist=True,
                   showfliers=False, medianprops=dict(color="k", lw=1.5))
    for p_, c in zip(bp["boxes"], [GS, "#c994c7"]):
        p_.set_facecolor(c); p_.set_alpha(.4); p_.set_edgecolor(c)
    for k, (v, c) in enumerate([(res["acc_pos"], GS), (res["auc_best"], "#c994c7")], start=1):
        a.scatter(np.full(len(v), k) + RNG.normal(0, .06, len(v)), v, s=18, color=c,
                  alpha=.8, linewidths=0)
    a.axhline(0, color="crimson", ls="--", lw=1.4, label="on the control curve")
    a.set_xticks([1, 2]); a.set_xticklabels(["position rule", "best plane"], fontsize=9)
    a.set_ylabel("switcher - control trend, at matched sep")
    a.legend(fontsize=8, frameon=False)
    title(a, "B5", "small residual below the curve",
          f"{np.median(res['acc_pos']):+.3f} (p={wilcoxon(res['acc_pos']).pvalue:.0e}) and "
          f"{np.median(res['auc_best']):+.3f} (p={wilcoxon(res['auc_best']).pvalue:.2f}), n={len(ov)}/40")

    # B6
    a = ax[6]
    bins = np.logspace(np.log10(0.02), np.log10(12), 45)
    a.hist(pool_sep, bins=bins, weights=np.ones(len(pool_sep)) / len(pool_sep), color=GC,
           alpha=.8, label=f"all whole-cell pairs (n={len(pool_sep)})")
    a.hist(sw["sep"], bins=bins, weights=np.ones(len(sw)) / len(sw), color=GS, alpha=.7,
           label="switcher remnant/partial (n=40)")
    a.axvline(pool_sep.min(), color="crimson", ls="--", lw=1.6,
              label=f"closest whole-cell pair ({pool_sep.min():.2f})")
    a.set_xscale("log"); a.set_xlabel("sep = centroid distance / mean spread")
    a.set_ylabel("fraction of pairs"); a.legend(fontsize=7.5, frameon=False)
    title(a, "B6", "switchers sit closer than any two whole cells",
          f"{(sw['sep'] < pool_sep.min()).sum()}/40 below the global minimum   "
          f"[may be algorithmic: C2]")

    # B8
    a = ax[7]
    r = sw.set_index("cell_id").loc[EXAMPLE]
    g = by_cell[EXAMPLE]
    mp = g["tracer_id"].to_numpy() == r["partial"]
    mr = g["tracer_id"].to_numpy() == EXAMPLE
    a.scatter(g.loc[mr, "x"], g.loc[mr, "y"], s=8, color=CR, alpha=.55, linewidths=0,
              label=f"remnant -> 'fibroblast' ({int(r.n_rem)})")
    a.scatter(g.loc[mp, "x"], g.loc[mp, "y"], s=8, color=CP, alpha=.75, linewidths=0,
              label=f"epithelial partial ({int(r.n_part)})")
    a.set_aspect("equal"); a.set_xlabel("x (um)"); a.set_ylabel("y (um)")
    a.legend(fontsize=7.5, frameon=False, markerscale=2)
    title(a, "B8", f"worked example {EXAMPLE}",
          f"18 epi markers all green, 5 fib markers all red; CNV r {r.r_rem:+.2f} vs {r.r_part:+.2f}")

    fig.suptitle("Claim-by-claim evidence  |  A5 load-bearing; B1/B3 describe the mechanism; "
                 "B4 shows why W1/W2 were withdrawn; B5/B6 are what survives the distance control",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = OUT / "fig_claims_summary.png"; fig.savefig(p, dpi=140); print("wrote", p)


if __name__ == "__main__":
    main()
