"""Distance control for the remnant-vs-partial spatial tests (panels B4-B6 of
fig_claims_summary.py, on their own).

The adjacent-pair controls compared switcher parts (~1.6 um apart) with whole cells 6-26 um
apart. Here, whole-cell pairs across the whole range of separation give the expected
position-rule accuracy at each sep (centroid distance / mean spread):
  (a) separability is a function of separation; switchers sit on/below that curve
  (b) residual below the curve for switchers inside the control range (17/40)
  (c) sep of every whole-cell pair within 40 um vs the switcher pairs
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns

OUT = C.Path(__file__).parent
RNG = np.random.default_rng(0)
GC, GS = "#4575b4", "#984ea3"
MIN_TX, TYPES = 200, ["epithelial", "fibroblast", "myeloid"]


def main():
    sw = pd.read_csv(OUT / "switcher_pairs_3d.csv")
    dm = pd.read_csv(OUT / "distance_matched.csv")
    ann = pd.read_csv(C.RUN / "annotations_tracer_whole.csv").set_index("entity_id")["annotation"]
    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["tracer_id", "_etype", "x", "y", "z"]))
    t["tracer_id"] = t["tracer_id"].astype(str)
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

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    ct, sr = dm[dm.group == "control"], dm[dm.group == "switcher"]

    a = ax[0]
    a.scatter(ct["sep"], ct["acc_pos"], s=22, color=GC, alpha=.6, linewidths=0, label="two whole cells")
    a.scatter(sr["sep"], sr["acc_pos"], s=30, color=GS, alpha=.85, marker="D", linewidths=0,
              label="switcher remnant vs partial")
    b = pd.cut(ct["sep"], [0, .6, .9, 1.2, 1.6, 2.2, 3, 4.5, 99])
    m = ct.groupby(b, observed=True).agg(s=("sep", "median"), v=("acc_pos", "median")).dropna()
    a.plot(m["s"], m["v"], color=GC, lw=2, marker="o", ms=4)
    a.set_xscale("log"); a.axhline(.5, color="0.6", ls=":", lw=1)
    a.set_xlabel("sep = centroid distance / mean spread")
    a.set_ylabel("position rule (balanced accuracy)")
    a.legend(fontsize=8, frameon=False, loc="lower right")
    a.set_title("(a) separability follows separation\nswitchers fall on or just below the whole-cell curve",
                fontsize=10, loc="left")

    a = ax[1]
    ov = sr[(sr.sep >= ct.sep.min()) & (sr.sep <= 2.0)]
    x = np.log10(ct.sep.values); o = np.argsort(x)
    res = {}
    for col in ["acc_pos", "auc_best"]:
        s_ = UnivariateSpline(x[o], ct[col].values[o], s=len(x) * .004)
        res[col] = ov[col].values - s_(np.log10(ov.sep.values))
    cols = [GS, "#c994c7"]
    bp = a.boxplot([res["acc_pos"], res["auc_best"]], widths=.5, patch_artist=True,
                   showfliers=False, medianprops=dict(color="k", lw=1.5))
    for p_, c in zip(bp["boxes"], cols):
        p_.set_facecolor(c); p_.set_alpha(.4); p_.set_edgecolor(c)
    for k, (v, c) in enumerate(zip([res["acc_pos"], res["auc_best"]], cols), start=1):
        a.scatter(np.full(len(v), k) + RNG.normal(0, .06, len(v)), v, s=18, color=c, alpha=.8, linewidths=0)
    a.axhline(0, color="crimson", ls="--", lw=1.4, label="on the whole-cell curve")
    a.set_xticks([1, 2]); a.set_xticklabels(["position rule", "best plane"])
    a.set_ylabel("switcher minus whole-cell curve, at the same sep")
    a.legend(fontsize=8, frameon=False)
    a.set_title(f"(b) small residual at matched separation (n={len(ov)}/40 in range)\n"
                f"position rule {np.median(res['acc_pos']):+.3f} (p={wilcoxon(res['acc_pos']).pvalue:.0e}), "
                f"best plane {np.median(res['auc_best']):+.3f} (p={wilcoxon(res['auc_best']).pvalue:.2f})",
                fontsize=10, loc="left")

    a = ax[2]
    bins = np.logspace(np.log10(0.02), np.log10(12), 45)
    a.hist(pool_sep, bins=bins, weights=np.ones(len(pool_sep)) / len(pool_sep), color=GC, alpha=.8,
           label=f"all whole-cell pairs within 40 um (n={len(pool_sep)})")
    a.hist(sw["sep"], bins=bins, weights=np.ones(len(sw)) / len(sw), color=GS, alpha=.7,
           label=f"switcher remnant vs partial (n={len(sw)})")
    a.axvline(pool_sep.min(), color="crimson", ls="--", lw=1.6,
              label=f"closest whole-cell pair ({pool_sep.min():.2f})")
    a.set_xscale("log"); a.set_xlabel("sep = centroid distance / mean spread")
    a.set_ylabel("fraction of pairs"); a.legend(fontsize=7.5, frameon=False)
    a.set_title(f"(c) switcher parts sit closer than any two whole cells\n"
                f"{(sw['sep'] < pool_sep.min()).sum()}/{len(sw)} below the closest whole-cell pair",
                fontsize=10, loc="left")

    fig.suptitle("Distance control: the adjacent-pair comparison was not matched on separation", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = OUT / "fig_distance_control.png"; fig.savefig(p, dpi=140); print("wrote", p)


if __name__ == "__main__":
    main()
