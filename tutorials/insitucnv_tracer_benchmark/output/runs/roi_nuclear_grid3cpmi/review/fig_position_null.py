"""Random-label null for the position rule (PI's design), for every switcher.

For each switcher (remnant vs top epithelial partial, transcripts inside the Xenium
cell; n = nA + nB from switcher_class_separation.csv):
  draw a single, intact Xenium (10x original) cell that is NOT one of the 163
  switchers and holds >= n transcripts; subsample exactly n of its transcripts;
  assign RANDOM labels with the switcher's own split (nA ones, nB zeros); run the
  identical LOO 10-NN position rule (fig_split_gene_vs_space.pos_loo / bal_acc).
Repeated N_NULL times per switcher -> a size- and balance-matched null distribution
of balanced accuracy under "the two labels have no spatial structure".

Tests
  (a) empirical p per switcher: (1 + #null >= obs) / (N_NULL + 1), BH-FDR over all
  (b) paired Wilcoxon signed-rank: observed vs that switcher's null median, overall
      and per class
Context panel: the random null is the FLOOR (no structure); two real adjacent whole
cells at the same separation are the CEILING (distance_matched.csv). Beating the
floor shows spatial structure exists; it does not by itself show two separate cells.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns
from fig_split_gene_vs_space import bal_acc, pos_loo

OUT = C.Path(__file__).parent
N_NULL = 1000
MIN_POOL = 20
CLASSES = ["fibroblast", "myeloid", "t_cell"]
CLAB = {"fibroblast": "fibroblast", "myeloid": "myeloid", "t_cell": "T cell"}
CC = {"fibroblast": "#984ea3", "myeloid": "#ff7f00", "t_cell": "#e31a1c"}


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1.0)
    return out


def main():
    rng = np.random.default_rng(0)
    sw = pd.read_csv(OUT / "switcher_class_separation.csv")
    all_sw = set(pd.read_csv(OUT / "immune_switchers.csv")["cell_id"])

    t = standardize_transcript_columns(pd.read_parquet(
        C.BENCH / "data/roi_transcripts.parquet", columns=["cell_id", "x", "y", "z"]))
    t["cell_id"] = t["cell_id"].astype(str)
    t = t.loc[~t["cell_id"].isin(UNASSIGNED_TOKENS) & ~t["cell_id"].isin(all_sw)]
    cells = {c: g[["x", "y", "z"]].to_numpy(float) for c, g in t.groupby("cell_id")}
    size = pd.Series({c: len(v) for c, v in cells.items()})
    print(f"null pool: {len(size)} non-switcher Xenium cells, median {size.median():.0f} tx")

    rows, null_store = [], {}
    for r in sw.itertuples(index=False):
        n, nA = int(r.nA + r.nB), int(r.nA)
        pool = size.index[size >= n].to_numpy()
        if len(pool) < MIN_POOL:                    # very large switchers: take the biggest cells
            pool = size.sort_values(ascending=False).index[:MIN_POOL].to_numpy()
        y = np.r_[np.ones(nA, int), np.zeros(n - nA, int)]
        null = np.empty(N_NULL)
        for k in range(N_NULL):
            X = cells[pool[rng.integers(len(pool))]]
            idx = rng.choice(len(X), size=n, replace=len(X) < n)
            yl = rng.permutation(y)
            null[k] = bal_acc(yl, pos_loo(X[idx], yl))
        null_store[r.cell_id] = null
        rows.append(dict(cls=r.cls, cell_id=r.cell_id, n=n, nA=nA, sep=r.sep, obs=r.acc_pos,
                         null_med=np.median(null), null_lo=np.quantile(null, .025),
                         null_hi=np.quantile(null, .975), null_sd=null.std(),
                         pool=len(pool), p_emp=(1 + (null >= r.acc_pos).sum()) / (N_NULL + 1)))
    res = pd.DataFrame(rows)
    res["q_bh"] = bh(res["p_emp"])
    res["z"] = (res["obs"] - res["null_med"]) / res["null_sd"]
    res.to_csv(OUT / "position_null.csv", index=False)
    np.savez_compressed(OUT / "position_null_draws.npz", **null_store)

    print(f"pool size per switcher: min {res.pool.min()}, median {res.pool.median():.0f}; "
          f"{int((res.pool == MIN_POOL).sum())} switchers fell back to the {MIN_POOL} largest cells")
    print(f"null median {res.null_med.median():.3f}, null SD median {res.null_sd.median():.4f}")
    w_all = wilcoxon(res["obs"] - res["null_med"])
    print(f"paired Wilcoxon, all {len(res)}: median obs-null {np.median(res.obs - res.null_med):+.3f}, "
          f"p={w_all.pvalue:.2e}")
    summ = []
    for c in CLASSES:
        d = res.loc[res["cls"] == c]
        w = wilcoxon(d["obs"] - d["null_med"])
        summ.append(dict(cls=c, n=len(d), obs_med=d.obs.median(), null_med=d.null_med.median(),
                         z_med=d.z.median(), wilcoxon_p=w.pvalue,
                         sig_q05=int((d.q_bh < 0.05).sum()), min_p=d.p_emp.min()))
    summ = pd.DataFrame(summ).set_index("cls")
    print(summ.round(4).to_string())

    # ---------------- figure ----------------
    fig, axs = plt.subplots(1, 3, figsize=(19.5, 6.0), gridspec_kw=dict(width_ratios=[1.6, 0.9, 1.1]))

    a = axs[0]
    x0 = 0
    for c in CLASSES:
        d = res.loc[res["cls"] == c].sort_values("obs").reset_index(drop=True)
        xs = x0 + np.arange(len(d))
        a.vlines(xs, d["null_lo"], d["null_hi"], color="0.7", lw=2.2)
        a.scatter(xs, d["null_med"], s=8, color="0.45", zorder=3)
        a.scatter(xs, d["obs"], s=26, color=CC[c], zorder=4, label=f"{CLAB[c]} (n={len(d)})")
        a.text(x0 + len(d) / 2, 0.79, CLAB[c], ha="center", fontsize=9, color=CC[c])
        x0 += len(d) + 4
    a.set_xticks([]); a.set_xlabel("switchers, sorted by observed accuracy within class")
    a.set_ylabel("position rule (balanced accuracy)"); a.set_ylim(0.44, 0.8)
    a.axhline(0.5, color="0.6", ls=":", lw=1)
    a.legend(handles=[plt.Line2D([], [], color="0.7", lw=3, label="null 95% interval (random labels)"),
                      plt.Line2D([], [], marker="o", ls="", color="0.45", ms=4, label="null median")]
             + a.get_legend_handles_labels()[0], fontsize=8, frameon=False, loc="lower right")
    a.set_title(f"(a) each switcher vs its own size- and balance-matched random-label null "
                f"({N_NULL} draws)\npaired Wilcoxon, all: median obs - null "
                f"{np.median(res.obs - res.null_med):+.3f}, p={w_all.pvalue:.1e}",
                fontsize=10, loc="left")

    a = axs[1]
    for k, c in enumerate(CLASSES):
        d = res.loc[res["cls"] == c]
        v = -np.log10(d["p_emp"])
        a.scatter(np.full(len(v), k) + np.random.default_rng(1).normal(0, 0.07, len(v)), v,
                  s=18, color=CC[c], alpha=0.8, linewidths=0)
        a.text(k, -np.log10(1 / (N_NULL + 1)) + 0.25,
               f"{int(summ.loc[c, 'sig_q05'])}/{len(d)}\nq<0.05", ha="center", fontsize=8.5)
    thr = res.loc[res["q_bh"] < 0.05, "p_emp"].max() if (res["q_bh"] < 0.05).any() else 0.05 / len(res)
    a.axhline(-np.log10(thr), color="crimson", ls="--", lw=1.2, label="BH FDR 5%")
    a.axhline(-np.log10(1 / (N_NULL + 1)), color="0.6", ls=":", lw=1,
              label=f"floor of resolution (1/{N_NULL + 1})")
    a.set_xticks(range(3)); a.set_xticklabels([CLAB[c] for c in CLASSES])
    a.set_ylabel("-log10 empirical p"); a.set_ylim(-0.1, -np.log10(1 / (N_NULL + 1)) + 0.9)
    a.legend(fontsize=8, frameon=False, loc="center right")
    a.set_title("(b) empirical p per switcher", fontsize=10, loc="left")

    a = axs[2]
    dm = pd.read_csv(OUT / "distance_matched.csv")
    ct = dm.loc[dm["group"] == "control"]
    a.scatter(ct["sep"], ct["acc_pos"], s=12, color="0.75", alpha=0.6, linewidths=0,
              label="two real whole cells (ceiling)")
    bb = pd.cut(ct["sep"], [0, .6, .9, 1.2, 1.6, 2.2, 3, 4.5, 99])
    mc = ct.groupby(bb, observed=True).agg(s=("sep", "median"), v=("acc_pos", "median")).dropna()
    a.plot(mc["s"], mc["v"], color="0.35", lw=2, marker="o", ms=4)
    o = np.argsort(res["sep"].values)
    a.fill_between(res["sep"].values[o], res["null_lo"].values[o], res["null_hi"].values[o],
                   color="#9ecae1", alpha=0.5, label="random-label null 95% (floor)")
    for c in CLASSES:
        d = res.loc[res["cls"] == c]
        a.scatter(d["sep"], d["obs"], s=26, color=CC[c], marker="D", alpha=0.9, linewidths=0,
                  label=f"{CLAB[c]} switchers")
    a.set_xscale("log"); a.set_xlabel("sep = centroid distance / mean spread")
    a.set_ylabel("position rule (balanced accuracy)")
    a.legend(fontsize=7.5, frameon=False, loc="upper left")
    a.set_title("(c) between the floor and the ceiling", fontsize=10, loc="left")

    fig.suptitle("Position rule against a random-label null built from single, intact Xenium cells "
                 "of matched size", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / "fig_position_null.png"; fig.savefig(p, dpi=150); print("wrote", p)


if __name__ == "__main__":
    main()
