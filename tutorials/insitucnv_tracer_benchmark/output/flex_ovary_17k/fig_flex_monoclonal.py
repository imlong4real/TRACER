"""Is the Flex tumour one clone or two?

Leiden on the epithelial cells' CNV profiles gave clone 0 (5,021) and clone 1 (2,315).
  (a) mean CNV profile of each clone along the genome; clone 1 fitted as alpha x clone 0;
      the residual (clone 1 - alpha x clone 0) underneath
  (b) per-cell CNV amplitude (projection onto the clone-0 profile) against UMIs per cell,
      for epithelial cells by clone and normal cells for reference
  (c) which expression clusters the clone-1 cells come from, with each cluster's
      epithelial marker score and median depth
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
COL = {"0": "#2166ac", "1": "#e66101", "normal": "0.6"}


def running_median(x, y, bins):
    idx = np.digitize(x, bins)
    xs, ys = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() >= 30:
            xs.append(np.median(x[m])); ys.append(np.median(y[m]))
    return np.array(xs), np.array(ys)


def main():
    c = ad.read_h5ad(OUT / "adata_flex_cnv.h5ad")
    o = c.obs.copy()
    o["n_umi"] = np.asarray(c.layers["counts"].sum(1)).ravel()
    X = c.obsm["X_cnv"]; X = X.toarray() if sp.issparse(X) else np.asarray(X)
    s0, s1 = (o.clone == "0").to_numpy(), (o.clone == "1").to_numpy()
    m0, m1 = X[s0].mean(0), X[s1].mean(0)
    alpha = (m1 @ m0) / (m0 @ m0); r = np.corrcoef(m0, m1)[0, 1]; res = m1 - alpha * m0
    print(f"clone1 vs clone0: r={r:.3f} alpha={alpha:.2f} residual SD {res.std():.4f} vs clone1 SD {m1.std():.4f}")
    amp = X @ m0 / (m0 @ m0)
    o["amp"] = amp
    print(o.groupby("clone")[["n_umi", "amp"]].median().round(3).to_string())

    fig = plt.figure(figsize=(20, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.25, 1], hspace=0.35, wspace=0.55)
    sub = gs[0, :].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.05)
    ax = fig.add_subplot(sub[0]); axr = fig.add_subplot(sub[1], sharex=ax)
    w = np.arange(len(m0))
    ax.plot(w, m0, color=COL["0"], lw=1.2, label=f"clone 0 (n={s0.sum():,})")
    ax.plot(w, m1, color=COL["1"], lw=1.2, label=f"clone 1 (n={s1.sum():,})")
    ax.plot(w, alpha * m0, color="k", lw=0.9, ls="--", label=f"{alpha:.2f} x clone 0 (best fit to clone 1)")
    axr.plot(w, res, color=COL["1"], lw=1.0)
    chr_pos = c.uns["cnv"]["chr_pos"]
    starts = sorted(chr_pos.items(), key=lambda kv: kv[1])
    for i, (ch, p) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(w)
        for a_ in (ax, axr):
            a_.axvline(p, color="0.85", lw=0.6)
        axr.text((p + end) / 2, -0.065, ch.replace("chr", ""), ha="center", fontsize=7,
                 transform=axr.get_xaxis_transform())
    for a_ in (ax, axr):
        a_.axhline(0, color="0.5", lw=0.6)
    ax.set_ylabel("mean inferCNV value"); ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax.set_title(f"(a) same shape, a third of the amplitude: clone 1 vs clone 0 r = {r:.2f}, best-fit scale {alpha:.2f}; "
                 f"residual SD {res.std():.3f} vs clone-1 profile SD {m1.std():.3f}", loc="left", fontsize=10)
    axr.set_ylabel("residual"); axr.set_xticks([]); axr.set_ylim(ax.get_ylim())

    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    bins = np.quantile(np.log10(o.n_umi), np.linspace(0, 1, 26))
    for cl in ["normal", "1", "0"]:
        m = (o.clone == cl).to_numpy()
        if cl == "normal":
            m &= rng.random(len(m)) < 0.3
        ax.scatter(o.n_umi[m], o.amp[m], s=3, alpha=0.35, color=COL[cl], lw=0,
                   label=f"{'normal (30% shown)' if cl == 'normal' else 'clone ' + cl}")
        mm = (o.clone == cl).to_numpy()
        xs, ys = running_median(np.log10(o.n_umi[mm].to_numpy()), o.amp[mm].to_numpy(), bins)
        ax.plot(10 ** xs, ys, color=COL[cl], lw=2.5)
    ax.set_xscale("log"); ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("UMIs per cell (autosomal genes)"); ax.set_ylabel("CNV amplitude (projection onto clone-0 profile)")
    ax.legend(fontsize=8, frameon=False, markerscale=4)
    ax.set_title("(b) amplitude rises with depth; clone 1 sits below clone 0 even at the same depth\n"
                 "(lines = running medians)  clone 0 median "
                 f"{o.loc[s0, 'n_umi'].median():.0f} UMIs, amplitude {o.loc[s0, 'amp'].median():.2f}; clone 1 "
                 f"{o.loc[s1, 'n_umi'].median():.0f} UMIs, amplitude {o.loc[s1, 'amp'].median():.2f}",
                 loc="left", fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    cm = pd.read_csv(OUT / "celltype_markers.csv", index_col=0); cm.index = cm.index.astype(str)
    epi = o[o.clone.isin(["0", "1"])]
    tab = pd.crosstab(epi.leiden.astype(str), epi.clone)
    tab = tab.loc[tab.sum(1) >= 50]
    frac1 = tab["1"] / tab.sum(1)
    order = frac1.sort_values().index
    y = np.arange(len(order))
    ax.barh(y, tab.loc[order, "0"], color=COL["0"], label="clone 0")
    ax.barh(y, tab.loc[order, "1"], left=tab.loc[order, "0"], color=COL["1"], label="clone 1")
    med_umi = epi.groupby(epi.leiden.astype(str))["n_umi"].median()
    ax.set_yticks(y, [f"cl {k}: {100*frac1[k]:.0f}% clone 1 | epi z {cm.loc[k, 'epithelial']:+.2f} | "
                      f"{med_umi[k]:.0f} UMIs" for k in order], fontsize=8)
    ax.set_xlabel("epithelial cells"); ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("(c) clone 1 = the expression clusters with weak epithelial\nscores and low depth",
                 loc="left", fontsize=10)

    fig.suptitle("Flex ovarian tumour: one clone, not two -- clone 1 is clone 0's profile at lower amplitude",
                 fontsize=13)
    fig.savefig(OUT / "fig_flex_monoclonal.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / "fig_flex_monoclonal.png")


if __name__ == "__main__":
    main()
