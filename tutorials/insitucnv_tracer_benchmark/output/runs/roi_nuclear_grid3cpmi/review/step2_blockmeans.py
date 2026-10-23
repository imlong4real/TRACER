"""Block-mean CNV profiles along the genome -- the aggregated read of the heatmap.

87.6% of per-cell X_cnv entries are exactly 0 (infercnvpy dynamic_threshold=1.5),
so the per-cell heatmap shows threshold crossings. Averaging within a block
recovers the profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import cnv_review_common as C
from step1_heatmap import BLOCK_ORDER, block_of, genome_axis

OUT = C.Path(__file__).parent
LINES = {
    "reference": ("#2b8000", "-"),
    "tumor / clone 0": ("#4575b4", "-"),
    "tumor / clone 4": ("#d73027", "-"),
}


def block_means(adata, X):
    blocks = adata.obs.apply(block_of, axis=1).to_numpy()
    out = {}
    for b in BLOCK_ORDER:
        m = blocks == b
        if m.sum() == 0:
            continue
        out[b] = (X[m].mean(axis=0), int(m.sum()))
    return out


def gx(wins, ext, offsets):
    return np.array([offsets[c] + (m - ext.loc[c, "lo"]) for c, m in
                     zip(wins["chromosome"], wins["bp_mid"])], dtype=float)


def main() -> None:
    data = {arm: C.load_arm(arm) for arm in ("raw", "tracer_whole")}
    wins_by_arm = {a: w for a, (_, w) in data.items()}
    ext, offsets, total = genome_axis(wins_by_arm)

    bm = {}
    for arm, (adata, wins) in data.items():
        bm[arm] = block_means(adata, C.dense_cnv(adata))
    vmax = max(np.abs(v[0]).max() for arm in bm for v in bm[arm].values())

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.9, 1.0, 0.9], hspace=0.30,
                          left=0.135, right=0.975, top=0.94, bottom=0.07)

    for k, (arm, title) in enumerate([("raw", "Xenium (original segmentation)"),
                                      ("tracer_whole", "TRACER-refined (whole cells)")]):
        wins = wins_by_arm[arm]
        x = gx(wins, ext, offsets)
        blocks = [b for b in BLOCK_ORDER if b in bm[arm]]
        M = np.vstack([bm[arm][b][0] for b in blocks])

        ax = fig.add_subplot(gs[2 * k])
        ax.pcolormesh(np.concatenate([[0], (x[:-1] + x[1:]) / 2, [total]]),
                      np.arange(len(blocks) + 1), M, cmap="RdBu_r",
                      norm=Normalize(-vmax, vmax), rasterized=True)
        ax.set_yticks(np.arange(len(blocks)) + 0.5)
        ax.set_yticklabels([f"{b}  (n={bm[arm][b][1]})" for b in blocks], fontsize=8)
        ax.invert_yaxis()
        for c in ext.index[1:]:
            ax.axvline(offsets[c], color="k", lw=0.5)
        ax.set_xticks([]); ax.set_xlim(0, total)
        ax.set_title(f"{title} -- mean CNV per row-block", fontsize=10, loc="left")

        axl = fig.add_subplot(gs[2 * k + 1])
        for b, (col, ls) in LINES.items():
            if b in bm[arm]:
                axl.plot(x, bm[arm][b][0], color=col, ls=ls, lw=0.9, label=f"{b} (n={bm[arm][b][1]})")
        ref = bm[arm]["reference"][0]
        axl.fill_between(x, -2 * ref.std(), 2 * ref.std(), color="0.85", zorder=0,
                         label=f"+/-2 SD of reference profile ({2*ref.std():.4f})")
        axl.axhline(0, color="k", lw=0.5)
        for c in ext.index[1:]:
            axl.axvline(offsets[c], color="k", lw=0.4, alpha=0.4)
        centers = [offsets[c] + (ext.loc[c, "hi"] - ext.loc[c, "lo"]) / 2 for c in ext.index]
        axl.set_xticks(centers)
        axl.set_xticklabels([c.replace("chr", "") for c in ext.index], fontsize=7)
        axl.set_xlim(0, total)
        axl.set_ylim(-vmax * 1.05, vmax * 1.05)
        axl.set_ylabel("mean CNV", fontsize=8)
        axl.tick_params(labelsize=7)
        axl.legend(fontsize=7, ncol=4, frameon=False, loc="upper left")
        if k == 1:
            axl.set_xlabel("chromosome", fontsize=9)

    fig.suptitle("Mean inferCNV profile by row-block, raw vs TRACER -- ovary ROI "
                 "(roi_nuclear_grid3cpmi, expression-kNN smoothing k=100)", fontsize=12, y=0.975)
    out = OUT / "cnv_blockmean_profiles.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)

    # clone 4 vs clone 0 contrast, per chromosome, both arms
    rows = []
    for arm in bm:
        wins = wins_by_arm[arm]
        ch = wins["chromosome"].to_numpy()
        for b in ("reference", "tumor / clone 0", "tumor / clone 4"):
            if b not in bm[arm]:
                continue
            s = pd.Series(bm[arm][b][0], index=ch).groupby(level=0).mean()
            rows.append(pd.Series(s, name=(arm, b)))
    tab = pd.DataFrame(rows).T
    tab = tab.reindex(sorted(tab.index, key=C.chromosome_rank))
    pd.set_option("display.width", 220)
    print("\nMean CNV per chromosome (block means):")
    print(tab.round(4).to_string())
    tab.round(5).to_csv(OUT / "cnv_blockmean_by_chrom.csv")


if __name__ == "__main__":
    main()
