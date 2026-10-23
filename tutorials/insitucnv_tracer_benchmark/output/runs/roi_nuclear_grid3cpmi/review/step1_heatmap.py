"""inferCNV heatmap: entities x genomic windows, both arms on a common bp axis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, LogNorm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

import cnv_review_common as C

OUT = C.Path(__file__).parent
CLONE_COLORS = {
    "0": "#4575b4", "4": "#d73027", "1": "#fdae61", "2": "#74add1",
    "3": "#fee090", "not_epithelial": "#d9d9d9", "no_truth": "#f7f7f7",
}
COMPARTMENT_COLORS = {"reference": "#7fbc41", "tumor": "#c51b7d", "unknown": "#bababa"}


def block_of(row) -> str:
    comp = str(row["compartment"])
    truth = str(row["truth_clone"])
    if comp == "reference":
        return "reference"
    if comp == "unknown":
        return "unknown"
    if truth in ("0", "4"):
        return f"tumor / clone {truth}"
    if truth in ("1", "2", "3"):
        return "tumor / clone 1,2,3"
    return "tumor / non-epi truth"


BLOCK_ORDER = [
    "reference",
    "tumor / clone 0",
    "tumor / clone 4",
    "tumor / clone 1,2,3",
    "tumor / non-epi truth",
    "unknown",
]


def order_rows(adata, X):
    """Block order fixed; within a block, hierarchical clustering on CNV."""
    blocks = adata.obs.apply(block_of, axis=1).to_numpy()
    idx, bounds, labels = [], [], []
    for b in BLOCK_ORDER:
        sel = np.flatnonzero(blocks == b)
        if sel.size == 0:
            continue
        if sel.size > 2:
            sub = X[sel]
            # Correlation distance is undefined for constant rows (an entity
            # whose whole CNV profile is flat). Cluster the rest, then append
            # the flat ones at the block's bottom rather than dropping them.
            var_ok = sub.std(axis=1) > 0
            if var_ok.sum() > 2:
                d = pdist(sub[var_ok], metric="correlation")
                d[~np.isfinite(d)] = np.nanmax(d[np.isfinite(d)]) if np.isfinite(d).any() else 0.0
                order = leaves_list(linkage(d, method="average"))
                sel = np.concatenate([sel[var_ok][order], sel[~var_ok]])
        idx.append(sel)
        labels.append(b)
        bounds.append(sel.size)
    return np.concatenate(idx), labels, np.cumsum([0] + bounds)


def genome_axis(wins_by_arm):
    """Common per-chromosome offsets across both arms."""
    allw = pd.concat(wins_by_arm.values())
    ext = allw.groupby("chromosome").agg(lo=("bp_start", "min"), hi=("bp_end", "max"))
    ext = ext.reindex(sorted(ext.index, key=C.chromosome_rank))
    span = (ext["hi"] - ext["lo"]).to_numpy(float)
    off = np.concatenate([[0.0], np.cumsum(span)])
    return ext, dict(zip(ext.index, off[:-1])), off[-1]


def chrom_edges(mids, lo, hi):
    """Non-overlapping cell edges from window midpoints (windows overlap 83%)."""
    if len(mids) == 1:
        return np.array([lo, hi], dtype=float)
    inner = (mids[:-1] + mids[1:]) / 2.0
    return np.concatenate([[lo], inner, [hi]])


def draw_panel(fig, gs, adata, wins, ext, offsets, vmax, title, show_x):
    X = C.dense_cnv(adata)
    row_idx, labels, bounds = order_rows(adata, X)
    X = X[row_idx]
    obs = adata.obs.iloc[row_idx]
    n = X.shape[0]

    ax_lab = fig.add_subplot(gs[0])
    ax_ann = fig.add_subplot(gs[1])
    ax_hm = fig.add_subplot(gs[2])

    # ---- heatmap, per chromosome (each arm has its own bin edges) ----
    yedges = np.arange(n + 1, dtype=float)
    for chrom, grp in wins.groupby("chromosome", sort=False):
        cols = grp.index.to_numpy()
        lo, hi = ext.loc[chrom, "lo"], ext.loc[chrom, "hi"]
        xe = chrom_edges(grp["bp_mid"].to_numpy(float), lo, hi) - lo + offsets[chrom]
        ax_hm.pcolormesh(xe, yedges, X[:, cols], cmap="RdBu_r",
                         norm=Normalize(-vmax, vmax), rasterized=True, shading="flat")
    for chrom in ext.index[1:]:
        ax_hm.axvline(offsets[chrom], color="k", lw=0.6)
    centers = [offsets[c] + (ext.loc[c, "hi"] - ext.loc[c, "lo"]) / 2 for c in ext.index]
    ax_hm.set_xticks(centers)
    ax_hm.set_xticklabels([c.replace("chr", "") for c in ext.index], fontsize=7)
    ax_hm.set_xlim(0, offsets[ext.index[-1]] + (ext.loc[ext.index[-1], "hi"] - ext.loc[ext.index[-1], "lo"]))
    ax_hm.set_ylim(n, 0)
    ax_hm.set_yticks([])
    if show_x:
        ax_hm.set_xlabel("chromosome (windows placed by genomic position)", fontsize=9)
    ax_hm.set_title(f"{title}   n={n} entities, {X.shape[1]} windows", fontsize=10, loc="left")

    # ---- annotation strips: compartment | truth clone | n_genes ----
    comp_cats = [c for c in COMPARTMENT_COLORS if (obs["compartment"].astype(str) == c).any()]
    clone_cats = [c for c in CLONE_COLORS if (obs["truth_clone"].astype(str) == c).any()]
    comp_code = pd.Categorical(obs["compartment"].astype(str), categories=comp_cats).codes
    clone_code = pd.Categorical(obs["truth_clone"].astype(str), categories=clone_cats).codes
    ax_ann.pcolormesh(np.array([0.0, 1.0]), yedges, comp_code[:, None],
                      cmap=ListedColormap([COMPARTMENT_COLORS[c] for c in comp_cats]), rasterized=True)
    ax_ann.pcolormesh(np.array([1.1, 2.1]), yedges, clone_code[:, None],
                      cmap=ListedColormap([CLONE_COLORS[c] for c in clone_cats]), rasterized=True)
    ng = obs["n_genes"].to_numpy(float)
    ax_ann.pcolormesh(np.array([2.2, 3.2]), yedges, ng[:, None], cmap="viridis",
                      norm=LogNorm(max(ng.min(), 1), ng.max()), rasterized=True)
    ax_ann.set_xlim(0, 3.2)
    ax_ann.set_ylim(n, 0)
    ax_ann.set_yticks([])
    ax_ann.set_xticks([0.5, 1.6, 2.7])
    ax_ann.set_xticklabels(["compartment", "truth clone", "n_genes"], fontsize=7, rotation=90)
    for s in ax_ann.spines.values():
        s.set_visible(False)

    # ---- bracketed block labels ----
    ax_lab.set_xlim(0, 1)
    ax_lab.set_ylim(n, 0)
    ax_lab.axis("off")
    # Bracket per block, with leader lines: small blocks (clone 1,2,3 and the
    # non-epi leak) are only a handful of rows, so their text must be pushed
    # apart or it overlaps its neighbours.
    centers = np.array([(bounds[i] + bounds[i + 1]) / 2 for i in range(len(labels))], dtype=float)
    min_gap = 0.055 * n
    placed = centers.copy()
    for _ in range(400):
        moved = False
        for i in range(len(placed) - 1):
            gap = placed[i + 1] - placed[i]
            if gap < min_gap:
                shift = (min_gap - gap) / 2
                placed[i] -= shift
                placed[i + 1] += shift
                moved = True
        if not moved:
            break
    placed = np.clip(placed, 0.03 * n, 0.97 * n)
    for i, lab in enumerate(labels):
        y0, y1 = bounds[i], bounds[i + 1]
        cnt = y1 - y0
        med = int(np.median(obs["n_genes"].to_numpy()[y0:y1]))
        ax_lab.plot([0.99, 0.93, 0.93, 0.99], [y0 + 0.5, y0 + 0.5, y1 - 0.5, y1 - 0.5],
                    color="0.25", lw=1.0, clip_on=False)
        ax_lab.plot([0.93, 0.90], [(y0 + y1) / 2, placed[i]], color="0.55", lw=0.7, clip_on=False)
        ax_lab.text(0.88, placed[i], f"{lab}\nn={cnt}, med genes={med}",
                    ha="right", va="center", fontsize=7.5)

    for i in bounds[1:-1]:
        for ax in (ax_hm, ax_ann):
            ax.axhline(i, color="w", lw=1.2)
    return labels, bounds, X, obs, comp_cats, clone_cats


def main() -> None:
    data = {arm: C.load_arm(arm) for arm in ("raw", "tracer_whole")}
    wins_by_arm = {a: w for a, (_, w) in data.items()}
    ext, offsets, _ = genome_axis(wins_by_arm)
    pooled = np.concatenate([C.dense_cnv(ad).ravel() for ad, _ in data.values()])
    vmax = float(np.quantile(np.abs(pooled), 0.99))
    print(f"shared colour limit +/-{vmax:.4f} (99th pct of |CNV| pooled over both arms)")

    fig = plt.figure(figsize=(15, 17))
    outer = fig.add_gridspec(2, 1, hspace=0.10, left=0.16, right=0.965, top=0.955, bottom=0.075)
    summaries = {}
    for k, (arm, title) in enumerate([("raw", "Xenium (original segmentation)"),
                                      ("tracer_whole", "TRACER-refined (whole cells)")]):
        adata, wins = data[arm]
        gs = outer[k].subgridspec(1, 3, width_ratios=[0.23, 0.045, 1.0], wspace=0.012)
        summaries[arm] = draw_panel(fig, gs, adata, wins, ext, offsets, vmax, title, show_x=(k == 1))

    cax = fig.add_axes([0.40, 0.030, 0.22, 0.010])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-vmax, vmax), cmap="RdBu_r"),
                 cax=cax, orientation="horizontal", label="inferCNV value (signed)")
    cax.tick_params(labelsize=7)

    handles = [Patch(facecolor=COMPARTMENT_COLORS[c], label=f"compartment: {c}") for c in COMPARTMENT_COLORS]
    handles += [Patch(facecolor=CLONE_COLORS[c], label=f"truth clone: {c}") for c in ("0", "4", "1", "2", "3", "not_epithelial")]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.965, 0.008),
               ncol=3, fontsize=7, frameon=False)
    fig.suptitle("inferCNV values, entities x genomic windows -- raw vs TRACER, ovary ROI "
                 "(roi_nuclear_grid3cpmi, expression-kNN smoothing k=100)", fontsize=12, y=0.985)
    out = OUT / "cnv_heatmap_raw_vs_tracer.png"
    fig.savefig(out, dpi=170)
    print("wrote", out)

    # ---- quantitative companion ----
    lines = []
    for arm, (labels, bounds, X, obs, _, _) in summaries.items():
        wins = wins_by_arm[arm]
        ref_lo, ref_hi = bounds[0], bounds[1]
        ref_sd = X[ref_lo:ref_hi].mean(axis=0).std()
        lines.append(f"\n=== {arm} === reference per-window SD of block mean = {ref_sd:.5f}")
        for i, lab in enumerate(labels):
            y0, y1 = bounds[i], bounds[i + 1]
            bm = X[y0:y1].mean(axis=0)
            n_out = int((np.abs(bm) > ref_sd).sum())
            top = pd.Series(bm, index=wins["chromosome"].to_numpy()).groupby(level=0).mean()
            top = top.reindex(sorted(top.index, key=C.chromosome_rank))
            hi3 = top.nlargest(3)
            lo3 = top.nsmallest(3)
            lines.append(
                f"  {lab:24s} n={y1-y0:4d} med_genes={int(np.median(obs['n_genes'].to_numpy()[y0:y1])):5d} "
                f"windows|mean|>refSD={n_out:3d}/{len(bm)}  "
                f"top+: {', '.join(f'{c}:{v:+.4f}' for c, v in hi3.items())}  "
                f"top-: {', '.join(f'{c}:{v:+.4f}' for c, v in lo3.items())}"
            )
    txt = "\n".join(lines)
    print(txt)
    (OUT / "cnv_heatmap_summary.txt").write_text(txt)


if __name__ == "__main__":
    main()
