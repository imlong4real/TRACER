"""inferCNV heatmaps for the SAME cell ids under both segmentations, grouped by
label-switch category, with matched spatial maps.

(a) top-left    Xenium CNV, rows = ids
(b) bottom-left TRACER CNV, SAME ids in the SAME row order
(c) top-right   Xenium centroids, coloured by category
(d) bottom-right TRACER centroids, coloured by category

Row blocks: epithelial -> fibroblast switchers, concordant fibroblast,
concordant epithelial. Ids come from the ID join (every whole-cell tracer_id is
a Xenium cell_id), so rows correspond one-to-one between the two heatmaps.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
import pyarrow.parquet as pq
warnings.filterwarnings("ignore")
import cnv_review_common as C
from step1_heatmap import genome_axis, chrom_edges

OUT = C.Path(__file__).parent
CAT = ["epithelial -> fibroblast", "concordant fibroblast", "concordant epithelial"]
CCOL = {"epithelial -> fibroblast": "#984ea3", "concordant fibroblast": "#ff7f00",
        "concordant epithelial": "#1b7837"}


def id_groups():
    ax = pd.read_csv(C.RUN / "annotations_raw.csv", dtype={"entity_id": str}).set_index("entity_id")["annotation"].astype(str)
    at = pd.read_csv(C.RUN / "annotations_tracer_whole.csv", dtype={"entity_id": str}).set_index("entity_id")["annotation"].astype(str)
    t = pq.read_table(C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
                      columns=["tracer_id", "_etype"]).to_pandas()
    whole = set(t.loc[t["_etype"].astype(str) == "cell", "tracer_id"].astype(str).unique())
    at = at.loc[at.index.isin(whole)]
    ids = ax.index.intersection(at.index)
    x, y = ax.reindex(ids), at.reindex(ids)
    return {
        CAT[0]: ids[(x == "epithelial") & (y == "fibroblast")],
        CAT[1]: ids[(x == "fibroblast") & (y == "fibroblast")],
        CAT[2]: ids[(x == "epithelial") & (y == "epithelial")],
    }


def order_within(X):
    """Hierarchical order on the Xenium profiles; flat rows appended at the end."""
    if len(X) < 3:
        return np.arange(len(X))
    sd = X.std(axis=1)
    ok = sd > 0
    idx = np.arange(len(X))
    if ok.sum() < 3:
        return idx
    d = pdist(X[ok], metric="correlation")
    d[~np.isfinite(d)] = np.nanmax(d[np.isfinite(d)]) if np.isfinite(d).any() else 0.0
    return np.concatenate([idx[ok][leaves_list(linkage(d, method="average"))], idx[~ok]])


def draw_heatmap(fig, gs, adata, wins, ext, offsets, total, order_ids, bounds, vmax, title, show_x):
    ax_lab = fig.add_subplot(gs[0]); ax_ann = fig.add_subplot(gs[1]); ax_hm = fig.add_subplot(gs[2])
    pos = {v: i for i, v in enumerate(adata.obs_names.astype(str))}
    rows = np.array([pos[i] for i in order_ids])
    X = C.dense_cnv(adata)[rows]
    ng = adata.obs["n_genes"].to_numpy()[rows]
    n = len(rows)
    yed = np.arange(n + 1, dtype=float)
    for chrom, grp in wins.groupby("chromosome", sort=False):
        cols = grp.index.to_numpy()
        lo, hi = ext.loc[chrom, "lo"], ext.loc[chrom, "hi"]
        xe = chrom_edges(grp["bp_mid"].to_numpy(float), lo, hi) - lo + offsets[chrom]
        ax_hm.pcolormesh(xe, yed, X[:, cols], cmap="RdBu_r", norm=Normalize(-vmax, vmax),
                         rasterized=True, shading="flat")
    for c in ext.index[1:]:
        ax_hm.axvline(offsets[c], color="k", lw=0.5)
    cen = [offsets[c] + (ext.loc[c, "hi"] - ext.loc[c, "lo"]) / 2 for c in ext.index]
    ax_hm.set_xticks(cen); ax_hm.set_xticklabels([c.replace("chr", "") for c in ext.index], fontsize=7)
    ax_hm.set_xlim(0, total); ax_hm.set_ylim(n, 0); ax_hm.set_yticks([])
    if show_x:
        ax_hm.set_xlabel("chromosome", fontsize=9)
    ax_hm.set_title(f"{title}   n={n} ids, {X.shape[1]} windows", fontsize=10, loc="left")

    cats = np.concatenate([[c] * (bounds[i + 1] - bounds[i]) for i, c in enumerate(CAT)])
    code = pd.Categorical(cats, categories=CAT).codes
    ax_ann.pcolormesh(np.array([0.0, 1.0]), yed, code[:, None],
                      cmap=ListedColormap([CCOL[c] for c in CAT]), rasterized=True)
    ax_ann.set_xlim(0, 1); ax_ann.set_ylim(n, 0); ax_ann.set_yticks([])
    ax_ann.set_xticks([0.5]); ax_ann.set_xticklabels(["category"], fontsize=7, rotation=90)
    for s in ax_ann.spines.values():
        s.set_visible(False)

    ax_lab.set_xlim(0, 1); ax_lab.set_ylim(n, 0); ax_lab.axis("off")
    centers = np.array([(bounds[i] + bounds[i + 1]) / 2 for i in range(len(CAT))], float)
    gap = 0.14 * n
    for _ in range(300):
        moved = False
        for i in range(len(centers) - 1):
            if centers[i + 1] - centers[i] < gap:
                sh = (gap - (centers[i + 1] - centers[i])) / 2
                centers[i] -= sh; centers[i + 1] += sh; moved = True
        if not moved:
            break
    centers = np.clip(centers, 0.05 * n, 0.95 * n)
    for i, c in enumerate(CAT):
        y0, y1 = bounds[i], bounds[i + 1]
        ax_lab.plot([0.99, 0.93, 0.93, 0.99], [y0 + .5, y0 + .5, y1 - .5, y1 - .5],
                    color="0.25", lw=1.0, clip_on=False)
        ax_lab.plot([0.93, 0.90], [(y0 + y1) / 2, centers[i]], color="0.55", lw=0.7, clip_on=False)
        ax_lab.text(0.88, centers[i], f"{c}\nn={y1-y0}, med genes={int(np.median(ng[y0:y1]))}",
                    ha="right", va="center", fontsize=7.5, color=CCOL[c])
    for b in bounds[1:-1]:
        ax_hm.axhline(b, color="k", lw=1.0); ax_ann.axhline(b, color="w", lw=1.0)


def main():
    groups = id_groups()
    data = {a: C.load_arm(a) for a in ("raw", "tracer_whole")}
    ext, offsets, total = genome_axis({a: w for a, (_, w) in data.items()})

    have = set(data["raw"][0].obs_names.astype(str)) & set(data["tracer_whole"][0].obs_names.astype(str))
    groups = {k: pd.Index([i for i in v if i in have]) for k, v in groups.items()}
    print({k: len(v) for k, v in groups.items()})

    Xr = C.dense_cnv(data["raw"][0])
    posr = {v: i for i, v in enumerate(data["raw"][0].obs_names.astype(str))}
    order_ids, bounds = [], [0]
    for c in CAT:
        ids = groups[c]
        sub = Xr[[posr[i] for i in ids]]
        order_ids.extend(ids[order_within(sub)])
        bounds.append(len(order_ids))
    order_ids = pd.Index(order_ids)

    pooled = np.concatenate([C.dense_cnv(ad).ravel() for ad, _ in data.values()])
    vmax = float(np.quantile(np.abs(pooled), 0.995))

    fig = plt.figure(figsize=(21, 15))
    outer = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], hspace=0.11, wspace=0.10,
                             left=0.055, right=0.985, top=0.935, bottom=0.075)
    for k, (arm, title) in enumerate([("raw", "(a) Xenium (original segmentation)"),
                                      ("tracer_whole", "(b) TRACER-refined (whole cells)")]):
        adata, wins = data[arm]
        gs = outer[k, 0].subgridspec(1, 3, width_ratios=[0.30, 0.035, 1.0], wspace=0.015)
        draw_heatmap(fig, gs, adata, wins, ext, offsets, total, order_ids, bounds, vmax,
                     title, show_x=(k == 1))

    for k, (arm, title) in enumerate([("raw", "(c) Xenium cell centroids"),
                                      ("tracer_whole", "(d) TRACER entity centroids")]):
        adata, _ = data[arm]
        a = fig.add_subplot(outer[k, 1])
        names = adata.obs_names.astype(str)
        xy = adata.obsm["spatial"]
        other = ~np.isin(names, order_ids.to_numpy())
        a.scatter(xy[other, 0], xy[other, 1], s=5, color="0.88", linewidths=0, label="other entities")
        for c in CAT[::-1]:
            m = np.isin(names, groups[c].to_numpy())
            a.scatter(xy[m, 0], xy[m, 1], s=16 if c == CAT[0] else 9, color=CCOL[c],
                      linewidths=0, label=f"{c} ({m.sum()})", zorder=4 if c == CAT[0] else 2)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.legend(fontsize=8, markerscale=1.8, frameon=False, loc="upper left")
        a.set_title(title, fontsize=10, loc="left")

    cax = fig.add_axes([0.20, 0.032, 0.18, 0.009])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-vmax, vmax), cmap="RdBu_r"), cax=cax,
                 orientation="horizontal", label="inferCNV value (signed)")
    cax.tick_params(labelsize=7)
    fig.suptitle("Same cell ids under both segmentations: CNV profiles (left) and position (right)\n"
                 "rows are in the SAME order in (a) and (b), ordered by clustering the Xenium profiles",
                 fontsize=13)
    p = OUT / "fig_switch_cnv.png"; fig.savefig(p, dpi=150); print("wrote", p)

    # quantitative companion
    print("\nmean |CNV| and correlation with the concordant-epithelial mean profile:")
    for arm in ("raw", "tracer_whole"):
        ad, w = data[arm]
        X = C.dense_cnv(ad); pos = {v: i for i, v in enumerate(ad.obs_names.astype(str))}
        prof = {c: X[[pos[i] for i in groups[c]]].mean(0) for c in CAT}
        base = prof[CAT[2]] - prof[CAT[2]].mean()
        print(f"  {arm}:")
        for c in CAT:
            v = prof[c] - prof[c].mean()
            r = v @ base / (np.linalg.norm(v) * np.linalg.norm(base) + 1e-12)
            print(f"    {c:28s} mean|CNV|={np.abs(prof[c]).mean():.4f}  RMS={np.sqrt((prof[c]**2).mean()):.4f}  r_vs_concEpi={r:+.3f}")


if __name__ == "__main__":
    main()
