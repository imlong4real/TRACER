"""Same as fig_switch_cnv, plus the epithelial PARTIAL recipients in (b) and (d).

Panel (b) is drawn from the inferCNV run that INCLUDES partials, so every row in
(b) is on the same footing. The partial block has no counterpart in (a) --
those entities exist only after TRACER -- so it is appended below a break.
"""
from __future__ import annotations
import sys, warnings, pickle
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
warnings.filterwarnings("ignore")
import cnv_review_common as C
from step1_heatmap import genome_axis, chrom_edges
from fig_switch_cnv import id_groups, CAT, order_within

OUT = C.Path(__file__).parent
PCAT = "epithelial PARTIAL recipients"
CCOL = {CAT[0]: "#984ea3", CAT[1]: "#ff7f00", CAT[2]: "#1b7837", PCAT: "#e7298a"}


def blocks_axes(fig, cell, n, labels, bounds, ng, vmax, wins, ext, offsets, total, X, title, show_x):
    import numpy as _np
    gs = cell.subgridspec(1, 3, width_ratios=[0.30, 0.035, 1.0], wspace=0.015)
    ax_lab = fig.add_subplot(gs[0]); ax_ann = fig.add_subplot(gs[1]); ax_hm = fig.add_subplot(gs[2])
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
    ax_hm.set_title(f"{title}\n   n={n} rows, {X.shape[1]} windows -- colour scale is PER PANEL, "
                    f"+/-{vmax:.3f} (these are two separate inferCNV runs; do not compare intensity "
                    f"across panels)", fontsize=9.5, loc="left")

    cats = np.concatenate([[l] * (bounds[i + 1] - bounds[i]) for i, l in enumerate(labels)])
    code = pd.Categorical(cats, categories=labels).codes
    ax_ann.pcolormesh(np.array([0.0, 1.0]), yed, code[:, None],
                      cmap=ListedColormap([CCOL[l] for l in labels]), rasterized=True)
    ax_ann.set_xlim(0, 1); ax_ann.set_ylim(n, 0); ax_ann.set_yticks([]); ax_ann.set_xticks([])
    for s in ax_ann.spines.values():
        s.set_visible(False)

    ax_lab.set_xlim(0, 1); ax_lab.set_ylim(n, 0); ax_lab.axis("off")
    centers = np.array([(bounds[i] + bounds[i + 1]) / 2 for i in range(len(labels))], float)
    gap = 0.13 * n
    for _ in range(400):
        moved = False
        for i in range(len(centers) - 1):
            if centers[i + 1] - centers[i] < gap:
                sh = (gap - (centers[i + 1] - centers[i])) / 2
                centers[i] -= sh; centers[i + 1] += sh; moved = True
        if not moved:
            break
    centers = np.clip(centers, 0.04 * n, 0.96 * n)
    for i, l in enumerate(labels):
        y0, y1 = bounds[i], bounds[i + 1]
        ax_lab.plot([0.99, 0.93, 0.93, 0.99], [y0 + .5, y0 + .5, y1 - .5, y1 - .5],
                    color="0.25", lw=1.0, clip_on=False)
        ax_lab.plot([0.93, 0.90], [(y0 + y1) / 2, centers[i]], color="0.55", lw=0.7, clip_on=False)
        rms = float(np.sqrt((X[y0:y1].mean(axis=0) ** 2).mean()))
        ax_lab.text(0.88, centers[i],
                    f"{l}\nn={y1-y0}, med genes={int(np.median(ng[y0:y1]))}\nprofile RMS={rms:.4f}",
                    ha="right", va="center", fontsize=7.5, color=CCOL[l])
    for b in bounds[1:-1]:
        ax_hm.axhline(b, color="k", lw=1.0)
    return ax_hm


def main():
    with open(OUT / "_partials_run.pkl", "rb") as fh:
        P = pickle.load(fh)
    g = id_groups()
    SWl, CFl, CEl = list(g[CAT[0]]), list(g[CAT[1]]), list(g[CAT[2]])
    epi_part = list(P["epi_part"].index)

    xen, wx = C.load_arm("raw")
    ext, offsets, total = genome_axis({"raw": wx, "tracer_whole": None} if False else
                                      {"raw": wx, "tr": C.window_intervals(P["var"])})
    wt = C.window_intervals(P["var"])

    posx = {v: i for i, v in enumerate(xen.obs_names.astype(str))}
    post = {v: i for i, v in enumerate(P["names"])}
    have = lambda ids: [i for i in ids if i in posx and i in post]
    SWl, CFl, CEl = have(SWl), have(CFl), have(CEl)
    epi_part = [i for i in epi_part if i in post]

    Xx = C.dense_cnv(xen); Xt = P["X"]
    order = []
    for ids in (SWl, CFl, CEl):
        sub = Xx[[posx[i] for i in ids]]
        order.append([ids[j] for j in order_within(sub)])
    sub = Xt[[post[i] for i in epi_part]]
    order_part = [epi_part[j] for j in order_within(sub)]

    ids_a = order[0] + order[1] + order[2]
    b_a = np.cumsum([0] + [len(o) for o in order])
    ids_b = ids_a + order_part
    b_b = np.concatenate([b_a, [len(ids_b)]])
    lab_a = [CAT[0], CAT[1], CAT[2]]
    lab_b = lab_a + [PCAT]

    Xa = Xx[[posx[i] for i in ids_a]]
    Xb = Xt[[post[i] for i in ids_b]]
    vmax_a = float(np.quantile(np.abs(Xa), 0.995))
    vmax_b = float(np.quantile(np.abs(Xb), 0.995))
    print(f"per-panel colour limits: (a) +/-{vmax_a:.4f}   (b) +/-{vmax_b:.4f}")
    ngx = xen.obs["n_genes"].to_numpy()[[posx[i] for i in ids_a]]
    ngt = P["obs"]["n_genes"].to_numpy()[[post[i] for i in ids_b]]

    fig = plt.figure(figsize=(21, 15.5))
    outer = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], hspace=0.11, wspace=0.10,
                             left=0.055, right=0.985, top=0.93, bottom=0.07)
    blocks_axes(fig, outer[0, 0], len(ids_a), lab_a, b_a, ngx, vmax_a, wx, ext, offsets, total,
                Xa, "(a) Xenium (original segmentation)", False)
    blocks_axes(fig, outer[1, 0], len(ids_b), lab_b, b_b, ngt, vmax_b, wt, ext, offsets, total,
                Xb, "(b) TRACER, run WITH partials -- pink block has no (a) counterpart", True)

    for k, (which, title) in enumerate([("xen", "(c) Xenium cell centroids"),
                                        ("tr", "(d) TRACER entity centroids, incl. partial recipients")]):
        a = fig.add_subplot(outer[k, 1])
        if which == "xen":
            nm = xen.obs_names.astype(str).to_numpy(); xy = xen.obsm["spatial"]
            cats = {CAT[0]: SWl, CAT[1]: CFl, CAT[2]: CEl}
        else:
            nm = P["names"]; xy = P["spatial"]
            cats = {CAT[0]: SWl, CAT[1]: CFl, CAT[2]: CEl, PCAT: epi_part}
        shown = np.concatenate([np.asarray(v) for v in cats.values()])
        oth = ~np.isin(nm, shown)
        a.scatter(xy[oth, 0], xy[oth, 1], s=4, color="0.90", linewidths=0, label="other entities")
        for c in list(cats)[::-1]:
            m = np.isin(nm, np.asarray(cats[c]))
            a.scatter(xy[m, 0], xy[m, 1], s=26 if c == PCAT else (16 if c == CAT[0] else 8),
                      color=CCOL[c], linewidths=0.4 if c == PCAT else 0,
                      edgecolor="k" if c == PCAT else "none",
                      label=f"{c} ({m.sum()})", zorder=5 if c == PCAT else (4 if c == CAT[0] else 2))
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.legend(fontsize=8, markerscale=1.6, frameon=False, loc="upper left")
        a.set_title(title, fontsize=10, loc="left")

    for xo, vm, nmv in ((0.10, vmax_a, "(a) Xenium"), (0.30, vmax_b, "(b) TRACER + partials")):
        cax = fig.add_axes([xo, 0.028, 0.15, 0.008])
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-vm, vm), cmap="RdBu_r"), cax=cax,
                     orientation="horizontal", label=f"inferCNV value, {nmv}")
        cax.tick_params(labelsize=7)
    fig.suptitle("The excised tumour signal is not destroyed -- it moves into PARTIAL entities that "
                 "the tracer_whole arm discards\n"
                 "rows 1-3 are the same ids in the same order in (a) and (b); each panel has its OWN "
                 "colour scale", fontsize=13)
    p = OUT / "fig_switch_cnv_partials.png"; fig.savefig(p, dpi=150); print("wrote", p)


if __name__ == "__main__":
    main()
