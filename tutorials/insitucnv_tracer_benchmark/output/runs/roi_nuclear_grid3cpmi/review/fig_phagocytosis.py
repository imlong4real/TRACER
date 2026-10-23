"""Do myeloid switchers carry a phagocytosis programme?

Myeloid switchers = Xenium epithelial cells whose same-ID TRACER whole cell is myeloid (62).
Their TRACER entity (the remnant, epithelial transcripts already moved to a partial) is
compared with concordant myeloid cells (myeloid in both segmentations), also on their TRACER
entity, so both sides come from the same segmentation.

Score per entity = share of transcripts from a curated myeloid phagocytosis set minus the
share from expression-matched control genes (10 per set gene, same expression bin; Seurat
module-score logic on counts). Myeloid annotation markers, ubiquitous lysosome/endocytosis
genes a tumour neighbour would supply, CD47 and secreted bridges are left out of the set.

Each switcher is matched to its 10 nearest concordant myeloid cells on (share of Xenium
epithelial cells within 30 um, log transcript count); the controls are randomly dropped to
the switcher's transcript total before scoring. Paired Wilcoxon on switcher - mean(matched).
Sanity first: concordant myeloid must outscore concordant fibroblast.
Dilution check (d): the same paired comparison for the myeloid annotation markers' share,
to see whether a difference is specific to phagocytosis or a weaker myeloid signal overall.
"""
from __future__ import annotations
import zlib
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon, mannwhitneyu
import cnv_review_common as C
from arm_cnv import drop_to

OUT = C.Path(__file__).parent
SET = ["MERTK", "AXL", "CD36", "MARCO", "TIMD4", "STAB1", "MRC1", "CD5L", "TREM2", "FCGR1A",
       "FCGR2A", "FCGR3A", "ITGAM", "CYBB", "NCF1", "NCF2", "LGMN", "SIRPA"]
EXCLUDE_FROM_CTRL = {"CD68", "CD14", "CSF1R", "CD163", "MSR1", "EPCAM", "MSLN", "PAX8", "CDH1"}
N_BINS, N_CTRL, K_MATCH, RADIUS = 25, 10, 10, 30.0
MARKERS = ["CD68", "CD14", "CSF1R", "CD163", "MSR1"]
COL = {"switcher": "#e66101", "myeloid": "#5e3c99", "fibroblast": "#b2abd2", "epithelial": "#1b7837"}


def main():
    t = ad.read_h5ad(C.RUN / "insitucnv_tracer_whole/adata_cnv.h5ad")
    x = ad.read_h5ad(C.RUN / "insitucnv_raw/adata_cnv.h5ad")
    sw = pd.read_csv(OUT / "immune_switchers.csv")
    sw_ids = sw.loc[sw["cls"] == "myeloid", "cell_id"].tolist()

    Y = t.layers["counts"]
    Y = (Y.tocsr() if sp.issparse(Y) else sp.csr_matrix(Y)).astype(np.int64)
    genes = t.var_names
    tot = np.asarray(Y.sum(1)).ravel()

    # control genes: same expression bin (mean share over all TRACER whole entities)
    share = sp.diags(1.0 / np.maximum(tot, 1)) @ Y
    mean_share = np.asarray(share.mean(0)).ravel()
    bins = pd.qcut(pd.Series(mean_share).rank(method="first"), N_BINS, labels=False).to_numpy()
    rng = np.random.default_rng(zlib.crc32(b"phagocytosis"))
    gi = {g: i for i, g in enumerate(genes)}
    set_idx = np.array([gi[g] for g in SET])
    banned = set(set_idx) | {gi[g] for g in EXCLUDE_FROM_CTRL if g in gi}
    ctrl_idx = []
    for i in set_idx:
        pool = [j for j in np.where(bins == bins[i])[0] if j not in banned]
        ctrl_idx.extend(rng.choice(pool, N_CTRL, replace=False))
    ctrl_idx = np.array(ctrl_idx)

    def score(rows):                           # rows: (n, G) dense int counts
        n = np.maximum(rows.sum(1), 1)
        return rows[:, set_idx].sum(1) / n - rows[:, ctrl_idx].sum(1) / n / N_CTRL

    mk_idx = np.array([gi[g] for g in MARKERS])

    def raw_share(rows, idx):
        return rows[:, idx].sum(1) / np.maximum(rows.sum(1), 1)

    def gene_shares(rows):
        return rows[:, set_idx] / np.maximum(rows.sum(1), 1)[:, None]

    # groups (same ID in both segmentations)
    common = x.obs_names.intersection(t.obs_names)
    ax_, at_ = x.obs.loc[common, "annotation"].astype(str), t.obs.loc[common, "annotation"].astype(str)
    conc = {k: common[(ax_ == k) & (at_ == k)].difference(sw_ids) for k in ["myeloid", "fibroblast", "epithelial"]}
    row = {e: i for i, e in enumerate(t.obs_names)}
    dense = lambda ids: Y[[row[e] for e in ids]].toarray()

    # sanity: concordant myeloid vs fibroblast (and epithelial for reference), no matching
    S = {k: score(dense(v)) for k, v in conc.items()}
    sanity_p = mannwhitneyu(S["myeloid"], S["fibroblast"], alternative="greater").pvalue
    print(f"sanity: concordant myeloid median {np.median(S['myeloid']):.4f} (n={len(S['myeloid'])}) vs "
          f"fibroblast {np.median(S['fibroblast']):.4f} (n={len(S['fibroblast'])}), "
          f"epithelial {np.median(S['epithelial']):.4f}; one-sided p={sanity_p:.2e}")
    if not np.median(S["myeloid"]) > np.median(S["fibroblast"]) or sanity_p >= 0.05:
        raise SystemExit("sanity check failed: the set does not mark myeloid cells here -- stopping")

    # matching covariates: share of Xenium epithelial cells within 30 um, log depth
    xy = np.asarray(x.obsm["spatial"])[:, :2]
    epi = (x.obs["annotation"].astype(str) == "epithelial").to_numpy()
    tree = cKDTree(xy)
    xrow = {e: i for i, e in enumerate(x.obs_names)}

    def tumour_share(ids):
        out = []
        for e in ids:
            i = xrow[e]
            nb = [j for j in tree.query_ball_point(xy[i], RADIUS) if j != i]
            out.append(epi[nb].mean() if nb else np.nan)
        return np.array(out)

    ctrl_ids = list(conc["myeloid"])
    cov_s = np.c_[tumour_share(sw_ids), np.log(tot[[row[e] for e in sw_ids]])]
    cov_c = np.c_[tumour_share(ctrl_ids), np.log(tot[[row[e] for e in ctrl_ids]])]
    mu, sd = cov_c.mean(0), cov_c.std(0)
    _, nn = cKDTree((cov_c - mu) / sd).query((cov_s - mu) / sd, k=K_MATCH)

    Ys, Yc = dense(sw_ids), dense(ctrl_ids)
    s_sw = score(Ys)
    rows_out, d_gene = [], []
    for k, e in enumerate(sw_ids):
        n_k = int(Ys[k].sum())
        mats = np.vstack([drop_to(Yc[j], n_k, rng) if Yc[j].sum() > n_k else Yc[j] for j in nn[k]])
        sc_m = score(mats)
        d_gene.append(gene_shares(Ys[[k]])[0] - gene_shares(mats).mean(0))
        ps, pm = raw_share(Ys[[k]], set_idx)[0], raw_share(mats, set_idx).mean()
        ms_, mm = raw_share(Ys[[k]], mk_idx)[0], raw_share(mats, mk_idx).mean()
        rows_out.append(dict(cell_id=e, n_tx=n_k, tumour_share_30um=cov_s[k, 0],
                             set_share=ps, set_share_matched=pm, marker_share=ms_,
                             marker_share_matched=mm,
                             score=s_sw[k], matched_mean=sc_m.mean(), diff=s_sw[k] - sc_m.mean(),
                             matched_tumour_share=cov_c[nn[k], 0].mean(),
                             matched_log_tx=cov_c[nn[k], 1].mean()))
    R = pd.DataFrame(rows_out)
    R.to_csv(OUT / "phagocytosis_scores.csv", index=False)
    w = wilcoxon(R["diff"])
    print(f"switchers n={len(R)}: score median {R.score.median():.4f} vs matched {R.matched_mean.median():.4f}; "
          f"median diff {R['diff'].median():+.4f}, {100*(R['diff'] > 0).mean():.0f}% positive, Wilcoxon p={w.pvalue:.2e}")
    print(f"matching: tumour share switcher {R.tumour_share_30um.median():.2f} vs matched "
          f"{R.matched_tumour_share.median():.2f}; log tx {np.median(cov_s[:, 1]):.2f} vs {R.matched_log_tx.median():.2f}")
    D = pd.DataFrame(d_gene, columns=SET)
    gstat = pd.DataFrame({"median_diff": D.median(), "frac_pos": (D > 0).mean(),
                          "p": [wilcoxon(D[g]).pvalue if (D[g] != 0).any() else 1.0 for g in SET]})
    print(gstat.sort_values("median_diff").round(5).to_string())

    # ---- figure
    l2 = lambda a, b: np.log2((a + 1e-4) / (b + 1e-4))
    fc_set, fc_mk = l2(R.set_share, R.set_share_matched), l2(R.marker_share, R.marker_share_matched)
    fc_ratio = fc_set - fc_mk
    p_ratio = wilcoxon(fc_ratio).pvalue
    print(f"dilution: log2 switcher/matched -- phago set {np.median(fc_set):+.2f}, myeloid markers "
          f"{np.median(fc_mk):+.2f}, set relative to markers {np.median(fc_ratio):+.2f} (Wilcoxon p={p_ratio:.2e})")
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1.25, 1]))
    ax = axs[0]
    groups = [("myeloid", S["myeloid"]), ("fibroblast", S["fibroblast"]), ("epithelial", S["epithelial"])]
    ax.boxplot([g[1] for g in groups], showfliers=False, widths=0.55, medianprops=dict(color="k"))
    for i, (nm, v) in enumerate(groups, 1):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, alpha=0.5, color=COL[nm], lw=0)
    ax.set_xticks([1, 2, 3], [f"concordant\n{nm} (n={len(v)})" for nm, v in groups])
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_ylabel("phagocytosis score (set share - matched control share)")
    ax.set_title(f"(a) sanity: does the set mark myeloid cells?\nmyeloid > fibroblast, one-sided p={sanity_p:.1e}",
                 fontsize=10, loc="left")

    ax = axs[1]
    for k in range(len(R)):
        ax.plot([0, 1], [R.matched_mean[k], R.score[k]], color="0.75", lw=0.6, zorder=1)
    ax.scatter(np.zeros(len(R)), R.matched_mean, s=14, color=COL["myeloid"], zorder=2, label="10 matched concordant myeloid (mean)")
    ax.scatter(np.ones(len(R)), R.score, s=14, color=COL["switcher"], zorder=2, label="myeloid switcher (TRACER remnant)")
    ax.set_xticks([0, 1], ["matched\ncontrols", "switchers"]); ax.set_xlim(-0.4, 1.4)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center")
    ax.set_title(f"(b) switchers vs matched controls (n={len(R)})\nmedian diff {R['diff'].median():+.4f}, "
                 f"{100*(R['diff'] > 0).mean():.0f}% higher, Wilcoxon p={w.pvalue:.1e}", fontsize=10, loc="left")
    ax.set_ylabel("phagocytosis score")

    ax = axs[2]
    o = gstat.sort_values("median_diff").index
    ax.barh(range(len(o)), 100 * gstat.loc[o, "median_diff"],
            color=[COL["switcher"] if v > 0 else COL["myeloid"] for v in gstat.loc[o, "median_diff"]])
    ax.set_yticks(range(len(o)), [f"{g}  ({100*gstat.loc[g, 'frac_pos']:.0f}% up)" for g in o], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("switcher - matched controls, % of transcripts (median over switchers)")
    ax.set_title("(c) which set genes differ", fontsize=10, loc="left")

    ax = axs[3]
    vals = [fc_mk, fc_set, fc_ratio]
    ax.boxplot(vals, showfliers=False, widths=0.55, medianprops=dict(color="k"))
    for i, v in enumerate(vals, 1):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, alpha=0.6, color=COL["switcher"], lw=0)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks([1, 2, 3], ["myeloid\nmarkers", "phagocytosis\nset", "set relative\nto markers"])
    ax.set_ylabel("log2 switcher / matched controls (transcript share)")
    ax.set_title(f"(d) specific, or a weaker myeloid signal overall?\nmarkers {np.median(fc_mk):+.2f}, set "
                 f"{np.median(fc_set):+.2f}, set vs markers {np.median(fc_ratio):+.2f} (p={p_ratio:.1e})",
                 fontsize=10, loc="left")

    fig.suptitle("Phagocytosis programme in myeloid switchers (TRACER remnant) vs concordant myeloid cells "
                 "matched on local tumour share and depth", fontsize=12)
    fig.tight_layout()
    p = OUT / "fig_phagocytosis.png"; fig.savefig(p, dpi=140); print("wrote", p)


if __name__ == "__main__":
    main()
