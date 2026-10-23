"""Descriptive figures for the 10x 17k ovarian cancer scFFPE Flex dataset.

Re-reads the matrix with the same QC/normalisation as run_flex_infercnv.py (all genes, not
just the autosomes kept for inferCNV) and takes cell_type, leiden and the UMAP from
adata_flex_cnv.h5ad.
  fig_flex_umap.png     UMAP by cell type; UMAP by Leiden cluster with each cluster's label
  fig_flex_dotplot.png  markers x cell type; markers x cluster (ordered by assigned type)
  fig_flex_qc.png       cells per type; UMIs and genes per cell by type; marker-score z by cluster
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUT = Path(__file__).resolve().parent
H5 = OUT.parents[1] / "data/flex_ovary_17k/17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5"
TYPES = ["epithelial", "fibroblast", "endothelial", "myeloid", "t_cell", "b_cell"]
MARKERS = {"epithelial": ["EPCAM", "PAX8", "WT1", "MUC16", "KRT8", "MSLN"],
           "fibroblast": ["COL1A1", "DCN", "LUM", "PDGFRB"],
           "endothelial": ["PECAM1", "VWF", "CDH5"],
           "myeloid": ["CD68", "CD14", "LYZ", "C1QA"],
           "t_cell": ["CD3D", "CD3E", "NKG7", "IL7R"],
           "b_cell": ["MS4A1", "CD79A", "JCHAIN"]}
PAL = {"epithelial": "#e66101", "fibroblast": "#1b7837", "endothelial": "#2166ac",
       "myeloid": "#b2182b", "t_cell": "#762a83", "b_cell": "#8c510a"}


def main():
    c = ad.read_h5ad(OUT / "adata_flex_cnv.h5ad", backed="r")
    obs, umap = c.obs[["leiden", "cell_type", "clone"]].copy(), np.asarray(c.obsm["X_umap"])
    a = sc.read_10x_h5(H5); a.var_names_make_unique()
    sc.pp.filter_cells(a, min_genes=200); sc.pp.filter_genes(a, min_cells=10)
    a = a[obs.index].copy()
    a.obs["n_umi"] = np.asarray(a.X.sum(1)).ravel(); a.obs["n_genes"] = np.asarray((a.X > 0).sum(1)).ravel()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    a.obs[["leiden", "cell_type"]] = obs[["leiden", "cell_type"]]
    a.obsm["X_umap"] = umap
    types = [t for t in TYPES if t in set(a.obs.cell_type)]
    a.obs["cell_type"] = pd.Categorical(a.obs["cell_type"], categories=types)
    a.uns["cell_type_colors"] = [PAL[t] for t in types]
    lab = a.obs.groupby("leiden", observed=True)["cell_type"].first()
    print(a.obs["cell_type"].value_counts().to_string())

    # ---- UMAP
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.8))
    sc.pl.umap(a, color="cell_type", ax=ax[0], show=False, frameon=False, s=6,
               title=f"cell type (n = {a.n_obs:,} cells after QC)")
    sc.pl.umap(a, color="leiden", ax=ax[1], show=False, frameon=False, s=6, legend_loc=None,
               title="expression Leiden clusters (resolution 1.0), labelled with assigned type")
    for k, xy in pd.DataFrame(umap, index=a.obs_names).groupby(a.obs["leiden"].to_numpy()).median().iterrows():
        ax[1].text(xy[0], xy[1], f"{k}\n{lab[k]}", fontsize=7, ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", alpha=0.8))
    fig.tight_layout(); fig.savefig(OUT / "fig_flex_umap.png", dpi=140); plt.close(fig)

    # ---- dotplots
    mk = {t: [g for g in MARKERS[t] if g in a.var_names] for t in TYPES}
    order = sorted(lab.index, key=lambda k: (types.index(lab[k]), int(k)))
    a.obs["cluster"] = pd.Categorical([f"{k} {lab[k]}" for k in a.obs["leiden"]],
                                      categories=[f"{k} {lab[k]}" for k in order])
    fig, ax = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    sc.pl.dotplot(a, mk, groupby="cell_type", standard_scale="var", ax=ax[0], show=False,
                  title="(a) markers by cell type (scaled per gene)")
    sc.pl.dotplot(a, mk, groupby="cluster", standard_scale="var", ax=ax[1], show=False,
                  title="(b) markers by expression cluster, ordered by assigned type")
    fig.savefig(OUT / "fig_flex_dotplot.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---- QC / composition / label evidence
    fig = plt.figure(figsize=(20, 5.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 1, 1, 1.3], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    vc = a.obs["cell_type"].value_counts().reindex(types)
    ax.barh(range(len(types)), vc.values, color=[PAL[t] for t in types])
    ax.set_yticks(range(len(types)), types); ax.invert_yaxis()
    for i, v in enumerate(vc.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    ax.set_xlabel("cells"); ax.set_title("(a) cells per type", loc="left", fontsize=10)
    for j, (col, nm) in enumerate([("n_umi", "UMIs per cell"), ("n_genes", "genes per cell")], start=1):
        ax = fig.add_subplot(gs[j])
        vals = [a.obs.loc[a.obs.cell_type == t, col].to_numpy() for t in types]
        vp = ax.violinplot([np.log10(v) for v in vals], showmedians=True, widths=0.8)
        for b, t in zip(vp["bodies"], types):
            b.set_facecolor(PAL[t]); b.set_alpha(0.6)
        ax.set_xticks(range(1, len(types) + 1), types, rotation=30)
        ax.set_ylabel(f"log10 {nm}")
        ax.set_title(f"({'bc'[j-1]}) {nm}; medians " + ", ".join(f"{int(np.median(v))}" for v in vals),
                     loc="left", fontsize=9)
    ax = fig.add_subplot(gs[3])
    cm = pd.read_csv(OUT / "celltype_markers.csv", index_col=0)
    cm.index = cm.index.astype(str)
    Z = cm.loc[order, TYPES]
    im = ax.imshow(Z.values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -2, 3))
    ax.set_xticks(range(len(TYPES)), TYPES, rotation=30)
    ax.set_yticks(range(len(order)), [f"{k} {lab[k]} (n={cm.loc[k, 'n']})" for k in order], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.04).set_label("mean marker-set score (z)")
    ax.set_title("(d) label evidence: each cluster takes its highest marker-set score", loc="left", fontsize=9)
    fig.savefig(OUT / "fig_flex_qc.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
