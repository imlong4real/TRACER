"""inferCNV (infercnvpy) on 10x's 17k human ovarian cancer scFFPE Flex dataset.

Steps
  1. QC (>=200 genes/cell, genes in >=10 cells), gene positions from GENCODE v50 by
     unversioned Ensembl ID (chrX/Y/M dropped).
  2. Cell typing: Leiden clusters on expression, each cluster labelled by the highest mean
     z-scored marker-set score (the spatial run's marker sets plus common scRNA markers that
     the Xenium 5K panel lacks).
  3. infercnvpy against all normal cells (fibroblast, myeloid, T, B, endothelial), window 100.
  4. Clones: epithelial cells only, CNV PCA + kNN rebuilt on the subset, Leiden.
  5. Arm-level karyotype per clone with review/arm_cnv.py (counts-based, median, z test).

Outputs here: adata_flex_cnv.h5ad, celltype_markers.csv, clone_arm_calls.csv, figures.
"""
from __future__ import annotations
import argparse, gzip, sys
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, infercnvpy as cnv, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
BENCH = OUT.parents[1]
REVIEW = BENCH / "output/runs/roi_nuclear_grid3cpmi/review"
sys.path.insert(0, str(REVIEW))
H5 = BENCH / "data/flex_ovary_17k/17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5"
GTF = BENCH / "data/gencode.v50.annotation.gtf.gz"
EXTRA = {"epithelial": ["KRT8", "KRT18", "KRT19", "KRT7", "WT1", "MUC16", "PAX8", "EPCAM"],
         "fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM"],
         "myeloid": ["LYZ", "C1QA", "C1QB", "AIF1", "TYROBP"],
         "t_cell": ["CD3D", "TRAC", "IL7R", "NKG7"],
         "b_cell": ["CD79B", "IGKC", "JCHAIN"],
         "endothelial": ["VWF", "PLVAP"]}
REFERENCE = ["fibroblast", "myeloid", "t_cell", "b_cell", "endothelial"]


def gene_positions(var):
    rows = []
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if p[2] != "gene":
                continue
            gid = p[8].split('gene_id "', 1)[1].split('"', 1)[0].split(".")[0]
            rows.append((gid, p[0], int(p[3]), int(p[4])))
    g = pd.DataFrame(rows, columns=["gid", "chromosome", "start", "end"]).drop_duplicates("gid").set_index("gid")
    pos = g.reindex(var["gene_ids"].str.split(".").str[0].to_numpy())
    return pos.set_index(var.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone-res", type=float, default=0.2)
    ap.add_argument("--window", type=int, default=100)
    args = ap.parse_args()

    a = sc.read_10x_h5(H5)
    a.var_names_make_unique()
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_genes(a, min_cells=10)
    pos = gene_positions(a.var)
    a.var["chromosome"] = pos["chromosome"].astype(str).to_numpy()
    a.var["start"] = pos["start"].astype(float).to_numpy()
    a.var["end"] = pos["end"].astype(float).to_numpy()
    keep = a.var["chromosome"].isin([f"chr{i}" for i in range(1, 23)])
    print(f"cells {a.n_obs}; genes {a.n_vars}, {keep.sum()} on autosomes with positions")
    a.layers["counts"] = a.X.copy()

    # --- cell typing
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=3000)
    sc.pp.pca(a, n_comps=50, mask_var="highly_variable")
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=30)
    sc.tl.leiden(a, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2)
    sc.tl.umap(a)
    ms = yaml.safe_load(open(BENCH / "config/panels.yaml"))["marker_sets"]
    sets = {k: sorted(set(map(str, ms.get(k, []))) | set(EXTRA[k])) for k in EXTRA}
    for k, gs in sets.items():
        gs = [g for g in gs if g in a.var_names]
        sc.tl.score_genes(a, gs, score_name=f"score_{k}")
    S = a.obs[[f"score_{k}" for k in sets]]
    Z = (S - S.mean()) / S.std()
    cl = Z.groupby(a.obs["leiden"], observed=True).mean()
    cl.columns = list(sets)
    lab = cl.idxmax(1)
    a.obs["cell_type"] = a.obs["leiden"].map(lab).astype(str)
    cl.assign(label=lab, n=a.obs["leiden"].value_counts()).round(2).to_csv(OUT / "celltype_markers.csv")
    print(a.obs["cell_type"].value_counts().to_string())

    # --- inferCNV on autosomes
    c = a[:, keep.to_numpy()].copy()
    c.obs["compartment"] = np.where(c.obs["cell_type"].isin(REFERENCE), "reference",
                                    np.where(c.obs["cell_type"] == "epithelial", "tumor", "unknown"))
    cnv.tl.infercnv(c, reference_key="cell_type", reference_cat=[t for t in REFERENCE if t in set(c.obs["cell_type"])],
                    window_size=args.window, n_jobs=4)
    cnv.tl.pca(c); cnv.pp.neighbors(c); cnv.tl.leiden(c, resolution=0.4); cnv.tl.cnv_score(c)
    print(c.obs.groupby("cell_type")["cnv_score"].median().round(4).to_string())

    # --- clones: epithelial only, CNV graph rebuilt on the subset
    e = c[c.obs["cell_type"] == "epithelial"].copy()
    cnv.tl.pca(e); cnv.pp.neighbors(e); cnv.tl.leiden(e, resolution=args.clone_res, key_added="clone")
    vc = e.obs["clone"].value_counts()
    small = vc.index[vc < 0.01 * e.n_obs]
    e.obs["clone"] = e.obs["clone"].astype(str).where(~e.obs["clone"].isin(small), "small")
    c.obs["clone"] = "normal"
    c.obs.loc[e.obs_names, "clone"] = e.obs["clone"].astype(str).to_numpy()
    c.obs.loc[(c.obs["clone"] == "normal") & (c.obs["compartment"] != "reference"), "clone"] = "other"
    print(c.obs["clone"].value_counts().to_string())

    # --- arm-level karyotype per clone (counts-based module from the spatial analysis)
    from arm_cnv import ArmCNV
    c.obs["annotation"] = c.obs["cell_type"]
    m = ArmCNV(c, stat="wmedian")
    lab_clone = c.obs["clone"].where(c.obs["clone"].ne("normal") & c.obs["clone"].ne("other"), "ref_" + c.obs["cell_type"])
    res = m.score_groups(lab_clone.to_numpy(), min_cells=50)
    res.to_csv(OUT / "clone_arm_calls.csv", index=False)
    for g, r in res.groupby("group"):
        print(f"{g:>16} n={r.n_cells.iloc[0]:>5}  gains {list(r.arm[r.call > 0])}  losses {list(r.arm[r.call < 0])}")

    c.write_h5ad(OUT / "adata_flex_cnv.h5ad")

    # --- figures
    sc.settings.figdir = OUT
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    sc.pl.umap(a, color="cell_type", ax=ax[0], show=False, title="cell type (marker scores per cluster)")
    c.obsm["X_umap"] = a[c.obs_names].obsm["X_umap"]
    sc.pl.umap(c, color="cnv_score", ax=ax[1], show=False, title="inferCNV score", cmap="magma")
    sc.pl.umap(c, color="clone", ax=ax[2], show=False, title="CNV clones (epithelial)")
    fig.tight_layout(); fig.savefig(OUT / "fig_flex_umap.png", dpi=130)
    order = c.obs["clone"].astype("category")
    c.obs["clone_cat"] = order
    cnv.pl.chromosome_heatmap(c, groupby="clone_cat", show=False, figsize=(16, 9))
    plt.savefig(OUT / "fig_flex_cnv_heatmap.png", dpi=130, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
