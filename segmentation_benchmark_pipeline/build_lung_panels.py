#!/usr/bin/env python3
"""Build 4 PMI panels from lung_cancer_50k.h5ad for the segbench benchmark.

  1 count1_pmi_balanced   count>=1 presence, naive PMI, cell-type-balanced
  2 count2_pmi_balanced   count>=2 presence, naive PMI, cell-type-balanced
  3 xgt1_pmi_balanced     Xgt1 presence,     naive PMI, cell-type-balanced
  4 vanilla_cpmi          existing lung_scrna_depth_corrected_pmi.parquet,
                          cPMI renamed to PMI (values untouched)

Balanced = equal cells per Cell_Cluster_level1, capped at the smallest type.
Gene universe = the 300 Xenium panel genes present in the reference.
Every panel is written with a `PMI` column because the pipeline selects
`metric_col = "PMI" if "PMI" in panel.columns else "NPMI"`.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
H5 = HERE / "lung_cancer_50k.h5ad"
TX = REPO / "tutorials" / "lung_cancer" / "data" / "lung_cancer_df.parquet"
VANILLA = REPO / "tutorials" / "lung_cancer" / "lung_scrna_depth_corrected_pmi.parquet"
OUT = HERE / "panels"
CELLTYPE = "Cell_Cluster_level1"
SEED = 0


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from tracer.conflict_reference import build_depth_corrected_reference
    OUT.mkdir(exist_ok=True)

    genes_tx = sorted(pd.read_parquet(TX, columns=["feature_name"])
                      .feature_name.astype(str).unique())
    log(f"Xenium panel: {len(genes_tx)} genes")

    A = ad.read_h5ad(H5)
    vn = np.array([str(g) for g in A.var_names])
    keep = [g for g in genes_tx if g in set(vn)]
    log(f"{len(keep)} present in the reference ({len(genes_tx)-len(keep)} missing)")

    ct = A.obs[CELLTYPE].astype(str).to_numpy()
    X = A.layers["counts"]
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    depth = np.asarray(X.sum(1)).ravel().astype(float)      # TRUE library, all genes
    col = {g: i for i, g in enumerate(vn)}
    sub = X[:, [col[g] for g in keep]].tocsr()

    vc = pd.Series(ct).value_counts()
    per = int(vc.min())
    log(f"{CELLTYPE}: {len(vc)} types, sizes {dict(vc)}")
    log(f"balanced draw: {per:,} cells x {len(vc)} types = {per*len(vc):,} cells")
    rng = np.random.default_rng(SEED)
    ref = np.concatenate([rng.choice(np.flatnonzero(ct == t), per, replace=False)
                          for t in vc.index])

    # log1p(CP10k) for the Xgt1 arm
    CP = (sp.diags(1e4 / np.maximum(depth, 1)) @ sub.astype(np.float64)).tocsr()
    L = CP.copy(); L.data = np.log1p(L.data)

    ARMS = [("count1_pmi_balanced", sub, 1),
            ("count2_pmi_balanced", sub, 2),
            ("xgt1_pmi_balanced",   L,   1)]
    for name, src, mc in ARMS:
        t0 = time.time()
        res = build_depth_corrected_reference(
            counts=src[ref], genes=np.asarray(keep, dtype=object),
            depth=depth[ref], min_count=mc, min_det_cells=25,
            n_depth_bins=25, depth_metric="total_counts")
        e = res.edges
        p = OUT / f"{name}.csv.gz"
        e[["gene_i", "gene_j", "PMI", "cPMI", "O", "E", "z"]].to_csv(p, index=False)
        gset = set(e.gene_i) | set(e.gene_j)
        log(f"{name:22s} {len(e):>7,} edges  {len(gset):>3} genes  "
            f"PMI[{e.PMI.min():+.2f},{e.PMI.max():+.2f}]  "
            f"pos {np.mean(e.PMI > 0.2):.0%} neg {np.mean(e.PMI < -0.2):.0%}  "
            f"[{time.time()-t0:.0f}s] -> {p.name}")

    v = pd.read_parquet(VANILLA)
    v = v.rename(columns={"cPMI": "PMI", "cNPMI": "NPMI"})
    p = OUT / "vanilla_cpmi.csv.gz"
    v.to_csv(p, index=False)
    gset = set(v.gene_i) | set(v.gene_j)
    log(f"{'vanilla_cpmi':22s} {len(v):>7,} edges  {len(gset):>3} genes  "
        f"PMI[{v.PMI.min():+.2f},{v.PMI.max():+.2f}]  "
        f"pos {np.mean(v.PMI > 0.2):.0%} neg {np.mean(v.PMI < -0.2):.0%}  "
        f"(cPMI renamed) -> {p.name}")


if __name__ == "__main__":
    main()
