#!/usr/bin/env python3
"""Specified 'vanilla' cPMI panel for the lung benchmark.

The inherited lung_scrna_depth_corrected_pmi.parquet has no recoverable build
provenance (no script, no sidecar, no PMI column). This rebuilds an equivalent
with fully recorded parameters, matching the code path and defaults used by
run_depthcorr_vs_Xgt1.py and build_wholetx_cpmi.py:

    all 50,000 cells, natural composition (NO balancing)
    presence  count >= 1
    min_det_cells=25, n_depth_bins=25, depth_metric="total_counts"
    gene universe = the 300 Xenium panel genes

Writes both estimator columns so the naive/corrected contrast is available on
this reference too, exactly as for the _rep arms.
"""
import time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "panels"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from tracer.conflict_reference import build_depth_corrected_reference
    OUT.mkdir(exist_ok=True)
    genes_tx = sorted(pd.read_parquet(
        REPO / "tutorials/lung_cancer/data/lung_cancer_df.parquet",
        columns=["feature_name"]).feature_name.astype(str).unique())
    A = ad.read_h5ad(HERE / "lung_cancer_50k.h5ad")
    vn = np.array([str(g) for g in A.var_names])
    keep = [g for g in genes_tx if g in set(vn)]
    X = A.layers["counts"]; X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    depth = np.asarray(X.sum(1)).ravel().astype(float)
    col = {g: i for i, g in enumerate(vn)}
    sub = X[:, [col[g] for g in keep]].tocsr()
    log(f"all {A.n_obs:,} cells, natural composition, {len(keep)} genes")

    res = build_depth_corrected_reference(
        counts=sub, genes=np.asarray(keep, dtype=object), depth=depth,
        min_count=1, min_det_cells=25, n_depth_bins=25,
        depth_metric="total_counts")
    e = res.edges
    log(f"built: {len(e):,} edges, {res.meta['n_genes_retained']} genes retained, "
        f"O med {np.median(e.O):.0f}, O max {e.O.max():,}")

    for name, cols in [("vanilla_spec_cpmi", "cPMI"), ("vanilla_spec_pmi", "PMI")]:
        out = e[["gene_i", "gene_j", cols, "O", "E", "z"]].rename(columns={cols: "PMI"})
        p = OUT / f"{name}.csv.gz"
        out.to_csv(p, index=False)
        log(f"{name:20s} pos {np.mean(e[cols]>0.2):.0%} neg {np.mean(e[cols]<-0.2):.0%}"
            f"  -> {p.name}")

    # how close is this to the inherited panel?
    old = pd.read_parquet(REPO / "tutorials/lung_cancer/lung_scrna_depth_corrected_pmi.parquet")
    k = lambda d: list(zip(np.minimum(d.gene_i, d.gene_j), np.maximum(d.gene_i, d.gene_j)))
    a = e.assign(key=k(e)).set_index("key"); b = old.assign(key=k(old)).set_index("key")
    sh = a.index.intersection(b.index)
    log(f"vs inherited panel: {len(sh):,} shared pairs of {len(e):,}/{len(old):,}; "
        f"pearson(cPMI) = {np.corrcoef(a.loc[sh,'cPMI'], b.loc[sh,'cPMI'])[0,1]:+.4f}, "
        f"pearson(O) = {np.corrcoef(a.loc[sh,'O'], b.loc[sh,'O'])[0,1]:+.4f}")


if __name__ == "__main__":
    main()
