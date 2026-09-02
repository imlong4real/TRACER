#!/usr/bin/env python3
"""Rebuild the three balanced lung panels with a REPLACEMENT-based balanced draw.

The first attempt capped every type at the rarest (Ciliated, 646) giving a
5,814-cell reference, against the vanilla panel's ~50,000. Median co-detection
O was 9-20 vs vanilla's 86 -- so that benchmark compared a well-estimated panel
against thinly-estimated ones, not one estimator against another.

Here every type is drawn to a common TARGET, sampling with replacement where
a type is smaller than the target. Reference size then matches vanilla's scale.

GUARD: `min_det_cells` counts ROWS, so replication would let a gene detected in
3 unique Ciliated cells clear a 25-cell floor. The gene filter is therefore
applied on UNIQUE cells of the draw, and the builder is then called with
min_det_cells=1 on the already-filtered gene set.
"""
import time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
H5 = HERE / "lung_cancer_50k.h5ad"
TX = REPO / "tutorials" / "lung_cancer" / "data" / "lung_cancer_df.parquet"
OUT = HERE / "panels"
CELLTYPE, SEED, MIN_DET = "Cell_Cluster_level1", 0, 25


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from tracer.conflict_reference import build_depth_corrected_reference
    OUT.mkdir(exist_ok=True)
    genes_tx = sorted(pd.read_parquet(TX, columns=["feature_name"])
                      .feature_name.astype(str).unique())
    A = ad.read_h5ad(H5)
    vn = np.array([str(g) for g in A.var_names])
    keep = [g for g in genes_tx if g in set(vn)]
    ct = A.obs[CELLTYPE].astype(str).to_numpy()
    X = A.layers["counts"]; X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    depth = np.asarray(X.sum(1)).ravel().astype(float)
    col = {g: i for i, g in enumerate(vn)}
    sub = X[:, [col[g] for g in keep]].tocsr()

    vc = pd.Series(ct).value_counts()
    target = int(round(vc.mean()))               # -> ~50k total, vanilla's scale
    log(f"{CELLTYPE}: {len(vc)} types; target {target:,}/type "
        f"-> {target*len(vc):,} reference cells")
    rng = np.random.default_rng(SEED)
    parts, rep = [], {}
    for t in vc.index:
        idx = np.flatnonzero(ct == t)
        draw = rng.choice(idx, target, replace=len(idx) < target)
        parts.append(draw)
        rep[t] = (len(idx), target / len(idx))
    ref = np.concatenate(parts)
    uniq = np.unique(ref)
    log(f"draw: {len(ref):,} rows, {len(uniq):,} unique cells "
        f"({len(uniq)/len(ref):.1%})")
    for t, (n, r) in rep.items():
        log(f"   {t:16s} n={n:>6,}  replication {r:>5.2f}x")

    # gene filter on UNIQUE cells, then min_det_cells=1 so replication cannot
    # manufacture support for a gene.
    for name, mc, use_log in [("count1_pmi_balanced_rep", 1, False),
                              ("count2_pmi_balanced_rep", 2, False),
                              ("xgt1_pmi_balanced_rep",   1, True)]:
        if use_log:
            CP = (sp.diags(1e4 / np.maximum(depth, 1)) @ sub.astype(np.float64)).tocsr()
            src = CP.copy(); src.data = np.log1p(src.data)
            det = np.asarray((src[uniq] >= 1).sum(0)).ravel()
        else:
            src = sub
            det = np.asarray((src[uniq] >= mc).sum(0)).ravel()
        ok = det >= MIN_DET
        genes_ok = [g for g, k in zip(keep, ok) if k]
        log(f"{name}: {ok.sum()}/{len(keep)} genes clear min_det_cells={MIN_DET} "
            f"on {len(uniq):,} unique cells")
        res = build_depth_corrected_reference(
            counts=src[ref][:, ok], genes=np.asarray(genes_ok, dtype=object),
            depth=depth[ref], min_count=mc if not use_log else 1,
            min_det_cells=1, n_depth_bins=25, depth_metric="total_counts")
        e = res.edges
        p = OUT / f"{name}.csv.gz"
        e[["gene_i", "gene_j", "PMI", "cPMI", "O", "E", "z"]].to_csv(p, index=False)
        gs = set(e.gene_i) | set(e.gene_j)
        log(f"{name:26s} {len(e):>7,} edges  {len(gs):>3} genes  "
            f"O med {np.median(e.O):>6.0f}  O max {e.O.max():>7,}  "
            f"pos {np.mean(e.PMI>0.2):.0%} neg {np.mean(e.PMI<-0.2):.0%} -> {p.name}")


if __name__ == "__main__":
    main()
