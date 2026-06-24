#!/usr/bin/env python
"""VISTA(VSIR)/VSIG4 vs hypoxia-surrogate spatial enrichment test.

Hypoxia surrogate = VEGFA + HIF1A (only hypoxia-response genes on the panel;
clearly limited).  Tests whether VSIR+/VSIG4+ immunoregulatory cells sit in
hypoxic neighbourhoods, on both original and TRACER cells, and quantifies
spatial enrichment.  Saves per-cell scores for figure + ROI selection.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.neighbors import KDTree

FIG2 = Path(__file__).resolve().parents[3] / "datasets/pancreas_cancer_xenium_10x/processed/fig2"
HYPOXIA = ["VEGFA", "HIF1A"]
K = 15            # spatial neighbours for local field
MIN_CT = 10


def prep(path):
    a = sc.read_h5ad(path)
    a = a[a.obs.n_counts >= MIN_CT].copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.tl.score_genes(a, HYPOXIA, score_name="hypoxia", ctrl_size=50)
    xy = a.obs[["centroid_x", "centroid_y"]].values
    tree = KDTree(xy)
    _, nn = tree.query(xy, k=K + 1)
    a.obs["hypoxia_local"] = a.obs["hypoxia"].values[nn[:, 1:]].mean(1)
    for g in ["VSIR", "VSIG4"]:
        c = a.layers["counts"][:, a.var_names.get_loc(g)]
        c = np.asarray(c.todense()).ravel() if hasattr(c, "todense") else np.asarray(c).ravel()
        a.obs[f"{g}_pos"] = c > 0
        a.obs[f"{g}_norm"] = np.asarray(a[:, g].X.todense()).ravel()
    return a


def report(name, a):
    print(f"\n===== {name} ({a.n_obs:,} cells, n_counts>={MIN_CT}) =====")
    for g in ["VSIR", "VSIG4"]:
        pos = a.obs[f"{g}_pos"].values
        hl = a.obs["hypoxia_local"].values
        u, p = mannwhitneyu(hl[pos], hl[~pos], alternative="greater")
        med_pos, med_neg = np.median(hl[pos]), np.median(hl[~pos])
        rho, prho = spearmanr(a.obs[f"{g}_norm"].values, hl)
        # effect size: rank-biserial
        rbc = 1 - 2 * u / (pos.sum() * (~pos).sum())
        print(f"  {g}+ cells: {pos.sum():,} ({pos.mean():.1%}) | "
              f"local-hypoxia median {g}+ {med_pos:.3f} vs {g}- {med_neg:.3f} "
              f"| MWU p={p:.1e} rbc={-rbc:+.3f} | Spearman(expr,hyp) rho={rho:+.3f}")
    # grid enrichment: bin tissue, correlate bin hypoxia vs VSIG4+/VSIR+ density
    xy = a.obs[["centroid_x", "centroid_y"]].values
    bs = 100.0
    bx = (xy[:, 0] // bs).astype(int); by = (xy[:, 1] // bs).astype(int)
    df = pd.DataFrame({"bx": bx, "by": by, "hyp": a.obs.hypoxia_local.values,
                       "vsir": a.obs.VSIR_pos.values, "vsig4": a.obs.VSIG4_pos.values})
    gb = df.groupby(["bx", "by"]).agg(hyp=("hyp", "mean"), vsir=("vsir", "mean"),
                                      vsig4=("vsig4", "mean"), n=("hyp", "size"))
    gb = gb[gb.n >= 10]
    for g in ["vsir", "vsig4"]:
        rho, p = spearmanr(gb["hyp"], gb[g])
        print(f"  grid({len(gb)} bins, 100µm): Spearman(bin hypoxia, {g.upper()}+ frac) "
              f"rho={rho:+.3f} p={p:.1e}")
    return a


def main():
    ao = report("ORIGINAL", prep(FIG2 / "original_annotated.h5ad"))
    at = report("TRACER", prep(FIG2 / "tracer_annotated.h5ad"))
    # save per-cell scores for the figure / ROI
    for name, a in [("original", ao), ("tracer", at)]:
        cols = ["centroid_x", "centroid_y", "centroid_z", "cell_type", "lt_conf",
                "entity_class" if "entity_class" in a.obs else "source",
                "hypoxia", "hypoxia_local", "VSIR_pos", "VSIG4_pos",
                "VSIR_norm", "VSIG4_norm", "n_counts"]
        cols = [c for c in cols if c in a.obs]
        a.obs[cols].to_parquet(FIG2 / f"vista_hypoxia_{name}.parquet")
    print("\nsaved per-cell vista/hypoxia score tables")


if __name__ == "__main__":
    main()
