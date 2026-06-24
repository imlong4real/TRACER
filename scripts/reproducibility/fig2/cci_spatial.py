#!/usr/bin/env python
"""Spatially-aware ligand-receptor enrichment: original vs TRACER.

For curated in-panel immune LR pairs we count directed spatial contacts
(<=R µm) where the sender expresses the ligand and its neighbour expresses the
receptor, vs a degree-preserving permutation null (shuffle receptor labels) ->
z-score / fold enrichment.  We also measure **sender-lineage purity**: the
fraction of ligand+ senders that belong to the biologically expected lineage.
TRACER (de-mixed) should raise purity (fewer implausible senders) and sharpen
plausible local interactions.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import KDTree

FIG2 = Path(__file__).resolve().parents[3] / "datasets/pancreas_cancer_xenium_10x/processed/fig2"
R = 30.0          # contact radius (µm)
KMAX = 12
NPERM = 100
MIN_CT = 10
rng = np.random.default_rng(0)

# ligand, receptor, expected sender lineage(s)
LR = [
    ("CD274", "PDCD1", {"Macrophage cell", "Ductal cell type 2"}),  # PD-L1 -> PD-1
    ("CD86", "CTLA4", {"Macrophage cell", "B cell"}),
    ("CD86", "CD28", {"Macrophage cell", "B cell"}),
    ("CD80", "CTLA4", {"Macrophage cell", "B cell"}),
    ("CCL19", "CCR7", {"Fibroblast cell", "Macrophage cell"}),
    ("CCL5", "CCR7", {"T cell", "Macrophage cell"}),
    ("CXCL9", "CXCR4", {"Macrophage cell"}),
    ("CXCL10", "CXCR4", {"Macrophage cell"}),
]


def edges(xy):
    tree = KDTree(xy)
    dist, idx = tree.query(xy, k=min(KMAX + 1, len(xy)))
    dist, idx = dist[:, 1:], idx[:, 1:]
    m = dist <= R
    src = np.repeat(np.arange(len(xy)), m.sum(1))
    dst = idx[m]
    return src, dst


def run(name):
    a = sc.read_h5ad(FIG2 / f"{name}_annotated.h5ad")
    a = a[(a.obs.n_counts >= MIN_CT) & (a.obs.lt_conf >= 0.5)].copy()
    xy = a.obs[["centroid_x", "centroid_y"]].values
    src, dst = edges(xy)
    ct = a.obs.cell_type.values
    Xc = a.X.tocsc()
    rows = []
    for lig, rec, send_lin in LR:
        Lp = np.asarray(Xc[:, a.var_names.get_loc(lig)].todense()).ravel() > 0
        Rp = np.asarray(Xc[:, a.var_names.get_loc(rec)].todense()).ravel() > 0
        obs = int(np.sum(Lp[src] & Rp[dst]))
        # degree-preserving null: shuffle receptor labels
        perm = np.empty(NPERM)
        Rp_copy = Rp.copy()
        for k in range(NPERM):
            rng.shuffle(Rp_copy)
            perm[k] = np.sum(Lp[src] & Rp_copy[dst])
        mu, sd = perm.mean(), perm.std() + 1e-9
        z = (obs - mu) / sd
        fold = obs / (mu + 1e-9)
        # sender-lineage purity among ligand+ cells that actually contact an R+ neighbour
        sender_cells = np.unique(src[(Lp[src] & Rp[dst])])
        if len(sender_cells):
            purity = np.mean([ct[c] in send_lin for c in sender_cells])
        else:
            purity = np.nan
        rows.append(dict(dataset=name, pair=f"{lig}->{rec}", obs=obs, exp=mu,
                         z=z, fold=fold, n_senders=len(sender_cells), sender_purity=purity))
    return pd.DataFrame(rows)


def main():
    res = pd.concat([run("original"), run("tracer")], ignore_index=True)
    res.to_csv(FIG2 / "cci_lr_comparison.csv", index=False)
    piv_z = res.pivot(index="pair", columns="dataset", values="z").round(1)
    piv_p = res.pivot(index="pair", columns="dataset", values="sender_purity").round(3)
    piv_f = res.pivot(index="pair", columns="dataset", values="fold").round(2)
    print("=== LR spatial enrichment z-score (obs vs permuted) ===")
    print(piv_z.to_string())
    print("\n=== fold enrichment ===")
    print(piv_f.to_string())
    print("\n=== sender-lineage purity (fraction of ligand+ senders in expected lineage) ===")
    print(piv_p.to_string())
    print("\nmean sender purity: original %.3f  tracer %.3f" %
          (res[res.dataset == "original"].sender_purity.mean(),
           res[res.dataset == "tracer"].sender_purity.mean()))


if __name__ == "__main__":
    main()
