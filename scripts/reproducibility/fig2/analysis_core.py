#!/usr/bin/env python
"""Merge label-transfer results, quantify TRACER 'cleanliness', compare composition.

Consumes the canonical label_transfer_spatial.py outputs and the foundation
matrices; produces annotated h5ads (used by every later panel), a composition
table (original vs TRACER complete / partial), and a cross-lineage contamination
("admixture") comparison testing whether TRACER cells are transcriptionally
cleaner.

Run:
    python scripts/reproducibility/fig2/analysis_core.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

ROOT = Path(__file__).resolve().parents[3]
FIG2 = ROOT / "datasets/pancreas_cancer_xenium_10x/processed/fig2"

# lineage marker panels within the Xenium 380-gene panel (for contamination)
LINEAGE_MARKERS = {
    "Ductal": ["EPCAM", "KRT7", "TFF2", "CFTR", "SERPINB3", "AGR3", "SPDEF", "GPX2"],
    "Acinar": ["AMY2A", "CPA3", "PRG4"],
    "Endocrine": ["CHGA", "INS", "GCG", "SST", "PPY", "PCSK2", "SCGN", "PPP1R1A"],
    "T/NK": ["CD3D", "CD3E", "CD2", "TRAC", "IL7R", "CD8A", "GZMB", "NKG7", "KLRD1"],
    "B/Plasma": ["MS4A1", "CD79A", "BANK1", "MZB1", "DERL3", "TNFRSF17", "TCL1A"],
    "Myeloid": ["CD68", "CD163", "VSIG4", "MARCO", "TREM2", "AIF1", "FCN1", "MS4A6A"],
    "Endothelial": ["PECAM1", "VWF", "EGFL7", "CLEC14A", "RAMP2", "SOX17"],
    "Fibroblast": ["PDGFRA", "VCAN", "SFRP2", "SFRP4", "FBLN1", "THBS2", "PCOLCE"],
    "Stellate": ["PDGFRB", "MYH11", "ACTA2", "DES", "HIGD1B"],
}


def contamination(adata):
    """1 - dominant-lineage marker fraction; higher = more cross-lineage admixture."""
    lins = list(LINEAGE_MARKERS)
    L = np.zeros((adata.n_obs, len(lins)))
    Xc = adata.X.tocsc()
    for j, lin in enumerate(lins):
        idx = [adata.var_names.get_loc(g) for g in LINEAGE_MARKERS[lin]
               if g in adata.var_names]
        L[:, j] = np.asarray(Xc[:, idx].sum(1)).ravel()
    tot = L.sum(1)
    dom = L.max(1)
    frac_off = np.where(tot > 0, 1.0 - dom / tot, np.nan)
    return frac_off, tot


def attach(adata, csv):
    ann = pd.read_csv(csv).set_index("cell_id")
    ann.index = ann.index.astype(str)
    adata.obs["cell_type"] = ann["transferred_label"].reindex(adata.obs_names).values
    adata.obs["lt_conf"] = ann["transfer_confidence"].reindex(adata.obs_names).values
    off, mtot = contamination(adata)
    adata.obs["contamination"] = off
    adata.obs["marker_total"] = mtot
    return adata


def main():
    ao = sc.read_h5ad(FIG2 / "original_cells.h5ad")
    at = sc.read_h5ad(FIG2 / "tracer_cells.h5ad")
    ao = attach(ao, FIG2 / "lt_original/pdac_orig_transferred_cell_annotations.csv")
    at = attach(at, FIG2 / "lt_tracer/pdac_tracer_transferred_cell_annotations.csv")
    ao.write_h5ad(FIG2 / "original_annotated.h5ad", compression="gzip")
    at.write_h5ad(FIG2 / "tracer_annotated.h5ad", compression="gzip")

    types = sorted(set(ao.obs.cell_type.dropna()) | set(at.obs.cell_type.dropna()))
    groups = {
        "original": ao.obs,
        "tracer_complete": at.obs[at.obs.entity_class == "complete"],
        "tracer_partial": at.obs[at.obs.entity_class == "partial"],
    }

    # ---- composition (confident cells only) ----
    CONF = 0.5
    rows = []
    for grp, obs in groups.items():
        sub = obs[(obs.lt_conf >= CONF)]
        vc = sub.cell_type.value_counts()
        for t in types:
            rows.append(dict(group=grp, cell_type=t, n=int(vc.get(t, 0)),
                             frac=float(vc.get(t, 0) / max(len(sub), 1))))
    comp = pd.DataFrame(rows)
    comp.to_csv(FIG2 / "composition_comparison.csv", index=False)
    piv = comp.pivot(index="cell_type", columns="group", values="frac").round(3)
    print("=== composition (frac of confident cells) ===")
    print(piv.to_string())

    # ---- cleanliness: contamination by group (confident, depth-matched) ----
    print("\n=== cross-lineage contamination (median; lower = cleaner) ===")
    clean_rows = []
    for grp, obs in groups.items():
        sub = obs[(obs.lt_conf >= CONF) & (obs.marker_total >= 3)]
        med = float(np.nanmedian(sub.contamination))
        print(f"  {grp:18} median contam {med:.3f} | median conf {sub.lt_conf.median():.3f}"
              f" | n={len(sub):,}")
        clean_rows.append(dict(group=grp, median_contam=med,
                               median_conf=float(sub.lt_conf.median()), n=len(sub)))
    pd.DataFrame(clean_rows).to_csv(FIG2 / "cleanliness_summary.csv", index=False)

    # ---- per cell type: contamination original vs tracer_complete ----
    print("\n=== contamination by cell type: original vs TRACER complete ===")
    o = ao.obs[(ao.obs.lt_conf >= CONF) & (ao.obs.marker_total >= 3)]
    c = at.obs[(at.obs.entity_class == "complete") & (at.obs.lt_conf >= CONF)
               & (at.obs.marker_total >= 3)]
    ct_rows = []
    for t in types:
        mo = float(np.nanmedian(o.loc[o.cell_type == t, "contamination"])) if (o.cell_type == t).any() else np.nan
        mc = float(np.nanmedian(c.loc[c.cell_type == t, "contamination"])) if (c.cell_type == t).any() else np.nan
        print(f"  {t:22} orig {mo:.3f} -> complete {mc:.3f}  (Δ {mo-mc:+.3f})")
        ct_rows.append(dict(cell_type=t, orig=mo, complete=mc, delta=mo-mc))
    pd.DataFrame(ct_rows).to_csv(FIG2 / "contamination_by_celltype.csv", index=False)

    # ---- confidence gain for partials (newly detectable) ----
    print("\n=== TRACER partial cells: confident detection by type ===")
    p = at.obs[(at.obs.entity_class == "partial")]
    pconf = p[p.lt_conf >= CONF]
    print(f"  partials total {len(p):,}; confident {len(pconf):,} ({len(pconf)/len(p):.1%})")
    print(pconf.cell_type.value_counts().to_string())


if __name__ == "__main__":
    main()
