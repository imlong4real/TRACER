#!/usr/bin/env python3
"""Build RCTD input h5ads restricted to the 1,656 HVG/NPMI gene panel.

Per the Figure-4 design decision, RCTD is run on the *same* 1,656
HVG/NPMI genes that drove TRACER's noseg reconstruction. This makes the
RCTD purity comparison gene-panel-matched across all four methods (so no
method is advantaged by carrying extra genes) and keeps RCTD tractable.

NB: this 1,656-gene restriction is *only* for RCTD. All other downstream
biological metrics (label transfer, marker validation, per-cell-type
Pearson to scRNA pseudobulk, gene/UMI counts) use the whole-transcriptome
matrices built by build_whole_transcriptome.py.

Inputs : the four whole-transcriptome cell/profile-by-gene h5ads.
Panel  : var_names of the TRACER HVG profile_by_gene.h5ad (1,656 genes).
Output : results/.../rctd/inputs/{method}_rctd_input.h5ad  (raw counts,
         cells x 1,656 genes, cells with >=1 panel UMI kept).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import anndata as ad

ROOT = Path(__file__).resolve().parents[4]
RES = ROOT / "results"
WT = RES / "kidney_visiumhd_noseg_bin2cell_benchmark/whole_transcriptome"
OUT = RES / "kidney_visiumhd_noseg_bin2cell_benchmark/rctd/inputs"
OUT.mkdir(parents=True, exist_ok=True)

WT_PATHS = {
    "10x_segmented": WT / "tenx_segmented_cell_by_gene.h5ad",
    "bin2cell_2um": WT / "bin2cell_cell_by_gene_whole_transcriptome.h5ad",
    "tracer_2um": RES / "tracer_noseg/kidney_visiumhd_2um/outputs/profile_by_gene_whole_transcriptome.h5ad",
    "tracer_8um": RES / "tracer_noseg/kidney_visiumhd_8um/outputs/profile_by_gene_whole_transcriptome.h5ad",
}
# Canonical 1,656 HVG/NPMI panel = var_names of TRACER's reconstruction matrix.
PANEL_H5AD = RES / "tracer_noseg/kidney_visiumhd_2um/outputs/profile_by_gene.h5ad"


def main():
    panel = list(ad.read_h5ad(PANEL_H5AD, backed="r").var_names)
    print(f"[rctd-in] panel: {len(panel)} HVG/NPMI genes")
    stats = {}
    for method, p in WT_PATHS.items():
        a = ad.read_h5ad(p)
        keep_genes = [g for g in panel if g in set(a.var_names)]
        sub = a[:, keep_genes].copy()
        X = sp.csr_matrix(sub.X)
        # Round to integer counts but store as float64: R's anndata maps a CSR
        # matrix to dgRMatrix, whose 'x' slot must be double (int32 -> error).
        X.data = np.rint(X.data).astype(np.float64)
        X.eliminate_zeros()
        umi = np.asarray(X.sum(1)).ravel()
        keep = umi >= 1
        sub = sub[keep].copy()
        sub.X = X[keep]
        # carry centroid coords so RCTD coords are real (optional, for QC)
        if "centroid_x" in sub.obs:
            sub.obs["x_centroid"] = sub.obs["centroid_x"].to_numpy()
            sub.obs["y_centroid"] = sub.obs["centroid_y"].to_numpy()
        outp = OUT / f"{method}_rctd_input.h5ad"
        sub.write_h5ad(outp)
        stats[method] = {"path": str(outp), "n_cells": int(sub.n_obs),
                         "n_panel_genes": int(sub.n_vars),
                         "median_panel_umi": float(np.median(np.asarray(sub.X.sum(1)).ravel()))}
        print(f"[rctd-in] {method}: {sub.n_obs:,} cells x {sub.n_vars} genes "
              f"(median panel UMI {stats[method]['median_panel_umi']:.0f}) -> {outp}")
    (OUT / "rctd_input_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
