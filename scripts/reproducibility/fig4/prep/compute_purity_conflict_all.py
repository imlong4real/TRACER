#!/usr/bin/env python3
"""Compute TRACER NPMI relative purity / conflict for all four methods, identically.

relative_purity + relative_conflict == 1 per cell (ReLU signal split), so the
per-method means form a clean stacked bar for Panel E.

We score every method's cells the SAME way: binary gene co-presence over the
kidney NPMI graph (`compute_purity_conflict_per_cc_relu`), using the already
gene-matched 1,656-gene RCTD input matrices. This makes the purity/conflict
comparison fair across 10x, bin2cell, TRACER 2 µm, and TRACER 8 µm (TRACER's own
stored scores come from transcript-level CCs and are not directly comparable to
segmented-cell methods).

Output: source_data/panel_E_purity_conflict.csv (per-method means + n) and
        source_data/panel_E_purity_conflict_percell_<method>.csv.gz
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fig4_config as C
from tracer.cc_scoring import build_pmi_matrix_from_long
from tracer.metrics import compute_cell_coherence

NPMI_CSV = C.RES / "kidney_visiumhd_rctd_tracer/reference/kidney_visiumhd_npmi.csv.gz"
RCTD_IN = C.RCTD / "inputs"
BATCH = 20000


def _load_npmi_long():
    d = pd.read_csv(NPMI_CSV)
    ren = {}
    for a in d.columns:
        la = a.lower()
        if la in ("gene_i", "gene_1", "genea", "gene_a", "source"): ren[a] = "gene_i"
        elif la in ("gene_j", "gene_2", "geneb", "gene_b", "target"): ren[a] = "gene_j"
        elif la in ("npmi", "weight", "value"): ren[a] = "NPMI"
    d = d.rename(columns=ren)
    assert {"gene_i", "gene_j", "NPMI"}.issubset(d.columns), f"npmi cols={list(d.columns)}"
    return d[["gene_i", "gene_j", "NPMI"]]


def main():
    print("[pc] building NPMI matrix ...", flush=True)
    npmi_long = _load_npmi_long()
    genes, gene_to_idx, npmi_mat, _ = build_pmi_matrix_from_long(npmi_long)
    print(f"[pc] NPMI graph: {len(genes)} genes", flush=True)

    means = []
    for method in C.METHOD_ORDER:
        key = {"10x_segmented": "10x_segmented", "bin2cell": "bin2cell_2um",
               "tracer_2um": "tracer_2um", "tracer_8um": "tracer_8um"}[method]
        a = ad.read_h5ad(RCTD_IN / f"{key}_rctd_input.h5ad")
        keep = [g for g in a.var_names if g in gene_to_idx]
        sub = a[:, keep]
        col_idx = np.array([gene_to_idx[g] for g in keep], dtype=np.int32)
        X = sp.csr_matrix(sub.X)
        n = X.shape[0]
        rp = np.zeros(n, np.float32); rc = np.zeros(n, np.float32); ss = np.zeros(n, np.float32)
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            M = (X[s:e].toarray() > 0).astype(np.float32)
            # count-based coherence at PMI enrichment cutoff (0.2); relative
            # purity/conflict = signal-normalized fractions of the pair counts.
            _coh, _p, _c, _ = compute_cell_coherence(M, col_idx, npmi_mat, threshold=0.2)
            _tot = _p + _c
            with np.errstate(invalid="ignore", divide="ignore"):
                r_p = np.where(_tot > 0, _p / _tot, np.nan)
                r_c = np.where(_tot > 0, _c / _tot, np.nan)
            rp[s:e], rc[s:e], ss[s:e] = r_p, r_c, _tot
        # save per-cell with cell_id so QC filtering can be applied downstream
        pc = pd.DataFrame({"cell_id": a.obs_names.astype(str),
                           "relative_purity": rp, "relative_conflict": rc,
                           "signal_strength": ss})
        pc.to_csv(C.SRCDIR / f"panel_E_purity_conflict_percell_{method}.csv.gz", index=False)
        has = ss > 0
        means.append({"method": C.METHOD_DISPLAY[method], "n_cells": int(has.sum()),
                      "mean_relative_purity": round(float(rp[has].mean()), 4),
                      "mean_relative_conflict": round(float(rc[has].mean()), 4),
                      "median_relative_purity": round(float(np.median(rp[has])), 4)})
        print(f"[pc] {method}: n={int(has.sum()):,} "
              f"mean_rel_purity={means[-1]['mean_relative_purity']} "
              f"mean_rel_conflict={means[-1]['mean_relative_conflict']}", flush=True)

    out = pd.DataFrame(means)
    out.to_csv(C.SRCDIR / "panel_E_purity_conflict.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
