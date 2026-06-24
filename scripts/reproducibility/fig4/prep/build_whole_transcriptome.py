#!/usr/bin/env python3
"""Build whole-transcriptome profile/cell-by-gene matrices for Figure 4.

Rationale
---------
TRACER noseg *reconstruction* (grouping / stitching) was driven by the
1,656 HVG/NPMI gene panel. That HVG-only matrix is correct for TRACER's
internal NPMI purity / conflict scoring, but it is NOT what we want for
downstream *biological* benchmarking (RCTD, label transfer, marker
validation, per-cell-type Pearson to scRNA pseudobulk, gene/UMI counts).

For those downstream metrics we re-aggregate every reconstructed profile
back to its contributing original VisiumHD bins and sum the *full*
original bin-by-gene matrix (18,132 genes) across those bins. This yields
a whole-transcriptome profile-by-gene matrix per method, on the identical
spatial units that TRACER/bin2cell produced.

Methods handled
---------------
  TRACER 2um  : square_002um bins, map = outputs/bin_to_profile_assignment.parquet
  TRACER 8um  : square_008um bins, map = outputs/bin_to_profile_assignment.parquet
  bin2cell    : square_002um bins, map = outputs/bin2cell_bin_to_cell_assignment.parquet
                (bin2cell_label, 0 = background dropped)
  10x seg     : already whole-transcriptome integer counts
                (segmented_outputs/filtered_feature_cell_matrix.h5) -> transpose to cells x genes

All outputs carry raw integer counts in .X, plus obs columns
{n_bins, centroid_x, centroid_y, transferred_label} where available.

Usage
-----
    python scripts/reproducibility/fig4/prep/build_whole_transcriptome.py [--methods ...]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "datasets/kidney_visiumhd_10x/segmented_outputs"
RES = ROOT / "results"
WT_DIR = RES / "kidney_visiumhd_noseg_bin2cell_benchmark/whole_transcriptome"
WT_DIR.mkdir(parents=True, exist_ok=True)

BIN_H5 = {
    "square_002um": DATA / "binned_outputs/square_002um/filtered_feature_bc_matrix.h5",
    "square_008um": DATA / "binned_outputs/square_008um/filtered_feature_bc_matrix.h5",
}


def _log(msg):
    print(f"[wt] {msg}", flush=True)


def _load_bins(square: str) -> ad.AnnData:
    """Load the original filtered VisiumHD bin matrix (bins x genes, raw counts)."""
    h5 = BIN_H5[square]
    if not h5.exists():
        raise FileNotFoundError(f"missing bin matrix: {h5}")
    _log(f"loading bins {square} <- {h5}")
    a = sc.read_10x_h5(h5)            # bins x genes, raw counts
    a.var_names_make_unique()
    # CSR for fast row slicing
    a.X = sp.csr_matrix(a.X)
    _log(f"  {a.shape[0]:,} bins x {a.shape[1]:,} genes")
    return a


def _aggregate(bins: ad.AnnData, bin_to_group: pd.DataFrame,
               group_col: str, bin_col: str = "bin_id") -> ad.AnnData:
    """Sum bin counts into groups via a sparse one-hot matmul.

    bin_to_group: long table with one row per (bin, group) membership.
    Returns AnnData groups x genes (raw integer counts), obs has n_bins.
    """
    bidx = pd.Index(bins.obs_names)
    m = bin_to_group[bin_to_group[bin_col].isin(bidx)].copy()
    dropped = len(bin_to_group) - len(m)
    if dropped:
        _log(f"  {dropped:,} assignment rows reference bins absent from the "
             f"filtered matrix (dropped)")
    groups = pd.Index(sorted(m[group_col].astype(str).unique()))
    g_pos = {g: i for i, g in enumerate(groups)}
    b_pos = pd.Series(np.arange(len(bidx)), index=bidx)

    rows = m[group_col].astype(str).map(g_pos).to_numpy()
    cols = b_pos.reindex(m[bin_col]).to_numpy()
    onehot = sp.csr_matrix(
        (np.ones(len(m), dtype=np.float32), (rows, cols)),
        shape=(len(groups), bins.n_obs),
    )
    X = onehot @ bins.X                      # groups x genes
    X = sp.csr_matrix(X)
    X.data = np.rint(X.data).astype(np.int32)
    X.eliminate_zeros()

    n_bins = np.asarray(m.groupby(group_col).size().reindex(groups).fillna(0),
                        dtype=np.int64)
    out = ad.AnnData(X=X, var=pd.DataFrame(index=bins.var_names))
    out.obs_names = groups
    out.obs["n_bins"] = n_bins
    out.obs["n_genes"] = np.asarray((X > 0).sum(1)).ravel()
    out.obs["total_counts"] = np.asarray(X.sum(1)).ravel().astype(np.int64)
    return out


def _attach_meta(adata: ad.AnnData, labels: pd.DataFrame, id_col: str,
                 label_col="transferred_label", conf_col="transfer_confidence",
                 cx="centroid_x", cy="centroid_y"):
    labels = labels.copy()
    labels[id_col] = labels[id_col].astype(str)
    labels = labels.set_index(id_col)
    idx = adata.obs_names
    for src, dst in [(label_col, "transferred_label"), (conf_col, "transfer_confidence"),
                     (cx, "centroid_x"), (cy, "centroid_y")]:
        if src in labels.columns:
            adata.obs[dst] = labels[src].reindex(idx).to_numpy()
    return adata


def build_tracer(run: str, square: str) -> ad.AnnData:
    rdir = RES / f"tracer_noseg/{run}"
    bp = pd.read_parquet(rdir / "outputs/bin_to_profile_assignment.parquet",
                         columns=["bin_id", "reconstructed_profile_id"])
    bp["reconstructed_profile_id"] = bp["reconstructed_profile_id"].astype(str)
    bins = _load_bins(square)
    out = _aggregate(bins, bp, "reconstructed_profile_id")
    lab = pd.read_csv(rdir / "label_transfer/reconstructed_profiles_with_labels.tsv.gz", sep="\t")
    out = _attach_meta(out, lab, "reconstructed_profile_id")
    return out


def build_bin2cell() -> ad.AnnData:
    bdir = RES / "bin2cell/kidney_visiumhd_2um"
    bp = pd.read_parquet(bdir / "outputs/bin2cell_bin_to_cell_assignment.parquet",
                         columns=["bin_id", "bin2cell_label"])
    bp = bp[bp["bin2cell_label"].astype(int) != 0].copy()   # 0 = background
    bp["bin2cell_label"] = bp["bin2cell_label"].astype(int).astype(str)
    bins = _load_bins("square_002um")
    out = _aggregate(bins, bp, "bin2cell_label")
    lab = pd.read_csv(bdir / "label_transfer/bin2cell_profiles_with_labels.tsv.gz", sep="\t")
    lab["cell_id"] = lab["cell_id"].astype(str)
    out = _attach_meta(out, lab, "cell_id")
    return out


def build_10x() -> ad.AnnData:
    h5 = DATA / "filtered_feature_cell_matrix.h5"
    _log(f"loading 10x segmented cells <- {h5}")
    a = sc.read_10x_h5(h5)               # cells x genes, raw counts
    a.var_names_make_unique()
    a.X = sp.csr_matrix(a.X).astype(np.int32)
    a.obs["n_bins"] = np.nan             # not bin-derived
    a.obs["n_genes"] = np.asarray((a.X > 0).sum(1)).ravel()
    a.obs["total_counts"] = np.asarray(a.X.sum(1)).ravel().astype(np.int64)
    annot = pd.read_csv(
        RES / "tracer_noseg/kidney_visiumhd_8um/validation_plots/_10x_labels/"
        "kidney_10x_seg_transferred_cell_annotations.csv")
    a = _attach_meta(a, annot, "cell_id")
    return a


TARGETS = {
    "tracer_2um": (lambda: build_tracer("kidney_visiumhd_2um", "square_002um"),
                   RES / "tracer_noseg/kidney_visiumhd_2um/outputs/profile_by_gene_whole_transcriptome.h5ad"),
    "tracer_8um": (lambda: build_tracer("kidney_visiumhd_8um", "square_008um"),
                   RES / "tracer_noseg/kidney_visiumhd_8um/outputs/profile_by_gene_whole_transcriptome.h5ad"),
    "bin2cell_2um": (build_bin2cell,
                     WT_DIR / "bin2cell_cell_by_gene_whole_transcriptome.h5ad"),
    "10x_segmented": (build_10x,
                      WT_DIR / "tenx_segmented_cell_by_gene.h5ad"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(TARGETS),
                    choices=list(TARGETS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    stats = {}
    for m in args.methods:
        builder, outp = TARGETS[m]
        if outp.exists() and not args.overwrite:
            _log(f"[skip] {m}: exists {outp} (use --overwrite)")
            a = ad.read_h5ad(outp, backed="r")
        else:
            t0 = time.time()
            a = builder()
            outp.parent.mkdir(parents=True, exist_ok=True)
            a.write_h5ad(outp)
            _log(f"[{m}] wrote {outp}  ({time.time()-t0:.1f}s)")
        stats[m] = {
            "path": str(outp),
            "n_profiles": int(a.n_obs),
            "n_genes": int(a.n_vars),
            "total_umis": int(np.asarray(a.X.sum()) if not a.isbacked else a.obs["total_counts"].sum()),
            "median_genes_per_profile": float(np.nanmedian(a.obs["n_genes"])),
            "median_umis_per_profile": float(np.nanmedian(a.obs["total_counts"])),
        }
        _log(f"[{m}] {stats[m]}")

    (WT_DIR / "whole_transcriptome_stats.json").write_text(json.dumps(stats, indent=2))
    _log(f"wrote stats -> {WT_DIR/'whole_transcriptome_stats.json'}")


if __name__ == "__main__":
    main()
