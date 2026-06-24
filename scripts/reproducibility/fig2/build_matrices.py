#!/usr/bin/env python
"""Foundation for Figure 2 biology: original-segmentation and TRACER cell x gene.

Joins the raw Xenium transcript table (gene identity, original 10x cell_id) with
the TRACER partition table (final assignment `label` + per-phase entity type) on
`transcript_id`, then builds two AnnData count matrices:

  * original_cells.h5ad  — cells = original 10x segmentation (cell_id)
  * tracer_cells.h5ad     — cells = TRACER `label`, tagged complete / partial /
                            neighboring via etype_at_finalize (0/1/5)

Both share the same transcript universe (qv>=20, real genes only), so downstream
comparisons are like-for-like.

Run:
    python scripts/reproducibility/fig2/build_matrices.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

ROOT = Path(__file__).resolve().parents[3]
DSET = ROOT / "datasets/pancreas_cancer_xenium_10x"
OUTS = DSET / "Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs"
TRANS = OUTS / "transcripts.parquet"
PART = DSET / "pdac_io_partition_sequential.parquet"
OUTDIR = DSET / "processed/fig2"
OUTDIR.mkdir(parents=True, exist_ok=True)

QV_MIN = 20.0
CTRL_PREFIX = ("NegControlProbe", "NegControlCodeword", "antisense", "BLANK",
               "UnassignedCodeword", "DeprecatedCodeword", "Intergenic",
               "genomic_control")
# etype_at_finalize -> TRACER entity class
ETYPE = {0: "complete", 1: "partial", 5: "neighboring", 3: "unassigned"}


def build_matrix(cell_ids, feat_codes, genes, x, y, z, qv):
    """Return (csr counts cells x genes, obs DataFrame) for given cell labels."""
    cell_cat = pd.Categorical(cell_ids)
    cc = cell_cat.codes.astype(np.int64)
    ones = np.ones(len(cc), dtype=np.float32)
    M = sp.coo_matrix((ones, (cc, feat_codes)),
                      shape=(len(cell_cat.categories), len(genes))).tocsr()
    M.sum_duplicates()
    df = pd.DataFrame({"cc": cc, "x": x, "y": y, "z": z, "qv": qv})
    g = df.groupby("cc", sort=True)
    obs = pd.DataFrame(index=pd.Index(cell_cat.categories.astype(str), name="cell"))
    obs["centroid_x"] = g["x"].mean().values
    obs["centroid_y"] = g["y"].mean().values
    obs["centroid_z"] = g["z"].mean().values
    obs["n_counts"] = np.asarray(M.sum(1)).ravel()
    obs["n_genes"] = M.getnnz(axis=1)
    obs["mean_qv"] = g["qv"].mean().values
    return M, obs


def main():
    print("Loading transcripts.parquet …")
    tx = pd.read_parquet(TRANS, columns=["transcript_id", "cell_id", "feature_name",
                                         "x_location", "y_location", "z_location", "qv"])
    print(f"  {len(tx):,} transcripts")
    print("Loading partition (TRACER assignment) …")
    pt = pd.read_parquet(PART, columns=["transcript_id", "label",
                                        "etype_at_finalize"])

    # align on transcript_id (fast path if identical order)
    if tx["transcript_id"].equals(pt["transcript_id"]):
        print("  transcript_id identical order -> direct attach")
        tx["label"] = pt["label"].values
        tx["etype"] = pt["etype_at_finalize"].values
    else:
        print("  merging on transcript_id …")
        pt = pt.set_index("transcript_id")
        tx = tx.join(pt, on="transcript_id")
        tx.rename(columns={"etype_at_finalize": "etype"}, inplace=True)

    # filter: QV and real genes
    feat = tx["feature_name"].astype(str)
    is_ctrl = feat.str.startswith(CTRL_PREFIX)
    keep = (tx["qv"].values >= QV_MIN) & (~is_ctrl.values)
    tx = tx[keep].reset_index(drop=True)
    print(f"  {len(tx):,} transcripts after qv>={QV_MIN:.0f} + real genes")

    genes = np.array(sorted(tx["feature_name"].astype(str).unique()))
    print(f"  {len(genes)} genes")
    feat_cat = pd.Categorical(tx["feature_name"].astype(str), categories=genes)
    feat_codes = feat_cat.codes.astype(np.int64)
    x = tx["x_location"].values; y = tx["y_location"].values
    z = tx["z_location"].values; qv = tx["qv"].values

    # ---------- ORIGINAL segmentation ----------
    cid = tx["cell_id"].astype(str).values
    orig_mask = ~np.isin(cid, ["-1", "UNASSIGNED", "", "nan"])
    print(f"\nORIGINAL: {orig_mask.sum():,} assigned transcripts")
    Mo, obso = build_matrix(cid[orig_mask], feat_codes[orig_mask], genes,
                            x[orig_mask], y[orig_mask], z[orig_mask], qv[orig_mask])
    ao = ad.AnnData(X=Mo, obs=obso, var=pd.DataFrame(index=genes))
    ao.obs["source"] = "original"
    ao.write_h5ad(OUTDIR / "original_cells.h5ad", compression="gzip")
    print(f"  original_cells.h5ad: {ao.shape}")

    # ---------- TRACER ----------
    lab = tx["label"].astype(str).values
    tr_mask = ~np.isin(lab, ["UNASSIGNED", "-1", "", "nan"])
    print(f"\nTRACER: {tr_mask.sum():,} assigned transcripts")
    Mt, obst = build_matrix(lab[tr_mask], feat_codes[tr_mask], genes,
                            x[tr_mask], y[tr_mask], z[tr_mask], qv[tr_mask])
    # per-cell entity class (constant within a TRACER cell)
    et = pd.DataFrame({"lab": lab[tr_mask], "etype": tx["etype"].values[tr_mask]})
    cell_et = et.groupby("lab")["etype"].agg(lambda s: s.mode().iloc[0])
    obst["etype_code"] = cell_et.reindex(obst.index).values
    obst["entity_class"] = obst["etype_code"].map(ETYPE).fillna("complete")
    at = ad.AnnData(X=Mt, obs=obst, var=pd.DataFrame(index=genes))
    at.obs["source"] = "tracer"
    at.write_h5ad(OUTDIR / "tracer_cells.h5ad", compression="gzip")
    print(f"  tracer_cells.h5ad: {at.shape}")
    print("  entity classes:", at.obs["entity_class"].value_counts().to_dict())


if __name__ == "__main__":
    main()
