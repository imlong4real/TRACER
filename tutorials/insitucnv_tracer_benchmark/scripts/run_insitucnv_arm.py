#!/usr/bin/env python3
"""Run infercnvpy/InSituCNV on one raw or TRACER entity arm.

This script builds AnnData from a transcript parquet so raw and TRACER arms use
the exact same retained transcript rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import UNASSIGNED_TOKENS, ensure_parent, read_gene_positions, require_columns, standardize_transcript_columns, write_json

TRACER_WHOLE_ETYPES = {"cell"}
TRACER_ALL_ETYPES = {"cell", "partial"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcripts", required=True, type=Path)
    p.add_argument("--arm", choices=["raw", "tracer_whole", "tracer_all"], required=True)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--gene-positions", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path, help="CSV from annotate_entities.py with entity_id,annotation.")
    p.add_argument("--min-transcripts", type=int, default=20)
    p.add_argument("--min-genes", type=int, default=10)
    p.add_argument("--reference-annotations", default="t_cell,b_cell,myeloid,fibroblast,endothelial")
    p.add_argument("--tumor-annotations", default="epithelial")
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--cluster-resolution", type=float, default=0.4)
    p.add_argument("--smoothing-neighbors", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_entity_col(df: pd.DataFrame, arm: str) -> str:
    if arm == "raw":
        return "cell_id"
    for col in ("cell_id_tracer", "stitched", "cell_id_finetuned"):
        if col in df.columns:
            return col
    raise SystemExit("No TRACER entity column found; looked for cell_id_tracer, stitched, cell_id_finetuned")


def filter_arm(df: pd.DataFrame, arm: str, entity_col: str) -> pd.DataFrame:
    out = df.copy()
    out[entity_col] = out[entity_col].astype(str)
    out = out.loc[~out[entity_col].isin(UNASSIGNED_TOKENS)].copy()
    if arm.startswith("tracer") and "_etype" in out.columns:
        keep = TRACER_WHOLE_ETYPES if arm == "tracer_whole" else TRACER_ALL_ETYPES
        out = out.loc[out["_etype"].astype(str).isin(keep)].copy()
    return out


def build_adata(df: pd.DataFrame, entity_col: str, annotations: pd.DataFrame, min_tx: int, min_genes: int):
    import anndata as ad
    from scipy import sparse as sp

    counts = df.groupby([entity_col, "feature_name"], observed=True).size().rename("count").reset_index()
    n_tx = counts.groupby(entity_col, observed=True)["count"].sum()
    n_genes = counts.groupby(entity_col, observed=True).size()
    keep_entities = n_tx.index[(n_tx >= min_tx) & (n_genes >= min_genes)].astype(str)
    counts = counts.loc[counts[entity_col].astype(str).isin(set(keep_entities))].copy()
    cell_cat = pd.Categorical(counts[entity_col].astype(str), categories=keep_entities)
    gene_cat = pd.Categorical(counts["feature_name"].astype(str))
    X = sp.csr_matrix(
        (counts["count"].to_numpy(np.float32), (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(gene_cat.categories)),
    )
    obs = pd.DataFrame(index=pd.Index(cell_cat.categories.astype(str), name="entity_id"))
    obs["n_transcripts"] = n_tx.reindex(obs.index).fillna(0).astype(int)
    obs["n_genes"] = n_genes.reindex(obs.index).fillna(0).astype(int)
    ann = annotations.set_index("entity_id")["annotation"].astype(str)
    obs["annotation"] = ann.reindex(obs.index).fillna("unknown")
    var = pd.DataFrame(index=pd.Index(gene_cat.categories.astype(str), name="gene"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    if {"x", "y"}.issubset(df.columns):
        spatial = df.groupby(entity_col, observed=True)[["x", "y"]].mean()
        adata.obsm["spatial"] = spatial.reindex(adata.obs_names).to_numpy(dtype=float)
    return adata


def add_gene_positions(adata, gene_positions: Path):
    pos = read_gene_positions(gene_positions).set_index("gene")
    common = adata.var_names.intersection(pos.index)
    adata._inplace_subset_var(adata.var_names.isin(common))
    pos = pos.loc[adata.var_names]
    adata.var["chromosome"] = "chr" + pos["chromosome"].astype(str).str.replace("^chr", "", regex=True).values
    adata.var["start"] = pos["start"].astype(int).values
    adata.var["end"] = pos["end"].astype(int).values
    return adata


def assign_compartment(adata, reference: set[str], tumor: set[str]) -> None:
    ann = adata.obs["annotation"].astype(str).str.lower()
    adata.obs["compartment"] = "unknown"
    adata.obs.loc[ann.isin(reference), "compartment"] = "reference"
    adata.obs.loc[ann.isin(tumor), "compartment"] = "tumor"


def build_spatial_neighbors(adata, n_neighbors: int) -> int:
    import scvelo as scv

    if "spatial" not in adata.obsm:
        raise SystemExit("InSituCNV smoothing requires entity spatial coordinates.")
    if adata.n_obs < 2:
        raise SystemExit("InSituCNV smoothing requires at least two entities.")
    effective_neighbors = min(n_neighbors, adata.n_obs - 1)
    scv.pp.neighbors(
        adata,
        n_neighbors=effective_neighbors,
        n_pcs=None,
        use_rep="spatial",
    )
    return effective_neighbors


def run_infercnv(adata, window_size: int, resolution: float, smoothing_neighbors: int):
    import scanpy as sc
    import infercnvpy as cnv
    import insitucnv as icv

    effective_neighbors = build_spatial_neighbors(adata, smoothing_neighbors)
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    icv.tl.smooth_data_for_cnv(adata, n_neighbors=effective_neighbors)
    if "M" not in adata.layers:
        raise SystemExit("InSituCNV smoothing did not create adata.layers['M']")
    adata.X = adata.layers["M"].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    cnv.tl.infercnv(
        adata,
        reference_key="compartment",
        reference_cat=["reference"],
        window_size=window_size,
        layer="M",
    )
    cnv.tl.pca(adata)
    cnv.pp.neighbors(adata)
    cnv.tl.leiden(adata, resolution=resolution)
    return adata


def summarize_chromosomes(adata) -> pd.DataFrame:
    from scipy import sparse as sp

    if "gene_values_cnv" not in adata.layers:
        return pd.DataFrame()
    X = adata.layers["gene_values_cnv"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    chrom = adata.var["chromosome"].astype(str).to_numpy()
    rows = []
    for comp in sorted(adata.obs["compartment"].astype(str).unique()):
        mask = adata.obs["compartment"].astype(str).to_numpy() == comp
        if not mask.any():
            continue
        per_gene = X[mask].mean(axis=0)
        row = pd.Series(per_gene, index=chrom).groupby(level=0).mean()
        row["compartment"] = comp
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df = standardize_transcript_columns(pd.read_parquet(args.transcripts))
    require_columns(df, {"feature_name", "cell_id"}, "transcripts")
    entity_col = resolve_entity_col(df, args.arm)
    df = filter_arm(df, args.arm, entity_col)
    annotations = pd.read_csv(args.annotations)
    require_columns(annotations, {"entity_id", "annotation"}, str(args.annotations))

    adata = build_adata(df, entity_col, annotations, args.min_transcripts, args.min_genes)
    adata = add_gene_positions(adata, args.gene_positions)
    assign_compartment(
        adata,
        {x.strip().lower() for x in args.reference_annotations.split(",") if x.strip()},
        {x.strip().lower() for x in args.tumor_annotations.split(",") if x.strip()},
    )
    summary = {
        "arm": args.arm,
        "entity_col": entity_col,
        "n_entities": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "compartment_counts": adata.obs["compartment"].value_counts().to_dict(),
        "median_transcripts": float(adata.obs["n_transcripts"].median()) if adata.n_obs else 0.0,
        "median_genes": float(adata.obs["n_genes"].median()) if adata.n_obs else 0.0,
        "dry_run": args.dry_run,
    }
    write_json(outdir / "arm_stats.json", summary)
    if args.dry_run:
        adata.obs.to_csv(outdir / "obs_preview.csv")
        print(json.dumps(summary, indent=2))
        return 0
    if int((adata.obs["compartment"] == "reference").sum()) == 0:
        raise SystemExit("No reference entities after QC/annotation; cannot run inferCNV.")
    adata = run_infercnv(adata, args.window_size, args.cluster_resolution, args.smoothing_neighbors)
    adata.write_h5ad(outdir / "adata_cnv.h5ad")
    chrom = summarize_chromosomes(adata)
    if not chrom.empty:
        chrom.to_csv(outdir / "chrom_cnv_by_compartment.csv", index=False)
    print(f"Wrote InSituCNV arm outputs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
