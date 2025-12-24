#!/usr/bin/env python3
"""
Run HOT-NERD Stages 4-5 (spatial coherence + final stitching) on breast cancer tissue.

This script reloads the df_stitched.parquet output from Stage 3 and completes
the pipeline with optimized Stage 5 apply_labels mapping.

Reads:
 - output/df_stitched.parquet (from Stage 3)
 - data/breast_cancer_npmi.csv

Writes:
 - output/df_finetuned.parquet

Run: python tutorials/breast_cancer/run_stages_4_5.py
"""
import json
import time
from pathlib import Path
import sys

import pandas as pd
import numpy as np


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "tutorials" / "breast_cancer" / "data"
    out_dir = repo_root / "tutorials" / "breast_cancer" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    npmi_csv = data_dir / "breast_cancer_npmi.csv"
    stitched_parquet = out_dir / "df_stitched.parquet"

    print("Starting HOT-NERD Stages 4-5 (spatial coherence + final stitching)")
    print(f"Reading df_stitched from: {stitched_parquet}")
    t0 = time.time()

    # Load df_stitched from Stage 3
    try:
        df_stitched = pd.read_parquet(stitched_parquet)
    except Exception as e:
        print("Failed to read parquet:", e)
        raise

    print("Loaded df_stitched rows:", len(df_stitched), "took", time.time() - t0, "s")

    print("Reading NPMI table from:", npmi_csv)
    t0 = time.time()
    df_npmi = pd.read_csv(npmi_csv)
    print("Loaded npmi rows:", len(df_npmi), "took", time.time() - t0, "s")

    # Import hotnerd functions lazily (after paths are resolved)
    sys.path.insert(0, str(repo_root / "src"))
    # Cython modules are auto-compiled via pyximport inside core.py
    
    from hotnerd import (
        enforce_spatial_coherence_fast,
        apply_stitching_to_transcripts_fast,
        prune_genes_by_npmi_greedy,
        build_dense_npmi_matrix,
    )
    import torch
    from torch_geometric.data import Data
    from scipy.spatial import cKDTree

    # Build aux dict (same as Stage 3)
    print("Building NPMI matrix...")
    genes, gene_to_idx, W = build_dense_npmi_matrix(df_npmi)
    aux = {
        "genes": genes,
        "gene_to_idx": gene_to_idx,
        "W": W,
    }
    print("NPMI matrix shape:", W.shape)

    # Fast cKDTree-based graph builder
    def build_graph_fast(
        df,
        *,
        k=10,
        dist_threshold=1.5,
        coord_cols=("x", "y", "z"),
    ):
        coords = df[list(coord_cols)].to_numpy(dtype=np.float32)
        N = coords.shape[0]

        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=min(k + 1, N))

        # Vectorized distance mask (exclude self)
        if indices.ndim == 1:
            indices = indices[:, None]
            distances = distances[:, None]
        mask = (distances <= dist_threshold)
        mask[:, 0] = False  # remove self

        src = np.repeat(np.arange(N), mask.sum(axis=1))
        tgt = indices[mask]

        edge_index = torch.from_numpy(np.vstack([src, tgt]).astype(np.int64))

        data = Data(pos=torch.from_numpy(coords), edge_index=edge_index)
        data.gene_name = df["feature_name"].to_numpy().astype(str)
        data.id = df["transcript_id"].to_numpy().astype(str)

        print(f"Constructed {edge_index.shape[1]:,} edges among {N:,} transcripts (k≤{k}, d≤{dist_threshold} µm)")
        return data

    # Stage 4: spatial coherence
    print("\nStage 4: enforce_spatial_coherence_fast (split spatially disjoint labels)")
    t0 = time.time()
    df_spatial = enforce_spatial_coherence_fast(
        df_stitched,
        build_graph_fn=build_graph_fast,
        entity_col="cell_id_stitched",
        coord_cols=("x", "y", "z"),
        k=5,
        dist_threshold=3.0,
        out_col="cell_id_spatial",
        show_progress=True,
    )
    print("Stage 4 done: rows=", len(df_spatial), "took", time.time() - t0, "s")

    # Stage 5: final stitching
    print("\nStage 5: apply_stitching_to_transcripts_fast (final stitching on split labels)")
    t0 = time.time()
    df_finetuned, entity_to_stitched_ft = apply_stitching_to_transcripts_fast(
        df_final=df_spatial,
        aux=aux,
        entity_col="cell_id_spatial",
        gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        purity_threshold=0.05,
        penalize_simplicity=True,
        deltaC_min=0.0,
        use_3d=True,
        out_col="cell_id_finetuned",
        show_progress=True,
    )
    print("Stage 5 done: rows=", len(df_finetuned), "took", time.time() - t0, "s")

    # Save output
    print("\nSaving df_finetuned to", out_dir / "df_finetuned.parquet")
    df_finetuned.to_parquet(out_dir / "df_finetuned.parquet", index=False)
    print("Done!")


if __name__ == "__main__":
    main()
