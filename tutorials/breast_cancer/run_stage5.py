#!/usr/bin/env python3
"""
Run only Stage 5 (final stitching) on pre-computed df_split.parquet
with optimized settings: use_3d=False, deltaC_min=0.05
"""
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
    split_fp = out_dir / "df_split.parquet"

    print(f"Reading df_split from: {split_fp}")
    t0 = time.time()
    df_split = pd.read_parquet(split_fp)
    print(f"Loaded {len(df_split):,} transcripts, took {time.time() - t0:.2f}s")

    print(f"Reading NPMI table from: {npmi_csv}")
    t0 = time.time()
    df_npmi = pd.read_csv(npmi_csv)
    print(f"Loaded {len(df_npmi):,} gene pairs, took {time.time() - t0:.2f}s")

    # Import hotnerd
    sys.path.insert(0, str(repo_root / "src"))
    from hotnerd import apply_stitching_to_transcripts_fast
    from hotnerd.core import build_dense_npmi_matrix

    # Rebuild aux dict from NPMI CSV
    print("Building NPMI matrix...")
    t0 = time.time()
    genes, gene_to_idx, W = build_dense_npmi_matrix(df_npmi)
    aux = {
        "genes": genes,
        "gene_to_idx": gene_to_idx,
        "W": W,
        "partial_map": {},
        "threshold": -0.1,
    }
    print(f"NPMI matrix built ({len(genes)} genes), took {time.time() - t0:.2f}s")

    # Stage 5: re-run stitching with spatial splits (optimized)
    print("=" * 70)
    print("Stage 5: apply_stitching_to_transcripts_fast (OPTIMIZED)")
    print("  use_3d=False (2D Delaunay, much faster)")
    print("=" * 70)
    t0 = time.time()
    df_finetuned, entity_to_stitched_ft = apply_stitching_to_transcripts_fast(
        df_final=df_split,
        aux=aux,
        entity_col="cell_id_spatial",
        gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        tau=0.05,
        use_relu=True,
        penalize_simplicity=True,
        deltaC_min=0,  # raised from 0.0 to prune low-value merges
        use_3d=False,     # 2D Delaunay instead of 3D (much faster)
        out_col="cell_id_finetuned",
        show_progress=True,
    )
    elapsed = time.time() - t0
    print(f"Stage 5 done: {len(df_finetuned):,} rows, took {elapsed:.2f}s ({elapsed/60:.1f} min)")

    # Save result
    finetuned_fp = out_dir / "df_finetuned.parquet"
    print(f"Saving df_finetuned to {finetuned_fp}")
    df_finetuned.to_parquet(finetuned_fp, index=False)
    print("✓ Done!")


if __name__ == "__main__":
    main()
