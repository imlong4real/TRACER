#!/usr/bin/env python3
"""
Run HOT-NERD pipeline on mouse ileum tissue data.

Reads:
 - data/mouse_gut_df.parquet  (Parquet file or directory)
 - data/mouse_gut_npmi.csv
Writes:
 - output/df_stitched.parquet
 - output/df_split.parquet
 - output/df_finetuned.parquet

This script prints progress messages for each stage.
"""
import json
import time
from pathlib import Path
import sys
import argparse

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Run HOT-NERD pipeline on mouse ileum tissue")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed (default: 42)")
    parser.add_argument("--run-smoke-test", action="store_true", help="Run deterministic reproducibility smoke test and exit")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "tutorials" / "mouse_ileum" / "data"
    out_dir = repo_root / "tutorials" / "mouse_ileum" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "mouse_gut_df.parquet"
    npmi_csv = data_dir / "mouse_gut_npmi.csv"

    print("Starting HOT-NERD run on mouse ileum tissue")
    print(f"Reading transcripts from: {parquet_path}")
    t0 = time.time()

    # Read transcripts (parquet can be a directory)
    try:
        df_transcripts = pd.read_parquet(parquet_path)
    except Exception as e:
        print("Failed to read parquet with pandas (pyarrow):", e)
        print("Trying fallback: pandas + fastparquet engine (if installed)...")
        try:
            df_transcripts = pd.read_parquet(parquet_path, engine="fastparquet")
        except Exception as e2:
            print("fastparquet fallback failed:", e2)
            # Try a lower-level pyarrow per-file read (more tolerant in some cases)
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                print("Attempting pyarrow per-file read fallback...")
                files = []
                if parquet_path.is_dir():
                    files = sorted(parquet_path.glob("*.parquet"))
                else:
                    files = [parquet_path]

                tables = []
                for f in files:
                    try:
                        pf = pq.ParquetFile(str(f))
                        tbl = pf.read()
                        tables.append(tbl)
                    except Exception as ef:
                        print(f"pyarrow failed on file {f}:", ef)
                        raise

                if not tables:
                    raise RuntimeError("No parquet files found for pyarrow fallback")

                combined = pa.concat_tables(tables)
                df_transcripts = combined.to_pandas()
                print("pyarrow per-file read succeeded; combined rows:", len(df_transcripts))
            except Exception as e_pa:
                print("pyarrow per-file fallback failed:", e_pa)
                print("Trying Dask to read the dataset lazily (requires dask)")
                try:
                    import dask.dataframe as dd

                    ddf = dd.read_parquet(str(parquet_path), engine="pyarrow")
                    print("Dask read succeeded. Attempting to compute into a pandas DataFrame (may require lots of RAM)")
                    try:
                        df_transcripts = ddf.compute()
                    except MemoryError:
                        print("MemoryError: dataset is too large to hold in memory as a single pandas DataFrame.")
                        print("Options:")
                        print(" - Run the tiled pipeline notebook `examples/large_tissue_pipeline.ipynb` to process by tile.")
                        print(" - Increase available memory or run on a Dask cluster and adapt the script to process tiles lazily.")
                        raise
                    except Exception as e3:
                        print("Failed to compute Dask dataframe:", e3)
                        raise
                except Exception as e_dd:
                    print("Dask fallback failed or not installed:", e_dd)
                    print("Cannot read Parquet. Please ensure the file/directory is a valid Parquet dataset, or install 'fastparquet' or 'dask'.")
                    raise

    print("Loaded transcripts rows:", len(df_transcripts), "took", time.time() - t0, "s")

    print(f"Reading NPMI table from: {npmi_csv}")
    t0 = time.time()
    df_npmi = pd.read_csv(npmi_csv)
    print("Loaded npmi rows:", len(df_npmi), "took", time.time() - t0, "s")

    # Import hotnerd functions lazily (after paths are resolved)
    sys.path.insert(0, str(repo_root / "src"))
    # Import reproducibility helpers from core
    from hotnerd.core import set_reproducibility_seed, reproducibility_smoke_test

    # Apply master seed for reproducibility
    set_reproducibility_seed(args.seed)

    # If requested, run smoke test and exit
    if args.run_smoke_test:
        print("Running reproducibility smoke test...")
        reproducibility_smoke_test(seed=args.seed)
        print("Smoke test passed")
        sys.exit(0)

    from hotnerd import (
        prune_transcripts_fast,
        annotate_unassigned_components_fast,
        apply_stitching_to_transcripts_memory_efficient,
        enforce_spatial_coherence_fast,
        prune_genes_by_npmi_greedy,
        build_graph
    )
    import torch
    # also seed torch RNG for extra determinism
    torch.manual_seed(args.seed)
    from torch_geometric.data import Data
    from scipy.spatial import cKDTree

    # Fast cKDTree-based graph builder (2-3x faster than sklearn NearestNeighbors)
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

    # Stage 1: conservative NPMI pruning
    print("Stage 1: prune_transcripts_fast (conservative NPMI)")
    t0 = time.time()
    df_pruned, aux = prune_transcripts_fast(
        df=df_transcripts,
        npmi_df=df_npmi,
        cell_id_col="cell_id",
        gene_col="feature_name",
        threshold=-0.05,
        unassigned_id="-1",
        n_jobs=-1,
        show_progress=True,
    )
    print("Stage 1 done: rows=", len(df_pruned), "took", time.time() - t0, "s")

    # Stage 2: annotate unassigned components
    print("Stage 2: annotate_unassigned_components_fast (build graph + CCs)")
    t0 = time.time()
    df_final = annotate_unassigned_components_fast(
        df_pruned=df_pruned,
        aux=aux,
        build_graph_fn=build_graph_fast,
        prune_fn=prune_genes_by_npmi_greedy,
        coord_cols=("x", "y", "z"),
        k=8,
        dist_threshold=1.5,
        min_comp_size=10,
        npmi_threshold=-0.1,
        unassigned_final_col="cell_id_npmi_cons_p2",
        cell_id_col="cell_id",
        gene_col="feature_name",
        transcript_id_col="transcript_id",
        show_progress=True,
    )
    print("Stage 2 done: rows=", len(df_final), "took", time.time() - t0, "s")

    # Stage 3: initial stitching
    print("Stage 3: apply_stitching_to_transcripts_memory_efficient (initial stitching)")
    t0 = time.time()
    df_stitched, entity_to_stitched = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_final,
        aux=aux,
        entity_col="cell_id_final",
        gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        purity_threshold=0.05,
        penalize_simplicity=True,
        deltaC_min=0.0,
        dist_threshold=20.0,
        use_3d=True,
        out_col="cell_id_stitched",
        show_progress=True,
        in_place=False,
        map_mode="categorical",
    )
    print("Stage 3 done: rows=", len(df_stitched), "took", time.time() - t0, "s")

    # Save intermediate stitched result
    stitched_fp = out_dir / "df_stitched.parquet"
    print(f"Saving df_stitched to {stitched_fp}")
    df_stitched.to_parquet(stitched_fp, index=False)

    # Stage 4: enforce spatial coherence (split large/multi-component labels)
    print("Stage 4: enforce_spatial_coherence_fast (split spatially disjoint labels)")
    t0 = time.time()
    df_split = enforce_spatial_coherence_fast(
        df_stitched=df_stitched,
        build_graph_fn=build_graph_fast,
        entity_col="cell_id_stitched",
        coord_cols=("x", "y", "z"),
        k=5,
        dist_threshold=20.0,
        out_col="cell_id_spatial",
        show_progress=True,
    )
    print("Stage 4 done: rows=", len(df_split), "took", time.time() - t0, "s")

    # Save spatial split result
    split_fp = out_dir / "df_split.parquet"
    print(f"Saving df_split to {split_fp}")
    df_split.to_parquet(split_fp, index=False)

    # Stage 5: re-run stitching with spatial splits
    print("Stage 5: apply_stitching_to_transcripts_memory_efficient (final stitching on split labels)")
    t0 = time.time()
    df_finetuned, entity_to_stitched_ft = apply_stitching_to_transcripts_memory_efficient(
        df_final=df_split,
        aux=aux,
        entity_col="cell_id_spatial",
        gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        purity_threshold=0.05,
        penalize_simplicity=True,
        deltaC_min=0.0,
        dist_threshold=20.0,
        use_3d=True,
        out_col="cell_id_finetuned",
        show_progress=True,
        in_place=False,
        map_mode="categorical",
    )
    print("Stage 5 done: rows=", len(df_finetuned), "took", time.time() - t0, "s")

    # Save final finetuned result
    finetuned_fp = out_dir / "df_finetuned.parquet"
    print(f"Saving df_finetuned to {finetuned_fp}")
    df_finetuned.to_parquet(finetuned_fp, index=False)

    # Save entity maps for debugging
    try:
        with open(out_dir / "entity_to_stitched.json", "w") as f:
            json.dump({str(k): str(v) for k, v in entity_to_stitched.items()}, f)
        with open(out_dir / "entity_to_stitched_finetuned.json", "w") as f:
            json.dump({str(k): str(v) for k, v in entity_to_stitched_ft.items()}, f)
    except Exception:
        print("Warning: failed to write entity mapping json files")

    print("Pipeline complete. Outputs:")
    print(" -", stitched_fp)
    print(" -", split_fp)
    print(" -", finetuned_fp)


if __name__ == "__main__":
    main()
