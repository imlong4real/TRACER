#!/usr/bin/env python3
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method already set
    pass
"""
Run HOT-NERD pipeline on melanoma tissue data.

Reads:
 - data/melanoma_transcripts_qv30.parquet
 - data/melanoma_nucleus_npmi.csv

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
import multiprocessing as mp

import pandas as pd
import numpy as np
import os
import smtplib
import ssl
from email.message import EmailMessage


mp.set_start_method("spawn", force=True)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "tutorials" / "melanoma" / "data"
    out_dir = repo_root / "tutorials" / "melanoma" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "melanoma_transcripts_qv30.parquet"
    npmi_csv = data_dir / "melanoma_nucleus_npmi.csv"

    if not parquet_path.exists():
        processed_path = data_dir / "processed" / "melanoma_transcripts_qv30.parquet"
        if processed_path.exists():
            parquet_path = processed_path
    if not npmi_csv.exists():
        processed_npmi = data_dir / "processed" / "melanoma_nucleus_npmi.csv"
        if processed_npmi.exists():
            npmi_csv = processed_npmi

    print("Starting HOT-NERD run on melanoma tissue")
    print(f"Reading transcripts from: {parquet_path}")
    t0 = time.time()

    # Read transcripts (parquet can be a directory)
    try:
        df_transcripts = pd.read_parquet(
            parquet_path,
            columns=[
                "transcript_id",
                "cell_id",
                "feature_name",
                "x_location",
                "y_location",
                "z_location",
            ],
        )
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

    if "x" not in df_transcripts.columns and "x_location" in df_transcripts.columns:
        df_transcripts = df_transcripts.rename(
            columns={"x_location": "x", "y_location": "y", "z_location": "z"}
        )
    if "z" not in df_transcripts.columns:
        df_transcripts["z"] = 0.0

    print("Loaded transcripts rows:", len(df_transcripts), "took", time.time() - t0, "s")

    print(f"Reading NPMI table from: {npmi_csv}")
    t0 = time.time()
    df_npmi = pd.read_csv(npmi_csv)
    print("Loaded npmi rows:", len(df_npmi), "took", time.time() - t0, "s")

    # Import hotnerd functions lazily (after paths are resolved)
    sys.path.insert(0, str(repo_root / "src"))
    # Cython modules are now auto-compiled via pyximport inside core.py on first import
    # No need to setup pyximport here again - it's already done in core.py

    from hotnerd import (
        prune_transcripts_fast,
        annotate_unassigned_components_fast,
        apply_stitching_to_transcripts_memory_efficient,
        enforce_spatial_coherence_fast,
        prune_genes_by_npmi_greedy,
        build_graph
    )

    def send_email(subject: str, body: str) -> None:
        """Send a short email using SMTP settings read from environment variables.

        Required environment variables:
          - EMAIL_SMTP_SERVER (e.g. smtp.gmail.com)
          - EMAIL_TO (recipient email)

        Optional:
          - EMAIL_SMTP_PORT (default 587)
          - EMAIL_SMTP_USER, EMAIL_SMTP_PASS (for auth)
          - EMAIL_FROM (defaults to EMAIL_SMTP_USER)
        """
        smtp_server = os.getenv("EMAIL_SMTP_SERVER")
        recipient = os.getenv("EMAIL_TO")
        if not smtp_server or not recipient:
            print("Email not configured (EMAIL_SMTP_SERVER or EMAIL_TO missing); skipping email: ", subject)
            return

        port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
        user = os.getenv("EMAIL_SMTP_USER")
        password = os.getenv("EMAIL_SMTP_PASS")
        sender = os.getenv("EMAIL_FROM", user or "hotnerd@localhost")

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        msg.set_content(body)

        context = ssl.create_default_context()
        try:
            with smtplib.SMTP(smtp_server, port, timeout=30) as server:
                server.starttls(context=context)
                if user and password:
                    server.login(user, password)
                server.send_message(msg)
            print(f"Sent email: {subject}")
        except Exception as e:
            print("Warning: failed to send email:", e)
    import torch
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
    pruned_fp = out_dir / "df_pruned.parquet"
    aux_fp = out_dir / "aux.pkl"

    if pruned_fp.exists() and aux_fp.exists():
        print("Found existing stage-1 outputs; loading df_pruned and aux from disk")
        df_pruned = pd.read_parquet(pruned_fp)
        import joblib

        aux = joblib.load(aux_fp)
    else:
        df_pruned, aux = prune_transcripts_fast(
        df=df_transcripts,
        npmi_df=df_npmi,
        cell_id_col="cell_id",
        gene_col="feature_name",
        threshold=-0.1,
        unassigned_id="UNASSIGNED",
        n_jobs=-1,
        show_progress=True,
    )
        # persist stage-1 outputs to allow resume
        try:
            df_pruned.to_parquet(pruned_fp, index=False)
            import joblib

            joblib.dump(aux, aux_fp)
            print(f"Saved stage-1 outputs: {pruned_fp}, {aux_fp}")
        except Exception as e:
            print("Warning: failed to save stage-1 outputs:", e)
    print("Stage 1 done: rows=", len(df_pruned), "took", time.time() - t0, "s")
    try:
        send_email(
            subject="HOT-NERD: Stage 1 complete",
            body=f"Stage 1 complete: pruned={len(df_pruned)} transcripts. Time: {time.time()-t0:.1f}s\nSaved: {pruned_fp if pruned_fp.exists() else 'not saved'}",
        )
    except Exception:
        pass

    # Stage 2: annotate unassigned components
    print("Stage 2: annotate_unassigned_components_fast (build graph + CCs)")
    t0 = time.time()
    final_fp = out_dir / "df_final.parquet"
    if final_fp.exists():
        print("Found existing stage-2 output; loading df_final from disk")
        df_final = pd.read_parquet(final_fp)
    else:
        df_final = annotate_unassigned_components_fast(
        df_pruned=df_pruned,
        aux=aux,
        build_graph_fn=build_graph_fast,
        prune_fn=prune_genes_by_npmi_greedy,
        coord_cols=("x", "y", "z"),
        k=8,
        dist_threshold=1.5,
        min_comp_size=100,
        npmi_threshold=-0.1,
        unassigned_final_col="cell_id_npmi_cons_p2",
        cell_id_col="cell_id",
        gene_col="feature_name",
        transcript_id_col="transcript_id",
        show_progress=True,
    )
        # persist stage-2 output
        try:
            df_final.to_parquet(final_fp, index=False)
            print(f"Saved stage-2 output: {final_fp}")
        except Exception as e:
            print("Warning: failed to save stage-2 output:", e)
    print("Stage 2 done: rows=", len(df_final), "took", time.time() - t0, "s")
    try:
        send_email(
            subject="HOT-NERD: Stage 2 complete",
            body=f"Stage 2 complete: annotated={len(df_final)} transcripts. Time: {time.time()-t0:.1f}s\nSaved: {final_fp if final_fp.exists() else 'not saved'}",
        )
    except Exception:
        pass

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
        deltaC_min=0.01,
        use_3d=True,
        dist_threshold=10.0,
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
    try:
        send_email(
            subject="HOT-NERD: Stage 3 complete",
            body=f"Stage 3 complete: stitched={len(df_stitched)} transcripts. Time: {time.time()-t0:.1f}s\nSaved: {stitched_fp}",
        )
    except Exception:
        pass

    # Stage 4: enforce spatial coherence (split large/multi-component labels)
    print("Stage 4: enforce_spatial_coherence_fast (split spatially disjoint labels)")
    t0 = time.time()
    df_split = enforce_spatial_coherence_fast(
        df_stitched=df_stitched,
        build_graph_fn=build_graph_fast,
        entity_col="cell_id_stitched",
        coord_cols=("x", "y", "z"),
        k=5,
        dist_threshold=10.0,
        out_col="cell_id_spatial",
        show_progress=True,
    )
    print("Stage 4 done: rows=", len(df_split), "took", time.time() - t0, "s")

    # Save spatial split result
    split_fp = out_dir / "df_split.parquet"
    print(f"Saving df_split to {split_fp}")
    df_split.to_parquet(split_fp, index=False)
    try:
        send_email(
            subject="HOT-NERD: Stage 4 complete",
            body=f"Stage 4 complete: split={len(df_split)} transcripts. Time: {time.time()-t0:.1f}s\nSaved: {split_fp}",
        )
    except Exception:
        pass

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
        deltaC_min=0.01,
        use_3d=True,
        dist_threshold=10.0,
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
    try:
        send_email(
            subject="HOT-NERD: Stage 5 complete",
            body=f"Stage 5 complete: finetuned={len(df_finetuned)} transcripts. Time: {time.time()-t0:.1f}s\nSaved: {finetuned_fp}",
        )
    except Exception:
        pass

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
    try:
        send_email(
            subject="HOT-NERD: Pipeline complete",
            body=f"Pipeline complete. Outputs:\n - {stitched_fp}\n - {split_fp}\n - {finetuned_fp}",
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
