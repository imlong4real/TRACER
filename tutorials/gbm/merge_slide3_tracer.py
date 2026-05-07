#!/usr/bin/env python3
"""Merge 8 TRACER-refined piece parquets into a single Slide 3 output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CATEGORY_COLS = ["cell_id_stitched", "cell_id_finetuned"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge TRACER piece outputs for Slide 3.")
    p.add_argument(
        "--manifest",
        default="tutorials/gbm/output/slide3_pieces/piece_run_manifest.csv",
        help="Piece run manifest CSV (repo-relative or absolute).",
    )
    p.add_argument(
        "--tracer-dir",
        default="tutorials/gbm/output/slide3_tracer",
        help="Directory containing per-piece TRACER output parquets.",
    )
    p.add_argument(
        "--output",
        default="tutorials/gbm/output/slide3_tracer_merged.parquet",
        help="Path for merged output parquet.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    manifest_path = Path(args.manifest)
    tracer_dir = Path(args.tracer_dir)
    output_path = Path(args.output)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not tracer_dir.is_dir():
        raise FileNotFoundError(f"Tracer output directory not found: {tracer_dir}")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest.sort_values("task_id").reset_index(drop=True)

    chunks: list[pd.DataFrame] = []
    total_rows = 0

    for _, row in manifest.iterrows():
        piece_name = Path(row["output_tracer_parquet"]).name
        piece_path = tracer_dir / piece_name

        if not piece_path.exists():
            raise FileNotFoundError(f"Missing piece parquet: {piece_path}")

        print(f"  Reading {piece_name} ...", flush=True)
        df = pd.read_parquet(piece_path)

        for col in CATEGORY_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df["piece_id"] = row["component_id"]
        df["slide_tissue_id"] = row["slide_tissue_id"]

        print(f"    {len(df):>12,} rows  (piece_id={row['component_id']})", flush=True)
        total_rows += len(df)
        chunks.append(df)

    print(f"\nConcatenating {len(chunks)} pieces ({total_rows:,} rows total) ...", flush=True)
    merged = pd.concat(chunks, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path} ...", flush=True)
    merged.to_parquet(output_path, index=False, compression="snappy")
    print(f"Done. {len(merged):,} rows written to {output_path}")


if __name__ == "__main__":
    main()
