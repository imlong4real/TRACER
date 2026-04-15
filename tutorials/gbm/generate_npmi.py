#!/usr/bin/env python3
"""
Generate a nucleus-based NPMI table for a GBM Xenium transcript parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REQUIRED_COLUMNS = {"feature_name", "cell_id", "qv", "overlaps_nucleus"}
EXCLUDE_IDS = {"-1", "UNASSIGNED", "DROP", "nan"}
COMMON_COLUMN_ALIASES = {
    "x_location": "x",
    "y_location": "y",
    "z_location": "z",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a GBM NPMI matrix from transcript parquet.")
    parser.add_argument("--input", required=True, help="Transcript parquet path.")
    parser.add_argument("--output", required=True, help="Output CSV path for the NPMI table.")
    parser.add_argument("--qv-min", type=float, default=30.0, help="Minimum Xenium qv to keep.")
    parser.add_argument("--low-pct", type=float, default=20.0, help="Lower percentile for confident nuclei.")
    parser.add_argument("--high-pct", type=float, default=80.0, help="Upper percentile for confident nuclei.")
    parser.add_argument(
        "--min-occurrences-per-context",
        type=int,
        default=2,
        help="Minimum copies of a gene in a nucleus before it counts as present.",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _normalize_common_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rename_map = {
        src: dst
        for src, dst in COMMON_COLUMN_ALIASES.items()
        if src in df.columns and dst not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df, rename_map


def main() -> None:
    args = _parse_args()

    import numpy as np
    import pandas as pd

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_parquet(input_path)
    df, rename_map = _normalize_common_columns(df)
    _validate_columns(df, REQUIRED_COLUMNS)

    df = df.copy()
    df["feature_name"] = df["feature_name"].astype(str).str.strip()
    df["cell_id"] = df["cell_id"].astype(str)
    df["qv"] = pd.to_numeric(df["qv"], errors="coerce")

    filtered = df[df["qv"] >= args.qv_min].copy()
    filtered = filtered[filtered["overlaps_nucleus"] == 1].copy()
    filtered = filtered[~filtered["cell_id"].isin(EXCLUDE_IDS)].copy()
    filtered = filtered[filtered["feature_name"] != ""].copy()

    if filtered.empty:
        raise ValueError("No transcripts remain after qv and nucleus-overlap filtering.")

    nuc_counts = filtered.groupby("cell_id").size()
    if nuc_counts.empty:
        raise ValueError("No valid nuclei remain after filtering.")

    low_thres = np.percentile(nuc_counts, args.low_pct)
    high_thres = np.percentile(nuc_counts, args.high_pct)
    good_ids = nuc_counts[(nuc_counts >= low_thres) & (nuc_counts <= high_thres)].index
    confident = filtered[filtered["cell_id"].isin(good_ids)].copy()

    if confident.empty:
        raise ValueError("No confident nuclei remain after percentile filtering.")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    from tracer.metrics import compute_npmi

    npmi_df = compute_npmi(
        confident,
        group_key="cell_id",
        min_occurrences_per_context=args.min_occurrences_per_context,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    npmi_df.to_csv(output_path, index=False)

    print(f"Input transcripts: {len(df):,}")
    if rename_map:
        print(f"Normalized columns: {rename_map}")
    print(f"After qv/nucleus filter: {len(filtered):,}")
    print(f"Confident nuclei: {len(good_ids):,}")
    print(f"Confident transcripts: {len(confident):,}")
    print(f"Saved NPMI table to: {output_path}")


if __name__ == "__main__":
    main()
