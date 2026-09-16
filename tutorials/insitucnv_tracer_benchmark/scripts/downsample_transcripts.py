#!/usr/bin/env python3
"""Apply gene-panel and detection-efficiency downsampling to transcript rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_parent, require_columns, standardize_transcript_columns, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcripts", required=True, type=Path)
    p.add_argument("--panel", required=True, type=Path, help="One retained gene per line.")
    p.add_argument("--detection-fraction", required=True, type=float)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--retained-ids-out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not (0 < args.detection_fraction <= 1):
        raise SystemExit("--detection-fraction must be in (0, 1]")
    panel = {g.strip() for g in args.panel.read_text().splitlines() if g.strip()}
    df = standardize_transcript_columns(pd.read_parquet(args.transcripts))
    require_columns(df, {"feature_name", "transcript_id"}, "transcripts")
    n0 = len(df)
    df = df.loc[df["feature_name"].astype(str).isin(panel)].copy()

    rng = np.random.default_rng(args.seed)
    if args.detection_fraction < 1:
        keep = rng.random(len(df)) < args.detection_fraction
        df = df.loc[keep].copy()

    ensure_parent(args.out)
    df.to_parquet(args.out, index=False)

    ids_out = args.retained_ids_out or args.out.with_suffix(args.out.suffix + ".retained_transcript_ids.tsv")
    ensure_parent(ids_out)
    pd.Series(df["transcript_id"].astype(str), name="transcript_id").to_csv(ids_out, sep="\t", index=False)

    write_json(
        args.out.with_suffix(args.out.suffix + ".summary.json"),
        {
            "input": str(args.transcripts),
            "panel": str(args.panel),
            "detection_fraction": args.detection_fraction,
            "seed": args.seed,
            "rows_input": n0,
            "rows_output": len(df),
            "genes_output": int(df["feature_name"].nunique()),
            "retained_ids": str(ids_out),
        },
    )
    print(f"Wrote {len(df):,} transcripts ({df['feature_name'].nunique():,} genes) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

