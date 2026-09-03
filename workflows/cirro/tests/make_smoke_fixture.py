#!/usr/bin/env python3
"""Materialize the deterministic segmented fixture used by TRACER's tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, "/app")

from tests.synthetic import (  # noqa: E402
    make_synthetic_npmi_panel_for_transcripts,
    make_synthetic_transcripts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    transcripts, truth = make_synthetic_transcripts(
        n_cells=8,
        voxels_per_cell_mean=80,
        tx_per_cell=25,
        n_genes=12,
        n_types=3,
        domain_z_um=10.0,
        nuclear_layers=2,
        seed=42,
    )
    panel = make_synthetic_npmi_panel_for_transcripts(transcripts, truth)
    transcripts = transcripts.rename(columns={"is_nuclear": "overlaps_nucleus"})
    transcripts["overlaps_nucleus"] = transcripts["overlaps_nucleus"].astype(np.uint8)
    transcripts["qv"] = np.float32(40.0)

    transcripts.to_parquet(args.outdir / "synthetic_xenium_transcripts.parquet", index=False)
    panel.to_csv(args.outdir / "synthetic_cpmi.csv.gz", index=False, compression="gzip")
    with (args.outdir / "ground_truth.json").open("w", encoding="utf-8") as handle:
        json.dump(truth, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
