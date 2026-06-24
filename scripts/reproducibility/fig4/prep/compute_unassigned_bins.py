#!/usr/bin/env python3
"""Compute the fraction of 2x2 µm bins left unassigned by each 2µm method,
on a COMMON denominator = the filtered in-tissue square_002um bin set.

  10x segmented : bin -> cell_id via barcode_mappings.parquet (None = unassigned)
  bin2cell      : bin -> bin2cell_label via bin2cell_bin_to_cell_assignment (0 = bg)
  TRACER 2 µm   : bin present in bin_to_profile_assignment

TRACER 8 µm is reported separately (8µm-bin grid) by the panel. Result cached to
source_data/unassigned_bins_2um.csv.
"""
from __future__ import annotations
import gzip
from pathlib import Path

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fig4_config as C

OUT = C.SRCDIR / "unassigned_bins_2um.csv"


def _filtered_2um_barcodes() -> pd.Index:
    bc = C.DATA / "binned_outputs/square_002um/filtered_feature_bc_matrix/barcodes.tsv.gz"
    with gzip.open(bc, "rt") as f:
        return pd.Index([l.strip() for l in f])


def main():
    bcs = _filtered_2um_barcodes()
    nbc = len(bcs)
    bcset = set(bcs)
    print(f"[unassigned] filtered 2µm in-tissue bins: {nbc:,}")
    rows = []

    # 10x
    bm = pd.read_parquet(C.DATA / "Visium_HD_Human_Kidney_FFPE_barcode_mappings.parquet",
                         columns=["square_002um", "cell_id"])
    bm = bm[bm["square_002um"].isin(bcset)]
    n_assigned = int(bm["cell_id"].notna().sum())
    rows.append(("10x", n_assigned, nbc))
    print(f"[unassigned] 10x: {n_assigned:,}/{nbc:,} assigned")
    del bm

    # bin2cell
    b2 = pd.read_parquet(C.RES / "bin2cell/kidney_visiumhd_2um/outputs/bin2cell_bin_to_cell_assignment.parquet",
                         columns=["bin_id", "bin2cell_label"])
    b2 = b2[b2["bin_id"].isin(bcset)]
    n_assigned = int((b2["bin2cell_label"].astype(int) != 0).sum())
    rows.append(("bin2cell", n_assigned, nbc))
    print(f"[unassigned] bin2cell: {n_assigned:,}/{nbc:,} assigned")
    del b2

    # TRACER 2 µm
    tr = pd.read_parquet(C.BIN_TO_PROFILE["tracer_2um"], columns=["bin_id"])
    n_assigned = int(tr[tr["bin_id"].isin(bcset)]["bin_id"].nunique())
    rows.append(("TRACER 2 µm", n_assigned, nbc))
    print(f"[unassigned] TRACER 2µm: {n_assigned:,}/{nbc:,} assigned")

    df = pd.DataFrame(rows, columns=["method", "n_assigned", "n_input_bins"])
    df["frac_unassigned"] = 1 - df["n_assigned"] / df["n_input_bins"]
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"[unassigned] wrote {OUT}")


if __name__ == "__main__":
    main()
