#!/usr/bin/env python3
"""Prepare a TRACER **seg-mode** input from VisiumHD bins + 10x nucleus masks.

VisiumHD has no per-molecule cell assignment, but 10x exports nucleus
polygons. This script overlays the 2 µm bin centers onto those polygons to
derive an initial nucleus seed per bin — the VisiumHD analogue of imaging
data's ``overlaps_nucleus`` — and explodes the bin×gene counts into the
transcript-like table that ``scripts/run_tracer.py`` (the canonical seg
pipeline) consumes.

Output parquet columns: ``transcript_id, feature_name, cell_id, bin_id,
x, y, z, overlaps_nucleus`` where:
  * ``cell_id`` = nucleus id for bins whose center falls in a nucleus, else
    ``"-1"`` (preserved so TRACER can rebuild residual/partial profiles);
  * ``overlaps_nucleus`` = 1 for nucleus-seeded bins, else 0 → makes
    ``run_segmented_pipeline`` take the nuclear-seed prune path (NOT noseg).

Coordinate frames
-----------------
TRACER x/y are the micron grid ``array_col*bin_size`` / ``array_row*bin_size``
(same as the noseg path, so seg/noseg are comparable). The polygon overlay
is done in **full-res pixel** space (``pxl_col/row_in_fullres``), which is
the frame the GeoJSON lives in — no transform needed there.

EXAMPLE
=======
::

    python scripts/prepare_visiumhd_seg_input.py \\
      --matrix-dir datasets/kidney_visiumhd_10x/segmented_outputs/binned_outputs/square_002um/filtered_feature_bc_matrix \\
      --spatial-dir datasets/kidney_visiumhd_10x/segmented_outputs/binned_outputs/square_002um/spatial \\
      --geojson datasets/kidney_visiumhd_10x/segmented_outputs/graphclust_annotated_nucleus_segmentations.geojson \\
      --npmi results/kidney_visiumhd_rctd_tracer/reference/kidney_visiumhd_npmi.csv.gz \\
      --roi-size-um 200 \\
      --out /tmp/kidney_seg_roi.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix-dir", required=True,
                   help="filtered_feature_bc_matrix dir (square_0NNum).")
    p.add_argument("--spatial-dir", required=True,
                   help="spatial/ dir with tissue_positions + scalefactors.")
    p.add_argument("--geojson", required=True,
                   help="10x nucleus segmentation GeoJSON (full-res pixels).")
    p.add_argument("--npmi", default=None,
                   help="NPMI panel csv(.gz); genes restrict the explode "
                        "(strongly recommended — without it all 18k genes "
                        "explode). Optional.")
    p.add_argument("--out", required=True, help="Output parquet path.")
    p.add_argument("--bin-size-um", type=float, default=2.0)
    p.add_argument("--roi-size-um", type=float, default=None,
                   help="If set, crop to a square ROI of this side (microns). "
                        "Start small (e.g. 200).")
    p.add_argument("--roi-center", type=float, nargs=2, default=None,
                   metavar=("X_UM", "Y_UM"),
                   help="ROI center in micron-grid coords; default = densest region.")
    p.add_argument("--multi-rule", default="nearest_centroid",
                   choices=["nearest_centroid", "smallest_id"],
                   help="Tie-break when a bin center is in >1 nucleus.")
    p.add_argument("--max-transcripts", type=int, default=20_000_000,
                   help="Guardrail: refuse explodes larger than this.")
    p.add_argument("--id-field", default="cell_id",
                   help="GeoJSON property holding the nucleus id.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t_start = time.time()

    from tracer.noseg_pipeline import (
        load_visiumhd_bins, subset_roi, explode_to_transcripts,
        load_npmi_panel, npmi_gene_set, _read_tissue_positions,
    )
    from tracer.visiumhd_seg import load_nucleus_polygons, assign_bins_to_nuclei

    # 1. Load bins + (optional) ROI crop.
    bins = load_visiumhd_bins(args.matrix_dir, args.spatial_dir,
                              expected_bin_size_um=args.bin_size_um)
    if args.roi_size_um is not None:
        center = tuple(args.roi_center) if args.roi_center else None
        bins = subset_roi(bins, size_um=args.roi_size_um, center=center)

    barcodes = np.asarray(bins.coords.index)

    # 2. Full-res pixel coords for the (ROI) bins — the polygon frame.
    tp = _read_tissue_positions(Path(args.spatial_dir)).set_index("barcode")
    need = {"pxl_col_in_fullres", "pxl_row_in_fullres"}
    if not need.issubset(tp.columns):
        raise SystemExit(
            f"tissue_positions lacks {need - set(tp.columns)} — cannot overlay "
            "nucleus polygons (need full-res pixel coords).")
    tp = tp.loc[barcodes]
    px = tp["pxl_col_in_fullres"].to_numpy(dtype=np.float64)
    py = tp["pxl_row_in_fullres"].to_numpy(dtype=np.float64)
    margin = 50.0  # px — pad bbox so edge nuclei aren't missed
    bbox = (px.min() - margin, py.min() - margin, px.max() + margin, py.max() + margin)

    # 3. Load nucleus polygons in the ROI bbox + overlay.
    print(f"[geo] loading nuclei within bbox {tuple(round(b) for b in bbox)} px")
    nuclei = load_nucleus_polygons(args.geojson, bbox=bbox, id_field=args.id_field)
    print(f"[geo] {len(nuclei.geoms):,} nuclei in bbox")
    cell_id_arr, overlaps_arr, ov_stats = assign_bins_to_nuclei(
        px, py, nuclei, multi_rule=args.multi_rule)
    print(f"[overlay] bins={ov_stats['n_bins']:,} assigned={ov_stats['n_assigned']:,} "
          f"({ov_stats['frac_assigned']:.1%}) ambiguous={ov_stats['n_ambiguous']:,} "
          f"({ov_stats['ambiguity_rate']:.2%} of assigned)")

    bin_cell_id = pd.Series(cell_id_arr, index=barcodes)
    bin_overlaps = pd.Series(overlaps_arr, index=barcodes)

    # 4. Gene panel (restrict explode) — fall back to all matrix genes.
    if args.npmi:
        panel = load_npmi_panel(args.npmi)
        panel_genes = npmi_gene_set(panel)
    else:
        panel_genes = set(np.asarray(bins.adata.var_names))
        print("[warn] no --npmi: exploding ALL genes (may be very large).")

    # 5. Explode with per-bin seed + overlaps_nucleus.
    df = explode_to_transcripts(
        bins, panel_genes=panel_genes, max_transcripts=args.max_transcripts,
        bin_cell_id=bin_cell_id, bin_overlaps_nucleus=bin_overlaps,
    )
    df["z"] = np.float32(0.0)  # VisiumHD is 2D

    # 6. Write parquet + metadata JSON.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    n_tx = len(df)
    assigned_tx = int((df["cell_id"].to_numpy() != "-1").sum())
    meta = {
        "source_matrix": str(args.matrix_dir),
        "geojson": str(args.geojson),
        "bin_size_um": args.bin_size_um,
        "roi_size_um": args.roi_size_um,
        "n_bins": int(ov_stats["n_bins"]),
        "n_transcript_rows": n_tx,
        "n_genes": int(df["feature_name"].nunique()),
        "n_nuclei_in_bbox": int(ov_stats["n_nuclei"]),
        "n_nuclei_seeded": int(bin_cell_id[bin_cell_id != "-1"].nunique()),
        "frac_bins_assigned": ov_stats["frac_assigned"],
        "frac_bins_unassigned": 1.0 - ov_stats["frac_assigned"],
        "frac_tx_nucleus_seeded": float(assigned_tx / n_tx) if n_tx else 0.0,
        "overlap_ambiguity_rate": ov_stats["ambiguity_rate"],
        "multi_rule": args.multi_rule,
        "coord_bounds_um": {
            "x_min": float(df["x"].min()), "x_max": float(df["x"].max()),
            "y_min": float(df["y"].min()), "y_max": float(df["y"].max()),
        },
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {n_tx:,} rows -> {out}")
    print(f"[done] metadata -> {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
