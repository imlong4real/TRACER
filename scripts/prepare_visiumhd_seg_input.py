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


def _read_microns_per_pixel(spatial_dir) -> float | None:
    """microns_per_pixel from a spatial/scalefactors_json.json, or None."""
    f = Path(spatial_dir) / "scalefactors_json.json"
    if not f.exists():
        return None
    try:
        return float(json.load(open(f)).get("microns_per_pixel"))
    except (ValueError, TypeError, KeyError):
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix-dir", required=True,
                   help="filtered_feature_bc_matrix dir (square_0NNum).")
    p.add_argument("--spatial-dir", required=True,
                   help="spatial/ dir with tissue_positions + scalefactors.")
    p.add_argument("--geojson", required=True,
                   help="10x NUCLEUS segmentation GeoJSON (full-res pixels). "
                        "Always supplies overlaps_nucleus.")
    p.add_argument("--cell-geojson", default=None,
                   help="10x CELL segmentation GeoJSON. When given, cell_id "
                        "comes from the CELL polygons (so cytoplasmic bins "
                        "join their cell) while overlaps_nucleus still comes "
                        "from --geojson. Without it cell_id == nucleus id, "
                        "which makes the whole-cell Prune scope a no-op: the "
                        "'cell' is then exactly the nucleus.")
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
                   help="GeoJSON property holding the nucleus/cell id. 10x "
                        "uses the SAME id namespace in both masks.")
    p.add_argument("--panel-genes-only", action="store_true",
                   help="Explode ONLY panel genes (legacy). Default: explode "
                        "ALL matrix genes, so off-panel genes exist to be "
                        "placed by the off-panel proximity rescue instead of "
                        "being dropped before the pipeline sees them.")
    p.add_argument("--allow-frame-mismatch", action="store_true",
                   help="Proceed even when the binned and segmented outputs "
                        "report different microns_per_pixel (i.e. came from "
                        "different spaceranger versions). Off by default: a "
                        "silent frame mismatch mis-registers every bin.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t_start = time.time()

    from tracer.noseg_pipeline import (
        load_visiumhd_bins, subset_roi, explode_to_transcripts,
        load_pmi_panel, pmi_gene_set, _read_tissue_positions,
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

    # 2b. Registration guard. The polygons live in the full-res microscopy
    # frame; the bins' pxl_*_in_fullres are only in that same frame when the
    # binned and segmented outputs came from the same spaceranger run. A
    # mismatch (e.g. binned 3.0.1 @ 5.7499 um/px vs segmented 4.0.1 @ 0.46428)
    # silently mis-registers every bin, so fail loudly rather than overlay.
    _bmpp = _read_microns_per_pixel(Path(args.spatial_dir))
    _smpp = _read_microns_per_pixel(Path(args.geojson).parent / "spatial")
    if _bmpp and _smpp:
        _scale = float(_bmpp) / float(_smpp)
        if abs(_scale - 1.0) < 0.02:
            print(f"[registration] same frame (scale={_scale:.4f}) — direct overlay")
        elif args.allow_frame_mismatch:
            print(f"[registration] WARNING frame mismatch scale={_scale:.4f} "
                  f"(binned {_bmpp} vs segmented {_smpp} um/px) — proceeding "
                  f"on --allow-frame-mismatch; overlay is UNRELIABLE")
        else:
            raise SystemExit(
                f"[registration] frame mismatch: binned microns_per_pixel="
                f"{_bmpp} vs segmented={_smpp} (scale={_scale:.4f}). These "
                f"outputs are from different spaceranger runs; the bin->polygon "
                f"overlay would be wrong. Use matching outputs, or pass "
                f"--allow-frame-mismatch to override.")

    # 3. Load nucleus polygons in the ROI bbox + overlay.
    print(f"[geo] loading nuclei within bbox {tuple(round(b) for b in bbox)} px")
    nuclei = load_nucleus_polygons(args.geojson, bbox=bbox, id_field=args.id_field)
    print(f"[geo] {len(nuclei.geoms):,} nuclei in bbox")
    cell_id_arr, overlaps_arr, ov_stats = assign_bins_to_nuclei(
        px, py, nuclei, multi_rule=args.multi_rule)
    print(f"[overlay] bins={ov_stats['n_bins']:,} assigned={ov_stats['n_assigned']:,} "
          f"({ov_stats['frac_assigned']:.1%}) ambiguous={ov_stats['n_ambiguous']:,} "
          f"({ov_stats['ambiguity_rate']:.2%} of assigned)")

    bin_overlaps = pd.Series(overlaps_arr, index=barcodes)

    if args.cell_geojson:
        # cell_id from the CELL mask; overlaps_nucleus stays nucleus-derived.
        # 10x ships both masks 1:1 on a shared id namespace (cell polygons are
        # ~5.8x the nucleus area), so a bin inside nucleus N should also fall
        # inside cell N -- checked below rather than assumed.
        cells = load_nucleus_polygons(args.cell_geojson, bbox=bbox,
                                      id_field=args.id_field)
        print(f"[geo] {len(cells.geoms):,} cell polygons in bbox")
        cell_arr, cell_ov, cstats = assign_bins_to_nuclei(
            px, py, cells, multi_rule=args.multi_rule)
        bin_cell_id = pd.Series(cell_arr, index=barcodes)
        nuc_only = pd.Series(cell_id_arr, index=barcodes)
        both = (nuc_only != "-1") & (bin_cell_id != "-1")
        disagree = int((nuc_only[both] != bin_cell_id[both]).sum())
        print(f"[overlay] cell-assigned bins={int((bin_cell_id != '-1').sum()):,} "
              f"({cstats['frac_assigned']:.1%})  nucleus bins="
              f"{int(bin_overlaps.astype(bool).sum()):,} "
              f"({ov_stats['frac_assigned']:.1%})")
        if disagree:
            print(f"[overlay] WARNING {disagree:,} bins sit in nucleus N but "
                  f"cell M != N ({disagree / max(int(both.sum()), 1):.2%} of "
                  f"nuclear bins) — check registration")
    else:
        bin_cell_id = pd.Series(cell_id_arr, index=barcodes)
        print("[overlay] cell_id == nucleus id (no --cell-geojson): every "
              "cytoplasmic bin enters as -1 and the whole-cell Prune scope "
              "is a no-op on this input.")

    # 4. Gene panel (restrict explode) — fall back to all matrix genes.
    all_genes = set(map(str, bins.adata.var_names))
    if args.npmi:
        panel = load_pmi_panel(args.npmi)
        panel_genes = pmi_gene_set(panel)
    else:
        panel_genes = set(all_genes)
    # Panel genes define the PMI edge list; explode genes decide which tx are
    # materialised. Restricting the explode to the panel DROPS off-panel genes
    # outright and idles the off-panel proximity rescue, so explode everything
    # unless asked otherwise.
    explode_genes = panel_genes if args.panel_genes_only else all_genes
    print(f"[explode-genes] data={len(all_genes):,} "
          f"in-panel={len(all_genes & panel_genes):,} "
          f"off-panel={len(all_genes - panel_genes):,} "
          f"exploding={'panel-only' if args.panel_genes_only else 'ALL'}")

    # 5. Explode with per-bin seed + overlaps_nucleus.
    df = explode_to_transcripts(
        bins, panel_genes=explode_genes, max_transcripts=args.max_transcripts,
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
