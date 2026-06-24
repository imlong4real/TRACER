#!/usr/bin/env python3
"""Central configuration for the Figure 4 reproducibility pipeline.

Figure 4 message
----------------
TRACER generalizes beyond imaging-based ST and reconstructs biologically
coherent cellular profiles from sequencing-based spatial transcriptomics
(VisiumHD), including from very small bin / near-pixel inputs without prior
segmentation.

Everything here is paths + constants + the shared lineage palette. No data
is loaded at import time. All paths are resolved relative to the repo root so
the pipeline is runnable from anywhere.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]            # repo root
DATA = ROOT / "datasets/kidney_visiumhd_10x/segmented_outputs"
RES = ROOT / "results"
FIG4 = ROOT / "scripts/reproducibility/fig4"
OUTDIR = FIG4 / "outputs"
SRCDIR = FIG4 / "source_data"
BENCH = RES / "kidney_visiumhd_noseg_bin2cell_benchmark"
WT = BENCH / "whole_transcriptome"
RCTD = BENCH / "rctd"

for _d in (OUTDIR, SRCDIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Imaging / spatial constants (from spaceranger scalefactors_json.json)
# ---------------------------------------------------------------------------
MPP = 0.2739013209140399          # microns per fullres pixel
HIRES_SCALEF = 0.13636674         # hires_pixel = fullres_pixel * scalef
HE_HIRES_PNG = DATA / "spatial/tissue_hires_image.png"
HE_BTF = DATA / "Visium_HD_Human_Kidney_FFPE_tissue_image.btf"   # full-res H&E (tiled BigTIFF)
CELL_SEG_GEOJSON = DATA / "cell_segmentations.geojson"

SPATIAL_DIR = {
    "tracer_2um": DATA / "binned_outputs/square_002um/spatial",
    "tracer_8um": DATA / "binned_outputs/square_008um/spatial",
}

# ---------------------------------------------------------------------------
# Lineage palette + ordering (shared across ALL panels)
# ---------------------------------------------------------------------------
# Internal label form matches the data ("FIB/VSMC/P"); DISPLAY map is for axes.
LINEAGES = ["PT", "TAL", "PC", "IC", "EC", "FIB/VSMC/P", "Myeloid",
            "Lymphoid", "POD"]

LINEAGE_DISPLAY = {
    "PT": "PT", "TAL": "TAL", "PC": "PC", "IC": "IC", "EC": "EC",
    "FIB/VSMC/P": "Fib/VSMC/P", "Myeloid": "Myeloid",
    "Lymphoid": "Lymphoid", "POD": "POD",
}

# Nature-style, print-friendly, mutually distinct.
PALETTE = {
    "PT":         "#2E6FB7",   # blue
    "TAL":        "#E8743B",   # orange
    "PC":         "#8E5BA6",   # purple
    "IC":         "#D14D9A",   # magenta
    "EC":         "#C0392B",   # red (endothelial)
    "FIB/VSMC/P": "#8C6D4F",   # brown
    "Myeloid":    "#E0B33A",   # gold
    "Lymphoid":   "#2CA089",   # teal
    "POD":        "#5BB36A",   # green
}
NA_COLOR = "#D9D9D9"           # unassigned / background units

# RCTD sanitizes "FIB/VSMC/P" -> "FIB_VSMC_P"; map back on read.
RCTD_LABEL_FIX = {"FIB_VSMC_P": "FIB/VSMC/P"}

# ---------------------------------------------------------------------------
# Methods: canonical display names + the cached artifacts each one uses.
# ---------------------------------------------------------------------------
METHOD_ORDER = ["10x_segmented", "bin2cell", "tracer_2um", "tracer_8um"]
METHOD_DISPLAY = {
    "10x_segmented": "10x",
    "bin2cell": "bin2cell",
    "tracer_2um": "TRACER 2 µm",
    "tracer_8um": "TRACER 8 µm",
}
METHOD_COLOR = {                # for benchmark bar/violin panels (E, F)
    "10x_segmented": "#7F8C8D",
    "bin2cell": "#16A085",
    "tracer_2um": "#2E6FB7",
    "tracer_8um": "#7D4FB7",
}

# Per-method label tables (lineage annotation; all whole-transcriptome).
LABELS = {
    "10x_segmented": RES / "tracer_noseg/kidney_visiumhd_8um/validation_plots/_10x_labels/kidney_10x_seg_transferred_cell_annotations.csv",
    "bin2cell": RES / "bin2cell/kidney_visiumhd_2um/label_transfer/bin2cell_profiles_with_labels.tsv.gz",
    "tracer_2um": RES / "tracer_noseg/kidney_visiumhd_2um/label_transfer_wt/reconstructed_profiles_with_labels.tsv.gz",
    "tracer_8um": RES / "tracer_noseg/kidney_visiumhd_8um/label_transfer_wt/reconstructed_profiles_with_labels.tsv.gz",
}
# HVG-based label tables (carry n_bins / TRACER NPMI purity-conflict, for G/F).
LABELS_HVG = {
    "tracer_2um": RES / "tracer_noseg/kidney_visiumhd_2um/label_transfer/reconstructed_profiles_with_labels.tsv.gz",
    "tracer_8um": RES / "tracer_noseg/kidney_visiumhd_8um/label_transfer/reconstructed_profiles_with_labels.tsv.gz",
}

# Whole-transcriptome cell/profile-by-gene matrices.
WT_H5AD = {
    "10x_segmented": WT / "tenx_segmented_cell_by_gene.h5ad",
    "bin2cell": WT / "bin2cell_cell_by_gene_whole_transcriptome.h5ad",
    "tracer_2um": RES / "tracer_noseg/kidney_visiumhd_2um/outputs/profile_by_gene_whole_transcriptome.h5ad",
    "tracer_8um": RES / "tracer_noseg/kidney_visiumhd_8um/outputs/profile_by_gene_whole_transcriptome.h5ad",
}

# Bin -> profile maps (for spatial bin maps + Panel G stitching example).
BIN_TO_PROFILE = {
    "tracer_2um": RES / "tracer_noseg/kidney_visiumhd_2um/outputs/bin_to_profile_assignment.parquet",
    "tracer_8um": RES / "tracer_noseg/kidney_visiumhd_8um/outputs/bin_to_profile_assignment.parquet",
}

# Runtime / memory benchmark metrics.
BENCH_METRICS = {
    "tracer_2um": RES / "tracer_noseg/kidney_visiumhd_2um/benchmark_metrics/runtime_memory.json",
    "tracer_8um": RES / "tracer_noseg/kidney_visiumhd_8um/benchmark_metrics/runtime_memory.json",
    "bin2cell": RES / "bin2cell/kidney_visiumhd_2um/benchmark_metrics/runtime_memory.json",
}

# scRNA reference (Schwann-excluded, 9 lineages, obs['lineage']).
REFERENCE_H5AD = RES / "tracer_noseg/_ref/kidney_ref_noschwann.h5ad"

# RCTD per-cell outputs (built by prep/run_rctd_all.sh). The runner names the
# bin2cell output dir `bin2cell_2um`, so map method key -> output dir.
_RCTD_DIR = {"10x_segmented": "10x_segmented", "bin2cell": "bin2cell_2um",
             "tracer_2um": "tracer_2um", "tracer_8um": "tracer_8um"}
RCTD_ASSIGN = {m: RCTD / _RCTD_DIR[m] / "rctd_cell_assignments_post.tsv"
               for m in METHOD_ORDER}

# Canonical marker genes per lineage for Panel D (validated against TRACER 2um).
MARKERS = {
    "PT":         ["LRP2", "CUBN", "SLC5A2", "SLC34A1", "ALDOB"],
    "TAL":        ["UMOD", "SLC12A1", "CLDN16", "KCNJ1"],
    "PC":         ["AQP2", "AQP3", "SCNN1G"],
    "IC":         ["ATP6V1B1", "ATP6V0D2", "SLC4A1", "FOXI1", "SLC26A4"],
    "EC":         ["PECAM1", "VWF", "KDR", "EMCN", "FLT1"],
    "FIB/VSMC/P": ["COL1A1", "COL3A1", "DCN", "ACTA2", "RGS5", "PDGFRB"],
    "Myeloid":    ["LYZ", "CD68", "C1QA", "C1QB", "CSF1R"],
    "Lymphoid":   ["PTPRC", "CD3D", "CD3E", "TRAC", "MS4A1", "CD79A"],
    "POD":        ["NPHS1", "NPHS2", "PODXL", "WT1", "MAFB"],
}


def he_micron_extent():
    """Return (xmax_um, ymax_um) of the H&E hires image in micron."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    w, h = Image.open(HE_HIRES_PNG).size
    return (w / HIRES_SCALEF * MPP, h / HIRES_SCALEF * MPP)
