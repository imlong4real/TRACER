# Figure 4 — reproducible pipeline (VisiumHD kidney, TRACER noseg)

**Message.** TRACER generalizes beyond imaging-based ST and reconstructs
biologically coherent cellular profiles from **sequencing-based** spatial
transcriptomics (VisiumHD), including from very small **2 µm / near-pixel** bins,
**without prior segmentation**.

Methods compared throughout: **10x** (segmented cells) · **bin2cell** ·
**TRACER 2 µm** · **TRACER 8 µm**.

---

## Key design decision: HVG reconstruction vs. whole-transcriptome biology

- TRACER noseg **reconstruction + NPMI purity/conflict scoring** were driven by
  the **1,656 HVG/NPMI** gene panel (correct, by design).
- All **downstream biology** (label transfer, marker validation, per-cell-type
  Pearson to scRNA pseudobulk, gene/UMI counts) uses **whole-transcriptome**
  profile-by-gene matrices, re-aggregated by summing the *original* VisiumHD
  bin counts (18,132 genes) across each profile's contributing bins.
- **RCTD-style purity** (Panel E violins) uses a fast **Python Poisson-EM
  deconvolution** (the implementation in `scripts/run_rctd_tracer_overlap.py`)
  on the matched **1,656 HVG/NPMI** panel for all four methods. It is validated
  against real **spacexr/RCTD** on 10x (entropy r≈0.96, max-weight r≈0.95,
  dominant-lineage agreement ≈92%; see `panel_E_supp_poisson_vs_spacexr`). The
  full spacexr runs (10x, bin2cell, TRACER 8µm) remain under `…/rctd/<method>/`.
- **Reference concordance** is reported as Pearson, Spearman and **Kendall τ**;
  Kendall is the main Panel E heatmap, Pearson/Spearman are supplemental.

See `results/kidney_visiumhd_noseg_bin2cell_benchmark/whole_transcriptome_reaggregation_summary.md`.

---

## Folder structure

```
scripts/reproducibility/fig4/
  README.md
  fig4_config.py          # paths, palette, method specs, constants
  utils.py                # cached loaders + plotting helpers (incl. full-res H&E tile crops)
  make_fig4.py            # orchestrate panels B–G + manifest + run summary
  panel_b_whole_tissue_maps.py
  panel_c_roi_validation.py
  panel_d_marker_validation.py
  panel_e_quantitative_benchmark.py
  panel_f_resolution_tradeoff.py
  panel_g_pixel_to_profile.py
  prep/
    build_whole_transcriptome.py     # re-aggregate WT matrices from original bins
    build_rctd_inputs.py             # 1,656-gene RCTD input h5ads
    run_rctd_all.sh                  # RCTD (spacexr) for all 4 methods
    run_poisson_em_rctd.py           # fast RCTD-style Poisson-EM + spacexr validation (Panel E)
    relabel_tracer_whole_transcriptome.py  # WT label transfer for TRACER 2/8 µm
    compute_purity_conflict_all.py   # NPMI relative purity/conflict per cell (Panel E)
    compute_unassigned_bins.py       # common-denominator unassigned 2µm bin fractions
    finalize_when_rctd_done.sh       # watcher: re-render Panel E when RCTD completes
  source_data/            # all plotting tables + caches
  outputs/                # panel_*.png / .svg + manifest + run summary
```

Panel A (graphical-abstract concept cartoon) is produced separately and is not
part of this code pipeline.

---

## Inputs used (exact paths)

| Purpose | Path |
|---|---|
| H&E full-res (BigTIFF, tiled) | `datasets/kidney_visiumhd_10x/segmented_outputs/Visium_HD_Human_Kidney_FFPE_tissue_image.btf` |
| H&E hires PNG (whole-tissue) | `datasets/.../segmented_outputs/spatial/tissue_hires_image.png` |
| 10x cell segmentations | `datasets/.../segmented_outputs/cell_segmentations.geojson` |
| 10x cell matrix | `datasets/.../segmented_outputs/filtered_feature_cell_matrix.h5` |
| 2 µm / 8 µm bin matrices | `datasets/.../binned_outputs/square_00{2,8}um/filtered_feature_bc_matrix.h5` |
| 2 µm bin→cell mapping (10x) | `datasets/.../Visium_HD_Human_Kidney_FFPE_barcode_mappings.parquet` |
| scRNA reference (Schwann-excluded, 9 lineages) | `results/tracer_noseg/_ref/kidney_ref_noschwann.h5ad` |
| TRACER 2/8 µm outputs | `results/tracer_noseg/kidney_visiumhd_{2,8}um/outputs/` |
| bin2cell outputs | `results/bin2cell/kidney_visiumhd_2um/` |

Whole-transcriptome matrices (built by prep): `tenx_segmented_cell_by_gene.h5ad`,
`bin2cell_cell_by_gene_whole_transcriptome.h5ad`,
`profile_by_gene_whole_transcriptome.h5ad` (TRACER 2/8 µm).

---

## How to regenerate (from repo root)

Activate the python env with scanpy/anndata (here: `.venv` → `spatial` conda env).

### 0. Prep (run once; outputs are cached/idempotent)

```bash
# whole-transcriptome re-aggregation (all 4 methods)
python scripts/reproducibility/fig4/prep/build_whole_transcriptome.py

# whole-transcriptome label transfer for TRACER 2 µm and 8 µm
python scripts/reproducibility/fig4/prep/relabel_tracer_whole_transcriptome.py

# 1,656-gene RCTD inputs, then RCTD for all four methods (R / spacexr)
python scripts/reproducibility/fig4/prep/build_rctd_inputs.py
bash   scripts/reproducibility/fig4/prep/run_rctd_all.sh

# fast RCTD-style Poisson-EM metrics for all 4 methods + spacexr validation
#   (used by Panel E violins; ~5 min total, no R required)
python scripts/reproducibility/fig4/prep/run_poisson_em_rctd.py

# NPMI relative purity/conflict per cell, all methods (Panel E stacked bar)
PYTHONPATH=src python scripts/reproducibility/fig4/prep/compute_purity_conflict_all.py

# common-denominator unassigned 2µm-bin fractions (Panel F)
python scripts/reproducibility/fig4/prep/compute_unassigned_bins.py
```

RCTD uses `/Users/lyuan13/anaconda3/envs/tracer_benchmark_r/bin/Rscript` and
forces `RETICULATE_PYTHON` to a python with `anndata`
(`/Users/lyuan13/anaconda3/envs/spatial/bin/python`).

### 1. All panels at once

```bash
cd scripts/reproducibility/fig4
python make_fig4.py            # panels B–G + manifest + run summary
```

### 2. Individual panels

```bash
cd scripts/reproducibility/fig4
python panel_b_whole_tissue_maps.py      # -> outputs/panel_B_whole_tissue_maps.{png,svg}
python panel_c_roi_validation.py         # -> outputs/panel_C_roi_validation.{png,svg} + per-ROI strips
python panel_d_marker_validation.py      # -> outputs/panel_D_marker_validation.{png,svg}
python panel_e_quantitative_benchmark.py # -> outputs/panel_E_quantitative_benchmark.{png,svg}
python panel_f_resolution_tradeoff.py    # -> outputs/panel_F_resolution_tradeoff.{png,svg}
python panel_g_pixel_to_profile.py       # -> outputs/panel_G_pixel_to_profile.{png,svg}
```

---

## Panels & outputs

| Panel | Content | Output | Source tables |
|---|---|---|---|
| B | Whole-tissue lineage maps over H&E, 4 methods | `panel_B_whole_tissue_maps.{png,svg}` | `panel_B_lineage_counts.csv` |
| C | 4 ROI zoom-ins (glomerulus/PT/TAL/IC) on full-res H&E × 5 columns | `panel_C_roi_validation.{png,svg}`, `panel_C_roi_<key>.{png,svg}` | `panel_C_roi_metadata.csv` |
| D | Canonical marker heatmaps (method × lineage), whole-transcriptome | `panel_D_marker_validation.{png,svg}` | `panel_D_marker_expression.csv` |
| E | **QC-filtered** (≥100 genes, ≥200 UMIs, ≥5 bins): retention bars · RCTD-style **Poisson-EM** entropy + max-weight half-violins (all 4 methods) · NPMI relative purity/conflict stacked bar · per-lineage **Kendall τ** to scRNA pseudobulk (magma, main) | `panel_E_quantitative_benchmark.{png,svg}` + `panel_E_sensitivity.{png,svg}` + supplements `panel_E_supp_concordance.{png,svg}` (Pearson+Spearman) and `panel_E_supp_poisson_vs_spacexr.{png,svg}` (validation) | `panel_E_qc_retention.csv`, `panel_E_sensitivity.csv`, `panel_E_concordance_to_reference.csv` (Pearson/Spearman/Kendall), `panel_E_rctd_summary.csv`, `panel_E_purity_conflict.csv`, `panel_E_poisson_vs_spacexr_validation.csv` |
| F | Runtime+memory, profile counts, genes/UMIs per profile (raincloud), unassigned bins (incl. 10x) | `panel_F_resolution_tradeoff.{png,svg}` | `panel_F_resolution_tradeoff.csv`, `unassigned_bins_2um.csv` |
| G | Bins-per-profile distribution + a dense, cell-type-diverse 2µm region on full-res H&E with **bins shared by >1 cell** (dominant_fraction<1) outlined, + per-cell reconstructed marker transcriptome | `panel_G_pixel_to_profile.{png,svg}` | `panel_G_bins_per_profile_distribution.csv`, `panel_G_region_metadata.csv`, `panel_G_region_entities.csv`, `panel_G_region_bins.csv.gz`, `panel_G_region_marker_matrix.csv` |

Manifest: `outputs/fig4_manifest.json` · Run summary: `outputs/fig4_run_summary.md`.

---

## Shared style

- Nature-style, white background, Arial, vector SVG (`svg.fonttype=none`).
- One lineage palette reused across all panels (`fig4_config.PALETTE`):
  PT `#2E6FB7` · TAL `#E8743B` · PC `#8E5BA6` · IC `#D14D9A` · EC `#C0392B` ·
  Fib/VSMC/P `#8C6D4F` · Myeloid `#E0B33A` · Lymphoid `#2CA089` · POD `#5BB36A`.
- Common spatial frame = fullres-pixel × MPP (0.27390 µm/px); TRACER bins mapped
  via `tissue_positions` `pxl_*_in_fullres`, bin2cell via its centroids, 10x via
  geojson centroids. Full-res H&E crops read only the tiles intersecting an ROI.

## Panel E QC

All Panel E metrics (RCTD entropy/max-weight, NPMI relative purity/conflict,
Pearson to scRNA pseudobulk) are computed on **QC-passing** profiles:
**primary** ≥100 genes, ≥200 UMIs, ≥5 bins (bin criterion skipped for 10x).
Retention rates and a **sensitivity analysis** under **strict** thresholds
(≥200 genes, ≥500 UMIs, ≥10 bins) are reported in `panel_E_qc_retention.csv`,
`panel_E_sensitivity.csv`, and `panel_E_sensitivity.{png,svg}`. NPMI relative
purity/conflict (`relative_purity + relative_conflict = 1`) is computed
identically for all methods from gene co-presence over the kidney NPMI graph
(`compute_purity_conflict_all.py`), so segmented and reconstructed methods are
directly comparable. Primary-QC retention: 10x 77%, bin2cell 49%,
TRACER 2 µm 31%, TRACER 8 µm 48%.

## Caveats

- RCTD (1,656-gene panel) is the only step requiring R/spacexr; it runs the four
  methods sequentially and is the slowest stage.
- TRACER whole-transcriptome coverage = profiles owning ≥1 bin (2 µm 98.5%,
  8 µm 61%); minority-fragment profiles are excluded from WT metrics.
- Panel F profile counts & unassigned-bin fractions reflect **input bin
  granularity, not quality**.
