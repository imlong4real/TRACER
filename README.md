<h1 align="center">TRACER</h1>
<p align="center"><b>Reconstructing biologically coherent cellular profiles from spatial transcriptomics</b></p>

<!-- badges: start -->
<p align="center">

[![Tests](https://github.com/imlong4real/TRACER/actions/workflows/test.yml/badge.svg)](https://github.com/imlong4real/TRACER/actions/workflows/test.yml)
[![Docker](https://github.com/imlong4real/TRACER/actions/workflows/build-docker.yml/badge.svg)](https://github.com/imlong4real/TRACER/actions/workflows/build-docker.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.03.08.710395-B31B1B)](https://www.biorxiv.org/content/10.64898/2026.03.08.710395v1)
[![ISMB 2026](https://img.shields.io/badge/ISMB-2026-purple)](https://www.iscb.org/ismb2026/whats-happening/track-details)
[![Statistical Bioinformatics Seminar](https://img.shields.io/badge/Statistical%20Bioinformatics-Seminar-red)](https://www.youtube.com/watch?v=CugrfP88tAk)

</p>
<!-- badges: end -->

TRACER is a **reference-optional framework for refining mixed spatial transcriptomic
profiles and reconstructing coherent anuclear or partial cells** from imaging- and
sequencing-based spatial transcriptomics. It learns a gene–gene coherence prior from
the data itself (or, optionally, a single-cell reference), prunes conflicting transcript
assignments, and rebuilds biologically consistent whole and partial cells.

---

## Why TRACER?

Modern spatial assays localize transcripts with high accuracy, yet the *cellular
profiles* derived from them are often **mixed, incomplete, or missing**. Tissue
thickness and 3D overlap blend neighboring cells; nuclear-anchored segmentation drops
anuclear fragments; rigid binning splits cells across bins; and segmentation errors
leak transcripts across boundaries. The result is a cell-by-gene matrix that
misrepresents the underlying biology.

<p align="center">
  <img src="assets/images/tracer_conceptual_failure_mode.png" alt="Spatial profiling failure modes addressed by TRACER" width="820"/>
</p>

TRACER targets these failure modes directly — recovering coherent profiles without
requiring a matched reference atlas.

## How TRACER works

TRACER scores transcript-to-cell assignments against a learned gene–gene coherence
prior, removes conflicting transcripts, and reconstructs residual/partial cells from
the leftover signal — preserving the input segmentation where it is trustworthy.

<p align="center">
  <img src="assets/images/tracer_workflow.png" alt="TRACER algorithm workflow" width="900"/>
</p>

## Features

- **Segmentation-mode refinement** — correct and clean profiles starting from existing cell masks.
- **No-segmentation reconstruction** — rebuild profiles from bins or local transcript neighborhoods (e.g. VisiumHD-style binned data).
- **Coherence-prior learning** — derive a PMI/NPMI gene–gene prior from the dataset itself, or optionally from a single-cell reference.
- **Conflict pruning** — demote transcripts whose genes are incompatible with the rest of their assigned cell.
- **Partial-cell reconstruction** — assemble coherent residual/partial cells from pruned and unassigned transcripts.
- **Broad input support** — designed for imaging-based platforms (Xenium, Xenium 5K, CosMx, MERFISH) via a standardized transcript table, and for VisiumHD-style binned data via the no-segmentation pipeline.
- **Laptop-friendly** — Cython-accelerated kernels keep typical ROIs runnable on CPU.

## Installation

Requires Python ≥ 3.9 and a C compiler (TRACER ships Cython extensions).

```bash
# development install (editable, with dev/test extras)
pip install -e ".[dev]"

# plain local install
pip install .
```

> PyPI release: _coming soon._

**Cython notes.** The extensions (`_cy_*.pyx`) compile automatically during install,
so a C toolchain must be present. On macOS, install the command-line tools first
(`xcode-select --install`). For a reproducible, faster-importing build you can build a
wheel instead: `python -m build --wheel` then `pip install dist/tracer-*.whl`.

Run the test suite:

```bash
python -m pytest
```

## Quick start

TRACER consumes a **standardized transcript table** plus a **gene–gene NPMI panel**
(both produced by the helper scripts in `scripts/`).

**Segmentation mode** — refine an existing segmentation:

```bash
python scripts/run_tracer.py \
  --transcripts transcripts.parquet \
  --npmi        npmi_panel.csv.gz \
  --platform    xenium \
  --outdir      results/tracer_seg \
  --sample-name my_sample \
  --seed 1
```

Useful extras: `--pmi-threshold`, `--g-z-um {<float>|auto}`, `--user-config
<file.toml>`, `--tau`, `--min-tx-per-cell-for-scores`, `--overwrite`.

**No-segmentation mode** — reconstruct profiles from VisiumHD-style bins, no nucleus prior:

```bash
python -m tracer.noseg_pipeline \
  --visiumhd-matrix path/to/binned_outputs/square_002um/filtered_feature_bc_matrix \
  --spatial-dir     path/to/binned_outputs/square_002um/spatial \
  --npmi            npmi_panel.csv.gz \
  --platform-config src/tracer/configs/platforms/noseg.toml \
  --outdir          results/tracer_noseg \
  --sample-name     my_visiumhd_sample \
  --bin-size-um 2 --seed 1
```

See the available options with `python scripts/run_tracer.py --help`. Add `--smoke`
to the no-segmentation run for a fast ROI-limited end-to-end check.

**VisiumHD seg mode** — cell/nucleus segmentation as the prior

VisiumHD ships both 10x cell and nucleus polygons. `prepare_visiumhd_seg_input.py`
overlays the bin centers onto those polygons to derive a `cell_id` and an
`overlaps_nucleus` seed per bin (the VisiumHD analogue of imaging
`overlaps_nucleus`), then the **same** `run_tracer.py` seg pipeline refines it.
Two steps:

```bash
# Step 1 — build a seg-mode transcript table from bins + cell/nucleus masks.
# Start with a small ROI (--roi-size-um) before scaling up.
python scripts/prepare_visiumhd_seg_input.py \
  --matrix-dir   path/to/segmented_outputs/binned_outputs/square_002um/filtered_feature_bc_matrix \
  --spatial-dir  path/to/segmented_outputs/binned_outputs/square_002um/spatial \
  --geojson      path/to/segmented_outputs/graphclust_annotated_nucleus_segmentations.geojson \
  --cell-geojson path/to/segmented_outputs/graphclust_annotated_cell_segmentations.geojson \
  --npmi         npmi.csv.gz \
  --roi-size-um  300 \
  --out          results/tracer/seg/roi.parquet

# Step 2 — run the standard seg pipeline on the prepared input.
python scripts/run_tracer.py \
  --transcripts results/tracer/seg/roi.parquet \
  --npmi        npmi.csv.gz \
  --platform    xenium \
  --outdir      results/tracer/seg/run \
  --sample-name seg \
  --seed 1
```

`--cell-geojson` is what makes the default `phase1.prune_scope = "cell"`
meaningful here. Without it `cell_id` is derived from the *nucleus* mask, so
`cell_id != "-1"` is exactly `overlaps_nucleus == 1`: the "whole cell" is the
nucleus, every cytoplasmic bin enters as `-1` and is invisible to Prune, and a
whole-cell seed is a no-op. With it, `cell_id` comes from the cell mask while
`overlaps_nucleus` stays nucleus-derived — on BTC HC01 that is 72.7% of bins
cell-assigned versus 14.8% nucleus-only. Pass `--panel-genes-only` to restrict
the explode to panel genes (legacy); the default explodes every matrix gene, so
off-panel transcripts exist for the off-panel proximity rescue to place.

The prep refuses to overlay when the binned and segmented outputs report
different `microns_per_pixel` (i.e. came from different spaceranger runs), since
the bin→polygon overlay would be silently mis-registered; override with
`--allow-frame-mismatch` if you know the frames are compatible.

Step 1 also writes a `<out>.meta.json` (bins, transcript rows, genes, nuclei,
cells, assigned/unassigned fractions, overlap ambiguity rate). The seeded input
makes `run_tracer.py` take the seeded prune path (whole cells + reconstructed
partials), unlike noseg mode. The `--multi-rule` flag
(`nearest_centroid` | `smallest_id`) controls the tie-break when a bin center
falls inside more than one polygon.

> VisiumHD is 2D (constant `z`), so the z-aware stages auto-degrade to 2D with
> a logged warning — no `--g-z-um` tuning is needed.

#### Platform presets and z-plane scaling

`--platform` selects a preset under `src/tracer/configs/platforms/`. Built-ins:
`xenium`, `atera` (Xenium 5K), `cosmx`, `merfish`, and `noseg`.

A key per-platform knob is **`stitch.g_z_um`** — the z-bin size (µm) used when
binning transcripts in z. Imaging platforms differ in how z is sampled:

| Platform | z sampling | `g_z_um` |
|---|---|---|
| Xenium / Atera | near-continuous optical depth (~20 µm span) | `1.0` |
| CosMx | discrete planes ~0.8 µm apart | `0.8` |
| MERFISH | discrete planes ~1.5 µm apart | `1.5` |
| VisiumHD / 2D | no z | auto-degrades to 2D |

Pick the right value three ways, in increasing precedence:

```bash
# 1. platform preset (recommended)
python scripts/run_tracer.py ... --platform merfish

# 2. adaptive — derive g_z_um from the observed z-plane spacing at run time
python scripts/run_tracer.py ... --g-z-um auto

# 3. explicit override (wins over user config, preset, and default)
python scripts/run_tracer.py ... --g-z-um 1.5
```

## Inputs

- **Transcript table** (`--transcripts`, Parquet): one row per transcript with
  spatial coordinates (`x`, `y`, and `z` for 3D; `z` is synthesized for 2D data),
  a gene name (`feature_name`), an initial segmentation label (`cell_id`; unassigned
  transcripts are allowed), and a `transcript_id`. An optional `overlaps_nucleus`
  flag enables nuclear-aware steps. Standardize raw vendor exports with
  `scripts/preprocess_xenium.py`.
- **NPMI panel** (`--npmi`, CSV/CSV.gz): long-format gene–gene co-occurrence statistics
  (`gene_i`, `gene_j`, `PMI`, `NPMI`), built with `scripts/build_npmi_from_scrna.py`
  from the dataset or a single-cell reference.
- **No-segmentation mode** additionally takes a VisiumHD feature–barcode matrix and its
  `spatial/` directory in place of a transcript table.

## Outputs

Written under `--outdir/outputs/`:

- **`transcripts_tracer_refined.parquet`** — per-transcript fate table: final entity
  label (`stitched`/`tracer_id`), entity type `_etype` (`cell` = refined whole cell,
  `partial` = reconstructed partial/residual cell, `unknown` = unassigned), and the
  original `cell_id` for reassignment auditing.
- **`cell_by_gene_tracer.h5ad`** — reconstructed cell-by-gene matrix over refined whole
  cells and reconstructed partial cells.
- **`cell_scores.tsv.gz`** — per-cell QC: purity and conflict (coherence) metrics.

Each run also records `run_summary.md` (stage-by-stage cell/partial/unassigned counts),
`config_receipt.json` (exact resolved config), and `runtime_memory.json`.

## Citation

If you use TRACER, please cite:

> **Reconstructing biologically coherent cellular profiles from imaging-based spatial transcriptomics**
> Long Yuan, Youyun Zheng, Shuming Zhang, Rameen Beroukhim, Atul Deshpande.
> bioRxiv (2026). DOI: [10.64898/2026.03.08.710395](https://doi.org/10.64898/2026.03.08.710395)

## Contact

For questions or collaboration:

- Long Yuan — `lyuan13[at]jhmi.edu`
- Atul Deshpande — `adeshpande[at]jhu.edu`

## License

Apache License 2.0 — see [LICENSE](LICENSE).
