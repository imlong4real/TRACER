# InSituCNV TRACER Degradation Benchmark

WIP benchmark for asking whether TRACER-refined Xenium profiles retain CNV signal
better than raw Xenium segmentation under reduced panel size and reduced detection
efficiency.

This folder is intentionally tutorial-local while the benchmark design is still
settling. Large data and outputs belong under `data/` and `output/`, both ignored.

## Design

- Dataset: public Xenium Prime ovarian adenocarcinoma.
- Primary comparison: raw Xenium cells vs TRACER whole cells.
- Secondary comparison: raw Xenium cells vs TRACER whole cells plus QC-passing
  partials.
- Downsampling is transcript-level. The same retained transcript IDs are used for
  the raw and TRACER arms.
- Xenium segmentation is fixed under downsampling. TRACER is rerun after each
  downsampling condition.
- Reduced panels are marker-preserving, not random, so reannotation remains
  biologically usable.
- Primary result: degradation curves showing concordance to full-depth/full-panel
  high-confidence CNV/subclone calls.

## Folder Layout

```text
config/
  downsampling.yaml
  ovarian_leiden_celltype_map.csv
  panels.yaml
scripts/
  common.py
  build_gene_positions.py
  make_full_ovary_celltyping.py
  make_full_ovary_cnv_preview.py
  select_roi.py
  make_marker_preserving_panels.py
  downsample_transcripts.py
  annotate_entities.py
  run_tracer_arm.py
  run_insitucnv_arm.py
  compare_degradation.py
sge/
  stage_xenium_ovary.sge
  make_full_ovary_celltyping.sge
  make_full_ovary_cnv_preview.sge
  run_roi_grid.sge
manifests/
  roi_manifest.csv
  run_manifest.csv
data/
output/
```

## Pilot Run Order

1. Stage the selected full Xenium Prime ovary files on argos with
   `sge/stage_xenium_ovary.sge`. This downloads only the files needed for the
   benchmark, not the full 144 GB output bundle.
2. Create or activate the `insitucnv_tracer_benchmark` conda environment.
3. Build `data/gene_positions_grch38.tsv` with `scripts/build_gene_positions.py`
   from the staged `gene_panel.json`. This tiny input is preferred over scanning
   the 31 GB transcript parquet just to collect unique genes.
4. Run a small full-ovary cell-typing smoke test. This validates the job,
   environment, and data wiring only; do not annotate from this subset.
5. Run full-ovary manuscript-style Leiden/UMAP/DEG cell typing on all cells.
6. Manually review the full-run Leiden DEGs, UMAP, and spatial plots, then edit
   or confirm `config/ovarian_leiden_celltype_map.csv`.
7. Apply the reviewed labels with `APPLY_ANNOTATIONS_ONLY=1`. This reuses the
   fixed `adata_umap.h5ad` and regenerates `adata_annotated.h5ad` without
   rerunning Leiden/UMAP.
8. Run the CNV preview with the generated gene-position table and reviewed
   annotated cell-typing object.
9. Use the clone-colored spatial plots to choose a ROI with multiple epithelial
   CNV clones plus reference cells, then extract the transcript-level ROI with
   `scripts/select_roi.py`. This is the first subsetting step and writes an ROI
   transcript parquet with TRACER-compatible columns.
10. Build marker-preserving panels from the ROI genes and gene-position table.
11. For every panel/detection condition, write a downsampled transcript parquet.
12. Run TRACER on each downsampled parquet.
13. Build raw/TRACER entity matrices, reannotate each arm, run InSituCNV, and save
   per-condition CNV summaries.
14. Compare each condition against the full-depth/full-panel reference calls.

## Stage Full Xenium Files

Run this first on argos:

```bash
cd $REPO
qsub tutorials/insitucnv_tracer_benchmark/sge/stage_xenium_ovary.sge
```

The job writes to `tutorials/insitucnv_tracer_benchmark/data/xenium_ovary/raw/`
and downloads these direct 10x-hosted files:

- `Xenium_Prime_Human_Ovary_FF_transcripts.parquet`
- `Xenium_Prime_Human_Ovary_FF_cells.parquet`
- `Xenium_Prime_Human_Ovary_FF_cell_feature_matrix.h5`
- `Xenium_Prime_Human_Ovary_FF_gene_panel.json`
- `Xenium_Prime_Human_Ovary_FF_analysis_summary.html`

It intentionally does not download `outs.zip`, `xe_outs.zip`, or H&E image files,
and it does not extract an ROI.

## Environment Use

Staging uses `curl` only, so `sge/stage_xenium_ovary.sge` does not need a conda
environment.

Use the shared benchmark environment for scripts that touch Xenium matrices,
parquet files, Scanpy, InSituCNV, infercnvpy, downsampling, or TRACER outputs:

```bash
cd $REPO
conda env create -f tutorials/insitucnv_tracer_benchmark/insitucnv_env.yml
conda activate insitucnv_tracer_benchmark
```

If the environment already exists, only run `conda activate
insitucnv_tracer_benchmark`. Full CNV/TRACER steps should run on argos, or inside
this same environment locally for ROI-scale tests.

## Build Gene Positions

After staging the 10x files, download the GENCODE human Release 50
GRCh38.p14 comprehensive gene annotation for reference chromosomes only. This is
the `CHR` `GTF` link on the GENCODE human release page:
https://www.gencodegenes.org/human/.

```bash
cd $REPO
mkdir -p tutorials/insitucnv_tracer_benchmark/data

curl -L \
  -o tutorials/insitucnv_tracer_benchmark/data/gencode.v50.annotation.gtf.gz \
  https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.annotation.gtf.gz
```

This `curl` download does not need the conda environment. Then activate
`insitucnv_tracer_benchmark` and build a GRCh38 gene-position table from the
staged gene panel:

```bash
cd $REPO
conda activate insitucnv_tracer_benchmark
python tutorials/insitucnv_tracer_benchmark/scripts/build_gene_positions.py \
  --genes-from tutorials/insitucnv_tracer_benchmark/data/xenium_ovary/raw/Xenium_Prime_Human_Ovary_FF_gene_panel.json \
  --gtf tutorials/insitucnv_tracer_benchmark/data/gencode.v50.annotation.gtf.gz \
  --out tutorials/insitucnv_tracer_benchmark/data/gene_positions_grch38.tsv
```

The script writes:

- `gene_positions_grch38.tsv` with `gene`, `gene_id`, `chromosome`, `start`,
  `end`, `strand`, `gene_type`, and `source`.
- `gene_positions_grch38.tsv.summary.json`.
- `gene_positions_grch38.tsv.missing_genes.txt` when input genes are absent from
  the GTF.
- `gene_positions_grch38.tsv.duplicate_gene_symbols.tsv` when the GTF contains
  duplicate symbols among the requested genes.

## Full-Ovary Cell Typing

Cell typing follows the ovarian InSituCNV manuscript notebook as a separate
stage: filter low-quality cells, normalize/log-transform counts, run PCA,
neighbors, Leiden, UMAP, Leiden DEGs, manually map Leiden clusters to cell
classes, then run cell-class DEGs. The manuscript cluster numbers are not reused
as truth because Leiden IDs can change across runs. Use the full-run
`leiden_marker_genes.csv`, `UMAP_leiden.pdf`, and `spatial_leiden.pdf` to review
or revise `config/ovarian_leiden_celltype_map.csv`.

Run a small smoke test first. This only checks code wiring; do not use subset
Leiden IDs to finalize the cell-type map:

```bash
cd $REPO
qsub -v MAX_CELLS=50000 \
  tutorials/insitucnv_tracer_benchmark/sge/make_full_ovary_celltyping.sge
```

Then run the full all-cell manuscript-style Leiden/DEG job:

```bash
cd $REPO
qsub tutorials/insitucnv_tracer_benchmark/sge/make_full_ovary_celltyping.sge
```

Manual annotation is the biological review step. The script creates
`leiden_marker_genes.csv`, `UMAP_leiden.pdf`, and `spatial_leiden.pdf`; a human
then interprets the cluster markers and records the final calls in
`config/ovarian_leiden_celltype_map.csv`. The current default map is derived
from the completed full-run cluster DEGs:

- Epithelial: `0`, `1`, `3`, `5`, `7`, `11`, `16`, `17`
- Low expression: `2`
- Fibroblasts: `4`, `8`, `9`
- T cells: `6`
- Monocytes: `10`, `12`, `14`
- Endothelial: `13`
- B cells: `15`
- Mesothelial: `18`

The CSV includes an `assignment_basis` column with marker evidence. The script
only consumes `leiden` and `cell_class`, so `assignment_basis` is documentation
for review and can be edited freely.

After the manual map is accepted, or any time you revise
`config/ovarian_leiden_celltype_map.csv`, do not rerun PCA/Leiden/UMAP. Reuse
the fixed full-run `adata_umap.h5ad` and regenerate only the reviewed cell-class
outputs:

```bash
cd $REPO
qsub -v APPLY_ANNOTATIONS_ONLY=1 \
  tutorials/insitucnv_tracer_benchmark/sge/make_full_ovary_celltyping.sge
```

This rewrites `adata_annotated.h5ad`, `celltyping_obs.csv.gz`,
`cell_class_counts.csv`, `cell_class_marker_genes.csv`, and the cell-class plots
without changing the Leiden clustering. Use `UMAP_H5AD=/path/to/adata_umap.h5ad`
if you need to apply a map to a non-default clustered object.

The cell-typing SGE script writes its own log at
`logs/insitucnv_tracer_full_ovary_celltyping.log`, requests `h_vmem=192G`, and
uses the argos conda base at
`/mnt/storage/dept/medonc/beroukhim/youyun/util/miniforge3` by default. Override
with `CONDA_BASE=/path/to/conda/base` or
`BENCH_PYTHON=/path/to/envs/insitucnv_tracer_benchmark/bin/python` if needed.

Cell-typing outputs are written to
`tutorials/insitucnv_tracer_benchmark/output/full_ovary_celltyping/`:

- `plots/celltyping/UMAP_leiden.pdf`
- `plots/celltyping/spatial_leiden.pdf`
- `plots/celltyping/dotplot_leiden_markers.pdf`
- `plots/celltyping/UMAP_cell_class.pdf`
- `plots/celltyping/spatial_cell_class.pdf`
- `plots/celltyping/dotplot_cell_class.pdf`
- `leiden_marker_genes.csv`
- `cell_class_marker_genes.csv`
- `leiden_counts.csv`
- `cell_class_counts.csv`
- `celltyping_obs.csv.gz`
- `adata_umap.h5ad`
- `adata_annotated.h5ad`
- `run_summary.json`

If a future full run produces new Leiden IDs, the script writes
`unmapped_leiden_clusters.txt` and exits after writing the review figures and
tables. Update the map from the new full-run DEGs, then rerun with
`APPLY_ANNOTATIONS_ONLY=1` until `adata_annotated.h5ad` is produced.

## Build Full-Data CNV Preview

The CNV preview consumes the reviewed full-ovary annotated object from the cell
typing stage. It does not redo cell typing.

The default CNV path is manuscript-faithful to the visible ovarian
InSituCNV-manuscript notebook:

- cell typing creates the expression/PCA neighbor graph with
  `sc.pp.neighbors(..., n_pcs=50)`;
- the CNV step reuses that existing graph for `icv.tl.smooth_data_for_cnv`
  instead of rebuilding a spatial neighbor graph;
- `infercnvpy` uses the manuscript reference classes `Monocytes`, `B cells`,
  `T cells`, `Endothelial`, `Low expression`, and `Fibroblasts`;
- epithelial CNV clusters are filtered to clones representing more than `0.1%`
  of epithelial cells for the default `cnv_epi` label, matching the manuscript
  notebook's selected-clone cutoff.

This means `infercnvpy` is run on expression-neighbor-smoothed values from
`adata.layers["M"]`. Physical `x/y` coordinates are used for plotting the CNV
labels back on the tissue, not for the default smoothing graph. A spatial
smoothing experiment is available only by explicitly submitting
`SMOOTHING_GRAPH=spatial`; use the default `SMOOTHING_GRAPH=existing` when
comparing to the paper workflow.

Run a CNV input smoke test after `adata_annotated.h5ad` exists:

```bash
cd $REPO
qsub -v DRY_RUN=1,MAX_CELLS=50000 \
  tutorials/insitucnv_tracer_benchmark/sge/make_full_ovary_cnv_preview.sge
```

The full CNV preview requires the generated GRCh38 gene-position table at
`tutorials/insitucnv_tracer_benchmark/data/gene_positions_grch38.tsv`, or a
different path passed with `GENE_POSITIONS=/path/to/gene_positions.tsv`. The
table must have `gene`, `chromosome`, `start`, and `end` columns.

Run the full CNV preview after the gene-position table and
`output/full_ovary_celltyping/adata_annotated.h5ad` both exist:

```bash
cd $REPO
qsub tutorials/insitucnv_tracer_benchmark/sge/make_full_ovary_cnv_preview.sge
```

To use a different reviewed annotation object, submit with
`CELLTYPING_H5AD=/path/to/adata_annotated.h5ad`.

CNV preview outputs are written to
`tutorials/insitucnv_tracer_benchmark/output/full_ovary_cnv_preview/`:

- `plots/spatial_cell_class.png`
- `plots/spatial_cnv_leiden.png`
- `plots/spatial_cnv_epi.png`
- `plots/spatial_cnv_epi_raw.png`
- `adata_obs.csv.gz`
- `cell_spatial_cnv_preview.csv.gz`
- `*_counts.csv`
- `run_summary.json`

`cnv_epi_raw` keeps all epithelial CNV Leiden clusters. `cnv_epi` applies the
manuscript `>0.1%` epithelial clone filter and labels smaller epithelial
clusters as `filtered_small_clone`. `run_summary.json` records the smoothing
graph mode, requested smoothing neighbor count, reference classes, and raw vs
filtered epithelial CNV cluster counts.

Use the clone-colored spatial plots to choose the ROI bbox, then run
`scripts/select_roi.py` on the full `transcripts.parquet`. The preview flags
write a 3-panel ROI check using the full-slide CNV metadata before you commit to
the downstream TRACER run:

```bash
cd $REPO
conda activate insitucnv_tracer_benchmark
python tutorials/insitucnv_tracer_benchmark/scripts/select_roi.py \
  --transcripts tutorials/insitucnv_tracer_benchmark/data/xenium_ovary/raw/Xenium_Prime_Human_Ovary_FF_transcripts.parquet \
  --cell-metadata tutorials/insitucnv_tracer_benchmark/output/full_ovary_cnv_preview/cell_spatial_cnv_preview.csv.gz \
  --cell-metadata-out tutorials/insitucnv_tracer_benchmark/data/roi_cell_metadata.csv \
  --out tutorials/insitucnv_tracer_benchmark/data/roi_transcripts.parquet \
  --preview-out tutorials/insitucnv_tracer_benchmark/output/full_ovary_cnv_preview/plots/roi_preview_x1750_2250_y14750_15250.png \
  --preview-summary-out tutorials/insitucnv_tracer_benchmark/output/full_ovary_cnv_preview/roi_preview_x1750_2250_y14750_15250_summary.json \
  --preview-counts-out tutorials/insitucnv_tracer_benchmark/output/full_ovary_cnv_preview/roi_preview_x1750_2250_y14750_15250_counts.csv \
  --xmin 1750 --xmax 2250 --ymin 14750 --ymax 15250 \
  --remove-controls
```

The preview uses `cell_spatial_cnv_preview.csv.gz` directly for the plotted cell
classes and CNV clone labels. The ROI parquet still comes from the full
transcript table, and `roi_cell_metadata.csv` remains limited to cells that have
ROI transcripts.

## Whole-Tissue CPMI + Single Reference Condition

This is the end-to-end path for ONE reference condition (full panel, full
detection): whole-tissue CPMI -> TRACER on the ROI -> InSituCNV (raw and TRACER
arms) -> compare. Each stage is an independent `qsub` so it can be validated
incrementally. CPMI and TRACER run inside `tracer_latest.sif`; InSituCNV, the
per-arm plots, and the comparison run in the `insitucnv_tracer_benchmark` conda
env (infercnvpy/insitucnv/scvelo are not in the container).

The CPMI and TRACER SGE scripts default to `SIF=$REPO/tracer_latest.sif`. If the
container lives elsewhere, submit with `SIF=/path/to/tracer_latest.sif`.

Stage 1 -- whole-tissue CPMI panels (`scripts/build_whole_tissue_cpmi.py`,
`sge/build_whole_tissue_cpmi.sge`). The builder streams the whole-tissue
transcript parquet, subsamples eligible cells, creates a cell x gene count matrix,
and calls `tracer.conflict_reference.build_depth_corrected_reference`. The default
starting point is **50k cells** and the gene universe is **not restricted to the
ROI**; it uses all observed genes that pass the CPMI filters. The output is still
TRACER-compatible: `PMI` stores `cPMI`, `NPMI` stores `cNPMI`, and `raw_PMI` is
retained for audit.

Build both candidate references first:

```bash
cd $REPO

# Nuclear-transcript CPMI, 50k eligible cells.
qsub tutorials/insitucnv_tracer_benchmark/sge/build_whole_tissue_cpmi.sge

# Whole-cell-transcript CPMI, 50k eligible cells.
qsub -v COUNT_SCOPE=whole_cell tutorials/insitucnv_tracer_benchmark/sge/build_whole_tissue_cpmi.sge
```

These write:

- `data/whole_tissue_cpmi_nuclear_50k.csv.gz`
- `data/whole_tissue_cpmi_whole_cell_50k.csv.gz`
- matching `.summary.json` files
- matching `.seed_support.json` files from `scripts/check_panel_seed_support.py`

The seed-support JSON is the immediate acceptance gate: per ROI cell, how many
within-nuclear-seed gene pairs have positive panel weight (`PMI > 0`, which is
`cPMI > 0` for these CPMI panels). TRACER's nuclear-seed prune skips absent pairs,
so the go/no-go is `verdict: HEALTHY` with few `frac_seeds_zero_support` and a
reasonable `median_positive_pairs_per_seed`. This is more important than global
panel positivity. If a panel is `CHECK`, do not trust downstream CNV from that
panel until you adjust `QV_MIN`, `MIN_OCC`, `SUBSAMPLE`, or `TOP_K_PER_GENE`.

Re-run the gate whenever the ROI is re-selected. It is a property of the
panel AND the ROI, so a gate result from an earlier ROI says nothing about the
current one. This bit us once: the `whole_tissue_cpmi_*_50k.csv.gz.seed_support.json`
files were written 2026-07-27 15:46, but `select_roi.py` regenerated
`data/roi_transcripts.parquet` at 20:27 the same day. Those gate results
describe a superseded 28,673-cell ROI; the final ROI has 1,898 cells, of which
1,793 are usable seeds. The panels used for the 2026-08-06/07 TRACER runs were
therefore never gated against the ROI they ran on. (They do pass -- re-checked
later -- but the record claimed evidence it did not have.) A quick sanity check
catches this class of error: `n_seeds` x `median_nuclear_genes_per_seed` cannot
exceed the ROI parquet's row count.

The gate is a FLOOR, not a ranking. It counts how much positive support exists,
not whether the weights are well calibrated, and a smaller panel scores lower
simply by having fewer edges. Measured on the final ROI, the unbalanced August
panels score slightly higher than the `grid3` rebuilds (nuclear median 85,793 vs
82,960; whole-cell 87,444 vs 83,853) while all four are `HEALTHY` with zero
zero-support seeds. Do not read that as the old draw being better -- note the
gap narrows at `p10` (6,523 vs 6,456; 6,750 vs 6,572), i.e. the support given up
sits in already-well-covered cells. Use the gate to reject unusable panels, and
judge draws on downstream CNV concordance instead.

Use the nuclear 50k CPMI panel by default for the first TRACER run. If the
whole-cell CPMI panel has better support and you intentionally want to test that
arm, pass it with `PANEL=...`.

Stage 2 -- TRACER on the ROI with that panel (`sge/run_tracer_roi.sge`, calling
`scripts/run_tracer.py`). Writes `output/runs/pilot_full_100/tracer/outputs/
transcripts_tracer_refined.parquet` (final label column `stitched`, plus
`_etype` for the whole/partial split).

```bash
qsub tutorials/insitucnv_tracer_benchmark/sge/run_tracer_roi.sge

# Optional: run TRACER with the whole-cell CPMI candidate instead.
qsub -v PANEL=tutorials/insitucnv_tracer_benchmark/data/whole_tissue_cpmi_whole_cell_50k.csv.gz \
  tutorials/insitucnv_tracer_benchmark/sge/run_tracer_roi.sge
```

Stage 3 -- InSituCNV per arm (`sge/run_insitucnv_reference.sge`). For each arm it
runs `annotate_entities.py` (marker-based, lowercase labels matching the
reference/tumor defaults), then `run_insitucnv_arm.py` (infercnvpy), then
`plot_cnv_arm.py` (CNV UMAP, marker dotplot by clone, per-clone chromosome CNV
heatmap). Run both arms:

```bash
qsub -v ARM=raw          tutorials/insitucnv_tracer_benchmark/sge/run_insitucnv_reference.sge
qsub -v ARM=tracer_whole tutorials/insitucnv_tracer_benchmark/sge/run_insitucnv_reference.sge
```

Stage 4 -- compare raw vs TRACER (`sge/compare_reference_condition.sge`, after
both arms finish). Writes `output/runs/pilot_full_100/compare/`:
`compare_metrics.csv` + `compare_summary.json` (tumor_signal, reference_flatness,
signal_to_baseline and raw->tracer deltas), `compare_chrom_cnv.png` (per-chromosome
CNV, raw vs TRACER), and `compare_umap_dotplot.png` (before/after clone montage).

```bash
qsub tutorials/insitucnv_tracer_benchmark/sge/compare_reference_condition.sge
```

## Minimal Local Smoke

```bash
conda activate insitucnv_tracer_benchmark

python tutorials/insitucnv_tracer_benchmark/scripts/make_marker_preserving_panels.py \
  --genes data/roi_genes.txt \
  --gene-positions data/gene_positions_grch38.tsv \
  --config tutorials/insitucnv_tracer_benchmark/config/panels.yaml \
  --outdir tutorials/insitucnv_tracer_benchmark/output/panels

python tutorials/insitucnv_tracer_benchmark/scripts/downsample_transcripts.py \
  --transcripts data/roi_transcripts.parquet \
  --panel output/panels/panel_1000.txt \
  --detection-fraction 0.5 \
  --seed 1 \
  --out output/runs/panel1000_detect50/transcripts.parquet
```

After ROI selection, the full ROI grid is driven by `manifests/run_manifest.csv`
and `sge/run_roi_grid.sge`.
