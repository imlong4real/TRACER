# GBM Minimal Tutorial

This folder provides the shortest end-to-end GBM workflow in this repo:

1. Generate a nucleus-based NPMI matrix.
2. Run TRACER with the same core parameters used in the lung cancer tutorial.
3. Compare original and TRACER-refined whole-cell profiles with unsupervised top-marker analysis.

## Expected Input

Use a transcript parquet with these columns:

- `feature_name`, `cell_id`
- `transcript_id`, `qv`, `overlaps_nucleus`
- coordinates can be either `x`, `y`, `z` or raw Xenium `x_location`, `y_location`, `z_location`
- `z` is optional; if neither `z` nor `z_location` exists, `run_gbm.py` fills it with `0.0`

## Commands

```bash
python tutorials/gbm/generate_npmi.py \
  --input path/to/gbm_transcripts.parquet \
  --output tutorials/gbm/data/gbm_npmi.csv

python tutorials/gbm/run_gbm.py \
  --input path/to/gbm_transcripts.parquet \
  --npmi tutorials/gbm/data/gbm_npmi.csv \
  --output tutorials/gbm/output/df_finetuned.parquet

python tutorials/gbm/compare_profiles.py \
  --input tutorials/gbm/output/df_finetuned.parquet \
  --npmi tutorials/gbm/data/gbm_npmi.csv \
  --outdir tutorials/gbm/output/profile_comparison
```

## Slide 3 Tissue Pieces

Run QC first. This detects major tissue components from the available Slide 3
Xenium outputs, writes a component summary, and creates a manual approval
template without writing per-piece transcript parquets.

```bash
python tutorials/gbm/prepare_slide3_pieces.py --qc-only
```

Review these files:

```text
tutorials/gbm/output/slide3_qc/component_plot.png
tutorials/gbm/output/slide3_qc/component_summary.csv
tutorials/gbm/output/slide3_qc/component_approval_template.csv
```

After manually setting `approved=yes` for the accepted rows in
`component_approval_template.csv`, write one transcript parquet per approved
piece and create the SGE task manifest.

```bash
python tutorials/gbm/prepare_slide3_pieces.py \
  --write-pieces \
  --approved-manifest tutorials/gbm/output/slide3_qc/component_approval_template.csv
```

For the current Patient4-only run, the manifest should contain 8 tasks:

```bash
qsub -t 1-8 tutorials/gbm/run_gbm_pieces.sge
```

If Patient6 Xenium output is added later and 12 pieces are approved, submit
`qsub -t 1-12 tutorials/gbm/run_gbm_pieces.sge` instead.

Once all piece jobs finish, merge the outputs and run profile comparison in a
single SGE job (4 cores, 128 GB):

```bash
qsub tutorials/gbm/run_gbm_compare.sge
```

This chains two steps automatically:

1. `merge_slide3_tracer.py` — concatenates the 8 per-piece TRACER parquets into
   `tutorials/gbm/output/slide3_tracer_merged.parquet`, tagging each row with
   `piece_id` and `slide_tissue_id` from the manifest.
2. `compare_profiles.py` — builds original and TRACER-finetuned cell profiles,
   computes purity/conflict scores, runs Leiden clustering, joins Patient4 cell
   type annotations from `Xenium_Annotations/adata_obs_annotated.csv`, and writes
   outputs to `tutorials/gbm/output/slide3_profile_comparison/`.

Review these outputs:

```text
tutorials/gbm/output/slide3_profile_comparison/profile_summary.csv
tutorials/gbm/output/slide3_profile_comparison/original_top_markers.csv
tutorials/gbm/output/slide3_profile_comparison/finetuned_top_markers.csv
tutorials/gbm/output/slide3_profile_comparison/original_marker_matrixplot.png
tutorials/gbm/output/slide3_profile_comparison/finetuned_marker_matrixplot.png
tutorials/gbm/output/slide3_profile_comparison/adata_orig.h5ad
tutorials/gbm/output/slide3_profile_comparison/adata_ft.h5ad
```

The two h5ad files contain the full AnnData objects (UMAP, PCA, Leiden clusters,
purity/conflict scores, cell type annotations) ready for interactive exploration.

## Interactive Analysis

`explore_slide3.qmd` is a Quarto notebook for visualizing and analysing the
h5ad outputs. Install dependencies once in the `segmentation` conda environment:

```bash
conda install -c conda-forge quarto seaborn
```

Then render from the repo root:

```bash
conda activate segmentation
quarto render tutorials/gbm/explore_slide3.qmd --to html
```

The self-contained `tutorials/gbm/explore_slide3.html` can be copied locally and
opened in any browser. The notebook covers:

1. **TRACER impact per cell type** — retention rate, transcript count delta, and
   purity/conflict distributions by cell type.
2. **UMAP visualizations** — original and finetuned cells coloured by cell type,
   purity, conflict, and Leiden cluster.
3. **Analysis A — cleaned out vs lumped in** — per-cell-type log2FC comparing
   mean raw expression before and after TRACER refinement.
4. **Analysis B — paired purity/conflict comparison** — Wilcoxon signed-rank
   tests and scatter plots for matched whole cells.

## InSituCNV subclone analysis (per segmentation arm)

Infer copy-number variation per cell with [InSituCNV](https://github.com/Moldia/InSituCNV)
(Jensen et al. 2025 — a thin wrapper around `infercnvpy`/inferCNV with scVelo
smoothing) and resolve **CNV subclones**, per segmentation arm.
`run_insitucnv_arm.py` runs end to end for one arm: **(1) inferCNV → (2) cluster
all cells on the CNV matrix → (3) select tumor CNV clusters as subclones and
profile each.** Run it on both arms — `raw` (original Xenium `cell_id`) and
`tracer` (the refined `cell_id_tracer`/`cell_id_finetuned` label from the piece
parquets) — and compare by eyeballing the per-subclone outputs. There is **no
bulk-tumor comparison**: pooling all tumor cells averages subclones away.

Tumor and reference cells come from your **existing** annotation
(`adata_obs_annotated.csv`), not de-novo clustering — one source of truth across
arms. `cancer_*` → tumor; myeloid / T-cell / vascular / oligo / neutrophil →
reference (the inferCNV baseline); else unknown. Raw arm: direct `cell_id` lookup.
Tracer arm: each refined cell inherits the **majority** annotation of its
constituent transcripts' original `cell_id`.

Runs in its own conda env (`insitucnv_env`), **not** the Apptainer container:

```bash
conda env create -f tutorials/gbm/insitucnv_env.yml
conda activate insitucnv_env
pip install git+https://github.com/Moldia/InSituCNV.git   # NOT -e /tmp (node-local)
```

Run each arm (annotations, RES, and depth-match are optional overrides):

```bash
qsub -v PIECE=04,ARM=raw    tutorials/gbm/run_insitucnv_arm.sge
qsub -v PIECE=04,ARM=tracer tutorials/gbm/run_insitucnv_arm.sge
```

Each arm writes to `tutorials/gbm/output/insitucnv/piece<NN>/<arm>/`:
`adata_cnv.h5ad`; `arm_summary.json` (depth, baseline flatness, per-chromosome
resolution, per-resolution subclones + events); `cnv_clusters_r{r}.csv` (every CNV
cluster: sizes, tumor/ref/unknown fractions, `is_subclone`, events);
`subclone_cohensd_r{r}.csv` (subclone × chromosome Cohen's d vs reference);
`subclone_chrom_cnv_r{r}.csv`; `subclone_assignments_r{r}.csv`; and
`plots/cnv_heatmap_r{r}.png` / `plots/spatial_clusters_r{r}.png`. A cheap
resolution re-sweep that skips inferCNV: add `-v FROM_H5AD=1,RES=0.03,0.08`.

**Subclones** are CNV clusters whose tumor-annotated fraction ≥
`--tumor-cluster-frac` (0.5). Each is profiled by per-chromosome mean CNV and
**Cohen's d vs reference cells** (intrinsic signal separation — not a bulk-WGS
correlation, since WGS is bulk and Xenium cells are polyclonal). Pick the
resolution whose subclones are distinct (e.g. separating the chr8+/chr13±/chr14+/
chr19− events that are subclonal in this tumor) and spatially coherent.

Caveats: (1) inferCNV windows are per-chromosome and the panel is sparse (~366
genes; ~16 of 23 chromosomes have <20 genes), so defaults are
`--window-size 10 --step 3`, `chrom_resolution.csv` reports genes+windows per
chromosome, and chromosomes below `--min-genes-per-chromosome` (10) are flagged
`low_resolution` and excluded from the Cohen's d. (2) Cohen's d and baseline
flatness shrink with per-cell depth, and TRACER makes fewer/denser cells, so run
both arms depth-matched (`-v DOWNSAMPLE=<shallower-arm-median>`, writes
`<arm>_ds<N>/`) before trusting a difference.

## Notes

- `generate_npmi.py` keeps `qv >= 30` transcripts, requires `overlaps_nucleus == 1`, and keeps confident nuclei between the 20th and 80th percentile of transcript counts.
- `run_gbm.py` uses the lung tutorial settings, including `deltaC_min=0.01` and `dist_threshold=5.0` for stitching and spatial refinement.
- `compare_profiles.py` outputs `profile_summary.csv`, top-marker CSVs, and matrixplots for original and finetuned whole cells.
- `prepare_slide3_pieces.py` excludes `P4_resection`; the Slide 3 piece workflow currently targets Patient4 and optionally Patient6 when its Xenium output is present.
