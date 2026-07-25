# GBM Minimal Tutorial

End-to-end GBM (Slide 3, Patient4) workflow on the cluster. TRACER stages run
inside the TRACER **Apptainer/Singularity** container; the InSituCNV stage runs in
a dedicated conda env. The SGE job scripts (`*.sge`) wrap `apptainer exec`/`conda`
internally, so you just submit them with `qsub`; the piece-preparation step is run
directly with `apptainer exec`.

Pipeline:

0. Build the container once.
1. Generate a nucleus-based NPMI/PMI matrix.
2. Detect Slide 3 tissue pieces and write one transcript parquet per approved piece.
3. Run TRACER on each piece.
4. Merge the pieces and compare original vs TRACER-refined profiles.
5. Infer CNV subclones with InSituCNV, per segmentation arm (raw vs TRACER).

This folder lives in the **`gbm` worktree**. Throughout, `REPO` is that worktree
root on the cluster:

```bash
REPO=/mnt/storage/dept/medonc/beroukhim/youyun/BTC_GBM/code/TRACER-gbm
```

The large, gitignored inputs/outputs (`tracer_latest.sif`, `tutorials/gbm/data/`,
`tutorials/gbm/output/`) are **symlinks** back to the canonical `…/code/TRACER`
checkout, so `$REPO`-relative paths resolve while the data lives in one place. The
`--bind /mnt/storage:/mnt/storage` in each apptainer command makes those symlink
targets reachable inside the container.

## 0. Build the Apptainer container

Normally already present in this worktree (symlinked from `…/code/TRACER`). To
build from scratch:

```bash
conda activate segmentation
cd /mnt/storage/dept/medonc/beroukhim/youyun/BTC_GBM/code/TRACER
apptainer pull tracer_latest.sif docker://ghcr.io/imlong4real/tracer:latest
```

This writes `tracer_latest.sif` at the canonical repo root; the `gbm` worktree
references it via a `tracer_latest.sif` symlink, which every command below uses.

## Expected Input

TRACER consumes a transcript parquet with these columns:

- `feature_name`, `cell_id`
- `transcript_id`, `qv`, `overlaps_nucleus`
- coordinates as `x`, `y`, `z` **or** raw Xenium `x_location`, `y_location`,
  `z_location` (`z` optional; filled with `0.0` if absent)

TRACER writes the refined per-transcript whole-cell label to **`cell_id_tracer`**
(plus `cell_id_stitched`), preserving the original `cell_id`. NOTE: runs before
mid-2026 used `cell_id_finetuned` for this column; the downstream merge/compare and
InSituCNV scripts accept either name.

## 1. NPMI / PMI

```bash
cd $REPO
qsub tutorials/gbm/generate_npmi.sge
```

Writes `tutorials/gbm/data/gbm_npmi_no_controls.csv` (keeps `qv >= 30`,
excludes Xenium control/deprecated features, uses nucleus-overlapping transcripts,
keeps confident nuclei between the 20th and 80th percentile of transcript counts,
and applies a 5th-95th percentile per-gene size-band filter).

### Piece 5 PMI/Prune diagnostic

Before replacing the production panel, generate a corrected spatial PMI panel
from piece 5 and run only the compiled nuclear-seed Prune stage:

```bash
cd $REPO
mkdir -p logs
qsub tutorials/gbm/run_gbm_piece5_pmi_diagnostic.sge
```

The job exports only accepted `W_sparse` edges. The complete bootstrap
`pair_ci` table remains a separate audit file and cannot enter Prune. It also
compares the corrected panel with the existing whole-slide panel. To include a
single-cell panel comparison, pass its cluster-visible path:

```bash
qsub -v SCRNA_PMI=/mnt/storage/path/to/pmi_panel_GBM_logcp10000_xge1_sub50k_long.csv \
  tutorials/gbm/run_gbm_piece5_pmi_diagnostic.sge
```

Outputs are isolated under
`tutorials/gbm/output/pmi_diagnostics/piece5/`. The key receipt is
`piece5_spatial_prune_summary.json`; the per-nucleus evidence is in
`piece5_spatial_prune_cells.csv`. Interpret retained original nuclei as:

- `>= 4,500`: piece scope or the old exporter explains the prior collapse.
- `2,000-4,499`: review edge coverage and per-seed support before proceeding.
- `< 2,000`: spatial PMI remains incompatible; test the single-cell panel for production.

## 2. Slide 3 tissue pieces

**QC first.** Detect major tissue components from the Slide 3 Xenium output, write a
component summary + review plot, and create a manual approval template (no per-piece
parquets yet):

```bash
cd $REPO
apptainer exec --bind $PWD:/app --bind /mnt/storage:/mnt/storage --pwd /app \
    tracer_latest.sif \
    python /app/tutorials/gbm/prepare_slide3_pieces.py \
      --qc-only \
      --xenium-output Patient4=/mnt/storage/dept/medonc/beroukhim/youyun/BTC_GBM/data/xenium/output-XETG00323__0023274__Patient4__20241004__181038 \
      --outdir $REPO/tutorials/gbm/output/slide3_qc
```

Review these files:

```text
tutorials/gbm/output/slide3_qc/component_plot.png
tutorials/gbm/output/slide3_qc/component_summary.csv
tutorials/gbm/output/slide3_qc/component_approval_template.csv
```

Set `approved=yes` for the accepted rows in `component_approval_template.csv`, then
**write one transcript parquet per approved piece** (this also writes the SGE task
manifest `slide3_pieces/piece_run_manifest.csv`):

```bash
apptainer exec --bind $PWD:/app --bind /mnt/storage:/mnt/storage --pwd /app \
    tracer_latest.sif \
    python /app/tutorials/gbm/prepare_slide3_pieces.py \
      --write-pieces \
      --xenium-output Patient4=/mnt/storage/dept/medonc/beroukhim/youyun/BTC_GBM/data/xenium/output-XETG00323__0023274__Patient4__20241004__181038 \
      --outdir $REPO/tutorials/gbm/output/slide3_qc \
      --approved-manifest $REPO/tutorials/gbm/output/slide3_qc/component_approval_template.csv \
      --pieces-outdir $REPO/tutorials/gbm/output/slide3_pieces
```

`prepare_slide3_pieces.py` excludes `P4_resection`; the workflow currently targets
Patient4 (add a second `--xenium-output Patient6=...` when its Xenium output is
available).

## 3. Run TRACER on each piece

For the current Patient4-only run the manifest has 8 tasks:

```bash
cd $REPO
mkdir -p tutorials/gbm/output logs
qsub -t 1-8 tutorials/gbm/run_gbm_pieces.sge
```

Each task reads its manifest row from `SGE_TASK_ID` and writes
`tutorials/gbm/output/slide3_tracer/slide3_piece_<NN>_Patient4_tracer.parquet`.

**Re-running a single failed task** (e.g. if one piece died on a full disk):

```bash
qsub -t 4 tutorials/gbm/run_gbm_pieces.sge
```

If Patient6 is added later and 12 pieces are approved, submit `qsub -t 1-12 ...`.

## 4. Merge + compare profiles

> ⚠️ **Preserve the previous run first.** The merge/compare step overwrites
> `slide3_tracer_merged.parquet` and `slide3_profile_comparison/`. To keep a prior
> TRACER run, rename it to a dated name before submitting:
>
> ```bash
> cd $REPO/tutorials/gbm/output
> mv slide3_tracer_merged.parquet   slide3_tracer_merged_$(date +%F).parquet
> mv slide3_profile_comparison      slide3_profile_comparison_$(date +%F)
> ```

```bash
cd $REPO
qsub tutorials/gbm/run_gbm_compare.sge
```

**Split across two environments.** Leiden clustering needs `igraph`/`leidenalg`,
which the pulled `tracer_latest.sif` lacks (they're downstream-only and the container
is refreshed only via a `main` merge + re-pull). So the comparison is split along its
dependency fault line and the SGE script chains **three** steps — the first two inside
the container, the third in the `insitucnv_env` conda env (the shared
"everything after TRACER" env; **create it first — see §5**, which now also pins
`umap-learn` for `sc.tl.umap`):

1. `merge_slide3_tracer.py` **[container]** — concatenates the per-piece TRACER parquets
   into `tutorials/gbm/output/slide3_tracer_merged.parquet`, tagging each row with
   `piece_id`/`slide_tissue_id` and normalizing the whole-cell column to `cell_id_tracer`.
2. `compare_profiles.py --stage prep` **[container, needs `tracer`]** — builds
   `original` (`cell_id`) vs TRACER-refined whole-cell AnnData (auto-detecting
   `cell_id_tracer` or `cell_id_finetuned`), computes purity/conflict scores, joins
   Patient4 cell-type annotations, QC-filters, and writes intermediates
   `adata_{orig,ft}_prepped.h5ad` + `prep_manifest.json`.
3. `compare_profiles.py --stage cluster` **[`insitucnv_env`, needs `igraph`]** — reads
   the prepped h5ads and runs the scanpy pipeline (normalize → PCA → neighbors → UMAP →
   Leiden → Wilcoxon markers), writing to `tutorials/gbm/output/slide3_profile_comparison/`:

```text
profile_summary.csv                           # per labeling: n_cells, mean transcripts/genes, purity, conflict, n_clusters
{original,finetuned}_top_markers.csv
{original,finetuned}_marker_matrixplot.png
adata_{orig,ft}.h5ad                          # full AnnData: UMAP, PCA, Leiden, purity/conflict, cell types
```

**Three-way comparison** (original vs a previous TRACER run vs the current run, aligned
by `piece_id`): preserve the previous run's merged parquet as
`slide3_tracer_merged_2025-05-07.parquet`, then submit `run_gbm_threeway.sge`. It uses the
same three-step split (`merge_slide3_tracer.py --allow-missing` → `compare_three_way.py
--stage prep` in the container → `--stage cluster` in `insitucnv_env`) and writes
`adata_{original,tracer_may,tracer_new}.h5ad` + matching `*_top_markers.csv` /
`*_marker_matrixplot.png` and a 3-row `profile_summary.csv`.

## 5. InSituCNV subclone analysis (per segmentation arm)

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

This stage runs in the `insitucnv_env` conda env, **not** the Apptainer container.
This is the shared "everything after TRACER" env — it also runs the `--stage cluster`
step of the profile comparison in §4 (Leiden/UMAP/markers). Create it once:

```bash
conda env create -f tutorials/gbm/insitucnv_env.yml
conda activate insitucnv_env
pip install git+https://github.com/Moldia/InSituCNV.git   # NOT -e /tmp (node-local)
```

(The `InSituCNV` pip install is only needed for §5; the §4 cluster step uses just the
conda packages — `scanpy`/`igraph`/`leidenalg`/`umap-learn`.)

Run each arm (annotations, RES, and depth-match are optional overrides):

```bash
cd $REPO
qsub -v PIECE=04,ARM=raw    tutorials/gbm/run_insitucnv_arm.sge
qsub -v PIECE=04,ARM=tracer tutorials/gbm/run_insitucnv_arm.sge
```

Each arm writes to `tutorials/gbm/output/insitucnv/piece<NN>/<arm>/`:
`adata_cnv.h5ad`; `arm_summary.json` (depth, baseline flatness, per-chromosome
resolution, per-resolution subclones + events); `cnv_clusters_r{r}.csv` (every CNV
cluster: sizes, tumor/ref/unknown fractions, `is_subclone`, events);
`subclone_cohensd_r{r}.csv` (subclone × chromosome Cohen's d vs reference);
`subclone_chrom_cnv_r{r}.csv`; `subclone_assignments_r{r}.csv`; and plots
`plots/cnv_heatmap_r{r}.png` (per-cell × genome), `plots/spatial_clusters_r{r}.png`,
plus two per-subclone summary heatmaps — `plots/subclone_chrom_cnv_heatmap_r{r}.png`
(subclone × chromosome mean CNV, with a reference baseline row) and
`plots/subclone_cohensd_heatmap_r{r}.png` (subclone × chromosome Cohen's d vs
reference). A cheap resolution re-sweep that skips inferCNV (and regenerates all of
these): add `-v FROM_H5AD=1,RES=0.03,0.08`.

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

## Single whole-slide run (no pieces)

To run TRACER over a whole transcript parquet in one job instead of per-piece:

```bash
qsub tutorials/gbm/run_gbm.sge   # writes tutorials/gbm/output/df_finetuned.parquet
```

then compare with `compare_profiles.py` via `run_gbm_compare.sge` (§4). To invoke it
directly, run the two stages in order — `--stage prep` in the container, then
`--stage cluster` in `insitucnv_env` — pointing `--input` at the whole-slide parquet.

## Interactive Analysis

`explore_slide3.qmd` is a Quarto notebook for visualizing the h5ad outputs. Install
dependencies once in the `segmentation` conda environment:

```bash
conda install -c conda-forge quarto seaborn
```

Then render from the repo root:

```bash
conda activate segmentation
quarto render tutorials/gbm/explore_slide3.qmd --to html
```

The self-contained `tutorials/gbm/explore_slide3.html` can be opened in any browser.
It covers TRACER impact per cell type, UMAP visualizations, per-cell-type log2FC
before/after refinement, and paired purity/conflict comparisons for matched whole
cells.

## Notes

- This is the `gbm` worktree: the SGE scripts hardcode `REPO=…/code/TRACER-gbm` and
  bind `/mnt/storage`. `tracer_latest.sif`, `data/`, and `output/` are symlinks to
  the canonical `…/code/TRACER` checkout (one shared data copy). Adjust `REPO` if
  the worktree lives elsewhere.
- `generate_npmi.py` keeps `qv >= 30`, excludes Xenium control/deprecated features,
  requires `overlaps_nucleus == 1`, keeps confident nuclei between the 20th and
  80th percentile of transcript counts, and applies a 5th-95th percentile
  per-gene size-band filter.
- `run_gbm.py` uses the lung-tutorial settings (`deltaC_min=0.01`,
  `dist_threshold=5.0`) and writes the whole-cell label to `cell_id_tracer`.
- The `compare_profiles.py` / InSituCNV scripts accept either `cell_id_tracer`
  (current) or `cell_id_finetuned` (older runs) as the TRACER whole-cell column.
