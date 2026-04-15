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

## Notes

- `generate_npmi.py` keeps `qv >= 30` transcripts, requires `overlaps_nucleus == 1`, and keeps confident nuclei between the 20th and 80th percentile of transcript counts.
- `run_gbm.py` uses the lung tutorial settings, including `deltaC_min=0.01` and `dist_threshold=5.0` for stitching and spatial refinement.
- `compare_profiles.py` outputs `profile_summary.csv`, top-marker CSVs, and matrixplots for original and finetuned whole cells.
