#!/usr/bin/env bash
# Run RCTD (spacexr, doublet mode) for all four Figure-4 methods on the
# matched 1,656 HVG/NPMI gene panel. Uses the Schwann-excluded 9-lineage
# kidney scRNA reference (obs['lineage']) so RCTD lineages match the
# Figure-4 palette. Runs smallest method first for an early throughput read.
#
# Usage: bash scripts/reproducibility/fig4/prep/run_rctd_all.sh
set -uo pipefail
cd "$(dirname "$0")/../../../.."   # repo root

RSCRIPT=/Users/lyuan13/anaconda3/envs/tracer_benchmark_r/bin/Rscript
# Force reticulate to a Python that has the `anndata` module. The repo ./.venv
# (system py3.9, no anndata) is auto-selected otherwise and RCTD fails on read.
export RETICULATE_PYTHON=/Users/lyuan13/anaconda3/envs/spatial/bin/python
# Sanitized reference: lineage 'FIB/VSMC/P' -> 'FIB_VSMC_P' (spacexr forbids '/').
REF=results/tracer_noseg/_ref/kidney_ref_noschwann_rctd.h5ad
IN=results/kidney_visiumhd_noseg_bin2cell_benchmark/rctd/inputs
OUTBASE=results/kidney_visiumhd_noseg_bin2cell_benchmark/rctd

run_one () {
  local method="$1"
  local outdir="$OUTBASE/$method"
  mkdir -p "$outdir"
  echo "=================================================================="
  echo "[rctd] $method  ->  $outdir   ($(date))"
  echo "=================================================================="
  "$RSCRIPT" scripts/run_rctd.R \
    --spatial-h5ad   "$IN/${method}_rctd_input.h5ad" \
    --reference-h5ad "$REF" \
    --reference-celltype-col lineage \
    --outdir "$outdir" \
    --doublet-mode doublet \
    --umi-min 10 \
    --umi-min-sigma 20 \
    --max-cores 8 \
    --seed 1 \
    2>&1 | tee "$outdir/rctd_run.log"
  echo "[rctd] $method done ($(date))"
}

for m in tracer_8um 10x_segmented bin2cell_2um tracer_2um; do
  run_one "$m"
done
echo "[rctd] ALL DONE ($(date))"
