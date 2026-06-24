#!/usr/bin/env bash
# Wait until RCTD has finished for all four methods, then re-render Panel E
# (RCTD entropy / max-weight half-violins for all methods) and refresh the
# Figure-4 manifest + run summary + reaggregation RCTD status.
set -uo pipefail
cd "$(dirname "$0")/../../../.."   # repo root
RCTD=results/kidney_visiumhd_noseg_bin2cell_benchmark/rctd
PY=/Users/lyuan13/anaconda3/envs/spatial/bin/python

need=(tracer_8um 10x_segmented bin2cell_2um tracer_2um)
while true; do
  missing=0
  for m in "${need[@]}"; do
    [ -f "$RCTD/$m/rctd_cell_assignments_post.tsv" ] || missing=1
  done
  [ "$missing" -eq 0 ] && break
  sleep 120
done

echo "[finalize] all RCTD complete -> re-rendering Panel E + manifest"
cd scripts/reproducibility/fig4
"$PY" panel_e_quantitative_benchmark.py
"$PY" make_fig4.py --panels E
echo "[finalize] done"
