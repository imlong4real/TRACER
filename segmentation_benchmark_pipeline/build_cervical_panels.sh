#!/bin/bash
# Build the cervical reference panel from the ADC/SCC-annotated scRNA atlas.
#
# ONE build_panels.py call. The RECIPE comes from tracer/configs/defaults.toml
# [panel] — resolved because --h5ad implies the with-reference preset:
#     strategy=rep (cell-type balanced, WITH replacement)
#     arms=[xgt1], promote=cPMI, depth_metric=n_genes
# Outputs BOTH `..._pmi_...` and `..._cpmi_...`; USE THE _cpmi_ ONE.
#
# INPUTS (verified 2026-09-08)
#   reference  23,906 cells x 20,615 genes; layers['counts'] is RAW COUNTS
#              (min 1.0, fully integral, max 16,939). X is log1p(CP10k) — do
#              NOT point the builder at X. No .raw.
#   celltype   `cell_type_fine`, 12 levels. Chosen over `cell_type` (11)
#              because it is the only column that splits Tumor Epithelial into
#              SCC (3,309) and ADC (1,354) — the contrast this atlas exists to
#              represent. Collapsing them would balance the two tumour
#              programmes into one stratum. `cell_type` and `cell_type_coarse`
#              are IDENTICAL (verified), so there is no third option.
#              Note `Unannotated` (1,966) is a real stratum here; with
#              replacement-balancing it gets sampled up to the mean type size,
#              so consider --exclude-obs cell_type_fine=Unannotated if that
#              is unwanted.
#
#   ⚠ TWO cervical spatial datasets exist and the gene universe is a CHOICE:
#       atera   17,420 genes  (whole-transcriptome; default)
#       xenium5k 4,863 genes  (5k panel)
#     Pass one as $1. This determines the panel entirely, so pick the platform
#     the panel will actually be USED on.
#
# Usage:  ./build_cervical_panels.sh [atera|xenium5k]
set -euo pipefail
cd "$(dirname "$0")"

# --- conda env -------------------------------------------------------------
# Must run under `genesis_env`: base python lacks pyarrow/anndata AND the
# Cython kernels are built against this env. `conda activate` trips `set -u`
# (its shell functions read unbound vars), so relax it just for the activation.
CONDA_SH="${CONDA_SH:-/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-genesis_env}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  [ -f "$CONDA_SH" ] || { echo "conda hook not found: $CONDA_SH (set CONDA_SH=)" >&2; exit 2; }
  set +u; . "$CONDA_SH"; conda activate "$CONDA_ENV"; set -u
fi
python - <<'PYCHK' || exit 2
import sys
miss = [m for m in ("pyarrow", "anndata", "scipy", "pandas")
        if __import__("importlib.util", fromlist=["util"]).find_spec(m) is None]
print(f"[env] {sys.executable}")
if miss:
    print(f"[env] MISSING: {', '.join(miss)} — wrong environment?"); sys.exit(1)
PYCHK

export PYTHONPATH="$(cd .. && pwd)/src"

PLATFORM="${1:-atera}"
case "$PLATFORM" in
  atera)    GENES=../Figures/ROIs_for_benchmarking/atera_cervical/roi_transcripts.parquet ;;
  xenium5k) GENES=../Figures/ROIs_for_benchmarking/xenium5k_cervical/roi_transcripts.parquet ;;
  *) echo "unknown platform '$PLATFORM' (want: atera | xenium5k)" >&2; exit 2 ;;
esac

REF=refs/cervical_scrna_adc_scc_marker_annotated.h5ad
CTCOL="${CTCOL:-cell_type_fine}"
OUT="${OUT:-panels_cervical_$PLATFORM}"

for f in "$REF" "$GENES"; do
  [ -f "$f" ] || { echo "missing input: $f" >&2; exit 2; }
done
echo "platform=$PLATFORM  genes=$GENES  celltype=$CTCOL  out=$OUT"

python - "$REF" "$GENES" <<'PY'
import sys, anndata as ad, pandas as pd
ref = ad.read_h5ad(sys.argv[1], backed="r")
sp = set(pd.read_parquet(sys.argv[2], columns=["feature_name"]).feature_name.astype(str))
rv = set(map(str, ref.var_names))
k = sp & rv
print(f"[genes] spatial panel {len(sp):,} | reference {len(rv):,} | INTERSECTION {len(k):,} "
      f"({100*len(k)/max(len(sp),1):.1f}% of the panel)")
if len(k) < 0.8 * len(sp):
    print(f"[genes] WARNING only {100*len(k)/len(sp):.1f}% of panel genes are in the "
          f"reference — check gene-symbol conventions before trusting the panel.")
PY

python build_panels.py --h5ad "$REF" --panel-genes "$GENES" --celltype-col "$CTCOL" --out "$OUT"

echo
echo "Done -> $OUT/  (recipe recorded in $OUT/panel_receipt.json)"
ls -la "$OUT"/*.csv.gz 2>/dev/null | awk '{printf "  %8.1f MB  %s\n",$5/1048576,$9}'
