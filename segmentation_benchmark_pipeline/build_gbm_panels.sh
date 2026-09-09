#!/bin/bash
# Build the GBM reference panel from Long's annotated scRNA atlas.
#
# ONE build_panels.py call. Everything about the RECIPE now comes from
# tracer/configs/defaults.toml [panel], resolved because --h5ad implies the
# with-reference preset:
#     strategy=rep (cell-type balanced, WITH replacement)
#     arms=[xgt1]  (count > (e-1)*lib/1e4)
#     promote=cPMI (what lands in the PMI column the pipeline reads)
#     depth_metric=n_genes  (xgt1 already conditions on library size)
# Outputs BOTH `..._pmi_...` and `..._cpmi_...`; USE THE _cpmi_ ONE.
#
# INPUTS (verified 2026-09-08)
#   reference  232,714 cells x 38,536 genes; layers['counts'] is RAW COUNTS
#              (min 1.0, fully integral, max 30,552). X is log1p(CP10k) — do
#              NOT point the builder at X.
#   celltype   `new_consensus`, 11 levels (T/NK, Myeloid, Glial-Neuronal,
#              Neoplastic, ...). Comparable granularity to the lung reference's
#              9. Alternatives if you want a different level:
#                gbmap_predicted__annotation_level_2   6
#                gbmap_predicted__annotation_level_3  20
#                cluster_majority                      7
#              NOT tumor_archetype / immune_subtype / vascular_subtype — each
#              has an NA / not_* catch-all covering the cells outside its
#              scope, so none is a complete partition and balancing on one
#              would lump most of the atlas into a single stratum.
#   genes      the Xenium GBM panel, taken from the spatial transcripts.
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

REF=refs/gbm_scrna_qc_with_tumor_archetype_immune_vascular_subtypes.h5ad
GENES_SRC=../tutorials/gbm/slide3_piece_05_Patient4.parquet
CTCOL="${CTCOL:-new_consensus}"
OUT="${OUT:-panels_gbm}"

for f in "$REF" "$GENES_SRC"; do
  [ -f "$f" ] || { echo "missing input: $f" >&2; exit 2; }
done

# The Xenium transcripts carry 541 distinct `feature_name` values, but 175 are
# NOT genes: 53 deprecated_codeword, 41 negative_control_codeword, 20
# negative_control_probe, 61 unassigned_codeword. Filter on `is_gene` -> 366
# real genes (365 in the reference; only BTBD11 missing). The builder would
# have dropped the controls anyway by intersecting with the reference, but they
# do not belong in a gene universe and their presence made the overlap
# pre-flight read a spurious 67.5%.
GENELIST=refs/gbm_panel_genes.txt
if [ ! -f "$GENELIST" ]; then
  python - "$GENES_SRC" "$GENELIST" <<'PYX'
import sys, pandas as pd
d = pd.read_parquet(sys.argv[1], columns=["feature_name", "is_gene"])
g = sorted(d.loc[d.is_gene.astype(bool), "feature_name"].astype(str).unique())
open(sys.argv[2], "w").write("\n".join(g) + "\n")
print(f"[genes] wrote {len(g):,} real genes -> {sys.argv[2]} (dropped "
      f"{d.feature_name.nunique() - len(g)} control/deprecated codewords)")
PYX
fi
GENES="$GENELIST"

# Pre-flight: a poor scRNA/panel gene overlap silently produces a thin panel.
python - "$REF" "$GENES" <<'PY'
import sys, anndata as ad, pandas as pd
ref = ad.read_h5ad(sys.argv[1], backed="r")
sp = set(l.strip() for l in open(sys.argv[2]) if l.strip())
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
