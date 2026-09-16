#!/usr/bin/env python3
"""Build a whole-tissue PMI/NPMI panel for TRACER from HIGH-CONFIDENCE NUCLEAR transcripts.

DESIGN
======
The panel is learned from *nuclear* transcripts, not whole-cell segmentation
counts. Nuclei are spatially compact, so nuclear co-occurrence reflects genes
that are genuinely co-expressed in the same cell with minimal segmentation
spillover. Learning PMI from whole-cell counts would instead teach TRACER that
spillover genes "belong together" -- baking in the contamination TRACER exists
to remove. So co-occurrence is measured over nuclear transcripts only.

This talks to ``tracer.metrics.compute_pmi_bootstrap`` DIRECTLY through its matrix
``counts=`` path (see src/tracer/metrics.py::_presence_from_counts and the
``counts is not None`` branch). It does NOT reuse tutorials/gbm/generate_npmi.py
(only the container-run pattern is borrowed from that folder). The pipeline works
in three stages:

  1. Stream the whole-tissue transcript parquet (2.7B rows) in batches, keeping
     only high-confidence nuclear gene transcripts:
         is_gene == True  AND  qv >= --qv-min  AND  overlaps_nucleus == 1
         AND cell_id not in {UNASSIGNED, -1, ...}
     and aggregate into a nuclear cell x gene COUNT matrix (no billion-row
     DataFrame; periodic consolidation bounds memory).
  2. Feed the matrix to ``compute_pmi_bootstrap(counts=(X, genes, cells),
     min_occurrences_per_context=--min-occurrences, persist_ci=True)``. The
     counts path binarizes presence as ``count >= min_occurrences`` and still
     populates ``result.pair_ci`` with the full-data point estimates
     ``legacy_pmi`` / ``legacy_npmi`` for every observed-cooccurrence pair.
  3. Emit a long-format panel ``gene_i, gene_j, PMI, NPMI`` (self-pairs dropped,
     non-finite rows dropped loudly) -- exactly what ``scripts/run_tracer.py``'s
     ``load_npmi_panel`` and ``run_segmented_pipeline`` consume (metric_col="PMI").

DENSITY GUARD
=============
TRACER's ``prune_transcripts_nuclear_seed`` SKIPS structurally-absent pairs, so
if the panel is too sparse, seeds get zero edges and cells collapse after Prune.
Nuclear-only can be sparse, so this script (a) defaults to presence at count>=1
(``--min-occurrences 1``), (b) does NOT apply a per-cell "confident nuclei"
percentile band, and (c) prints presence-matrix density stats (median present
genes/cell, genes seen in 0 cells) and panel positivity so the panel can be
sanity-checked BEFORE running TRACER.

EXAMPLE (inside tracer_latest.sif on argos)
===========================================
::

    python tutorials/insitucnv_tracer_benchmark/scripts/build_whole_tissue_npmi.py \\
      --transcripts tutorials/insitucnv_tracer_benchmark/data/xenium_ovary/raw/Xenium_Prime_Human_Ovary_FF_transcripts.parquet \\
      --out tutorials/insitucnv_tracer_benchmark/data/whole_tissue_npmi.csv.gz \\
      --qv-min 20 --min-occurrences 1 --seed 1

LOCAL DRY-RUN (no tracer import; validates the streaming aggregator only)
========================================================================
::

    python .../build_whole_tissue_npmi.py --transcripts <parquet> \\
      --out /tmp/npmi_dryrun.csv --qv-min 20 --max-row-groups 3 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Path bootstrap so `import tracer` resolves from the repo checkout (harmless
# no-op inside the .sif, where tracer is already installed).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from common import UNASSIGNED_TOKENS  # benchmark-local helper
except Exception:
    UNASSIGNED_TOKENS = frozenset(
        {"UNASSIGNED", "Unassigned", "unassigned", "DROP", "nan", "None",
         "", "0", "-1", "NA", "<NA>"}
    )

_GENE_KEY_MULT = 100_000  # composite key = cell_code * MULT + gene_code (genes << MULT)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--transcripts", required=True, type=Path,
                   help="Whole-tissue transcript parquet (Xenium).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output long-format panel csv(.gz): gene_i,gene_j,PMI,NPMI.")

    # Transcript columns / high-confidence nuclear filter.
    p.add_argument("--cell-id-col", default="cell_id")
    p.add_argument("--gene-col", default="feature_name")
    p.add_argument("--qv-col", default="qv")
    p.add_argument("--nucleus-col", default="overlaps_nucleus")
    p.add_argument("--is-gene-col", default="is_gene")
    p.add_argument("--qv-min", type=float, default=20.0,
                   help="Minimum Xenium qv to keep (default: 20).")
    p.add_argument("--no-nucleus-filter", action="store_true",
                   help="Do NOT restrict to nucleus-overlapping transcripts (NOT recommended).")
    p.add_argument("--panel-genes", type=Path, default=None,
                   help="Optional gene universe: one symbol per line, or tsv/csv with a "
                        "'gene' column. Genes outside it are skipped. Default: all observed genes.")

    # Cell subsampling (optional; default uses all nuclear cells).
    p.add_argument("--subsample-cells", type=int, default=0,
                   help="Randomly keep at most N nuclear cells (seeded). 0 = all cells.")

    # PMI/NPMI computation (compute_pmi_bootstrap counts= path).
    p.add_argument("--min-occurrences", type=int, default=1,
                   help="presence(cell,gene) = nuclear count >= this. 1 => dense (recommended).")
    p.add_argument("--metric", choices=["npmi", "pmi"], default="npmi",
                   help="Metric stored in W_sparse. Both PMI and NPMI are always emitted.")
    p.add_argument("--pmi-formula", default="jeffreys")
    p.add_argument("--max-bootstraps", type=int, default=200)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--show-progress", action="store_true")

    # Streaming controls.
    p.add_argument("--batch-size", type=int, default=8_000_000,
                   help="Rows per parquet read batch (default: 8M).")
    p.add_argument("--consolidate-every", type=int, default=20,
                   help="Aggregate accumulated triplets every N batches to bound memory.")
    p.add_argument("--max-row-groups", type=int, default=0,
                   help="Cap the number of row groups read (0 = all). For smoke tests.")
    p.add_argument("--dry-run", action="store_true",
                   help="Stream + build the matrix + print density, then exit BEFORE "
                        "importing/calling tracer. Use to validate locally.")
    return p.parse_args()


def load_panel_genes(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    if path.suffix.lower() in {".tsv", ".csv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        gcol = "gene" if "gene" in df.columns else df.columns[0]
        genes = {str(g) for g in df[gcol].astype(str)}
    else:
        with open(path) as fh:
            genes = {ln.strip() for ln in fh if ln.strip()}
    log(f"Panel gene universe from {path.name}: {len(genes):,} genes")
    return genes


# ---------------------------------------------------------------------------
# Stage 1 — stream nuclear transcripts -> cell x gene count matrix
# ---------------------------------------------------------------------------
def stream_nuclear_counts(args: argparse.Namespace, panel: set[str] | None):
    import pyarrow.parquet as pq

    cols = [args.cell_id_col, args.gene_col, args.qv_col, args.nucleus_col]
    pf = pq.ParquetFile(args.transcripts)
    schema_names = set(pf.schema.names)
    has_is_gene = args.is_gene_col in schema_names
    if has_is_gene:
        cols.append(args.is_gene_col)
    for c in (args.cell_id_col, args.gene_col, args.qv_col, args.nucleus_col):
        if c not in schema_names:
            raise SystemExit(f"Column {c!r} not in parquet schema: {sorted(schema_names)}")

    total_rows = pf.metadata.num_rows
    n_row_groups = pf.metadata.num_row_groups
    log(f"Streaming {total_rows:,} rows / {n_row_groups:,} row groups from {args.transcripts.name}")
    log(f"Filter: is_gene==True({'yes' if has_is_gene else 'col-absent -> skipped'}), "
        f"{args.qv_col}>={args.qv_min}, "
        f"{'nucleus==1' if not args.no_nucleus_filter else 'nucleus filter OFF'}, "
        f"exclude UNASSIGNED cells")

    cell_to_idx: dict[str, int] = {}
    gene_to_idx: dict[str, int] = {}
    trip_keys: list[np.ndarray] = []   # composite cell*MULT+gene
    trip_vals: list[np.ndarray] = []   # counts
    consolidated_keys = np.zeros(0, dtype=np.int64)
    consolidated_vals = np.zeros(0, dtype=np.int64)

    def consolidate():
        nonlocal trip_keys, trip_vals, consolidated_keys, consolidated_vals
        if not trip_keys and consolidated_keys.size == 0:
            return
        keys = np.concatenate([consolidated_keys] + trip_keys) if trip_keys else consolidated_keys
        vals = np.concatenate([consolidated_vals] + trip_vals) if trip_vals else consolidated_vals
        # sum duplicate (cell,gene) keys
        order = np.argsort(keys, kind="stable")
        keys = keys[order]; vals = vals[order]
        uniq, start = np.unique(keys, return_index=True)
        summed = np.add.reduceat(vals, start) if uniq.size else vals
        consolidated_keys, consolidated_vals = uniq, summed
        trip_keys, trip_vals = [], []

    n_kept = 0
    n_batches = 0
    rg_done = 0
    stop = False
    for rg in range(n_row_groups):
        if stop:
            break
        batch_iter = pf.iter_batches(batch_size=args.batch_size, columns=cols, row_groups=[rg])
        for rb in batch_iter:
            df = rb.to_pandas()
            if has_is_gene:
                df = df[df[args.is_gene_col].astype("boolean").fillna(False).to_numpy(dtype=bool)]
            qv = pd.to_numeric(df[args.qv_col], errors="coerce")
            keep = (qv >= args.qv_min).to_numpy(dtype=bool, copy=True)
            if not args.no_nucleus_filter:
                nuc = pd.to_numeric(df[args.nucleus_col], errors="coerce")
                keep = keep & (nuc == 1).to_numpy(dtype=bool)
            df = df[keep]
            cid = df[args.cell_id_col].astype(str)
            gene = df[args.gene_col].astype(str)
            good = ~cid.isin(UNASSIGNED_TOKENS)
            if panel is not None:
                good &= gene.isin(panel)
            cid = cid[good]; gene = gene[good]
            if cid.empty:
                continue
            grp = pd.DataFrame({"c": cid.to_numpy(), "g": gene.to_numpy()})
            gc = grp.groupby(["c", "g"], sort=False).size()
            c_arr = gc.index.get_level_values(0).to_numpy()
            g_arr = gc.index.get_level_values(1).to_numpy()
            cnt = gc.to_numpy(dtype=np.int64)
            c_codes = np.fromiter((cell_to_idx.setdefault(c, len(cell_to_idx)) for c in c_arr),
                                  dtype=np.int64, count=c_arr.size)
            g_codes = np.fromiter((gene_to_idx.setdefault(g, len(gene_to_idx)) for g in g_arr),
                                  dtype=np.int64, count=g_arr.size)
            trip_keys.append(c_codes * _GENE_KEY_MULT + g_codes)
            trip_vals.append(cnt)
            n_kept += int(cnt.sum())
            n_batches += 1
            if n_batches % args.consolidate_every == 0:
                consolidate()
        rg_done += 1
        if rg_done % 200 == 0:
            log(f"  row groups {rg_done:,}/{n_row_groups:,}; kept nuclear tx so far: {n_kept:,}; "
                f"cells={len(cell_to_idx):,} genes={len(gene_to_idx):,}")
        if args.max_row_groups and rg_done >= args.max_row_groups:
            log(f"  stopping early at {rg_done} row groups (--max-row-groups)")
            stop = True
    consolidate()

    n_cells = len(cell_to_idx)
    n_genes = len(gene_to_idx)
    if n_cells == 0 or n_genes < 2:
        raise SystemExit(f"After filtering: {n_cells} cells, {n_genes} genes — nothing to compute.")
    rows = (consolidated_keys // _GENE_KEY_MULT).astype(np.int64)
    cols_ = (consolidated_keys % _GENE_KEY_MULT).astype(np.int64)
    X = sp.csr_matrix((consolidated_vals.astype(np.float32), (rows, cols_)),
                      shape=(n_cells, n_genes))
    genes = np.empty(n_genes, dtype=object)
    for g, i in gene_to_idx.items():
        genes[i] = g
    cells = np.empty(n_cells, dtype=object)
    for c, i in cell_to_idx.items():
        cells[i] = c
    log(f"Nuclear count matrix: {n_cells:,} cells x {n_genes:,} genes; "
        f"nnz={X.nnz:,}; kept nuclear tx={n_kept:,}")
    return X.tocsr(), genes.astype(str), cells.astype(str)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    panel = load_panel_genes(args.panel_genes)
    X, genes, cells = stream_nuclear_counts(args, panel)

    # Optional cell subsample (after the full nuclear matrix is built).
    if args.subsample_cells and 0 < args.subsample_cells < X.shape[0]:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(X.shape[0], size=args.subsample_cells, replace=False))
        X = X[idx].tocsr()
        cells = cells[idx]
        log(f"Subsampled to {X.shape[0]:,} nuclear cells (seed={args.seed})")

    # Density diagnostics — the thing that broke pruning on GBM.
    pres = (X >= args.min_occurrences)
    pres.eliminate_zeros()
    genes_per_cell = np.asarray(pres.sum(axis=1)).ravel()
    cells_per_gene = np.asarray(pres.sum(axis=0)).ravel()
    stats = {
        "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "min_occurrences": int(args.min_occurrences),
        "qv_min": float(args.qv_min),
        "nucleus_filter": (not args.no_nucleus_filter),
        "nnz_presence": int(pres.nnz),
        "median_present_genes_per_cell": float(np.median(genes_per_cell)) if genes_per_cell.size else 0.0,
        "mean_present_genes_per_cell": float(genes_per_cell.mean()) if genes_per_cell.size else 0.0,
        "genes_expressed_in_0_cells": int((cells_per_gene == 0).sum()),
        "median_cells_per_gene": float(np.median(cells_per_gene)) if cells_per_gene.size else 0.0,
    }
    log("Presence-matrix density: " + json.dumps(stats))

    if args.dry_run:
        _write_summary(args.out, {"dry_run": True, "transcripts": str(args.transcripts), **stats})
        log("DRY RUN complete (nuclear matrix assembled; tracer not invoked).")
        return 0

    from tracer.metrics import compute_pmi_bootstrap

    t0 = time.time()
    result = compute_pmi_bootstrap(
        counts=(X, genes, cells),
        min_occurrences_per_context=int(args.min_occurrences),
        metric=args.metric,
        pmi_formula=args.pmi_formula,
        max_bootstraps=int(args.max_bootstraps),
        tau=float(args.tau),
        seed=int(args.seed),
        persist_ci=True,
        show_progress=bool(args.show_progress),
    )
    runtime = time.time() - t0
    log(f"compute_pmi_bootstrap done in {runtime:.1f}s; genes={len(result.genes):,}; "
        f"pair_ci rows={0 if result.pair_ci is None else len(result.pair_ci):,}")

    ci = result.pair_ci
    if ci is None or ci.empty:
        raise SystemExit(
            "compute_pmi_bootstrap returned no pairs — panel would be empty. "
            "Nuclear co-occurrence may be too sparse; lower --qv-min or --min-occurrences, "
            "or add more cells."
        )

    panel_df = pd.DataFrame({
        "gene_i": ci["gene_i"].astype(str).to_numpy(),
        "gene_j": ci["gene_j"].astype(str).to_numpy(),
        "PMI": ci["legacy_pmi"].to_numpy(dtype=np.float64),
        "NPMI": ci["legacy_npmi"].to_numpy(dtype=np.float64),
    })
    panel_df = panel_df[panel_df["gene_i"] != panel_df["gene_j"]]

    finite = np.isfinite(panel_df["PMI"]) & np.isfinite(panel_df["NPMI"])
    n_drop = int((~finite).sum())
    if n_drop:
        log(f"Dropping {n_drop:,} non-finite pairs ({100 * n_drop / max(len(panel_df),1):.1f}%)")
    panel_df = panel_df.loc[finite].reset_index(drop=True)
    if panel_df.empty:
        raise SystemExit("All pairs were non-finite; refusing to write an empty panel.")

    if args.out.suffix == ".gz" or args.out.name.endswith(".csv.gz"):
        panel_df.to_csv(args.out, index=False, compression="gzip")
    else:
        panel_df.to_csv(args.out, index=False)
    log(f"Wrote {len(panel_df):,} panel pairs -> {args.out}")

    pos = int((panel_df["PMI"] > 0).sum())
    _write_summary(args.out, {
        "dry_run": False,
        "transcripts": str(args.transcripts),
        "out": str(args.out),
        "runtime_seconds": round(runtime, 1),
        "n_panel_pairs": int(len(panel_df)),
        "n_positive_pmi_pairs": pos,
        "frac_positive_pmi": round(pos / len(panel_df), 4),
        "n_nonfinite_dropped": n_drop,
        "pmi_median": round(float(panel_df["PMI"].median()), 4),
        "npmi_median": round(float(panel_df["NPMI"].median()), 4),
        **stats,
    })
    return 0


def _write_summary(out: Path, payload: dict) -> None:
    sp_path = Path(str(out) + ".summary.json")
    with sp_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    log(f"Wrote summary -> {sp_path}")


if __name__ == "__main__":
    raise SystemExit(main())
