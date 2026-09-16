#!/usr/bin/env python3
"""Build a whole-tissue depth-corrected PMI panel for TRACER.

This is the CPMI replacement for the older whole-tissue PMI/NPMI benchmark
panel. It streams the full Xenium transcript parquet, aggregates eligible
transcripts into a cell x gene count matrix, then calls
``tracer.conflict_reference.build_depth_corrected_reference``.

Two count scopes are supported:

``nuclear``
    Uses only high-confidence nucleus-overlapping transcripts
    (``overlaps_nucleus == 1``). This is the least circular reference for
    segmentation cleanup because it avoids whole-cell segmentation spillover.

``whole_cell``
    Uses all high-confidence transcripts assigned to a Xenium cell. This is a
    deliberately separate comparison arm because it can encode segmentation
    spillover that TRACER is intended to remove.

The output is long-format and TRACER-compatible: ``PMI`` stores CPMI and
``NPMI`` stores cNPMI so the existing TRACER loader and pruning code can consume
the panel without changing the segmentation internals. The raw uncorrected PMI
is retained as ``raw_PMI`` for audit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Path bootstrap so local checkout imports work inside and outside Apptainer.
# ``tracer.__init__`` pulls optional heavy dependencies, so the CPMI builder is
# loaded directly from its file if a normal package import fails.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCH_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from common import UNASSIGNED_TOKENS
except Exception:
    UNASSIGNED_TOKENS = frozenset(
        {
            "UNASSIGNED",
            "Unassigned",
            "unassigned",
            "DROP",
            "nan",
            "None",
            "",
            "0",
            "-1",
            "NA",
            "<NA>",
        }
    )

_GENE_KEY_MULT = 100_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--transcripts",
        required=True,
        type=Path,
        help="Whole-tissue transcript parquet (Xenium).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output csv(.gz). Default: "
            "data/whole_tissue_cpmi_<count_scope>_<subsample>.csv.gz."
        ),
    )
    p.add_argument(
        "--count-scope",
        choices=["nuclear", "whole_cell"],
        default="nuclear",
        help="Transcript scope used to build the cell x gene matrix.",
    )

    # Transcript columns / filters.
    p.add_argument("--cell-id-col", default="cell_id")
    p.add_argument("--gene-col", default="feature_name")
    p.add_argument("--qv-col", default="qv")
    p.add_argument("--nucleus-col", default="overlaps_nucleus")
    p.add_argument("--is-gene-col", default="is_gene")
    p.add_argument(
        "--qv-min",
        type=float,
        default=20.0,
        help="Minimum Xenium qv to keep.",
    )
    p.add_argument(
        "--panel-genes",
        type=Path,
        default=None,
        help=(
            "Optional gene universe: one symbol per line, or tsv/csv with a "
            "'gene' column. Default: all observed genes that pass CPMI filters."
        ),
    )

    # Cell subsampling.
    p.add_argument(
        "--subsample-cells",
        type=int,
        default=50_000,
        help="Randomly keep at most N eligible cells. 0 = all eligible cells.",
    )
    p.add_argument("--seed", type=int, default=1)

    # Presence and CPMI controls.
    p.add_argument(
        "--min-occurrences",
        type=int,
        default=1,
        help="presence(cell,gene) = count >= this.",
    )
    p.add_argument("--min-det-cells", type=int, default=25)
    p.add_argument("--n-depth-bins", type=int, default=25)
    p.add_argument(
        "--depth-metric",
        choices=["total_counts", "n_genes"],
        default="total_counts",
    )
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--min-cooccur", type=int, default=2)
    p.add_argument("--min-expected-neg", type=float, default=5.0)
    p.add_argument("--max-observed-neg", type=int, default=1)
    p.add_argument("--neg-recovery-top-m", type=int, default=2500)
    p.add_argument(
        "--top-k-per-gene",
        type=int,
        default=0,
        help="Keep top K strongest absolute-CPMI partners per gene. 0 = no top-k filter.",
    )
    p.add_argument("--clip", type=float, default=4.0)
    p.add_argument("--no-clip", action="store_true")

    # Streaming controls.
    p.add_argument("--batch-size", type=int, default=8_000_000)
    p.add_argument("--consolidate-every", type=int, default=20)
    p.add_argument(
        "--max-row-groups",
        type=int,
        default=0,
        help="Cap row groups read in each pass. 0 = all. Useful for smoke tests.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stream and build the count matrix, then exit before CPMI calculation.",
    )
    return p.parse_args()


def default_out_path(args: argparse.Namespace) -> Path:
    if args.subsample_cells == 0:
        label = "all"
    elif args.subsample_cells % 1000 == 0:
        label = f"{args.subsample_cells // 1000}k"
    else:
        label = str(args.subsample_cells)
    return _BENCH_ROOT / "data" / f"whole_tissue_cpmi_{args.count_scope}_{label}.csv.gz"


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


def load_cpmi_builder():
    try:
        from tracer.conflict_reference import build_depth_corrected_reference

        return build_depth_corrected_reference
    except ImportError:
        module_path = _REPO_ROOT / "src" / "tracer" / "conflict_reference.py"
        spec = importlib.util.spec_from_file_location(
            "_tracer_conflict_reference", module_path
        )
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.build_depth_corrected_reference


def parquet_schema(args: argparse.Namespace):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(args.transcripts)
    schema_names = set(pf.schema.names)
    required = {args.cell_id_col, args.gene_col, args.qv_col}
    if args.count_scope == "nuclear":
        required.add(args.nucleus_col)
    missing = sorted(required - schema_names)
    if missing:
        raise SystemExit(
            f"Missing required parquet columns {missing}; available: {sorted(schema_names)}"
        )
    return pf, schema_names


def batch_columns(args: argparse.Namespace, schema_names: set[str]) -> list[str]:
    cols = [args.cell_id_col, args.gene_col, args.qv_col]
    if args.count_scope == "nuclear":
        cols.append(args.nucleus_col)
    if args.is_gene_col in schema_names:
        cols.append(args.is_gene_col)
    return list(dict.fromkeys(cols))


def filter_batch(
    df: pd.DataFrame,
    args: argparse.Namespace,
    panel: set[str] | None,
    *,
    has_is_gene: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if has_is_gene:
        is_gene = df[args.is_gene_col].astype("boolean").fillna(False)
        df = df[is_gene.to_numpy(dtype=bool)]

    qv = pd.to_numeric(df[args.qv_col], errors="coerce")
    keep = (qv >= args.qv_min).to_numpy(dtype=bool, copy=True)
    if args.count_scope == "nuclear":
        nuc = pd.to_numeric(df[args.nucleus_col], errors="coerce")
        keep = keep & (nuc == 1).to_numpy(dtype=bool)
    df = df[keep]

    cid = df[args.cell_id_col].astype(str)
    gene = df[args.gene_col].astype(str)
    good = ~cid.isin(UNASSIGNED_TOKENS)
    if panel is not None:
        good &= gene.isin(panel)
    return cid[good].to_numpy(dtype=object), gene[good].to_numpy(dtype=object)


def collect_sampled_cells(
    args: argparse.Namespace,
    panel: set[str] | None,
) -> set[str] | None:
    if args.subsample_cells <= 0:
        return None

    pf, schema_names = parquet_schema(args)
    cols = batch_columns(args, schema_names)
    has_is_gene = args.is_gene_col in schema_names
    candidates: set[str] = set()
    n_row_groups = pf.metadata.num_row_groups
    log(
        f"Pass 1: collecting eligible {args.count_scope} cells from "
        f"{n_row_groups:,} row groups"
    )
    for rg in range(n_row_groups):
        if args.max_row_groups and rg >= args.max_row_groups:
            log(f"  stopping cell-collection pass at {rg} row groups")
            break
        for rb in pf.iter_batches(
            batch_size=args.batch_size, columns=cols, row_groups=[rg]
        ):
            cid, _gene = filter_batch(
                rb.to_pandas(), args, panel, has_is_gene=has_is_gene
            )
            candidates.update(map(str, cid))
        if (rg + 1) % 200 == 0:
            log(f"  row groups {rg + 1:,}/{n_row_groups:,}; cells={len(candidates):,}")

    if not candidates:
        raise SystemExit("No eligible cells found for CPMI subsampling.")

    cells = np.array(sorted(candidates), dtype=object)
    if args.subsample_cells >= len(cells):
        log(f"Requested {args.subsample_cells:,} cells; using all {len(cells):,}.")
        return set(map(str, cells))

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(cells), size=args.subsample_cells, replace=False))
    sampled = set(map(str, cells[idx]))
    log(
        f"Subsampled {len(sampled):,}/{len(cells):,} eligible cells "
        f"(seed={args.seed})"
    )
    return sampled


def stream_counts(
    args: argparse.Namespace,
    panel: set[str] | None,
    keep_cells: set[str] | None,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, dict]:
    pf, schema_names = parquet_schema(args)
    cols = batch_columns(args, schema_names)
    has_is_gene = args.is_gene_col in schema_names
    total_rows = pf.metadata.num_rows
    n_row_groups = pf.metadata.num_row_groups
    log(
        f"Pass 2: streaming {total_rows:,} rows / {n_row_groups:,} row groups "
        f"from {args.transcripts.name}"
    )
    log(
        "Filter: "
        f"is_gene==True({'yes' if has_is_gene else 'col-absent -> skipped'}), "
        f"{args.qv_col}>={args.qv_min}, "
        f"{'nucleus==1' if args.count_scope == 'nuclear' else 'assigned whole-cell'}, "
        "exclude UNASSIGNED cells"
    )

    cell_to_idx: dict[str, int] = {}
    gene_to_idx: dict[str, int] = {}
    trip_keys: list[np.ndarray] = []
    trip_vals: list[np.ndarray] = []
    consolidated_keys = np.zeros(0, dtype=np.int64)
    consolidated_vals = np.zeros(0, dtype=np.int64)

    def consolidate() -> None:
        nonlocal trip_keys, trip_vals, consolidated_keys, consolidated_vals
        if not trip_keys and consolidated_keys.size == 0:
            return
        keys = (
            np.concatenate([consolidated_keys] + trip_keys)
            if trip_keys
            else consolidated_keys
        )
        vals = (
            np.concatenate([consolidated_vals] + trip_vals)
            if trip_vals
            else consolidated_vals
        )
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        vals = vals[order]
        uniq, start = np.unique(keys, return_index=True)
        summed = np.add.reduceat(vals, start) if uniq.size else vals
        consolidated_keys, consolidated_vals = uniq, summed
        trip_keys, trip_vals = [], []

    n_kept = 0
    n_batches = 0
    rg_done = 0
    for rg in range(n_row_groups):
        if args.max_row_groups and rg >= args.max_row_groups:
            log(f"  stopping aggregation pass at {rg} row groups")
            break
        for rb in pf.iter_batches(
            batch_size=args.batch_size, columns=cols, row_groups=[rg]
        ):
            cid, gene = filter_batch(
                rb.to_pandas(), args, panel, has_is_gene=has_is_gene
            )
            if keep_cells is not None:
                in_sample = pd.Series(cid, dtype="object").isin(keep_cells).to_numpy()
                cid = cid[in_sample]
                gene = gene[in_sample]
            if cid.size == 0:
                continue
            grp = pd.DataFrame({"c": cid, "g": gene})
            gc = grp.groupby(["c", "g"], sort=False).size()
            c_arr = gc.index.get_level_values(0).to_numpy()
            g_arr = gc.index.get_level_values(1).to_numpy()
            cnt = gc.to_numpy(dtype=np.int64)
            c_codes = np.fromiter(
                (cell_to_idx.setdefault(str(c), len(cell_to_idx)) for c in c_arr),
                dtype=np.int64,
                count=c_arr.size,
            )
            g_codes = np.fromiter(
                (gene_to_idx.setdefault(str(g), len(gene_to_idx)) for g in g_arr),
                dtype=np.int64,
                count=g_arr.size,
            )
            if len(gene_to_idx) >= _GENE_KEY_MULT:
                raise SystemExit(
                    f"Observed {len(gene_to_idx):,} genes, exceeding key multiplier "
                    f"{_GENE_KEY_MULT:,}."
                )
            trip_keys.append(c_codes * _GENE_KEY_MULT + g_codes)
            trip_vals.append(cnt)
            n_kept += int(cnt.sum())
            n_batches += 1
            if n_batches % args.consolidate_every == 0:
                consolidate()
        rg_done += 1
        if rg_done % 200 == 0:
            log(
                f"  row groups {rg_done:,}/{n_row_groups:,}; kept tx={n_kept:,}; "
                f"cells={len(cell_to_idx):,}; genes={len(gene_to_idx):,}"
            )
    consolidate()

    n_cells = len(cell_to_idx)
    n_genes = len(gene_to_idx)
    if n_cells == 0 or n_genes < 2:
        raise SystemExit(
            f"After filtering: {n_cells} cells, {n_genes} genes; nothing to compute."
        )
    rows = (consolidated_keys // _GENE_KEY_MULT).astype(np.int64)
    cols_ = (consolidated_keys % _GENE_KEY_MULT).astype(np.int64)
    X = sp.csr_matrix(
        (consolidated_vals.astype(np.float32), (rows, cols_)),
        shape=(n_cells, n_genes),
    )

    genes = np.empty(n_genes, dtype=object)
    for g, i in gene_to_idx.items():
        genes[i] = g
    cells = np.empty(n_cells, dtype=object)
    for c, i in cell_to_idx.items():
        cells[i] = c

    pres = (X >= args.min_occurrences)
    pres.eliminate_zeros()
    genes_per_cell = np.asarray(pres.sum(axis=1)).ravel()
    cells_per_gene = np.asarray(pres.sum(axis=0)).ravel()
    stats = {
        "count_scope": args.count_scope,
        "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "subsample_cells_requested": int(args.subsample_cells),
        "min_occurrences": int(args.min_occurrences),
        "qv_min": float(args.qv_min),
        "nnz_counts": int(X.nnz),
        "nnz_presence": int(pres.nnz),
        "kept_transcripts": int(n_kept),
        "median_present_genes_per_cell": float(np.median(genes_per_cell))
        if genes_per_cell.size
        else 0.0,
        "mean_present_genes_per_cell": float(genes_per_cell.mean())
        if genes_per_cell.size
        else 0.0,
        "genes_expressed_in_0_cells": int((cells_per_gene == 0).sum()),
        "median_cells_per_gene": float(np.median(cells_per_gene))
        if cells_per_gene.size
        else 0.0,
    }
    log(
        f"{args.count_scope} count matrix: {n_cells:,} cells x {n_genes:,} genes; "
        f"nnz={X.nnz:,}; kept tx={n_kept:,}"
    )
    log("Presence-matrix density: " + json.dumps(stats))
    return X.tocsr(), genes.astype(str), cells.astype(str), stats


def canonicalize_edges(edges: pd.DataFrame) -> pd.DataFrame:
    df = edges.copy()
    df["gene_i"] = df["gene_i"].astype(str)
    df["gene_j"] = df["gene_j"].astype(str)
    swap = df["gene_i"] > df["gene_j"]
    if swap.any():
        gi = df.loc[swap, "gene_i"].copy()
        df.loc[swap, "gene_i"] = df.loc[swap, "gene_j"].to_numpy()
        df.loc[swap, "gene_j"] = gi.to_numpy()

    dup = df.duplicated(["gene_i", "gene_j"], keep=False)
    if not dup.any():
        return df.reset_index(drop=True)

    conflicts: list[tuple[str, str]] = []
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"gene_i", "gene_j"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    for (gi, gj), grp in df.loc[dup].groupby(["gene_i", "gene_j"], sort=False):
        for col in numeric_cols:
            vals = grp[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size and (vals.max() - vals.min()) > 1e-6:
                conflicts.append((gi, gj))
                break
    if conflicts:
        preview = ", ".join(f"{a}/{b}" for a, b in conflicts[:5])
        raise SystemExit(
            f"Conflicting duplicate CPMI rows after canonicalization: {preview}"
        )
    n_dup_rows = int(dup.sum() - df.loc[dup, ["gene_i", "gene_j"]].drop_duplicates().shape[0])
    log(f"Dropping {n_dup_rows:,} exact duplicate canonical CPMI rows")
    return df.drop_duplicates(["gene_i", "gene_j"], keep="first").reset_index(drop=True)


def make_tracer_panel(edges: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    edges = canonicalize_edges(edges)
    panel = edges.rename(columns={"PMI": "raw_PMI"}).copy()
    panel["PMI"] = panel["cPMI"]
    panel["NPMI"] = panel["cNPMI"]
    panel = panel[panel["gene_i"] != panel["gene_j"]]
    finite = np.isfinite(panel["PMI"]) & np.isfinite(panel["NPMI"])
    n_drop = int((~finite).sum())
    panel = panel.loc[finite].reset_index(drop=True)
    columns = [
        "gene_i",
        "gene_j",
        "PMI",
        "NPMI",
        "cPMI",
        "cNPMI",
        "raw_PMI",
        "O",
        "E",
        "z",
    ]
    remaining = [c for c in panel.columns if c not in columns]
    return panel[columns + remaining], n_drop


def write_csv(df: pd.DataFrame, path: Path) -> None:
    compression = "gzip" if path.suffix == ".gz" or path.name.endswith(".csv.gz") else None
    df.to_csv(path, index=False, compression=compression)


def write_summary(out: Path, payload: dict) -> None:
    sp_path = Path(str(out) + ".summary.json")
    with sp_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    log(f"Wrote summary -> {sp_path}")


def main() -> int:
    args = parse_args()
    if args.out is None:
        args.out = default_out_path(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    panel_genes = load_panel_genes(args.panel_genes)
    keep_cells = collect_sampled_cells(args, panel_genes)
    X, genes, cells, density_stats = stream_counts(args, panel_genes, keep_cells)

    if args.dry_run:
        write_summary(
            args.out,
            {
                "dry_run": True,
                "transcripts": str(args.transcripts),
                "out": str(args.out),
                **density_stats,
            },
        )
        log("DRY RUN complete (count matrix assembled; CPMI not computed).")
        return 0

    build_depth_corrected_reference = load_cpmi_builder()
    clip = None if args.no_clip else float(args.clip)
    top_k = int(args.top_k_per_gene) if args.top_k_per_gene > 0 else None

    t0 = time.time()
    result = build_depth_corrected_reference(
        counts=X,
        genes=genes,
        min_count=int(args.min_occurrences),
        min_det_cells=int(args.min_det_cells),
        n_depth_bins=int(args.n_depth_bins),
        depth_metric=args.depth_metric,
        eps=float(args.eps),
        min_cooccur=int(args.min_cooccur),
        min_expected_neg=float(args.min_expected_neg),
        max_observed_neg=int(args.max_observed_neg),
        neg_recovery_top_m=int(args.neg_recovery_top_m),
        top_k_per_gene=top_k,
        clip=clip,
    )
    runtime = time.time() - t0
    panel_df, n_drop = make_tracer_panel(result.edges)
    if panel_df.empty:
        raise SystemExit("CPMI builder produced no finite edges; refusing to write panel.")

    write_csv(panel_df, args.out)
    log(f"Wrote {len(panel_df):,} CPMI panel pairs -> {args.out}")

    pos = int((panel_df["PMI"] > 0).sum())
    neg = int((panel_df["PMI"] < 0).sum())
    summary = {
        "dry_run": False,
        "transcripts": str(args.transcripts),
        "out": str(args.out),
        "serialized_weight": "PMI=cPMI; NPMI=cNPMI",
        "runtime_seconds": round(runtime, 1),
        "n_panel_pairs": int(len(panel_df)),
        "n_positive_cpmi_pairs": pos,
        "n_negative_cpmi_pairs": neg,
        "frac_positive_cpmi": round(pos / len(panel_df), 4),
        "frac_negative_cpmi": round(neg / len(panel_df), 4),
        "n_nonfinite_dropped": n_drop,
        "cpmi_median": round(float(panel_df["PMI"].median()), 4),
        "cnpmi_median": round(float(panel_df["NPMI"].median()), 4),
        "raw_pmi_median": round(float(panel_df["raw_PMI"].median()), 4),
        "builder_meta": result.meta,
        **density_stats,
    }
    write_summary(args.out, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
