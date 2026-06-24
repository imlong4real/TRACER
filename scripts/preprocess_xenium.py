#!/usr/bin/env python3
"""Standardize a Xenium-style transcripts parquet for TRACER + benchmark methods.

Reads a 10x Xenium ``transcripts.parquet`` (or Xenium5k, or Atera), validates
the schema, optionally filters by QC and removes control probes, and writes a
standardized parquet whose columns are::

    x, y, z, feature_name, cell_id, transcript_id, qv, overlaps_nucleus

The script never drops unassigned transcripts unless ``--drop-unassigned`` is
explicitly passed. ``cell_id`` is cast to ``str`` and sentinel labels
("UNASSIGNED", negative integers, blank, etc.) are mapped to the canonical
unassigned token ``"-1"``.

Two-pass streaming design (memory-safe for 16 GB+ Atera inputs):

  Pass 1 — scan ``feature_name`` [+ ``is_gene``] [+ ``qv``]; collect the gene
           set surviving the QC filter and (if requested) control-probe
           regex; collect per-gene transcript counts.
  Pass 2 — stream rows, rename coords (``x_location → x`` etc.) to the
           canonical schema, filter, write a new parquet via PyArrow.

Reading uses fastparquet (avoids the "Repetition level histogram size
mismatch" error pyarrow raises on some Xenium-produced files). Writing uses
pyarrow.parquet.ParquetWriter.

Outputs (next to ``--out`` or under ``--summary-dir``):

  <out>                                  standardized parquet
  preprocessing_summary.json             counts, filter steps, schema
  gene_counts.tsv                        gene → kept_transcript_count
  transcript_assignment_summary.tsv      assigned vs unassigned breakdown
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from fastparquet import ParquetFile as FastParquetFile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COORD_ALIASES: dict[str, str] = {
    "x_location": "x",
    "y_location": "y",
    "z_location": "z",
}

# Default control-probe regex (applies when the input lacks an ``is_gene``
# column). Covers 10x Xenium / Xenium5k / Atera naming patterns.
DEFAULT_CONTROL_REGEX = (
    r"^(?:Neg|BLANK|Blank|Unassigned|Deprecated|Control"
    r"|antisense_|UnassignedCodeword_"
    r"|NegControlProbe_|NegControlCodeword_)"
)

# Sentinels that should collapse to canonical "-1" in cell_id after str-cast.
UNASSIGNED_SENTINELS = frozenset({
    "UNASSIGNED", "Unassigned", "unassigned", "DROP", "DROP_",
    "nan", "None", "", "0", "-1", "NA", "<NA>",
})

CANONICAL_COLUMNS = (
    "x", "y", "z", "feature_name", "cell_id", "transcript_id", "qv",
    "overlaps_nucleus",
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str, *, flush: bool = True) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=flush)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Input transcripts parquet (Xenium/Xenium5k/Atera).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output path for standardized parquet.")
    p.add_argument("--qv-min", type=float, default=None,
                   help="Minimum qv (rows kept where qv > this). Skipped if "
                        "the input has no qv column.")
    p.add_argument("--remove-control-probes", action="store_true",
                   help="Drop control probes (regex + is_gene if present).")
    p.add_argument("--control-regex", default=DEFAULT_CONTROL_REGEX,
                   help="Regex applied to feature_name when "
                        "--remove-control-probes is set.")
    p.add_argument("--platform", default=None,
                   choices=(None, "xenium", "xenium5k", "atera", "auto"),
                   help="Optional platform preset; currently informational "
                        "(written into preprocessing_summary.json).")
    p.add_argument("--drop-unassigned", action="store_true",
                   help="Drop transcripts whose cell_id is unassigned. "
                        "DEFAULT IS FALSE — unassigned transcripts are kept.")
    p.add_argument("--summary-dir", type=Path, default=None,
                   help="Where to write summary JSON/TSV (default: --out parent).")
    p.add_argument("--row-group-progress", type=int, default=50,
                   help="Log every N row groups during streaming.")
    p.add_argument("--dry-run", action="store_true",
                   help="Inspect schema, run pass 1, print what would be "
                        "written. Do NOT write parquet/summaries.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------
@dataclass
class InputSchema:
    """What the input has vs. what's canonical."""
    available: list[str]
    coord_cols: dict[str, str]              # canonical → input column name
    has_feature_name: bool
    has_cell_id: bool
    has_transcript_id: bool
    has_qv: bool
    has_overlaps_nucleus: bool
    has_is_gene: bool
    has_z: bool

    def report(self) -> dict[str, Any]:
        return {
            "input_columns": self.available,
            "canonical_to_input_map": self.coord_cols,
            "has_z": self.has_z,
            "has_transcript_id": self.has_transcript_id,
            "has_qv": self.has_qv,
            "has_overlaps_nucleus": self.has_overlaps_nucleus,
            "has_is_gene": self.has_is_gene,
        }


def detect_schema(columns: list[str]) -> InputSchema:
    available = list(columns)

    # Resolve coordinate columns. Accept canonical (x/y/z) or Xenium-style
    # (x_location/y_location/z_location). If neither present, fail loud.
    coord_cols: dict[str, str] = {}
    for canon, xen in (("x", "x_location"), ("y", "y_location"), ("z", "z_location")):
        if canon in available:
            coord_cols[canon] = canon
        elif xen in available:
            coord_cols[canon] = xen
    if "x" not in coord_cols or "y" not in coord_cols:
        raise SystemExit(
            f"Input parquet must have x/y coordinates (canonical or "
            f"Xenium-style x_location/y_location). Found columns: {available}"
        )
    has_z = "z" in coord_cols

    if "feature_name" not in available:
        raise SystemExit(
            f"Input parquet must have 'feature_name'. Found: {available}"
        )
    if "cell_id" not in available:
        raise SystemExit(
            f"Input parquet must have 'cell_id'. Found: {available}"
        )
    return InputSchema(
        available=available,
        coord_cols=coord_cols,
        has_feature_name=True,
        has_cell_id=True,
        has_transcript_id="transcript_id" in available,
        has_qv="qv" in available,
        has_overlaps_nucleus="overlaps_nucleus" in available,
        has_is_gene="is_gene" in available,
        has_z=has_z,
    )


# ---------------------------------------------------------------------------
# Pass 1 — collect gene set + per-gene counts (memory-safe streaming)
# ---------------------------------------------------------------------------
def _qv_mask(chunk: pd.DataFrame, qv_min: float | None) -> pd.Series:
    if qv_min is None or "qv" not in chunk.columns:
        return pd.Series(True, index=chunk.index)
    return chunk["qv"] > qv_min


def _control_mask(chunk: pd.DataFrame, *, remove_control: bool,
                  control_regex: re.Pattern, has_is_gene: bool) -> pd.Series:
    """Returns mask of rows to KEEP (i.e. False = drop as control)."""
    if not remove_control:
        return pd.Series(True, index=chunk.index)
    keep = pd.Series(True, index=chunk.index)
    if has_is_gene and "is_gene" in chunk.columns:
        keep &= chunk["is_gene"].astype(bool)
    if "feature_name" in chunk.columns:
        keep &= ~chunk["feature_name"].astype(str).str.match(control_regex)
    return keep


def pass1_collect(
    fp: FastParquetFile,
    *,
    qv_min: float | None,
    remove_control: bool,
    control_regex: re.Pattern,
    has_is_gene: bool,
    progress_every: int,
) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    """Scan feature_name [+ qv] [+ is_gene] [+ cell_id] to collect:
      - per-kept-gene transcript count
      - per-control-gene transcript count (informational)
      - overall counts dict (assigned/unassigned/total before+after filter)
    """
    scan_cols = ["feature_name", "cell_id"]
    if qv_min is not None and "qv" in fp.columns:
        scan_cols.append("qv")
    if has_is_gene:
        scan_cols.append("is_gene")

    rg_seen = 0
    n_rg = len(fp.row_groups)
    total_rows = 0
    rows_passing_qv = 0
    rows_after_control = 0
    assigned_in = 0
    assigned_out = 0
    gene_counts: dict[str, int] = {}
    control_counts: dict[str, int] = {}
    t0 = time.time()
    log(f"[pass 1] scanning {n_rg} row groups; columns={scan_cols}")

    for chunk in fp.iter_row_groups(columns=scan_cols):
        rg_seen += 1
        if chunk is None or chunk.empty:
            continue
        total_rows += len(chunk)
        cid_is_assigned = ~chunk["cell_id"].astype(str).isin(UNASSIGNED_SENTINELS)
        assigned_in += int(cid_is_assigned.sum())

        qv_keep = _qv_mask(chunk, qv_min)
        rows_passing_qv += int(qv_keep.sum())

        ctrl_keep = _control_mask(
            chunk, remove_control=remove_control,
            control_regex=control_regex, has_is_gene=has_is_gene,
        )
        rows_after_control += int((qv_keep & ctrl_keep).sum())

        # gene counts on rows that survive BOTH filters
        kept = chunk.loc[qv_keep & ctrl_keep, "feature_name"].astype(str)
        for g, n in kept.value_counts().items():
            gene_counts[g] = gene_counts.get(g, 0) + int(n)
        # informational: count control probes dropped (only when remove_control)
        if remove_control:
            ctrl = chunk.loc[qv_keep & ~ctrl_keep, "feature_name"].astype(str)
            for g, n in ctrl.value_counts().items():
                control_counts[g] = control_counts.get(g, 0) + int(n)
        # assigned-after counts the surviving rows that were assigned in input
        assigned_out += int((cid_is_assigned & qv_keep & ctrl_keep).sum())

        if rg_seen % progress_every == 0:
            log(f"[pass 1]  rg {rg_seen}/{n_rg}: scanned {total_rows:,}, "
                f"kept {rows_after_control:,} ({time.time()-t0:.1f}s)")

    log(f"[pass 1] done — {rg_seen} row groups in {time.time()-t0:.1f}s")
    counts = {
        "total_rows": total_rows,
        "rows_after_qv": rows_passing_qv,
        "rows_after_control_probes": rows_after_control,
        "assigned_in": assigned_in,
        "assigned_after_filters": assigned_out,
        "unassigned_in": total_rows - assigned_in,
        "n_genes_kept": len(gene_counts),
        "n_genes_dropped_as_control": len(control_counts),
    }
    return gene_counts, control_counts, counts


# ---------------------------------------------------------------------------
# Pass 2 — stream-filter, rename, write
# ---------------------------------------------------------------------------
def _standardize_cell_id(s: pd.Series) -> pd.Series:
    """Cast to str; map sentinel / negative-int unassigned labels → ``-1``."""
    if pd.api.types.is_integer_dtype(s):
        out = pd.Series(np.where(s < 0, "-1", s.astype(str)),
                        index=s.index, name="cell_id")
    else:
        out = s.astype(str)
        out = out.where(~out.isin(UNASSIGNED_SENTINELS), "-1")
    return out


def _standardize_chunk(
    chunk: pd.DataFrame,
    *,
    schema: InputSchema,
    qv_min: float | None,
    remove_control: bool,
    control_regex: re.Pattern,
    drop_unassigned: bool,
    next_transcript_id_start: int,
) -> tuple[pd.DataFrame, int]:
    """Apply filters + rename + standardize one chunk."""
    keep = _qv_mask(chunk, qv_min) & _control_mask(
        chunk, remove_control=remove_control,
        control_regex=control_regex, has_is_gene=schema.has_is_gene,
    )
    if not keep.any():
        return chunk.iloc[0:0], next_transcript_id_start
    chunk = chunk.loc[keep].copy()

    # Rename coords
    coord_rename = {v: k for k, v in schema.coord_cols.items() if k != v}
    if coord_rename:
        chunk = chunk.rename(columns=coord_rename)

    # Ensure canonical column set
    if "z" not in chunk.columns:
        chunk["z"] = np.float32(0.0)
    else:
        chunk["z"] = chunk["z"].astype(np.float32)
    chunk["x"] = chunk["x"].astype(np.float32)
    chunk["y"] = chunk["y"].astype(np.float32)
    chunk["feature_name"] = chunk["feature_name"].astype(str)

    chunk["cell_id"] = _standardize_cell_id(chunk["cell_id"])
    if drop_unassigned:
        chunk = chunk.loc[chunk["cell_id"] != "-1"].copy()

    if "transcript_id" not in chunk.columns:
        n = len(chunk)
        chunk["transcript_id"] = np.arange(
            next_transcript_id_start, next_transcript_id_start + n,
            dtype=np.int64,
        )
        next_transcript_id_start += n
    else:
        try:
            chunk["transcript_id"] = chunk["transcript_id"].astype(np.int64)
        except (OverflowError, TypeError):
            chunk["transcript_id"] = chunk["transcript_id"].astype(str)

    if "qv" in chunk.columns:
        chunk["qv"] = chunk["qv"].astype(np.float32)

    if "overlaps_nucleus" in chunk.columns:
        chunk["overlaps_nucleus"] = chunk["overlaps_nucleus"].astype(np.uint8)
    else:
        chunk["overlaps_nucleus"] = np.uint8(0)

    out_cols = [c for c in CANONICAL_COLUMNS if c in chunk.columns]
    return chunk[out_cols].reset_index(drop=True), next_transcript_id_start


def pass2_write(
    fp: FastParquetFile,
    out_path: Path,
    *,
    schema: InputSchema,
    qv_min: float | None,
    remove_control: bool,
    control_regex: re.Pattern,
    drop_unassigned: bool,
    progress_every: int,
) -> dict[str, int]:
    rg_seen = 0
    n_rg = len(fp.row_groups)
    total_written = 0
    total_assigned = 0
    next_tid = 0
    writer: pq.ParquetWriter | None = None
    t0 = time.time()
    log(f"[pass 2] streaming to {out_path}")

    try:
        for chunk in fp.iter_row_groups():
            rg_seen += 1
            if chunk is None or chunk.empty:
                continue
            std, next_tid = _standardize_chunk(
                chunk, schema=schema, qv_min=qv_min,
                remove_control=remove_control, control_regex=control_regex,
                drop_unassigned=drop_unassigned,
                next_transcript_id_start=next_tid,
            )
            if std.empty:
                continue
            tbl = pa.Table.from_pandas(std, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), tbl.schema,
                                          compression="snappy")
            writer.write_table(tbl)
            total_written += len(std)
            total_assigned += int((std["cell_id"] != "-1").sum())
            if rg_seen % progress_every == 0:
                log(f"[pass 2]  rg {rg_seen}/{n_rg}: written {total_written:,} "
                    f"({time.time()-t0:.1f}s)")
    finally:
        if writer is not None:
            writer.close()
    log(f"[pass 2] done — wrote {total_written:,} rows in {time.time()-t0:.1f}s")
    return {
        "rows_written": total_written,
        "rows_assigned_after": total_assigned,
        "rows_unassigned_after": total_written - total_assigned,
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def write_summaries(
    summary_dir: Path,
    *,
    args: argparse.Namespace,
    schema: InputSchema,
    pass1_counts: dict[str, Any],
    pass2_counts: dict[str, int] | None,
    gene_counts: dict[str, int],
    control_counts: dict[str, int],
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)

    # gene_counts.tsv (kept genes only, sorted)
    gc = (
        pd.DataFrame(
            sorted(gene_counts.items(), key=lambda kv: -kv[1]),
            columns=["feature_name", "n_transcripts"],
        )
        .reset_index(drop=True)
    )
    gc.to_csv(summary_dir / "gene_counts.tsv", sep="\t", index=False)

    # transcript_assignment_summary.tsv
    rows = [
        ("total_input_rows", pass1_counts["total_rows"]),
        ("assigned_input", pass1_counts["assigned_in"]),
        ("unassigned_input", pass1_counts["unassigned_in"]),
        ("rows_after_qv", pass1_counts["rows_after_qv"]),
        ("rows_after_control_filter", pass1_counts["rows_after_control_probes"]),
        ("n_genes_kept", pass1_counts["n_genes_kept"]),
        ("n_genes_dropped_as_control", pass1_counts["n_genes_dropped_as_control"]),
    ]
    if pass2_counts is not None:
        rows.extend([
            ("rows_written", pass2_counts["rows_written"]),
            ("assigned_after_filters", pass2_counts["rows_assigned_after"]),
            ("unassigned_after_filters", pass2_counts["rows_unassigned_after"]),
        ])
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(
        summary_dir / "transcript_assignment_summary.tsv",
        sep="\t", index=False,
    )

    # preprocessing_summary.json
    summary = {
        "input": str(args.input),
        "out": str(args.out) if not args.dry_run else None,
        "dry_run": bool(args.dry_run),
        "platform": args.platform,
        "qv_min": args.qv_min,
        "remove_control_probes": bool(args.remove_control_probes),
        "control_regex": args.control_regex,
        "drop_unassigned": bool(args.drop_unassigned),
        "schema": schema.report(),
        "canonical_columns_written": [
            c for c in CANONICAL_COLUMNS
            if (c in ("x", "y", "feature_name", "cell_id"))
            or (c == "z")
            or (c == "transcript_id")
            or (c == "qv" and schema.has_qv)
            or (c == "overlaps_nucleus")
        ],
        "counts": {
            "pass1": pass1_counts,
            "pass2": pass2_counts,
        },
        "top_kept_genes": [
            {"feature_name": g, "n_transcripts": int(n)}
            for g, n in sorted(gene_counts.items(),
                               key=lambda kv: -kv[1])[:20]
        ],
        "top_control_genes_dropped": [
            {"feature_name": g, "n_transcripts": int(n)}
            for g, n in sorted(control_counts.items(),
                               key=lambda kv: -kv[1])[:20]
        ],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(summary_dir / "preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary_dir = args.summary_dir or args.out.parent

    control_regex = re.compile(args.control_regex)
    log(f"Input  : {args.input} ({args.input.stat().st_size/1e9:.2f} GB)")
    log(f"Output : {args.out}")
    log(f"qv-min : {args.qv_min!r}; remove-control: {args.remove_control_probes}; "
        f"keep-unassigned: {not args.drop_unassigned}; dry-run: {args.dry_run}")

    fp = FastParquetFile(str(args.input))
    schema = detect_schema(fp.columns)
    log(f"Schema : {schema.report()}")

    gene_counts, control_counts, pass1_counts = pass1_collect(
        fp,
        qv_min=args.qv_min,
        remove_control=args.remove_control_probes,
        control_regex=control_regex,
        has_is_gene=schema.has_is_gene,
        progress_every=args.row_group_progress,
    )

    log(f"Pass 1 result: total={pass1_counts['total_rows']:,}  "
        f"after_qv={pass1_counts['rows_after_qv']:,}  "
        f"after_ctrl={pass1_counts['rows_after_control_probes']:,}  "
        f"assigned_in={pass1_counts['assigned_in']:,}  "
        f"unassigned_in={pass1_counts['unassigned_in']:,}  "
        f"genes_kept={pass1_counts['n_genes_kept']}  "
        f"genes_dropped_ctrl={pass1_counts['n_genes_dropped_as_control']}")

    pass2_counts = None
    if not args.dry_run:
        pass2_counts = pass2_write(
            fp, args.out,
            schema=schema,
            qv_min=args.qv_min,
            remove_control=args.remove_control_probes,
            control_regex=control_regex,
            drop_unassigned=args.drop_unassigned,
            progress_every=args.row_group_progress,
        )

    write_summaries(
        summary_dir,
        args=args,
        schema=schema,
        pass1_counts=pass1_counts,
        pass2_counts=pass2_counts,
        gene_counts=gene_counts,
        control_counts=control_counts,
    )
    log(f"Summaries → {summary_dir}/{{preprocessing_summary.json, "
        f"gene_counts.tsv, transcript_assignment_summary.tsv}}")

    if args.dry_run:
        log("DRY RUN — no parquet written.")
    else:
        out_pq = pq.ParquetFile(str(args.out))
        log(f"Output  : {args.out} — {out_pq.metadata.num_rows:,} rows, "
            f"cols={list(out_pq.schema_arrow.names)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
