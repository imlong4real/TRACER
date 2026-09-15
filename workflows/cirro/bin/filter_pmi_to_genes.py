#!/usr/bin/env python3
"""Filter a long PMI/cPMI table to an observed platform gene set.

This is a deterministic, value-preserving adapter operation: rows are kept
only when both endpoints occur in the transcript input. PMI values and all
other textual column values are preserved; no panel estimation, thresholding,
or platform-specific tuning is performed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_genes(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "feature_name" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a feature_name column")
        genes = {
            str(row["feature_name"])
            for row in reader
            if row.get("feature_name") not in (None, "")
        }
    if not genes:
        raise ValueError(f"No platform genes found in {path}")
    return genes


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pmi", required=True, type=Path)
    result.add_argument("--gene-counts", required=True, type=Path)
    result.add_argument("--out", required=True, type=Path)
    result.add_argument("--receipt", required=True, type=Path)
    result.add_argument("--source", default=None)
    result.add_argument("--source-sha256", required=True)
    result.add_argument("--chunksize", type=int, default=1_000_000)
    result.add_argument("--progress-rows", type=int, default=10_000_000)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.chunksize < 1:
        raise ValueError("--chunksize must be positive")
    platform_genes = load_genes(args.gene_counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    source_rows = 0
    effective_rows = 0
    self_pair_rows = 0
    noncanonical_pair_rows = 0
    panel_genes: set[str] = set()
    wrote_header = False
    header_columns: list[str] | None = None
    next_progress = args.progress_rows

    # mtime=0 makes the filtered derivative reproducible across runs.
    with args.out.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_output, compresslevel=6, mtime=0
        ) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as output:
                for chunk in pd.read_csv(
                    args.pmi,
                    chunksize=args.chunksize,
                    dtype=str,
                    keep_default_na=False,
                ):
                    if header_columns is None:
                        header_columns = list(chunk.columns)
                    if not {"gene_i", "gene_j"}.issubset(chunk.columns):
                        raise ValueError(
                            f"PMI table must contain gene_i and gene_j; found {list(chunk.columns)}"
                        )
                    if "PMI" not in chunk.columns and "NPMI" not in chunk.columns:
                        raise ValueError("PMI table must contain PMI or NPMI")

                    gene_i = chunk["gene_i"].astype(str)
                    gene_j = chunk["gene_j"].astype(str)
                    source_rows += len(chunk)
                    panel_genes.update(gene_i.unique())
                    panel_genes.update(gene_j.unique())
                    self_pair_rows += int((gene_i == gene_j).sum())
                    noncanonical_pair_rows += int((gene_i > gene_j).sum())

                    keep = gene_i.isin(platform_genes) & gene_j.isin(platform_genes)
                    kept = chunk.loc[keep]
                    if not kept.empty:
                        kept.to_csv(output, index=False, header=not wrote_header)
                        wrote_header = True
                        effective_rows += len(kept)
                    if args.progress_rows > 0 and source_rows >= next_progress:
                        print(
                            f"PMI overlap scan: source_rows={source_rows:,} "
                            f"effective_edges={effective_rows:,}",
                            flush=True,
                        )
                        while next_progress <= source_rows:
                            next_progress += args.progress_rows

    if not wrote_header:
        raise ValueError("PMI/platform overlap produced no gene-pair rows")

    overlap_genes = sorted(platform_genes.intersection(panel_genes))
    overlap_payload = "".join(f"{gene}\n" for gene in overlap_genes).encode("utf-8")
    metric = "PMI"
    if header_columns is not None and "PMI" not in header_columns and "NPMI" in header_columns:
        metric = "NPMI"

    receipt = {
        "filter_policy": "retain rows where gene_i and gene_j are both observed platform genes; preserve all values",
        "panel_rebuilt_or_retuned": False,
        "metric": metric,
        "source_pmi": {
            "source": args.source,
            "effective_staged_path": str(args.pmi.resolve()),
            "sha256": args.source_sha256,
            "size_bytes": args.pmi.stat().st_size,
            "pair_rows": int(source_rows),
            "gene_count": len(panel_genes),
            "self_pair_rows": int(self_pair_rows),
            "noncanonical_gene_order_rows": int(noncanonical_pair_rows),
        },
        "platform": {
            "gene_count": len(platform_genes),
            "gene_counts_path": str(args.gene_counts.resolve()),
        },
        "overlap": {
            "gene_count": len(overlap_genes),
            "genes_sha256": hashlib.sha256(overlap_payload).hexdigest(),
            "genes": overlap_genes,
        },
        "effective_pmi": {
            "path": str(args.out.resolve()),
            "sha256": sha256_file(args.out),
            "size_bytes": args.out.stat().st_size,
            "gene_pair_edge_count": int(effective_rows),
        },
    }
    write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
