#!/usr/bin/env python3
"""Build a gene-position table for only the genes present in an input dataset."""

from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from common import ensure_parent, normalize_chromosome, write_json


DEFAULT_OUT = Path("tutorials/insitucnv_tracer_benchmark/data/gene_positions_grch38.tsv")
GENE_COLUMN_CANDIDATES = ("gene", "gene_name", "feature_name", "symbol")
STANDARD_CHROMOSOMES = {*(str(i) for i in range(1, 23)), "X", "Y", "MT"}
JSON_GENE_CONTEXT_KEYS = {
    "gene",
    "genes",
    "gene_panel",
    "gene_panels",
    "feature",
    "features",
    "target",
    "targets",
    "panel",
    "payload",
}
H5_FEATURE_NAME_PATHS = (
    "matrix/features/name",
    "features/name",
    "gene_names",
    "genes",
    "feature_names",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--genes-from",
        required=True,
        type=Path,
        help="Gene list, CSV/TSV, 10x gene_panel.json, transcript parquet, or 10x matrix h5.",
    )
    p.add_argument(
        "--gene-column",
        default="feature_name",
        help="Column to read from parquet/CSV inputs. Defaults to Xenium transcript feature_name.",
    )
    p.add_argument("--gtf", type=Path, default=None, help="Local GTF or GTF.GZ annotation file.")
    p.add_argument(
        "--gtf-url",
        default=None,
        help="Optional URL to download if --gtf is absent or points to a missing file.",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def normalize_gene(value: object) -> str | None:
    if value is None:
        return None
    gene = str(value).strip()
    if not gene or gene.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return gene


def unique_genes(values: list[object]) -> list[str]:
    genes: list[str] = []
    seen: set[str] = set()
    for value in values:
        gene = normalize_gene(value)
        if gene is None:
            continue
        key = gene.upper()
        if key in seen:
            continue
        seen.add(key)
        genes.append(gene)
    return genes


def read_gene_list(path: Path) -> list[str]:
    genes: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.replace(",", "\t").split("\t", maxsplit=1)[0].strip()
            if token.lower() in set(GENE_COLUMN_CANDIDATES):
                continue
            genes.append(token)
    return unique_genes(genes)


def read_delimited_genes(path: Path, gene_column: str) -> list[str]:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(path, sep=sep)
    column = choose_gene_column(df.columns, gene_column, str(path))
    values = df[column].tolist()
    if len(df.columns) == 1 and str(column).lower() not in set(GENE_COLUMN_CANDIDATES):
        values.insert(0, column)
    return unique_genes(values)


def read_parquet_genes(path: Path, gene_column: str) -> list[str]:
    try:
        df = pd.read_parquet(path, columns=[gene_column])
        column = gene_column
    except Exception:
        df = pd.read_parquet(path)
        column = choose_gene_column(df.columns, gene_column, str(path))
    return unique_genes(df[column].tolist())


def choose_gene_column(columns: pd.Index | list[str], preferred: str, label: str) -> str:
    column_set = set(columns)
    if preferred in column_set:
        return preferred
    for candidate in GENE_COLUMN_CANDIDATES:
        if candidate in column_set:
            return candidate
    if len(columns) == 1:
        return list(columns)[0]
    raise SystemExit(
        f"{label} has no gene column {preferred!r}; tried {GENE_COLUMN_CANDIDATES}; "
        f"present={list(columns)}"
    )


def read_json_genes(path: Path) -> list[str]:
    with path.open() as handle:
        payload = json.load(handle)
    return unique_genes(collect_json_genes(payload))


def collect_json_genes(node: object, in_gene_context: bool = False) -> list[object]:
    genes: list[object] = []
    if isinstance(node, dict):
        lowered = {str(key).lower(): key for key in node}
        for candidate in GENE_COLUMN_CANDIDATES:
            original_key = lowered.get(candidate)
            if original_key is not None:
                genes.append(node[original_key])
                break
        if in_gene_context and "name" in lowered and any(
            key in lowered for key in ("id", "gene_id", "ensembl_id", "feature_id")
        ):
            genes.append(node[lowered["name"]])
        for key, value in node.items():
            next_context = in_gene_context or str(key).lower() in JSON_GENE_CONTEXT_KEYS
            genes.extend(collect_json_genes(value, in_gene_context=next_context))
    elif isinstance(node, list):
        if all(isinstance(value, str) for value in node):
            genes.extend(node)
        else:
            for value in node:
                genes.extend(collect_json_genes(value, in_gene_context=in_gene_context))
    return genes


def read_h5_genes(path: Path) -> list[str]:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(
            "Reading H5 gene lists requires h5py; use the benchmark conda env."
        ) from exc

    with h5py.File(path, "r") as handle:
        for dataset_path in H5_FEATURE_NAME_PATHS:
            if dataset_path in handle:
                values = decode_h5_values(handle[dataset_path][()])
                return unique_genes(values)
    raise SystemExit(
        f"Could not find feature names in {path}; tried H5 datasets {list(H5_FEATURE_NAME_PATHS)}"
    )


def decode_h5_values(values) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode())
        else:
            decoded.append(str(value))
    return decoded


def read_input_genes(path: Path, gene_column: str) -> list[str]:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".json":
        return read_json_genes(path)
    if suffix in {".h5", ".hdf5"}:
        return read_h5_genes(path)
    if suffix == ".parquet":
        return read_parquet_genes(path, gene_column)
    if suffix in {".csv", ".tsv", ".tab"}:
        return read_delimited_genes(path, gene_column)
    if suffix in {".txt", ".list"} or "genes" in name:
        return read_gene_list(path)
    raise SystemExit(
        f"Unsupported --genes-from type for {path}; expected list/CSV/TSV/parquet/json/h5."
    )


def resolve_gtf(gtf: Path | None, gtf_url: str | None, out: Path) -> Path:
    if gtf is not None and gtf.exists():
        return gtf
    if gtf_url is None:
        if gtf is None:
            raise SystemExit("Provide --gtf or --gtf-url.")
        raise SystemExit(f"Missing --gtf file: {gtf}")

    if gtf is None:
        parsed = urlparse(gtf_url)
        filename = Path(parsed.path).name or "annotation.gtf.gz"
        gtf = out.parent / filename

    ensure_parent(gtf)
    print(f"Downloading GTF from {gtf_url} to {gtf}", flush=True)
    urllib.request.urlretrieve(gtf_url, gtf)
    return gtf


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def parse_gtf_attributes(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for field in raw.rstrip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        if " " in field:
            key, value = field.split(" ", maxsplit=1)
        elif "=" in field:
            key, value = field.split("=", maxsplit=1)
        else:
            continue
        attrs[key.strip()] = value.strip().strip('"')
    return attrs


def parse_matching_gtf(gtf: Path, genes: list[str]) -> pd.DataFrame:
    gene_keys = {gene.upper() for gene in genes}
    rows: list[dict[str, object]] = []
    with open_text(gtf) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = parse_gtf_attributes(parts[8])
            gene_name = attrs.get("gene_name") or attrs.get("gene") or attrs.get("Name")
            if gene_name is None or gene_name.upper() not in gene_keys:
                continue
            rows.append(
                {
                    "gtf_gene": gene_name,
                    "gene_id": attrs.get("gene_id", ""),
                    "chromosome": normalize_chromosome(parts[0]),
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "strand": parts[6],
                    "gene_type": attrs.get("gene_type") or attrs.get("gene_biotype", ""),
                    "source": parts[1],
                }
            )
    return pd.DataFrame(rows)


def rank_annotation_rows(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["_chrom_rank"] = ranked["chromosome"].map(chromosome_sort_key)
    ranked["_is_standard"] = ranked["chromosome"].map(is_standard_chromosome)
    ranked["_is_protein_coding"] = ranked["gene_type"].eq("protein_coding")
    return ranked.sort_values(
        ["_is_standard", "_is_protein_coding", "_chrom_rank", "start", "end", "gene_id"],
        ascending=[False, False, True, True, True, True],
        kind="mergesort",
    )


def is_standard_chromosome(chromosome: str) -> bool:
    chrom = normalize_chromosome(chromosome)
    return chrom in STANDARD_CHROMOSOMES


def chromosome_sort_key(chromosome: str) -> tuple[int, int | str]:
    chrom = normalize_chromosome(chromosome)
    try:
        return (0, int(chrom))
    except ValueError:
        if chrom == "X":
            return (1, 23)
        if chrom == "Y":
            return (1, 24)
        if chrom == "MT":
            return (1, 25)
        return (2, chrom)


def build_gene_positions(
    genes: list[str], gtf_rows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if gtf_rows.empty:
        return (
            pd.DataFrame(
                columns=[
                    "gene",
                    "gene_id",
                    "chromosome",
                    "start",
                    "end",
                    "strand",
                    "gene_type",
                    "source",
                ]
            ),
            pd.DataFrame(),
            genes,
        )

    ranked = rank_annotation_rows(gtf_rows)
    ranked["_gene_key"] = ranked["gtf_gene"].str.upper()
    selected = ranked.drop_duplicates("_gene_key", keep="first").set_index("_gene_key")
    output_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for gene in genes:
        key = gene.upper()
        if key not in selected.index:
            missing.append(gene)
            continue
        row = selected.loc[key]
        output_rows.append(
            {
                "gene": gene,
                "gene_id": row["gene_id"],
                "chromosome": row["chromosome"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "strand": row["strand"],
                "gene_type": row["gene_type"],
                "source": row["source"],
            }
        )
    duplicate_rows = duplicate_symbol_rows(gtf_rows)
    return pd.DataFrame(output_rows), duplicate_rows, missing


def duplicate_symbol_rows(gtf_rows: pd.DataFrame) -> pd.DataFrame:
    if gtf_rows.empty:
        return pd.DataFrame()
    duplicate_mask = (
        gtf_rows.groupby(gtf_rows["gtf_gene"].str.upper())["gtf_gene"].transform("size")
        > 1
    )
    duplicates = gtf_rows.loc[duplicate_mask].copy()
    if duplicates.empty:
        return duplicates
    duplicates = rank_annotation_rows(duplicates).drop(
        columns=["_chrom_rank", "_is_standard", "_is_protein_coding"]
    )
    return duplicates.rename(columns={"gtf_gene": "gene"})


def sidecar_path(out: Path, suffix: str) -> Path:
    return out.with_suffix(out.suffix + suffix)


def write_outputs(
    args: argparse.Namespace,
    positions: pd.DataFrame,
    duplicates: pd.DataFrame,
    missing: list[str],
    summary: dict,
) -> None:
    ensure_parent(args.out)
    positions.to_csv(args.out, sep="\t", index=False)
    write_json(sidecar_path(args.out, ".summary.json"), summary)

    missing_out = sidecar_path(args.out, ".missing_genes.txt")
    if missing:
        with missing_out.open("w") as handle:
            for gene in missing:
                handle.write(f"{gene}\n")
    elif missing_out.exists():
        missing_out.unlink()

    duplicate_out = sidecar_path(args.out, ".duplicate_gene_symbols.tsv")
    if not duplicates.empty:
        duplicates.to_csv(duplicate_out, sep="\t", index=False)
    elif duplicate_out.exists():
        duplicate_out.unlink()


def main() -> int:
    args = parse_args()
    genes = read_input_genes(args.genes_from, args.gene_column)
    if not genes:
        raise SystemExit(f"No genes found in {args.genes_from}")

    gtf = resolve_gtf(args.gtf, args.gtf_url, args.out)
    gtf_rows = parse_matching_gtf(gtf, genes)
    positions, duplicates, missing = build_gene_positions(genes, gtf_rows)

    duplicate_symbol_count = (
        int(duplicates["gene"].str.upper().nunique()) if not duplicates.empty else 0
    )
    summary = {
        "genes_from": str(args.genes_from),
        "gene_column": args.gene_column,
        "gtf": str(gtf),
        "gtf_url": args.gtf_url,
        "out": str(args.out),
        "input_gene_count": len(genes),
        "matched_count": int(len(positions)),
        "missing_count": int(len(missing)),
        "duplicate_symbol_count": duplicate_symbol_count,
        "missing_genes": str(sidecar_path(args.out, ".missing_genes.txt")) if missing else None,
        "duplicate_gene_symbols": str(sidecar_path(args.out, ".duplicate_gene_symbols.tsv"))
        if duplicate_symbol_count
        else None,
    }
    write_outputs(args, positions, duplicates, missing, summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {len(positions):,} gene positions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
