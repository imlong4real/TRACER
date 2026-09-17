#!/usr/bin/env python3
"""Shared helpers for the WIP InSituCNV/TRACER benchmark scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


UNASSIGNED_TOKENS = frozenset(
    {"UNASSIGNED", "Unassigned", "unassigned", "DROP", "nan", "None", "", "0", "-1", "NA", "<NA>"}
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a small YAML config, falling back to JSON syntax if PyYAML is absent."""
    path = Path(path)
    try:
        import yaml
    except ImportError:
        with path.open() as handle:
            return json.load(handle)
    with path.open() as handle:
        data = yaml.safe_load(handle)
    return data or {}


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def standardize_transcript_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common Xenium coordinate/id column variants."""
    rename = {}
    for src, dst in (
        ("x_location", "x"),
        ("y_location", "y"),
        ("z_location", "z"),
        ("feature_name", "feature_name"),
        ("cell_id", "cell_id"),
    ):
        if src in df.columns and dst not in df.columns:
            rename[src] = dst
    if rename:
        df = df.rename(columns=rename)
    if "z" not in df.columns:
        df["z"] = 0.0
    if "transcript_id" not in df.columns:
        df["transcript_id"] = range(len(df))
    if "overlaps_nucleus" not in df.columns:
        df["overlaps_nucleus"] = 0
    return df


def require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise SystemExit(f"{label} missing required columns {sorted(missing)}; present={list(df.columns)}")


def normalize_chromosome(value: object) -> str:
    chrom = str(value).strip()
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    if chrom == "M":
        chrom = "MT"
    return chrom


def chromosome_rank(value: object) -> tuple[int, int | str]:
    chrom = normalize_chromosome(value)
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


def read_gene_positions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    if "gene" not in df.columns and "gene_name" in df.columns:
        df = df.rename(columns={"gene_name": "gene"})
    require_columns(df, {"gene", "chromosome", "start", "end"}, str(path))
    df["gene"] = df["gene"].astype(str)
    df["chromosome"] = df["chromosome"].map(normalize_chromosome)
    return df.drop_duplicates("gene", keep="first")


def cnv_bin_chromosomes(adata) -> tuple[Any, Any]:
    """Return ``(values, chrom_label_per_column)`` for an infercnvpy result.

    infercnvpy writes its CNV estimate to ``adata.obsm["X_cnv"]`` -- a
    cells x genomic-BIN matrix, NOT cells x genes -- and records where each
    chromosome starts along that axis in ``adata.uns["cnv"]["chr_pos"]``
    (chromosome -> first column index). On this pipeline's output X_cnv is
    (n_cells, 354) against 4,968 genes, so the bin axis cannot be indexed with
    ``adata.var``.

    Callers previously looked for a ``layers["gene_values_cnv"]``, which
    nothing in this pipeline produces. Both chromosome summaries therefore
    returned empty silently: no ``chrom_cnv_by_compartment.csv`` was written,
    which is the input ``compare_reference_condition.py`` requires, and the
    per-clone chromosome heatmap was skipped with a one-line notice.

    Returns ``(None, None)`` when the CNV matrix or its chromosome map is
    absent, so callers can degrade as before rather than raise.
    """
    import numpy as np
    import scipy.sparse as sp

    X = adata.obsm.get("X_cnv") if hasattr(adata, "obsm") else None
    chr_pos = (adata.uns.get("cnv") or {}).get("chr_pos") if hasattr(adata, "uns") else None
    if X is None or not chr_pos:
        return None, None
    X = X.toarray() if sp.issparse(X) else np.asarray(X)

    # chr_pos gives only the START column of each chromosome; expand to a label
    # per column by taking each chromosome's span up to the next start.
    items = sorted(((str(k), int(v)) for k, v in dict(chr_pos).items()), key=lambda kv: kv[1])
    labels = np.empty(X.shape[1], dtype=object)
    for i, (name, start) in enumerate(items):
        stop = items[i + 1][1] if i + 1 < len(items) else X.shape[1]
        labels[start:stop] = name
    return X, labels


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")

