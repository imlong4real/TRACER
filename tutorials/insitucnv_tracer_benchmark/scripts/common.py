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


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = ensure_parent(path)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")

