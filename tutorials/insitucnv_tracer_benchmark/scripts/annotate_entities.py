#!/usr/bin/env python3
"""Annotate cells/entities from marker genes and optional raw-cell provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import UNASSIGNED_TOKENS, ensure_parent, load_yaml, require_columns, standardize_transcript_columns, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcripts", required=True, type=Path)
    p.add_argument("--entity-col", default="cell_id")
    p.add_argument("--config", required=True, type=Path, help="Panel config with marker_sets.")
    p.add_argument("--raw-annotations", type=Path, default=None, help="Optional CSV with cell_id,annotation.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--min-marker-hits", type=int, default=2)
    return p.parse_args()


def load_raw_annotation_map(path: Path | None) -> pd.Series | None:
    if path is None:
        return None
    ann = pd.read_csv(path)
    require_columns(ann, {"cell_id", "annotation"}, str(path))
    ann["cell_id"] = ann["cell_id"].astype(str)
    ann["annotation"] = ann["annotation"].astype(str)
    return ann.drop_duplicates("cell_id").set_index("cell_id")["annotation"]


def marker_scores(df: pd.DataFrame, entity_col: str, marker_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(entity_col, observed=True)["feature_name"]
    marker_sets = {k: set(map(str, v)) for k, v in marker_sets.items()}
    for entity, genes in grouped:
        counts = genes.astype(str).value_counts()
        row = {"entity_id": str(entity), "n_transcripts": int(counts.sum()), "n_genes": int(len(counts))}
        best_label = "unknown"
        best_hits = 0
        best_score = 0.0
        for label, markers in marker_sets.items():
            hits = int(counts.index.isin(markers).sum())
            score = float(counts.loc[counts.index.intersection(markers)].sum()) / max(float(counts.sum()), 1.0)
            row[f"marker_hits_{label}"] = hits
            row[f"marker_score_{label}"] = score
            if (hits, score) > (best_hits, best_score):
                best_label, best_hits, best_score = label, hits, score
        row["marker_annotation"] = best_label if best_hits > 0 else "unknown"
        row["marker_hits_best"] = best_hits
        row["marker_score_best"] = best_score
        rows.append(row)
    return pd.DataFrame(rows)


def majority_raw_annotation(df: pd.DataFrame, entity_col: str, raw_map: pd.Series) -> pd.DataFrame:
    if "cell_id" not in df.columns:
        return pd.DataFrame(columns=["entity_id", "majority_raw_annotation", "raw_annotation_fraction"])
    tmp = df[[entity_col, "cell_id"]].copy()
    tmp["raw_annotation"] = tmp["cell_id"].astype(str).map(raw_map)
    tmp = tmp.dropna(subset=["raw_annotation"])
    if tmp.empty:
        return pd.DataFrame(columns=["entity_id", "majority_raw_annotation", "raw_annotation_fraction"])
    counts = tmp.groupby([entity_col, "raw_annotation"], observed=True).size().rename("n").reset_index()
    totals = counts.groupby(entity_col, observed=True)["n"].sum()
    idx = counts.sort_values(["n", "raw_annotation"], ascending=[False, True]).groupby(entity_col, observed=True).head(1)
    idx["raw_annotation_fraction"] = idx["n"] / idx[entity_col].map(totals)
    return idx.rename(columns={entity_col: "entity_id", "raw_annotation": "majority_raw_annotation"})[
        ["entity_id", "majority_raw_annotation", "raw_annotation_fraction"]
    ].assign(entity_id=lambda x: x["entity_id"].astype(str))


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    marker_sets = cfg.get("marker_sets") or {}
    if not marker_sets:
        raise SystemExit(f"{args.config} has no marker_sets")
    df = standardize_transcript_columns(pd.read_parquet(args.transcripts))
    require_columns(df, {args.entity_col, "feature_name"}, "transcripts")
    df[args.entity_col] = df[args.entity_col].astype(str)
    df = df.loc[~df[args.entity_col].isin(UNASSIGNED_TOKENS)].copy()

    out = marker_scores(df, args.entity_col, marker_sets)
    raw_map = load_raw_annotation_map(args.raw_annotations)
    if raw_map is not None:
        out = out.merge(majority_raw_annotation(df, args.entity_col, raw_map), on="entity_id", how="left")
    out["annotation"] = np.where(
        out["marker_hits_best"] >= args.min_marker_hits,
        out["marker_annotation"],
        out.get("majority_raw_annotation", "unknown"),
    )
    out["annotation"] = pd.Series(out["annotation"]).fillna("unknown")
    ensure_parent(args.out)
    out.to_csv(args.out, index=False)
    write_json(args.out.with_suffix(args.out.suffix + ".summary.json"), {"entities": len(out), "annotations": out["annotation"].value_counts().to_dict()})
    print(f"Wrote {len(out):,} entity annotations to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

