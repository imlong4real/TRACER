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
    p.add_argument("--rank-by", choices=("score", "hits"), default="score",
                   help="Pick the winning marker set by expression fraction "
                        "(default) or by distinct-hit count (legacy; biased "
                        "toward larger marker sets).")
    return p.parse_args()


def load_raw_annotation_map(path: Path | None) -> pd.Series | None:
    if path is None:
        return None
    ann = pd.read_csv(path)
    require_columns(ann, {"cell_id", "annotation"}, str(path))
    ann["cell_id"] = ann["cell_id"].astype(str)
    ann["annotation"] = ann["annotation"].astype(str)
    return ann.drop_duplicates("cell_id").set_index("cell_id")["annotation"]


def marker_scores(df: pd.DataFrame, entity_col: str, marker_sets: dict[str, list[str]],
                  rank_by: str = "score") -> pd.DataFrame:
    """Score each entity against every marker set and take the winner.

    `rank_by` decides the winner:

    ``score``  fraction of the entity's transcripts belonging to the set.
    ``hits``   number of DISTINCT marker genes detected (the original rule).

    Ranking on `hits` is biased by marker-set SIZE, because a larger set can
    accumulate more distinct hits from ambient/low-level reads alone. On the
    ovary ROI that let a 4-gene epithelial set outscore a 5-gene fibroblast
    set for true fibroblasts, and reference recall was 28%. Ranking on the
    expression fraction instead -- the quantity that actually discriminates --
    takes it to 84%. Set `rank_by="hits"` to reproduce the old behaviour.
    """
    if rank_by not in {"score", "hits"}:
        raise SystemExit(f"rank_by must be 'score' or 'hits', got {rank_by!r}")
    rows = []
    grouped = df.groupby(entity_col, observed=True)["feature_name"]
    marker_sets = {k: set(map(str, v)) for k, v in marker_sets.items()}
    for entity, genes in grouped:
        counts = genes.astype(str).value_counts()
        row = {"entity_id": str(entity), "n_transcripts": int(counts.sum()), "n_genes": int(len(counts))}
        per_label = {}
        for label, markers in marker_sets.items():
            hits = int(counts.index.isin(markers).sum())
            score = float(counts.loc[counts.index.intersection(markers)].sum()) / max(float(counts.sum()), 1.0)
            row[f"marker_hits_{label}"] = hits
            row[f"marker_score_{label}"] = score
            per_label[label] = (hits, score)
        # Ties break on the other quantity, then on label for determinism.
        def key(item):
            label, (hits, score) = item
            return (score, hits, label) if rank_by == "score" else (hits, score, label)
        best_label, (best_hits, best_score) = max(per_label.items(), key=key)
        keep = best_score > 0 if rank_by == "score" else best_hits > 0
        row["marker_annotation"] = best_label if keep else "unknown"
        row["marker_hits_best"] = best_hits
        row["marker_score_best"] = best_score
        rows.append(row)
    return pd.DataFrame(rows)


def majority_raw_annotation(df: pd.DataFrame, entity_col: str, raw_map: pd.Series) -> pd.DataFrame:
    if "cell_id" not in df.columns:
        return pd.DataFrame(columns=["entity_id", "majority_raw_annotation", "raw_annotation_fraction"])
    # Dedupe: on the raw arm `entity_col` IS "cell_id", and selecting it twice
    # yields duplicate columns, so `tmp["cell_id"]` returns a DataFrame whose
    # .map() rejects a dict. That made --raw-annotations unusable for the raw
    # arm while working fine for TRACER entities ("stitched").
    tmp = df[list(dict.fromkeys([entity_col, "cell_id"]))].copy()
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

    out = marker_scores(df, args.entity_col, marker_sets, rank_by=args.rank_by)
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

