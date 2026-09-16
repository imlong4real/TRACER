#!/usr/bin/env python3
"""Acceptance gate for a TRACER gene-pair panel: per-seed positive support.

This is the exact check from the GBM over-pruning postmortem, where a too-sparse
panel left ~38% of prune seeds with ZERO panel edges (median 0 positive pairs/
seed) and cells collapsed after Prune. TRACER's ``prune_transcripts_nuclear_seed``
anchors each cell on its NUCLEAR gene set and admits transcripts by positive
panel weight to that seed, SKIPPING structurally-absent pairs -- so what matters
is not global panel positivity but whether individual seeds have enough positive
within-seed edges.

For each ROI cell it takes the nuclear gene set (``overlaps_nucleus==1``,
``qv>=--qv-min``, ``is_gene``), then counts within-set gene pairs whose selected
panel metric is > 0. For CPMI panels this is usually ``PMI``, where the builder
serialized cPMI into the historical TRACER weight column. Cells with fewer than
``--min-nuclear-genes`` unique nuclear genes are reported separately (TRACER
falls back to a whole-cell prune for those).

Runs in a plain scientific-Python env (pandas + pyarrow); it does NOT import
tracer, so it can validate a freshly-built panel BEFORE the expensive TRACER run.

EXAMPLE
=======
::

    python check_panel_seed_support.py \\
      --panel   tutorials/insitucnv_tracer_benchmark/data/whole_tissue_cpmi_nuclear_50k.csv.gz \\
      --transcripts tutorials/insitucnv_tracer_benchmark/data/roi_transcripts.parquet \\
      --qv-min 20

GO/NO-GO
========
Healthy (like the collab panel that fixed GBM): frac_seeds_zero_support near 0,
median_positive_pairs_per_seed in the hundreds. Danger (like the sparse GBM
spatial panel): frac_seeds_zero_support ~0.38, median 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from common import UNASSIGNED_TOKENS
except Exception:
    UNASSIGNED_TOKENS = frozenset(
        {"UNASSIGNED", "Unassigned", "unassigned", "DROP", "nan", "None",
         "", "0", "-1", "NA", "<NA>"}
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", required=True, type=Path, help="TRACER panel csv(.gz): gene_i,gene_j,PMI[,NPMI].")
    p.add_argument("--transcripts", required=True, type=Path, help="ROI transcript parquet.")
    p.add_argument("--metric-col", default="PMI", help="Panel column used as the edge weight. CPMI panels serialize cPMI as PMI.")
    p.add_argument("--qv-min", type=float, default=20.0)
    p.add_argument("--min-nuclear-genes", type=int, default=3,
                   help="Seeds with fewer unique nuclear genes fall back to whole-cell prune in TRACER.")
    p.add_argument("--cell-id-col", default="cell_id")
    p.add_argument("--gene-col", default="feature_name")
    p.add_argument("--qv-col", default="qv")
    p.add_argument("--nucleus-col", default="overlaps_nucleus")
    p.add_argument("--is-gene-col", default="is_gene")
    p.add_argument("--out", type=Path, default=None, help="Optional summary json path.")
    return p.parse_args()


def load_positive_adjacency(panel_path: Path, metric_col: str) -> dict[str, set[str]]:
    df = pd.read_csv(panel_path)
    if not {"gene_i", "gene_j"}.issubset(df.columns):
        raise SystemExit(f"Panel missing gene_i/gene_j; columns={list(df.columns)}")
    col = metric_col if metric_col in df.columns else ("NPMI" if "NPMI" in df.columns else None)
    if col is None:
        raise SystemExit(f"Panel has neither {metric_col} nor NPMI; columns={list(df.columns)}")
    pos = df[np.isfinite(df[col]) & (df[col] > 0)]
    adj: dict[str, set[str]] = {}
    gi = pos["gene_i"].astype(str).to_numpy()
    gj = pos["gene_j"].astype(str).to_numpy()
    for a, b in zip(gi, gj):
        if a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)   # panels are one-direction; symmetrize
    print(f"Panel: {len(df):,} pairs; {len(pos):,} with {col}>0; {len(adj):,} genes have a positive partner")
    return adj


def nuclear_gene_sets(args: argparse.Namespace) -> dict[str, set[str]]:
    import pyarrow.parquet as pq
    cols = [args.cell_id_col, args.gene_col, args.qv_col, args.nucleus_col]
    pf = pq.ParquetFile(args.transcripts)
    if args.is_gene_col in pf.schema.names:
        cols.append(args.is_gene_col)
    frames = []
    for rb in pf.iter_batches(batch_size=8_000_000, columns=cols):
        df = rb.to_pandas()
        if args.is_gene_col in df.columns:
            df = df[df[args.is_gene_col].astype("boolean").fillna(False).to_numpy(dtype=bool)]
        qv = pd.to_numeric(df[args.qv_col], errors="coerce")
        nuc = pd.to_numeric(df[args.nucleus_col], errors="coerce")
        keep = (qv >= args.qv_min).to_numpy(dtype=bool, copy=True) & (nuc == 1).to_numpy(dtype=bool)
        df = df[keep]
        cid = df[args.cell_id_col].astype(str)
        good = ~cid.isin(UNASSIGNED_TOKENS)
        frames.append(pd.DataFrame({"c": cid[good].to_numpy(),
                                    "g": df[args.gene_col].astype(str)[good].to_numpy()}))
    allnuc = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["c", "g"])
    sets = allnuc.groupby("c")["g"].agg(set)
    return sets.to_dict()


def count_positive_pairs(gene_set: set[str], adj: dict[str, set[str]]) -> int:
    edges = 0
    for g in gene_set:
        partners = adj.get(g)
        if partners:
            edges += len(partners & gene_set)
    return edges // 2   # each undirected edge counted twice


def main() -> int:
    args = parse_args()
    adj = load_positive_adjacency(args.panel, args.metric_col)
    seeds = nuclear_gene_sets(args)
    if not seeds:
        raise SystemExit("No nuclear transcripts survived filtering; cannot assess seed support.")

    n_nuc_genes = np.array([len(s) for s in seeds.values()])
    is_seed = n_nuc_genes >= args.min_nuclear_genes
    pos_pairs = np.array([count_positive_pairs(s, adj) if len(s) >= args.min_nuclear_genes else -1
                          for s in seeds.values()])
    seed_pp = pos_pairs[is_seed]

    summary = {
        "panel": str(args.panel),
        "transcripts": str(args.transcripts),
        "metric_col": args.metric_col,
        "qv_min": args.qv_min,
        "min_nuclear_genes": args.min_nuclear_genes,
        "n_cells_with_nuclear_tx": int(len(seeds)),
        "frac_cells_below_min_nuclear_genes": round(float((~is_seed).mean()), 4),
        "n_seeds": int(is_seed.sum()),
        "median_nuclear_genes_per_seed": float(np.median(n_nuc_genes[is_seed])) if is_seed.any() else 0.0,
        "median_positive_pairs_per_seed": float(np.median(seed_pp)) if seed_pp.size else 0.0,
        "mean_positive_pairs_per_seed": float(seed_pp.mean()) if seed_pp.size else 0.0,
        "frac_seeds_zero_support": round(float((seed_pp == 0).mean()), 4) if seed_pp.size else 1.0,
        "p10_positive_pairs_per_seed": float(np.percentile(seed_pp, 10)) if seed_pp.size else 0.0,
    }
    verdict = ("HEALTHY" if summary["frac_seeds_zero_support"] < 0.05
               and summary["median_positive_pairs_per_seed"] >= 5 else "CHECK")
    summary["verdict"] = verdict
    print(json.dumps(summary, indent=2))
    print(f"\nVERDICT: {verdict}  "
          f"(zero-support seeds={summary['frac_seeds_zero_support']:.1%}, "
          f"median positive pairs/seed={summary['median_positive_pairs_per_seed']:.0f})")
    if verdict != "HEALTHY":
        print("  -> Panel may starve TRACER's nuclear-seed prune (cf. GBM collapse). "
              "Consider lowering --qv-min / --min-occurrences or adding cells when building the panel.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            json.dump(summary, fh, indent=2, default=str)
            fh.write("\n")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
