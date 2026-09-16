#!/usr/bin/env python3
"""Create marker-preserving, chromosome-balanced gene panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import chromosome_rank, ensure_parent, load_yaml, read_gene_positions, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--genes", required=True, type=Path, help="One gene per line, or transcript parquet.")
    p.add_argument("--gene-positions", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def read_gene_universe(path: Path) -> tuple[set[str], pd.Series | None]:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=["feature_name"])
        counts = df["feature_name"].astype(str).value_counts()
        return set(counts.index), counts
    genes = [g.strip() for g in path.read_text().splitlines() if g.strip()]
    return set(genes), None


def flatten_markers(config: dict) -> list[str]:
    out: list[str] = []
    for genes in (config.get("marker_sets") or {}).values():
        out.extend(str(g) for g in genes)
    out.extend(str(g) for g in config.get("priority_genes") or [])
    return list(dict.fromkeys(out))


def chromosome_balanced_fill(
    candidates: pd.DataFrame,
    n_needed: int,
    counts: pd.Series | None,
    rng: np.random.Generator,
) -> list[str]:
    if n_needed <= 0:
        return []
    work = candidates.copy()
    if counts is not None:
        work["count"] = work["gene"].map(counts).fillna(0).astype(float)
    else:
        work["count"] = 1.0
    work = work.sort_values(["chromosome", "count", "gene"], ascending=[True, False, True])
    chroms = sorted(work["chromosome"].unique(), key=chromosome_rank)
    selected: list[str] = []
    by_chrom = {c: list(work.loc[work["chromosome"] == c, "gene"]) for c in chroms}
    while len(selected) < n_needed:
        progressed = False
        for chrom in chroms:
            genes = by_chrom[chrom]
            if genes:
                selected.append(genes.pop(0))
                progressed = True
                if len(selected) >= n_needed:
                    break
        if not progressed:
            break
    if len(selected) < n_needed:
        remaining = sorted(set(work["gene"]) - set(selected))
        rng.shuffle(remaining)
        selected.extend(remaining[: n_needed - len(selected)])
    return selected[:n_needed]


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    universe, counts = read_gene_universe(args.genes)
    positions = read_gene_positions(args.gene_positions)
    positions = positions.loc[positions["gene"].isin(universe)].copy()
    rng = np.random.default_rng(args.seed)

    markers = [g for g in flatten_markers(cfg) if g in set(positions["gene"])]
    panel_sizes = cfg.get("panel_sizes") or ["full", 2000, 1000, 500]
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary = []
    full = sorted(set(positions["gene"]))
    for size in panel_sizes:
        if str(size) == "full":
            panel = full
            out_name = "panel_full.txt"
        else:
            n = int(size)
            base = list(dict.fromkeys(markers))
            base = base[:n]
            remaining = positions.loc[~positions["gene"].isin(base)].copy()
            fill = chromosome_balanced_fill(remaining, n - len(base), counts, rng)
            panel = list(dict.fromkeys(base + fill))[:n]
            out_name = f"panel_{n}.txt"
        out = args.outdir / out_name
        ensure_parent(out)
        out.write_text("\n".join(panel) + "\n")
        summary.append({"panel": out_name, "requested_size": size, "n_genes": len(panel), "n_markers": len(set(panel) & set(markers))})

    pd.DataFrame(summary).to_csv(args.outdir / "panel_summary.csv", index=False)
    write_json(args.outdir / "panel_summary.json", {"panels": summary, "n_universe_positioned": len(full)})
    print(f"Wrote {len(summary)} panels to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

