#!/usr/bin/env python3
"""Generate a nucleus-based NPMI panel for a GBM Xenium transcript parquet.

Updated for the 2026-05-27 TRACER refactor: the old point-estimate
``tracer.metrics.compute_npmi`` was retired in favor of the bootstrapped
``tracer.metrics.compute_pmi_bootstrap`` (active-sampler bootstrap with
per-pair CIs). This script keeps the GBM "confident nuclei" recipe but now
delegates the nucleus restriction, sentinel-context exclusion, and the
per-cell transcript-count percentile band to ``compute_pmi_bootstrap``'s
built-in pre-filters:

    - nuclear_only=True, nucleus_col="overlaps_nucleus"   (was: == 1 mask)
    - exclude_contexts default {-1, UNASSIGNED, DROP, ...} (was: EXCLUDE_IDS)
    - percentile_filter=(low_pct, high_pct)               (was: np.percentile band)

Only the Xenium ``qv`` quality cut has no built-in equivalent, so it stays
here as an explicit pre-step.

The output is a long-format panel CSV with columns ``gene_i, gene_j, PMI,
NPMI`` (one direction per pair; self-pairs dropped) — exactly what
``run_gbm.py`` / ``run_segmented_pipeline`` consume.

EXAMPLE
=======
::

    python tutorials/gbm/generate_npmi.py \\
      --input  tutorials/gbm/data/transcripts.parquet \\
      --output tutorials/gbm/data/gbm_npmi.csv \\
      --qv-min 30 --low-pct 20 --high-pct 80 --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED_COLUMNS = {"feature_name", "cell_id", "qv", "overlaps_nucleus"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a GBM NPMI panel from a transcript parquet.")
    parser.add_argument("--input", required=True, help="Transcript parquet path.")
    parser.add_argument("--output", required=True, help="Output CSV path for the NPMI panel.")
    parser.add_argument("--qv-min", type=float, default=30.0, help="Minimum Xenium qv to keep.")
    parser.add_argument("--low-pct", type=float, default=20.0, help="Lower percentile for confident nuclei.")
    parser.add_argument("--high-pct", type=float, default=80.0, help="Upper percentile for confident nuclei.")
    parser.add_argument(
        "--min-occurrences-per-context",
        type=int,
        default=2,
        help="Minimum copies of a gene in a nucleus before it counts as present.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Bootstrap RNG seed (reproducibility).")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Forward show_progress=True to compute_pmi_bootstrap.",
    )
    return parser.parse_args()


def _validate_columns(df, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def main() -> None:
    args = _parse_args()

    import numpy as np
    import pandas as pd

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    from tracer.metrics import compute_pmi_bootstrap

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_parquet(input_path)
    _validate_columns(df, REQUIRED_COLUMNS)

    df = df.copy()
    df["feature_name"] = df["feature_name"].astype(str).str.strip()
    df["cell_id"] = df["cell_id"].astype(str)
    df["qv"] = pd.to_numeric(df["qv"], errors="coerce")

    # qv cut: the only filter without a compute_pmi_bootstrap equivalent.
    n_in = len(df)
    df = df[(df["qv"] >= args.qv_min) & (df["feature_name"] != "")].copy()
    if df.empty:
        raise ValueError("No transcripts remain after qv / empty-gene filtering.")

    # compute_pmi_bootstrap applies, in order: sentinel-context exclusion
    # (default {-1, UNASSIGNED, DROP, nan, None, ""}), nuclear-only
    # restriction, then the per-cell tx-count percentile band — i.e. the
    # GBM "confident nuclei" selection, vectorized inside the library.
    result = compute_pmi_bootstrap(
        df,
        group_key="cell_id",
        feature_col="feature_name",
        count_col=None,  # one row per transcript; presence built by groupby
        min_occurrences_per_context=args.min_occurrences_per_context,
        nuclear_only=True,
        nucleus_col="overlaps_nucleus",
        percentile_filter=(args.low_pct, args.high_pct),
        metric="pmi",
        seed=args.seed,
        persist_ci=True,  # required to populate result.pair_ci
        show_progress=args.show_progress,
    )

    ci = result.pair_ci
    if ci is None or ci.empty:
        raise ValueError(
            "compute_pmi_bootstrap returned no settled pairs — check qv/percentile "
            "filters and that the panel has enough nucleus-overlapping transcripts."
        )

    # pair_ci stores gene indices into result.genes (the function may reorder
    # its internal gene set), so map indices -> names via result.genes. Use
    # the full-data point estimates (legacy_pmi / legacy_npmi); per the
    # docstring, the bootstrap CI bounds are biased for magnitude/ranking.
    genes = np.asarray(result.genes, dtype=str)
    i_idx = ci["gene_i_idx"].to_numpy(dtype=np.int64)
    j_idx = ci["gene_j_idx"].to_numpy(dtype=np.int64)
    panel = pd.DataFrame(
        {
            "gene_i": genes[i_idx],
            "gene_j": genes[j_idx],
            "PMI": ci["legacy_pmi"].to_numpy(dtype=np.float64),
            "NPMI": ci["legacy_npmi"].to_numpy(dtype=np.float64),
        }
    )
    panel = panel[panel["gene_i"] != panel["gene_j"]].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)

    print(f"Input transcripts: {n_in:,}")
    print(f"After qv>={args.qv_min} / non-empty gene: {len(df):,}")
    print(f"Genes in result: {len(result.genes):,}")
    print(f"NPMI pairs written: {len(panel):,}")
    print(f"Saved NPMI panel to: {output_path}")


if __name__ == "__main__":
    main()
