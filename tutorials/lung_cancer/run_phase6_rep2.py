#!/usr/bin/env python3
"""Run Phase 6 reassignment on df_finetuned_rep2.parquet and compare results.

Usage:
  python tutorials/lung_cancer/run_phase6_rep2.py [--seed 42]

This script will:
 - load `tutorials/lung_cancer/output/df_finetuned_rep2.parquet`
 - run `reassign_unassigned_to_nearby_entities_fast` (Phase 6) from `src/tracer/core.py`
 - save the updated file back to `df_finetuned_rep2.parquet`
 - compare exact (bitwise) match rates for `cell_id_stitched`, `cell_id_finetuned`, and
   `cell_id_finetuned_2` between `df_finetuned.parquet` and `df_finetuned_rep2.parquet`.
"""
from __future__ import annotations
import argparse
import importlib.util
import time
from pathlib import Path
import pandas as pd


def load_core_module() -> object:
    core_path = Path("src/tracer/core.py")
    spec = importlib.util.spec_from_file_location("tracer_core_local", str(core_path))
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def run_phase6_on_rep2(seed: int = 42) -> None:
    core = load_core_module()
    core.set_reproducibility_seed(seed)

    out_dir = Path("tutorials/lung_cancer/output")
    fp_a = out_dir / "df_finetuned.parquet"
    fp_b = out_dir / "df_finetuned_rep2.parquet"

    if not fp_b.exists():
        raise FileNotFoundError(f"Missing file: {fp_b}")

    print(f"Loading: {fp_b}")
    df = pd.read_parquet(fp_b)
    print(f"Rows: {len(df):,}")

    print("Running Phase 6 reassignment...")
    t0 = time.time()
    df_out, n_reassigned, stats = core.reassign_unassigned_to_nearby_entities_fast(
        df,
        entity_summary=None,
        entity_col="cell_id_finetuned",
        gene_col="feature_name",
        coord_cols=("x", "y", "z"),
        out_col="cell_id_finetuned_2",
        dist_threshold=20.0,
        only_partial_component=False,
        show_progress=False,
    )
    dt = time.time() - t0
    print(f"Phase 6 done: reassigned={n_reassigned:,} transcripts. Took {dt:.1f}s")
    print("stats:", stats)

    print(f"Saving updated file -> {fp_b}")
    df_out.to_parquet(fp_b, index=False)

    # Compare columns
    print("\nComparing columns between df_finetuned.parquet and df_finetuned_rep2.parquet")
    cols = ["cell_id_stitched", "cell_id_finetuned", "cell_id_finetuned_2"]
    a = pd.read_parquet(fp_a)
    b = pd.read_parquet(fp_b)

    for col in cols:
        a_has = col in a.columns
        b_has = col in b.columns
        if not (a_has and b_has):
            print(f"{col}: missing in A?{not a_has} B?{not b_has}")
            continue
        arr_a = a[col].astype(str).to_numpy()
        arr_b = b[col].astype(str).to_numpy()
        if arr_a.shape != arr_b.shape:
            print(f"{col}: length mismatch {arr_a.shape} vs {arr_b.shape}")
            continue
        total = arr_a.shape[0]
        same = (arr_a == arr_b)
        n_same = int(same.sum())
        pct = 100.0 * n_same / total if total > 0 else 0.0
        print(f"{col}: total={total:,}, matches={n_same:,}, match%={pct:.6f}")
        if n_same < total:
            idxs = (~same).nonzero()[0][:5]
            print(" sample mismatches:")
            for i in idxs:
                print(f"  row {i}: A={arr_a[i]}  B={arr_b[i]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_phase6_on_rep2(seed=args.seed)


if __name__ == "__main__":
    main()
