from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STAGE_COLUMNS = [
    ("Original", "cell_id"),
    ("Stage 1", "cell_id_npmi_cons_p2"),
    ("Stage 2", "cell_id_final"),
    ("Stage 3 (TRACER stitched)", "cell_id_stitched"),
    ("Stage 4", "cell_id_spatial"),
    ("Stage 5 (TRACER finetuned)", "cell_id_finetuned"),
]

DATASETS = [
    (
        "Breast cancer Xenium",
        Path("tutorials/breast_cancer/output/df_finetuned_run4.parquet"),
        25,
    ),
    (
        "Lung cancer Xenium",
        Path("tutorials/lung_cancer/output/df_finetuned_5um.parquet"),
        10,
    ),
    (
        "Mouse ileum MERFISH",
        Path("tutorials/mouse_ileum/output/df_finetuned.parquet"),
        10,
    ),
]


def infer_entity_type(entity_id: str) -> str:
    """
    Returns one of: 'cell', 'partial', 'component', 'drop', 'unknown'
    """
    if entity_id is None or (isinstance(entity_id, float) and np.isnan(entity_id)):
        return "unknown"
    s = str(entity_id)
    # Interpreting requested logic as intended equality checks.
    if s == "DROP" or s == "UNASSIGNED" or s == "-1":
        return "drop"
    if s.startswith("UNASSIGNED_"):
        return "pseudocell"
    if "-" in s:
        return "partial_cell"
    return "whole_cell"


def _parquet_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema.names)


def compute_counts() -> pd.DataFrame:
    rows: list[dict] = []

    for dataset, parquet_path, min_comp_size in DATASETS:
        available_cols = _parquet_columns(parquet_path)
        required_cols = [col for _, col in STAGE_COLUMNS if col in available_cols]
        df = pd.read_parquet(parquet_path, columns=required_cols)

        for stage_name, col in STAGE_COLUMNS:
            if col not in df.columns:
                rows.append(
                    {
                        "dataset": dataset,
                        "min_comp_size": min_comp_size,
                        "stage": stage_name,
                        "column": col,
                        "unassigned": np.nan,
                        "whole_cell": np.nan,
                        "partial_cell": np.nan,
                        "pseudocell": np.nan,
                        "unknown": np.nan,
                        "total_transcripts": np.nan,
                        "note": "column_missing",
                    }
                )
                continue

            types = df[col].map(infer_entity_type)
            value_counts = types.value_counts(dropna=False)

            rows.append(
                {
                    "dataset": dataset,
                    "min_comp_size": min_comp_size,
                    "stage": stage_name,
                    "column": col,
                    "unassigned": int(value_counts.get("drop", 0)),
                    "whole_cell": int(value_counts.get("whole_cell", 0)),
                    "partial_cell": int(value_counts.get("partial_cell", 0)),
                    "pseudocell": int(value_counts.get("pseudocell", 0)),
                    "unknown": int(value_counts.get("unknown", 0)),
                    "total_transcripts": int(len(df)),
                    "note": "",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (stored in output metadata only).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("tutorials/entity_type_counts_all_stages.csv"),
        help="Path to write the summary table as CSV.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    result = compute_counts()
    result.to_csv(args.output_csv, index=False)

    print(f"Seed: {args.seed}")
    print(f"Saved: {args.output_csv}")
    print(result.to_csv(index=False))


if __name__ == "__main__":
    main()
