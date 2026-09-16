#!/usr/bin/env python3
"""Compare downsampled CNV runs against full-depth reference runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_parent, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--reference-run-id", required=True)
    p.add_argument("--outdir", required=True, type=Path)
    return p.parse_args()


def load_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def load_chrom(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "compartment" not in df.columns:
        return pd.DataFrame()
    return df.set_index("compartment")


def vector_corr(a: pd.Series, b: pd.Series) -> float:
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return float("nan")
    x = a.loc[common].astype(float).to_numpy()
    y = b.loc[common].astype(float).to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def arm_paths(row: pd.Series) -> dict[str, Path]:
    return {
        "raw": Path(row["insitucnv_raw_outdir"]),
        "tracer_whole": Path(row["insitucnv_tracer_whole_outdir"]),
        "tracer_all": Path(row["insitucnv_tracer_all_outdir"]),
    }


def summarize_arm(run_id: str, row: pd.Series, arm: str, outdir: Path, ref_chrom: pd.DataFrame | None) -> dict:
    stats = load_stats(outdir / "arm_stats.json")
    chrom = load_chrom(outdir / "chrom_cnv_by_compartment.csv")
    out = {
        "run_id": run_id,
        "arm": arm,
        "panel_size": row.get("panel_size"),
        "detection_fraction": row.get("detection_fraction"),
        "n_entities": stats.get("n_entities"),
        "n_genes": stats.get("n_genes"),
        "median_transcripts": stats.get("median_transcripts"),
        "median_genes": stats.get("median_genes"),
        "reference_flatness": np.nan,
        "tumor_signal_abs_mean": np.nan,
        "tumor_reference_cnv_corr_to_full": np.nan,
    }
    if not chrom.empty:
        numeric = chrom.drop(columns=[c for c in chrom.columns if c == "compartment"], errors="ignore")
        if "reference" in numeric.index:
            out["reference_flatness"] = float(numeric.loc["reference"].astype(float).std())
        if "tumor" in numeric.index:
            out["tumor_signal_abs_mean"] = float(numeric.loc["tumor"].astype(float).abs().mean())
        if ref_chrom is not None and "tumor" in numeric.index and "tumor" in ref_chrom.index:
            out["tumor_reference_cnv_corr_to_full"] = vector_corr(numeric.loc["tumor"], ref_chrom.loc["tumor"])
    return out


def main() -> int:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.reference_run_id not in set(manifest["run_id"].astype(str)):
        raise SystemExit(f"reference run {args.reference_run_id!r} not found in {args.manifest}")
    manifest["run_id"] = manifest["run_id"].astype(str)
    ref_row = manifest.loc[manifest["run_id"] == args.reference_run_id].iloc[0]
    ref_chroms = {
        arm: load_chrom(path / "chrom_cnv_by_compartment.csv")
        for arm, path in arm_paths(ref_row).items()
    }

    rows = []
    for _, row in manifest.iterrows():
        for arm, path in arm_paths(row).items():
            rows.append(summarize_arm(str(row["run_id"]), row, arm, path, ref_chroms.get(arm)))
    out = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.outdir / "degradation_summary.csv", index=False)
    write_json(
        args.outdir / "degradation_summary.json",
        {
            "reference_run_id": args.reference_run_id,
            "n_rows": len(out),
            "arms": sorted(out["arm"].dropna().unique()),
        },
    )
    print(f"Wrote degradation summary to {args.outdir / 'degradation_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

