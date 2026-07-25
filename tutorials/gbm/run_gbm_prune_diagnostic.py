#!/usr/bin/env python3
"""Run only TRACER's nuclear-seed Prune stage for a GBM piece."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tutorials.gbm.run_gbm import load_npmi_panel, load_transcripts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--npmi", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="JSON summary path (default: <output-stem>_summary.json).",
    )
    parser.add_argument(
        "--cell-report-output",
        type=Path,
        default=None,
        help="Per-original-cell CSV path (default: <output-stem>_cells.csv).",
    )
    parser.add_argument("--platform", default="xenium")
    parser.add_argument("--user-config", type=Path, default=None)
    parser.add_argument("--pmi-threshold", type=float, default=None)
    parser.add_argument("--seed-coherence-floor", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--comparison-panel",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional PMI panel to compare against; may be repeated.",
    )
    return parser.parse_args()


def _companion_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}{suffix}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _panel_seed_support(df: pd.DataFrame, aux: dict) -> pd.DataFrame:
    """Count accepted PMI edges represented in each original nuclear seed."""
    assigned = df.loc[df["cell_id"].astype(str) != "-1"].copy()
    assigned["_cell_str"] = assigned["cell_id"].astype(str)
    assigned["_is_nuc"] = assigned["overlaps_nucleus"].astype(bool)

    base = assigned.groupby("_cell_str", sort=False, observed=True).agg(
        original_transcripts=("_cell_str", "size"),
        nuclear_transcripts=("_is_nuc", "sum"),
    )

    gene_to_idx = aux["gene_to_idx"]
    nuclear_genes = assigned.loc[
        assigned["_is_nuc"], ["_cell_str", "feature_name"]
    ].copy()
    nuclear_genes["_gene_idx"] = nuclear_genes["feature_name"].astype(str).map(gene_to_idx)
    nuclear_genes = nuclear_genes.dropna(subset=["_gene_idx"]).drop_duplicates(
        ["_cell_str", "_gene_idx"]
    )
    nuclear_genes["_gene_idx"] = nuclear_genes["_gene_idx"].astype(np.int32)

    W = aux["W"].tocsr()
    support_rows = []
    for cell_id, group in nuclear_genes.groupby("_cell_str", sort=False, observed=True):
        gene_ids = group["_gene_idx"].to_numpy(dtype=np.int32)
        support_rows.append(
            (
                str(cell_id),
                int(len(gene_ids)),
                int(W[gene_ids][:, gene_ids].nnz) if len(gene_ids) > 1 else 0,
            )
        )
    support = pd.DataFrame(
        support_rows,
        columns=[
            "_cell_str",
            "nuclear_genes_in_panel",
            "accepted_nuclear_gene_pairs",
        ],
    ).set_index("_cell_str")
    base = base.join(support, how="left")
    base[["nuclear_genes_in_panel", "accepted_nuclear_gene_pairs"]] = base[
        ["nuclear_genes_in_panel", "accepted_nuclear_gene_pairs"]
    ].fillna(0).astype(np.int64)
    return base


def _build_cell_report(df_out: pd.DataFrame, aux: dict) -> pd.DataFrame:
    report = _panel_seed_support(df_out, aux)
    assigned = df_out.loc[df_out["cell_id"].astype(str) != "-1"].copy()
    assigned["_cell_str"] = assigned["cell_id"].astype(str)
    assigned["_label_str"] = assigned["tracer_id"].astype(str)
    assigned["_is_main"] = assigned["_label_str"] == assigned["_cell_str"]
    assigned["_is_partial"] = assigned["_etype"].astype(str) == "partial"
    assigned["_is_unassigned"] = assigned["_label_str"] == "-1"
    outcome = assigned.groupby("_cell_str", sort=False, observed=True).agg(
        main_transcripts=("_is_main", "sum"),
        partial_transcripts=("_is_partial", "sum"),
        unassigned_transcripts=("_is_unassigned", "sum"),
    )
    report = report.join(outcome, how="left")
    report["main_retained"] = report["main_transcripts"] > 0
    report["has_accepted_nuclear_pair"] = report["accepted_nuclear_gene_pairs"] > 0
    return report.reset_index(names="cell_id")


def _entity_counts(df_out: pd.DataFrame) -> dict[str, int]:
    labels = df_out["tracer_id"].astype(str)
    etypes = df_out["_etype"].astype(str)
    return {
        "main_cells": int(labels.loc[etypes == "cell"].nunique()),
        "partial_entities": int(labels.loc[etypes == "partial"].nunique()),
        "component_entities": int(labels.loc[etypes == "component"].nunique()),
        "unassigned_transcripts": int((labels == "-1").sum()),
        "assigned_transcripts": int((labels != "-1").sum()),
    }


def _compare_panels(panel: pd.DataFrame, specs: list[str]) -> dict[str, dict]:
    if not specs:
        return {}
    primary_metric = "PMI" if "PMI" in panel.columns else "NPMI"
    primary = panel[["gene_i", "gene_j", primary_metric]].rename(
        columns={primary_metric: "primary_value"}
    )
    comparisons = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"--comparison-panel must use NAME=PATH syntax; got {spec!r}"
            )
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path)
        if not name or not raw_path:
            raise ValueError(f"Invalid --comparison-panel value: {spec!r}")
        other = load_npmi_panel(path)
        other_metric = "PMI" if "PMI" in other.columns else "NPMI"
        other_values = other[["gene_i", "gene_j", other_metric]].rename(
            columns={other_metric: "comparison_value"}
        )
        shared = primary.merge(
            other_values,
            on=["gene_i", "gene_j"],
            how="inner",
            validate="one_to_one",
        )
        pearson_raw = (
            shared["primary_value"].corr(shared["comparison_value"], method="pearson")
            if len(shared) > 1
            else np.nan
        )
        spearman_raw = (
            shared["primary_value"].corr(shared["comparison_value"], method="spearman")
            if len(shared) > 1
            else np.nan
        )
        pearson = float(pearson_raw) if np.isfinite(pearson_raw) else None
        spearman = float(spearman_raw) if np.isfinite(spearman_raw) else None
        sign_agreement = (
            float(
                (
                    np.sign(shared["primary_value"])
                    == np.sign(shared["comparison_value"])
                ).mean()
            )
            if len(shared)
            else None
        )
        comparisons[name] = {
            "path": str(path),
            "metric": other_metric,
            "rows": len(other),
            "genes": len(
                set(other["gene_i"].astype(str)) | set(other["gene_j"].astype(str))
            ),
            "shared_edges": len(shared),
            "primary_edge_overlap_fraction": (
                float(len(shared) / len(primary)) if len(primary) else 0.0
            ),
            "pearson": pearson,
            "spearman": spearman,
            "sign_agreement": sign_agreement,
            "sha256": _sha256(path),
        }
    return comparisons


def main() -> None:
    args = _parse_args()

    from tracer.config import load_config, to_dict as config_to_dict
    from tracer.core import set_reproducibility_seed
    import tracer
    import tracer.pipeline as pipeline
    import tracer.pruning as pruning
    from tracer.pruning import prune_transcripts_nuclear_seed

    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    set_reproducibility_seed(args.seed)

    report_path = args.report_output or _companion_path(args.output, "_summary.json")
    cell_report_path = (
        args.cell_report_output or _companion_path(args.output, "_cells.csv")
    )
    for path in (args.output, report_path, cell_report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading transcripts from: {args.input}")
    df = load_transcripts(args.input)
    print(f"Reading accepted PMI panel from: {args.npmi}")
    panel = load_npmi_panel(args.npmi)
    cfg = load_config(path=args.user_config, platform=args.platform)
    phase1 = cfg.phase1

    pmi_threshold = (
        float(args.pmi_threshold)
        if args.pmi_threshold is not None
        else float(pipeline.PMI_THR)
    )
    seed_coherence_floor = (
        float(args.seed_coherence_floor)
        if args.seed_coherence_floor is not None
        else float(pipeline.SEED_COHERENCE_FLOOR)
    )
    metric_col = "PMI" if "PMI" in panel.columns else "NPMI"

    print(f"tracer: {tracer.__file__}")
    print(f"tracer.pipeline: {pipeline.__file__}")
    print(f"tracer.pruning: {pruning.__file__}")
    print(
        "Prune parameters: "
        f"threshold={pmi_threshold} "
        f"seed_coherence_floor={seed_coherence_floor} "
        f"nuclear_only_admit={pipeline.NUCLEAR_ONLY_ADMIT} "
        f"tx_weighted={pipeline.TX_WEIGHTED_PRUNE} "
        f"veto_mode={phase1.veto_mode}"
    )

    original_cell_ids = set(
        df.loc[df["cell_id"].astype(str) != "-1", "cell_id"].astype(str)
    )
    original_transcript_ids = (
        df["transcript_id"].to_numpy(copy=True)
        if "transcript_id" in df.columns
        else np.arange(len(df), dtype=np.int64)
    )

    t0 = time.time()
    df_out, aux = prune_transcripts_nuclear_seed(
        df,
        panel,
        cell_id_col="cell_id",
        out_col="tracer_id",
        gene_col="feature_name",
        nuclear_col="overlaps_nucleus",
        threshold=pmi_threshold,
        unassigned_id="-1",
        metric_col=metric_col,
        nan_fill=0.0,
        min_nuclear_genes=3,
        seed_coherence_floor=seed_coherence_floor,
        nuclear_only_admit=pipeline.NUCLEAR_ONLY_ADMIT,
        tx_weighted=pipeline.TX_WEIGHTED_PRUNE,
        veto_mode=phase1.veto_mode,
        mean_admit_threshold=phase1.mean_admit_threshold,
        min_admit_threshold=phase1.min_admit_threshold,
        aggregator_percentile=phase1.aggregator_percentile,
        real_signal_threshold=phase1.real_signal_threshold,
        neg_npmi_threshold=phase1.neg_npmi_threshold,
        n_jobs=-1,
        show_progress=False,
    )
    wall_seconds = time.time() - t0

    if len(df_out) != len(df):
        raise RuntimeError(
            f"Prune changed transcript row count: {len(df):,} -> {len(df_out):,}"
        )
    if "transcript_id" in df_out.columns and not np.array_equal(
        df_out["transcript_id"].to_numpy(), original_transcript_ids
    ):
        raise RuntimeError("Prune changed transcript_id values or row order.")
    if not np.array_equal(
        df_out["cell_id"].astype(str).to_numpy(),
        df["cell_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Prune changed original cell_id values.")

    cell_report = _build_cell_report(df_out, aux)
    cell_report.to_csv(cell_report_path, index=False)
    df_out.to_parquet(args.output, index=False)

    retained = int(cell_report["main_retained"].sum())
    n_original = len(original_cell_ids)
    zero_supported = int((~cell_report["has_accepted_nuclear_pair"]).sum())
    entity_counts = _entity_counts(df_out)
    panel_genes = set(panel["gene_i"].astype(str)) | set(panel["gene_j"].astype(str))
    input_genes = set(df["feature_name"].astype(str))
    panel_comparisons = _compare_panels(panel, args.comparison_panel)
    if retained >= 4_500:
        decision = "spatial_scope_or_exporter_supported"
    elif retained < 2_000:
        decision = "spatial_pmi_still_incompatible"
    else:
        decision = "intermediate_review_required"

    supported_pairs = cell_report["accepted_nuclear_gene_pairs"].to_numpy()
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(args.input),
        "npmi": str(args.npmi),
        "output": str(args.output),
        "cell_report": str(cell_report_path),
        "runtime": {
            "wall_seconds": wall_seconds,
            "tracer_module": getattr(tracer, "__file__", None),
            "pipeline_module": getattr(pipeline, "__file__", None),
            "pruning_module": getattr(pruning, "__file__", None),
        },
        "parameters": {
            "platform": args.platform,
            "seed": args.seed,
            "metric_col": metric_col,
            "pmi_threshold": pmi_threshold,
            "seed_coherence_floor": seed_coherence_floor,
            "min_nuclear_genes": 3,
            "nuclear_only_admit": bool(pipeline.NUCLEAR_ONLY_ADMIT),
            "tx_weighted": bool(pipeline.TX_WEIGHTED_PRUNE),
            "phase1": config_to_dict(cfg)["phase1"],
        },
        "input_counts": {
            "transcripts": len(df),
            "original_cells": n_original,
            "assigned_transcripts": int((df["cell_id"].astype(str) != "-1").sum()),
            "unassigned_transcripts": int((df["cell_id"].astype(str) == "-1").sum()),
            "nucleus_overlapping_transcripts": int(
                df["overlaps_nucleus"].astype(bool).sum()
            ),
            "genes": len(input_genes),
        },
        "panel": {
            "rows": len(panel),
            "genes": len(panel_genes),
            "input_genes_missing_from_panel": sorted(input_genes - panel_genes),
            "comparisons": panel_comparisons,
        },
        "prune": {
            **entity_counts,
            "original_cell_ids_retained": retained,
            "original_cell_ids_lost": n_original - retained,
            "original_cell_retention_fraction": (
                float(retained / n_original) if n_original else 0.0
            ),
        },
        "seed_panel_support": {
            "cells_with_zero_accepted_nuclear_pairs": zero_supported,
            "fraction_with_zero_accepted_nuclear_pairs": (
                float(zero_supported / n_original) if n_original else 0.0
            ),
            "accepted_nuclear_pairs_min": int(np.min(supported_pairs)),
            "accepted_nuclear_pairs_median": float(np.median(supported_pairs)),
            "accepted_nuclear_pairs_p90": float(np.percentile(supported_pairs, 90)),
            "accepted_nuclear_pairs_max": int(np.max(supported_pairs)),
        },
        "decision": decision,
        "decision_thresholds": {
            "scope_or_exporter_supported_min_retained": 4_500,
            "spatial_pmi_incompatible_max_retained_exclusive": 2_000,
        },
    }
    summary["checksums"] = {
        "npmi_sha256": _sha256(args.npmi),
        "output_sha256": _sha256(args.output),
        "cell_report_sha256": _sha256(cell_report_path),
    }
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(
        "Prune complete: "
        f"retained original cells={retained:,}/{n_original:,}; "
        f"main cells={entity_counts['main_cells']:,}; "
        f"partials={entity_counts['partial_entities']:,}; "
        f"unassigned tx={entity_counts['unassigned_transcripts']:,}; "
        f"zero-supported seeds={zero_supported:,}; "
        f"wall={wall_seconds:.1f}s"
    )
    print(f"Decision: {decision}")
    print(f"Saved Prune parquet to: {args.output}")
    print(f"Saved per-cell report to: {cell_report_path}")
    print(f"Saved summary to: {report_path}")


if __name__ == "__main__":
    main()
