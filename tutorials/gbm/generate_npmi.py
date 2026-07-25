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
    - per_gene_percentile_filter=(5, 95)                  (production size-band filter)

The Xenium ``qv`` quality cut, raw sentinel-cell canonicalization, and
control/deprecated feature exclusion stay here as explicit pre-steps.

The operational output is a long-format panel CSV with columns ``gene_i,
gene_j, PMI, NPMI`` containing exactly the accepted edges stored in
``PmiBootstrapResult.W_sparse``. The complete ``pair_ci`` table is written
separately for diagnostics; rejected, unsettled, and low-evidence rows are
never promoted to operational PMI edges.

EXAMPLE
=======
::

    python tutorials/gbm/generate_npmi.py \\
      --input  tutorials/gbm/data/transcripts.parquet \\
      --output tutorials/gbm/data/gbm_npmi_no_controls.csv \\
      --qv-min 30 --low-pct 20 --high-pct 80 --seed 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_COLUMNS = {"feature_name", "cell_id", "qv", "overlaps_nucleus"}
UNASSIGNED_TOKENS = frozenset(
    {"UNASSIGNED", "Unassigned", "unassigned", "DROP", "nan", "None", "", "0", "-1", "NA"}
)
DEFAULT_EXCLUDE_FEATURE_PREFIXES = (
    "NegControl",
    "DeprecatedCodeword",
    "BLANK",
    "Unassigned",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a GBM NPMI panel from a transcript parquet.")
    parser.add_argument("--input", required=True, help="Transcript parquet path.")
    parser.add_argument("--output", required=True, help="Output CSV path for the NPMI panel.")
    parser.add_argument(
        "--audit-output",
        default=None,
        help="Optional full pair_ci audit CSV (default: <output-stem>_pair_ci.csv).",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON run summary (default: <output-stem>_summary.json).",
    )
    parser.add_argument("--qv-min", type=float, default=30.0, help="Minimum Xenium qv to keep.")
    parser.add_argument("--low-pct", type=float, default=20.0, help="Lower percentile for confident nuclei.")
    parser.add_argument("--high-pct", type=float, default=80.0, help="Upper percentile for confident nuclei.")
    parser.add_argument(
        "--per-gene-low-pct",
        type=float,
        default=5.0,
        help="Lower per-gene size-band percentile for PMI admittance.",
    )
    parser.add_argument(
        "--per-gene-high-pct",
        type=float,
        default=95.0,
        help="Upper per-gene size-band percentile for PMI admittance.",
    )
    parser.add_argument(
        "--no-per-gene-percentile-filter",
        action="store_true",
        help="Disable the per-gene size-band PMI admittance filter.",
    )
    parser.add_argument(
        "--min-occurrences-per-context",
        type=int,
        default=2,
        help="Minimum copies of a gene in a nucleus before it counts as present.",
    )
    parser.add_argument(
        "--exclude-feature-prefix",
        action="append",
        default=list(DEFAULT_EXCLUDE_FEATURE_PREFIXES),
        help=(
            "Feature-name prefix to exclude before PMI generation. "
            "May be repeated; defaults exclude Xenium control/deprecated features."
        ),
    )
    parser.add_argument(
        "--keep-control-features",
        action="store_true",
        help="Do not apply the default feature-prefix exclusions.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Bootstrap RNG seed (reproducibility).")
    parser.add_argument(
        "--max-bootstraps",
        type=int,
        default=10_000,
        help="Maximum active-sampler bootstrap iterations.",
    )
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


def _canonicalize_cell_id(series):
    """Map raw Xenium unassigned sentinels to TRACER's canonical "-1" label."""
    import pandas as pd

    values = series.astype(str).str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_sentinel = (
        numeric.notna()
        & (numeric <= 0)
        & values.str.fullmatch(r"[+-]?\d+(?:\.0+)?", na=False)
    )
    sentinel = values.isin(UNASSIGNED_TOKENS) | numeric_sentinel
    return values.where(~sentinel, "-1")


def _feature_prefix_mask(series, prefixes: tuple[str, ...]):
    if not prefixes:
        return series.astype(str) == "\0"
    return series.astype(str).str.startswith(prefixes, na=False)


def _companion_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}{suffix}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value):
    """Compact numpy/pandas-heavy diagnostics into JSON-safe summaries."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else value
        summary = {"size": int(value.size)}
        if finite.size and np.issubdtype(value.dtype, np.number):
            summary.update(
                {
                    "min": float(np.min(finite)),
                    "median": float(np.median(finite)),
                    "max": float(np.max(finite)),
                }
            )
        return summary
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _extract_accepted_panel(result):
    """Return the operational W_sparse edges and an annotated pair_ci audit."""
    import numpy as np
    import pandas as pd

    genes = np.asarray(result.genes, dtype=str)
    W = result.W_sparse.tocoo(copy=False)
    if W.shape != (len(genes), len(genes)):
        raise ValueError(
            f"W_sparse shape {W.shape} does not match {len(genes)} result genes."
        )

    i_idx = np.minimum(W.row, W.col).astype(np.int64, copy=False)
    j_idx = np.maximum(W.row, W.col).astype(np.int64, copy=False)
    if np.any(i_idx == j_idx):
        raise ValueError("W_sparse unexpectedly contains self-pairs.")
    if not np.isfinite(W.data).all():
        raise ValueError("W_sparse contains non-finite accepted PMI values.")

    accepted = pd.DataFrame(
        {
            "gene_i_idx": i_idx,
            "gene_j_idx": j_idx,
            "PMI": W.data.astype(np.float64, copy=False),
        }
    ).sort_values(["gene_i_idx", "gene_j_idx"], ignore_index=True)
    if accepted.duplicated(["gene_i_idx", "gene_j_idx"]).any():
        raise ValueError("W_sparse contains duplicate undirected gene pairs.")

    ci = result.pair_ci
    if ci is None or ci.empty:
        raise ValueError(
            "compute_pmi_bootstrap returned no pair_ci audit rows; "
            "persist_ci=True is required."
        )
    audit = ci.copy()
    for col in ("gene_i_idx", "gene_j_idx"):
        audit[col] = pd.to_numeric(audit[col], errors="raise").astype(np.int64)
    audit_i = np.minimum(audit["gene_i_idx"], audit["gene_j_idx"])
    audit_j = np.maximum(audit["gene_i_idx"], audit["gene_j_idx"])
    audit["gene_i_idx"] = audit_i
    audit["gene_j_idx"] = audit_j
    if audit.duplicated(["gene_i_idx", "gene_j_idx"]).any():
        raise ValueError("pair_ci contains duplicate undirected gene pairs.")

    lookup = audit[
        ["gene_i_idx", "gene_j_idx", "legacy_npmi", "kind"]
    ].rename(columns={"legacy_npmi": "NPMI"})
    panel = accepted.merge(
        lookup,
        on=["gene_i_idx", "gene_j_idx"],
        how="left",
        validate="one_to_one",
    )
    if panel["NPMI"].isna().any():
        missing = int(panel["NPMI"].isna().sum())
        raise ValueError(
            f"{missing} accepted W_sparse edges lack a finite legacy_npmi in pair_ci."
        )
    panel["NPMI"] = pd.to_numeric(panel["NPMI"], errors="raise")
    if not np.isfinite(panel["NPMI"]).all():
        raise ValueError("Accepted panel contains non-finite NPMI values.")
    panel.insert(0, "gene_i", genes[panel["gene_i_idx"].to_numpy()])
    panel.insert(1, "gene_j", genes[panel["gene_j_idx"].to_numpy()])

    accepted_values = accepted.rename(columns={"PMI": "accepted_metric_value"})
    audit = audit.merge(
        accepted_values,
        on=["gene_i_idx", "gene_j_idx"],
        how="left",
        validate="one_to_one",
    )
    audit["accepted_in_w_sparse"] = audit["accepted_metric_value"].notna()

    panel = panel[["gene_i", "gene_j", "PMI", "NPMI"]]
    return panel, audit


def main() -> None:
    args = _parse_args()

    import numpy as np
    import pandas as pd

    from tracer.metrics import compute_pmi_bootstrap

    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = (
        Path(args.audit_output)
        if args.audit_output
        else _companion_path(output_path, "_pair_ci.csv")
    )
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else _companion_path(output_path, "_summary.json")
    )

    df = pd.read_parquet(input_path)
    _validate_columns(df, REQUIRED_COLUMNS)

    df = df.copy()
    df["feature_name"] = df["feature_name"].astype(str).str.strip()
    raw_cell_id = df["cell_id"].astype(str).str.strip()
    df["cell_id"] = _canonicalize_cell_id(df["cell_id"])
    df["qv"] = pd.to_numeric(df["qv"], errors="coerce")

    exclude_prefixes = (
        tuple()
        if args.keep_control_features
        else tuple(dict.fromkeys(str(p) for p in args.exclude_feature_prefix if str(p)))
    )
    per_gene_percentile_filter = (
        None
        if args.no_per_gene_percentile_filter
        else (args.per_gene_low_pct, args.per_gene_high_pct)
    )

    # qv / feature filters: explicit pre-steps without compute_pmi_bootstrap equivalents.
    n_in = len(df)
    remapped_to_unassigned = int(((raw_cell_id != "-1") & (df["cell_id"] == "-1")).sum())
    keep = (df["qv"] >= args.qv_min) & (df["feature_name"] != "")
    n_dropped_qv_or_empty = int((~keep).sum())
    if exclude_prefixes:
        control_mask = _feature_prefix_mask(df["feature_name"], exclude_prefixes)
        n_dropped_prefix = int((keep & control_mask).sum())
        keep &= ~control_mask
    else:
        n_dropped_prefix = 0
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError("No transcripts remain after qv / gene-prefix / empty-gene filtering.")

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
        per_gene_percentile_filter=per_gene_percentile_filter,
        metric="pmi",
        seed=args.seed,
        max_bootstraps=args.max_bootstraps,
        persist_ci=True,  # required to populate result.pair_ci
        show_progress=args.show_progress,
    )

    panel, audit = _extract_accepted_panel(result)
    if exclude_prefixes:
        leaked = _feature_prefix_mask(panel["gene_i"], exclude_prefixes) | _feature_prefix_mask(
            panel["gene_j"], exclude_prefixes
        )
        if leaked.any():
            examples = sorted(
                set(panel.loc[leaked, "gene_i"].astype(str))
                | set(panel.loc[leaked, "gene_j"].astype(str))
            )
            examples = [g for g in examples if g.startswith(exclude_prefixes)]
            raise RuntimeError(
                "Excluded feature prefixes leaked into the PMI panel: "
                + ", ".join(examples[:20])
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    audit.to_csv(audit_path, index=False)

    n_genes = len(result.genes)
    n_possible_pairs = n_genes * (n_genes - 1) // 2
    kind_counts = {
        str(k): int(v)
        for k, v in audit["kind"].astype(str).value_counts(dropna=False).items()
    }
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "audit_output": str(audit_path),
        "parameters": {
            "qv_min": args.qv_min,
            "context_percentile_filter": [args.low_pct, args.high_pct],
            "per_gene_percentile_filter": per_gene_percentile_filter,
            "min_occurrences_per_context": args.min_occurrences_per_context,
            "seed": args.seed,
            "max_bootstraps": args.max_bootstraps,
            "metric": "pmi",
            "nuclear_only": True,
        },
        "filtering": {
            "input_transcripts": n_in,
            "cell_id_remapped_to_unassigned": remapped_to_unassigned,
            "dropped_qv_or_empty_gene": n_dropped_qv_or_empty,
            "dropped_excluded_feature_prefix": n_dropped_prefix,
            "transcripts_after_explicit_filters": len(df),
            "excluded_feature_prefixes": list(exclude_prefixes),
        },
        "panel": {
            "result_genes": n_genes,
            "possible_undirected_pairs": n_possible_pairs,
            "accepted_edges": len(panel),
            "w_sparse_nnz": int(result.W_sparse.nnz),
            "accepted_pair_coverage": (
                float(len(panel) / n_possible_pairs) if n_possible_pairs else 0.0
            ),
            "pair_ci_rows": len(audit),
            "pair_ci_kind_counts": kind_counts,
        },
        "bootstrap_diagnostics": _json_safe(result.diagnostics),
    }
    summary["checksums"] = {
        "panel_sha256": _sha256(output_path),
        "audit_sha256": _sha256(audit_path),
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n")

    print(f"Input transcripts: {n_in:,}")
    print(f"cell_id remapped-to--1: {remapped_to_unassigned:,}")
    print(f"Dropped by qv<{args.qv_min} or empty gene: {n_dropped_qv_or_empty:,}")
    print(f"Excluded feature prefixes: {', '.join(exclude_prefixes) if exclude_prefixes else '(none)'}")
    print(f"Dropped by excluded feature prefix after qv/empty filter: {n_dropped_prefix:,}")
    print(f"After qv / gene-prefix / non-empty gene filters: {len(df):,}")
    print(f"Per-gene percentile filter: {per_gene_percentile_filter}")
    print(f"Genes in result: {len(result.genes):,}")
    print(f"Accepted W_sparse edges written: {len(panel):,}")
    print(f"Full pair_ci audit rows: {len(audit):,}")
    print(f"Saved accepted PMI panel to: {output_path}")
    print(f"Saved pair audit to: {audit_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
