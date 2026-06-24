#!/usr/bin/env python3
"""Run ovrlpy as an orthogonal vertical-integrity diagnostic.

Wraps the ovrlpy ``Ovrlp.process_coordinates → fit_transcripts → compute_VSI``
pipeline and emits per-pixel VSI plus the summary metrics needed for
benchmark comparison. ovrlpy is treated as an **orthogonal diagnostic**:
it is NOT a TRACER optimization target.

USAGE
=====
Post-method only::

    python scripts/run_ovrlpy.py \\
      --transcripts results/tracer/lung_xenium/outputs/transcripts_tracer_refined.parquet \\
      --outdir results/benchmark/lung_xenium/ovrlpy_post \\
      --gene-mode all_shared_genes \\
      --reference-h5ad <lung scRNA>

Pre + post (paired comparison)::

    python scripts/run_ovrlpy.py \\
      --transcripts results/tracer/lung_xenium/outputs/transcripts_tracer_refined.parquet \\
      --transcripts-pre datasets/lung_cancer_xenium_10x/filtered_df_standardized.parquet \\
      --outdir results/benchmark/lung_xenium/ovrlpy_pre_post \\
      --gene-mode all_shared_genes \\
      --reference-h5ad <lung scRNA>

If ovrlpy / polars are not importable, the script exits 1 with a clear
message rather than producing a silent stub.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--transcripts", required=True, type=Path,
                   help="Method-output transcripts parquet (post).")
    p.add_argument("--label-col", default=None,
                   help="Label column (default: auto-detect 'stitched' then 'cell_id').")
    p.add_argument("--transcripts-pre", type=Path, default=None,
                   help="Optional pre-method transcripts for paired pre/post analysis.")
    p.add_argument("--pre-label-col", default="cell_id")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--gene-mode",
                   choices=("all_shared_genes", "spatial_panel", "intersect_reference"),
                   default="spatial_panel",
                   help="Gene set restriction. 'spatial_panel' uses all genes in "
                        "transcripts; 'intersect_reference' intersects with "
                        "--reference-h5ad var_names; 'all_shared_genes' uses the "
                        "union of pre+post spatial genes (intersected with reference "
                        "if --reference-h5ad provided).")
    p.add_argument("--reference-h5ad", type=Path, default=None,
                   help="Optional reference h5ad used to filter genes when "
                        "gene-mode involves 'reference'.")
    p.add_argument("--min-transcripts", type=int, default=50,
                   help="Min transcripts per cell for VSI computation.")
    p.add_argument("--low-vsi-threshold", type=float, default=0.5,
                   help="Threshold below which a cell is classified low-VSI.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument(
        "--entity-mode",
        choices=("whole_cells_only", "all_entities", "partial_cells_only"),
        default="whole_cells_only",
        help=(
            "Which TRACER entity classes (column `_etype` on post transcripts) "
            "to feed into ovrlpy. ovrlpy scores vertical coherence per "
            "segmentation unit; partial/component/unassigned entities are not "
            "fully-resolved biological cells and would distort VSI if treated as "
            "such. Modes: 'whole_cells_only' (default — _etype=='cell' only); "
            "'partial_cells_only' (_etype in {'partial','component'} — "
            "sensitivity analysis); 'all_entities' (no filter; legacy behavior)."
        ),
    )
    p.add_argument(
        "--entity-col", default="_etype",
        help="Column in --transcripts that carries the entity class label.",
    )
    return p


# Canonical entity-class buckets. Anything not in `whole` or `partial` is
# treated as an unassigned/dropped unit and excluded from the main ovrlpy run.
WHOLE_CELL_LABELS = {"cell"}
PARTIAL_CELL_LABELS = {"partial", "component", "rescued_component"}
EXCLUDED_LABELS = {"unknown", "unassigned", "UNASSIGNED", "drop", "DROP", "-1"}


def filter_by_entity_mode(
    df: pd.DataFrame, *, mode: str, entity_col: str, tag: str,
    log: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Subset a transcripts DataFrame by entity class.

    Returns (filtered_df, counts_dict). counts_dict tracks how many transcripts
    fell into each bucket so the caller can report what was excluded.
    """
    counts: dict[str, int] = {}
    if entity_col not in df.columns:
        # Pre-TRACER raw inputs do not carry _etype — every transcript is just
        # whatever the upstream segmentation produced. Treat them all as
        # "whole" so pre-runs in whole_cells_only mode are not silently empty.
        counts["no_entity_col"] = int(len(df))
        log.info("[%s/entity_mode=%s] no '%s' column — passing all %d transcripts through.",
                 tag, mode, entity_col, len(df))
        return df, counts

    et = df[entity_col].astype(str)
    n_total = int(len(df))
    in_whole   = et.isin(WHOLE_CELL_LABELS)
    in_partial = et.isin(PARTIAL_CELL_LABELS)
    n_whole, n_partial = int(in_whole.sum()), int(in_partial.sum())
    n_other = n_total - n_whole - n_partial
    counts = {
        "n_total": n_total,
        "n_whole_cells_transcripts": n_whole,
        "n_partial_cells_transcripts": n_partial,
        "n_other_transcripts": n_other,
    }

    if mode == "whole_cells_only":
        keep = in_whole
    elif mode == "partial_cells_only":
        keep = in_partial
    elif mode == "all_entities":
        keep = pd.Series(True, index=df.index)
    else:
        raise SystemExit(f"unknown --entity-mode {mode!r}")

    n_kept = int(keep.sum())
    counts["n_kept_after_entity_filter"] = n_kept
    counts["n_excluded_by_entity_filter"] = n_total - n_kept
    log.info(
        "[%s/entity_mode=%s] entity breakdown: whole=%d partial=%d other=%d "
        "→ kept %d / excluded %d",
        tag, mode, n_whole, n_partial, n_other, n_kept, n_total - n_kept,
    )
    return df.loc[keep].copy(), counts


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(outdir: Path) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("run_ovrlpy")
    log.setLevel(logging.INFO)
    log.propagate = False
    if log.handlers:
        return log
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s :: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(outdir / "ovrlpy.log", mode="a"); fh.setFormatter(fmt)
    log.addHandler(sh); log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def detect_label_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise SystemExit(f"--label-col {requested!r} not in transcripts.")
        return requested
    for c in ("stitched", "cell_id"):
        if c in df.columns:
            return c
    raise SystemExit(
        f"Could not detect label column. Columns: {list(df.columns)}"
    )


def load_transcripts(path: Path, *, log: logging.Logger) -> pd.DataFrame:
    log.info("Loading transcripts: %s", path)
    df = pd.read_parquet(path)
    if "z" not in df.columns:
        df["z"] = np.float32(0.0)
    for c in ("x", "y", "z", "feature_name"):
        if c not in df.columns:
            raise SystemExit(f"transcripts parquet missing {c!r}; cols={list(df.columns)}")
    return df


def determine_gene_set(
    args, df_post: pd.DataFrame, df_pre: pd.DataFrame | None,
    log: logging.Logger,
) -> set[str]:
    if args.gene_mode == "spatial_panel":
        genes = set(df_post["feature_name"].astype(str).unique())
        if df_pre is not None:
            genes |= set(df_pre["feature_name"].astype(str).unique())
        return genes
    if args.gene_mode == "all_shared_genes":
        genes = set(df_post["feature_name"].astype(str).unique())
        if df_pre is not None:
            genes |= set(df_pre["feature_name"].astype(str).unique())
        if args.reference_h5ad is not None:
            import anndata as ad
            ref = ad.read_h5ad(args.reference_h5ad, backed="r")
            genes &= set(np.asarray(ref.var_names, dtype=str))
            log.info("  intersected with reference: %d genes", len(genes))
        return genes
    if args.gene_mode == "intersect_reference":
        if args.reference_h5ad is None:
            raise SystemExit("--gene-mode intersect_reference requires --reference-h5ad")
        import anndata as ad
        ref = ad.read_h5ad(args.reference_h5ad, backed="r")
        spatial = set(df_post["feature_name"].astype(str).unique())
        if df_pre is not None:
            spatial &= set(df_pre["feature_name"].astype(str).unique())
        return spatial & set(np.asarray(ref.var_names, dtype=str))
    raise SystemExit(f"unknown gene-mode {args.gene_mode!r}")


# ---------------------------------------------------------------------------
# ovrlpy invocation
# ---------------------------------------------------------------------------
@dataclass
class OvrlpyRun:
    tag: str
    n_input_transcripts: int
    n_cells_in_input: int
    n_pixels_scored: int
    mean_vsi: float
    median_vsi: float
    fraction_low_vsi: float
    n_low_vsi: int
    per_pixel_parquet: str | None = None


def run_one(
    df: pd.DataFrame, *, label_col: str, tag: str,
    gene_set: set[str], outdir: Path, seed: int, n_workers: int,
    min_transcripts: int, low_vsi_threshold: float,
    log: logging.Logger,
) -> OvrlpyRun:
    try:
        import ovrlpy
        import polars as pl
    except ImportError as e:
        raise SystemExit(
            f"ovrlpy / polars not importable ({e}). Install them in the "
            f"current Python environment before running this script."
        )

    sub = df.loc[df["feature_name"].astype(str).isin(gene_set), :].copy()
    n_input = len(sub)
    # ovrlpy expects polars DataFrame with x/y/z/gene/cell_id.
    keep = sub[["x", "y", "z", "feature_name", label_col]].rename(
        columns={"feature_name": "gene", label_col: "cell_id"},
    )
    keep["cell_id"] = keep["cell_id"].astype(str)
    tx_pl = pl.from_pandas(keep)
    log.info("[ovrlpy/%s] running on %d transcripts (%d genes, %d cells)",
             tag, n_input, len(gene_set), keep["cell_id"].nunique())

    ov = ovrlpy.Ovrlp(tx_pl, random_state=seed, n_workers=n_workers)
    ov.process_coordinates()
    ov.fit_transcripts(min_transcripts=min_transcripts)
    ov.compute_VSI(min_transcripts=min_transcripts)
    per_pixel = ovrlpy.cell_integrity_from_transcripts(
        ov, cell_id="cell_id", unassigned="-1",
    )
    pdf = per_pixel.to_pandas() if hasattr(per_pixel, "to_pandas") else per_pixel

    int_col = "integrity" if "integrity" in pdf.columns else "vsi"
    if int_col not in pdf.columns:
        raise SystemExit(
            f"ovrlpy output has no 'integrity' or 'vsi' column. "
            f"Columns: {list(pdf.columns)}"
        )
    mean_vsi = float(pdf[int_col].mean())
    median_vsi = float(pdf[int_col].median())
    n_low = int((pdf[int_col] < low_vsi_threshold).sum())
    frac_low = float(n_low / max(1, len(pdf)))

    pp_path = outdir / f"ovrlpy_per_pixel_{tag}.parquet"
    pdf.to_parquet(pp_path, index=False)

    log.info("[ovrlpy/%s] mean_vsi=%.3f median_vsi=%.3f frac_low<%.2f=%.3f (n_low=%d)",
             tag, mean_vsi, median_vsi, low_vsi_threshold, frac_low, n_low)
    return OvrlpyRun(
        tag=tag,
        n_input_transcripts=int(n_input),
        n_cells_in_input=int(keep["cell_id"].nunique()),
        n_pixels_scored=int(len(pdf)),
        mean_vsi=mean_vsi, median_vsi=median_vsi,
        fraction_low_vsi=frac_low, n_low_vsi=n_low,
        per_pixel_parquet=str(pp_path),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_problem_score_map(
    pp_path: Path, *, outdir: Path, tag: str, log: logging.Logger,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib unavailable; skipping plot for %s", tag)
        return
    df = pd.read_parquet(pp_path)
    int_col = "integrity" if "integrity" in df.columns else "vsi"
    if "x" not in df.columns or "y" not in df.columns:
        log.warning("per-pixel parquet has no x/y; skipping spatial plot for %s", tag)
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df["x"], df["y"], c=1.0 - df[int_col],
                    s=2, cmap="Reds", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.set_title(f"ovrlpy problem score (1 - VSI) — {tag}")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / f"ovrlpy_problem_score_map_{tag}.png", dpi=150)
    fig.savefig(outdir / f"ovrlpy_problem_score_map_{tag}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_argparser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.outdir)
    log.info("=== run_ovrlpy.py === out=%s", args.outdir)

    df_post = load_transcripts(args.transcripts, log=log)
    post_label = detect_label_col(df_post, args.label_col)
    df_pre = None
    if args.transcripts_pre is not None:
        df_pre = load_transcripts(args.transcripts_pre, log=log)
        if args.pre_label_col not in df_pre.columns:
            raise SystemExit(
                f"--pre-label-col {args.pre_label_col!r} not in pre transcripts."
            )

    # Apply entity-mode filter BEFORE gene-set inference so that any genes
    # only seen in excluded entities are not counted toward the panel.
    df_post, post_entity_counts = filter_by_entity_mode(
        df_post, mode=args.entity_mode, entity_col=args.entity_col,
        tag="post", log=log,
    )
    pre_entity_counts: dict[str, int] = {}
    if df_pre is not None:
        df_pre, pre_entity_counts = filter_by_entity_mode(
            df_pre, mode=args.entity_mode, entity_col=args.entity_col,
            tag="pre", log=log,
        )

    gene_set = determine_gene_set(args, df_post, df_pre, log)
    log.info("Gene set: %d genes (mode=%s)", len(gene_set), args.gene_mode)

    runs: dict[str, OvrlpyRun] = {}
    t0 = time.time()
    runs["post"] = run_one(
        df_post, label_col=post_label, tag="post",
        gene_set=gene_set, outdir=args.outdir, seed=args.seed,
        n_workers=args.n_workers, min_transcripts=args.min_transcripts,
        low_vsi_threshold=args.low_vsi_threshold, log=log,
    )
    if df_pre is not None:
        runs["pre"] = run_one(
            df_pre, label_col=args.pre_label_col, tag="pre",
            gene_set=gene_set, outdir=args.outdir, seed=args.seed,
            n_workers=args.n_workers, min_transcripts=args.min_transcripts,
            low_vsi_threshold=args.low_vsi_threshold, log=log,
        )
    runtime = time.time() - t0

    # Per-cell metrics TSV: emit the per-pixel summary alongside.
    rows = []
    for tag, run in runs.items():
        rows.append({"tag": tag, **{k: v for k, v in asdict(run).items()
                                     if k != "per_pixel_parquet"}})
    pd.DataFrame(rows).to_csv(args.outdir / "ovrlpy_cell_metrics.tsv",
                              sep="\t", index=False)
    # Tag-specific aliases requested by the benchmark spec.
    if "pre" in runs:
        pd.DataFrame([{"tag": "pre", **{k: v for k, v in asdict(runs["pre"]).items()
                                         if k != "per_pixel_parquet"}}]
                     ).to_csv(args.outdir / "ovrlpy_pre_cell_metrics.tsv",
                              sep="\t", index=False)
    if "post" in runs:
        suffix = "whole_cell" if args.entity_mode == "whole_cells_only" else args.entity_mode
        pd.DataFrame([{"tag": "post", **{k: v for k, v in asdict(runs["post"]).items()
                                          if k != "per_pixel_parquet"}}]
                     ).to_csv(args.outdir / f"ovrlpy_post_{suffix}_metrics.tsv",
                              sep="\t", index=False)

    if "pre" in runs and "post" in runs:
        pre, post = runs["pre"], runs["post"]
        pp_rows = [
            ("mean_vsi", pre.mean_vsi, post.mean_vsi, post.mean_vsi - pre.mean_vsi),
            ("median_vsi", pre.median_vsi, post.median_vsi, post.median_vsi - pre.median_vsi),
            ("fraction_low_vsi", pre.fraction_low_vsi, post.fraction_low_vsi,
             post.fraction_low_vsi - pre.fraction_low_vsi),
            ("n_low_vsi", pre.n_low_vsi, post.n_low_vsi, post.n_low_vsi - pre.n_low_vsi),
            ("n_pixels_scored", pre.n_pixels_scored, post.n_pixels_scored,
             post.n_pixels_scored - pre.n_pixels_scored),
        ]
        pd.DataFrame(pp_rows, columns=["metric", "pre", "post", "delta"]).to_csv(
            args.outdir / "ovrlpy_pre_post_metrics.tsv", sep="\t", index=False,
        )
        log.info("[pre/post] mean_vsi: %.3f → %.3f (Δ %+.3f); fraction_low: %.3f → %.3f (Δ %+.3f)",
                 pre.mean_vsi, post.mean_vsi, post.mean_vsi - pre.mean_vsi,
                 pre.fraction_low_vsi, post.fraction_low_vsi,
                 post.fraction_low_vsi - pre.fraction_low_vsi)

    # Plots
    for tag, run in runs.items():
        if run.per_pixel_parquet is not None:
            plot_problem_score_map(
                Path(run.per_pixel_parquet), outdir=args.outdir,
                tag=tag, log=log,
            )

    # Run summary
    summary = {
        "command": " ".join(sys.argv),
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
        "n_genes_used": len(gene_set),
        "gene_mode": args.gene_mode,
        "entity_mode": args.entity_mode,
        "entity_filter_counts": {
            "post": post_entity_counts,
            "pre":  pre_entity_counts,
        },
        "runs": {tag: asdict(run) for tag, run in runs.items()},
        "runtime_seconds": runtime,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.outdir / "ovrlpy_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("DONE — outputs at %s", args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
