#!/usr/bin/env python3
"""Run the TRACER GBM segmentation-refinement pipeline on a Xenium parquet.

This is the config-driven entry point introduced by the 2026-05-27 TRACER
refactor. It replaces the previous hand-wired 5-stage script: the whole
prune -> rescue -> stitch -> split -> stitch workflow now lives behind
``tracer.pipeline.run_segmented_pipeline``, parameterized by a platform
config loaded with ``tracer.config.load_config(platform="xenium")``.

It mirrors the production runner ``scripts/run_tracer.py`` but keeps the
GBM-specific input handling (raw Xenium ``*_location`` column names) and, by
default, writes a single refined transcript parquet that the downstream GBM
tutorial scripts (``merge_slide3_tracer.py`` / ``compare_profiles.py`` /
``plot_chromosome_heatmap.py``) consume.

The pipeline writes the final per-transcript label to a column named
``stitched``. For backward compatibility with the downstream scripts (which
read ``cell_id_finetuned`` / ``cell_id_stitched``) we also emit those names as
aliases of ``stitched``. The original input ``cell_id`` is preserved.

EXAMPLE
=======
::

    python tutorials/gbm/run_gbm.py \\
      --input  tutorials/gbm/data/transcripts.parquet \\
      --npmi   tutorials/gbm/data/gbm_npmi.csv \\
      --output tutorials/gbm/output/df_finetuned.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"x", "y", "feature_name", "cell_id"}
COMMON_COLUMN_ALIASES = {
    "x_location": "x",
    "y_location": "y",
    "z_location": "z",
}
# Tokens that mark a transcript as not belonging to a real cell.
UNASSIGNED_TOKENS = frozenset(
    {"UNASSIGNED", "Unassigned", "unassigned", "DROP", "nan", "None", "", "0", "-1", "NA"}
)


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("run_gbm")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    log.propagate = False
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(h)
    return log


class _RuntimeExceeded(RuntimeError):
    pass


def _install_watchdog(max_runtime_sec: float | None, log: logging.Logger):
    """Arm a SIGALRM watchdog that aborts with a clear message instead of hanging silently."""
    if not max_runtime_sec or max_runtime_sec <= 0:
        return lambda: None
    if not hasattr(signal, "SIGALRM"):
        log.warning("SIGALRM unavailable on this platform; --max-runtime-sec ignored.")
        return lambda: None

    def _handler(signum, frame):
        raise _RuntimeExceeded(
            f"TRACER pipeline exceeded the --max-runtime-sec budget ({max_runtime_sec:.0f}s). "
            f"Re-run with TRACER_STAGE_VERBOSE=1 to see which stage stalled."
        )

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(max_runtime_sec))
    log.info("Watchdog armed: max runtime %.0fs", max_runtime_sec)
    return lambda: signal.setitimer(signal.ITIMER_REAL, 0.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRACER on a GBM transcript parquet.")
    parser.add_argument("--input", required=True, help="Transcript parquet path (raw Xenium ok).")
    parser.add_argument("--npmi", required=True, help="NPMI panel CSV from generate_npmi.py.")
    parser.add_argument("--output", required=True, help="Output parquet for the refined transcripts.")
    parser.add_argument(
        "--platform",
        default="xenium",
        help="TRACER platform preset (src/tracer/configs/platforms/<name>.toml).",
    )
    parser.add_argument(
        "--pmi-threshold",
        type=float,
        default=None,
        help="Override the in-pipeline PMI prune threshold (default: from config).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Reproducibility seed.")
    parser.add_argument(
        "--user-config",
        type=Path,
        default=None,
        help="Optional user-override TOML on top of defaults+platform.",
    )
    parser.add_argument(
        "--max-runtime-sec",
        type=float,
        default=None,
        help="Abort with a clear error if the pipeline exceeds this wall-clock budget.",
    )
    parser.add_argument(
        "--emit-cell-outputs",
        action="store_true",
        help="Also write a cell-by-gene h5ad + per-cell purity/conflict scores "
        "next to --output (off by default; downstream scripts build their own).",
    )
    return parser.parse_args()


def load_transcripts(path: Path) -> pd.DataFrame:
    """Load + standardize a (possibly raw Xenium) GBM transcript parquet."""
    df = pd.read_parquet(path)

    rename_map = {
        src: dst
        for src, dst in COMMON_COLUMN_ALIASES.items()
        if src in df.columns and dst not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"Normalized columns: {rename_map}")

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. "
            "Accepted coordinate names are x/y/z or raw Xenium "
            "x_location/y_location/z_location."
        )

    df = df.copy()
    if "z" not in df.columns:
        df["z"] = np.float32(0.0)
    if "transcript_id" not in df.columns:
        df["transcript_id"] = np.arange(len(df), dtype=np.int64)
    # The pipeline runs nuclear-seed pruning when this column is present
    # (10x Xenium standard output). Default to 0 so it degrades gracefully.
    if "overlaps_nucleus" not in df.columns:
        df["overlaps_nucleus"] = np.uint8(0)

    df["feature_name"] = df["feature_name"].astype(str).str.strip()
    df["cell_id"] = df["cell_id"].astype(str)
    for c in ("x", "y", "z"):
        df[c] = df[c].astype(np.float32)
    df["overlaps_nucleus"] = df["overlaps_nucleus"].astype(np.uint8)

    n_assigned = int((df["cell_id"] != "-1").sum())
    print(
        f"Loaded {len(df):,} transcripts, {df['feature_name'].nunique():,} genes; "
        f"assigned={n_assigned:,}, unassigned={len(df) - n_assigned:,}, "
        f"nucleus-overlapping={int(df['overlaps_nucleus'].sum()):,}"
    )
    return df


def load_npmi_panel(path: Path) -> pd.DataFrame:
    """Load the long-format NPMI panel and symmetric-expand it.

    Same contract as ``scripts/run_tracer.py``: requires ``gene_i``/``gene_j``;
    the pipeline prefers a ``PMI`` column and falls back to ``NPMI``.
    """
    df = pd.read_csv(path)
    if not {"gene_i", "gene_j"}.issubset(df.columns):
        raise SystemExit(f"NPMI panel missing gene_i/gene_j; columns: {list(df.columns)}")
    if df.duplicated(["gene_i", "gene_j"]).any():
        print("NPMI panel has duplicate pairs — keeping first occurrence.")
        df = df.drop_duplicates(["gene_i", "gene_j"], keep="first")
    rev = df.copy()
    rev["gene_i"], rev["gene_j"] = df["gene_j"].values, df["gene_i"].values
    panel = pd.concat([df, rev], ignore_index=True)
    panel = panel.loc[panel["gene_i"] != panel["gene_j"]].reset_index(drop=True)
    print(
        f"NPMI panel: {len(panel):,} rows after symmetric expansion "
        f"(PMI={'yes' if 'PMI' in panel.columns else 'no'}, "
        f"NPMI={'yes' if 'NPMI' in panel.columns else 'no'})"
    )
    return panel


def _emit_cell_outputs(
    df_out: pd.DataFrame, panel: pd.DataFrame, output_path: Path, *, tau: float = 0.05
) -> None:
    """Optional: write a cell-by-gene h5ad + per-cell scores (mirrors run_tracer.py)."""
    import anndata as ad
    import scipy.sparse as sp
    from tracer.metrics import (
        build_cell_gene_matrix,
        build_npmi_matrix,
        compute_cell_conflict_relu,
        compute_cell_purity_relu,
    )

    keep = ~df_out["stitched"].astype(str).isin(UNASSIGNED_TOKENS)
    work = df_out.loc[keep, ["stitched", "feature_name", "x", "y", "z"]].rename(
        columns={"stitched": "cell_id"}
    )

    cell_ids, _genes, M, col_idx = build_cell_gene_matrix(
        work, min_transcripts=5, genes_npm=panel, cell_col="cell_id",
        exclude_ids=set(UNASSIGNED_TOKENS),
    )
    npmi_mat, _gix = build_npmi_matrix(panel)
    _, _, _, pur_df = compute_cell_purity_relu(
        M=M, col_idx=col_idx, npmi_mat=npmi_mat, tau=tau, cell_ids=cell_ids
    )
    _, _, _, conf_df = compute_cell_conflict_relu(
        M=M, col_idx=col_idx, npmi_mat=npmi_mat, tau=tau, cell_ids=cell_ids
    )
    scores = pur_df.rename(columns={"cell_purity_relu": "purity_score"})[
        ["cell_id", "purity_score", "signal_strength", "relative_purity", "relative_conflict"]
    ].merge(
        conf_df.rename(columns={"cell_conflict_relu": "conflict_score"})[["cell_id", "conflict_score"]],
        on="cell_id", how="outer",
    )

    cg = work.groupby(["cell_id", "feature_name"], observed=True).size().rename("count").reset_index()
    cell_cat = pd.Categorical(cg["cell_id"])
    gene_cat = pd.Categorical(cg["feature_name"])
    X = sp.csr_matrix(
        (cg["count"].to_numpy(np.int32), (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(gene_cat.categories)),
    )
    obs = scores.set_index("cell_id").reindex(cell_cat.categories.astype(str))
    var = pd.DataFrame(index=pd.Index(gene_cat.categories.astype(str), name="feature_name"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()

    stem = output_path.with_suffix("")
    h5ad_path = Path(f"{stem}_cell_by_gene.h5ad")
    scores_path = Path(f"{stem}_cell_scores.tsv.gz")
    adata.write_h5ad(h5ad_path)
    scores.to_csv(scores_path, sep="\t", index=False, compression="gzip")
    print(f"Wrote {h5ad_path} ({adata.n_obs:,} cells) and {scores_path}")


def main() -> None:
    args = _parse_args()
    log = _setup_logging()

    from tracer.core import set_reproducibility_seed
    import tracer.pipeline as pipeline
    from tracer.config import load_config

    os.environ.setdefault("TRACER_STAGE_VERBOSE", "1")
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    set_reproducibility_seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Reading transcripts from: {input_path}")
    df = load_transcripts(input_path)
    print(f"Reading NPMI panel from: {args.npmi}")
    panel = load_npmi_panel(Path(args.npmi))

    cfg = load_config(path=args.user_config, platform=args.platform)
    if args.pmi_threshold is not None:
        # PMI prune threshold lives at module level for legacy reasons; the
        # pipeline reads it as the Stage-1 prune threshold.
        pipeline.PMI_THR = float(args.pmi_threshold)
        log.info("PMI threshold override: pipeline.PMI_THR = %.4f", pipeline.PMI_THR)

    log.info("Calling run_segmented_pipeline (df=%d rows, panel=%d pairs, platform=%s)",
             len(df), len(panel), args.platform)
    disarm = _install_watchdog(args.max_runtime_sec, log)
    t0 = time.time()
    try:
        df_out, progression = pipeline.run_segmented_pipeline(df=df, npmi_panel=panel, cfg=cfg)
    except _RuntimeExceeded as exc:
        log.error("%s", exc)
        raise SystemExit(2)
    finally:
        disarm()
    wall = time.time() - t0

    for s in progression:
        log.info(
            "[stage] %-22s cells=%-7s partials=%-7s components=%-7s unassigned=%-9s %.2fs",
            s.get("stage", ""), f"{s.get('n_cells', 0):,}",
            f"{s.get('n_partials', 0):,}", f"{s.get('n_components', 0):,}",
            f"{s.get('n_unassigned_tx', 0):,}", (s.get("stage_seconds") or 0.0),
        )
    log.info("TRACER done — %d stages, output rows=%d, wall=%.1fs",
             len(progression), len(df_out), wall)

    # Final per-transcript label is `stitched`; expose legacy aliases so the
    # downstream GBM scripts keep working unchanged.
    df_out["cell_id_finetuned"] = df_out["stitched"].astype(str)
    df_out["cell_id_stitched"] = df_out["stitched"].astype(str)

    n_cells = df_out.loc[~df_out["stitched"].astype(str).isin(UNASSIGNED_TOKENS), "stitched"].nunique()
    log.info("Final entities (stitched): %d", n_cells)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    log.info("Saved refined transcripts to: %s", output_path)

    if args.emit_cell_outputs:
        _emit_cell_outputs(df_out, panel, output_path)


if __name__ == "__main__":
    main()
