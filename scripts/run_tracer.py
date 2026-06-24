#!/usr/bin/env python3
"""Slim TRACER runner: standardized transcripts + NPMI panel → refined transcripts.

This is the production entry point introduced in the 2026-05-27 refactor.
Replaces the previous kitchen-sink ``run_tracer.py`` (kept as
``scripts/run_tracer_legacy.py`` for reference). The new script does ONE thing:
load inputs, run the canonical TRACER pipeline, write the standard outputs.

What this script DOES:
    - Load a standardized transcripts parquet (produced by preprocess_xenium.py).
    - Load an NPMI panel csv(.gz) (produced by build_npmi_from_scrna.py).
    - Load + override platform config (tracer.config.load_config).
    - Run ``tracer.pipeline.run_segmented_pipeline``.
    - Compute per-cell purity/conflict via tracer.metrics.
    - Emit:
        outputs/transcripts_tracer_refined.parquet
        outputs/cell_by_gene_tracer.h5ad
        outputs/cell_scores.tsv.gz
        run_summary.md
        runtime_memory.json
        config_receipt.json

What this script DOES NOT DO:
    - Compute NPMI (use scripts/build_npmi_from_scrna.py).
    - Run ovrlpy (use scripts/run_ovrlpy.py).
    - Run RCTD (use scripts/run_rctd.R).
    - Label transfer (use scripts/label_transfer_spatial.py).
    - Compute benchmark metrics (use scripts/get_metric.py).

These were the responsibilities of the legacy script. Each is now a
separate, composable module.

EXAMPLE
=======
::

    python scripts/run_tracer.py \\
      --transcripts datasets/lung_cancer_xenium_10x/filtered_df.parquet \\
      --npmi results/reference_npmi/lung_cancer_npmi.csv.gz \\
      --pmi-threshold 0.2 \\
      --platform xenium \\
      --outdir results/tracer/lung_xenium \\
      --sample-name lung_xenium --seed 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import socket
import subprocess
import sys
import time
import platform as _platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — let the script run from any cwd.
# ---------------------------------------------------------------------------
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
                   help="Standardized transcripts parquet from preprocess_xenium.py.")
    p.add_argument("--npmi", required=True, type=Path,
                   help="NPMI panel csv(.gz) from build_npmi_from_scrna.py.")
    p.add_argument("--pmi-threshold", type=float, default=None,
                   help="Override the in-pipeline PMI threshold (default: from "
                        "platform/user config).")
    p.add_argument("--platform", default="xenium",
                   help="Platform preset name (matches src/tracer/configs/platforms/<name>.toml).")
    p.add_argument("--defaults-config", type=Path, default=None,
                   help="Documentation/provenance — currently informational.")
    p.add_argument("--platform-config", type=Path, default=None,
                   help="Documentation/provenance — currently informational.")
    p.add_argument("--user-config", type=Path, default=None,
                   help="Optional user-override TOML on top of defaults+platform.")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--sample-name", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--min-tx-per-cell-for-scores", type=int, default=5,
                   help="Min transcripts/cell for cell-level purity/conflict scoring.")
    p.add_argument("--tau", type=float, default=0.05,
                   help="NPMI threshold for purity/conflict relu (default 0.05).")
    p.add_argument("--overwrite", action="store_true",
                   help="If outdir exists, overwrite contents.")
    return p


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(outdir: Path) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("run_tracer")
    log.setLevel(logging.INFO)
    log.propagate = False
    if log.handlers:
        return log
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(outdir / "run.log", mode="a"); fh.setFormatter(fmt)
    log.addHandler(sh); log.addHandler(fh)
    return log


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Runtime accounting
# ---------------------------------------------------------------------------
@dataclass
class StageTime:
    name: str
    seconds: float
    peak_rss_gb: float


class Timer:
    def __init__(self, log: logging.Logger):
        self.log = log
        self.stages: list[StageTime] = []

    def time(self, name: str):
        return _StageCtx(name, self.log, self)


class _StageCtx:
    def __init__(self, name: str, log: logging.Logger, timer: Timer):
        self.name = name; self.log = log; self.timer = timer; self.t0 = 0.0
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.log.info("[stage start] %s", self.name)
        return self
    def __exit__(self, *exc):
        secs = time.perf_counter() - self.t0
        rss = _rss_gb()
        self.timer.stages.append(StageTime(self.name, secs, rss))
        self.log.info("[stage done]  %s — %.2fs  peak_rss=%.2f GB",
                      self.name, secs, rss)


def _rss_gb() -> float:
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024 ** 3)
    except Exception:
        try:
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return (r if sys.platform == "darwin" else r * 1024) / (1024 ** 3)
        except Exception:
            return float("nan")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_transcripts(path: Path, log: logging.Logger) -> pd.DataFrame:
    log.info("Loading transcripts: %s", path)
    df = pd.read_parquet(path)
    required = {"x", "y", "feature_name", "cell_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Transcripts parquet missing required columns {missing}. "
            f"Use scripts/preprocess_xenium.py to standardize first. "
            f"Present columns: {sorted(df.columns)}"
        )
    if "z" not in df.columns:
        df["z"] = np.float32(0.0)
    if "overlaps_nucleus" not in df.columns:
        df["overlaps_nucleus"] = np.uint8(0)
    if "transcript_id" not in df.columns:
        df["transcript_id"] = np.arange(len(df), dtype=np.int64)
    df["cell_id"] = df["cell_id"].astype(str)
    for c in ("x", "y", "z"):
        df[c] = df[c].astype(np.float32)
    df["feature_name"] = df["feature_name"].astype(str)
    df["overlaps_nucleus"] = df["overlaps_nucleus"].astype(np.uint8)
    n_assigned = int((df["cell_id"] != "-1").sum())
    log.info("Loaded: %d rows, %d genes, assigned=%d, unassigned=%d",
             len(df), df["feature_name"].nunique(),
             n_assigned, len(df) - n_assigned)
    return df


def load_npmi_panel(path: Path, log: logging.Logger) -> pd.DataFrame:
    log.info("Loading NPMI panel: %s", path)
    df = pd.read_csv(path)
    if not {"gene_i", "gene_j"}.issubset(df.columns):
        raise SystemExit(
            f"NPMI panel missing gene_i/gene_j; columns: {list(df.columns)}"
        )
    # Pipeline expects long-format with both directions per pair.
    if (df.duplicated(["gene_i", "gene_j"]).any()):
        log.warning("NPMI panel has duplicate pairs — keeping first occurrence.")
        df = df.drop_duplicates(["gene_i", "gene_j"], keep="first")
    # Emit symmetric form (i, j) and (j, i) so downstream lookups work.
    rev = df.copy()
    rev["gene_i"], rev["gene_j"] = df["gene_j"].values, df["gene_i"].values
    panel = pd.concat([df, rev], ignore_index=True)
    # Drop self-pairs if present (i == j).
    panel = panel.loc[panel["gene_i"] != panel["gene_j"]].reset_index(drop=True)
    log.info("NPMI panel: %d rows after symmetric expansion; PMI: %s, NPMI: %s",
             len(panel),
             "yes" if "PMI" in panel.columns else "no",
             "yes" if "NPMI" in panel.columns else "no")
    return panel


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_tracer(df: pd.DataFrame, panel: pd.DataFrame, *,
               platform_name: str, user_config: Path | None,
               pmi_threshold_override: float | None,
               log: logging.Logger):
    """Apply config + invoke the canonical SEG pipeline."""
    from tracer.config import load_config
    import tracer.pipeline as pipeline

    cfg = load_config(path=user_config, platform=platform_name)
    if pmi_threshold_override is not None:
        # PMI threshold lives at the module level (`pipeline.PMI_THR`) for
        # legacy reasons; mirror it onto the config so the receipt reflects
        # the override.
        pipeline.PMI_THR = float(pmi_threshold_override)
        log.info("PMI threshold override: pipeline.PMI_THR = %.4f",
                 pipeline.PMI_THR)
    log.info("Calling run_segmented_pipeline (df=%d rows, panel=%d pairs)",
             len(df), len(panel))
    os.environ.setdefault("TRACER_STAGE_VERBOSE", "1")
    df_out, progression = pipeline.run_segmented_pipeline(
        df=df, npmi_panel=panel, cfg=cfg,
    )
    log.info("TRACER done — %d final stages; output rows=%d",
             len(progression), len(df_out))
    return df_out, progression, cfg


# ---------------------------------------------------------------------------
# Per-cell scores + cell-by-gene
# ---------------------------------------------------------------------------
UNASSIGNED_TOKENS = frozenset({
    "UNASSIGNED", "Unassigned", "unassigned",
    "DROP", "nan", "None", "", "0", "-1", "NA",
})


def build_outputs(
    df_post: pd.DataFrame, *,
    npmi_panel: pd.DataFrame, log: logging.Logger,
    label_col: str = "stitched", min_tx: int = 5, tau: float = 0.05,
) -> tuple[pd.DataFrame, "anndata.AnnData"]:
    """Compute per-cell purity/conflict + build cell-by-gene AnnData."""
    import anndata as ad
    import scipy.sparse as sp
    from tracer.metrics import (
        build_cell_gene_matrix, build_npmi_matrix,
        compute_cell_purity_relu, compute_cell_conflict_relu,
    )

    if "_etype" in df_post.columns:
        keep_mask = df_post["_etype"].astype(str).isin({"cell", "partial", "component"})
    else:
        keep_mask = ~df_post[label_col].astype(str).isin(UNASSIGNED_TOKENS)
    work = df_post.loc[keep_mask, [label_col, "feature_name", "x", "y", "z"]].copy()
    work = work.rename(columns={label_col: "cell_id"})

    cell_ids, _genes_cell, M, col_idx = build_cell_gene_matrix(
        work, min_transcripts=min_tx, genes_npm=npmi_panel,
        cell_col="cell_id", exclude_ids=set(UNASSIGNED_TOKENS),
    )
    npmi_mat, _gix = build_npmi_matrix(npmi_panel)
    _, _, _, pur_df = compute_cell_purity_relu(
        M=M, col_idx=col_idx, npmi_mat=npmi_mat, tau=tau, cell_ids=cell_ids,
    )
    _, _, _, conf_df = compute_cell_conflict_relu(
        M=M, col_idx=col_idx, npmi_mat=npmi_mat, tau=tau, cell_ids=cell_ids,
    )
    scores = (
        pur_df.rename(columns={"cell_purity_relu": "purity_score"})
              [["cell_id", "purity_score", "signal_strength",
                "relative_purity", "relative_conflict"]]
        .merge(
            conf_df.rename(columns={"cell_conflict_relu": "conflict_score"})
                   [["cell_id", "conflict_score"]],
            on="cell_id", how="outer",
        )
    )
    log.info("Per-cell scores: %d cells with purity, %d cells total in cell-by-gene",
             int(scores["purity_score"].notna().sum()), len(cell_ids))

    # Cell-by-gene AnnData (counts layer + score obs).
    cg = (
        work.groupby(["cell_id", "feature_name"], observed=True).size()
            .rename("count").reset_index()
    )
    cell_cat = pd.Categorical(cg["cell_id"])
    gene_cat = pd.Categorical(cg["feature_name"])
    X = sp.csr_matrix(
        (cg["count"].to_numpy(dtype=np.int32),
         (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(gene_cat.categories)),
    )
    obs = scores.set_index("cell_id").reindex(cell_cat.categories.astype(str))
    var = pd.DataFrame(index=pd.Index(gene_cat.categories.astype(str),
                                       name="feature_name"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return scores, adata


# ---------------------------------------------------------------------------
# Output dump
# ---------------------------------------------------------------------------
def write_outputs(
    df_post: pd.DataFrame, scores: pd.DataFrame, adata, *,
    outdir: Path, sample_name: str, args, cfg, panel_path: Path,
    transcripts_path: Path, progression: list[dict[str, Any]],
    timer: Timer, log: logging.Logger,
) -> None:
    outputs = outdir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    df_post.to_parquet(outputs / "transcripts_tracer_refined.parquet",
                       index=False, compression="snappy")
    adata.write_h5ad(outputs / "cell_by_gene_tracer.h5ad")
    scores.to_csv(outputs / "cell_scores.tsv.gz", sep="\t", index=False,
                  compression="gzip")
    log.info("Wrote outputs to %s/outputs/", outdir)

    # config_receipt.json
    from tracer.config import to_dict as cfg_to_dict
    receipt = {
        "command": " ".join(sys.argv),
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
        "sample_name": sample_name,
        "platform_name": args.platform,
        "config": cfg_to_dict(cfg),
        "inputs": {
            "transcripts": str(transcripts_path),
            "transcripts_sha1": file_sha1(transcripts_path),
            "transcripts_rows": int(len(df_post)),
            "npmi": str(panel_path),
            "npmi_sha1": file_sha1(panel_path),
        },
        "host": {
            "hostname": socket.gethostname(),
            "python": sys.version.split()[0],
            "platform": _platform.platform(),
            "executable": sys.executable,
        },
        "git_commit": git_commit_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(outdir / "config_receipt.json", "w") as f:
        json.dump(receipt, f, indent=2, default=str)

    # runtime_memory.json
    rm = {
        "sample_name": sample_name,
        "stages": [asdict(s) for s in timer.stages],
        "total_seconds": float(sum(s.seconds for s in timer.stages)),
        "peak_rss_gb_observed": float(max((s.peak_rss_gb for s in timer.stages), default=0.0)),
    }
    with open(outdir / "runtime_memory.json", "w") as f:
        json.dump(rm, f, indent=2)

    # run_summary.md
    md_lines = [
        f"# TRACER run summary — {sample_name}",
        "",
        f"- Date (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Platform preset: `{args.platform}`",
        f"- Git commit: `{git_commit_hash()}`",
        f"- Seed: {args.seed}",
        f"- PMI threshold override: {args.pmi_threshold}",
        f"- Transcripts: `{transcripts_path}` ({len(df_post):,} final rows)",
        f"- NPMI panel: `{panel_path}`",
        "",
        "## Stage progression",
        "",
        "| Stage | n_cells | n_partials | n_components | n_unassigned_tx | seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in progression:
        md_lines.append(
            f"| {s.get('stage', '')} | "
            f"{s.get('n_cells', 0):,} | {s.get('n_partials', 0):,} | "
            f"{s.get('n_components', 0):,} | {s.get('n_unassigned_tx', 0):,} | "
            f"{(s.get('stage_seconds') or 0):.2f} |"
        )
    md_lines += [
        "",
        "## Top-level outputs",
        "",
        "- `outputs/transcripts_tracer_refined.parquet`",
        "- `outputs/cell_by_gene_tracer.h5ad`",
        "- `outputs/cell_scores.tsv.gz`",
        "",
        "## Runtime",
        "",
        f"- Total wall time: {rm['total_seconds']:.1f} s",
        f"- Peak RSS observed: {rm['peak_rss_gb_observed']:.2f} GB",
    ]
    with open(outdir / "run_summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    log.info("Wrote run_summary.md, config_receipt.json, runtime_memory.json")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_argparser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if any(args.outdir.iterdir()) and not args.overwrite:
        # Allow re-runs: only block when the canonical outputs already exist.
        sentinel = args.outdir / "outputs" / "transcripts_tracer_refined.parquet"
        if sentinel.exists():
            raise SystemExit(
                f"outdir {args.outdir} already contains a TRACER run "
                f"({sentinel}). Pass --overwrite to replace it."
            )

    log = setup_logging(args.outdir)
    log.info("=== run_tracer.py ===")
    log.info("Sample: %s; platform: %s; seed: %d",
             args.sample_name, args.platform, args.seed)

    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    timer = Timer(log)

    with timer.time("load_transcripts"):
        df = load_transcripts(args.transcripts, log)
    with timer.time("load_npmi"):
        panel = load_npmi_panel(args.npmi, log)
    with timer.time("run_pipeline"):
        df_post, progression, cfg = run_tracer(
            df, panel,
            platform_name=args.platform,
            user_config=args.user_config,
            pmi_threshold_override=args.pmi_threshold,
            log=log,
        )
    with timer.time("build_outputs"):
        scores, adata = build_outputs(
            df_post, npmi_panel=panel, log=log,
            label_col="stitched",
            min_tx=args.min_tx_per_cell_for_scores, tau=args.tau,
        )
    with timer.time("write_outputs"):
        write_outputs(
            df_post, scores, adata,
            outdir=args.outdir, sample_name=args.sample_name,
            args=args, cfg=cfg,
            panel_path=args.npmi, transcripts_path=args.transcripts,
            progression=progression, timer=timer, log=log,
        )

    log.info("DONE. Total wall: %.1fs",
             sum(s.seconds for s in timer.stages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
