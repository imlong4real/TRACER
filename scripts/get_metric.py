#!/usr/bin/env python3
"""Method-agnostic benchmark metrics for spatial-segmentation outputs.

Given the transcript-level parquet of ANY segmentation method (raw Xenium,
TRACER, Baysor, proseg, Segger, SPLIT/RCTD, cellAdmix, or any custom
method with a ``cell_id``/``stitched`` column), compute the standardized
benchmark metrics needed for the TRACER publication and per-method
comparison.

Metric categories (each runs independently; failures are isolated):

  A. Runtime / memory       — parse --runtime-json or accept missing
  B. Transcript assignment  — pre/post counts of assigned vs unassigned
  C. Cell / partial counts  — full cells, partial cells, components
  D. Cell QC                — tx/cell, genes/cell, % mito, purity/conflict
  E. Label transfer         — KNN transfer from scRNA reference (same
                              model for pre + post, gene set, seed)
  F. Reference consistency  — per-celltype pseudo-bulk Pearson r vs scRNA
  G. Marker specificity     — top-N markers per celltype, log2FC, paired
                              Wilcoxon (one-sided) original vs TRACER
  H. NPMI coherence         — purity/conflict if --npmi provided
  I. Benchmark summary      — aggregate into method_summary.json +
                              benchmark_summary.tsv

USAGE
=====
For TRACER-refined output::

    python scripts/get_metric.py --method TRACER \\
      --transcripts results/tracer/lung_xenium/outputs/transcripts_tracer_refined.parquet \\
      --original   datasets/lung_cancer_xenium_10x/filtered_df.parquet \\
      --reference-h5ad <scRNA h5ad> \\
      --outdir results/benchmark/lung_xenium/TRACER \\
      --min-transcripts-per-cell 10 --max-transcripts-per-cell 900

For the raw segmentation::

    python scripts/get_metric.py --method original \\
      --transcripts datasets/lung_cancer_xenium_10x/filtered_df.parquet \\
      --reference-h5ad <scRNA h5ad> \\
      --outdir results/benchmark/lung_xenium/original \\
      --min-transcripts-per-cell 10 --max-transcripts-per-cell 900
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


UNASSIGNED_TOKENS = frozenset({
    "UNASSIGNED", "Unassigned", "unassigned",
    "DROP", "nan", "None", "", "0", "-1", "NA",
})


# ===========================================================================
# Logging
# ===========================================================================
def setup_logging(outdir: Path) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("get_metric")
    log.setLevel(logging.INFO)
    log.propagate = False
    if log.handlers:
        return log
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s :: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(outdir / "get_metric.log", mode="a"); fh.setFormatter(fmt)
    log.addHandler(sh); log.addHandler(fh)
    return log


# ===========================================================================
# CLI
# ===========================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--method", required=True,
                   help="Method label for outputs (TRACER, original, Baysor, proseg, ...).")
    p.add_argument("--transcripts", required=True, type=Path,
                   help="Method-output transcripts parquet.")
    p.add_argument("--label-col", default=None,
                   help="Column to use as final cell label (default: auto-detect "
                        "'stitched' then 'cell_id').")
    p.add_argument("--original", type=Path, default=None,
                   help="Optional: pre-method transcripts parquet (enables pre/post).")
    p.add_argument("--transcripts-pre", dest="transcripts_pre_deprecated",
                   type=Path, default=None,
                   help="DEPRECATED alias for --original. Will be removed.")
    p.add_argument("--original-label-col", default="cell_id",
                   help="Label column on --original (default: cell_id).")
    p.add_argument("--reference-h5ad", required=True, type=Path)
    p.add_argument("--reference-celltype-col", default="cell_type_harmonized",
                   help="obs column carrying the reference cell-type label.")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--npmi", type=Path, default=None,
                   help="Optional NPMI csv(.gz) for purity/conflict + coherence.")
    p.add_argument("--runtime-json", type=Path, default=None,
                   help="Optional runtime/memory JSON to copy into outdir.")
    p.add_argument("--reference-marker-table", type=Path, default=None,
                   help="Optional precomputed marker table (tsv with cell_type, gene, "
                        "rank). If absent, markers are computed from reference h5ad.")
    p.add_argument("--min-transcripts-per-cell", type=int, default=10)
    p.add_argument("--max-transcripts-per-cell", type=int, default=900)
    p.add_argument("--n-top-markers", type=int, default=30)
    p.add_argument("--n-anchors-per-celltype", type=int, default=300,
                   help="Reference anchor cells per celltype for KNN label transfer.")
    p.add_argument("--knn-k", type=int, default=15,
                   help="K for KNN label transfer.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip", default="",
                   help="Comma-separated metric categories to skip (A,B,C,D,E,F,G,H).")
    p.add_argument("--platform", default=None,
                   help="Optional platform tag for the summary table.")
    p.add_argument("--dataset-name", default=None,
                   help="Optional dataset identifier for the summary table.")
    return p


# ===========================================================================
# Inputs
# ===========================================================================
def detect_label_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise SystemExit(f"--label-col {requested!r} not in transcripts.")
        return requested
    for c in ("stitched", "cell_id"):
        if c in df.columns:
            return c
    raise SystemExit(
        f"Could not detect label column. Pass --label-col explicitly. "
        f"Columns present: {list(df.columns)}"
    )


def load_transcripts(path: Path, *, log: logging.Logger) -> pd.DataFrame:
    log.info("Loading transcripts: %s", path)
    df = pd.read_parquet(path)
    for c in ("feature_name",):
        if c not in df.columns:
            raise SystemExit(f"transcripts parquet missing {c!r}; cols={list(df.columns)}")
    df["feature_name"] = df["feature_name"].astype(str)
    return df


# ===========================================================================
# A. Runtime / memory
# ===========================================================================
def metric_runtime(args, outdir: Path, log: logging.Logger) -> dict[str, Any]:
    if args.runtime_json is None:
        log.warning("[A runtime] no --runtime-json provided; emitting placeholder.")
        out = {
            "method": args.method, "runtime_seconds": None,
            "peak_memory_gb": None, "source": "not_provided",
        }
    else:
        with open(args.runtime_json) as f:
            data = json.load(f)
        # Recognize the run_tracer.py format: {"stages": [...], "total_seconds": ...}
        if isinstance(data, dict) and "stages" in data:
            out = {
                "method": args.method,
                "runtime_seconds": float(data.get("total_seconds", 0.0)),
                "peak_memory_gb": float(data.get("peak_rss_gb_observed", 0.0)),
                "stages": data["stages"],
                "source": str(args.runtime_json),
            }
        else:
            out = dict(data)
            out.setdefault("method", args.method)
            out["source"] = str(args.runtime_json)
    with open(outdir / "runtime_memory.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("[A runtime] wrote runtime_memory.json")
    return out


# ===========================================================================
# B. Transcript assignment
# ===========================================================================
def metric_assignment(
    df_post: pd.DataFrame, label_col: str,
    df_pre: pd.DataFrame | None, pre_label_col: str | None,
    *, outdir: Path, log: logging.Logger,
) -> dict[str, Any]:
    post_labels = df_post[label_col].astype(str)
    is_assigned_post = ~post_labels.isin(UNASSIGNED_TOKENS)
    if "_etype" in df_post.columns:
        is_assigned_post = df_post["_etype"].astype(str).isin(
            {"cell", "partial", "component"})
    n_total_post = int(len(df_post))
    n_assigned_post = int(is_assigned_post.sum())
    rows = [
        ("n_total_transcripts_post", n_total_post),
        ("n_assigned_transcripts_post", n_assigned_post),
        ("n_unassigned_transcripts_post", n_total_post - n_assigned_post),
        ("frac_assigned_post", n_assigned_post / max(1, n_total_post)),
    ]
    pre_summary = {}
    if df_pre is not None:
        pre_labels = df_pre[pre_label_col].astype(str)
        is_assigned_pre = ~pre_labels.isin(UNASSIGNED_TOKENS)
        n_total_pre = int(len(df_pre))
        n_assigned_pre = int(is_assigned_pre.sum())
        rows += [
            ("n_total_transcripts_pre", n_total_pre),
            ("n_assigned_transcripts_pre", n_assigned_pre),
            ("n_unassigned_transcripts_pre", n_total_pre - n_assigned_pre),
            ("frac_assigned_pre", n_assigned_pre / max(1, n_total_pre)),
            ("delta_assigned_post_minus_pre", n_assigned_post - n_assigned_pre),
            ("delta_unassigned_post_minus_pre",
             (n_total_post - n_assigned_post) - (n_total_pre - n_assigned_pre)),
        ]
        pre_summary = {
            "n_total_transcripts_pre": n_total_pre,
            "n_assigned_transcripts_pre": n_assigned_pre,
        }
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(
        outdir / "transcript_assignment.tsv", sep="\t", index=False,
    )
    log.info("[B assignment] post assigned=%d / %d (%.1f%%); pre %s",
             n_assigned_post, n_total_post,
             100 * n_assigned_post / max(1, n_total_post),
             "n/a" if df_pre is None else
             f"assigned={pre_summary['n_assigned_transcripts_pre']} / {pre_summary['n_total_transcripts_pre']}")
    return dict(rows)


# ===========================================================================
# C. Cell / partial-cell counts
# ===========================================================================
def metric_cell_counts(
    df_post: pd.DataFrame, label_col: str, *,
    min_tx: int, max_tx: int, outdir: Path, log: logging.Logger,
) -> dict[str, Any]:
    sub = df_post.loc[~df_post[label_col].astype(str).isin(UNASSIGNED_TOKENS)].copy()
    counts = sub[label_col].astype(str).value_counts()
    # Etype counts when available
    n_cells = n_partials = n_components = None
    if "_etype" in df_post.columns:
        et = (
            df_post.loc[~df_post[label_col].astype(str).isin(UNASSIGNED_TOKENS),
                        ["_etype", label_col]]
            .drop_duplicates(label_col)
        )
        n_cells = int((et["_etype"] == "cell").sum())
        n_partials = int((et["_etype"] == "partial").sum())
        n_components = int((et["_etype"] == "component").sum())
    n_total_entities = int(len(counts))

    kept = counts[(counts >= min_tx) & (counts <= max_tx)]
    n_kept = int(len(kept))
    rows = [
        ("n_total_entities_pre_filter", n_total_entities),
        ("n_total_entities_post_filter", n_kept),
        ("min_transcripts_per_cell", int(min_tx)),
        ("max_transcripts_per_cell", int(max_tx)),
        ("n_cells", n_cells),
        ("n_partials", n_partials),
        ("n_components", n_components),
        ("median_tx_per_entity_pre_filter",
         float(counts.median()) if len(counts) else 0.0),
        ("median_tx_per_entity_post_filter",
         float(kept.median()) if len(kept) else 0.0),
    ]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(
        outdir / "cell_count_summary.tsv", sep="\t", index=False,
    )
    log.info("[C cell counts] entities pre=%d, post-filter=%d "
             "(min=%d, max=%d). cells=%s partials=%s components=%s",
             n_total_entities, n_kept, min_tx, max_tx,
             n_cells, n_partials, n_components)
    return {
        "n_total_entities_pre_filter": n_total_entities,
        "n_total_entities_post_filter": n_kept,
        "n_cells": n_cells, "n_partials": n_partials, "n_components": n_components,
        "kept_cell_ids": set(kept.index.astype(str)),
    }


# ===========================================================================
# Cell × gene matrix builder (used by D / E / F / G / H)
# ===========================================================================
def build_cellxgene(
    df: pd.DataFrame, label_col: str, *,
    keep_ids: set[str] | None = None,
    log: logging.Logger,
):
    """Build a cells × genes counts AnnData restricted to keep_ids (if given)."""
    import anndata as ad
    sub = df.loc[~df[label_col].astype(str).isin(UNASSIGNED_TOKENS)].copy()
    if keep_ids is not None:
        sub = sub.loc[sub[label_col].astype(str).isin(keep_ids)]
    cg = (
        sub.groupby([label_col, "feature_name"], observed=True).size()
           .rename("count").reset_index()
    )
    cell_cat = pd.Categorical(cg[label_col].astype(str))
    gene_cat = pd.Categorical(cg["feature_name"].astype(str))
    X = sp.csr_matrix(
        (cg["count"].to_numpy(dtype=np.int32),
         (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(gene_cat.categories)),
    )
    obs = pd.DataFrame(index=pd.Index(cell_cat.categories.astype(str), name="cell_id"))
    var = pd.DataFrame(index=pd.Index(gene_cat.categories.astype(str), name="feature_name"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    log.info("Built cell×gene matrix: %d cells × %d genes", adata.n_obs, adata.n_vars)
    return adata


# ===========================================================================
# D. Cell QC
# ===========================================================================
def metric_cell_qc(
    adata, *, outdir: Path, log: logging.Logger,
    npmi_panel: pd.DataFrame | None = None,
    tau: float = 0.05,
) -> dict[str, Any]:
    counts = np.asarray(adata.X.sum(axis=1)).ravel()
    nz = adata.X > 0
    n_genes = np.asarray(nz.sum(axis=1)).ravel()

    # Mito gene fraction (MT- prefix). Skip if none in panel.
    mito_mask = np.array([str(g).upper().startswith("MT-")
                          for g in adata.var_names], dtype=bool)
    if mito_mask.any():
        mito_counts = np.asarray(adata.X[:, mito_mask].sum(axis=1)).ravel()
        pct_mito = 100 * mito_counts / np.maximum(counts, 1)
    else:
        pct_mito = np.full(adata.n_obs, np.nan)

    # Optional: NPMI-derived purity / conflict scores.
    purity_score = conflict_score = rel_pur = rel_conf = signal = None
    if npmi_panel is not None and len(adata) > 0:
        purity_score, conflict_score, rel_pur, rel_conf, signal = _compute_purity_conflict(
            adata, npmi_panel, tau=tau, log=log,
        )

    per_cell = pd.DataFrame({
        "cell_id": adata.obs_names,
        "n_transcripts": counts.astype(np.int64),
        "n_genes_detected": n_genes.astype(np.int64),
        "pct_mito": pct_mito.astype(np.float32),
    })
    if purity_score is not None:
        per_cell["purity_score"] = purity_score
        per_cell["conflict_score"] = conflict_score
        per_cell["relative_purity"] = rel_pur
        per_cell["relative_conflict"] = rel_conf
        per_cell["signal_strength"] = signal
    per_cell.to_csv(outdir / "cell_qc.tsv", sep="\t", index=False)

    summary_rows = [
        ("n_cells_filtered", int(adata.n_obs)),
        ("median_transcripts_per_cell", float(np.median(counts))),
        ("median_genes_per_cell", float(np.median(n_genes))),
        ("mean_transcripts_per_cell", float(np.mean(counts))),
        ("mean_genes_per_cell", float(np.mean(n_genes))),
        ("median_pct_mito", float(np.nanmedian(pct_mito)) if mito_mask.any() else None),
    ]
    if purity_score is not None:
        summary_rows += [
            ("median_purity_score", float(np.nanmedian(purity_score))),
            ("median_conflict_score", float(np.nanmedian(conflict_score))),
            ("median_relative_purity", float(np.nanmedian(rel_pur))),
            ("median_relative_conflict", float(np.nanmedian(rel_conf))),
            ("median_signal_strength", float(np.nanmedian(signal))),
        ]
    pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(
        outdir / "cell_qc_summary.tsv", sep="\t", index=False,
    )
    log.info("[D cell QC] %d cells; median tx=%.0f; median genes=%.0f",
             adata.n_obs, np.median(counts), np.median(n_genes))
    return {r[0]: r[1] for r in summary_rows}


def _compute_purity_conflict(adata, npmi_panel, *, tau: float, log: logging.Logger):
    """Apply tracer.metrics relu purity/conflict to an AnnData."""
    from tracer.metrics import (
        build_cell_gene_matrix, build_npmi_matrix,
        compute_cell_purity_relu, compute_cell_conflict_relu,
    )
    # Reconstruct a transcripts-style df from the AnnData (one row per
    # nonzero entry, replicated `count` times for purity's tx-counting).
    coo = adata.X.tocoo()
    cell_ids_arr = np.asarray(adata.obs_names, dtype=str)
    genes_arr = np.asarray(adata.var_names, dtype=str)
    # Repeat rows by count for the matrix builder.
    rep = coo.data.astype(np.int64)
    cell_idx_rep = np.repeat(coo.row, rep)
    gene_idx_rep = np.repeat(coo.col, rep)
    df = pd.DataFrame({
        "cell_id": cell_ids_arr[cell_idx_rep],
        "feature_name": genes_arr[gene_idx_rep],
        "x": 0.0, "y": 0.0, "z": 0.0,
    })
    cids, _, M, col_idx = build_cell_gene_matrix(
        df, min_transcripts=1, genes_npm=npmi_panel,
        cell_col="cell_id",
    )
    npmi_mat, _ = build_npmi_matrix(npmi_panel)
    _, _, _, pur_df = compute_cell_purity_relu(
        M=M, col_idx=col_idx, npmi_mat=npmi_mat, tau=tau, cell_ids=cids,
    )
    _, _, _, conf_df = compute_cell_conflict_relu(
        M=M, col_idx=col_idx, npmi_mat=npmi_mat, tau=tau, cell_ids=cids,
    )
    # Align to AnnData obs order
    name_to_pos = {c: i for i, c in enumerate(adata.obs_names.astype(str))}
    out_pur = np.full(adata.n_obs, np.nan, dtype=np.float64)
    out_conf = np.full(adata.n_obs, np.nan, dtype=np.float64)
    out_rpur = np.full(adata.n_obs, np.nan, dtype=np.float64)
    out_rconf = np.full(adata.n_obs, np.nan, dtype=np.float64)
    out_sig = np.full(adata.n_obs, np.nan, dtype=np.float64)
    for _, row in pur_df.iterrows():
        pos = name_to_pos.get(str(row["cell_id"]))
        if pos is not None:
            out_pur[pos] = row["cell_purity_relu"]
            out_rpur[pos] = row["relative_purity"]
            out_rconf[pos] = row["relative_conflict"]
            out_sig[pos] = row["signal_strength"]
    for _, row in conf_df.iterrows():
        pos = name_to_pos.get(str(row["cell_id"]))
        if pos is not None:
            out_conf[pos] = row["cell_conflict_relu"]
    return out_pur, out_conf, out_rpur, out_rconf, out_sig


# ===========================================================================
# Reference scRNA loader (shared between E + F + G)
# ===========================================================================
@dataclass
class ReferenceData:
    counts_csr: sp.csr_matrix
    var_names: np.ndarray
    obs: pd.DataFrame
    celltype_col: str

    @property
    def n_cells(self) -> int:
        return self.counts_csr.shape[0]


def load_reference(h5ad_path: Path, celltype_col: str,
                   log: logging.Logger) -> ReferenceData:
    import anndata as ad
    log.info("Loading reference: %s", h5ad_path)
    a = ad.read_h5ad(h5ad_path)
    if celltype_col not in a.obs.columns:
        raise SystemExit(
            f"Reference h5ad has no obs column {celltype_col!r}. "
            f"Available: {list(a.obs.columns)[:20]}"
        )
    # Pick raw counts (prefer layers['counts']).
    if "counts" in a.layers:
        X = a.layers["counts"]
    elif a.raw is not None:
        X = a.raw.X
    else:
        X = a.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    log.info("  shape=%s; celltype_col=%s; %d celltypes",
             X.shape, celltype_col, a.obs[celltype_col].nunique())
    return ReferenceData(
        counts_csr=X.astype(np.float32),
        var_names=np.asarray(a.var_names, dtype=str),
        obs=a.obs.copy(), celltype_col=celltype_col,
    )


# ===========================================================================
# E. Label transfer (KNN-based; same model / anchors for pre + post)
# ===========================================================================
def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _log_normalize(X: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    """CPM-style normalization + log1p, returning a dense float32 array.
    Same as scanpy.pp.normalize_total + log1p but spelled out so it has
    a stable, dependency-light contract."""
    counts = np.asarray(X.sum(axis=1)).ravel()
    counts[counts == 0] = 1.0
    Xn = X.multiply(1.0 / counts[:, None]).multiply(target_sum)
    if sp.issparse(Xn):
        Xn = Xn.tocsr()
        Xn.data = np.log1p(Xn.data)
    else:
        Xn = np.log1p(Xn)
    return Xn


def _pick_anchors(ref: ReferenceData, *, per_type: int, seed: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample at most ``per_type`` reference cells per celltype."""
    labels = ref.obs[ref.celltype_col].astype(str).to_numpy()
    keep_idx = []
    keep_lab = []
    for ct in np.unique(labels):
        if ct in ("nan", "NaN", "None", ""):
            continue
        rows = np.where(labels == ct)[0]
        if len(rows) > per_type:
            rows = rng.choice(rows, size=per_type, replace=False)
        keep_idx.append(rows)
        keep_lab.append(np.full(len(rows), ct, dtype=object))
    if not keep_idx:
        raise SystemExit("No reference anchors found.")
    return np.concatenate(keep_idx), np.concatenate(keep_lab)


def transfer_labels(
    adata_query, ref: ReferenceData, *,
    seed: int, k: int, per_type: int, log: logging.Logger,
) -> pd.DataFrame:
    """KNN-based cell type transfer. Returns DataFrame with columns
    [cell_id, predicted_celltype, confidence, n_neighbors_voting]."""
    if adata_query.n_obs == 0:
        return pd.DataFrame(columns=["cell_id", "predicted_celltype",
                                       "confidence", "n_neighbors_voting"])

    # Restrict to genes present in both query and reference.
    q_genes = np.asarray(adata_query.var_names, dtype=str)
    r_genes = ref.var_names
    shared = np.intersect1d(q_genes, r_genes)
    if len(shared) < 5:
        raise SystemExit(
            f"Only {len(shared)} shared genes between query and reference."
        )
    log.info("[E label transfer] shared genes = %d", len(shared))

    q_gene_pos = {g: i for i, g in enumerate(q_genes)}
    r_gene_pos = {g: i for i, g in enumerate(r_genes)}
    q_idx = np.array([q_gene_pos[g] for g in shared], dtype=np.int64)
    r_idx = np.array([r_gene_pos[g] for g in shared], dtype=np.int64)

    # Build reference anchors (subsample per celltype).
    rng = np.random.default_rng(seed)
    anchor_rows, anchor_labels = _pick_anchors(
        ref, per_type=per_type, seed=seed, rng=rng,
    )
    ref_X_sub = ref.counts_csr[anchor_rows][:, r_idx].tocsr()
    log.info("  anchors: %d cells across %d celltypes",
             ref_X_sub.shape[0], len(np.unique(anchor_labels)))

    # Normalize + L2.
    ref_norm = _log_normalize(ref_X_sub)
    ref_dense = ref_norm.toarray() if sp.issparse(ref_norm) else ref_norm
    ref_l2 = _l2_normalize_rows(ref_dense.astype(np.float32))

    q_X = adata_query.X[:, q_idx]
    q_X = sp.csr_matrix(q_X) if not sp.issparse(q_X) else q_X.tocsr()
    q_norm = _log_normalize(q_X)
    q_dense = q_norm.toarray() if sp.issparse(q_norm) else q_norm
    q_l2 = _l2_normalize_rows(q_dense.astype(np.float32))

    # Cosine similarity = dot product on L2-normalized vectors.
    # NearestNeighbors handles up to ~100k cells in seconds.
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nbrs.fit(ref_l2)
    distances, indices = nbrs.kneighbors(q_l2)
    neighbor_labels = anchor_labels[indices]

    # Majority vote per query cell.
    pred = []
    conf = []
    n_vote = []
    for row in neighbor_labels:
        vals, counts = np.unique(row, return_counts=True)
        top = np.argmax(counts)
        pred.append(vals[top])
        conf.append(counts[top] / k)
        n_vote.append(int(counts[top]))
    out = pd.DataFrame({
        "cell_id": np.asarray(adata_query.obs_names, dtype=str),
        "predicted_celltype": pred,
        "confidence": np.asarray(conf, dtype=np.float32),
        "n_neighbors_voting": np.asarray(n_vote, dtype=np.int32),
    })
    log.info("[E label transfer] %d cells annotated; median confidence=%.2f",
             len(out), float(np.median(out["confidence"])))
    return out


# ===========================================================================
# F. Reference scRNA consistency (per-celltype pseudo-bulk Pearson)
# ===========================================================================
def metric_reference_consistency(
    adata_query, ann: pd.DataFrame, ref: ReferenceData,
    *, method: str, outdir: Path, log: logging.Logger,
    min_cells_per_type: int = 5,
) -> pd.DataFrame:
    from scipy.stats import pearsonr

    # Shared gene set
    q_genes = np.asarray(adata_query.var_names, dtype=str)
    r_genes = ref.var_names
    shared = np.intersect1d(q_genes, r_genes)
    q_pos = np.array([list(q_genes).index(g) for g in shared], dtype=np.int64)
    r_pos = np.array([list(r_genes).index(g) for g in shared], dtype=np.int64)
    log.info("[F consistency] shared genes = %d", len(shared))

    q_X = adata_query.X[:, q_pos]
    q_X = sp.csr_matrix(q_X) if not sp.issparse(q_X) else q_X.tocsr()
    q_norm = _log_normalize(q_X)
    q_dense = q_norm.toarray() if sp.issparse(q_norm) else q_norm

    r_X = ref.counts_csr[:, r_pos]
    r_X = sp.csr_matrix(r_X) if not sp.issparse(r_X) else r_X.tocsr()
    r_norm = _log_normalize(r_X)
    # Don't densify the whole reference matrix; mean by group using groupby on rows
    ref_labels = ref.obs[ref.celltype_col].astype(str).to_numpy()

    # Map query cell → predicted celltype
    name_to_ct = dict(zip(ann["cell_id"].astype(str), ann["predicted_celltype"]))
    q_labels = np.asarray([name_to_ct.get(c, None)
                            for c in adata_query.obs_names.astype(str)], dtype=object)

    rows = []
    celltypes = sorted({c for c in q_labels if c is not None and c == c})
    for ct in celltypes:
        q_mask = q_labels == ct
        r_mask = ref_labels == ct
        n_q = int(q_mask.sum())
        n_r = int(r_mask.sum())
        if n_q < min_cells_per_type or n_r < min_cells_per_type:
            continue
        q_bulk = q_dense[q_mask].mean(axis=0)
        # Reference pseudo-bulk: mean of log-normalized sparse rows
        rsel = r_norm[r_mask]
        r_bulk = np.asarray(rsel.mean(axis=0)).ravel()
        # Pearson on shared-gene vectors
        if np.std(q_bulk) == 0 or np.std(r_bulk) == 0:
            r_val, p_val = np.nan, np.nan
        else:
            r_val, p_val = pearsonr(q_bulk, r_bulk)
        rows.append({
            "method": method,
            "cell_type": ct,
            "n_spatial_cells": n_q,
            "n_reference_cells": n_r,
            "n_genes_used": int(len(shared)),
            "pearson_r": float(r_val),
            "pearson_p": float(p_val),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "reference_consistency_by_celltype.tsv",
              sep="\t", index=False)
    log.info("[F consistency] %d celltypes scored; median r = %.3f",
             len(df), float(df["pearson_r"].median()) if len(df) else float("nan"))
    return df


def plot_reference_consistency(df: pd.DataFrame, outdir: Path,
                                log: logging.Logger) -> None:
    if df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("[F plots] matplotlib unavailable.")
        return
    fig, ax = plt.subplots(figsize=(max(4, 0.4 * len(df)), 4))
    df_sorted = df.sort_values("pearson_r", ascending=True)
    ax.scatter(df_sorted["pearson_r"], range(len(df_sorted)),
               s=20 + df_sorted["n_spatial_cells"].clip(0, 200) / 5,
               c="steelblue", alpha=0.85)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["cell_type"], fontsize=8)
    ax.axvline(0, color="grey", ls="--", lw=0.5)
    ax.set_xlabel("Pearson r (spatial vs scRNA pseudo-bulk)")
    ax.set_title(f"Reference consistency — {df['method'].iloc[0]}")
    fig.tight_layout()
    fig.savefig(outdir / "reference_consistency_dotplot.png", dpi=150)
    fig.savefig(outdir / "reference_consistency_dotplot.pdf")
    plt.close(fig)


# ===========================================================================
# G. Marker specificity (log2FC)
# ===========================================================================
def compute_reference_markers(
    ref: ReferenceData, *, n_top: int,
    log: logging.Logger,
) -> pd.DataFrame:
    """Return a (cell_type, gene, rank, scrna_log2fc) table from scanpy
    rank_genes_groups (Wilcoxon) on the reference scRNA."""
    try:
        import scanpy as sc
    except ImportError:
        log.warning("[G markers] scanpy unavailable; cannot compute markers.")
        return pd.DataFrame(columns=["cell_type", "gene", "rank", "scrna_log2fc"])
    import anndata as ad
    a = ad.AnnData(
        X=ref.counts_csr.copy(),
        obs=ref.obs[[ref.celltype_col]].copy(),
        var=pd.DataFrame(index=pd.Index(ref.var_names, name="feature_name")),
    )
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    # Skip empty / "Patient*-specific" groups by minimum size
    counts = a.obs[ref.celltype_col].value_counts()
    keep = counts[counts >= 30].index.tolist()
    a = a[a.obs[ref.celltype_col].isin(keep)].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.rank_genes_groups(a, ref.celltype_col, method="wilcoxon")
    rows = []
    rg = a.uns["rank_genes_groups"]
    for ct in rg["names"].dtype.names:
        names = rg["names"][ct][:n_top]
        scores = rg["logfoldchanges"][ct][:n_top]
        for rank, (g, lfc) in enumerate(zip(names, scores), start=1):
            rows.append({"cell_type": ct, "gene": str(g), "rank": rank,
                          "scrna_log2fc": float(lfc)})
    df = pd.DataFrame(rows)
    log.info("[G markers] computed %d marker rows across %d celltypes",
             len(df), df["cell_type"].nunique() if len(df) else 0)
    return df


def metric_marker_specificity(
    adata_query, ann: pd.DataFrame, markers: pd.DataFrame, *,
    method: str, outdir: Path, log: logging.Logger,
    min_cells_per_type: int = 5,
) -> pd.DataFrame:
    """For each celltype, compute log2FC of top-N marker genes in that
    celltype vs all other celltypes in the SAME method."""
    if markers.empty:
        return pd.DataFrame()

    # log-normalize the query matrix
    X = sp.csr_matrix(adata_query.X) if not sp.issparse(adata_query.X) else adata_query.X.tocsr()
    Xn = _log_normalize(X)
    Xn_dense = Xn.toarray() if sp.issparse(Xn) else Xn
    var_names = np.asarray(adata_query.var_names, dtype=str)
    var_pos = {g: i for i, g in enumerate(var_names)}

    name_to_ct = dict(zip(ann["cell_id"].astype(str), ann["predicted_celltype"]))
    q_labels = np.asarray([name_to_ct.get(c, None)
                            for c in adata_query.obs_names.astype(str)], dtype=object)

    rows = []
    for ct in markers["cell_type"].unique():
        in_ct = q_labels == ct
        out_ct = (q_labels != ct) & np.array([x is not None and x == x
                                              for x in q_labels])
        n_in = int(in_ct.sum())
        n_out = int(out_ct.sum())
        if n_in < min_cells_per_type or n_out < min_cells_per_type:
            continue
        ct_markers = markers.loc[markers["cell_type"] == ct]
        for _, mr in ct_markers.iterrows():
            g = mr["gene"]
            if g not in var_pos:
                continue
            j = var_pos[g]
            mu_in = float(Xn_dense[in_ct, j].mean())
            mu_out = float(Xn_dense[out_ct, j].mean())
            # log2FC on log1p data — same as scanpy's reported logfoldchanges.
            l2fc = (mu_in - mu_out) / np.log(2.0)
            rows.append({
                "method": method, "cell_type": ct, "gene": g,
                "rank": int(mr["rank"]),
                "scrna_log2fc": float(mr["scrna_log2fc"]),
                "spatial_log2fc": float(l2fc),
                "spatial_mean_in": mu_in, "spatial_mean_out": mu_out,
                "n_cells_in_celltype": n_in, "n_cells_other": n_out,
            })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "marker_specificity_log2fc.tsv",
              sep="\t", index=False)
    log.info("[G specificity] %d marker rows across %d celltypes",
             len(df), df["cell_type"].nunique() if len(df) else 0)
    return df


def plot_marker_specificity(df: pd.DataFrame, outdir: Path,
                             log: logging.Logger) -> None:
    if df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    by_ct = df.groupby("cell_type")["spatial_log2fc"].median().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(max(4, 0.4 * len(by_ct)), 4))
    ax.scatter(by_ct.values, range(len(by_ct)), s=30, c="firebrick", alpha=0.85)
    ax.axvline(0, color="grey", ls="--", lw=0.5)
    ax.set_yticks(range(len(by_ct)))
    ax.set_yticklabels(by_ct.index, fontsize=8)
    ax.set_xlabel("Median spatial log2FC (markers vs other celltypes)")
    ax.set_title(f"Marker specificity — {df['method'].iloc[0]}")
    fig.tight_layout()
    fig.savefig(outdir / "marker_specificity_dotplot.png", dpi=150)
    fig.savefig(outdir / "marker_specificity_dotplot.pdf")
    plt.close(fig)


def metric_tcell_marker_log2fc(
    adata_query, ann: pd.DataFrame, *,
    method: str, outdir: Path, log: logging.Logger,
    tcell_pattern: tuple[str, ...] = ("T cells", "T_cells", "tT cells",
                                       "CD8 T", "CD4 T"),
    min_cells_per_type: int = 5,
) -> pd.DataFrame:
    """Compute log2FC of canonical T-cell markers in T-cell-classified
    spatial cells vs all others."""
    CANON_TCELL_MARKERS = [
        "CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4",
        "TRAC", "TRBC1", "TRBC2", "NKG7", "GZMA", "GZMB", "GZMK",
        "PRF1", "IL7R", "LEF1", "TCF7", "TBX21", "FOXP3", "CTLA4", "PDCD1",
    ]
    var_names = np.asarray(adata_query.var_names, dtype=str)
    in_panel = [m for m in CANON_TCELL_MARKERS if m in set(var_names)]
    if not in_panel:
        log.info("[G T-cell] no canonical T-cell markers in panel.")
        return pd.DataFrame()
    name_to_ct = dict(zip(ann["cell_id"].astype(str), ann["predicted_celltype"]))
    q_labels = np.asarray([name_to_ct.get(c, None)
                            for c in adata_query.obs_names.astype(str)], dtype=object)
    is_tcell = np.array([any(p.lower() in (str(x).lower() if x is not None else "")
                              for p in tcell_pattern)
                          for x in q_labels])
    other_mask = ~is_tcell & np.array([x is not None and x == x
                                        for x in q_labels])
    if is_tcell.sum() < min_cells_per_type or other_mask.sum() < min_cells_per_type:
        log.info("[G T-cell] too few cells (T-cell=%d, other=%d)",
                 int(is_tcell.sum()), int(other_mask.sum()))
        return pd.DataFrame()

    X = sp.csr_matrix(adata_query.X) if not sp.issparse(adata_query.X) else adata_query.X.tocsr()
    Xn = _log_normalize(X)
    Xn_dense = Xn.toarray() if sp.issparse(Xn) else Xn
    var_pos = {g: i for i, g in enumerate(var_names)}

    rows = []
    for g in in_panel:
        j = var_pos[g]
        mu_in = float(Xn_dense[is_tcell, j].mean())
        mu_out = float(Xn_dense[other_mask, j].mean())
        rows.append({
            "method": method, "gene": g,
            "spatial_log2fc": (mu_in - mu_out) / np.log(2.0),
            "spatial_mean_in_tcells": mu_in,
            "spatial_mean_in_other": mu_out,
            "n_tcells": int(is_tcell.sum()),
            "n_other": int(other_mask.sum()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "tcell_marker_log2fc.tsv", sep="\t", index=False)
    log.info("[G T-cell] %d markers in panel; n_tcells=%d, n_other=%d",
             len(df), int(is_tcell.sum()), int(other_mask.sum()))
    return df


# ===========================================================================
# H. NPMI coherence
# ===========================================================================
def metric_npmi_coherence(
    adata_query, npmi_panel: pd.DataFrame, *,
    outdir: Path, log: logging.Logger, tau: float = 0.05,
) -> dict[str, float]:
    """Reuse tracer.metrics relu purity/conflict aggregated for the dataset."""
    pur, conf, rel_pur, rel_conf, sig = _compute_purity_conflict(
        adata_query, npmi_panel, tau=tau, log=log,
    )
    out = {
        "median_purity_score":   float(np.nanmedian(pur)),
        "median_conflict_score": float(np.nanmedian(conf)),
        "median_relative_purity":   float(np.nanmedian(rel_pur)),
        "median_relative_conflict": float(np.nanmedian(rel_conf)),
        "median_signal_strength":   float(np.nanmedian(sig)),
        "fraction_high_conflict_cells":   float(np.nanmean(rel_conf > 0.4)),
        "fraction_low_purity_cells":      float(np.nanmean(rel_pur < 0.4)),
    }
    pd.DataFrame(list(out.items()), columns=["metric", "value"]).to_csv(
        outdir / "npmi_coherence_metrics.tsv", sep="\t", index=False,
    )
    log.info("[H NPMI] median rel_purity=%.3f rel_conflict=%.3f signal=%.3f",
             out["median_relative_purity"], out["median_relative_conflict"],
             out["median_signal_strength"])
    return out


# ===========================================================================
# Cell-type frequency tables (pre/post)
# ===========================================================================
def write_celltype_frequency(
    ann_pre: pd.DataFrame | None, ann_post: pd.DataFrame,
    *, outdir: Path, log: logging.Logger,
) -> None:
    def _freq(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["method", "cell_type", "n_cells", "fraction"])
        c = df["predicted_celltype"].value_counts()
        return pd.DataFrame({
            "method": label,
            "cell_type": c.index.astype(str),
            "n_cells": c.values.astype(int),
            "fraction": (c.values / c.values.sum()).astype(np.float32),
        })

    post_freq = _freq(ann_post, "post")
    post_freq.to_csv(outdir / "post_celltype_frequency.tsv",
                      sep="\t", index=False)
    if ann_pre is not None:
        pre_freq = _freq(ann_pre, "pre")
        pre_freq.to_csv(outdir / "pre_celltype_frequency.tsv",
                         sep="\t", index=False)
        combined = pre_freq.merge(post_freq, on="cell_type", how="outer",
                                    suffixes=("_pre", "_post"))
        combined.to_csv(outdir / "celltype_frequency_pre_post.tsv",
                         sep="\t", index=False)


# ===========================================================================
# I. Benchmark summary
# ===========================================================================
def assemble_summary(
    *, args, transcripts: dict[str, Any], cell_counts: dict[str, Any],
    cell_qc: dict[str, Any], runtime: dict[str, Any],
    consistency_df: pd.DataFrame, marker_df: pd.DataFrame,
    tcell_df: pd.DataFrame, npmi_metrics: dict[str, float] | None,
    n_celltypes_detected: int, outdir: Path, log: logging.Logger,
) -> None:
    summary = {
        "method": args.method,
        "dataset": args.dataset_name,
        "platform": args.platform,
        "n_input_transcripts": transcripts.get("n_total_transcripts_post"),
        "n_output_transcripts": transcripts.get("n_assigned_transcripts_post"),
        "n_unassigned_pre": transcripts.get("n_unassigned_transcripts_pre"),
        "n_unassigned_post": transcripts.get("n_unassigned_transcripts_post"),
        "n_cells_pre_filter": cell_counts.get("n_total_entities_pre_filter"),
        "n_cells_post_filter": cell_counts.get("n_total_entities_post_filter"),
        "n_partial_cells_post_filter": cell_counts.get("n_partials"),
        "n_components_post_filter": cell_counts.get("n_components"),
        "median_transcripts_per_cell": cell_qc.get("median_transcripts_per_cell"),
        "median_genes_per_cell": cell_qc.get("median_genes_per_cell"),
        "median_pct_mito": cell_qc.get("median_pct_mito"),
        "n_celltypes_detected": n_celltypes_detected,
        "median_reference_pearson_r": float(consistency_df["pearson_r"].median())
            if not consistency_df.empty else None,
        "median_marker_log2fc": float(marker_df["spatial_log2fc"].median())
            if not marker_df.empty else None,
        "tcell_marker_log2fc_median": float(tcell_df["spatial_log2fc"].median())
            if not tcell_df.empty else None,
        "runtime_seconds": runtime.get("runtime_seconds"),
        "peak_memory_gb": runtime.get("peak_memory_gb"),
        "relative_purity":  npmi_metrics.get("median_relative_purity")
            if npmi_metrics else None,
        "relative_conflict": npmi_metrics.get("median_relative_conflict")
            if npmi_metrics else None,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(outdir / "method_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(list(summary.items()), columns=["metric", "value"]).to_csv(
        outdir / "benchmark_summary.tsv", sep="\t", index=False,
    )
    log.info("[I summary] wrote method_summary.json + benchmark_summary.tsv")


# ===========================================================================
# Driver
# ===========================================================================
def main() -> int:
    args = build_argparser().parse_args()
    # Backwards-compat: --transcripts-pre is a deprecated alias for --original.
    if getattr(args, "transcripts_pre_deprecated", None) is not None:
        if args.original is not None and args.original != args.transcripts_pre_deprecated:
            raise SystemExit(
                "Both --original and --transcripts-pre were provided with different "
                "paths. --transcripts-pre is deprecated; pass only --original."
            )
        print("[get_metric] WARNING: --transcripts-pre is deprecated; use --original.",
              file=sys.stderr, flush=True)
        if args.original is None:
            args.original = args.transcripts_pre_deprecated
    args.outdir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.outdir)
    log.info("=== get_metric.py ===")
    log.info("Method: %s; dataset: %s; platform: %s; seed: %d",
             args.method, args.dataset_name, args.platform, args.seed)

    skip_set = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    # Load inputs
    df_post = load_transcripts(args.transcripts, log=log)
    label_col = detect_label_col(df_post, args.label_col)
    log.info("Label column: %s", label_col)
    df_pre = None
    if args.original is not None:
        df_pre = load_transcripts(args.original, log=log)
        if args.original_label_col not in df_pre.columns:
            raise SystemExit(
                f"--original-label-col {args.original_label_col!r} missing; "
                f"cols={list(df_pre.columns)}"
            )

    npmi_panel = None
    if args.npmi is not None:
        log.info("Loading NPMI panel: %s", args.npmi)
        np_df = pd.read_csv(args.npmi)
        # Symmetric expansion
        rev = np_df.copy()
        rev["gene_i"], rev["gene_j"] = np_df["gene_j"].values, np_df["gene_i"].values
        npmi_panel = pd.concat([np_df, rev], ignore_index=True)
        npmi_panel = npmi_panel.loc[npmi_panel["gene_i"] != npmi_panel["gene_j"]]

    # --- A: runtime
    runtime = {} if "A" in skip_set else metric_runtime(args, args.outdir, log)

    # --- B: transcript assignment
    transcripts = {} if "B" in skip_set else metric_assignment(
        df_post, label_col, df_pre, args.original_label_col,
        outdir=args.outdir, log=log,
    )

    # --- C: cell counts + min/max filter
    counts_info = metric_cell_counts(
        df_post, label_col,
        min_tx=args.min_transcripts_per_cell,
        max_tx=args.max_transcripts_per_cell,
        outdir=args.outdir, log=log,
    )
    keep_ids = counts_info["kept_cell_ids"]

    # --- Build filtered cell×gene AnnData (post)
    adata_post = build_cellxgene(df_post, label_col, keep_ids=keep_ids, log=log)

    # Pre-method counterpart (apply SAME filter)
    adata_pre = None
    if df_pre is not None:
        pre_label = args.original_label_col
        pre_counts = (
            df_pre.loc[~df_pre[pre_label].astype(str).isin(UNASSIGNED_TOKENS),
                        pre_label]
                  .astype(str).value_counts()
        )
        pre_keep_ids = set(pre_counts[(pre_counts >= args.min_transcripts_per_cell)
                                       & (pre_counts <= args.max_transcripts_per_cell)
                                       ].index.astype(str))
        adata_pre = build_cellxgene(df_pre, pre_label, keep_ids=pre_keep_ids, log=log)

    # --- D: cell QC (per-cell)
    qc_summary = {} if "D" in skip_set else metric_cell_qc(
        adata_post, outdir=args.outdir, log=log, npmi_panel=npmi_panel,
    )

    # --- Load reference (shared by E, F, G)
    ref = None
    ann_post = pd.DataFrame()
    ann_pre = None
    if "E" not in skip_set or "F" not in skip_set or "G" not in skip_set:
        try:
            ref = load_reference(args.reference_h5ad, args.reference_celltype_col, log)
        except Exception as e:
            log.warning("Reference load failed (%s); E/F/G will be skipped.", e)
            ref = None

    # --- E: label transfer (use SAME reference + same anchors for pre/post)
    if ref is not None and "E" not in skip_set:
        ann_post = transfer_labels(
            adata_post, ref, seed=args.seed,
            k=args.knn_k, per_type=args.n_anchors_per_celltype, log=log,
        )
        ann_post.to_csv(args.outdir / "post_celltype_annotations.tsv",
                        sep="\t", index=False)
        if adata_pre is not None:
            ann_pre = transfer_labels(
                adata_pre, ref, seed=args.seed,
                k=args.knn_k, per_type=args.n_anchors_per_celltype, log=log,
            )
            ann_pre.to_csv(args.outdir / "pre_celltype_annotations.tsv",
                            sep="\t", index=False)
        write_celltype_frequency(ann_pre, ann_post, outdir=args.outdir, log=log)

    # --- F: reference consistency
    consistency_df = pd.DataFrame()
    if ref is not None and "F" not in skip_set and not ann_post.empty:
        consistency_df = metric_reference_consistency(
            adata_post, ann_post, ref,
            method=args.method, outdir=args.outdir, log=log,
        )
        plot_reference_consistency(consistency_df, args.outdir, log)

    # --- G: marker specificity + T-cell markers
    marker_df = pd.DataFrame()
    tcell_df = pd.DataFrame()
    if ref is not None and "G" not in skip_set and not ann_post.empty:
        if args.reference_marker_table is not None:
            markers_table = pd.read_csv(args.reference_marker_table, sep="\t")
        else:
            markers_table = compute_reference_markers(
                ref, n_top=args.n_top_markers, log=log,
            )
        if not markers_table.empty:
            markers_table.to_csv(
                args.outdir / "reference_markers_used.tsv",
                sep="\t", index=False,
            )
            marker_df = metric_marker_specificity(
                adata_post, ann_post, markers_table,
                method=args.method, outdir=args.outdir, log=log,
            )
            plot_marker_specificity(marker_df, args.outdir, log)
        tcell_df = metric_tcell_marker_log2fc(
            adata_post, ann_post, method=args.method,
            outdir=args.outdir, log=log,
        )

    # --- H: NPMI coherence
    npmi_metrics = None
    if npmi_panel is not None and "H" not in skip_set:
        npmi_metrics = metric_npmi_coherence(
            adata_post, npmi_panel, outdir=args.outdir, log=log,
        )

    # --- I: assemble summary
    n_celltypes = int(ann_post["predicted_celltype"].nunique()) if not ann_post.empty else 0
    assemble_summary(
        args=args,
        transcripts=transcripts, cell_counts=counts_info,
        cell_qc=qc_summary, runtime=runtime,
        consistency_df=consistency_df, marker_df=marker_df,
        tcell_df=tcell_df, npmi_metrics=npmi_metrics,
        n_celltypes_detected=n_celltypes,
        outdir=args.outdir, log=log,
    )
    log.info("DONE — outputs at %s", args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
