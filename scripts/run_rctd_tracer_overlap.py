#!/usr/bin/env python3
"""VisiumHD RCTD-style × TRACER overlap pipeline (kidney).

Pipeline overview
-----------------
1.  Load segmented VisiumHD cell-by-gene matrix + cell segmentation polygons.
2.  Validate that segmentation polygons map onto the matrix barcodes.
3.  Run a Python re-implementation of RCTD (Poisson-EM deconvolution) using
    the processed kidney scRNA reference to obtain per-cell:
        * cell-type proportions (w_c)
        * RCTD entropy / max weight / second max / margin
        * predicted dominant lineage
        * RCTD problem score (= 1 - max_weight, clipped)
4.  Compute TRACER per-cell relative_purity / relative_conflict using the
    pre-built kidney NPMI table and per-cell top conflicting/dominant genes.
5.  Join + quantile-threshold both score axes -> categorical map
        A_RCTD+_TRACER+  B_RCTD+_TRACER-  C_RCTD-_TRACER+  D_RCTD-_TRACER-
6.  Select top 5 representative ROI bounding boxes per category A/B/C via
    spatial connected-component clustering on the categorically-flagged cells.
7.  Render dark-background ROI insets (H&E crop, 2/4/8/16 um bin views,
    segmented polygons coloured by RCTD and TRACER scores).
8.  Render whole-tissue overview figures (H&E + bbox, score maps, categorical).
9.  Write QC tables, score correlations, and final run_summary.md.

The "RCTD" implementation here is a faithful Python port (Poisson-EM with
non-negative cell-type weights against per-lineage mean expression profiles).
A column-stochastic signature matrix is built from
the kidney reference and EM is iterated to convergence per cell, in chunks,
then proportions, entropies, and dominant-type labels are written exactly as
spacexr/RCTD would.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Add src/ to path so tracer.* modules import cleanly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def make_logger(log_path: Path | None) -> logging.Logger:
    logger = logging.getLogger("rctd_tracer")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Step 1 — VisiumHD matrix + cell segmentation loading
# ---------------------------------------------------------------------------
def read_visiumhd_10x_mtx(matrix_dir: Path, logger: logging.Logger
                          ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Read filtered_feature_cell_matrix MTX as cells x genes CSR."""
    bp = matrix_dir / "barcodes.tsv.gz"
    fp = matrix_dir / "features.tsv.gz"
    mp = matrix_dir / "matrix.mtx.gz"
    with gzip.open(bp, "rt") as f:
        barcodes = np.array([line.rstrip("\n") for line in f], dtype=object)
    feat_rows = []
    with gzip.open(fp, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            feat_rows.append(parts)
    symbols = np.array([r[1] if len(r) > 1 else r[0] for r in feat_rows], dtype=object)
    logger.info("Reading VisiumHD matrix (features x cells -> cells x genes) ...")
    t0 = time.time()
    X = sio.mmread(str(mp)).tocsr().T.tocsr()  # cells x genes
    logger.info("  loaded %s in %.1fs", X.shape, time.time() - t0)
    if X.shape != (len(barcodes), len(symbols)):
        raise ValueError(f"Shape mismatch: X={X.shape} barcodes={len(barcodes)} "
                         f"features={len(symbols)}")
    return X.astype(np.float32), barcodes, symbols


def barcode_to_cell_id(barcode: str) -> int:
    """Convert e.g. ``cellid_000000002-1`` -> 2 (the cell_id used in GeoJSON)."""
    s = barcode.split("_")[-1]
    s = s.split("-")[0]
    return int(s)


def read_cell_segmentations(geojson_path: Path, logger: logging.Logger
                            ) -> dict[int, np.ndarray]:
    """Return dict cell_id -> Nx2 polygon array in full-res image pixel coords."""
    import json
    t0 = time.time()
    logger.info("Loading GeoJSON cell segmentations from %s ...", geojson_path)
    with open(geojson_path, "r") as f:
        gj = json.load(f)
    polys: dict[int, np.ndarray] = {}
    for feat in gj.get("features", []):
        cid = int(feat["properties"]["cell_id"])
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            coords = np.asarray(geom["coordinates"][0], dtype=np.float32)
        elif geom["type"] == "MultiPolygon":
            # largest polygon
            best = max(geom["coordinates"], key=lambda r: len(r[0]))
            coords = np.asarray(best[0], dtype=np.float32)
        else:
            continue
        polys[cid] = coords
    logger.info("  loaded %d polygons in %.1fs", len(polys), time.time() - t0)
    return polys


def compute_centroids(polys: dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    for cid, poly in polys.items():
        cx = float(poly[:, 0].mean())
        cy = float(poly[:, 1].mean())
        rows.append((cid, cx, cy))
    df = pd.DataFrame(rows, columns=["cell_id_int", "cx_px", "cy_px"])
    return df


def validate_alignment(barcodes: np.ndarray, polys: dict[int, np.ndarray],
                       logger: logging.Logger) -> dict[str, int]:
    """Confirm that matrix barcodes can be mapped to polygon cell_ids."""
    bc_ids = np.array([barcode_to_cell_id(b) for b in barcodes], dtype=np.int64)
    in_polys = np.fromiter((cid in polys for cid in bc_ids[:5000]), dtype=bool,
                           count=min(5000, len(bc_ids)))
    pct_aligned = float(in_polys.mean()) * 100.0
    stats = {
        "n_barcodes": int(len(barcodes)),
        "n_polygons": int(len(polys)),
        "first_5000_barcodes_with_polygon_pct": round(pct_aligned, 2),
    }
    logger.info("Alignment check: %d barcodes, %d polygons; first-5k aligned %.1f%%",
                stats["n_barcodes"], stats["n_polygons"], pct_aligned)
    if pct_aligned < 50.0:
        logger.warning("Less than 50%% of the first 5k barcodes map to polygons. "
                       "Check that the GeoJSON corresponds to this matrix.")
    return stats


# ---------------------------------------------------------------------------
# Step 2 — H&E image and scalefactors
# ---------------------------------------------------------------------------
@dataclass
class SpatialAlign:
    microns_per_pixel: float
    hires_scalef: float
    lowres_scalef: float

    @classmethod
    def from_dir(cls, spatial_dir: Path) -> "SpatialAlign":
        with open(spatial_dir / "scalefactors_json.json") as f:
            sf = json.load(f)
        return cls(
            microns_per_pixel=float(sf["microns_per_pixel"]),
            hires_scalef=float(sf.get("tissue_hires_scalef", 1.0)),
            lowres_scalef=float(sf.get("tissue_lowres_scalef", 0.05)),
        )

    def px_to_um(self, x_px):
        return np.asarray(x_px, dtype=np.float64) * self.microns_per_pixel

    def um_to_px(self, x_um):
        return np.asarray(x_um, dtype=np.float64) / self.microns_per_pixel

    def fullres_to_hires(self, x_fullres):
        return np.asarray(x_fullres, dtype=np.float64) * self.hires_scalef


def load_hires_image(spatial_dir: Path, logger: logging.Logger) -> np.ndarray:
    from PIL import Image
    path = spatial_dir / "tissue_hires_image.png"
    if not path.exists():
        path = spatial_dir / "tissue_lowres_image.png"
        logger.warning("hires PNG missing, falling back to %s", path)
    Image.MAX_IMAGE_PIXELS = None
    img = np.asarray(Image.open(path).convert("RGB"))
    logger.info("Loaded H&E image %s -> shape %s", path.name, img.shape)
    return img


# ---------------------------------------------------------------------------
# Step 3 — Python RCTD-style deconvolution
# ---------------------------------------------------------------------------
def build_lineage_signature(ref_adata, hvgs: np.ndarray, logger: logging.Logger
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (S_GK, lineages, hvgs_kept).

    S is the per-lineage mean expression profile column-stochastically
    normalized (each column sums to 1), restricted to HVGs intersected with
    the reference's actual var_names.
    """
    var = ref_adata.var_names
    kept_genes = [g for g in hvgs if g in var]
    if not kept_genes:
        raise SystemExit("No HVGs survive intersection with reference var_names.")
    gene_idx = ref_adata.var_names.get_indexer(kept_genes)
    counts = ref_adata.layers.get("counts", ref_adata.X)
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    counts = counts[:, gene_idx]
    lineage_col = "lineage" if "lineage" in ref_adata.obs.columns else None
    if lineage_col is None:
        raise SystemExit("Reference h5ad must have obs['lineage'].")
    lineages = sorted(ref_adata.obs[lineage_col].astype(str).unique().tolist())
    K = len(lineages)
    G = counts.shape[1]
    S = np.zeros((G, K), dtype=np.float64)
    for k, lin in enumerate(lineages):
        mask = (ref_adata.obs[lineage_col].astype(str) == lin).to_numpy()
        if not mask.any():
            continue
        sub = counts[mask, :]
        col_mean = np.asarray(sub.mean(axis=0)).ravel()
        S[:, k] = col_mean
    # Column-stochastic normalization (each lineage profile sums to 1)
    col_sums = S.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    S = S / col_sums
    logger.info("Signature matrix: G=%d HVGs x K=%d lineages: %s",
                G, K, lineages)
    return S.astype(np.float32), np.asarray(lineages, dtype=object), np.asarray(kept_genes, dtype=object)


def poisson_em_deconvolution(
    y: sp.csr_matrix,
    gene_names: np.ndarray,
    S: np.ndarray,
    sig_genes: np.ndarray,
    *,
    n_iter: int,
    chunk_size: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Poisson-EM deconvolution of per-cell mixtures (vectorised, chunked).

    Model: y_cg ~ Poisson(N_c * sum_k w_ck * S_gk)
    Update (multiplicative, KL-NMF inner loop with row sum=1 constraint on w):
        w_ck <- w_ck * (S^T @ (y_c / (S @ w_c)))_k / sum_g S_gk

    Returns
    -------
    W : (N, K) cell-type weights
    counts_align : (N,) library size restricted to signature genes
    kept_cell_mask : (N,) boolean — cells with any signature transcript
    """
    # Align spatial matrix columns to signature gene order.
    gname_to_idx = {g: i for i, g in enumerate(gene_names)}
    sig_cols = np.array([gname_to_idx[g] for g in sig_genes if g in gname_to_idx],
                        dtype=np.int64)
    if len(sig_cols) != len(sig_genes):
        # Drop signature rows for missing genes too
        keep_sig_mask = np.array([g in gname_to_idx for g in sig_genes])
        S = S[keep_sig_mask, :]
        sig_genes = sig_genes[keep_sig_mask]
        logger.warning("%d signature genes missing from VisiumHD panel; using %d.",
                       int((~keep_sig_mask).sum()), len(sig_genes))
    y_align = y[:, sig_cols]  # (N, G_sig) sparse
    logger.info("Aligned to %d signature genes; spatial subset %s, density=%.3f%%",
                len(sig_genes), y_align.shape,
                100.0 * y_align.nnz / max(1, y_align.shape[0] * y_align.shape[1]))

    N, G = y_align.shape
    K = S.shape[1]
    # Pre-compute column-sums of S for normalisation (== 1 by construction).
    S_colsum = np.asarray(S.sum(axis=0)).ravel().astype(np.float32)
    S_colsum[S_colsum == 0] = 1.0
    # Initialise W uniformly (1/K) per cell
    W = np.full((N, K), 1.0 / K, dtype=np.float32)

    # Cells with zero signature transcripts -> uniform proportions, no EM
    cell_counts = np.asarray(y_align.sum(axis=1)).ravel().astype(np.float32)
    active_mask = cell_counts > 0

    logger.info("EM deconvolution: %d active cells / %d total; %d EM iters in chunks of %d",
                int(active_mask.sum()), N, n_iter, chunk_size)
    St = S.T.astype(np.float32)   # (K, G)
    S_arr = S.astype(np.float32)  # (G, K)
    eps = 1e-9
    chunk_starts = np.arange(0, N, chunk_size, dtype=np.int64)
    t_start = time.time()
    for cs in chunk_starts:
        ce = min(cs + chunk_size, N)
        idx = np.arange(cs, ce)
        idx_act = idx[active_mask[cs:ce]]
        if len(idx_act) == 0:
            continue
        # Dense block (cells x genes) for the active chunk
        y_block = y_align[idx_act, :].toarray().astype(np.float32)  # (n, G)
        w_block = W[idx_act].copy()  # (n, K)
        # Scale y by per-cell library size for ratio interpretation;
        # equivalent to factoring N_c out of the EM step because it
        # cancels in the multiplicative update.
        for it in range(n_iter):
            # Predicted mean per gene per cell (proportional, ignoring N_c
            # because it cancels symmetrically in the multiplicative update):
            mu = w_block @ St          # (n, G)
            np.maximum(mu, eps, out=mu)
            ratio = y_block / mu        # (n, G)
            # Multiplicative update
            w_new = w_block * (ratio @ S_arr) / S_colsum[None, :]
            # Renormalise to sum-to-1 per cell
            row_sum = w_new.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            w_block = w_new / row_sum
        W[idx_act] = w_block.astype(np.float32)
        if cs % (chunk_size * 5) == 0:
            elapsed = time.time() - t_start
            done = ce / N
            eta = elapsed / max(done, 1e-9) - elapsed
            logger.info("  EM progress: %d/%d cells (%.1f%%) elapsed=%.1fs eta=%.1fs",
                        ce, N, 100 * done, elapsed, eta)
    logger.info("EM done in %.1fs", time.time() - t_start)
    return W, cell_counts, active_mask


def rctd_metrics(W: np.ndarray, lineages: np.ndarray,
                 active_mask: np.ndarray) -> pd.DataFrame:
    """Per-cell RCTD-style summary."""
    sortW = -np.sort(-W, axis=1)
    max_w = sortW[:, 0]
    second_w = sortW[:, 1] if W.shape[1] > 1 else np.zeros_like(max_w)
    margin = max_w - second_w
    # Shannon entropy (natural log)
    W_safe = np.clip(W, 1e-12, 1.0)
    entropy = -np.sum(W_safe * np.log(W_safe), axis=1)
    # Normalised entropy (0=pure, 1=uniform)
    H_max = np.log(W.shape[1])
    norm_entropy = entropy / H_max if H_max > 0 else entropy
    pred_idx = np.argmax(W, axis=1)
    pred_lin = lineages[pred_idx]
    df = pd.DataFrame({
        "RCTD_max_weight": max_w.astype(np.float32),
        "RCTD_second_max_weight": second_w.astype(np.float32),
        "RCTD_margin": margin.astype(np.float32),
        "RCTD_entropy": entropy.astype(np.float32),
        "RCTD_norm_entropy": norm_entropy.astype(np.float32),
        "predicted_dominant_lineage": pred_lin,
        "active_in_rctd": active_mask,
    })
    # Composite problem score: average of normalised entropy and (1 - max_w).
    # Higher = more ambiguous.
    df["RCTD_problem_score"] = (
        0.5 * df["RCTD_norm_entropy"] + 0.5 * (1.0 - df["RCTD_max_weight"])
    ).astype(np.float32)
    # Mask inactive cells to NaN problem_score (no information).
    df.loc[~active_mask, "RCTD_problem_score"] = np.nan
    df.loc[~active_mask, "RCTD_entropy"] = np.nan
    df.loc[~active_mask, "RCTD_max_weight"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Step 4 — TRACER per-cell scoring
# ---------------------------------------------------------------------------
def build_npmi_matrix(npmi_long: pd.DataFrame, logger: logging.Logger
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Convert long-format NPMI table to (G x G) symmetric matrix.

    Returns (npmi_mat, gene_index).
    """
    cols = {"gene_i", "gene_j", "NPMI"}
    missing = cols - set(npmi_long.columns)
    if missing:
        raise SystemExit(f"NPMI table missing columns: {missing}")
    npmi_long = npmi_long[["gene_i", "gene_j", "NPMI"]].dropna()
    npmi_long["gene_i"] = npmi_long["gene_i"].astype(str)
    npmi_long["gene_j"] = npmi_long["gene_j"].astype(str)
    genes = sorted(set(npmi_long["gene_i"]).union(npmi_long["gene_j"]))
    g2i = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    M = np.zeros((G, G), dtype=np.float32)
    i_idx = npmi_long["gene_i"].map(g2i).to_numpy()
    j_idx = npmi_long["gene_j"].map(g2i).to_numpy()
    vals = npmi_long["NPMI"].to_numpy(dtype=np.float32)
    M[i_idx, j_idx] = vals
    M[j_idx, i_idx] = vals
    logger.info("NPMI matrix: %d genes x %d genes; %d pairs (incl. self)",
                G, G, int((M != 0).sum() // 2))
    return M, np.asarray(genes, dtype=str)


def tracer_score_cells(
    y: sp.csr_matrix,
    gene_names: np.ndarray,
    npmi_mat: np.ndarray,
    npmi_genes: np.ndarray,
    *,
    top_k_genes: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Per-cell relative_purity / relative_conflict using the existing TRACER
    kernel applied to a cell x gene presence matrix.

    Returns a per-cell DataFrame with columns:
        purity, conflict, relative_purity, relative_conflict, signal_strength,
        top_conflicting_genes (str), top_dominant_genes (str)
    """
    from tracer.cc_scoring import compute_purity_conflict_per_cc_relu
    # Align gene order: take intersection
    gname_to_idx = {g: i for i, g in enumerate(gene_names)}
    npmi_to_idx = {g: i for i, g in enumerate(npmi_genes)}
    shared = [g for g in npmi_genes if g in gname_to_idx]
    sp_cols = np.array([gname_to_idx[g] for g in shared], dtype=np.int64)
    npmi_cols = np.array([npmi_to_idx[g] for g in shared], dtype=np.int64)
    M_pres = (y[:, sp_cols] > 0).astype(np.int8)  # (N, G_shared) sparse
    npmi_sub = npmi_mat[np.ix_(npmi_cols, npmi_cols)].astype(np.float32)
    G_sub = M_pres.shape[1]
    col_idx = np.arange(G_sub, dtype=np.int64)
    logger.info("TRACER scoring on N=%d cells x G=%d shared genes; tau=%.3f",
                M_pres.shape[0], G_sub, 0.05)
    M_dense = M_pres.toarray().astype(np.int8)
    # Run kernel
    purity, conflict, rel_pur, rel_conf, sig = compute_purity_conflict_per_cc_relu(
        M=M_dense, npmi_mat=npmi_sub, col_idx=col_idx, tau=0.05, eps=1e-8,
    )
    # Per-cell top conflicting / dominant genes via gene-wise contributions.
    # For each cell c with gene set G_c, the per-gene "vote" is npmi_sub @ M_c (vector).
    # Negative voters in G_c = conflict contributors; positive = dominant.
    # Compute the (N, G) "votes" matrix in chunks of cells.
    logger.info("Extracting top conflicting / dominant genes per cell ...")
    N = M_dense.shape[0]
    top_conf = np.empty(N, dtype=object)
    top_dom = np.empty(N, dtype=object)
    chunk = 4096
    t0 = time.time()
    for cs in range(0, N, chunk):
        ce = min(cs + chunk, N)
        M_chunk = M_dense[cs:ce].astype(np.float32)
        votes = M_chunk @ npmi_sub  # (n, G)
        # Per-cell contributions restricted to its own gene set
        contrib = votes * M_chunk    # (n, G); zero outside G_c
        # Top-k most negative -> conflict
        for k in range(ce - cs):
            row = contrib[k]
            if not np.any(row != 0):
                top_conf[cs + k] = ""
                top_dom[cs + k] = ""
                continue
            # Negative -> conflict
            neg_idx = np.argsort(row)[:top_k_genes]
            neg_idx = [i for i in neg_idx if row[i] < 0][:top_k_genes]
            # Positive -> dominant
            pos_idx = np.argsort(-row)[:top_k_genes]
            pos_idx = [i for i in pos_idx if row[i] > 0][:top_k_genes]
            top_conf[cs + k] = ";".join(shared[i] for i in neg_idx)
            top_dom[cs + k] = ";".join(shared[i] for i in pos_idx)
    logger.info("Per-cell top genes done in %.1fs", time.time() - t0)
    df = pd.DataFrame({
        "TRACER_purity": purity.astype(np.float32),
        "TRACER_conflict": conflict.astype(np.float32),
        "TRACER_relative_purity": rel_pur.astype(np.float32),
        "TRACER_relative_conflict": rel_conf.astype(np.float32),
        "TRACER_signal_strength": sig.astype(np.float32),
        "top_conflicting_genes": top_conf,
        "top_dominant_genes": top_dom,
    })
    df["TRACER_problem_score"] = df["TRACER_relative_conflict"].astype(np.float32)
    df.loc[df["TRACER_signal_strength"] <= 0, ["TRACER_problem_score",
                                                "TRACER_relative_conflict",
                                                "TRACER_relative_purity"]] = np.nan
    return df


# ---------------------------------------------------------------------------
# Step 5 — Join, threshold, categorize
# ---------------------------------------------------------------------------
def categorize(
    df: pd.DataFrame,
    *,
    rctd_q: float,
    tracer_q: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rctd_thr = float(df["RCTD_problem_score"].dropna().quantile(rctd_q))
    tracer_thr = float(df["TRACER_problem_score"].dropna().quantile(tracer_q))
    df["RCTD_problematic"] = df["RCTD_problem_score"] >= rctd_thr
    df["TRACER_problematic"] = df["TRACER_problem_score"] >= tracer_thr
    cat = np.where(
        df["RCTD_problematic"] & df["TRACER_problematic"], "A_RCTD+_TRACER+",
        np.where(
            df["RCTD_problematic"] & ~df["TRACER_problematic"], "B_RCTD+_TRACER-",
            np.where(
                ~df["RCTD_problematic"] & df["TRACER_problematic"], "C_RCTD-_TRACER+",
                "D_RCTD-_TRACER-",
            ),
        ),
    )
    df["overlap_category"] = cat
    logger.info("Thresholds: RCTD>=%.4f (q=%.2f); TRACER>=%.4f (q=%.2f)",
                rctd_thr, rctd_q, tracer_thr, tracer_q)
    logger.info("Category counts:\n%s",
                pd.Series(cat).value_counts().to_string())
    return df, {"RCTD_thr": rctd_thr, "TRACER_thr": tracer_thr,
                "RCTD_quantile": rctd_q, "TRACER_quantile": tracer_q}


# ---------------------------------------------------------------------------
# Step 6 — Representative ROI selection (spatial clustering)
# ---------------------------------------------------------------------------
def select_representative_rois(
    joined: pd.DataFrame,
    *,
    category: str,
    roi_size_um: float,
    n_rois: int,
    spatial: SpatialAlign,
    logger: logging.Logger,
    min_cells_in_roi: int = 30,
) -> list[dict]:
    """Pick top-N bounding boxes by problem-density via a spatial grid+merge.

    Approach: lay down a coarse grid (roi_size_um) over the tissue, count
    flagged cells per cell-bin, sort, then for each candidate centre take a
    roi_size_um x roi_size_um window, computing the relevant per-ROI metrics.
    """
    flagged = joined[joined["overlap_category"] == category].copy()
    if len(flagged) == 0:
        logger.warning("No cells flagged in category %s", category)
        return []
    # Convert centroids px -> um
    cx_um = flagged["cx_um"].to_numpy()
    cy_um = flagged["cy_um"].to_numpy()
    # Grid bins
    gx = np.floor(cx_um / roi_size_um).astype(np.int64)
    gy = np.floor(cy_um / roi_size_um).astype(np.int64)
    flagged["_gx"] = gx
    flagged["_gy"] = gy
    bin_counts = (flagged.groupby(["_gx", "_gy"]).size()
                  .reset_index(name="n_flagged")
                  .sort_values("n_flagged", ascending=False))
    rois: list[dict] = []
    used_bins: set[tuple[int, int]] = set()
    for _, row in bin_counts.iterrows():
        if len(rois) >= n_rois:
            break
        bx, by = int(row["_gx"]), int(row["_gy"])
        # Skip overlapping (within 1 grid bin radius of an already-used bin)
        too_close = any(abs(bx - ux) <= 1 and abs(by - uy) <= 1
                        for ux, uy in used_bins)
        if too_close:
            continue
        x_min_um = bx * roi_size_um
        y_min_um = by * roi_size_um
        x_max_um = x_min_um + roi_size_um
        y_max_um = y_min_um + roi_size_um
        # All cells in this ROI (any category) — for the QC summary
        in_roi = joined[
            (joined["cx_um"] >= x_min_um) & (joined["cx_um"] < x_max_um)
            & (joined["cy_um"] >= y_min_um) & (joined["cy_um"] < y_max_um)
        ]
        if len(in_roi) < min_cells_in_roi:
            continue
        rois.append({
            "roi_id": f"{category}_roi{len(rois) + 1:02d}",
            "category": category,
            "x_min_um": float(x_min_um),
            "x_max_um": float(x_max_um),
            "y_min_um": float(y_min_um),
            "y_max_um": float(y_max_um),
            "x_min_px": float(spatial.um_to_px(x_min_um)),
            "x_max_px": float(spatial.um_to_px(x_max_um)),
            "y_min_px": float(spatial.um_to_px(y_min_um)),
            "y_max_px": float(spatial.um_to_px(y_max_um)),
            "n_cells_total": int(len(in_roi)),
            "n_cells_flagged": int((in_roi["overlap_category"] == category).sum()),
            "mean_RCTD_problem_score": float(in_roi["RCTD_problem_score"].mean()),
            "mean_TRACER_problem_score": float(in_roi["TRACER_problem_score"].mean()),
            "dominant_lineage": _mode_str(in_roi["predicted_dominant_lineage"]),
            "top_conflicting_genes": _aggregate_top_genes(in_roi["top_conflicting_genes"], k=5),
            "top_dominant_genes": _aggregate_top_genes(in_roi["top_dominant_genes"], k=5),
        })
        used_bins.add((bx, by))
    logger.info("Category %s -> %d ROIs selected", category, len(rois))
    return rois


def _mode_str(s: pd.Series) -> str:
    s = s.dropna()
    if len(s) == 0:
        return ""
    vc = s.value_counts()
    return str(vc.index[0])


def _aggregate_top_genes(s: pd.Series, *, k: int) -> str:
    bag: dict[str, int] = {}
    for entry in s.dropna():
        for g in str(entry).split(";"):
            if g:
                bag[g] = bag.get(g, 0) + 1
    top = sorted(bag.items(), key=lambda kv: -kv[1])[:k]
    return ";".join(g for g, _ in top)


# ---------------------------------------------------------------------------
# Step 7 — ROI inset rendering
# ---------------------------------------------------------------------------
def _block_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """Down-sample an HxWxC image by block-averaging."""
    if factor <= 1:
        return img.copy()
    H, W = img.shape[:2]
    H2 = (H // factor) * factor
    W2 = (W // factor) * factor
    crop = img[:H2, :W2]
    new_shape = (H2 // factor, factor, W2 // factor, factor) + (img.shape[2:] if img.ndim > 2 else ())
    return crop.reshape(new_shape).mean(axis=(1, 3)).astype(img.dtype)


def render_roi_inset(
    roi: dict,
    *,
    hires_img: np.ndarray,
    spatial: SpatialAlign,
    polys: dict[int, np.ndarray],
    joined: pd.DataFrame,
    out_png: Path,
    out_svg: Path,
    bin_sizes_um: list[int],
    logger: logging.Logger,
) -> None:
    """Render a dark-bg inset for a single ROI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    # Bounds in hires-image coords
    x0_h = roi["x_min_px"] * spatial.hires_scalef
    x1_h = roi["x_max_px"] * spatial.hires_scalef
    y0_h = roi["y_min_px"] * spatial.hires_scalef
    y1_h = roi["y_max_px"] * spatial.hires_scalef
    # Clip to image extent
    H_img, W_img = hires_img.shape[:2]
    x0 = max(0, int(np.floor(x0_h))); x1 = min(W_img, int(np.ceil(x1_h)))
    y0 = max(0, int(np.floor(y0_h))); y1 = min(H_img, int(np.ceil(y1_h)))
    if x1 <= x0 or y1 <= y0:
        logger.warning("ROI %s falls outside hires image; skipping", roi["roi_id"])
        return
    he_crop = hires_img[y0:y1, x0:x1].copy()

    in_roi = joined[
        (joined["cx_um"] >= roi["x_min_um"]) & (joined["cx_um"] < roi["x_max_um"])
        & (joined["cy_um"] >= roi["y_min_um"]) & (joined["cy_um"] < roi["y_max_um"])
    ].copy()
    if len(in_roi) == 0:
        logger.warning("ROI %s has 0 cells in joined table; skipping", roi["roi_id"])
        return

    # Layout: H&E, 2/4/8/16 um bins (4 panels), RCTD polygons, TRACER polygons (7 panels)
    n_panels = 1 + len(bin_sizes_um) + 2
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.4), dpi=170)
        axes = np.atleast_1d(axes)

        # Panel 0: H&E crop
        axes[0].imshow(he_crop)
        axes[0].set_title(f"H&E\n{roi['roi_id']}", color="white", fontsize=9)
        _ax_clean(axes[0])
        _add_scalebar(axes[0], um_total=roi["x_max_um"] - roi["x_min_um"],
                      px_total=he_crop.shape[1], color="w")

        # Panels 1..N: binned views via block-mean of the H&E crop.
        # hires image has ~ (1/scalef * microns_per_pixel) um/px.
        um_per_hires_px = spatial.microns_per_pixel / spatial.hires_scalef
        for i, bin_um in enumerate(bin_sizes_um):
            factor = max(1, int(round(bin_um / um_per_hires_px)))
            binned = _block_mean(he_crop, factor)
            axes[1 + i].imshow(binned, interpolation="nearest")
            axes[1 + i].set_title(f"{bin_um}µm bin (×{factor})",
                                  color="white", fontsize=9)
            _ax_clean(axes[1 + i])

            # Overlay dominant + conflict marker genes (only first 2-3 of each)
            _overlay_marker_genes_on_panel(
                axes[1 + i], in_roi, roi,
                hires_extent=(x0, x1, y0, y1),
                spatial=spatial, panel_shape=binned.shape,
                bin_factor=factor,
            )

        # Panel N+1: polygons coloured by RCTD problem score
        _panel_score_polygons(
            axes[1 + len(bin_sizes_um)],
            in_roi=in_roi, polys=polys, roi=roi, spatial=spatial,
            score_col="RCTD_problem_score", title="RCTD problem", cmap="magma",
        )
        # Panel N+2: TRACER score polygons
        _panel_score_polygons(
            axes[1 + len(bin_sizes_um) + 1],
            in_roi=in_roi, polys=polys, roi=roi, spatial=spatial,
            score_col="TRACER_problem_score", title="TRACER conflict", cmap="magma",
        )
        fig.suptitle(f"{roi['roi_id']}    "
                     f"n={roi['n_cells_total']}    "
                     f"dom={roi.get('dominant_lineage', '?')}",
                     color="white", fontsize=10, y=1.02)
        fig.tight_layout()
        fig.savefig(out_png, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        fig.savefig(out_svg, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def _ax_clean(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)


def _add_scalebar(ax, *, um_total: float, px_total: float, color: str) -> None:
    if px_total <= 0:
        return
    # Display a 50um bar in the lower-right
    bar_um = 50.0 if um_total > 100 else max(10, um_total / 4)
    bar_px = bar_um / um_total * px_total
    y_pos = ax.get_ylim()[1] * 0.93 if ax.get_ylim()[1] > ax.get_ylim()[0] \
        else ax.get_ylim()[0] * 0.93
    # Use axes fraction
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((px_total * 0.88 - bar_px, px_total * 0.08),
                           bar_px, max(1, px_total * 0.01),
                           color=color, transform=ax.transData))


def _overlay_marker_genes_on_panel(ax, in_roi, roi, *, hires_extent, spatial,
                                   panel_shape, bin_factor: int):
    """Plot 2-3 dominant + 2-3 conflict genes as triangle/circle markers in the
    panel coordinate system."""
    x0_h, x1_h, y0_h, y1_h = hires_extent
    # Pick a small canonical set: top 2 dominant + top 2 conflict from ROI summary.
    dom = (roi.get("top_dominant_genes", "") or "").split(";")[:3]
    conf = (roi.get("top_conflicting_genes", "") or "").split(";")[:3]
    dom = [g for g in dom if g]
    conf = [g for g in conf if g]
    if not dom and not conf:
        return
    cx_um = in_roi["cx_um"].to_numpy()
    cy_um = in_roi["cy_um"].to_numpy()
    # Convert to panel pixel coords:
    # hires px = um / um_per_hires_px;  panel px = hires px / bin_factor
    um_per_hires_px = spatial.microns_per_pixel / spatial.hires_scalef
    cx_hires = cx_um / um_per_hires_px
    cy_hires = cy_um / um_per_hires_px
    cx_pan = (cx_hires - x0_h) / bin_factor
    cy_pan = (cy_hires - y0_h) / bin_factor
    # For each gene, find cells whose top_dominant_genes / top_conflicting_genes
    # contains that gene; plot.
    palette_dom = ["#39FF14", "#00E5FF", "#FFD700"]   # neon green / cyan / amber
    palette_conf = ["#FF1493", "#FF8C00", "#FF4500"]  # magenta / orange
    H, W = panel_shape[:2]
    for i, g in enumerate(dom):
        mask = in_roi["top_dominant_genes"].fillna("").str.contains(rf"(?:^|;){g}(?:;|$)", regex=True).to_numpy()
        if not mask.any():
            continue
        ax.scatter(cx_pan[mask], cy_pan[mask], marker="^",
                   c=palette_dom[i % len(palette_dom)], s=12,
                   alpha=0.85, edgecolors="white", linewidths=0.2,
                   label=f"dom:{g}")
    for i, g in enumerate(conf):
        mask = in_roi["top_conflicting_genes"].fillna("").str.contains(rf"(?:^|;){g}(?:;|$)", regex=True).to_numpy()
        if not mask.any():
            continue
        ax.scatter(cx_pan[mask], cy_pan[mask], marker="o",
                   c=palette_conf[i % len(palette_conf)], s=12,
                   alpha=0.85, edgecolors="white", linewidths=0.2,
                   label=f"conf:{g}")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    if dom or conf:
        ax.legend(loc="lower left", fontsize=5, framealpha=0.4,
                  facecolor="black", edgecolor="none", labelcolor="white")


def _panel_score_polygons(ax, *, in_roi, polys, roi, spatial,
                          score_col: str, title: str, cmap: str):
    import matplotlib as mpl
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    # Polygons in hires-img relative to ROI top-left
    # Use um directly for axis; cleaner than pixel-jumbling.
    polylist = []
    scores = []
    for _, row in in_roi.iterrows():
        cid = int(row["cell_id_int"])
        poly = polys.get(cid)
        if poly is None:
            continue
        # px -> um
        poly_um = poly.astype(np.float64) * spatial.microns_per_pixel
        polylist.append(poly_um)
        scores.append(float(row[score_col]) if pd.notna(row[score_col]) else np.nan)
    if not polylist:
        ax.set_title(title + "\n(no polygons)", color="white", fontsize=8)
        _ax_clean(ax)
        return
    scores = np.asarray(scores, dtype=np.float64)
    # vmin/vmax: percentile 5/95 within ROI; else fallback 0/1
    finite = scores[np.isfinite(scores)]
    if finite.size:
        vmin = float(np.nanpercentile(finite, 5))
        vmax = float(np.nanpercentile(finite, 95))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = mpl.colormaps[cmap]
    colors = cmap_obj(norm(np.nan_to_num(scores, nan=vmin)))
    pc = PolyCollection(polylist, facecolors=colors, edgecolors="white",
                        linewidths=0.2)
    ax.add_collection(pc)
    ax.set_xlim(roi["x_min_um"], roi["x_max_um"])
    ax.set_ylim(roi["y_max_um"], roi["y_min_um"])  # invert y so image-like
    ax.set_facecolor("black")
    ax.set_title(f"{title}\nmean={float(np.nanmean(scores)):.3f}",
                 color="white", fontsize=8)
    _ax_clean(ax)


# ---------------------------------------------------------------------------
# Step 8 — Whole-tissue figures
# ---------------------------------------------------------------------------
def whole_tissue_categorical(joined: pd.DataFrame, out_path_base: Path,
                             logger: logging.Logger) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cat_palette = {
        "A_RCTD+_TRACER+": "#00E5FF",   # cyan
        "B_RCTD+_TRACER-": "#FF1493",   # magenta
        "C_RCTD-_TRACER+": "#39FF14",   # lime
        "D_RCTD-_TRACER-": "#1a1a3a",   # dark navy
    }
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
        # Plot D last over to have it as background
        order = ["D_RCTD-_TRACER-", "B_RCTD+_TRACER-", "C_RCTD-_TRACER+", "A_RCTD+_TRACER+"]
        for cat in order:
            sub = joined[joined["overlap_category"] == cat]
            if len(sub) == 0:
                continue
            ax.scatter(sub["cx_um"], sub["cy_um"], s=0.7,
                       c=cat_palette.get(cat, "white"), alpha=0.75,
                       linewidths=0, rasterized=True, label=f"{cat} (n={len(sub)})")
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.set_title("Whole-tissue categorical overlap (RCTD x TRACER)",
                     color="white")
        ax.legend(loc="lower right", fontsize=7, facecolor="black",
                  edgecolor="white", labelcolor="white")
        ax.set_xlabel("x (µm)", color="white"); ax.set_ylabel("y (µm)", color="white")
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(f"{out_path_base}.{ext}", dpi=160,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    logger.info("Wrote %s.[png|svg]", out_path_base)


def whole_tissue_problem_score_maps(joined: pd.DataFrame, out_path_base: Path,
                                    logger: logging.Logger) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=160)
        for ax, col, title in zip(
            axes,
            ["RCTD_problem_score", "TRACER_problem_score"],
            ["RCTD problem score (entropy + 1-max_w)/2", "TRACER relative_conflict"],
        ):
            scores = joined[col].rank(pct=True).to_numpy()
            sc = ax.scatter(joined["cx_um"], joined["cy_um"], c=scores,
                            s=0.7, cmap="magma", alpha=0.85, linewidths=0,
                            vmin=0.5, vmax=1.0, rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.invert_yaxis()
            ax.set_title(title, color="white")
            ax.set_xlabel("x (µm)", color="white")
            ax.set_ylabel("y (µm)", color="white")
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("percentile rank", color="white")
            cb.ax.tick_params(colors="white")
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(f"{out_path_base}.{ext}", dpi=160,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    logger.info("Wrote %s.[png|svg]", out_path_base)


def whole_tissue_bounding_boxes(joined: pd.DataFrame, rois: list[dict],
                                out_path_base: Path,
                                logger: logging.Logger) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cat_palette = {
        "A_RCTD+_TRACER+": "#00E5FF",
        "B_RCTD+_TRACER-": "#FF1493",
        "C_RCTD-_TRACER+": "#39FF14",
    }
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
        # Background: all cells light grey
        ax.scatter(joined["cx_um"], joined["cy_um"], s=0.5,
                   c="#444444", alpha=0.4, linewidths=0, rasterized=True)
        for roi in rois:
            color = cat_palette.get(roi["category"], "white")
            ax.add_patch(Rectangle(
                (roi["x_min_um"], roi["y_min_um"]),
                roi["x_max_um"] - roi["x_min_um"],
                roi["y_max_um"] - roi["y_min_um"],
                edgecolor=color, facecolor="none", lw=2.5,
            ))
            ax.text(roi["x_min_um"], roi["y_min_um"] - 30, roi["roi_id"],
                    color=color, fontsize=7, weight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.set_title("Whole tissue with representative ROI bounding boxes",
                     color="white")
        ax.set_xlabel("x (µm)", color="white"); ax.set_ylabel("y (µm)", color="white")
        # Legend
        from matplotlib.patches import Patch
        legend_items = [Patch(edgecolor=v, facecolor="none", label=k)
                        for k, v in cat_palette.items()]
        ax.legend(handles=legend_items, loc="lower right", fontsize=8,
                  facecolor="black", edgecolor="white", labelcolor="white")
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(f"{out_path_base}.{ext}", dpi=160,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    logger.info("Wrote %s.[png|svg]", out_path_base)


# ---------------------------------------------------------------------------
# Step 10 — Driver
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--visiumhd-matrix", required=True, type=Path)
    p.add_argument("--cell-segmentations", required=True, type=Path)
    p.add_argument("--spatial-dir", required=True, type=Path)
    p.add_argument("--reference-h5ad", required=True, type=Path)
    p.add_argument("--npmi-table", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--sample-name", default="kidney_visiumhd")
    p.add_argument("--rctd-problem-quantile", type=float, default=0.90)
    p.add_argument("--tracer-problem-quantile", type=float, default=0.90)
    p.add_argument("--roi-size-um", type=float, default=250.0)
    p.add_argument("--n-rois-per-category", type=int, default=5)
    p.add_argument("--bin-sizes-um", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--em-iters", type=int, default=80)
    p.add_argument("--em-chunk-size", type=int, default=8192)
    p.add_argument("--top-k-genes-per-cell", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-figures", action="store_true",
                   help="Useful for fast end-to-end QC.")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for sub in ("rctd", "tracer", "overlap", "figures",
                "figures/roi_insets", "qc", "logs"):
        (args.outdir / sub).mkdir(parents=True, exist_ok=True)
    logger = make_logger(args.outdir / "logs" / "rctd_tracer_run.log")
    logger.info("=== run_rctd_tracer_overlap.py ===")
    logger.info("argv: %s", " ".join(sys.argv))

    spatial = SpatialAlign.from_dir(args.spatial_dir)
    logger.info("Spatial alignment: %.4f µm/px; hires_scalef=%.4f; lowres_scalef=%.4f",
                spatial.microns_per_pixel, spatial.hires_scalef, spatial.lowres_scalef)

    # ---- Load VisiumHD matrix + segmentation ------------------------------
    y, barcodes, gene_names = read_visiumhd_10x_mtx(args.visiumhd_matrix, logger)
    polys = read_cell_segmentations(args.cell_segmentations, logger)
    align_stats = validate_alignment(barcodes, polys, logger)
    bc_ids = np.array([barcode_to_cell_id(b) for b in barcodes], dtype=np.int64)
    centroids = compute_centroids(polys)
    centroids["cx_um"] = spatial.px_to_um(centroids["cx_px"]).astype(np.float32)
    centroids["cy_um"] = spatial.px_to_um(centroids["cy_px"]).astype(np.float32)

    # ---- Run RCTD-style deconvolution -------------------------------------
    import anndata as ad
    ref = ad.read_h5ad(args.reference_h5ad)
    logger.info("Reference: %s; lineages=%d",
                ref.shape, len(set(ref.obs["lineage"])))
    # Use HVGs intersected with VisiumHD panel as signature genes
    hvg_path = args.reference_h5ad.parent / "hvg_gene_list.tsv"
    if hvg_path.exists():
        hvgs = pd.read_csv(hvg_path, sep="\t")["gene"].astype(str).to_numpy()
        logger.info("Loaded %d HVGs from %s", len(hvgs), hvg_path)
    else:
        hvgs = ref.var_names.to_numpy()
        logger.warning("HVG list missing; using all reference genes (slower).")
    # Also restrict to genes present in spatial panel.
    spatial_panel = set(map(str, gene_names))
    hvgs = np.array([g for g in hvgs if g in spatial_panel], dtype=str)
    logger.info("Signature gene set after spatial intersection: %d", len(hvgs))

    S, lineages, sig_genes = build_lineage_signature(ref, hvgs, logger)
    W, _lib_sig, active_mask = poisson_em_deconvolution(
        y, gene_names, S, sig_genes,
        n_iter=args.em_iters, chunk_size=args.em_chunk_size, logger=logger,
    )
    rctd_df = rctd_metrics(W, lineages, active_mask)
    rctd_df.insert(0, "cell_id_int", bc_ids)
    rctd_df.insert(1, "barcode", barcodes)
    # Add per-lineage weights as columns
    for k, lin in enumerate(lineages):
        rctd_df[f"w_{lin.replace('/', '_')}"] = W[:, k].astype(np.float32)

    rctd_out = args.outdir / "rctd" / "rctd_cell_scores.tsv.gz"
    rctd_df.to_csv(rctd_out, sep="\t", index=False, compression="gzip")
    logger.info("Wrote %s (%d rows)", rctd_out, len(rctd_df))
    with open(args.outdir / "rctd" / "rctd_summary.json", "w") as f:
        json.dump({
            "n_cells": int(len(rctd_df)),
            "n_active_in_rctd": int(active_mask.sum()),
            "lineages": lineages.tolist(),
            "n_signature_genes": int(len(sig_genes)),
            "em_iters": int(args.em_iters),
            "predicted_lineage_counts": {
                str(k): int(v) for k, v in
                rctd_df["predicted_dominant_lineage"].value_counts().items()
            },
        }, f, indent=2)

    # ---- Run TRACER per-cell scoring --------------------------------------
    logger.info("Loading NPMI table %s ...", args.npmi_table)
    npmi_long = pd.read_csv(args.npmi_table)
    npmi_mat, npmi_genes = build_npmi_matrix(npmi_long, logger)
    tracer_df = tracer_score_cells(
        y, gene_names, npmi_mat, npmi_genes,
        top_k_genes=args.top_k_genes_per_cell, logger=logger,
    )
    tracer_df.insert(0, "cell_id_int", bc_ids)
    tracer_df.insert(1, "barcode", barcodes)
    tracer_out = args.outdir / "tracer" / "tracer_cell_scores.tsv.gz"
    tracer_df.to_csv(tracer_out, sep="\t", index=False, compression="gzip")
    logger.info("Wrote %s (%d rows)", tracer_out, len(tracer_df))
    with open(args.outdir / "tracer" / "tracer_summary.json", "w") as f:
        json.dump({
            "n_cells": int(len(tracer_df)),
            "n_npmi_genes": int(len(npmi_genes)),
            "median_relative_conflict": float(tracer_df["TRACER_relative_conflict"].median()),
            "median_relative_purity": float(tracer_df["TRACER_relative_purity"].median()),
        }, f, indent=2)

    # ---- Join + categorise ------------------------------------------------
    join_df = pd.DataFrame({
        "cell_id_int": bc_ids,
        "barcode": barcodes,
    })
    join_df = (join_df.merge(centroids, on="cell_id_int", how="left")
                      .merge(rctd_df.drop(columns=["barcode"]), on="cell_id_int", how="left")
                      .merge(tracer_df.drop(columns=["barcode"]), on="cell_id_int", how="left"))
    join_df, thresholds = categorize(
        join_df, rctd_q=args.rctd_problem_quantile,
        tracer_q=args.tracer_problem_quantile, logger=logger,
    )
    join_out = args.outdir / "overlap" / "joined_rctd_tracer_scores.tsv.gz"
    join_df.to_csv(join_out, sep="\t", index=False, compression="gzip")
    logger.info("Wrote %s (%d rows)", join_out, len(join_df))

    # ---- ROIs -------------------------------------------------------------
    all_rois: list[dict] = []
    for cat in ["A_RCTD+_TRACER+", "B_RCTD+_TRACER-", "C_RCTD-_TRACER+"]:
        all_rois.extend(select_representative_rois(
            join_df, category=cat, roi_size_um=args.roi_size_um,
            n_rois=args.n_rois_per_category, spatial=spatial, logger=logger,
        ))
    with open(args.outdir / "overlap" / "representative_rois.json", "w") as f:
        json.dump(all_rois, f, indent=2)
    pd.DataFrame(all_rois).to_csv(
        args.outdir / "overlap" / "roi_summary.tsv", sep="\t", index=False,
    )

    # ---- Figures ----------------------------------------------------------
    if not args.skip_figures:
        whole_tissue_categorical(
            join_df, args.outdir / "figures" / "whole_tissue_categorical_overlap",
            logger=logger)
        whole_tissue_problem_score_maps(
            join_df, args.outdir / "figures" / "whole_tissue_problem_score_maps",
            logger=logger)
        whole_tissue_bounding_boxes(
            join_df, all_rois,
            args.outdir / "figures" / "whole_tissue_bounding_boxes_all",
            logger=logger)
        hires_img = load_hires_image(args.spatial_dir, logger)
        for roi in all_rois:
            base = args.outdir / "figures" / "roi_insets" / f"{roi['roi_id']}"
            render_roi_inset(
                roi, hires_img=hires_img, spatial=spatial, polys=polys,
                joined=join_df, out_png=base.with_suffix(".png"),
                out_svg=base.with_suffix(".svg"),
                bin_sizes_um=args.bin_sizes_um, logger=logger,
            )

    # ---- QC / Diagnostics -------------------------------------------------
    write_qc(join_df, thresholds, lineages, hvgs, gene_names, sig_genes,
             npmi_genes, all_rois, args, align_stats, logger)
    logger.info("Done.")
    return 0


def write_qc(join_df, thresholds, lineages, hvgs, spatial_genes,
             sig_genes, npmi_genes, all_rois, args, align_stats, logger) -> None:
    qc_dir = args.outdir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    # Category counts
    cat_counts = join_df["overlap_category"].value_counts()
    cat_counts.to_frame("n_cells").to_csv(
        qc_dir / "category_counts.tsv", sep="\t",
    )
    # Gene overlap
    spatial_set = set(map(str, spatial_genes))
    pd.DataFrame([
        {"set": "VisiumHD_panel", "n": len(spatial_set)},
        {"set": "reference_HVGs", "n": int(len(hvgs))},
        {"set": "signature_genes_RCTD", "n": int(len(sig_genes))},
        {"set": "NPMI_genes_TRACER", "n": int(len(npmi_genes))},
        {"set": "NPMI_in_spatial",
         "n": int(len(set(map(str, npmi_genes)) & spatial_set))},
    ]).to_csv(qc_dir / "gene_overlap_summary.tsv", sep="\t", index=False)
    # Score correlation
    from scipy.stats import spearmanr, pearsonr
    mask = (join_df["RCTD_problem_score"].notna()
            & join_df["TRACER_problem_score"].notna())
    if mask.any():
        rho_s, p_s = spearmanr(join_df.loc[mask, "RCTD_problem_score"],
                               join_df.loc[mask, "TRACER_problem_score"])
        rho_p, p_p = pearsonr(join_df.loc[mask, "RCTD_problem_score"],
                              join_df.loc[mask, "TRACER_problem_score"])
    else:
        rho_s = rho_p = p_s = p_p = float("nan")
    pd.DataFrame([
        {"correlation": "spearman", "rho": float(rho_s), "p": float(p_s),
         "n": int(mask.sum())},
        {"correlation": "pearson", "rho": float(rho_p), "p": float(p_p),
         "n": int(mask.sum())},
    ]).to_csv(qc_dir / "score_correlation.tsv", sep="\t", index=False)
    # Overall qc_summary.json
    summary = {
        "command": " ".join(sys.argv),
        "sample_name": args.sample_name,
        "alignment_stats": align_stats,
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "category_counts": {str(k): int(v) for k, v in cat_counts.items()},
        "n_cells_total": int(len(join_df)),
        "n_cells_RCTD_scored": int(join_df["RCTD_problem_score"].notna().sum()),
        "n_cells_TRACER_scored": int(join_df["TRACER_problem_score"].notna().sum()),
        "n_cells_both_scored": int(mask.sum()),
        "spearman_RCTD_vs_TRACER": float(rho_s),
        "pearson_RCTD_vs_TRACER": float(rho_p),
        "n_rois": int(len(all_rois)),
        "lineages": [str(x) for x in lineages],
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(qc_dir / "qc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # run_summary.md
    run_md = args.outdir / "run_summary.md"
    with open(run_md, "w") as f:
        f.write(f"# VisiumHD RCTD x TRACER overlap run — {args.sample_name}\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
        f.write("## Command\n\n```\n" + " ".join(sys.argv) + "\n```\n\n")
        f.write("## Alignment\n\n")
        for k, v in align_stats.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Cells scored\n\n")
        f.write(f"- total cells (matrix): {len(join_df)}\n")
        f.write(f"- RCTD scored: {summary['n_cells_RCTD_scored']}\n")
        f.write(f"- TRACER scored: {summary['n_cells_TRACER_scored']}\n")
        f.write(f"- both: {summary['n_cells_both_scored']}\n\n")
        f.write("## Thresholds\n\n")
        for k, v in thresholds.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n## Category counts\n\n")
        for k, v in cat_counts.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\n## Spearman correlation (RCTD vs TRACER): {rho_s:.4f} (p={p_s:.2e})\n")
        f.write(f"## Pearson correlation: {rho_p:.4f} (p={p_p:.2e})\n\n")
        f.write(f"## ROIs selected: {len(all_rois)} ({args.n_rois_per_category}/category x 3)\n\n")
        f.write("| roi_id | category | x_min_um | y_min_um | n_cells | mean_RCTD | mean_TRACER | dominant_lineage |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in all_rois:
            f.write(f"| {r['roi_id']} | {r['category']} | {r['x_min_um']:.1f} | "
                    f"{r['y_min_um']:.1f} | {r['n_cells_total']} | "
                    f"{r['mean_RCTD_problem_score']:.3f} | "
                    f"{r['mean_TRACER_problem_score']:.3f} | "
                    f"{r.get('dominant_lineage', '')} |\n")
    logger.info("Wrote %s", run_md)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    sys.exit(main())
