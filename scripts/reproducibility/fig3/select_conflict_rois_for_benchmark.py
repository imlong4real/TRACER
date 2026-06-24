#!/usr/bin/env python3
"""Fig3 cross-platform ROI selection for segmentation benchmarking.

For each spatial platform we:
  1. Load a matched scRNA reference. Build a column-stochastic per-lineage
     signature and score every spatial cell with the **Python RCTD
     (Poisson-EM) re-implementation** (``run_rctd_tracer_overlap``) to obtain a
     per-cell **RCTD problem score** = 0.5*norm_entropy + 0.5*(1 - max_weight).
     A deterministic scRNA PMI/NPMI panel is also emitted as a reference
     artifact (``npmi_panel.csv.gz``).
  2. Select ONE high-problem-score ROI by a predefined, reproducible algorithm
     (spatial smoothing of the RCTD problem score over a grid + ranked
     candidate windows). No visual cherry-picking. Targets ~1,000-2,000 cells
     and ~1-2M transcripts.
  3. Export a benchmark-ready, standardized transcript parquet. Boundary cells
     are kept whole: every transcript assigned to an ROI cell is included even
     if it lies outside the geometric ROI box, so no cell is cropped.
  4. Render whole-tissue RCTD problem-score maps (PNG+SVG) and a per-cell
     unique-RGB inset (ROI for the selection platforms; whole tissue for
     MERFISH).

Platforms:
  cosmx_nsclc       lung scRNA  vs CosMx NSCLC (Lung5_Rep1)        select ROI
  merfish_mouse_ileum gut scRNA vs MERFISH ileum (already an ROI) keep whole
  atera_cervical    cervical scRNA vs Atera (Xenium WTA-scale)    HVG + select
  xenium5k_cervical cervical scRNA vs Xenium 5K                   HVG + select

This script ONLY prepares standardized, conflict-enriched ROI inputs for
TRACER, cellAdmix, SPLIT, Baysor, proseg and Segger. It does NOT run any
benchmark algorithm.

Usage
-----
    python scripts/reproducibility/fig3/select_conflict_rois_for_benchmark.py \
        --only merfish_mouse_ileum            # one platform
    python scripts/reproducibility/fig3/select_conflict_rois_for_benchmark.py  # all
    python scripts/reproducibility/fig3/select_conflict_rois_for_benchmark.py \
        --summarize-only                       # rebuild aggregate summary only
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- repo imports -----------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT),
           str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging                 # noqa: E402
import anndata as ad           # noqa: E402
import duckdb                  # noqa: E402

# Python RCTD (Poisson-EM) re-implementation — ROI selection is driven by the
# per-cell RCTD problem score, not the TRACER NPMI conflict score.
from run_rctd_tracer_overlap import (   # noqa: E402
    build_lineage_signature,
    poisson_em_deconvolution,
    rctd_metrics,
)

_LOG = logging.getLogger("fig3_rctd")
if not _LOG.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    _LOG.addHandler(_h)
_LOG.setLevel(logging.INFO)

OUT_ROOT = _REPO_ROOT / "results" / "fig3_cross_platform_roi_benchmark"

# Standardized benchmark schema for roi_transcripts.parquet
BENCH_REQUIRED_COLS = ["x", "y", "feature_name", "cell_id", "platform", "sample", "roi_id"]
BENCH_OPTIONAL_COLS = ["transcript_id", "z", "overlaps_nucleus"]
UNASSIGNED = "UNASSIGNED"          # standardized sentinel for unassigned tx

# regexes for gene-class exclusion (whole-transcriptome references)
import re                       # noqa: E402
MITO_RE = re.compile(r"^(MT-|mt-|Mt-|MT\.)")
RIBO_RE = re.compile(r"^(RPS|RPL|Rps|Rpl|MRPS|MRPL)")
CONTROL_RE = re.compile(
    r"^(Neg|BLANK|Blank|Unassigned|Deprecated|Control|antisense_"
    r"|UnassignedCodeword_|NegControlProbe_|NegControlCodeword_"
    r"|SystemControl|DeprecatedCodeword_|Intergenic)"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# Platform configuration
# ===========================================================================
@dataclass
class Platform:
    key: str
    platform: str                 # platform label written into parquet
    sample: str
    reference_h5ad: str
    ref_celltype_col: str
    spatial_parquet: str
    # column names in the spatial parquet
    x_col: str
    y_col: str
    z_col: str | None
    feature_col: str = "feature_name"
    cell_col: str = "cell_id"
    transcript_id_col: str | None = "transcript_id"
    overlaps_nucleus_col: str | None = "overlaps_nucleus"
    is_gene_col: str | None = None
    unassigned_values: tuple[str, ...] = (UNASSIGNED,)
    # gene-panel strategy
    use_hvg: bool = False
    n_hvg: int = 2000
    exclude_mito_ribo: bool = False
    # centroid source: "cells_parquet" or "transcripts"
    centroid_source: str = "transcripts"
    cells_parquet: str | None = None
    cells_id_col: str = "cell_id"
    cells_x_col: str = "x_centroid"
    cells_y_col: str = "y_centroid"
    cells_ntx_col: str | None = "transcript_counts"
    # optional spatial cell-type map (cell_id -> type) for diversity
    celltype_csv: str | None = None
    celltype_csv_id_col: str | None = None
    celltype_csv_type_col: str | None = None
    celltype_csv_id_is_int: bool = False
    # selection control
    select_roi: bool = True       # if False, the whole dataset is the ROI
    # NPMI / scoring params
    min_cells_expressed: int = 25
    score_min_transcripts: int = 5


def build_platforms() -> dict[str, Platform]:
    D = str(_REPO_ROOT / "datasets")
    return {
        "cosmx_nsclc": Platform(
            key="cosmx_nsclc", platform="CosMx", sample="Lung5_Rep1",
            reference_h5ad=f"{D}/lung_cancer_scrna_36973297/lung_cancer_50k.h5ad",
            ref_celltype_col="Cell_Cluster_level1",
            spatial_parquet=f"{D}/lung_cancer_cosmx_bruker/processed/transcripts/lung5_rep1_transcripts.parquet",
            x_col="x", y_col="y", z_col="z",
            unassigned_values=("0",),
            use_hvg=False, exclude_mito_ribo=False,
            centroid_source="transcripts",
            celltype_csv=f"{D}/lung_cancer_cosmx_bruker/processed/annotations/lung5_rep1_cell_annotations.csv",
            celltype_csv_id_col="cell", celltype_csv_type_col="cell_type",
            select_roi=True, min_cells_expressed=25, score_min_transcripts=5,
        ),
        "merfish_mouse_ileum": Platform(
            key="merfish_mouse_ileum", platform="MERFISH", sample="mouse_ileum",
            reference_h5ad=f"{D}/gut_scrna_GSE92332/processed/h5ad/gut_scrna_gse92332_ileum_annotated.h5ad",
            ref_celltype_col="cell_type",
            spatial_parquet=f"{D}/gut_MERFISH_petukhov_2021/mouse_gut_df.parquet",
            x_col="x", y_col="y", z_col="z",
            unassigned_values=("0",),
            use_hvg=False, exclude_mito_ribo=False,
            centroid_source="transcripts",
            celltype_csv=f"{D}/gut_MERFISH_petukhov_2021/cell_assignment.csv",
            celltype_csv_id_col="cell", celltype_csv_type_col="leiden_final",
            celltype_csv_id_is_int=True,
            select_roi=False, min_cells_expressed=15, score_min_transcripts=5,
        ),
        "atera_cervical": Platform(
            key="atera_cervical", platform="Atera", sample="atera_cervical",
            reference_h5ad=f"{D}/cervical_cancer_scrna_10x/processed/h5ad/cervical_scrna_adc_scc_marker_annotated.h5ad",
            ref_celltype_col="cell_type",
            spatial_parquet=f"{D}/cervical_cancer_atera_10x/filtered_df.parquet",
            x_col="x_location", y_col="y_location", z_col="z_location",
            is_gene_col="is_gene", unassigned_values=(UNASSIGNED,),
            use_hvg=True, n_hvg=2000, exclude_mito_ribo=True,
            centroid_source="cells_parquet",
            cells_parquet=f"{D}/cervical_cancer_atera_10x/cells.parquet",
            select_roi=True, min_cells_expressed=100, score_min_transcripts=10,
        ),
        "xenium5k_cervical": Platform(
            key="xenium5k_cervical", platform="Xenium5K", sample="xenium5k_cervical",
            reference_h5ad=f"{D}/cervical_cancer_scrna_10x/processed/h5ad/cervical_scrna_adc_scc_marker_annotated.h5ad",
            ref_celltype_col="cell_type",
            spatial_parquet=f"{D}/cervical_cancer_xenium5k_10x/filtered_df.parquet",
            x_col="x_location", y_col="y_location", z_col="z_location",
            is_gene_col="is_gene", unassigned_values=(UNASSIGNED,),
            use_hvg=True, n_hvg=2000, exclude_mito_ribo=True,
            centroid_source="cells_parquet",
            cells_parquet=f"{D}/cervical_cancer_xenium5k_10x/cells.parquet",
            select_roi=True, min_cells_expressed=50, score_min_transcripts=10,
        ),
    }


# ===========================================================================
# Reference loading + raw-count validation
# ===========================================================================
def _looks_raw(X) -> bool:
    Xc = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    if Xc.nnz == 0:
        return False
    nz = Xc.data
    if nz.size > 50000:
        nz = np.random.default_rng(0).choice(nz, 50000, replace=False)
    nzf = nz.astype(np.float64)
    return (nzf >= 0).all() and np.allclose(nzf, np.round(nzf), atol=1e-8)


def load_reference(cfg: Platform):
    """Return (counts_csr cells x genes [raw], var_names, obs)."""
    log(f"  loading reference {Path(cfg.reference_h5ad).name}")
    a = ad.read_h5ad(cfg.reference_h5ad)
    for name, X, var in (("layers['counts']", a.layers.get("counts"), a.var_names),
                         ("raw.X", a.raw.X if a.raw is not None else None,
                          a.raw.var_names if a.raw is not None else None),
                         ("X", a.X, a.var_names)):
        if X is None:
            continue
        if _looks_raw(X):
            log(f"    raw counts from {name}")
            csr = (sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()).astype(np.float64)
            return csr, np.asarray(var, dtype=str), a.obs.copy()
    raise SystemExit(f"No raw counts found in {cfg.reference_h5ad}")


# ===========================================================================
# Gene panel selection (overlap; optional HVG)
# ===========================================================================
def spatial_gene_panel(cfg: Platform) -> set[str]:
    con = duckdb.connect()
    where = f"WHERE {cfg.is_gene_col}" if cfg.is_gene_col else ""
    rows = con.execute(
        f"SELECT DISTINCT {cfg.feature_col} FROM read_parquet('{cfg.spatial_parquet}') {where}"
    ).fetchnumpy()[cfg.feature_col]
    con.close()
    genes = {str(g) for g in rows}
    genes = {g for g in genes if not CONTROL_RE.search(g)}
    return genes


def select_genes(cfg: Platform, counts: sp.csr_matrix, ref_var: np.ndarray,
                 spatial_genes: set[str]) -> tuple[list[str], dict]:
    """Pick the gene set used to build NPMI. Returns (genes, stats)."""
    pres = counts.copy()
    pres.data = (pres.data > 0).astype(np.int8)
    n_cells_per_gene = np.asarray(pres.sum(axis=0)).ravel().astype(np.int64)
    ncpg = dict(zip(ref_var.tolist(), n_cells_per_gene.tolist()))

    overlap = sorted(set(ref_var.tolist()) & spatial_genes)
    stats = {
        "n_spatial_genes": int(len(spatial_genes)),
        "n_reference_genes": int(len(ref_var)),
        "n_overlap_genes": int(len(overlap)),
    }

    # control + (optional) mito/ribo + low-detection filters
    cand = []
    for g in overlap:
        if CONTROL_RE.search(g):
            continue
        if cfg.exclude_mito_ribo and (MITO_RE.search(g) or RIBO_RE.search(g)):
            continue
        if ncpg.get(g, 0) < cfg.min_cells_expressed:
            continue
        cand.append(g)

    if not cfg.use_hvg:
        stats["gene_selection"] = "all_overlap_filtered"
        stats["n_genes_for_npmi"] = int(len(cand))
        return cand, stats

    # HVG among candidate overlap genes (seurat flavor on log-norm or computed)
    import scanpy as sc
    idx = [i for i, g in enumerate(ref_var) if g in set(cand)]
    sub = ad.AnnData(X=counts[:, idx].copy(),
                     var=pd.DataFrame(index=np.asarray(ref_var)[idx]))
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    n_top = min(cfg.n_hvg, sub.n_vars - 1)
    sc.pp.highly_variable_genes(sub, n_top_genes=n_top, flavor="seurat")
    hvg = sub.var_names[sub.var["highly_variable"].to_numpy()].tolist()
    stats["gene_selection"] = f"hvg_top{n_top}_of_overlap"
    stats["n_candidate_after_filters"] = int(len(cand))
    stats["n_genes_for_npmi"] = int(len(hvg))
    return hvg, stats


# ===========================================================================
# Deterministic NPMI from raw-count presence
# ===========================================================================
def compute_npmi(counts: sp.csr_matrix, ref_var: np.ndarray, gene_set: list[str],
                 *, min_cells: int):
    """Presence-based PMI/NPMI over gene_set. Returns
    (genes_universe, npmi_mat float32 GxG, npmi_long DataFrame)."""
    gs = set(gene_set)
    idx = [i for i, g in enumerate(ref_var) if g in gs]
    genes = np.asarray(ref_var)[idx].astype(str)
    pres = counts[:, idx].copy()
    pres.data = (pres.data > 0).astype(np.float64)
    pres.eliminate_zeros()
    n_i = np.asarray(pres.sum(axis=0)).ravel().astype(np.int64)
    keep = n_i >= min_cells
    pres = pres[:, keep].tocsr()
    genes = genes[keep]
    n_i = n_i[keep]
    G = len(genes)
    N = pres.shape[0]
    if G < 2:
        raise SystemExit(f"Only {G} genes survived NPMI filtering.")

    # Co-occurrence counts (genes x genes) via sparse Gram matrix.
    C = (pres.T @ pres).toarray()
    p_i = n_i / N

    iu, ju = np.triu_indices(G, k=1)
    cij = C[iu, ju].astype(np.float64)
    mask = cij > 0
    iu, ju, cij = iu[mask], ju[mask], cij[mask]
    p_ij = cij / N
    p_prod = p_i[iu] * p_i[ju]
    pmi = np.log(p_ij / p_prod)
    npmi = pmi / (-np.log(p_ij))

    npmi_mat = np.eye(G, dtype=np.float32)
    npmi_mat[iu, ju] = npmi.astype(np.float32)
    npmi_mat[ju, iu] = npmi.astype(np.float32)

    npmi_long = pd.DataFrame({
        "gene_i": genes[iu], "gene_j": genes[ju],
        "n_cells_i": n_i[iu], "n_cells_j": n_i[ju],
        "n_cells_ij": cij.astype(np.int64),
        "p_i": p_i[iu], "p_j": p_i[ju], "p_ij": p_ij,
        "PMI": pmi, "NPMI": npmi,
    })
    return genes, npmi_mat, npmi_long


# ===========================================================================
# Spatial cell x gene COUNT matrix (cells x genes) via duckdb (out-of-core)
# ===========================================================================
def build_counts(cfg: Platform, genes_universe: np.ndarray):
    """Return (cell_ids np.ndarray[str], Y csr float32 cells x G) of integer
    transcript counts over genes_universe (the RCTD signature gene order)."""
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    vocab = pd.DataFrame({"gene": genes_universe.astype(str),
                          "gidx": np.arange(len(genes_universe), dtype=np.int32)})
    con.register("vocab", vocab)
    unassigned = ", ".join("'" + u + "'" for u in cfg.unassigned_values)
    gene_filter = f"AND t.{cfg.is_gene_col}" if cfg.is_gene_col else ""

    base = f"""
        SELECT CAST(t.{cfg.cell_col} AS VARCHAR) AS cid, t.{cfg.feature_col} AS fn
        FROM read_parquet('{cfg.spatial_parquet}') t
        WHERE CAST(t.{cfg.cell_col} AS VARCHAR) NOT IN ({unassigned}) {gene_filter}
    """
    con.execute(f"CREATE TEMP VIEW assigned AS {base}")
    con.execute("""
        CREATE TEMP TABLE cnt AS
        SELECT a.cid AS cid, v.gidx AS gidx, COUNT(*)::INTEGER AS n
        FROM assigned a JOIN vocab v ON a.fn = v.gene
        GROUP BY a.cid, v.gidx
    """)
    con.execute("""
        CREATE TEMP TABLE cmap AS
        SELECT cid, (row_number() OVER (ORDER BY cid) - 1)::INTEGER AS row
        FROM (SELECT DISTINCT cid FROM cnt)
    """)
    coded = con.execute("""
        SELECT cm.row AS row, c.gidx AS col, c.n AS n
        FROM cnt c JOIN cmap cm ON c.cid = cm.cid
    """).fetchnumpy()
    cell_ids = con.execute("SELECT cid FROM cmap ORDER BY row").fetchnumpy()["cid"].astype(str)
    con.close()

    rows = coded["row"].astype(np.int32)
    cols = coded["col"].astype(np.int32)
    vals = coded["n"].astype(np.float32)
    n_cells = len(cell_ids)
    G = len(genes_universe)
    Y = sp.coo_matrix((vals, (rows, cols)), shape=(n_cells, G)).tocsr()
    return cell_ids, Y


def rctd_score_cells(counts_ref: sp.csr_matrix, ref_var: np.ndarray,
                     ref_lineage: np.ndarray, genes_universe: np.ndarray,
                     Y: sp.csr_matrix, *, n_iter: int = 30, chunk: int = 20_000):
    """Score spatial cells with the Python RCTD (Poisson-EM) re-implementation.

    Returns a DataFrame (one row per spatial cell, aligned to Y rows) with
    RCTD_problem_score, RCTD_max_weight, RCTD_norm_entropy and
    predicted_dominant_lineage.
    """
    # Build a lightweight reference AnnData carrying raw counts + obs['lineage'].
    ref = ad.AnnData(
        X=counts_ref.copy(),
        obs=pd.DataFrame({"lineage": np.asarray(ref_lineage, dtype=str)}),
        var=pd.DataFrame(index=np.asarray(ref_var, dtype=str)),
    )
    ref.layers["counts"] = counts_ref.copy()
    S, lineages, sig_genes = build_lineage_signature(
        ref, np.asarray(genes_universe, dtype=str), _LOG)
    W, _counts_align, active_mask = poisson_em_deconvolution(
        Y, np.asarray(genes_universe, dtype=str), S, sig_genes,
        n_iter=n_iter, chunk_size=chunk, logger=_LOG)
    met = rctd_metrics(W, lineages, active_mask)
    return met


# ===========================================================================
# Cell centroids + transcript counts + (optional) cell types
# ===========================================================================
def cell_table(cfg: Platform, scored_cell_ids: np.ndarray, rctd: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"cell_id": scored_cell_ids.astype(str)})
    for c in ("RCTD_problem_score", "RCTD_max_weight", "RCTD_norm_entropy",
              "RCTD_margin", "predicted_dominant_lineage"):
        df[c] = rctd[c].to_numpy()

    con = duckdb.connect(); con.execute("PRAGMA threads=4")
    if cfg.centroid_source == "cells_parquet":
        cells = con.execute(f"""
            SELECT CAST({cfg.cells_id_col} AS VARCHAR) AS cell_id,
                   {cfg.cells_x_col} AS x, {cfg.cells_y_col} AS y,
                   {cfg.cells_ntx_col} AS n_tx
            FROM read_parquet('{cfg.cells_parquet}')
        """).fetch_df()
    else:
        unassigned = ", ".join("'" + u + "'" for u in cfg.unassigned_values)
        gene_filter = f"AND {cfg.is_gene_col}" if cfg.is_gene_col else ""
        cells = con.execute(f"""
            SELECT CAST({cfg.cell_col} AS VARCHAR) AS cell_id,
                   avg({cfg.x_col}) AS x, avg({cfg.y_col}) AS y, count(*) AS n_tx
            FROM read_parquet('{cfg.spatial_parquet}')
            WHERE CAST({cfg.cell_col} AS VARCHAR) NOT IN ({unassigned}) {gene_filter}
            GROUP BY 1
        """).fetch_df()
    con.close()
    cells["cell_id"] = cells["cell_id"].astype(str)
    df = df.merge(cells, on="cell_id", how="left")
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)

    # RCTD predicted dominant lineage provides cell-type diversity uniformly
    # across platforms (used by the ROI selector's diversity tiebreak).
    df["cell_type"] = df["predicted_dominant_lineage"]
    return df


# ===========================================================================
# Reproducible ROI selector (grid smoothing + integral-image window scan)
# ===========================================================================
def _integral(a: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))


def _win_sum(I: np.ndarray, r0, r1, c0, c1) -> np.ndarray:
    return I[r1, c1] - I[r0, c1] - I[r1, c0] + I[r0, c0]


def select_roi(cells: pd.DataFrame, *, score_col="RCTD_problem_score",
               target_cells=1500, cell_band=(1000, 2000),
               tx_target=1.5e6, tx_band=(0.8e6, 2.2e6), conflict_hi_q=0.75,
               seed=1, border_frac=0.03):
    """Predefined ROI selection by high RCTD problem score. Returns (roi, ranking_df).

    The integral-image scan pre-filters on a slightly widened cell band (the
    fine-grid bin edges only approximate cell membership); candidates are then
    re-counted exactly and hard-filtered to ``cell_band`` before ranking."""
    coarse_band = (cell_band[0] * 0.8, cell_band[1] * 1.2)
    x = cells["x"].to_numpy(); y = cells["y"].to_numpy()
    rc = np.nan_to_num(cells[score_col].to_numpy(), nan=0.0)
    ntx = np.nan_to_num(cells["n_tx"].to_numpy(), nan=0.0)
    has_type = cells["cell_type"].notna().any()
    type_codes = pd.factorize(cells["cell_type"])[0] if has_type else None

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    W, H = xmax - xmin, ymax - ymin
    area = max(W * H, 1.0)
    density = len(cells) / area
    hi_thr = np.quantile(rc[rc > 0], conflict_hi_q) if (rc > 0).any() else np.inf
    is_hi = (rc >= hi_thr).astype(np.float64)

    bx = border_frac * W
    by = border_frac * H

    records = []
    # Sweep several window scales so we can hit the cell/transcript target.
    for scale in (0.8, 1.0, 1.25, 1.6):
        L = float(np.sqrt(target_cells / max(density, 1e-12)) * scale)
        L = min(L, 0.6 * min(W, H))
        if L <= 0:
            continue
        nb = 6                              # fine bins per window side
        b = L / nb
        gx = max(int(np.ceil(W / b)), nb + 1)
        gy = max(int(np.ceil(H / b)), nb + 1)
        ix = np.clip(((x - xmin) / b).astype(int), 0, gx - 1)
        iy = np.clip(((y - ymin) / b).astype(int), 0, gy - 1)
        shape = (gy, gx)
        C = np.zeros(shape); S = np.zeros(shape); T = np.zeros(shape); Hh = np.zeros(shape)
        np.add.at(C, (iy, ix), 1.0)
        np.add.at(S, (iy, ix), rc)
        np.add.at(T, (iy, ix), ntx)
        np.add.at(Hh, (iy, ix), is_hi)
        IC, IS, IT, IH = _integral(C), _integral(S), _integral(T), _integral(Hh)

        stride = max(1, nb // 3)
        r0s = np.arange(0, gy - nb + 1, stride)
        c0s = np.arange(0, gx - nb + 1, stride)
        for r0 in r0s:
            for c0 in c0s:
                r1, c1 = r0 + nb, c0 + nb
                n_cells = _win_sum(IC, r0, r1, c0, c1)
                if n_cells < coarse_band[0] or n_cells > coarse_band[1]:
                    continue
                x0 = xmin + c0 * b; x1 = xmin + c1 * b
                y0 = ymin + r0 * b; y1 = ymin + r1 * b
                # avoid tissue edges
                if x0 < xmin + bx or x1 > xmax - bx or y0 < ymin + by or y1 > ymax - by:
                    continue
                s_rc = _win_sum(IS, r0, r1, c0, c1)
                n_hi = _win_sum(IH, r0, r1, c0, c1)
                n_tx = _win_sum(IT, r0, r1, c0, c1)
                mean_rc = s_rc / max(n_cells, 1)
                records.append((x0, x1, y0, y1, float(n_cells), float(n_tx),
                                float(mean_rc), float(n_hi), float(L)))

    if not records:
        # Fallback: densest-conflict single window at base scale, relaxed bands.
        L = float(np.sqrt(target_cells / max(density, 1e-12)))
        x0 = float(np.clip(np.median(x[rc >= hi_thr]) - L / 2, xmin, xmax - L)) if np.isfinite(hi_thr) else xmin
        y0 = float(np.clip(np.median(y[rc >= hi_thr]) - L / 2, ymin, ymax - L)) if np.isfinite(hi_thr) else ymin
        records.append((x0, x0 + L, y0, y0 + L, np.nan, np.nan, np.nan, np.nan, L))

    cols = ["xmin", "xmax", "ymin", "ymax", "n_cells", "n_transcripts",
            "mean_problem_score", "n_high_problem_cells", "window_side"]
    cand = pd.DataFrame.from_records(records, columns=cols)

    # Exact recompute + diversity for the strongest candidates (top by mean score).
    cand = cand.sort_values("mean_problem_score", ascending=False).head(150).reset_index(drop=True)
    div = np.zeros(len(cand)); exact_cells = np.zeros(len(cand)); exact_hi = np.zeros(len(cand))
    exact_rc = np.zeros(len(cand)); exact_tx = np.zeros(len(cand))
    for i, row in cand.iterrows():
        m = ((x >= row.xmin) & (x < row.xmax) & (y >= row.ymin) & (y < row.ymax))
        exact_cells[i] = m.sum()
        exact_tx[i] = ntx[m].sum()
        exact_rc[i] = rc[m].mean() if m.any() else 0.0
        exact_hi[i] = is_hi[m].sum()
        if has_type and m.any():
            vc = np.bincount(type_codes[m][type_codes[m] >= 0])
            p = vc[vc > 0] / vc.sum()
            div[i] = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
        else:
            div[i] = np.nan
    cand["n_cells"] = exact_cells
    cand["n_transcripts"] = exact_tx
    cand["mean_problem_score"] = exact_rc
    cand["n_high_problem_cells"] = exact_hi
    cand["celltype_diversity"] = div
    cand["tx_distance"] = (cand["n_transcripts"] - tx_target).abs()
    cand["cell_distance"] = (cand["n_cells"] - target_cells).abs()

    # Hard-enforce the exact cell band (fall back to the widened coarse band if
    # no window lands exactly inside, so an ROI is always produced).
    in_band = cand["n_cells"].between(cell_band[0], cell_band[1])
    if in_band.any():
        cand = cand[in_band].reset_index(drop=True)
    else:
        in_coarse = cand["n_cells"].between(coarse_band[0], coarse_band[1])
        if in_coarse.any():
            cand = cand[in_coarse].reset_index(drop=True)

    # Rank: 1) mean RCTD problem score 2) #high-problem cells 3) tx near target
    #       4) cell near target 5) lineage diversity
    div_key = cand["celltype_diversity"].fillna(-1.0)
    cand = cand.assign(_div=div_key).sort_values(
        by=["mean_problem_score", "n_high_problem_cells", "tx_distance",
            "cell_distance", "_div"],
        ascending=[False, False, True, True, False],
    ).drop(columns="_div").reset_index(drop=True)
    cand.insert(0, "rank", np.arange(1, len(cand) + 1))

    top = cand.iloc[0]
    reason = (f"highest mean RCTD problem score ({top.mean_problem_score:.3f}) among "
              f"reproducible candidate windows with {int(top.n_cells)} cells / "
              f"{int(top.n_transcripts):,} transcripts (target ~1.5k cells, ~1.5M tx); "
              f"{int(top.n_high_problem_cells)} high-problem cells "
              f"(>= q{int(conflict_hi_q*100)} RCTD problem score).")
    roi = dict(xmin=float(top.xmin), xmax=float(top.xmax),
               ymin=float(top.ymin), ymax=float(top.ymax),
               window_side=float(top.window_side), reason=reason,
               high_conflict_threshold=float(hi_thr) if np.isfinite(hi_thr) else None)
    return roi, cand


# ===========================================================================
# ROI transcript extraction (whole boundary cells) via duckdb
# ===========================================================================
def extract_roi_transcripts(cfg: Platform, roi: dict | None, roi_cell_ids: np.ndarray,
                            out_parquet: Path) -> dict:
    con = duckdb.connect(); con.execute("PRAGMA threads=4")
    tid = (f"t.{cfg.transcript_id_col} AS transcript_id"
           if cfg.transcript_id_col else "NULL AS transcript_id")
    zc = f"t.{cfg.z_col} AS z" if cfg.z_col else "NULL AS z"
    onuc = (f"t.{cfg.overlaps_nucleus_col} AS overlaps_nucleus"
            if cfg.overlaps_nucleus_col else "NULL AS overlaps_nucleus")
    unassigned = ", ".join("'" + u + "'" for u in cfg.unassigned_values)
    cell_norm = (f"CASE WHEN CAST(t.{cfg.cell_col} AS VARCHAR) IN ({unassigned}) "
                 f"THEN '{UNASSIGNED}' ELSE CAST(t.{cfg.cell_col} AS VARCHAR) END")

    select = f"""
        SELECT t.{cfg.x_col} AS x, t.{cfg.y_col} AS y,
               t.{cfg.feature_col} AS feature_name,
               {cell_norm} AS cell_id,
               {tid}, {zc}, {onuc},
               '{cfg.platform}' AS platform, '{cfg.sample}' AS sample,
               '{cfg.key}_roi' AS roi_id
        FROM read_parquet('{cfg.spatial_parquet}') t
    """
    if roi is None:
        where = ""  # whole dataset is the ROI (MERFISH)
    else:
        con.register("roi_ids", pd.DataFrame({"cid": roi_cell_ids.astype(str)}))
        where = f"""
            WHERE {cell_norm} IN (SELECT cid FROM roi_ids)
               OR (CAST(t.{cfg.cell_col} AS VARCHAR) IN ({unassigned})
                   AND t.{cfg.x_col} >= {roi['xmin']} AND t.{cfg.x_col} < {roi['xmax']}
                   AND t.{cfg.y_col} >= {roi['ymin']} AND t.{cfg.y_col} < {roi['ymax']})
        """
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({select} {where}) TO '{out_parquet}' (FORMAT PARQUET)")

    # exact counts from the written parquet
    n_tx = con.execute(f"SELECT count(*) FROM read_parquet('{out_parquet}')").fetchone()[0]
    n_unassigned = con.execute(
        f"SELECT count(*) FROM read_parquet('{out_parquet}') WHERE cell_id='{UNASSIGNED}'"
    ).fetchone()[0]
    n_cells = con.execute(
        f"SELECT count(DISTINCT cell_id) FROM read_parquet('{out_parquet}') "
        f"WHERE cell_id <> '{UNASSIGNED}'"
    ).fetchone()[0]
    bounds = con.execute(
        f"SELECT min(x), max(x), min(y), max(y) FROM read_parquet('{out_parquet}')"
    ).fetchone()
    con.close()
    return dict(n_transcripts=int(n_tx), n_unassigned=int(n_unassigned),
                n_cells=int(n_cells),
                frac_unassigned=float(n_unassigned) / max(int(n_tx), 1),
                x_min=float(bounds[0]), x_max=float(bounds[1]),
                y_min=float(bounds[2]), y_max=float(bounds[3]))


# ===========================================================================
# CIELAB per-cell RGB inset
# ===========================================================================
def cielab_palette(n: int, seed: int) -> np.ndarray:
    from skimage.color import lab2rgb
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    hue = (i * 137.508) % 360.0                       # golden-angle hue spacing
    L = 60.0 + (rng.random(n) - 0.5) * 26.0           # ~47-73 lightness
    C = 42.0 + (rng.random(n) - 0.5) * 24.0           # chroma
    a = C * np.cos(np.deg2rad(hue))
    bb = C * np.sin(np.deg2rad(hue))
    lab = np.stack([L, a, bb], axis=1).reshape(-1, 1, 3)
    rgb = lab2rgb(lab).reshape(-1, 3)
    rng.shuffle(rgb)                                  # decorrelate spatial neighbors
    return np.clip(rgb, 0, 1)


def render_rgb_inset(roi_parquet: Path, out_base: Path, *, roi: dict | None,
                     title_prefix: str, seed: int, max_points: int = 1_200_000):
    con = duckdb.connect()
    df = con.execute(
        f"SELECT x, y, cell_id FROM read_parquet('{roi_parquet}')"
    ).fetch_df()
    con.close()
    assigned = df[df["cell_id"] != UNASSIGNED]
    unassigned = df[df["cell_id"] == UNASSIGNED]
    uniq = assigned["cell_id"].unique()
    cmap = {c: i for i, c in enumerate(uniq)}
    pal = cielab_palette(len(uniq), seed)

    a = assigned
    if len(a) > max_points:
        a = a.sample(max_points, random_state=seed)
    colors = pal[a["cell_id"].map(cmap).to_numpy()]

    xr = (df["x"].min(), df["x"].max()); yr = (df["y"].min(), df["y"].max())
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        if len(unassigned):
            uu = unassigned if len(unassigned) <= max_points else unassigned.sample(max_points, random_state=seed)
            ax.scatter(uu["x"], uu["y"], s=0.4, c="#555555", alpha=0.25,
                       lw=0, rasterized=True)
        ax.scatter(a["x"], a["y"], s=0.7, c=colors, alpha=0.55, lw=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(*xr); ax.set_ylim(*yr)
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        ax.set_title(f"{title_prefix}\n"
                     f"x=[{xr[0]:.0f},{xr[1]:.0f}] y=[{yr[0]:.0f},{yr[1]:.0f}] | "
                     f"{len(uniq):,} cells, {len(df):,} transcripts",
                     fontsize=10)
        for ext in ("png", "svg"):
            fig.savefig(out_base.with_suffix("." + ext), bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        plt.close(fig)


# ===========================================================================
# QC plots
# ===========================================================================
def qc_whole_tissue_rctd(cells: pd.DataFrame, out_base: Path, title: str,
                         roi: dict | None = None):
    """Full-tissue map coloured by per-cell RCTD problem score (raw score +
    percentile rank). Optional ROI bbox overlay."""
    ps = np.nan_to_num(cells["RCTD_problem_score"].to_numpy(), nan=0.0)
    rank = pd.Series(ps).rank(pct=True).to_numpy()
    xx = cells["x"].to_numpy(); yy = cells["y"].to_numpy()
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
        for ax, vals, cm, lab, vlo, vhi in (
            (axes[0], ps, "magma", "RCTD problem score", 0.0, float(np.nanmax(ps)) or 1.0),
            (axes[1], rank, "magma", "RCTD problem score (percentile)", 0.5, 1.0)):
            order = np.argsort(vals)
            scv = ax.scatter(xx[order], yy[order], c=vals[order], s=2, cmap=cm,
                             lw=0, rasterized=True, vmin=vlo, vmax=vhi)
            ax.set_aspect("equal"); ax.invert_yaxis()
            ax.set_title(lab); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(scv, ax=ax, shrink=0.7, label=lab)
            if roi is not None:
                ax.add_patch(Rectangle((roi["xmin"], roi["ymin"]),
                                       roi["xmax"] - roi["xmin"],
                                       roi["ymax"] - roi["ymin"],
                                       fill=False, ec="cyan", lw=1.8))
        fig.suptitle(title, fontsize=11, color="white")
        for ext in ("png", "svg"):
            fig.savefig(out_base.with_suffix("." + ext), dpi=300,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def qc_bbox(cells: pd.DataFrame, roi: dict, out_base: Path, title: str):
    ps = np.nan_to_num(cells["RCTD_problem_score"].to_numpy(), nan=0.0)
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
        order = np.argsort(ps)
        sc = ax.scatter(cells["x"].to_numpy()[order], cells["y"].to_numpy()[order],
                        c=ps[order], s=2, cmap="magma", lw=0, rasterized=True,
                        vmin=0, vmax=float(np.nanmax(ps)) or 1.0)
        ax.add_patch(Rectangle((roi["xmin"], roi["ymin"]),
                               roi["xmax"] - roi["xmin"], roi["ymax"] - roi["ymin"],
                               fill=False, ec="cyan", lw=2.2))
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(sc, ax=ax, shrink=0.7, label="RCTD problem score")
        for ext in ("png", "svg"):
            fig.savefig(out_base.with_suffix("." + ext), dpi=300,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ===========================================================================
# Schema validation
# ===========================================================================
def validate_schema(roi_parquet: Path, out_json: Path) -> dict:
    con = duckdb.connect()
    cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{roi_parquet}')").fetch_df()
    present = cols["column_name"].tolist()
    n = con.execute(f"SELECT count(*) FROM read_parquet('{roi_parquet}')").fetchone()[0]
    distinct_unassigned = con.execute(
        f"SELECT DISTINCT cell_id FROM read_parquet('{roi_parquet}') "
        f"WHERE cell_id IN ('0','-1','{UNASSIGNED}','nan','None')"
    ).fetch_df()["cell_id"].tolist()
    con.close()
    report = {
        "parquet": str(roi_parquet),
        "n_rows": int(n),
        "columns_present": present,
        "required_columns_ok": all(c in present for c in BENCH_REQUIRED_COLS),
        "missing_required": [c for c in BENCH_REQUIRED_COLS if c not in present],
        "optional_present": [c for c in BENCH_OPTIONAL_COLS if c in present],
        "unassigned_sentinel": UNASSIGNED,
        "unassigned_sentinels_observed": distinct_unassigned,
        "unassigned_consistent": set(distinct_unassigned).issubset({UNASSIGNED}),
    }
    out_json.write_text(json.dumps(report, indent=2))
    return report


# ===========================================================================
# Per-platform driver
# ===========================================================================
def process_platform(cfg: Platform) -> dict:
    log(f"=== {cfg.key} ({cfg.platform}) ===")
    outdir = OUT_ROOT / cfg.key
    outdir.mkdir(parents=True, exist_ok=True)

    counts, ref_var, ref_obs = load_reference(cfg)
    if cfg.ref_celltype_col not in ref_obs.columns:
        raise SystemExit(f"Reference missing celltype col '{cfg.ref_celltype_col}'")
    ref_lineage = ref_obs[cfg.ref_celltype_col].astype(str).to_numpy()
    spatial_genes = spatial_gene_panel(cfg)
    genes_for_npmi, gene_stats = select_genes(cfg, counts, ref_var, spatial_genes)
    log(f"  genes: spatial={gene_stats['n_spatial_genes']} ref={gene_stats['n_reference_genes']} "
        f"overlap={gene_stats['n_overlap_genes']} npmi={gene_stats['n_genes_for_npmi']}")

    genes_universe, npmi_mat, npmi_long = compute_npmi(
        counts, ref_var, genes_for_npmi, min_cells=cfg.min_cells_expressed)
    log(f"  NPMI: {len(genes_universe)} genes, {len(npmi_long):,} gene pairs")
    npmi_long.to_csv(outdir / "npmi_panel.csv.gz", index=False, compression="gzip")
    if cfg.use_hvg:
        pd.DataFrame({"gene": genes_universe}).to_csv(
            outdir / "hvg_gene_list.tsv", sep="\t", index=False)

    log("  building spatial cell x gene count matrix (duckdb)")
    cell_ids, Y = build_counts(cfg, genes_universe)
    log(f"  RCTD Poisson-EM on {Y.shape[0]:,} cells x {Y.shape[1]} genes")
    rctd = rctd_score_cells(counts, ref_var, ref_lineage, genes_universe, Y)

    cells = cell_table(cfg, cell_ids, rctd)
    # apply scoring min-transcripts floor
    cells = cells[cells["n_tx"] >= cfg.score_min_transcripts].reset_index(drop=True)
    log(f"  scored cells with centroids: {len(cells):,}")

    # --- ROI selection ----------------------------------------------------
    if cfg.select_roi:
        roi, ranking = select_roi(cells, seed=1)
        ranking.to_csv(outdir / "candidate_roi_ranking.tsv", sep="\t", index=False)
        roi_cells = cells[(cells["x"] >= roi["xmin"]) & (cells["x"] < roi["xmax"]) &
                          (cells["y"] >= roi["ymin"]) & (cells["y"] < roi["ymax"])].copy()
        roi_cell_ids = roi_cells["cell_id"].to_numpy()
        log(f"  selected ROI x=[{roi['xmin']:.0f},{roi['xmax']:.0f}] "
            f"y=[{roi['ymin']:.0f},{roi['ymax']:.0f}] cells={len(roi_cells):,}")
    else:
        roi = None
        roi_cells = cells.copy()
        roi_cell_ids = roi_cells["cell_id"].to_numpy()
        log(f"  whole dataset used as ROI: cells={len(roi_cells):,}")

    # --- transcript extraction (whole boundary cells) ---------------------
    roi_parquet = outdir / "roi_transcripts.parquet"
    extract = extract_roi_transcripts(cfg, roi, roi_cell_ids, roi_parquet)
    log(f"  ROI transcripts: {extract['n_transcripts']:,} "
        f"({extract['frac_unassigned']*100:.1f}% unassigned); "
        f"cells: {extract['n_cells']:,}")

    # roi cells / scores tables
    roi_cells_out = roi_cells[["cell_id", "x", "y", "n_tx", "cell_type"]].copy()
    roi_cells_out.to_csv(outdir / "roi_cells.tsv.gz", sep="\t", index=False,
                         compression="gzip")
    roi_cells[["cell_id", "x", "y", "n_tx", "RCTD_problem_score",
               "RCTD_max_weight", "RCTD_norm_entropy", "RCTD_margin",
               "predicted_dominant_lineage", "cell_type"]].to_csv(
        outdir / "roi_cell_scores.tsv.gz", sep="\t", index=False, compression="gzip")

    # --- schema validation ------------------------------------------------
    schema = validate_schema(roi_parquet, outdir / "benchmark_input_schema_validation.json")

    # --- figures ----------------------------------------------------------
    # Full-tissue RCTD problem-score map (all platforms, incl. MERFISH).
    qc_whole_tissue_rctd(cells, outdir / "qc_whole_tissue_rctd_problem_score",
                         f"{cfg.platform} {cfg.sample} — RCTD problem score", roi)
    # Per-cell unique-RGB inset (ROI for selection platforms; whole tissue for MERFISH).
    inset_title = f"{cfg.platform} {cfg.sample} {'ROI' if roi else 'whole tissue'} — per-cell CIELAB transcripts"
    render_rgb_inset(roi_parquet, outdir / "qc_roi_inset_rgb_cells",
                     roi=roi, title_prefix=inset_title, seed=1)
    if cfg.select_roi:
        qc_bbox(cells, roi, outdir / "qc_selected_roi_bbox",
                f"{cfg.platform} {cfg.sample} — selected high-problem ROI (RCTD)")

    # --- summary ----------------------------------------------------------
    roi_ps = float(np.nanmean(roi_cells["RCTD_problem_score"].to_numpy()))
    roi_mw = float(np.nanmedian(roi_cells["RCTD_max_weight"].to_numpy()))
    summary = {
        "platform": cfg.platform,
        "key": cfg.key,
        "sample": cfg.sample,
        "reference_h5ad": cfg.reference_h5ad,
        "n_spatial_genes": gene_stats["n_spatial_genes"],
        "n_scrna_genes": gene_stats["n_reference_genes"],
        "n_overlap_genes": gene_stats["n_overlap_genes"],
        "n_genes_for_npmi": int(len(genes_universe)),
        "gene_selection": gene_stats.get("gene_selection"),
        "roi_selected": bool(cfg.select_roi),
        "roi_xmin": extract["x_min"], "roi_xmax": extract["x_max"],
        "roi_ymin": extract["y_min"], "roi_ymax": extract["y_max"],
        "n_cells": extract["n_cells"],
        "n_transcripts": extract["n_transcripts"],
        "frac_unassigned_transcripts": extract["frac_unassigned"],
        "mean_rctd_problem_score": roi_ps,
        "median_rctd_max_weight": roi_mw,
        "roi_selection_reason": (roi["reason"] if roi else
                                 "dataset is already an ROI with <1M transcripts; "
                                 "used in full without sub-selection"),
        "output_parquet": str(roi_parquet),
        "schema_required_ok": schema["required_columns_ok"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (outdir / "roi_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"  done {cfg.key}: {extract['n_cells']:,} cells / "
        f"{extract['n_transcripts']:,} transcripts; mean RCTD problem={roi_ps:.3f}")
    return summary


# ===========================================================================
# Aggregate summary
# ===========================================================================
def write_aggregate_summary():
    rows = []
    for sub in sorted(OUT_ROOT.glob("*/roi_summary.json")):
        rows.append(json.loads(sub.read_text()))
    if not rows:
        log("No per-platform summaries found; nothing to aggregate.")
        return
    cols = ["platform", "sample", "reference_h5ad", "n_spatial_genes",
            "n_scrna_genes", "n_overlap_genes", "n_genes_for_npmi",
            "roi_xmin", "roi_xmax", "roi_ymin", "roi_ymax", "n_cells",
            "n_transcripts", "frac_unassigned_transcripts",
            "mean_rctd_problem_score", "median_rctd_max_weight",
            "roi_selection_reason", "output_parquet"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(OUT_ROOT / "roi_selection_summary.tsv", sep="\t", index=False)

    lines = ["# Fig3 cross-platform ROI selection summary (RCTD problem score)", "",
             f"_Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}._", "",
             "Standardized ROI inputs for TRACER, cellAdmix, SPLIT, Baysor, proseg "
             "and Segger. ROIs are selected by **highest RCTD (Poisson-EM) problem "
             "score**. Benchmark algorithms are NOT run here.",
             ""]
    for r in rows:
        lines += [
            f"## {r['platform']} — {r['sample']}", "",
            f"- Reference: `{Path(r['reference_h5ad']).name}`",
            f"- Genes: spatial={r['n_spatial_genes']}, scRNA={r['n_scrna_genes']}, "
            f"overlap={r['n_overlap_genes']}, RCTD signature/NPMI={r['n_genes_for_npmi']} "
            f"({r.get('gene_selection')})",
            f"- ROI x: [{r['roi_xmin']:.1f}, {r['roi_xmax']:.1f}]  "
            f"y: [{r['roi_ymin']:.1f}, {r['roi_ymax']:.1f}]",
            f"- **Cells: {r['n_cells']:,}** | **Transcripts: {r['n_transcripts']:,}** "
            f"| unassigned: {r['frac_unassigned_transcripts']*100:.1f}%",
            f"- Mean RCTD problem score: {r.get('mean_rctd_problem_score', float('nan')):.3f} | "
            f"median RCTD max weight: {r.get('median_rctd_max_weight', float('nan')):.3f}",
            f"- Reason: {r['roi_selection_reason']}",
            f"- Parquet: `{r['output_parquet']}`", "",
        ]
    (OUT_ROOT / "roi_selection_summary.md").write_text("\n".join(lines))
    log(f"Wrote aggregate summary for {len(rows)} platforms → {OUT_ROOT}")


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    platforms = build_platforms()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(platforms),
                   help="Run only these platform keys.")
    p.add_argument("--summarize-only", action="store_true",
                   help="Only rebuild the aggregate summary from existing outputs.")
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        write_aggregate_summary()
        return 0

    keys = args.only or list(platforms)
    for k in keys:
        try:
            process_platform(platforms[k])
        except Exception as e:
            import traceback
            log(f"ERROR in {k}: {e}")
            traceback.print_exc()
    write_aggregate_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
