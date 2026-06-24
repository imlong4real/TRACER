#!/usr/bin/env python3
"""label_transfer_spatial.py

Reference-guided label transfer for TRACER spatial transcriptomics datasets.

Supports three reference types:
    * lung_gse127465          — Zilionis 2019 NSCLC scRNA-seq (mtx + tsv.gz)
    * pancreas_tosti_2020     — Tosti 2020 pancreas snRNA-seq (Conos/Pagoda2
                                template; Python fallback used here)
    * cervical_atera_plus_scrna — Atera labeled h5ad reference + optional scRNA h5

Method (Python fallback — used when R/Conos is not available)
    1. Build per-cell sparse count matrix from the spatial transcript parquet.
    2. Read the reference, collapse annotations using the per-reference
       harmonization map (e.g. Tosti notebook's Acinar collapse).
    3. Intersect genes between reference and query (gene-symbol level).
    4. Anchor selection ("clean_marker"): keep, per class, cells with
       (a) sufficient transcript depth, (b) detectable expression of the class's
       top marker genes, and (c) cap by --max_reference_cells_per_type.
    5. Log-normalize both, compute per-class centroids in the shared gene space.
    6. Score each query cell by cosine similarity to every class centroid;
       softmax → class probabilities.
    7. Predicted class = argmax; confidence = max softmax probability.
    8. Save transferred annotations, confidence, anchor list, shared genes,
       low-confidence cells, and a QC summary.

The script DOES NOT silently emit low-quality labels: a QC summary reports
shared-gene count, per-class anchor counts, confidence distribution, and the
fraction of cells below `--low_confidence_threshold`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io

import matplotlib
matplotlib.use("Agg")

import anndata as ad


warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[label_transfer] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Harmonization tables
# ---------------------------------------------------------------------------
LUNG_FINE_TO_COARSE: Dict[str, str] = {
    # Zilionis / GSE127465 -> harmonized labels (apply AFTER stripping b/t prefix)
    "Neutrophils":   "neutrophil",
    "Endothelial cells": "endothelial",
    "Fibroblasts":   "fibroblast",
    "Plasma cells":  "plasma",
    "mDC": "DC",
    "pDC": "DC",
    "DC":  "DC",
    "MoMacDC": "macrophage",        # Zilionis mixed monocyte/macrophage/DC compartment
    "Monocytes": "monocyte",
    "NK cells":  "NK",
    "B cells":   "B cells",
    "Mast cells": "mast",
    "RBC": "RBC",
    "Platelets": "Platelets",
    "Basophils": "basophil",
    "Myeloid precursor-like": "myeloid",
    "Smooth muscle cells": "smooth muscle",
    # Lung epithelium (kept as their own labels — useful for normal lung,
    # malignant cells get caught by Patient*-specific rule below)
    "Type I cells":  "epithelial",
    "Type II cells": "epithelial",
    "Club cells":    "epithelial",
    "Ciliated cells": "epithelial",
    "ND": "unknown",                 # 'not determined' — drop downstream
}

LUNG_T_PREFIXES = ("T cell", "T cells", "T_", "T-")
LUNG_TUMOR_PREFIXES = ("Patient_", "Patient ")


def harmonize_lung(label: str) -> Tuple[str, str]:
    """Return (fine, coarse) labels.

    fine  -> keeps subset distinctions where possible (e.g. pDC, mDC, T subsets)
    coarse -> collapses to broad lineage (neutrophil/endothelial/fibroblast/...)

    Handles GSE127465 Zilionis 2019 labels:
      - "Major cell type": bB cells, bMonocytes, bNK cells, bNeutrophils,
        bPlasma cells, bPlatelets, bRBC, bT cells, bpDC, plus 't' (tumor) /
        'n' (normal) prefixes from the tumor cohort.
      - "Minor subset": bN1..bN6, bMono1..3, bMonoDC, bCD4T1, bCD8T1, bpDC ...
    Strip the 'b'/'t'/'n' tissue prefix before mapping.
    """
    if label is None:
        return ("unknown", "unknown")
    s = str(label).strip()
    if not s or s.lower() in ("nan", "na", "none"):
        return ("unknown", "unknown")

    # Strip GSE127465 tissue prefix ('b' = blood, 't' = tumor, 'n' = normal lung)
    # but only when the next char is an uppercase letter / known token: this keeps
    # already-cleaned labels (e.g. "Plasma cells") intact.
    stripped = s
    if len(s) >= 2 and s[0] in ("b", "t", "n") and s[1].isupper():
        stripped = s[1:]
    elif s in ("bpDC", "tpDC"):
        stripped = s[1:]

    # Catch Patient-specific malignant labels in all common formats
    # ("Patient_1", "Patient 1", "Patient1-specific", "Patient_1-specific", ...)
    s_low = s.replace(" ", "").lower()
    if s_low.startswith("patient"):
        return ("malignant", "malignant")

    # Direct mapping on the cleaned label
    key = stripped
    if key in LUNG_FINE_TO_COARSE:
        coarse = LUNG_FINE_TO_COARSE[key]
        # Keep DC subtype in fine
        if key in ("pDC", "mDC"):
            fine = key
        elif s in ("bNeutrophils", "tNeutrophils"):
            fine = s  # preserve b/t neutrophil distinction in fine
        else:
            fine = coarse
        return (fine, coarse)
    if s.startswith(LUNG_TUMOR_PREFIXES) or stripped.startswith(LUNG_TUMOR_PREFIXES):
        return ("malignant", "malignant")
    if (stripped.startswith(LUNG_T_PREFIXES) or stripped.startswith("CD4")
            or stripped.startswith("CD8") or stripped.startswith("Treg")):
        return (stripped, "T cells")
    if key in ("Platelets", "bPlatelets"):
        return ("Platelets", "Platelets")
    # Minor-subset short codes (bN1..N6, bMono1..3, bCD4T1, bCD8T1, bMonoDC, …)
    # — fall back to a sensible coarse based on prefix.
    if stripped.startswith("N") and stripped[1:].isdigit():
        return (s, "neutrophil")
    if stripped.startswith("Mono"):
        return (s, "monocyte")
    if stripped.startswith("MonoDC"):
        return (s, "DC")
    return (stripped, stripped)


# Tosti 2020 collapse (matches the notebook's `cell_annot_adj`)
PANCREAS_TOSTI_COLLAPSE: Dict[str, str] = {
    # filled by prefix logic below for Acinar*
    "Alpha": "Alpha/Beta/Delta/Gamma",
    "Beta":  "Alpha/Beta/Delta/Gamma",
    "Delta": "Alpha/Beta/Delta/Gamma",
    "Gamma": "Alpha/Beta/Delta/Gamma",
    "Activated Stellate": "Stellate",
    "Quiescent Stellate": "Stellate",
    "Schwann": "Stellate",
    "Ductal": "Ductal",
    "MUC5B+ Ductal": "Ductal",
}


def harmonize_pancreas_tosti(label: str) -> Tuple[str, str]:
    if label is None:
        return ("unknown", "unknown")
    s = str(label).strip()
    if not s or s.lower() in ("nan", "na", "none"):
        return ("unknown", "unknown")
    if s.startswith("Acinar"):
        return ("Acinar", "Acinar")
    if s in PANCREAS_TOSTI_COLLAPSE:
        coarse = PANCREAS_TOSTI_COLLAPSE[s]
        return (coarse, coarse)
    return (s, s)


def harmonize_passthrough(label: str) -> Tuple[str, str]:
    if label is None:
        return ("unknown", "unknown")
    s = str(label).strip()
    if not s:
        return ("unknown", "unknown")
    return (s, s)


HARMONIZERS = {
    "lung_nsclc": harmonize_lung,
    "lung_gse127465": harmonize_lung,
    "pancreas_tosti_2020": harmonize_pancreas_tosti,
    "passthrough": harmonize_passthrough,
    "cervical_atera": harmonize_passthrough,
}


# ---------------------------------------------------------------------------
# Query-side: build cell × gene from transcript parquet
# ---------------------------------------------------------------------------
def build_query_from_parquet(parquet_path: str, *, min_transcripts: int,
                             max_transcripts: Optional[int],
                             use_duckdb: bool = True,
                             intermediate_dir: Optional[Path] = None,
                             ) -> ad.AnnData:
    """Build a cell-by-gene AnnData from a TRACER transcript parquet."""
    log(f"building query cell-by-gene from {parquet_path}")
    if use_duckdb:
        import duckdb
        intermediate_dir = Path(intermediate_dir or "/tmp/label_transfer_interm")
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        interm = intermediate_dir / "_query_counts.parquet"
        con = duckdb.connect()
        # Detect schema
        esc = parquet_path.replace("'", "''")
        desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{esc}') LIMIT 0").fetchall()
        cols = [r[0] for r in desc]
        if "cell_id" not in cols or "feature_name" not in cols:
            raise RuntimeError(f"query parquet missing cell_id/feature_name; cols={cols}")
        has_is_gene = "is_gene" in cols
        extra = " AND is_gene = TRUE" if has_is_gene else ""
        uvals = "'UNASSIGNED','Unassigned','unassigned','0','','-1','NA'"
        log(f"  DuckDB aggregating counts to {interm}")
        t0 = time.time()
        con.execute(f"""
            COPY (
              SELECT CAST(cell_id AS VARCHAR) AS cell_id, feature_name, COUNT(*) AS count
              FROM read_parquet('{esc}')
              WHERE cell_id IS NOT NULL
                AND CAST(cell_id AS VARCHAR) NOT IN ({uvals})
                AND feature_name IS NOT NULL
                AND length(CAST(feature_name AS VARCHAR)) > 0
                {extra}
              GROUP BY cell_id, feature_name
            )
            TO '{interm}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        log(f"  aggregation done in {time.time()-t0:.1f}s")
        cg = pd.read_parquet(interm)
    else:
        import pyarrow.parquet as pq
        tbl = pq.read_table(parquet_path, columns=["cell_id", "feature_name"])
        df = tbl.to_pandas()
        cell = df["cell_id"].astype(str)
        keep = cell.notna() & ~cell.isin({"UNASSIGNED", "Unassigned", "unassigned",
                                          "0", "", "-1", "NA"})
        df = df[keep]
        cg = df.groupby(["cell_id", "feature_name"], observed=True).size().reset_index(name="count")

    cell_cat = cg["cell_id"].astype("category")
    gene_cat = cg["feature_name"].astype("category")
    rows = cell_cat.cat.codes.to_numpy()
    cols = gene_cat.cat.codes.to_numpy()
    data = cg["count"].to_numpy(dtype=np.int32)
    n_cells = len(cell_cat.cat.categories)
    n_genes = len(gene_cat.cat.categories)
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes))

    obs = pd.DataFrame(index=pd.Index(cell_cat.cat.categories.astype(str), name="cell_id"))
    var = pd.DataFrame(index=pd.Index(gene_cat.cat.categories.astype(str), name="feature_name"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names_make_unique()

    # QC + filter
    total = np.asarray(adata.X.sum(axis=1)).ravel()
    nz = (adata.X > 0).astype(np.int32)
    n_g = np.asarray(nz.sum(axis=1)).ravel()
    adata.obs["total_counts"] = total
    adata.obs["n_genes_by_counts"] = n_g
    keep = total >= min_transcripts
    if max_transcripts is not None:
        keep &= total <= max_transcripts
    log(f"  cells before filter: {adata.n_obs:,}  after: {int(keep.sum()):,}  "
        f"[{min_transcripts}, {max_transcripts}]")
    adata = adata[keep].copy()
    return adata


# ---------------------------------------------------------------------------
# Reference loaders
# ---------------------------------------------------------------------------
def load_reference_lung_gse127465(*, scrna_gene_names: str, scrna_metadata: str,
                                  scrna_counts: str) -> ad.AnnData:
    """Read the Zilionis 2019 NSCLC scRNA reference (mtx + tsv.gz)."""
    log("loading lung GSE127465 reference...")
    # Genes
    with gzip.open(scrna_gene_names, "rt") as f:
        # File can be 1 column (just gene name) or 2 columns
        first = f.readline().strip().split("\t")
        f.seek(0)
        names_df = pd.read_csv(f, sep="\t", header=None)
    gene_names = names_df.iloc[:, 0].astype(str).str.strip().tolist()
    log(f"  genes: {len(gene_names):,}")

    # Metadata
    meta = pd.read_csv(scrna_metadata, sep="\t", compression="infer")
    log(f"  metadata: {meta.shape}; columns sample: {list(meta.columns)[:10]}")

    # Counts (mtx, genes × cells per common scRNA convention — may need transpose)
    log(f"  reading counts mtx: {scrna_counts}")
    with gzip.open(scrna_counts, "rt") as f:
        m = scipy.io.mmread(f)
    m = m.tocsr()
    log(f"  mtx shape: {m.shape}")
    # The filename declares 54773 cells × 41861 genes → cells × genes layout.
    if m.shape[0] == len(gene_names) and m.shape[1] == len(meta):
        m = m.T.tocsr()
    elif m.shape[0] != len(meta):
        # Try transpose if it matches the other way
        if m.shape[1] == len(meta):
            m = m.T.tocsr()
    n_cells, n_genes = m.shape
    if n_genes != len(gene_names) or n_cells != len(meta):
        raise RuntimeError(f"mtx shape {m.shape} doesn't match metadata ({len(meta)}) / gene names ({len(gene_names)})")
    var = pd.DataFrame(index=pd.Index(gene_names, name="feature_name"))
    # var index unique
    var.index = pd.Index(pd.Series(var.index).astype(str), name="feature_name")
    if var.index.duplicated().any():
        var.index = pd.Index(
            pd.Series(var.index).astype(str) + "__" + var.index.to_series().groupby(var.index).cumcount().astype(str),
            name="feature_name")
    obs = meta.copy()
    obs.index = pd.Index(obs.iloc[:, 0].astype(str), name="cell_id")
    # Build AnnData. The mtx is normalized (per filename) — we'll log1p but skip
    # the row-normalize step to avoid double normalization.
    adata = ad.AnnData(X=m.astype(np.float32), obs=obs, var=var)
    # Cluster column for label transfer — auto-detect.
    # GSE127465 metadata uses "Major cell type" (with spaces), "Minor subset".
    # We must NOT fall through to a generic "first object col with <100 uniques"
    # rule, because that picks the "Patient" column (9 values: p1..p7 etc).
    candidates = [
        "Major cell type", "Minor subset", "Most likely LM22 cell type",
        "Major_cell_type", "Minor_cell_type", "Sub_cell_type", "Major-cell_type",
        "Cluster", "cell_type", "annotation",
    ]
    cluster_col = None
    for c in candidates:
        if c in obs.columns:
            cluster_col = c
            break
    if cluster_col is None:
        raise RuntimeError(
            f"lung reference: no recognized cluster column found. "
            f"Tried {candidates}. Available: {list(obs.columns)}")
    adata.obs["original_cell_type"] = obs[cluster_col].astype(str).values
    log(f"  using cluster column: {cluster_col!r}  "
        f"({adata.obs['original_cell_type'].nunique()} unique labels)")
    return adata


def load_reference_tosti(*, reference_expr_matrix: str, reference_meta: str,
                         max_cells_per_class: Optional[int] = 5000,
                         random_seed: int = 0,
                         shared_genes: Optional[set] = None,
                         ) -> ad.AnnData:
    """Read Tosti 2020 pancreas snRNA-seq (exprMatrix.tsv.gz + meta.tsv).

    The full matrix is ~30k genes × 112k cells which is too large to load
    densely. We:
      1. Pre-subsample cells per class to `max_cells_per_class` from `meta.tsv`
         (anchor selection happens later but this caps before reading the heavy
         expression file).
      2. Stream-read the gzipped exprMatrix line-by-line (one gene per line),
         keeping only the selected columns and building a sparse COO matrix.
      3. If `shared_genes` is provided, drop genes that aren't in the spatial
         panel — usually shrinks rows from ~30k to ~300.
    """
    log("loading Tosti 2020 pancreas reference (streaming)...")
    rng = np.random.default_rng(random_seed)
    meta = pd.read_csv(reference_meta, sep="\t")
    meta.columns = [c.strip() for c in meta.columns]
    if "Cell" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "Cell"})
    if "Cluster" not in meta.columns:
        raise RuntimeError(f"Tosti meta missing Cluster column. cols={list(meta.columns)}")
    log(f"  meta: {meta.shape}; clusters: {meta['Cluster'].nunique()}")

    # Subsample meta per class
    if max_cells_per_class is not None and max_cells_per_class > 0:
        sampled = []
        for cluster, group in meta.groupby("Cluster", observed=True):
            if len(group) > max_cells_per_class:
                idx = rng.choice(len(group), size=max_cells_per_class, replace=False)
                sampled.append(group.iloc[idx])
            else:
                sampled.append(group)
        meta = pd.concat(sampled).reset_index(drop=True)
        log(f"  subsampled meta: {meta.shape}  (≤{max_cells_per_class} per class)")
    selected = set(meta["Cell"].astype(str).tolist())

    # Stream exprMatrix line by line; build COO (gene_i, cell_i_in_subset, val)
    log(f"  streaming exprMatrix: {reference_expr_matrix}")
    t0 = time.time()
    # Determine compression and open accordingly
    if reference_expr_matrix.endswith(".gz"):
        f = gzip.open(reference_expr_matrix, "rt", encoding="utf-8")
    else:
        f = open(reference_expr_matrix, "rt", encoding="utf-8")
    try:
        header = f.readline().rstrip("\n").split("\t")
        # First col is gene id; remaining are cell ids
        cell_cols_all = header[1:]
        # Build indices for selected columns (in matrix-column order)
        sel_mask = np.array([c in selected for c in cell_cols_all], dtype=bool)
        sel_idx = np.where(sel_mask)[0]
        sel_cells = [cell_cols_all[i] for i in sel_idx]
        if not sel_cells:
            raise RuntimeError("None of the meta cells matched exprMatrix columns")
        log(f"  selected cells: {len(sel_cells):,} / {len(cell_cols_all):,}")

        gene_ids: List[str] = []
        rows, cols, vals = [], [], []
        n_genes_kept = 0
        n_lines = 0
        for line in f:
            n_lines += 1
            if not line:
                continue
            parts = line.rstrip("\n").split("\t")
            gene = parts[0].split("|", 1)[0]
            if shared_genes is not None and gene not in shared_genes:
                continue
            # Parse only selected columns
            data = parts[1:]
            if len(data) != len(cell_cols_all):
                continue
            for j, ci in enumerate(sel_idx):
                v = data[ci]
                if v and v != "0" and v != "0.0":
                    try:
                        fv = float(v)
                    except ValueError:
                        continue
                    if fv != 0.0:
                        rows.append(n_genes_kept)
                        cols.append(j)
                        vals.append(fv)
            gene_ids.append(gene)
            n_genes_kept += 1
            if n_lines % 2000 == 0:
                log(f"    scanned {n_lines:,} gene rows  (kept {n_genes_kept:,})  "
                    f"({time.time()-t0:.1f}s)")
    finally:
        f.close()
    log(f"  streamed {n_lines:,} gene rows, kept {n_genes_kept:,}  "
        f"({time.time()-t0:.1f}s)")
    if n_genes_kept == 0:
        raise RuntimeError("No genes kept from Tosti reference (shared_genes filter too strict?)")

    n_cells = len(sel_cells)
    mat = sp.coo_matrix(
        (np.asarray(vals, dtype=np.float32),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n_genes_kept, n_cells),
    ).tocsr()
    # genes × cells → cells × genes
    mat = mat.T.tocsr()
    log(f"  matrix (cells × genes): {mat.shape}  nnz={mat.nnz:,}")

    var = pd.DataFrame(index=pd.Index(gene_ids, name="feature_name"))
    var.index = pd.Index(pd.Series(var.index).astype(str), name="feature_name")
    if var.index.duplicated().any():
        var.index = pd.Index(
            pd.Series(var.index).astype(str) + "__" +
            var.index.to_series().groupby(var.index).cumcount().astype(str),
            name="feature_name")
    obs = meta.set_index("Cell")
    obs.index = obs.index.astype(str)
    obs.index.name = "cell_id"
    obs = obs.reindex(sel_cells)
    adata = ad.AnnData(X=mat, obs=obs, var=var)
    adata.obs["original_cell_type"] = obs["Cluster"].astype(str).values
    log(f"  ref cells: {adata.n_obs:,}  genes: {adata.n_vars:,}  "
        f"unique clusters: {adata.obs['original_cell_type'].nunique()}")
    return adata


def load_reference_cervical(*, reference_h5ad: Optional[str],
                            scrna_10x_h5: Optional[str]) -> ad.AnnData:
    """Load Atera h5ad reference; optionally augment with 10x scRNA h5.

    Prefers `adata.layers['counts']` over `adata.X` when present, because the
    pre-built Atera h5ad has scaled (mean-centered, negative) values in `.X`.
    Centroid+cosine label transfer requires raw or library-normalized counts.
    """
    log("loading cervical references...")
    refs = []
    if reference_h5ad:
        log(f"  reading h5ad: {reference_h5ad}")
        a = ad.read_h5ad(reference_h5ad)
        if "cell_type" not in a.obs.columns:
            raise RuntimeError(f"reference h5ad missing obs['cell_type']; obs cols={list(a.obs.columns)}")
        if "counts" in a.layers:
            log("    using raw counts layer (X is scaled / log-normalized)")
            a = ad.AnnData(X=a.layers["counts"], obs=a.obs.copy(), var=a.var.copy())
        else:
            # Detect mean-centered/scaled X — negative values indicate scaling.
            try:
                Xs = a.X
                sample = Xs[:1000].toarray() if sp.issparse(Xs) else np.asarray(Xs[:1000])
                if float(sample.min()) < 0:
                    log("    WARN: reference X has negative values (looks scaled). "
                        "Label transfer quality may degrade — provide a counts layer.")
            except Exception:
                pass
        a.obs["original_cell_type"] = a.obs["cell_type"].astype(str).values
        a.obs["__source"] = "atera_wta"
        log(f"    cells: {a.n_obs:,}  genes: {a.n_vars:,}  "
            f"types: {a.obs['original_cell_type'].nunique()}")
        refs.append(a)
    if scrna_10x_h5:
        log(f"  reading 10x h5: {scrna_10x_h5}")
        try:
            import scanpy as sc
            a = sc.read_10x_h5(scrna_10x_h5)
            a.var_names_make_unique()
            # No cell type labels in the raw 10x matrix; skip for label_transfer
            log(f"    10x h5 has {a.n_obs:,} cells × {a.n_vars:,} genes "
                f"(no labels — used for gene-set extension only)")
        except Exception as e:
            log(f"    WARN: could not read 10x h5: {e}")
    if not refs:
        raise RuntimeError("cervical reference loader: no labeled reference provided")
    if len(refs) == 1:
        return refs[0]
    return ad.concat(refs, axis=0, join="outer", merge="same")


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------
def select_anchors(ref: ad.AnnData, *, label_col: str = "transferred_label_input",
                   strategy: str = "clean_marker",
                   max_per_type: int = 5000,
                   min_per_type: int = 50,
                   random_seed: int = 0) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Pick a clean, balanced anchor set from the reference.

    For each label class:
      - drop cells with very low total counts (< 5th percentile within class)
      - rank by a "marker score" = sum of log1p(top per-class genes)
      - cap to max_per_type (random subsample within high-quality pool)
      - require min_per_type or warn
    """
    rng = np.random.default_rng(random_seed)
    labels = pd.Series(ref.obs[label_col].astype(str).values, index=ref.obs_names)
    cats = sorted(labels.dropna().unique())
    log(f"  anchor selection: {len(cats)} classes; strategy={strategy}")

    keep = np.zeros(ref.n_obs, dtype=bool)

    # total counts (use raw .X)
    total_counts = np.asarray(ref.X.sum(axis=1)).ravel()
    if strategy in ("clean_marker", "marker") and ref.n_vars > 10:
        # Marker score uses a quick per-class top-gene mean: rank cells by
        # how strongly they hit their own class's top genes.
        # Use log1p(X) for stability.
        import scipy.sparse as sps
        X = ref.X
        if not sps.issparse(X):
            X = sps.csr_matrix(X)
        Xlog = X.copy().astype(np.float32)
        Xlog.data = np.log1p(Xlog.data)
        # mean per class
        cat_index = pd.Categorical(labels, categories=cats)
        codes = cat_index.codes
        n_classes = len(cats)
        per_class_means = np.zeros((n_classes, ref.n_vars), dtype=np.float32)
        per_class_counts = np.zeros(n_classes, dtype=np.int64)
        for ci, cat in enumerate(cats):
            idx = np.where(codes == ci)[0]
            if len(idx) == 0:
                continue
            per_class_means[ci] = np.asarray(Xlog[idx].mean(axis=0)).ravel()
            per_class_counts[ci] = len(idx)
        # Find top 25 genes per class (by mean over class - mean over others)
        global_mean = Xlog.mean(axis=0)
        global_mean = np.asarray(global_mean).ravel()
        for ci, cat in enumerate(cats):
            n_ci = int(per_class_counts[ci])
            if n_ci == 0:
                continue
            # specificity score
            score = per_class_means[ci] - global_mean
            n_top = min(25, ref.n_vars)
            top_genes = np.argpartition(score, -n_top)[-n_top:]
            # cells of this class
            idx = np.where(codes == ci)[0]
            marker_score = np.asarray(Xlog[idx][:, top_genes].sum(axis=1)).ravel()
            depth = total_counts[idx]
            if n_ci > min_per_type:
                # keep cells with depth above 10th percentile and marker_score
                # above median
                depth_thr = np.percentile(depth, 10)
                marker_thr = np.percentile(marker_score, 50)
                ok = (depth >= depth_thr) & (marker_score >= marker_thr)
                if ok.sum() < min_per_type:
                    ok = depth >= depth_thr  # relax
                idx_ok = idx[ok]
            else:
                idx_ok = idx
            if len(idx_ok) > max_per_type:
                idx_ok = rng.choice(idx_ok, size=max_per_type, replace=False)
            keep[idx_ok] = True
    else:
        # naive cap per class
        for cat in cats:
            idx = np.where(labels.values == cat)[0]
            if len(idx) > max_per_type:
                idx = rng.choice(idx, size=max_per_type, replace=False)
            keep[idx] = True

    anchor_summary = pd.DataFrame({
        "cell_type": cats,
        "n_in_reference": [int((labels.values == c).sum()) for c in cats],
        "n_anchor": [int(((labels.values == c) & keep).sum()) for c in cats],
    })
    log(f"  anchors kept: {int(keep.sum()):,} / {ref.n_obs:,}")
    sub = ref[keep].copy()
    return sub, anchor_summary


# ---------------------------------------------------------------------------
# Label transfer (Python KNN over per-class centroids + neighbor smoothing)
# ---------------------------------------------------------------------------
def _row_normalize_log(X: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    """Normalize total counts per cell, then log1p. Returns dense float32."""
    X = X.copy().astype(np.float32)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv = sp.diags(target_sum / row_sums)
    X = inv @ X
    X.data = np.log1p(X.data)
    return X


def _l2_normalize(X: np.ndarray, axis: int = 1) -> np.ndarray:
    n = np.linalg.norm(X, axis=axis, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def transfer_labels_python(query: ad.AnnData, ref: ad.AnnData,
                           *, label_col: str,
                           temperature: float = 0.05,
                           softmax_low_confidence: float = 0.4
                           ) -> Tuple[pd.DataFrame, dict]:
    """Cosine-centroid label transfer in shared-gene space.

    Returns (annotation_df, qc_dict). annotation_df is per-query-cell:
        cell_id, transferred_label, transfer_confidence, top2_label, top2_confidence
    """
    log("transferring labels (Python centroid + cosine)")
    shared = sorted(set(query.var_names) & set(ref.var_names))
    log(f"  shared genes: {len(shared):,}  (query={query.n_vars}, ref={ref.n_vars})")
    if len(shared) < 5:
        raise RuntimeError(f"too few shared genes ({len(shared)}); aborting.")

    q = query[:, shared].copy()
    r = ref[:, shared].copy()

    # Normalize + log1p
    Xq = _row_normalize_log(q.X if sp.issparse(q.X) else sp.csr_matrix(q.X))
    Xr = _row_normalize_log(r.X if sp.issparse(r.X) else sp.csr_matrix(r.X))

    # Per-class centroid in normalized log-space
    labels = pd.Categorical(r.obs[label_col].astype(str).values)
    cats = list(labels.categories)
    n_classes = len(cats)
    log(f"  computing {n_classes} class centroids in {Xr.shape[1]} genes")
    centroids = np.zeros((n_classes, Xr.shape[1]), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int64)
    for ci, cat in enumerate(cats):
        idx = np.where(labels.codes == ci)[0]
        if len(idx) == 0:
            continue
        centroids[ci] = np.asarray(Xr[idx].mean(axis=0)).ravel()
        counts[ci] = len(idx)

    # Cosine similarity = normalized dot product
    centroids_n = _l2_normalize(centroids, axis=1)
    # Densify query in chunks
    chunk = 25000
    n_q = Xq.shape[0]
    sims = np.empty((n_q, n_classes), dtype=np.float32)
    for start in range(0, n_q, chunk):
        stop = min(n_q, start + chunk)
        block = Xq[start:stop].toarray()
        block = _l2_normalize(block, axis=1)
        sims[start:stop] = block @ centroids_n.T

    # Softmax over classes with temperature
    # Numerically stable
    z = sims / max(temperature, 1e-6)
    z -= z.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)

    pred_idx = probs.argmax(axis=1)
    conf = probs[np.arange(n_q), pred_idx]
    # top-2
    p2 = probs.copy()
    p2[np.arange(n_q), pred_idx] = -1
    second_idx = p2.argmax(axis=1)
    second_conf = probs[np.arange(n_q), second_idx]

    annot = pd.DataFrame({
        "cell_id": q.obs_names.astype(str).values,
        "transferred_label": [cats[i] for i in pred_idx],
        "transfer_confidence": conf.astype(np.float64),
        "second_label": [cats[i] for i in second_idx],
        "second_confidence": second_conf.astype(np.float64),
        "shared_genes": int(len(shared)),
    })

    # QC summary
    qc = {
        "n_query_cells": int(n_q),
        "n_reference_cells": int(Xr.shape[0]),
        "n_shared_genes": int(len(shared)),
        "n_classes": int(n_classes),
        "low_confidence_threshold": float(softmax_low_confidence),
        "low_confidence_count": int((conf < softmax_low_confidence).sum()),
        "low_confidence_fraction": float((conf < softmax_low_confidence).mean()),
        "mean_confidence": float(conf.mean()),
        "median_confidence": float(np.median(conf)),
    }
    # per-class top1 counts
    pred_series = pd.Series(annot["transferred_label"])
    qc["per_class_pred_counts"] = pred_series.value_counts().to_dict()
    log(f"  pred counts (top1): {qc['per_class_pred_counts']}")
    log(f"  mean confidence: {qc['mean_confidence']:.3f}, "
        f"low-conf fraction (<{softmax_low_confidence}): {qc['low_confidence_fraction']:.1%}")
    return annot, qc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--query_transcript_parquet", default=None,
                   help="Transcript parquet (Xenium/CosMx/MERFISH).")
    p.add_argument("--query_h5ad", default=None,
                   help="Pre-built cell-by-gene AnnData (already QC'd). Used "
                        "in place of --query_transcript_parquet for scRNA / "
                        "h5ad-format queries.")
    p.add_argument("--query_10x_h5", default=None,
                   help="Cellranger filtered_feature_bc_matrix.h5 query.")
    p.add_argument("--reference_type", required=True,
                   choices=["lung_gse127465", "pancreas_tosti_2020",
                            "cervical_atera_plus_scrna"])
    # Lung GSE127465 inputs
    p.add_argument("--scrna_gene_names", default=None)
    p.add_argument("--scrna_metadata",  default=None)
    p.add_argument("--scrna_counts",    default=None)
    # Pancreas Tosti inputs
    p.add_argument("--reference_rds", default=None,
                   help="(unused by Python fallback — kept for CLI parity)")
    p.add_argument("--reference_expr_matrix", default=None)
    p.add_argument("--reference_meta",         default=None)
    p.add_argument("--reference_umap",         default=None,
                   help="(unused by Python fallback)")
    p.add_argument("--template_notebook",      default=None,
                   help="(unused by Python fallback — kept for CLI parity)")
    # Cervical inputs
    p.add_argument("--reference_h5ad", default=None)
    p.add_argument("--scrna_10x_h5",   default=None)

    p.add_argument("--outdir", required=True)
    p.add_argument("--sample_prefix", required=True)
    p.add_argument("--min_transcripts", type=int, default=10)
    p.add_argument("--max_transcripts", type=int, default=900)
    p.add_argument("--label_harmonization", default=None,
                   help="Override harmonization key (default chosen from "
                        "reference_type)")
    p.add_argument("--anchor_selection", choices=["clean_marker", "naive"],
                   default="clean_marker")
    p.add_argument("--max_reference_cells_per_type", type=int, default=5000)
    p.add_argument("--min_reference_cells_per_type", type=int, default=50)
    p.add_argument("--low_confidence_threshold", type=float, default=0.4)
    p.add_argument("--softmax_temperature", type=float, default=0.05)
    p.add_argument("--random_seed", type=int, default=0)
    return p.parse_args()


def _pick_harmonizer(name: Optional[str], ref_type: str):
    if name and name in HARMONIZERS:
        return HARMONIZERS[name]
    if ref_type in HARMONIZERS:
        return HARMONIZERS[ref_type]
    return harmonize_passthrough


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Query ----------
    interm = outdir / "_intermediate"
    n_query_sources = sum(bool(x) for x in
                          (args.query_transcript_parquet,
                           args.query_h5ad, args.query_10x_h5))
    if n_query_sources != 1:
        raise SystemExit("Provide exactly one of --query_transcript_parquet, "
                         "--query_h5ad, --query_10x_h5")
    if args.query_transcript_parquet:
        query = build_query_from_parquet(
            args.query_transcript_parquet,
            min_transcripts=args.min_transcripts,
            max_transcripts=args.max_transcripts,
            intermediate_dir=interm,
        )
    elif args.query_h5ad:
        log(f"loading query h5ad: {args.query_h5ad}")
        query = ad.read_h5ad(args.query_h5ad)
        if "counts" in query.layers:
            query = ad.AnnData(X=query.layers["counts"],
                                obs=query.obs.copy(), var=query.var.copy())
        query.var_names_make_unique()
        query.obs_names_make_unique()
        total = np.asarray(query.X.sum(axis=1)).ravel()
        nz = (query.X > 0).astype(np.int32)
        query.obs["total_counts"] = total
        query.obs["n_genes_by_counts"] = np.asarray(nz.sum(axis=1)).ravel()
        keep = (total >= args.min_transcripts) & (total <= args.max_transcripts)
        query = query[keep].copy()
        log(f"  query h5ad after [{args.min_transcripts}, {args.max_transcripts}] "
            f"filter: {query.n_obs:,} cells × {query.n_vars:,} genes")
    else:
        import scanpy as sc
        log(f"loading query 10x h5: {args.query_10x_h5}")
        query = sc.read_10x_h5(args.query_10x_h5)
        query.var_names_make_unique()
        total = np.asarray(query.X.sum(axis=1)).ravel()
        nz = (query.X > 0).astype(np.int32)
        query.obs["total_counts"] = total
        query.obs["n_genes_by_counts"] = np.asarray(nz.sum(axis=1)).ravel()
        keep = (total >= args.min_transcripts) & (total <= args.max_transcripts)
        query = query[keep].copy()
        log(f"  query 10x h5 after [{args.min_transcripts}, {args.max_transcripts}] "
            f"filter: {query.n_obs:,} cells × {query.n_vars:,} genes")
    log(f"query AnnData: {query.shape}")

    # ---------- 2) Reference ----------
    if args.reference_type == "lung_gse127465":
        if not (args.scrna_gene_names and args.scrna_metadata and args.scrna_counts):
            raise SystemExit("lung_gse127465 requires --scrna_gene_names, --scrna_metadata, --scrna_counts")
        ref = load_reference_lung_gse127465(
            scrna_gene_names=args.scrna_gene_names,
            scrna_metadata=args.scrna_metadata,
            scrna_counts=args.scrna_counts,
        )
    elif args.reference_type == "pancreas_tosti_2020":
        if not (args.reference_expr_matrix and args.reference_meta):
            raise SystemExit("pancreas_tosti_2020 requires --reference_expr_matrix and --reference_meta")
        # Pre-filter genes to spatial panel — shrinks Tosti rows from ~30k to ~300.
        shared = set(query.var_names.astype(str).tolist())
        ref = load_reference_tosti(
            reference_expr_matrix=args.reference_expr_matrix,
            reference_meta=args.reference_meta,
            max_cells_per_class=args.max_reference_cells_per_type,
            random_seed=args.random_seed,
            shared_genes=shared,
        )
    elif args.reference_type == "cervical_atera_plus_scrna":
        ref = load_reference_cervical(
            reference_h5ad=args.reference_h5ad,
            scrna_10x_h5=args.scrna_10x_h5,
        )
    else:
        raise SystemExit(f"unknown reference type: {args.reference_type}")

    # ---------- 3) Harmonization ----------
    harm_name = args.label_harmonization or args.reference_type
    harmonizer = _pick_harmonizer(harm_name, args.reference_type)
    log(f"harmonization: {harm_name}")
    raw = ref.obs["original_cell_type"].astype(str).values
    fine, coarse = zip(*[harmonizer(x) for x in raw])
    ref.obs["cell_type"] = list(fine)
    ref.obs["cell_type_coarse"] = list(coarse)
    # Drop "unknown" labels from reference — they would corrupt transfer
    keep_known = ref.obs["cell_type"].values != "unknown"
    if int(keep_known.sum()) < ref.n_obs:
        log(f"  dropping {int((~keep_known).sum())} reference cells with 'unknown' label")
        ref = ref[keep_known].copy()
    log(f"reference after harmonize: {ref.n_obs:,} cells, "
        f"{ref.obs['cell_type'].nunique()} fine classes, "
        f"{ref.obs['cell_type_coarse'].nunique()} coarse")

    # Use the coarse label for the actual label-transfer (more robust given
    # small gene panels). The fine label is preserved per cell for reporting
    # if we ever want to translate back.
    ref.obs["transferred_label_input"] = ref.obs["cell_type_coarse"].astype(str).values

    # ---------- 4) Anchor selection ----------
    ref_anchor, anchor_summary = select_anchors(
        ref, label_col="transferred_label_input",
        strategy=args.anchor_selection,
        max_per_type=args.max_reference_cells_per_type,
        min_per_type=args.min_reference_cells_per_type,
        random_seed=args.random_seed,
    )
    anchor_path = outdir / f"{args.sample_prefix}_reference_anchor_cells.csv"
    out = anchor_summary.copy()
    out.to_csv(anchor_path, index=False)
    # Also write the list of actual anchor cell IDs
    anchor_ids = pd.DataFrame({
        "cell_id": ref_anchor.obs_names.astype(str).values,
        "cell_type": ref_anchor.obs["transferred_label_input"].astype(str).values,
    })
    anchor_ids.to_csv(outdir / f"{args.sample_prefix}_reference_anchor_cell_ids.csv",
                      index=False)

    # ---------- 5) Transfer ----------
    annot, qc = transfer_labels_python(
        query, ref_anchor,
        label_col="transferred_label_input",
        temperature=args.softmax_temperature,
        softmax_low_confidence=args.low_confidence_threshold,
    )
    # Re-attach query QC + (placeholder) original_cell_type/x/y/z if available
    annot = annot.set_index("cell_id")
    annot["n_transcripts"] = query.obs["total_counts"].reindex(annot.index).values
    annot["n_genes_by_counts"] = query.obs["n_genes_by_counts"].reindex(annot.index).values
    annot["original_cell_type"] = np.nan
    annot["cell_type"] = annot["transferred_label"]
    annot["cell_type_coarse"] = annot["transferred_label"]
    annot = annot.reset_index()
    annot_path = outdir / f"{args.sample_prefix}_transferred_cell_annotations.csv"
    annot.to_csv(annot_path, index=False)
    log(f"wrote: {annot_path}")

    # confidence CSV (separate, just to make discovery easy)
    annot[["cell_id", "transferred_label", "transfer_confidence",
           "second_label", "second_confidence", "shared_genes"]].to_csv(
        outdir / f"{args.sample_prefix}_label_transfer_confidence.csv", index=False)
    # low-confidence
    low = annot[annot["transfer_confidence"] < args.low_confidence_threshold]
    low.to_csv(outdir / f"{args.sample_prefix}_low_confidence_cells.csv", index=False)
    # shared genes text
    shared_genes = sorted(set(query.var_names) & set(ref.var_names))
    with open(outdir / f"{args.sample_prefix}_shared_genes.txt", "w") as f:
        f.write("\n".join(shared_genes))

    # ---------- 6) QC summary ----------
    qc_summary = {
        "sample_prefix": args.sample_prefix,
        "reference_type": args.reference_type,
        "method": "python_cosine_centroid_softmax",
        "conos_pagoda2_available": False,
        "n_query_cells": qc["n_query_cells"],
        "n_reference_cells": qc["n_reference_cells"],
        "n_shared_genes": qc["n_shared_genes"],
        "n_classes": qc["n_classes"],
        "anchor_selection": args.anchor_selection,
        "max_reference_cells_per_type": args.max_reference_cells_per_type,
        "min_reference_cells_per_type": args.min_reference_cells_per_type,
        "low_confidence_threshold": qc["low_confidence_threshold"],
        "low_confidence_count": qc["low_confidence_count"],
        "low_confidence_fraction": qc["low_confidence_fraction"],
        "mean_confidence": qc["mean_confidence"],
        "median_confidence": qc["median_confidence"],
        "softmax_temperature": args.softmax_temperature,
    }
    pd.DataFrame(list(qc_summary.items()), columns=["metric", "value"]).to_csv(
        outdir / f"{args.sample_prefix}_label_transfer_qc_summary.csv", index=False)
    (outdir / f"{args.sample_prefix}_label_transfer_meta.json").write_text(
        json.dumps(qc_summary, indent=2, default=str))

    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
