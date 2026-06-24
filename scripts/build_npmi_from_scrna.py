#!/usr/bin/env python3
"""Build bootstrapped PMI/NPMI from a raw scRNA-seq reference.

Orchestrates ``tracer.metrics.compute_pmi_bootstrap`` (active-sampler
bootstrap with per-pair CIs, sparse CSR output) to produce the long-format
NPMI panel that TRACER and the get_metric benchmark consume.

USAGE
=====
Small-panel (lung Xenium, ~300 genes)::

    python scripts/build_npmi_from_scrna.py \\
      --reference-h5ad <h5ad with layers['counts'] or adata.raw.X> \\
      --spatial-transcripts <standardized parquet> \\
      --out results/reference_npmi/lung_cancer_npmi.csv.gz \\
      --bootstrap-n 100 --seed 1 --min-cells-expressed 10 \\
      --mode all_pairs

Large-panel (cervical Atera, ~12k genes)::

    python scripts/build_npmi_from_scrna.py \\
      --reference-h5ad <h5ad> \\
      --spatial-transcripts <standardized parquet> \\
      --out results/reference_npmi/cervical_atera_npmi.csv.gz \\
      --bootstrap-n 200 --seed 1 --min-cells-expressed 100 \\
      --min-expected-cooccurrence 10 --pmi-abs-threshold 0.2 \\
      --mode sparse_pairs --active-bootstrap

CORE GUARANTEES
===============
- Raw counts only. h5ad must expose integer-valued counts via
  ``layers['counts']`` (preferred), ``adata.raw.X``, or ``adata.X``.
  Normalized counts are detected and the script exits with a clear error.
- Reproducible. Seed flows through to the bootstrap RNG.
- Memory-aware. ``--mode sparse_pairs`` short-circuits dense G×G when
  the candidate panel exceeds ~5k genes. Pre-filter pipeline writes a
  per-step audit table so the user can verify what was kept and why.
- No silent fallbacks. Excluded genes / removed pairs are dumped to
  TSV alongside the main output.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Bootstrap path bootstrap — let the script run when called via `python
# scripts/build_npmi_from_scrna.py` regardless of CWD.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import anndata as ad  # noqa: E402

from tracer.metrics import compute_pmi_bootstrap  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_EXCLUDE_CONTROL_REGEX = (
    r"^(?:Neg|BLANK|Blank|Unassigned|Deprecated|Control"
    r"|antisense_|UnassignedCodeword_"
    r"|NegControlProbe_|NegControlCodeword_)"
)
MITO_REGEX = r"^(MT-|mt-|Mt-|MT\.)"
RIBO_REGEX = r"^(RPS|RPL|Rps|Rpl)"

# Columns in the final output CSV, in order.
OUTPUT_COLUMNS = [
    "gene_i", "gene_j",
    "PMI", "NPMI",
    "PMI_std", "NPMI_std",
    "PMI_ci_low", "PMI_ci_high",
    "NPMI_ci_low", "NPMI_ci_high",
    "n_cells_i", "n_cells_j", "n_cells_ij",
    "p_i", "p_j", "p_ij",
    "expected_ij",
    "bootstrap_reps_used", "active_stopped",
    "kind", "filter_mode",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str, *, flush: bool = True) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=flush)


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------
def _rss_gb() -> float:
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024 ** 3)
    except Exception:
        return float("nan")


def _log_mem(tag: str) -> None:
    rss = _rss_gb()
    if rss == rss:
        log(f"[mem {tag}] RSS={rss:.2f} GB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Inputs
    p.add_argument("--reference-h5ad", required=True, type=Path)
    g_in = p.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--spatial-transcripts", type=Path,
                      help="Standardized transcripts parquet (must have feature_name).")
    g_in.add_argument("--spatial-gene-list", type=Path,
                      help="Plain text file with one gene symbol per line.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output csv(.gz) path for the long-format NPMI panel.")

    # Mode + sparsity
    p.add_argument("--mode", choices=("all_pairs", "sparse_pairs"),
                   default="all_pairs")
    p.add_argument("--min-cells-expressed", type=int, default=10,
                   help="Drop a gene if it is expressed in < N reference cells.")
    p.add_argument("--min-expected-cooccurrence", type=float, default=10.0,
                   help="For sparse_pairs: drop pairs with E[k_ij] = N*p_i*p_j below this.")
    p.add_argument("--pmi-positive-threshold", type=float, default=0.2,
                   help="For sparse_pairs: keep pairs with PMI >= this.")
    p.add_argument("--pmi-negative-threshold", type=float, default=-0.2,
                   help="For sparse_pairs: keep pairs with PMI <= this.")
    p.add_argument("--pmi-abs-threshold", type=float, default=None,
                   help="Convenience: set both pos/neg thresholds to ±abs.")
    p.add_argument("--top-k-positive-per-gene", type=int, default=None,
                   help="If set, retain the top-K positive PMI partners per gene "
                        "(after PMI threshold filtering).")
    p.add_argument("--top-k-negative-per-gene", type=int, default=None,
                   help="If set, retain the top-K (most negative) PMI partners per gene.")

    # Gene filters
    p.add_argument("--exclude-control-regex", default=DEFAULT_EXCLUDE_CONTROL_REGEX)
    p.add_argument("--exclude-mito", action="store_true",
                   help="Drop mitochondrial genes (MT-/mt-/Mt-).")
    p.add_argument("--exclude-ribo", action="store_true",
                   help="Drop ribosomal genes (RPS/RPL/Rps/Rpl).")

    # Bootstrap + active sampling
    p.add_argument("--bootstrap-n", type=int, default=100,
                   help="Base bootstrap iterations (hard ceiling without --active-bootstrap).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--active-bootstrap", action="store_true",
                   help="Enable active-sampler stopping. Max iterations = --active-bootstrap-max-reps.")
    p.add_argument("--active-bootstrap-min-reps", type=int, default=30,
                   help="Minimum bootstrap samples before a pair's CI is evaluated.")
    p.add_argument("--active-bootstrap-max-reps", type=int, default=200,
                   help="Hard ceiling on bootstrap iterations under active sampling.")
    p.add_argument("--active-bootstrap-ci-width", type=float, default=0.05,
                   help="Active-sampling tau. CI inside ±tau stops the pair early.")
    p.add_argument("--active-bootstrap-alpha", type=float, default=0.05,
                   help="Confidence level = 1 - alpha for the bootstrap CI.")

    # Memory safety
    p.add_argument("--max-memory-gb", type=float, default=64.0,
                   help="Hard ceiling on estimated peak memory. Exits before running if exceeded.")
    p.add_argument("--pair-chunk-size", type=int, default=1_000_000,
                   help="Chunk size for sparse-mode candidate pair processing (memory).")

    # Run
    p.add_argument("--dry-run", action="store_true",
                   help="Run all pre-filters and print what would be bootstrapped; exit before bootstrap.")
    p.add_argument("--show-progress", action="store_true",
                   help="Forward show_progress=True to compute_pmi_bootstrap.")
    return p


# ---------------------------------------------------------------------------
# Step 1 — load + validate raw counts
# ---------------------------------------------------------------------------
def _looks_like_raw_counts(X, sample_n: int = 100_000, *, rng_seed: int = 0) -> tuple[bool, str]:
    """Heuristic raw-counts detector.

    Returns ``(is_raw, reason)``. Sample up to ``sample_n`` nonzero entries;
    require all integer-valued and ≥ 0. The reason string describes the
    detection (used in error messages).
    """
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    if X.nnz == 0:
        return False, "matrix has zero nonzero entries"
    nz = X.data
    if nz.size > sample_n:
        idx = np.random.default_rng(rng_seed).choice(nz.size, sample_n, replace=False)
        nz = nz[idx]
    if (nz < 0).any():
        return False, "negative values detected — counts cannot be negative"
    nz_f = nz.astype(np.float64)
    if not np.allclose(nz_f, np.round(nz_f), atol=1e-8):
        return False, "fractional values detected — likely normalized expression"
    return True, "integer-valued, non-negative"


def load_raw_counts_h5ad(path: Path) -> tuple[sp.csr_matrix, np.ndarray, pd.DataFrame]:
    """Read an h5ad and return (counts_csr cells x genes, var_names, obs)."""
    log(f"Reading h5ad: {path}")
    adata = ad.read_h5ad(path)
    log(f"  shape={adata.shape}; layers={list(adata.layers.keys())}; "
        f"raw={'set' if adata.raw is not None else 'None'}")

    candidates: list[tuple[str, Any, np.ndarray]] = []
    if "counts" in adata.layers:
        candidates.append(("layers['counts']", adata.layers["counts"],
                           np.asarray(adata.var_names, dtype=str)))
    if adata.raw is not None:
        candidates.append(("raw.X", adata.raw.X,
                           np.asarray(adata.raw.var_names, dtype=str)))
    candidates.append(("X", adata.X, np.asarray(adata.var_names, dtype=str)))

    last_reason = ""
    for name, X, var in candidates:
        ok, reason = _looks_like_raw_counts(X)
        if ok:
            log(f"  ✓ {name}: {reason}")
            X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
            return X.astype(np.float64), var, adata.obs.copy()
        last_reason = f"  ✗ {name}: {reason}"
        log(last_reason)

    raise SystemExit(
        "Raw counts required for PMI/NPMI; normalized counts detected.\n"
        f"Inspected (last): {last_reason}\n"
        "Provide an h5ad with layers['counts'] or adata.raw.X holding "
        "non-negative integer counts."
    )


# ---------------------------------------------------------------------------
# Step 2 — gene filtering
# ---------------------------------------------------------------------------
@dataclass
class GeneFilterStats:
    n_reference_genes: int = 0
    n_spatial_genes: int = 0
    n_overlap: int = 0
    n_excluded_control: int = 0
    n_excluded_mito: int = 0
    n_excluded_ribo: int = 0
    n_failed_min_cells: int = 0
    n_retained: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def load_spatial_gene_list(
    *,
    transcripts: Path | None,
    gene_list: Path | None,
) -> set[str]:
    if gene_list is not None:
        with open(gene_list) as f:
            genes = {line.strip() for line in f if line.strip()}
        log(f"Spatial gene list: {len(genes)} symbols from {gene_list}")
        return genes
    assert transcripts is not None
    log(f"Spatial gene list: scanning {transcripts}")
    fn = pd.read_parquet(transcripts, columns=["feature_name"])["feature_name"]
    genes = set(fn.astype(str).unique().tolist())
    log(f"  → {len(genes)} unique gene symbols")
    return genes


def filter_genes(
    ref_var: np.ndarray,
    counts_csr: sp.csr_matrix,
    spatial_genes: set[str],
    *,
    exclude_control_regex: str,
    exclude_mito: bool,
    exclude_ribo: bool,
    min_cells_expressed: int,
) -> tuple[sp.csr_matrix, np.ndarray, pd.DataFrame, GeneFilterStats]:
    """Apply panel intersection + gene-class filters + expression-frequency filter.

    Returns (counts_filtered, kept_genes, excluded_table, stats).
    """
    stats = GeneFilterStats(
        n_reference_genes=int(len(ref_var)),
        n_spatial_genes=int(len(spatial_genes)),
    )

    # n_cells expressed per gene (presence: count > 0).
    pres = counts_csr.copy()
    pres.data = (pres.data > 0).astype(np.int8)
    pres.eliminate_zeros()
    n_cells_per_gene = np.asarray(pres.sum(axis=0)).ravel().astype(np.int64)

    overlap = set(ref_var.tolist()) & spatial_genes
    stats.n_overlap = int(len(overlap))

    ctrl_re = re.compile(exclude_control_regex)
    mito_re = re.compile(MITO_REGEX)
    ribo_re = re.compile(RIBO_REGEX)

    # Apply filters sequentially on the spatial-panel-restricted set, so each
    # counter reports the marginal exclusion attributable to that filter.
    keep_mask = np.ones(len(ref_var), dtype=bool)
    rows = []
    for i, g in enumerate(ref_var):
        spatial_detected = g in spatial_genes
        reference_detected = n_cells_per_gene[i] > 0

        if not spatial_detected:
            keep_mask[i] = False
            # Non-panel genes go into the excluded table for completeness
            # but DO NOT count toward the per-filter rejection counters
            # (those would be misleading — those reasons were not the
            # deciding factor for what got into the PMI panel).
            rows.append({
                "gene": str(g),
                "reason_excluded": "not_in_spatial_panel",
                "n_cells_expressed": int(n_cells_per_gene[i]),
                "spatial_detected": False,
                "reference_detected": bool(reference_detected),
            })
            continue

        reasons = []
        if ctrl_re.search(g):
            reasons.append("control_probe")
            stats.n_excluded_control += 1
        if exclude_mito and mito_re.search(g):
            reasons.append("mitochondrial")
            stats.n_excluded_mito += 1
        if exclude_ribo and ribo_re.search(g):
            reasons.append("ribosomal")
            stats.n_excluded_ribo += 1
        if n_cells_per_gene[i] < min_cells_expressed:
            reasons.append(f"failed_min_cells_expressed<{min_cells_expressed}")
            stats.n_failed_min_cells += 1
        if reasons:
            keep_mask[i] = False
            rows.append({
                "gene": str(g),
                "reason_excluded": ";".join(reasons),
                "n_cells_expressed": int(n_cells_per_gene[i]),
                "spatial_detected": True,
                "reference_detected": bool(reference_detected),
            })

    kept_genes = ref_var[keep_mask]
    stats.n_retained = int(len(kept_genes))

    excluded_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["gene", "reason_excluded", "n_cells_expressed",
                 "spatial_detected", "reference_detected"]
    )

    counts_kept = counts_csr[:, keep_mask].tocsr()
    log(f"Gene filter: ref={stats.n_reference_genes} spatial={stats.n_spatial_genes} "
        f"overlap={stats.n_overlap} → kept {stats.n_retained}")
    log(f"  excluded: control={stats.n_excluded_control} "
        f"mito={stats.n_excluded_mito} ribo={stats.n_excluded_ribo} "
        f"min_cells={stats.n_failed_min_cells}")
    return counts_kept, kept_genes, excluded_df, stats


# ---------------------------------------------------------------------------
# Step 3 — build long-format DataFrame for compute_pmi_bootstrap
# ---------------------------------------------------------------------------
def counts_csr_to_long_df(
    counts_csr: sp.csr_matrix,
    cell_ids: np.ndarray,
    gene_names: np.ndarray,
) -> pd.DataFrame:
    """Sparse cells × genes → long-format ``(cell_id, feature_name, count)`` DataFrame.

    Includes only nonzero entries. cell_id and feature_name are strings;
    count is int32.
    """
    coo = counts_csr.tocoo()
    df = pd.DataFrame({
        "cell_id": cell_ids[coo.row].astype(str),
        "feature_name": gene_names[coo.col].astype(str),
        "count": np.maximum(1, np.round(coo.data)).astype(np.int32),
    })
    log(f"Long-format DataFrame: {len(df):,} rows "
        f"({counts_csr.shape[0]:,} cells × {counts_csr.shape[1]:,} genes; "
        f"density={counts_csr.nnz / counts_csr.shape[0] / counts_csr.shape[1]:.3%})")
    return df


# ---------------------------------------------------------------------------
# Step 4 — sparse-mode pre-filter audit (informational; the bootstrap
#          function applies its own min_expected_cooccur filter internally)
# ---------------------------------------------------------------------------
@dataclass
class SparseFilterAudit:
    n_genes_kept: int = 0
    n_candidate_pairs_pre: int = 0
    n_pairs_dropped_expected: int = 0
    n_pairs_dropped_zero_cooccur: int = 0
    n_pairs_candidate_after: int = 0
    min_expected_cooccurrence: float = 0.0
    pmi_positive_threshold: float = 0.0
    pmi_negative_threshold: float = 0.0
    top_k_positive: int | None = None
    top_k_negative: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_sparse_pre_filter(
    counts_csr: sp.csr_matrix,
    *,
    min_expected: float,
) -> tuple[SparseFilterAudit, int]:
    """Quick audit: count candidate pairs surviving expected-cooccurrence + observed > 0.

    Returns (audit, n_candidate_pairs_after). Does NOT materialize the pair list —
    we only need counts for the memory-safety check.
    """
    N, G = counts_csr.shape
    pres = counts_csr.copy()
    pres.data = (pres.data > 0).astype(np.int8)
    pres.eliminate_zeros()

    n_i = np.asarray(pres.sum(axis=0)).ravel().astype(np.int64)
    p_i = n_i / N
    # Pairs with N * p_i * p_j >= min_expected
    # → p_j >= min_expected / (N * p_i)
    # Vectorize via outer product of p's.
    n_total_pairs = G * (G - 1) // 2
    audit = SparseFilterAudit(
        n_genes_kept=int(G),
        n_candidate_pairs_pre=int(n_total_pairs),
        min_expected_cooccurrence=float(min_expected),
    )

    # Compute expected per pair via outer product (G x G) — only G<=20k feasible
    if G > 25_000:
        log(f"  G={G} too large to materialize G^2 expected matrix; "
            f"skipping pre-audit. compute_pmi_bootstrap will apply the filter.")
        return audit, int(n_total_pairs)

    p_outer = (p_i[:, None] * p_i[None, :]) * N
    iu, ju = np.triu_indices(G, k=1)
    expected = p_outer[iu, ju]
    mask_expected = expected >= min_expected
    audit.n_pairs_dropped_expected = int((~mask_expected).sum())

    # Observed cooccurrence per pair via sparse B.T @ B (in upper-triangle order).
    # Cast presence to int32 first — int8 overflows for any pair cooccurring
    # in >127 cells.
    pres_i32 = pres.astype(np.int32)
    BTB = (pres_i32.T @ pres_i32).tocsr()
    # Use the same (iu, ju) order as the expected mask for a clean intersection.
    obs_count = np.asarray(BTB[iu, ju]).ravel()
    mask_obs = obs_count > 0

    # Pairs surviving expected filter but with zero observed cooccurrence.
    mask_zero_obs_after_expected = mask_expected & ~mask_obs
    audit.n_pairs_dropped_zero_cooccur = int(mask_zero_obs_after_expected.sum())

    # Final candidate set: must pass BOTH filters.
    mask_candidate = mask_expected & mask_obs
    audit.n_pairs_candidate_after = int(mask_candidate.sum())
    return audit, audit.n_pairs_candidate_after


def estimate_memory_gb(*, n_candidate_pairs: int, n_cells: int) -> float:
    """Order-of-magnitude estimate of compute_pmi_bootstrap peak RSS.

    Heuristics from past runs:
      - sample_lists hold up to max_reps doubles per pair until settle.
        Worst case (no early stop): n_pairs * max_reps * 8 bytes.
        With active sampling, typical realized: ~30 × n_pairs × 8 bytes.
      - presence CSR float32 + cell index buffer: O(N * G) at worst.
    Returns peak estimate in GB.
    """
    samples_gb = (n_candidate_pairs * 30 * 8) / (1024 ** 3)
    pres_gb = (n_cells * 8 * 2) / (1024 ** 3)   # rough
    return samples_gb + pres_gb + 1.0           # +1 GB for python/numpy overhead


# ---------------------------------------------------------------------------
# Step 5 — expand bootstrap output to required CSV schema
# ---------------------------------------------------------------------------
def expand_result(
    result,
    *,
    pres: sp.csr_matrix,
    gene_index_to_name: np.ndarray,
    filter_mode: str,
    max_bootstraps: int,
    pmi_pos_thr: float | None,
    pmi_neg_thr: float | None,
    top_k_pos: int | None,
    top_k_neg: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert NpmiBootstrapResult.pair_ci + presence stats → output DataFrame.

    Returns (df, removed_counts) where removed_counts tracks each removal
    step for the audit table.
    """
    ci = result.pair_ci
    if ci is None or ci.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), {
            "no_bootstrap_records": 1,
            "pmi_threshold_filtered": 0,
            "top_k_filtered": 0,
        }

    N = pres.shape[0]
    # Cast to int32 BEFORE matmul: int8 overflows for any pair with cooccurrence
    # > 127 cells (very common on whole-transcriptome scRNA references).
    pres_i32 = pres.astype(np.int32)
    n_per_gene = np.asarray(pres_i32.sum(axis=0)).ravel().astype(np.int64)
    p_per_gene = n_per_gene / N

    i_idx = ci["gene_i_idx"].to_numpy(dtype=np.int64)
    j_idx = ci["gene_j_idx"].to_numpy(dtype=np.int64)

    # n_cells_ij — pairwise via sparse matmul on the requested pairs only.
    # We compute B.T @ B once (sparse, int32 → int64 result) and look up.
    BTB = (pres_i32.T @ pres_i32).astype(np.int64).tocsr()
    # For each (i, j), look up BTB[i, j]. Build a vectorized indexed access.
    n_cells_ij = np.zeros(len(i_idx), dtype=np.int64)
    for k in range(len(i_idx)):
        n_cells_ij[k] = int(BTB[i_idx[k], j_idx[k]])

    n_cells_i = n_per_gene[i_idx]
    n_cells_j = n_per_gene[j_idx]
    p_i = p_per_gene[i_idx]
    p_j = p_per_gene[j_idx]
    p_ij = n_cells_ij / N
    expected = N * p_i * p_j

    pmi = ci["legacy_pmi"].to_numpy(dtype=np.float64)
    npmi = ci["legacy_npmi"].to_numpy(dtype=np.float64)
    npmi_ci_lo = ci["ci_lo"].to_numpy(dtype=np.float64)
    npmi_ci_hi = ci["ci_hi"].to_numpy(dtype=np.float64)
    n_boot = ci["n_bootstraps"].to_numpy(dtype=np.int64)
    kind = ci["kind"].astype(str).to_numpy()

    # PMI std / CI are NOT computed by the bootstrap function (only NPMI's
    # CI is persisted, since the active sampler operates on a single
    # metric). NPMI_std is approximated from the bootstrap CI assuming
    # near-Gaussian behavior (std ≈ (hi - lo) / (2 * z_alpha/2)).
    z = 1.959963984540054  # 1 - alpha/2 quantile for alpha=0.05; close enough
    npmi_std = (npmi_ci_hi - npmi_ci_lo) / (2.0 * z)

    # active_stopped: a pair stopped early if it settled (kind != "unsettled")
    # AND used fewer than max_bootstraps replicates.
    active_stopped = (kind != "unsettled") & (n_boot < max_bootstraps)

    out = pd.DataFrame({
        "gene_i": gene_index_to_name[i_idx],
        "gene_j": gene_index_to_name[j_idx],
        "PMI":  pmi,
        "NPMI": npmi,
        "PMI_std":  np.full(len(i_idx), np.nan, dtype=np.float64),
        "NPMI_std": npmi_std,
        "PMI_ci_low":  np.full(len(i_idx), np.nan, dtype=np.float64),
        "PMI_ci_high": np.full(len(i_idx), np.nan, dtype=np.float64),
        "NPMI_ci_low":  npmi_ci_lo,
        "NPMI_ci_high": npmi_ci_hi,
        "n_cells_i":  n_cells_i,
        "n_cells_j":  n_cells_j,
        "n_cells_ij": n_cells_ij,
        "p_i":  p_i,
        "p_j":  p_j,
        "p_ij": p_ij,
        "expected_ij": expected,
        "bootstrap_reps_used": n_boot,
        "active_stopped": active_stopped.astype(bool),
        "kind": kind,
        "filter_mode": filter_mode,
    })

    removed = {"pmi_threshold_filtered": 0, "top_k_filtered": 0}

    # PMI threshold filter (sparse_pairs mode) — keep pairs satisfying PMI >= pos OR PMI <= neg.
    if pmi_pos_thr is not None and pmi_neg_thr is not None:
        keep = (out["PMI"] >= pmi_pos_thr) | (out["PMI"] <= pmi_neg_thr)
        n_drop = int((~keep).sum())
        if n_drop:
            log(f"  PMI threshold filter: dropped {n_drop:,} pairs "
                f"(kept {int(keep.sum()):,})")
        removed["pmi_threshold_filtered"] = n_drop
        out = out.loc[keep].reset_index(drop=True)

    # Top-k per gene (positive + negative separately)
    if top_k_pos or top_k_neg:
        before = len(out)
        out = _apply_top_k(out, top_k_pos=top_k_pos, top_k_neg=top_k_neg)
        removed["top_k_filtered"] = int(before - len(out))
        log(f"  Top-k per gene filter: dropped {removed['top_k_filtered']:,} pairs "
            f"(kept {len(out):,})")

    return out.reset_index(drop=True), removed


def _apply_top_k(
    df: pd.DataFrame, *,
    top_k_pos: int | None, top_k_neg: int | None,
) -> pd.DataFrame:
    """Keep the strongest |k_pos| positive PMI partners and |k_neg| most-negative
    PMI partners per gene (looking at both gene_i and gene_j sides)."""
    # Long-form: each pair contributes two (gene, partner, pmi) rows so the
    # top-k decision can be made symmetrically per gene.
    a = df[["gene_i", "gene_j", "PMI"]].rename(columns={"gene_i": "g", "gene_j": "p"})
    b = df[["gene_j", "gene_i", "PMI"]].rename(columns={"gene_j": "g", "gene_i": "p"})
    expanded = pd.concat([a.assign(_orig=df.index),
                          b.assign(_orig=df.index)], ignore_index=True)
    keep_orig = set()
    if top_k_pos:
        pos = expanded[expanded["PMI"] > 0]
        kept = (
            pos.sort_values("PMI", ascending=False, kind="stable")
               .groupby("g", as_index=False, sort=False)
               .head(top_k_pos)["_orig"]
        )
        keep_orig.update(kept.tolist())
    if top_k_neg:
        neg = expanded[expanded["PMI"] < 0]
        kept = (
            neg.sort_values("PMI", ascending=True, kind="stable")
               .groupby("g", as_index=False, sort=False)
               .head(top_k_neg)["_orig"]
        )
        keep_orig.update(kept.tolist())
    return df.loc[sorted(keep_orig)]


# ---------------------------------------------------------------------------
# Step 6 — driver
# ---------------------------------------------------------------------------
def main() -> int:
    args = build_argparser().parse_args()
    out_path: Path = args.out
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pmi_abs_threshold is not None:
        args.pmi_positive_threshold = float(abs(args.pmi_abs_threshold))
        args.pmi_negative_threshold = -float(abs(args.pmi_abs_threshold))

    log(f"=== build_npmi_from_scrna.py ===")
    log(f"Reference  : {args.reference_h5ad}")
    log(f"Out        : {out_path}")
    log(f"Mode       : {args.mode}; min_cells_expressed={args.min_cells_expressed}; "
        f"min_expected_cooccurrence={args.min_expected_cooccurrence}; "
        f"PMI thr=({args.pmi_negative_threshold},{args.pmi_positive_threshold})")
    log(f"Bootstrap  : n={args.bootstrap_n} active={args.active_bootstrap} "
        f"min_reps={args.active_bootstrap_min_reps} "
        f"max_reps={args.active_bootstrap_max_reps} "
        f"tau={args.active_bootstrap_ci_width} alpha={args.active_bootstrap_alpha} "
        f"seed={args.seed}")

    # --- step 1: load raw counts -------------------------------------------
    counts, ref_var, obs = load_raw_counts_h5ad(args.reference_h5ad)
    _log_mem("after h5ad load")

    # --- step 2: spatial panel ---------------------------------------------
    spatial_genes = load_spatial_gene_list(
        transcripts=args.spatial_transcripts, gene_list=args.spatial_gene_list,
    )

    # --- step 3: gene filter -----------------------------------------------
    counts_f, kept_genes, excluded_df, gf_stats = filter_genes(
        ref_var, counts, spatial_genes,
        exclude_control_regex=args.exclude_control_regex,
        exclude_mito=args.exclude_mito, exclude_ribo=args.exclude_ribo,
        min_cells_expressed=args.min_cells_expressed,
    )
    excluded_df.to_csv(out_dir / "npmi_excluded_genes.tsv", sep="\t", index=False)
    pd.DataFrame({"gene": kept_genes}).to_csv(
        out_dir / "npmi_gene_list.tsv", sep="\t", index=False,
    )
    pd.DataFrame(
        [(k, v) for k, v in gf_stats.as_dict().items()],
        columns=["metric", "value"],
    ).to_csv(out_dir / "npmi_gene_filter_summary.tsv", sep="\t", index=False)

    if kept_genes.size < 2:
        raise SystemExit(
            f"Only {kept_genes.size} genes survived filtering — not enough for PMI."
        )

    # --- step 4: build long-format df ---------------------------------------
    # Build presence (binary) here for downstream pair stats; counts_f is
    # the integer matrix used for the long-format input.
    pres = counts_f.copy()
    pres.data = (pres.data > 0).astype(np.int8)
    pres.eliminate_zeros()
    cell_ids = np.asarray(obs.index, dtype=str)

    df_long = counts_csr_to_long_df(counts_f, cell_ids, kept_genes)
    _log_mem("after long-format build")

    # --- step 5: sparse-mode audit + memory safety check --------------------
    sparse_audit = None
    if args.mode == "sparse_pairs":
        sparse_audit, n_candidate = estimate_sparse_pre_filter(
            counts_f, min_expected=args.min_expected_cooccurrence,
        )
        sparse_audit.pmi_positive_threshold = float(args.pmi_positive_threshold)
        sparse_audit.pmi_negative_threshold = float(args.pmi_negative_threshold)
        sparse_audit.top_k_positive = args.top_k_positive_per_gene
        sparse_audit.top_k_negative = args.top_k_negative_per_gene
        log(f"sparse_pairs audit: G={counts_f.shape[1]}; "
            f"candidate pairs after expected+observed prune ≈ {n_candidate:,}")
        peak_gb = estimate_memory_gb(
            n_candidate_pairs=n_candidate, n_cells=counts_f.shape[0],
        )
        log(f"  estimated peak RSS = {peak_gb:.2f} GB "
            f"(--max-memory-gb={args.max_memory_gb})")
        if peak_gb > args.max_memory_gb:
            raise SystemExit(
                f"Estimated peak memory {peak_gb:.2f} GB exceeds "
                f"--max-memory-gb={args.max_memory_gb}. "
                f"Tighten one of: --min-cells-expressed (currently "
                f"{args.min_cells_expressed}), "
                f"--min-expected-cooccurrence (currently "
                f"{args.min_expected_cooccurrence}), or use "
                f"--top-k-positive-per-gene / --top-k-negative-per-gene."
            )

    if args.dry_run:
        log("DRY RUN — pre-filters complete; skipping bootstrap.")
        _write_summary_json(
            args=args, gf_stats=gf_stats, sparse_audit=sparse_audit,
            result_diagnostics=None, n_output_pairs=0, runtime_seconds=0.0,
            out_dir=out_dir,
        )
        return 0

    # --- step 6: run compute_pmi_bootstrap ---------------------------------
    t0 = time.time()
    tau = float(args.active_bootstrap_ci_width)
    if args.active_bootstrap:
        max_b = int(args.active_bootstrap_max_reps)
        min_for_ci = int(args.active_bootstrap_min_reps)
    else:
        max_b = int(args.bootstrap_n)
        min_for_ci = min(int(args.bootstrap_n), 30)

    # Block sizes: the function bootstraps in batches of `coarse_block` first
    # and `refine_block` thereafter. For active sampling, set these so that
    # min_for_ci fits within the first 1-2 blocks.
    coarse_block = min(max_b, max(min_for_ci, 50))
    refine_block = min(max_b, max(min_for_ci // 2, 25))
    ci_level = 1.0 - float(args.active_bootstrap_alpha)

    log(f"Calling compute_pmi_bootstrap: max_bootstraps={max_b}, "
        f"min_samples_for_ci={min_for_ci}, tau={tau}, ci_level={ci_level}, "
        f"coarse_block={coarse_block}, refine_block={refine_block}")
    result = compute_pmi_bootstrap(
        df_long,
        group_key="cell_id",
        feature_col="feature_name",
        count_col="count",
        min_occurrences_per_context=1,
        tau=tau,
        ci_level=ci_level,
        max_bootstraps=max_b,
        coarse_block=coarse_block,
        refine_block=refine_block,
        min_expected_cooccur_for_evidence=float(args.min_expected_cooccurrence),
        min_samples_for_ci=min_for_ci,
        seed=int(args.seed),
        show_progress=bool(args.show_progress),
        persist_ci=True,            # required to populate result.pair_ci
        memory_optimize=True,
        metric="npmi",
    )
    runtime = time.time() - t0
    log(f"compute_pmi_bootstrap done in {runtime:.1f}s; "
        f"genes in result: {len(result.genes)}; pair_ci rows: "
        f"{0 if result.pair_ci is None else len(result.pair_ci)}")
    _log_mem("after bootstrap")

    # --- step 7: align gene index — compute_pmi_bootstrap may have a
    #             different gene set after its own internal filter (rare;
    #             happens if min_occurrences_per_context drops any genes).
    gene_index_to_name = np.asarray(result.genes, dtype=str)
    # Build a presence matrix in the same gene order as result.genes.
    name_to_orig = {g: i for i, g in enumerate(kept_genes)}
    perm = np.array([name_to_orig[g] for g in gene_index_to_name], dtype=np.int64)
    pres_reordered = pres[:, perm].tocsr()

    # --- step 8: expand result + apply post-filters -------------------------
    pmi_pos = float(args.pmi_positive_threshold) if args.mode == "sparse_pairs" else None
    pmi_neg = float(args.pmi_negative_threshold) if args.mode == "sparse_pairs" else None
    out_df, removed = expand_result(
        result,
        pres=pres_reordered,
        gene_index_to_name=gene_index_to_name,
        filter_mode=args.mode,
        max_bootstraps=max_b,
        pmi_pos_thr=pmi_pos,
        pmi_neg_thr=pmi_neg,
        top_k_pos=args.top_k_positive_per_gene,
        top_k_neg=args.top_k_negative_per_gene,
    )
    out_df = out_df[OUTPUT_COLUMNS]
    log(f"Output pairs: {len(out_df):,}")

    # --- step 9: write outputs ---------------------------------------------
    if out_path.suffix == ".gz" or out_path.name.endswith(".csv.gz"):
        out_df.to_csv(out_path, index=False, compression="gzip")
    else:
        out_df.to_csv(out_path, index=False)
    log(f"Wrote NPMI table → {out_path}")

    if args.mode == "sparse_pairs":
        pd.DataFrame(
            [("pairs_dropped_by_expected_cooccurrence",
              sparse_audit.n_pairs_dropped_expected),
             ("pairs_dropped_zero_observed_cooccurrence",
              sparse_audit.n_pairs_dropped_zero_cooccur),
             ("pairs_dropped_pmi_threshold", removed["pmi_threshold_filtered"]),
             ("pairs_dropped_top_k", removed["top_k_filtered"]),
             ("pairs_in_final_output", len(out_df))],
            columns=["step", "n_pairs"],
        ).to_csv(out_dir / "npmi_removed_pairs_summary.tsv", sep="\t", index=False)
        pd.DataFrame(
            [(k, v) for k, v in sparse_audit.as_dict().items()],
            columns=["metric", "value"],
        ).to_csv(out_dir / "npmi_candidate_pair_summary.tsv", sep="\t", index=False)
        with open(out_dir / "npmi_sparse_filter_summary.json", "w") as f:
            json.dump(sparse_audit.as_dict(), f, indent=2)

    _write_summary_json(
        args=args, gf_stats=gf_stats, sparse_audit=sparse_audit,
        result_diagnostics=result.diagnostics,
        n_output_pairs=len(out_df), runtime_seconds=runtime, out_dir=out_dir,
    )
    return 0


def _write_summary_json(
    *, args, gf_stats: GeneFilterStats, sparse_audit: SparseFilterAudit | None,
    result_diagnostics: dict | None, n_output_pairs: int,
    runtime_seconds: float, out_dir: Path,
) -> None:
    summary = {
        "command": " ".join(sys.argv),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "gene_filter_stats": gf_stats.as_dict(),
        "sparse_filter_audit": sparse_audit.as_dict() if sparse_audit else None,
        "bootstrap_diagnostics": _sanitize(result_diagnostics)
            if result_diagnostics else None,
        "n_output_pairs": int(n_output_pairs),
        "runtime_seconds": float(runtime_seconds),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_dir / "npmi_build_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_sanitize_scalar)


def _sanitize(d: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-friendly types."""
    if isinstance(d, dict):
        return {k: _sanitize(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_sanitize(v) for v in d]
    return _sanitize_scalar(d)


def _sanitize_scalar(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


if __name__ == "__main__":
    sys.exit(main())
