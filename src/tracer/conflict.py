"""Reusable, auditable TRACER relative-conflict scoring.

Single source of truth for the Fig. 1 conflict calculation so Atera and
VisiumHD score cells the *same* way. It wraps the validated primitives in
:mod:`tracer.metrics` / :mod:`tracer._kernels` and adds:

* an explicit, recorded **gene background** (the NPMI gene-gene table). The
  background should be the widest available whole-transcriptome NPMI table for
  the dataset, NOT an HVG-restricted subset, unless an HVG run is being scored
  deliberately as a comparison.
* an optional **top-k present-in-profile** approximation: for each cell, only
  the ``top_k`` strongest (largest |ReLU-adjusted NPMI|) interactions *among
  the genes present in that cell* are summed. This bounds per-cell work at
  whole-transcriptome scale and, by construction, uses top interactions that
  are PRESENT in the profile (never top interactions absent from the cell).
* full **audit metadata** (number of background genes, per-cell gene / signal
  statistics, tau, top_k, runtime) returned alongside the scores.

The relative conflict itself is unchanged:
``relative_conflict = neg_relu / (pos_relu + neg_relu)`` where
``pos_relu``/``neg_relu`` are the summed symmetric-ReLU(NPMI, tau) positive /
negative contributions of the retained present-gene pairs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import build_cell_gene_matrix, build_pmi_matrix
from ._kernels import pair_aggregate_dense, pair_aggregate_topk

# NPMI-value column preference (first present wins).
_NPMI_COLS = ("NPMI", "npmi", "NPMI_median", "value")


@dataclass
class ConflictResult:
    """Per-cell conflict scores + provenance for the audit report."""

    scores: pd.DataFrame               # cell_id + relative_conflict + companions
    audit: dict = field(default_factory=dict)


def load_pmi_long(path, *, npmi_col: str | None = None) -> pd.DataFrame:
    """Load a long-format NPMI table (parquet or csv[.gz]) as gene_i/gene_j/NPMI.

    The NPMI value column is auto-detected from :data:`_NPMI_COLS` unless
    ``npmi_col`` is given. Raises with a clear message if no value column is
    found so a mis-specified table fails loudly rather than silently scoring 0.
    """
    path = str(path)
    cols_needed = ["gene_i", "gene_j"]
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        have = set(pq.read_schema(path).names)
    else:
        have = set(pd.read_csv(path, nrows=0).columns)
    if npmi_col is None:
        npmi_col = next((c for c in _NPMI_COLS if c in have), None)
    if npmi_col is None or npmi_col not in have:
        raise ValueError(
            f"No NPMI value column found in {path}. Looked for {_NPMI_COLS}; "
            f"available columns: {sorted(have)}")
    missing = [c for c in cols_needed if c not in have]
    if missing:
        raise ValueError(f"{path} missing gene columns {missing} (has {sorted(have)})")
    usecols = cols_needed + [npmi_col]
    df = (pd.read_parquet(path, columns=usecols) if path.endswith(".parquet")
          else pd.read_csv(path, usecols=usecols))
    df = df.rename(columns={npmi_col: "NPMI"})
    df["gene_i"] = df["gene_i"].astype(str)
    df["gene_j"] = df["gene_j"].astype(str)
    return df


def score_relative_conflict(
    filtered_df: pd.DataFrame,
    npmi_long: pd.DataFrame,
    *,
    tau: float = 0.05,
    min_transcripts: int = 10,
    top_k: int | None = 500,
    cell_col: str = "cell_id",
    feature_col: str = "feature_name",
    exclude_ids=None,
    conflict_percentile: float = 50.0,
    background_name: str = "whole_transcriptome",
) -> ConflictResult:
    """Score per-cell TRACER relative conflict against an NPMI background.

    Parameters
    ----------
    filtered_df : long-format transcripts with ``cell_col`` and ``feature_col``.
    npmi_long : gene-gene NPMI table (``gene_i``, ``gene_j``, ``NPMI``). This is
        the **gene background**; pass the whole-transcriptome table for
        production, or an HVG-restricted table only for an explicit comparison.
    tau : symmetric-ReLU dead-zone half-width.
    top_k : if not ``None``/<=0, keep only the ``top_k`` strongest present-gene
        interactions per cell (present-in-profile approximation). ``None`` =
        full aggregation over every present-gene pair.
    conflict_percentile : percentile (within the scored set) used for the
        boolean ``is_conflict`` flag. Fig. 1 uses the 50th percentile.
    background_name : label recorded in the audit (e.g. ``"whole_transcriptome"``
        or ``"HVG"``).

    Returns
    -------
    :class:`ConflictResult` with a per-cell ``scores`` DataFrame (columns
    ``cell_id, relative_conflict, relative_purity, cell_conflict_relu,
    signal_strength, n_present_genes, n_signal_pairs, is_conflict``) and an
    ``audit`` dict.
    """
    t0 = time.time()
    if feature_col != "feature_name":
        filtered_df = filtered_df.rename(columns={feature_col: "feature_name"})

    npmi_mat, gene_to_idx = build_pmi_matrix(npmi_long)
    n_background_genes = int(npmi_mat.shape[0])

    cell_ids, genes_cell, M, col_idx = build_cell_gene_matrix(
        filtered_df, min_transcripts=min_transcripts, genes_npm=npmi_long,
        cell_col=cell_col, exclude_ids=exclude_ids)
    n_cells = int(len(cell_ids))
    if n_cells == 0:
        raise ValueError("No cells passed build_cell_gene_matrix filters.")

    use_topk = top_k is not None and int(top_k) > 0
    eps = 1e-8
    if use_topk:
        k_arr, pos_relu, neg_relu, nsig = pair_aggregate_topk(
            M, col_idx, npmi_mat, tau=tau, top_k=int(top_k))
    else:
        k_arr, _n_pos, _sum_neg, pos_relu, neg_relu = pair_aggregate_dense(
            M, col_idx, npmi_mat, threshold=0.0, tau=tau)
        # count of signal pairs is not returned by the dense kernel; approximate
        nsig = np.full(n_cells, -1, dtype=np.int64)

    n_pairs_total = k_arr * (k_arr - 1) // 2
    has_pairs = n_pairs_total > 0
    total_abs = pos_relu + neg_relu

    relative_conflict = np.full(n_cells, np.nan)
    relative_purity = np.full(n_cells, np.nan)
    cell_conflict_relu = np.full(n_cells, np.nan)
    signal_strength = np.full(n_cells, np.nan)

    cell_conflict_relu[has_pairs] = neg_relu[has_pairs] / n_pairs_total[has_pairs]
    signal_strength[has_pairs] = total_abs[has_pairs]
    has_signal = has_pairs & (total_abs > eps)
    relative_conflict[has_signal] = neg_relu[has_signal] / total_abs[has_signal]
    relative_purity[has_signal] = pos_relu[has_signal] / total_abs[has_signal]

    valid = ~np.isnan(cell_conflict_relu)
    thr = float(np.nanpercentile(cell_conflict_relu[valid], conflict_percentile)) if valid.any() else np.nan
    is_conflict = np.zeros(n_cells, dtype=bool)
    is_conflict[valid] = cell_conflict_relu[valid] >= thr

    scores = pd.DataFrame({
        "cell_id": np.asarray(cell_ids, dtype=str),
        "relative_conflict": relative_conflict,
        "relative_purity": relative_purity,
        "cell_conflict_relu": cell_conflict_relu,
        "signal_strength": signal_strength,
        "n_present_genes": k_arr,
        "n_signal_pairs": nsig,
        "is_conflict": is_conflict,
    })

    kk = k_arr[k_arr > 0]
    ns = nsig[nsig >= 0]
    audit = {
        "background_name": background_name,
        "n_background_genes": n_background_genes,
        "n_cells_scored": n_cells,
        "tau": float(tau),
        "min_transcripts": int(min_transcripts),
        "top_k": (int(top_k) if use_topk else None),
        "scoring": ("top_k_present_in_profile" if use_topk else "full_present_pairs"),
        "conflict_percentile": float(conflict_percentile),
        "conflict_threshold": thr,
        "present_genes_per_cell": {
            "median": float(np.median(kk)) if kk.size else 0.0,
            "p90": float(np.percentile(kk, 90)) if kk.size else 0.0,
            "max": int(kk.max()) if kk.size else 0,
        },
        "signal_pairs_per_cell": ({
            "median": float(np.median(ns)) if ns.size else 0.0,
            "p90": float(np.percentile(ns, 90)) if ns.size else 0.0,
            "max": int(ns.max()) if ns.size else 0,
            "frac_cells_exceeding_top_k": (float(np.mean(ns > int(top_k))) if (use_topk and ns.size) else 0.0),
        } if ns.size else None),
        "runtime_s": round(time.time() - t0, 2),
    }
    return ConflictResult(scores=scores, audit=audit)
