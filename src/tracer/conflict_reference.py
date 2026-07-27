"""Depth-normalized whole-transcriptome conflict-reference builders.

The naive whole-transcriptome PMI/NPMI co-detection prior is globally biased by
scRNA **library/detection-depth heterogeneity**: high-depth cells co-detect
almost everything, so the independence expectation ``p_i * p_j`` is systematically
too low and nearly every gene pair gets a positive PMI. Empirically the observed
median PMI is ``~+0.66`` while the depth-only Chung-Lu prediction
``log(1 + CV_depth**2)`` accounts for the bulk of it. That inflation compresses
the negative (conflict) tail and hides genuine mutual-exclusion edges.

This module builds a **depth-conditioned null** and expresses each edge as a
depth-corrected PMI so positive/negative coherence is measured *relative to what
depth alone predicts*, preserving whole-transcriptome coverage while removing the
library-depth-driven false positives.

Two builders:

``build_depth_corrected_reference``
    Fits a depth-conditioned detection null ``P(gene detected | cell depth)`` with
    quantile depth bins (a piecewise-constant spline; conditional gene
    independence *within* a depth stratum), then for each gene pair computes the
    observed co-detection ``O`` and the expected co-detection ``E`` under that
    null and reports ``cPMI = log((O + eps) / (E + eps))`` and a Poisson z-score
    ``(O - E) / sqrt(E + eps)``. It also **recovers strong mutual-exclusion edges
    that sparse ``M^T M`` misses** (pairs with ``E >= min_expected_neg`` but
    ``O <= max_observed_neg``), searched only among the well-detected genes that
    can reach that expected count (never by enumerating all pairs), and can keep
    the ``top_k_per_gene`` strongest edges by ``|cPMI|``.

``build_rarefied_reference``
    Sensitivity option (not the default path). Rarefies every cell to a fixed
    target UMI depth, recomputes PMI/NPMI per bootstrap, and keeps only edges
    whose sign is stable across bootstraps -- a depth-equalized cross-check on the
    corrected reference.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "ReferenceResult",
    "build_depth_corrected_reference",
    "build_rarefied_reference",
]


@dataclass
class ReferenceResult:
    """A gene-gene conflict reference plus provenance for the audit report."""

    edges: pd.DataFrame                # gene_i, gene_j, weight columns
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #
def _counts_and_genes(adata, counts, genes, *, layer="counts", var_symbol="symbol"):
    """Return (csr counts [cells x genes], gene-name array) from an AnnData or an
    explicit (counts, genes) pair. Counts are the raw UMI matrix, not presence."""
    if adata is not None:
        X = adata.layers[layer] if (layer and layer in getattr(adata, "layers", {})) else adata.X
        g = (adata.var[var_symbol].to_numpy() if var_symbol in adata.var.columns
             else np.asarray(adata.var_names))
        return sp.csr_matrix(X), np.asarray(g, dtype=str)
    if counts is None or genes is None:
        raise ValueError("Pass either `adata` or both `counts` and `genes`.")
    return sp.csr_matrix(counts), np.asarray(genes, dtype=str)


def _depth_bins(depth, n_bins):
    """Assign each cell to a quantile depth bin; returns (bin_of_cell, n_bins_used).

    Quantile edges collapse when depth is highly tied; we merge empty bins so
    ``n_bins_used`` reflects the realized strata."""
    depth = np.asarray(depth, float)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(depth, qs))
    # digitize into (len(edges)-1) bins; clip the top edge inclusive
    b = np.clip(np.digitize(depth, edges[1:-1], right=False), 0, len(edges) - 2)
    # compact bin ids
    uniq, b = np.unique(b, return_inverse=True)
    return b.astype(np.int64), int(len(uniq))


def _expected_cooccur_chunked(R_scaled, R, ii, jj, *, chunk=8_000_000):
    """E_ij = sum_d n_d * r_id * r_jd for the pair index arrays (ii, jj).

    ``R`` is (G x D) detection-rate matrix, ``R_scaled = R * n_d`` (broadcast over
    D). Computed in row chunks so peak memory stays bounded on a laptop."""
    E = np.empty(len(ii), np.float64)
    for s in range(0, len(ii), chunk):
        sl = slice(s, s + chunk)
        E[sl] = np.einsum("kd,kd->k", R_scaled[ii[sl]], R[jj[sl]], optimize=True)
    return E


# --------------------------------------------------------------------------- #
# depth-corrected reference
# --------------------------------------------------------------------------- #
def build_depth_corrected_reference(
    adata=None,
    *,
    counts=None,
    genes=None,
    depth=None,
    layer="counts",
    var_symbol="symbol",
    min_count: int = 1,
    min_det_cells: int = 25,
    n_depth_bins: int = 25,
    depth_metric: str = "total_counts",
    eps: float = 1.0,
    min_cooccur: int = 2,
    min_expected_neg: float = 5.0,
    max_observed_neg: int = 1,
    neg_recovery_top_m: int = 2500,
    top_k_per_gene: int | None = None,
    clip: float | None = 4.0,
) -> ReferenceResult:
    """Build a depth-corrected whole-transcriptome conflict reference.

    Parameters
    ----------
    adata / counts+genes : the scRNA reference. ``adata`` uses ``layer`` (raw
        counts) and ``var[var_symbol]``; or pass a cells x genes count matrix and
        a matching gene-name array.
    depth : optional explicit per-cell depth covariate. Default derives it from
        ``depth_metric`` (``"total_counts"`` = UMI sum, ``"n_genes"`` = detected
        gene count).
    min_det_cells : keep genes detected in at least this many cells.
    n_depth_bins : quantile strata for the depth-conditioned null.
    eps : Laplace pseudo-count in ``log((O+eps)/(E+eps))`` (add-one by default;
        it sets the floor of the mutual-exclusion signal at ``O=0``).
    min_expected_neg / max_observed_neg / neg_recovery_top_m : recover strong
        negative edges (``E >= min_expected_neg`` and ``O <= max_observed_neg``)
        among the top ``neg_recovery_top_m`` best-detected genes -- the only genes
        whose expected co-detection can clear the threshold -- instead of
        enumerating all pairs.
    top_k_per_gene : if set, keep only each gene's ``top_k`` strongest edges by
        ``|cPMI|`` (edges present/relevant to the gene, never absent partners).
    clip : clip ``cPMI`` to ``[-clip, clip]`` (``None`` disables) so the ``O=0``
        tail stays bounded.

    Returns
    -------
    ReferenceResult with ``edges`` (``gene_i, gene_j, O, E, z, PMI, cPMI, cNPMI``)
    and an ``meta`` audit dict.
    """
    t0 = time.time()
    X, gene_names = _counts_and_genes(adata, counts, genes, layer=layer, var_symbol=var_symbol)
    Ccells_all = X.shape[0]
    if depth is None:
        depth = (np.asarray(X.sum(1)).ravel() if depth_metric == "total_counts"
                 else np.asarray((X >= min_count).sum(1)).ravel())
    else:
        depth = np.asarray(depth, float).ravel()

    # binary detection + gene filter
    P = (X >= min_count).astype(np.float32)
    P.eliminate_zeros()
    kc = np.asarray(P.sum(0)).ravel()
    keep = kc >= min_det_cells
    P = P.tocsc()[:, keep].tocsr()
    gene_names = gene_names[keep]
    C, G = P.shape
    marg = np.asarray(P.sum(0)).ravel()

    # depth-conditioned null: per-bin detection rate r_gd, expected E_ij = sum_d n_d r_id r_jd
    binof, D = _depth_bins(depth, n_depth_bins)
    n_d = np.bincount(binof, minlength=D).astype(np.float64)
    # bin-indicator (cells x D); R = (P^T @ B) / n_d  -> (G x D) detection rates
    B = sp.csr_matrix((np.ones(C, np.float32), (np.arange(C), binof)), shape=(C, D))
    R = np.asarray((P.T @ B).todense(), np.float64) / np.maximum(n_d, 1.0)[None, :]
    R_scaled = R * n_d[None, :]

    # observed co-detection (co >= min_cooccur) from sparse M^T M
    co = (P.T @ P).tocoo()
    m = (co.row < co.col) & (co.data >= min_cooccur)
    ii = co.row[m].astype(np.int64)
    jj = co.col[m].astype(np.int64)
    O = co.data[m].astype(np.float64)
    del co
    n_observed = int(len(O))
    E = _expected_cooccur_chunked(R_scaled, R, ii, jj)

    # ---- recover strong mutual-exclusion edges missed by sparse M^T M --------
    # Only well-detected genes can reach E >= min_expected_neg, so search the
    # dense top-M block (never all pairs).
    M = int(min(neg_recovery_top_m, G))
    top = np.argsort(-marg)[:M]
    top_set = np.empty(G, bool); top_set[:] = False; top_set[top] = True
    Emm = (R_scaled[top] @ R[top].T)              # M x M dense expected
    # observed within the top block, from the already-computed observed arrays
    in_top = top_set[ii] & top_set[jj]
    pos_in_top = -np.ones(G, np.int64); pos_in_top[top] = np.arange(M)
    Omm = np.zeros((M, M), np.float64)
    ai = pos_in_top[ii[in_top]]; aj = pos_in_top[jj[in_top]]
    Omm[ai, aj] = O[in_top]; Omm[aj, ai] = O[in_top]
    iu = np.triu_indices(M, k=1)
    miss = (Emm[iu] >= min_expected_neg) & (Omm[iu] <= max_observed_neg)
    ri = top[iu[0][miss]]; rj = top[iu[1][miss]]; rE = Emm[iu][miss]
    rO = Omm[iu][miss]
    n_recovered = int(len(ri))
    # concatenate recovered (may double-count observed pairs already having O<=1;
    # those weren't in the observed set unless co>=min_cooccur, so O>=2 there --
    # recovered O<=1 are disjoint from observed by construction).
    ii = np.concatenate([ii, ri]); jj = np.concatenate([jj, rj])
    O = np.concatenate([O, rO]); E = np.concatenate([E, rE])

    # ---- corrected association ----------------------------------------------
    cPMI = np.log((O + eps) / (E + eps))
    z = (O - E) / np.sqrt(E + eps)
    # naive (uncorrected) PMI for reference/comparison
    p_ij = (O + eps) / (C + 2 * eps)
    p_i = (marg[ii] + eps) / (C + 2 * eps)
    p_j = (marg[jj] + eps) / (C + 2 * eps)
    PMI = np.log(p_ij / (p_i * p_j))
    # bounded normalization symmetric in sign: divide by the pair's -log(p_min)
    # scale (independent of O so the O->0 tail is not erased like observed-NPMI).
    p_min = (np.minimum(marg[ii], marg[jj]) + eps) / (C + 2 * eps)
    denom = -np.log(p_min)
    cNPMI = np.where(denom > 0, cPMI / denom, 0.0)
    if clip is not None:
        cPMI = np.clip(cPMI, -clip, clip)
        cNPMI = np.clip(cNPMI, -1.0, 1.0)

    edges = pd.DataFrame({
        "gene_i": gene_names[ii], "gene_j": gene_names[jj],
        "O": O.astype(np.int64), "E": E.astype(np.float32),
        "z": z.astype(np.float32), "PMI": PMI.astype(np.float32),
        "cPMI": cPMI.astype(np.float32), "cNPMI": cNPMI.astype(np.float32),
    })

    # ---- top_k_per_gene by |cPMI| (present/relevant partners only) ----------
    if top_k_per_gene is not None and int(top_k_per_gene) > 0:
        k = int(top_k_per_gene)
        both = pd.concat([
            edges.assign(g=edges.gene_i),
            edges.assign(g=edges.gene_j),
        ], ignore_index=True)
        both["absw"] = both.cPMI.abs()
        rank = both.groupby("g", sort=False)["absw"].rank(method="first", ascending=False)
        # an edge survives if it ranks in the top-k of EITHER endpoint. `both` is
        # two stacked copies of `edges`, so row t and row t+len(edges) are the two
        # endpoint views of edge t.
        both_idx = np.tile(np.arange(len(edges)), 2)
        survive = np.zeros(len(edges), bool)
        survive[both_idx[(rank <= k).to_numpy()]] = True
        edges = edges.loc[survive].reset_index(drop=True)

    meta = {
        "builder": "depth_corrected",
        "n_cells": int(C),
        "n_cells_input": int(Ccells_all),
        "n_genes_input": int(len(keep)),
        "n_genes_retained": int(G),
        "depth_metric": depth_metric,
        "depth_cv": float(np.std(depth) / max(np.mean(depth), 1e-9)),
        "chung_lu_inflation_1p_cv2": float(1 + (np.std(depth) / max(np.mean(depth), 1e-9)) ** 2),
        "n_depth_bins_used": int(D),
        "min_det_cells": int(min_det_cells),
        "eps": float(eps),
        "n_observed_pairs": n_observed,
        "n_recovered_neg_pairs": n_recovered,
        "neg_recovery": {"min_expected": float(min_expected_neg),
                          "max_observed": int(max_observed_neg),
                          "top_m_genes": int(M)},
        "n_edges": int(len(edges)),
        "top_k_per_gene": (int(top_k_per_gene) if top_k_per_gene else None),
        "cpmi_median": float(np.median(edges.cPMI)),
        "raw_pmi_median": float(np.median(edges.PMI)),
        "runtime_s": round(time.time() - t0, 1),
    }
    return ReferenceResult(edges=edges, meta=meta)


# --------------------------------------------------------------------------- #
# rarefied reference (sensitivity option)
# --------------------------------------------------------------------------- #
def _rarefy_csr(X, target, rng):
    """Downsample each cell's counts to `target` total UMIs (multinomial, with
    replacement -- a fast standard rarefaction approximation). Cells with total
    <= target keep their counts. Returns a binary presence CSR."""
    X = X.tocsr()
    C = X.shape[0]
    rows, cols = [], []
    indptr, indices, data = X.indptr, X.indices, X.data
    for c in range(C):
        s, e = indptr[c], indptr[c + 1]
        if e == s:
            continue
        cnt = data[s:e].astype(np.float64)
        tot = cnt.sum()
        cols_c = indices[s:e]
        if tot <= target:
            present = cols_c
        else:
            draw = rng.multinomial(int(target), cnt / tot)
            present = cols_c[draw > 0]
        rows.append(np.full(len(present), c)); cols.append(present)
    r = np.concatenate(rows); cc = np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(r), np.int8), (r, cc)), shape=X.shape)


def build_rarefied_reference(
    adata=None,
    *,
    counts=None,
    genes=None,
    layer="counts",
    var_symbol="symbol",
    target_depth: float | None = None,
    target_percentile: float = 10.0,
    n_bootstraps: int = 10,
    min_det_cells: int = 25,
    min_cooccur: int = 2,
    eps: float = 1.0,
    alpha: float = 0.1,
    sign_stability: float = 0.8,
    seed: int = 0,
) -> ReferenceResult:
    """Depth-equalized rarefied PMI/NPMI reference (sensitivity cross-check).

    Every cell is rarefied to ``target_depth`` UMIs (default: the
    ``target_percentile`` of total counts) so per-cell depth no longer varies,
    then PMI/NPMI is recomputed each of ``n_bootstraps`` times. Only edges whose
    sign is consistent in at least ``sign_stability`` of the bootstraps are kept;
    the reported ``NPMI``/``PMI`` are bootstrap means over the stable edges.
    """
    t0 = time.time()
    X, gene_names = _counts_and_genes(adata, counts, genes, layer=layer, var_symbol=var_symbol)
    X = X.tocsr()
    tot = np.asarray(X.sum(1)).ravel()
    if target_depth is None:
        target_depth = float(np.percentile(tot[tot > 0], target_percentile))
    # fix the gene set on full-depth detection so edge identity is stable
    kc = np.asarray((X >= 1).sum(0)).ravel()
    keep = kc >= min_det_cells
    Xk = X.tocsc()[:, keep].tocsr()
    gk = gene_names[keep]
    C, G = Xk.shape

    rng = np.random.default_rng(seed)
    # accumulate every bootstrap's edges, then group once (vectorized) rather than
    # a per-edge Python dict (millions of edges x bootstraps).
    keys_all, pmi_all, npmi_all = [], [], []
    for b in range(n_bootstraps):
        Pb = _rarefy_csr(Xk, target_depth, rng)
        marg = np.asarray(Pb.sum(0)).ravel()
        co = (Pb.T @ Pb).tocoo()
        mm = (co.row < co.col) & (co.data >= min_cooccur)
        ii = co.row[mm].astype(np.int64); jj = co.col[mm].astype(np.int64)
        cv = co.data[mm].astype(np.float64)
        Na = C + 2 * alpha
        pij = (cv + alpha) / Na
        pipj = ((marg[ii] + alpha) / Na) * ((marg[jj] + alpha) / Na)
        pmi = np.log(pij / pipj)
        keys_all.append(ii * np.int64(G) + jj)
        pmi_all.append(pmi)
        npmi_all.append(pmi / (-np.log(pij)))
    acc = pd.DataFrame({
        "key": np.concatenate(keys_all),
        "PMI": np.concatenate(pmi_all),
        "NPMI": np.concatenate(npmi_all),
    })
    acc["sgn"] = np.sign(acc.PMI)
    g = acc.groupby("key", sort=False)
    stat = g.agg(PMI=("PMI", "mean"), NPMI=("NPMI", "mean"),
                 sgnsum=("sgn", "sum"), n_boot=("PMI", "size")).reset_index()
    need = max(2, int(np.ceil(sign_stability * n_bootstraps)))
    stat["sign_frac"] = stat.sgnsum.abs() / stat.n_boot
    stat = stat[(stat.n_boot >= need) & (stat.sign_frac >= sign_stability)]
    ii = (stat.key.to_numpy() // G).astype(np.int64)
    jj = (stat.key.to_numpy() % G).astype(np.int64)
    edges = pd.DataFrame({"gene_i": gk[ii], "gene_j": gk[jj],
                          "PMI": stat.PMI.to_numpy(np.float32),
                          "NPMI": stat.NPMI.to_numpy(np.float32),
                          "sign_frac": stat.sign_frac.to_numpy(np.float32),
                          "n_boot": stat.n_boot.to_numpy(np.int32)})
    meta = {
        "builder": "rarefied",
        "n_cells": int(C), "n_genes_retained": int(G),
        "target_depth_umi": float(target_depth),
        "target_percentile": float(target_percentile),
        "n_bootstraps": int(n_bootstraps),
        "sign_stability": float(sign_stability),
        "min_det_cells": int(min_det_cells),
        "n_edges": int(len(edges)),
        "npmi_median": float(edges.NPMI.median()) if len(edges) else float("nan"),
        "pmi_median": float(edges.PMI.median()) if len(edges) else float("nan"),
        "runtime_s": round(time.time() - t0, 1),
    }
    return ReferenceResult(edges=edges, meta=meta)
