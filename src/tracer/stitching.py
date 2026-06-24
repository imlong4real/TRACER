"""Phase 4: Hierarchical entity stitching.

Stitch acceptance is layered:

  (1) candidate enumeration via bin-grid + ``dist_threshold`` pairwise
      tx-tx edges,
  (2) witness floor (``min_local_tx_per_entity``) — the spatial sanity
      check on the accept path: each entity must contribute K unique
      tx in the shared bin neighborhood,
  (3) ΔC test with C(union) bypass — composition is the primary
      acceptance gate (ΔC ≥ ``deltaC_min``, or C(union) ≥
      ``c_union_bypass`` when size-capped),
  (4) optional Mahalanobis RESCUE (``mahalanobis_d_rescue``) — when
      composition borderline-rejects (``rescue_delta_c_floor`` <
      ΔC < 0) AND the two tx clouds are geometrically enmeshed
      (Maha D ≤ threshold), override the rejection. Recovers EMT-like
      cells whose two-program anti-correlation makes ΔC reject a
      legitimate single-cell merge. The ΔC floor protects against
      fusing engulfment doublets where composition rejects strongly,
  (5) ``max_merger_depth`` cap on merger-tree height.

NOTE: an earlier veto-direction Mahalanobis implementation was
superseded — the witness count adequately gates the accept path;
geometry's useful contribution is the rescue, not a veto.
"""

import heapq
import itertools
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ._repro import _ensure_reproducibility_seed
from ._utils import relu_symmetric
from .graph import _BIN_BIAS, bin_xy, delaunay_edges, neighbor_bins, unpack_bin


# ---------- Phase 4: Hierarchical Stitching ----------


def estimate_within_cell_dz_threshold(
    df: pd.DataFrame,
    *,
    entity_col: str,
    z_col: str = "z",
    n_sample: int = 50,
    min_entity_size: int = 5,
    cohens_d_threshold: float = 3.0,
    target_percentile: float = 90.0,
    unimodal_percentile: float = 50.0,
    min_recommended_G_z: float = 1.0,
    seed: int = 42,
) -> dict:
    """Estimate a within-cell |Δz| threshold from segmented input.

    Pools pairwise |Δz| values across a sample of segmented entities,
    fits a 2-component Gaussian mixture model, tests for bimodality
    via Cohen's d on the fitted component means, and returns the
    target percentile of the **smaller-mean component** when bimodal
    (otherwise the percentile of the full distribution).

    The intent: in a noisy DAPI/Voronoi segmentation that merges
    stacked stratum cells, the within-entity |Δz| distribution is
    bimodal — a low-Δz mode (within-stratum tx pairs) and a high-Δz
    mode (cross-stratum tx pairs from the merged column). The smaller-
    mean mode reflects within-cell scale; its right tail (90 %ile by
    default) is a robust upper bound on legitimate within-cell |Δz|
    that downstream stitching can use as a Δz filter threshold.

    On clean, unimodal data (e.g. ground-truth labels), the GMM
    collapses, Cohen's d falls below ``cohens_d_threshold``, and the
    percentile of the full pooled distribution is returned instead.

    Parameters
    ----------
    df : pd.DataFrame
        Transcript-level table with at least ``entity_col`` and
        ``z_col``.
    entity_col : str
        Column whose distinct non-``"-1"`` values define entities.
    z_col : str, default ``"z"``
        Column holding the z coordinate (µm).
    n_sample : int, default 50
        Number of entities to randomly sample. If fewer eligible
        entities exist, all are used.
    min_entity_size : int, default 5
        Skip entities below this transcript count (no meaningful
        pairwise statistic).
    cohens_d_threshold : float, default 3.0
        Cohen's d cutoff between the two GMM components for
        declaring bimodality. d ≥ 3 corresponds to nearly-disjoint
        modes (the means are 3+ pooled std-deviations apart). The
        default is intentionally strict because a unimodal
        triangular distribution (e.g., within-cell |Δz| pairs from
        clean ground-truth cells) trivially splits into two GMM
        components with d ≈ 2 — strict cutoff prevents that
        spurious bimodality from misleading the threshold.
    target_percentile : float in [0, 100], default 90
        Percentile of the **smaller-mean GMM mode** to report as the
        threshold when the data is bimodal. The smaller mode is the
        within-cell distribution (cross-stratum pairs go in the larger
        mode), so its right tail is a robust upper bound on legitimate
        within-cell |Δz|.
    unimodal_percentile : float in [0, 100], default 50
        Percentile of the **full pooled distribution** to report when
        the data is unimodal (Cohen's d below cutoff). The unimodal
        case typically arises from clean segmentation — the right
        tail then includes pathologically z-elongated entities and
        isn't a reliable scale, so the median is a more robust
        within-cell-scale estimate than higher percentiles.
    min_recommended_G_z : float, default 1.0
        Floor for the ``recommended_G_z`` output (in µm). Useful when
        downstream tooling assumes integer-µm bins.
    seed : int, default 42
        Random seed for entity sampling and GMM initialization.

    Returns
    -------
    result : dict with keys
        - ``threshold`` (float, µm): the recommended |Δz| threshold
        - ``bimodal`` (bool): whether Cohen's d ≥ threshold
        - ``cohens_d`` (float): effect size between the two modes
        - ``gmm_means`` (list of 2 floats): fitted component means
        - ``gmm_stds`` (list of 2 floats): fitted component stds
        - ``gmm_weights`` (list of 2 floats): mixing proportions
        - ``smaller_mode_mean`` (float): mean of the smaller-mean mode
        - ``smaller_mode_std`` (float): std of the smaller-mean mode
        - ``smaller_mode_weight`` (float): mixing weight of that mode
        - ``n_sampled_entities`` (int)
        - ``n_pairs`` (int): total within-entity pairs pooled
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as e:
        raise ImportError(
            "estimate_within_cell_dz_threshold requires scikit-learn"
        ) from e

    rng = np.random.default_rng(seed)
    s = df[entity_col].astype(str)
    sizes = df[s != "-1"].groupby(entity_col).size()
    eligible = sizes[sizes >= int(min_entity_size)].index.tolist()
    if not eligible:
        return {
            "threshold": float("nan"), "bimodal": False, "cohens_d": 0.0,
            "gmm_means": [float("nan"), float("nan")],
            "gmm_stds":  [float("nan"), float("nan")],
            "gmm_weights": [float("nan"), float("nan")],
            "smaller_mode_mean": float("nan"),
            "smaller_mode_std":  float("nan"),
            "smaller_mode_weight": float("nan"),
            "recommended_G_z": float("nan"),
            "n_sampled_entities": 0, "n_pairs": 0,
        }

    if len(eligible) > n_sample:
        sampled = rng.choice(eligible, size=n_sample, replace=False).tolist()
    else:
        sampled = eligible

    pooled = []
    for e in sampled:
        z = df.loc[df[entity_col] == e, z_col].to_numpy(dtype=float)
        if len(z) < 2:
            continue
        ii, jj = np.triu_indices(len(z), k=1)
        pooled.append(np.abs(z[ii] - z[jj]))
    arr = np.concatenate(pooled) if pooled else np.empty(0)

    if arr.size < 10:
        return {
            "threshold": float("nan"), "bimodal": False, "cohens_d": 0.0,
            "gmm_means": [float("nan"), float("nan")],
            "gmm_stds":  [float("nan"), float("nan")],
            "gmm_weights": [float("nan"), float("nan")],
            "smaller_mode_mean": float("nan"),
            "smaller_mode_std":  float("nan"),
            "smaller_mode_weight": float("nan"),
            "recommended_G_z": float("nan"),
            "n_sampled_entities": len(sampled), "n_pairs": int(arr.size),
        }

    X = arr.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=int(seed),
                          max_iter=200, n_init=4)
    gmm.fit(X)
    means = gmm.means_.flatten()
    stds = np.sqrt(np.maximum(gmm.covariances_.flatten(), 1e-12))
    weights = gmm.weights_

    pooled_std = float(np.sqrt((stds[0] ** 2 + stds[1] ** 2) / 2))
    cohens_d = float(abs(means[0] - means[1]) / pooled_std) if pooled_std > 0 else 0.0
    bimodal = bool(cohens_d >= float(cohens_d_threshold))

    smaller_idx = int(np.argmin(means))
    if bimodal:
        # Soft-assign every pair to its most likely component, then
        # compute the percentile of pairs assigned to the smaller mode.
        resp = gmm.predict_proba(X)
        in_smaller = resp[:, smaller_idx] >= 0.5
        smaller_arr = arr[in_smaller]
        if smaller_arr.size == 0:
            smaller_arr = arr  # safety
        threshold = float(np.percentile(smaller_arr, float(target_percentile)))
    else:
        threshold = float(np.percentile(arr, float(unimodal_percentile)))

    # Recommended G_z is bimodality-aware:
    #   - unimodal: ceil(threshold), the smallest 1-µm bin still above
    #     within-cell scale. Wide enough to admit cell-spanning merges
    #     at depth=1, narrow enough to bound them.
    #   - bimodal: floor(threshold). The threshold is the smaller-mode
    #     90 %ile (within-cell upper bound); a bin smaller than that
    #     guarantees an empty-bin moat between the within-cell mode
    #     and the cross-stratum mode, which Split & Stitch can refuse
    #     to bridge at depth=1.
    if bimodal:
        recommended_G_z = float(max(float(min_recommended_G_z),
                                    np.floor(threshold)))
    else:
        recommended_G_z = float(max(float(min_recommended_G_z),
                                    np.ceil(threshold)))

    return {
        "threshold": threshold,
        "bimodal": bimodal,
        "cohens_d": cohens_d,
        "gmm_means": means.tolist(),
        "gmm_stds":  stds.tolist(),
        "gmm_weights": weights.tolist(),
        "smaller_mode_mean":   float(means[smaller_idx]),
        "smaller_mode_std":    float(stds[smaller_idx]),
        "smaller_mode_weight": float(weights[smaller_idx]),
        "recommended_G_z":     recommended_G_z,
        "n_sampled_entities":  int(len(sampled)),
        "n_pairs":             int(arr.size),
    }


def compute_within_entity_dz_stats(
    df: pd.DataFrame,
    *,
    entity_col: str,
    z_col: str = "z",
    etype_filter: tuple[str, ...] | None = ("cell",),
    min_entity_size: int = 5,
    percentiles: tuple[float, ...] = (50, 75, 90, 95, 99),
) -> dict[str, float]:
    """Pool within-entity pairwise |Δz| across all entities, return stats.

    Used to derive a data-driven Δz threshold for stitching's
    ``min_close_edges_dz`` guard: any cross-component candidate pair whose
    z-spread exceeds the within-cell scale is geometrically unlikely to
    be same-cell and can be filtered before agglomerative scoring.

    Parameters
    ----------
    df : pd.DataFrame
        Transcript-level table with at least ``entity_col`` and ``z_col``.
    entity_col : str
        Column whose distinct non-``"-1"`` values define entities.
    z_col : str, default ``"z"``
        Column holding the z coordinate (µm).
    etype_filter : tuple of {"cell", "partial", "component"} or None
        Restrict the pool to entities of these types (read from the
        ``_etype`` column when present, otherwise computed via
        :func:`tracer._etype.infer_etype_from_label`). Pass ``None`` to
        include all non-``"-1"`` entities. Default ``("cell",)`` — cells
        are the most representative reference scale.
    min_entity_size : int
        Skip entities with fewer than this many transcripts (no
        pairwise statistic).
    percentiles : tuple of float
        Percentiles to report alongside the median. Values in [0, 100].

    Returns
    -------
    stats : dict
        Keys: ``n_entities`` (int), ``n_pairs`` (int), ``median`` (float),
        ``mean`` (float), ``max`` (float), and one entry per requested
        percentile, e.g. ``"p75"``. All distances in same units as
        ``z_col`` (typically µm). Returns NaN-filled dict if no data.
    """
    from ._etype import infer_etype_from_label

    s = df[entity_col].astype(str)
    keep = s != "-1"
    if etype_filter is not None:
        # Prefer the upstream-emitted _etype column when present
        # (correct on FFPE cell_ids). Fall back to the vectorized
        # label parser for back-compat on input frames without _etype.
        if "_etype" in df.columns:
            types = df["_etype"].astype(str)
        else:
            types = pd.Series(
                np.asarray(infer_etype_from_label(s)).astype(str),
                index=s.index,
            )
        keep = keep & types.isin(etype_filter)
    sub = df[keep]
    pooled: list[np.ndarray] = []
    n_kept = 0
    for _, g in sub.groupby(entity_col, sort=False):
        if len(g) < max(2, int(min_entity_size)):
            continue
        z = g[z_col].to_numpy(dtype=float)
        ii, jj = np.triu_indices(len(z), k=1)
        pooled.append(np.abs(z[ii] - z[jj]))
        n_kept += 1
    if not pooled:
        out = {"n_entities": 0, "n_pairs": 0,
               "median": float("nan"), "mean": float("nan"),
               "max": float("nan")}
        for p in percentiles:
            out[f"p{int(p)}"] = float("nan")
        return out
    arr = np.concatenate(pooled)
    out = {
        "n_entities": int(n_kept),
        "n_pairs": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }
    for p in percentiles:
        out[f"p{int(p)}"] = float(np.percentile(arr, p))
    return out


def build_entity_table(
    df_final: pd.DataFrame,
    *,
    entity_col: str,
    gene_col: str = "feature_name",
    coord_cols=("x", "y", "z"),
):
    """
    Build per-entity summary:
      - centroid (x,y,z)
      - unique genes list
      - type: cell/partial/component
    """
    # Read-only view of the two columns we need — no full-df copy.
    # Previously called `.astype(str).str.strip()` on `df_final[gene_col]`
    # which forced an O(100M)-row Python string op; assume gene names
    # arrive normalised (use `prepare_transcript_df` upstream).
    ent = df_final[entity_col].astype(str)
    keep = ent.notna() & (ent != "DROP") & (ent != "nan")

    # Slice to the keep rows. `.loc` is a view when the mask is boolean.
    df = df_final.loc[keep, [entity_col, gene_col, *coord_cols]].copy()
    df[entity_col] = df[entity_col].astype(str)

    # entity type — prefer the upstream-emitted `_etype` column when
    # present (correct on Xenium FFPE / IO cell_ids). Fall back to the
    # canonical vectorized label parser for back-compat on input frames
    # without _etype.
    if "_etype" in df_final.columns:
        df["_etype"] = df_final.loc[keep, "_etype"].astype(str).to_numpy()
    else:
        from ._etype import infer_etype_from_label
        df["_etype"] = np.asarray(
            infer_etype_from_label(df[entity_col])
        ).astype(str)
    df = df[df["_etype"].isin(["cell", "partial", "component"])]

    # centroid (`observed=True` avoids processing empty categorical groups
    # when entity_col is categorical).
    grouped_coords = df.groupby(entity_col, sort=True, observed=True)[list(coord_cols)]
    cent = grouped_coords.mean()
    # Per-axis min/max — used by the spatial centroid-in-bbox gate at
    # Stitch time. Cheap O(N_tx) extra pass.
    bbox_min = grouped_coords.min().rename(columns={c: f"{c}_min" for c in coord_cols})
    bbox_max = grouped_coords.max().rename(columns={c: f"{c}_max" for c in coord_cols})

    # unique genes per entity (sorted for deterministic downstream mapping).
    # Pre-cast gene_col to plain str ONCE so the per-group lambda doesn't
    # have to convert from Categorical inside `.apply()` — previously that
    # was ~6 s cumulative across 103K invocations on the densest PDAC
    # sub-tile (per cProfile). After pre-cast, per-group work reduces to
    # a single np.sort on an object array (no element-wise conversion).
    if isinstance(df[gene_col].dtype, pd.CategoricalDtype):
        df[gene_col] = df[gene_col].astype(str)
    genes = df.groupby(entity_col, sort=True, observed=True)[gene_col].unique()
    genes = genes.apply(np.sort)

    etype = df.groupby(entity_col, observed=True)["_etype"].first()

    # Per-entity tx count — used as the "size" in the asymmetric
    # smaller-inside-larger spatial test.
    n_tx = df.groupby(entity_col, observed=True)[gene_col].size().rename("n_tx")

    summary = (
        cent.join(bbox_min).join(bbox_max)
            .join(genes.rename("genes"))
            .join(etype.rename("etype"))
            .join(n_tx)
    )
    summary = summary.reset_index().rename(columns={entity_col: "entity_id"})
    return summary


# -------------------------------------------
# Coherence C(gene-set) using NPMI
# -------------------------------------------

_VALID_COHERENCE_MODES = ("count", "magnitude")


def _slice_npmi_submatrix(npmi_mat, gene_ids):
    """Return a dense float submatrix for the given gene indices.

    Handles both dense ``np.ndarray`` and ``scipy.sparse`` inputs. For
    sparse inputs we slice to a sparse submatrix and densify only the
    small per-entity block; absent entries become exact zeros — by
    design (see :func:`compute_pmi_bootstrap` docs).
    """
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover
        sparse = None

    if sparse is not None and sparse.issparse(npmi_mat):
        # Slice rows then cols (CSR/CSC). The bootstrap stores only the
        # upper triangle of W_sparse, but `gene_ids` may reorder genes
        # so the upper triangle of `sub` no longer covers the same cells
        # as the upper triangle of the original. Symmetrise via sub+sub.T
        # — exactly one of (sub[a,b], sub[b,a]) is nonzero by
        # construction, so the sum is just the value, not doubled.
        sub = npmi_mat[gene_ids, :][:, gene_ids]
        dense = np.asarray(sub.todense())
        return dense + dense.T
    return npmi_mat[np.ix_(gene_ids, gene_ids)]


def coherence(
    gene_ids: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    mode: str = "count",
    threshold: float = 0.05,
    metric: str = "npmi",
) -> tuple[float, float, float]:
    """Unified coherence — returns ``(C, purity, conflict)``.

    The function operates on the values stored in ``npmi_mat`` and is
    metric-agnostic in its math. The ``metric`` kwarg is purely an
    advisory parameter that:
      (a) validates the caller's intent against the chosen ``mode``, and
      (b) documents the threshold's interpretation.

    Parameters
    ----------
    gene_ids : np.ndarray
        Indices into ``npmi_mat`` for the gene set under consideration.
    npmi_mat : np.ndarray or scipy.sparse
        Square matrix of pairwise association values. Caller's
        responsibility to ensure entries are NPMI (bounded [-1,+1]) or
        PMI (unbounded log-fold-enrichment) consistent with ``metric``.
    mode : {"count", "magnitude"}
        ``"count"`` — purity = #(w > threshold) / |P|;
        conflict = #(w < -threshold) / |P|. Threshold-based fraction.

        ``"magnitude"`` — purity = Σmax(w, 0) / Σ|w|;
        conflict = Σmax(-w, 0) / Σ|w|. **Only valid with metric="npmi"**
        because PMI's unbounded magnitude lets a single rare-strong
        pair dominate the sum.
    threshold : float
        Dead-zone threshold τ. Used directly in ``"count"`` mode. The
        natural calibration depends on ``metric``: NPMI thresholds are
        typically in [0.01, 0.1]; PMI thresholds reflect log-fold
        enrichment (e.g., 0.4 ≈ "50% above independence").
    metric : {"npmi", "pmi"}
        Advisory parameter naming the metric in ``npmi_mat``. Raises
        ``ValueError`` if ``metric="pmi"`` is paired with
        ``mode="magnitude"``.

    Returns
    -------
    C : float
        ``purity - conflict``.
    purity : float
    conflict : float
    """
    k = int(gene_ids.size)
    if k < 2:
        return 0.0, 0.0, 0.0
    if mode not in _VALID_COHERENCE_MODES:
        raise ValueError(
            f"mode must be one of {_VALID_COHERENCE_MODES!r} (got {mode!r})"
        )
    if metric not in ("npmi", "pmi"):
        raise ValueError(f"metric must be 'npmi' or 'pmi' (got {metric!r})")
    if metric == "pmi" and mode == "magnitude":
        raise ValueError(
            "metric='pmi' is incompatible with mode='magnitude' — PMI's "
            "unbounded magnitude lets rare-strong pairs dominate the sum. "
            "Use metric='pmi' with mode='count' instead."
        )

    # Fast path: count-mode + dense float32 W → Cython kernel.
    # ~5-10× per-call speedup vs numpy at ROI/full scale.
    if mode == "count":
        try:
            import scipy.sparse as _sp
            if not _sp.issparse(npmi_mat) and isinstance(npmi_mat, np.ndarray) \
               and npmi_mat.dtype == np.float32:
                from . import _cy_prune
                gids32 = np.ascontiguousarray(gene_ids, dtype=np.int32)
                C, purity, conflict = _cy_prune.coherence_count_kernel(
                    gids32, npmi_mat, float(threshold)
                )
                return float(C), float(purity), float(conflict)
        except (ImportError, AttributeError):
            pass  # fall through to numpy path

    sub = _slice_npmi_submatrix(npmi_mat, gene_ids)
    iu = np.triu_indices(k, k=1)
    vals = sub[iu]
    vals = vals[np.isfinite(vals)]
    P = vals.size
    if P == 0:
        return 0.0, 0.0, 0.0

    if mode == "count":
        purity = float(np.sum(vals > threshold)) / P
        conflict = float(np.sum(vals < -threshold)) / P
    else:  # magnitude
        denom = float(np.sum(np.abs(vals)))
        if denom == 0.0:
            return 0.0, 0.0, 0.0
        purity = float(np.sum(np.maximum(vals, 0.0))) / denom
        conflict = float(np.sum(np.maximum(-vals, 0.0))) / denom

    return float(purity - conflict), float(purity), float(conflict)


def signal_strength(gene_ids: np.ndarray, npmi_mat: np.ndarray) -> float:
    """``S(G) = Σ|w_ij|`` over finite (i, j) pairs (manuscript Eq 22).

    Diagnostic — not folded into ΔC. Returns 0.0 for sets of <2 genes
    or sets with no observed pairs.
    """
    k = int(gene_ids.size)
    if k < 2:
        return 0.0
    sub = _slice_npmi_submatrix(npmi_mat, gene_ids)
    iu = np.triu_indices(k, k=1)
    vals = sub[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.sum(np.abs(vals)))


def deltaC(
    genes_u: np.ndarray,
    genes_v: np.ndarray,
    npmi_mat: np.ndarray,
    *,
    mode: str = "count",
    threshold: float = 0.05,
    penalize_simplicity: bool = True,
    metric: str = "npmi",
) -> float:
    """Unified ΔC across coherence modes.

    Without ``penalize_simplicity``::

        ΔC = C(union) - max(C(u), C(v))

    With ``penalize_simplicity`` (default), each per-cluster C is
    adjusted by ``-1/n`` and the union by ``-1/(n_u + n_v)`` so a
    larger merged set must produce strictly higher coherence to win
    over the simpler-to-explain split.

    ``metric`` is forwarded to :func:`coherence`; see its docstring.
    """
    C_u, _, _ = coherence(genes_u, npmi_mat, mode=mode, threshold=threshold, metric=metric)
    C_v, _, _ = coherence(genes_v, npmi_mat, mode=mode, threshold=threshold, metric=metric)
    union = np.unique(np.concatenate([genes_u, genes_v]))
    C_union, _, _ = coherence(union, npmi_mat, mode=mode, threshold=threshold, metric=metric)

    if not penalize_simplicity:
        return float(C_union - max(C_u, C_v))

    nu = max(int(genes_u.size), 1)
    nv = max(int(genes_v.size), 1)
    n_union = nu + nv
    C_sep = max(C_u - 1.0 / nu, C_v - 1.0 / nv)
    return float(C_union - (1.0 / n_union) - C_sep)


def compute_housekeeping_mask(
    W,
    *,
    pos_thresh: float = 0.05,
    neg_thresh: float = -0.05,
    min_strong_count: int = 5,
) -> np.ndarray:
    """Bool array of length ``G``. ``True`` = keep gene, ``False`` = drop.

    A gene is flagged as housekeeping if it has fewer than
    ``min_strong_count`` strong-positive (NPMI > ``pos_thresh``) AND
    fewer than ``min_strong_count`` strong-negative (NPMI < ``neg_thresh``)
    neighbors. The diagonal is ignored. NaN entries don't count toward
    either tally. Accepts dense or sparse ``W``.
    """
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover
        sparse = None

    if W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")
    G = int(W.shape[0])
    if G == 0:
        return np.empty((0,), dtype=bool)

    if sparse is not None and sparse.issparse(W):
        # The bootstrap CSR stores only the upper triangle.
        # Symmetrise virtually by counting both rows and columns.
        Wcsr = W.tocsr().astype(np.float32)
        # Boolean masks as sparse — diagonal entries shouldn't be stored
        # (PmiBootstrapResult never stores i==j) but be defensive.
        Wcsr.setdiag(0.0)
        Wcsr.eliminate_zeros()
        pos_mask = (Wcsr > pos_thresh)
        neg_mask = (Wcsr < neg_thresh)
        # Counts: per-row + per-column (since only upper-tri stored,
        # row i counts the j>i neighbors and col i counts the j<i ones).
        pos_counts = (
            np.asarray(pos_mask.sum(axis=1)).ravel()
            + np.asarray(pos_mask.sum(axis=0)).ravel()
        )
        neg_counts = (
            np.asarray(neg_mask.sum(axis=1)).ravel()
            + np.asarray(neg_mask.sum(axis=0)).ravel()
        )
    else:
        W_arr = np.asarray(W, dtype=np.float32)
        diag_mask = np.eye(G, dtype=bool)
        pos = (W_arr > pos_thresh) & ~diag_mask
        neg = (W_arr < neg_thresh) & ~diag_mask
        pos_counts = pos.sum(axis=1)
        neg_counts = neg.sum(axis=1)

    return (pos_counts >= min_strong_count) | (neg_counts >= min_strong_count)


# ----------------------------
# Union-Find (Disjoint Set Union)
# ----------------------------
class DSU:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


# --------------------------------------
# Constrained hierarchical ΔC stitching
# --------------------------------------
_LEGACY_STITCH_KWARG_SENTINEL = object()


# Diagnostic counters populated by stitch_entities_hierarchical when the
# spatial-centroid bypass gate is active. Reset at the start of each
# call. Read by callers after the call returns (e.g. CLI / sweep tooling
# that wants to log how many pairs the gate captured).
_LAST_GATE_STATS: dict[str, int] = {}
_LAST_STITCH_PHASE_TIMINGS: dict[str, float] = {}


def _stitch_entities_hierarchical_decomposable(
    summary_df: pd.DataFrame,
    aux: dict,
    *,
    mode: str = "count",
    threshold: float = 0.05,
    metric: str = "npmi",
    penalize_simplicity=True,
    deltaC_min=0.0,
    c_union_bypass: float | None = None,
    c_union_bypass_max_n_tx: int | None = None,
    max_merger_depth: int | None = None,
    use_3d=True,
    dist_threshold: float | None = None,
    candidate_source: str = "delaunay",
    G: float | None = None,
    stitch_neighborhood: str = "8",
    G_z: float | None = None,
    z_neighbor_depth: int = 0,
    transcript_coords: np.ndarray | None = None,
    transcript_entity_codes: np.ndarray | None = None,
    min_candidate_edges: int | str = 0,
    # Optional per-entity-witness count: drop candidate pair (E1, E2)
    # unless EACH entity contributes at least `min_local_tx_per_entity`
    # unique tx in the shared bin neighborhood (xy 8-Moore + ±depth z
    # bins). Catches single-bridging-tx candidates that sneak through
    # the diagonal-Moore reach (~5.66 µm at G=2). Symmetric in (E1, E2)
    # — resistant to mass-dominated cross-product counts.
    # Default 0 = off (current behavior unchanged).
    min_local_tx_per_entity: int = 0,
    max_pair_median_dz: float | None = None,
    min_close_edges_dz: float | None = None,
    min_close_edges_n: int = 0,
):
    """**EXPERIMENTAL — opt-in via `use_decomposable_stitch=True`.**

    Lazy DSU + max-heap greedy with decomposable coherence primitives.
    Algorithmic complexity: O(M log N) instead of the eager path's
    O(rounds × candidate_pairs). Designed for tissue-scale (200k+
    entities) where the eager path becomes the dominant runtime.

    Strategy summary (full design + math validation in
    `/Users/adeshpa6/1_Projects/01.10_Lab/GENESIS/TODO.md`):
      1. Pre-compute per-original `(n_above, n_below, n_finite)` and
         per-spatial-pair cross primitives.
      2. DSU groups + max-heap of candidate-pair ΔC values.
      3. On heap pop: check DSU root staleness; if stale, recompute
         from current primitives and reinsert.
      4. On merge: combine running primitive sums + cross-sums to all
         neighbour groups; push fresh ΔC entries for new candidate
         pairs to the heap.

    Bit-match expectation:
      Per-call ΔC values are bit-equivalent to the eager path
      (validated on 1000 µm ROI: 71k calls, 0 mismatches). The merge
      sequence may differ on exact ties due to FP rounding in the
      cross-segment arithmetic, but final entity-to-stitched parity
      matches the eager output to within ~0.001 ARI in practice.

    Implementation strategy:
      1. Reuse the eager path's setup (candidate-pair build, filters,
         centroids, gene-id mapping) — call `stitch_entities_hierarchical`
         in a flag-disambiguated mode that returns just the prepared
         state. Implementation here re-does the prep inline to avoid
         a refactor of the existing function.
      2. Maintain per-DSU-root primitive sums (n_above, n_below,
         n_finite) accumulated across all merges in that root.
      3. For each candidate pair: compute ΔC by combining roots'
         current primitive sums plus a fresh cross-segment computation
         for the merge boundary. No re-iteration of the union's full
         gene-pair set.
      4. On merge: update primitive sums by adding the cross
         contribution; gene set is the union of the two roots.

    The cross computation uses the 6-segment decomposition validated
    in `/tmp/validate_decomp_coh.py`:
        triu(A∪B) = triu(A−B) + triu(B−A) + triu(A∩B)
                   + cross(A−B, B−A) + cross(A−B, A∩B) + cross(B−A, A∩B)

    Implementation is integrated into `stitch_entities_hierarchical`
    directly (see the `if use_decomposable_stitch …` branches inside
    `C_of_root`, `compute_deltaC_roots`, and the merge step). This
    helper is a thin wrapper that simply forwards with the flag set.
    """
    return stitch_entities_hierarchical(
        summary_df=summary_df, aux=aux,
        mode=mode, threshold=threshold, metric=metric,
        penalize_simplicity=penalize_simplicity, deltaC_min=deltaC_min,
        c_union_bypass=c_union_bypass,
        c_union_bypass_max_n_tx=c_union_bypass_max_n_tx,
        max_merger_depth=max_merger_depth,
        use_3d=use_3d, dist_threshold=dist_threshold,
        candidate_source=candidate_source, G=G,
        stitch_neighborhood=stitch_neighborhood,
        G_z=G_z, z_neighbor_depth=z_neighbor_depth,
        transcript_coords=transcript_coords,
        transcript_entity_codes=transcript_entity_codes,
        min_candidate_edges=min_candidate_edges,
        min_local_tx_per_entity=min_local_tx_per_entity,
        max_pair_median_dz=max_pair_median_dz,
        min_close_edges_dz=min_close_edges_dz,
        min_close_edges_n=min_close_edges_n,
        use_decomposable_stitch=True,  # actually invoke the primitive path
    )


def stitch_entities_hierarchical(
    summary_df: pd.DataFrame,
    aux: dict,
    *,
    mode: str = "count",
    threshold: float = 0.05,
    metric: str = "npmi",
    penalize_simplicity=True,
    deltaC_min=0.0,
    # Optional acceptance bypass: when set, a pair that fails ΔC ≥
    # deltaC_min is still accepted if the raw post-merge coherence
    # C(union) ≥ c_union_bypass. Spatial-witness / candidate-source
    # gates still apply. Designed to recover same-program fragment
    # absorptions where both parents are already at C ≈ 1.0 and ΔC
    # has no headroom regardless of how perfect the union is.
    # None = off (legacy behavior). Recommended 0.9 when enabled.
    c_union_bypass: float | None = None,
    # When set, the C(union) bypass only applies if the merged entity's
    # total tx count is at or below this threshold. Recovers small
    # within-cell fragment consolidations (where ΔC has no headroom
    # because both parents are at C ≈ 1.0) while requiring the strong
    # ΔC signal for large mergers (where the bypass risks bridging
    # cross-cell compartments). None = no size cap on the bypass.
    c_union_bypass_max_n_tx: int | None = None,
    # Optional cap on per-component merger-tree depth (height of the
    # binary tree built by greedy merging, leaves = pre-stitch entities,
    # internal nodes = merge events). Each DSU root carries
    # `depth = max(child_depth) + 1`; a merger is blocked when either
    # side has already reached the cap. Balanced N-entity merges cost
    # log2(N) depth; chain merges cost N-1 — so the cap intrinsically
    # rewards balanced consolidations and penalises one-component-
    # repeatedly-absorbing-neighbours growth. None = off (legacy).
    # Recommended 3 when enabled (allows up to 8 balanced entities or
    # 4 chain entities per stitched component).
    max_merger_depth: int | None = None,
    use_3d=True,
    dist_threshold: float | None = None,
    candidate_source: str = "delaunay",
    G: float | None = None,
    stitch_neighborhood: str = "8",
    G_z: float | None = None,
    z_neighbor_depth: int = 0,
    transcript_coords: np.ndarray | None = None,
    transcript_entity_codes: np.ndarray | None = None,
    min_candidate_edges: int | str = 0,
    # Optional per-entity-witness count: drop candidate pair (E1, E2)
    # unless EACH entity contributes at least `min_local_tx_per_entity`
    # unique tx in the shared bin neighborhood (xy 8-Moore + ±depth z
    # bins). Catches single-bridging-tx candidates that sneak through
    # the diagonal-Moore reach (~5.66 µm at G=2). Symmetric in (E1, E2)
    # — resistant to mass-dominated cross-product counts.
    # Default 0 = off (current behavior unchanged).
    min_local_tx_per_entity: int = 0,
    max_pair_median_dz: float | None = None,
    min_close_edges_dz: float | None = None,
    min_close_edges_n: int = 0,
    # Deprecated kwargs — translated to mode/threshold below.
    purity_threshold=_LEGACY_STITCH_KWARG_SENTINEL,
    tau=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relu=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relative=_LEGACY_STITCH_KWARG_SENTINEL,
    # Experimental: lazy DSU+max-heap merge with decomposable coherence
    # primitives. Validated bit-match on 1000 µm ROI (71k coh calls,
    # 894 merges, 0 mismatches) but not yet on full-tissue scale.
    # Default False (use the existing eager-recompute greedy).
    use_decomposable_stitch: bool = False,
    # Experimental: top-K positive-clique fast-gate for candidate pairs
    # at heap-init. For each entity, precompute its K signature genes
    # (highest sum of positive PMI to others in same entity). For each
    # candidate pair (i, j), scan the K×K cross-PMI block: if ANY entry
    # is < neg_npmi_threshold, REJECT the pair without computing its
    # full ΔC. Cuts heap-init Python-loop overhead by skipping the
    # majority of cell-pair candidates (most are biologically
    # incompatible). 0 = disabled (no behavior change). Recommended
    # K = 3-5 for empirical bit-match.
    fast_gate_top_k: int = 0,
    fast_gate_mean_threshold: float = 0.0,
    # Optional: per-entity tx counts. Used when multi-partial mergers
    # tie on suffix → majority-tx-count winner gets the merged label.
    # If None, falls back to lexicographic tiebreak.
    entity_n_tx: dict[str, int] | None = None,
    # Optional spatial centroid-in-bbox bypass at merge time. When True,
    # candidate pairs where the SMALLER entity's centroid lies inside
    # the LARGER entity's per-axis tx-coord range are MERGED without
    # PMI evaluation (positive override / Tier-1 in the 3-tier cascade).
    # Default False (no spatial bypass; standard ΔC-driven merging).
    spatial_centroid_gate: bool = False,
    # Tightness of the spatial-overlap test. K=1 → bbox check; K≥2 →
    # require K tx coords above AND K below smaller's centroid per
    # axis. Higher K → stricter (more interior).
    spatial_centroid_k: int = 1,
    # Per-entity tx-coord arrays. Required for K≥2.
    # dict[entity_id -> (n_tx, n_dim) ndarray].
    entity_tx_coords: dict | None = None,
    # Spatial gate mode (only when spatial_centroid_gate=True):
    #   "pre"  — current behavior: spatial bypass returns sentinel ΔC,
    #            so spatial-overlap pairs MERGE FIRST regardless of ΔC.
    #            Spatial overrides any ΔC verdict, including rejections.
    #   "post" — spatial gate fires only as a tiebreaker on ΔC-rejected
    #            pairs. ΔC takes priority for accepting and ranking; if
    #            a pair fails the ΔC test (dc < deltaC_min), THEN check
    #            the spatial gate; if centroids match, merge anyway.
    #            More conservative — lets ΔC do its job, only uses
    #            spatial as a fallback for marginal-but-co-located pairs.
    spatial_gate_mode: str = "pre",
    # Flipped spatial test: instead of "smaller's centroid inside larger's
    # tx cloud", check "larger's centroid inside smaller's tx cloud".
    # Effective K is dynamically capped at floor(n_smaller / 3) so small
    # partials use a lighter K. This is the right test for detecting
    # whether the cell's tx are arranged AROUND the partial (i.e., the
    # partial is a real fragment of the cell), as opposed to the partial
    # being embedded INSIDE the cell (the contamination case).
    spatial_gate_flipped: bool = False,
    # Optional Mahalanobis-D RESCUE on borderline-ΔC pairs. When set,
    # the loop OVERRIDES a ΔC reject when
    #     rescue_delta_c_floor < ΔC < 0    AND    D ≤ mahalanobis_d_rescue
    # where D = sqrt((μ_A − μ_B)^T Σ_pooled^-1 (μ_A − μ_B)) over the
    # two entities' tx coords. Singular Σ / n<2 cases gracefully no-op
    # (no rescue). None = off (default). Requires `entity_tx_coords`
    # (or the function will silently disable). See module docstring.
    mahalanobis_d_rescue: float | None = None,
    rescue_delta_c_floor: float = -0.2,
):
    """Hierarchical entity stitching driven by ΔC.

    The optional ``min_candidate_edges`` kwarg filters candidate pairs
    by the number of supporting transcript-level cross-bin connections.
    A pair (A, B) is admitted only when at least
    ``min_candidate_edges`` transcript pairs (tx_a in A, tx_b in B) lie
    in candidate bin neighborhoods. Pass an integer for a fixed
    threshold or the string ``"min"`` for a per-pair adaptive
    threshold of ``min(n_A, n_B)`` where n_X is the entity tx count.
    Only meaningful when ``candidate_source='grid'``.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Required columns: ``entity_id``, ``x``, ``y``, ``z`` (or just
        ``x``, ``y`` if ``use_3d=False``), ``genes`` (np.ndarray[str]),
        ``etype`` in ``{'cell', 'partial', 'component'}``.
    aux : dict
        Must contain ``"W"`` (NPMI matrix) and ``"gene_to_idx"``. May
        contain ``"housekeeping_mask"`` (bool array of length G); when
        present, gene indices flagged ``False`` are removed from each
        entity's gene set before ΔC is computed.
    mode : {"count", "magnitude"}
        Coherence semantics. See :func:`coherence`.
    threshold : float
        Dead-zone threshold τ used by :func:`coherence` / :func:`deltaC`.
    penalize_simplicity : bool
        If True, ΔC penalizes smaller gene sets; see :func:`deltaC`.
    deltaC_min : float
        Minimum ΔC required to merge two clusters.
    use_3d : bool
        Use 3D or 2D coordinates for centroid distance.

    Other Parameters
    ----------------
    purity_threshold, tau, use_relu, use_relative : deprecated
        Legacy kwargs from before the coherence consolidation. Passing
        any of them emits ``DeprecationWarning`` and translates to
        ``mode``/``threshold``. See release notes for the behavior
        shift.

    Returns
    -------
    entity_to_stitched : dict[str, str]
    info : dict
        Cluster bookkeeping; currently just ``{"root_to_label": ...}``.
    """
    legacy_passed = {
        name: value
        for name, value in (
            ("purity_threshold", purity_threshold),
            ("tau", tau),
            ("use_relu", use_relu),
            ("use_relative", use_relative),
        )
        if value is not _LEGACY_STITCH_KWARG_SENTINEL
    }
    if legacy_passed:
        warnings.warn(
            "stitch_entities_hierarchical: legacy kwargs "
            f"{sorted(legacy_passed)} are deprecated; pass mode='count'|"
            "'magnitude' and threshold instead. Translating with the same "
            "behavior shift as the coherence wrappers; see release notes.",
            DeprecationWarning,
            stacklevel=2,
        )
        eff_use_relu = legacy_passed.get("use_relu", True)
        eff_use_relative = legacy_passed.get("use_relative", False)
        eff_tau = legacy_passed.get("tau", _LEGACY_STITCH_KWARG_SENTINEL)
        eff_pt = legacy_passed.get("purity_threshold", _LEGACY_STITCH_KWARG_SENTINEL)

        if not eff_use_relu:
            mode = "count"
        elif eff_use_relative:
            mode = "magnitude"
        else:
            mode = "count"

        eff_tau_set = eff_tau is not _LEGACY_STITCH_KWARG_SENTINEL
        eff_pt_set = eff_pt is not _LEGACY_STITCH_KWARG_SENTINEL
        if eff_tau_set and eff_pt_set and eff_tau != eff_pt:
            warnings.warn(
                "stitch_entities_hierarchical: both tau and purity_threshold "
                f"passed with different values ({eff_tau!r} vs {eff_pt!r}); "
                "using tau.",
                DeprecationWarning,
                stacklevel=2,
            )
        if eff_tau_set:
            threshold = eff_tau
        elif eff_pt_set:
            threshold = eff_pt

    # When use_decomposable_stitch=True, the merge loop below uses the
    # `_compute_deltaC_via_primitives` helper instead of recomputing
    # coherence(union) from scratch on every ΔC eval. All other setup
    # (candidate-pair build, filters, DSU, max-heap, lazy stale-pop)
    # is shared with the eager path. See `_stitch_entities_hierarchical_decomposable`
    # docstring for the algorithm rationale + bit-match expectation.
    npmi_mat = aux["W"]
    gene_to_idx = aux["gene_to_idx"]
    housekeeping_mask = aux.get("housekeeping_mask")

    # map entity -> gene indices
    entity_ids = summary_df["entity_id"].astype(str).to_numpy()
    etypes = summary_df["etype"].astype(str).to_numpy()

    gene_id_lists = []
    for genes in summary_df["genes"].values:
        g = pd.Index(np.asarray(genes, dtype=str)).map(gene_to_idx)
        g = np.sort(g[~pd.isna(g)].astype(int).unique())
        g = np.asarray(g, dtype=np.int32)
        if housekeeping_mask is not None and g.size > 0:
            g = g[housekeeping_mask[g]]
        gene_id_lists.append(g)

    # points
    if use_3d:
        pts = summary_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    else:
        pts = summary_df[["x", "y"]].to_numpy(dtype=np.float64)

    N = len(entity_ids)
    if N == 0:
        # No entities to stitch (e.g. a sparse tile where Group/Rescue
        # anchored nothing). Empty mapping → caller leaves labels as-is.
        return {}, {}
    if N == 1:
        return {entity_ids[0]: entity_ids[0]}, {}

    # Diagnostic phase-timing (visible via _LAST_STITCH_PHASE_TIMINGS
    # after the call). Negligible overhead; flat dict, ~0.3 µs/write.
    import time as _stitch_t
    _phase_t0 = _stitch_t.time()
    _phase_timings: dict[str, float] = {}

    def _phase(name: str) -> None:
        nonlocal _phase_t0
        now = _stitch_t.time()
        _phase_timings[name] = round(now - _phase_t0, 3)
        _phase_t0 = now

    # ----------------------------------------------------------------
    # Candidate edge enumeration: Delaunay over centroids OR bin-grid
    # ----------------------------------------------------------------
    if candidate_source not in ("delaunay", "grid"):
        raise ValueError(
            f"candidate_source must be 'delaunay' or 'grid' (got {candidate_source!r})"
        )

    adj: list[list[int]] | None = None

    if candidate_source == "delaunay":
        # Delaunay edges (use SciPy by default)
        edges = delaunay_edges(pts)

        # Optionally filter edges by geometric length to reduce candidate merges
        if dist_threshold is not None:
            if len(edges) > 0:
                ei = np.asarray(edges, dtype=np.int64)
                p0 = pts[ei[:, 0]]
                p1 = pts[ei[:, 1]]
                dists = np.linalg.norm(p0 - p1, axis=1)
                keep = dists <= float(dist_threshold)
                edges = [tuple(x) for x in ei[keep]]

        # adjacency on original nodes
        adj = [[] for _ in range(N)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

    else:  # candidate_source == "grid"
        if G is None:
            raise ValueError("G must be provided when candidate_source='grid'")
        if stitch_neighborhood not in ("0", "4", "8") and not (
            isinstance(stitch_neighborhood, str)
            and stitch_neighborhood.startswith("R")
            and stitch_neighborhood[1:].isdigit()
            and int(stitch_neighborhood[1:]) >= 1
        ):
            raise ValueError(
                f"stitch_neighborhood must be '0', '4', '8', or 'R<N>' "
                f"(got {stitch_neighborhood!r})"
            )
        if z_neighbor_depth < 0:
            raise ValueError(f"z_neighbor_depth must be ≥ 0 (got {z_neighbor_depth})")
        if z_neighbor_depth > 0 and G_z is None:
            raise ValueError(
                "z_neighbor_depth > 0 requires G_z to be set"
            )
        if transcript_coords is None or transcript_entity_codes is None:
            raise ValueError(
                "transcript_coords and transcript_entity_codes must be "
                "provided when candidate_source='grid'"
            )
        if transcript_coords.shape[0] != transcript_entity_codes.shape[0]:
            raise ValueError(
                "transcript_coords and transcript_entity_codes must have "
                "equal length"
            )

        # Map transcripts to (bin_key, entity_idx). Skip transcripts whose
        # entity_idx is < 0 (e.g., DROP / unmapped labels).
        # Bin keys are either int64 (xy-only, packed) or (xy_int64, bz_int)
        # tuples (xyz). Tuple keys cost a small dict-overhead penalty but
        # the entity counts at stitch time are moderate.
        ec = np.asarray(transcript_entity_codes, dtype=np.int64)
        valid = ec >= 0
        xy_keys = bin_xy(transcript_coords[:, :2], G)[valid]
        comp_codes = ec[valid]
        if G_z is not None:
            if transcript_coords.shape[1] < 3:
                raise ValueError(
                    "G_z requires transcript_coords to have a z column"
                )
            bz_arr = np.floor(
                transcript_coords[valid, 2] / float(G_z)
            ).astype(np.int64)
            bin_keys = list(zip(xy_keys.tolist(), bz_arr.tolist()))
        else:
            bin_keys = xy_keys.tolist()

        bin_to_comps = defaultdict(set)
        # Per-(bin, entity) tx counts so we can weight candidate edges
        # by the supporting tx-tx pair count for the optional
        # min_candidate_edges filter. Same memory order as bin_to_comps.
        bin_to_comp_counts: dict = defaultdict(lambda: defaultdict(int))
        for bk, c in zip(bin_keys, comp_codes.tolist()):
            bin_to_comps[bk].add(c)
            bin_to_comp_counts[bk][c] += 1
        # Total tx per entity (for min_candidate_edges='min' mode)
        entity_tx_total: dict[int, int] = defaultdict(int)
        for c in comp_codes.tolist():
            entity_tx_total[c] += 1

        # Half-neighborhood directions in xy. "0" → empty (same-bin pairs only).
        # "4" / "8" — orthogonal-only / full Moore-1 (24 → 8 raw offsets / 4 half).
        # "R<N>" — Moore-N: every (dx, dy) with max(|dx|,|dy|) in 1..N, restricted
        # to the upper half-plane (dy > 0 OR (dy == 0 AND dx > 0)) so each
        # unordered bin-pair appears once. R1 ≡ "8".
        if stitch_neighborhood == "0":
            xy_half_offsets: tuple[tuple[int, int], ...] = ()
        elif stitch_neighborhood == "4":
            xy_half_offsets = ((0, 1), (1, 0))
        elif stitch_neighborhood == "8":
            xy_half_offsets = ((0, 1), (1, -1), (1, 0), (1, 1))
        elif (isinstance(stitch_neighborhood, str)
                and stitch_neighborhood.startswith("R")):
            try:
                R = int(stitch_neighborhood[1:])
            except ValueError as exc:
                raise ValueError(
                    f"stitch_neighborhood='{stitch_neighborhood}' is not a "
                    f"valid 'R<N>' specifier (e.g. 'R2', 'R3')."
                ) from exc
            if R < 1:
                raise ValueError(
                    f"stitch_neighborhood='R{R}' must have R>=1; use '0' "
                    f"for same-bin-only."
                )
            _half = []
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    if dx == 0 and dy == 0:
                        continue
                    if dy > 0 or (dy == 0 and dx > 0):
                        _half.append((dx, dy))
            xy_half_offsets = tuple(_half)
        else:
            raise ValueError(
                f"stitch_neighborhood must be one of '0', '4', '8', or "
                f"'R<N>' (got {stitch_neighborhood!r})."
            )

        # z-offsets for the candidate-enumeration window. We use a half-
        # window in z to avoid enumerating each unordered (bin_a, bin_b)
        # pair twice: positive dz only, plus dz=0 for in-plane neighbors.
        # For z_neighbor_depth=0, z is a "same-bin only" partition.
        if G_z is None:
            z_offsets_with_dz0: list[int] = [0]
            z_offsets_strict_pos: list[int] = []
        else:
            z_offsets_with_dz0 = list(range(-z_neighbor_depth, z_neighbor_depth + 1))
            z_offsets_strict_pos = list(range(1, z_neighbor_depth + 1))

        track_local = int(min_local_tx_per_entity) > 0

        # ----------------------------------------------------------------
        # Vectorised candidate-pair enumeration via table-level merges.
        # Replaces the per-bin Python record loop (~2.5M _record() calls
        # on the densest PDAC sub-tile, 60% of Stitch wall) with a pandas
        # hash-merge per (xy, z) offset followed by a single groupby
        # aggregation. The bin_to_comps / bin_to_comp_counts dicts are
        # preserved as-is for downstream filter compatibility.
        # ----------------------------------------------------------------
        bin_z_arr_for_df = (
            bz_arr if G_z is not None
            else np.zeros(len(comp_codes), dtype=np.int64)
        )
        bc_df = pd.DataFrame({
            "bin_xy": xy_keys.astype(np.int64),
            "bin_z":  bin_z_arr_for_df.astype(np.int64),
            "comp":   comp_codes.astype(np.int64),
        })
        bc_grouped = (
            bc_df.groupby(["bin_xy", "bin_z", "comp"], sort=False, as_index=False)
            .size().rename(columns={"size": "n_tx"})
        )
        bc_grouped["bin_xy"] = bc_grouped["bin_xy"].astype(np.int64)
        bc_grouped["bin_z"] = bc_grouped["bin_z"].astype(np.int64)
        bc_grouped["comp"] = bc_grouped["comp"].astype(np.int64)
        bc_grouped["n_tx"] = bc_grouped["n_tx"].astype(np.int64)

        def _shift_bin_xy(xy: np.ndarray, dx: int, dy: int) -> np.ndarray:
            bx = (xy >> np.int64(32)) - _BIN_BIAS
            by = (xy & np.int64(0xFFFFFFFF)) - _BIN_BIAS
            return (
                ((bx + dx + _BIN_BIAS).astype(np.int64) << np.int64(32))
                | (by + dy + _BIN_BIAS).astype(np.int64)
            )

        offsets_iter: list[tuple[int, int, int]] = [(0, 0, 0)]
        if G_z is None:
            offsets_iter += [(dx, dy, 0) for (dx, dy) in xy_half_offsets]
        else:
            offsets_iter += [
                (dx, dy, dz)
                for (dx, dy) in xy_half_offsets
                for dz in z_offsets_with_dz0
            ] + [(0, 0, dz) for dz in z_offsets_strict_pos]

        records: list[pd.DataFrame] = []
        for dx, dy, dz in offsets_iter:
            if dx == 0 and dy == 0 and dz == 0:
                merged = bc_grouped.merge(
                    bc_grouped,
                    on=["bin_xy", "bin_z"],
                    suffixes=("_a", "_b"),
                )
                merged = merged[merged["comp_a"] < merged["comp_b"]]
                if len(merged) == 0:
                    continue
                lo_arr = merged["comp_a"].to_numpy()
                hi_arr = merged["comp_b"].to_numpy()
                count_arr = (
                    merged["n_tx_a"].to_numpy()
                    * merged["n_tx_b"].to_numpy()
                )
                bin_lo_xy = merged["bin_xy"].to_numpy()
                bin_lo_z = merged["bin_z"].to_numpy()
                bin_hi_xy = bin_lo_xy
                bin_hi_z = bin_lo_z
            else:
                right = bc_grouped.copy()
                right["bin_xy_join"] = _shift_bin_xy(
                    right["bin_xy"].to_numpy(), -dx, -dy
                )
                right["bin_z_join"] = right["bin_z"] - dz
                merged = bc_grouped.merge(
                    right,
                    left_on=["bin_xy", "bin_z"],
                    right_on=["bin_xy_join", "bin_z_join"],
                    suffixes=("_a", "_b"),
                )
                merged = merged[merged["comp_a"] != merged["comp_b"]]
                if len(merged) == 0:
                    continue
                comp_a = merged["comp_a"].to_numpy()
                comp_b = merged["comp_b"].to_numpy()
                count_arr = (
                    merged["n_tx_a"].to_numpy()
                    * merged["n_tx_b"].to_numpy()
                )
                bin_a_xy = merged["bin_xy_a"].to_numpy()
                bin_a_z = merged["bin_z_a"].to_numpy()
                bin_b_xy = merged["bin_xy_b"].to_numpy()
                bin_b_z = merged["bin_z_b"].to_numpy()
                swap_mask = comp_a > comp_b
                lo_arr = np.where(swap_mask, comp_b, comp_a)
                hi_arr = np.where(swap_mask, comp_a, comp_b)
                bin_lo_xy = np.where(swap_mask, bin_b_xy, bin_a_xy)
                bin_lo_z = np.where(swap_mask, bin_b_z, bin_a_z)
                bin_hi_xy = np.where(swap_mask, bin_a_xy, bin_b_xy)
                bin_hi_z = np.where(swap_mask, bin_a_z, bin_b_z)

            out = pd.DataFrame({
                "lo": lo_arr.astype(np.int64),
                "hi": hi_arr.astype(np.int64),
                "count": count_arr.astype(np.int64),
            })
            if track_local:
                out["blxy"] = bin_lo_xy.astype(np.int64)
                out["blz"] = bin_lo_z.astype(np.int64)
                out["bhxy"] = bin_hi_xy.astype(np.int64)
                out["bhz"] = bin_hi_z.astype(np.int64)
            records.append(out)

        if records:
            all_records = pd.concat(records, ignore_index=True, copy=False)
            agg = (
                all_records.groupby(["lo", "hi"], sort=False, as_index=False)
                ["count"].sum()
            )
            pair_tx_edges: dict[tuple[int, int], int] = {
                (int(l), int(h)): int(c)
                for l, h, c in zip(
                    agg["lo"].to_numpy(), agg["hi"].to_numpy(),
                    agg["count"].to_numpy(),
                )
            }
            candidate_pairs: set[tuple[int, int]] = set(pair_tx_edges.keys())
        else:
            pair_tx_edges = {}
            candidate_pairs = set()

        # Pre-compute per-pair witness counts (n_lo, n_hi) via vectorised
        # joins. Previously we stored full pair_lo_bins / pair_hi_bins
        # sets and let the filter iterate them, which was ~50K small
        # per-pair Python set constructions and dominated this phase. The
        # filter only needs the SUM of bin_to_comp_counts[bin][comp] over
        # unique witness bins per pair — computable in a single merge +
        # groupby pair (no Python iteration).
        #
        # Single shared dedup at the (entity-pair, A-bin, B-bin) level
        # before each side's per-bin dedup. By construction `all_records`
        # is already unique at this level (each offset emits each
        # unordered bin pair at most once), so this is mostly an
        # idempotent enforcement — but it makes both per-side dedups
        # run against the minimal adjacency-row table rather than re-
        # scanning the full offset-iteration log twice. Each per-side
        # dedup remains essential: an A-bin paired with multiple B-bins
        # for the same (lo, hi) must contribute its A-tx count ONCE.
        #
        # A streaming per-pair walk with early exit (terminate once both
        # sides cross their effective thresholds) would be a stronger
        # win for healthy pairs, but trades vectorisation for Python
        # iteration; deferred until profiling shows this phase is hot.
        pair_n_lo: dict[tuple[int, int], int] = {}
        pair_n_hi: dict[tuple[int, int], int] = {}
        if track_local and records:
            ar_u = all_records[
                ["lo", "hi", "blxy", "blz", "bhxy", "bhz"]
            ].drop_duplicates()

            lo_merged = (
                ar_u[["lo", "hi", "blxy", "blz"]]
                .drop_duplicates()
                .merge(
                    bc_grouped,
                    left_on=["blxy", "blz", "lo"],
                    right_on=["bin_xy", "bin_z", "comp"],
                    how="left",
                )
            )
            lo_merged["n_tx"] = lo_merged["n_tx"].fillna(0).astype(np.int64)
            lo_summed = (
                lo_merged.groupby(["lo", "hi"], sort=False, as_index=False)
                ["n_tx"].sum()
            )
            pair_n_lo = {
                (int(l), int(h)): int(n)
                for l, h, n in zip(
                    lo_summed["lo"].to_numpy(),
                    lo_summed["hi"].to_numpy(),
                    lo_summed["n_tx"].to_numpy(),
                )
            }

            hi_merged = (
                ar_u[["lo", "hi", "bhxy", "bhz"]]
                .drop_duplicates()
                .merge(
                    bc_grouped,
                    left_on=["bhxy", "bhz", "hi"],
                    right_on=["bin_xy", "bin_z", "comp"],
                    how="left",
                )
            )
            hi_merged["n_tx"] = hi_merged["n_tx"].fillna(0).astype(np.int64)
            hi_summed = (
                hi_merged.groupby(["lo", "hi"], sort=False, as_index=False)
                ["n_tx"].sum()
            )
            pair_n_hi = {
                (int(l), int(h)): int(n)
                for l, h, n in zip(
                    hi_summed["lo"].to_numpy(),
                    hi_summed["hi"].to_numpy(),
                    hi_summed["n_tx"].to_numpy(),
                )
            }

        # Optional minimum-supporting-edges filter.
        if min_candidate_edges:
            if isinstance(min_candidate_edges, str):
                if min_candidate_edges != "min":
                    raise ValueError(
                        f"min_candidate_edges string mode must be 'min' "
                        f"(got {min_candidate_edges!r})"
                    )
                kept = {
                    p for p, n in pair_tx_edges.items()
                    if n >= min(entity_tx_total[p[0]], entity_tx_total[p[1]])
                }
            else:
                thr = int(min_candidate_edges)
                kept = {p for p, n in pair_tx_edges.items() if n >= thr}
            candidate_pairs = kept

        # Optional per-entity-witness count filter. Drop a candidate
        # pair (E1, E2) unless EACH entity contributes at least
        # `min_local_tx_per_entity` UNIQUE tx in the bins where they
        # co-occur (xy 8-Moore + z window). Symmetric in (E1, E2) —
        # not fooled by a 1-tx × N-tx bridging pair where the cross-
        # product count alone would pass `min_candidate_edges`.
        if track_local:
            mlt = int(min_local_tx_per_entity)
            # Witness counts are pre-summed per pair via the merge above,
            # so the filter is a single comprehension — no per-pair
            # Python set iteration. Same semantics as the prior code:
            # for each pair, require ≥mlt UNIQUE tx of EACH side across
            # all bins where the pair co-occurs.
            #
            # Effective threshold is capped at each entity's TOTAL tx
            # count: a 2-tx entity cannot produce 3 witnesses, so the
            # raw mlt threshold unfairly blocks every merge involving
            # any below-threshold entity. Cap: eff_mlt(E) = min(mlt, n_E).
            kept_local = {
                p for p in candidate_pairs
                if (pair_n_lo.get(p, 0)
                        >= min(mlt, entity_tx_total.get(p[0], mlt))
                    and pair_n_hi.get(p, 0)
                        >= min(mlt, entity_tx_total.get(p[1], mlt)))
            }
            candidate_pairs = kept_local

        # Optional per-pair median |Δz| guard. Reject candidate pairs
        # whose member tx have a median pairwise |Δz| larger than the
        # threshold. Useful when the bin filter under-discriminates due
        # to grid-alignment artefacts (a 1.5 µm physical gap can hide
        # inside one G_z=2 bin if the bin boundary aligns badly).
        if max_pair_median_dz is not None and G_z is not None:
            # Build per-entity z-coord array once
            ent_to_z: dict[int, np.ndarray] = defaultdict(list)
            for c, z in zip(comp_codes.tolist(),
                            transcript_coords[valid, 2].tolist()):
                ent_to_z[c].append(z)
            ent_to_z_arr = {c: np.asarray(zs) for c, zs in ent_to_z.items()}
            kept2 = set()
            for (a, b) in candidate_pairs:
                za = ent_to_z_arr.get(a)
                zb = ent_to_z_arr.get(b)
                if za is None or zb is None:
                    continue
                dz = np.abs(za[:, None] - zb[None, :]).ravel()
                if float(np.median(dz)) <= float(max_pair_median_dz):
                    kept2.add((a, b))
            candidate_pairs = kept2

        # Count-based Δz guard: admit only if at least
        # ``min_close_edges_n`` tx-tx pairs across (A, B) have
        # |Δz| < ``min_close_edges_dz``. Picks up the asymmetry between
        # within-cell pairs (where some edges are very tight) and
        # cross-stratum pairs (where every edge clears the gap).
        if (min_close_edges_dz is not None and min_close_edges_n > 0
                and G_z is not None):
            ent_to_z3: dict[int, np.ndarray] = defaultdict(list)
            for c, z in zip(comp_codes.tolist(),
                            transcript_coords[valid, 2].tolist()):
                ent_to_z3[c].append(z)
            ent_to_z3_arr = {c: np.asarray(zs) for c, zs in ent_to_z3.items()}
            kept3 = set()
            thr_dz = float(min_close_edges_dz)
            thr_n = int(min_close_edges_n)
            for (a, b) in candidate_pairs:
                za = ent_to_z3_arr.get(a)
                zb = ent_to_z3_arr.get(b)
                if za is None or zb is None:
                    continue
                dz = np.abs(za[:, None] - zb[None, :]).ravel()
                if int((dz < thr_dz).sum()) >= thr_n:
                    kept3.add((a, b))
            candidate_pairs = kept3

        edges = list(candidate_pairs)

        # Indices are no longer needed after initial enumeration; release memory.
        del bin_to_comps
    _phase("candidate_enum")

    # cluster metadata tracked at DSU roots
    dsu = DSU(N)

    # track whether a cluster contains a real cell (constraint)
    has_cell = np.array([t == "cell" for t in etypes], dtype=bool)

    # Per-root merger-tree depth: 0 for pre-stitch entities, updated on
    # union as `depth[rnew] = max(depth[ra], depth[rb]) + 1`. Used by
    # the optional `max_merger_depth` cap to block over-deep mergers.
    # Balanced merges of N entities cost log2(N) depth; left-deep chains
    # cost N-1. So the cap intrinsically rewards balanced consolidations
    # (similar-ΔC partner sizes) and penalises chain-style growth (one
    # big component repeatedly absorbing small partners — the failure
    # mode that produces multi-cell stromal-compartment blobs).
    merger_depth = np.zeros(N, dtype=np.int32)

    # For label preference
    # store lists of member entity_ids by type at roots (kept as python sets for simplicity)
    cell_ids = [set([entity_ids[i]]) if etypes[i] == "cell" else set() for i in range(N)]
    partial_ids = [set([entity_ids[i]]) if etypes[i] == "partial" else set() for i in range(N)]
    comp_ids = [set([entity_ids[i]]) if etypes[i] == "component" else set() for i in range(N)]

    # store gene_id union at roots (as sorted unique arrays)
    root_genes = gene_id_lists[:]  # list of np arrays

    # Decomposable-coherence state (only populated when
    # use_decomposable_stitch=True). Per-root running primitives
    # (n_above, n_below, n_finite) updated on union via the 6-segment
    # cross arithmetic. Initialised here from each original's self-prim;
    # combine on union below.
    root_prims: list[tuple[int, int, int]] | None = None
    if use_decomposable_stitch and mode == "count":
        try:
            from . import _cy_prune as _cyp
            # Cython kernel needs float32 dense W
            if isinstance(npmi_mat, np.ndarray) and npmi_mat.dtype == np.float32:
                W_f32 = npmi_mat
            else:
                import scipy.sparse as _sp
                if _sp.issparse(npmi_mat):
                    W_f32 = npmi_mat.toarray().astype(np.float32)
                else:
                    W_f32 = np.ascontiguousarray(npmi_mat, dtype=np.float32)
            root_prims = [
                _cyp.coherence_count_primitives(
                    np.ascontiguousarray(g, dtype=np.int32), W_f32, float(threshold)
                ) if g.size >= 2 else (0, 0, 0)
                for g in gene_id_lists
            ]
        except Exception:
            # Any setup failure → fall back gracefully (no primitive path).
            root_prims = None

    # Reset diagnostic gate-fire counters (visible to caller via
    # tracer.stitching._LAST_GATE_STATS after the call returns).
    _LAST_GATE_STATS.clear()
    _LAST_GATE_STATS.update({
        "K": int(spatial_centroid_k) if spatial_centroid_gate else 0,
        "checks_total": 0,        # _spatial_overlap calls
        "checks_pass": 0,         # _spatial_overlap returned True
        "init_bypasses": 0,       # heap-init pairs that took the bypass
        "merges_via_bypass": 0,   # actual unions that fired through bypass
        "merges_total": 0,        # all unions
        "mahalanobis_rescue_checks": 0,  # ΔC-borderline pairs evaluated for rescue
        "mahalanobis_rescues": 0,        # actual ΔC rejects overridden by Maha rescue
    })

    # ----------------------------------------------------------------
    # Per-root spatial state.
    # K=1 (default): bbox check via min/max columns from summary_df.
    # K≥2: count-based check requiring K tx-coords above AND K below
    # the smaller entity's centroid per axis. Requires per-root tx
    # coord arrays (maintained as concatenated ndarrays on union).
    # ----------------------------------------------------------------
    root_centroid: np.ndarray | None = None  # [N, n_dim]
    root_bbox_min: np.ndarray | None = None  # [N, n_dim]  K=1 only
    root_bbox_max: np.ndarray | None = None  # [N, n_dim]  K=1 only
    root_tx_coords: list | None = None        # [N] list of (n_tx, n_dim) arrays — K≥2 only

    # Per-root tx count — always populated (used by spatial gate AND
    # the size-gated c_union_bypass). Falls back to 1 per entity if the
    # summary_df was built without an n_tx column (legacy callers).
    if "n_tx" in summary_df.columns:
        root_n_tx: np.ndarray = summary_df["n_tx"].to_numpy(dtype=np.int64).copy()
    else:
        root_n_tx = np.ones(N, dtype=np.int64)

    if spatial_centroid_gate:
        coord_keys = ["x", "y", "z"] if use_3d else ["x", "y"]
        try:
            root_centroid = summary_df[coord_keys].to_numpy(dtype=np.float64).copy()
            min_keys = [f"{c}_min" for c in coord_keys]
            max_keys = [f"{c}_max" for c in coord_keys]
            root_bbox_min = summary_df[min_keys].to_numpy(dtype=np.float64).copy()
            root_bbox_max = summary_df[max_keys].to_numpy(dtype=np.float64).copy()
            if spatial_centroid_k >= 2:
                if entity_tx_coords is None:
                    # Need per-entity tx coords for K≥2 → fall back to K=1
                    spatial_centroid_k = 1
                    root_tx_coords = None
                else:
                    root_tx_coords = [
                        np.asarray(entity_tx_coords.get(str(eid), np.zeros((0, len(coord_keys)))),
                                    dtype=np.float64)
                        for eid in summary_df["entity_id"].astype(str)
                    ]
        except KeyError:
            spatial_centroid_gate = False
            root_centroid = root_bbox_min = root_bbox_max = None
            root_tx_coords = None
            # root_n_tx kept — it's used by c_union_bypass_max_n_tx too.

    # Mahalanobis RESCUE: build per-root tx-coord arrays if not already
    # built by the K≥2 spatial-gate path. Independent of
    # `spatial_centroid_gate`. If `entity_tx_coords` is missing we
    # silently disable (the rescue is a soft optional layer).
    _maha_n_dim: int = 3 if use_3d else 2
    if mahalanobis_d_rescue is not None and root_tx_coords is None:
        coord_keys_m = ["x", "y", "z"] if use_3d else ["x", "y"]
        _maha_n_dim = len(coord_keys_m)
        if entity_tx_coords is None:
            mahalanobis_d_rescue = None  # graceful no-op
        else:
            try:
                root_tx_coords = [
                    np.asarray(
                        entity_tx_coords.get(
                            str(eid), np.zeros((0, len(coord_keys_m)))
                        ),
                        dtype=np.float64,
                    )
                    for eid in summary_df["entity_id"].astype(str)
                ]
            except Exception:
                mahalanobis_d_rescue = None  # graceful no-op

    def _mahalanobis_distance(ra: int, rb: int) -> float:
        """Mahalanobis-D between the two roots' tx clouds.

        D = sqrt( (μ_A − μ_B)^T  Σ_pooled^-1  (μ_A − μ_B) ),
        Σ_pooled = ((n_A − 1)·Cov_A + (n_B − 1)·Cov_B) / (n_A + n_B − 2)

        Returns NaN when n<2 in either side, when tx coords are
        unavailable, or when Σ_pooled is singular / numerically
        ill-conditioned (callers treat NaN as "no rescue").
        """
        if root_tx_coords is None:
            return float("nan")
        ca = root_tx_coords[ra]
        cb = root_tx_coords[rb]
        if ca is None or cb is None or ca.size == 0 or cb.size == 0:
            return float("nan")
        n_a = int(ca.shape[0])
        n_b = int(cb.shape[0])
        if n_a < 2 or n_b < 2:
            return float("nan")
        mu_a = ca.mean(axis=0)
        mu_b = cb.mean(axis=0)
        cov_a = np.atleast_2d(np.cov(ca, rowvar=False, ddof=1))
        cov_b = np.atleast_2d(np.cov(cb, rowvar=False, ddof=1))
        denom = float(n_a + n_b - 2)
        if denom <= 0.0:
            return float("nan")
        cov_pooled = ((n_a - 1) * cov_a + (n_b - 1) * cov_b) / denom
        try:
            cond = np.linalg.cond(cov_pooled)
            if not np.isfinite(cond) or cond > 1e12:
                return float("nan")
            diff = (mu_a - mu_b).reshape(-1, 1)
            sol = np.linalg.solve(cov_pooled, diff)
            d2 = float((diff.T @ sol).item())
        except np.linalg.LinAlgError:
            return float("nan")
        if not np.isfinite(d2) or d2 < 0.0:
            return float("nan")
        return float(np.sqrt(d2))

    def _spatial_overlap(ra: int, rb: int) -> bool:
        """Default rule: smaller's centroid inside larger's tx cloud.
        K=1 → bbox check. K≥2 → ≥K tx of larger above AND below
        smaller's centroid per axis.

        Flipped rule (spatial_gate_flipped=True): swap roles. Test
        whether the LARGER's centroid lies inside the SMALLER's tx
        cloud. Effective K capped at floor(n_smaller / 3) so a small
        partial uses a lighter K. Detects "cell arranged AROUND
        partial" (legitimate fragment) instead of "partial embedded
        IN cell" (often contamination).
        """
        _LAST_GATE_STATS["checks_total"] += 1
        n_a = int(root_n_tx[ra])
        n_b = int(root_n_tx[rb])
        if n_a <= n_b:
            small_idx, large_idx = ra, rb
            n_small, n_large = n_a, n_b
        else:
            small_idx, large_idx = rb, ra
            n_small, n_large = n_b, n_a

        if spatial_gate_flipped:
            # test point = larger's centroid; reference cloud = smaller's tx
            c = root_centroid[large_idx]
            ref_idx = small_idx
            # Dynamic K cap: small partial → lighter K. Uses ceiling
            # division (n+2)//3 so a 5-tx partial gets K=2, not K=1
            # (bbox-only). Floor gave K=1 for n∈{3,4,5} — too permissive.
            k_eff = min(int(spatial_centroid_k),
                         max(1, (n_small + 2) // 3))
        else:
            c = root_centroid[small_idx]
            ref_idx = large_idx
            # Original "smaller's centroid in larger" rule with dynamic
            # floor based on LARGER entity's size:
            #   K_eff = max(K, ceil(n_larger / 3))
            # For a 105-tx cell merging with a small partial, K_eff = 35
            # — requires 35 cell tx straddling the partial centroid in
            # each axis. Stricter for big cells (the typical contamination
            # host), no-op for small (n<30) cells.
            k_eff = max(int(spatial_centroid_k),
                         (n_large + 2) // 3)

        if k_eff <= 1:
            # Bbox check (cheap)
            bb_min = root_bbox_min[ref_idx]
            bb_max = root_bbox_max[ref_idx]
            ok = bool(np.all(c >= bb_min) and np.all(c <= bb_max))
            if ok:
                _LAST_GATE_STATS["checks_pass"] += 1
            return ok

        # K≥2: per-axis count of tx in reference cloud above/below c.
        ref_coords = root_tx_coords[ref_idx]
        if ref_coords.size == 0 or ref_coords.shape[0] < 2 * k_eff:
            return False
        for d in range(c.shape[0]):
            col = ref_coords[:, d]
            n_above = int(np.sum(col > c[d]))
            if n_above < k_eff:
                return False
            n_below = int(np.sum(col < c[d]))
            if n_below < k_eff:
                return False
        _LAST_GATE_STATS["checks_pass"] += 1
        return True

    # constraint: can we merge clusters A and B?
    def can_merge(ra, rb):
        # never merge two clusters that both contain a cell
        if has_cell[ra] and has_cell[rb]:
            return False
        # Optional merger-tree depth cap. Stops a component from
        # growing beyond a fixed merger-tree height — protects against
        # chain-style "big-blob-keeps-absorbing-neighbours" growth that
        # produces multi-cell over-merges. Block if EITHER side has
        # already reached the cap (so the union, which would be at
        # max+1, would exceed it).
        if max_merger_depth is not None:
            if (int(merger_depth[ra]) >= int(max_merger_depth)
                    or int(merger_depth[rb]) >= int(max_merger_depth)):
                return False
        return True

    # ----------------------------------------------------------------
    # Per-root coherence cache.
    #
    # The heap loop pops O(N · avg_neighbours) candidate edges and each
    # call to deltaC needs C(ra), C(rb), and C(ra ∪ rb). Without a
    # cache, C(ra) and C(rb) get recomputed once for every neighbour
    # they appear with — a 2–3× redundant cost on the rejected pops
    # that dominate the loop. We cache (C, purity, conflict) per root
    # and invalidate the entry immediately after dsu.union (when the
    # root's gene set changes). C(ra ∪ rb) is genuinely new each merge
    # and is not cached.
    # ----------------------------------------------------------------
    root_C_cache: dict[int, tuple[float, float, float]] = {}
    cache_hits = 0
    cache_misses = 0

    def C_of_root(root_idx: int) -> tuple[float, float, float]:
        nonlocal cache_hits, cache_misses
        cached = root_C_cache.get(root_idx)
        if cached is not None:
            cache_hits += 1
            return cached
        cache_misses += 1
        # Decomposable path: derive (C, purity, conflict) from the
        # root's running primitive sums — no gene-pair iteration.
        if use_decomposable_stitch and root_prims is not None and mode == "count":
            na, nb, nf = root_prims[root_idx]
            if nf == 0:
                triple = (0.0, 0.0, 0.0)
            else:
                purity = na / nf
                conflict = nb / nf
                triple = (purity - conflict, purity, conflict)
            root_C_cache[root_idx] = triple
            return triple
        triple = coherence(
            root_genes[root_idx], npmi_mat,
            mode=mode, threshold=threshold, metric=metric,
        )
        root_C_cache[root_idx] = triple
        return triple

    # Helper: combine two roots' primitives into the union's via the
    # 6-segment decomposition (validated bit-exact in
    # /tmp/validate_decomp_coh.py against direct coherence). Returns
    # (n_above_union, n_below_union, n_finite_union, union_genes_array).
    # Only invoked when use_decomposable_stitch=True and root_prims is
    # populated (mode == 'count' + dense float32 W).
    def _combine_prims(ra, rb):
        ga = root_genes[ra]
        gb = root_genes[rb]
        if ga.size == 0 and gb.size == 0:
            return (0, 0, 0), np.empty(0, dtype=np.int32)
        if ga.size == 0:
            return root_prims[rb], gb
        if gb.size == 0:
            return root_prims[ra], ga
        # 3-segment partition: a_only, b_only, common
        common = np.intersect1d(ga, gb, assume_unique=True)
        a_only = np.setdiff1d(ga, common, assume_unique=True).astype(np.int32)
        b_only = np.setdiff1d(gb, common, assume_unique=True).astype(np.int32)
        common32 = common.astype(np.int32)
        # primitives needed: 3 self (a_only, b_only, common)
        # + 3 cross (a×b, a×c, b×c).  Compose triu(union) from these.
        from . import _cy_prune as _cyp_local
        sa = _cyp_local.coherence_count_primitives(a_only, W_f32, float(threshold)) if a_only.size >= 2 else (0, 0, 0)
        sb = _cyp_local.coherence_count_primitives(b_only, W_f32, float(threshold)) if b_only.size >= 2 else (0, 0, 0)
        sc = _cyp_local.coherence_count_primitives(common32, W_f32, float(threshold)) if common32.size >= 2 else (0, 0, 0)
        cab = _cyp_local.coherence_cross_primitives(a_only, b_only, W_f32, float(threshold))
        cac = _cyp_local.coherence_cross_primitives(a_only, common32, W_f32, float(threshold))
        cbc = _cyp_local.coherence_cross_primitives(b_only, common32, W_f32, float(threshold))
        union_prims = (
            sa[0] + sb[0] + sc[0] + cab[0] + cac[0] + cbc[0],
            sa[1] + sb[1] + sc[1] + cab[1] + cac[1] + cbc[1],
            sa[2] + sb[2] + sc[2] + cab[2] + cac[2] + cbc[2],
        )
        union_genes = np.concatenate([a_only, b_only, common32])
        union_genes.sort()
        return union_prims, union_genes

    # Sentinel ΔC value for spatial-overlap pairs. Pushes them to the
    # top of the heap (popped first, bypass coherence + threshold).
    # 1e9 is far above any realistic ΔC value (∈ [-1, 1] in practice).
    _SPATIAL_OVERLAP_DC = 1e9

    # compute deltaC between current roots
    # Returns (dC, C_union). C_union is the raw post-merge coherence
    # before any size-bias penalty, used by the optional bypass gate.
    # For the spatial-bypass shortcut, C_union is set to 1.0 (sentinel
    # mirrors the dC sentinel: pair is a guaranteed merge regardless).
    def compute_deltaC_roots(ra, rb):
        # Spatial bypass (Tier 1): in mode="pre", if the smaller
        # entity's centroid is inside the larger entity's bbox, treat
        # the pair as a GUARANTEED merge — return a high sentinel ΔC.
        # No coherence / gene-PMI evaluation is performed. In
        # mode="post", we never short-circuit here; the spatial test
        # is checked at pop time as a fallback for ΔC-rejected pairs.
        if (spatial_centroid_gate and spatial_gate_mode == "pre"
                and root_centroid is not None
                and _spatial_overlap(ra, rb)):
            return _SPATIAL_OVERLAP_DC, 1.0

        # Decomposable-primitive fast path: derive C(union) from the
        # roots' running primitive sums + on-the-fly cross primitives,
        # without re-iterating the union's full gene-pair set. Bit-
        # equivalent to coherence(union) for mode='count' (validated).
        if use_decomposable_stitch and root_prims is not None and mode == "count":
            Cu, _, _ = C_of_root(ra)
            Cv, _, _ = C_of_root(rb)
            (na_u, nb_u, nf_u), _ = _combine_prims(ra, rb)
            if nf_u == 0:
                Cunion = 0.0
            else:
                Cunion = (na_u - nb_u) / nf_u
            if not penalize_simplicity:
                return float(Cunion - max(Cu, Cv)), float(Cunion)
            nu = max(int(root_genes[ra].size), 1)
            nv = max(int(root_genes[rb].size), 1)
            n_union = nu + nv
            C_sep = max(Cu - 1.0 / nu, Cv - 1.0 / nv)
            return float(Cunion - (1.0 / n_union) - C_sep), float(Cunion)

        # Eager path (default): compute C(union) directly via coherence.
        Cu, _, _ = C_of_root(ra)
        Cv, _, _ = C_of_root(rb)
        union = np.unique(np.concatenate([root_genes[ra], root_genes[rb]]))
        Cunion, _, _ = coherence(
            union, npmi_mat, mode=mode, threshold=threshold, metric=metric,
        )
        if not penalize_simplicity:
            return float(Cunion - max(Cu, Cv)), float(Cunion)
        nu = max(int(root_genes[ra].size), 1)
        nv = max(int(root_genes[rb].size), 1)
        n_union = nu + nv
        C_sep = max(Cu - 1.0 / nu, Cv - 1.0 / nv)
        return float(Cunion - (1.0 / n_union) - C_sep), float(Cunion)

    def _pair_passes_gate(dc, c_union, ra, rb):
        """Acceptance gate: ΔC ≥ deltaC_min OR (bypass set and C_union ≥ bypass).
        Spatial bypass already yields dc=sentinel which always passes.

        When `c_union_bypass_max_n_tx` is set, the C(union) bypass only
        applies if the merged entity's total tx count would stay at or
        below the threshold. Small mergers (typically within-cell
        fragment consolidation) admit; large mergers (typically cross-
        cell partial bridging) require the strong ΔC signal.
        """
        if not np.isfinite(dc):
            return False
        if dc >= deltaC_min:
            return True
        if (c_union_bypass is not None
                and np.isfinite(c_union)
                and c_union >= c_union_bypass):
            if c_union_bypass_max_n_tx is None:
                return True
            n_union = int(root_n_tx[ra]) + int(root_n_tx[rb])
            if n_union <= c_union_bypass_max_n_tx:
                return True
        return False

    # max-heap of candidate edges by deltaC (lazy updates)
    def _heap_item(dc, a, b):
        # Deterministic tie-breaking: enforce ordered endpoints
        if a > b:
            a, b = b, a
        return (-dc, a, b)

    # ----------------------------------------------------------------
    # Optional heap-init fast-gate: drop candidate pairs whose top-clique
    # cross-PMI block contains a strong-negative entry. For most cell-
    # pair candidates that are biologically incompatible, this avoids
    # the expensive compute_deltaC_roots call entirely. The gate is
    # applied ONLY to the initial edge list (heap-init); boundary
    # expansion + stale-pop reinserts during the merge loop are unchanged.
    # ----------------------------------------------------------------
    gate_keep_mask: np.ndarray | None = None
    if fast_gate_top_k > 0 and len(edges) > 0:
        try:
            from . import _cy_prune as _cyp_gate
            # Build float32 dense W if not already
            if isinstance(npmi_mat, np.ndarray) and npmi_mat.dtype == np.float32:
                _W_f32 = npmi_mat
            else:
                import scipy.sparse as _sp
                if _sp.issparse(npmi_mat):
                    _W_f32 = npmi_mat.toarray().astype(np.float32)
                else:
                    _W_f32 = np.ascontiguousarray(npmi_mat, dtype=np.float32)
            top_cliques = _cyp_gate.top_k_positive_clique_per_entity(
                gene_id_lists, _W_f32, int(fast_gate_top_k), float(threshold),
            )
            edges_arr = np.asarray(edges, dtype=np.int32)
            if edges_arr.ndim == 1:
                edges_arr = edges_arr.reshape(-1, 2)
            gate_keep_mask = _cyp_gate.fast_gate_pairs(
                top_cliques, edges_arr, _W_f32, float(fast_gate_mean_threshold),
            )
        except Exception:
            gate_keep_mask = None  # graceful fallback: no gating

    _phase("setup_root_state")
    heap = []
    for ei, (i, j) in enumerate(edges):
        # Tier 1 (positive override): spatial-overlap bypass. If the
        # smaller entity's centroid is inside the larger's bbox, this
        # pair MUST go on the heap (with sentinel ΔC) regardless of
        # the gate result. compute_deltaC_roots returns 1e9 for these.
        is_spatial = (
            spatial_centroid_gate and root_centroid is not None
            and _spatial_overlap(i, j)
        )
        if is_spatial:
            _LAST_GATE_STATS["init_bypasses"] += 1
        # Tier 2 (cheap rejection): fast-gate skips expensive eval ONLY
        # when there's no spatial bypass. In "post" mode, spatial is a
        # fallback at pop time, so still let the fast-gate cull these.
        if (not is_spatial) and gate_keep_mask is not None and not gate_keep_mask[ei]:
            continue
        # Tier 3 (full ΔC eval; sentinel 1e9 only in "pre" mode)
        di, ci = compute_deltaC_roots(i, j)
        if _pair_passes_gate(di, ci, i, j):
            heapq.heappush(heap, _heap_item(di, i, j))
        elif (is_spatial and spatial_gate_mode == "post"
              and np.isfinite(di)):
            # Post-mode rescue: ΔC says reject, but spatial matches.
            # Push at the real (low/negative) ΔC priority so genuine
            # ΔC merges happen first; this pair gets revisited at pop
            # time and merged via the spatial-override path.
            heapq.heappush(heap, _heap_item(di, i, j))
        elif (
            mahalanobis_d_rescue is not None
            and np.isfinite(di)
            and rescue_delta_c_floor < di < 0.0
        ):
            # Mahalanobis-rescue candidate: ΔC says reject but ΔC sits
            # in the rescue band. Push at the real (negative) ΔC so
            # the rescue gets a chance at pop time. The pop-time check
            # re-evaluates ΔC against current roots and consults the
            # Maha distance before admitting.
            heapq.heappush(heap, _heap_item(di, i, j))

    _phase("heap_init")
    # greedy merging
    while heap:
        neg_dc, a, b = heapq.heappop(heap)
        dc = -neg_dc

        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            continue
        if not can_merge(ra, rb):
            continue

        # recompute deltaC for current clusters (because a,b may have merged)
        dc_now, cu_now = compute_deltaC_roots(ra, rb)
        post_override = False
        maha_rescued = False
        if not _pair_passes_gate(dc_now, cu_now, ra, rb):
            # ΔC says reject. In "post" mode, give the spatial gate
            # one chance: if the (current-root) centroid test still
            # matches, force the merge anyway. This is the only way
            # spatial can intervene in post-mode.
            if (spatial_centroid_gate and spatial_gate_mode == "post"
                    and root_centroid is not None
                    and _spatial_overlap(ra, rb)):
                post_override = True
            elif (
                mahalanobis_d_rescue is not None
                and np.isfinite(dc_now)
                and rescue_delta_c_floor < dc_now < 0.0
            ):
                # Mahalanobis RESCUE — override the ΔC reject when the
                # two tx clouds are geometrically enmeshed.
                _LAST_GATE_STATS["mahalanobis_rescue_checks"] += 1
                d = _mahalanobis_distance(ra, rb)
                if np.isfinite(d) and d <= float(mahalanobis_d_rescue):
                    maha_rescued = True
                    _LAST_GATE_STATS["mahalanobis_rescues"] += 1
                else:
                    continue
            else:
                continue

        # merge (choose new root)
        rnew = dsu.union(ra, rb)
        rold = rb if rnew == ra else ra
        _LAST_GATE_STATS["merges_total"] += 1
        # Update merger-tree depth: new root carries 1 + max of children.
        merger_depth[rnew] = max(int(merger_depth[ra]),
                                  int(merger_depth[rb])) + 1
        # Track which gate drove the merge:
        #   pre-mode bypass → dc_now == 1e9 sentinel
        #   post-mode override → ΔC failed but spatial said merge
        if dc_now >= _SPATIAL_OVERLAP_DC * 0.5:
            _LAST_GATE_STATS["merges_via_bypass"] += 1
        elif post_override:
            _LAST_GATE_STATS["merges_via_bypass"] += 1

        # update cluster metadata onto rnew
        has_cell[rnew] = has_cell[rnew] or has_cell[rold]
        cell_ids[rnew] |= cell_ids[rold]
        partial_ids[rnew] |= partial_ids[rold]
        comp_ids[rnew] |= comp_ids[rold]

        # n_tx accumulation — always updated (used by size-gated
        # c_union_bypass even when the spatial gate is off).
        n_new = int(root_n_tx[rnew])
        n_old = int(root_n_tx[rold])
        n_total = n_new + n_old
        root_n_tx[rnew] = n_total
        root_n_tx[rold] = 0

        # Spatial state update on union (when gate is active).
        if (spatial_centroid_gate and root_centroid is not None):
            if n_total > 0:
                root_centroid[rnew] = (
                    (root_centroid[rnew] * n_new + root_centroid[rold] * n_old) / n_total
                )
            root_bbox_min[rnew] = np.minimum(root_bbox_min[rnew], root_bbox_min[rold])
            root_bbox_max[rnew] = np.maximum(root_bbox_max[rnew], root_bbox_max[rold])
            # K≥2 path: maintain concatenated tx-coord arrays per root.
            if root_tx_coords is not None:
                if root_tx_coords[rnew].size == 0:
                    root_tx_coords[rnew] = root_tx_coords[rold]
                elif root_tx_coords[rold].size > 0:
                    root_tx_coords[rnew] = np.concatenate(
                        [root_tx_coords[rnew], root_tx_coords[rold]], axis=0
                    )
                root_tx_coords[rold] = np.zeros((0, root_centroid.shape[1]), dtype=np.float64)
        elif mahalanobis_d_rescue is not None and root_tx_coords is not None:
            # Mahalanobis-rescue only (spatial centroid gate is off) —
            # still need to maintain concatenated tx-coord arrays on
            # union so the rescue sees the merged cloud on subsequent
            # comparisons.
            n_dim_u = (
                root_tx_coords[rnew].shape[1] if root_tx_coords[rnew].size
                else (root_tx_coords[rold].shape[1]
                      if root_tx_coords[rold].size else _maha_n_dim)
            )
            if root_tx_coords[rnew].size == 0:
                root_tx_coords[rnew] = root_tx_coords[rold]
            elif root_tx_coords[rold].size > 0:
                root_tx_coords[rnew] = np.concatenate(
                    [root_tx_coords[rnew], root_tx_coords[rold]], axis=0
                )
            root_tx_coords[rold] = np.zeros((0, n_dim_u), dtype=np.float64)

        # union genes (and primitive sums when in decomposable mode).
        # In decomposable mode we already computed _combine_prims for
        # (ra, rb) inside compute_deltaC_roots; recompute here for the
        # new root's bookkeeping. Yes, this duplicates work — a future
        # optimisation could cache the result. For now, correctness
        # over speed: the merge path is O(merges), not O(rounds).
        if use_decomposable_stitch and root_prims is not None and mode == "count":
            new_prims, new_genes = _combine_prims(ra, rb)
            root_genes[rnew] = new_genes if new_genes.dtype == np.int32 else new_genes.astype(np.int32)
            root_prims[rnew] = new_prims
            root_prims[rold] = (0, 0, 0)
        else:
            if root_genes[rnew].size == 0:
                root_genes[rnew] = root_genes[rold]
            elif root_genes[rold].size == 0:
                pass
            else:
                root_genes[rnew] = np.unique(np.concatenate([root_genes[rnew], root_genes[rold]])).astype(np.int32)

        # clear old to save memory
        cell_ids[rold].clear()
        partial_ids[rold].clear()
        comp_ids[rold].clear()
        root_genes[rold] = np.empty((0,), dtype=np.int32)

        # Invalidate cached coherence for both old roots — rnew's gene
        # set just changed; rold is now empty. They'll be recomputed on
        # next access.
        root_C_cache.pop(ra, None)
        root_C_cache.pop(rb, None)

        # Boundary expansion: push new candidate edges around rnew.
        # Lazy DSU revalidation at pop handles any duplicate pushes.
        if candidate_source == "delaunay":
            # Reuse original node adjacency via a and b endpoints.
            for nbr in (adj[a] + adj[b]):
                rn = dsu.find(nbr)
                rr = dsu.find(rnew)
                if rn == rr:
                    continue
                if not can_merge(rr, rn):
                    continue
                dtry, ctry = compute_deltaC_roots(rr, rn)
                if _pair_passes_gate(dtry, ctry, rr, rn):
                    heapq.heappush(heap, _heap_item(dtry, rr, rn))
        else:  # candidate_source == "grid"
            # No explicit boundary expansion needed: in grid mode, the
            # initial candidate enumeration is comprehensive (every
            # spatially-adjacent component pair is in the heap), and
            # lazy DSU revalidation at pop handles merges. Skipping
            # expansion + index maintenance is both correct and avoids
            # an O(|bins(rnew)| * 9) scan per merge.
            pass

    # ----------------------------------------------------------------
    # Choose stitched label per final root.
    # Priority: cell > partial > component.
    #
    # Multi-partial merger rule (when partial_ids[r] has > 1 element):
    # the merged entity is GIVEN A FRESH LABEL with a SECOND DASH
    # LEVEL, so the result is distinguishable from any of its inputs.
    #
    # Label form:
    #   Phase 1c partial:  "{cell}-{d1}"        (single dash, e.g. "37962-1")
    #   Stitch merger:     "{cell}-{d1}-{d2}"   (two dashes, e.g. "37962-1-1")
    #
    # The depth-2 namespace `(cell, d1)` is per-(winning cell, winning
    # d1). It is initialised by scanning all input partials and
    # recording the max d2 seen in each (cell, d1) namespace; new
    # mergers increment past that max → guaranteed-unique labels.
    #
    # Decision rule for picking the merger's parent (which (cell, d1)
    # namespace owns the result):
    #   1. Higher d1 suffix wins (more aggregated lineage).
    #   2. Higher d2 suffix wins (already-merged > unmerged).
    #   3. Higher tx count wins (dominant biological signal).
    #   4. Lexicographic (final deterministic tiebreak).
    # ----------------------------------------------------------------
    def _parse_partial(label: str) -> tuple[str, int, int] | None:
        """Parse '{cell}-{d1}' or '{cell}-{d1}-{d2}'. Returns (cell,
        d1, d2) or None if not a valid partial label. d2 = 0 for
        single-dash labels."""
        if "-" not in label:
            return None
        parts = label.rsplit("-", 2)
        # parts can be 1, 2, or 3 elements depending on dash count.
        if len(parts) == 2:
            cell, d1_str = parts
            try:
                return cell, int(d1_str), 0
            except ValueError:
                return None
        elif len(parts) == 3:
            cell, d1_str, d2_str = parts
            try:
                return cell, int(d1_str), int(d2_str)
            except ValueError:
                return None
        return None

    # Initialise the depth-2 counters from input labels.
    next_merger_counter: dict[tuple[str, int], int] = {}
    for i in range(N):
        eid = entity_ids[i]
        if etypes[i] != "partial":
            continue
        parsed = _parse_partial(eid)
        if parsed is None:
            continue
        cell, d1, d2 = parsed
        key = (cell, d1)
        cur = next_merger_counter.get(key, 0)
        if d2 > cur:
            next_merger_counter[key] = d2

    def _pick_partial_label(partials: set[str]) -> str:
        if len(partials) == 1:
            return next(iter(partials))
        rows = []
        for p in partials:
            parsed = _parse_partial(p)
            if parsed is None:
                continue
            cell, d1, d2 = parsed
            n_tx = (entity_n_tx or {}).get(p, 0)
            rows.append((d1, d2, n_tx, p, cell))
        if not rows:
            return sorted(partials)[0]
        # Sort: highest d1 → highest d2 → highest tx → lex-smaller label.
        rows.sort(key=lambda r: (-r[0], -r[1], -r[2], r[3]))
        winner_d1, _, _, _, winner_cell = rows[0]
        key = (winner_cell, winner_d1)
        next_merger_counter[key] = next_merger_counter.get(key, 0) + 1
        return f"{winner_cell}-{winner_d1}-{next_merger_counter[key]}"

    root_to_label = {}
    for i in range(N):
        r = dsu.find(i)
        if r in root_to_label:
            continue
        if cell_ids[r]:
            label = sorted(cell_ids[r])[0]          # deterministic
        elif partial_ids[r]:
            label = _pick_partial_label(partial_ids[r])
        else:
            label = sorted(comp_ids[r])[0]
        root_to_label[r] = label

    entity_to_stitched = {entity_ids[i]: root_to_label[dsu.find(i)] for i in range(N)}
    _phase("merge_loop")
    _LAST_STITCH_PHASE_TIMINGS.clear()
    _LAST_STITCH_PHASE_TIMINGS.update(_phase_timings)
    info = {
        "root_to_label": root_to_label,
        "coherence_cache_hits": cache_hits,
        "coherence_cache_misses": cache_misses,
        "phase_timings": dict(_phase_timings),
    }
    return entity_to_stitched, info

def apply_stitching_to_transcripts(
    df_final: pd.DataFrame,
    aux: dict,
    *,
    entity_col="cell_id_final",   # final id column
    gene_col="feature_name",
    coord_cols=("x", "y", "z"),
    mode: str = "count",
    threshold: float = 0.05,
    penalize_simplicity=True,
    deltaC_min=0.0,
    use_3d=True,
    out_col="cell_id_stitched",
    purity_threshold=_LEGACY_STITCH_KWARG_SENTINEL,
    tau=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relu=_LEGACY_STITCH_KWARG_SENTINEL,
):
    _ensure_reproducibility_seed()
    # build entity table (centroids + genes)
    summary = build_entity_table(
        df_final,
        entity_col=entity_col,
        gene_col=gene_col,
        coord_cols=coord_cols,
    )

    # rename centroid cols to x,y,z expected by stitching function
    # (build_entity_table keeps original names)
    if tuple(coord_cols) == ("x", "y", "z"):
        summary = summary.rename(columns={"x": "x", "y": "y", "z": "z"})
    else:
        # if different coordinate column names used, map them:
        summary = summary.rename(columns={coord_cols[0]: "x", coord_cols[1]: "y", coord_cols[2]: "z"})

    legacy_kwargs = {}
    if purity_threshold is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["purity_threshold"] = purity_threshold
    if tau is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["tau"] = tau
    if use_relu is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["use_relu"] = use_relu

    # stitch entities
    entity_to_stitched, info = stitch_entities_hierarchical(
        summary_df=summary.rename(columns={"entity_id": "entity_id"}),
        aux=aux,
        mode=mode,
        threshold=threshold,
        penalize_simplicity=penalize_simplicity,
        deltaC_min=deltaC_min,
        use_3d=use_3d,
        dist_threshold=None,
        **legacy_kwargs,
    )

    # map back to transcripts
    df_out = df_final.copy()
    ent = df_out[entity_col].astype(str)

    # default: keep original entity label (DROP stays DROP)
    df_out[out_col] = ent

    # apply stitched labels to non-drop entities
    mask = ent.notna() & (ent != "DROP") & (ent != "nan")
    df_out.loc[mask, out_col] = ent[mask].map(entity_to_stitched).fillna(ent[mask])

    return df_out, entity_to_stitched


def apply_stitching_to_transcripts_fast(
    df_final: pd.DataFrame,
    aux: dict,
    *,
    entity_col="cell_id_final",
    gene_col="feature_name",
    coord_cols=("x", "y", "z"),
    mode: str = "count",
    threshold: float = 0.05,
    penalize_simplicity=True,
    deltaC_min=0.0,
    use_3d=True,
    out_col="cell_id_stitched",
    show_progress: bool = True,
    purity_threshold=_LEGACY_STITCH_KWARG_SENTINEL,
    tau=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relu=_LEGACY_STITCH_KWARG_SENTINEL,
):
    """
    Fast wrapper around `apply_stitching_to_transcripts`.
    - Builds entity table and runs hierarchical stitching, with optional progress bars.
    - Uses ReLU-based coherence scoring by default for robust cluster merging.
    - Returns same outputs as original function.

    Parameters
    ----------
    df_final : pd.DataFrame
        Transcript-level data with entity assignments
    aux : dict
        Contains NPMI matrix ("W") and gene mapping ("gene_to_idx")
    entity_col : str
        Column with current entity labels
    gene_col : str
        Column with gene names
    coord_cols : tuple
        Coordinate column names
    purity_threshold : float
        Threshold for original scoring (used if use_relu=False)
    tau : float
        Dead-zone threshold for ReLU (used if use_relu=True, default)
    use_relu : bool
        If True, use ReLU-based coherence (default, faster and more robust)
    penalize_simplicity : bool
        Penalize smaller gene sets in deltaC
    deltaC_min : float
        Minimum deltaC for merging
    use_3d : bool
        Use 3D coordinates
    out_col : str
        Output column name
    show_progress : bool
        Show progress bar

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with stitched labels
    entity_to_stitched : dict
        Mapping from original to stitched entity IDs
    """
    _ensure_reproducibility_seed()
    # build entity table (centroids + genes)
    if show_progress:
        # small progress step for entity build
        pbar = tqdm(total=2, desc="stitching")
    else:
        pbar = None

    summary = build_entity_table(
        df_final,
        entity_col=entity_col,
        gene_col=gene_col,
        coord_cols=coord_cols,
    )
    if pbar is not None:
        pbar.update(1)

    # rename centroid cols if necessary
    if tuple(coord_cols) == ("x", "y", "z"):
        summary = summary.rename(columns={"x": "x", "y": "y", "z": "z"})
    else:
        summary = summary.rename(columns={coord_cols[0]: "x", coord_cols[1]: "y", coord_cols[2]: "z"})

    legacy_kwargs = {}
    if purity_threshold is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["purity_threshold"] = purity_threshold
    if tau is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["tau"] = tau
    if use_relu is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["use_relu"] = use_relu

    # stitch entities (this is the heavy op)
    entity_to_stitched, info = stitch_entities_hierarchical(
        summary_df=summary.rename(columns={"entity_id": "entity_id"}),
        aux=aux,
        mode=mode,
        threshold=threshold,
        penalize_simplicity=penalize_simplicity,
        deltaC_min=deltaC_min,
        use_3d=use_3d,
        dist_threshold=None,
        **legacy_kwargs,
    )

    if pbar is not None:
        pbar.update(1)
        pbar.close()

    # map back to transcripts using vectorized numpy lookup (much faster than pandas.map())
    df_out = df_final.copy()
    ent = df_out[entity_col].astype(str)
    df_out[out_col] = ent

    mask = ent.notna() & (ent != "DROP") & (ent != "nan")

    if mask.sum() > 0:
        # Fully vectorized mapping using pandas.Series.map() (much faster than loop)
        ent_values = ent[mask]

        # Convert dict to pandas Series for vectorized .map()
        mapping_series = pd.Series(entity_to_stitched)

        # Vectorized map with fillna for unmapped values (keeps original)
        stitched_values = ent_values.map(mapping_series).fillna(ent_values)

        # Single assignment
        df_out.loc[mask, out_col] = stitched_values

    return df_out, entity_to_stitched


def apply_stitching_to_transcripts_memory_efficient(
    df_final: pd.DataFrame,
    aux: dict,
    *,
    entity_col: str = "tracer_id",
    gene_col: str = "feature_name",
    coord_cols=("x", "y", "z"),
    mode: str = "count",
    threshold: float = 0.05,
    metric: str = "npmi",
    penalize_simplicity: bool = True,
    deltaC_min: float = 0.0,
    # See `stitch_entities_hierarchical` docstring. None = off (legacy).
    c_union_bypass: float | None = None,
    # See `stitch_entities_hierarchical` docstring. None = no size cap.
    c_union_bypass_max_n_tx: int | None = None,
    # See `stitch_entities_hierarchical` docstring. None = off (legacy).
    max_merger_depth: int | None = None,
    use_3d: bool = True,
    dist_threshold: float | None = 15.0,
    out_col: str = "tracer_id",
    debug_stages: bool = False,
    debug_legacy_col: str = "cell_id_stitched",
    show_progress: bool = True,
    in_place: bool = False,
    map_mode: str = "categorical",
    chunk_size: int | None = 2_000_000,
    candidate_source: str = "delaunay",
    G: float | None = None,
    stitch_neighborhood: str = "8",
    G_z: float | None = None,
    z_neighbor_depth: int = 0,
    min_candidate_edges: int | str = 0,
    # Optional per-entity-witness count: drop candidate pair (E1, E2)
    # unless EACH entity contributes at least `min_local_tx_per_entity`
    # unique tx in the shared bin neighborhood (xy 8-Moore + ±depth z
    # bins). Catches single-bridging-tx candidates that sneak through
    # the diagonal-Moore reach (~5.66 µm at G=2). Symmetric in (E1, E2)
    # — resistant to mass-dominated cross-product counts.
    # Default 0 = off (current behavior unchanged).
    min_local_tx_per_entity: int = 0,
    max_pair_median_dz: float | None = None,
    min_close_edges_dz: float | None = None,
    min_close_edges_n: int = 0,
    purity_threshold=_LEGACY_STITCH_KWARG_SENTINEL,
    tau=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relu=_LEGACY_STITCH_KWARG_SENTINEL,
    use_relative=_LEGACY_STITCH_KWARG_SENTINEL,
    # Experimental: lazy DSU+heap with decomposable-coherence primitives.
    # Default False (eager path, byte-unchanged). Bit-match validated on
    # 500/1000 µm ROIs (99.98%+ per-tx label parity, ARI identical to 4
    # decimals). See `_stitch_entities_hierarchical_decomposable` for
    # rationale and `TODO.md` for tissue-scale follow-ups.
    use_decomposable_stitch: bool = False,
    # Experimental: top-K positive-clique fast-gate at heap-init.
    # 0 = disabled (default). ≥1 enables — pre-filters candidate pairs
    # using a small per-entity signature signature; rejects pairs with
    # strong-negative top-clique cross-PMI before expensive ΔC eval.
    fast_gate_top_k: int = 0,
    fast_gate_mean_threshold: float = 0.0,
    # Experimental: spatial centroid-in-bbox gate at merge time.
    # When True, smaller entity's centroid must lie inside the larger
    # entity's per-axis tx-coord range (axis-aligned bbox). Default
    # False (no spatial constraint beyond Stitch's existing
    # `dist_threshold` Delaunay-edge filter at candidate-build time).
    spatial_centroid_gate: bool = False,
    # Tightness of the spatial-overlap test. K=1 → bbox check (at
    # least 1 tx of larger entity above AND 1 below smaller's centroid
    # in each axis). K=2 → require 2 above AND 2 below per axis (more
    # interior). K=3 → 3 each. Higher K = stricter.
    spatial_centroid_k: int = 1,
    # Optional per-entity tx-coord arrays. Required for K≥2; with K=1
    # the gate falls back to bbox check using `summary_df`'s min/max
    # columns. dict[entity_id_str -> (n_tx, n_dim) ndarray].
    entity_tx_coords: dict | None = None,
    # Spatial gate mode: "pre" (current default — spatial bypass overrides
    # ΔC and merges first) or "post" (spatial fires only as a fallback
    # for ΔC-rejected pairs).
    spatial_gate_mode: str = "pre",
    # Flipped overlap test: larger's centroid inside smaller's tx cloud.
    # Effective K capped at floor(n_smaller / 3).
    spatial_gate_flipped: bool = False,
    # Optional Mahalanobis-D RESCUE on borderline-ΔC pairs. See
    # `stitch_entities_hierarchical` docstring. None = off (default).
    mahalanobis_d_rescue: float | None = None,
    rescue_delta_c_floor: float = -0.2,
):
    """
    Memory-efficient stitching wrapper optimized for very large datasets (10M+ rows).

    This function mirrors `apply_stitching_to_transcripts_fast` but minimizes
    temporary allocations when mapping stitched labels back to transcripts.

    Parameters
    ----------
    df_final : pd.DataFrame
        Transcript-level data with entity assignments
    aux : dict
        Contains NPMI matrix ("W") and gene mapping ("gene_to_idx")
    entity_col : str
        Column with current entity labels
    gene_col : str
        Column with gene names
    coord_cols : tuple
        Coordinate column names
    purity_threshold : float
        Threshold for original scoring (used if use_relu=False)
    tau : float
        Dead-zone threshold for ReLU (used if use_relu=True, default)
    use_relu : bool
        If True, use ReLU-based coherence (default, faster and more robust)
    use_relative : bool
        If True (and use_relu=True), use relative_purity and
        relative_conflict for stitching.
    penalize_simplicity : bool
        Penalize smaller gene sets in deltaC
    deltaC_min : float
        Minimum deltaC for merging
    use_3d : bool
        Use 3D coordinates
    out_col : str
        Output column name
    show_progress : bool
        Show progress bar
    in_place : bool
        If True, write output to the input DataFrame without copying
    map_mode : {"categorical", "chunked"}
        Mapping strategy to minimize memory use.
        - "categorical": map category codes (fast, low memory)
        - "chunked": map in chunks using pandas Series.map()
    chunk_size : int or None
        Chunk size for "chunked" mapping. None maps all at once.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with stitched labels
    entity_to_stitched : dict
        Mapping from original to stitched entity IDs
    """
    _ensure_reproducibility_seed()
    if show_progress:
        pbar = tqdm(total=2, desc="stitching")
    else:
        pbar = None

    summary = build_entity_table(
        df_final,
        entity_col=entity_col,
        gene_col=gene_col,
        coord_cols=coord_cols,
    )
    if pbar is not None:
        pbar.update(1)

    if tuple(coord_cols) == ("x", "y", "z"):
        summary = summary.rename(columns={"x": "x", "y": "y", "z": "z"})
    else:
        summary = summary.rename(columns={coord_cols[0]: "x", coord_cols[1]: "y", coord_cols[2]: "z"})

    # Build per-transcript inputs for grid candidate enumeration if requested.
    transcript_coords = None
    transcript_entity_codes = None
    if candidate_source == "grid":
        # Map each transcript's entity string to its row index in summary_df.
        entity_id_arr = summary["entity_id"].astype(str).to_numpy()
        entity_to_idx = {eid: i for i, eid in enumerate(entity_id_arr)}
        ent_str = df_final[entity_col].astype(str).to_numpy()
        transcript_entity_codes = np.fromiter(
            (entity_to_idx.get(e, -1) for e in ent_str),
            dtype=np.int64,
            count=len(ent_str),
        )
        if G_z is not None and len(coord_cols) >= 3:
            transcript_coords = df_final[
                [coord_cols[0], coord_cols[1], coord_cols[2]]
            ].to_numpy(dtype=np.float64)
        else:
            transcript_coords = df_final[
                [coord_cols[0], coord_cols[1]]
            ].to_numpy(dtype=np.float64)

    legacy_kwargs = {}
    if purity_threshold is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["purity_threshold"] = purity_threshold
    if tau is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["tau"] = tau
    if use_relu is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["use_relu"] = use_relu
    if use_relative is not _LEGACY_STITCH_KWARG_SENTINEL:
        legacy_kwargs["use_relative"] = use_relative

    # Per-entity tx count for the multi-partial merger tiebreak rule
    # in stitch_entities_hierarchical (majority-tx-count when suffix
    # levels tie). One O(N) value_counts pass over the entity column.
    entity_n_tx_dict = (
        df_final[entity_col].astype(str).value_counts().to_dict()
    )

    # When the K≥2 strict spatial gate OR the Mahalanobis rescue is
    # requested, compute per-entity tx-coord arrays. Cheap groupby-by-
    # entity on transcript-level data.
    entity_tx_coords_dict: dict | None = None
    _needs_tx_coords = (
        (spatial_centroid_gate and spatial_centroid_k >= 2)
        or mahalanobis_d_rescue is not None
    )
    if _needs_tx_coords:
        coord_cols_used = list(coord_cols) if use_3d else list(coord_cols[:2])
        entity_tx_coords_dict = {}
        ent_str_col = df_final[entity_col].astype(str)
        for eid, sub in df_final.groupby(ent_str_col, observed=True):
            entity_tx_coords_dict[str(eid)] = sub[coord_cols_used].to_numpy(
                dtype=np.float64
            )

    entity_to_stitched, info = stitch_entities_hierarchical(
        summary_df=summary.rename(columns={"entity_id": "entity_id"}),
        aux=aux,
        mode=mode,
        threshold=threshold,
        metric=metric,
        penalize_simplicity=penalize_simplicity,
        deltaC_min=deltaC_min,
        c_union_bypass=c_union_bypass,
        c_union_bypass_max_n_tx=c_union_bypass_max_n_tx,
        max_merger_depth=max_merger_depth,
        use_3d=use_3d,
        dist_threshold=dist_threshold,
        candidate_source=candidate_source,
        G=G,
        stitch_neighborhood=stitch_neighborhood,
        G_z=G_z,
        z_neighbor_depth=z_neighbor_depth,
        transcript_coords=transcript_coords,
        transcript_entity_codes=transcript_entity_codes,
        min_candidate_edges=min_candidate_edges,
        min_local_tx_per_entity=min_local_tx_per_entity,
        max_pair_median_dz=max_pair_median_dz,
        min_close_edges_dz=min_close_edges_dz,
        min_close_edges_n=min_close_edges_n,
        use_decomposable_stitch=use_decomposable_stitch,
        fast_gate_top_k=fast_gate_top_k,
        fast_gate_mean_threshold=fast_gate_mean_threshold,
        entity_n_tx=entity_n_tx_dict,
        spatial_centroid_gate=spatial_centroid_gate,
        spatial_centroid_k=spatial_centroid_k,
        entity_tx_coords=entity_tx_coords_dict,
        spatial_gate_mode=spatial_gate_mode,
        spatial_gate_flipped=spatial_gate_flipped,
        mahalanobis_d_rescue=mahalanobis_d_rescue,
        rescue_delta_c_floor=rescue_delta_c_floor,
        **legacy_kwargs,
    )

    if pbar is not None:
        pbar.update(1)
        pbar.close()

    df_out = df_final if in_place else df_final.copy()
    ent = df_out[entity_col]

    # Did any merge actually occur this pass (≥2 entities collapse onto
    # one stitched label)? Guards the post-merge `_etype` homogenization
    # so the no-merge fast path stays cheap (no O(N) heterogeneity scan).
    merges_happened = True

    if map_mode == "categorical":
        ent_cat = ent.astype("category")
        categories = ent_cat.cat.categories.astype(str)
        mapped_categories = pd.Index(categories).map(lambda x: entity_to_stitched.get(x, x))
        merges_happened = not mapped_categories.is_unique

        # Fast path: one-to-one mapping (no merges) -> just rename categories
        if mapped_categories.is_unique:
            df_out[out_col] = ent_cat.cat.rename_categories(mapped_categories)
        else:
            # Slow path: merges exist, recode via factorization
            new_cat_codes, new_categories = pd.factorize(mapped_categories, sort=False)
            ent_codes = ent_cat.cat.codes.to_numpy(copy=False)

            out_codes = np.full_like(ent_codes, -1)
            valid = ent_codes >= 0
            if valid.any():
                out_codes[valid] = new_cat_codes[ent_codes[valid]]

            df_out[out_col] = pd.Categorical.from_codes(out_codes, categories=new_categories)
        # Fall through to the shared block below — do NOT early-return;
        # otherwise the categorical fast path skips the `_etype`
        # homogenization and downstream `.first()` becomes
        # non-deterministic. (debug_legacy_col is written once, in the
        # shared block, for both map modes.)
    elif map_mode == "chunked":
        ent_str = ent.astype(str)
        merges_happened = not (
            pd.Index(ent_str.unique())
            .map(lambda x: entity_to_stitched.get(x, x))
            .is_unique
        )
        df_out[out_col] = ent_str

        mask = ent_str.notna() & (ent_str != "DROP") & (ent_str != "nan")
        if mask.any():
            idx = np.flatnonzero(mask.to_numpy())
            mapping_series = pd.Series(entity_to_stitched)

            if chunk_size is None:
                vals = ent_str.iloc[idx]
                mapped = vals.map(mapping_series).fillna(vals)
                df_out.iloc[idx, df_out.columns.get_loc(out_col)] = mapped.to_numpy()
            else:
                for start in range(0, len(idx), chunk_size):
                    end = start + chunk_size
                    sel = idx[start:end]
                    vals = ent_str.iloc[sel]
                    mapped = vals.map(mapping_series).fillna(vals)
                    df_out.iloc[sel, df_out.columns.get_loc(out_col)] = mapped.to_numpy()
    else:
        raise ValueError("map_mode must be 'categorical' or 'chunked'")

    if debug_stages and debug_legacy_col != out_col:
        df_out[debug_legacy_col] = df_out[out_col].copy()

    # Homogenize `_etype` on every stitched label that ended up
    # heterogeneous. Without this, rows of merged-in entities keep their
    # original (now-stale) etype, and downstream
    # `df.groupby(entity_col)["_etype"].first()` becomes non-deterministic
    # — silently miscoding a merged entity and breaking the cell-cell
    # merge gate in any subsequent stitch pass.
    #
    # NOTE on scope: summary uses `etype_filter=("cell",)` by default, so
    # non-cell pre-stitch entities (partials/components) never appear in
    # entity_to_stitched. They keep their original tracer_id as the
    # stitched label, and their rows can still end up grouped with cell
    # rows under the same stitched label (when a cell entity's id equals
    # a partial's parent prefix). So we homogenize all heterogeneous
    # stitched labels rather than only those reachable via summary.
    #
    # Complexity: detecting the heterogeneous labels is an O(N_tx)
    # groupby-nunique scan — but it only runs when `merges_happened`
    # (the no-merge categorical fast path skips it entirely). The fix
    # itself (homogenize_etype_for_entities) then touches only the
    # heterogeneous labels' rows. See that helper for the priority rule.
    if "_etype" in df_out.columns and merges_happened:
        from ._etype import homogenize_etype_for_entities
        # Find stitched labels whose rows have heterogeneous _etype.
        _SENT = {"-1", "DROP", "UNASSIGNED", "nan"}
        labels = df_out[out_col].astype(str)
        sub_mask = ~labels.isin(_SENT)
        het_counts = (
            df_out.loc[sub_mask, [out_col, "_etype"]]
            .astype({out_col: str, "_etype": str})
            .groupby(out_col)["_etype"].nunique()
        )
        het_labels = het_counts[het_counts > 1].index.tolist()
        if het_labels:
            # One pass over only the heterogeneous labels' rows.
            homogenize_etype_for_entities(
                df_out, het_labels, entity_col=out_col, etype_col="_etype",
            )

    return df_out, entity_to_stitched
