"""Phase-1-time Mahalanobis-gated remerge.

Sibling of the Stitch-time ``mahalanobis_d_rescue`` (see
``tracer.stitching``). Applies the same rule one stage earlier — right
after Phase 1 QC, before Rescue/Group/Stitch — to catch EMT-like
over-splits at the source.

Rule (per candidate entity pair sharing an xy 8-Moore + z-window bin):

    floor < ΔC < 0    AND    D ≤ threshold    →    merge (DSU union)

ΔC is gated first; pairs outside the (floor, 0) band short-circuit before
Mahalanobis-D computation, keeping the per-pair cost cheap on dense ROIs.

The pipeline runner calls ``phase1_maha_remerge`` between Phase1-QC and
Rescue when ``cfg.phase1.maha_remerge_d`` is set. When ``None``,
behavior is bit-exact unchanged.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Sentinel labels treated as "no entity" (skip in candidate enumeration).
_SENT: set[str] = {"-1", "DROP", "UNASSIGNED", "nan", "__GUARD_SKIP__"}


# ---------------------------------------------------------------
# Disjoint-set union for chain-correct relabeling
# ---------------------------------------------------------------
class _DSU:
    __slots__ = ("parent",)

    def __init__(self, items: Iterable[str]):
        self.parent: dict[str, str] = {x: x for x in items}

    def find(self, x: str) -> str:
        # Path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Deterministic: smaller string label survives (matches Stitch's
        # endpoint-ordering convention).
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


# ---------------------------------------------------------------
# Mahalanobis-D between two tx clouds
# (lifted from tracer.stitching._mahalanobis_distance — same math)
# ---------------------------------------------------------------
def _maha_d(ca: np.ndarray, cb: np.ndarray) -> float:
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


# ---------------------------------------------------------------
# Candidate-pair enumeration (xy 8-Moore + ±z_neighbor_depth z bins)
# ---------------------------------------------------------------
def _candidate_pairs(
    df: pd.DataFrame,
    *,
    label_col: str,
    bin_size_um: float,
    g_z_um: float,
    z_depth: int,
) -> list[tuple[str, str]]:
    """Two entities are candidates if they share at least one
    xy 8-Moore-neighbor bin within ±z_depth z-bins.
    """
    labels = df[label_col].astype(str).to_numpy()
    keep = ~pd.Series(labels).isin(_SENT).to_numpy()
    if not keep.any():
        return []

    xs = df.loc[keep, "x"].to_numpy()
    ys = df.loc[keep, "y"].to_numpy()
    zs = df.loc[keep, "z"].to_numpy() if "z" in df.columns else np.zeros(int(keep.sum()))
    lab_k = labels[keep]

    bx = np.floor(xs / bin_size_um).astype(np.int64)
    by = np.floor(ys / bin_size_um).astype(np.int64)
    bz = np.floor(zs / max(g_z_um, 1e-9)).astype(np.int64)

    # bin -> set(labels)
    bin_to_labs: dict[tuple[int, int, int], set[str]] = {}
    for i in range(lab_k.size):
        key = (int(bx[i]), int(by[i]), int(bz[i]))
        bin_to_labs.setdefault(key, set()).add(lab_k[i])

    # For each bin, expand neighborhood (3x3 xy * (2z_depth+1) z) and
    # union the label sets. Pairs within the joined set are candidates.
    offsets_xy = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    offsets_z = list(range(-z_depth, z_depth + 1))

    cand: set[tuple[str, str]] = set()
    for key in bin_to_labs.keys():
        bxk, byk, bzk = key
        joined: set[str] = set()
        for dx, dy in offsets_xy:
            for dz in offsets_z:
                nb = (bxk + dx, byk + dy, bzk + dz)
                if nb in bin_to_labs:
                    joined |= bin_to_labs[nb]
        joined_list = sorted(joined)
        for i in range(len(joined_list)):
            for j in range(i + 1, len(joined_list)):
                cand.add((joined_list[i], joined_list[j]))
    return sorted(cand)


# ---------------------------------------------------------------
# Panel → W matrix builder
# ---------------------------------------------------------------
def _build_W_from_panel(panel) -> tuple[np.ndarray, dict[str, int]]:
    """Build (W, g2i) from a panel DataFrame OR a pre-built aux dict.

    Accepts:
      - dict with keys {"W", "g2i"} or {"W", "gene_to_idx"} → reuse directly.
      - DataFrame with columns (gene_i, gene_j) and one of
        (value | PMI | NPMI | weight) → build dense symmetric W with NaN diag.
    """
    if isinstance(panel, dict):
        W = panel.get("W")
        g2i = panel.get("g2i") or panel.get("gene_to_idx")
        if W is None or g2i is None:
            raise ValueError(
                "panel dict must have 'W' and 'g2i'/'gene_to_idx' keys"
            )
        return W, g2i

    pdf = panel.copy()
    if "value" in pdf.columns:
        pdf = pdf.rename(columns={"value": "weight"})
    elif "NPMI" in pdf.columns:
        pdf = pdf.rename(columns={"NPMI": "weight"})
    elif "PMI" in pdf.columns:
        pdf = pdf.rename(columns={"PMI": "weight"})
    elif "weight" not in pdf.columns:
        raise ValueError(
            "panel must have 'value', 'NPMI', 'PMI', or 'weight' column"
        )
    pdf["gene_i"] = pdf["gene_i"].astype(str)
    pdf["gene_j"] = pdf["gene_j"].astype(str)
    genes = sorted(set(pdf.gene_i) | set(pdf.gene_j))
    g2i = {g: i for i, g in enumerate(genes)}
    W = np.full((len(genes), len(genes)), np.nan, dtype=np.float32)
    gi = pdf.gene_i.map(g2i).to_numpy(np.int64)
    gj = pdf.gene_j.map(g2i).to_numpy(np.int64)
    v = pdf.weight.to_numpy(np.float32)
    W[gi, gj] = v
    W[gj, gi] = v
    np.fill_diagonal(W, np.nan)
    return W, g2i


# ---------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------
def phase1_maha_remerge(
    df: pd.DataFrame,
    panel_or_aux,
    *,
    threshold: float,
    floor: float,
    cfg=None,
    entity_col: str = "tracer_id",
    gene_col: str = "feature_name",
    bin_size_um: float = 2.0,
    g_z_um: float = 1.0,
    z_neighbor_depth: int = 1,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Apply Mahalanobis-gated rescue at Phase-1-QC output.

    For each candidate entity pair sharing a bin neighborhood
    (xy 8-Moore + z-window):
      1. Compute ΔC (same coherence call signature as Stitch).
      2. If ΔC not in (floor, 0): skip (no Maha computation).
      3. Compute Mahalanobis D over the two entities' tx coords.
      4. If D <= threshold: union via DSU.

    Parameters
    ----------
    df : pd.DataFrame
        Post-Phase1-QC transcript-level dataframe with at least
        ``entity_col``, ``gene_col``, and coord columns (x, y, [z]).
    panel_or_aux : pd.DataFrame | dict
        Either the bootstrap NPMI/PMI panel (DataFrame with gene_i/gene_j
        + value/PMI/NPMI/weight) or a pre-built aux dict
        ``{"W": ..., "g2i": ...}``.
    threshold : float
        Mahalanobis-D ceiling for rescue (typically ~1.0).
    floor : float
        ΔC magnitude bound (must be ≤ 0; typically -0.2).
    cfg : PipelineConfig | None
        When provided, ΔC parameters (pmi_threshold, etc.) are read from
        ``cfg.phase1`` / ``cfg.stitch`` to match the surrounding
        pipeline. When ``None``, sensible defaults are used.

    Returns
    -------
    df_out : pd.DataFrame
        Same columns/order as ``df``; ``entity_col`` is remapped to
        DSU-root labels.
    stats : dict
        ``n_candidates``, ``n_dc_in_band``, ``n_rescues``, ``pairs``
        (list of dicts: src/dst labels, ΔC, D).
    """
    from tracer.stitching import deltaC  # local import — avoids cycle

    # Pull coherence/ΔC parameters from cfg when available; otherwise use
    # documented defaults that match the bench helper.
    pmi_thr = 0.2
    mode = "count"
    metric = "pmi"
    penalize_simplicity = True
    if cfg is not None:
        _p1 = getattr(cfg, "phase1", None)
        if _p1 is not None:
            pmi_thr = float(getattr(_p1, "pmi_threshold", pmi_thr))

    df = df.copy()

    W, g2i = _build_W_from_panel(panel_or_aux)

    # Group tx by entity, build per-entity gene arrays + tx-coord arrays.
    labels = df[entity_col].astype(str).to_numpy()
    keep_mask = ~pd.Series(labels).isin(_SENT).to_numpy()
    keep_df = df.loc[keep_mask].reset_index(drop=True)
    if keep_df.empty:
        return df, dict(n_candidates=0, n_dc_in_band=0, n_rescues=0, pairs=[])

    coord_cols = ["x", "y"] + (["z"] if "z" in df.columns else [])
    by_label = keep_df.groupby(entity_col, sort=False)
    ent_genes: dict[str, np.ndarray] = {}
    ent_coords: dict[str, np.ndarray] = {}
    for lbl, sub in by_label:
        gids = np.array(
            [g2i[g] for g in sub[gene_col].astype(str).unique() if g in g2i],
            dtype=np.int64,
        )
        ent_genes[str(lbl)] = gids
        ent_coords[str(lbl)] = sub[coord_cols].to_numpy(np.float64)

    cand = _candidate_pairs(
        df, label_col=entity_col,
        bin_size_um=bin_size_um, g_z_um=g_z_um, z_depth=z_neighbor_depth,
    )

    dsu = _DSU(ent_genes.keys())
    n_dc_in_band = 0
    pairs_rescued: list[dict] = []
    for a, b in cand:
        ga, gb = ent_genes.get(a), ent_genes.get(b)
        if ga is None or gb is None:
            continue
        if ga.size < 2 or gb.size < 2:
            # Coherence undefined for k<2 — skip (matches stitching).
            continue
        # ΔC first — cheap short-circuit before Maha.
        dc = deltaC(
            ga, gb, W,
            mode=mode, threshold=pmi_thr, metric=metric,
            penalize_simplicity=penalize_simplicity,
        )
        if not np.isfinite(dc):
            continue
        if not (floor < dc < 0.0):
            continue
        n_dc_in_band += 1
        d = _maha_d(ent_coords[a], ent_coords[b])
        if not np.isfinite(d):
            continue
        if d <= float(threshold):
            dsu.union(a, b)
            pairs_rescued.append(dict(
                a=a, b=b, deltaC=float(dc), maha_D=float(d),
                root=dsu.find(a),
            ))

    if pairs_rescued:
        remap = {lbl: dsu.find(lbl) for lbl in ent_genes.keys()}
        df.loc[keep_mask, entity_col] = df.loc[keep_mask, entity_col].astype(str).map(
            lambda x: remap.get(x, x)
        )
        # Homogenize _etype on every affected merged entity. Without this
        # the merged entity carries mixed-_etype rows (cell tx from one
        # side, partial tx from the other), and downstream
        # `groupby(entity_col)["_etype"].first()` becomes non-deterministic
        # — silently miscoding the entity for the cell-cell merge gate in
        # Stitch. See `tracer._etype.homogenize_etype_for_entity` for the
        # priority rule.
        if "_etype" in df.columns:
            from ._etype import homogenize_etype_for_entities
            affected_roots = {dsu.find(p["a"]) for p in pairs_rescued}
            # Batch: resolve all merged roots in one call (single column
            # scan to locate their rows), not a full rescan per root.
            homogenize_etype_for_entities(
                df, affected_roots, entity_col=entity_col, etype_col="_etype",
            )

    stats = dict(
        n_candidates=len(cand),
        n_dc_in_band=int(n_dc_in_band),
        n_rescues=len(pairs_rescued),
        pairs=pairs_rescued,
    )
    if verbose:
        print(
            f"[phase1_maha_remerge] candidates={stats['n_candidates']} "
            f"dc_in_band={stats['n_dc_in_band']} "
            f"rescues={stats['n_rescues']}",
            flush=True,
        )
        for p in pairs_rescued:
            print(
                f"    merge: {p['a']} + {p['b']}  ΔC={p['deltaC']:.4f}  "
                f"D={p['maha_D']:.3f}  → {p['root']}",
                flush=True,
            )
    return df, stats


__all__ = ["phase1_maha_remerge"]
