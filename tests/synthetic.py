"""Synthetic data generators shared across tests (and reusable from
``benchmarks/``).

Three generators:

- :func:`make_synthetic_npmi_panel` — plants 5 known gene-pair structures
  (positive, negative, independent, dropout-high, dropout-low) for testing
  the bootstrap NPMI panel builder. Returns a long-format
  ``(cell_id, feature_name)`` DataFrame.

- :func:`make_synthetic_transcripts` — plants ``n_cells`` cells arranged
  on a coarse xy grid, each cell drawn from one of ``n_types`` archetype
  gene-panel templates with strong within-type co-occurrence.
  Returns a transcript-level DataFrame matching the schema TRACER pipeline
  stages expect (``transcript_id``, ``feature_name``, ``cell_id``, ``x``,
  ``y``, ``z``) plus a ``ground_truth`` dict.

- :func:`make_synthetic_npmi_panel_for_transcripts` — builds a
  long-format NPMI / PMI panel consistent with the planted gene-coherence
  structure of :func:`make_synthetic_transcripts`. Suitable as the
  ``npmi_df`` argument to ``prune_transcripts_fast`` and downstream
  pipeline stages.

All three are deterministic given a seed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------
# 1. NPMI bootstrap correctness fixture
# -----------------------------------------------------------------------

def make_synthetic_npmi_panel(N: int = 1000, G: int = 20, seed: int = 42
                              ) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a synthetic transcript table with planted gene-pair structures.

    Returns
    -------
    df : pd.DataFrame
        Long-format with columns ``cell_id``, ``feature_name``. Each cell is
        repeated ``min_occurrences_per_context=2`` so the bootstrap has
        enough observations.
    M : np.ndarray of shape ``(N, G)``
        The cell × gene presence matrix (0/1) used to construct ``df``.

    Planted structures (gene indices):
      - 0, 1: strong positive co-occurrence (~40% jointly present).
      - 2, 3: strong mutual exclusivity (~80% have exactly one).
      - 4, 5: independent at rate 0.3 each.
      - 6, 7: rare and independent (~0.02 each), zero observed cooccur,
        E[cooccur] ≈ 0.4 (< 10) → indeterminate under the bootstrap rule.
      - 8, 9: high marginal (~0.4 each), zero observed cooccur,
        E[cooccur] ≈ 160 (≫ 10) → ``neg_one`` sentinel.
      - 10..G-1: noise at 10%.
    """
    rng = np.random.default_rng(seed)

    M = np.zeros((N, G), dtype=np.int8)

    # Genes 0,1: strong positive — both in 40% of cells.
    coexp = rng.random(N) < 0.4
    M[coexp, 0] = 1
    M[coexp, 1] = 1
    M[rng.random(N) < 0.02, 0] = 1
    M[rng.random(N) < 0.02, 1] = 1

    # Genes 2,3: mutual exclusivity, exactly one of them in 80% of cells.
    which = rng.random(N) < 0.4
    in_pair = rng.random(N) < 0.8
    M[in_pair & which, 2] = 1
    M[in_pair & ~which, 3] = 1

    # Genes 4,5: independent at rate 0.3.
    M[rng.random(N) < 0.3, 4] = 1
    M[rng.random(N) < 0.3, 5] = 1

    # Genes 6,7: rare and independent (~0.02 each).
    M[rng.random(N) < 0.02, 6] = 1
    M[rng.random(N) < 0.02, 7] = 1
    both = (M[:, 6] == 1) & (M[:, 7] == 1)
    if both.any():
        idxs = np.flatnonzero(both)
        half = len(idxs) // 2
        M[idxs[:half], 7] = 0
        M[idxs[half:], 6] = 0

    # Genes 8,9: high marginal (~0.4 each), zero cooccur.
    p89 = rng.random(N)
    g8 = p89 < 0.4
    g9 = (p89 >= 0.4) & (p89 < 0.8)
    M[g8, 8] = 1
    M[g9, 9] = 1

    # Other genes: noise at 10%.
    for g in range(10, G):
        M[rng.random(N) < 0.1, g] = 1

    rows = []
    for cid in range(N):
        for g in range(G):
            if M[cid, g]:
                rows.append((str(cid), f"gene_{g:02d}"))
                rows.append((str(cid), f"gene_{g:02d}"))
    df = pd.DataFrame(rows, columns=["cell_id", "feature_name"])
    return df, M


# -----------------------------------------------------------------------
# 2. Synthetic transcripts (for end-to-end pipeline tests)
# -----------------------------------------------------------------------

_SIX_NEIGHBORS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
                  (0, 0, -1), (0, 0, 1))


def _deepest_voxel(
    voxels: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Find the cell voxel whose minimum 6-connected distance to any
    non-cell voxel is maximum — i.e., the morphologically deepest
    interior point of the cell.

    Computed via multi-source BFS from the cell's boundary voxels (cell
    voxels with at least one non-cell 6-neighbor) inward. The voxel
    with the largest BFS distance is the deepest. For roughly-spherical
    cells this is near the geometric centroid; for irregular or
    elongated shapes it's the inradius point and can be substantially
    further from the boundary than the centroid.
    """
    from collections import deque

    cell_set = set(voxels)
    if len(cell_set) == 1:
        return voxels[0]

    # Boundary = cell voxels with at least one non-cell 6-neighbor.
    dist: dict[tuple[int, int, int], int] = {}
    q: deque[tuple[int, int, int]] = deque()
    for vx in voxels:
        for dx, dy, dz in _SIX_NEIGHBORS:
            nbr = (vx[0] + dx, vx[1] + dy, vx[2] + dz)
            if nbr not in cell_set:
                dist[vx] = 0
                q.append(vx)
                break
    if not q:
        # Pathological: cell completely surrounded by other cell voxels
        # (no boundary at all). Shouldn't happen with a finite grid.
        return voxels[0]

    while q:
        vx = q.popleft()
        for dx, dy, dz in _SIX_NEIGHBORS:
            nbr = (vx[0] + dx, vx[1] + dy, vx[2] + dz)
            if nbr in cell_set and nbr not in dist:
                dist[nbr] = dist[vx] + 1
                q.append(nbr)

    return max(dist, key=dist.get)


def _flood_fill_cell(seed_voxel, target, owner, cell_idx, nuclear_layers):
    """BFS-grow a cell from ``seed_voxel`` until it reaches ``target``
    voxels or all reachable neighbours are exhausted.

    Returns
    -------
    voxels : list[tuple[int, int, int]]
        All voxels assigned to this cell, including the seed.
    nuclear_voxels : set[tuple[int, int, int]]
        Subset that were assigned in the first ``nuclear_layers`` BFS
        layers (seed + first ``nuclear_layers - 1`` neighbour rings).

    Mutates ``owner`` in place.
    """
    nx, ny, nz = owner.shape
    voxels: list[tuple[int, int, int]] = [seed_voxel]
    nuclear_voxels: set[tuple[int, int, int]] = {seed_voxel}
    owner[seed_voxel] = cell_idx
    frontier = [seed_voxel]
    layer = 1  # seed is layer 0
    while len(voxels) < target and frontier:
        next_frontier: list[tuple[int, int, int]] = []
        for vx in frontier:
            for dx, dy, dz in _SIX_NEIGHBORS:
                nvx = (vx[0] + dx, vx[1] + dy, vx[2] + dz)
                if not (0 <= nvx[0] < nx and 0 <= nvx[1] < ny and 0 <= nvx[2] < nz):
                    continue
                if owner[nvx] != -1:
                    continue
                owner[nvx] = cell_idx
                voxels.append(nvx)
                if layer < nuclear_layers:
                    nuclear_voxels.add(nvx)
                next_frontier.append(nvx)
                if len(voxels) >= target:
                    break
            if len(voxels) >= target:
                break
        layer += 1
        frontier = next_frontier
    return voxels, nuclear_voxels


def make_synthetic_transcripts(
    n_cells: int = 8,
    voxels_per_cell_mean: int = 100,
    voxels_per_cell_jitter: float = 0.2,
    tx_per_cell: int = 25,
    n_genes: int = 12,
    n_types: int = 3,
    voxel_size_um: float = 1.0,
    domain_z_um: float = 10.0,
    nuclear_layers: int = 2,
    section_z_range_um: tuple[float, float] | None = None,
    cross_type_noise_pct: float = 0.20,
    enforce_inter_cell_gap: bool = False,
    max_cell_z_extent_um: float | None = None,
    n_z_strata: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a transcript-level DataFrame with voxel-grid cell shapes.

    Layout
    ------
    The 3D domain is gridded at ``voxel_size_um`` (default 1 µm).
    ``n_cells`` seed voxels are placed uniformly at random; each cell
    is grown by 6-connected BFS flood-fill until it reaches its target
    voxel count (drawn uniformly from
    ``[voxels_per_cell_mean × (1 ± voxels_per_cell_jitter)]``). Voxels
    are exclusively owned — adjacent cells touch face-to-face but
    never share a voxel.

    The first ``nuclear_layers`` BFS layers (seed + immediate ring +
    next ring at ``nuclear_layers=2``) are tagged as **nuclear**;
    transcripts from those voxels get ``is_nuclear=True``.

    xy domain auto-sized to fit cells with ~30 % slack:
    ``domain_xy_um = sqrt(n_cells × voxels_per_cell_mean × 1.3 / nz)``.

    Transcripts are placed at uniformly-random positions within their
    owning voxel (NOT snapped to bin centers).

    Each cell is assigned a type ``c % n_types``; ``tx_per_cell``
    transcripts are sampled, each gene drawn from the cell's archetype
    panel (``1 − cross_type_noise_pct`` weight) or a cross-type
    noise gene.

    Section extraction
    ------------------
    If ``section_z_range_um=(z_lo, z_hi)`` is set, transcripts with
    ``z`` outside that interval are dropped. Cells partially in the
    section keep their visible transcripts under the original
    ``cell_id``. The section boundary is independent of voxel
    boundaries.

    Returns
    -------
    df : pd.DataFrame
        Columns ``transcript_id`` (str), ``feature_name`` (str),
        ``cell_id`` (str — ground truth), ``x``, ``y``, ``z`` (float32),
        ``is_nuclear`` (bool).

    ground_truth : dict
        ``n_cells, n_types, voxel_size_um, domain_xy_um, domain_z_um,
        section_z_range_um, n_clipped_cells``,
        ``cell_to_type: dict[str -> int]``,
        ``type_to_genes: dict[int -> list[str]]``,
        ``gene_to_type: dict[str -> int]``,
        ``n_voxels_per_cell: dict[str -> int]``,
        ``n_nuclear_voxels_per_cell: dict[str -> int]``,
        ``cell_centers: list[(x, y, z)]`` (centroid of voxel set in µm).
    """
    rng = np.random.default_rng(seed)

    if n_genes % n_types != 0:
        raise ValueError(
            f"n_genes={n_genes} must be divisible by n_types={n_types} "
            "for clean archetype panels."
        )
    genes_per_type = n_genes // n_types
    gene_names = [f"gene_{i:02d}" for i in range(n_genes)]
    type_to_genes: dict[int, list[str]] = {
        t: gene_names[t * genes_per_type: (t + 1) * genes_per_type]
        for t in range(n_types)
    }
    gene_to_type = {g: t for t, gs in type_to_genes.items() for g in gs}

    # 1. Domain dimensions — 4× slack so seeds are well-spaced and
    # round-robin flood-fill produces compact (more spherical) cells
    # with multi-hop interior depth. At 2× slack the density was 44%
    # and most cells ended up amoeboid with max BFS depth ≤ 1, which
    # left no room for an interior nucleus.
    nz = max(1, int(round(domain_z_um / voxel_size_um)))
    target_volume = n_cells * voxels_per_cell_mean * 4.0
    n_xy = max(2, int(np.ceil(np.sqrt(target_volume / nz))))
    nx = ny = n_xy
    domain_xy_um = nx * voxel_size_um

    if nx * ny * nz < n_cells:
        raise ValueError(
            f"Domain ({nx}×{ny}×{nz}) too small for {n_cells} cell seeds; "
            "raise voxels_per_cell_mean or domain_z_um."
        )

    # 2. Ownership grid (-1 = unowned)
    owner = np.full((nx, ny, nz), -1, dtype=np.int32)

    # 3. Seed cells
    if n_z_strata is not None and n_z_strata > 1:
        # Stratified z-seeding: place cells in distinct z-layers at SHARED
        # xy positions, so cells stack vertically. Used to test the z-bin
        # filter in stitching — random seeding rarely produces z-stacked
        # cells given moderate cell counts.
        if n_cells % n_z_strata != 0:
            raise ValueError(
                f"n_cells ({n_cells}) must be divisible by "
                f"n_z_strata ({n_z_strata}) for symmetric stacking"
            )
        cells_per_stratum = n_cells // n_z_strata
        if cells_per_stratum > nx * ny:
            raise ValueError(
                f"cells_per_stratum ({cells_per_stratum}) exceeds available "
                f"xy positions ({nx * ny})"
            )
        # Pick random xy positions; use the same set in every stratum so
        # cells share xy across strata.
        xy_flat = rng.choice(nx * ny, size=cells_per_stratum, replace=False)
        stratum_height = nz / n_z_strata
        seeds = []
        for s in range(n_z_strata):
            z_seed = int((s + 0.5) * stratum_height)
            for xy_idx in xy_flat:
                x_seed = int(xy_idx // ny)
                y_seed = int(xy_idx % ny)
                seeds.append((x_seed, y_seed, z_seed))
        # Order: cell 0 = stratum 0 / xy 0, cell 1 = stratum 0 / xy 1, …,
        # cell cells_per_stratum = stratum 1 / xy 0, etc.
    else:
        flat_idxs = rng.choice(nx * ny * nz, size=n_cells, replace=False)
        seeds = [tuple(int(c) for c in np.unravel_index(i, (nx, ny, nz)))
                 for i in flat_idxs]

    # 4. Flood-fill each cell to its target size
    lo = max(1, int(voxels_per_cell_mean * (1 - voxels_per_cell_jitter)))
    hi = max(lo + 1, int(voxels_per_cell_mean * (1 + voxels_per_cell_jitter)) + 1)
    targets = rng.integers(lo, hi, size=n_cells)

    # Round-robin BFS: each cell expands one layer per round, so no
    # single cell hogs the volume. Boxed-in seeds still fail to grow,
    # but the round-robin ensures fairness vs sequential greedy.
    cell_voxels: list[list[tuple[int, int, int]]] = [[s] for s in seeds]
    cell_frontier: list[list[tuple[int, int, int]]] = [[s] for s in seeds]
    for c, s in enumerate(seeds):
        owner[s] = c

    def _has_other_cell_neighbor(nvx: tuple[int, int, int], c: int) -> bool:
        """Return True if any 6-neighbor of ``nvx`` is owned by a cell
        other than ``c``. Used to enforce a 1-voxel gap between cells."""
        for ddx, ddy, ddz in _SIX_NEIGHBORS:
            nbr = (nvx[0] + ddx, nvx[1] + ddy, nvx[2] + ddz)
            if not (0 <= nbr[0] < nx and 0 <= nbr[1] < ny and 0 <= nbr[2] < nz):
                continue
            o = owner[nbr]
            if o != -1 and o != c:
                return True
        return False

    # z-extent cap: cells can grow at most ±max_z_half_extent voxels
    # from their seed's z. Constrains cells to thin z bands so multiple
    # cells can stack at the same xy across distinct z layers — the
    # realistic dense-tissue case.
    if max_cell_z_extent_um is not None:
        max_z_half_extent = max(
            0, int(round(max_cell_z_extent_um / (2.0 * voxel_size_um)))
        )
    else:
        max_z_half_extent = None

    while True:
        progress = False
        for c in range(n_cells):
            if len(cell_voxels[c]) >= targets[c]:
                continue
            if not cell_frontier[c]:
                continue
            next_frontier: list[tuple[int, int, int]] = []
            for vx in cell_frontier[c]:
                for dx, dy, dz in _SIX_NEIGHBORS:
                    nvx = (vx[0] + dx, vx[1] + dy, vx[2] + dz)
                    if not (0 <= nvx[0] < nx and 0 <= nvx[1] < ny and 0 <= nvx[2] < nz):
                        continue
                    if owner[nvx] != -1:
                        continue
                    if enforce_inter_cell_gap and _has_other_cell_neighbor(nvx, c):
                        # Skip — claiming this voxel would put us face-to-face
                        # with a different cell. Leaves a 1-voxel buffer.
                        continue
                    if max_z_half_extent is not None:
                        if abs(nvx[2] - seeds[c][2]) > max_z_half_extent:
                            continue
                    owner[nvx] = c
                    cell_voxels[c].append(nvx)
                    next_frontier.append(nvx)
                    progress = True
                    if len(cell_voxels[c]) >= targets[c]:
                        break
                if len(cell_voxels[c]) >= targets[c]:
                    break
            cell_frontier[c] = next_frontier
        if not progress:
            break

    # 4b. Re-center the nucleus to the cell's deepest interior voxel.
    # The seed-based assignment in the BFS above places the nucleus
    # wherever the seed voxel happened to land — often near the cell
    # boundary if the cell grew preferentially in one direction. The
    # geometric centroid isn't enough either: for an elongated or
    # crescent-shaped cell, the centroid voxel can still be near the
    # surface (or even fall outside the voxel set entirely, in which
    # case "closest cell voxel" is on the boundary).
    #
    # Instead, pick the voxel whose **minimum distance to any non-cell
    # voxel** is maximum — the morphologically deepest interior point.
    # Computed via multi-source BFS from the boundary inward. This
    # guarantees the nucleus has at least ``min(deepest_dist,
    # nuclear_layers)`` voxels of cytoplasm padding to the cell
    # surface in every direction.
    cell_nuclear_voxels: list[set[tuple[int, int, int]]] = []
    for c in range(n_cells):
        voxels = cell_voxels[c]
        if not voxels:
            cell_nuclear_voxels.append(set())
            continue
        center_voxel = _deepest_voxel(voxels)
        cell_voxel_set = set(voxels)
        # BFS from center within the cell's voxels for nuclear_layers rounds.
        nuc: set[tuple[int, int, int]] = {center_voxel}
        frontier = [center_voxel]
        for _ in range(max(0, nuclear_layers - 1)):
            next_f: list[tuple[int, int, int]] = []
            for vx in frontier:
                for dx, dy, dz in _SIX_NEIGHBORS:
                    nvx = (vx[0] + dx, vx[1] + dy, vx[2] + dz)
                    if nvx in cell_voxel_set and nvx not in nuc:
                        nuc.add(nvx)
                        next_f.append(nvx)
            if not next_f:
                break
            frontier = next_f
        cell_nuclear_voxels.append(nuc)

    # 5. Place transcripts
    rows = []
    tx_id = 0
    cell_centers: list[tuple[float, float, float]] = []
    cell_to_type: dict[str, int] = {}
    n_voxels_per_cell: dict[str, int] = {}
    n_nuclear_voxels_per_cell: dict[str, int] = {}

    # Type assignment. In stratified-z mode, cells stacked at the same
    # xy share a type (so the z-bin filter has work to do — same-type
    # stacked cells can't be split by PMI alone). Otherwise, use the
    # standard round-robin c % n_types.
    if n_z_strata is not None and n_z_strata > 1:
        cells_per_stratum = n_cells // n_z_strata

        def _type_for(c_idx: int) -> int:
            xy_idx = c_idx % cells_per_stratum
            return xy_idx % n_types
    else:
        def _type_for(c_idx: int) -> int:
            return c_idx % n_types

    for c in range(n_cells):
        voxels = cell_voxels[c]
        nuclear = cell_nuclear_voxels[c]
        if not voxels:
            continue
        ctype = _type_for(c)
        cell_to_type[str(c)] = ctype
        n_voxels_per_cell[str(c)] = len(voxels)
        n_nuclear_voxels_per_cell[str(c)] = len(nuclear)
        # Centroid (µm)
        arr = np.asarray(voxels, dtype=np.float64)
        cx = float((arr[:, 0].mean() + 0.5) * voxel_size_um)
        cy = float((arr[:, 1].mean() + 0.5) * voxel_size_um)
        cz = float((arr[:, 2].mean() + 0.5) * voxel_size_um)
        cell_centers.append((cx, cy, cz))

        archetype_genes = type_to_genes[ctype]
        other_genes = [g for g in gene_names if g not in archetype_genes]

        for _ in range(tx_per_cell):
            voxel = voxels[int(rng.integers(len(voxels)))]
            x = (voxel[0] + rng.random()) * voxel_size_um
            y = (voxel[1] + rng.random()) * voxel_size_um
            z = (voxel[2] + rng.random()) * voxel_size_um
            if rng.random() < cross_type_noise_pct:
                gene = other_genes[int(rng.integers(len(other_genes)))]
            else:
                gene = archetype_genes[int(rng.integers(len(archetype_genes)))]
            is_nuc = voxel in nuclear
            rows.append((
                str(tx_id), gene, str(c),
                float(x), float(y), float(z), bool(is_nuc),
            ))
            tx_id += 1

    df = pd.DataFrame(
        rows,
        columns=["transcript_id", "feature_name", "cell_id",
                 "x", "y", "z", "is_nuclear"],
    )
    df = df.astype({"x": np.float32, "y": np.float32, "z": np.float32})

    # 6. Optional tissue-section extraction (independent of voxel grid)
    n_clipped_cells = 0
    if section_z_range_um is not None:
        z_lo, z_hi = section_z_range_um
        cells_before = set(df["cell_id"].astype(str))
        df = df[(df["z"] >= z_lo) & (df["z"] < z_hi)].reset_index(drop=True)
        cells_after = set(df["cell_id"].astype(str))
        # Cells whose tx are partially clipped: present in both, but
        # in cells_after with fewer rows. Approximate: count cells
        # entirely lost vs partially lost.
        for cid in cells_before:
            n_full = sum(1 for c in range(n_cells) if str(c) == cid) * tx_per_cell
            n_remaining = int((df["cell_id"].astype(str) == cid).sum())
            if 0 < n_remaining < n_full:
                n_clipped_cells += 1

    ground_truth: dict[str, Any] = {
        "n_cells": n_cells,
        "n_types": n_types,
        "voxel_size_um": voxel_size_um,
        "domain_xy_um": domain_xy_um,
        "domain_z_um": domain_z_um,
        "section_z_range_um": section_z_range_um,
        "n_clipped_cells": n_clipped_cells,
        "cell_centers": cell_centers,
        "cell_to_type": cell_to_type,
        "type_to_genes": type_to_genes,
        "gene_to_type": gene_to_type,
        "n_voxels_per_cell": n_voxels_per_cell,
        "n_nuclear_voxels_per_cell": n_nuclear_voxels_per_cell,
    }
    return df, ground_truth


def make_synthetic_npmi_panel_for_transcripts(
    df: pd.DataFrame,
    ground_truth: dict[str, Any],
    *,
    same_type_pmi: float = 1.0,
    cross_type_pmi: float = -1.0,
    metric: str = "pmi",
) -> pd.DataFrame:
    """Build a long-format NPMI / PMI panel matching the planted structure.

    For every (g_i, g_j) pair in the transcript dataframe's gene
    vocabulary, emit one row:

      - within the same archetype panel → ``same_type_pmi``
      - across panels → ``cross_type_pmi``

    The column name ``NPMI`` is used (matching the legacy CSV convention
    that ``prune_transcripts_fast`` and other downstream stages expect).
    The values are interpreted in whatever metric the pipeline is configured
    for (``metric`` is for documentation only — the column name is fixed).

    Defaults (``+1.0`` / ``-1.0``) are calibrated for ``metric="pmi"``:
    ``+1.0 > ln(1.5) ≈ 0.405`` and ``-1.0 < ln(1/3) ≈ -1.099`` — clearly on
    the positive / negative side of the modern thresholds.
    """
    _ = metric  # for documentation; column name is fixed.
    gene_to_type = ground_truth["gene_to_type"]
    genes = sorted(set(df["feature_name"].astype(str)))

    rows = []
    for i, gi in enumerate(genes):
        for gj in genes[i + 1:]:
            ti = gene_to_type.get(gi)
            tj = gene_to_type.get(gj)
            if ti is not None and tj is not None and ti == tj:
                v = same_type_pmi
            else:
                v = cross_type_pmi
            rows.append((gi, gj, float(v)))
    return pd.DataFrame(rows, columns=["gene_i", "gene_j", "NPMI"])
