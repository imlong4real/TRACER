"""Tests for the Mahalanobis-D RESCUE in Stitch.

The rescue overrides a ΔC reject when:
    rescue_delta_c_floor < ΔC < 0   AND   D ≤ mahalanobis_d_rescue

i.e. when composition borderline-rejects but the two tx clouds are
geometrically enmeshed (low Maha D ⇒ one cell with anti-correlated
sub-programs, not two distinct cells). The ΔC floor protects against
fusing engulfment doublets where composition rejects strongly.

Canonical cases (DESIGN doc):
  EMT cell, ΔC = −0.10, D = 0.0    → ACCEPT via rescue
  jikammne doublet, ΔC = −0.49, D = 0.5  → REJECT (ΔC ≤ floor)
  jiecahje CAF+TAM, ΔC ≪ 0, D = 1.59     → REJECT (ΔC ≤ floor AND D too high)
  Two adjacent M2 macrophages, ΔC = +0.05, D = 2.0 → ACCEPT (ΔC ≥ 0; rescue logic doesn't apply)
  Same-cell fragment rejoin, ΔC = +0.5, D = 0.3 → ACCEPT (ΔC ≥ 0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracer.config import StitchConfig
from tracer.stitching import (
    _LAST_GATE_STATS,
    apply_stitching_to_transcripts_memory_efficient,
)


# ---------------------------------------------------------------------
# Helpers — partial-overlap gene panels giving tunable ΔC.
#
# Two entities A and B share a fraction of their gene panels. The PMI
# matrix is +0.6 within either program (block A and block B), and
# `cross` between cross-program genes. Tunable ΔC:
#
#   overlap=0.70, cross=+0.00  → ΔC ≈ -0.065 (in EMT rescue zone)
#   overlap=0.50, cross=-0.40  → ΔC ≈ -0.426 (below default floor -0.2)
#   overlap=0.95, cross=+0.30  → ΔC ≈ +0.050 (positive zone)
# ---------------------------------------------------------------------

def _build_aux_and_pools(prog_size: int = 10,
                        overlap_frac: float = 0.70,
                        cross: float = 0.0,
                        within: float = 0.6) -> tuple[dict, list, list]:
    """Build the PMI aux and per-entity gene pools.

    Returns (aux, pool_A, pool_B) where pool_X is the list of gene
    names entity X should draw transcripts from.
    """
    shared = int(round(overlap_frac * prog_size))
    unique_a = prog_size - shared
    unique_b = prog_size - shared
    G = unique_a + shared + unique_b
    # Layout: [unique_a | shared | unique_b]
    genes = [f"G{i:03d}" for i in range(G)]
    a_idx = list(range(0, unique_a + shared))
    b_idx = list(range(unique_a, unique_a + shared + unique_b))
    pool_A = [genes[i] for i in a_idx]
    pool_B = [genes[i] for i in b_idx]

    W = np.full((G, G), float(cross), dtype=np.float32)
    for i in a_idx:
        W[i, a_idx] = within
    for i in b_idx:
        W[i, b_idx] = within
    np.fill_diagonal(W, np.nan)
    g2i = {g: i for i, g in enumerate(genes)}
    aux = {"W": W, "gene_to_idx": g2i}
    return aux, pool_A, pool_B


def _gauss_entity(label: str, n: int, center, sigma, gene_pool, rng):
    """Build a transcript-level fragment for one entity (etype=partial,
    so two of them are stitch-eligible — Stitch refuses to merge two
    cells)."""
    pts = rng.normal(loc=center, scale=sigma, size=(n, 3))
    return pd.DataFrame({
        "transcript_id": np.arange(n) + rng.integers(0, 10_000_000),
        "tracer_id": [label] * n,
        # Force at least one of each gene in the pool so the entity's
        # gene-set is exactly the pool (rather than an RNG subsample).
        "feature_name": (
            list(gene_pool)
            + list(rng.choice(gene_pool, size=max(0, n - len(gene_pool))))
        )[:n],
        "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
        "_etype": ["partial"] * n,
    })


# ---------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------

def test_config_default_ships_on():
    """Default Stitch Mahalanobis-D rescue ships at D=1.0."""
    cfg = StitchConfig()
    assert cfg.mahalanobis_d_rescue == 1.0
    assert cfg.rescue_delta_c_floor == -0.2


def test_config_can_disable():
    """Setting mahalanobis_d_rescue to None disables the rescue (back-compat)."""
    cfg = StitchConfig(mahalanobis_d_rescue=None)
    assert cfg.mahalanobis_d_rescue is None


def test_config_validates_positive():
    with pytest.raises(ValueError):
        StitchConfig(mahalanobis_d_rescue=0.0)
    with pytest.raises(ValueError):
        StitchConfig(mahalanobis_d_rescue=-1.0)
    # > 0 ok
    StitchConfig(mahalanobis_d_rescue=1e-6)
    StitchConfig(mahalanobis_d_rescue=1.0)
    StitchConfig(mahalanobis_d_rescue=None)


def test_config_floor_must_be_nonpositive():
    # Positive floor not allowed (rescue zone is ΔC < 0 by definition)
    with pytest.raises(ValueError):
        StitchConfig(rescue_delta_c_floor=0.5)
    # Zero is fine (edge)
    StitchConfig(rescue_delta_c_floor=0.0)
    # Negative is fine
    StitchConfig(rescue_delta_c_floor=-0.5)
    # Non-finite rejected
    with pytest.raises(ValueError):
        StitchConfig(rescue_delta_c_floor=float("nan"))
    with pytest.raises(ValueError):
        StitchConfig(rescue_delta_c_floor=float("-inf"))


# ---------------------------------------------------------------------
# EMT proxy — concentric clouds, borderline-negative ΔC (in band).
# Without rescue: split. With rescue: merge.
# ---------------------------------------------------------------------

def test_emt_concentric_rescue_fires():
    """Two concentric clouds (D ≈ 0) with overlap=0.70 / cross=0.0 →
    ΔC ≈ -0.065 (in (-0.2, 0)). Without rescue → split; with rescue
    (d=1.0) → merge."""
    rng = np.random.default_rng(0)
    aux, pool_A, pool_B = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.70, cross=0.0,
    )

    a = _gauss_entity("a-1-1", 60, center=(0.0, 0.0, 0.0),
                      sigma=2.0, gene_pool=pool_A, rng=rng)
    b = _gauss_entity("b-1-1", 60, center=(0.0, 0.0, 0.0),
                      sigma=2.0, gene_pool=pool_B, rng=rng)
    df = pd.concat([a, b], ignore_index=True)

    common_kwargs = dict(
        candidate_source="grid", G=2.0, G_z=2.0, z_neighbor_depth=1,
        dist_threshold=10.0,
        deltaC_min=0.03,
        max_merger_depth=None,
        c_union_bypass=None,    # disable bypass so only ΔC/rescue decides
        show_progress=False,
    )

    # WITHOUT rescue → 2 entities.
    df_off, _ = apply_stitching_to_transcripts_memory_efficient(
        df.copy(), aux=aux, **common_kwargs,
    )
    n_off = df_off["tracer_id"].astype(str).nunique()
    assert n_off == 2, (
        f"Without rescue, ΔC in (-0.2, 0) should keep them split; "
        f"got n_ent={n_off}, stats={dict(_LAST_GATE_STATS)}"
    )
    assert _LAST_GATE_STATS.get("mahalanobis_rescues", 0) == 0
    assert _LAST_GATE_STATS.get("mahalanobis_rescue_checks", 0) == 0

    # WITH rescue at d=1.0 (concentric → D ≈ 0 ≤ 1.0; ΔC in band).
    df_on, _ = apply_stitching_to_transcripts_memory_efficient(
        df.copy(), aux=aux,
        mahalanobis_d_rescue=1.0,
        rescue_delta_c_floor=-0.2,
        **common_kwargs,
    )
    n_on = df_on["tracer_id"].astype(str).nunique()
    assert n_on == 1, (
        f"With Maha rescue (D≈0 ≤ 1.0, ΔC in rescue band), the "
        f"EMT-like merger should fire; got n_ent={n_on}, "
        f"stats={dict(_LAST_GATE_STATS)}"
    )
    assert _LAST_GATE_STATS.get("mahalanobis_rescues", 0) >= 1


# ---------------------------------------------------------------------
# Engulfment proxy — strongly-negative ΔC, low D.
# Default floor (-0.2) protects → no rescue.
# Permissive floor (-1.0) would rescue — confirms the floor is what
# protects, not some other mechanism.
# ---------------------------------------------------------------------

def test_engulfment_doublet_protected_by_floor():
    """overlap=0.50 / cross=-0.4 → ΔC ≈ -0.426 (well below -0.2 floor).
    Concentric → D low. Default floor → split; permissive floor → merge."""
    rng = np.random.default_rng(1)
    aux, pool_A, pool_B = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.50, cross=-0.4,
    )

    a = _gauss_entity("a-1-1", 60, center=(0.0, 0.0, 0.0),
                      sigma=2.0, gene_pool=pool_A, rng=rng)
    b = _gauss_entity("b-1-1", 60, center=(0.3, 0.0, 0.0),
                      sigma=2.0, gene_pool=pool_B, rng=rng)
    df = pd.concat([a, b], ignore_index=True)

    common_kwargs = dict(
        candidate_source="grid", G=2.0, G_z=2.0, z_neighbor_depth=1,
        dist_threshold=10.0,
        deltaC_min=0.03,
        max_merger_depth=None,
        c_union_bypass=None,
        show_progress=False,
        mahalanobis_d_rescue=1.0,
    )

    # Default floor (-0.2) → ΔC ≈ -0.4 below floor → no rescue.
    df_floor, _ = apply_stitching_to_transcripts_memory_efficient(
        df.copy(), aux=aux,
        rescue_delta_c_floor=-0.2,
        **common_kwargs,
    )
    n_floor = df_floor["tracer_id"].astype(str).nunique()
    assert n_floor == 2, (
        f"With ΔC well below the -0.2 floor, the rescue must not fire; "
        f"got n_ent={n_floor}, stats={dict(_LAST_GATE_STATS)}"
    )
    assert _LAST_GATE_STATS.get("mahalanobis_rescues", 0) == 0

    # Permissive floor (-1.0) → ΔC ≈ -0.4 now in band AND D low → rescue.
    df_loose, _ = apply_stitching_to_transcripts_memory_efficient(
        df.copy(), aux=aux,
        rescue_delta_c_floor=-1.0,
        **common_kwargs,
    )
    n_loose = df_loose["tracer_id"].astype(str).nunique()
    assert n_loose == 1, (
        f"Permissive floor (-1.0) should let the rescue fire on this "
        f"low-D pair; got n_ent={n_loose}, stats={dict(_LAST_GATE_STATS)}"
    )
    assert _LAST_GATE_STATS.get("mahalanobis_rescues", 0) >= 1


# ---------------------------------------------------------------------
# Side-by-side proxy — high D, borderline ΔC. Rescue must NOT fire.
# ---------------------------------------------------------------------

def test_side_by_side_blocked_by_high_d():
    """ΔC borderline (in (-0.2, 0)), but D is large because clouds are
    offset. Rescue must NOT fire (D > threshold)."""
    rng = np.random.default_rng(2)
    aux, pool_A, pool_B = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.70, cross=0.0,
    )

    # Side-by-side along x: offset 6 µm, tight σ → D large.
    a = _gauss_entity("a-1-1", 60, center=(0.0, 0.0, 0.0),
                      sigma=1.0, gene_pool=pool_A, rng=rng)
    b = _gauss_entity("b-1-1", 60, center=(6.0, 0.0, 0.0),
                      sigma=1.0, gene_pool=pool_B, rng=rng)
    df = pd.concat([a, b], ignore_index=True)

    df_strict, _ = apply_stitching_to_transcripts_memory_efficient(
        df.copy(), aux=aux,
        candidate_source="grid", G=2.0, G_z=2.0, z_neighbor_depth=1,
        dist_threshold=10.0,
        deltaC_min=0.03,
        max_merger_depth=None,
        c_union_bypass=None,
        show_progress=False,
        mahalanobis_d_rescue=1.0,       # strict
        rescue_delta_c_floor=-0.2,
    )
    n_strict = df_strict["tracer_id"].astype(str).nunique()
    assert n_strict == 2, (
        f"Side-by-side clouds (high D) should not rescue; got "
        f"n_ent={n_strict}, stats={dict(_LAST_GATE_STATS)}"
    )
    assert _LAST_GATE_STATS.get("mahalanobis_rescues", 0) == 0


# ---------------------------------------------------------------------
# Positive ΔC — composition decides; rescue logic does not apply.
# ---------------------------------------------------------------------

def test_positive_delta_c_accepts_independent_of_d():
    """ΔC ≥ 0 → composition accepts. Rescue must not even be consulted,
    regardless of D."""
    rng = np.random.default_rng(3)
    # Use overlap 0.95 / cross +0.30 → ΔC ≈ +0.05.
    aux, pool_A, pool_B = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.95, cross=0.30,
    )

    # Spatially well-separated → D would be high if consulted.
    a = _gauss_entity("a-1-1", 60, center=(0.0, 0.0, 0.0),
                      sigma=1.0, gene_pool=pool_A, rng=rng)
    b = _gauss_entity("b-1-1", 60, center=(5.0, 0.0, 0.0),
                      sigma=1.0, gene_pool=pool_B, rng=rng)
    df = pd.concat([a, b], ignore_index=True)

    df_out, _ = apply_stitching_to_transcripts_memory_efficient(
        df.copy(), aux=aux,
        candidate_source="grid", G=2.0, G_z=2.0, z_neighbor_depth=1,
        dist_threshold=10.0,
        deltaC_min=0.03,
        max_merger_depth=None,
        c_union_bypass=None,
        show_progress=False,
        mahalanobis_d_rescue=1.0,
        rescue_delta_c_floor=-0.2,
    )
    n_out = df_out["tracer_id"].astype(str).nunique()
    assert n_out == 1, (
        f"Positive ΔC should accept regardless of D; got n_ent={n_out}, "
        f"stats={dict(_LAST_GATE_STATS)}"
    )
    # Rescue must not have been consulted (positive zone bypasses it).
    assert _LAST_GATE_STATS.get("mahalanobis_rescues", 0) == 0


def test_default_none_does_not_invoke_rescue():
    """With mahalanobis_d_rescue=None (default), the rescue check is
    never invoked. Guards back-compat."""
    rng = np.random.default_rng(4)
    aux, pool_A, pool_B = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.70, cross=0.0,
    )
    a = _gauss_entity("a-1-1", 60, center=(0.0, 0.0, 0.0),
                      sigma=2.0, gene_pool=pool_A, rng=rng)
    b = _gauss_entity("b-1-1", 60, center=(0.0, 0.0, 0.0),
                      sigma=2.0, gene_pool=pool_B, rng=rng)
    df = pd.concat([a, b], ignore_index=True)
    apply_stitching_to_transcripts_memory_efficient(
        df.copy(), aux=aux,
        candidate_source="grid", G=2.0, G_z=2.0, z_neighbor_depth=1,
        dist_threshold=10.0,
        deltaC_min=0.03,
        max_merger_depth=None,
        c_union_bypass=None,
        show_progress=False,
        # mahalanobis_d_rescue not passed → defaults to None
    )
    assert _LAST_GATE_STATS.get("mahalanobis_rescues", 0) == 0
    assert _LAST_GATE_STATS.get("mahalanobis_rescue_checks", 0) == 0


# ---------------------------------------------------------------------
# Post-merge _etype homogenization in Stitch (the original Long bug)
# ---------------------------------------------------------------------

def test_stitch_homogenizes_merged_etype():
    """Merging a cell with a partial yields a stitched label whose rows
    share a single `_etype` (cell wins). Reproduces the failure mode
    where a heterogeneous merged entity makes `groupby.first()`
    non-deterministic and bypasses the cell-cell merge gate."""
    from tracer._etype import ETYPE_DTYPE
    rng = np.random.default_rng(0)
    aux, pool_A, pool_B = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.70, cross=0.0,
    )
    a = _gauss_entity("a-1-1", 60, center=(0.0, 0.0, 0.0),
                      sigma=2.0, gene_pool=pool_A, rng=rng)
    b = _gauss_entity("b-1-1", 60, center=(0.0, 0.0, 0.0),
                      sigma=2.0, gene_pool=pool_B, rng=rng)
    a["_etype"] = "cell"   # A is a cell; B keeps the default "partial"
    df = pd.concat([a, b], ignore_index=True)
    df["_etype"] = df["_etype"].astype(ETYPE_DTYPE)

    df_on, _ = apply_stitching_to_transcripts_memory_efficient(
        df.copy(), aux=aux,
        mahalanobis_d_rescue=1.0, rescue_delta_c_floor=-0.2,
        candidate_source="grid", G=2.0, G_z=2.0, z_neighbor_depth=1,
        dist_threshold=10.0, deltaC_min=0.03, max_merger_depth=None,
        c_union_bypass=None, show_progress=False,
    )
    lab = df_on["tracer_id"].astype(str)
    assert lab.nunique() == 1, (
        f"cell+partial should merge into one entity; got n={lab.nunique()}"
    )
    per_label = df_on.groupby(lab)["_etype"].agg(lambda s: set(s.astype(str)))
    assert per_label.iloc[0] == {"cell"}, (
        f"merged _etype not homogenized: {per_label.iloc[0]}"
    )
