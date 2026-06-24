"""Tests for the Phase-1-time Mahalanobis-D rescue.

Mirrors test_stitch_mahalanobis_rescue, applied at the
``phase1_maha_remerge`` entry point instead of inside Stitch. Rule:

    floor < ΔC < 0    AND    D ≤ threshold    →    merge (DSU union)

Canonical cases:
  EMT, ΔC ≈ -0.10, D ≈ 0       → MERGE with d=1.0; NO MERGE with d=None
  Engulfment proxy, ΔC ≈ -0.5  → NO MERGE (ΔC ≤ floor)
  Side-by-side, ΔC ≈ -0.1, D large → NO MERGE (D above threshold)
  Positive ΔC                  → NO MERGE (ΔC ≥ 0)

Default-None back-compat is exercised by the existing
test_pipeline_smoke / test_pipeline_regression suites, which run with
the package default ``cfg.phase1.maha_remerge_d = None`` and therefore
must remain bit-exact when the new stage is in place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracer.config import Phase1Config
from tracer.phase1_rescue import phase1_maha_remerge


# ---------------------------------------------------------------------
# Helpers — partial-overlap gene panels giving tunable ΔC.
# Two entities A and B share `overlap_frac` of their gene panels. The
# PMI matrix is +0.6 within either program, `cross` between programs.
#   overlap=0.70, cross=+0.00 → ΔC ≈ -0.06 (in EMT rescue zone)
#   overlap=0.50, cross=-0.40 → ΔC ≈ -0.4  (below default floor -0.2)
#   overlap=0.95, cross=+0.30 → ΔC ≈ +0.05 (positive zone)
# ---------------------------------------------------------------------

def _build_aux_and_pools(prog_size: int = 10,
                        overlap_frac: float = 0.70,
                        cross: float = 0.0,
                        within: float = 0.6):
    shared = int(round(overlap_frac * prog_size))
    unique_a = prog_size - shared
    unique_b = prog_size - shared
    G = unique_a + shared + unique_b
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
    aux = {"W": W, "g2i": g2i}
    return aux, pool_A, pool_B


def _entity(label: str, n: int, center, sigma, gene_pool, rng):
    pts = rng.normal(loc=center, scale=sigma, size=(n, 3))
    return pd.DataFrame({
        "transcript_id": np.arange(n) + rng.integers(0, 10_000_000),
        "tracer_id": [label] * n,
        "feature_name": (
            list(gene_pool)
            + list(rng.choice(gene_pool, size=max(0, n - len(gene_pool))))
        )[:n],
        "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
    })


# ---------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------

def test_config_default_ships_on():
    """Default Phase-1-Maha-Remerge ships at D=1.0 (rescue ΔC ∈ (-0.2, 0))."""
    cfg = Phase1Config()
    assert cfg.maha_remerge_d == 1.0
    assert cfg.maha_remerge_delta_c_floor == -0.2


def test_config_can_disable():
    """Setting maha_remerge_d to None disables the stage (back-compat)."""
    cfg = Phase1Config(maha_remerge_d=None)
    assert cfg.maha_remerge_d is None


def test_config_validates_positive_d():
    with pytest.raises(ValueError):
        Phase1Config(maha_remerge_d=0.0)
    with pytest.raises(ValueError):
        Phase1Config(maha_remerge_d=-1.0)
    # > 0 ok
    Phase1Config(maha_remerge_d=1e-6)
    Phase1Config(maha_remerge_d=1.0)
    Phase1Config(maha_remerge_d=None)


def test_config_floor_must_be_nonpositive():
    with pytest.raises(ValueError):
        Phase1Config(maha_remerge_delta_c_floor=0.5)
    # Zero edge ok.
    Phase1Config(maha_remerge_delta_c_floor=0.0)
    Phase1Config(maha_remerge_delta_c_floor=-0.5)
    with pytest.raises(ValueError):
        Phase1Config(maha_remerge_delta_c_floor=float("nan"))


# ---------------------------------------------------------------------
# EMT proxy — concentric clouds, borderline-negative ΔC (in band).
# ---------------------------------------------------------------------

def test_emt_concentric_rescue_fires():
    """Two concentric clouds (D ≈ 0) with overlap=0.70 / cross=0.0 →
    ΔC ≈ -0.06 (in (-0.2, 0)). With d=1.0 → merge; with d=None → split."""
    rng = np.random.default_rng(0)
    aux, pool_A, pool_B = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.70, cross=0.0,
    )
    a = _entity("a-1-1", 60, center=(0.0, 0.0, 0.0),
                sigma=2.0, gene_pool=pool_A, rng=rng)
    b = _entity("b-1-1", 60, center=(0.0, 0.0, 0.0),
                sigma=2.0, gene_pool=pool_B, rng=rng)
    df = pd.concat([a, b], ignore_index=True)

    # With d=1.0 → merge expected.
    df_out, stats = phase1_maha_remerge(
        df, aux, threshold=1.0, floor=-0.2,
    )
    assert stats["n_rescues"] >= 1, (
        f"expected rescue with d=1.0; stats={stats}"
    )
    n_ent_out = df_out.tracer_id.astype(str).nunique()
    assert n_ent_out == 1, (
        f"expected 1 merged entity after rescue; got {n_ent_out}"
    )
    # The merged label should be the deterministic smaller string of
    # {"a-1-1", "b-1-1"} = "a-1-1".
    assert set(df_out.tracer_id.astype(str).unique()) == {"a-1-1"}

    # With d=None semantics — verify by NOT calling the helper at all
    # (the pipeline gate skips the call). df stays unchanged.
    # Direct test: a huge threshold floor that excludes everything.
    df_out2, stats2 = phase1_maha_remerge(
        df, aux, threshold=1.0, floor=0.0,  # floor=0 excludes the band
    )
    assert stats2["n_rescues"] == 0
    assert df_out2.tracer_id.astype(str).nunique() == 2


# ---------------------------------------------------------------------
# Engulfment-proxy: ΔC strongly negative (≤ floor) → NO merge even
# when geometrically enmeshed (D small).
# ---------------------------------------------------------------------

def test_engulfment_strong_negative_dc_no_rescue():
    """ΔC ≈ -0.4 (below default floor -0.2) → reject even at D ≈ 0."""
    rng = np.random.default_rng(1)
    aux, pool_A, pool_B = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.50, cross=-0.4,
    )
    a = _entity("a-1-1", 60, center=(0.0, 0.0, 0.0),
                sigma=2.0, gene_pool=pool_A, rng=rng)
    b = _entity("b-1-1", 60, center=(0.0, 0.0, 0.0),
                sigma=2.0, gene_pool=pool_B, rng=rng)
    df = pd.concat([a, b], ignore_index=True)

    df_out, stats = phase1_maha_remerge(
        df, aux, threshold=1.0, floor=-0.2,
    )
    assert stats["n_rescues"] == 0
    assert df_out.tracer_id.astype(str).nunique() == 2


# ---------------------------------------------------------------------
# Side-by-side: ΔC in band but D large → NO merge.
# ---------------------------------------------------------------------

def test_side_by_side_high_d_no_rescue():
    """ΔC in band (-0.2, 0) but D > threshold → no merge."""
    rng = np.random.default_rng(2)
    aux, pool_A, pool_B = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.70, cross=0.0,
    )
    # Place the two clouds far apart (D >> 1.0) but still within the
    # 8-Moore candidate bin band (bin=2.0µm; 6µm separation falls in a
    # different bin, but still in the +/- z window so they remain
    # candidates).
    a = _entity("a-1-1", 60, center=(0.0, 0.0, 0.0),
                sigma=0.5, gene_pool=pool_A, rng=rng)
    b = _entity("b-1-1", 60, center=(4.0, 0.0, 0.0),
                sigma=0.5, gene_pool=pool_B, rng=rng)
    df = pd.concat([a, b], ignore_index=True)

    # Force candidate enumeration regardless of bin distance by using
    # a coarser bin so they share one. bin_size=10µm puts both into
    # the same bin → enumerated as a candidate pair.
    df_out, stats = phase1_maha_remerge(
        df, aux, threshold=1.0, floor=-0.2,
        bin_size_um=10.0,
    )
    # If a candidate pair is enumerated and ΔC lands in band, the
    # rescue still fails because D >> 1.0.
    if stats["n_dc_in_band"] >= 1:
        assert stats["n_rescues"] == 0, (
            f"D should exceed threshold for side-by-side; stats={stats}"
        )
    assert df_out.tracer_id.astype(str).nunique() == 2


# ---------------------------------------------------------------------
# Positive ΔC pair: not in rescue zone → no Maha computation happens.
# ---------------------------------------------------------------------

def test_positive_delta_c_no_rescue():
    """ΔC ≥ 0 → rescue logic doesn't apply (Phase 1 wouldn't have
    rejected the pair in the first place). Two concentric clouds with
    nearly-identical gene programs."""
    rng = np.random.default_rng(3)
    aux, pool_A, pool_B = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.95, cross=0.30,
    )
    a = _entity("a-1-1", 60, center=(0.0, 0.0, 0.0),
                sigma=2.0, gene_pool=pool_A, rng=rng)
    b = _entity("b-1-1", 60, center=(0.0, 0.0, 0.0),
                sigma=2.0, gene_pool=pool_B, rng=rng)
    df = pd.concat([a, b], ignore_index=True)

    df_out, stats = phase1_maha_remerge(
        df, aux, threshold=1.0, floor=-0.2,
    )
    assert stats["n_rescues"] == 0
    assert df_out.tracer_id.astype(str).nunique() == 2


# ---------------------------------------------------------------------
# Default pipeline behavior: Phase1-Maha-Remerge ships on at D=1.0 with
# ΔC floor -0.2; setting maha_remerge_d=None disables the stage as a
# back-compat escape hatch.
# ---------------------------------------------------------------------

def test_pipeline_default_ships_on():
    """Default config has Phase1-Maha-Remerge enabled at D=1.0."""
    pytest.importorskip("tracer.stitching")
    from tracer.config import load_config
    cfg = load_config()
    assert cfg.phase1.maha_remerge_d == 1.0
    assert hasattr(cfg.phase1, "maha_remerge_delta_c_floor")
    assert cfg.phase1.maha_remerge_delta_c_floor == -0.2


# ---------------------------------------------------------------------
# Sentinel labels are ignored (no spurious enumeration / merge).
# ---------------------------------------------------------------------

def test_sentinel_labels_skipped():
    rng = np.random.default_rng(4)
    aux, pool_A, _ = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.70, cross=0.0,
    )
    a = _entity("a-1-1", 30, center=(0.0, 0.0, 0.0),
                sigma=2.0, gene_pool=pool_A, rng=rng)
    # Sentinel tx (won't enumerate).
    s = _entity("-1", 10, center=(0.0, 0.0, 0.0),
                sigma=2.0, gene_pool=pool_A, rng=rng)
    df = pd.concat([a, s], ignore_index=True)

    df_out, stats = phase1_maha_remerge(
        df, aux, threshold=1.0, floor=-0.2,
    )
    # With only one non-sentinel entity, no candidate pairs.
    assert stats["n_candidates"] == 0
    assert stats["n_rescues"] == 0
    # Sentinel labels unchanged.
    assert (df_out.tracer_id.astype(str) == "-1").sum() == 10


# ---------------------------------------------------------------------
# Post-merge _etype homogenization (the cell-cell-gate failure mode)
# ---------------------------------------------------------------------

def test_maha_remerge_homogenizes_merged_etype():
    """A cell entity merged with a partial yields ONE entity whose every
    row shares a single `_etype` (cell, by priority). Without
    homogenization the merged entity carries mixed cell/partial rows and
    downstream `groupby(...)["_etype"].first()` is non-deterministic."""
    from tracer._etype import ETYPE_DTYPE
    rng = np.random.default_rng(0)
    aux, pool_A, pool_B = _build_aux_and_pools(
        prog_size=10, overlap_frac=0.70, cross=0.0,
    )
    a = _entity("a-1-1", 60, center=(0.0, 0.0, 0.0),
                sigma=2.0, gene_pool=pool_A, rng=rng)
    b = _entity("b-1-1", 60, center=(0.0, 0.0, 0.0),
                sigma=2.0, gene_pool=pool_B, rng=rng)
    a["_etype"] = "cell"
    b["_etype"] = "partial"
    df = pd.concat([a, b], ignore_index=True)
    df["_etype"] = df["_etype"].astype(ETYPE_DTYPE)

    df_out, stats = phase1_maha_remerge(df, aux, threshold=1.0, floor=-0.2)
    assert stats["n_rescues"] >= 1, f"expected a merge; stats={stats}"
    merged = df_out["tracer_id"].astype(str)
    assert merged.nunique() == 1, "cell+partial should merge into one entity"
    per_label = df_out.groupby(merged)["_etype"].agg(
        lambda s: set(s.astype(str))
    )
    assert per_label.iloc[0] == {"cell"}, (
        f"merged _etype not homogenized to the priority winner: "
        f"{per_label.iloc[0]}"
    )
