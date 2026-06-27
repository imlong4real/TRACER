"""Tests for `tracer.zscale.resolve_g_z_um` — platform-aware z-bin sizing.

Covers the heuristic branches with synthetic z distributions modelled on
the real platforms:
  * Xenium     — continuous z (many distinct values)  → auto keeps 1.0
  * MERFISH    — 9 planes spaced 1.5 µm               → auto = 1.5
  * CosMx      — 11 planes spaced 0.8 µm              → auto = 0.8
  * degenerate — single-plane / missing z             → 2D, depth 0
plus explicit-float honoring, the spacing-mismatch warning, and the
legacy `None` passthrough.
"""
from __future__ import annotations

import numpy as np
import pytest

from tracer.zscale import ZScaleResolution, resolve_g_z_um, z_plane_stats


# Synthetic z arrays modelled on the real datasets.
def _planes(values, per_plane=50, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    z = np.repeat(np.asarray(values, dtype=float), per_plane)
    if jitter:
        z = z + rng.normal(0, jitter, size=z.size)
    return z


MERFISH_Z = _planes([2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0, 14.5])
COSMX_Z = _planes([-0.8, 0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4, 7.2])
XENIUM_Z = np.random.default_rng(0).uniform(10.0, 30.0, size=20000)  # continuous


# --------------------------------------------------------------------------
# z_plane_stats
# --------------------------------------------------------------------------
def test_plane_stats_discrete():
    n, sp = z_plane_stats(MERFISH_Z)
    assert n == 9
    assert sp == pytest.approx(1.5, abs=1e-6)
    n, sp = z_plane_stats(COSMX_Z)
    assert n == 11
    assert sp == pytest.approx(0.8, abs=1e-6)


def test_plane_stats_continuous_and_empty():
    n, sp = z_plane_stats(XENIUM_Z)
    assert n > 1000
    assert np.isfinite(sp)
    # None / empty → no planes, nan spacing.
    n_none, sp_none = z_plane_stats(None)
    assert n_none == 0 and np.isnan(sp_none)
    n0, sp0 = z_plane_stats(np.array([]))
    assert n0 == 0 and np.isnan(sp0)


def test_plane_stats_jitter_collapses_to_one_plane():
    # A nominally single plane with tiny float jitter must read as 1 plane.
    z = _planes([5.0], per_plane=200, jitter=1e-9, seed=3)
    n, sp = z_plane_stats(z)
    assert n == 1
    assert np.isnan(sp)


# --------------------------------------------------------------------------
# auto mode
# --------------------------------------------------------------------------
def test_auto_discrete_merfish():
    r = resolve_g_z_um(MERFISH_Z, "auto")
    assert r.mode == "auto-discrete"
    assert r.g_z_um == pytest.approx(1.5)
    assert r.z_neighbor_depth_override is None
    assert not r.warnings


def test_auto_discrete_cosmx():
    r = resolve_g_z_um(COSMX_Z, "auto")
    assert r.mode == "auto-discrete"
    assert r.g_z_um == pytest.approx(0.8)


def test_auto_continuous_keeps_xenium_default():
    r = resolve_g_z_um(XENIUM_Z, "auto")
    assert r.mode == "auto-continuous"
    assert r.g_z_um == pytest.approx(1.0)


def test_auto_respects_custom_continuous_default():
    r = resolve_g_z_um(XENIUM_Z, "auto", continuous_default_um=2.0)
    assert r.g_z_um == pytest.approx(2.0)


# --------------------------------------------------------------------------
# explicit float
# --------------------------------------------------------------------------
def test_explicit_float_honored():
    r = resolve_g_z_um(MERFISH_Z, 1.5)
    assert r.mode == "explicit"
    assert r.g_z_um == pytest.approx(1.5)
    assert not r.warnings


def test_explicit_too_small_warns_on_discrete():
    # The exact MERFISH-with-default-1.0 failure: g_z < plane spacing.
    r = resolve_g_z_um(MERFISH_Z, 1.0)
    assert r.mode == "explicit"
    assert r.g_z_um == pytest.approx(1.0)
    assert r.warnings and "spacing" in r.warnings[0]


def test_explicit_geq_spacing_no_warning():
    # CosMx with default 1.0 ≥ 0.8 spacing → connects, no warning.
    r = resolve_g_z_um(COSMX_Z, 1.0)
    assert not r.warnings


def test_preset_equal_spacing_no_warning_under_float32_jitter():
    # The CosMx 0.8 preset must not trip its own warning when coords are
    # float32 (observed spacing reads as ~0.80000001 > 0.8).
    z32 = COSMX_Z.astype(np.float32)
    r = resolve_g_z_um(z32, 0.8)
    assert not r.warnings


# --------------------------------------------------------------------------
# degenerate / 2D
# --------------------------------------------------------------------------
def test_single_plane_degrades_to_2d():
    z = _planes([5.0], per_plane=100)
    r = resolve_g_z_um(z, "auto")
    assert r.mode == "degenerate-2d"
    assert r.z_neighbor_depth_override == 0
    assert r.warnings


def test_missing_z_degrades_to_2d():
    r = resolve_g_z_um(None, 1.0)
    assert r.mode == "degenerate-2d"
    assert r.z_neighbor_depth_override == 0


def test_all_nan_degrades_to_2d():
    r = resolve_g_z_um(np.full(50, np.nan), "auto")
    assert r.mode == "degenerate-2d"


# --------------------------------------------------------------------------
# legacy None passthrough
# --------------------------------------------------------------------------
def test_none_is_legacy_auto_when_z_present():
    r = resolve_g_z_um(MERFISH_Z, None)
    assert r.mode == "legacy-auto"
    assert r.g_z_um is None  # caller substitutes within-cell auto_Gz


# --------------------------------------------------------------------------
# error handling
# --------------------------------------------------------------------------
def test_bad_string_rejected():
    with pytest.raises(ValueError, match="auto"):
        resolve_g_z_um(MERFISH_Z, "magic")


def test_returns_frozen_resolution():
    r = resolve_g_z_um(COSMX_Z, "auto")
    assert isinstance(r, ZScaleResolution)
    with pytest.raises(Exception):
        r.g_z_um = 9.0  # frozen
