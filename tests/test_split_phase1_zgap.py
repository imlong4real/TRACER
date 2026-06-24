"""Regression tests for the Split-Phase1 z-gap splitter
(`_spatial_split_phase1_entities`).

Guards the fix for the alphanumeric-cell_id no-op bug: the original
numeric-only gate regex (`^\\d+(-\\d+){0,2}$`) rejected every entity on
datasets whose cell_ids contain dashes (e.g. PDAC "jikammne-1"), so the
whole z-gap stage silently did nothing. The splitter now gates on the
`_etype` column (cell/partial) when present, and mints child labels by
appending a collision-free suffix to the parent label — prefix-agnostic.

The legacy numeric-label path (no `_etype` column) is asserted unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tests._pipeline_runner import _spatial_split_phase1_entities


def _two_z_clusters(label: str, n_low: int, n_high: int,
                    with_etype: bool, etype: str = "cell") -> pd.DataFrame:
    """One entity ``label`` with ``n_low`` tx at z≈1.0 and ``n_high`` tx at
    z≈8.0 — a clean z-gap of ~7 µm, well above the 2.0 µm threshold."""
    n = n_low + n_high
    z = np.r_[np.full(n_low, 1.0) + np.linspace(0, 0.3, n_low),
              np.full(n_high, 8.0) + np.linspace(0, 0.3, n_high)]
    df = pd.DataFrame({
        "tracer_id": [label] * n,
        "x": np.linspace(0, 1, n),
        "y": np.linspace(0, 1, n),
        "z": z,
    })
    if with_etype:
        df["_etype"] = etype
    return df


def test_alphanumeric_with_etype_splits():
    """Primary regression: dash-containing cell_id with `_etype` must split."""
    df = _two_z_clusters("abc-1", n_low=5, n_high=4, with_etype=True)
    out, stats = _spatial_split_phase1_entities(df, entity_col="tracer_id")

    assert stats["entities_split"] == 1
    labels = set(out["tracer_id"])
    assert labels == {"abc-1", "abc-1-1"}
    # Larger (5-tx) group keeps the original label; smaller (4-tx) is minted.
    assert (out["tracer_id"] == "abc-1").sum() == 5
    assert (out["tracer_id"] == "abc-1-1").sum() == 4
    # Minted child rows are reclassified as partial.
    minted = out["tracer_id"] == "abc-1-1"
    assert (out.loc[minted, "_etype"] == "partial").all()


def test_numeric_with_etype_splits():
    """Parity: numeric cell_id with `_etype` splits as before."""
    df = _two_z_clusters("5", n_low=5, n_high=4, with_etype=True)
    out, stats = _spatial_split_phase1_entities(df, entity_col="tracer_id")

    assert stats["entities_split"] == 1
    assert set(out["tracer_id"]) == {"5", "5-1"}
    assert (out["tracer_id"] == "5").sum() == 5
    assert (out["tracer_id"] == "5-1").sum() == 4


def test_numeric_no_etype_legacy_path_splits():
    """Back-compat: with no `_etype` column the legacy numeric-regex path
    still recognizes and splits numeric labels."""
    df = _two_z_clusters("5", n_low=5, n_high=4, with_etype=False)
    assert "_etype" not in df.columns
    out, stats = _spatial_split_phase1_entities(df, entity_col="tracer_id")

    assert stats["entities_split"] == 1
    assert set(out["tracer_id"]) == {"5", "5-1"}


def test_alphanumeric_no_etype_legacy_noop():
    """Documents the boundary: without `_etype` the function cannot
    disambiguate a dash-containing cell_id, so the legacy regex path
    leaves it untouched. (Production always supplies `_etype`.)"""
    df = _two_z_clusters("abc-1", n_low=5, n_high=4, with_etype=False)
    out, stats = _spatial_split_phase1_entities(df, entity_col="tracer_id")

    assert stats["entities_split"] == 0
    assert set(out["tracer_id"]) == {"abc-1"}


def test_no_zgap_is_left_untouched():
    """Control: an entity with no z-gap is not split."""
    df = _two_z_clusters("abc-1", n_low=5, n_high=0, with_etype=True)
    out, stats = _spatial_split_phase1_entities(df, entity_col="tracer_id")

    assert stats["entities_split"] == 0
    assert set(out["tracer_id"]) == {"abc-1"}


def test_existing_child_suffix_no_collision():
    """Minted labels avoid colliding with a pre-existing child suffix."""
    base = _two_z_clusters("abc-1", n_low=5, n_high=4, with_etype=True)
    # Add an unrelated existing child "abc-1-1" (single tx, won't be split).
    existing = pd.DataFrame({
        "tracer_id": ["abc-1-1"],
        "x": [0.5], "y": [0.5], "z": [4.0],
        "_etype": ["partial"],
    })
    df = pd.concat([base, existing], ignore_index=True)
    out, stats = _spatial_split_phase1_entities(df, entity_col="tracer_id")

    assert stats["entities_split"] == 1
    # The new group must be "abc-1-2", not collide with existing "abc-1-1".
    assert (out["tracer_id"] == "abc-1-2").sum() == 4
    assert (out["tracer_id"] == "abc-1-1").sum() == 1
