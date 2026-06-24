"""Unit tests for `_phase1_rerank_within_parent_etype`.

The function re-ranks depth-1 entities under each parent cell_id by
nuclear-tx count and promotes the largest to the main `{cell_id}` label.
Pure relabeling; no tx demotion, no coordinate changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests._pipeline_runner import _phase1_rerank_within_parent_etype


def _df(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal test frame: rows of (entity, cell_id, nuclear)."""
    return pd.DataFrame(
        rows, columns=["tracer_id", "cell_id", "overlaps_nucleus"]
    )


def test_no_partials_is_noop():
    """One depth-1 entity under parent → no relabel."""
    df = _df([
        ("42", "42", True),
        ("42", "42", True),
        ("42", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    assert (out["tracer_id"] == df["tracer_id"]).all()
    assert stats["n_parents_reranked"] == 0
    assert stats["n_tx_relabeled"] == 0


def test_single_swap_promotes_larger_partial():
    """Partial `42-1` has 5 nuclear tx, main `42` has 3 → swap."""
    df = _df([
        ("42",   "42", True),
        ("42",   "42", True),
        ("42",   "42", True),
        ("42-1", "42", True),
        ("42-1", "42", True),
        ("42-1", "42", True),
        ("42-1", "42", True),
        ("42-1", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 5, "42-1": 3}
    assert stats["n_parents_reranked"] == 1
    assert stats["n_tx_relabeled"] == 8


def test_tie_keeps_original_main():
    """Main `42` and partial `42-1` both have 4 nuclear tx → strict >
    means original main wins; no relabel."""
    df = _df([
        ("42",   "42", True),  ("42",   "42", True),
        ("42",   "42", True),  ("42",   "42", True),
        ("42-1", "42", True),  ("42-1", "42", True),
        ("42-1", "42", True),  ("42-1", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 4, "42-1": 4}
    assert stats["n_parents_reranked"] == 0


def test_three_way_reorder():
    """Main 42 has 2, partial 42-1 has 4, partial 42-2 has 7 →
    new order: 42-2 (7) → main; 42-1 (4) → -1; 42 (2) → -2."""
    df = _df([
        ("42",   "42", True), ("42",   "42", True),
        ("42-1", "42", True), ("42-1", "42", True),
        ("42-1", "42", True), ("42-1", "42", True),
        ("42-2", "42", True), ("42-2", "42", True),
        ("42-2", "42", True), ("42-2", "42", True),
        ("42-2", "42", True), ("42-2", "42", True),
        ("42-2", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 7, "42-1": 4, "42-2": 2}
    assert stats["n_parents_reranked"] == 1
    assert stats["n_tx_relabeled"] == 13


def test_subpartial_follows_parent_with_bump_on_collision():
    """Promoted partial brings its sub-partials along; deposed main
    bumps past the reserved sub-suffix slots.

    Before:
      42       × 2   (main)
      42-1     × 4   (partial; direct tx)
      42-1-1   × 2   (sub-partial of 42-1)
      subtree-size: 42=2, 42-1=6 → 42-1 wins.

    After:
      42       × 4   (was 42-1 direct)
      42-1     × 2   (was 42-1-1; sub-partial of new main)
      42-2     × 2   (was 42; bumped past the reserved -1 slot)
    """
    df = _df([
        ("42",     "42", True), ("42",     "42", True),
        ("42-1",   "42", True), ("42-1",   "42", True),
        ("42-1",   "42", True), ("42-1",   "42", True),
        ("42-1-1", "42", True), ("42-1-1", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 4, "42-1": 2, "42-2": 2}
    assert stats["n_parents_reranked"] == 1
    assert stats["n_tx_relabeled"] == 8


def test_unassigned_labels_untouched():
    """Labels matching UNASSIGNED_*, -1, etc. are not candidates for
    rerank under any parent."""
    df = _df([
        ("42",            "42", True), ("42",            "42", True),
        ("42-1",          "42", True), ("42-1",          "42", True),
        ("42-1",          "42", True), ("42-1",          "42", True),
        ("UNASSIGNED_7", "42", True),
        ("-1",            "42", False),
        ("UNASSIGNED",   "42", True),
    ])
    out, stats = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    assert counts["42"] == 4
    assert counts["42-1"] == 2
    assert counts["UNASSIGNED_7"] == 1
    assert counts["-1"] == 1
    assert counts["UNASSIGNED"] == 1


def test_cyto_tx_dont_count_toward_size():
    """Only nuclear tx count toward the size used for ranking.
    Partial has more total tx but fewer nuclear → main wins."""
    df = _df([
        ("42",   "42", True),  ("42",   "42", True),  ("42",   "42", True),
        ("42-1", "42", True),  ("42-1", "42", True),
        ("42-1", "42", False), ("42-1", "42", False),
        ("42-1", "42", False), ("42-1", "42", False), ("42-1", "42", False),
    ])
    out, stats = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 3, "42-1": 7}
    assert stats["n_parents_reranked"] == 0


def test_idempotent():
    """Running rerank twice in a row is a no-op the second time."""
    df = _df([
        ("42",   "42", True),
        ("42",   "42", True),
        ("42-1", "42", True),  ("42-1", "42", True),
        ("42-1", "42", True),  ("42-1", "42", True),
    ])
    out1, stats1 = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    assert stats1["n_parents_reranked"] == 1
    out2, stats2 = _phase1_rerank_within_parent_etype(
        out1, entity_col="tracer_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    assert (out2["tracer_id"] == out1["tracer_id"]).all()
    assert stats2["n_parents_reranked"] == 0


def test_margin_tx_blocks_close_swaps():
    """margin_tx=3 blocks a +1 difference."""
    df = _df([
        ("42",   "42", True), ("42",   "42", True), ("42",   "42", True),
        ("42-1", "42", True), ("42-1", "42", True),
        ("42-1", "42", True), ("42-1", "42", True),
    ])
    out, stats = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id",
        nuclear_col="overlaps_nucleus", margin_tx=3,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 3, "42-1": 4}
    assert stats["n_parents_reranked"] == 0


def test_qc_after_rerank_demotes_old_main_below_min_tx():
    """If old main had 2 tx (below min_tx=3) and partial had 5, after
    Rerank the partial wears `42` (survives QC); the old main wears
    `42-1` with 2 tx and gets demoted by the next Phase 1-QC pass."""
    from tests._pipeline_runner import (
        _phase1_rerank_within_parent_etype,
        _qc_demote_small_phase1_entities,
        PHASE1_QC_MIN_TX,
    )
    df = _df([
        ("42",   "42", True), ("42",   "42", True),
        ("42-1", "42", True), ("42-1", "42", True),
        ("42-1", "42", True), ("42-1", "42", True), ("42-1", "42", True),
    ])
    rerk, _ = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id",
        nuclear_col="overlaps_nucleus", margin_tx=1,
    )
    qcd, _qc_stats = _qc_demote_small_phase1_entities(
        rerk, entity_col="tracer_id", min_size=PHASE1_QC_MIN_TX,
        unassigned_id="-1",
    )
    counts = qcd["tracer_id"].value_counts().to_dict()
    # Rerank: old 42-1 (5) → "42"; old 42 (2) → "42-1"
    # QC: "42-1" has 2 tx < 3 → demote to "-1"
    assert counts["42"] == 5
    assert counts.get("-1", 0) == 2
    assert "42-1" not in counts
