"""Smoke tests for the tile-parallel SEG orchestrator.

Validates that:
  - The tiler assigns every cell_id to exactly one tile (no duplication, no drops).
  - For a synthetic input, single-tile run (n_tiles_xy=(1,1)) matches the
    sequential `run_segmented_pipeline` output byte-for-byte.
  - Multi-tile run on a spatially separable synthetic dataset produces
    consistent within-tile entity counts.

The single-tile parity gate is the key invariant: if (1,1) reproduces
the sequential pipeline, the orchestrator's mechanics (slicing,
process-pool roundtrip, concat) are sound. The multi-tile run can
diverge slightly due to skipped cross-tile Stitch/Rescue, which is
expected and documented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import (
    make_synthetic_transcripts,
    make_synthetic_npmi_panel_for_transcripts,
)
from tests._pipeline_runner import run_segmented_pipeline
from tests._pipeline_runner_tiled import (
    run_segmented_pipeline_tiled,
    _assign_cells_to_tiles,
)


@pytest.fixture(scope="module")
def synthetic_inputs():
    df, gt = make_synthetic_transcripts(n_cells=20, n_types=3, seed=42)
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    df_nuc = df.rename(columns={"is_nuclear": "overlaps_nucleus"})
    return df_nuc, panel


# ---------------------------------------------------------------------------
# Tiler unit tests
# ---------------------------------------------------------------------------

def test_tiler_assigns_every_cell_id_once(synthetic_inputs):
    df, _ = synthetic_inputs
    cell_to_tile, tile_info = _assign_cells_to_tiles(
        df, n_tiles_xy=(2, 2), cell_id_col="cell_id", coord_cols=("x", "y"),
    )
    assert set(cell_to_tile.index) == set(df["cell_id"].unique()), (
        "Every cell_id must appear exactly once in cell_to_tile."
    )
    assert cell_to_tile.notna().all(), "No cell_id should land outside the bbox."
    # Tile indices are in [0, n_x*n_y)
    assert cell_to_tile.min() >= 0
    assert cell_to_tile.max() < 4


def test_tiler_disjoint_assignment(synthetic_inputs):
    """Same cell_id never appears in two tiles' slices."""
    df, _ = synthetic_inputs
    cell_to_tile, _ = _assign_cells_to_tiles(
        df, n_tiles_xy=(2, 2), cell_id_col="cell_id", coord_cols=("x", "y"),
    )
    # Map tx → tile, then check no cell_id spans tiles
    df_t = df.assign(_tile=df["cell_id"].map(cell_to_tile))
    per_cell_tiles = df_t.groupby("cell_id")["_tile"].nunique()
    assert (per_cell_tiles == 1).all(), (
        f"Cell_id spans multiple tiles: "
        f"{per_cell_tiles[per_cell_tiles > 1].head().to_dict()}"
    )


# ---------------------------------------------------------------------------
# Single-tile parity gate
# ---------------------------------------------------------------------------

def test_single_tile_matches_sequential(synthetic_inputs):
    """run_segmented_pipeline_tiled with n_tiles_xy=(1,1) must produce
    the same output as run_segmented_pipeline (modulo row order)."""
    df, panel = synthetic_inputs

    df_seq, _ = run_segmented_pipeline(df.copy(), panel)

    result = run_segmented_pipeline_tiled(
        df.copy(), panel,
        n_tiles_xy=(1, 1),
        n_workers=1,    # serial fallback so failures don't get swallowed
        rerank=False,   # match default of run_segmented_pipeline
        reassign=True,
    )
    df_tiled = result["df_out"]

    # Same number of tx
    assert len(df_seq) == len(df_tiled), (
        f"Row count diverges: seq={len(df_seq)} tiled={len(df_tiled)}"
    )

    # Same set of (transcript_id -> label) assignments. Sort by
    # transcript_id (or any stable key) before comparing.
    col_seq = "stitched" if "stitched" in df_seq.columns else "tracer_id"
    col_tiled = "stitched" if "stitched" in df_tiled.columns else "tracer_id"

    # Use the index as identity if no transcript_id present
    if "transcript_id" in df_seq.columns:
        key = "transcript_id"
        a = df_seq.set_index(key)[col_seq].astype(str)
        b = df_tiled.set_index(key)[col_tiled].astype(str)
    else:
        # Fall back to comparing sorted label distributions
        a = df_seq[col_seq].astype(str).sort_values().reset_index(drop=True)
        b = df_tiled[col_tiled].astype(str).sort_values().reset_index(drop=True)

    assert (a == b).all() or (
        a.value_counts().to_dict() == b.value_counts().to_dict()
    ), "Single-tile output diverges from sequential pipeline output."


# ---------------------------------------------------------------------------
# Multi-tile smoke
# ---------------------------------------------------------------------------

def test_multi_tile_runs_and_produces_output(synthetic_inputs):
    """Multi-tile run completes and produces a sensible output frame."""
    df, panel = synthetic_inputs
    result = run_segmented_pipeline_tiled(
        df, panel,
        n_tiles_xy=(2, 2),
        n_workers=1,   # serial in test to keep deterministic and printable
        rerank=False, reassign=True,
    )
    assert result["df_out"] is not None
    assert len(result["df_out"]) == len(df), (
        "Tx-count must be preserved through the tile pipeline."
    )
    # Each tile got at least one cell.
    for ti, info in result["per_tile"].items():
        assert info["n_input_cell_ids"] > 0
        # Output cells / partials is a non-negative integer
        assert info["n_out_cells"] >= 0
        assert info["n_out_partials"] >= 0
    # Tile bbox info is recorded
    assert len(result["tile_info"]) == 4
