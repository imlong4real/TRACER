"""Stitch merge-time guards and neighbour re-queueing (2026-11-19).

Ported from the GBM Stitch prototype and made the SEG defaults together
with the looser ΔC_min (-0.3), Moore-2 reach and depth cap 5:

* ``merged_c_min``           merged coherence C(union) must be >= this;
* ``cell_delta_c_min``       a merge involving a whole cell needs ΔC >= this;
* ``cumulative_delta_c_min`` ΔC summed over a group's merge tree >= this;
* ``requeue_neighbours``     after a merge, re-score the new group against
                             its neighbours (grid candidates).

The synthetic pipeline fixtures never reach these guards (their stage
labels are identical with and without them), so each guard is pinned here
on hand-built entity pairs with a known ΔC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tracer.pipeline as P
from tracer.config import StitchConfig, load_config
from tracer.stitching import (
    _LAST_GATE_STATS,
    apply_stitching_to_transcripts_memory_efficient,
)
from tests.test_stitch_mahalanobis_rescue import _build_aux_and_pools, _gauss_entity

# overlap 0.70 / cross -0.3 gives ΔC ≈ -0.18: accepted at deltaC_min -0.3,
# rejected by the -0.1 whole-cell guard.
KW = dict(candidate_source="grid", G=2.0, G_z=2.0, z_neighbor_depth=1,
          dist_threshold=10.0, deltaC_min=-0.3, max_merger_depth=None,
          c_union_bypass=None, mahalanobis_d_rescue=None, show_progress=False)


def _pair(etype_a="partial", seed=0):
    rng = np.random.default_rng(seed)
    aux, pool_a, pool_b = _build_aux_and_pools(prog_size=10, overlap_frac=0.70, cross=-0.3)
    a = _gauss_entity("a-1-1", 60, (0.0, 0.0, 0.0), 2.0, pool_a, rng)
    b = _gauss_entity("b-1-1", 60, (1.0, 0.0, 0.0), 2.0, pool_b, rng)
    if etype_a == "cell":
        a["tracer_id"], a["_etype"] = "a", "cell"
    return aux, pd.concat([a, b], ignore_index=True)


def _n(df, aux, **kw):
    out, _ = apply_stitching_to_transcripts_memory_efficient(df.copy(), aux=aux, **{**KW, **kw})
    return out["tracer_id"].astype(str).nunique()


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
def test_pair_merges_without_guards():
    aux, df = _pair()
    assert _n(df, aux) == 1


def test_merged_c_min_blocks_low_union_coherence():
    aux, df = _pair()
    assert _n(df, aux, merged_c_min=0.0) == 1          # C(union) is positive here
    assert _n(df, aux, merged_c_min=0.999) == 2        # but not this high
    assert _LAST_GATE_STATS["blocked_merged_c"] >= 1


def test_cell_guard_applies_only_to_whole_cells():
    aux, df = _pair(etype_a="partial")
    assert _n(df, aux, cell_delta_c_min=-0.1) == 1     # two partials: guard silent
    aux, df = _pair(etype_a="cell")
    assert _n(df, aux) == 1                            # cell + partial, no guard
    assert _n(df, aux, cell_delta_c_min=-0.1) == 2     # ΔC ≈ -0.18 < -0.1
    assert _LAST_GATE_STATS["blocked_cell_guard"] >= 1
    assert _n(df, aux, cell_delta_c_min=-0.25) == 1    # looser guard lets it through


def test_cumulative_budget_counts_the_whole_merge_tree():
    """One merge at ΔC ≈ -0.18 fits a -0.3 budget but not a -0.1 one."""
    aux, df = _pair()
    assert _n(df, aux, cumulative_delta_c_min=-0.3) == 1
    assert _n(df, aux, cumulative_delta_c_min=-0.1) == 2
    assert _LAST_GATE_STATS["blocked_cumulative_dc"] >= 1


def _three_programs(core=7, uniq=3, cross=-0.3, within=0.6):
    """Three programs sharing `core` genes, each with `uniq` own genes that
    conflict (cross) with the other programs' own genes."""
    genes = [f"G{i:03d}" for i in range(core + 3 * uniq)]
    shared = list(range(core))
    W = np.full((len(genes),) * 2, cross, dtype=np.float32)
    pools = []
    for k in range(3):
        idx = shared + list(range(core + k * uniq, core + (k + 1) * uniq))
        pools.append([genes[i] for i in idx])
        for i in idx:
            W[i, idx] = within
    np.fill_diagonal(W, np.nan)
    return {"W": W, "gene_to_idx": {g: i for i, g in enumerate(genes)}}, pools


def test_cumulative_budget_stops_a_chain():
    """Three fragments in a row with three different programs. With a loose
    per-merge gate both merges happen (first ΔC ≈ -0.18, sum ≈ -0.55). A
    -0.3 budget allows the first merge and blocks the second: it is the
    sum over the merge tree that is limited, not each merge."""
    aux, pools = _three_programs()
    rng = np.random.default_rng(3)
    df = pd.concat([_gauss_entity(f"{k}-1-1", 60, (3.0 * i, 0.0, 0.0), 1.0, pools[i], rng)
                    for i, k in enumerate("abc")], ignore_index=True)
    assert _n(df, aux, deltaC_min=-0.9) == 1
    assert _LAST_GATE_STATS["merges_total"] == 2
    assert _n(df, aux, deltaC_min=-0.9, cumulative_delta_c_min=-0.3) == 2
    assert _LAST_GATE_STATS["merges_total"] == 1
    assert _LAST_GATE_STATS["blocked_cumulative_dc"] >= 1


@pytest.mark.parametrize("off", [None, float("-inf")])
def test_guards_off_spellings_change_nothing(off):
    aux, df = _pair(etype_a="cell")
    base = _n(df, aux)
    assert _n(df, aux, merged_c_min=off, cell_delta_c_min=off,
              cumulative_delta_c_min=off) == base


# --------------------------------------------------------------------------
# Re-queueing
# --------------------------------------------------------------------------
def test_requeue_rescores_neighbours_after_a_merge():
    rng = np.random.default_rng(3)
    aux, pool_a, pool_b = _build_aux_and_pools(prog_size=10, overlap_frac=0.70, cross=-0.3)
    parts = [_gauss_entity(f"{k}-1-1", 60, (3.0 * i, 0.0, 0.0), 2.0,
                           pool_a if i % 2 == 0 else pool_b, rng)
             for i, k in enumerate("abc")]
    df = pd.concat(parts, ignore_index=True)
    _n(df, aux, requeue_neighbours=False)
    assert _LAST_GATE_STATS["requeued_pairs"] == 0
    _n(df, aux, requeue_neighbours=True)
    assert _LAST_GATE_STATS["requeued_pairs"] >= 1


def test_requeue_is_grid_only():
    rng = np.random.default_rng(5)
    aux, pool_a, pool_b = _build_aux_and_pools(prog_size=10, overlap_frac=0.70, cross=-0.3)
    df = pd.concat([_gauss_entity(f"e{i}-1-1", 40, (3.0 * (i % 3), 3.0 * (i // 3), 0.5 * i),
                                  1.0, pool_a if i % 2 else pool_b, rng)
                    for i in range(6)], ignore_index=True)
    _n(df, aux, requeue_neighbours=True, candidate_source="delaunay")
    assert _LAST_GATE_STATS["requeued_pairs"] == 0


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def test_seg_defaults_are_the_chosen_set():
    s = load_config().stitch
    assert (s.deltaC_min, s.neighborhood, s.max_merger_depth) == (-0.3, "R2", 5)
    assert (s.merged_c_min, s.cell_delta_c_min, s.cumulative_delta_c_min) == (0.0, -0.1, -0.3)
    assert s.requeue_neighbours is True
    assert s.rescue_delta_c_floor == -0.2
    assert StitchConfig() == s


def test_noseg_keeps_previous_stitch():
    s = load_config(platform="noseg").stitch
    assert (s.deltaC_min, s.neighborhood, s.max_merger_depth) == (0.03, "8", 3)
    assert (s.merged_c_min, s.cell_delta_c_min, s.cumulative_delta_c_min) == (None, None, None)
    assert s.requeue_neighbours is False
    assert P._resolve_pipeline_cfg(None, noseg=True).stitch == s


def test_minus_inf_means_off():
    s = StitchConfig(merged_c_min=float("-inf"), cell_delta_c_min=float("-inf"),
                     cumulative_delta_c_min=float("-inf"))
    assert (s.merged_c_min, s.cell_delta_c_min, s.cumulative_delta_c_min) == (None, None, None)


@pytest.mark.parametrize("field,value", [("merged_c_min", 1.5), ("cell_delta_c_min", -2.0),
                                         ("cumulative_delta_c_min", float("nan"))])
def test_guard_values_validated(field, value):
    with pytest.raises(ValueError, match=field):
        StitchConfig(**{field: value})


@pytest.mark.parametrize("nb,ok", [("8", True), ("4", True), ("0", True), ("R2", True),
                                   ("R3", True), ("R0", False), ("Rx", False), ("9", False)])
def test_neighborhood_validated(nb, ok):
    if ok:
        StitchConfig(neighborhood=nb)
    else:
        with pytest.raises(ValueError, match="neighborhood"):
            StitchConfig(neighborhood=nb)
