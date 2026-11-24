"""The shared prune pieces (unified-prune steps 3-4).

* ``_seed_entity`` / ``seed_entity``: the seed core (count transcripts per
  gene, tx-weighted greedy). Phase 1 builds its 1a seed and 1c sub-seed
  with it; the Group cascade builds each window's seed with it.
* ``_prune_entity`` / ``prune_entities``: Phase-1's per-entity unit (seed
  -> admission -> one carve), on top of the seed core.

These tests pin:
* ``prune_entities`` with Phase-1 settings is the Phase-1 kernel, code
  for code;
* ``seed_entity`` reproduces the cascade's original per-window rule
  (np.unique counts -> greedy_prune_retained), and committing the seed
  genes' transcripts gives the cascade's membership.
"""
from __future__ import annotations

import numpy as np
import pytest

from tracer import _cy_prune
from tracer._utils import symmetric_csr
from tracer.pruning import build_sparse_pmi_matrix_from_long
from tests.segmentation_sim import simulate_dapi_voronoi_segmentation
from tests.synthetic import (
    make_synthetic_npmi_panel_for_transcripts,
    make_synthetic_transcripts,
)

THR = 0.2


@pytest.fixture(scope="module")
def setup():
    df, gt = make_synthetic_transcripts(
        n_cells=30, voxels_per_cell_mean=120, tx_per_cell=60, n_genes=24,
        n_types=4, domain_z_um=10.0, nuclear_layers=2, seed=7,
    )
    df = simulate_dapi_voronoi_segmentation(df)
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    genes, g2i, W = build_sparse_pmi_matrix_from_long(panel, metric_col="NPMI")
    S = symmetric_csr(W)
    gidx = df["feature_name"].map(g2i).fillna(-1).to_numpy(np.int32)
    nuc = df["is_nuclear"].to_numpy(np.uint8)
    cells = df["cell_id"].astype(str).to_numpy()
    lists = [np.flatnonzero(cells == c).astype(np.int32)
             for c in sorted(set(cells) - {"-1"})]
    csr = (S.indptr.astype(np.int32), S.indices.astype(np.int32),
           S.data.astype(np.float32))
    return lists, gidx, nuc, csr


@pytest.mark.parametrize("veto_mode", [0, 1, 2])
@pytest.mark.parametrize("nuclear_seed_only", [0, 1])
@pytest.mark.parametrize("nuclear_only_admit", [0, 1])
def test_phase1_settings_match_phase1_kernel(setup, veto_mode, nuclear_seed_only,
                                             nuclear_only_admit):
    lists, gidx, nuc, (ip, ix, dt) = setup
    common = dict(tx_weighted=1, veto_mode=veto_mode, min_admit_threshold=0.0,
                  mean_admit_threshold=0.2, aggregator_percentile=25.0,
                  real_signal_threshold=0.2, neg_npmi_threshold=-0.2,
                  fallback_whole_cell_admit=1)
    ref = _cy_prune.prune_cells_nuclear_seed_sparse(
        lists, gidx, nuc, ip, ix, dt, THR, 3, 0,
        nuclear_only_admit=nuclear_only_admit,
        nuclear_seed_only=nuclear_seed_only, **common)
    got = _cy_prune.prune_entities(
        lists, gidx, nuc, ip, ix, dt, THR,
        min_seed_genes=3, max_carves=1,
        seed_from_mask=nuclear_seed_only, admit_mask_only=nuclear_only_admit,
        **common)
    assert np.array_equal(ref, got)
    assert (got == 0).any() and (got == 1).any(), "fixture exercises no carve"


def test_seed_entity_reproduces_cascade_rule(setup):
    """The cascade's original rule: retained = greedy_prune_retained over
    np.unique(genes, return_counts=True); committed = tx of retained genes."""
    lists, gidx, nuc, (ip, ix, dt) = setup
    n_changed = 0
    for rows in lists:
        g = gidx[rows]
        g = g[g >= 0].astype(np.int32)
        if g.size == 0:
            continue
        ug, uc = np.unique(g, return_counts=True)
        ref = _cy_prune.greedy_prune_retained(
            ug.astype(np.int32), uc.astype(np.int32),
            W_indptr=ip, W_indices=ix, W_data=dt, threshold=THR)
        got = _cy_prune.seed_entity(g, ip, ix, dt, THR)
        assert np.array_equal(ref, got)
        assert np.all(np.diff(got) > 0), "seed must be ascending"
        n_changed += int((~np.isin(g, got)).sum())
    assert n_changed > 0, "no entity lost a gene; the comparison is vacuous"


def test_phase1_whole_cell_main_is_seed_genes_plus_admitted(setup):
    """With admission and carve off, Phase-1's whole-cell main is exactly
    the transcripts of seed_entity's genes: Phase 1 and the cascade build
    seeds with the same core."""
    lists, gidx, nuc, (ip, ix, dt) = setup
    codes = _cy_prune.prune_entities(
        lists, gidx, nuc, ip, ix, dt, THR, min_seed_genes=3, max_carves=0,
        seed_from_mask=0, veto_mode=0, neg_npmi_threshold=1e9)
    for rows in lists:
        g = gidx[rows]
        ok = g >= 0
        if not ok.any():
            continue
        seed = _cy_prune.seed_entity(g[ok].astype(np.int32), ip, ix, dt, THR)
        main = codes[rows] == 0
        # every seed-gene transcript is main ...
        assert main[ok & np.isin(g, seed)].all()


def test_max_carves_validated(setup):
    lists, gidx, nuc, (ip, ix, dt) = setup
    with pytest.raises(ValueError):
        _cy_prune.prune_entities(lists, gidx, nuc, ip, ix, dt, THR, max_carves=2)


def test_dense_w_conversion_matches_dense_kernel():
    """The cascade converts a dense (NaN->0 filled) W to CSR once. Every
    entry, zeros included, must stay stored, so the greedy prune sees the
    same observed pairs as the dense kernel did."""
    from tracer.density_cascade import _csr_arrays_for_prune
    rng = np.random.default_rng(1)
    G = 40
    A = rng.normal(0, 0.5, (G, G)).astype(np.float32)
    A = (A + A.T) / 2
    A[rng.random((G, G)) < 0.3] = 0.0          # observed zeros
    np.fill_diagonal(A, 0.0)
    ip, ix, dt = _csr_arrays_for_prune(A)
    assert dt.size == G * G
    for _ in range(50):
        g = np.unique(rng.choice(G, size=rng.integers(2, 15), replace=False)).astype(np.int32)
        c = rng.integers(1, 6, size=g.size).astype(np.int32)
        dense = _cy_prune.greedy_prune_retained(g, c, W_dense=A, threshold=THR)
        sparse = _cy_prune.greedy_prune_retained(g, c, W_indptr=ip, W_indices=ix,
                                                 W_data=dt, threshold=THR)
        assert np.array_equal(dense, sparse)
