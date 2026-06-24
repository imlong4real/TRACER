"""Smoke tests for the segmented + no-segmentation pipelines on synthetic
transcripts.

The synthetic input plants 8 cells in a 3D voxel grid, 3 type archetypes
(gene panels of 4 genes each), 25 tx per cell, 80% archetype-coherent +
20% cross-type noise. Cells get amoeboid shapes via 6-connected flood-fill
and partially overlap in xy (z separates them — biologically realistic
dense tissue).

Two scenarios per workflow:
- **FullVolume** (sanity baseline): all 8 cells intact in a 10 µm domain.
  Easy mode — every cell has its full tx count and nucleus. Catches gross
  regressions in the trivial path.
- **Section** (primary stress test): the middle 5 µm slab is extracted,
  yielding a mix of fully-contained cells and z-clipped partial cells.
  This is the realistic scenario TRACER's rescue/stitch logic is built
  for. Discriminating assertions (ARI, coverage, partial-vs-whole
  recovery, no phantom cell) live here.

Test classes:
- ``TestSegmentedFullVolume`` / ``TestSegmentedSection``
- ``TestNoSegFullVolume`` / ``TestNoSegSection``
- ``TestSegVsNoSegConsistency``: cross-mode partition agreement on full
  volume.
"""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from tests.synthetic import (
    make_synthetic_transcripts,
    make_synthetic_npmi_panel_for_transcripts,
)
from tests._pipeline_runner import run_segmented_pipeline, run_noseg_pipeline


# Voxel-grid layout: 8 cells, ~80 voxels each (1 µm voxels), full
# 10 µm z volume. Cells get amoeboid shapes via 6-conn flood-fill.
CELLS_KW = dict(
    n_cells=8,
    voxels_per_cell_mean=80,
    tx_per_cell=25,
    n_genes=12,
    n_types=3,
    domain_z_um=10.0,
    nuclear_layers=2,
)
TX_PER_CELL = CELLS_KW["tx_per_cell"]

# Tissue-section sub-volume: middle 5 µm of the 10 µm domain.
SECTION_Z = (2.5, 7.5)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def synthetic_inputs():
    """Full 10 µm volume — every cell intact."""
    df, gt = make_synthetic_transcripts(**CELLS_KW, seed=42)
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    return df, panel, gt


@pytest.fixture(scope="module")
def section_inputs():
    """5 µm slab extracted from the 10 µm volume — mix of whole + clipped cells."""
    df, gt = make_synthetic_transcripts(
        **CELLS_KW, section_z_range_um=SECTION_Z, seed=42,
    )
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    return df, panel, gt


@pytest.fixture(scope="module")
def seg_result(synthetic_inputs):
    df, panel, gt = synthetic_inputs
    df_out, prog = run_segmented_pipeline(df, panel)
    return df_out, prog, gt


@pytest.fixture(scope="module")
def noseg_result(synthetic_inputs):
    df, panel, gt = synthetic_inputs
    df_out, prog = run_noseg_pipeline(df, panel)
    return df_out, prog, gt


@pytest.fixture(scope="module")
def seg_section_result(section_inputs):
    df, panel, gt = section_inputs
    df_out, prog = run_segmented_pipeline(df, panel)
    return df_out, prog, gt


@pytest.fixture(scope="module")
def noseg_section_result(section_inputs):
    df, panel, gt = section_inputs
    df_out, prog = run_noseg_pipeline(df, panel)
    return df_out, prog, gt


def _final_label_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    s = df[col].astype(str)
    n_tx_total = len(s)
    n_unas_tx = int((s == "-1").sum())
    n_assigned = n_tx_total - n_unas_tx
    return {"total": n_tx_total, "assigned": n_assigned, "unassigned": n_unas_tx}


def _partition_by_clipping(df_in: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Return (well_preserved, heavily_clipped) cell-id sets, based on
    how many of the planted ``TX_PER_CELL`` transcripts survived
    sectioning.

    - well-preserved: ≥ 80% of tx survived (cell is mostly inside the slab)
    - heavily-clipped: ≤ 40% of tx survived (cell barely intersects)

    Cells in the middle band are excluded from the comparison — they're
    too noisy to give a clean signal either way.
    """
    counts = df_in["cell_id"].astype(str).value_counts()
    well = set(counts[counts >= 0.8 * TX_PER_CELL].index)
    clipped = set(counts[(counts > 0) & (counts <= 0.4 * TX_PER_CELL)].index)
    return well, clipped


# ============================================================================
# Segmented — full volume (sanity baseline)
# ============================================================================

class TestSegmentedFullVolume:
    """Easy-mode sanity baseline: all cells intact, all nuclei present.
    Catches gross regressions in the trivial path."""

    def test_pipeline_runs_end_to_end(self, seg_result):
        df_out, prog, _ = seg_result
        assert "stitched" in df_out.columns
        # 8 stages of progression captured
        assert len(prog) >= 7

    def test_recovers_planted_truth(self, seg_result):
        """On whole cells with planted ground truth, segmented TRACER
        should refine cleanly (high ARI)."""
        df_out, _, _ = seg_result
        truth = df_out["cell_id"].astype(str).values
        out = df_out["stitched"].astype(str).values
        mask = out != "-1"
        if mask.sum() < 2:
            pytest.skip("not enough assigned tx")
        ari = adjusted_rand_score(truth[mask], out[mask])
        assert ari > 0.5, f"ARI(seg-output, ground_truth) = {ari:.3f} < 0.5"


# ============================================================================
# Segmented — 5 µm section (primary stress test)
# ============================================================================

class TestSegmentedSection:
    """Primary discriminating tests. Section input contains both
    fully-contained cells AND z-clipped partial cells — the realistic
    scenario TRACER's rescue/stitch logic targets."""

    def test_section_pipeline_runs(self, seg_section_result):
        df_out, prog, _ = seg_section_result
        assert "stitched" in df_out.columns
        assert len(prog) >= 7

    def test_section_z_bounds_respected(self, section_inputs):
        df, _, _ = section_inputs
        z_lo, z_hi = SECTION_Z
        assert (df["z"] >= z_lo).all()
        assert (df["z"] < z_hi).all()

    def test_section_has_mix_of_well_and_clipped(self, section_inputs):
        """Sanity: the section actually contains the mix we're testing —
        some well-preserved cells (≥80% tx surviving) and some heavily
        clipped (≤40%)."""
        df, _, gt = section_inputs
        well, clipped = _partition_by_clipping(df)
        assert len(well) > 0, (
            "section has no well-preserved cells — bad section bounds"
        )
        assert len(clipped) > 0, (
            "section has no heavily-clipped cells — bad section bounds"
        )
        assert gt["n_clipped_cells"] >= 1

    def test_final_entity_count_in_range(self, seg_section_result):
        df_out, _, _ = seg_section_result
        s = df_out["stitched"].astype(str)
        n_distinct = (s != "-1").groupby(s).any().sum()
        # 8 planted cells; clipped cells may fragment or be lost.
        assert 4 <= n_distinct <= 16, (
            f"Expected 4..16 final entities, got {n_distinct}"
        )

    def test_coverage_above_50pct(self, seg_section_result):
        df_out, _, _ = seg_section_result
        c = _final_label_counts(df_out, "stitched")
        coverage = c["assigned"] / c["total"]
        assert coverage > 0.5, f"Coverage {coverage:.1%} below 50%"

    def test_recovers_planted_truth(self, seg_section_result):
        """ARI threshold relaxed for clipped-cell input (some cells lose
        most of their tx and have no nucleus left)."""
        df_out, _, _ = seg_section_result
        truth = df_out["cell_id"].astype(str).values
        out = df_out["stitched"].astype(str).values
        mask = out != "-1"
        if mask.sum() < 2:
            pytest.skip("not enough assigned tx")
        ari = adjusted_rand_score(truth[mask], out[mask])
        assert ari > 0.4, f"ARI={ari:.3f} below 0.4 (section relaxed bound)"

    def test_well_preserved_recovered_better_than_clipped(
        self, section_inputs, seg_section_result,
    ):
        """The signal we actually care about: TRACER should preserve
        well-intact cells more reliably than heavily-clipped ones. We
        assert per-class assignment rate, not absolute ARI — clipped
        cells are expected to be noisier and may even be entirely
        unassignable."""
        df_in, _, _ = section_inputs
        df_out, _, _ = seg_section_result
        well, clipped = _partition_by_clipping(df_in)
        if not well or not clipped:
            pytest.skip("section did not produce both well-preserved + clipped cells")

        # Map transcript_id -> ground-truth cell_id (from input)
        gt_cid = df_in.set_index("transcript_id")["cell_id"].astype(str)
        out_lbl = df_out.set_index("transcript_id")["stitched"].astype(str)
        idx = gt_cid.index.intersection(out_lbl.index)
        gt_cid = gt_cid.loc[idx]
        out_lbl = out_lbl.loc[idx]

        well_mask = gt_cid.isin(well)
        clipped_mask = gt_cid.isin(clipped)
        well_assigned = (out_lbl[well_mask] != "-1").mean()
        clipped_assigned = (out_lbl[clipped_mask] != "-1").mean()

        # Well-preserved cells should retain assignment at least as well
        # as clipped ones. Allow a small slack — cell-by-cell variance
        # is real (e.g. a "well-preserved" cell may be in low-tx region).
        assert well_assigned >= clipped_assigned - 0.10, (
            f"Well-preserved cell assignment rate ({well_assigned:.1%}) is "
            f"much worse than heavily-clipped ({clipped_assigned:.1%}) — "
            f"pipeline is breaking the easy case."
        )
        # And well-preserved cells should be substantially recovered.
        assert well_assigned > 0.5, (
            f"Well-preserved cell assignment rate {well_assigned:.1%} "
            f"below 50% — pipeline is failing on mostly-intact cells."
        )

    def test_no_phantom_cell(self, seg_section_result):
        """Regression test for the SHIELD_LABEL absorption bug
        (commit 6aa3f3e). No single entity should hold > 85% of
        assigned tx."""
        df_out, _, _ = seg_section_result
        s = df_out["stitched"].astype(str)
        s = s[s != "-1"]
        if len(s) == 0:
            pytest.skip("no assigned tx")
        max_share = s.value_counts().iloc[0] / len(s)
        assert max_share <= 0.85, (
            f"Phantom-cell symptom: a single entity holds "
            f"{100*max_share:.1f}% of assigned tx (>85% is the SHIELD_LABEL "
            f"bug fingerprint)."
        )


# ============================================================================
# No-segmentation — full volume (sanity baseline)
# ============================================================================

class TestNoSegFullVolume:
    def test_noseg_pipeline_runs_end_to_end(self, noseg_result):
        df_out, _, _ = noseg_result
        assert "stitched" in df_out.columns

    def test_noseg_handles_xy_only_input(self, synthetic_inputs):
        """VHD / 2D-grid inputs lack a z column. The noseg pipeline
        should accept them by synthesising z=0 so all 3D-aware stages
        run unchanged."""
        df, panel, _ = synthetic_inputs
        df_xy = df.drop(columns=["z"])
        assert "z" not in df_xy.columns
        df_out, prog = run_noseg_pipeline(df_xy, panel)
        assert "stitched" in df_out.columns
        # No-op assertion: pipeline should produce some assigned tx.
        assigned = (df_out["stitched"].astype(str) != "-1").sum()
        assert assigned > 0, "noseg on xy-only input produced no assigned tx"


# ============================================================================
# No-segmentation — 5 µm section (primary stress test)
# ============================================================================

class TestNoSegSection:
    def test_noseg_section_runs(self, noseg_section_result):
        df_out, _, _ = noseg_section_result
        assert "stitched" in df_out.columns

    def test_noseg_finds_components(self, noseg_section_result):
        """Group + Stitch should recover a non-trivial number of
        components from the planted cells, even on clipped input."""
        df_out, _, gt = noseg_section_result
        s = df_out["stitched"].astype(str)
        n_distinct = (s != "-1").groupby(s).any().sum()
        assert n_distinct >= 3, (
            f"Expected ≥3 components, got {n_distinct} "
            f"(should find most of {gt['n_cells']} planted cells)"
        )

    def test_no_phantom_cell(self, noseg_section_result):
        """Regression test for the SHIELD_LABEL absorption bug
        (commit 6aa3f3e). Pre-fix, pre_stage2_rescue absorbed nearby
        unassigned tx into the __GUARD_SKIP__ shield label, producing
        a phantom "cell" containing thousands of tx.

        Note: the no-seg pipeline doesn't actually invoke pre_stage2_rescue
        (it has no entities to rescue into) — but this test serves as a
        general "pathological merger" regression check that would also
        fail if Stitch over-merged."""
        df_out, _, _ = noseg_section_result
        s = df_out["stitched"].astype(str)
        s = s[s != "-1"]
        if len(s) == 0:
            pytest.skip("no assigned tx")
        max_share = s.value_counts().iloc[0] / len(s)
        # Threshold 0.85 catches the bug's ~100% fingerprint while
        # tolerating legitimate dense-tissue consolidation under noseg
        # (where Stitch can merge many adjacent same-type bin-cliques
        # into a single super-component).
        assert max_share <= 0.85, (
            f"Phantom-cell symptom: a single entity holds "
            f"{100*max_share:.1f}% of assigned tx (>85% is the SHIELD_LABEL "
            f"bug fingerprint)."
        )


# ============================================================================
# Cross-mode consistency
# ============================================================================

class TestSegVsNoSegConsistency:
    """Runs BOTH pipelines on the same synthetic input and asserts
    non-trivial partition agreement.

    Note: under noseg, all tx start unassigned, so the cell_id used by
    the segmented pipeline is replaced. Both partitions are compared on
    the assigned-in-both subset.
    """

    def test_partition_agreement_above_chance(self, seg_result, noseg_result):
        seg_out, _, _ = seg_result
        noseg_out, _, _ = noseg_result

        seg_lbl = seg_out.set_index("transcript_id")["stitched"].astype(str)
        noseg_lbl = noseg_out.set_index("transcript_id")["stitched"].astype(str)

        idx = seg_lbl.index.intersection(noseg_lbl.index)
        a = seg_lbl.loc[idx]
        b = noseg_lbl.loc[idx]
        # ARI on assigned-in-both
        mask = (a != "-1") & (b != "-1")
        if mask.sum() < 2:
            pytest.skip("not enough assigned-in-both tx")
        ari = adjusted_rand_score(a[mask].values, b[mask].values)
        ami = adjusted_mutual_info_score(a[mask].values, b[mask].values)
        assert ari > 0.10, (
            f"ARI(seg, noseg) = {ari:.3f} below chance threshold (0.10)"
        )
        assert ami > 0.10, (
            f"AMI(seg, noseg) = {ami:.3f} below chance threshold (0.10)"
        )


# ============================================================================
# Phase1-Rerank opt-in smoke tests
# ============================================================================

def test_rerank_off_omits_stage_seg_smoke(monkeypatch, synthetic_inputs):
    """With PHASE1_RERANK_ENABLED=False, Phase1-Rerank is not recorded
    in the SEG progression. Default is now True (promoted 2026-05-13)
    so this test monkey-patches it off."""
    import tests._pipeline_runner as runner
    monkeypatch.setattr(runner, "PHASE1_RERANK_ENABLED", False)
    df, panel, _gt = synthetic_inputs
    _df_out, progression = runner.run_segmented_pipeline(df, panel)
    stage_names = [p["stage"] for p in progression]
    assert "Phase1-Rerank" not in stage_names


def test_rerank_on_records_stage_seg_smoke(monkeypatch, synthetic_inputs):
    """Flipping PHASE1_RERANK_ENABLED=True records the new stage between
    Split-Phase1 and Phase1-QC in the SEG progression.

    The synthetic fixture uses ``is_nuclear``; rename it to
    ``overlaps_nucleus`` so the nuclear-seed prune path (and rerank) fires.
    """
    import tests._pipeline_runner as runner
    monkeypatch.setattr(runner, "PHASE1_RERANK_ENABLED", True)
    df, panel, _gt = synthetic_inputs
    # Expose the nuclear flag under the name the pipeline expects so that
    # the nuclear-seed prune path (and therefore Phase1-Rerank) is taken.
    df_nuc = df.rename(columns={"is_nuclear": "overlaps_nucleus"})
    _df_out, progression = runner.run_segmented_pipeline(df_nuc, panel)
    stage_names = [p["stage"] for p in progression]
    idx_split = stage_names.index("Split-Phase1")
    idx_qc = stage_names.index("Phase1-QC")
    idx_rerank = stage_names.index("Phase1-Rerank")
    assert idx_split < idx_rerank < idx_qc


def test_rerank_composes_with_reassign_1c(monkeypatch, synthetic_inputs):
    """Both opt-in stages on simultaneously: Phase1-Reassign-1c sits
    between Prune and Split-Phase1; Phase1-Rerank sits between
    Split-Phase1 and Phase1-QC. Confirms the order is exactly
    Prune → Phase1-Reassign-1c → Split-Phase1 → Phase1-Rerank → Phase1-QC."""
    import tests._pipeline_runner as runner
    monkeypatch.setattr(runner, "PHASE1_REASSIGN_AFTER_1C", True)
    monkeypatch.setattr(runner, "PHASE1_RERANK_ENABLED", True)
    df, panel, _gt = synthetic_inputs
    # Force the nuclear-seed path: rename `is_nuclear` -> `overlaps_nucleus`
    # so the column guard passes and both opt-in stages fire.
    df = df.rename(columns={"is_nuclear": "overlaps_nucleus"})
    _df_out, progression = runner.run_segmented_pipeline(df, panel)
    stage_names = [p["stage"] for p in progression]
    idx_prune = stage_names.index("Prune")
    idx_reassign = stage_names.index("Phase1-Reassign-1c")
    idx_split = stage_names.index("Split-Phase1")
    idx_rerank = stage_names.index("Phase1-Rerank")
    idx_qc = stage_names.index("Phase1-QC")
    assert idx_prune < idx_reassign < idx_split < idx_rerank < idx_qc


# ============================================================================
# Phase B equivalence: cfg=None  ≡  cfg=load_config()
# ============================================================================
#
# When ``cfg is None`` (the legacy call path), ``_resolve_pipeline_cfg``
# builds a ``PipelineConfig`` from the current module-global constants.
# With the TOML defaults in lock-step with the dataclass defaults
# (verified by ``test_config.py``), ``cfg=load_config()`` must produce
# bit-identical per-tx labels. These tests are the Phase-A → Phase-B
# safety net: any future wiring change that breaks this equivalence
# will fail here.


def test_seg_pipeline_cfg_none_matches_load_config(synthetic_inputs):
    """SEG runner: ``cfg=None`` and ``cfg=load_config()`` produce
    bit-identical per-tx labels on the FullVolume fixture."""
    from tracer.config import load_config

    df, panel, _gt = synthetic_inputs
    df_a, _ = run_segmented_pipeline(df.copy(), panel)
    df_b, _ = run_segmented_pipeline(df.copy(), panel, cfg=load_config())
    a = df_a["stitched"].astype(str).reset_index(drop=True)
    b = df_b["stitched"].astype(str).reset_index(drop=True)
    n_diff = int((a != b).sum())
    assert n_diff == 0, (
        f"SEG cfg=None vs cfg=load_config(): {n_diff}/{len(a)} tx labels "
        "differ — Phase-B wiring is no longer bit-exact with the "
        "module-global fallback. Inspect `_resolve_pipeline_cfg` and "
        "the call sites for drift."
    )


def test_noseg_pipeline_cfg_none_matches_load_config(synthetic_inputs):
    """NOSEG runner: same equivalence guarantee as SEG."""
    from tracer.config import load_config

    df, panel, _gt = synthetic_inputs
    df_a, _ = run_noseg_pipeline(df.copy(), panel)
    df_b, _ = run_noseg_pipeline(df.copy(), panel, cfg=load_config())
    a = df_a["stitched"].astype(str).reset_index(drop=True)
    b = df_b["stitched"].astype(str).reset_index(drop=True)
    n_diff = int((a != b).sum())
    assert n_diff == 0, (
        f"NOSEG cfg=None vs cfg=load_config(): {n_diff}/{len(a)} tx "
        "labels differ — Phase-B wiring drift detected."
    )


def test_seg_pipeline_witness_rank_policy_engages(synthetic_inputs):
    """Smoke test: ``rescue.rank_policy = "witness"`` actually changes
    per-tx labels relative to ``"distance"`` on the FullVolume fixture.

    This is a *liveness* check — it guards against the witness branch
    silently degrading to a no-op (e.g., due to a refactor that misroutes
    the rank dispatch). It does NOT make a claim about which policy is
    better; that's the bench's job.
    """
    from dataclasses import replace as dc_replace

    from tracer.config import load_config

    df, panel, _gt = synthetic_inputs
    # Build the distance-policy config explicitly via dc_replace —
    # the default config has rank_policy="witness" since 2026-05-15.
    cfg_witness = load_config()
    cfg_dist = dc_replace(
        cfg_witness,
        rescue=dc_replace(cfg_witness.rescue, rank_policy="distance"),
        final_rescue=dc_replace(cfg_witness.final_rescue, rank_policy="distance"),
    )
    assert cfg_witness.rescue.rank_policy == "witness"
    assert cfg_dist.rescue.rank_policy == "distance"

    df_a, _ = run_segmented_pipeline(df.copy(), panel, cfg=cfg_dist)
    df_b, _ = run_segmented_pipeline(df.copy(), panel, cfg=cfg_witness)
    a = df_a["stitched"].astype(str).reset_index(drop=True)
    b = df_b["stitched"].astype(str).reset_index(drop=True)
    n_diff = int((a != b).sum())
    # Witness branch must take effect on at least one tx — on this
    # dense fixture it changes ~9% in practice. The lower bound is set
    # at 1 so the test catches the worst regression (witness = noop)
    # without overfitting to a specific synthetic.
    assert n_diff >= 1, (
        "rescue.rank_policy='witness' produced identical labels to "
        "'distance' — the witness branch is not being exercised. "
        "Check `reassign_unassigned_grid_pool`'s rank-policy dispatch "
        "and `_resolve_pipeline_cfg`/call-site plumbing."
    )
