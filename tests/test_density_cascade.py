"""Smoke tests for `tracer.density_cascade`.

Catches regressions in:
  - `auto_floor_from_coverage` rule (target_cov, hard_min, R-Moore dilation)
  - `density_cascade_phase1` end-to-end on a small ROI
  - `cascade_as_residual_handler` label format (partial-form `cascade_<n>-1`)
  - label-parse consistency: cascade labels must classify as 'partial'

Run via: pytest tests/test_density_cascade.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

from tracer.density_cascade import (
    auto_floor_from_coverage,
    auto_thresholds,
    cascade_as_residual_handler,
    density_cascade_phase1,
    _build_grid,
    _moore_dilate,
)


# ============================================================================
# auto_floor_from_coverage — pure-function unit tests
# ============================================================================
class TestAutoFloor:
    def test_empty_grid_returns_hard_min(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        floor, curve = auto_floor_from_coverage(grid, target_cov=0.65)
        assert floor == 2  # hard_min default
        assert curve == []

    def test_single_dense_bin_picks_max_threshold(self):
        # One bin with 5 tx; target=0.65 → 65 % coverage of 5 tx = >= 4 tx.
        # Anchoring at any threshold up to 5 captures the full 5 tx via
        # the R=1 Moore-dilated mask, so coverage = 100 %. Largest n is 5.
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[5, 5] = 5
        floor, curve = auto_floor_from_coverage(grid, target_cov=0.65)
        assert floor == 5  # picks the strictest threshold meeting target
        # Curve should be descending thresholds from 5 down
        assert curve[0][0] == 5
        assert curve[0][1] == 1.0  # 100 % coverage at thr=5

    def test_sparse_pool_falls_back_to_hard_min(self):
        # 100 isolated single-tx bins → no R=1 nbhd captures more than 1
        # bin's worth → coverage at thr=2 is 0 (no bins reach thr).
        grid = np.zeros((100, 100), dtype=np.int32)
        for i in range(0, 100, 10):
            grid[i, i] = 1
        floor, _ = auto_floor_from_coverage(grid, target_cov=0.65,
                                              hard_min=2)
        assert floor == 2  # never reaches 65 %, falls back

    def test_dense_uniform_grid_picks_high_floor(self):
        # 50x50 grid, every bin = 10 tx → bin_tail at thr=10 is 100 %, R=1
        # dilated mask covers the full grid → coverage = 100 %.
        # Largest n satisfying coverage>=0.65 is 10.
        grid = np.full((50, 50), 10, dtype=np.int32)
        floor, curve = auto_floor_from_coverage(grid, target_cov=0.65,
                                                  hard_min=2)
        assert floor == 10
        assert curve[0][1] == 1.0  # full coverage at thr=10

    def test_target_cov_threshold_changes_floor(self):
        # Pyramid: bin counts increase toward center.
        grid = np.zeros((20, 20), dtype=np.int32)
        grid[10, 10] = 8
        grid[9:12, 9:12] = np.maximum(grid[9:12, 9:12], 3)
        grid[8:13, 8:13] = np.maximum(grid[8:13, 8:13], 2)
        # 8 tx in centre + 9 bins of >= 3 + 25 bins of >= 2.
        f_low, _ = auto_floor_from_coverage(grid, target_cov=0.50)
        f_high, _ = auto_floor_from_coverage(grid, target_cov=0.99)
        # Higher coverage target requires walking further down the cascade
        # → smaller floor.
        assert f_high <= f_low

    def test_R2_caps_anchor_count_differently(self):
        # Same grid, R=2 territory dilates wider so coverage at any
        # threshold >= R=1 coverage. Floor at R=2 should be >= floor at R=1
        # (reach target with stricter threshold under wider territory).
        grid = np.zeros((30, 30), dtype=np.int32)
        rng = np.random.default_rng(seed=0)
        idx = rng.integers(0, 30, size=(50, 2))
        for x, y in idx:
            grid[x, y] = rng.integers(2, 6)
        f_r1, _ = auto_floor_from_coverage(grid, target_cov=0.65, R=1)
        f_r2, _ = auto_floor_from_coverage(grid, target_cov=0.65, R=2)
        assert f_r2 >= f_r1


# ============================================================================
# Moore dilation — sanity check the helper
# ============================================================================
class TestMooreDilate:
    def test_R1_dilates_3x3(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        out = _moore_dilate(mask, R=1)
        assert out.sum() == 9  # 3x3 centered on (5,5)
        assert out[4:7, 4:7].all()

    def test_R2_dilates_5x5(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        out = _moore_dilate(mask, R=2)
        assert out.sum() == 25
        assert out[3:8, 3:8].all()


# ============================================================================
# density_cascade_phase1 + cascade_as_residual_handler — integration
# ============================================================================
class TestCascadeIntegration:
    @pytest.fixture(scope="class")
    def roi(self):
        # `data_loader` lives in benchmarks/ (untracked in some
        # checkouts including CI). Skip these integration tests when
        # the helper isn't importable; the unit-level cascade behavior
        # is exercised by TestAutoFloor / TestMooreDilate above.
        pytest.importorskip(
            "data_loader",
            reason="benchmarks/data_loader.py unavailable (not on sys.path)",
        )
        from data_loader import load_roi_df, DEFAULT_PROJECT_DIR
        # The lung-cancer ROI parquet + NPMI panel are large data files
        # not committed to the repo (DEFAULT_PROJECT_DIR points at a
        # developer-local tutorials/ tree). On CI — and any checkout
        # without those files — the loads raise FileNotFoundError; skip
        # rather than error. importorskip above only catches "module
        # missing", not "data missing".
        try:
            df = load_roi_df(half_side_um=250.0,
                              roi_center_xy=(1818.7, 2186.8)).reset_index(drop=True)
            panel = pd.read_csv(
                DEFAULT_PROJECT_DIR / "data" / "lung_cancer_npmi.csv"
            )
        except FileNotFoundError as e:
            pytest.skip(
                f"lung_cancer integration data unavailable: {e}"
            )
        for c in ("transcript_id", "cell_id", "feature_name"):
            df[c] = df[c].astype(str)
        df["overlaps_nucleus"] = df["overlaps_nucleus"].astype(bool)
        genes = set(df["feature_name"].unique())
        panel = panel[panel["gene_i"].isin(genes)
                       & panel["gene_j"].isin(genes)]
        return df, panel

    def test_cascade_runs_end_to_end_with_auto_thresholds(self, roi):
        df, panel = roi
        out = density_cascade_phase1(
            df, panel, G=2.0, thresholds="auto",
            territory_radius_bins=1,
            pmi_threshold=0.05, min_anchor_tx=3,
            auto_target_cov=0.65, auto_hard_min=2,
        )
        assert out["n_anchors"] > 0
        assert out["n_tx_assigned"] > 0
        assert out["n_tx_assigned"] <= out["n_tx_valid"]
        assert isinstance(out["thresholds_used"], list)
        # Auto-threshold should have produced a descending list
        thrs = out["thresholds_used"]
        if len(thrs) >= 2:
            assert thrs[0] > thrs[-1]
        assert out["coverage_curve"]  # non-empty diagnostic curve

    def test_auto_thresholds_helper(self, roi):
        df, _ = roi
        thrs = auto_thresholds(df, G=2.0, target_cov=0.65, R=1, hard_min=2)
        assert len(thrs) >= 1
        # Floor (last entry) should be >= hard_min
        assert thrs[-1] >= 2

    def test_residual_handler_emits_partial_labels(self, roi):
        df, panel = roi
        # Synthetic post-Rescue residual: mark half the tx as "-1"
        df_in = df.copy()
        df_in["tracer_id"] = "real_cell"
        n = len(df_in)
        df_in.loc[df_in.index[: n // 2], "tracer_id"] = "-1"

        # Build minimal aux dict
        from tracer.pruning import build_sparse_pmi_matrix_from_long
        _, gene_to_idx, W = build_sparse_pmi_matrix_from_long(
            panel, metric_col="PMI")
        aux = {"gene_to_idx": gene_to_idx, "W": W}

        df_out = cascade_as_residual_handler(
            df_in, aux=aux, panel=panel,
            entity_col="tracer_id",
            thresholds="auto",
            auto_target_cov=0.65, auto_hard_min=2,
        )
        # Non-residual labels untouched
        assert (df_out.loc[df_out["tracer_id"] == "real_cell"]
                .shape[0] >= 1)
        # Cascade labels must be in partial form: cascade_<n>-1
        cas_labels = df_out.loc[
            df_out["tracer_id"].astype(str).str.startswith("cascade_"),
            "tracer_id"
        ].astype(str)
        if len(cas_labels) > 0:
            assert all("-" in lbl for lbl in cas_labels.unique()), (
                "Cascade labels must contain '-' for Stitch partial-eligibility"
            )
            # All should end with "-1" (depth-1 partial)
            assert all(lbl.endswith("-1") for lbl in cas_labels.unique())


# ============================================================================
# Pipeline-level smoke: cascade label classifies as 'partial'
# ============================================================================
class TestClassifyConsistency:
    def test_cascade_partial_label_classified_as_partial(self):
        """Cascade labels (``cascade_<N>-1`` and ``cascade_<N>-1-1``)
        must be classified as ``partial`` by the canonical label parser
        so Stitch / per-stage accounting count them in the partial pool.
        """
        from tracer._etype import infer_etype_from_label
        kinds = list(
            np.asarray(
                infer_etype_from_label(
                    pd.Series(["cascade_5-1", "cascade_42-1", "cascade_5-1-1"])
                )
            ).astype(str)
        )
        assert kinds == ["partial", "partial", "partial"]
