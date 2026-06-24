"""Tests for the Phase 1b hybrid admission gate.

Covers the unified ``_admission_test`` Cython helper exposed through
``prune_transcripts_nuclear_seed(..., veto_mode=...)``.

The unit-level test uses a synthetic panel mirroring the EPCAM-vs-
macrophage case from the design spec: a candidate gene whose mean PMI
to the seed is positive (because of many housekeeping/IFN positives)
but whose PMI against a minority of opposing-lineage markers is
strongly negative. Under ``veto_mode="mean"`` the candidate admits;
under ``veto_mode="hybrid"`` the percentile gate vetoes it.

The integration test reads the PDAC EMT 50 µm ROI from the developer
project tree and verifies the EPCAM-jikageak case directly. It skips
gracefully when the parquet is not on disk, following the pattern in
``tests/test_density_cascade.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on the import path when run outside the project root.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tracer.pruning import prune_transcripts_nuclear_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic case: mean dilution vs hybrid percentile gate
# ---------------------------------------------------------------------------
def _build_synthetic_emt_panel(
    n_seed_pos: int = 7,
    n_seed_neg: int = 3,
    pos_pmi: float = 0.5,
    neg_pmi: float = -0.7,
):
    """Synthesize a tiny PMI panel + transcript df mirroring the
    EPCAM-vs-macrophage 1b case.

    - Seed: 10 unique nuclear genes (g0..g9). 7 are "housekeeping/IFN"
      positives toward CAND (pos_pmi); 3 are "opposing lineage" with
      strongly negative PMI to CAND (neg_pmi).
    - CAND: a cytoplasmic transcript (gene id 10) for a single cell.
      Under default ``veto_mode="mean"`` it admits because the mean
      across all 10 seed genes is ~+0.14; under ``"hybrid"`` it should
      veto because p25 of real-signal PMIs is well below the admit
      threshold (0.2 default).

    Returns ``(df, npmi_panel)``.
    """
    gene_names = [f"g{i}" for i in range(n_seed_pos + n_seed_neg)] + ["CAND"]
    n_genes = len(gene_names)

    # Build symmetric PMI long-form. Within-seed pairs all positive
    # (so the greedy 1a prune keeps everything in the seed).
    rows = []
    seed_gids = list(range(n_seed_pos + n_seed_neg))
    # Strong positive PMI between every seed pair: keeps the seed.
    for i in range(len(seed_gids)):
        for j in range(i + 1, len(seed_gids)):
            rows.append((gene_names[seed_gids[i]], gene_names[seed_gids[j]], 0.4))
    # CAND vs seed: pos for the first n_seed_pos, neg for the rest.
    cand_name = "CAND"
    for k in range(n_seed_pos):
        rows.append((gene_names[k], cand_name, pos_pmi))
    for k in range(n_seed_pos, n_seed_pos + n_seed_neg):
        rows.append((gene_names[k], cand_name, neg_pmi))
    panel = pd.DataFrame(rows, columns=["gene_i", "gene_j", "PMI"])

    # Build the transcript dataframe: one cell with one transcript per
    # seed gene (all nuclear), plus one cytoplasmic CAND transcript.
    tx_rows = []
    tid = 0
    for k in range(n_seed_pos + n_seed_neg):
        tx_rows.append({
            "transcript_id": f"t{tid}",
            "cell_id": "C1",
            "feature_name": gene_names[k],
            "x_location": 0.0 + k * 0.1,
            "y_location": 0.0,
            "overlaps_nucleus": True,
        })
        tid += 1
    # Candidate transcript (cytoplasmic — eligible for 1b admission)
    tx_rows.append({
        "transcript_id": f"t{tid}",
        "cell_id": "C1",
        "feature_name": cand_name,
        "x_location": 1.0,
        "y_location": 0.0,
        "overlaps_nucleus": False,
    })
    df = pd.DataFrame(tx_rows)
    return df, panel


def _cand_admitted(out_df: pd.DataFrame) -> bool:
    """True when the CAND transcript was admitted to the main cell."""
    row = out_df[out_df["feature_name"] == "CAND"].iloc[0]
    return str(row["tracer_id"]) == "C1"


@pytest.mark.parametrize(
    "veto_mode,expected_admit",
    [
        ("mean", True),
        ("hybrid", False),
    ],
)
def test_phase1b_admission_mean_vs_hybrid(veto_mode, expected_admit):
    """Synthetic EMT-like cell: mean admits CAND, hybrid vetoes it.

    Diluted-mean case: 7 positive PMI pairs (housekeeping-like, +0.5)
    + 3 negative PMI pairs (opposing-lineage, -0.7). Mean ≈ +0.14
    > legacy threshold (the prune ``threshold`` arg is 1e-5 here for
    the bad-edge step), so mean admits. p25 of real-signal PMIs is
    well below the hybrid ``mean_admit_threshold`` (0.2 default), so
    hybrid vetoes.
    """
    df, panel = _build_synthetic_emt_panel()
    # Use a tiny bad-edge threshold so the synthetic positive within-seed
    # PMIs all pass the 1a prune cleanly (no incidental drops).
    df_out, _aux = prune_transcripts_nuclear_seed(
        df, panel,
        cell_id_col="cell_id",
        gene_col="feature_name",
        nuclear_col="overlaps_nucleus",
        threshold=0.0,
        unassigned_id="-1",
        metric_col="PMI",
        nan_fill=0.0,
        min_nuclear_genes=3,
        nuclear_only_admit=False,
        veto_mode=veto_mode,
        mean_admit_threshold=0.2,
        aggregator_percentile=25.0,
        real_signal_threshold=0.05,
    )
    admitted = _cand_admitted(df_out)
    if expected_admit:
        assert admitted, (
            f"Under veto_mode='{veto_mode}' CAND should admit to C1 "
            f"(mean dilution lets it pass), but tracer_id was "
            f"{df_out[df_out['feature_name'] == 'CAND']['tracer_id'].iloc[0]!r}"
        )
    else:
        assert not admitted, (
            f"Under veto_mode='{veto_mode}' CAND must NOT admit to C1 "
            f"(hybrid p25 gate should veto), but it admitted."
        )


# ---------------------------------------------------------------------------
# Integration: PDAC EMT 50 µm ROI — EPCAM-vs-jikageak-1
# ---------------------------------------------------------------------------
class TestPDACEMTHybridAdmission:
    @pytest.fixture(scope="class")
    def roi(self):
        # `data_loader` lives in benchmarks/ (untracked in some
        # checkouts including CI). Skip these integration tests when
        # the helper isn't importable; the synthetic unit test above
        # exercises the gate semantics.
        pytest.importorskip(
            "data_loader",
            reason="benchmarks/data_loader.py unavailable (not on sys.path)",
        )
        from data_loader import load_roi_df, DEFAULT_PROJECT_DIR  # type: ignore[import-not-found]

        # PDAC EMT 50 µm ROI: x∈[10500,10550], y∈[1750,1800]. The
        # design spec calls out cell jikageak-1 specifically.
        try:
            df = load_roi_df(
                half_side_um=25.0,
                roi_center_xy=(10525.0, 1775.0),
                project="pdac",
            ).reset_index(drop=True)
            panel = pd.read_csv(
                DEFAULT_PROJECT_DIR / "data" / "pdac_npmi.csv"
            )
        except (FileNotFoundError, TypeError, ValueError) as e:
            pytest.skip(f"PDAC integration data unavailable: {e}")
        for c in ("transcript_id", "cell_id", "feature_name"):
            df[c] = df[c].astype(str)
        df["overlaps_nucleus"] = df["overlaps_nucleus"].astype(bool)
        # Guard: ROI must contain the target cell.
        if "jikageak-1" not in set(df["cell_id"]):
            pytest.skip("jikageak-1 not present in this ROI snapshot")
        return df, panel

    def test_hybrid_rejects_epcam_in_jikageak_1(self, roi):
        df, panel = roi
        df_out, _ = prune_transcripts_nuclear_seed(
            df, panel,
            cell_id_col="cell_id",
            gene_col="feature_name",
            nuclear_col="overlaps_nucleus",
            threshold=0.2, unassigned_id="-1",
            metric_col="PMI", nan_fill=0.0,
            min_nuclear_genes=3,
            seed_coherence_floor=0.10,
            nuclear_only_admit=True,
            veto_mode="hybrid",
            mean_admit_threshold=0.5,
            aggregator_percentile=25.0,
            real_signal_threshold=0.05,
        )
        # Identify EPCAM tx originally assigned to jikageak-1 by input
        # cell_id. Under hybrid, none of them should remain in the main
        # entity for jikageak-1 (they should be demoted to partial or
        # unassigned).
        mask = (df_out["feature_name"] == "EPCAM") & (df_out["cell_id"] == "jikageak-1")
        if not mask.any():
            pytest.skip("No EPCAM tx mapped to jikageak-1 in this ROI snapshot")
        labels = set(df_out.loc[mask, "tracer_id"].astype(str))
        assert "jikageak-1" not in labels, (
            "Under veto_mode='hybrid' EPCAM should not be admitted to "
            f"jikageak-1's main entity. Labels were: {sorted(labels)}"
        )
