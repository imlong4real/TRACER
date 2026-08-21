"""Snapshot regression tests for pipeline outputs on synthetic data.

Each test fingerprints the pipeline output (entity counts, partition
ARI/AMI vs ground truth, per-stage progression) and compares against
``tests/references/<variant>.json``. If the current output diverges
beyond the per-metric tolerance, the test fails with a structured diff
plus an explicit instruction for regenerating the reference if the
change is intentional.

Maintainer workflow:

  1. CI runs ``pytest`` → references are enforced.
  2. A pipeline change diverges from a reference → CI fails with a
     diff in the test log.
  3. Maintainer reviews: regression (fix code) or improvement (update
     reference).
  4. To update: ``TRACER_UPDATE_REFERENCES=1 pytest tests/test_pipeline_regression.py``
     locally, then commit ``tests/references/*.json``.
"""
from __future__ import annotations

import pytest
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from tests._regression_helpers import assert_matches_reference
from tests._pipeline_runner import run_segmented_pipeline, run_noseg_pipeline
from tests.synthetic import (
    make_synthetic_transcripts,
    make_synthetic_npmi_panel_for_transcripts,
)


CELLS_KW = dict(
    n_cells=8,
    voxels_per_cell_mean=80,
    tx_per_cell=25,
    n_genes=12,
    n_types=3,
    domain_z_um=10.0,
    nuclear_layers=2,
)
SECTION_Z = (2.5, 7.5)


# Per-metric tolerances. Counts: exact equality (deterministic).
# Partition metrics: small ε for float-PMI ranking edge cases.
TOLERANCES_COUNTS = {
    "n_cells": 0,
    "n_partials": 0,
    "n_components": 0,
    "n_unassigned_tx": 0,
}
TOLERANCES_PARTITION = {
    "ari_vs_truth": 0.02,
    "ami_vs_truth": 0.02,
    "ari_seg_vs_noseg": 0.02,
    "ami_seg_vs_noseg": 0.02,
    "coverage_pct": 0.5,  # half a percentage point
}


@pytest.fixture(scope="module")
def synthetic_inputs():
    df, gt = make_synthetic_transcripts(**CELLS_KW, seed=42)
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    return df, panel, gt


def _fingerprint(df_out, progression, gt) -> dict:
    """Compute the fingerprint dict to compare against the reference."""
    from tracer._etype import infer_etype_from_label

    s = df_out["tracer_id"].astype(str)
    if "_etype" in df_out.columns:
        etypes = df_out["_etype"].astype(str)
    else:
        etypes = pd.Series(
            np.asarray(infer_etype_from_label(s)).astype(str),
            index=df_out.index,
        )
    types = etypes.where(
        etypes.isin(["cell", "partial", "component"]),
        other="unassigned",
    )
    n_ent = s.groupby(types).nunique().to_dict()
    n_tx = types.value_counts().to_dict()

    n_total = len(s)
    n_assigned = n_total - int(n_tx.get("unassigned", 0))
    coverage_pct = round(100 * n_assigned / max(n_total, 1), 2)

    truth = df_out["cell_id"].astype(str).values
    out = s.values
    mask = (out != "-1") & (truth != "-1")
    if mask.sum() >= 2:
        ari_truth = round(float(adjusted_rand_score(truth[mask], out[mask])), 4)
        ami_truth = round(float(adjusted_mutual_info_score(truth[mask], out[mask])), 4)
    else:
        ari_truth = ami_truth = float("nan")

    # Strip non-deterministic timing fields (_ts / stage_seconds) from
    # the progression before fingerprinting — they change every run.
    progression_clean = [
        {k: v for k, v in stage.items() if k not in {"_ts", "stage_seconds"}}
        for stage in progression
    ]
    return {
        "n_cells": int(n_ent.get("cell", 0)),
        "n_partials": int(n_ent.get("partial", 0)),
        "n_components": int(n_ent.get("component", 0)),
        "n_unassigned_tx": int(n_tx.get("unassigned", 0)),
        "coverage_pct": coverage_pct,
        "ari_vs_truth": ari_truth,
        "ami_vs_truth": ami_truth,
        "stage_progression": progression_clean,
    }


def test_regression_segmented(synthetic_inputs):
    df, panel, gt = synthetic_inputs
    df_out, prog = run_segmented_pipeline(df, panel)
    fp = _fingerprint(df_out, prog, gt)
    tol = {**TOLERANCES_COUNTS, **TOLERANCES_PARTITION}
    assert_matches_reference("segmented", fp, tol)


def test_regression_noseg(synthetic_inputs):
    df, panel, gt = synthetic_inputs
    df_out, prog = run_noseg_pipeline(df, panel)
    fp = _fingerprint(df_out, prog, gt)
    # Under noseg, cell_id was overwritten to "-1" so ari_vs_truth is
    # NaN (no ground-truth cell_id to compare on the merged DataFrame).
    # We still record other metrics.
    tol = {**TOLERANCES_COUNTS, **TOLERANCES_PARTITION}
    assert_matches_reference("noseg", fp, tol)


OFFPANEL_DROP = 3   # genes present in transcripts but absent from the panel


@pytest.fixture(scope="module")
def offpanel_inputs(synthetic_inputs):
    """Same transcripts, but a panel that omits some of their genes.

    This is the shape the off-panel rescue targets: housekeeping genes such
    as ACTB self-eliminate from a PMI panel (ubiquitous co-detection carries
    no information), so their transcripts exist in the data with no PMI edge
    to anything. Every PMI-driven gate is a no-op on them, and before
    ``RescueConfig.offpanel_first_entity`` they could never be rescued and
    were discarded at Finalize.

    The stock ``synthetic_inputs`` panel covers the full gene vocabulary, so
    it cannot exercise that path at all -- which is exactly why the existing
    references do not move when the behaviour changes.
    """
    df, panel, gt = synthetic_inputs
    genes = sorted(set(df["feature_name"].astype(str)))
    dropped = set(genes[:OFFPANEL_DROP])
    slim = panel[~panel["gene_i"].isin(dropped) & ~panel["gene_j"].isin(dropped)]
    return df, slim.reset_index(drop=True), gt, sorted(dropped)


def test_regression_segmented_offpanel(offpanel_inputs):
    """Guards the off-panel (zero-PMI) rescue and the entity accounting around
    it. Transcripts of the dropped genes carry no PMI evidence whatsoever, so
    their fate is decided purely by whether off-panel proximity assignment is
    enabled -- making this fingerprint sensitive to a default the other
    regression variants cannot see.

    What this fixture currently pins (vs. the pre-``offpanel_first_entity``
    behaviour), and why it is worth keeping:

    * ``n_unassigned_tx`` 37 -> 16, ``coverage_pct`` 81.5 -> 92.0. The change
      lands entirely in Post-Group Rescue; every earlier stage is identical.
    * It is strictly additive to the existing partition: all 163 already-
      assigned transcripts keep their exact label, and ARI restricted to that
      identical subset is unchanged at 0.9185.
    * But whole-fixture ``ari_vs_truth`` FALLS 0.9185 -> 0.7759, because all
      21 newly-assigned transcripts belong to synthetic cell "3" -- a cell
      neither branch manages to reconstruct (``n_cells`` is 7, not 8, on both
      sides). With no "cell 3" entity in their Moore neighborhood to rejoin,
      proximity assignment scatters them across cells 1/4/5/7, so 0 of 21 land
      in their true cell.

    That is the failure mode this snapshot exists to keep visible: when a cell
    is dissolved upstream, off-panel proximity rescue converts a clean
    "unassigned" signal into false-positive assignments in its neighbours --
    which is the contamination TRACER exists to remove. The fixture is
    deliberately harsh (3 of 12 genes dropped), so the ratio is not a
    prediction for real panels; the behaviour it exposes is the point.
    """
    df, panel, gt, dropped = offpanel_inputs
    panel_genes = set(panel["gene_i"]) | set(panel["gene_j"])
    assert set(dropped) and not (set(dropped) & panel_genes)
    assert df["feature_name"].isin(dropped).any()

    df_out, prog = run_segmented_pipeline(df, panel)
    fp = _fingerprint(df_out, prog, gt)
    tol = {**TOLERANCES_COUNTS, **TOLERANCES_PARTITION}
    assert_matches_reference("segmented_offpanel", fp, tol)


def test_regression_segmented_section():
    """Regression on tissue-section-extracted slab. Different fingerprint
    than full-volume run because clipped cells lose tx."""
    df, gt = make_synthetic_transcripts(
        **CELLS_KW, section_z_range_um=SECTION_Z, seed=42,
    )
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    df_out, prog = run_segmented_pipeline(df, panel)
    fp = _fingerprint(df_out, prog, gt)
    fp["n_clipped_cells"] = gt["n_clipped_cells"]
    tol = {**TOLERANCES_COUNTS, **TOLERANCES_PARTITION, "n_clipped_cells": 0}
    assert_matches_reference("segmented_section", fp, tol)


def test_regression_seg_vs_noseg(synthetic_inputs):
    """Cross-mode partition agreement between segmented and no-seg
    runs on the same input."""
    df, panel, gt = synthetic_inputs
    seg_out, _ = run_segmented_pipeline(df, panel)
    noseg_out, _ = run_noseg_pipeline(df, panel)

    seg_lbl = seg_out.set_index("transcript_id")["tracer_id"].astype(str)
    noseg_lbl = noseg_out.set_index("transcript_id")["tracer_id"].astype(str)
    idx = seg_lbl.index.intersection(noseg_lbl.index)
    a = seg_lbl.loc[idx]
    b = noseg_lbl.loc[idx]
    mask = (a != "-1") & (b != "-1")
    if mask.sum() >= 2:
        ari = round(float(adjusted_rand_score(a[mask].values, b[mask].values)), 4)
        ami = round(float(adjusted_mutual_info_score(a[mask].values, b[mask].values)), 4)
    else:
        ari = ami = float("nan")

    fp = {"ari_seg_vs_noseg": ari, "ami_seg_vs_noseg": ami,
          "n_assigned_in_both": int(mask.sum())}
    tol = {"ari_seg_vs_noseg": 0.02, "ami_seg_vs_noseg": 0.02,
           "n_assigned_in_both": 5}
    assert_matches_reference("seg_vs_noseg", fp, tol)
