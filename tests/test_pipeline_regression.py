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

    s = df_out["stitched"].astype(str)
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

    seg_lbl = seg_out.set_index("transcript_id")["stitched"].astype(str)
    noseg_lbl = noseg_out.set_index("transcript_id")["stitched"].astype(str)
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
