"""Regression lock for the SEG nuclear-seed Prune under the 2026-08-17
default config (``phase1.nuclear_only_admit=False``,
``phase1.admit_independent=False``).

Why this file exists
--------------------
The pipeline-regression fixtures in ``test_pipeline_regression.py`` build
transcripts carrying an ``is_nuclear`` column, which routes
``run_segmented_pipeline`` through ``prune_transcripts_fast`` — NOT
``prune_transcripts_nuclear_seed``, where ``nuclear_only_admit`` /
``admit_independent`` actually act (the SEG nuclear-seed branch is taken
only when the transcript frame has an ``overlaps_nucleus`` column; see
``pipeline.run_segmented_pipeline`` ~L1649). So no existing test
exercises the changed behavior. This file closes that gap.

What is locked
--------------
1. ``nuclear_only_admit=False`` — cytoplasm is ADMITTED at Prune.
   Under the default config, cytoplasmic transcripts (``overlaps_nucleus
   =False``) whose gene fits a cell's nuclear seed by PMI are admitted to
   that cell at the Prune stage (``_etype=="cell"`` for the cell's own
   ``cell_id``). Re-running the SAME input with the OLD default
   ``nuclear_only_admit=True`` defers every one of them (they leave the
   Prune stage unassigned, ``tracer_id=="-1"``). The two runs must
   differ on that population — this is the sensitivity that catches a
   default revert.

The behavior is asserted on the **Prune-stage snapshot** (captured via a
wrap of ``pipeline._record_stage``), not the final pipeline output,
because downstream Rescue/Group/Stitch re-route deferred cytoplasm and
would mask the Prune-stage difference these knobs control.

Behavior NOT locked here (deliberately skipped) — see the skipped test at
the bottom for ``admit_independent`` and the reasoning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tracer import pipeline as _pipeline  # noqa: E402
from tracer.config import load_config  # noqa: E402

from tests.synthetic import (  # noqa: E402
    make_synthetic_transcripts,
    make_synthetic_npmi_panel_for_transcripts,
)

# Unassigned sentinels (mirrors the etype "unknown" family).
_UNASSIGNED = {"-1", "-1.0", "DROP", "UNASSIGNED", "nan", "None", ""}


# ----------------------------------------------------------------------
# Fixture — a SEG (nuclear-seed) input that reliably takes the
# nuclear-seed prune branch AND gives each cell >= 3 unique nuclear genes
# (so the whole-cell FALLBACK path is NOT taken; the fallback path admits
# cytoplasm regardless of nuclear_only_admit and would neutralize the
# knob). nuclear_layers=3 + tx_per_cell=40 yields ~6 unique nuclear genes
# per cell here. 240 tx total → the whole pipeline runs in ~1-2 s.
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_inputs():
    df, gt = make_synthetic_transcripts(
        n_cells=6,
        voxels_per_cell_mean=120,
        tx_per_cell=40,
        n_genes=12,
        n_types=3,
        domain_z_um=14.0,
        nuclear_layers=3,
        seed=7,
    )
    # The SEG nuclear-seed prune keys off ``overlaps_nucleus``; the
    # synthetic generator names the flag ``is_nuclear``. (Same rename
    # trick as test_nuclear_only_admit_wiring.py.)
    df = df.rename(columns={"is_nuclear": "overlaps_nucleus"})
    assert "overlaps_nucleus" in df.columns
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    return df, panel


def _capture_prune_snapshot(df, panel, cfg):
    """Run ``run_segmented_pipeline`` and return the Prune-stage snapshot
    DataFrame (``tracer_id`` + ``_etype`` + the pristine ``cell_id`` /
    ``overlaps_nucleus`` columns).

    Wraps ``pipeline._record_stage`` for the duration of the run;
    restores it in ``finally`` so the module global is never left
    patched.
    """
    captured: dict = {}
    original = _pipeline._record_stage

    def _spy(progression, stage_name, stage_df, col):
        if stage_name == "Prune" and "prune" not in captured:
            captured["prune"] = stage_df.copy()
        return original(progression, stage_name, stage_df, col)

    _pipeline._record_stage = _spy
    try:
        _pipeline.run_segmented_pipeline(df.copy(), panel, cfg=cfg)
    finally:
        _pipeline._record_stage = original

    assert "prune" in captured, "Prune stage was never recorded"
    return captured["prune"]


def _n_cytoplasm_admitted_to_own_cell(prune_df) -> int:
    """Count cytoplasmic transcripts (``overlaps_nucleus==False``) that
    the Prune stage admitted to their OWN cell as a main entity
    (``_etype=="cell"`` and ``tracer_id == cell_id``)."""
    cyto = prune_df[~prune_df["overlaps_nucleus"].astype(bool)]
    is_cell = cyto["_etype"].astype(str) == "cell"
    own = cyto["tracer_id"].astype(str) == cyto["cell_id"].astype(str)
    return int((is_cell & own).sum())


@pytest.fixture(scope="module")
def prune_default(synthetic_inputs):
    """Prune snapshot under the DEFAULT config (load_config →
    nuclear_only_admit=False)."""
    df, panel = synthetic_inputs
    cfg = load_config()
    # Guard: this test only locks the CURRENT default. If the default is
    # reverted to True the assertions below fail (which is the point),
    # but this makes the assumption explicit.
    assert cfg.phase1.resolve_scope()[1] is False  # admit half: whole-cell
    return _capture_prune_snapshot(df, panel, cfg)


@pytest.fixture(scope="module")
def prune_nuclear_only(synthetic_inputs):
    """Prune snapshot under the OLD default nuclear_only_admit=True
    (forced on the frozen dataclass) on the SAME input."""
    df, panel = synthetic_inputs
    cfg = load_config()
    object.__setattr__(cfg.phase1, "prune_scope", "nuclear")
    return _capture_prune_snapshot(df, panel, cfg)


# ----------------------------------------------------------------------
# Behavior 1 — nuclear_only_admit=False admits cytoplasm at Prune.
# ----------------------------------------------------------------------
def test_default_admits_cytoplasm_at_prune(prune_default):
    """Under the default (nuclear_only_admit=False) many cytoplasmic tx
    are admitted to their own cell already at the Prune stage."""
    n_default = _n_cytoplasm_admitted_to_own_cell(prune_default)
    # This fixture yields 162; assert a healthy floor so a partial
    # regression (or a default flip) is caught, without hard-coding the
    # exact count.
    assert n_default >= 50, (
        "default (nuclear_only_admit=False) should admit cytoplasm at "
        f"Prune, but only {n_default} cytoplasmic tx were admitted to "
        "their own cell — has phase1.nuclear_only_admit been reverted "
        "to True?"
    )


def test_nuclear_only_defers_all_cytoplasm_at_prune(prune_nuclear_only):
    """Under nuclear_only_admit=True, NO cytoplasmic tx is admitted to a
    cell at Prune — identity is established from nuclear tx only."""
    n_nuc = _n_cytoplasm_admitted_to_own_cell(prune_nuclear_only)
    assert n_nuc == 0, (
        "nuclear_only_admit=True must defer all cytoplasm at Prune, but "
        f"{n_nuc} cytoplasmic tx were admitted as a cell main."
    )


def test_cytoplasm_admission_differs_between_configs(
    prune_default, prune_nuclear_only
):
    """The two configs must DIFFER on the cytoplasmic population — the
    core sensitivity that guards the nuclear_only_admit default. If the
    default were reverted to True, prune_default would behave like
    prune_nuclear_only and this strict inequality would fail."""
    n_default = _n_cytoplasm_admitted_to_own_cell(prune_default)
    n_nuc = _n_cytoplasm_admitted_to_own_cell(prune_nuclear_only)
    assert n_default > n_nuc, (
        f"expected default to admit more cytoplasm than nuclear-only, "
        f"got default={n_default} nuclear_only={n_nuc}"
    )


def test_individual_cytoplasmic_tx_flips_when_reverted(
    synthetic_inputs, prune_default, prune_nuclear_only
):
    """Targeted per-transcript lock: pick a specific cytoplasmic tx that
    the default admits to its cell, and assert the SAME tx is left
    unassigned once nuclear_only_admit is forced back to True."""
    df, _ = synthetic_inputs

    dflt = prune_default.set_index("transcript_id")
    nuc = prune_nuclear_only.set_index("transcript_id")

    # Candidate cytoplasmic tx admitted to own cell under the default.
    cyto = prune_default[~prune_default["overlaps_nucleus"].astype(bool)]
    admitted = cyto[
        (cyto["_etype"].astype(str) == "cell")
        & (cyto["tracer_id"].astype(str) == cyto["cell_id"].astype(str))
    ]
    assert not admitted.empty, "no cytoplasmic tx admitted under default"

    probe_tid = admitted["transcript_id"].iloc[0]
    own_cell = str(dflt.loc[probe_tid, "cell_id"])

    # Default: admitted to its own cell as a `cell` main.
    assert str(dflt.loc[probe_tid, "tracer_id"]) == own_cell
    assert str(dflt.loc[probe_tid, "_etype"]) == "cell"

    # nuclear_only_admit=True: NOT admitted to that cell — left
    # unassigned (routed differently) by the Prune stage.
    reverted_id = str(nuc.loc[probe_tid, "tracer_id"])
    assert reverted_id != own_cell
    assert reverted_id in _UNASSIGNED, (
        f"probe cytoplasmic tx {probe_tid} (gene "
        f"{dflt.loc[probe_tid, 'feature_name']!r}, cell {own_cell}) was "
        f"expected unassigned under nuclear_only_admit=True, got "
        f"tracer_id={reverted_id!r}"
    )


def test_default_config_flags(synthetic_inputs):
    """Documents (and guards) the two defaults this file assumes. A
    revert of either surfaces here immediately."""
    cfg = load_config()
    assert cfg.phase1.resolve_scope()[1] is False  # admit half: whole-cell
    assert cfg.phase1.admit_independent is False


# ----------------------------------------------------------------------
# Behavior 2 — admit_independent=False collapsed-seed doublet split.
# SKIPPED: not deterministically constructible in a minimal synthetic
# fixture. `admit_independent` is only consulted on the "empty
# real-signal" (orthogonal) admission branch in
# `_cy_prune._admission_test` (returns `_ADMIT_INDEPENDENT` iff the
# candidate gene has NO seed pair with |PMI| > real_signal_threshold).
# Producing that split requires the greedy 1a seed to collapse to an
# orthogonal anchor while a >=3-gene second program is ORTHOGONAL to the
# survivors — but the anti-correlation that collapses a program always
# leaves the surviving program IN the seed, so the rejected program is
# anti-correlated (real-signal veto, fires under BOTH configs) rather
# than orthogonal. Empirically the synthetic panel (all edges +/-1, no
# orthogonal/absent pairs) yields a bit-identical Prune partition under
# admit_independent True vs False across seeds. Locking this behavior
# needs a real high-plex panel fixture with partial-orthogonality; out
# of scope for a fast synthetic regression test. Only the deterministic
# nuclear_only_admit behavior is locked above.
# ----------------------------------------------------------------------
@pytest.mark.skip(
    reason="admit_independent doublet split is not deterministically "
    "reproducible in a minimal synthetic fixture (requires orthogonal "
    "real-signal the +/-1 synthetic panel cannot express); only "
    "nuclear_only_admit is locked. See module docstring."
)
def test_admit_independent_false_splits_collapsed_seed_doublet():
    raise NotImplementedError
