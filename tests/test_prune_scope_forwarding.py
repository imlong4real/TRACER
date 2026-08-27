"""Wiring test: ``cfg.phase1.nuclear_only_admit`` must actually control
the SEG Prune stage.

Before the pipeline fix, ``run_segmented_pipeline`` forwarded the module
constant ``NUCLEAR_ONLY_ADMIT`` (hard-wired True) to
``prune_transcripts_nuclear_seed`` and ignored ``cfg.phase1.
nuclear_only_admit`` entirely. This test monkeypatches the prune helper
to capture the forwarded ``nuclear_only_admit`` kwarg and asserts it
equals the config value for both True and False.

RED (pre-fix): the False run still forwards True → assertion fails.
GREEN (post-fix): the forwarded value tracks ``cfg.phase1.
nuclear_only_admit`` for both.
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
from tracer.config import PipelineConfig  # noqa: E402

from tests.synthetic import (  # noqa: E402
    make_synthetic_transcripts,
    make_synthetic_npmi_panel_for_transcripts,
)


class _StopCapture(Exception):
    """Sentinel raised after the kwarg is captured to short-circuit the run."""


@pytest.fixture(scope="module")
def synthetic_inputs():
    df, gt = make_synthetic_transcripts(
        n_cells=6,
        voxels_per_cell_mean=60,
        tx_per_cell=20,
        n_genes=12,
        n_types=3,
        domain_z_um=10.0,
        nuclear_layers=2,
        seed=7,
    )
    # The SEG nuclear-seed prune keys off ``overlaps_nucleus``; the
    # synthetic generator names the flag ``is_nuclear``.
    if "overlaps_nucleus" not in df.columns and "is_nuclear" in df.columns:
        df = df.rename(columns={"is_nuclear": "overlaps_nucleus"})
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    return df, panel


def _capture_nuclear_only_admit(df, panel, want: bool, monkeypatch) -> bool:
    """Run the SEG pipeline with ``cfg.phase1.nuclear_only_admit=want``,
    intercept the prune call, and return the forwarded kwarg value."""
    captured: dict[str, bool] = {}

    def _spy(*args, **kwargs):
        captured["nuclear_only_admit"] = kwargs.get("nuclear_only_admit")
        raise _StopCapture

    monkeypatch.setattr(_pipeline, "prune_transcripts_nuclear_seed", _spy)

    cfg = PipelineConfig()
    # Configs are frozen; poke the field the same way the task specifies.
    object.__setattr__(cfg.phase1, "prune_scope",
                       "nuclear" if want else "cell")

    with pytest.raises(_StopCapture):
        _pipeline.run_segmented_pipeline(df, panel, cfg=cfg)

    assert "nuclear_only_admit" in captured, "prune spy was never reached"
    return captured["nuclear_only_admit"]


@pytest.mark.parametrize("want", [True, False])
def test_nuclear_only_admit_is_forwarded_from_cfg(synthetic_inputs, monkeypatch, want):
    df, panel = synthetic_inputs
    got = _capture_nuclear_only_admit(df, panel, want, monkeypatch)
    assert got == want, (
        f"cfg.phase1.nuclear_only_admit={want} but Prune received "
        f"nuclear_only_admit={got!r} — the field is not wired through."
    )


# ---------------------------------------------------------------------------
# Signature guard for the data-gated integration call site
# ---------------------------------------------------------------------------
def test_pdac_integration_kwargs_are_a_valid_signature(synthetic_inputs):
    """The PDAC ROI test's kwargs must remain callable without the ROI data.

    ``TestPDACEMTHybridAdmission`` sits behind ``pytest.importorskip`` on
    ``benchmarks/data_loader.py``, which is absent in CI and in most
    checkouts — so a keyword retired from
    ``prune_transcripts_nuclear_seed`` can rot there indefinitely without
    failing a single run. That is exactly what happened to
    ``seed_coherence_floor``: removed from the signature, left in the call,
    green everywhere, and a hard ``TypeError`` for anyone who did have the
    data on their path.

    Binding the same kwargs against a synthetic frame closes the hole: the
    call is cheap, needs no external data, and fails loudly in CI the moment
    the signature and the integration call site drift apart.
    """
    from tracer.pruning import prune_transcripts_nuclear_seed
    from tests.test_phase1b_hybrid_admission import PDAC_PRUNE_KWARGS

    df, panel = synthetic_inputs
    # The kwargs pin `metric_col="PMI"` (the PDAC panel convention) while the
    # synthetic generator emits NPMI. Alias rather than override the kwarg, so
    # every value is bound exactly as the integration test binds it.
    panel = panel.copy()
    if "PMI" not in panel.columns:
        panel["PMI"] = panel["NPMI"]
    # Runs the real prune — a TypeError here means the kwargs no longer match.
    out, aux = prune_transcripts_nuclear_seed(
        df.copy(), panel, **PDAC_PRUNE_KWARGS)
    assert "tracer_id" in out.columns
    assert len(out) == len(df)
    assert isinstance(aux, dict)
