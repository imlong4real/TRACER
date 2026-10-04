"""Wiring test: ``cfg.phase1.tx_weighted_prune`` must actually control the SEG Prune stage.

Before the fix, ``run_segmented_pipeline`` forwarded the module constant ``TX_WEIGHTED_PRUNE`` (hard-wired True)
to ``prune_transcripts_nuclear_seed`` and ignored ``cfg.phase1.tx_weighted_prune`` entirely, so
``tx_weighted_prune = false`` in a user config silently did nothing. This test intercepts the prune call and
asserts the forwarded ``tx_weighted`` kwarg tracks the config value for both True and False.

RED (pre-fix): the False run still forwards True.  GREEN (post-fix): forwarded value == cfg value.
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
        n_cells=6, voxels_per_cell_mean=60, tx_per_cell=20, n_genes=12, n_types=3,
        domain_z_um=10.0, nuclear_layers=2, seed=7,
    )
    if "overlaps_nucleus" not in df.columns and "is_nuclear" in df.columns:
        df = df.rename(columns={"is_nuclear": "overlaps_nucleus"})
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    return df, panel


@pytest.mark.parametrize("want", [True, False])
def test_tx_weighted_prune_is_forwarded_from_cfg(synthetic_inputs, monkeypatch, want):
    df, panel = synthetic_inputs
    captured: dict[str, bool] = {}

    def _spy(*args, **kwargs):
        captured["tx_weighted"] = kwargs.get("tx_weighted")
        raise _StopCapture

    monkeypatch.setattr(_pipeline, "prune_transcripts_nuclear_seed", _spy)
    cfg = PipelineConfig()
    object.__setattr__(cfg.phase1, "tx_weighted_prune", want)   # configs are frozen
    with pytest.raises(_StopCapture):
        _pipeline.run_segmented_pipeline(df, panel, cfg=cfg)
    assert "tx_weighted" in captured, "prune spy was never reached"
    assert captured["tx_weighted"] == want, (
        f"cfg.phase1.tx_weighted_prune={want} but Prune received tx_weighted={captured['tx_weighted']!r} "
        "— the field is not wired through.")
