"""Wiring tests: ``cfg.split_phase1.*`` and ``cfg.{rescue,final_rescue}.
z_bound_um`` must actually reach the pipeline.

Before the fix, ``run_segmented_pipeline`` called
``_spatial_split_phase1_entities`` with the module constants
``SPLIT_PHASE1_DZ`` / ``SPLIT_PHASE1_MIN_TX`` / ``SPLIT_PHASE1_MIN_ENTITY``,
and no Rescue / Post-Group Rescue / Final Rescue call site passed
``z_bound=`` — so both TOML sections were parsed, validated and then
ignored.

Three layers of evidence:

1. ``_resolve_pipeline_cfg(None)`` maps the (monkeypatchable) module
   constants into ``cfg.split_phase1``, so legacy callers keep working.
2. Spies on the stage helpers capture the forwarded kwargs and assert
   each stage receives its own config value.
3. End-to-end: a non-default value changes the per-tx partition, while
   the explicit default reproduces the default run bit-exactly.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace as dc_replace
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
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    return df, panel


def _labels(fn, df, panel, cfg=None):
    out, _ = fn(df.copy(), panel, cfg=cfg)
    return out["tracer_id"].astype(str).reset_index(drop=True)


def _n_diff(a, b) -> int:
    return int((a != b).sum())


# ---------------------------------------------------------------------------
# 1. cfg=None fallback maps module constants
# ---------------------------------------------------------------------------
def test_resolver_defaults_match_load_config():
    cfg_none = _pipeline._resolve_pipeline_cfg(None)
    cfg_toml = load_config()
    assert cfg_none.split_phase1 == cfg_toml.split_phase1
    assert cfg_none.rescue.z_bound_um is None
    assert cfg_none.final_rescue.z_bound_um is None
    assert cfg_toml.rescue.z_bound_um is None
    assert cfg_toml.final_rescue.z_bound_um is None


def test_resolver_reads_split_phase1_module_constants(monkeypatch):
    monkeypatch.setattr(_pipeline, "SPLIT_PHASE1_DZ", 0.75)
    monkeypatch.setattr(_pipeline, "SPLIT_PHASE1_MIN_TX", 2)
    monkeypatch.setattr(_pipeline, "SPLIT_PHASE1_MIN_ENTITY", 4)
    sp = _pipeline._resolve_pipeline_cfg(None).split_phase1
    assert (sp.dz_threshold_um, sp.min_tx, sp.min_entity_size) == (0.75, 2, 4)


# ---------------------------------------------------------------------------
# 2. Forwarding: each stage receives its own cfg value
# ---------------------------------------------------------------------------
def test_split_phase1_kwargs_forwarded_from_cfg(synthetic_inputs, monkeypatch):
    df, panel = synthetic_inputs
    captured: list[dict] = []
    real = _pipeline._spatial_split_phase1_entities

    def _spy(*args, **kwargs):
        captured.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(_pipeline, "_spatial_split_phase1_entities", _spy)
    base = load_config()
    cfg = dc_replace(base, split_phase1=dc_replace(
        base.split_phase1, dz_threshold_um=3.5, min_tx=2, min_entity_size=6))
    _pipeline.run_segmented_pipeline(df.copy(), panel, cfg=cfg)

    assert len(captured) == 1, "Split-Phase1 spy was not called exactly once"
    kw = captured[0]
    assert kw["dz_threshold"] == 3.5
    assert kw["min_size"] == 2
    assert kw["min_entity_size"] == 6


def _zbound_spies(monkeypatch):
    """Record the ``z_bound`` kwarg of every pipeline-level rescue call."""
    calls: list[tuple[str, object]] = []
    real_guarded = _pipeline.guarded_rescue
    real_pool = _pipeline.reassign_unassigned_grid_pool

    def _guarded(*args, **kwargs):
        calls.append(("guarded_rescue", kwargs.get("z_bound", "MISSING")))
        return real_guarded(*args, **kwargs)

    def _pool(*args, **kwargs):
        calls.append(("final_rescue", kwargs.get("z_bound", "MISSING")))
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(_pipeline, "guarded_rescue", _guarded)
    monkeypatch.setattr(_pipeline, "reassign_unassigned_grid_pool", _pool)
    return calls


def _zbound_cfg(rescue_zb, final_zb):
    base = load_config()
    return dc_replace(
        base,
        rescue=dc_replace(base.rescue, z_bound_um=rescue_zb),
        final_rescue=dc_replace(base.final_rescue, z_bound_um=final_zb),
    )


@pytest.mark.parametrize("runner", ["seg", "noseg"])
def test_z_bound_forwarded_to_every_rescue_stage(synthetic_inputs, monkeypatch,
                                                 runner):
    df, panel = synthetic_inputs
    calls = _zbound_spies(monkeypatch)
    cfg = _zbound_cfg(1.25, 3.75)
    assert cfg.rescue.post_group_passes > 0  # Post-Group Rescue is exercised
    fn = (_pipeline.run_segmented_pipeline if runner == "seg"
          else _pipeline.run_noseg_pipeline)
    fn(df.copy(), panel, cfg=cfg)

    guarded = [zb for name, zb in calls if name == "guarded_rescue"]
    final = [zb for name, zb in calls if name == "final_rescue"]
    assert guarded, "no Rescue / Post-Group Rescue call was observed"
    assert final, "no Final Rescue call was observed"
    assert all(zb == 1.25 for zb in guarded), guarded
    assert all(zb == 3.75 for zb in final), final


# ---------------------------------------------------------------------------
# 3. End-to-end behavior
# ---------------------------------------------------------------------------
def test_split_phase1_dz_threshold_changes_partition(synthetic_inputs):
    df, panel = synthetic_inputs
    base = load_config()
    assert base.split_phase1.dz_threshold_um == 2.0
    cfg = dc_replace(base, split_phase1=dc_replace(
        base.split_phase1, dz_threshold_um=0.5))
    a = _labels(_pipeline.run_segmented_pipeline, df, panel, cfg=base)
    b = _labels(_pipeline.run_segmented_pipeline, df, panel, cfg=cfg)
    assert _n_diff(a, b) >= 1, (
        "split_phase1.dz_threshold_um=0.5 produced labels identical to the "
        "2.0 default — the key is not reaching Split-Phase1."
    )


def test_split_phase1_module_constant_still_honoured_when_cfg_none(
        synthetic_inputs, monkeypatch):
    """Legacy path: monkeypatching SPLIT_PHASE1_DZ with cfg=None must give
    the same partition as passing the equivalent cfg explicitly."""
    df, panel = synthetic_inputs
    base = load_config()
    cfg = dc_replace(base, split_phase1=dc_replace(
        base.split_phase1, dz_threshold_um=0.5))
    want = _labels(_pipeline.run_segmented_pipeline, df, panel, cfg=cfg)
    monkeypatch.setattr(_pipeline, "SPLIT_PHASE1_DZ", 0.5)
    got = _labels(_pipeline.run_segmented_pipeline, df, panel, cfg=None)
    assert _n_diff(want, got) == 0


@pytest.mark.parametrize("runner", ["seg", "noseg"])
def test_z_bound_changes_partition(synthetic_inputs, runner):
    """A near-zero z-bound on Rescue and Final Rescue must change labels;
    the explicit default (bin_size_um * sqrt(2)) must not."""
    df, panel = synthetic_inputs
    fn = (_pipeline.run_segmented_pipeline if runner == "seg"
          else _pipeline.run_noseg_pipeline)
    base = load_config()
    a = _labels(fn, df, panel, cfg=base)
    tight = _labels(fn, df, panel, cfg=_zbound_cfg(0.01, 0.01))
    assert _n_diff(a, tight) >= 1, (
        f"{runner}: z_bound_um=0.01 produced labels identical to the "
        "default — the key is not reaching the rescue stages."
    )
    G = base.rescue.bin_size_um
    assert base.final_rescue.bin_size_um == G
    explicit = _labels(fn, df, panel,
                       cfg=_zbound_cfg(G * math.sqrt(2.0), G * math.sqrt(2.0)))
    assert _n_diff(a, explicit) == 0
