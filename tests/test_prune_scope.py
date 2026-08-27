"""``cfg.phase1.prune_scope`` — one knob for the whole Prune stage.

Rationale
---------
Seed source (Phase 1a) and admission (Phase 1b/1c) were governed by two
independent booleans, spanning four states of which only two are coherent:

    nuclear seed + nuclear admit   -> legacy, coherent
    cell    seed + cell    admit   -> default, coherent
    nuclear seed + cell    admit   -> SPLIT-BRAIN (the shipped bug: a thin
                                      nuclear seed gating whole-cell
                                      admission, dropping whole cells at
                                      ``seed_coherence_floor``)
    cell    seed + nuclear admit   -> untested, no demand

``prune_scope`` makes the incoherent states unreachable through the
supported API: one enum sets both halves together.

The legacy booleans are REMOVED from the config, not deprecated. Keeping
either as an override would re-open the hole: set only the seed half and
you are back to nuclear-seed + cell-admit. A stale config fails loudly and
the error names its replacement — see ``tests/test_retired_config_keys.py``.

The kernel keeps both as FUNCTION arguments
(``prune_transcripts_nuclear_seed(nuclear_seed_only=, nuclear_only_admit=)``),
so mixed A/B remains possible below the config layer.
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
    pass


@pytest.fixture(scope="module")
def synthetic_inputs():
    df, gt = make_synthetic_transcripts(
        n_cells=6, voxels_per_cell_mean=60, tx_per_cell=20, n_genes=12,
        n_types=3, domain_z_um=10.0, nuclear_layers=2, seed=7,
    )
    if "overlaps_nucleus" not in df.columns and "is_nuclear" in df.columns:
        df = df.rename(columns={"is_nuclear": "overlaps_nucleus"})
    return df, make_synthetic_npmi_panel_for_transcripts(df, gt)


def _forwarded(df, panel, monkeypatch, **phase1) -> dict:
    captured: dict = {}

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        raise _StopCapture

    monkeypatch.setattr(_pipeline, "prune_transcripts_nuclear_seed", _spy)
    cfg = PipelineConfig()
    for k, v in phase1.items():
        object.__setattr__(cfg.phase1, k, v)
    with pytest.raises(_StopCapture):
        _pipeline.run_segmented_pipeline(df, panel, cfg=cfg)
    assert captured, "prune spy was never reached"
    return captured


def test_prune_scope_exists_and_defaults_to_cell():
    cfg = PipelineConfig()
    assert hasattr(cfg.phase1, "prune_scope"), "phase1.prune_scope is missing"
    assert cfg.phase1.prune_scope == "cell", (
        f"prune_scope must default to 'cell'; got {cfg.phase1.prune_scope!r}"
    )


def test_prune_scope_rejects_unknown_value():
    with pytest.raises(ValueError):
        PipelineConfig(); cfg = PipelineConfig()
        object.__setattr__(cfg.phase1, "prune_scope", "nucleus")
        cfg.phase1.__post_init__()


@pytest.mark.parametrize("scope,want_nuclear", [("cell", False), ("nuclear", True)])
def test_prune_scope_sets_BOTH_halves(synthetic_inputs, monkeypatch, scope,
                                      want_nuclear):
    """One enum drives seed AND admission — no split-brain reachable."""
    df, panel = synthetic_inputs
    kw = _forwarded(df, panel, monkeypatch, prune_scope=scope)
    assert kw.get("nuclear_seed_only") == want_nuclear, (
        f"prune_scope={scope!r} must give nuclear_seed_only={want_nuclear}; "
        f"got {kw.get('nuclear_seed_only')!r}")
    assert kw.get("nuclear_only_admit") == want_nuclear, (
        f"prune_scope={scope!r} must give nuclear_only_admit={want_nuclear}; "
        f"got {kw.get('nuclear_only_admit')!r}")


def test_default_config_is_whole_cell_prune(synthetic_inputs, monkeypatch):
    """The shipped default stays bit-compatible: whole-cell seed + admit."""
    df, panel = synthetic_inputs
    kw = _forwarded(df, panel, monkeypatch)
    assert kw.get("nuclear_seed_only") is False
    assert kw.get("nuclear_only_admit") is False


def test_deprecated_per_half_keys_are_rejected_by_the_loader():
    """An old config naming the removed booleans must fail LOUDLY.

    Keeping them as deprecated overrides re-opened the split-brain state
    (set only the seed half -> nuclear seed + whole-cell admit), which is
    exactly what prune_scope exists to prevent. They are removed, and the
    TOML loader rejects unknown keys.
    """
    from dataclasses import fields as _fields
    from tracer.config import Phase1Config

    names = {f.name for f in _fields(Phase1Config)}
    assert "nuclear_only_admit" not in names, (
        "nuclear_only_admit must be REMOVED, not deprecated — a lone "
        "override re-creates the split-brain state.")
    assert "nuclear_seed_only" not in names, (
        "nuclear_seed_only must be REMOVED, not deprecated.")
    with pytest.raises(TypeError):
        Phase1Config(nuclear_only_admit=True)


def test_no_config_can_express_a_mixed_scope():
    """The whole point: seed and admit halves can never disagree."""
    from tracer.config import Phase1Config
    for scope in ("nuclear", "cell"):
        seed, admit = Phase1Config(prune_scope=scope).resolve_scope()
        assert seed == admit, (
            f"prune_scope={scope!r} produced a MIXED scope "
            f"(seed={seed}, admit={admit}) — split-brain is reachable again.")
