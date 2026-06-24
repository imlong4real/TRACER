"""Tests for `tracer.config`. Phase A — scaffolding only.

These tests verify:
  1. Default-instantiated `PipelineConfig` is valid.
  2. `defaults.toml` parses and yields the same values as the dataclass
     defaults — code and TOML stay in lock-step.
  3. Layered overrides patch correctly.
  4. `[final_rescue] inherit = "rescue"` works.
  5. Validation errors fire for bad values / unknown keys.
  6. Receipt round-trip (`dump_receipt` → JSON → load → equal).
"""
from __future__ import annotations

import json
import textwrap
from dataclasses import asdict
from pathlib import Path

import pytest

from tracer.config import (
    GroupConfig,
    Phase1Config,
    Phase1QcConfig,
    Phase1RerankConfig,
    PipelineConfig,
    RescueConfig,
    SplitPhase1Config,
    dump_receipt,
    load_config,
    to_dict,
)


# --------------------------------------------------------------------------
# 1. Default instantiation works and respects __post_init__ invariants.
# --------------------------------------------------------------------------


def test_default_pipelineconfig_instantiates():
    cfg = PipelineConfig()
    assert cfg.rescue.veto_mode == "hybrid"
    assert cfg.rescue.mean_admit_threshold == 0.5
    assert cfg.phase1.tx_weighted_prune is True
    # final_rescue defaults derive from rescue
    assert cfg.final_rescue.veto_mode == "hybrid"
    assert cfg.final_rescue.small_entity_guard_n == 0


def test_dataclasses_are_frozen():
    cfg = PipelineConfig()
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.rescue.veto_mode = "min"  # type: ignore[misc]


# --------------------------------------------------------------------------
# 2. defaults.toml matches the dataclass defaults.
# --------------------------------------------------------------------------


def test_defaults_toml_matches_dataclass_defaults():
    """Loaded `defaults.toml` must equal a default-instantiated config —
    keeps the human-readable TOML in sync with the canonical code defaults.
    """
    cfg_from_code = PipelineConfig()
    cfg_from_toml = load_config()  # no path, no platform → defaults.toml
    assert to_dict(cfg_from_toml) == to_dict(cfg_from_code), (
        "defaults.toml has drifted from PipelineConfig() defaults. "
        "Update one or the other so they agree."
    )


# --------------------------------------------------------------------------
# 3. Layered override.
# --------------------------------------------------------------------------


def test_user_override_patches_keys(tmp_path: Path):
    user = tmp_path / "user.toml"
    user.write_text(textwrap.dedent("""\
        [rescue]
        veto_mode = "min"
        max_passes = 5

        [phase1]
        tx_weighted_prune = false
    """))
    cfg = load_config(path=user)
    # patched keys
    assert cfg.rescue.veto_mode == "min"
    assert cfg.rescue.max_passes == 5
    assert cfg.phase1.tx_weighted_prune is False
    # unpatched keys still at defaults
    assert cfg.rescue.mean_admit_threshold == 0.5
    assert cfg.phase1.pmi_threshold == 0.2


def test_user_override_runs_validation(tmp_path: Path):
    user = tmp_path / "bad.toml"
    user.write_text("[rescue]\nveto_mode = \"bogus\"\n")
    with pytest.raises(ValueError, match="veto_mode"):
        load_config(path=user)


def test_unknown_key_rejected(tmp_path: Path):
    user = tmp_path / "typo.toml"
    user.write_text("[rescue]\nmin_admit_thresold = 0.5\n")  # missing 'h' in 'threshold'
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path=user)


# --------------------------------------------------------------------------
# 4. Inherit semantics.
# --------------------------------------------------------------------------


def test_final_rescue_inherits_from_rescue(tmp_path: Path):
    user = tmp_path / "inherit.toml"
    user.write_text(textwrap.dedent("""\
        [rescue]
        max_passes = 7
        veto_mode = "mean"

        [final_rescue]
        inherit = "rescue"
        max_passes = 1
    """))
    cfg = load_config(path=user)
    # rescue: as set
    assert cfg.rescue.max_passes == 7
    assert cfg.rescue.veto_mode == "mean"
    # final_rescue: inherits veto_mode but overrides max_passes
    assert cfg.final_rescue.veto_mode == "mean"
    assert cfg.final_rescue.max_passes == 1


def test_inherit_self_rejected(tmp_path: Path):
    user = tmp_path / "self.toml"
    user.write_text("[rescue]\ninherit = \"rescue\"\n")
    with pytest.raises(ValueError, match="inherits from itself"):
        load_config(path=user)


def test_inherit_missing_target_rejected(tmp_path: Path):
    user = tmp_path / "missing.toml"
    user.write_text("[final_rescue]\ninherit = \"nonexistent\"\n")
    with pytest.raises(ValueError, match="missing section"):
        load_config(path=user)


# --------------------------------------------------------------------------
# 5. Direct dataclass validation.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs,match", [
    ({"veto_mode": "wrong"}, "veto_mode"),
    ({"max_passes": 0}, "max_passes"),
    ({"bin_size_um": -1.0}, "bin_size_um"),
])
def test_rescue_invariants(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RescueConfig(**kwargs)


def test_phase1_invariants():
    with pytest.raises(ValueError, match="pmi_threshold"):
        Phase1Config(pmi_threshold=2.0)
    with pytest.raises(ValueError, match="seed_coherence_floor"):
        Phase1Config(seed_coherence_floor=-0.1)


@pytest.mark.parametrize("kwargs, match", [
    ({"margin_tx": 0}, "margin_tx"),
    ({"margin_tx": -1}, "margin_tx"),
])
def test_phase1_rerank_invariants(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Phase1RerankConfig(**kwargs)


def test_phase1_rerank_defaults():
    cfg = Phase1RerankConfig()
    assert cfg.enabled is True  # promoted to default-on 2026-05-13
    assert cfg.margin_tx == 1


@pytest.mark.parametrize("kwargs, match", [
    ({"seg_cascade_target_cov": 0.0}, "seg_cascade_target_cov"),
    ({"seg_cascade_target_cov": 1.5}, "seg_cascade_target_cov"),
    ({"noseg_cascade_target_cov": -0.1}, "noseg_cascade_target_cov"),
    ({"seg_cascade_hard_min": 1}, "seg_cascade_hard_min"),
    ({"noseg_cascade_hard_min": 0}, "noseg_cascade_hard_min"),
    ({"cascade_bin_size_um": -1.0}, "cascade_bin_size_um"),
    ({"cascade_territory_radius_bins": 0}, "cascade_territory_radius_bins"),
    ({"cascade_pmi_threshold": 2.0}, "cascade_pmi_threshold"),
    ({"cascade_min_anchor_tx": 0}, "cascade_min_anchor_tx"),
    ({"legacy_bin_size_um": 0.0}, "legacy_bin_size_um"),
    ({"legacy_neighborhood": "diagonal"}, "legacy_neighborhood"),
    ({"legacy_min_comp_size": 0}, "legacy_min_comp_size"),
])
def test_group_invariants(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GroupConfig(**kwargs)


# --------------------------------------------------------------------------
# 6. Receipt round-trip.
# --------------------------------------------------------------------------


def test_receipt_roundtrip(tmp_path: Path):
    cfg = PipelineConfig(
        rescue=RescueConfig(veto_mode="mean", max_passes=2),
        phase1=Phase1Config(pmi_threshold=0.10, tx_weighted_prune=False),
    )
    receipt = tmp_path / "receipt.json"
    dump_receipt(cfg, receipt)

    loaded = json.loads(receipt.read_text())
    assert loaded["rescue"]["veto_mode"] == "mean"
    assert loaded["rescue"]["max_passes"] == 2
    assert loaded["phase1"]["pmi_threshold"] == 0.10
    assert loaded["phase1"]["tx_weighted_prune"] is False
    # unchanged sections still present
    assert loaded["phase1_qc"]["min_tx"] == 3

    # And the loaded dict matches to_dict(cfg) by value (to_dict
    # normalizes tuples → lists so JSON round-trip is lossless).
    from tracer.config import to_dict
    assert loaded == to_dict(cfg)


# --------------------------------------------------------------------------
# 7. Unknown platform raises.
# --------------------------------------------------------------------------


def test_unknown_platform_raises():
    with pytest.raises(FileNotFoundError, match="Unknown platform"):
        load_config(platform="atlantis")
