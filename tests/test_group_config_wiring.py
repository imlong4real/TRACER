"""Group, Split-Phase1 and Phase1-QC settings come from config (unified-prune
step 6).

Before step 6 the runners read module constants for these stages, so a
config value for ``split_phase1``, ``phase1_qc`` or (on the SEG path)
``group`` loaded and validated but did nothing. These tests pin that:

* ``cfg=None`` resolves to the same values as the bundled defaults, and
  both give the same labels at every stage;
* Cascade-Split and Cascade-QC switch on and off separately (0 = off, the
  TOML spelling);
* every setting reaches the stage that uses it;
* the retired spatial-CC backend is never reached and its exports warn.
"""
from __future__ import annotations

import dataclasses

import pytest

import tracer.pipeline as P
from tracer.config import load_config
from tests.test_stage_label_regression import _run, inputs  # noqa: F401

SECTIONS = ("group", "split_phase1", "phase1_qc")
RUNNERS = {"seg": P.run_segmented_pipeline, "noseg": P.run_noseg_pipeline}


def _replace(cfg, **sections):
    return dataclasses.replace(cfg, **{
        name: dataclasses.replace(getattr(cfg, name), **kw)
        for name, kw in sections.items()
    })


def test_cfg_none_resolves_to_bundled_defaults():
    resolved, loaded = P._resolve_pipeline_cfg(None), load_config()
    for name in SECTIONS:
        assert getattr(resolved, name) == getattr(loaded, name), name


@pytest.mark.parametrize("mode", ["seg", "noseg"])
def test_cfg_none_and_bundled_defaults_give_same_labels(monkeypatch, inputs, mode):
    df, panel = inputs
    a, _ = _run(monkeypatch, RUNNERS[mode], df, panel)
    monkeypatch.undo()  # else the second capture wraps the first
    b, _ = _run(monkeypatch, RUNNERS[mode], df, panel, load_config())
    assert a == b


@pytest.mark.parametrize("mode", ["seg", "noseg"])
@pytest.mark.parametrize("split_dz,qc_min", [(2.0, 5), (0, 5), (2.0, 0), (0, 0)])
def test_cascade_post_steps_switch_separately(monkeypatch, inputs, mode,
                                              split_dz, qc_min):
    df, panel = inputs
    cfg = _replace(load_config(), group=dict(cascade_split_dz_um=split_dz,
                                             cascade_qc_min_tx=qc_min))
    stages, _ = _run(monkeypatch, RUNNERS[mode], df, panel, cfg)
    names = [s["stage"] for s in stages]
    assert ("Cascade-Split" in names) == bool(split_dz)
    assert ("Cascade-QC" in names) == bool(qc_min)


def test_cascade_post_steps_accept_none():
    cfg = _replace(load_config(), group=dict(cascade_split_dz_um=None,
                                             cascade_qc_min_tx=None))
    assert cfg.group.cascade_split_dz_um is None


@pytest.mark.parametrize("field,value", [("cascade_split_dz_um", -1.0),
                                         ("cascade_qc_min_tx", -1)])
def test_cascade_post_steps_validated(field, value):
    with pytest.raises(ValueError, match=field):
        _replace(load_config(), group={field: value})


def test_toml_can_switch_cascade_qc_off(tmp_path):
    p = tmp_path / "user.toml"
    p.write_text("[group]\ncascade_qc_min_tx = 0\n")
    cfg = load_config(path=p)
    assert cfg.group.cascade_qc_min_tx == 0
    assert cfg.group.cascade_split_dz_um == 2.0


@pytest.mark.parametrize("mode", ["seg", "noseg"])
def test_settings_reach_their_stages(monkeypatch, inputs, mode):
    """Spy on the split/QC/cascade calls and check each gets its cfg value."""
    df, panel = inputs
    calls = {"split": [], "qc": [], "cascade": []}
    orig_split = P._spatial_split_phase1_entities
    orig_qc = P._qc_demote_small_phase1_entities
    orig_cascade = P.cascade_as_residual_handler

    def split(*a, **kw):
        calls["split"].append(kw)
        return orig_split(*a, **kw)

    def qc(*a, **kw):
        calls["qc"].append(kw)
        return orig_qc(*a, **kw)

    def cascade(*a, **kw):
        calls["cascade"].append(kw)
        return orig_cascade(*a, **kw)

    monkeypatch.setattr(P, "_spatial_split_phase1_entities", split)
    monkeypatch.setattr(P, "_qc_demote_small_phase1_entities", qc)
    monkeypatch.setattr(P, "cascade_as_residual_handler", cascade)

    cfg = _replace(
        load_config(),
        split_phase1=dict(dz_threshold_um=3.5, min_tx=2, min_entity_size=3),
        phase1_qc=dict(min_tx=4),
        group=dict(cascade_split_dz_um=1.5, cascade_qc_min_tx=7,
                   cascade_min_anchor_tx=4, cascade_pmi_threshold=0.25,
                   seg_cascade_target_cov=0.6, noseg_cascade_target_cov=0.7),
    )
    _run(monkeypatch, RUNNERS[mode], df, panel, cfg)

    casc_split = [k for k in calls["split"] if k.get("label_prefix") == "cascade_"]
    casc_qc = [k for k in calls["qc"] if k.get("label_prefix") == "cascade_"]
    assert [k["dz_threshold"] for k in casc_split] == [1.5]
    assert (casc_split[0]["min_size"], casc_split[0]["min_entity_size"]) == (2, 3)
    assert [k["min_size"] for k in casc_qc] == [7]

    (c,) = calls["cascade"]
    assert c["min_anchor_tx"] == 4
    assert c["pmi_threshold"] == 0.25
    assert c["auto_target_cov"] == (0.6 if mode == "seg" else 0.7)

    if mode == "seg":
        p1_split = [k for k in calls["split"] if k.get("label_prefix") is None]
        p1_qc = [k for k in calls["qc"] if k.get("label_prefix") is None]
        assert [k["dz_threshold"] for k in p1_split] == [3.5]
        assert (p1_split[0]["min_size"], p1_split[0]["min_entity_size"]) == (2, 3)
        assert [k["min_size"] for k in p1_qc] == [4]


@pytest.mark.parametrize("name", ["annotate_unassigned_components",
                                  "annotate_unassigned_components_fast",
                                  "prune_genes_by_pmi_greedy"])
def test_legacy_group_functions_warn(name):
    """Still exported for one release, but deprecated (step 7)."""
    import numpy as np
    import tracer
    fn = getattr(tracer, name)
    with pytest.warns(DeprecationWarning, match="retired spatial-CC Group"):
        try:
            if name == "prune_genes_by_pmi_greedy":
                fn(np.array([0, 1]), np.zeros((2, 2)))
            else:
                fn(None, None, build_graph_fn=None, prune_fn=None)
        except Exception:
            pass  # the warning fires before the body runs


def test_pipeline_never_calls_legacy_group(monkeypatch, inputs):
    """Neither runner reaches the retired backend; Group is the cascade."""
    import warnings
    df, panel = inputs
    for mode, runner in RUNNERS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            stages, raw = _run(monkeypatch, runner, df, panel)
        monkeypatch.undo()
        assert any(l.startswith("cascade_") for l in raw["Group"]), mode
        assert not any(l.startswith("UNASSIGNED_") for l in raw["Group"]), mode
