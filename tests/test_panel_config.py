"""Tests for `PanelConfig` — the depth-corrected (cPMI) panel recipe.

The pipeline resolves a panel's edge weight as `"PMI" if "PMI" in columns`
(pipeline.py:1640, 2037). It cannot be told which estimator that column
holds. So whichever estimator `build_panels.py` writes into `PMI` is the one
TRACER consumes, and getting that default wrong silently degrades every run.

Two recipes, settled by benchmark and recorded in the panel-recipe notes:

  WITH an scRNA reference   `xgt1_cpmi_balanced_rep`
      cell-type balanced with replacement, xgt1 presence, cPMI promoted,
      depth binned on n_genes (xgt1 already conditions on library size).

  WITHOUT one (nuclear)     `nucgrid_pmi`
      label-free grid3 balancing, naive PMI promoted. cPMI is *better per
      cell* here (0.2845 vs 0.3576 size-matched) but the nuclear reference
      is evidence-starved (O median 13 vs 86), so it admits too little.
      Interim until reference depth is solved -- NOT an endorsement of
      naive PMI, which is depth-confounded (67% of pairs positive).

`promote` and `strategy` INVERT between them, so a single default is wrong
for one case or the other, and silently so. These tests pin both.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tracer.config import PanelConfig, load_config


def test_reference_recipe_is_the_default():
    """Bare defaults must be the with-reference recipe."""
    c = PanelConfig()
    assert c.strategy == "rep"
    assert tuple(c.arms) == ("xgt1",)
    assert c.promote == "cPMI"          # the estimator TRACER will consume
    assert c.depth_metric == "n_genes"
    assert c.min_abs_value is None      # truncation is not neutral; opt-in only


def test_defaults_toml_matches_the_dataclass():
    """Code and TOML stay in lock-step, as for every other section."""
    c = load_config().panel
    d = PanelConfig()
    assert (c.strategy, tuple(c.arms), c.promote, c.depth_metric) == \
           (d.strategy, tuple(d.arms), d.promote, d.depth_metric)


def test_nuclear_preset_inverts_promote_and_strategy():
    """The no-reference recipe is grid3 + naive PMI, not rep + cPMI."""
    c = load_config(panel_preset="nuclear").panel
    assert c.strategy == "grid3"        # label-free; on scRNA grid3 LOSES to labels
    assert c.promote == "PMI"           # interim: cPMI is evidence-starved on nuclei


def test_nuclear_preset_does_not_leak_into_the_default():
    """Loading the nuclear preset must not mutate the shared default."""
    load_config(panel_preset="nuclear")
    assert load_config().panel.promote == "cPMI"


def test_unknown_preset_names_the_available_ones():
    with pytest.raises(FileNotFoundError, match="nuclear"):
        load_config(panel_preset="does-not-exist")


def test_promote_rejects_a_non_estimator():
    with pytest.raises(ValueError, match="promote"):
        PanelConfig(promote="NPMI")


def test_total_counts_with_xgt1_warns():
    """xgt1 already conditions on library size -- n_genes is the right covariate."""
    with pytest.warns(UserWarning, match="n_genes"):
        PanelConfig(arms=("xgt1",), depth_metric="total_counts")


def test_min_abs_value_warns_that_truncation_is_not_neutral():
    """Measured on lung: a 12.2% conjunctive cut moved cells +654 and
    whole-cell entropy 0.227 -> 0.252. Absent edges are permissive."""
    with pytest.warns(UserWarning, match="not neutral|permissive"):
        PanelConfig(min_abs_value=0.2)


def test_user_toml_overrides_the_preset():
    p = Path(__import__("tempfile").mkdtemp()) / "u.toml"
    p.write_text(textwrap.dedent("""
        [panel]
        depth_metric = "total_counts"
        arms = ["count2"]
    """))
    c = load_config(path=p, panel_preset="nuclear").panel
    assert c.depth_metric == "total_counts"
    assert tuple(c.arms) == ("count2",)
    assert c.strategy == "grid3"        # preset value survives where not overridden
