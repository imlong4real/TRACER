"""Regression tests for the no-segmentation CLI entry point.

``tests/test_pipeline_smoke.py`` exercises the *library* function
``tracer.pipeline.run_noseg_pipeline``. It does not touch
``tracer.noseg_pipeline._run_one``, the wrapper the ``python -m
tracer.noseg_pipeline`` CLI runs in every tile worker. That gap let the
pipeline's move to a single ``tracer_id`` output column break the CLI while
the whole suite stayed green: ``_run_one`` still read ``df_final["stitched"]``,
which ``_canonicalize_output`` had dropped, so every tile died with
``KeyError: 'stitched'``.

These tests pin the seam between the two.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tracer.noseg_pipeline as noseg
from tracer.config import load_config
from tracer.pipeline import _canonicalize_output
from tests.synthetic import (
    make_synthetic_transcripts,
    make_synthetic_npmi_panel_for_transcripts,
)


CELLS_KW = dict(
    n_cells=6,
    voxels_per_cell_mean=60,
    tx_per_cell=20,
    n_genes=12,
    n_types=3,
    domain_z_um=10.0,
    nuclear_layers=2,
)


def _noseg_inputs():
    """Synthetic transcripts shaped like the CLI's exploded bin table."""
    df, gt = make_synthetic_transcripts(**CELLS_KW, seed=7)
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    # The CLI explodes bins, so every transcript carries a bin_id and the
    # segmentation prior is discarded before the pipeline sees it.
    df = df.copy()
    df["bin_id"] = (df["x"].round().astype(int).astype(str) + "_"
                    + df["y"].round().astype(int).astype(str))
    df["cell_id"] = "-1"
    return df, panel


def test_pipeline_output_is_canonicalized_to_tracer_id():
    """The contract _run_one has to consume: tracer_id present, stitched gone."""
    df = pd.DataFrame({
        "transcript_id": [1, 2, 3],
        "cell_id": ["-1", "-1", "-1"],
        "stitched": ["a", "b", "UNASSIGNED"],
        "_etype": ["cell", "cell", "unknown"],
    })
    out = _canonicalize_output(df, pd.Series(["-1", "-1", "-1"], index=[1, 2, 3]))
    assert "tracer_id" in out.columns
    assert "stitched" not in out.columns


def test_run_one_accepts_canonicalized_frame(monkeypatch):
    """_run_one must read the canonical label column, not the dropped one.

    Stubs the pipeline so the test pins the column contract alone: the frame
    returned here is exactly the shape `_canonicalize_output` produces.
    """
    src = pd.DataFrame({
        "transcript_id": [1, 2, 3, 4],
        "bin_id": ["b1", "b1", "b2", "b2"],
        "x": [0.0, 0.0, 1.0, 1.0],
        "y": [0.0, 0.0, 1.0, 1.0],
        "feature_name": ["GA", "GB", "GA", "GB"],
    })
    canonical = src.assign(tracer_id=["e1", "e1", "e2", "-1"])

    monkeypatch.setattr(noseg, "run_noseg_pipeline",
                        lambda df, panel, cfg=None: (canonical, []))
    out = noseg._run_one(src, panel=None, cfg=None, tile_tag=None)

    assert "stitched" in out.columns, "downstream consumers expect `stitched`"
    # the "-1" transcript is unassigned and must be dropped
    assert len(out) == 3
    assert set(out["stitched"]) == {"e1", "e2"}


def test_run_one_still_accepts_legacy_stitched_frame(monkeypatch):
    """A frame that carries only the legacy column must keep working."""
    src = pd.DataFrame({
        "transcript_id": [1, 2],
        "bin_id": ["b1", "b2"],
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "feature_name": ["GA", "GB"],
    })
    legacy = src.assign(stitched=["e1", "UNASSIGNED"])

    monkeypatch.setattr(noseg, "run_noseg_pipeline",
                        lambda df, panel, cfg=None: (legacy, []))
    out = noseg._run_one(src, panel=None, cfg=None, tile_tag=None)
    assert list(out["stitched"]) == ["e1"]


def test_run_one_applies_tile_tag(monkeypatch):
    src = pd.DataFrame({
        "transcript_id": [1],
        "bin_id": ["b1"],
        "x": [0.0], "y": [0.0],
        "feature_name": ["GA"],
    })
    monkeypatch.setattr(noseg, "run_noseg_pipeline",
                        lambda df, panel, cfg=None: (src.assign(tracer_id=["e1"]), []))
    out = noseg._run_one(src, panel=None, cfg=None, tile_tag="t3")
    assert list(out["stitched"]) == ["t3::e1"]


def test_run_one_end_to_end_on_synthetic_transcripts():
    """Full path: real run_noseg_pipeline through _run_one, as the CLI runs it.

    This is the test that would have caught the KeyError.
    """
    df, panel = _noseg_inputs()
    cfg = load_config(platform="noseg")

    out = noseg._run_one(df, panel, cfg, tile_tag="tile0")

    assert not out.empty, "no-seg reconstruction produced zero real profiles"
    assert list(out.columns) == ["bin_id", "x", "y", "feature_name", "stitched"]
    assert out["stitched"].str.startswith("tile0::").all()
    # every emitted label is a real entity, never an unassigned sentinel
    bare = out["stitched"].str.replace("tile0::", "", regex=False)
    assert not bare.isin(list(noseg._UNASSIGNED_LABELS)).any()


def test_aggregate_profiles_consumes_run_one_output():
    """_run_one's output must satisfy aggregate_profiles' label_col contract."""
    df, panel = _noseg_inputs()
    cfg = load_config(platform="noseg")

    out = noseg._run_one(df, panel, cfg, tile_tag=None)
    res = noseg.aggregate_profiles(out, panel, label_col="stitched")

    assert res.adata.n_obs > 0
    assert len(res.scores) == res.adata.n_obs
    assert res.adata.n_obs == out["stitched"].nunique()
