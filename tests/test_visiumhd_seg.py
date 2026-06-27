"""Tests for polygon-derived ``overlaps_nucleus`` (VisiumHD seg mode).

Uses tiny synthetic nucleus polygons + bin centers — no scanpy / VisiumHD
IO needed. Covers: point-in-polygon assignment, the unassigned sentinel,
multi-overlap tie-break determinism, ambiguity stats, and the explode hook
that threads per-bin labels into the transcript table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon  # noqa: E402

from tracer.visiumhd_seg import (  # noqa: E402
    NucleusPolygons, assign_bins_to_nuclei,
)


def _square(cx, cy, half, cid):
    return Polygon([(cx - half, cy - half), (cx + half, cy - half),
                    (cx + half, cy + half), (cx - half, cy + half)])


def _nuclei(specs):
    geoms = [_square(cx, cy, h, cid) for (cx, cy, h, cid) in specs]
    cents = np.array([[cx, cy] for (cx, cy, _, _) in specs], dtype=float)
    ids = np.array([str(cid) for (_, _, _, cid) in specs], dtype=object)
    return NucleusPolygons(geoms=geoms, cell_ids=ids, centroids=cents)


# --------------------------------------------------------------------------
# Basic point-in-polygon assignment.
# --------------------------------------------------------------------------
def test_inside_and_outside():
    nuc = _nuclei([(10, 10, 2, "7"), (50, 50, 2, "9")])
    bx = np.array([10.0, 50.0, 30.0])   # in 7, in 9, in nothing
    by = np.array([10.0, 50.0, 30.0])
    cid, ov, stats = assign_bins_to_nuclei(bx, by, nuc)
    assert list(cid) == ["7", "9", "-1"]
    assert list(ov) == [1, 1, 0]
    assert stats["n_assigned"] == 2
    assert stats["n_unassigned"] == 1
    assert stats["frac_assigned"] == pytest.approx(2 / 3)
    assert stats["n_ambiguous"] == 0


def test_empty_inputs():
    nuc = _nuclei([(10, 10, 2, "7")])
    cid, ov, stats = assign_bins_to_nuclei(np.array([]), np.array([]), nuc)
    assert cid.size == 0 and ov.size == 0
    assert stats["n_bins"] == 0

    empty = NucleusPolygons(geoms=[], cell_ids=np.array([], dtype=object),
                            centroids=np.empty((0, 2)))
    cid, ov, stats = assign_bins_to_nuclei(np.array([1.0]), np.array([1.0]), empty)
    assert list(cid) == ["-1"] and list(ov) == [0]


# --------------------------------------------------------------------------
# Multi-overlap determinism (overlapping polygons).
# --------------------------------------------------------------------------
def test_multi_overlap_nearest_centroid():
    # Two overlapping squares; the bin sits closer to nucleus "1" centroid.
    nuc = _nuclei([(0, 0, 10, "1"), (8, 0, 10, "2")])
    bx, by = np.array([1.0]), np.array([0.0])  # inside both; nearer (0,0)
    cid, ov, stats = assign_bins_to_nuclei(bx, by, nuc, multi_rule="nearest_centroid")
    assert cid[0] == "1"
    assert stats["n_ambiguous"] == 1
    assert stats["ambiguity_rate"] == pytest.approx(1.0)


def test_multi_overlap_smallest_id():
    nuc = _nuclei([(0, 0, 10, "5"), (8, 0, 10, "2")])
    bx, by = np.array([4.0]), np.array([0.0])  # inside both
    cid, _, _ = assign_bins_to_nuclei(bx, by, nuc, multi_rule="smallest_id")
    assert cid[0] == "2"


def test_multi_overlap_is_deterministic():
    nuc = _nuclei([(0, 0, 10, "1"), (0, 0, 10, "2")])  # identical → centroid tie
    bx, by = np.array([0.0]), np.array([0.0])
    out = {assign_bins_to_nuclei(bx, by, nuc)[0][0] for _ in range(5)}
    assert out == {"1"}  # tie broken by smallest id, every time


# --------------------------------------------------------------------------
# Explode hook: per-bin labels thread into the transcript table.
# --------------------------------------------------------------------------
def test_explode_threads_seg_labels(monkeypatch):
    from tracer.noseg_pipeline import explode_to_transcripts, BinTable

    # 2 bins x 2 genes; bin A nucleus-seeded, bin B unassigned.
    import anndata as ad
    import scipy.sparse as sp
    X = sp.csr_matrix(np.array([[2, 0], [0, 3]], dtype=np.float32))
    adata = ad.AnnData(X=X)
    adata.obs_names = ["A", "B"]
    adata.var_names = ["GENE1", "GENE2"]
    coords = pd.DataFrame(
        {"x_um": [0.0, 2.0], "y_um": [0.0, 0.0],
         "array_row": [0, 0], "array_col": [0, 1]},
        index=pd.Index(["A", "B"], name="bin_id"))
    bins = BinTable(adata=adata, coords=coords, bin_size_um=2.0, microns_per_pixel=0.27)

    bin_cell_id = pd.Series({"A": "42", "B": "-1"})
    bin_ov = pd.Series({"A": 1, "B": 0}, dtype=np.uint8)
    df = explode_to_transcripts(
        bins, panel_genes={"GENE1", "GENE2"},
        bin_cell_id=bin_cell_id, bin_overlaps_nucleus=bin_ov)

    assert "overlaps_nucleus" in df.columns
    a = df[df.bin_id == "A"]
    b = df[df.bin_id == "B"]
    assert len(a) == 2 and (a.cell_id == "42").all() and (a.overlaps_nucleus == 1).all()
    assert len(b) == 3 and (b.cell_id == "-1").all() and (b.overlaps_nucleus == 0).all()


def test_explode_default_is_noseg():
    # Without the hooks, behavior is unchanged (cell_id all -1, no ov col).
    from tracer.noseg_pipeline import explode_to_transcripts, BinTable
    import anndata as ad
    import scipy.sparse as sp
    adata = ad.AnnData(X=sp.csr_matrix(np.array([[1]], dtype=np.float32)))
    adata.obs_names = ["A"]; adata.var_names = ["GENE1"]
    coords = pd.DataFrame({"x_um": [0.0], "y_um": [0.0],
                           "array_row": [0], "array_col": [0]},
                          index=pd.Index(["A"], name="bin_id"))
    bins = BinTable(adata=adata, coords=coords, bin_size_um=2.0, microns_per_pixel=0.27)
    df = explode_to_transcripts(bins, panel_genes={"GENE1"})
    assert (df.cell_id == "-1").all()
    assert "overlaps_nucleus" not in df.columns
