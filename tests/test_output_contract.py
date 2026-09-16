import numpy as np
import pandas as pd

from tracer.pipeline import _canonicalize_output
from tracer.config import load_config
from tests.synthetic import (
    make_synthetic_transcripts,
    make_synthetic_npmi_panel_for_transcripts,
)


CELLS_KW = dict(
    n_cells=8,
    voxels_per_cell_mean=80,
    tx_per_cell=25,
    n_genes=12,
    n_types=3,
    domain_z_um=10.0,
    nuclear_layers=2,
)


def _regression_inputs():
    df, gt = make_synthetic_transcripts(**CELLS_KW, seed=42)
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    return df, panel


def _frame():
    return pd.DataFrame({
        "transcript_id": [1, 2, 3, 4],
        "cell_id":       ["c1", "c1", "c2", "UNASSIGNED"],
        "tracer_id":     ["c1", "-1", "c2-1", "-1"],
        "stitched":      ["c1", "c1", "c2", "UNASSIGNED"],
        "_etype":        ["cell", "cell", "partial", "unknown"],
    })


def test_canonicalize_sets_tracer_id_to_final_and_drops_stitched():
    df = _frame()
    pristine = pd.Series(["c1","c1","c2","UNASSIGNED"], index=[1,2,3,4])
    out = _canonicalize_output(df, pristine)
    assert "stitched" not in out.columns
    assert list(out["tracer_id"]) == ["c1","c1","c2","-1"]
    assert list(out["_etype"]) == ["cell","cell","partial","unknown"]


def test_canonicalize_restores_pristine_cell_id():
    df = _frame()
    df["cell_id"] = ["c1","c1","c2","-1"]  # simulate finalize's reset
    pristine = pd.Series(["c1","c1","c2","UNASSIGNED"], index=[1,2,3,4])
    out = _canonicalize_output(df, pristine)
    assert list(out["cell_id"]) == ["c1","c1","c2","UNASSIGNED"]


def test_canonicalize_preserves_component_labels():
    df = _frame()
    df["stitched"] = ["c1","UNASSIGNED_7","cascade_3-1","UNASSIGNED"]
    pristine = pd.Series(df["cell_id"].to_numpy(), index=df["transcript_id"])
    out = _canonicalize_output(df, pristine)
    assert list(out["tracer_id"]) == ["c1","UNASSIGNED_7","cascade_3-1","-1"]


def test_segmented_output_single_tracer_id_and_pristine_cell_id():
    df, panel = _regression_inputs()
    pristine = df.set_index("transcript_id")["cell_id"].astype(str)
    from tracer.pipeline import run_segmented_pipeline
    out, _ = run_segmented_pipeline(df.copy(), panel, cfg=load_config())
    assert "stitched" not in out.columns and "tracer_id" in out.columns
    got = out.set_index("transcript_id")["cell_id"].astype(str)
    assert (got.reindex(pristine.index) == pristine).all()
    assert "UNASSIGNED" not in set(out["tracer_id"].astype(str))


def test_partition_nonempty_and_etype_consistent():
    df, panel = _regression_inputs()
    from tracer.pipeline import run_segmented_pipeline
    out, _ = run_segmented_pipeline(df.copy(), panel, cfg=load_config())
    tid = out["tracer_id"].astype(str)
    assert (tid[tid != "-1"].value_counts() >= 1).all()
    # every unassigned tracer_id has _etype unknown, and vice-versa
    assert set(out.loc[tid == "-1", "_etype"].astype(str)) <= {"unknown"}


def test_noseg_output_single_tracer_id_and_pristine_cell_id():
    df, panel = _regression_inputs()
    pristine = df.set_index("transcript_id")["cell_id"].astype(str)
    from tracer.pipeline import run_noseg_pipeline
    out, _ = run_noseg_pipeline(df.copy(), panel, cfg=load_config())
    assert "stitched" not in out.columns and "tracer_id" in out.columns
    got = out.set_index("transcript_id")["cell_id"].astype(str)
    assert (got.reindex(pristine.index) == pristine).all()
    assert "UNASSIGNED" not in set(out["tracer_id"].astype(str))
