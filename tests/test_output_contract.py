import numpy as np
import pandas as pd

from tracer.pipeline import _canonicalize_output


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
