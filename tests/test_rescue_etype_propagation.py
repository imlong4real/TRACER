"""propagate_etype_to_moved: after rescue relabels unassigned tx into a target
entity, the target must stay HOMOGENEOUS. The target's etype is the dominant
non-unknown etype among its PRE-EXISTING members (rows not just moved), so a
stray/stale etype on a moved row can neither define the target nor survive."""
import numpy as np
import pandas as pd
from tracer._etype import propagate_etype_to_moved


def test_moved_rows_adopt_target_and_stale_contaminant_overwritten():
    # Target 'P' = 3 pre-existing partial tx (rows 0,1,2). Rows 3,4 were just
    # moved into P: row 3 carries a STALE 'cell' etype, row 4 'unknown'.
    df = pd.DataFrame({
        "tracer_id": ["P", "P", "P", "P", "P"],
        "_etype":    ["partial", "partial", "partial", "cell", "unknown"],
    })
    propagate_etype_to_moved(
        df, np.array([3, 4]), np.array(["P", "P"]), entity_col="tracer_id",
    )
    # Whole entity homogeneous partial; the stale 'cell' is gone.
    assert (df["_etype"] == "partial").all()


def test_contaminant_on_moved_row_does_not_define_target():
    # Even if the moved contaminant row appears FIRST in df order, the target's
    # type is decided by pre-existing members, not the moved row.
    df = pd.DataFrame({
        "tracer_id": ["P", "P", "P", "P"],
        "_etype":    ["cell", "partial", "partial", "partial"],  # row 0 = moved stale cell
    })
    propagate_etype_to_moved(
        df, np.array([0]), np.array(["P"]), entity_col="tracer_id",
    )
    assert (df["_etype"] == "partial").all()


def test_moved_into_genuine_cell_becomes_cell():
    df = pd.DataFrame({
        "tracer_id": ["C", "C", "C", "C"],
        "_etype":    ["cell", "cell", "unknown", "unknown"],  # rows 2,3 moved in
    })
    propagate_etype_to_moved(
        df, np.array([2, 3]), np.array(["C", "C"]), entity_col="tracer_id",
    )
    assert (df["_etype"] == "cell").all()


def test_no_preexisting_nonunknown_member_leaves_moved_unchanged():
    # A target whose only non-moved members are 'unknown' can't be typed;
    # moved rows are left as-is (no fabricated etype).
    df = pd.DataFrame({
        "tracer_id": ["U", "U", "U"],
        "_etype":    ["unknown", "partial", "unknown"],  # row 1 moved in
    })
    propagate_etype_to_moved(
        df, np.array([1]), np.array(["U"]), entity_col="tracer_id",
    )
    # row 1 keeps 'partial' (no pre-existing type to overwrite it with)
    assert df.loc[1, "_etype"] == "partial"
