"""finalize_unassigned must enforce the published invariant
    sentinel entity label  ==>  _etype == 'unknown'
so no dropped tx is left carrying a stale 'cell'/'partial' etype."""
import pandas as pd
from tracer.spatial import finalize_unassigned


def test_finalize_resets_etype_for_collapsed_rows():
    df = pd.DataFrame({
        "stitched": ["C", "-1", "group_rejected", "C"],
        "cell_id":  ["5", "5", "5", "5"],
        "_etype":   ["cell", "cell", "partial", "cell"],  # rows 1,2 stale
    })
    finalize_unassigned(df, col="stitched")
    dropped = df["stitched"] == "UNASSIGNED"
    assert dropped.sum() == 2
    assert (df.loc[dropped, "_etype"] == "unknown").all()      # reset
    assert (df.loc[~dropped, "_etype"] == "cell").all()        # real entities kept


def test_finalize_leaves_real_entity_etypes_untouched():
    df = pd.DataFrame({
        "stitched": ["A", "A-tr-1", "B"],
        "cell_id":  ["A", "A", "B"],
        "_etype":   ["cell", "partial", "cell"],
    })
    finalize_unassigned(df, col="stitched")
    assert (df["stitched"] != "UNASSIGNED").all()              # nothing dropped
    assert df["_etype"].tolist() == ["cell", "partial", "cell"]
