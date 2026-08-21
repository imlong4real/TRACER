"""set_entity_etype / swap_entity_etypes: all-tx etype changes so an entity is
never left mixed."""
import pandas as pd
from tracer._etype import set_entity_etype, swap_entity_etypes


def _df():
    return pd.DataFrame({
        "tracer_id": ["A", "A", "A", "B", "B", "C"],
        "_etype":    ["cell", "cell", "cell", "partial", "partial", "component"],
    })


def test_set_entity_etype_all_tx():
    df = _df()
    set_entity_etype(df, "A", "partial")
    assert (df.loc[df.tracer_id == "A", "_etype"] == "partial").all()
    assert (df.loc[df.tracer_id == "B", "_etype"] == "partial").all()  # untouched


def test_swap_entity_etypes_all_tx():
    df = _df()
    swap_entity_etypes(df, "A", "B")           # A(cell)<->B(partial)
    assert (df.loc[df.tracer_id == "A", "_etype"] == "partial").all()
    assert (df.loc[df.tracer_id == "B", "_etype"] == "cell").all()
    assert (df.loc[df.tracer_id == "C", "_etype"] == "component").all()  # untouched


def test_swap_resolves_mixed_side():
    # a pre-mixed entity still swaps to a single value (dominant), not mixed
    df = _df()
    df.loc[2, "_etype"] = "partial"            # A now mixed: 2 cell + 1 partial
    swap_entity_etypes(df, "A", "B")
    assert df.loc[df.tracer_id == "A", "_etype"].nunique() == 1   # not mixed
    assert (df.loc[df.tracer_id == "B", "_etype"] == "cell").all()  # A dominant = cell


def test_missing_entity_is_noop():
    df = _df()
    swap_entity_etypes(df, "A", "ZZZ")         # ZZZ absent
    assert (df.loc[df.tracer_id == "A", "_etype"] == "cell").all()
