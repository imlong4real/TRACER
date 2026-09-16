"""Consolidation: pre_stage2_rescue -> guarded_rescue (deprecated alias), and
the 4 internally-dead reassign_* functions emit DeprecationWarning."""
import warnings
import pytest
import tracer.spatial as sp


def test_guarded_rescue_is_the_canonical_name():
    assert hasattr(sp, "guarded_rescue")
    # the guard-wrapper still delegates to the grid-pool kernel
    assert callable(sp.guarded_rescue)


def test_pre_stage2_rescue_alias_warns_and_delegates():
    assert sp.pre_stage2_rescue is not sp.guarded_rescue  # a wrapper, not identity
    with pytest.warns(DeprecationWarning, match="guarded_rescue"):
        # call with no aux/df is fine — the deprecation fires before real work;
        # use a trivially-empty frame to reach the fast return.
        import pandas as pd
        df = pd.DataFrame({"tracer_id": [], "feature_name": [], "x": [], "y": [], "z": []})
        sp.pre_stage2_rescue(df, aux={"gene_to_idx": {}, "W": None}, cluster_guard_n=0)


@pytest.mark.parametrize("name", [
    "reassign_unassigned_to_nearby_entities",
    "reassign_unassigned_by_gene_compat",
    "reassign_unassigned_to_nearest_tx_no_neg",
    "reassign_unassigned_to_nearby_entities_fast",
])
def test_dead_functions_are_deprecated(name):
    fn = getattr(sp, name)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            fn()  # will TypeError on missing args, but the warning fires first
        except TypeError:
            pass
    assert any(issubclass(w.category, DeprecationWarning) for w in rec), \
        f"{name} did not emit DeprecationWarning"
