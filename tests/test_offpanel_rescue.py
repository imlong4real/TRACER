"""Off-panel (zero-PMI) transcripts — genes absent from the PMI panel (e.g.
housekeeping ACTB) carry no PMI signal, so the PMI-gated rescue excludes them
(`una_g_idx >= 0`) and they are stranded forever. Opt-in `offpanel_first_entity`
assigns each off-panel tx to the nearest assigned entity in its Moore
neighborhood (proximity only, no PMI) — mirroring how a near-zero in-panel gene
already defer-admits. Default OFF preserves bit-exact behavior."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tracer.spatial import reassign_unassigned_grid_pool, pre_stage2_rescue


def _aux():
    g2i = {"g0": 0, "g1": 1}
    W = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float32)
    return {"W": sp.csr_matrix(W), "gene_to_idx": g2i}


def _df():
    # E1: 3 assigned tx (gene g0) near origin; one UNASSIGNED off-panel gX adjacent.
    rows = [
        ("E1", "g0", 0.0, 0.0, 0.0),
        ("E1", "g0", 0.5, 0.0, 0.0),
        ("E1", "g0", 0.0, 0.5, 0.0),
        ("-1", "gX", 1.0, 0.0, 0.0),   # off-panel gene, unassigned, adjacent to E1
    ]
    return pd.DataFrame(rows, columns=["tracer_id", "feature_name", "x", "y", "z"])


def test_offpanel_assigned_to_nearest_when_enabled():
    out, n, stats = reassign_unassigned_grid_pool(
        _df(), _aux(), entity_col="tracer_id", out_col="tracer_id",
        G=5.0, unassigned_labels={"-1"}, offpanel_first_entity=True,
    )
    off = out[out["feature_name"] == "gX"]
    assert (off["tracer_id"] == "E1").all()
    assert stats.get("offpanel_reassigned", 0) == 1


def test_offpanel_stays_unassigned_by_default():
    out, n, stats = reassign_unassigned_grid_pool(
        _df(), _aux(), entity_col="tracer_id", out_col="tracer_id",
        G=5.0, unassigned_labels={"-1"},
    )
    off = out[out["feature_name"] == "gX"]
    assert (off["tracer_id"] == "-1").all()          # unchanged: off-panel excluded
    assert stats.get("offpanel_reassigned", 0) == 0


def test_offpanel_never_rescues_control_probes():
    # Control/blank probes are ALSO panel-absent (g_idx<0) but must NEVER be
    # proximity-rescued — they are instrument noise, not genes. A real off-panel
    # gene (gX) attaches; a NegControl/UnassignedCodeword adjacent to the same
    # entity must stay unassigned even with the flag on.
    rows = [
        ("E1", "g0", 0.0, 0.0, 0.0), ("E1", "g0", 0.5, 0.0, 0.0),
        ("E1", "g0", 0.0, 0.5, 0.0),
        ("-1", "gX", 1.0, 0.0, 0.0),                       # real off-panel gene
        ("-1", "NegControlProbe_00042", 1.0, 0.5, 0.0),    # control probe
        ("-1", "UnassignedCodeword_0231", 0.5, 1.0, 0.0),  # unassigned codeword
    ]
    df = pd.DataFrame(rows, columns=["tracer_id", "feature_name", "x", "y", "z"])
    out, n, stats = reassign_unassigned_grid_pool(
        df, _aux(), entity_col="tracer_id", out_col="tracer_id",
        G=5.0, unassigned_labels={"-1"}, offpanel_first_entity=True,
    )
    assert (out[out["feature_name"] == "gX"]["tracer_id"] == "E1").all()   # real gene rescued
    ctrl = out[out["feature_name"].str.startswith(("NegControl", "UnassignedCodeword"))]
    assert (ctrl["tracer_id"] == "-1").all()               # controls untouched
    assert stats.get("offpanel_reassigned", 0) == 1        # only gX, not the 2 controls


def test_pre_stage2_reclaims_offpanel_when_enabled():
    # off-panel handling must be part of the STANDARD rescue loop, not only
    # Final Rescue — pre_stage2_rescue (Main + Post-Group) delegates to
    # reassign_unassigned_grid_pool and must pass the flag through.
    out, n, n_skip, stats = pre_stage2_rescue(
        _df(), _aux(), entity_col="tracer_id", out_col="tracer_id",
        G=5.0, cluster_guard_n=0, offpanel_first_entity=True,
    )
    off = out[out["feature_name"] == "gX"]
    assert (off["tracer_id"] == "E1").all()


def test_pre_stage2_offpanel_off_by_default():
    out, n, n_skip, stats = pre_stage2_rescue(
        _df(), _aux(), entity_col="tracer_id", out_col="tracer_id",
        G=5.0, cluster_guard_n=0,
    )
    off = out[out["feature_name"] == "gX"]
    assert (off["tracer_id"] == "-1").all()


def test_offpanel_no_neighbor_stays_unassigned():
    # off-panel tx with NO assigned entity in its neighborhood is left alone
    df = _df()
    df.loc[df["feature_name"] == "gX", ["x", "y", "z"]] = [1000.0, 1000.0, 0.0]
    out, n, stats = reassign_unassigned_grid_pool(
        df, _aux(), entity_col="tracer_id", out_col="tracer_id",
        G=5.0, unassigned_labels={"-1"}, offpanel_first_entity=True,
    )
    off = out[out["feature_name"] == "gX"]
    assert (off["tracer_id"] == "-1").all()
