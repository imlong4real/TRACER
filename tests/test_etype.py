"""Unit tests for `tracer._etype` foundation helpers.

These tests verify the foundation alone — no stage emitters yet.
Parity vs the legacy `infer_entity_type` parser is checked on integer
cell_ids; on dash-containing FFPE-style cell_ids the helper
*intentionally* reproduces the legacy bug (so it serves as a
regression baseline for the column-based emitters that follow).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracer._etype import (
    ETYPE_CATEGORIES,
    ETYPE_DTYPE,
    empty_etype,
    etype_from_codes,
    infer_etype_from_label,
    infer_entity_type_etype,
)


def test_categories_canonical():
    assert ETYPE_CATEGORIES == ["cell", "partial", "component", "drop", "unknown"]


def test_dtype_uses_canonical_categories():
    assert list(ETYPE_DTYPE.categories) == ETYPE_CATEGORIES
    assert not ETYPE_DTYPE.ordered


def test_empty_etype():
    e = empty_etype(5)
    assert isinstance(e, pd.Categorical)
    assert e.dtype == ETYPE_DTYPE
    assert (np.asarray(e) == "unknown").all()


def test_etype_from_codes_basic():
    codes = np.array([0, 1, 2, 0, 1], dtype=np.int8)
    e = etype_from_codes(codes)
    assert list(np.asarray(e)) == ["cell", "partial", "unknown", "cell", "partial"]


def test_etype_from_codes_fallback_maps_to_unknown():
    codes = np.array([0, 1, 2, 3], dtype=np.int8)
    e = etype_from_codes(codes)
    # Codes 2 and 3 both → unknown
    assert list(np.asarray(e)) == ["cell", "partial", "unknown", "unknown"]


def test_infer_etype_from_label_integer_ids():
    labels = pd.Series(["42", "42-1", "42-1-1", "UNASSIGNED_3", "-1", "DROP", "nan"])
    e = infer_etype_from_label(labels)
    assert list(np.asarray(e)) == [
        "cell", "partial", "partial", "component", "unknown", "unknown", "unknown"
    ]


def test_infer_etype_from_label_rejected_sentinels():
    labels = pd.Series(["prune_rejected", "group_rejected", "demote_rejected"])
    e = infer_etype_from_label(labels)
    assert list(np.asarray(e)) == ["unknown", "unknown", "unknown"]


def test_infer_etype_from_label_ffpe_dash_in_cell_id_documents_legacy_bug():
    """PDAC-style alphanumeric cell_id with native `-1` suffix.

    The legacy parsing rule misclassifies the main as a partial because
    the cell_id contains a dash. This test DOCUMENTS the legacy bug so
    that stage emitters using kernel codes (which avoid the bug
    entirely) can verify they produce CORRECT classifications even on
    these labels.
    """
    labels = pd.Series(["adohnpem-1", "adohnpem-1-1"])
    e = infer_etype_from_label(labels)
    # Legacy says both are "partial" (the bug — `adohnpem-1` is really
    # a main, but the parser sees a dash and calls it a partial).
    assert list(np.asarray(e)) == ["partial", "partial"]


def test_concat_preserves_dtype():
    a = pd.Series(empty_etype(3))
    b = pd.Series(pd.Categorical(["cell", "partial", "unknown"], dtype=ETYPE_DTYPE))
    c = pd.concat([a, b]).reset_index(drop=True)
    assert c.dtype == ETYPE_DTYPE


def test_can_assign_categorical_value_via_loc():
    df = pd.DataFrame({"x": [1, 2, 3]})
    df["_etype"] = empty_etype(3)
    df.loc[df["x"] == 2, "_etype"] = "cell"
    assert df["_etype"].dtype == ETYPE_DTYPE
    assert list(df["_etype"].astype(str)) == ["unknown", "cell", "unknown"]


def test_invalid_category_assignment_raises_or_becomes_nan():
    """Assigning a string not in ETYPE_CATEGORIES should NOT silently
    produce a wrong category. pandas raises TypeError or sets NaN."""
    df = pd.DataFrame({"_etype": empty_etype(2)})
    # pandas behavior on out-of-category assignment differs by version;
    # whichever happens, the value must NOT silently become a valid
    # different category.
    try:
        df.loc[0, "_etype"] = "not_a_real_category"
    except (TypeError, ValueError):
        return  # acceptable: pandas refused
    # If pandas allowed the assignment, it must have NaN'd it.
    assert pd.isna(df["_etype"].iloc[0]) or str(df["_etype"].iloc[0]) != "cell"


def test_infer_entity_type_etype_reads_column():
    df = pd.DataFrame({
        "tracer_id": ["42", "42-1", "UNASSIGNED_7"],
        "_etype": pd.Categorical(
            ["cell", "partial", "component"], dtype=ETYPE_DTYPE
        ),
    })
    kinds = infer_entity_type_etype(df)
    assert list(kinds) == ["cell", "partial", "component"]


# ---------------------------------------------------------------------------
# Step 2 — Phase 1 emitter parity
# ---------------------------------------------------------------------------


def test_phase1_emitter_writes_etype_consistent_with_labels():
    """On integer cell_ids, the Phase 1 emitter's `_etype` column must
    agree with `infer_etype_from_label` applied to the produced label
    column. This is the parity gate: if it diverges on integer
    cell_ids, the emitter is buggy. (On FFPE/IO dash-containing
    cell_ids, the emitter would be CORRECT and the parity helper would
    be WRONG — the inversion that motivates the refactor — but we
    don't exercise that case here; that's the job of the PDAC
    re-bench in Step 4.)
    """
    from tests.synthetic import (
        make_synthetic_transcripts,
        make_synthetic_npmi_panel_for_transcripts,
    )
    from tracer.pruning import prune_transcripts_fast

    df, gt = make_synthetic_transcripts(
        n_cells=10, n_types=2, seed=42,
    )
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    df_out, _aux = prune_transcripts_fast(
        df.copy(), panel,
        cell_id_col="cell_id", gene_col="feature_name",
        threshold=0.05, unassigned_id="-1",
        nan_fill=0.0, n_jobs=-1, show_progress=False,
    )
    # `_etype` column is present and a Categorical
    assert "_etype" in df_out.columns
    assert df_out["_etype"].dtype == ETYPE_DTYPE

    # Parity against legacy label parsing on integer cell_ids.
    # The label column produced by prune_transcripts_fast is `tracer_id`
    # by default (or whatever `out_col` is set to). Check the columns
    # to find it.
    label_cols = [c for c in df_out.columns if c in ("tracer_id", "out", "_out")]
    # Default out_col in the package is `tracer_id`; if that's missing,
    # fall back to whatever the function actually wrote. Worst case
    # we'll find no match and the test surfaces a real bug.
    assert label_cols, f"expected a label column; got {list(df_out.columns)}"
    label_col = label_cols[0]

    legacy_kinds = infer_etype_from_label(df_out[label_col])
    new_kinds = df_out["_etype"]
    # Compare as strings (Categorical equality requires same dtype)
    assert (
        np.asarray(legacy_kinds).astype(str) == np.asarray(new_kinds).astype(str)
    ).all(), (
        "Phase 1 _etype emitter diverges from legacy label classification "
        "on integer cell_ids — this is the parity gate failure."
    )


def test_phase1_nuclear_seed_path_writes_etype_from_kernel_codes():
    """The Cython-batched nuclear-seed prune path (production for
    Xenium FFPE / IO) writes `_etype` directly from kernel codes,
    bypassing label-string parsing. On integer cell_ids the result
    must agree with `infer_etype_from_label` applied to the label
    column (parity gate); the bug-free behavior on FFPE cell_ids is
    exercised separately in the PDAC re-bench (Step 4)."""
    from tests.synthetic import (
        make_synthetic_transcripts,
        make_synthetic_npmi_panel_for_transcripts,
    )
    from tracer.pruning import prune_transcripts_nuclear_seed

    df, gt = make_synthetic_transcripts(
        n_cells=10, n_types=2, seed=42,
    )
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    # The nuclear-seed path requires the column the runner expects.
    df = df.rename(columns={"is_nuclear": "overlaps_nucleus"})

    df_out, _aux = prune_transcripts_nuclear_seed(
        df.copy(), panel,
        cell_id_col="cell_id", gene_col="feature_name",
        nuclear_col="overlaps_nucleus",
        threshold=0.05, unassigned_id="-1",
        metric_col="NPMI",  # synthetic panel uses NPMI column name
        nan_fill=0.0, n_jobs=-1, show_progress=False,
    )
    assert "_etype" in df_out.columns
    assert df_out["_etype"].dtype == ETYPE_DTYPE

    label_cols = [c for c in df_out.columns if c == "tracer_id"]
    assert label_cols, f"expected tracer_id; got {list(df_out.columns)}"
    label_col = label_cols[0]

    legacy_kinds = infer_etype_from_label(df_out[label_col])
    new_kinds = df_out["_etype"]
    assert (
        np.asarray(legacy_kinds).astype(str) == np.asarray(new_kinds).astype(str)
    ).all(), (
        "Nuclear-seed _etype emitter diverges from legacy classification "
        "on integer cell_ids — kernel-code mapping is buggy."
    )


def test_phase1_family_etype_parity_end_to_end_seg_smoke():
    """End-to-end parity gate for Step 3 emitters.

    Runs the full SEG pipeline on integer cell_ids and verifies the
    final ``_etype`` column agrees with ``infer_etype_from_label``
    applied to the *active final partition column* (``stitched`` after
    Final Rescue / Finalize). Covers every emitter that writes
    ``_etype``: Phase 1 family (Prune, Reassign-1c, Split-Phase1,
    Phase1-QC, Phase1-Rerank), Mid-QC (Split-Unassigned, Demote-low-C),
    Stitch, Demote, and Final Rescue.

    On integer cell_ids the legacy parsing is correct, so parity is
    the right invariant. On FFPE/IO cell_ids the legacy parsing is
    WRONG (the bug that motivates this whole refactor); we verify
    that case via the PDAC re-bench in Step 4.

    NOTE: Comparing against ``stitched`` (not ``tracer_id``) is
    correct: ``tracer_id`` is the pre-Stitch column and is no longer
    mutated after Stitch hands off to ``stitched``. ``_etype`` tracks
    the active partition.
    """
    from tests.synthetic import (
        make_synthetic_transcripts,
        make_synthetic_npmi_panel_for_transcripts,
    )
    import tests._pipeline_runner as runner
    from tests._pipeline_runner import run_segmented_pipeline

    df, gt = make_synthetic_transcripts(n_cells=15, n_types=3, seed=42)
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    # Force the nuclear-seed prune path so we exercise the Cython
    # kernel-code emitter (production-relevant), not just the legacy
    # whole-cell prune.
    df_nuc = df.rename(columns={"is_nuclear": "overlaps_nucleus"})

    # Snapshot defaults; restore at end.
    orig_rerank = runner.PHASE1_RERANK_ENABLED
    orig_reassign = runner.PHASE1_REASSIGN_AFTER_1C
    try:
        runner.PHASE1_RERANK_ENABLED = True  # exercise the rerank emitter
        runner.PHASE1_REASSIGN_AFTER_1C = True
        df_out, _prog = run_segmented_pipeline(df_nuc, panel)
    finally:
        runner.PHASE1_RERANK_ENABLED = orig_rerank
        runner.PHASE1_REASSIGN_AFTER_1C = orig_reassign

    assert "_etype" in df_out.columns, (
        "End-to-end pipeline must carry _etype through to the final "
        "output (Phase 1 family emitters should populate it)."
    )
    assert df_out["_etype"].dtype == ETYPE_DTYPE

    # Parity vs legacy on the active partition column. After Stitch,
    # the live label column is `stitched`; `tracer_id` freezes at the
    # pre-Stitch state. Final Rescue can promote tx whose tracer_id is
    # still "-1" to a cascade partial — _etype correctly reflects the
    # new stitched-column assignment.
    final_col = "stitched" if "stitched" in df_out.columns else "tracer_id"
    legacy = infer_etype_from_label(df_out[final_col])
    new = df_out["_etype"]
    legacy_arr = np.asarray(legacy).astype(str)
    new_arr = np.asarray(new).astype(str)

    if not (legacy_arr == new_arr).all():
        mism = (legacy_arr != new_arr)
        labels = np.asarray(df_out[final_col]).astype(str)
        sample = labels[mism][:10]
        legacy_samp = legacy_arr[mism][:10]
        new_samp = new_arr[mism][:10]
        msg = (
            f"_etype diverges from label-parse on {mism.sum()}/{len(mism)} tx. "
            f"Sample (label / legacy / new): "
            + ", ".join(f"{l!r}/{lk}/{nk}"
                        for l, lk, nk in zip(sample, legacy_samp, new_samp))
        )
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Step 4 — etype-aware rerank reader (cell_id-based parent identification)
# ---------------------------------------------------------------------------


def _build_rerank_test_frame(cell_id: str, *, n_main: int, n_partial: int,
                              n_subpartial: int = 0,
                              partial_idx: int = 1,
                              subpartial_idx: int = 1) -> pd.DataFrame:
    """Build a minimal DataFrame for rerank testing.

    Returns a DataFrame with columns: tracer_id, cell_id, overlaps_nucleus,
    _etype. All rows under one parent ``cell_id``. ``tracer_id`` follows
    the legacy dash convention: main = cell_id, partial = ``f'{cell_id}-{partial_idx}'``,
    sub-partial = ``f'{cell_id}-{partial_idx}-{subpartial_idx}'``.
    """
    from tracer._etype import ETYPE_DTYPE
    rows = []
    # main
    for _ in range(n_main):
        rows.append((cell_id, cell_id, True, "cell"))
    # partial
    partial_lab = f"{cell_id}-{partial_idx}"
    for _ in range(n_partial):
        rows.append((partial_lab, cell_id, True, "partial"))
    # sub-partial
    sub_lab = f"{cell_id}-{partial_idx}-{subpartial_idx}"
    for _ in range(n_subpartial):
        rows.append((sub_lab, cell_id, True, "partial"))
    df = pd.DataFrame(rows, columns=["tracer_id", "cell_id", "overlaps_nucleus", "_etype"])
    df["_etype"] = df["_etype"].astype(ETYPE_DTYPE)
    return df


def test_rerank_etype_integer_cell_id_parity_swap():
    """Integer cell_ids — etype-aware rerank performs the expected
    single-swap when the depth-1 partial outscores the main."""
    from tests._pipeline_runner import _phase1_rerank_within_parent_etype
    df = _build_rerank_test_frame("42", n_main=3, n_partial=5)
    out_etype, stats_etype = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id", cell_id_col="cell_id", margin_tx=1,
    )
    assert stats_etype["n_parents_reranked"] == 1
    assert stats_etype["n_tx_relabeled"] == 8
    # Swap should have happened (5 > 3).
    counts = out_etype["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 5, "42-1": 3}


def test_rerank_etype_integer_cell_id_parity_tie():
    """Tie keeps the original main label."""
    from tests._pipeline_runner import _phase1_rerank_within_parent_etype
    df = _build_rerank_test_frame("42", n_main=4, n_partial=4)
    out_etype, stats_etype = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id", cell_id_col="cell_id", margin_tx=1,
    )
    assert stats_etype["n_parents_reranked"] == 0
    counts = out_etype["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 4, "42-1": 4}


def test_rerank_etype_subpartial_follows_with_bump_on_collision():
    """Sub-partial follows depth-1 ancestor; deposed main bumps to next
    free suffix slot."""
    from tests._pipeline_runner import _phase1_rerank_within_parent_etype
    df = _build_rerank_test_frame(
        "42", n_main=2, n_partial=4, n_subpartial=2,
    )
    out_etype, _ = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id", cell_id_col="cell_id", margin_tx=1,
    )
    counts = out_etype["tracer_id"].value_counts().to_dict()
    assert counts == {"42": 4, "42-1": 2, "42-2": 2}


def test_rerank_etype_works_on_ffpe_dash_in_cell_id():
    """The killer test: PDAC-style cell_id ``adohnpem-1`` reranks
    correctly. Pre-refactor legacy regex rerank silently no-oped on
    FFPE cell_ids because the dash in the cell_id confused the parent-
    identification regex. The etype-aware sibling reads ``cell_id_col``
    directly, so it works regardless of cell_id format."""
    from tests._pipeline_runner import _phase1_rerank_within_parent_etype
    df = _build_rerank_test_frame("adohnpem-1", n_main=3, n_partial=5)
    out_etype, stats_etype = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id", cell_id_col="cell_id", margin_tx=1,
    )
    assert stats_etype["n_parents_reranked"] == 1
    assert stats_etype["n_tx_relabeled"] == 8
    counts = out_etype["tracer_id"].value_counts().to_dict()
    assert counts == {"adohnpem-1": 5, "adohnpem-1-1": 3}


def test_rerank_etype_handles_pdac_subpartial():
    """Sub-partial follow + bump-on-collision works on PDAC cell_ids."""
    from tests._pipeline_runner import _phase1_rerank_within_parent_etype
    df = _build_rerank_test_frame(
        "adohnpem-1", n_main=2, n_partial=4, n_subpartial=2,
    )
    out, stats = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id", cell_id_col="cell_id", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    # After swap: was 42-1 (4 + 2 sub) → "adohnpem-1"; sub follows;
    # deposed main (2 tx) bumps past the reserved sub-suffix slot.
    assert counts == {"adohnpem-1": 4, "adohnpem-1-1": 2, "adohnpem-1-2": 2}
    assert stats["n_parents_reranked"] == 1


def test_rerank_etype_unassigned_label_skipped():
    """Rows with cell_id == sentinel are skipped; UNASSIGNED entities ignored."""
    from tests._pipeline_runner import _phase1_rerank_within_parent_etype
    from tracer._etype import ETYPE_DTYPE
    df = pd.DataFrame(
        [
            ("42",            "42",  True,  "cell"),
            ("42-1",          "42",  True,  "partial"),
            ("42-1",          "42",  True,  "partial"),
            ("42-1",          "42",  True,  "partial"),
            ("42-1",          "42",  True,  "partial"),
            ("42-1",          "42",  True,  "partial"),
            ("UNASSIGNED_7", "42",  True,  "component"),
            ("-1",            "-1",  False, "unknown"),
        ],
        columns=["tracer_id", "cell_id", "overlaps_nucleus", "_etype"],
    )
    df["_etype"] = df["_etype"].astype(ETYPE_DTYPE)
    out, stats = _phase1_rerank_within_parent_etype(
        df, entity_col="tracer_id", cell_id_col="cell_id", margin_tx=1,
    )
    counts = out["tracer_id"].value_counts().to_dict()
    # main 42 (1 tx) vs partial 42-1 (5 tx) → swap.
    # UNASSIGNED_7 and -1 untouched.
    assert counts["42"] == 5
    assert counts["42-1"] == 1
    assert counts["UNASSIGNED_7"] == 1
    assert counts["-1"] == 1




# ---------------------------------------------------------------------
# Post-merge _etype homogenization (homogenize_etype_for_entity / _entities)
# ---------------------------------------------------------------------

def _etyped(labels, etypes):
    from tracer._etype import ETYPE_DTYPE
    df = pd.DataFrame({"tracer_id": labels, "_etype": etypes})
    df["_etype"] = df["_etype"].astype(ETYPE_DTYPE)
    return df


def test_homogenize_noop_on_homogeneous_entity():
    from tracer._etype import homogenize_etype_for_entity
    df = _etyped(["42", "42", "42"], ["cell", "cell", "cell"])
    homogenize_etype_for_entity(df, "42")
    assert df["_etype"].astype(str).tolist() == ["cell", "cell", "cell"]


@pytest.mark.parametrize(
    "etypes, winner",
    [
        (["cell", "partial"], "cell"),
        (["partial", "component"], "partial"),
        (["component", "drop"], "component"),
        (["drop", "unknown"], "drop"),
        (["unknown", "partial", "cell"], "cell"),
        (["component", "cell", "partial"], "cell"),
    ],
)
def test_homogenize_priority_selection(etypes, winner):
    """cell > partial > component > drop > unknown."""
    from tracer._etype import homogenize_etype_for_entity
    df = _etyped(["e"] * len(etypes), etypes)
    homogenize_etype_for_entity(df, "e")
    assert set(df["_etype"].astype(str)) == {winner}


def test_homogenize_absent_label_is_noop():
    from tracer._etype import homogenize_etype_for_entity
    df = _etyped(["42", "42"], ["cell", "partial"])
    before = df["_etype"].astype(str).tolist()
    homogenize_etype_for_entity(df, "999")  # label not present
    assert df["_etype"].astype(str).tolist() == before


def test_homogenize_missing_etype_column_is_noop():
    from tracer._etype import homogenize_etype_for_entity
    df = pd.DataFrame({"tracer_id": ["42", "42"]})
    homogenize_etype_for_entity(df, "42")  # no _etype column → no error
    assert "_etype" not in df.columns


def test_homogenize_batch_resolves_each_entity_independently():
    """One call, one pass; each entity gets its own priority winner;
    entities not listed are untouched."""
    from tracer._etype import homogenize_etype_for_entities
    df = _etyped(
        ["A", "A", "B", "B", "C", "C", "D", "D"],
        ["cell", "partial", "partial", "component",
         "unknown", "unknown", "cell", "partial"],
    )
    homogenize_etype_for_entities(df, ["A", "B", "C"])  # D omitted
    got = df.groupby("tracer_id")["_etype"].agg(lambda s: set(s.astype(str)))
    assert got["A"] == {"cell"}
    assert got["B"] == {"partial"}
    assert got["C"] == {"unknown"}
    assert got["D"] == {"cell", "partial"}  # untouched


def test_homogenize_batch_empty_labels_is_noop():
    from tracer._etype import homogenize_etype_for_entities
    df = _etyped(["A", "A"], ["cell", "partial"])
    before = df["_etype"].astype(str).tolist()
    homogenize_etype_for_entities(df, [])
    assert df["_etype"].astype(str).tolist() == before
