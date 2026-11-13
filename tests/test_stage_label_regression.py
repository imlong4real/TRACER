"""Per-transcript, per-stage label regression (unified-prune step 1).

``test_pipeline_regression.py`` pins entity *counts* and ARI on the final
output. That is too coarse for the unified-prune refactor
(``docs/design/unified_prune.md``), whose extraction steps must leave every
transcript's label unchanged at every stage. This module pins exactly that:
for each stage the pipeline records, it hashes the ``(transcript_id, label,
etype)`` triples and compares against ``tests/references/stage_labels_*.json``.

Two hashes per stage:

* ``label_sha``     -- the raw labels. Catches renames as well as moves.
* ``partition_sha`` -- labels relabelled by order of first appearance, plus
  etype. Unchanged when only entity *names* change, so a failure message can
  say "names changed, partition did not".

Variants
--------
``seg_nuclear``   SEG with ``overlaps_nucleus`` -> nuclear-seed Prune (1a/1b/1c),
                  Reassign-1c, Rerank, and the SEG residual cascade. This is the
                  path whose ``aux["W"]`` is upper-triangle only.
``seg_wholecell`` SEG without a nucleus column -> ``prune_transcripts_fast``.
``noseg``         NOSEG -> cascade driven by ``cfg.group``.

The existing synthetic fixtures carry ``is_nuclear`` but not
``overlaps_nucleus`` and a panel with only an ``NPMI`` column, so they never
reach the nuclear-seed path nor the production ``metric_col="PMI"`` branch.
These fixtures add both. ``test_fixture_exercises_target_paths`` fails if a
generator change stops them from doing so.

Updating
--------
``TRACER_UPDATE_REFERENCES=1 pytest tests/test_stage_label_regression.py``
rewrites the references (same switch as the count-level regression tests).
Only do that for an intentional behaviour change, and say so in the commit.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import tracer.pipeline as P
from tests._regression_helpers import REFERENCE_DIR
from tests.segmentation_sim import simulate_dapi_voronoi_segmentation
from tests.synthetic import (
    make_synthetic_npmi_panel_for_transcripts,
    make_synthetic_transcripts,
)

# 30 cells x 60 tx over 24 genes / 4 types, DAPI + Voronoi segmentation.
# ~1,800 tx; each variant runs in well under a second. At this size the SEG
# run yields 1c partials, Split-Phase1 splits, Mid-QC demotions and ~10
# cascade entities at Group, i.e. every stage the refactor touches moves.
GEN_KW = dict(
    n_cells=30,
    voxels_per_cell_mean=120,
    tx_per_cell=60,
    n_genes=24,
    n_types=4,
    domain_z_um=10.0,
    nuclear_layers=2,
    seed=7,
)


def _panel(df, gt) -> pd.DataFrame:
    panel = make_synthetic_npmi_panel_for_transcripts(df, gt)
    # Production panels carry PMI; the pipeline only takes the PMI branch
    # when that column exists.
    return panel.rename(columns={"NPMI": "PMI"})


@pytest.fixture(scope="module")
def inputs():
    df, gt = make_synthetic_transcripts(**GEN_KW)
    seg = simulate_dapi_voronoi_segmentation(df)
    seg["overlaps_nucleus"] = seg["is_nuclear"].astype(int)
    return seg, _panel(df, gt)


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------
def _stage_hashes(tid: np.ndarray, label: np.ndarray, etype: np.ndarray) -> dict:
    order = np.argsort(tid, kind="stable")
    tid, label, etype = tid[order], label[order], etype[order]
    raw = "\n".join(f"{t}\t{l}\t{e}" for t, l, e in zip(tid, label, etype))
    # First-appearance relabelling in transcript_id order. Unassigned
    # ("-1") keeps its own token so it never aliases an entity.
    canon, seen = [], {}
    for l in label:
        canon.append("U" if l == "-1" else seen.setdefault(l, str(len(seen))))
    part = "\n".join(f"{c}\t{e}" for c, e in zip(canon, etype))
    return {
        "n_tx": int(len(tid)),
        "n_labels": int(len(set(label))),
        "label_sha": hashlib.sha256(raw.encode()).hexdigest(),
        "partition_sha": hashlib.sha256(part.encode()).hexdigest(),
    }


def _run(monkeypatch, runner, df, panel) -> tuple[list[dict], dict]:
    """Run ``runner`` and return per-stage hashes plus raw captures."""
    stages, raw = [], {}
    orig = P._record_stage

    def capture(progression, stage_name, frame, col):
        tid = frame["transcript_id"].astype(str).to_numpy()
        lab = frame[col].astype(str).to_numpy()
        if "_etype" in frame.columns:
            et = frame["_etype"].astype(str).to_numpy()
        else:
            et = np.full(len(lab), "")
        stages.append({"stage": stage_name, **_stage_hashes(tid, lab, et)})
        raw[stage_name] = lab
        return orig(progression, stage_name, frame, col)

    monkeypatch.setattr(P, "_record_stage", capture)
    runner(df.copy(), panel)
    return stages, raw


# --------------------------------------------------------------------------
# Reference comparison
# --------------------------------------------------------------------------
def _assert_stages(name: str, stages: list[dict]) -> None:
    path = REFERENCE_DIR / f"stage_labels_{name}.json"
    if os.environ.get("TRACER_UPDATE_REFERENCES") or not path.exists():
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fixture": GEN_KW, "stages": stages}, indent=2) + "\n")
        pytest.skip(f"Wrote reference: {path}")

    ref = json.loads(path.read_text())
    assert ref["fixture"] == GEN_KW, (
        f"{path.name} was generated with a different fixture; regenerate it."
    )
    ref_stages = ref["stages"]
    ref_names = [s["stage"] for s in ref_stages]
    cur_names = [s["stage"] for s in stages]
    if ref_names != cur_names:
        pytest.fail(
            f"Stage sequence changed for {name}:\n  reference: {ref_names}\n"
            f"  current:   {cur_names}", pytrace=False,
        )
    for r, c in zip(ref_stages, stages):
        if r == c:
            continue
        if r["partition_sha"] == c["partition_sha"]:
            what = "entity NAMES changed; the partition and etypes are identical"
        else:
            what = "the PARTITION (or etypes) changed"
        pytest.fail(
            f"{name}: first divergence at stage '{c['stage']}' -- {what}.\n"
            f"  reference: {r}\n  current:   {c}\n\n"
            "If intentional, regenerate with\n"
            "  TRACER_UPDATE_REFERENCES=1 pytest tests/test_stage_label_regression.py\n"
            f"and commit tests/references/{path.name}.",
            pytrace=False,
        )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_stage_labels_seg_nuclear(monkeypatch, inputs):
    df, panel = inputs
    stages, _ = _run(monkeypatch, P.run_segmented_pipeline, df, panel)
    _assert_stages("seg_nuclear", stages)


def test_stage_labels_seg_wholecell(monkeypatch, inputs):
    df, panel = inputs
    stages, _ = _run(
        monkeypatch, P.run_segmented_pipeline,
        df.drop(columns=["overlaps_nucleus"]), panel,
    )
    _assert_stages("seg_wholecell", stages)


def test_stage_labels_noseg(monkeypatch, inputs):
    df, panel = inputs
    stages, _ = _run(monkeypatch, P.run_noseg_pipeline, df, panel)
    _assert_stages("noseg", stages)


def test_fixture_exercises_target_paths(monkeypatch, inputs):
    """Guard the fixture itself: the refactor-relevant paths must fire."""
    df, panel = inputs
    seen = {}
    orig_nuc = P.prune_transcripts_nuclear_seed

    def spy(*a, **k):
        out = orig_nuc(*a, **k)
        W = sp.csr_matrix(out[1]["W"])
        seen["W_lower_nnz"] = sp.tril(W, -1).nnz
        seen["W_upper_nnz"] = sp.triu(W, 1).nnz
        return out

    monkeypatch.setattr(P, "prune_transcripts_nuclear_seed", spy)
    stages, raw = _run(monkeypatch, P.run_segmented_pipeline, df, panel)

    assert seen, "nuclear-seed Prune did not run (overlaps_nucleus ignored?)"
    # Documents today's W contract (upper triangle only). Unified-prune
    # step 2 symmetrizes W; update this assertion in that commit.
    assert seen["W_upper_nnz"] > 0 and seen["W_lower_nnz"] == 0

    names = [s["stage"] for s in stages]
    for st in ("Prune", "Phase1-Reassign-1c", "Split-Phase1", "Phase1-QC",
               "Group", "Mid-QC", "Stitch"):
        assert st in names, f"stage {st} not recorded"

    group = set(raw["Group"])
    n_cascade = sum(l.startswith("cascade") for l in group)
    assert n_cascade >= 3, f"only {n_cascade} cascade entities at Group"
    by = {s["stage"]: s for s in stages}
    assert by["Split-Phase1"]["n_labels"] > by["Prune"]["n_labels"], "Split-Phase1 split nothing"
