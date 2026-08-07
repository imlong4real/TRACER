"""Tests for the PMI-named scoring API and the count-based cell-quality metric.

Covers the three surfaces introduced by the npmi->pmi refactor:

* ``metrics.build_pmi_matrix`` / ``cc_scoring.build_pmi_matrix_from_long`` —
  the metric-agnostic panel readers (previously hardcoded to an ``"NPMI"``
  column, which forced ``run_tracer`` to rename ``PMI``->``NPMI`` before
  scoring).
* ``metrics.compute_cell_coherence`` — the new default per-cell metric.
* ``run_tracer --score-mode {count,magnitude}``.

The load-bearing claim is the equivalence one: ``compute_cell_coherence`` must
be the *same* function the segmentation uses internally
(``stitching.coherence(mode="count")``). If those drift apart, per-cell QC and
the pipeline silently stop agreeing on what "coherent" means, which is the
exact confusion the refactor set out to remove. That is checked directly
rather than by re-deriving the formula, so a change to either implementation
alone fails the test.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tracer.cc_scoring import build_pmi_matrix_from_long
from tracer.metrics import build_pmi_matrix, compute_cell_coherence
from tracer.stitching import coherence

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _panel(rows, metric_col="PMI"):
    return pd.DataFrame(rows, columns=["gene_i", "gene_j", metric_col])


def _sym_matrix(rng, G, scale):
    """Symmetric, zero-diagonal association matrix."""
    W = rng.normal(0.0, scale, size=(G, G))
    W = np.triu(W, 1)
    return (W + W.T).astype(np.float64)


def _presence(rng, n_cells, G, max_genes=12, min_genes=0):
    M = np.zeros((n_cells, G), dtype=np.int32)
    for c in range(n_cells):
        k = int(rng.integers(min_genes, min(G, max_genes) + 1))
        if k:
            M[c, rng.choice(G, size=k, replace=False)] = 1
    return M


# --------------------------------------------------------------------------- #
# build_pmi_matrix — metric-agnostic panel reader
# --------------------------------------------------------------------------- #
class TestBuildPmiMatrix:
    def test_reads_pmi_column(self):
        mat, gi = build_pmi_matrix(_panel([("A", "B", 0.4)], "PMI"))
        assert mat[gi["A"], gi["B"]] == pytest.approx(0.4)

    def test_reads_npmi_column(self):
        mat, gi = build_pmi_matrix(_panel([("A", "B", 0.4)], "NPMI"))
        assert mat[gi["A"], gi["B"]] == pytest.approx(0.4)

    def test_prefers_pmi_when_both_present(self):
        """A bootstrap panel can carry both; PMI is the pipeline metric, so it
        must win — otherwise scoring silently runs on a different scale than
        the partition it is scoring."""
        df = pd.DataFrame({"gene_i": ["A"], "gene_j": ["B"],
                           "PMI": [0.9], "NPMI": [0.1]})
        mat, gi = build_pmi_matrix(df)
        assert mat[gi["A"], gi["B"]] == pytest.approx(0.9)

    def test_symmetric_and_absent_pairs_zero(self):
        mat, gi = build_pmi_matrix(
            _panel([("A", "B", 0.4), ("A", "C", -0.3)]))
        assert mat[gi["A"], gi["B"]] == mat[gi["B"], gi["A"]]
        assert mat[gi["A"], gi["C"]] == mat[gi["C"], gi["A"]]
        assert mat[gi["B"], gi["C"]] == 0.0          # unobserved pair
        assert np.allclose(mat, mat.T)

    def test_one_directional_input_not_doubled(self):
        """Assignment, not accumulation: a symmetric panel must not inflate.

        Guards the same class of bug the sparse builder had (COO summing
        duplicate cells), from the dense side.
        """
        one = _panel([("A", "B", 0.4)])
        both = pd.concat([one, _panel([("B", "A", 0.4)])], ignore_index=True)
        m1, g1 = build_pmi_matrix(one)
        m2, g2 = build_pmi_matrix(both)
        assert m1[g1["A"], g1["B"]] == pytest.approx(0.4)
        assert m2[g2["A"], g2["B"]] == pytest.approx(0.4)

    def test_gene_index_covers_both_columns(self):
        _mat, gi = build_pmi_matrix(_panel([("A", "B", 0.1), ("C", "D", 0.2)]))
        assert set(gi) == {"A", "B", "C", "D"}


class TestBuildPmiMatrixFromLong:
    """cc_scoring twin — same metric-agnostic contract, different return shape."""

    @pytest.mark.parametrize("col", ["PMI", "NPMI"])
    def test_reads_either_metric(self, col):
        genes, gi, mat, col_idx = build_pmi_matrix_from_long(
            _panel([("A", "B", 0.4)], col))
        assert mat[gi["A"], gi["B"]] == pytest.approx(0.4, rel=1e-6)
        assert mat[gi["B"], gi["A"]] == pytest.approx(0.4, rel=1e-6)
        assert list(genes) == ["A", "B"]
        assert list(col_idx) == [0, 1]

    def test_prefers_pmi_when_both_present(self):
        df = pd.DataFrame({"gene_i": ["A"], "gene_j": ["B"],
                           "PMI": [0.9], "NPMI": [0.1]})
        _genes, gi, mat, _ci = build_pmi_matrix_from_long(df)
        assert mat[gi["A"], gi["B"]] == pytest.approx(0.9, rel=1e-6)


# --------------------------------------------------------------------------- #
# compute_cell_coherence — must equal stitching.coherence(mode="count")
# --------------------------------------------------------------------------- #
class TestCellCoherenceEquivalence:
    @pytest.mark.parametrize(
        "G,scale,tau,metric",
        [
            (40, 0.30, 0.05, "npmi"),   # bounded NPMI scale
            (40, 0.90, 0.20, "pmi"),    # unbounded PMI scale
            (90, 0.60, 0.20, "pmi"),    # wider panel
            (30, 0.50, 0.00, "pmi"),    # tau=0: every signed pair counts
        ],
    )
    def test_matches_segmentation_coherence(self, G, scale, tau, metric):
        rng = np.random.default_rng(7)
        W = _sym_matrix(rng, G, scale)
        M = _presence(rng, 120, G)
        col_idx = np.arange(G, dtype=np.int32)

        coh, pur, con, _df = compute_cell_coherence(
            M=M, col_idx=col_idx, npmi_mat=W, threshold=tau)

        for c in range(M.shape[0]):
            gids = np.flatnonzero(M[c] != 0).astype(np.int64)
            C_ref, p_ref, q_ref = coherence(
                gids, W, mode="count", threshold=tau, metric=metric)
            assert pur[c] == pytest.approx(p_ref, abs=1e-6), f"purity cell {c}"
            assert con[c] == pytest.approx(q_ref, abs=1e-6), f"conflict cell {c}"
            assert coh[c] == pytest.approx(C_ref, abs=1e-6), f"C cell {c}"

    def test_degenerate_cells_are_zero(self):
        """<2 present genes has no pairs; both implementations return zeros
        rather than dividing by an empty pair set."""
        rng = np.random.default_rng(1)
        G = 12
        W = _sym_matrix(rng, G, 0.8)
        M = np.zeros((2, G), dtype=np.int32)
        M[1, 3] = 1                       # one gene; row 0 stays empty
        coh, pur, con, _ = compute_cell_coherence(
            M=M, col_idx=np.arange(G, dtype=np.int32), npmi_mat=W,
            threshold=0.2)
        assert list(coh) == [0.0, 0.0]
        assert list(pur) == [0.0, 0.0]
        assert list(con) == [0.0, 0.0]


class TestCellCoherenceContract:
    def test_scores_df_schema_and_identities(self):
        rng = np.random.default_rng(3)
        G, n = 25, 40
        W = _sym_matrix(rng, G, 0.8)
        M = _presence(rng, n, G, min_genes=2)
        ids = [f"c{i}" for i in range(n)]
        coh, pur, con, df = compute_cell_coherence(
            M=M, col_idx=np.arange(G, dtype=np.int32), npmi_mat=W,
            threshold=0.2, cell_ids=ids)

        assert list(df.columns) == [
            "cell_id", "purity_score", "conflict_score", "coherence",
            "relative_purity", "relative_conflict", "signal_strength",
        ]
        assert df["cell_id"].tolist() == ids
        # coherence is exactly purity - conflict
        assert np.allclose(df["coherence"], df["purity_score"]
                           - df["conflict_score"])
        # count-based metrics are bounded, unlike the retired ReLU magnitudes
        assert ((df["purity_score"] >= 0) & (df["purity_score"] <= 1)).all()
        assert ((df["conflict_score"] >= 0) & (df["conflict_score"] <= 1)).all()
        assert ((df["coherence"] >= -1) & (df["coherence"] <= 1)).all()
        # signal_strength is the fraction clearing tau either way
        assert np.allclose(df["signal_strength"],
                           df["purity_score"] + df["conflict_score"])
        assert np.allclose(coh, df["coherence"])
        assert np.allclose(pur, df["purity_score"])
        assert np.allclose(con, df["conflict_score"])

    def test_no_scores_df_without_cell_ids(self):
        rng = np.random.default_rng(5)
        G = 10
        out = compute_cell_coherence(
            M=_presence(rng, 3, G), col_idx=np.arange(G, dtype=np.int32),
            npmi_mat=_sym_matrix(rng, G, 0.5), threshold=0.2)
        assert out[3] is None

    def test_relative_scores_nan_without_signal(self):
        """A cell whose pairs all sit inside the dead zone has no signal to
        apportion; the relative columns must be NaN, not 0/0 -> a silent 0."""
        G = 6
        W = np.full((G, G), 0.01, dtype=np.float64)
        np.fill_diagonal(W, 0.0)
        M = np.ones((1, G), dtype=np.int32)
        _c, _p, _q, df = compute_cell_coherence(
            M=M, col_idx=np.arange(G, dtype=np.int32), npmi_mat=W,
            threshold=0.2, cell_ids=["c0"])
        assert df["signal_strength"].iloc[0] == pytest.approx(0.0)
        assert np.isnan(df["relative_purity"].iloc[0])
        assert np.isnan(df["relative_conflict"].iloc[0])


# --------------------------------------------------------------------------- #
# run_tracer --score-mode
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def run_tracer_mod():
    """Load scripts/run_tracer.py as a module (it is a script, not a package)."""
    path = _REPO_ROOT / "scripts" / "run_tracer.py"
    spec = importlib.util.spec_from_file_location("_run_tracer_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _score_inputs(seed=11, n_cells=12, tx_per_cell=8):
    """A small (df_post, panel) pair shaped like post-pipeline output."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(10)]
    rows = []
    for c in range(n_cells):
        for _ in range(tx_per_cell):
            rows.append({
                "stitched": f"cell{c}",
                "feature_name": genes[int(rng.integers(len(genes)))],
                "x": float(rng.uniform(0, 50)),
                "y": float(rng.uniform(0, 50)),
                "z": float(rng.normal(0, 1)),
            })
    df_post = pd.DataFrame(rows)
    pairs = [(genes[i], genes[j], float(rng.normal(0, 0.6)))
             for i in range(len(genes)) for j in range(i + 1, len(genes))]
    panel = pd.DataFrame(pairs, columns=["gene_i", "gene_j", "PMI"])
    return df_post, panel


class TestScoreMode:
    def test_count_mode_matches_compute_cell_coherence(self, run_tracer_mod):
        """The default path must be the metric itself, not a reimplementation."""
        import logging
        df_post, panel = _score_inputs()
        log = logging.getLogger("test_score_mode")
        scores, _adata = run_tracer_mod.build_outputs(
            df_post, npmi_panel=panel, log=log, min_tx=1, score_mode="count")

        from tracer.metrics import build_cell_gene_matrix
        work = df_post.rename(columns={"stitched": "cell_id"})[
            ["cell_id", "feature_name", "x", "y", "z"]]
        cell_ids, _g, M, col_idx = build_cell_gene_matrix(
            work, min_transcripts=1, genes_npm=panel, cell_col="cell_id",
            exclude_ids=set(run_tracer_mod.UNASSIGNED_TOKENS))
        mat, _gi = build_pmi_matrix(panel)
        _c, _p, _q, ref = compute_cell_coherence(
            M=M, col_idx=col_idx, npmi_mat=mat, threshold=0.2,
            cell_ids=cell_ids)

        merged = scores.merge(ref, on="cell_id", suffixes=("", "_ref"))
        assert len(merged) == len(ref)
        for c in ["purity_score", "conflict_score", "coherence"]:
            assert np.allclose(merged[c], merged[f"{c}_ref"], equal_nan=True)

    def test_both_modes_share_one_schema(self, run_tracer_mod):
        """Downstream consumers read cell_scores.tsv.gz without knowing the
        mode, so the columns must not depend on it."""
        import logging
        df_post, panel = _score_inputs()
        log = logging.getLogger("test_score_mode")
        cnt, _ = run_tracer_mod.build_outputs(
            df_post, npmi_panel=panel, log=log, min_tx=1, score_mode="count")
        mag, _ = run_tracer_mod.build_outputs(
            df_post, npmi_panel=panel, log=log, min_tx=1, score_mode="magnitude")
        assert list(cnt.columns) == list(mag.columns)
        assert set(cnt["cell_id"]) == set(mag["cell_id"])

    def test_unknown_mode_rejected(self, run_tracer_mod):
        import logging
        df_post, panel = _score_inputs()
        with pytest.raises(SystemExit):
            run_tracer_mod.build_outputs(
                df_post, npmi_panel=panel, log=logging.getLogger("t"),
                min_tx=1, score_mode="bogus")

    def test_tau_autoselects_by_panel_metric(self, run_tracer_mod):
        """tau=None must track the panel's metric scale: 0.2 on PMI, 0.05 on
        NPMI. Verified behaviourally — auto must equal the explicit threshold
        it claims to pick, and differ from the other one."""
        import logging
        log = logging.getLogger("test_score_mode")
        df_post, panel = _score_inputs()

        auto, _ = run_tracer_mod.build_outputs(
            df_post, npmi_panel=panel, log=log, min_tx=1, tau=None)
        at_02, _ = run_tracer_mod.build_outputs(
            df_post, npmi_panel=panel, log=log, min_tx=1, tau=0.2)
        at_005, _ = run_tracer_mod.build_outputs(
            df_post, npmi_panel=panel, log=log, min_tx=1, tau=0.05)
        assert np.allclose(auto["coherence"], at_02["coherence"])
        assert not np.allclose(auto["coherence"], at_005["coherence"])

        npmi_panel = panel.rename(columns={"PMI": "NPMI"})
        auto_n, _ = run_tracer_mod.build_outputs(
            df_post, npmi_panel=npmi_panel, log=log, min_tx=1, tau=None)
        at_005_n, _ = run_tracer_mod.build_outputs(
            df_post, npmi_panel=npmi_panel, log=log, min_tx=1, tau=0.05)
        assert np.allclose(auto_n["coherence"], at_005_n["coherence"])
