from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tutorials.gbm.generate_npmi import _extract_accepted_panel
from tutorials.gbm.run_gbm import load_npmi_panel
from tutorials.gbm.run_gbm_prune_diagnostic import _panel_seed_support


def test_extract_accepted_panel_uses_only_w_sparse_edges():
    genes = np.asarray(["A", "B", "C"])
    W = sp.csr_matrix(
        (
            np.asarray([0.4, -1.0], dtype=np.float32),
            (np.asarray([0, 0]), np.asarray([1, 2])),
        ),
        shape=(3, 3),
    )
    pair_ci = pd.DataFrame(
        [
            (0, 1, "A", "B", "pos", 0.2, 0.4),
            (0, 2, "A", "C", "neg_one", -1.0, -1.0),
            (1, 2, "B", "C", "dead_zone", 0.01, 0.02),
        ],
        columns=[
            "gene_i_idx",
            "gene_j_idx",
            "gene_i",
            "gene_j",
            "kind",
            "legacy_npmi",
            "legacy_pmi",
        ],
    )
    panel, audit = _extract_accepted_panel(
        SimpleNamespace(genes=genes, W_sparse=W, pair_ci=pair_ci)
    )

    assert list(panel[["gene_i", "gene_j"]].itertuples(index=False, name=None)) == [
        ("A", "B"),
        ("A", "C"),
    ]
    np.testing.assert_allclose(panel["PMI"], [0.4, -1.0])
    np.testing.assert_allclose(panel["NPMI"], [0.2, -1.0])
    accepted = audit.set_index(["gene_i", "gene_j"])["accepted_in_w_sparse"]
    assert bool(accepted.loc[("A", "B")])
    assert bool(accepted.loc[("A", "C")])
    assert not bool(accepted.loc[("B", "C")])


def test_load_npmi_panel_collapses_symmetric_rows_without_doubling(tmp_path):
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        [
            ("A", "B", 0.4),
            ("B", "A", 0.4),
        ],
        columns=["gene_i", "gene_j", "PMI"],
    ).to_csv(path, index=False)

    panel = load_npmi_panel(path)

    assert len(panel) == 1
    assert panel.loc[0, "gene_i"] == "A"
    assert panel.loc[0, "gene_j"] == "B"
    assert panel.loc[0, "PMI"] == 0.4


def test_panel_seed_support_counts_stored_edges_per_nucleus():
    df = pd.DataFrame(
        {
            "cell_id": ["c1", "c1", "c2", "c2", "-1"],
            "feature_name": ["A", "B", "A", "C", "B"],
            "overlaps_nucleus": [1, 1, 1, 1, 1],
        }
    )
    W = sp.csr_matrix(
        (
            np.asarray([0.4], dtype=np.float32),
            (np.asarray([0]), np.asarray([1])),
        ),
        shape=(3, 3),
    )
    aux = {
        "gene_to_idx": {"A": 0, "B": 1, "C": 2},
        "W": W,
    }

    report = _panel_seed_support(df, aux)

    assert report.loc["c1", "accepted_nuclear_gene_pairs"] == 1
    assert report.loc["c2", "accepted_nuclear_gene_pairs"] == 0
