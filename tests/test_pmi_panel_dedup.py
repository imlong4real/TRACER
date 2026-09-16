"""Regression: a symmetric/duplicated PMI panel must not inflate W.

Guards the run_tracer panel-doubling bug. `run_tracer.load_npmi_panel` used to
expand a one-directional panel to both (i, j) and (j, i); those both fold to the
same (i < j) cell in `build_sparse_pmi_matrix_from_long`, and the COO->CSR build
SUMS duplicate coordinates -> every PMI doubled (2x), silently halving every
calibrated threshold. The dedup in `build_sparse_pmi_matrix_from_long` must
collapse duplicate cells to one value so no caller can trip this.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from tracer.pruning import build_sparse_pmi_matrix_from_long


def _panel(rows):
    return pd.DataFrame(rows, columns=["gene_i", "gene_j", "PMI"])


def test_symmetric_panel_not_inflated():
    """A both-directions panel yields the SAME W as the one-directional one."""
    one_dir = _panel([("A", "B", 0.4), ("A", "C", -0.2), ("B", "C", 0.9)])
    rev = one_dir.copy()
    rev["gene_i"], rev["gene_j"] = one_dir["gene_j"].values, one_dir["gene_i"].values
    doubled = pd.concat([one_dir, rev], ignore_index=True)

    _, _, W1 = build_sparse_pmi_matrix_from_long(one_dir, metric_col="PMI")
    _, _, W2 = build_sparse_pmi_matrix_from_long(doubled, metric_col="PMI")

    assert W1.nnz == W2.nnz == 3
    assert (W1 - W2).nnz == 0
    # values equal the raw single PMI (no 2x)
    assert np.allclose(np.sort(W1.data), np.sort([0.4, -0.2, 0.9]).astype(np.float32))


def test_conflicting_duplicates_warn_and_keep_first():
    """Duplicate pair with DIFFERENT values -> warn, keep first (malformed panel)."""
    panel = _panel([("A", "B", 0.5), ("B", "A", 0.9)])  # same undirected pair
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, _, W = build_sparse_pmi_matrix_from_long(panel, metric_col="PMI")
    assert any("conflicting" in str(x.message) for x in w)
    assert W.nnz == 1
    assert float(W.tocoo().data[0]) == pytest.approx(0.5)  # first kept, not summed


def test_identical_duplicates_silent_and_not_doubled():
    """Duplicate pair with the SAME value -> no warning, single (not doubled) value."""
    panel = _panel([("A", "B", 0.5), ("B", "A", 0.5)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, _, W = build_sparse_pmi_matrix_from_long(panel, metric_col="PMI")
    assert not any("conflicting" in str(x.message) for x in w)
    assert W.nnz == 1
    assert float(W.tocoo().data[0]) == pytest.approx(0.5)  # 0.5, not 1.0


def test_clean_one_directional_panel_is_untouched():
    """No duplicates -> dedup is a pure no-op (every pair preserved verbatim)."""
    panel = _panel([("A", "B", 0.4), ("A", "C", -0.2), ("B", "C", 0.9), ("A", "D", 0.1)])
    _, _, W = build_sparse_pmi_matrix_from_long(panel, metric_col="PMI")
    assert W.nnz == 4
    assert np.allclose(np.sort(W.data), np.sort([0.4, -0.2, 0.9, 0.1]).astype(np.float32))
