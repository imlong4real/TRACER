"""`_classify_counts` — counts must be integers; everything else is rejected.

ONE RULE. CP10k, TPM, log1p(CP10k) and any mix are non-integral, so
integrality alone separates the good case from every bad one.

Two alternatives were tried and dropped, both recorded here so they are not
re-proposed:

  * "row sum == 1e4 or 1e6" names the normalisation but is not robust —
    normalise to CP10k and THEN drop genes and row sums land nowhere near 1e4
    (measured on GBM: median 3,378, 0.0% of cells within tolerance).

  * "non-integral AND max < 15 => log-transformed" keys on magnitude, and a
    deep cell's CP10k values are all small (100k-UMI cell, 100-count gene ->
    CP10k = 10), so genuinely scaled data can look logged.

We reject per-cell RESCALING too, even though it is provably harmless under
the default recipe (a 38%-TPM GBM reference gave a BIT-IDENTICAL panel,
because CP10k is invariant to it). The invariance belongs to `xgt1` +
`n_genes`, not to the file — `count1`/`count2` and `total_counts` do read the
raw magnitude. `--allow-non-integral-counts` is the deliberate escape hatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracer.panel_builder import _classify_counts, _report_counts_verdict


def counts_matrix(n=200, g=50, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.random(n, g, density=0.3, format="csr", random_state=seed)
    X.data = np.floor(rng.gamma(2.0, 20.0, size=X.data.size)) + 1.0
    return X


def test_integer_counts_pass():
    assert _classify_counts(counts_matrix())["status"] == "ok"


def test_log_transformed_subset_is_rejected():
    X = counts_matrix().tocsr()
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    X.data[rows < 40] = np.log1p(X.data[rows < 40])
    r = _classify_counts(X)
    assert r["status"] == "non_integral" and r["n_non_integral"] == 40


def test_per_cell_rescaling_is_rejected_too():
    """Harmless under xgt1+n_genes, but the file is still not counts."""
    X = counts_matrix()
    scale = np.ones(X.shape[0]); scale[:60] = 1e6 / np.asarray(X.sum(1)).ravel()[:60]
    r = _classify_counts((sp.diags(scale) @ X).tocsr())
    assert r["status"] == "non_integral"


def test_cp10k_then_gene_filtered_is_caught():
    """The case the row-sum rule misses entirely."""
    X = counts_matrix()
    lib = np.asarray(X.sum(1)).ravel()
    cp = (sp.diags(1e4 / lib) @ X).tocsr()[:, ::3].tocsr()
    assert _classify_counts(cp)["status"] == "non_integral"


def test_offending_batch_is_named():
    X = counts_matrix().tocsr()
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    X.data[rows < 40] = np.log1p(X.data[rows < 40])
    obs = pd.DataFrame({"batch": ["bad"] * 40 + ["good"] * (X.shape[0] - 40)})
    assert "bad" in (_classify_counts(X, obs=obs)["offending"] or "")


def test_report_raises_on_non_integral_and_names_the_escape_hatch():
    X = counts_matrix().tocsr()
    X.data = X.data + 0.5
    with pytest.raises(SystemExit, match="allow-non-integral-counts"):
        _report_counts_verdict(_classify_counts(X))


def test_escape_hatch_downgrades_to_a_warning():
    X = counts_matrix().tocsr()
    X.data = X.data + 0.5
    with pytest.warns(RuntimeWarning):
        _report_counts_verdict(_classify_counts(X), allow_non_integral=True)


def test_clean_matrix_reports_nothing():
    _report_counts_verdict(_classify_counts(counts_matrix()))   # must not raise
