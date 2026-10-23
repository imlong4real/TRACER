"""The canonical reference-panel recipe, end to end.

``xgt1_cpmi_balanced_rep`` is the default for an scRNA reference:

    counts (true library size over ALL genes)
      -> cell-type balanced draw WITH replacement            (balancing)
      -> presence: log1p(CP10k) >= 1                          (normalisation)
      -> gene support counted on DISTINCT cells
      -> depth covariate n_genes, quantile strata             (expectation)
      -> O, E, cPMI = log((O+1)/(E+1))
      -> cPMI promoted into the PMI column TRACER reads

These tests pin each step against an independent recomputation, through the
library entry point and the CLI, and follow the promoted values into the
pipeline's own weight matrix.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from tracer.config import PanelConfig, load_config
from tracer.conflict_reference import _depth_bins
from tracer.panel_builder import (
    PRESENCE_RULE, balanced_draw, build_reference_panels, load_reference,
    presence_space,
)

ROOT = Path(__file__).resolve().parents[1]
TYPES = {"T": 300, "B": 180, "Mac": 60, "Fib": 40}      # unequal on purpose
N_PANEL, N_OFF = 30, 200


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        f"_{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _reference(seed=0):
    """Counts for 4 cell types over 30 panel + 200 off-panel genes.

    Libraries span ~1k-40k UMIs, so the xgt1 rule differs from count >= 1
    for the deeper cells.
    """
    rng = np.random.default_rng(seed)
    labels = np.concatenate([[t] * n for t, n in TYPES.items()])
    rng.shuffle(labels)
    n, g = len(labels), N_PANEL + N_OFF
    base = rng.gamma(0.6, 1.0, size=g)
    prog = {t: rng.choice(N_PANEL, 8, replace=False) for t in TYPES}
    lib = rng.lognormal(np.log(6000), 0.9, size=n)
    rows = []
    for c in range(n):
        mu = base.copy()
        mu[prog[labels[c]]] *= 25.0
        rows.append(rng.poisson(lib[c] * mu / mu.sum()))
    X = sp.csr_matrix(np.asarray(rows, dtype=np.float64))
    panel = [f"P{i:02d}" for i in range(N_PANEL)]
    genes = panel + [f"O{i:03d}" for i in range(N_OFF)]
    obs = pd.DataFrame({"cell_type": labels}, index=[f"c{i}" for i in range(n)])
    return X, genes, panel, obs


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    X, genes, panel, obs = _reference()
    d = tmp_path_factory.mktemp("ref")
    A = ad.AnnData(X=X.copy(), obs=obs.copy(), var=pd.DataFrame(index=genes))
    A.layers["counts"] = X.copy()
    h5 = d / "ref.h5ad"
    A.write_h5ad(h5)
    (d / "panel_genes.txt").write_text("\n".join(panel) + "\n")
    return {"dir": d, "h5ad": h5, "panel_txt": d / "panel_genes.txt",
            "X": X, "genes": genes, "panel": panel, "obs": obs}


def _sub(ref):
    idx = [ref["genes"].index(g) for g in ref["panel"]]
    return ref["X"][:, idx].tocsr()


def _true_depth(ref):
    return np.asarray(ref["X"].sum(1)).ravel()


# --------------------------------------------------------------------------
# the recipe, recomputed independently
# --------------------------------------------------------------------------
def _independent_cpmi(sub, panel, depth, labels, *, seed=0, min_det=25, bins=25):
    """First-principles recomputation of xgt1_cpmi_balanced_rep."""
    # balancing: types in value_counts order, mean-size target, replacement
    # only where a type is smaller than the target
    vc = pd.Series(labels).value_counts()
    target = int(round(vc.mean()))
    rng = np.random.default_rng(seed)
    ref = np.concatenate([
        rng.choice(np.flatnonzero(labels == t), target,
                   replace=(labels == t).sum() < target) for t in vc.index])
    # normalisation: CP10k with the TRUE library size; presence >= 1
    dense = sub.toarray()
    present = np.log1p(dense * (1e4 / depth[:, None])) >= 1.0
    # support on distinct cells of the draw
    ok = present[np.unique(ref)].sum(0) >= min_det
    # covariate: n_genes under the same rule, over all panel genes
    cov = present.sum(1).astype(float)
    P = present[ref][:, ok].astype(float)
    b, D = _depth_bins(cov[ref], bins)
    B = np.zeros((len(ref), D)); B[np.arange(len(ref)), b] = 1
    n_d = B.sum(0)
    R = (P.T @ B) / np.maximum(n_d, 1)
    O = P.T @ P
    E = (R * n_d) @ R.T
    names = np.asarray(panel)[ok]
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if O[i, j] >= 2:
                out[(names[i], names[j])] = (
                    O[i, j], np.clip(np.log((O[i, j] + 1) / (E[i, j] + 1)), -4, 4))
    return out, set(names)


def test_default_recipe_matches_an_independent_recomputation(ref):
    sub, depth, labels = _sub(ref), _true_depth(ref), ref["obs"]["cell_type"].to_numpy()
    ps = build_reference_panels(sub, ref["panel"], depth, ref["obs"],
                                celltype_col="cell_type")
    assert ps.tracer_input == {"xgt1": "xgt1_cpmi_balanced_rep"}
    assert ps.pmi_column == {"xgt1_pmi_balanced_rep": "PMI",
                             "xgt1_cpmi_balanced_rep": "cPMI"}
    exp, genes_ok = _independent_cpmi(sub, ref["panel"], depth, labels)
    p = ps.panels["xgt1_cpmi_balanced_rep"]
    # recovered-negative edges (O <= 1) are an estimator extra; compare the
    # observed-pair edges, which must match one for one
    got = {(a, b): (o, v) for a, b, o, v in
           zip(p.gene_i, p.gene_j, p.O, p.PMI) if o >= 2}
    assert set(got) == set(exp)
    for k, (o, v) in exp.items():
        assert got[k][0] == o
        assert got[k][1] == pytest.approx(v, abs=1e-5)
    assert set(p.gene_i) | set(p.gene_j) <= genes_ok
    # the naive companion keeps naive PMI in PMI and cPMI alongside
    naive = ps.panels["xgt1_pmi_balanced_rep"]
    np.testing.assert_allclose(naive["cPMI"].to_numpy(), p["PMI"].to_numpy())
    assert not np.allclose(naive["PMI"].to_numpy(), p["PMI"].to_numpy())


def test_recipe_defaults_are_effective_not_just_declared(ref):
    """What PanelConfig() says is what build_reference_panels does."""
    c = load_config().panel
    assert (c.strategy, tuple(c.arms), c.promote, c.depth_metric) == \
           ("rep", ("xgt1",), "cPMI", "n_genes")
    ps = build_reference_panels(_sub(ref), ref["panel"], _true_depth(ref),
                                ref["obs"], cfg=c, celltype_col="cell_type")
    f = ps.facts
    assert f["strategy"] == "rep" and f["draw"]["replacement"] is True
    assert f["arms"]["xgt1"]["presence_rule"] == PRESENCE_RULE["xgt1"]
    assert "distinct" in f["arms"]["xgt1"]["gene_support"]
    assert f["arms"]["xgt1"]["depth_covariate"].startswith("n_genes")
    assert f["arms"]["xgt1"]["estimator_meta"]["depth_metric"] == "n_genes"


# --------------------------------------------------------------------------
# presence threshold
# --------------------------------------------------------------------------
def test_xgt1_rule_is_inclusive_and_never_meets_its_boundary():
    """log1p(CP10k) >= 1 and > 1 select the same cells on integer counts.

    Equality needs count/lib == (e-1)/1e4 exactly; e-1 is irrational, and in
    floating point the nearest integer counts stay far from the boundary for
    every library size up to 2M UMIs.
    """
    lib = np.arange(1, 2_000_001, dtype=np.float64)
    c_star = np.ceil((np.e - 1.0) * lib / 1e4)
    for c in (c_star - 1, c_star):           # the counts either side of it
        m = c >= 1
        x = np.log1p((1e4 / lib[m]) * c[m])  # the cp10k_log1p arithmetic
        assert not np.any(x == 1.0)
        np.testing.assert_array_equal(x >= 1.0, x > 1.0)
        np.testing.assert_array_equal(x >= 1.0,
                                      c[m] * 1e4 >= (np.e - 1.0) * lib[m])
    vals, thr = presence_space(sp.csr_matrix([[1.0, 2.0]]), np.array([3.0]), "xgt1")
    assert thr == 1


def test_presence_uses_the_true_library_size(ref):
    """Off-panel genes raise the CP10k denominator, so a count-1 panel gene
    in a deep cell is absent under xgt1; a panel-only sum would call it
    present."""
    sub, depth = _sub(ref), _true_depth(ref)
    panel_only = np.asarray(sub.sum(1)).ravel()
    assert (depth > panel_only).all()
    v_true, t = presence_space(sub, depth, "xgt1")
    v_pan, _ = presence_space(sub, panel_only, "xgt1")
    assert (v_true >= t).sum() < (v_pan >= t).sum()
    deep = np.flatnonzero(depth > 1e4 / (np.e - 1.0))
    assert deep.size
    ones = (sub[deep] == 1).toarray()
    assert ones.any() and not (v_true[deep].toarray()[ones] >= 1).any()


def test_load_reference_takes_depth_over_all_genes_before_restricting(ref):
    info = {}
    sub, keep, depth, _ = load_reference(ref["h5ad"], ref["panel"], info_out=info)
    np.testing.assert_allclose(depth, _true_depth(ref))
    assert keep == ref["panel"] and sub.shape == (len(depth), N_PANEL)
    assert info["depth_computed_before_panel_restriction"] is True
    assert "all 230 genes" in info["depth_source"]
    assert info["matrix_source"] == "layers['counts']"


def test_depth_obs_overrides_the_row_sum(ref, tmp_path):
    A = ad.read_h5ad(ref["h5ad"])
    A.obs["nCount_RNA"] = _true_depth(ref) * 2.0
    p = tmp_path / "r.h5ad"; A.write_h5ad(p)
    info = {}
    _, _, depth, _ = load_reference(p, ref["panel"], depth_obs="nCount_RNA",
                                    info_out=info)
    np.testing.assert_allclose(depth, 2.0 * _true_depth(ref))
    assert info["depth_source"] == "obs['nCount_RNA']"


# --------------------------------------------------------------------------
# normalised / logged input
# --------------------------------------------------------------------------
def _logged_h5ad(ref, path, *, bogus_counts_layer=False):
    depth = _true_depth(ref)
    L = (sp.diags(1e4 / depth) @ ref["X"]).tocsr()
    L.data = np.log1p(L.data)
    A = ad.AnnData(X=L, obs=ref["obs"].assign(nCount_RNA=depth),
                   var=pd.DataFrame(index=ref["genes"]))
    if bogus_counts_layer:
        bogus = L.copy()
        bogus.data = bogus.data * 1.5      # a layer that is NOT what X holds
        A.layers["counts"] = bogus
    A.write_h5ad(path)
    return path


def test_log1p_cp10k_input_recovers_the_counts_and_the_panel(ref, tmp_path):
    p = _logged_h5ad(ref, tmp_path / "logged.h5ad")
    sub, keep, depth, obs = load_reference(p, ref["panel"], depth_obs="nCount_RNA",
                                           x_is_log1p_cp10k=True)
    np.testing.assert_array_equal(sub.toarray(), _sub(ref).toarray())
    a = build_reference_panels(sub, keep, depth, obs, celltype_col="cell_type")
    b = build_reference_panels(_sub(ref), ref["panel"], _true_depth(ref),
                               ref["obs"], celltype_col="cell_type")
    for name in b.panels:
        pd.testing.assert_frame_equal(a.panels[name], b.panels[name])


def test_logged_input_without_the_flag_is_rejected(ref, tmp_path):
    p = _logged_h5ad(ref, tmp_path / "logged.h5ad")
    with pytest.raises(SystemExit, match="NON-INTEGRAL"):
        load_reference(p, ref["panel"])


def test_the_log1p_flag_reads_X_not_a_counts_layer(ref, tmp_path):
    p = _logged_h5ad(ref, tmp_path / "both.h5ad", bogus_counts_layer=True)
    info = {}
    sub, *_ = load_reference(p, ref["panel"], depth_obs="nCount_RNA",
                             x_is_log1p_cp10k=True, info_out=info)
    np.testing.assert_array_equal(sub.toarray(), _sub(ref).toarray())
    assert info["matrix_source"].startswith("X ")


def test_per_cell_rescaling_is_harmless_under_the_default_recipe(ref):
    """CP10k input (allowed explicitly) gives the counts panel: xgt1 + n_genes
    only ever see CP10k, which a per-cell rescaling cannot change."""
    sub, depth = _sub(ref), _true_depth(ref)
    cp = (sp.diags(1e4 / depth) @ ref["X"]).tocsr()
    idx = [ref["genes"].index(g) for g in ref["panel"]]
    a = build_reference_panels(cp[:, idx].tocsr(), ref["panel"],
                               np.asarray(cp.sum(1)).ravel(), ref["obs"],
                               celltype_col="cell_type")
    b = build_reference_panels(sub, ref["panel"], depth, ref["obs"],
                               celltype_col="cell_type")
    pd.testing.assert_frame_equal(a.panels["xgt1_cpmi_balanced_rep"],
                                  b.panels["xgt1_cpmi_balanced_rep"])


# --------------------------------------------------------------------------
# balancing, sampling and support
# --------------------------------------------------------------------------
def test_sampling_is_deterministic_and_seeded(ref):
    sub, depth = _sub(ref), _true_depth(ref)
    run = lambda s: build_reference_panels(
        sub, ref["panel"], depth, ref["obs"], cfg=PanelConfig(seed=s),
        celltype_col="cell_type")
    a, b, c = run(0), run(0), run(1)
    for name in a.panels:
        pd.testing.assert_frame_equal(a.panels[name], b.panels[name])
    assert a.facts["draw"] == b.facts["draw"]
    assert a.facts["draw"]["strata"] != c.facts["draw"]["strata"]
    assert not a.panels["xgt1_cpmi_balanced_rep"].equals(c.panels["xgt1_cpmi_balanced_rep"])
    # the documented RNG order: one draw per type, in value_counts order
    labels = ref["obs"]["cell_type"].to_numpy()
    rng = np.random.default_rng(0)
    vc = pd.Series(labels).value_counts()
    t = int(round(vc.mean()))
    exp = np.concatenate([rng.choice(np.flatnonzero(labels == k), t,
                                     replace=(labels == k).sum() < t) for k in vc.index])
    np.testing.assert_array_equal(balanced_draw(labels, strategy="rep", seed=0), exp)


def test_rep_and_capped_are_different_recipes_with_different_names(ref):
    sub, depth = _sub(ref), _true_depth(ref)
    rep = build_reference_panels(sub, ref["panel"], depth, ref["obs"],
                                 cfg=PanelConfig(strategy="rep"), celltype_col="cell_type")
    cap = build_reference_panels(sub, ref["panel"], depth, ref["obs"],
                                 cfg=PanelConfig(strategy="capped"), celltype_col="cell_type")
    assert rep.tracer_input["xgt1"] == "xgt1_cpmi_balanced_rep"
    assert cap.tracer_input["xgt1"] == "xgt1_cpmi_balanced"
    mean_t = int(round(np.mean(list(TYPES.values()))))
    assert rep.facts["draw"]["n_rows"] == mean_t * len(TYPES)
    assert rep.facts["draw"]["n_distinct_cells"] < rep.facts["draw"]["n_rows"]
    assert cap.facts["draw"]["n_rows"] == min(TYPES.values()) * len(TYPES)
    assert cap.facts["draw"]["n_distinct_cells"] == cap.facts["draw"]["n_rows"]
    assert cap.facts["draw"]["replacement"] is False


def test_replicated_rows_cannot_manufacture_gene_support():
    """A gene seen in 10 distinct Fib cells, replicated to ~150 rows, must not
    clear min_det_cells=25."""
    rng = np.random.default_rng(3)
    labels = np.array(["A"] * 400 + ["Fib"] * 10)
    X = rng.poisson(2.0, size=(len(labels), 6)).astype(float)
    X[:, 5] = 0.0
    X[labels == "Fib", 5] = 3.0                    # Fib-only gene
    sub = sp.csr_matrix(X)
    depth = X.sum(1) + 500.0                       # shallow: xgt1 == count >= 1
    genes = [f"g{i}" for i in range(6)]
    obs = pd.DataFrame({"ct": labels})
    ps = build_reference_panels(sub, genes, depth, obs, celltype_col="ct",
                                cfg=PanelConfig(min_det_cells=25))
    ref = balanced_draw(labels, strategy="rep", seed=0)
    assert (X[ref, 5] > 0).sum() >= 25            # the rows WOULD clear it
    p = ps.panels["xgt1_cpmi_balanced_rep"]
    assert "g5" not in set(p.gene_i) | set(p.gene_j)
    assert ps.facts["arms"]["xgt1"]["n_genes_supported"] == 5


def test_nuclear_recipe_is_unchanged():
    c = load_config(panel_preset="nuclear").panel
    assert (c.strategy, c.promote) == ("grid3", "PMI")
    rng = np.random.default_rng(5)
    sub = sp.csr_matrix(rng.poisson(1.5, size=(500, 12)).astype(float))
    depth = np.asarray(sub.sum(1)).ravel()
    ps = build_reference_panels(sub, [f"g{i}" for i in range(12)], depth, cfg=c)
    assert ps.tracer_input == {"xgt1": "xgt1_pmi_grid3"}
    assert ps.pmi_column == {"xgt1_pmi_grid3": "PMI"}     # interim naive PMI


# --------------------------------------------------------------------------
# CLI, receipt, and downstream consumption
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cli_build(ref, tmp_path_factory):
    out = tmp_path_factory.mktemp("panels")
    bp = _load_script("build_panels")
    argv = ["build_panels.py", "--h5ad", str(ref["h5ad"]),
            "--panel-genes", str(ref["panel_txt"]),
            "--celltype-col", "cell_type", "--out", str(out)]
    old = sys.argv
    sys.argv = argv
    try:
        bp.main()
    finally:
        sys.argv = old
    return out, bp


def test_cli_writes_the_recipe_and_names_the_tracer_input(ref, cli_build):
    out, _ = cli_build
    rec = json.loads((out / "panel_receipt.json").read_text())
    assert rec["resolved_panel_config"]["strategy"] == "rep"
    assert rec["resolved_panel_config"]["arms"] == ["xgt1"]
    assert rec["resolved_panel_config"]["depth_metric"] == "n_genes"
    assert rec["effective_promote"] == "cPMI"
    outs = {o["file"]: o for o in rec["outputs"]}
    assert outs["xgt1_cpmi_balanced_rep.csv.gz"] == {
        "file": "xgt1_cpmi_balanced_rep.csv.gz", "pmi_column": "cPMI", "tracer_input": True}
    assert outs["xgt1_pmi_balanced_rep.csv.gz"]["tracer_input"] is False
    assert rec["reference"]["depth_computed_before_panel_restriction"] is True
    # the CLI output is the library output
    ps = build_reference_panels(_sub(ref), ref["panel"], _true_depth(ref),
                                ref["obs"], celltype_col="cell_type")
    pd.testing.assert_frame_equal(
        pd.read_csv(out / "xgt1_cpmi_balanced_rep.csv.gz"),
        ps.panels["xgt1_cpmi_balanced_rep"], check_dtype=False, atol=1e-6)


def test_receipt_describes_the_latest_run(ref, cli_build, monkeypatch):
    out, bp = cli_build
    monkeypatch.setattr(sys, "argv", [
        "build_panels.py", "--h5ad", str(ref["h5ad"]),
        "--panel-genes", str(ref["panel_txt"]), "--celltype-col", "cell_type",
        "--out", str(out), "--seed", "7"])
    bp.main()
    rec = json.loads((out / "panel_receipt.json").read_text())
    assert rec["resolved_panel_config"]["seed"] == 7
    monkeypatch.setattr(sys, "argv", [
        "build_panels.py", "--h5ad", str(ref["h5ad"]),
        "--panel-genes", str(ref["panel_txt"]), "--celltype-col", "cell_type",
        "--out", str(out)])
    bp.main()                                   # restore seed 0 for later tests


def test_promoted_cpmi_is_what_the_pipeline_reads(cli_build, monkeypatch):
    """Follow the cPMI values from the panel file into the weight matrix the
    pipeline builds, and into the run receipt."""
    out, _ = cli_build
    rt = _load_script("run_tracer")
    log = logging.getLogger("test_reference_recipe")
    panel_path = out / "xgt1_cpmi_balanced_rep.csv.gz"
    panel = rt.load_npmi_panel(panel_path, log)
    prov = rt.panel_provenance(panel_path)
    assert prov["pmi_column"] == "cPMI" and prov["tracer_input"] is True
    assert rt.panel_provenance(out / "xgt1_pmi_balanced_rep.csv.gz")["tracer_input"] is False

    import tracer.pipeline as pl
    from tests.synthetic import make_synthetic_transcripts

    seen = {}
    real = pl.prune_transcripts_fast

    def spy(df, npmi_panel, **kw):
        res = real(df, npmi_panel, **kw)
        seen.update(metric_col=kw.get("metric_col"), aux=res[1])
        return res

    monkeypatch.setattr(pl, "prune_transcripts_fast", spy)
    genes = sorted(set(panel.gene_i) | set(panel.gene_j))[:12]
    df, _ = make_synthetic_transcripts(n_cells=4, voxels_per_cell_mean=40,
                                       tx_per_cell=15, n_genes=12, seed=2)
    df["feature_name"] = df["feature_name"].map(
        dict(zip([f"gene_{i:02d}" for i in range(12)], genes)))
    pl.run_segmented_pipeline(df.drop(columns="is_nuclear"), panel)
    assert seen["metric_col"] == "PMI"
    W, g2i = seen["aux"]["W"], seen["aux"]["gene_to_idx"]
    raw = pd.read_csv(panel_path)
    for a, b, v in zip(raw.gene_i, raw.gene_j, raw.PMI):
        assert W[g2i[a], g2i[b]] == pytest.approx(np.float32(v))
        assert W[g2i[b], g2i[a]] == pytest.approx(np.float32(v))


def test_other_entry_points_accept_a_cpmi_panel(cli_build):
    """noseg loader and the tiled global stitch read PMI like the pipeline."""
    out, _ = cli_build
    from tracer.noseg_pipeline import load_pmi_panel
    from tracer.pipeline_tiled import run_segmented_pipeline_tiled
    from tests.synthetic import make_synthetic_transcripts

    panel = load_pmi_panel(out / "xgt1_cpmi_balanced_rep.csv.gz")
    assert "NPMI" not in panel.columns
    genes = sorted(set(panel.gene_i) | set(panel.gene_j))[:12]
    df, _ = make_synthetic_transcripts(n_cells=6, voxels_per_cell_mean=40,
                                       tx_per_cell=15, n_genes=12, seed=4)
    df["feature_name"] = df["feature_name"].map(
        dict(zip([f"gene_{i:02d}" for i in range(12)], genes)))
    res = run_segmented_pipeline_tiled(df.drop(columns="is_nuclear"), panel,
                                       n_tiles_xy=(2, 1), n_workers=1,
                                       global_stitch=True)
    assert len(res["df_out"]) == len(df)
