"""Build PMI/cPMI reference panels from a single-cell reference.

The estimator lives next door in :mod:`tracer.conflict_reference`
(``build_depth_corrected_reference``), which emits the naive and depth-corrected
values in one pass so a PMI-vs-cPMI comparison differs in exactly one term:

    PMI  = log(p_ij / (p_i * p_j))                    marginal-product null
    cPMI = log((O + eps) / (E + eps)),  E = sum_d n_d * r_id * r_jd

THIS module holds everything around that: how the reference is drawn, how
presence is called, how the counts matrix is validated, and the on-disk panel
contract. The CLI is ``scripts/build_panels.py`` — the same split as
``scripts/build_npmi_from_scrna.py`` over ``tracer.metrics.compute_pmi_bootstrap``.

DRAW STRATEGIES
---------------
rep      Cell-type balanced WITH REPLACEMENT to target = round(mean type size).
         Reference size then matches the unbalanced panel's scale.
         GUARD: `min_det_cells` counts ROWS, so replication alone would let a
         gene detected in 3 unique cells clear a 25-cell floor. The gene filter
         is therefore applied to the UNIQUE cells of the draw and the builder is
         called with min_det_cells=1 on the already-filtered gene set.

capped   Cell-type balanced WITHOUT replacement, every type capped at the rarest.
         Superseded by `rep`: on lung this collapsed a 50k atlas to 5,814 cells
         (median co-detection O of 9-20 vs 86 unbalanced), so a balanced-vs-
         unbalanced comparison measured sample size rather than estimator.
         Kept because it reproduces the earlier panels.

grid3/4  LABEL-FREE balancing on a marginal n_genes x depth grid, replacement
         draw. The nuclear recipe — the single largest nuclear improvement
         measured (cPMI 0.4077 -> 0.2848). On scRNA it LOSES to cell-type
         labels (0.2976 vs 0.2361), so it is for references with no labels.

vanilla  All cells, natural composition, no balancing.

PRESENCE ARMS
-------------
A gene is PRESENT in a cell when its value in the arm's space is >= the arm's
threshold (`presence_space`). The same rule feeds gene support, the n_genes
covariate and the estimator.

count1   raw counts >= 1
count2   raw counts >= 2
xgt1     log1p(CP10k) >= 1, CP10k computed with the TRUE library size; i.e.
         count >= (e-1)*lib/1e4. The name reads "> 1": on integer counts the two
         never disagree, because equality needs count/lib == (e-1)/1e4 exactly
         and e-1 is irrational (tests/test_reference_recipe.py checks it in
         floating point too). For any cell below ~5,820 UMIs this is identical
         to count >= 1. NOTE this makes the arm INVARIANT to per-cell
         rescaling, since CP10k is.

Depth is always the TRUE library size over all genes in the reference, computed
before restricting to the panel gene set.

THE RECIPE
----------
`build_reference_panels` runs the whole with-reference recipe -- draw, presence,
distinct-cell gene support, depth covariate, estimator, promotion -- and returns
the panels without writing them. `scripts/build_panels.py` is a thin CLI over it.
Use it rather than `build_depth_corrected_reference` directly: that estimator's
own defaults are count1 + total_counts + no balancing, and its `edges["PMI"]` is
the NAIVE estimator, which is what TRACER would consume if handed `edges` as is.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

_LOG = logging.getLogger(__name__)


def log(m):
    """Timestamped progress line. Kept module-level so the CLI and the library
    emit one consistent stream."""
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


EDGE_COLS = ["gene_i", "gene_j", "PMI", "cPMI", "O", "E", "z"]

ARM_SPECS = {           # name -> (min_count, log1p_cp10k)
    "count1": (1, False),
    "count2": (2, False),
    "xgt1": (1, True),
}

#: Output-name suffix per strategy, preserving the historical panel names.
SUFFIX = {"rep": "_pmi_balanced_rep", "capped": "_pmi_balanced",
          "grid3": "_pmi_grid3", "grid4": "_pmi_grid4"}


def read_panel_genes(path: Path) -> list[str]:
    """Gene universe to restrict the reference to.

    A transcripts parquet (its `feature_name` column) or a text file with one
    gene per line. Sorted, because the sort order fixes the gene indexing and
    therefore the panel's row order.
    """
    if path.suffix == ".parquet":
        g = pd.read_parquet(path, columns=["feature_name"]).feature_name
    else:
        g = pd.Series([l.strip() for l in path.read_text().splitlines() if l.strip()])
    return sorted(g.astype(str).unique())


def load_transcripts_reference(path: Path, *, nucleus_only: bool,
                               panel_genes: list[str] | None):
    """Build a cells x genes reference from a standardized transcript table.

    Self-contained alternative to an external scRNA reference: the panel is
    estimated from the assay's own cells. `nucleus_only` keeps transcripts
    inside a nucleus (`overlaps_nucleus == 1`), the segmentation-independent
    subset — this is the `bootstrap.nuclear_only` path of the xenium preset.

    NOTE the depth here is the cell's own transcript count over the panel
    genes, not a true library size: an in-situ panel has no off-panel genes to
    sum. So `depth` and the count matrix are coupled in a way they are not for
    an scRNA reference, and the depth-binned null is correspondingly weaker.
    """
    cols = ["cell_id", "feature_name"]
    d = pd.read_parquet(path)
    missing = {"cell_id", "feature_name"} - set(d.columns)
    if missing:
        raise SystemExit(f"transcripts table missing {sorted(missing)}")
    n0 = len(d)
    if nucleus_only:
        if "overlaps_nucleus" not in d.columns:
            raise SystemExit("--nucleus-only needs an `overlaps_nucleus` column")
        d = d.loc[pd.to_numeric(d["overlaps_nucleus"], errors="coerce").fillna(0) == 1]
    cid = d["cell_id"].astype(str)
    d = d.loc[~cid.isin({"UNASSIGNED", "-1", "", "nan", "None", "NA"})]
    log(f"transcripts {n0:,} -> {len(d):,} "
        f"({'nuclear + ' if nucleus_only else ''}assigned)")

    gene = d["feature_name"].astype(str)
    if panel_genes is not None:
        d = d.loc[gene.isin(set(panel_genes))]
        gene = d["feature_name"].astype(str)
    cell_cat = pd.Categorical(d["cell_id"].astype(str))
    gene_cat = pd.Categorical(gene)
    cg = pd.DataFrame({"c": cell_cat.codes, "g": gene_cat.codes})
    agg = cg.groupby(["c", "g"], observed=True).size()
    ci = agg.index.get_level_values(0).to_numpy()
    gi = agg.index.get_level_values(1).to_numpy()
    X = sp.csr_matrix((agg.to_numpy(np.float64), (ci, gi)),
                      shape=(len(cell_cat.categories), len(gene_cat.categories)))
    keep = [str(g) for g in gene_cat.categories]
    # Panel-only "library size" — see the note above.
    depth = np.asarray(X.sum(1)).ravel().astype(float)
    log(f"reference {X.shape[0]:,} cells x {len(keep)} genes; "
        f"tx/cell median {np.median(depth):.0f} p25 {np.percentile(depth, 25):.0f}")
    return X, keep, depth, pd.DataFrame(index=pd.Index(cell_cat.categories))


def _classify_counts(X, *, obs=None, tol: float = 1e-6):
    """Are these integer counts? Everything else is rejected.

    ONE RULE: counts are integers. CP10k, TPM, log1p(CP10k) and any mix of
    them are non-integral, so integrality alone separates the good case from
    every bad one.

    Two alternatives I tried and dropped:

    * "row sum == 1e4 or 1e6" names the normalisation but is not robust:
      normalise to CP10k and THEN drop genes -- an ordinary preprocessing
      order -- and row sums land nowhere near 1e4 (measured: median 3,378,
      0.0% of cells within tolerance). Integrality still catches that file.

    * "non-integral AND max < 15 => log-transformed" keys on magnitude, and a
      deep cell's CP10k values are all small (a 100k-UMI cell with a 100-count
      gene gives CP10k = 10), so genuinely scaled data can look logged.

    We reject even a per-cell RESCALING, which is provably harmless under the
    default recipe -- CP10k is invariant to it, and a 38%-TPM GBM reference
    produced a BIT-IDENTICAL panel. That invariance belongs to `xgt1` +
    `n_genes`, not to the file: the same reference changes the panel under
    `count1`/`count2` or `depth_metric=total_counts`. Rejecting once beats
    tolerating it until someone changes the arm. `--allow-non-integral-counts`
    is the escape hatch.
    """
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    n = X.shape[0]
    out = {"n_cells": n, "status": "ok", "n_non_integral": 0, "offending": None}
    if X.data.size == 0:
        return out
    d = X.data.astype(np.float64, copy=False)
    frac = np.abs(d - np.round(d))
    if frac.max() <= tol:
        return out
    rows = np.repeat(np.arange(n), np.diff(X.indptr))
    cell_frac = np.zeros(n)
    np.maximum.at(cell_frac, rows, frac)
    bad = cell_frac > tol
    out.update(status="non_integral", n_non_integral=int(bad.sum()),
               max_deviation=float(frac.max()),
               offending=_name_batches(bad, obs))
    return out


def _name_batches(mask, obs):
    """Which batch are the offending cells concentrated in, if any?"""
    if obs is None:
        return None
    for col in ("Project", "orig.ident", "batch", "sample", "sample_id"):
        if col in getattr(obs, "columns", ()):
            v = obs[col].astype(str).to_numpy()
            hits = sorted(k for k in np.unique(v) if mask[v == k].mean() > 0.5)
            if hits:
                return f"{col}: {', '.join(hits[:6])}"
    return None


def _report_counts_verdict(v, *, allow_non_integral: bool = False) -> None:
    """Accept integer counts; otherwise stop, unless explicitly overridden."""
    if v["status"] == "ok":
        log(f"counts check PASSED ({v['n_cells']:,} cells, all integral)")
        return
    where = f" Concentrated in {v['offending']}." if v.get("offending") else ""
    msg = (f"counts check FAILED: {v['n_non_integral']:,}/{v['n_cells']:,} cells "
           f"hold NON-INTEGRAL values (max deviation "
           f"{v.get('max_deviation', 0):.3f}); this is not a counts matrix."
           f"{where}")
    if allow_non_integral:
        import warnings
        warnings.warn(msg + " Proceeding under --allow-non-integral-counts.",
                      RuntimeWarning, stacklevel=2)
        log("WARNING: " + msg + " Proceeding (--allow-non-integral-counts).")
        return
    raise SystemExit(
        msg + "\n  Fix one of:\n"
        "    * point --layer at a real counts layer\n"
        "    * --x-is-log1p-cp10k --depth-obs COL  (recover counts from X)\n"
        "    * --exclude-obs COL=VALUE             (drop the offending batch)\n"
        "    * --allow-non-integral-counts         (proceed anyway; safe ONLY "
        "for a per-cell RESCALING under xgt1+n_genes, NEVER for log1p)")


def _check_count_floor(X, raw, obs, tol: float = 1e-3):
    """Warn when a reconstruction does not land on integer counts.

    For true UMI data the smallest STORED value per cell must be exactly 1
    (zeros are structurally absent from a sparse matrix, so the floor is the
    smallest observed count). A floor elsewhere means X was normalised against
    a different library than the depth column, or the underlying values are not
    counts at all. On the PDAC atlas this isolated project CA001063 — whose
    values are continuous (TPM-like), so no rescaling recovers counts — while
    the other five projects reconstructed exactly.
    """
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    mn = np.full(X.shape[0], np.inf)
    np.minimum.at(mn, rows, X.data)
    bad = np.abs(mn - 1.0) > tol
    dev = float(np.abs(raw - np.round(raw)).mean())
    if not bad.any():
        log(f"count-floor check PASSED (all cells min=1, mean integrality "
            f"deviation {dev:.2e})")
        return
    import warnings
    msg = (f"count-floor check FAILED for {int(bad.sum()):,}/{len(bad):,} cells "
           f"(their smallest reconstructed value is not 1; median "
           f"{np.median(mn[bad]):.3f}). Their presence calls are NOT comparable "
           f"to the rest.")
    for col in ("Project", "orig.ident", "batch", "sample"):
        if col in obs:
            v = obs[col].astype(str).to_numpy()
            frac = {k: float(bad[v == k].mean()) for k in np.unique(v)}
            hits = sorted((k for k, f in frac.items() if f > 0.5))
            if hits:
                msg += (f" Concentrated in {col}: {', '.join(hits)} — consider "
                        f"--exclude-obs {col}={hits[0]}")
                break
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    log("WARNING: " + msg)


def load_reference(h5ad: Path, panel_genes: list[str], *,
                   depth_obs: str | None = None, x_is_log1p_cp10k: bool = False,
                   exclude_obs: list[str] | None = None,
                   allow_non_integral: bool = False,
                   verdict_out: dict | None = None,
                   info_out: dict | None = None):
    """Return (counts_over_panel_genes, kept_gene_names, true_depth, obs).

    ``x_is_log1p_cp10k`` recovers integer counts from an atlas that ships
    normalised values in ``X`` (as the PDAC atlas does), inverting
    ``log1p(count / lib * 1e4)`` with the true library size from ``depth_obs``.
    Without it such an atlas is silently treated as counts and the `xgt1` arm
    log-transforms already-logged data.

    ``depth_obs`` names an obs column holding the true library size. Prefer it
    whenever ``X`` has been gene-filtered: summing the retained genes
    understates depth, and depth is exactly what the cPMI null conditions on.

    ``info_out``, if given, receives where the matrix and the depth came from
    and the panel-gene overlap, for the receipt.
    """
    A = ad.read_h5ad(h5ad)
    for spec in (exclude_obs or []):
        if "=" not in spec:
            raise SystemExit(f"--exclude-obs wants COL=VALUE, got {spec!r}")
        col, val = spec.split("=", 1)
        if col not in A.obs:
            raise SystemExit(f"--exclude-obs column {col!r} not in obs")
        m = A.obs[col].astype(str).to_numpy() != val
        log(f"--exclude-obs {col}={val}: dropping {int((~m).sum()):,} cells")
        A = A[m].copy()
    vn = np.array([str(g) for g in A.var_names])
    keep = [g for g in panel_genes if g in set(vn)]
    if x_is_log1p_cp10k:
        # The flag describes X. A `counts` layer, if present, is not what was
        # declared to be log1p(CP10k), so it must not be inverted.
        X, matrix_source = A.X, "X (log1p CP10k, inverted with depth_obs)"
        if "counts" in A.layers:
            log("NOTE --x-is-log1p-cp10k: reading X; the `counts` layer is ignored")
    elif "counts" in A.layers:
        X, matrix_source = A.layers["counts"], "layers['counts']"
    else:
        X, matrix_source = A.X, "X"
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    if depth_obs is not None:
        if depth_obs not in A.obs:
            raise SystemExit(f"--depth-obs {depth_obs!r} not in obs; "
                             f"have {sorted(A.obs.columns)[:12]}...")
        depth = A.obs[depth_obs].to_numpy(dtype=float)
        depth_source = f"obs[{depth_obs!r}]"
    else:
        # Depth over ALL genes — the true library size, not the panel subset.
        depth = np.asarray(X.sum(1)).ravel().astype(float)
        depth_source = f"row sums of {matrix_source} over all {X.shape[1]:,} genes"
    if x_is_log1p_cp10k:
        X = X.astype(np.float64).copy()
        X.data = np.expm1(X.data)                       # -> CP10k
        X = (sp.diags(depth / 1e4) @ X).tocsr()         # -> counts
        raw = X.data.copy()
        X.data = np.round(X.data)
        X.eliminate_zeros()
        log(f"recovered counts from log1p(CP10k): min {X.data.min():.0f} "
            f"max {X.data.max():,.0f}, {X.nnz:,} nonzero")
        _check_count_floor(X, raw, A.obs)
    # Sanity-check whatever we are about to treat as counts — on the FULL
    # matrix, before restricting to panel genes, because a subset's maximum is
    # lower than the whole cell's and the log-vs-continuous split keys on it.
    _counts_verdict = _classify_counts(X, obs=A.obs)
    _report_counts_verdict(_counts_verdict, allow_non_integral=allow_non_integral)
    if verdict_out is not None:
        verdict_out.update(_counts_verdict)

    col = {g: i for i, g in enumerate(vn)}
    sub = X[:, [col[g] for g in keep]].tocsr()
    log(f"reference {A.n_obs:,} cells; panel {len(panel_genes)} genes, "
        f"{len(keep)} present ({len(panel_genes) - len(keep)} missing)")
    if info_out is not None:
        info_out.update(
            path=str(h5ad), n_cells=int(A.n_obs), n_genes_total=int(X.shape[1]),
            matrix_source=matrix_source, depth_source=depth_source,
            depth_computed_before_panel_restriction=True,
            n_panel_genes=len(panel_genes), n_panel_genes_present=len(keep),
            exclude_obs=list(exclude_obs or []))
    return sub, keep, depth, A.obs


def balanced_draw(celltypes: np.ndarray, *, strategy: str, seed: int) -> np.ndarray:
    """Row indices of the reference draw. RNG is consumed in value_counts order."""
    vc = pd.Series(celltypes).value_counts()
    rng = np.random.default_rng(seed)
    if strategy == "capped":
        per = int(vc.min())
        log(f"{len(vc)} types, sizes {dict(vc)}")
        log(f"capped draw: {per:,} cells x {len(vc)} types = {per * len(vc):,}")
        return np.concatenate([
            rng.choice(np.flatnonzero(celltypes == t), per, replace=False)
            for t in vc.index])

    target = int(round(vc.mean()))
    log(f"{len(vc)} types; target {target:,}/type -> {target * len(vc):,} cells")
    parts = []
    for t in vc.index:
        idx = np.flatnonzero(celltypes == t)
        parts.append(rng.choice(idx, target, replace=len(idx) < target))
        log(f"   {str(t):16s} n={len(idx):>6,}  replication {target / len(idx):>5.2f}x")
    ref = np.concatenate(parts)
    uniq = np.unique(ref)
    log(f"draw: {len(ref):,} rows, {len(uniq):,} unique ({len(uniq) / len(ref):.1%})")
    return ref


def grid_draw(sub, depth, *, k: int, seed: int) -> np.ndarray:
    """Label-free stratified draw on an n_genes x depth grid, WITH replacement.

    Uses MARGINAL quantile edges on each axis, so the joint cells are unequal —
    that is the whole point. `nested` re-quantiles depth *within* each n_genes
    bin, which forces the strata back to equal size and makes an equal draw
    identical to uniform sampling (the qcut no-op: measured +0.8 pairs, p=0.54
    over 5 seeds, vs +3.0, p=0.005 for this grid).

    Drawn to a common target with replacement rather than capped at the
    smallest cell: capping is what collapsed the first lung reference to 5,814
    cells and turned a balance comparison into a sample-size comparison.
    """
    ngenes = np.asarray((sub > 0).sum(1)).ravel().astype(float)
    qs = np.linspace(0, 1, k + 1)[1:-1]
    gi = np.digitize(ngenes, np.quantile(ngenes, qs))
    di = np.digitize(depth, np.quantile(depth, qs))
    key = gi * k + di
    occupied = [c for c in np.unique(key)]
    sizes = np.array([int((key == c).sum()) for c in occupied])
    target = int(round(sizes.mean()))
    log(f"grid{k}: {len(occupied)}/{k*k} cells occupied, sizes {sizes.min():,}"
        f"-{sizes.max():,} (ratio {sizes.max()/max(sizes.min(),1):.1f}x); "
        f"target {target:,}/cell -> {target*len(occupied):,} rows")
    rng = np.random.default_rng(seed)
    parts = []
    for c in occupied:
        idx = np.flatnonzero(key == c)
        parts.append(rng.choice(idx, target, replace=len(idx) < target))
    ref = np.concatenate(parts)
    uniq = np.unique(ref)
    log(f"draw: {len(ref):,} rows, {len(uniq):,} unique ({len(uniq)/len(ref):.1%})")
    return ref


def cp10k_log1p(sub, depth):
    """log1p(counts per 10k), the `xgt1` presence space."""
    CP = (sp.diags(1e4 / np.maximum(depth, 1)) @ sub.astype(np.float64)).tocsr()
    L = CP.copy()
    L.data = np.log1p(L.data)
    return L


#: Human-readable presence rule per arm, for logs and receipts.
PRESENCE_RULE = {
    "count1": "count >= 1",
    "count2": "count >= 2",
    "xgt1": "log1p(count * 1e4 / library_size) >= 1, i.e. count >= (e-1) * library_size / 1e4",
}


def presence_space(sub, depth, arm: str):
    """``(values, threshold)`` for an arm: present iff ``values >= threshold``.

    ``depth`` must be the TRUE library size (all genes), which is what makes
    CP10k mean counts per 10k of the cell rather than of the panel.
    """
    mc, use_log = ARM_SPECS[arm]
    if use_log:
        return cp10k_log1p(sub, depth), 1
    return sub, mc


def panel_frame(edges, *, promote: str | None = None,
                min_abs: float | None = None, name: str = "panel") -> pd.DataFrame:
    """The panel exactly as `write_panel` writes it (see there)."""
    if promote is None:
        df = edges[EDGE_COLS]
    else:
        df = (edges[["gene_i", "gene_j", promote, "O", "E", "z"]]
              .rename(columns={promote: "PMI"}))
    if min_abs is not None and min_abs > 0:
        t = float(min_abs)
        # Conjunctive: keep the pair if EITHER estimator clears the threshold,
        # so both emitted panels share one row set.
        informative = ((edges["PMI"].abs() > t) | (edges["cPMI"].abs() > t)).to_numpy()
        n0 = len(df)
        df = df.loc[informative].reset_index(drop=True)
        log(f"{name}: dropped {n0 - len(df):,}/{n0:,} edges "
            f"({(n0 - len(df)) / max(n0, 1):.1%}) with |PMI|<={t} AND |cPMI|<={t}")
    return df


def write_panel(edges, out: Path, name: str, *, promote: str | None = None,
                min_abs: float | None = None):
    """Write one panel. `promote` copies that estimator into the `PMI` slot.

    `min_abs` drops the "limbo band" — pairs too weak to assert either
    association or mutual exclusion. TRACER reads `PMI_THR = 0.2` as an effect
    size (O/E >= e^0.2, i.e. 22% above chance), so truncating at the same value
    leaves exactly the pairs that can influence a decision.

    A pair is dropped only when BOTH estimators are uninformative
    (|PMI| <= min_abs AND |cPMI| <= min_abs). Cutting on the promoted metric
    alone would discard pairs the other estimator still calls real, and would
    leave the PMI and cPMI panels with different row sets — which breaks the
    one-term-difference property that makes them comparable.

    NOT score-neutral: measured +1,057 cells and -32,153 unassigned tx on the
    lung ROI. Removing near-zero values raises the mean and 25th-percentile
    aggregates that Rescue/Stitch admission gates read, so candidates that
    previously failed now pass. Coherence is unaffected (its denominator
    already excludes |w| <= rst), but the aggregators are not.
    """
    _write_frame(panel_frame(edges, promote=promote, min_abs=min_abs, name=name),
                 out, name)


def _write_frame(df, out: Path, name: str) -> Path:
    """Write one panel frame and log its shape and sign balance."""
    p = out / f"{name}.csv.gz"
    df.to_csv(p, index=False)
    v = df["PMI"]
    gs = set(df.gene_i) | set(df.gene_j)
    log(f"{name:26s} {len(df):>7,} edges  {len(gs):>3} genes  "
        f"O med {np.median(df.O):>6.0f}  "
        f"pos {np.mean(v > 0.2):.0%} neg {np.mean(v < -0.2):.0%} -> {p.name}")
    return p


# --------------------------------------------------------------------------- #
# The recipe: draw -> presence -> distinct-cell support -> covariate ->
# estimator -> promotion. Library entry point; the CLI is a thin shell.
# --------------------------------------------------------------------------- #
@dataclass
class PanelSet:
    """Everything one recipe run produces, before anything is written.

    ``panels`` maps output name -> frame exactly as ``write_panel`` writes it;
    ``pmi_column`` names the estimator in each frame's ``PMI`` column, the
    only column the pipeline reads; ``tracer_input`` maps arm -> the output
    TRACER should consume under the recipe's ``promote``.
    """

    panels: dict[str, pd.DataFrame]
    pmi_column: dict[str, str]
    tracer_input: dict[str, str]
    edges: dict[str, pd.DataFrame]
    facts: dict = field(default_factory=dict)


def _strata_summary(labels, ref) -> dict:
    """Per-stratum available / drawn / distinct-drawn counts."""
    lab = pd.Series(np.asarray(labels).astype(str))
    avail = lab.value_counts()
    drawn = lab.iloc[ref].value_counts()
    uniq = lab.iloc[np.unique(ref)].value_counts()
    return {str(k): {"available": int(avail[k]), "drawn_rows": int(drawn.get(k, 0)),
                     "distinct_cells": int(uniq.get(k, 0))} for k in avail.index}


def build_reference_panels(sub, genes, depth, obs=None, *, cfg=None,
                           celltype_col: str | None = None,
                           emit_cpmi: bool | None = None,
                           prefix: str = "") -> PanelSet:
    """Run the reference-panel recipe in ``cfg`` (default: `PanelConfig()`).

    Parameters
    ----------
    sub : cells x panel-genes RAW counts (CSR), e.g. from `load_reference`.
    genes : panel gene names matching ``sub``'s columns.
    depth : TRUE per-cell library size over ALL genes, established before the
        restriction to panel genes. It is the CP10k denominator; a panel-only
        sum would understate it.
    obs : cell metadata; needs ``celltype_col`` for strategy rep/capped.
    cfg : `tracer.config.PanelConfig`. The default is the with-reference
        recipe ``xgt1_cpmi_balanced_rep``: cell-type balanced WITH replacement,
        xgt1 presence, cPMI promoted, depth binned on n_genes.
    emit_cpmi : also produce the cPMI-promoted panel (non-vanilla strategies).
        Defaults to ``cfg.promote == "cPMI"``.

    Three things are kept apart: CP10k normalisation (presence only, xgt1),
    cell-type balancing (which rows enter the estimator), and the
    depth-conditioned expectation (the estimator's null, binned on the
    covariate). Gene support is counted on DISTINCT cells, so replicated rows
    cannot carry a gene over ``min_det_cells``.
    """
    from tracer.config import PanelConfig
    from tracer.conflict_reference import build_depth_corrected_reference

    cfg = cfg if cfg is not None else PanelConfig()
    if emit_cpmi is None:
        emit_cpmi = cfg.promote == "cPMI"
    genes = list(genes)
    depth = np.asarray(depth, dtype=float)
    sub = sub.tocsr() if sp.issparse(sub) else sp.csr_matrix(sub)
    if sub.shape != (len(depth), len(genes)):
        raise ValueError(f"counts {sub.shape} do not match {len(depth)} depths "
                         f"x {len(genes)} genes")
    strategy = cfg.strategy
    if strategy in ("rep", "capped") and (obs is None or celltype_col is None
                                          or celltype_col not in obs):
        raise ValueError(f"strategy {strategy!r} balances on cell types: pass "
                         f"obs and a celltype_col present in it")

    panels: dict[str, pd.DataFrame] = {}
    pmi_column: dict[str, str] = {}
    tracer_input: dict[str, str] = {}
    edges_by_arm: dict[str, pd.DataFrame] = {}
    facts: dict = {"strategy": strategy, "seed": int(cfg.seed), "arms": {}}

    if strategy == "vanilla":
        ref = np.arange(sub.shape[0])
        facts["draw"] = {"n_rows": int(sub.shape[0]),
                         "n_distinct_cells": int(sub.shape[0]),
                         "replacement": False}
        log(f"all {sub.shape[0]:,} cells, natural composition, {len(genes)} genes")
    else:
        if strategy.startswith("grid"):
            ref = grid_draw(sub, depth, k=int(strategy[-1]), seed=cfg.seed)
            labels = None
        else:
            labels = obs[celltype_col].astype(str).to_numpy()
            ref = balanced_draw(labels, strategy=strategy, seed=cfg.seed)
        facts["draw"] = {
            "n_rows": int(len(ref)), "n_distinct_cells": int(len(np.unique(ref))),
            "replacement": strategy != "capped",
            "balanced_on": celltype_col if labels is not None else
                           f"grid{strategy[-1]} (n_genes x depth, label-free)",
        }
        if labels is not None:
            facts["draw"]["strata"] = _strata_summary(labels, ref)
    uniq = np.unique(ref)

    for arm in cfg.arms:
        mc, use_log = ARM_SPECS[arm]
        src, thr = presence_space(sub, depth, arm)
        arm_facts = {"presence_rule": PRESENCE_RULE[arm]}

        if strategy == "vanilla":
            counts, gene_names, min_det = src, genes, cfg.min_det_cells
            arm_facts["gene_support"] = (
                f"estimator's min_det_cells={cfg.min_det_cells} over all cells")
        elif strategy != "capped":
            # Gene admission is decided on DISTINCT cells, then the builder is
            # called with min_det_cells=1 so replication cannot manufacture
            # support for a gene it would not otherwise clear.
            det = np.asarray((src[uniq] >= thr).sum(0)).ravel()
            ok = det >= cfg.min_det_cells
            gene_names = [g for g, k in zip(genes, ok) if k]
            log(f"{arm}: {ok.sum()}/{len(genes)} genes clear "
                f"min_det_cells={cfg.min_det_cells} on {len(uniq):,} unique cells")
            counts, min_det = src[ref][:, ok], 1
            arm_facts["gene_support"] = (
                f"min_det_cells={cfg.min_det_cells} counted on "
                f"{len(uniq):,} distinct drawn cells")
        else:
            counts, gene_names, min_det = src[ref], genes, cfg.min_det_cells
            arm_facts["gene_support"] = (
                f"estimator's min_det_cells={cfg.min_det_cells} on the drawn "
                f"rows (no replacement, so rows are distinct cells)")

        # `depth_metric` in the builder only fires when `depth` is None, and we
        # always pass depth explicitly -- so the covariate has to be built here
        # or the flag is silently inert. n_genes = transcriptome complexity
        # under THIS arm's presence rule; total_counts = true library size.
        # After Xgt1 (a depth-RELATIVE presence threshold) the library-size
        # null is largely redundant, which is what makes n_genes worth having.
        if cfg.depth_metric == "n_genes":
            cov = np.asarray((src >= thr).sum(1)).ravel().astype(float)
            arm_facts["depth_covariate"] = (
                f"n_genes: panel genes present by '{PRESENCE_RULE[arm]}', per "
                f"cell, over all {len(genes)} panel genes in the reference")
        else:
            cov = depth
            arm_facts["depth_covariate"] = "total_counts: true library size"
        res = build_depth_corrected_reference(
            counts=counts, genes=np.asarray(gene_names, dtype=object),
            depth=cov if strategy == "vanilla" else cov[ref],
            min_count=1 if use_log else mc, min_det_cells=min_det,
            n_depth_bins=cfg.n_depth_bins, depth_metric=cfg.depth_metric,
            top_k_per_gene=cfg.top_k_per_gene)
        edges_by_arm[arm] = res.edges
        arm_facts["n_genes_supported"] = len(gene_names)
        arm_facts["estimator_meta"] = res.meta

        if strategy == "vanilla":
            # Each estimator gets its own file, promoted into `PMI`. The
            # count1 arm keeps the legacy `vanilla_spec_*` names so existing
            # panels reproduce; other arms are prefixed.
            tag = "" if arm == "count1" else f"{arm}_"
            for name, est in (("vanilla_spec_cpmi", "cPMI"), ("vanilla_spec_pmi", "PMI")):
                n = prefix + tag + name
                panels[n] = panel_frame(res.edges, promote=est,
                                        min_abs=cfg.min_abs_value, name=n)
                pmi_column[n] = est
            tracer_input[arm] = prefix + tag + (
                "vanilla_spec_cpmi" if cfg.promote == "cPMI" else "vanilla_spec_pmi")
        else:
            n = prefix + arm + SUFFIX[strategy]
            panels[n] = panel_frame(res.edges, min_abs=cfg.min_abs_value, name=n)
            pmi_column[n] = "PMI"          # naive; cPMI rides along unread
            tracer_input[arm] = n
            if emit_cpmi:
                nc = prefix + arm + SUFFIX[strategy].replace("_pmi_", "_cpmi_")
                panels[nc] = panel_frame(res.edges, promote="cPMI",
                                         min_abs=cfg.min_abs_value, name=nc)
                pmi_column[nc] = "cPMI"
                if cfg.promote == "cPMI":
                    tracer_input[arm] = nc
        facts["arms"][arm] = arm_facts

    return PanelSet(panels=panels, pmi_column=pmi_column,
                    tracer_input=tracer_input, edges=edges_by_arm, facts=facts)
