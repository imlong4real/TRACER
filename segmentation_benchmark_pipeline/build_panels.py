#!/usr/bin/env python3
"""Build PMI/cPMI reference panels from a single-cell reference.

Replaces the three dataset-specific builders (build_lung_panels_rep.py,
build_lung_panels.py, build_lung_vanilla_spec.py), which were hardcoded to the
lung h5ad, the lung transcripts parquet and Cell_Cluster_level1.

Every strategy calls `build_depth_corrected_reference` once per arm, which emits
the naive and depth-corrected estimators in the same pass, so a PMI-vs-cPMI
comparison differs in exactly one term:

    PMI  = log(p_ij / (p_i * p_j))                    marginal-product null
    cPMI = log((O + eps) / (E + eps)),  E = sum_d n_d * r_id * r_jd

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

vanilla  All cells, natural composition, no balancing.

PRESENCE ARMS (`--arms`)
------------------------
count1   raw counts, min_count=1
count2   raw counts, min_count=2
xgt1     log1p(CP10k) with min_count=1, i.e. count > (e-1)*lib/1e4. For any cell
         below ~5,820 UMIs this is identical to count >= 1.

Depth is always the TRUE library size over all genes in the reference, computed
before restricting to the panel gene set.

EXAMPLES
--------
    python build_panels.py --h5ad lung_cancer_50k.h5ad \
        --panel-genes ../tutorials/lung_cancer/data/lung_cancer_df.parquet \
        --celltype-col Cell_Cluster_level1 --strategy rep --out panels

    python build_panels.py --h5ad lung_cancer_50k.h5ad \
        --panel-genes ../tutorials/lung_cancer/data/lung_cancer_df.parquet \
        --strategy vanilla --out panels
"""
import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

#: Written to every panel. The pipeline resolves its edge weight as
#: `"PMI" if "PMI" in columns else "NPMI"` and cannot be told which estimator
#: that column holds, so the promoted-column variants below matter.
EDGE_COLS = ["gene_i", "gene_j", "PMI", "cPMI", "O", "E", "z"]

ARM_SPECS = {           # name -> (min_count, log1p_cp10k)
    "count1": (1, False),
    "count2": (2, False),
    "xgt1": (1, True),
}

#: Output-name suffix per strategy, preserving the historical panel names.
SUFFIX = {"rep": "_pmi_balanced_rep", "capped": "_pmi_balanced",
          "grid3": "_pmi_grid3", "grid4": "_pmi_grid4"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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
                   exclude_obs: list[str] | None = None):
    """Return (counts_over_panel_genes, kept_gene_names, true_depth, obs).

    ``x_is_log1p_cp10k`` recovers integer counts from an atlas that ships
    normalised values in ``X`` (as the PDAC atlas does), inverting
    ``log1p(count / lib * 1e4)`` with the true library size from ``depth_obs``.
    Without it such an atlas is silently treated as counts and the `xgt1` arm
    log-transforms already-logged data.

    ``depth_obs`` names an obs column holding the true library size. Prefer it
    whenever ``X`` has been gene-filtered: summing the retained genes
    understates depth, and depth is exactly what the cPMI null conditions on.
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
    X = A.layers["counts"] if "counts" in A.layers else A.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    if depth_obs is not None:
        if depth_obs not in A.obs:
            raise SystemExit(f"--depth-obs {depth_obs!r} not in obs; "
                             f"have {sorted(A.obs.columns)[:12]}...")
        depth = A.obs[depth_obs].to_numpy(dtype=float)
    else:
        # Depth over ALL genes — the true library size, not the panel subset.
        depth = np.asarray(X.sum(1)).ravel().astype(float)
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
    col = {g: i for i, g in enumerate(vn)}
    sub = X[:, [col[g] for g in keep]].tocsr()
    log(f"reference {A.n_obs:,} cells; panel {len(panel_genes)} genes, "
        f"{len(keep)} present ({len(panel_genes) - len(keep)} missing)")
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


def _write_receipt(args, pcfg) -> None:
    """Record the resolved recipe beside the panels.

    A bare `.csv.gz` carries no provenance — which is how a panel built with
    an off-by-2x top-k rule ended up mislabelled `topk1000`. This states the
    recipe, which estimator sits in the `PMI` column, and the exact command.
    """
    import json, sys
    from dataclasses import asdict
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec = out / f"{args.prefix}panel_receipt.json"
    if rec.exists():
        return
    rec.write_text(json.dumps({
        "panel_preset": args._panel_preset,
        "effective_promote": args._effective_promote,
        "canonical_panel_substring":
            "_cpmi_" if args._effective_promote == "cPMI" else "_pmi_",
        "resolved_panel_config": asdict(pcfg),
        "effective_args": {k: (str(v) if isinstance(v, Path) else v)
                           for k, v in vars(args).items()
                           if not k.startswith("_")},
        "command": " ".join(sys.argv),
    }, indent=2, default=str) + "\n")
    log(f"receipt -> {rec}")


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
    p = out / f"{name}.csv.gz"
    df.to_csv(p, index=False)
    v = df["PMI"] if promote is not None or min_abs else edges[promote or "PMI"]
    gs = set(df.gene_i) | set(df.gene_j)
    log(f"{name:26s} {len(df):>7,} edges  {len(gs):>3} genes  "
        f"O med {np.median(df.O):>6.0f}  "
        f"pos {np.mean(v > 0.2):.0%} neg {np.mean(v < -0.2):.0%} -> {p.name}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5ad", type=Path,
                     help="Single-cell reference with a `counts` layer.")
    src.add_argument("--transcripts", type=Path,
                     help="Standardized transcript table (cell_id, "
                          "feature_name). Estimates the panel from the assay's "
                          "own cells instead of an external reference.")
    ap.add_argument("--nucleus-only", action="store_true",
                    help="With --transcripts, keep only overlaps_nucleus == 1 "
                         "— the segmentation-independent subset.")
    ap.add_argument("--panel-genes", type=Path, default=None,
                    help="Transcripts parquet (feature_name column) or a "
                         "one-gene-per-line text file. Required with --h5ad; "
                         "with --transcripts defaults to the observed genes.")
    ap.add_argument("--out", type=Path, required=True, help="Output directory.")
    ap.add_argument("--strategy",
                    choices=("rep", "capped", "vanilla", "grid3", "grid4"),
                    default=None,
                    help="Reference draw. rep/capped need --celltype-col; "
                         "grid3/grid4 are LABEL-FREE (n_genes x depth grid, "
                         "marginal edges, replacement draw); vanilla is none.")
    ap.add_argument("--celltype-col", default=None,
                    help="obs column to balance on. Required unless "
                         "--strategy vanilla.")
    ap.add_argument("--arms", nargs="+", default=None,
                    choices=sorted(ARM_SPECS), help="Presence arms to build.")
    ap.add_argument("--panel-preset", default=None,
                    help="Panel RECIPE preset (tracer/configs/panels/*.toml). "
                         "Inferred when omitted: --h5ad -> the with-reference "
                         "recipe (rep + xgt1 + cPMI + n_genes); "
                         "--transcripts --nucleus-only -> 'nuclear' "
                         "(grid3 + naive PMI). Explicit flags always win.")
    ap.add_argument("--panel-config", type=Path, default=None,
                    help="User-override TOML, top of the layer stack.")
    ap.add_argument("--prefix", default="",
                    help="Prepended to every output panel name.")
    ap.add_argument("--emit-cpmi", action="store_true", default=None,
                    help="Also write a `<arm>_cpmi_balanced...` panel per arm, "
                         "the same build with cPMI promoted into the `PMI` "
                         "column so the pipeline consumes the depth-corrected "
                         "estimator. Adds no information — the default panel "
                         "already carries both columns — but drops into a run "
                         "without editing. Implied for --strategy vanilla.")
    ap.add_argument("--no-emit-cpmi", dest="emit_cpmi", action="store_false",
                    help="Suppress the cPMI-promoted panel (it is written by "
                         "default under the with-reference recipe).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min-det-cells", type=int, default=None,
                    help="A gene must be detected in this many cells (default 25).")
    ap.add_argument("--n-depth-bins", type=int, default=None)
    ap.add_argument("--depth-metric", default=None,
                    choices=("total_counts", "n_genes"),
                    help="Covariate the cPMI null is binned on. Inert for the "
                         "naive-PMI column, which never reads E.")
    ap.add_argument("--depth-obs", default=None,
                    help="obs column holding the TRUE library size. Use it when "
                         "X has been gene-filtered (e.g. nCount_RNA).")
    ap.add_argument("--top-k-per-gene", type=int, default=None, metavar="K",
                    help="Builder-native relative cut: keep each gene's K "
                         "strongest partners by |cPMI| (ONE ranking per gene "
                         "over all its edges; an edge survives if in the top-K "
                         "of either endpoint). Pure rank cut — no magnitude "
                         "threshold; combine with --min-abs-value if wanted.")
    ap.add_argument("--min-abs-value", type=float, default=None, metavar="TAU",
                    help="Drop edges with |value| <= TAU (the uninformative "
                         "limbo band). Use 0.2 to match TRACER's PMI_THR, "
                         "making the panel exactly the pairs that can change a "
                         "decision. Applied to each panel's promoted metric.")
    ap.add_argument("--exclude-obs", action="append", metavar="COL=VALUE",
                    help="Drop reference cells where obs[COL] == VALUE. "
                         "Repeatable.")
    ap.add_argument("--x-is-log1p-cp10k", action="store_true",
                    help="X holds log1p(CP10k), not counts; invert it using "
                         "--depth-obs to recover integer counts.")
    args = ap.parse_args()

    # ---- resolve the panel recipe -------------------------------------
    # The recipe lives in tracer/configs (defaults.toml [panel], plus
    # configs/panels/<preset>.toml). It is inferred from the SOURCE flags,
    # which already determine it unambiguously, and any explicit CLI flag
    # overrides it. See PanelConfig for why `promote` matters most: the
    # pipeline reads whatever estimator sits in the `PMI` column.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tracer.config import load_config

    preset = args.panel_preset
    if preset is None and args.transcripts and args.nucleus_only:
        preset = "nuclear"
    if preset is None and args.transcripts and not args.nucleus_only:
        ap.error("--transcripts without --nucleus-only is ambiguous: pass "
                 "--panel-preset {reference,nuclear} explicitly")
    pcfg = load_config(path=args.panel_config,
                       panel_preset=None if preset in (None, "reference") else preset).panel

    for field, val in (("strategy", pcfg.strategy), ("arms", list(pcfg.arms)),
                       ("depth_metric", pcfg.depth_metric), ("seed", pcfg.seed),
                       ("min_det_cells", pcfg.min_det_cells),
                       ("n_depth_bins", pcfg.n_depth_bins)):
        if getattr(args, field) is None:
            setattr(args, field, val)
    if args.min_abs_value is None:
        args.min_abs_value = pcfg.min_abs_value
    if args.top_k_per_gene is None:
        args.top_k_per_gene = pcfg.top_k_per_gene
    if args.emit_cpmi is None:
        args.emit_cpmi = (pcfg.promote == "cPMI")

    # Report the RESOLVED promotion, not the config's preference: under
    # --no-emit-cpmi no cPMI panel is written, so pointing at one would lie.
    effective = "cPMI" if (pcfg.promote == "cPMI" and args.emit_cpmi) else "PMI"
    canonical = "_cpmi_" if effective == "cPMI" else "_pmi_"
    log(f"panel recipe: preset={preset or 'reference'}  strategy={args.strategy}  "
        f"arms={args.arms}  promote={effective}  depth_metric={args.depth_metric}")
    log(f"  -> the panel TRACER should consume is the one with '{canonical}' "
        f"in its name (the pipeline reads the PMI column blindly)")
    if pcfg.promote == "cPMI" and not args.emit_cpmi:
        log("  WARNING --no-emit-cpmi: only the NAIVE-PMI panel will be written. "
            "TRACER will consume the depth-CONFOUNDED estimator.")
    if pcfg.promote == "PMI":
        log("  NOTE nuclear recipe: naive PMI is INTERIM here (depth-confounded, "
            "67% of pairs positive). cPMI is better per cell but the nuclear "
            "reference is evidence-starved. Fix with more nuclei, then switch.")
    args._panel_cfg = pcfg
    args._panel_preset = preset or "reference"
    args._effective_promote = effective

    if args.strategy in ("rep", "capped") and not args.celltype_col:
        ap.error("--celltype-col is required for --strategy rep/capped")
    if args.h5ad and not args.panel_genes:
        ap.error("--panel-genes is required with --h5ad")
    if args.nucleus_only and not args.transcripts:
        ap.error("--nucleus-only only applies to --transcripts")
    if args.x_is_log1p_cp10k and not args.depth_obs:
        ap.error("--x-is-log1p-cp10k needs --depth-obs to invert the scaling")

    from tracer.conflict_reference import build_depth_corrected_reference
    args.out.mkdir(parents=True, exist_ok=True)

    genes = read_panel_genes(args.panel_genes) if args.panel_genes else None
    if args.transcripts:
        sub, keep, depth, obs = load_transcripts_reference(
            args.transcripts, nucleus_only=args.nucleus_only, panel_genes=genes)
    else:
        sub, keep, depth, obs = load_reference(
            args.h5ad, genes, depth_obs=args.depth_obs,
            x_is_log1p_cp10k=args.x_is_log1p_cp10k,
            exclude_obs=args.exclude_obs)

    if args.strategy == "vanilla":
        # All cells, natural composition. Both estimators are written as
        # separate files, each with its own value promoted into `PMI`.
        # Honours --arms (it used to hardcode min_count=1 and ignore the
        # flag). The count1 arm keeps the legacy `vanilla_spec_*` names so
        # existing panels reproduce bit-identically; other arms are prefixed.
        log(f"all {sub.shape[0]:,} cells, natural composition, {len(keep)} genes")
        for arm in args.arms:
            mc, use_log = ARM_SPECS[arm]
            src = cp10k_log1p(sub, depth) if use_log else sub
            if args.depth_metric == "n_genes":
                cov = np.asarray((src >= (1 if use_log else mc)).sum(1)).ravel().astype(float)
            else:
                cov = depth
            res = build_depth_corrected_reference(
                counts=src, genes=np.asarray(keep, dtype=object), depth=cov,
                min_count=1 if use_log else mc, min_det_cells=args.min_det_cells,
                n_depth_bins=args.n_depth_bins, depth_metric=args.depth_metric,
            top_k_per_gene=args.top_k_per_gene)
            _write_receipt(args, pcfg)
            tag = "" if arm == "count1" else f"{arm}_"
            for name, est in (("vanilla_spec_cpmi", "cPMI"), ("vanilla_spec_pmi", "PMI")):
                write_panel(res.edges, args.out, args.prefix + tag + name,
                            promote=est, min_abs=args.min_abs_value)
        return

    if args.strategy.startswith("grid"):
        ref = grid_draw(sub, depth, k=int(args.strategy[-1]), seed=args.seed)
    else:
        ct = obs[args.celltype_col].astype(str).to_numpy()
        ref = balanced_draw(ct, strategy=args.strategy, seed=args.seed)
    uniq = np.unique(ref)

    for arm in args.arms:
        mc, use_log = ARM_SPECS[arm]
        src = cp10k_log1p(sub, depth) if use_log else sub
        thr = 1 if use_log else mc

        if args.strategy != "capped":
            # Gene admission is decided on DISTINCT cells, then the builder is
            # called with min_det_cells=1 so replication cannot manufacture
            # support for a gene it would not otherwise clear.
            det = np.asarray((src[uniq] >= thr).sum(0)).ravel()
            ok = det >= args.min_det_cells
            genes_ok = [g for g, k in zip(keep, ok) if k]
            log(f"{arm}: {ok.sum()}/{len(keep)} genes clear "
                f"min_det_cells={args.min_det_cells} on {len(uniq):,} unique cells")
            counts, gene_names, min_det = src[ref][:, ok], genes_ok, 1
        else:
            counts, gene_names, min_det = src[ref], keep, args.min_det_cells

        # `depth_metric` in the builder only fires when `depth` is None, and we
        # always pass depth explicitly — so the covariate has to be built here
        # or the flag is silently inert. n_genes = transcriptome complexity
        # under THIS arm's presence rule; total_counts = true library size.
        # After Xgt1 (a depth-RELATIVE presence threshold) the library-size
        # null is largely redundant, which is what makes n_genes worth having.
        if args.depth_metric == "n_genes":
            cov = np.asarray((src >= thr).sum(1)).ravel().astype(float)
        else:
            cov = depth
        res = build_depth_corrected_reference(
            counts=counts, genes=np.asarray(gene_names, dtype=object),
            depth=cov[ref], min_count=1 if use_log else mc,
            min_det_cells=min_det, n_depth_bins=args.n_depth_bins,
            depth_metric=args.depth_metric,
            top_k_per_gene=args.top_k_per_gene)
        _write_receipt(args, pcfg)
        write_panel(res.edges, args.out, args.prefix + arm + SUFFIX[args.strategy],
                    min_abs=args.min_abs_value)
        if args.emit_cpmi:
            cpmi_name = arm + SUFFIX[args.strategy].replace("_pmi_", "_cpmi_")
            write_panel(res.edges, args.out, args.prefix + cpmi_name,
                        promote="cPMI", min_abs=args.min_abs_value)


if __name__ == "__main__":
    main()
