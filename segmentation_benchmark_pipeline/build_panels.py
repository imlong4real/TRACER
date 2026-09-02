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
SUFFIX = {"rep": "_pmi_balanced_rep", "capped": "_pmi_balanced"}


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


def load_reference(h5ad: Path, panel_genes: list[str]):
    """Return (counts_over_panel_genes, kept_gene_names, true_depth, obs)."""
    A = ad.read_h5ad(h5ad)
    vn = np.array([str(g) for g in A.var_names])
    keep = [g for g in panel_genes if g in set(vn)]
    X = A.layers["counts"] if "counts" in A.layers else A.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    # Depth over ALL genes — the true library size, not the panel subset.
    depth = np.asarray(X.sum(1)).ravel().astype(float)
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


def cp10k_log1p(sub, depth):
    """log1p(counts per 10k), the `xgt1` presence space."""
    CP = (sp.diags(1e4 / np.maximum(depth, 1)) @ sub.astype(np.float64)).tocsr()
    L = CP.copy()
    L.data = np.log1p(L.data)
    return L


def write_panel(edges, out: Path, name: str, *, promote: str | None = None):
    """Write one panel. `promote` copies that estimator into the `PMI` slot."""
    if promote is None:
        df = edges[EDGE_COLS]
    else:
        df = (edges[["gene_i", "gene_j", promote, "O", "E", "z"]]
              .rename(columns={promote: "PMI"}))
    p = out / f"{name}.csv.gz"
    df.to_csv(p, index=False)
    v = edges[promote or "PMI"]
    gs = set(edges.gene_i) | set(edges.gene_j)
    log(f"{name:26s} {len(edges):>7,} edges  {len(gs):>3} genes  "
        f"O med {np.median(edges.O):>6.0f}  "
        f"pos {np.mean(v > 0.2):.0%} neg {np.mean(v < -0.2):.0%} -> {p.name}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", type=Path, required=True,
                    help="Single-cell reference with a `counts` layer.")
    ap.add_argument("--panel-genes", type=Path, required=True,
                    help="Transcripts parquet (feature_name column) or a "
                         "one-gene-per-line text file.")
    ap.add_argument("--out", type=Path, required=True, help="Output directory.")
    ap.add_argument("--strategy", choices=("rep", "capped", "vanilla"),
                    default="rep", help="Reference draw (default: rep).")
    ap.add_argument("--celltype-col", default=None,
                    help="obs column to balance on. Required unless "
                         "--strategy vanilla.")
    ap.add_argument("--arms", nargs="+", default=["count1", "count2", "xgt1"],
                    choices=sorted(ARM_SPECS), help="Presence arms to build.")
    ap.add_argument("--prefix", default="",
                    help="Prepended to every output panel name.")
    ap.add_argument("--emit-cpmi", action="store_true",
                    help="Also write a `<arm>_cpmi_balanced...` panel per arm, "
                         "the same build with cPMI promoted into the `PMI` "
                         "column so the pipeline consumes the depth-corrected "
                         "estimator. Adds no information — the default panel "
                         "already carries both columns — but drops into a run "
                         "without editing. Implied for --strategy vanilla.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-det-cells", type=int, default=25,
                    help="A gene must be detected in this many cells (default 25).")
    ap.add_argument("--n-depth-bins", type=int, default=25)
    ap.add_argument("--depth-metric", default="total_counts")
    args = ap.parse_args()

    if args.strategy != "vanilla" and not args.celltype_col:
        ap.error("--celltype-col is required unless --strategy vanilla")

    from tracer.conflict_reference import build_depth_corrected_reference
    args.out.mkdir(parents=True, exist_ok=True)

    genes = read_panel_genes(args.panel_genes)
    sub, keep, depth, obs = load_reference(args.h5ad, genes)

    if args.strategy == "vanilla":
        # All cells, natural composition. Both estimators are written as
        # separate files, each with its own value promoted into `PMI`.
        log(f"all {sub.shape[0]:,} cells, natural composition, {len(keep)} genes")
        res = build_depth_corrected_reference(
            counts=sub, genes=np.asarray(keep, dtype=object), depth=depth,
            min_count=1, min_det_cells=args.min_det_cells,
            n_depth_bins=args.n_depth_bins, depth_metric=args.depth_metric)
        for name, est in (("vanilla_spec_cpmi", "cPMI"), ("vanilla_spec_pmi", "PMI")):
            write_panel(res.edges, args.out, args.prefix + name, promote=est)
        return

    ct = obs[args.celltype_col].astype(str).to_numpy()
    ref = balanced_draw(ct, strategy=args.strategy, seed=args.seed)
    uniq = np.unique(ref)

    for arm in args.arms:
        mc, use_log = ARM_SPECS[arm]
        src = cp10k_log1p(sub, depth) if use_log else sub
        thr = 1 if use_log else mc

        if args.strategy == "rep":
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

        res = build_depth_corrected_reference(
            counts=counts, genes=np.asarray(gene_names, dtype=object),
            depth=depth[ref], min_count=1 if use_log else mc,
            min_det_cells=min_det, n_depth_bins=args.n_depth_bins,
            depth_metric=args.depth_metric)
        write_panel(res.edges, args.out, args.prefix + arm + SUFFIX[args.strategy])
        if args.emit_cpmi:
            cpmi_name = arm + SUFFIX[args.strategy].replace("_pmi_", "_cpmi_")
            write_panel(res.edges, args.out, args.prefix + cpmi_name, promote="cPMI")


if __name__ == "__main__":
    main()
