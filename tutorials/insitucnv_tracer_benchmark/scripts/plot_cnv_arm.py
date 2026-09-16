#!/usr/bin/env python3
"""Per-arm CNV visualization: UMAP + marker dotplot by clone, from an adata_cnv.h5ad.

Consumes the ``adata_cnv.h5ad`` written by ``run_insitucnv_arm.py`` (which runs
infercnvpy: ``obsm['X_cnv']``, ``obs['cnv_leiden']`` = CNV clones,
``obs['compartment']`` / ``obs['annotation']``, ``layers['gene_values_cnv']``,
and ``var['chromosome']``). It emits, for ONE arm (raw or tracer_whole):

  - ``<label>_umap.png``            : UMAP colored by cnv clone / compartment / annotation
  - ``<label>_dotplot_clone.png``   : marker-gene dotplot grouped by cnv clone
  - ``<label>_chrom_cnv_by_clone.csv`` + ``<label>_chrom_cnv_by_clone.png``
                                      : mean CNV per (clone x chromosome) heatmap

Only scanpy/anndata are required (UMAP is computed with scanpy on the CNV
embedding, so infercnvpy is not needed here). Markers come from a panel config
(``panels.yaml``: ``marker_sets``).

EXAMPLE
=======
::

    python plot_cnv_arm.py \\
      --adata output/runs/pilot_full_100/insitucnv_raw/adata_cnv.h5ad \\
      --marker-config tutorials/insitucnv_tracer_benchmark/config/panels.yaml \\
      --label raw --outdir output/runs/pilot_full_100/insitucnv_raw/plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_yaml(path: Path) -> dict:
    try:
        import yaml
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        with open(path) as fh:
            return json.load(fh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adata", required=True, type=Path, help="adata_cnv.h5ad from run_insitucnv_arm.py")
    p.add_argument("--marker-config", type=Path, default=None,
                   help="panels.yaml with marker_sets (for the dotplot). If absent, dotplot is skipped.")
    p.add_argument("--label", required=True, help="Arm label used to prefix output files (e.g. raw, tracer_whole).")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--clone-col", default="cnv_leiden")
    p.add_argument("--n-neighbors", type=int, default=15)
    return p.parse_args()


def resolve_clone_col(adata, requested: str) -> str:
    if requested in adata.obs.columns:
        return requested
    for c in ("cnv_leiden", "leiden", "cnv_clusters"):
        if c in adata.obs.columns:
            return c
    raise SystemExit(f"No clone column found (looked for {requested}, cnv_leiden, leiden). "
                     f"obs cols: {list(adata.obs.columns)}")


def compute_umap(adata, clone_col: str, n_neighbors: int):
    import scanpy as sc
    n_neighbors = min(n_neighbors, max(2, adata.n_obs - 1))
    if "X_cnv" in adata.obsm:
        rep = np.asarray(adata.obsm["X_cnv"].todense()) if hasattr(adata.obsm["X_cnv"], "todense") \
            else np.asarray(adata.obsm["X_cnv"])
        adata.obsm["X_cnv_dense"] = rep
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_cnv_dense")
    else:
        # Fall back to expression PCA if the CNV embedding is missing.
        sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1, adata.n_obs - 1))
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)


def summarize_chrom_by_clone(adata, clone_col: str) -> pd.DataFrame:
    from scipy import sparse as sp
    if "gene_values_cnv" not in adata.layers or "chromosome" not in adata.var.columns:
        return pd.DataFrame()
    X = adata.layers["gene_values_cnv"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    chrom = adata.var["chromosome"].astype(str).to_numpy()
    clones = adata.obs[clone_col].astype(str).to_numpy()
    rows = {}
    for cl in sorted(pd.unique(clones)):
        mask = clones == cl
        per_gene = X[mask].mean(axis=0)
        rows[cl] = pd.Series(per_gene, index=chrom).groupby(level=0).mean()
    return pd.DataFrame(rows).T  # clone x chromosome


def main() -> int:
    import scanpy as sc
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(args.outdir)
    sc.settings.autoshow = False

    adata = sc.read_h5ad(args.adata)
    clone_col = resolve_clone_col(adata, args.clone_col)
    adata.obs[clone_col] = adata.obs[clone_col].astype("category")
    print(f"[{args.label}] {adata.n_obs} entities x {adata.n_vars} genes; "
          f"clones={adata.obs[clone_col].nunique()} ({clone_col})")

    # --- UMAP ---------------------------------------------------------------
    compute_umap(adata, clone_col, args.n_neighbors)
    color = [c for c in (clone_col, "compartment", "annotation") if c in adata.obs.columns]
    fig = sc.pl.umap(adata, color=color, ncols=len(color), show=False, return_fig=True,
                     title=[f"{args.label}: {c}" for c in color])
    fig.savefig(args.outdir / f"{args.label}_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- marker dotplot by clone -------------------------------------------
    if args.marker_config is not None and args.marker_config.exists():
        cfg = load_yaml(args.marker_config)
        marker_sets = cfg.get("marker_sets") or {}
        # Preserve group structure so the dotplot brackets markers by cell type.
        var_markers = {grp: [g for g in genes if g in adata.var_names]
                       for grp, genes in marker_sets.items()}
        var_markers = {g: v for g, v in var_markers.items() if v}
        if var_markers:
            dp = sc.pl.dotplot(adata, var_markers, groupby=clone_col, show=False, return_fig=True)
            dp.savefig(args.outdir / f"{args.label}_dotplot_clone.png", dpi=150, bbox_inches="tight")
            plt.close("all")
        else:
            print(f"[{args.label}] no marker genes present in var_names; skipping dotplot")
    else:
        print(f"[{args.label}] no marker config; skipping dotplot")

    # --- per-clone chromosome CNV heatmap ----------------------------------
    chrom = summarize_chrom_by_clone(adata, clone_col)
    if not chrom.empty:
        # Order chromosomes 1..22, X, Y.
        def crank(c):
            c = c.replace("chr", "")
            return (0, int(c)) if c.isdigit() else (1, {"X": 23, "Y": 24, "MT": 25}.get(c, 99))
        chrom = chrom[sorted(chrom.columns, key=crank)]
        chrom.to_csv(args.outdir / f"{args.label}_chrom_cnv_by_clone.csv")
        fig, ax = plt.subplots(figsize=(max(6, 0.35 * chrom.shape[1]), max(3, 0.4 * chrom.shape[0])))
        vmax = float(np.nanmax(np.abs(chrom.to_numpy()))) or 1e-3
        im = ax.imshow(chrom.to_numpy(), aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(chrom.shape[1])); ax.set_xticklabels(chrom.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(chrom.shape[0])); ax.set_yticklabels(chrom.index, fontsize=7)
        ax.set_title(f"{args.label}: mean CNV by clone x chromosome")
        fig.colorbar(im, ax=ax, shrink=0.6)
        fig.savefig(args.outdir / f"{args.label}_chrom_cnv_by_clone.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        print(f"[{args.label}] gene_values_cnv/chromosome missing; skipping chrom heatmap")

    print(f"[{args.label}] wrote plots to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
