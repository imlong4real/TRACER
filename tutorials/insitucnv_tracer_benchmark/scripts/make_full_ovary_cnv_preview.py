#!/usr/bin/env python3
"""Build full-slide InSituCNV spatial previews from annotated cell types."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import read_gene_positions, write_json


DEFAULT_CELLTYPING_H5AD = Path(
    "tutorials/insitucnv_tracer_benchmark/output/full_ovary_celltyping/adata_annotated.h5ad"
)
DEFAULT_OUTDIR = Path("tutorials/insitucnv_tracer_benchmark/output/full_ovary_cnv_preview")
REFERENCE_CLASSES = ["Monocytes", "B cells", "T cells", "Endothelial", "Low expression", "Fibroblasts"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--celltyping-h5ad", type=Path, default=DEFAULT_CELLTYPING_H5AD)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--gene-positions",
        type=Path,
        default=Path("tutorials/insitucnv_tracer_benchmark/data/gene_positions_grch38.tsv"),
        help="TSV/CSV with gene, chromosome, start, end. Required unless --dry-run is used.",
    )
    p.add_argument("--smoothing-neighbors", type=int, default=200)
    p.add_argument(
        "--smoothing-graph",
        choices=("existing", "expression", "spatial"),
        default="existing",
        help=(
            "Neighbor graph used by InSituCNV smoothing. 'existing' reuses the "
            "cell-typing expression/PCA graph and matches the manuscript workflow; "
            "'expression' rebuilds a Scanpy expression/PCA graph; 'spatial' is an "
            "explicit non-manuscript experiment using physical coordinates."
        ),
    )
    p.add_argument(
        "--expression-neighbors",
        type=int,
        default=15,
        help="Nearest neighbors for --smoothing-graph expression; Scanpy's manuscript default is 15.",
    )
    p.add_argument("--neighbor-n-pcs", type=int, default=50, help="PCs for --smoothing-graph expression.")
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--cnv-resolution", type=float, default=0.4)
    p.add_argument("--epi-cnv-resolution", type=float, default=0.2)
    p.add_argument(
        "--min-epi-clone-pct",
        type=float,
        default=0.1,
        help="Keep epithelial CNV clusters representing more than this percent of epithelial cells.",
    )
    p.add_argument("--max-cells", type=int, default=None, help="Optional deterministic subset for smoke tests.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--write-h5ad", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Validate annotated input and summaries without CNV.")
    p.add_argument("--no-invert-y", action="store_true")
    return p.parse_args()


def log_step(message: str) -> None:
    print(f"[make_full_ovary_cnv_preview] {message}", flush=True)


def normalize_gene_names(names) -> pd.Index:
    return pd.Index([str(x).upper() for x in names])


def load_annotated_adata(path: Path, max_cells: int | None, seed: int):
    import scanpy as sc

    log_step(f"Reading annotated cell typing object from {path}")
    adata = sc.read_h5ad(path)
    adata.obs_names = adata.obs_names.astype(str)
    if "cell class" not in adata.obs and "cell_class" in adata.obs:
        adata.obs["cell class"] = adata.obs["cell_class"].astype(str)
    if "cell class" not in adata.obs:
        raise SystemExit(f"{path} is missing adata.obs['cell class']; run make_full_ovary_celltyping.py first.")
    if "counts" not in adata.layers:
        if "raw" in adata.layers:
            adata.layers["counts"] = adata.layers["raw"].copy()
        else:
            raise SystemExit(f"{path} is missing raw/count layers needed for InSituCNV.")
    if "spatial" not in adata.obsm:
        if {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
            adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        elif {"x", "y"}.issubset(adata.obs.columns):
            adata.obs["x_centroid"] = adata.obs["x"].astype(float)
            adata.obs["y_centroid"] = adata.obs["y"].astype(float)
            adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        else:
            raise SystemExit(f"{path} is missing spatial coordinates.")
    if "x_centroid" not in adata.obs or "y_centroid" not in adata.obs:
        adata.obs["x_centroid"] = adata.obsm["spatial"][:, 0]
        adata.obs["y_centroid"] = adata.obsm["spatial"][:, 1]
    if max_cells is not None and adata.n_obs > max_cells:
        log_step(f"Subsetting from {adata.n_obs} to {max_cells} cells with seed={seed}")
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
        adata = adata[keep].copy()
    return adata


def add_qc_metrics(adata) -> None:
    from scipy import sparse as sp

    X = adata.layers["counts"]
    adata.obs["n_counts"] = np.asarray(X.sum(axis=1)).ravel()
    if sp.issparse(X):
        adata.obs["n_genes"] = np.diff(X.tocsr().indptr)
    else:
        adata.obs["n_genes"] = np.count_nonzero(np.asarray(X), axis=1)


def add_gene_positions(adata, gene_positions: Path):
    pos = read_gene_positions(gene_positions).set_index("gene")
    var_upper = normalize_gene_names(adata.var_names)
    pos_by_upper = pos.copy()
    pos_by_upper.index = normalize_gene_names(pos_by_upper.index)
    pos_by_upper = pos_by_upper.loc[~pos_by_upper.index.duplicated(keep="first")]
    keep = var_upper.isin(pos_by_upper.index)
    if int(keep.sum()) == 0:
        raise SystemExit(f"No genes in matrix overlap {gene_positions}")
    adata._inplace_subset_var(keep)
    matched = pos_by_upper.loc[normalize_gene_names(adata.var_names)]
    adata.var["chromosome"] = "chr" + matched["chromosome"].astype(str).str.replace("^chr", "", regex=True).to_numpy()
    adata.var["start"] = matched["start"].astype(int).to_numpy()
    adata.var["end"] = matched["end"].astype(int).to_numpy()
    return adata


def existing_neighbor_params(adata) -> dict[str, object]:
    neighbors = adata.uns.get("neighbors", {})
    params = dict(neighbors.get("params", {})) if isinstance(neighbors, dict) else {}
    return {str(key): value for key, value in params.items()}


def has_existing_neighbors(adata) -> bool:
    return "neighbors" in adata.uns and "connectivities" in adata.obsp


def require_existing_neighbors(adata) -> dict[str, object]:
    params = existing_neighbor_params(adata)
    if not has_existing_neighbors(adata):
        raise SystemExit(
            "Missing the cell-typing neighbor graph needed for manuscript-style InSituCNV smoothing. "
            "Rerun make_full_ovary_celltyping.py, or explicitly use --smoothing-graph expression/spatial."
        )
    return {"mode": "existing", "params": params}


def build_expression_neighbors(adata, n_neighbors: int, n_pcs: int) -> dict[str, object]:
    import scanpy as sc

    if adata.n_obs < 2:
        raise SystemExit("InSituCNV smoothing requires at least two cells.")
    pcs = min(int(n_pcs), adata.n_obs - 1, adata.n_vars - 1)
    if pcs < 1:
        raise SystemExit(f"Need at least two cells and genes for PCA; found {adata.n_obs} cells, {adata.n_vars} genes.")
    effective_neighbors = min(int(n_neighbors), adata.n_obs - 1)
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=pcs)
    sc.pp.neighbors(adata, n_neighbors=effective_neighbors, n_pcs=pcs)
    return {"mode": "expression", "params": existing_neighbor_params(adata)}


def build_spatial_neighbors(adata, n_neighbors: int) -> dict[str, object]:
    import scanpy as sc

    if "spatial" not in adata.obsm:
        raise SystemExit("Spatial smoothing requires adata.obsm['spatial'].")
    if adata.n_obs < 2:
        raise SystemExit("Spatial smoothing requires at least two cells.")
    effective_neighbors = min(int(n_neighbors), adata.n_obs - 1)
    sc.pp.neighbors(adata, n_neighbors=effective_neighbors, n_pcs=None, use_rep="spatial")
    return {"mode": "spatial", "params": existing_neighbor_params(adata)}


def prepare_smoothing_graph(
    adata,
    smoothing_graph: str,
    smoothing_neighbors: int,
    expression_neighbors: int,
    neighbor_n_pcs: int,
) -> dict[str, object]:
    if smoothing_graph == "existing":
        return require_existing_neighbors(adata)
    if smoothing_graph == "expression":
        return build_expression_neighbors(adata, expression_neighbors, neighbor_n_pcs)
    if smoothing_graph == "spatial":
        return build_spatial_neighbors(adata, smoothing_neighbors)
    raise ValueError(f"Unsupported smoothing graph mode: {smoothing_graph}")


def run_cnv(
    adata,
    smoothing_neighbors: int,
    smoothing_graph: str,
    expression_neighbors: int,
    neighbor_n_pcs: int,
    window_size: int,
    cnv_resolution: float,
    epi_resolution: float,
    min_epi_clone_pct: float,
):
    import infercnvpy as cnv
    import insitucnv as icv
    import scanpy as sc

    present_reference_classes = [
        label for label in REFERENCE_CLASSES if adata.obs["cell class"].astype(str).isin([label]).any()
    ]
    if not present_reference_classes:
        raise SystemExit(f"No reference cells in classes {REFERENCE_CLASSES}; cannot run inferCNV.")
    graph_summary = prepare_smoothing_graph(
        adata,
        smoothing_graph=smoothing_graph,
        smoothing_neighbors=smoothing_neighbors,
        expression_neighbors=expression_neighbors,
        neighbor_n_pcs=neighbor_n_pcs,
    )
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata)
    icv.tl.smooth_data_for_cnv(adata, n_neighbors=min(int(smoothing_neighbors), adata.n_obs - 1))
    if "M" not in adata.layers:
        raise SystemExit("InSituCNV smoothing did not create adata.layers['M']")
    adata.X = adata.layers["M"].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    cnv.tl.infercnv(
        adata,
        reference_key="cell class",
        reference_cat=present_reference_classes,
        window_size=window_size,
        layer="M",
    )
    adata.uns["reference_classes_used"] = present_reference_classes
    adata.uns["smoothing_graph"] = graph_summary
    run_cnv_clustering(adata, "cnv_leiden", cnv_resolution)
    adata.obs["cnv_epi_raw"] = "not_epithelial"
    adata.obs["cnv_epi"] = "not_epithelial"
    epi_mask = adata.obs["cell class"].astype(str).to_numpy() == "Epithelial"
    if int(epi_mask.sum()) > 0:
        adata_epi = adata[epi_mask].copy()
        run_cnv_clustering(adata_epi, "cnv_epi", epi_resolution)
        raw_labels = adata_epi.obs["cnv_epi"].astype(str)
        adata.obs.loc[adata_epi.obs_names, "cnv_epi_raw"] = raw_labels.to_numpy()
        clone_pct = raw_labels.value_counts(normalize=True).mul(100)
        selected_clones = clone_pct[clone_pct > float(min_epi_clone_pct)].index.astype(str).tolist()
        selected_mask = raw_labels.isin(selected_clones)
        adata.obs.loc[adata_epi.obs_names, "cnv_epi"] = "filtered_small_clone"
        selected_obs_names = raw_labels.index[selected_mask.to_numpy()]
        adata.obs.loc[selected_obs_names, "cnv_epi"] = raw_labels.loc[selected_mask].to_numpy()
        adata.uns["cnv_epi_n_cells"] = int(adata_epi.n_obs)
        adata.uns["cnv_epi_raw_n_clusters"] = int(raw_labels.nunique())
        adata.uns["cnv_epi_selected_n_clusters"] = int(len(selected_clones))
        adata.uns["cnv_epi_min_clone_pct"] = float(min_epi_clone_pct)
        adata.uns["cnv_epi_selected_clones"] = selected_clones
        adata.uns["cnv_epi_filtered_small_clone_n_cells"] = int((~selected_mask).sum())
    else:
        adata.uns["cnv_epi_n_cells"] = 0
        adata.uns["cnv_epi_raw_n_clusters"] = 0
        adata.uns["cnv_epi_selected_n_clusters"] = 0
        adata.uns["cnv_epi_min_clone_pct"] = float(min_epi_clone_pct)
        adata.uns["cnv_epi_selected_clones"] = []
        adata.uns["cnv_epi_filtered_small_clone_n_cells"] = 0
    return adata


def run_cnv_clustering(adata, key: str, resolution: float) -> None:
    import infercnvpy as cnv

    cnv.tl.pca(adata)
    cnv.pp.neighbors(adata)
    try:
        cnv.tl.leiden(adata, resolution=resolution, key_added=key)
    except TypeError:
        cnv.tl.leiden(adata, resolution=resolution)
        if "cnv_leiden" in adata.obs and key != "cnv_leiden":
            adata.obs[key] = adata.obs["cnv_leiden"].astype(str)
    if key not in adata.obs and "cnv_leiden" in adata.obs:
        adata.obs[key] = adata.obs["cnv_leiden"].astype(str)
    if key not in adata.obs:
        raise SystemExit(f"infercnvpy leiden did not create {key} or cnv_leiden")


def write_spatial_csv(adata, outdir: Path) -> None:
    requested = ["x_centroid", "y_centroid", "cell class", "leiden", "n_counts", "n_genes"]
    for optional in ("cnv_leiden", "cnv_epi", "cnv_epi_raw"):
        if optional in adata.obs:
            requested.append(optional)
    out = adata.obs[[col for col in requested if col in adata.obs]].copy()
    out.insert(0, "cell_id", adata.obs_names.astype(str))
    out = out.rename(columns={"x_centroid": "x", "y_centroid": "y", "cell class": "cell_class"})
    out.to_csv(outdir / "cell_spatial_cnv_preview.csv.gz", index=False)


def write_adata_obs_csv(adata, outdir: Path) -> None:
    requested = [
        "x_centroid",
        "y_centroid",
        "cell class",
        "leiden",
        "cnv_leiden",
        "cnv_epi",
        "cnv_epi_raw",
        "n_counts",
        "n_genes",
    ]
    out = adata.obs[[col for col in requested if col in adata.obs]].copy()
    out.insert(0, "cell_id", adata.obs_names.astype(str))
    out = out.rename(columns={"x_centroid": "x", "y_centroid": "y", "cell class": "cell_class"})
    out.to_csv(outdir / "adata_obs.csv.gz", index=False)


def write_cluster_summaries(adata, outdir: Path) -> None:
    for col in ("leiden", "cell class", "cnv_leiden", "cnv_epi", "cnv_epi_raw"):
        if col in adata.obs:
            label = col.replace(" ", "_")
            adata.obs[col].astype(str).value_counts().rename_axis(label).reset_index(name="n_cells").to_csv(
                outdir / f"{label}_counts.csv", index=False
            )


def categorical_palette(values: pd.Series) -> dict[str, tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt

    levels = sorted(pd.Series(values.astype(str)).dropna().unique())
    cmap = plt.get_cmap("tab20")
    return {level: cmap(i % cmap.N) for i, level in enumerate(levels)}


def plot_spatial(adata, color_key: str, out: Path, invert_y: bool, mask=None) -> None:
    import matplotlib.pyplot as plt

    if color_key not in adata.obs:
        return
    if mask is None:
        mask = np.ones(adata.n_obs, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return
    labels = adata.obs.loc[mask, color_key].astype(str)
    palette = categorical_palette(labels)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    ax.scatter(
        adata.obs.loc[mask, "x_centroid"].to_numpy(dtype=float),
        adata.obs.loc[mask, "y_centroid"].to_numpy(dtype=float),
        c=list(labels.map(palette)),
        s=0.2,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    ax.set_xlabel("x centroid")
    ax.set_ylabel("y centroid")
    ax.set_title(color_key)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=4, label=label)
        for label, color in palette.items()
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not args.celltyping_h5ad.exists():
        raise SystemExit(f"Missing --celltyping-h5ad: {args.celltyping_h5ad}")
    if not args.dry_run and not args.gene_positions.exists():
        raise SystemExit(f"Missing --gene-positions table required for CNV: {args.gene_positions}")

    adata = load_annotated_adata(args.celltyping_h5ad, args.max_cells, args.seed)
    add_qc_metrics(adata)
    summary = {
        "celltyping_h5ad": str(args.celltyping_h5ad),
        "n_cells": int(adata.n_obs),
        "n_genes_matrix": int(adata.n_vars),
        "cell_class_counts": adata.obs["cell class"].astype(str).value_counts().to_dict(),
        "gene_positions": str(args.gene_positions),
        "gene_positions_exists": args.gene_positions.exists(),
        "dry_run": bool(args.dry_run),
        "max_cells": args.max_cells,
        "smoothing_graph_requested": args.smoothing_graph,
        "existing_neighbors_present": has_existing_neighbors(adata),
        "existing_neighbor_params": existing_neighbor_params(adata),
    }
    if args.dry_run and args.smoothing_graph == "existing":
        require_existing_neighbors(adata)
    if not args.dry_run:
        adata = add_gene_positions(adata, args.gene_positions)
        summary["n_genes_with_positions"] = int(adata.n_vars)
        adata = run_cnv(
            adata,
            smoothing_neighbors=args.smoothing_neighbors,
            smoothing_graph=args.smoothing_graph,
            expression_neighbors=args.expression_neighbors,
            neighbor_n_pcs=args.neighbor_n_pcs,
            window_size=args.window_size,
            cnv_resolution=args.cnv_resolution,
            epi_resolution=args.epi_cnv_resolution,
            min_epi_clone_pct=args.min_epi_clone_pct,
        )
        summary["cnv_epi_n_cells"] = int(adata.uns.get("cnv_epi_n_cells", 0))
        summary["cnv_epi_raw_n_clusters"] = int(adata.uns.get("cnv_epi_raw_n_clusters", 0))
        summary["cnv_epi_selected_n_clusters"] = int(adata.uns.get("cnv_epi_selected_n_clusters", 0))
        summary["cnv_epi_min_clone_pct"] = float(adata.uns.get("cnv_epi_min_clone_pct", args.min_epi_clone_pct))
        summary["cnv_epi_filtered_small_clone_n_cells"] = int(
            adata.uns.get("cnv_epi_filtered_small_clone_n_cells", 0)
        )
        summary["cnv_leiden_n_clusters"] = (
            int(adata.obs["cnv_leiden"].astype(str).nunique()) if "cnv_leiden" in adata.obs else 0
        )
        summary["reference_classes_used"] = list(adata.uns.get("reference_classes_used", []))
        summary["smoothing_neighbors_requested"] = int(args.smoothing_neighbors)
        summary["smoothing_graph"] = dict(adata.uns.get("smoothing_graph", {}))

    write_spatial_csv(adata, args.outdir)
    write_adata_obs_csv(adata, args.outdir)
    write_cluster_summaries(adata, args.outdir)
    plot_spatial(adata, "cell class", plot_dir / "spatial_cell_class.png", invert_y=not args.no_invert_y)
    plot_spatial(adata, "cnv_leiden", plot_dir / "spatial_cnv_leiden.png", invert_y=not args.no_invert_y)
    if "cnv_epi" in adata.obs:
        selected_epi_mask = ~adata.obs["cnv_epi"].astype(str).isin({"not_epithelial", "filtered_small_clone"})
        plot_spatial(
            adata,
            "cnv_epi",
            plot_dir / "spatial_cnv_epi.png",
            invert_y=not args.no_invert_y,
            mask=selected_epi_mask,
        )
    if "cnv_epi_raw" in adata.obs:
        raw_epi_mask = adata.obs["cnv_epi_raw"].astype(str) != "not_epithelial"
        plot_spatial(
            adata,
            "cnv_epi_raw",
            plot_dir / "spatial_cnv_epi_raw.png",
            invert_y=not args.no_invert_y,
            mask=raw_epi_mask,
        )
    write_json(args.outdir / "run_summary.json", summary)
    if args.write_h5ad:
        adata.write_h5ad(args.outdir / "adata_full_ovary_cnv_preview.h5ad")
    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote full ovary CNV preview outputs to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
