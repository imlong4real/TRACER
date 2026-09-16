#!/usr/bin/env python3
"""Run manuscript-style full-ovary Xenium cell typing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import UNASSIGNED_TOKENS, require_columns, write_json


DEFAULT_SAMPLE = "Xenium_Prime_Human_Ovary_FF"
DEFAULT_RAW_DIR = Path("tutorials/insitucnv_tracer_benchmark/data/xenium_ovary/raw")
DEFAULT_OUTDIR = Path("tutorials/insitucnv_tracer_benchmark/output/full_ovary_celltyping")
DEFAULT_ANNOTATION_MAP = Path("tutorials/insitucnv_tracer_benchmark/config/ovarian_leiden_celltype_map.csv")
DEFAULT_OVARIAN_LEIDEN_MAP = {
    "0": "Epithelial",
    "1": "Epithelial",
    "2": "Low expression",
    "3": "Epithelial",
    "4": "Fibroblasts",
    "5": "Epithelial",
    "6": "T cells",
    "7": "Epithelial",
    "8": "Fibroblasts",
    "9": "Fibroblasts",
    "10": "Monocytes",
    "11": "Epithelial",
    "12": "Monocytes",
    "13": "Endothelial",
    "14": "Monocytes",
    "15": "B cells",
    "16": "Epithelial",
    "17": "Epithelial",
    "18": "Mesothelial",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--sample-name", default=DEFAULT_SAMPLE)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--annotation-map", type=Path, default=DEFAULT_ANNOTATION_MAP)
    p.add_argument("--umap-h5ad", type=Path, default=None, help="Clustered h5ad to reuse with --apply-annotations-only.")
    p.add_argument(
        "--apply-annotations-only",
        action="store_true",
        help="Reuse adata_umap.h5ad and only regenerate cell-class labels, plots, tables, and adata_annotated.h5ad.",
    )
    p.add_argument("--min-genes", type=int, default=5)
    p.add_argument("--min-counts", type=int, default=10)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--leiden-resolution", type=float, default=1.0)
    p.add_argument("--top-leiden-genes", type=int, default=50)
    p.add_argument("--top-cell-class-genes", type=int, default=50)
    p.add_argument("--max-cells", type=int, default=None, help="Optional deterministic subset for smoke tests only.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--spot-size", type=float, default=60.0)
    p.add_argument("--invert-y", action="store_true")
    return p.parse_args()


def sample_path(raw_dir: Path, sample_name: str, suffix: str) -> Path:
    return raw_dir / f"{sample_name}_{suffix}"


def log_step(message: str) -> None:
    print(f"[make_full_ovary_celltyping] {message}", flush=True)


def load_cells(path: Path) -> pd.DataFrame:
    log_step(f"Reading cells from {path}")
    cells = pd.read_parquet(path)
    require_columns(cells, {"cell_id", "x_centroid", "y_centroid"}, str(path))
    cells["cell_id"] = cells["cell_id"].astype(str)
    return cells.drop_duplicates("cell_id", keep="first").set_index("cell_id")


def load_adata(matrix_h5: Path, cells: pd.DataFrame, max_cells: int | None, seed: int):
    import scanpy as sc

    log_step(f"Reading matrix from {matrix_h5}")
    adata = sc.read_10x_h5(str(matrix_h5), gex_only=False)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    common = adata.obs_names.intersection(cells.index.astype(str))
    if len(common) == 0:
        raise SystemExit("No overlap between matrix cell IDs and cells.parquet cell_id values.")
    adata = adata[common].copy()
    adata.obs = adata.obs.join(cells, how="left")
    adata = adata[adata.obs["x_centroid"].notna() & adata.obs["y_centroid"].notna()].copy()
    if max_cells is not None and adata.n_obs > max_cells:
        log_step(f"Subsetting from {adata.n_obs} to {max_cells} cells with seed={seed}")
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
        adata = adata[keep].copy()
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    adata.layers["raw"] = adata.X.copy()
    adata.layers["counts"] = adata.X.copy()
    return adata


def load_clustered_adata(path: Path):
    import scanpy as sc

    log_step(f"Reading clustered cell-typing object from {path}")
    adata = sc.read_h5ad(path)
    adata.obs_names = adata.obs_names.astype(str)
    if "leiden" not in adata.obs:
        raise SystemExit(f"{path} is missing adata.obs['leiden']; rerun full cell typing first.")
    if "counts" not in adata.layers:
        if "raw" in adata.layers:
            adata.layers["counts"] = adata.layers["raw"].copy()
        else:
            raise SystemExit(f"{path} is missing raw/count layers needed by downstream CNV.")
    if "spatial" not in adata.obsm:
        if {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
            adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        elif {"x", "y"}.issubset(adata.obs.columns):
            adata.obs["x_centroid"] = adata.obs["x"].astype(float)
            adata.obs["y_centroid"] = adata.obs["y"].astype(float)
            adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        else:
            raise SystemExit(f"{path} is missing spatial coordinates.")
    return adata


def add_qc_metrics(adata) -> None:
    from scipy import sparse as sp

    X = adata.layers["raw"]
    adata.obs["n_counts"] = np.asarray(X.sum(axis=1)).ravel()
    if sp.issparse(X):
        adata.obs["n_genes"] = np.diff(X.tocsr().indptr)
    else:
        adata.obs["n_genes"] = np.count_nonzero(np.asarray(X), axis=1)


def effective_pcs(adata, requested: int) -> int:
    upper = min(int(requested), adata.n_obs - 1, adata.n_vars - 1)
    if upper < 1:
        raise SystemExit(f"Need at least two cells and genes for PCA; found {adata.n_obs} cells, {adata.n_vars} genes.")
    return upper


def load_annotation_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_OVARIAN_LEIDEN_MAP)
    path = Path(path)
    if not path.exists():
        if path == DEFAULT_ANNOTATION_MAP:
            return dict(DEFAULT_OVARIAN_LEIDEN_MAP)
        raise SystemExit(f"Missing --annotation-map: {path}")
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    cols = {str(col).strip().lower().replace(" ", "_"): col for col in df.columns}
    leiden_col = cols.get("leiden")
    class_col = cols.get("cell_class") or cols.get("cell_type")
    if leiden_col is None or class_col is None:
        raise SystemExit(f"{path} must contain leiden and cell_class columns; present={list(df.columns)}")
    mapping: dict[str, str] = {}
    for _, row in df[[leiden_col, class_col]].dropna().iterrows():
        leiden = str(row[leiden_col]).strip()
        cell_class = str(row[class_col]).strip()
        if leiden and cell_class not in UNASSIGNED_TOKENS:
            mapping[leiden] = cell_class
    if not mapping:
        raise SystemExit(f"No usable Leiden-to-cell-class rows in {path}")
    return mapping


def rank_genes_groups_df(adata, top_n: int | None) -> pd.DataFrame:
    import scanpy as sc

    deg = sc.get.rank_genes_groups_df(adata, group=None)
    if "group" in deg.columns:
        deg["group"] = deg["group"].astype(str)
        if top_n is not None and top_n > 0:
            deg = deg.groupby("group", group_keys=False).head(top_n)
    return deg


def write_counts(adata, column: str, out: Path) -> None:
    counts = adata.obs[column].astype(str).value_counts()
    counts.rename_axis(column.replace(" ", "_")).reset_index(name="n_cells").to_csv(out, index=False)


def write_obs(adata, out: Path) -> None:
    cols = ["x_centroid", "y_centroid", "leiden", "cell class", "n_counts", "n_genes"]
    obs = adata.obs[[col for col in cols if col in adata.obs]].copy()
    obs.insert(0, "cell_id", adata.obs_names.astype(str))
    obs = obs.rename(columns={"x_centroid": "x", "y_centroid": "y", "cell class": "cell_class"})
    obs.to_csv(out, index=False)


def write_unmapped_leiden_clusters(adata, unmapped: list[str], out: Path) -> None:
    if not unmapped:
        if out.exists():
            out.unlink()
        return
    counts = adata.obs["leiden"].astype(str).value_counts()
    with out.open("w") as handle:
        handle.write("leiden\tn_cells\n")
        for leiden in unmapped:
            handle.write(f"{leiden}\t{int(counts.get(leiden, 0))}\n")


def configure_figures() -> None:
    import matplotlib
    import scanpy as sc

    matplotlib.use("Agg", force=True)
    sc.set_figure_params(
        scanpy=True,
        dpi=200,
        dpi_save=200,
        frameon=False,
        vector_friendly=True,
        fontsize=13,
        figsize=(5, 5),
        format="pdf",
        transparent=False,
        ipython_format="png2x",
    )
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def save_umap(adata, color: str, out: Path, legend_loc: str | None = None) -> None:
    import matplotlib.pyplot as plt
    import scanpy as sc

    kwargs = {"color": color, "frameon": False, "show": False, "s": 0.3}
    if legend_loc is not None:
        kwargs["legend_loc"] = legend_loc
    sc.pl.umap(adata, **kwargs)
    plt.savefig(out, bbox_inches="tight")
    plt.close("all")


def save_spatial(adata, color: str, out: Path, spot_size: float, invert_y: bool) -> None:
    import matplotlib.pyplot as plt
    import scanpy as sc

    try:
        sc.pl.spatial(adata, color=color, spot_size=spot_size, frameon=False, show=False)
        if invert_y:
            plt.gca().invert_yaxis()
        plt.savefig(out, bbox_inches="tight")
        plt.close("all")
        return
    except Exception as exc:
        out.with_suffix(out.suffix + ".scanpy_spatial_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        plt.close("all")

    labels = adata.obs[color].astype(str)
    levels = sorted(labels.dropna().unique())
    cmap = plt.get_cmap("tab20")
    palette = {level: cmap(i % cmap.N) for i, level in enumerate(levels)}
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.scatter(
        adata.obs["x_centroid"].to_numpy(dtype=float),
        adata.obs["y_centroid"].to_numpy(dtype=float),
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
    ax.set_title(color)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_ranked_dotplot(adata, out: Path, n_genes: int) -> None:
    import matplotlib.pyplot as plt
    import scanpy as sc

    sc.pl.rank_genes_groups_dotplot(adata, n_genes=n_genes, cmap="Blues", show=False)
    plt.savefig(out, bbox_inches="tight")
    plt.close("all")


def apply_annotation(adata, annotation_map: dict[str, str], outdir: Path) -> list[str]:
    leiden = adata.obs["leiden"].astype(str)
    mapped = leiden.map(annotation_map)
    unmapped = sorted(leiden[mapped.isna()].unique(), key=lambda value: (len(str(value)), str(value)))
    adata.obs["cell class"] = pd.Categorical(mapped.fillna("Unassigned").astype(str))
    write_unmapped_leiden_clusters(adata, unmapped, outdir / "unmapped_leiden_clusters.txt")
    return unmapped


def write_cell_class_outputs(
    adata,
    annotation_map: dict[str, str],
    outdir: Path,
    plot_dir: Path,
    top_cell_class_genes: int,
    spot_size: float,
    invert_y: bool,
    summary: dict,
) -> list[str]:
    import scanpy as sc

    unmapped = apply_annotation(adata, annotation_map, outdir)
    summary["annotation_map_n_entries"] = int(len(annotation_map))
    summary["unmapped_leiden_clusters"] = unmapped
    summary["cell_class_counts"] = adata.obs["cell class"].astype(str).value_counts().to_dict()

    log_step("Writing cell-class figures and marker table")
    save_umap(adata, "cell class", plot_dir / "UMAP_cell_class.pdf")
    save_spatial(adata, "cell class", plot_dir / "spatial_cell_class.pdf", spot_size, invert_y=invert_y)
    sc.tl.dendrogram(adata, groupby="cell class")
    sc.tl.rank_genes_groups(adata, groupby="cell class", n_genes=top_cell_class_genes)
    rank_genes_groups_df(adata, top_cell_class_genes).to_csv(outdir / "cell_class_marker_genes.csv", index=False)
    save_ranked_dotplot(adata, plot_dir / "dotplot_cell_class.pdf", n_genes=5)
    write_counts(adata, "cell class", outdir / "cell_class_counts.csv")
    write_obs(adata, outdir / "celltyping_obs.csv.gz")
    return unmapped


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.outdir / "plots" / "celltyping"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.apply_annotations_only:
        umap_h5ad = args.umap_h5ad or args.outdir / "adata_umap.h5ad"
        for path in (umap_h5ad, args.annotation_map):
            if not path.exists():
                raise SystemExit(f"Missing required input: {path}")
        configure_figures()
        adata = load_clustered_adata(umap_h5ad)
        summary = {
            "mode": "apply_annotations_only",
            "sample_name": args.sample_name,
            "umap_h5ad": str(umap_h5ad),
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "annotation_map": str(args.annotation_map),
        }
        annotation_map = load_annotation_map(args.annotation_map)
        unmapped = write_cell_class_outputs(
            adata,
            annotation_map,
            args.outdir,
            plot_dir,
            args.top_cell_class_genes,
            args.spot_size,
            args.invert_y,
            summary,
        )
        if unmapped:
            summary["status"] = "unmapped_leiden_clusters"
            write_json(args.outdir / "run_summary.json", summary)
            print(json.dumps(summary, indent=2))
            raise SystemExit(
                "Leiden cell typing produced clusters missing from the annotation map: "
                f"{', '.join(unmapped)}. Review leiden_marker_genes.csv and update {args.annotation_map}."
            )
        if "lognorm" not in adata.layers:
            adata.layers["lognorm"] = adata.X.copy()
        adata.write(args.outdir / "adata_annotated.h5ad")
        summary["status"] = "ok"
        write_json(args.outdir / "run_summary.json", summary)
        print(json.dumps(summary, indent=2))
        print(f"Updated cell-class outputs from {umap_h5ad}")
        return 0

    matrix_h5 = sample_path(args.raw_dir, args.sample_name, "cell_feature_matrix.h5")
    cells_path = sample_path(args.raw_dir, args.sample_name, "cells.parquet")
    for path in (matrix_h5, cells_path, args.annotation_map):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    import scanpy as sc

    configure_figures()
    cells = load_cells(cells_path)
    adata = load_adata(matrix_h5, cells, args.max_cells, args.seed)
    summary = {
        "sample_name": args.sample_name,
        "n_cells_loaded": int(adata.n_obs),
        "n_genes_loaded": int(adata.n_vars),
        "min_genes": args.min_genes,
        "min_counts": args.min_counts,
        "max_cells": args.max_cells,
        "annotation_map": str(args.annotation_map),
    }

    log_step(f"Filtering cells with min_genes={args.min_genes}, min_counts={args.min_counts}")
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_cells(adata, min_counts=args.min_counts)
    add_qc_metrics(adata)
    summary["n_cells_after_filter"] = int(adata.n_obs)
    summary["n_genes_after_filter"] = int(adata.n_vars)

    pcs = effective_pcs(adata, args.n_pcs)
    summary["n_pcs_used"] = int(pcs)
    adata.X = adata.layers["raw"].copy()
    log_step("Normalizing total counts")
    sc.pp.normalize_total(adata)
    log_step("Log-transforming counts")
    sc.pp.log1p(adata)
    log_step(f"Running PCA with n_comps={pcs}")
    sc.pp.pca(adata, n_comps=pcs)
    log_step(f"Building neighbors with n_pcs={pcs}")
    sc.pp.neighbors(adata, n_pcs=pcs)
    log_step(f"Running Leiden with resolution={args.leiden_resolution}")
    sc.tl.leiden(adata, resolution=args.leiden_resolution)
    log_step("Running UMAP")
    sc.tl.umap(adata)
    adata.write(args.outdir / "adata_umap.h5ad")

    log_step("Writing Leiden figures and marker table")
    save_umap(adata, "leiden", plot_dir / "UMAP_leiden.pdf", legend_loc="on data")
    save_spatial(adata, "leiden", plot_dir / "spatial_leiden.pdf", args.spot_size, invert_y=args.invert_y)
    sc.tl.rank_genes_groups(adata, groupby="leiden", n_genes=args.top_leiden_genes)
    rank_genes_groups_df(adata, args.top_leiden_genes).to_csv(args.outdir / "leiden_marker_genes.csv", index=False)
    save_ranked_dotplot(adata, plot_dir / "dotplot_leiden_markers.pdf", n_genes=3)
    write_counts(adata, "leiden", args.outdir / "leiden_counts.csv")

    annotation_map = load_annotation_map(args.annotation_map)
    unmapped = write_cell_class_outputs(
        adata,
        annotation_map,
        args.outdir,
        plot_dir,
        args.top_cell_class_genes,
        args.spot_size,
        args.invert_y,
        summary,
    )

    if unmapped:
        summary["status"] = "unmapped_leiden_clusters"
        write_json(args.outdir / "run_summary.json", summary)
        print(json.dumps(summary, indent=2))
        raise SystemExit(
            "Leiden cell typing produced clusters missing from the annotation map: "
            f"{', '.join(unmapped)}. Review leiden_marker_genes.csv and update {args.annotation_map}."
        )

    adata.layers["lognorm"] = adata.X.copy()
    adata.write(args.outdir / "adata_annotated.h5ad")
    summary["status"] = "ok"
    write_json(args.outdir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote full ovary cell-typing outputs to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
