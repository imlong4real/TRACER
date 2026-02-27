from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import geopandas as gpd
from shapely.geometry import Polygon

from spatialdata import SpatialData
from spatialdata.models import TableModel, PointsModel, ShapesModel

from tracer.metrics import compute_npmi


def _build_polygons_from_vertices(df: pd.DataFrame) -> gpd.GeoDataFrame:
    polys = []
    cell_ids = []
    for cell_id, sub in df.groupby("cell_id"):
        coords = list(zip(sub["vertex_x"], sub["vertex_y"]))
        if len(coords) >= 3:
            polys.append(Polygon(coords))
            cell_ids.append(str(cell_id))
    return gpd.GeoDataFrame({"cell_id": cell_ids, "geometry": polys})


def _build_spatialdata_manual(ds_dir: Path) -> SpatialData:
    h5_path = ds_dir / "cell_feature_matrix.h5"
    adata = sc.read_10x_h5(str(h5_path), gex_only=True)
    adata.var_names_make_unique()
    adata.obs["cell_id"] = adata.obs_names.astype(str)

    shapes = {}
    region = "cells"

    cell_boundaries = ds_dir / "cell_boundaries.parquet"
    if cell_boundaries.exists():
        cb = pd.read_parquet(cell_boundaries)
        cb["cell_id"] = cb["cell_id"].astype(str)
        gdf = _build_polygons_from_vertices(cb)
        gdf = gdf.set_index("cell_id")
        shapes["cell_boundaries"] = ShapesModel.parse(gdf)
        region = "cell_boundaries"

    nucleus_boundaries = ds_dir / "nucleus_boundaries.parquet"
    if nucleus_boundaries.exists():
        nb = pd.read_parquet(nucleus_boundaries)
        nb["cell_id"] = nb["cell_id"].astype(str)
        gdf_nb = _build_polygons_from_vertices(nb)
        gdf_nb = gdf_nb.set_index("cell_id")
        shapes["nucleus_boundaries"] = ShapesModel.parse(gdf_nb)

    adata.obs["region"] = region
    table = TableModel.parse(
        adata,
        region=region,
        region_key="region",
        instance_key="cell_id",
    )

    transcripts_path = ds_dir / "transcripts.parquet"
    transcripts = pd.read_parquet(transcripts_path)
    points = PointsModel.parse(
        transcripts,
        coordinates={"x": "x_location", "y": "y_location", "z": "z_location"},
        feature_key="feature_name",
        instance_key="cell_id",
    )

    return SpatialData(tables={"table": table}, points={"transcripts": points}, shapes=shapes)


def main() -> None:
    data_root = Path("/Users/lyuan13/Desktop/HOT-NERD/tutorials/melanoma/data")
    out_dir = data_root / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted({p.parent for p in data_root.rglob("experiment.xenium")})
    if not dataset_dirs:
        raise FileNotFoundError("No experiment.xenium found under melanoma/data")

    summary_rows = []

    for ds_dir in dataset_dirs:
        prefix = ds_dir.name if ds_dir != data_root else "melanoma"
        print(f"\n=== Processing {prefix} ===")

        print("parsing the data... ", end="")
        sdata = _build_spatialdata_manual(ds_dir)
        print("done")

        print("writing the data... ", end="")
        zarr_path = out_dir / f"{prefix}.zarr"
        sdata.write(zarr_path, overwrite=True)
        print("done")

        genes = sdata.tables["table"].var.index
        n_genes = int(len(genes))
        n_cells = int(sdata.tables["table"].obs.shape[0])

        transcripts = sdata.points["transcripts"].compute()
        total_transcripts = int(len(transcripts))

        vsir_present = "VSIR" in genes
        vsig4_present = "VSIG4" in genes
        selplg_present = "SELPLG" in genes

        summary_rows.append(
            {
                "sample": prefix,
                "n_genes": n_genes,
                "n_cells": n_cells,
                "total_transcripts": total_transcripts,
                "VSIR_present": vsir_present,
                "VSIG4_present": vsig4_present,
                "SELPLG_present": selplg_present,
            }
        )

        filtered_df = transcripts[transcripts["qv"] > 30].copy()
        filtered_df = filtered_df[filtered_df["feature_name"].isin(genes)].copy()

        filtered_out = out_dir / f"{prefix}_transcripts_qv30.parquet"
        filtered_df.to_parquet(filtered_out, index=False)

        nuc_df = filtered_df[
            (filtered_df["cell_id"] != "UNASSIGNED")
            & (filtered_df["overlaps_nucleus"] == 1)
        ].copy()
        nuc_df["cell_id"] = nuc_df["cell_id"].astype(str)

        nuc_counts = nuc_df.groupby("cell_id").size()
        low_thres = np.percentile(nuc_counts, 20)
        high_thres = np.percentile(nuc_counts, 80)
        print("Transcript count thresholds:", low_thres, high_thres)

        good_nuc_ids = nuc_counts[(nuc_counts >= low_thres) & (nuc_counts <= high_thres)].index
        nuc_df_confident = nuc_df[nuc_df["cell_id"].isin(good_nuc_ids)].copy()
        print("Number of confident nuclei:", len(good_nuc_ids))

        npmi_df = compute_npmi(nuc_df_confident, group_key="cell_id")
        npmi_out = out_dir / f"{prefix}_nucleus_npmi.csv"
        npmi_df.to_csv(npmi_out, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
