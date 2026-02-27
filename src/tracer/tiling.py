#----------------------·•●  🧽  ●•·-------------------------
#                   TRACER Tiling Module
#----------------------·•●──────●•·-------------------------
# Author: Long Yuan
# Affiliation: Johns Hopkins University
# Email: lyuan13@jhmi.edu
#-----------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
import pymetis
from shapely.strtree import STRtree
import os
from typing import Optional

#
def metis_partition_cells(
    sdata,
    *,
    k: int = 10,
    nparts: int = 100,
    # centroid source
    centroid_source: str = "cell_circles",  # "cell_circles" (fast) or "transcripts" (fallback)
    # transcript-based centroid options (only used if centroid_source="transcripts")
    qv_min: float = 30,
    exclude_unassigned: bool = True,
    # kNN options
    metric: str = "euclidean",
    algorithm: str = "auto",
    n_jobs: int = -1,
    # metis options
    seed: int = 1,
):
    """
    Partition cells using PyMetis on a kNN graph built from cell centroids.

    Returns a dict with:
      - cell_ids (np.ndarray, shape [n_cells])
      - coords (np.ndarray, shape [n_cells, 2])
      - parts (np.ndarray, shape [n_cells])
      - edge_cuts (int)
      - gdf (GeoDataFrame with points + partition)
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if nparts < 2:
        raise ValueError("nparts must be >= 2")

    # -----------------------------
    # Get centroids + cell_ids
    # -----------------------------
    if centroid_source == "cell_circles":
        # SpatialData usually stores circles with center coords or geometry as circles/points.
        # We'll try common patterns.
        gdf = sdata.shapes["cell_circles"]
        gdf = gdf.sort_index()

        cell_ids = gdf.index.astype(str).to_numpy()
        x = gdf.geometry.x.to_numpy(dtype=np.float64)
        y = gdf.geometry.y.to_numpy(dtype=np.float64)
        coords = np.column_stack([x, y])
        n_cells = coords.shape[0]

        if n_cells == 0:
            raise ValueError("cell_circles is empty.")
        if k + 1 > n_cells:
            raise ValueError(f"k={k} too large for n_cells={n_cells} (need k+1 <= n_cells).")

        libsize = None  # optional; not always available here

    elif centroid_source == "transcripts":
        # 
        transcripts = sdata.points["transcripts"].compute()

        df = transcripts
        if "qv" in df.columns:
            df = df[df["qv"] > qv_min]

        if exclude_unassigned and "cell_id" in df.columns:
            df = df[df["cell_id"] != "UNASSIGNED"]

        # centroid + library size
        cell_summary = (
            df.groupby("cell_id")
              .agg(
                  x=("x", "mean"),
                  y=("y", "mean"),
                  library_size=("feature_name", "size"),
              )
        )

        cell_ids = cell_summary.index.astype(str).to_numpy()
        coords = cell_summary[["x", "y"]].to_numpy(dtype=np.float64)
        libsize = cell_summary["library_size"].to_numpy(dtype=np.int64)

    else:
        raise ValueError('centroid_source must be "cell_circles" or "transcripts"')

    n_cells = coords.shape[0]
    if n_cells == 0:
        raise ValueError("No cells found for partitioning.")
    if k + 1 > n_cells:
        raise ValueError(f"k={k} too large for n_cells={n_cells} (need k+1 <= n_cells).")

    # -----------------------------
    # Build kNN adjacency 
    # -----------------------------
    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        metric=metric,
        algorithm=algorithm,
        n_jobs=n_jobs,
    ).fit(coords)

    _, indices = nbrs.kneighbors(coords, return_distance=True)
    neigh = indices[:, 1:]  # drop self (first col)

    # adjacency list for PyMetis expects an undirected graph.
    # We symmetrize efficiently using Python sets per node.
    adj_sets = [set() for _ in range(n_cells)]
    for i in range(n_cells):
        js = neigh[i]
        adj_sets[i].update(js)
        for j in js:
            adj_sets[j].add(i)

    adjacency = [sorted(list(s)) for s in adj_sets]

    # -----------------------------
    # Run PyMetis
    # -----------------------------
    # Note: pymetis accepts "seed" in newer builds; if yours doesn't, remove it.
    try:
        edge_cuts, parts = pymetis.part_graph(
            nparts=nparts,
            adjacency=adjacency,
            seed=seed,
        )
    except TypeError:
        edge_cuts, parts = pymetis.part_graph(
            nparts=nparts,
            adjacency=adjacency,
        )

    parts = np.asarray(parts, dtype=np.int32)

    # -----------------------------
    # Make a compact GeoDataFrame (points)
    # -----------------------------
    out = gpd.GeoDataFrame(
        {
            "cell_id": cell_ids,
            "partition": parts,
        },
        geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
        crs=getattr(sdata, "crs", None),  # SpatialData may not expose CRS directly
    )

    if libsize is not None:
        out["library_size"] = libsize

    return {
        "cell_ids": cell_ids,
        "coords": coords,
        "parts": parts,
        "edge_cuts": int(edge_cuts),
        "gdf": out,
        "k": int(k),
        "nparts": int(nparts),
        "centroid_source": centroid_source,
    }

#
def build_metis_partition_hulls(
    gdf_cells: gpd.GeoDataFrame,
    *,
    partition_col: str = "partition",
    min_cells: int = 3,
):
    """
    Build convex hulls for each METIS partition.

    Returns GeoDataFrame:
        columns: [partition, geometry]
    """
    hull_rows = []

    for p, sub in gdf_cells.groupby(partition_col, sort=True):
        if len(sub) < min_cells:
            continue

        hull = MultiPoint(sub.geometry.values).convex_hull
        hull_rows.append((int(p), hull))

    hull_gdf = gpd.GeoDataFrame(
        {"partition": [r[0] for r in hull_rows]},
        geometry=[r[1] for r in hull_rows],
        crs=gdf_cells.crs,
    )

    return hull_gdf

# Plotting functions 
def plot_metis_partitions(
    gdf_cells: gpd.GeoDataFrame,
    *,
    ax=None,
    s: float = 1.0,
    alpha: float = 0.6,
    title: Optional[str] = None,
):
    """
    Fast scatter plot of METIS partitions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    gdf_cells.plot(
        ax=ax,
        column="partition",
        markersize=s,
        alpha=alpha,
        categorical=True,
        legend=False,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    if title is not None:
        ax.set_title(title)

    return ax

# Plotting functions
def plot_metis_hulls(
    hull_gdf: gpd.GeoDataFrame,
    *,
    ax,
    annotate: bool = False,
    annotate_max: int = 200,
    linewidth: float = 1.0,
):
    """
    Overlay convex hull boundaries and optional labels.
    """
    hull_gdf.boundary.plot(ax=ax, linewidth=linewidth, edgecolor="black")

    if annotate:
        hg = hull_gdf if len(hull_gdf) <= annotate_max else hull_gdf.sample(annotate_max, random_state=0)
        for _, row in hg.iterrows():
            x, y = row.geometry.representative_point().coords[0]
            ax.text(
                x, y,
                str(int(row["partition"])),
                fontsize=9,
                ha="center",
                va="center",
                weight="bold",
            )

# Hybrid transcript chunking
def chunk_transcripts(
    sdata,
    *,
    cell_partition_gdf: gpd.GeoDataFrame,
    hull_gdf: gpd.GeoDataFrame,
    out_dir: str = "chunks",
    qv_min: float = 30,
    cell_id_col: str = "cell_id",
    partition_col: str = "partition",
    x_col: str = "x",
    y_col: str = "y",
    unassigned_label: str = "UNASSIGNED",
    verbose: bool = True,
):
    """
    Hybrid transcript chunking:
      1) Fast cell_id -> partition mapping
      2) Hull-based assignment ONLY for unassigned transcripts

    """

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------
    # Build cell_id -> partition lookup
    # -------------------------------------------------
    cell_to_part = (
        cell_partition_gdf[[partition_col]]
        .astype({partition_col: int})
        .to_dict()[partition_col]
    )

    # -------------------------------------------------
    # Load + filter transcripts
    # -------------------------------------------------
    df = sdata.points["transcripts"].compute()

    if "qv" in df.columns:
        df = df[df["qv"] >= qv_min]

    if verbose:
        print(f"Using {len(df):,} transcripts after qv >= {qv_min}")

    # -------------------------------------------------
    # Cell-based assignment 
    # -------------------------------------------------
    df = df.copy()
    df["partition"] = df[cell_id_col].map(cell_to_part)

    # -------------------------------------------------
    # Identify transcripts needing hull assignment
    # -------------------------------------------------
    needs_hull = (
        df["partition"].isna() |
        (df[cell_id_col] == unassigned_label)
    )

    n_hull = int(needs_hull.sum())
    if verbose:
        print(f"Transcripts requiring hull-based assignment: {n_hull:,}")

    # -------------------------------------------------
    # Hull-based assignment ONLY for unassigned
    # -------------------------------------------------
    if n_hull > 0:
        df_hull = df.loc[needs_hull, [x_col, y_col]].copy()
        points = gpd.points_from_xy(df_hull[x_col], df_hull[y_col])

        hulls = hull_gdf.geometry.values
        part_ids = hull_gdf[partition_col].astype(int).values
        tree = STRtree(hulls)

        assigned = np.full(len(points), -1, dtype=np.int32)

        for i, pt in enumerate(points):
            hit_idxs = tree.query(pt)   # indices (Shapely 2.x)
            for h_idx in hit_idxs:
                if hulls[h_idx].contains(pt):
                    assigned[i] = part_ids[h_idx]
                    break

        df.loc[needs_hull, "partition"] = assigned

    # -------------------------------------------------
    # Final cleanup
    # -------------------------------------------------
    df = df[df["partition"].notna()]
    df["partition"] = df["partition"].astype(int)

    # -------------------------------------------------
    # Reassign any remaining -1 to nearest hull 
    # -------------------------------------------------
    remaining = df["partition"] < 0
    n_remaining = int(remaining.sum())

    if n_remaining > 0:
        if verbose:
            print(f"Fallback assigning {n_remaining:,} transcripts to nearest hull")

        hull_centroids = np.vstack([
            (geom.centroid.x, geom.centroid.y)
            for geom in hull_gdf.geometry
        ])
        hull_parts = hull_gdf["partition"].astype(int).values

        pts = df.loc[remaining, [x_col, y_col]].values

        nn = NearestNeighbors(n_neighbors=1).fit(hull_centroids)
        _, idx = nn.kneighbors(pts)

        df.loc[remaining, "partition"] = hull_parts[idx[:, 0]]

    df["partition"] = df["partition"].astype(int)

    # -------------------------------------------------
    # Write chunks
    # -------------------------------------------------
    summary = []

    for p, df_p in df.groupby("partition", sort=True):
        out_path = f"{out_dir}/chunk_{int(p):03d}.parquet"
        df_p.drop(columns="partition").to_parquet(out_path, index=False)

        summary.append({
            "chunk_id": int(p),
            "n_transcripts": len(df_p),
            "n_cells": df_p[cell_id_col].nunique(),
        })

        if verbose:
            print(
                f"Saved chunk_{int(p):03d}.parquet  "
                f"({len(df_p):,} transcripts, {df_p[cell_id_col].nunique():,} cells)"
            )

    summary_df = pd.DataFrame(summary).sort_values("chunk_id")
    summary_df.to_csv(f"{out_dir}/chunk_summary.csv", index=False)

    return summary_df
