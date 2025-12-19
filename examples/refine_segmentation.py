"""Example: refine segmentation visualization

This script builds a small synthetic dataset and runs the HOT-NERD
pipeline to demonstrate how an initial segmentation image (`10X.png`)
can be refined to a clustered result (`v1_example.png`).

Run from the repository root after installing the package in editable
mode (recommended):

    pip install -e .
    python examples/refine_segmentation.py

Or run without installing by adding the `src` folder to `PYTHONPATH`:

    PYTHONPATH=./src python examples/refine_segmentation.py
"""
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
    from shapely.geometry import box
except Exception:
    raise SystemExit("This example requires geopandas and shapely. Install via pip install geopandas shapely")

# Ensure repo/src is on path when running from examples/
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    from hotnerd import (
        build_graph,
        add_edge_prob_stats,
        to_networkx,
        prune_graph,
        purity_conflict_from_cc,
        build_npmi_matrix_from_long,
        stitch_connected_components,
        plot_cc,
        calculate_rankings,
        calculate_thresholds,
    )
except Exception as e:
    raise SystemExit(
        "Failed to import hotnerd. Either install the package (pip install -e .) or ensure src/ is on PYTHONPATH.\n"
        f"Import error: {e}"
    )


def make_synthetic_data(n_transcripts=800, n_cells=40, n_genes=6, seed=0):
    rng = np.random.default_rng(seed)

    genes = [f"G{i+1}" for i in range(n_genes)]

    # Transcript-level table
    x = rng.normal(loc=0.0, scale=100.0, size=n_transcripts)
    y = rng.normal(loc=0.0, scale=100.0, size=n_transcripts)
    z = rng.normal(loc=0.0, scale=10.0, size=n_transcripts)
    cell_id = rng.integers(0, n_cells, size=n_transcripts)
    feature_name = rng.choice(genes, size=n_transcripts)

    df_local = pd.DataFrame({
        "x": x,
        "y": y,
        "z": z,
        "cell_id": cell_id,
        "feature_name": feature_name,
    })

    # create simple cell polygons (squares) for plotting
    cell_centers = {
        cid: (float(x[cell_id == cid].mean() or 0.0), float(y[cell_id == cid].mean() or 0.0))
        for cid in range(n_cells)
    }
    polys = []
    ids = []
    size = 40.0
    for cid, (cx, cy) in cell_centers.items():
        polys.append(box(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2))
        ids.append(cid)

    cell_gdf = gpd.GeoDataFrame({"cell_id": ids, "geometry": polys}).set_index("cell_id")

    # pairwise NPMI / conditional table (long form)
    rows = []
    ranked_rows = []
    for a, b in combinations(genes, 2):
        npmi = float(rng.uniform(-0.1, 0.6))
        p_i = float(rng.uniform(0.01, 0.2))
        p_j = float(rng.uniform(0.01, 0.2))
        p_ij = min(p_i, p_j) * float(rng.uniform(0.01, 0.8))
        p_j_given_i = p_ij / p_i if p_i > 0 else 0.0
        p_i_given_j = p_ij / p_j if p_j > 0 else 0.0
        count = int(rng.integers(1, 50))

        rows.append({"gene_i": a, "gene_j": b, "NPMI": npmi})

        # add both directions for ranked_df lookup convenience
        for gi, gj in [(a, b), (b, a)]:
            ranked_rows.append(
                {
                    "gene_i": gi,
                    "gene_j": gj,
                    "P_i": p_i,
                    "P_j": p_j,
                    "P_ij": p_ij,
                    "NPMI": npmi,
                    "P_j_given_i": p_j_given_i,
                    "P_i_given_j": p_i_given_j,
                    "count_ij": count,
                }
            )

    pair_counts_df = pd.DataFrame(rows)
    ranked_df = pd.DataFrame(ranked_rows)

    return df_local, cell_gdf, ranked_df, pair_counts_df


def main():
    out_dir = os.path.abspath("examples/output")
    os.makedirs(out_dir, exist_ok=True)

    # If example CSVs exist in this folder, load them; otherwise use synthetic data
    trans_fp = os.path.join(HERE, "example_transcripts.csv")
    npmi_fp = os.path.join(HERE, "example_npmi_long.csv")
    cell_fp = os.path.join(HERE, "example_cell_geometries.csv")

    if os.path.exists(trans_fp) and os.path.exists(npmi_fp) and os.path.exists(cell_fp):
        print("Loading example data from CSVs...")
        df_local = pd.read_csv(trans_fp)

        # pairwise NPMI long table
        pair_counts_df = pd.read_csv(npmi_fp)

        # ranked_df expected by calculate_rankings should include the directional
        # fields P_j_given_i and P_i_given_j    
        ranked_df = pair_counts_df.copy()

        # load cell geometries (WKT in 'geometry' column)
        cell_df = pd.read_csv(cell_fp)
        try:
            from shapely import wkt

            cell_df["geometry"] = cell_df["geometry"].apply(wkt.loads)
        except Exception:
            # fallback: assume geometry column already parsed
            pass

        cell_gdf = gpd.GeoDataFrame(cell_df, geometry="geometry").set_index("cell_id")
    else:
        df_local, cell_gdf, ranked_df, pair_counts_df = make_synthetic_data()

    # Save initial segmentation visualization (10X.png)
    fig, ax = plt.subplots(figsize=(8, 8))
    cell_gdf.plot(ax=ax, column=cell_gdf.index, cmap="tab20", edgecolor="black", linewidth=0.6)
    ax.set_axis_off()
    fig.savefig(os.path.join(out_dir, "10X.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Run pipeline
    data = build_graph(df_local, k=40, dist_threshold=2)

    # compute rankings to populate QC columns (e.g. 'drop') used downstream
    ranked_df = calculate_rankings(ranked_df, min_support=1)
    ranked_df = calculate_thresholds(ranked_df, min_support=1, top_ind=0)

    data = add_edge_prob_stats(data, ranked_df)
    G = to_networkx(data)
    G_pruned = prune_graph(
        G.copy(),
        ranked_df,
        distance_thresholds=(1.0, 2.0),
        directed=False,
        min_npmi_thresh=0.01,
        max_npmi_thresh=0.5,
        min_cond_prob_thresh=0.01,
        max_cond_prob_thresh=0.5,
        prune_npmi_thresh=0.2,
    )

    summary_df, M_cc, genes, node_to_component = purity_conflict_from_cc(
        G_pruned=G_pruned,
        npmi_long=pair_counts_df,
        df_local=df_local,
        purity_threshold=0.05,
        min_cc_size=15,
        return_matrix=True,
        return_node_mapping=True,
    )

    genes_all, gene_to_idx, npmi_mat, col_idx = build_npmi_matrix_from_long(pair_counts_df)

    summary_df_stitched, G_cc = stitch_connected_components(
        summary_df=summary_df,
        M_cc=M_cc,
        npmi_mat=npmi_mat,
        col_idx=col_idx,
        purity_threshold=0.05,
        penalize_simplicity=True,
        use_3d=True,
    )

    comp_to_stitched = dict(
        zip(summary_df_stitched["component_id"], summary_df_stitched["stitched_cc_id"])
    )

    node_to_stitched = {n: comp_to_stitched[node_to_component[n]] for n in node_to_component}

    fig, ax = plot_cc(
        df_local=df_local,
        G_pruned=G_pruned,
        cell_gdf=cell_gdf,
        node_to_group=node_to_stitched,
        group_name="Segmented Cell",
        s=10,
        alpha_pts=0.8,
        force_show_legend=False,
        alpha_edges=0.6,
        edge_lw=0.6,
        alpha_poly=0.15,
        linewidth_poly=0.6,
    )

    fig.savefig(os.path.join(out_dir, "v1_example.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", os.path.join(out_dir, "10X.png"), os.path.join(out_dir, "v1_example.png"))


if __name__ == "__main__":
    main()
