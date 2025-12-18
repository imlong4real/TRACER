import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def plot_cc(
    df_local,
    G_pruned,
    cell_gdf,
    *,
    node_to_group=None,          #e.g. stitched_cc_id
    group_name="CC",
    x_col="x",
    y_col="y",
    cell_id_col="cell_id",
    figsize=(10, 10),
    s=0.5,
    alpha_pts=0.7,
    alpha_edges=0.15,
    edge_lw=0.3,
    alpha_poly=0.15,
    linewidth_poly=0.6,
    max_legend_groups=20,
    force_show_legend=False,
    max_edges_draw=200_000,
    random_state=0,
):
    """
    Visualize transcript graph colored by connected components or stitched groups.

    Parameters
    ----------
    df_local : DataFrame
        Transcript-level table (index must align with graph node IDs).
    G_pruned : networkx.Graph
        Pruned transcript graph.
    cell_gdf : GeoDataFrame
        Cell geometry polygons indexed by cell_id.
    node_to_group : dict[int, int], optional
        Mapping from graph node → group ID (e.g. stitched_cc_id).
        If None, graph connected components are used.
    group_name : str
        Name used in title/legend (e.g. "CC", "Stitched CC").
    """

    rng = np.random.default_rng(random_state)

    # Determine node → group mapping
    if node_to_group is None:
        # fallback: connected components
        if G_pruned.is_directed():
            comps = nx.weakly_connected_components(G_pruned)
        else:
            comps = nx.connected_components(G_pruned)

        node_to_group = {}
        for gid, nodes in enumerate(comps):
            for n in nodes:
                node_to_group[n] = gid
        group_name = "CC"

    # Build plotting dataframe
    plot_df = df_local[[x_col, y_col, cell_id_col]].copy()
    plot_df["group_id"] = plot_df.index.map(node_to_group)
    plot_df = plot_df[plot_df["group_id"].notna()].copy()
    plot_df["group_id"] = plot_df["group_id"].astype(int)

    group_ids = sorted(plot_df["group_id"].unique())
    n_groups = len(group_ids)

    # Node → row index mapping (critical)
    node_to_row = {node_id: i for i, node_id in enumerate(plot_df.index)}

    # Color assignment
    if n_groups <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(n_groups)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / n_groups) for i in range(n_groups)]

    group_to_color = dict(zip(group_ids, colors))

    # Subset cell polygons to ROI
    roi_cell_ids = plot_df[cell_id_col].astype(str).unique()
    cell_gdf_sub = cell_gdf.loc[
        cell_gdf.index.astype(str).isin(roi_cell_ids)
    ]

    # Prepare edges
    edges = [
        (u, v)
        for u, v in G_pruned.edges()
        if u in node_to_row and v in node_to_row
    ]

    if len(edges) > max_edges_draw:
        edges = rng.choice(edges, size=max_edges_draw, replace=False)

    if len(edges) > 0:
        edges = np.asarray(edges, dtype=int)
        xy = plot_df[[x_col, y_col]].to_numpy()

        src = np.fromiter(
            (node_to_row[u] for u in edges[:, 0]),
            dtype=np.int64,
            count=len(edges),
        )
        tgt = np.fromiter(
            (node_to_row[v] for v in edges[:, 1]),
            dtype=np.int64,
            count=len(edges),
        )

        segments = np.stack([xy[src], xy[tgt]], axis=1)

        edge_colors = [
            group_to_color[node_to_group[u]] for u in edges[:, 0]
        ]

        edge_lc = LineCollection(
            segments,
            colors=edge_colors,
            linewidths=edge_lw,
            alpha=alpha_edges,
            rasterized=True,
        )
    else:
        edge_lc = None

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    if edge_lc is not None:
        ax.add_collection(edge_lc)

    for gid in group_ids:
        sub = plot_df[plot_df["group_id"] == gid]
        ax.scatter(
            sub[x_col].values,
            sub[y_col].values,
            s=s,
            alpha=alpha_pts,
            c=[group_to_color[gid]],
            linewidths=0,
            rasterized=True,
        )

    # cell boundaries
    cell_gdf_sub.boundary.plot(
        ax=ax,
        linewidth=linewidth_poly,
        edgecolor="black",
        alpha=alpha_poly,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(f"{group_name}s (n={n_groups})")

    # Legend
    show_legend = force_show_legend or (n_groups <= max_legend_groups)

    if show_legend:
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=group_to_color[g],
                markeredgewidth=0,
                label=f"{group_name} {g}",
            )
            for g in group_ids
        ]

        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            ncol=3 if (force_show_legend and n_groups > max_legend_groups) else 1,
        )

    plt.tight_layout()
    return fig, ax