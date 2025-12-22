import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d

def convex_hull_faces_3d(points):
    """
    points: (N,3)
    Returns list of triangular faces or None.
    """
    if points.shape[0] < 4:
        return None
    try:
        hull = ConvexHull(points)
    except Exception:
        return None
    return [points[s] for s in hull.simplices]

def plot_3d_surface(
    ax,
    points,
    faces,
    *,
    color="tab:blue",
    alpha=0.25,
    point_size=2,
    elev=30,
    azim=45,
    xlim=None,
    ylim=None,
):
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=point_size, alpha=0.6
    )

    if faces is not None:
        poly = Poly3DCollection(
            faces,
            facecolor=color,
            edgecolor="k",
            linewidths=0.2,
            alpha=alpha,
        )
        ax.add_collection3d(poly)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

def plot_3d_convex_cell(
    df,
    *,
    cell_id,
    cell_col="cell_id",
    stitched_col="cell_id_stitched",
    coord_cols=("x", "y", "z"),
    views=((90, -90), (0, -90), (30, 45)),
    out_png="cell_3d_comparison.png",
):
    """
    Returns a 3x3 subplot:
      Row 1: original segmentation (cell_id)
      Row 2: refined core (cell_id_stitched == cell_id)
      Row 3: refined partial (cell_id-1), if exists
    Columns: three viewpoints
    """

    df = df.copy()
    cell_id = str(cell_id)

    # ---------- Original segmentation ----------
    df_orig = df[df[cell_col].astype(str) == cell_id]
    pts_orig = df_orig[list(coord_cols)].to_numpy(float)

    faces_orig = convex_hull_faces_3d(pts_orig)

    # record original XY bounds
    xlim = (pts_orig[:, 0].min(), pts_orig[:, 0].max())
    ylim = (pts_orig[:, 1].min(), pts_orig[:, 1].max())

    # ---------- Refined core ----------
    df_core = df[df[stitched_col].astype(str) == cell_id]
    pts_core = df_core[list(coord_cols)].to_numpy(float)
    faces_core = convex_hull_faces_3d(pts_core)

    # ---------- Refined partial (cell_id-1) ----------
    partial_id = f"{cell_id}-1"
    df_partial = df[df[stitched_col].astype(str) == partial_id]
    has_partial = len(df_partial) > 0

    if has_partial:
        pts_partial = df_partial[list(coord_cols)].to_numpy(float)
        faces_partial = convex_hull_faces_3d(pts_partial)

    # ---------- Plot ----------
    fig = plt.figure(figsize=(15, 12))

    row_titles = [
        f"Original cell {cell_id}",
        "Refined core",
        "Refined partial",
    ]

    for r in range(3):
        for c, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(3, 3, r * 3 + c + 1, projection="3d")

            if r == 0:
                plot_3d_surface(
                    ax, pts_orig, faces_orig,
                    color="tab:gray", elev=elev, azim=azim,
                    xlim=xlim, ylim=ylim
                )
            elif r == 1:
                plot_3d_surface(
                    ax, pts_core, faces_core,
                    color="tab:blue", elev=elev, azim=azim,
                    xlim=xlim, ylim=ylim
                )
            else:
                if not has_partial:
                    ax.set_visible(False)
                    continue
                plot_3d_surface(
                    ax, pts_partial, faces_partial,
                    color="tab:orange", elev=elev, azim=azim,
                    xlim=xlim, ylim=ylim
                )

            if c == 0:
                ax.set_title(row_titles[r])

    fig.suptitle(
        f"Cell {cell_id}: Original vs Refined Core- vs Partial-Cell",
        y=0.95,
        fontsize=14
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_png

#
def estimate_alpha_auto(points, k=20, scale=1.8):
    """
    Estimate alpha from median kNN distance.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    dists, _ = nbrs.kneighbors(points)
    d = np.median(dists[:, 1:])  # exclude self
    return scale * d

def alpha_shape_mesh(points, k=20, scale=1.8):
    """
    Build concave hull mesh with automatic alpha.

    Parameters
    ----------
    k : int
        Number of neighbors used by the kNN estimator for alpha.
    scale : float
        Scale factor applied to the median kNN distance to get alpha.
    """
    alpha = estimate_alpha_auto(points, k=k, scale=scale)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha
    )
    mesh.compute_vertex_normals()
    return mesh, alpha


def plot_mesh_matplotlib(ax, mesh, color="tab:blue", alpha=0.35):
    """
    Render Open3D mesh in matplotlib 3D axis.
    """
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    faces = verts[tris]

    poly = Poly3DCollection(
        faces, facecolor=color, edgecolor="k", linewidths=0.15, alpha=alpha
    )
    ax.add_collection3d(poly)


def plot_3d_concave_cell(
    df,
    cell_id,
    *,
    coord_cols=("x","y","z"),
    id_orig="cell_id",
    id_stitched="cell_id_stitched",
    k=20,
    scale=1.8,
    out_png="cell_124838_refinement_3x3.png"
):
    """
    Publication-style 3×3 figure using concave hulls.
    """
    cid = str(cell_id)
    cid_partial = f"{cid}-1"

    # -------- extract data --------
    df = df.copy()
    df[id_orig] = df[id_orig].astype(str)
    df[id_stitched] = df[id_stitched].astype(str)

    rows = [
        ("Original segmentation", df[df[id_orig] == cid]),
        ("Refined core cell",     df[df[id_stitched] == cid]),
        ("Partial cell",          df[df[id_stitched] == cid_partial]),
    ]

    # Compute XY range from ORIGINAL segmentation
    xy = rows[0][1][["x","y"]].to_numpy()
    xmin, xmax = xy[:,0].min(), xy[:,0].max()
    ymin, ymax = xy[:,1].min(), xy[:,1].max()

    views = [(90,-90), (0,-90), (30,45)]

    fig = plt.figure(figsize=(12, 12))

    for r, (label, sub) in enumerate(rows):
        if sub.shape[0] < 10:
            continue

        pts = sub[list(coord_cols)].to_numpy(float)
        mesh, alpha_used = alpha_shape_mesh(pts, k=k, scale=scale)

        for c, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(3, 3, r*3 + c + 1, projection="3d")

            ax.scatter(
                pts[:,0], pts[:,1], pts[:,2],
                s=2, alpha=0.5
            )
            plot_mesh_matplotlib(ax, mesh)

            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            # z scaling
            zc = pts[:,2].mean()
            zr = (pts[:,2].max() - pts[:,2].min()) / 2
            ax.set_zlim(zc - zr, zc + zr)

            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

            if c == 0:
                ax.set_ylabel(label, fontsize=11)
            if r == 0:
                ax.set_title(f"View {c+1}", fontsize=11)

    fig.suptitle(
        f"Cell {cid}: Original vs Refined Core- vs Partial-Cell",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    return out_png
#
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