#!/usr/bin/env python
"""Figure 2 — 3D ROI: TRACER reconstructs partial cells & new cross-type contacts.

Open3D builds a smooth cell-like surface for every cell with enough transcripts
(convex hull of its point cloud, Loop-subdivided + Laplacian-smoothed). Cells
with too few / coplanar transcripts cannot form a hull and fall back to a small
ellipsoid; these sparse/degenerate cells are rendered de-emphasised (small,
low-alpha, no white edge) and filtered below a transcript-count threshold, so
they no longer appear as large bright "white ovals".

Meshes are rendered on a dark canvas with matplotlib 3D (Open3D's filament
offscreen renderer is unavailable headless on macOS).

Per-cell diagnostics are exported to outputs/fig2_3d_roi_cells.csv.

Run:
    python scripts/reproducibility/fig2/build_3d_roi.py
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig2_style as S

ROOT = Path(__file__).resolve().parents[3]
DSET = ROOT / "datasets/pancreas_cancer_xenium_10x"
FIG2 = DSET / "processed/fig2"
TRANS = DSET / "Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs/transcripts.parquet"
PART = DSET / "pdac_io_partition_sequential.parquet"
OUT = Path(__file__).resolve().parent / "outputs"

W = 80.0                 # ROI window (µm)
IMMUNE = {"T cell", "Macrophage cell", "B cell", "Endothelial cell"}
MIN_TX_CELL = 3          # minimum transcripts to consider a cell at all
HULL_MIN_PTS = 6         # >= this many non-coplanar points -> smooth hull
FALLBACK_RENDER_MIN_TX = 5   # fallback cells below this are filtered from the figure


def pick_roi(at):
    """Window maximising confident partial immune cells embedded near ductal cells."""
    o = at.obs
    part_imm = o[(o.entity_class == "partial") & (o.lt_conf >= 0.5) & o.cell_type.isin(IMMUNE)]
    duct = o[(o.cell_type.str.startswith("Ductal")) & (o.lt_conf >= 0.5)]
    pi = part_imm[["centroid_x", "centroid_y"]].values
    du = duct[["centroid_x", "centroid_y"]].values
    best = None
    xs = np.arange(o.centroid_x.min(), o.centroid_x.max() - W, 60)
    ys = np.arange(o.centroid_y.min(), o.centroid_y.max() - W, 60)
    for x0 in xs:
        for y0 in ys:
            np_ = ((pi[:, 0] >= x0) & (pi[:, 0] < x0+W) & (pi[:, 1] >= y0) & (pi[:, 1] < y0+W)).sum()
            nd = ((du[:, 0] >= x0) & (du[:, 0] < x0+W) & (du[:, 1] >= y0) & (du[:, 1] < y0+W)).sum()
            score = min(np_, 8) + 0.05 * min(nd, 60) if (np_ >= 4 and nd >= 8) else -1
            if best is None or score > best[0]:
                best = (score, x0, y0, np_, nd)
    return best


def cell_mesh(pts):
    """Smooth cell-like surface from a transcript point cloud via Open3D.

    Returns (vertices, triangles, fallback) where fallback=True means the point
    cloud was too sparse/coplanar for a convex hull and a small ellipsoid was
    substituted.
    """
    pts_u = np.unique(pts, axis=0)
    degenerate = (len(pts_u) < HULL_MIN_PTS or
                  np.linalg.matrix_rank(pts_u - pts_u.mean(0)) < 3)
    if degenerate:
        std = pts_u.std(0) if len(pts_u) > 1 else np.array([2.0, 2.0, 1.0])
        radii = np.clip(std * 1.5, 1.0, 3.0)
        m = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=6)
        V = np.asarray(m.vertices) * radii + pts_u.mean(0)
        return V, np.asarray(m.triangles), True
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_u))
    hull, _ = pc.compute_convex_hull()
    hull = hull.subdivide_loop(number_of_iterations=1)
    hull = hull.filter_smooth_laplacian(number_of_iterations=8, lambda_filter=0.5)
    hull.compute_vertex_normals()
    return np.asarray(hull.vertices), np.asarray(hull.triangles), False


def shade(color, normals, light=np.array([0.3, 0.4, 1.0]), gain=1.0):
    light = light / np.linalg.norm(light)
    b = (0.45 + 0.55 * np.clip(normals @ light, 0, 1)) * gain
    return np.clip(np.array(plt.matplotlib.colors.to_rgb(color))[None] * b[:, None], 0, 1)


def _face_normals(tris):
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm = np.linalg.norm(fn, axis=1, keepdims=True); nrm[nrm == 0] = 1
    return fn / nrm


def render(ax, cells, zscale, title, highlight=None):
    ax.set_facecolor(S.BG)
    order = sorted(range(len(cells)), key=lambda i: -cells[i]["cz"])  # far -> near
    for i in order:
        c = cells[i]
        V = c["V"].copy(); V[:, 2] *= zscale
        tris = V[c["T"]]
        fn = _face_normals(tris)
        if c["fallback"]:
            # de-emphasised: dim, semi-transparent ellipsoid, no white edge
            fc = shade(c["color"], fn, gain=0.7)
            pc = Poly3DCollection(tris, facecolors=np.c_[fc, np.full(len(fc), 0.28)],
                                  edgecolors=(1, 1, 1, 0.0), linewidths=0.0)
        elif c["partial"]:
            fc = shade(c["color"], fn)
            pc = Poly3DCollection(tris, facecolors=np.c_[fc, np.full(len(fc), 0.85)],
                                  edgecolors=(1, 1, 1, 0.22), linewidths=0.3)
        else:  # complete hull
            fc = shade(c["color"], fn)
            pc = Poly3DCollection(tris, facecolors=np.c_[fc, np.full(len(fc), 0.42)],
                                  edgecolors=(1, 1, 1, 0.04), linewidths=0.08)
        ax.add_collection3d(pc)
    if highlight:
        for (p, q) in highlight:
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2]*zscale, q[2]*zscale],
                    color="#ffffff", lw=1.3, ls=(0, (2, 2)), alpha=0.8, zorder=10)
    ax.set_title(title, color=S.INK, fontsize=13, fontweight="bold", pad=2, loc="left")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
    ax.grid(False)
    ax.view_init(elev=34, azim=-62)


def make_cells(df, idcol, ctmap, source, clsmap=None):
    """Build per-cell meshes + diagnostics records."""
    cells = []
    for cid, g in df.groupby(idcol):
        if cid in ("-1", "UNASSIGNED", "", "nan") or len(g) < MIN_TX_CELL:
            continue
        ct = ctmap.get(cid, None)
        if ct is None or isinstance(ct, float):
            continue
        pts = g[["x", "y", "z"]].values.astype(float)
        V, T, fallback = cell_mesh(pts)
        cells.append(dict(
            source=source, id=str(cid), cell_type=ct,
            entity_class=(clsmap.get(cid) if clsmap else "original"),
            n_transcripts=int(len(g)), fallback=bool(fallback),
            partial=(clsmap.get(cid) == "partial") if clsmap else False,
            color=S.CELLTYPE_COLORS.get(ct, "#888888"),
            centroid=pts.mean(0), cz=pts[:, 1].mean(), V=V, T=T))
    return cells


def finalize(cells):
    """Assign 3D depth rank and the 'rendered' flag (de-emphasis/filter rule)."""
    order = sorted(range(len(cells)), key=lambda i: -cells[i]["cz"])
    for rank, i in enumerate(order):
        cells[i]["rank_3d"] = rank
    for c in cells:
        c["rendered"] = (not c["fallback"]) or (c["n_transcripts"] >= FALLBACK_RENDER_MIN_TX)
    return cells


def diag_frame(cells):
    rows = []
    for c in cells:
        cx, cy, cz = c["centroid"]
        rows.append(dict(source=c["source"], id=c["id"], cell_type=c["cell_type"],
                         entity_class=c["entity_class"], n_transcripts=c["n_transcripts"],
                         rank_3d=c["rank_3d"], fallback=c["fallback"], rendered=c["rendered"],
                         centroid_x=round(cx, 2), centroid_y=round(cy, 2),
                         centroid_z=round(cz, 2), color=c["color"]))
    return pd.DataFrame(rows)


def main():
    at = sc.read_h5ad(FIG2 / "tracer_annotated.h5ad")
    ao = sc.read_h5ad(FIG2 / "original_annotated.h5ad")
    score, x0, y0, npart, nduct = pick_roi(at)
    print(f"ROI x0={x0:.0f} y0={y0:.0f} W={W:.0f} | partial-immune={npart} ductal={nduct}")

    tx = pd.read_parquet(TRANS, columns=["x_location", "y_location", "z_location",
                                         "cell_id", "qv"])
    lab = pd.read_parquet(PART, columns=["label"])["label"].values
    m = ((tx.x_location.values >= x0) & (tx.x_location.values < x0+W) &
         (tx.y_location.values >= y0) & (tx.y_location.values < y0+W) &
         (tx.qv.values >= 20))
    sub = tx[m].copy(); sub["label"] = lab[m]
    sub["x"] = sub.x_location - x0; sub["y"] = sub.y_location - y0; sub["z"] = sub.z_location
    print(f"  ROI transcripts: {len(sub):,}")

    ct_o = ao.obs["cell_type"].to_dict()
    ct_t = at.obs["cell_type"].to_dict()
    cls_t = at.obs["entity_class"].to_dict()

    orig_cells = finalize(make_cells(sub, "cell_id", ct_o, "original"))
    trac = sub[sub.label.isin(set(cls_t))]
    trac_cells = finalize(make_cells(trac, "label", ct_t, "tracer", cls_t))

    # ---- diagnostics export + console log ----
    diag = pd.concat([diag_frame(orig_cells), diag_frame(trac_cells)], ignore_index=True)
    diag.to_csv(OUT / "fig2_3d_roi_cells.csv", index=False)
    for name, cc in [("original", orig_cells), ("tracer", trac_cells)]:
        fb = [c for c in cc if c["fallback"]]
        drop = [c for c in cc if not c["rendered"]]
        print(f"  {name}: {len(cc)} cells | hull {sum(not c['fallback'] for c in cc)} | "
              f"fallback {len(fb)} | filtered(<{FALLBACK_RENDER_MIN_TX}tx) {len(drop)} | "
              f"partial {sum(c['partial'] for c in cc)}")
    print("  diagnostics -> outputs/fig2_3d_roi_cells.csv")
    print("  fallback cells by type:\n",
          diag[diag.fallback].groupby("cell_type").size().to_string())

    draw_o = [c for c in orig_cells if c["rendered"]]
    draw_t = [c for c in trac_cells if c["rendered"]]

    # highlight: rendered partial immune cell <-> nearest complete ductal cell
    highlight = []
    duct = [c for c in draw_t if c["cell_type"].startswith("Ductal") and not c["partial"]]
    for c in draw_t:
        if c["partial"] and c["cell_type"] in IMMUNE and duct:
            d = min(duct, key=lambda q: np.linalg.norm(q["centroid"][:2] - c["centroid"][:2]))
            if np.linalg.norm(d["centroid"][:2] - c["centroid"][:2]) < 25:
                highlight.append((c["centroid"], d["centroid"]))

    S.use_dark()
    zscale = 1.7
    fig = plt.figure(figsize=(15.5, 7.6), facecolor=S.BG)
    axA = fig.add_subplot(121, projection="3d"); axA.set_box_aspect((1, 1, 0.6))
    axB = fig.add_subplot(122, projection="3d"); axB.set_box_aspect((1, 1, 0.6))
    render(axA, draw_o, zscale, "a   Original 10x segmentation")
    render(axB, draw_t, zscale, "b   TRACER (complete + reconstructed partial)", highlight)
    for ax in (axA, axB):
        ax.set_xlim(0, W); ax.set_ylim(0, W); ax.set_zlim(0, sub.z.max()*zscale)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg = [Patch(facecolor=S.CELLTYPE_COLORS[t], label=t.replace(" cell", ""))
           for t in ["Ductal cell type 2", "Macrophage cell", "T cell", "B cell",
                     "Endothelial cell", "Fibroblast cell", "Stellate cell"]]
    leg += [Line2D([0], [0], marker="o", color="none", markerfacecolor="#888",
                   markeredgecolor="none", alpha=0.4, markersize=8,
                   label="sparse partial (fallback)"),
            Line2D([0], [0], color="w", lw=1.3, ls=(0, (2, 2)),
                   label="new immune–tumour contact")]
    fig.legend(handles=leg, loc="lower center", ncol=9, frameon=False,
               fontsize=9, labelcolor=S.INK, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("TRACER reconstructs partial cells and exposes new immune–tumour proximities",
                 color=S.INK, fontsize=14.5, fontweight="bold", y=1.0)
    fig.text(0.5, 0.95, f"{W:.0f} µm ROI · {sum(c['partial'] for c in draw_t)} "
             f"reconstructed partial cells shown · z exaggerated {zscale:.1f}×",
             ha="center", color=S.INK_SOFT, fontsize=10.5)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.82, bottom=0.08, wspace=0.0)
    S.save(fig, str(OUT / "fig2_3d_roi"))
    print("wrote fig2_3d_roi.png/.svg")


if __name__ == "__main__":
    main()
