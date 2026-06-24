#!/usr/bin/env python3
"""Shared loaders + plotting helpers for the Figure 4 pipeline.

All heavy spatial joins (TRACER bins -> H&E micron frame, 10x geojson
centroids) are cached to source_data/ as parquet so re-running individual
panels is cheap. Every loader prints the exact path it reads.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fig4_config as C


# ---------------------------------------------------------------------------
# Style + saving
# ---------------------------------------------------------------------------
def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.linewidth": 0.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    return plt


def save_fig(fig, name: str, dpi: int = 300):
    """Save a panel as both PNG and SVG into outputs/. Returns the paths."""
    paths = []
    for ext in ("png", "svg"):
        p = C.OUTDIR / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  [saved] {name}.png / .svg")
    return paths


def log(msg):
    print(f"[fig4] {msg}", flush=True)


# ---------------------------------------------------------------------------
# H&E
# ---------------------------------------------------------------------------
def load_he():
    """Return (image_array, extent=[0,xmax_um,ymax_um,0]) for imshow overlay."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = np.asarray(Image.open(C.HE_HIRES_PNG).convert("RGB"))
    xmax, ymax = C.he_micron_extent()
    return img, [0.0, xmax, ymax, 0.0]


_BTF = None


def _btf_page():
    global _BTF
    if _BTF is None:
        import tifffile
        _BTF = tifffile.TiffFile(C.HE_BTF)
    return _BTF, _BTF.series[0].pages[0]


def he_crop_um(x0_um, y0_um, x1_um, y1_um):
    """Decode a full-resolution H&E crop for a micron bounding box, reading
    only the tiles that intersect it (instant on the tiled BigTIFF).

    Returns (rgb_array, extent=[x0_um, x1_um, y1_um, y0_um]) for imshow with
    micron coordinates (origin upper)."""
    tf, p = _btf_page()
    tw, tl = p.tilewidth, p.tilelength
    H, W = p.imagelength, p.imagewidth
    across = (W + tw - 1) // tw
    # micron -> fullres pixel, clamped
    x0 = max(0, int(x0_um / C.MPP)); x1 = min(W, int(np.ceil(x1_um / C.MPP)))
    y0 = max(0, int(y0_um / C.MPP)); y1 = min(H, int(np.ceil(y1_um / C.MPP)))
    tx0, tx1 = x0 // tw, (x1 - 1) // tw
    ty0, ty1 = y0 // tl, (y1 - 1) // tl
    out = np.full(((ty1 - ty0 + 1) * tl, (tx1 - tx0 + 1) * tw, 3), 255, np.uint8)
    fh = tf.filehandle
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            idx = ty * across + tx
            cnt = p.databytecounts[idx]
            if cnt == 0:
                continue
            fh.seek(p.dataoffsets[idx]); data = fh.read(cnt)
            tile, _, shp = p.decode(data, idx)
            tile = np.asarray(tile).reshape(shp[-3], shp[-2], shp[-1])
            out[(ty - ty0) * tl:(ty - ty0 + 1) * tl,
                (tx - tx0) * tw:(tx - tx0 + 1) * tw] = tile[:, :, :3]
    ox, oy = tx0 * tw, ty0 * tl
    crop = out[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return crop, [x0 * C.MPP, x1 * C.MPP, y1 * C.MPP, y0 * C.MPP]


# ---------------------------------------------------------------------------
# Per-method spatial points in the common H&E-micron frame
# ---------------------------------------------------------------------------
def _tissue_positions(square_spatial: Path) -> pd.DataFrame:
    tp = pd.read_parquet(square_spatial / "tissue_positions.parquet")
    return tp.set_index("barcode")[["pxl_col_in_fullres", "pxl_row_in_fullres"]]


def _tracer_points(method: str) -> pd.DataFrame:
    """Each TRACER bin placed in H&E micron, colored by its profile lineage."""
    run = "kidney_visiumhd_2um" if method == "tracer_2um" else "kidney_visiumhd_8um"
    ba = pd.read_parquet(C.BIN_TO_PROFILE[method],
                         columns=["bin_id", "reconstructed_profile_id"])
    ba["reconstructed_profile_id"] = ba["reconstructed_profile_id"].astype(str)
    lab = pd.read_csv(C.LABELS[method], sep="\t",
                      usecols=["reconstructed_profile_id", "transferred_label"])
    lab["reconstructed_profile_id"] = lab["reconstructed_profile_id"].astype(str)
    tp = _tissue_positions(C.SPATIAL_DIR[method])
    m = ba.merge(lab, on="reconstructed_profile_id", how="left").join(tp, on="bin_id")
    m = m.dropna(subset=["pxl_col_in_fullres", "transferred_label"])
    return pd.DataFrame({
        "lineage": m["transferred_label"].to_numpy(),
        "mx": m["pxl_col_in_fullres"].to_numpy() * C.MPP,
        "my": m["pxl_row_in_fullres"].to_numpy() * C.MPP,
    })


def _bin2cell_points() -> pd.DataFrame:
    d = pd.read_csv(C.LABELS["bin2cell"], sep="\t",
                    usecols=["transferred_label", "centroid_x", "centroid_y"])
    d = d.dropna(subset=["transferred_label", "centroid_x"])
    return pd.DataFrame({"lineage": d["transferred_label"].to_numpy(),
                         "mx": d["centroid_x"].to_numpy(),
                         "my": d["centroid_y"].to_numpy()})


def _tenx_points() -> pd.DataFrame:
    annot = pd.read_csv(C.LABELS["10x_segmented"])
    annot["cid"] = annot["cell_id"].astype(str).str.extract(r"(\d+)").astype(int)
    cent = _tenx_centroids()                      # cid -> (x_fullres, y_fullres)
    annot = annot[annot["cid"].isin(cent.index)].copy()
    xy = cent.reindex(annot["cid"]).to_numpy()
    return pd.DataFrame({"lineage": annot["transferred_label"].to_numpy(),
                         "mx": xy[:, 0] * C.MPP, "my": xy[:, 1] * C.MPP})


def _tenx_centroids() -> pd.DataFrame:
    """Parse cell_segmentations.geojson once; cache cid->fullres centroid."""
    cache = C.SRCDIR / "tenx_cell_centroids_fullres.parquet"
    if cache.exists():
        return pd.read_parquet(cache).set_index("cid")[["x", "y"]]
    log(f"parsing {C.CELL_SEG_GEOJSON} (one-time)")
    gj = json.loads(C.CELL_SEG_GEOJSON.read_text())
    rows = []
    for f in gj["features"]:
        cid = int(f["properties"]["cell_id"])
        xy = np.asarray(f["geometry"]["coordinates"][0], float).mean(0)
        rows.append((cid, xy[0], xy[1]))
    df = pd.DataFrame(rows, columns=["cid", "x", "y"])
    df.to_parquet(cache)
    return df.set_index("cid")[["x", "y"]]


def tracer2um_bin_frame(cache: bool = True) -> pd.DataFrame:
    """Every TRACER 2µm bin with its profile id, lineage, and H&E-micron coords
    [profile_id, lineage, mx, my, bin_id]. Cached (needed for Panel G neighbours)."""
    cpath = C.SRCDIR / "tracer_2um_bins_he.parquet"
    if cache and cpath.exists():
        return pd.read_parquet(cpath)
    ba = pd.read_parquet(C.BIN_TO_PROFILE["tracer_2um"],
                         columns=["bin_id", "reconstructed_profile_id",
                                  "dominant_fraction", "n_tx_in_bin"])
    ba["reconstructed_profile_id"] = ba["reconstructed_profile_id"].astype(str)
    lab = pd.read_csv(C.LABELS["tracer_2um"], sep="\t",
                      usecols=["reconstructed_profile_id", "transferred_label"])
    lab["reconstructed_profile_id"] = lab["reconstructed_profile_id"].astype(str)
    tp = _tissue_positions(C.SPATIAL_DIR["tracer_2um"])
    m = ba.merge(lab, on="reconstructed_profile_id", how="left").join(tp, on="bin_id")
    m = m.dropna(subset=["pxl_col_in_fullres"])
    out = pd.DataFrame({
        "profile_id": m["reconstructed_profile_id"].to_numpy(),
        "lineage": m["transferred_label"].to_numpy(),
        "mx": m["pxl_col_in_fullres"].to_numpy() * C.MPP,
        "my": m["pxl_row_in_fullres"].to_numpy() * C.MPP,
        "bin_id": m["bin_id"].to_numpy(),
        # dominant_fraction < 1  ==>  this 2µm bin is shared by >1 reconstructed cell
        "dominant_fraction": m["dominant_fraction"].to_numpy(),
        "n_tx_in_bin": m["n_tx_in_bin"].to_numpy(),
    })
    if cache:
        out.to_parquet(cpath)
    return out


def method_points(method: str, cache: bool = True) -> pd.DataFrame:
    """DataFrame[lineage, mx, my] (H&E micron). Cached per method."""
    cpath = C.SRCDIR / f"points_{method}.parquet"
    if cache and cpath.exists():
        return pd.read_parquet(cpath)
    if method in ("tracer_2um", "tracer_8um"):
        df = _tracer_points(method)
    elif method == "bin2cell":
        df = _bin2cell_points()
    elif method == "10x_segmented":
        df = _tenx_points()
    else:
        raise ValueError(method)
    if cache:
        df.to_parquet(cpath)
    return df


# ---------------------------------------------------------------------------
# Label tables (per-unit lineage + counts)
# ---------------------------------------------------------------------------
def load_labels(method: str) -> pd.DataFrame:
    """Unified per-unit table: lineage, confidence, n_transcripts, n_genes,
    n_bins (NaN where not bin-derived)."""
    if method == "10x_segmented":
        d = pd.read_csv(C.LABELS[method])
        return pd.DataFrame({"lineage": d["transferred_label"],
                             "confidence": d["transfer_confidence"],
                             "n_transcripts": d["n_transcripts"],
                             "n_genes": d["n_genes_by_counts"],
                             "n_bins": np.nan})
    if method == "bin2cell":
        d = pd.read_csv(C.LABELS[method], sep="\t")
        return pd.DataFrame({"lineage": d["transferred_label"],
                             "confidence": d["transfer_confidence"],
                             "n_transcripts": d["n_transcripts"],
                             "n_genes": d.get("n_genes"),
                             "n_bins": np.nan})
    # tracer (WT labels carry n_bins / n_transcripts from HVG scores merge)
    d = pd.read_csv(C.LABELS[method], sep="\t")
    return pd.DataFrame({"lineage": d["transferred_label"],
                         "confidence": d["transfer_confidence"],
                         "n_transcripts": d["n_transcripts"],
                         "n_genes": d["n_genes"],
                         "n_bins": d["n_bins"]})


# ---------------------------------------------------------------------------
# Whole-transcriptome matrices + reference pseudobulk
# ---------------------------------------------------------------------------
def load_wt(method: str, backed=True):
    import anndata as ad
    return ad.read_h5ad(C.WT_H5AD[method], backed="r" if backed else None)


def qc_table(method: str) -> pd.DataFrame:
    """Per-cell QC metrics (cell_id index): n_genes, n_umis, n_bins, from the
    whole-transcriptome matrix obs. n_bins is NaN for 10x segmented cells."""
    import anndata as ad
    a = ad.read_h5ad(C.WT_H5AD[method], backed="r")
    df = pd.DataFrame({
        "n_genes": np.asarray(a.obs["n_genes"], float),
        "n_umis": np.asarray(a.obs["total_counts"], float),
        "n_bins": np.asarray(a.obs["n_bins"], float),
    }, index=a.obs_names.astype(str))
    return df


def qc_pass_ids(method: str, min_genes: int, min_umis: int, min_bins: int) -> set:
    """cell_ids passing QC. The n_bins criterion is skipped where n_bins is NaN
    (10x segmented cells are not bin-derived)."""
    q = qc_table(method)
    ok = (q["n_genes"] >= min_genes) & (q["n_umis"] >= min_umis)
    bins_ok = q["n_bins"].isna() | (q["n_bins"] >= min_bins)
    ok = ok & bins_ok
    return set(q.index[ok])


def lineage_mean_expression(method: str, genes: list[str], normalize=True,
                            keep_ids: set | None = None) -> pd.DataFrame:
    """Mean (CP10k-normalized) expression of `genes` per lineage for a method.

    Uses the whole-transcriptome matrix + that method's lineage labels. If
    `keep_ids` is given, only those cell_ids are used (QC filtering).
    Returns DataFrame genes x lineage.
    """
    import anndata as ad
    a = ad.read_h5ad(C.WT_H5AD[method])
    if keep_ids is not None:
        mask = a.obs_names.astype(str).isin(keep_ids)
        a = a[mask].copy()
    lab = _wt_labels(method).reindex(a.obs_names)
    genes = [g for g in genes if g in set(a.var_names)]
    sub = a[:, genes]
    import scipy.sparse as sp
    X = sub.X
    X = sp.csr_matrix(X).astype(np.float64)
    if normalize:
        tot = np.asarray(a.X.sum(1)).ravel().astype(np.float64)
        tot[tot == 0] = 1.0
        X = X.multiply(1.0 / tot[:, None]).tocsr() * 1e4
        X = X.log1p() if hasattr(X, "log1p") else _log1p_sparse(X)
    df = pd.DataFrame(X.toarray(), index=a.obs_names, columns=genes)
    df["lineage"] = lab.to_numpy()
    out = df.dropna(subset=["lineage"]).groupby("lineage")[genes].mean().T
    return out


def _log1p_sparse(X):
    X = X.tocsr().copy()
    X.data = np.log1p(X.data)
    return X


def _wt_labels(method: str) -> pd.Series:
    """obs_name -> lineage for a method's WT matrix."""
    if method == "10x_segmented":
        d = pd.read_csv(C.LABELS[method])
        return d.set_index(d["cell_id"].astype(str))["transferred_label"]
    if method == "bin2cell":
        d = pd.read_csv(C.LABELS[method], sep="\t")
        return d.set_index(d["cell_id"].astype(str))["transferred_label"]
    d = pd.read_csv(C.LABELS[method], sep="\t")
    return d.set_index(d["reconstructed_profile_id"].astype(str))["transferred_label"]


def reference_pseudobulk(normalize=True) -> pd.DataFrame:
    """scRNA reference per-lineage pseudobulk (CP10k log1p mean). genes x lineage.
    Cached to source_data."""
    cache = C.SRCDIR / "reference_pseudobulk.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import anndata as ad
    import scipy.sparse as sp
    a = ad.read_h5ad(C.REFERENCE_H5AD)
    X = a.layers["counts"] if "counts" in a.layers else a.X
    X = sp.csr_matrix(X).astype(np.float64)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1.0
    Xn = X.multiply(1.0 / tot[:, None]).tocsr() * 1e4
    Xn.data = np.log1p(Xn.data)
    # Sparse group-mean: indicator (lineage x cells) @ Xn / n_per_lineage.
    lin = pd.Categorical(a.obs["lineage"].astype(str))
    ind = sp.csr_matrix((np.ones(len(lin)), (lin.codes, np.arange(len(lin)))),
                        shape=(len(lin.categories), len(lin)))
    sums = ind @ Xn                                    # lineages x genes
    counts = np.asarray(ind.sum(1)).ravel()
    means = np.asarray(sums.todense()) / counts[:, None]
    pb = pd.DataFrame(means.T, index=a.var_names, columns=list(lin.categories))
    pb.to_parquet(cache)
    return pb


def load_rctd(method: str) -> pd.DataFrame | None:
    """RCTD per-cell: cell_id, dominant_celltype, max_weight, entropy.
    Returns None if the run is missing (panel falls back)."""
    p = C.RCTD_ASSIGN[method]
    if not p.exists():
        return None
    d = pd.read_csv(p, sep="\t")
    d["dominant_celltype"] = d["dominant_celltype"].replace(C.RCTD_LABEL_FIX)
    return d


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def half_violin(ax, data, positions, colors, side="right", width=0.7,
                show_median=True, log=False):
    """Raincloud-style half violins with a median tick. `data` is a list of
    1-D arrays aligned with `positions`/`colors`."""
    import numpy as _np
    clean = [_np.asarray(d, float) for d in data]
    clean = [d[_np.isfinite(d)] for d in clean]
    if log:
        clean = [_np.log10(d[d > 0] + 1.0) for d in clean]
    parts = ax.violinplot(clean, positions=positions, showextrema=False, widths=width)
    for b, pos, col in zip(parts["bodies"], positions, colors):
        v = b.get_paths()[0].vertices
        if side == "right":
            v[:, 0] = _np.clip(v[:, 0], pos, _np.inf)
        else:
            v[:, 0] = _np.clip(v[:, 0], -_np.inf, pos)
        b.set_facecolor(col); b.set_alpha(0.8); b.set_edgecolor("k"); b.set_linewidth(0.4)
    if show_median:
        for d, pos in zip(clean, positions):
            if len(d):
                m = _np.median(d)
                ax.plot([pos - 0.18, pos + 0.02], [m, m], color="k", lw=1.3, zorder=6)
    return clean


def per_profile_counts(method: str) -> pd.DataFrame:
    """obs-level n_genes and total_counts (UMIs) per unit for a method's WT matrix."""
    import anndata as ad
    a = ad.read_h5ad(C.WT_H5AD[method], backed="r")
    return pd.DataFrame({"n_genes": np.asarray(a.obs["n_genes"]),
                         "n_umis": np.asarray(a.obs["total_counts"])})


def lineage_handles(lineages=None):
    """Matplotlib legend handles for the shared lineage palette."""
    import matplotlib.lines as mlines
    lineages = lineages or C.LINEAGES
    return [mlines.Line2D([], [], marker="o", linestyle="", markersize=5,
                          color=C.PALETTE[l], label=C.LINEAGE_DISPLAY[l])
            for l in lineages]
