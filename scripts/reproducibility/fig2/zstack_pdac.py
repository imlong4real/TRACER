#!/usr/bin/env python
"""3-D z-stack reconstruction view of the post-TRACER PDAC Xenium sample.

Two transcript "stacks" are rendered in a single 3-D scene:

  * lower stack — complete PDAC cells (etype == 0, original cells) at their
    native z depth, i.e. the intact reconstructed volume.
  * upper stack — TRACER-reconstructed *partial* cells (etype == 1), lifted
    above the lower stack so the rescued material reads as its own layer.

Every cell is given a unique, perceptually-uniform colour derived from its
``cell_id`` in CIELAB space (golden-angle hue walk at controlled L*/chroma,
converted via skimage ``lab2rgb``).  Dark background, Nature-style minimal
chrome.

Outputs (PNG + SVG, >=300 dpi) -> ``scripts/reproducibility/fig2/outputs``.

Run:
    python scripts/reproducibility/fig2/zstack_pdac.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)
from skimage.color import lab2rgb

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "datasets/pancreas_cancer_xenium_10x/pdac_io_partition_sequential.parquet"
OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

# spatial ROI (microns) — dense, balanced patch of complete + partial cells
ROI = dict(x0=4600, y0=3500, size=300)

BG = "#0a0d12"          # near-black background
PANEL = "#11161d"       # faint stack base panels
STACK_GAP = 1.55        # upper-stack lift as a multiple of the z span


# --------------------------------------------------------------------------- #
# Perceptually-uniform CIELAB palette
# --------------------------------------------------------------------------- #
def cielab_palette(n: int, seed: int = 7) -> np.ndarray:
    """`n` well-separated sRGB colours sampled in CIELAB via a golden-angle walk."""
    idx = np.arange(n)
    hue = (idx * 137.508) % 360.0                      # golden-angle hue spread
    L = 62.0 + 20.0 * np.sin(idx * 0.7)                # gentle L* modulation
    C = 42.0 + 12.0 * np.cos(idx * 1.3)                # chroma modulation
    rad = np.deg2rad(hue)
    lab = np.stack([L, C * np.cos(rad), C * np.sin(rad)], axis=1)
    rgb = lab2rgb(lab[None, :, :])[0]                  # (n,3) in [0,1]
    rng = np.random.default_rng(seed)
    rng.shuffle(rgb)                                   # decorrelate spatial order
    return np.clip(rgb, 0, 1)


def color_lookup(cell_ids: np.ndarray) -> dict[str, np.ndarray]:
    uniq = np.unique(cell_ids)
    pal = cielab_palette(len(uniq))
    return dict(zip(uniq, pal))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_roi() -> pd.DataFrame:
    df = pd.read_parquet(DATA, columns=["x", "y", "z", "cell_id", "etype_at_finalize"])
    x0, y0, s = ROI["x0"], ROI["y0"], ROI["size"]
    m = (df.x >= x0) & (df.x < x0 + s) & (df.y >= y0) & (df.y < y0 + s)
    df = df[m & df.etype_at_finalize.isin([0, 1])].copy()
    df["x"] -= x0
    df["y"] -= y0
    return df


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def scatter_stack(ax, sub, z_off, cmap, base_z, panel_extent):
    cols = np.array([cmap[c] for c in sub.cell_id.values])
    ax.scatter(
        sub.x.values, sub.y.values, sub.z.values + z_off,
        c=cols, s=7.0, alpha=0.95, depthshade=False,
        edgecolors="none", rasterized=True,
    )
    # faint base panel under the stack for a "slab" read
    xx, yy = np.meshgrid(np.linspace(0, panel_extent, 2), np.linspace(0, panel_extent, 2))
    ax.plot_surface(xx, yy, np.full_like(xx, base_z), color=PANEL,
                    alpha=0.55, shade=False, zorder=0)


def main():
    df = load_roi()
    cmap = color_lookup(df.cell_id.values)

    zmin, zmax = df.z.min(), df.z.max()
    zspan = zmax - zmin
    upper_off = zspan * STACK_GAP
    s = ROI["size"]

    orig = df[df.etype_at_finalize == 0]
    part = df[df.etype_at_finalize == 1]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": "#e8edf2",
        "axes.labelcolor": "#cfd6dd",
    })

    fig = plt.figure(figsize=(12.5, 12.0), facecolor=BG)
    # 3-D scene occupies the left ~74 %; a clean right gutter holds the labels
    ax = fig.add_axes([0.0, 0.04, 0.74, 0.92], projection="3d")
    ax.set_facecolor(BG)

    scatter_stack(ax, orig, 0.0, cmap, zmin - 0.06 * zspan, s)
    scatter_stack(ax, part, upper_off, cmap, zmin + upper_off - 0.06 * zspan, s)

    # ---- title block (figure-level, centred over the scene) ----
    scene_cx = 0.37
    fig.text(scene_cx, 0.965, "PDAC Xenium — TRACER z-stack reconstruction",
             ha="center", va="top", fontsize=19, color="#f2f5f8",
             fontweight="bold")
    fig.text(scene_cx, 0.928,
             f"300 µm ROI · {orig.cell_id.nunique()} complete + "
             f"{part.cell_id.nunique()} partial cells · "
             f"unique CIELAB colour per cell_id",
             ha="center", va="top", fontsize=11.5, color="#9aa6b2")

    # ---- slab labels in the right gutter, with thin leaders to each slab ----
    leader = dict(arrowstyle="-", color="#5d6875", lw=0.9,
                  connectionstyle="arc3,rad=-0.12")
    ax.annotate("TRACER-reconstructed\npartial cells",
                xy=(0.62, 0.64), xytext=(0.77, 0.66),
                xycoords="figure fraction", textcoords="figure fraction",
                ha="left", va="center", color="#e8eef4", fontsize=13,
                fontstyle="italic", linespacing=1.35, arrowprops=leader)
    ax.annotate("Complete cells",
                xy=(0.58, 0.31), xytext=(0.77, 0.31),
                xycoords="figure fraction", textcoords="figure fraction",
                ha="left", va="center", color="#e8eef4", fontsize=13,
                fontstyle="italic", arrowprops=leader)

    # minimal chrome
    ax.set_xlabel("x (µm)", fontsize=10, labelpad=2)
    ax.set_ylabel("y (µm)", fontsize=10, labelpad=2)
    ax.set_zticks([])
    ax.set_xlim(0, s)
    ax.set_ylim(0, s)
    ax.set_zlim(zmin - 0.1 * zspan, zmin + upper_off + 1.1 * zspan)
    ax.set_box_aspect((1, 1, 1.5))

    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_visible(False)
    ax.grid(False)
    ax.xaxis.line.set_color((1, 1, 1, 0.18))
    ax.yaxis.line.set_color((1, 1, 1, 0.18))
    ax.zaxis.line.set_color((1, 1, 1, 0.0))
    ax.tick_params(colors="#7d8893", labelsize=8)
    ax.view_init(elev=18, azim=-58)

    for ext in ("png", "svg"):
        fig.savefig(OUTDIR / f"pdac_zstack_reconstruction.{ext}", dpi=300,
                    facecolor=BG)
    plt.close(fig)
    print(f"wrote pdac_zstack_reconstruction.png / .svg  "
          f"({len(orig):,} + {len(part):,} transcripts)")


if __name__ == "__main__":
    main()
