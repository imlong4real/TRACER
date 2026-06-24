"""Shared dark-background Nature-style theme for all Figure 2 panels.

Keeps the alluvial / z-stack aesthetic: deep charcoal canvas, light ink,
vivid but balanced palettes, minimal chrome.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- canvas ----
BG = "#0c1016"          # deep charcoal-navy
PANEL = "#11161d"       # faint panel fill
INK = "#eef3f8"         # primary light text
INK_SOFT = "#9fb0c0"    # secondary text
RULE = "#243140"        # hairlines / separators
GRID = "#1c2530"

# ---- 10 atlas cell types (consistent colour across every panel) ----
CELLTYPE_COLORS = {
    "Ductal cell type 1": "#4c9bff",   # azure
    "Ductal cell type 2": "#ff9e3d",   # amber
    "Acinar cell":        "#3ddc84",   # emerald
    "Endocrine cell":     "#ff5d5d",   # red
    "T cell":             "#b487ff",   # violet
    "B cell":             "#c0875a",   # tan
    "Macrophage cell":    "#ff8ad8",   # pink
    "Endothelial cell":   "#29e0e0",   # cyan
    "Fibroblast cell":    "#d4d447",   # olive
    "Stellate cell":      "#9aa7b4",   # slate
}
# entity-class colours (match alluvial)
ENTITY_COLORS = {
    "original":   "#4c9bff",
    "complete":   "#4c9bff",
    "partial":    "#3ddc84",
    "neighboring": "#ffb02e",
}

# sequential colormap for expression / scores on dark bg
SEQ_CMAP = "magma"


def use_dark():
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": RULE,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
        "figure.dpi": 120,
    })


def style_ax(ax, spines=("left", "bottom")):
    ax.set_facecolor(BG)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines)
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def save(fig, path_stem, dpi=300):
    for ext in ("png", "svg"):
        fig.savefig(f"{path_stem}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
