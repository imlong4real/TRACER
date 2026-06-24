#!/usr/bin/env python
"""Transcript-fate alluvial diagrams for the post-TRACER PDAC Xenium sample.

Reproduces Figure 2 alluvial panels:

  * Tier A  — coarse TRACER phases:  Prune -> Group -> Stitch
  * Tier B  — full detailed phase ladder

Each transcript carries an entity-class label at every TRACER phase
(``etype_at_<phase>``).  The codes map to:

    0 -> original cell      1 -> partial
    3 -> unassigned         5 -> neighboring cell

The diagram tracks how transcripts flow between these classes from one phase
to the next.  Node columns sit at each phase state; the *transition* labels
(Prune / Group / ...) sit over the ribbons that connect adjacent states.

Outputs (PNG + SVG, >=300 dpi) land in ``scripts/reproducibility/fig2/outputs``.

Run:
    python scripts/reproducibility/fig2/alluvial_pdac.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.collections import PolyCollection

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "datasets/pancreas_cancer_xenium_10x/pdac_io_partition_sequential.parquet"
OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

# etype code -> entity class
CLASS_OF = {0: "original cell", 1: "partial", 3: "unassigned", 5: "neighboring cell"}

# fixed top -> bottom slot order
CLASS_ORDER = ["neighboring cell", "original cell", "partial", "unassigned"]

# --------------------------------------------------------------------------- #
# Dark, high-contrast "Nature-style" theme
# --------------------------------------------------------------------------- #
BG = "#0c1016"            # deep charcoal-navy canvas
INK = "#eef3f8"           # primary light text
INK_SOFT = "#9fb0c0"      # secondary / subtitle text
RULE = "#243140"          # subtle separator / hairlines

# vivid, well-separated entity-class palette tuned for a dark canvas
PALETTE = {
    "neighboring cell": "#ffb02e",  # amber
    "original cell": "#4c9bff",     # azure
    "partial": "#3ddc84",           # emerald
    "unassigned": "#7e8ca0",        # slate
}

RIBBON_ALPHA = 0.58      # flow opacity over the dark canvas
GRADIENT = True          # blend ribbon colour source -> target along the flow

# node-column geometry (axis fraction units)
NODE_W = 0.018        # half is drawn either side of the column centre
SLOT_GAP = 0.013      # vertical gap between class slots within a column
N_POINTS = 96         # samples per ribbon edge


def _lighten(hex_color: str, amt: float) -> tuple[float, float, float]:
    """Blend a colour toward white by ``amt`` (0..1)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amt, g + (1 - g) * amt, b + (1 - b) * amt)

# Tier A: (state label used internally, parquet phase suffix)
TIER_A_COLS = ["input", "phase1", "group", "stitch"]
TIER_A_LABELS = ["Prune", "Group", "Stitch"]

TIER_B_COLS = [
    "input", "phase1", "rescue", "group",
    "post_group_rescue", "stitch", "demote", "final_rescue",
]
TIER_B_LABELS = [
    "Prune", "Rescue", "Group", "Post-Group Rescue",
    "Stitch", "Demote", "Final Rescue",
]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_class_frame(phases: list[str]) -> tuple[pd.DataFrame, int]:
    """Return a frame of per-transcript entity classes for the given phases."""
    cols = [f"etype_at_{p}" for p in phases]
    raw = pd.read_parquet(DATA, columns=cols)
    out = pd.DataFrame(index=raw.index)
    for p, c in zip(phases, cols):
        out[p] = raw[c].map(CLASS_OF).astype("category")
    return out, len(out)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def slot_bounds(counts: dict[str, int], total: int) -> dict[str, tuple[float, float]]:
    """Top/bottom y for each (always 4) class slot, stacked from y=1 downward.

    Slots are reserved for absent classes too, so a class always occupies the
    same vertical band across every column.
    """
    usable = 1.0 - SLOT_GAP * (len(CLASS_ORDER) - 1)
    bounds: dict[str, tuple[float, float]] = {}
    y = 1.0
    for cls in CLASS_ORDER:
        h = counts.get(cls, 0) / total * usable
        bounds[cls] = (y, y - h)  # (top, bottom)
        y -= h + SLOT_GAP
    return bounds


def draw_ribbon(ax, x0, x1, src, dst, c_src, c_dst):
    """Smooth ribbon between source band ``src`` and target band ``dst``.

    ``src``/``dst`` are (top, bottom) tuples in axis-y units.  The ribbon is
    built from per-segment quads whose colour blends from the source class to
    the target class along the flow, so a transcript's fate change is legible
    in the ribbon itself.
    """
    t = np.linspace(0.0, 1.0, N_POINTS)
    ease = (1.0 - np.cos(t * np.pi)) / 2.0  # smooth ease-in-out
    xs = x0 + (x1 - x0) * t
    top = src[0] + (dst[0] - src[0]) * ease
    bot = src[1] + (dst[1] - src[1]) * ease

    rgb0 = np.array(mcolors.to_rgb(c_src))
    rgb1 = np.array(mcolors.to_rgb(c_dst))
    # colour blend weighted toward the source for most of the run, easing late
    w = ease ** 1.35 if GRADIENT else np.zeros_like(ease)
    wmid = (w[:-1] + w[1:]) / 2.0

    quads, colors = [], []
    for i in range(len(xs) - 1):
        quads.append([
            (xs[i], bot[i]), (xs[i + 1], bot[i + 1]),
            (xs[i + 1], top[i + 1]), (xs[i], top[i]),
        ])
        colors.append(rgb0 * (1 - wmid[i]) + rgb1 * wmid[i])
    pc = PolyCollection(quads, facecolors=colors, edgecolors="face",
                        linewidths=0.0, antialiaseds=True,
                        alpha=RIBBON_ALPHA, zorder=1)
    ax.add_collection(pc)


# --------------------------------------------------------------------------- #
# Main alluvial renderer
# --------------------------------------------------------------------------- #
def render_alluvial(frame, phases, trans_labels, total, title, stem):
    n_col = len(phases)
    xcent = np.linspace(0.04, 0.96, n_col)

    # per-column class counts & slot bounds
    counts = [frame[p].value_counts().to_dict() for p in phases]
    bounds = [slot_bounds(c, total) for c in counts]

    fig, ax = plt.subplots(figsize=(2.15 * n_col + 4.0, 7.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    usable = 1.0 - SLOT_GAP * (len(CLASS_ORDER) - 1)

    # ---- ribbons (drawn first, beneath nodes) ----
    for i in range(n_col - 1):
        flow = (
            frame.groupby([phases[i], phases[i + 1]], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        x_src = xcent[i] + NODE_W
        x_dst = xcent[i + 1] - NODE_W

        out_off = {c: bounds[i][c][0] for c in CLASS_ORDER}     # running top, source side
        in_off = {c: bounds[i + 1][c][0] for c in CLASS_ORDER}  # running top, target side
        for s in CLASS_ORDER:
            if s not in flow.index:
                continue
            for d in CLASS_ORDER:
                if d not in flow.columns:
                    continue
                cnt = int(flow.at[s, d])
                if cnt == 0:
                    continue
                h = cnt / total * usable
                src = (out_off[s], out_off[s] - h)
                dst = (in_off[d], in_off[d] - h)
                draw_ribbon(ax, x_src, x_dst, src, dst, PALETTE[s], PALETTE[d])
                out_off[s] -= h
                in_off[d] -= h

    # ---- nodes (soft-rounded bars with a luminous edge) ----
    pad = 0.006
    for i, b in enumerate(bounds):
        for cls in CLASS_ORDER:
            top, bot = b[cls]
            if top - bot <= 1e-4:
                continue
            ax.add_patch(
                FancyBboxPatch(
                    (xcent[i] - NODE_W, bot + pad * 0), 2 * NODE_W, top - bot,
                    boxstyle="round,pad=0,rounding_size=0.006",
                    mutation_aspect=0.18,
                    facecolor=PALETTE[cls], edgecolor=_lighten(PALETTE[cls], 0.45),
                    lw=0.7, zorder=3,
                )
            )

    # ---- transition labels (over the ribbons between columns) ----
    for i, lab in enumerate(trans_labels):
        xm = (xcent[i] + xcent[i + 1]) / 2.0
        ax.text(xm, 1.052, lab, ha="center", va="bottom",
                fontsize=13, fontstyle="italic", fontweight="bold",
                color=INK)

    # ---- separator rule beneath the diagram ----
    ax.plot([0.02, 0.98], [-0.05, -0.05], color=RULE, lw=1.0,
            zorder=4, clip_on=False, solid_capstyle="round")

    # ---- title (with subtitle) ----
    ax.text(0.5, 1.155, f"PDAC Xenium · transcript fate — {title}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=19, fontweight="bold", color=INK)
    ax.text(0.5, 1.108, f"n = {total:,} transcripts",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11.5, color=INK_SOFT)

    # ---- legend ----
    handles = [Patch(facecolor=PALETTE[c], edgecolor=_lighten(PALETTE[c], 0.45),
                     lw=0.6, label=c)
               for c in ["neighboring cell", "original cell", "partial", "unassigned"]]
    leg = ax.legend(handles=handles, title="ENTITY CLASS", loc="upper center",
                    bbox_to_anchor=(0.5, -0.055), ncol=4, frameon=False,
                    fontsize=12, title_fontsize=11.5, handlelength=1.1,
                    handleheight=1.1, columnspacing=2.4, labelcolor=INK)
    leg.get_title().set_color(INK_SOFT)
    leg._legend_box.align = "center"

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.14, 1.18)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.12)

    for ext in ("png", "svg"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor=BG)
    plt.close(fig)
    print(f"  wrote {stem}.png / .svg")


# --------------------------------------------------------------------------- #
def main():
    print("Tier A …")
    fa, total = load_class_frame(TIER_A_COLS)
    render_alluvial(fa, TIER_A_COLS, TIER_A_LABELS, total,
                    "Tier A", "pdac_full_tier_a")

    print("Tier B …")
    fb, total_b = load_class_frame(TIER_B_COLS)
    render_alluvial(fb, TIER_B_COLS, TIER_B_LABELS, total_b,
                    "Tier B", "pdac_full_tier_b")

    assert total == total_b, "tier totals disagree"
    print(f"Done. n={total:,} transcripts.")


if __name__ == "__main__":
    main()
