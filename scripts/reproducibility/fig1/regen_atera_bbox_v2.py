#!/usr/bin/env python3
"""Replot the Atera whole-tissue bounding-boxes figure with a legend that
matches the visual schema of ``whole_tissue_categorical_overlap_v2.svg``.

What changes vs the v1 bbox figure
----------------------------------
* Legend handles are **filled circles** (matplotlib.lines.Line2D with
  ``marker='o'``, ``markerfacecolor=<cat colour>``, ``markeredgecolor='white'``,
  ``markersize=12``) — identical to the categorical_overlap_v2 figure.
* Labels use the same ``ovrlpy ✓ | TRACER ✓`` etc. wording with cell counts
  annotated as ``n=xxx,xxx`` (from the cached joined-cell table).
* Dark background, lower-right anchor, white edge, slightly transparent
  frame — all matching the categorical_overlap_v2 file.

Background scatter (whole tissue) is preserved from v1; ROI bounding boxes
remain drawn as 2.5pt-wide coloured rectangles with no text annotations
inside the canvas. ROI metadata is read from the cached
``representative_rois.json``.

Output
------
``results/ovrlpy_tracer/cervical_atera_full_memoryaware/final_figures_fixed/
  whole_tissue_bounding_boxes_all_v2.{svg,png}``

(The v1 SVG/PNG is preserved.)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


CAT_PALETTE = {
    "A_ovrlpy+_tracer+":  "#00E5FF",
    "B_ovrlpy-_tracer+":  "#FF1493",
    "C_ovrlpy+_tracer-":  "#39FF14",
    "D_concordant_clean": "#1a1a3a",
}
# Legend labels — match the kidney bbox style (compound ✓/✕ wording, but
# the concordant-clean D category is dropped from the legend since the
# whole-tissue background already represents it).
ATERA_LABEL = {
    "C_ovrlpy+_tracer-":  "ovrlpy ✕ | TRACER ✓",
    "B_ovrlpy-_tracer+":  "ovrlpy ✓ | TRACER ✕",
    "A_ovrlpy+_tracer+":  "ovrlpy ✕ | TRACER ✕",
}
ATERA_LEGEND_ORDER = [
    "A_ovrlpy+_tracer+",
    "B_ovrlpy-_tracer+",
    "C_ovrlpy+_tracer-",
]


def main() -> int:
    atera_dir = Path("results/ovrlpy_tracer/cervical_atera_full_memoryaware")
    joined_path = atera_dir / "tables" / "ovrlpy_tracer_cell_level_comparison.tsv"
    rois_path = atera_dir / "representative_rois.json"
    out_base = atera_dir / "final_figures_fixed" / "whole_tissue_bounding_boxes_all_v2"

    print(f"Loading {joined_path}")
    df = pd.read_csv(joined_path, sep="\t", usecols=["category", "cx", "cy"])
    counts = df["category"].value_counts()
    print(f"  category counts: {counts.to_dict()}")

    print(f"Loading {rois_path}")
    rois_by_cat = json.load(open(rois_path))
    n_rois = sum(len(v) for v in rois_by_cat.values()
                 if isinstance(v, list))
    print(f"  {n_rois} ROIs across {len(rois_by_cat)} categories")

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
        # Whole tissue dim scatter (matches v1 spirit)
        ax.scatter(df["cx"], df["cy"], s=0.5, c="#444444",
                   alpha=0.4, linewidths=0, rasterized=True)
        # Coloured ROI rectangles, no in-canvas text labels
        for cat, roi_list in rois_by_cat.items():
            color = CAT_PALETTE.get(cat, "white")
            if not isinstance(roi_list, list):
                continue
            for roi in roi_list:
                ax.add_patch(Rectangle(
                    (roi["xmin"], roi["ymin"]),
                    roi["xmax"] - roi["xmin"],
                    roi["ymax"] - roi["ymin"],
                    edgecolor=color, facecolor="none", lw=2.5,
                ))
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_title("Atera — whole tissue with representative ROI bounding boxes",
                     color="white")
        ax.set_xlabel("x (µm)", color="white")
        ax.set_ylabel("y (µm)", color="white")

        # Legend matches the kidney bbox v2: filled rectangular Patch
        # handles, no per-category cell-count text, and only the three
        # non-concordant categories (A/B/C) — the concordant-clean D is
        # represented by the background scatter so it is omitted from the
        # legend.
        legend_handles = [
            Patch(facecolor=CAT_PALETTE[c], edgecolor="white", linewidth=1.0,
                  label=ATERA_LABEL[c])
            for c in ATERA_LEGEND_ORDER
        ]
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=10, facecolor="black", edgecolor="white",
                  labelcolor="white", handletextpad=0.6, framealpha=0.85,
                  handlelength=2.0, handleheight=1.4)
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(f"{out_base}.{ext}", dpi=160, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"Wrote {out_base}.svg + .png")
    return 0


if __name__ == "__main__":
    main()
