#!/usr/bin/env python3
"""Panel C — ROI biological validation (zoom-ins on real kidney histology).

Four biologically intuitive ROIs, each auto-selected as the densest patch of
its hallmark lineage in the TRACER 2um reconstruction:
  - Glomerulus        (POD lights up glomeruli)
  - Cortical PT-rich  (PT broad / abundant)
  - TAL-enriched      (TAL spatially restricted)
  - Collecting-duct / IC region (IC localized)

For each ROI we show, side by side: H&E | 10x segmented | bin2cell |
TRACER 2um | TRACER 8um, all in the common micron frame.

Message: reconstructed cell types align with real kidney histology.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

import fig4_config as C
import utils as U

ROI_SIZE_UM = 320.0
ROI_SPEC = [   # (key, hallmark lineage, rationale)
    ("glomerulus", "POD", "Podocytes mark glomerular tufts; POD should form compact clusters."),
    ("cortex_PT", "PT", "Proximal tubule dominates cortex; PT should be broad and abundant."),
    ("TAL_enriched", "TAL", "Thick ascending limb is spatially restricted; TAL focal enrichment."),
    ("collecting_duct_IC", "IC", "Intercalated cells localize to collecting ducts; IC focal, not diffuse."),
]


def _pick_roi(pts_t2: pd.DataFrame, lineage: str, half: float,
              exclude: list[tuple[float, float]]):
    """Densest ROI center for a lineage via 2D histogram, avoiding prior picks."""
    sub = pts_t2[pts_t2["lineage"] == lineage]
    binsz = half  # grid resolution ~ ROI half-size
    xedges = np.arange(sub["mx"].min(), sub["mx"].max() + binsz, binsz)
    yedges = np.arange(sub["my"].min(), sub["my"].max() + binsz, binsz)
    H, xe, ye = np.histogram2d(sub["mx"], sub["my"], bins=[xedges, yedges])
    # penalize cells near previously chosen ROI centers
    cx = (xe[:-1] + xe[1:]) / 2
    cy = (ye[:-1] + ye[1:]) / 2
    for (ex, ey) in exclude:
        for i, x in enumerate(cx):
            for j, y in enumerate(cy):
                if abs(x - ex) < 3 * half and abs(y - ey) < 3 * half:
                    H[i, j] = 0
    i, j = np.unravel_index(np.argmax(H), H.shape)
    return float(cx[i]), float(cy[j]), int(H[i, j])


def _draw_roi(ax, df, cx, cy, half, crop, ext, title, s, show_he_only=False):
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    ax.imshow(crop, extent=ext, zorder=0)
    if not show_he_only:
        sub = df[(df.mx >= x0) & (df.mx <= x1) & (df.my >= y0) & (df.my <= y1)]
        order = [l for l in C.LINEAGES if l in set(sub["lineage"].dropna())]
        for l in order:
            s2 = sub[sub["lineage"] == l]
            ax.scatter(s2["mx"], s2["my"], s=s, linewidths=0, color=C.PALETTE[l],
                       zorder=2, alpha=0.9)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)  # y inverted (micron grows down)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def make():
    plt = U.setup_style()
    pts = {m: U.method_points(m) for m in C.METHOD_ORDER}
    half = ROI_SIZE_UM / 2

    chosen, meta = [], []
    for key, lin, why in ROI_SPEC:
        cx, cy, n = _pick_roi(pts["tracer_2um"], lin, half, chosen)
        chosen.append((cx, cy))
        meta.append({"roi": key, "hallmark_lineage": lin, "center_x_um": round(cx, 1),
                     "center_y_um": round(cy, 1), "size_um": ROI_SIZE_UM,
                     "tracer2um_hallmark_units_in_patch": n, "rationale": why})
    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(C.SRCDIR / "panel_C_roi_metadata.csv", index=False)
    print(meta_df.to_string(index=False))

    order = ["10x_segmented", "bin2cell", "tracer_2um", "tracer_8um"]
    sdot = {"10x_segmented": 9, "bin2cell": 9, "tracer_2um": 5, "tracer_8um": 16}
    # full-resolution H&E crop per ROI (instant; reads only intersecting tiles)
    crops = {key: U.he_crop_um(cx - half, cy - half, cx + half, cy + half)
             for (key, _, _), (cx, cy) in zip(ROI_SPEC, chosen)}

    nrow = len(ROI_SPEC)
    fig, axes = plt.subplots(nrow, 5, figsize=(13, 2.8 * nrow))
    for r, ((key, lin, why), (cx, cy)) in enumerate(zip(ROI_SPEC, chosen)):
        crop, ext = crops[key]
        rowlab = f"{key}\n({C.LINEAGE_DISPLAY[lin]})"
        _draw_roi(axes[r, 0], None, cx, cy, half, crop, ext,
                  "H&E" if r == 0 else "", 0, show_he_only=True)
        axes[r, 0].set_ylabel(rowlab, fontsize=8, rotation=90, labelpad=6)
        for c, m in enumerate(order, start=1):
            title = C.METHOD_DISPLAY[m] if r == 0 else ""
            _draw_roi(axes[r, c], pts[m], cx, cy, half, crop, ext, title, sdot[m])
    fig.legend(handles=U.lineage_handles(), loc="lower center", ncol=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.2,
               columnspacing=0.9)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05,
                        wspace=0.04, hspace=0.08)
    U.save_fig(fig, "panel_C_roi_validation")

    # individual per-ROI strips
    for (key, lin, why), (cx, cy) in zip(ROI_SPEC, chosen):
        crop, ext = crops[key]
        f, axs = plt.subplots(1, 5, figsize=(13, 2.9))
        _draw_roi(axs[0], None, cx, cy, half, crop, ext, "H&E", 0, show_he_only=True)
        for c, m in enumerate(order, start=1):
            _draw_roi(axs[c], pts[m], cx, cy, half, crop, ext, C.METHOD_DISPLAY[m], sdot[m])
        f.suptitle(f"ROI: {key} — hallmark {C.LINEAGE_DISPLAY[lin]}", fontsize=9)
        f.legend(handles=U.lineage_handles(), loc="lower center", ncol=9,
                 frameon=False, bbox_to_anchor=(0.5, -0.05))
        f.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.06, wspace=0.04)
        U.save_fig(f, f"panel_C_roi_{key}")


if __name__ == "__main__":
    make()
