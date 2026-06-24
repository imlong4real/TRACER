#!/usr/bin/env python3
"""Panel B — whole-tissue lineage maps for all four methods over H&E.

Side-by-side 10x segmented | bin2cell | TRACER 2um | TRACER 8um, each as a
lineage-colored scatter on a faint H&E background (white canvas), all placed
in the common H&E micron frame so they are directly comparable.

Message: TRACER recovers plausible kidney architecture at tissue scale; both
2um and 8um yield interpretable profiles rather than noise.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import fig4_config as C
import utils as U


def _draw(ax, df, he, ext, title, s, bbox):
    ax.imshow(he, extent=ext, alpha=0.45, zorder=0)
    order = [l for l in C.LINEAGES if l in set(df["lineage"].dropna())]
    for l in order:
        sub = df[df["lineage"] == l]
        ax.scatter(sub["mx"], sub["my"], s=s, linewidths=0,
                   color=C.PALETTE[l], zorder=2, rasterized=True)
    x0, x1, y0, y1 = bbox
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)   # y inverted (micron grows down)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6)


def make():
    plt = U.setup_style()
    he, ext = U.load_he()
    pts = {m: U.method_points(m) for m in C.METHOD_ORDER}
    s = {"10x_segmented": 0.7, "bin2cell": 0.7, "tracer_2um": 0.25, "tracer_8um": 1.1}
    # crop axes to the shared data bounding box (the VisiumHD capture area)
    allp = pd.concat([pts[m][["mx", "my"]] for m in C.METHOD_ORDER])
    mx0, mx1 = allp["mx"].quantile([0.001, 0.999])
    my0, my1 = allp["my"].quantile([0.001, 0.999])
    pad = 0.02 * max(mx1 - mx0, my1 - my0)
    bbox = (mx0 - pad, mx1 + pad, my0 - pad, my1 + pad)

    # source table: lineage counts per method
    rows = []
    for m in C.METHOD_ORDER:
        vc = pts[m]["lineage"].value_counts()
        for l in C.LINEAGES:
            rows.append({"method": C.METHOD_DISPLAY[m], "lineage": l,
                         "n_units": int(vc.get(l, 0))})
    src = pd.DataFrame(rows).pivot(index="lineage", columns="method",
                                   values="n_units").reindex(C.LINEAGES)
    src.to_csv(C.SRCDIR / "panel_B_lineage_counts.csv")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
    for ax, m in zip(axes, C.METHOD_ORDER):
        _draw(ax, pts[m], he, ext,
              f"{C.METHOD_DISPLAY[m]}  (n={len(pts[m]):,})", s[m], bbox)
    fig.legend(handles=U.lineage_handles(), loc="lower center", ncol=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04), handletextpad=0.2,
               columnspacing=0.9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.06, wspace=0.04)
    U.save_fig(fig, "panel_B_whole_tissue_maps")
    print(src.to_string())


if __name__ == "__main__":
    make()
