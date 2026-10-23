"""Two figures: (1) spatial cell-type maps; (2) compartment disagreement + Sankey flows."""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Patch
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns
from seg_spatial import project_annotation, compartment, CT, COL

OUT = C.Path(__file__).parent
TITLE = {"raw": "Xenium (original segmentation)", "tracer_whole": "TRACER-refined (whole cells)"}


def ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.55):
    """Cubic-Bezier ribbon from (x0, y0a..y0b) to (x1, y1a..y1b)."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
             (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def sankey(ax, piv, colours, title, gap_frac=0.015, unit=""):
    """piv: DataFrame, rows = left nodes, cols = right nodes, values = flow."""
    piv = piv.loc[piv.sum(1) > 0, piv.sum(0) > 0]
    total = piv.values.sum()
    gap = gap_frac * total
    lh = piv.sum(1); rh = piv.sum(0)
    ly, ry = {}, {}
    y = 0.0
    for n in piv.index:
        ly[n] = y; y += lh[n] + gap
    left_top = y - gap
    y = 0.0
    for n in piv.columns:
        ry[n] = y; y += rh[n] + gap
    right_top = y - gap
    top = max(left_top, right_top)
    lo = (top - left_top) / 2; ro = (top - right_top) / 2

    lcur = {n: ly[n] + lo for n in piv.index}
    rcur = {n: ry[n] + ro for n in piv.columns}
    for n in piv.index:
        ax.add_patch(plt.Rectangle((0.0, ly[n] + lo), 0.055 * total, lh[n],
                                   color=colours.get(n, "#999999")))
        ax.text(-0.01 * total, ly[n] + lo + lh[n] / 2, f"{n}  {int(lh[n]):,}{unit}",
                ha="right", va="center", fontsize=7.5)
    for n in piv.columns:
        ax.add_patch(plt.Rectangle((0.945 * total, ry[n] + ro), 0.055 * total, rh[n],
                                   color=colours.get(n, "#999999")))
        ax.text(1.015 * total, ry[n] + ro + rh[n] / 2, f"{n}  {int(rh[n]):,}{unit}",
                ha="left", va="center", fontsize=7.5)
    for a in piv.index:
        for b in piv.columns:
            v = piv.loc[a, b]
            if v <= 0:
                continue
            ribbon(ax, 0.055 * total, 0.945 * total, lcur[a], lcur[a] + v,
                   rcur[b], rcur[b] + v, colours.get(a, "#999999"))
            lcur[a] += v; rcur[b] += v
    ax.set_xlim(-0.30 * total, 1.32 * total)
    ax.set_ylim(top * 1.02, -0.02 * top)
    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left")


def main():
    # ---------------- figure 1: spatial cell-type maps ----------------
    ents = {}
    for arm in ("raw", "tracer_whole"):
        ad, _ = C.load_arm(arm)
        ents[arm] = (ad.obsm["spatial"], ad.obs["annotation"].astype(str).to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.4))
    for ax, arm in zip(axes, ("raw", "tracer_whole")):
        xy, lab = ents[arm]
        for c in CT:
            m = lab == c
            if m.sum():
                ax.scatter(xy[m, 0], xy[m, 1], s=14, color=COL[c], label=f"{c} ({m.sum()})", linewidths=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8.5, markerscale=1.8, frameon=False, loc="upper left")
        ax.set_title(f"{TITLE[arm]}\nn={len(lab)} entities", fontsize=11)
    fig.suptitle("Spatial cell-type maps, marker-based annotation -- ovary ROI (500 x 500 um)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p1 = OUT / "fig_spatial_celltype_maps.png"; fig.savefig(p1, dpi=160); print("wrote", p1)

    # ---------------- figure 2: disagreement map + Sankeys ----------------
    meta = pd.read_csv(C.BENCH / "data/roi_cell_metadata.csv", dtype={"cell_id": str}).set_index("cell_id")
    ann = {a: project_annotation(a) for a in ("raw", "tracer_whole")}
    common = ann["raw"].index.intersection(ann["tracer_whole"].index).intersection(meta.index)
    xr, xt = ann["raw"].reindex(common), ann["tracer_whole"].reindex(common)
    xy = meta.loc[common, ["x", "y"]].to_numpy()
    cr, ctr = compartment(xr), compartment(xt)
    cat = np.where((cr == "tumor") & (ctr == "tumor"), "agree: tumour",
          np.where((cr == "reference") & (ctr == "reference"), "agree: reference",
          np.where((cr == "tumor") & (ctr == "reference"), "Xenium tumour -> TRACER reference",
          np.where((cr == "reference") & (ctr == "tumor"), "Xenium reference -> TRACER tumour",
                   "involves unknown"))))
    DC = {"agree: tumour": "#a8ddb5", "agree: reference": "#cccccc", "involves unknown": "#f6e8a0",
          "Xenium tumour -> TRACER reference": "#e31a1c", "Xenium reference -> TRACER tumour": "#1f78b4"}

    fig = plt.figure(figsize=(19, 8.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.12, 1.12], wspace=0.10,
                          left=0.03, right=0.97, top=0.88, bottom=0.05)
    ax = fig.add_subplot(gs[0])
    for c in ("agree: tumour", "agree: reference", "involves unknown",
              "Xenium reference -> TRACER tumour", "Xenium tumour -> TRACER reference"):
        m = cat == c
        if m.sum():
            ax.scatter(xy[m, 0], xy[m, 1], s=15 if "->" in c else 9, color=DC[c],
                       label=f"{c} ({m.sum()})", linewidths=0, zorder=3 if "->" in c else 1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=1.7, frameon=False, loc="upper left")
    ax.set_title(f"(a) where the arms disagree on compartment\n{len(common)} shared 10X cells", fontsize=10)

    # cell-level Sankey
    piv = pd.crosstab(xr, xt).reindex(index=CT, columns=CT).fillna(0)
    sankey(fig.add_subplot(gs[1]), piv, COL,
           "(b) per-CELL label flow (shared cells, >=50% transcript ownership)", unit="")

    # transcript-level Sankey
    tdf = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "_etype"]))
    tdf["cell_id"] = tdf["cell_id"].astype(str); tdf["tracer_id"] = tdf["tracer_id"].astype(str)
    ax_ = pd.read_csv(C.RUN / "annotations_raw.csv").set_index("entity_id")["annotation"].astype(str)
    at_ = pd.read_csv(C.RUN / "annotations_tracer_whole.csv").set_index("entity_id")["annotation"].astype(str)
    et = tdf["_etype"].astype(str)
    tdf["L"] = np.where(tdf["cell_id"].isin(UNASSIGNED_TOKENS), "unassigned",
                        tdf["cell_id"].map(ax_).fillna("unknown"))
    tdf["R"] = np.where(tdf["tracer_id"].isin(UNASSIGNED_TOKENS), "unassigned",
               np.where(et == "partial", "partial (discarded)", tdf["tracer_id"].map(at_).fillna("unknown")))
    tp = pd.crosstab(tdf["L"], tdf["R"])
    lo = [c for c in CT + ["unassigned"] if c in tp.index]
    ro = [c for c in CT + ["partial (discarded)", "unassigned"] if c in tp.columns]
    tp = tp.reindex(index=lo, columns=ro).fillna(0)
    TC = dict(COL); TC["unassigned"] = "#8c8c8c"; TC["partial (discarded)"] = "#fdae61"
    sankey(fig.add_subplot(gs[2]), tp, TC,
           "(c) per-TRANSCRIPT flow (all 4.18M, including discarded partials)", unit="")
    fig.suptitle("Cell-type reassignment between Xenium and TRACER -- who becomes what?", fontsize=13)
    p2 = OUT / "fig_disagreement_sankey.png"; fig.savefig(p2, dpi=160); print("wrote", p2)
    print("\nper-cell flow:\n" + piv.astype(int).to_string())
    print(f"\nepithelial -> fibroblast: cells {int(piv.loc['epithelial','fibroblast'])}, "
          f"transcripts {int(tp.loc['epithelial','fibroblast']):,}")


if __name__ == "__main__":
    main()
