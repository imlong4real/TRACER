"""Cell-type reassignment, Xenium -> TRACER, joined by ID.

All 1,821 whole-cell tracer_ids ARE Xenium cell_ids (partials get fresh
`cascade_*` ids), so the id join is the complete, threshold-free linkage:
1,821 of the 1,897 ROI cells. The earlier version of this figure instead asked
"which entity owns >=50% of this cell", which drops 299 cells because the
same-id entity holds a median of only 48.6% of its cell's transcripts.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns
from seg_spatial import CT, COL
from fig_spatial_and_sankey import sankey

OUT = C.Path(__file__).parent
REFSET = {"fibroblast", "myeloid", "t_cell", "b_cell", "endothelial"}


def comp(s):
    return s.map(lambda v: "tumor" if v == "epithelial" else ("reference" if v in REFSET else "unknown"))


def main():
    ax = pd.read_csv(C.RUN / "annotations_raw.csv", dtype={"entity_id": str}).set_index("entity_id")["annotation"].astype(str)
    at = pd.read_csv(C.RUN / "annotations_tracer_whole.csv", dtype={"entity_id": str}).set_index("entity_id")["annotation"].astype(str)
    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "_etype"]))
    t["cell_id"] = t["cell_id"].astype(str); t["tracer_id"] = t["tracer_id"].astype(str)
    et = t["_etype"].astype(str)
    whole = set(t.loc[et == "cell", "tracer_id"].unique())
    at_w = at.loc[at.index.isin(whole)]

    ids = ax.index.intersection(at_w.index)                       # the id join
    lost = ax.index.difference(at_w.index)                        # cells whose id does not survive
    xr, xt = ax.reindex(ids), at_w.reindex(ids)
    print(f"ROI Xenium cells {len(ax)} | whole-cell TRACER entities {len(at_w)} | id join {len(ids)} | "
          f"Xenium ids with no whole-cell entity {len(lost)}")

    meta = pd.read_csv(C.BENCH / "data/roi_cell_metadata.csv", dtype={"cell_id": str}).set_index("cell_id")
    xy = meta.reindex(ids)[["x", "y"]].to_numpy()
    cr, ct_ = comp(xr), comp(xt)
    cat = np.where((cr == "tumor") & (ct_ == "tumor"), "agree: tumour",
          np.where((cr == "reference") & (ct_ == "reference"), "agree: reference",
          np.where((cr == "tumor") & (ct_ == "reference"), "Xenium tumour -> TRACER reference",
          np.where((cr == "reference") & (ct_ == "tumor"), "Xenium reference -> TRACER tumour",
                   "involves unknown"))))
    DC = {"agree: tumour": "#a8ddb5", "agree: reference": "#cccccc", "involves unknown": "#f6e8a0",
          "Xenium tumour -> TRACER reference": "#e31a1c", "Xenium reference -> TRACER tumour": "#1f78b4"}

    fig = plt.figure(figsize=(19, 8.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.12, 1.12], wspace=0.10,
                          left=0.03, right=0.97, top=0.87, bottom=0.05)
    a = fig.add_subplot(gs[0])
    for c in ("agree: tumour", "agree: reference", "involves unknown",
              "Xenium reference -> TRACER tumour", "Xenium tumour -> TRACER reference"):
        m = cat == c
        if m.sum():
            a.scatter(xy[m, 0], xy[m, 1], s=16 if "->" in c else 9, color=DC[c],
                      label=f"{c} ({m.sum()})", linewidths=0, zorder=3 if "->" in c else 1)
    ml = np.isin(ids, lost)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.legend(fontsize=8, markerscale=1.7, frameon=False, loc="upper left")
    a.set_title(f"(a) where the arms disagree on compartment\n"
                f"all {len(ids)} id-matched cells (76 Xenium ids have no whole-cell entity)", fontsize=10)

    piv = pd.crosstab(xr, xt).reindex(index=CT, columns=CT).fillna(0)
    sankey(fig.add_subplot(gs[1]), piv, COL,
           f"(b) per-CELL label flow, joined by ID (n={len(ids)}, no threshold)")

    tdf = t.copy()
    tdf["L"] = np.where(tdf["cell_id"].isin(UNASSIGNED_TOKENS), "unassigned",
                        tdf["cell_id"].map(ax).fillna("unknown"))
    tdf["R"] = np.where(tdf["tracer_id"].isin(UNASSIGNED_TOKENS), "unassigned",
               np.where(et == "partial", "partial (discarded)", tdf["tracer_id"].map(at).fillna("unknown")))
    tp = pd.crosstab(tdf["L"], tdf["R"])
    lo = [c for c in CT + ["unassigned"] if c in tp.index]
    ro = [c for c in CT + ["partial (discarded)", "unassigned"] if c in tp.columns]
    tp = tp.reindex(index=lo, columns=ro).fillna(0)
    TC = dict(COL); TC["unassigned"] = "#8c8c8c"; TC["partial (discarded)"] = "#fdae61"
    sankey(fig.add_subplot(gs[2]), tp, TC, "(c) per-TRANSCRIPT flow (all 4.18M)")

    fig.suptitle("Cell-type reassignment between Xenium and TRACER, joined by ID "
                 "(every whole-cell tracer_id IS a Xenium cell_id)", fontsize=13)
    p = OUT / "fig_disagreement_sankey.png"; fig.savefig(p, dpi=160); print("wrote", p)

    print("\nper-CELL flow, ID join (rows Xenium, cols TRACER):\n" + piv.astype(int).to_string())
    print("\ncompartment-level:\n" + pd.Series(cat).value_counts().to_string())
    print(f"\nthe 76 Xenium ids with no whole-cell entity, by Xenium label:\n"
          + ax.reindex(lost).value_counts().to_string())


if __name__ == "__main__":
    main()
