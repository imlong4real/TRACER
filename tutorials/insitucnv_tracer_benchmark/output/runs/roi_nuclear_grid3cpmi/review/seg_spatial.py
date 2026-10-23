"""Spatial cell-type maps, Xenium vs TRACER, plus where the two disagree.

Entities differ between arms, so the disagreement panels work on a shared
universe of 10X cell_ids: each cell takes the annotation of whichever entity
owns >=50% of its transcripts.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns

OUT = C.Path(__file__).parent
CT = ["epithelial", "fibroblast", "myeloid", "t_cell", "b_cell", "endothelial", "unknown"]
COL = {"epithelial": "#2e7d32", "fibroblast": "#d73027", "myeloid": "#7b52ab", "t_cell": "#8d6e63",
       "b_cell": "#1f77b4", "endothelial": "#ff7f0e", "unknown": "#d0d0d0"}
REFSET = {"fibroblast", "myeloid", "t_cell", "b_cell", "endothelial"}


def project_annotation(arm, min_frac=0.5):
    spec = C.ARMS[arm]; ecol = spec["entity_col"]
    cols = list(dict.fromkeys([ecol, "cell_id"]))
    df = standardize_transcript_columns(pd.read_parquet(spec["transcripts"], columns=cols))
    df[ecol] = df[ecol].astype(str); df["cell_id"] = df["cell_id"].astype(str)
    df["_e"] = df[ecol]
    if "_etype" in df.columns:
        pass
    df = df.loc[~df["_e"].isin(UNASSIGNED_TOKENS) & ~df["cell_id"].isin(UNASSIGNED_TOKENS)]
    ann = pd.read_csv(C.RUN / f"annotations_{arm}.csv").set_index("entity_id")["annotation"].astype(str)
    df["_a"] = df["_e"].map(ann)
    df = df.dropna(subset=["_a"])
    vc = df.groupby(["cell_id", "_a"], observed=True).size().rename("n").reset_index()
    tot = df.groupby("cell_id").size().rename("tot")
    vc = vc.sort_values(["n", "_a"], ascending=[True, False])
    top = vc.groupby("cell_id").tail(1).set_index("cell_id")
    frac = top["n"] / tot.reindex(top.index)
    return top.loc[frac >= min_frac, "_a"]


def compartment(s):
    return s.map(lambda v: "tumor" if v == "epithelial" else ("reference" if v in REFSET else "unknown"))


def main():
    meta = pd.read_csv(C.BENCH / "data/roi_cell_metadata.csv", dtype={"cell_id": str}).set_index("cell_id")
    ann = {a: project_annotation(a) for a in ("raw", "tracer_whole")}
    entity_xy, entity_ann = {}, {}
    for arm in ("raw", "tracer_whole"):
        ad, _ = C.load_arm(arm)
        entity_xy[arm] = ad.obsm["spatial"]
        entity_ann[arm] = ad.obs["annotation"].astype(str).to_numpy()

    common = ann["raw"].index.intersection(ann["tracer_whole"].index).intersection(meta.index)
    xr, xt = ann["raw"].reindex(common), ann["tracer_whole"].reindex(common)
    xy = meta.loc[common, ["x", "y"]].to_numpy()
    cr, ctr = compartment(xr), compartment(xt)

    # local epithelial fraction from the full-slide cell_class (independent of both arms)
    epi_truth = (meta.loc[common, "cell_class"].astype(str) == "Epithelial").to_numpy()
    _, idx = cKDTree(xy).query(xy, k=min(21, len(xy)))
    local_epi = epi_truth[idx[:, 1:]].mean(axis=1)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.16, left=0.04, right=0.98, top=0.91, bottom=0.07)

    panels = [("raw", "Xenium (original segmentation)"), ("tracer_whole", "TRACER-refined (whole cells)")]
    for k, (arm, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        pts, lab = entity_xy[arm], entity_ann[arm]
        for c in CT:
            m = lab == c
            if m.sum():
                ax.scatter(pts[m, 0], pts[m, 1], s=11, color=COL[c], label=f"{c} ({m.sum()})", linewidths=0)
        ax.legend(fontsize=7.5, markerscale=1.7, frameon=False, loc="upper left")
        ax.set_title(f"{title}\nentities at their own centroids, n={len(lab)}", fontsize=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    tmap = {"Epithelial": "epithelial", "Fibroblasts": "fibroblast", "Monocytes": "myeloid",
            "T cells": "t_cell", "B cells": "b_cell", "Endothelial": "endothelial",
            "Low expression": "unknown"}
    cc = meta.loc[common, "cell_class"].astype(str).map(tmap).fillna("unknown")
    for c in CT:
        m = (cc == c).to_numpy()
        if m.sum():
            ax.scatter(xy[m, 0], xy[m, 1], s=11, color=COL[c], label=f"{c} ({m.sum()})", linewidths=0)
    ax.legend(fontsize=7.5, markerscale=1.7, frameon=False, loc="upper left")
    ax.set_title("full-slide cell_class (original segmentation)\nnot independent of Xenium", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    # ---- disagreement map, compartment level ----
    cat = np.where((cr == "tumor") & (ctr == "tumor"), "agree: tumour",
          np.where((cr == "reference") & (ctr == "reference"), "agree: reference",
          np.where((cr == "tumor") & (ctr == "reference"), "Xenium tumour -> TRACER reference",
          np.where((cr == "reference") & (ctr == "tumor"), "Xenium reference -> TRACER tumour",
                   "involves unknown"))))
    DC = {"agree: tumour": "#c7e9c0", "agree: reference": "#cfcfcf",
          "Xenium tumour -> TRACER reference": "#e31a1c",
          "Xenium reference -> TRACER tumour": "#1f78b4", "involves unknown": "#f0e442"}
    ax = fig.add_subplot(gs[1, 0])
    for c in ("agree: tumour", "agree: reference", "involves unknown",
              "Xenium reference -> TRACER tumour", "Xenium tumour -> TRACER reference"):
        m = cat == c
        if m.sum():
            ax.scatter(xy[m, 0], xy[m, 1], s=13 if "->" in c else 8, color=DC[c],
                       label=f"{c} ({m.sum()})", linewidths=0, zorder=3 if "->" in c else 1)
    ax.legend(fontsize=7.5, markerscale=1.7, frameon=False, loc="upper left")
    ax.set_title(f"(d) where the two arms disagree (compartment level)\n{len(common)} shared 10X cells", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    # ---- fragmentation: how much of a cell does its owning TRACER entity hold? ----
    ax = fig.add_subplot(gs[1, 1])
    spec = C.ARMS["tracer_whole"]
    d2 = standardize_transcript_columns(pd.read_parquet(spec["transcripts"], columns=["tracer_id", "cell_id"]))
    d2["tracer_id"] = d2["tracer_id"].astype(str); d2["cell_id"] = d2["cell_id"].astype(str)
    d2 = d2.loc[~d2["tracer_id"].isin(UNASSIGNED_TOKENS) & ~d2["cell_id"].isin(UNASSIGNED_TOKENS)]
    own = d2.groupby(["cell_id", "tracer_id"], observed=True).size().rename("n").reset_index()
    tot2 = d2.groupby("cell_id").size().rename("tot")
    best = (own["n"] / own["cell_id"].map(tot2)).groupby(own["cell_id"]).max()
    ax.hist(best.to_numpy(), bins=np.linspace(0, 1, 41), color="#08519c", alpha=0.85)
    ax.axvline(0.5, color="crimson", ls="--", lw=1.2,
               label=f"0.5 ownership cutoff\nmedian = {best.median():.2f}")
    ax.set_xlabel("largest share of a 10X cell's transcripts held by one TRACER entity")
    ax.set_ylabel("10X cells")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(e) TRACER fragments cells: the median 10X cell has only ~53%\n"
                 "     of its transcripts in its biggest TRACER entity", fontsize=10, loc="left")

    # ---- crosstab ----
    ax = fig.add_subplot(gs[1, 2])
    ctab = pd.crosstab(xr, xt).reindex(index=CT, columns=CT).fillna(0).astype(int)
    im = ax.imshow(ctab.values, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(ctab.values.max(), 2)))
    ax.set_xticks(range(len(CT))); ax.set_xticklabels(CT, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CT))); ax.set_yticklabels(CT, fontsize=8)
    for i in range(len(CT)):
        for j in range(len(CT)):
            v = ctab.values[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7.5,
                        color="w" if v > ctab.values.max() * 0.25 else "k")
    ax.set_xlabel("TRACER"); ax.set_ylabel("Xenium")
    ax.set_title("(f) per-cell label crosstab", fontsize=10, loc="left")

    fig.suptitle("Where do Xenium and TRACER disagree about cell type? -- ovary ROI", fontsize=13)
    out = OUT / "seg_spatial_celltype.png"; fig.savefig(out, dpi=150); print("wrote", out)

    print(f"\nshared cells: {len(common)}")
    print(pd.Series(cat).value_counts().to_string())
    print("\ncrosstab (rows Xenium, cols TRACER):\n" + ctab.to_string())
    print(f"\nTRACER cell-ownership share: median {best.median():.3f}, "
          f"{100*(best<0.5).mean():.1f}% of cells have NO entity owning half of them")
    flip = cat == "Xenium tumour -> TRACER reference"
    sw = pd.DataFrame({"local_epi": local_epi, "flip": flip,
                       "xenium": xr.to_numpy(), "tracer": xt.to_numpy(),
                       "truth_class": meta.loc[common, "cell_class"].to_numpy()})
    print("\nfor cells that switched tumour->reference, what does the full slide call them?")
    print(sw.loc[sw.flip, "truth_class"].value_counts().to_string())
    sw.to_csv(OUT / "seg_spatial_switches.csv", index=False)


if __name__ == "__main__":
    main()
