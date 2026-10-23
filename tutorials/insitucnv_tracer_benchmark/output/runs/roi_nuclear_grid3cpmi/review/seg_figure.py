"""Figure for the segmentation-level comparison (no CNV)."""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns
from seg_compare import load, qc_table, LABELS

OUT = C.Path(__file__).parent
COL = {"epithelial":"#2e7d32","fibroblast":"#d73027","myeloid":"#7b52ab","t_cell":"#8d6e63",
       "b_cell":"#1f77b4","endothelial":"#ff7f0e","unknown":"#bdbdbd"}


def main():
    data = {}
    for arm in ("raw","tracer_whole"):
        df, ecol = load(arm)
        sub = df.loc[~df[ecol].isin(UNASSIGNED_TOKENS)]
        if "_etype" in df.columns:
            sub = sub.loc[sub["_etype"].astype(str)=="cell"]
        t = qc_table(sub, ecol)
        ann = pd.read_csv(C.RUN/f"annotations_{arm}.csv").set_index("entity_id")["annotation"].astype(str)
        t["annotation"] = ann.reindex(t.index.astype(str)).fillna("unknown").to_numpy()
        data[arm] = (df, ecol, t)

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28, left=0.06, right=0.98, top=0.90, bottom=0.08)

    # (a) transcript accounting
    ax = fig.add_subplot(gs[0,0])
    tot = len(data["raw"][0])
    raw_un = int(data["raw"][0][data["raw"][1]].isin(UNASSIGNED_TOKENS).sum())
    tdf = data["tracer_whole"][0]
    et = tdf["_etype"].astype(str).value_counts()
    bars = {
        "Xenium": [tot-raw_un, 0, raw_un],
        "TRACER": [int(et.get("cell",0)), int(et.get("partial",0)), int(et.get("unknown",0))],
    }
    labels = ["assigned to whole cells","assigned to PARTIALS\n(discarded by tracer_whole)","unassigned"]
    cols = ["#2e7d32","#fdae61","#bdbdbd"]
    bottom = np.zeros(2)
    for i,(lab,c) in enumerate(zip(labels, cols)):
        vals = np.array([bars["Xenium"][i], bars["TRACER"][i]])/tot*100
        ax.bar(["Xenium","TRACER"], vals, bottom=bottom, color=c, label=lab)
        for j,v in enumerate(vals):
            if v > 3: ax.text(j, bottom[j]+v/2, f"{v:.1f}%", ha="center", va="center", fontsize=8)
        bottom += vals
    ax.set_ylabel("% of the ROI's 4,176,533 transcripts"); ax.set_ylim(0,100)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(0.02,0.35))
    ax.set_title("(a) where the transcripts go", fontsize=10, loc="left")

    # (b) entity counts
    ax = fig.add_subplot(gs[0,1])
    n = {"raw":[1897,1891], "TRACER":[1821,1818]}
    x=np.arange(2); w=0.35
    ax.bar(x-w/2, [n["raw"][0], n["TRACER"][0]], w, label="pre-QC", color="#9ecae1")
    ax.bar(x+w/2, [n["raw"][1], n["TRACER"][1]], w, label="post-QC", color="#08519c")
    for i,(a,b) in enumerate([n["raw"], n["TRACER"]]):
        ax.text(i-w/2, a+15, str(a), ha="center", fontsize=8)
        ax.text(i+w/2, b+15, str(b), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["Xenium","TRACER\n(whole cells)"])
    ax.set_ylabel("entities"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("(b) cell count is essentially unchanged (-3.9%)", fontsize=10, loc="left")

    # (c) composition
    ax = fig.add_subplot(gs[0,2])
    truth = pd.read_csv(C.BENCH/"data/roi_cell_metadata.csv")["cell_class"].value_counts()
    tmap={"Epithelial":"epithelial","Fibroblasts":"fibroblast","Monocytes":"myeloid","T cells":"t_cell",
          "B cells":"b_cell","Endothelial":"endothelial","Low expression":"unknown"}
    series = {
        "full-slide\ncell_class": truth.rename(index=tmap).groupby(level=0).sum().reindex(LABELS).fillna(0),
        "Xenium": data["raw"][2]["annotation"].value_counts().reindex(LABELS).fillna(0),
        "TRACER": data["tracer_whole"][2]["annotation"].value_counts().reindex(LABELS).fillna(0),
    }
    bottom=np.zeros(3)
    for lab in LABELS:
        vals=np.array([series[k][lab]/series[k].sum()*100 for k in series])
        ax.bar(list(series), vals, bottom=bottom, color=COL[lab], label=lab)
        for j,v in enumerate(vals):
            if v>4: ax.text(j, bottom[j]+v/2, f"{v:.0f}", ha="center", va="center", fontsize=7)
        bottom+=vals
    ax.set_ylabel("% of entities"); ax.set_ylim(0,100)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0,0.02))
    ax.set_title("(c) cell-type composition", fontsize=10, loc="left")

    # (d) depth per annotation
    ax = fig.add_subplot(gs[1,0:2])
    order=[l for l in LABELS if l!="unknown"]
    pos=np.arange(len(order))
    for off,(arm,c) in zip((-0.2,0.2), (("raw","#6baed6"),("tracer_whole","#08519c"))):
        t=data[arm][2]
        vals=[t.loc[(t.annotation==l)&t.passes_qc,"n_genes"].to_numpy() for l in order]
        vals=[v if len(v) else np.array([np.nan]) for v in vals]
        bp=ax.boxplot(vals, positions=pos+off, widths=0.34, patch_artist=True, showfliers=False)
        for b in bp["boxes"]: b.set_facecolor(c[1] if isinstance(c,tuple) else c); b.set_alpha(0.85)
        for m in bp["medians"]: m.set_color("k")
    ax.axhline(700, color="crimson", ls="--", lw=1, label="n_genes = 700 (panel e threshold)")
    ax.set_xticks(pos); ax.set_xticklabels(order); ax.set_yscale("log"); ax.set_ylabel("n_genes per entity")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(d) depth per cell type -- light = Xenium, dark = TRACER; stromal depth falls furthest", fontsize=10, loc="left")

    # (e) composition vs depth threshold
    ax = fig.add_subplot(gs[1,2])
    ths=np.arange(0,1300,50)
    for arm,c,ls in (("raw","#6baed6","-"),("tracer_whole","#08519c","-")):
        t=data[arm][2]
        fr=[ (t.loc[t.passes_qc & (t.n_genes>=x),"annotation"]=="epithelial").mean()*100 for x in ths]
        ax.plot(ths, fr, color=c, ls=ls, lw=2, label={"raw":"Xenium","tracer_whole":"TRACER"}[arm])
    ax.axhline(54.5, color="k", ls=":", lw=1, label="full-slide epithelial share (54.5%)")
    ax.set_xlabel("keep entities with n_genes >= x"); ax.set_ylabel("% epithelial")
    ax.legend(fontsize=8, frameon=False); ax.set_ylim(0,100)
    ax.set_title("(e) epithelial share vs depth cutoff", fontsize=10, loc="left")

    fig.suptitle("Segmentation-level comparison, raw Xenium vs TRACER -- ovary ROI (no CNV involved)", fontsize=13)
    out=OUT/"seg_compare.png"; fig.savefig(out, dpi=150); print("wrote", out)


if __name__ == "__main__":
    main()
