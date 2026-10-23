"""Epithelial and fibroblast scores side by side, per category of entity.

Categories: ids that go epithelial (Xenium) -> fibroblast (TRACER), plus
concordant-fibroblast and concordant-epithelial controls, each under both
segmentations.

Score = the annotator's own definition: fraction of the entity's transcripts
belonging to the marker set. Top row uses all transcripts; bottom row uses only
transcripts inside a nucleus, which cannot be a neighbour's spill-over.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns

OUT = C.Path(__file__).parent
ms = yaml.safe_load(open(C.BENCH / "config/panels.yaml"))["marker_sets"]
EPI, FIB = set(map(str, ms["epithelial"])), set(map(str, ms["fibroblast"]))
NAMES = ["epithelial -> fibroblast\nXENIUM", "epithelial -> fibroblast\nTRACER",
         "concordant fibroblast\nXENIUM", "concordant fibroblast\nTRACER",
         "concordant epithelial\nXENIUM", "concordant epithelial\nTRACER"]
EC, FC = "#2e7d32", "#d73027"


def scores(df, col):
    d = df.loc[~df[col].isin(UNASSIGNED_TOKENS)]
    tot = d.groupby(col, observed=True).size().rename("n")
    e = d.loc[d["feature_name"].astype(str).isin(EPI)].groupby(col, observed=True).size()
    f = d.loc[d["feature_name"].astype(str).isin(FIB)].groupby(col, observed=True).size()
    o = pd.DataFrame({"n": tot})
    o["epi"] = e.reindex(o.index).fillna(0) / o["n"]
    o["fib"] = f.reindex(o.index).fillna(0) / o["n"]
    o.index = o.index.astype(str)
    return o


def paired_panel(a, sets, title, ylab):
    w = 0.34
    rng = np.random.default_rng(0)
    for off, key, col in ((-w / 2, "epi", EC), (w / 2, "fib", FC)):
        data = [s[key].dropna().to_numpy() for s in sets]
        bp = a.boxplot(data, positions=np.arange(len(sets)) + off, widths=w * 0.88,
                       patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(col); b.set_alpha(0.80)
        for m in bp["medians"]:
            m.set_color("k"); m.set_linewidth(1.4)
        for i, v in enumerate(data):
            a.scatter(rng.normal(i + off, 0.035, len(v)), v, s=3, color="k", alpha=0.20, zorder=3)
    top = max(np.percentile(s[k].dropna().to_numpy(), 98)
              for s in sets for k in ("epi", "fib")) * 1.30
    a.set_ylim(-top * 0.16, top)
    for i, s in enumerate(sets):
        a.text(i - w / 2, -top * 0.045, f"{s['epi'].median():.4f}", ha="center", fontsize=7, color=EC)
        a.text(i + w / 2, -top * 0.045, f"{s['fib'].median():.4f}", ha="center", fontsize=7, color=FC)
        a.text(i, -top * 0.115, f"n={len(s['epi'].dropna())}", ha="center", fontsize=7.5)
    a.axhline(0, color="0.7", lw=0.6)
    for i in (1.5, 3.5):
        a.axvline(i, color="0.85", lw=1.0)
    a.set_xticks(range(len(sets)))
    a.set_xticklabels(NAMES, fontsize=8)
    a.set_ylabel(ylab, fontsize=9.5)
    a.set_title(title, fontsize=11, loc="left")
    a.legend(handles=[Patch(facecolor=EC, alpha=0.8, label="epithelial marker score"),
                      Patch(facecolor=FC, alpha=0.8, label="fibroblast marker score")],
             fontsize=9, frameon=False, loc="upper center", ncol=2)


def main():
    ax = pd.read_csv(C.RUN / "annotations_raw.csv", dtype={"entity_id": str}).set_index("entity_id")["annotation"].astype(str)
    at = pd.read_csv(C.RUN / "annotations_tracer_whole.csv", dtype={"entity_id": str}).set_index("entity_id")["annotation"].astype(str)
    both = ax.index.intersection(at.index)
    lab = pd.DataFrame({"x": ax.reindex(both), "t": at.reindex(both)})
    SW = lab.index[(lab.x == "epithelial") & (lab.t == "fibroblast")]
    CF = lab.index[(lab.x == "fibroblast") & (lab.t == "fibroblast")]
    CE = lab.index[(lab.x == "epithelial") & (lab.t == "epithelial")]

    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "_etype", "feature_name", "overlaps_nucleus"]))
    t["cell_id"] = t["cell_id"].astype(str); t["tracer_id"] = t["tracer_id"].astype(str)
    tw = t.loc[t["_etype"].astype(str) == "cell"]
    sx, st = scores(t, "cell_id"), scores(tw, "tracer_id")
    nx = scores(t.loc[t["overlaps_nucleus"].astype(bool)], "cell_id")
    nt = scores(tw.loc[tw["overlaps_nucleus"].astype(bool)], "tracer_id")

    A = [sx.reindex(SW), st.reindex(SW), sx.reindex(CF), st.reindex(CF), sx.reindex(CE), st.reindex(CE)]
    N = [nx.reindex(SW), nt.reindex(SW), nx.reindex(CF), nt.reindex(CF), nx.reindex(CE), nt.reindex(CE)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 11.5))
    paired_panel(axes[0], A, "(a) ALL transcripts", "marker score\n(fraction of the entity's transcripts)")
    paired_panel(axes[1], N, "(b) NUCLEAR transcripts only -- cannot be a neighbour's spill-over",
                 "marker score\n(fraction of the entity's nuclear transcripts)")
    fig.suptitle("Epithelial vs fibroblast marker score, same id under each segmentation\n"
                 "left pair = the ids that switch label; middle and right pairs = concordant controls",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / "fig_switch_paired_scores.png"; fig.savefig(p, dpi=160); print("wrote", p)

    for nm, S in (("ALL transcripts", A), ("NUCLEAR only", N)):
        tab = pd.DataFrame({n.replace("\n", " "): dict(n_ids=len(s["epi"].dropna()),
                                                       median_tx=s["n"].median(),
                                                       epi=s["epi"].median(), fib=s["fib"].median(),
                                                       epi_over_fib=s["epi"].median() / max(s["fib"].median(), 1e-9))
                            for n, s in zip(NAMES, S)}).T
        print(f"\n{nm}:\n" + tab.round(4).to_string())


if __name__ == "__main__":
    main()
