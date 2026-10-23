"""Paired marker scores for the SAME id under Xenium vs TRACER segmentation.

Group of interest : ids labelled epithelial by Xenium and fibroblast by TRACER.
Controls          : ids concordant fibroblast, and concordant epithelial,
                    each shown under both segmentations.

Score = the annotator's own definition (annotate_entities.marker_scores):
fraction of the entity's transcripts belonging to the marker set. Shown for all
transcripts and for NUCLEAR transcripts only -- a transcript inside a nucleus
cannot be a neighbour's spill-over, so the nuclear panels discriminate
"contaminated fibroblast that TRACER cleaned" from "epithelial cell whose
markers TRACER removed".
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns

OUT = C.Path(__file__).parent
ms = yaml.safe_load(open(C.BENCH / "config/panels.yaml"))["marker_sets"]
EPI, FIB = set(map(str, ms["epithelial"])), set(map(str, ms["fibroblast"]))
NAMES = ["switch\nXenium", "switch\nTRACER", "conc. fib\nXenium", "conc. fib\nTRACER",
         "conc. epi\nXenium", "conc. epi\nTRACER"]
COLS = ["#e31a1c", "#e31a1c", "#d9a0a0", "#d9a0a0", "#2e7d32", "#2e7d32"]
ALPHA = [0.45, 0.95] * 3


def scores(df, col):
    d = df.loc[~df[col].isin(UNASSIGNED_TOKENS)]
    tot = d.groupby(col, observed=True).size().rename("n")
    e = d.loc[d["feature_name"].astype(str).isin(EPI)].groupby(col, observed=True).size()
    f = d.loc[d["feature_name"].astype(str).isin(FIB)].groupby(col, observed=True).size()
    o = pd.DataFrame({"n": tot})
    o["n_epi"] = e.reindex(o.index).fillna(0).astype(int)
    o["n_fib"] = f.reindex(o.index).fillna(0).astype(int)
    o["epi"] = o["n_epi"] / o["n"]
    o["fib"] = o["n_fib"] / o["n"]
    o.index = o.index.astype(str)
    return o


def box(a, sets, key, ylab, title, ymin=-0.02):
    data = [s[key].dropna().to_numpy() for s in sets]
    bp = a.boxplot(data, patch_artist=True, showfliers=False, widths=0.62)
    for b, c, al in zip(bp["boxes"], COLS, ALPHA):
        b.set_facecolor(c); b.set_alpha(al)
    for m in bp["medians"]:
        m.set_color("k")
    rng = np.random.default_rng(0)
    for i, v in enumerate(data):
        a.scatter(rng.normal(i + 1, 0.055, len(v)), v, s=3.5, color="k", alpha=0.22, zorder=3)
    a.set_xticklabels(NAMES, fontsize=7.5)
    a.set_ylabel(ylab, fontsize=9)
    a.set_title(title, fontsize=9.5, loc="left")
    top = max(np.percentile(v, 97) for v in data if len(v)) * 1.25
    a.set_ylim(ymin * top / 0.02 if ymin < 0 else ymin, top)
    for i, v in enumerate(data):
        a.text(i + 1, a.get_ylim()[0] * 0.55, f"n={len(v)}\nmed {np.median(v):.4f}",
               ha="center", fontsize=6.5)


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
    nuc = t["overlaps_nucleus"].astype(bool)

    sx, st = scores(t, "cell_id"), scores(tw, "tracer_id")
    nx = scores(t.loc[nuc], "cell_id")
    nt = scores(tw.loc[tw["overlaps_nucleus"].astype(bool)], "tracer_id")
    S_all = [sx.reindex(SW), st.reindex(SW), sx.reindex(CF), st.reindex(CF), sx.reindex(CE), st.reindex(CE)]
    S_nuc = [nx.reindex(SW), nt.reindex(SW), nx.reindex(CF), nt.reindex(CF), nx.reindex(CE), nt.reindex(CE)]

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26, left=0.06, right=0.98, top=0.88, bottom=0.07)

    box(fig.add_subplot(gs[0, 0]), S_all, "epi", "epithelial score\n(fraction of transcripts)",
        "(a) epithelial score, ALL transcripts")
    box(fig.add_subplot(gs[0, 1]), S_all, "fib", "fibroblast score\n(fraction of transcripts)",
        "(b) fibroblast score, ALL transcripts")
    box(fig.add_subplot(gs[1, 0]), S_nuc, "epi", "epithelial score\n(NUCLEAR transcripts only)",
        "(d) epithelial score, NUCLEAR transcripts only")
    box(fig.add_subplot(gs[1, 1]), S_nuc, "fib", "fibroblast score\n(NUCLEAR transcripts only)",
        "(e) fibroblast score, NUCLEAR transcripts only")

    # (c) paired movement, properly scaled
    a = fig.add_subplot(gs[0, 2])
    A, B = sx.reindex(SW), st.reindex(SW)
    a.scatter(sx.reindex(CE)["epi"], sx.reindex(CE)["fib"], s=9, color="#2e7d32", alpha=0.25,
              label=f"conc. epithelial, Xenium ({len(CE)})", linewidths=0)
    a.scatter(st.reindex(CF)["epi"], st.reindex(CF)["fib"], s=9, color="#d9a0a0", alpha=0.5,
              label=f"conc. fibroblast, TRACER ({len(CF)})", linewidths=0)
    ok = A["epi"].notna() & B["epi"].notna()
    for i in SW[ok]:
        a.annotate("", xy=(B.loc[i, "epi"], B.loc[i, "fib"]), xytext=(A.loc[i, "epi"], A.loc[i, "fib"]),
                   arrowprops=dict(arrowstyle="->", color="#e31a1c", lw=0.6, alpha=0.6))
    a.scatter(A["epi"], A["fib"], s=18, facecolor="none", edgecolor="#e31a1c", lw=0.9,
              label="switchers, Xenium", zorder=5)
    a.scatter(B["epi"], B["fib"], s=22, color="#e31a1c", marker="^", label="switchers, TRACER", zorder=5)
    a.set_xlim(-0.001, 0.028); a.set_ylim(-0.002, 0.05)
    a.set_xlabel("epithelial score"); a.set_ylabel("fibroblast score")
    a.legend(fontsize=7, frameon=False, loc="upper right")
    a.set_title("(c) each switcher's move: tail = Xenium, head = TRACER", fontsize=9.5, loc="left")

    # (f) absolute marker counts
    a = fig.add_subplot(gs[1, 2])
    x = np.arange(6); w = 0.38
    me = [s["n_epi"].median() for s in S_all]; mf = [s["n_fib"].median() for s in S_all]
    a.bar(x - w / 2, me, w, color="#2e7d32", alpha=0.85, label="epithelial marker transcripts")
    a.bar(x + w / 2, mf, w, color="#d73027", alpha=0.85, label="fibroblast marker transcripts")
    for i in range(6):
        a.text(i - w / 2, me[i] + 0.4, f"{me[i]:.0f}", ha="center", fontsize=7.5)
        a.text(i + w / 2, mf[i] + 0.4, f"{mf[i]:.0f}", ha="center", fontsize=7.5)
    a.set_xticks(x); a.set_xticklabels(NAMES, fontsize=7.5)
    a.set_ylabel("median marker transcripts per entity")
    a.legend(fontsize=8, frameon=False)
    a.set_title("(f) absolute counts (fractions hide the size change)", fontsize=9.5, loc="left")

    fig.suptitle("Marker scores for the SAME id under Xenium vs TRACER segmentation -- "
                 "group of interest: epithelial (Xenium) -> fibroblast (TRACER), n=78",
                 fontsize=13)
    p = OUT / "fig_switch_scores.png"; fig.savefig(p, dpi=155); print("wrote", p)

    tab = pd.DataFrame({
        n.replace("\n", " "): dict(n_ids=len(s.dropna(subset=["epi"])), median_tx=s["n"].median(),
                                   epi_score=s["epi"].median(), fib_score=s["fib"].median(),
                                   n_epi=s["n_epi"].median(), n_fib=s["n_fib"].median())
        for n, s in zip(NAMES, S_all)}).T
    print("\nALL transcripts:\n" + tab.round(4).to_string())
    tab2 = pd.DataFrame({
        n.replace("\n", " "): dict(n_ids=len(s.dropna(subset=["epi"])), median_nuc_tx=s["n"].median(),
                                   epi_score=s["epi"].median(), fib_score=s["fib"].median(),
                                   n_epi=s["n_epi"].median(), n_fib=s["n_fib"].median())
        for n, s in zip(NAMES, S_nuc)}).T
    print("\nNUCLEAR transcripts only:\n" + tab2.round(4).to_string())


if __name__ == "__main__":
    main()
