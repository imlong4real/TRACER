"""How do Xenium cell_ids map onto TRACER tracer_ids, and does that mapping
manufacture the apparent 'epithelial -> fibroblast' conversion?
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import cnv_review_common as C

OUT = C.Path(__file__).parent


def main():
    brk = pd.read_csv(OUT / "converted_destination_breakdown.csv")
    sib = pd.read_csv(OUT / "converted_sibling_entities.csv")
    prov = pd.read_csv(OUT / "converted_entity_provenance.csv", dtype={"tracer_id": str, "dom_all": str})

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.30, left=0.06, right=0.98, top=0.87, bottom=0.10)

    # (a) how many whole-cell entities are centred on each Xenium cell
    ax = fig.add_subplot(gs[0, 0])
    counts = {"0\n(cell dissolves)": 239, "1\n(clean 1:1)": 1584, "2": 71, "3": 3, "86": 1}
    bars = ax.bar(list(counts), list(counts.values()),
                  color=["#d73027", "#2e7d32", "#fdae61", "#fdae61", "#7b52ab"])
    for b, v in zip(bars, counts.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 20, f"{v:,}", ha="center", fontsize=9)
    ax.set_yscale("symlog"); ax.set_ylabel("Xenium cells (log)")
    ax.set_xlabel("whole-cell TRACER entities whose dominant source is that cell")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("(a) the mapping is mostly 1:1 -- but 239 cells (12.6%) have\n"
                 "     NO TRACER entity centred on them", fontsize=10, loc="left")

    # (b) which Xenium cells lose their entity
    ax = fig.add_subplot(gs[0, 1])
    lost = {"epithelial": (102, 1029), "fibroblast": (14, 404), "myeloid": (18, 209),
            "t_cell": (4, 30), "unknown": (101, 207)}
    names = list(lost); pct = [100 * lost[k][0] / lost[k][1] for k in names]
    ax.bar(names, pct, color=["#2e7d32", "#d73027", "#7b52ab", "#8d6e63", "#bdbdbd"], alpha=0.9)
    for i, k in enumerate(names):
        ax.text(i, pct[i] + 0.8, f"{lost[k][0]}/{lost[k][1]}", ha="center", fontsize=8)
    ax.set_ylabel("% of cells with no entity centred on them")
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    ax.set_title("(b) which Xenium cells dissolve", fontsize=10, loc="left")

    # (c) ID reuse
    ax = fig.add_subplot(gs[0, 2])
    same = int((prov["tracer_id"] == prov["dom_all"]).sum()); n = len(prov)
    ax.bar(["same string", "different"], [same, n - same], color=["#e31a1c", "#cccccc"])
    ax.text(0, same + 1.5, f"{same}/{n}\n({100*same/n:.1f}%)", ha="center", fontsize=9)
    ax.set_ylabel("converted entities")
    ax.set_title("(c) the 'fibroblast' entity INHERITS the Xenium cell's ID string\n"
                 "     -- so any ID- or dominant-source link pairs them by construction",
                 fontsize=10, loc="left")

    # (d) where the cell's transcripts go
    ax = fig.add_subplot(gs[1, 0])
    cols = ["pct_converted", "pct_other_whole", "pct_partial", "pct_unassigned"]
    labs = ["the ID-inheriting\n'fibroblast' entity", "OTHER whole-cell\nentities", "partials\n(discarded)", "unassigned"]
    med = [brk[c].median() for c in cols]
    ax.bar(labs, med, color=["#e31a1c", "#2e7d32", "#fdae61", "#8c8c8c"], alpha=0.9)
    for i, v in enumerate(med):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("% of the Xenium cell's transcripts (median)")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("(d) the ID-inheriting entity holds only ~27% of the cell", fontsize=10, loc="left")

    # (e) where the EPITHELIAL MARKERS go -- the decisive panel
    ax = fig.add_subplot(gs[1, 1])
    tot = brk[["epi_conv", "epi_other_whole", "epi_partial", "epi_unassigned"]].sum()
    pct = tot / tot.sum() * 100
    bars = ax.bar(labs, pct.to_numpy(), color=["#e31a1c", "#2e7d32", "#fdae61", "#8c8c8c"], alpha=0.9)
    for b, v, c in zip(bars, pct.to_numpy(), tot.to_numpy()):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%\n({int(c)})", ha="center", fontsize=8.5)
    ax.set_ylabel("% of the cells' epithelial marker transcripts")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(0, 85)
    ax.set_title("(e) the epithelial identity is NOT kept as a partial --\n"
                 "     it is absorbed by entities centred on NEIGHBOURING cells",
                 fontsize=10, loc="left")

    # (f) is a sibling entity centred on the same cell?
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    txt = (
        "What actually happens (82 cases)\n"
        "--------------------------------\n"
        "Xenium cell X: genuinely epithelial\n"
        "  single nucleus, normal size, epithelial by\n"
        "  its own nuclear transcripts.\n\n"
        "TRACER redistributes X's transcripts:\n"
        f"   27%  -> an entity that KEEPS X's id string\n"
        f"           ({same}/{n} cases), 0 epithelial markers,\n"
        "           some fibroblast markers -> 'fibroblast'\n"
        "   44%  -> whole-cell entities centred on\n"
        "           NEIGHBOURING cells (median 8 of them)\n"
        "   23%  -> partials, discarded by tracer_whole\n\n"
        f"Another entity centred on X at all: only "
        f"{100*sib.any_centred.mean():.0f}%\n"
        f"  ...annotated epithelial: {100*(sib.centred_epithelial>0).mean():.0f}%\n\n"
        "So X does not become a fibroblast.\n"
        "X dissolves, and the remnant carrying its ID\n"
        "is what gets called fibroblast.\n\n"
        "=> 'epithelial -> fibroblast' is an artefact of\n"
        "   forcing a 1:1 link onto a many-to-many\n"
        "   redistribution."
    )
    ax.text(0, 1, txt, va="top", ha="left", fontsize=8.8, family="monospace")

    fig.suptitle("How are Xenium cell_ids linked to TRACER tracer_ids -- and does the link create the "
                 "apparent cell-type conversion?", fontsize=13)
    p = OUT / "fig_id_linkage.png"; fig.savefig(p, dpi=155); print("wrote", p)


if __name__ == "__main__":
    main()
