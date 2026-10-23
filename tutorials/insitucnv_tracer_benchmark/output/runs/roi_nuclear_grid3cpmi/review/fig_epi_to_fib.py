"""Are the Xenium-epithelial cells that TRACER calls fibroblast really fibroblasts?"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import cnv_review_common as C

OUT = C.Path(__file__).parent
G = ["converted\n(X:epi -> T:fib)", "concordant\nfibroblast", "concordant\nepithelial"]


def main():
    ent = pd.read_csv(OUT / "nuclear_scores_tracer_entities.csv", index_col=0)
    ent.index = ent.index.astype(str)
    geo = pd.read_csv(OUT / "converted_split_geometry.csv", dtype={"tracer_id": str, "cell_id": str})
    conv = pd.read_csv(OUT / "converted_entities.csv", dtype=str)["tracer_id"]

    # regenerate group memberships cheaply
    from nuclear_test import marker_sets
    ax = pd.read_csv(C.RUN / "annotations_raw.csv").set_index("entity_id")["annotation"].astype(str)
    at = pd.read_csv(C.RUN / "annotations_tracer_whole.csv").set_index("entity_id")["annotation"].astype(str)
    prov = pd.read_csv(OUT / "converted_entity_provenance.csv", dtype={"tracer_id": str, "dom_all": str})

    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28, left=0.06, right=0.98, top=0.88, bottom=0.09)

    # (a) nuclear-only marker scores
    ax1 = fig.add_subplot(gs[0, 0])
    sets = {}
    sets[G[0]] = ent.reindex(conv.astype(str)).dropna(subset=["n_tx"])
    # concordant groups from annotations + provenance of ALL tracer entities is costly;
    # approximate with annotation label alone, which is exact for these two groups.
    fibs = pd.Index(at.index[at == "fibroblast"].astype(str)).difference(conv.astype(str))
    epis = pd.Index(at.index[at == "epithelial"].astype(str))
    sets[G[1]] = ent.reindex(fibs).dropna(subset=["n_tx"])
    sets[G[2]] = ent.reindex(epis).dropna(subset=["n_tx"])
    w = 0.36
    for off, col, lab in ((-w/2, "#2e7d32", "epithelial markers"), (w/2, "#d73027", "fibroblast markers")):
        key = "s_epithelial" if "epi" in lab else "s_fibroblast"
        vals = [sets[g][key].dropna().to_numpy() for g in G]
        bp = ax1.boxplot(vals, positions=np.arange(3) + off, widths=w * 0.92,
                         patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(col); b.set_alpha(0.85)
        for m in bp["medians"]:
            m.set_color("k")
        ax1.plot([], [], color=col, lw=6, alpha=0.85, label=lab)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(G, fontsize=8)
    ax1.set_ylabel("marker score, NUCLEAR transcripts only")
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title("(a) nuclear-only identity: the converted entities look\n     like fibroblasts, not epithelium",
                  fontsize=9.5, loc="left")

    # (b) within-cell marker split
    ax2 = fig.add_subplot(gs[0, 1])
    d = geo
    x = np.arange(2)
    for off, col, key, lab in ((-0.18, "#2e7d32", ("epi_piece", "epi_rest"), "epithelial markers"),
                               (0.18, "#d73027", ("fib_piece", "fib_rest"), "fibroblast markers")):
        vals = [d[key[0]].to_numpy(), d[key[1]].to_numpy()]
        bp = ax2.boxplot(vals, positions=x + off, widths=0.32, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(col); b.set_alpha(0.85)
        for m in bp["medians"]:
            m.set_color("k")
        ax2.plot([], [], color=col, lw=6, alpha=0.85, label=lab)
    ax2.set_xticks(x); ax2.set_xticklabels(["carved piece\n(-> 'fibroblast')", "remainder\n(stays epithelial)"], fontsize=8)
    ax2.set_ylabel("marker transcripts per cell")
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_title(f"(b) inside ONE Xenium epithelial cell, the split is perfect:\n"
                  f"     0 epithelial markers in the piece ({int(d.epi_piece.sum())} observed vs 388 expected)",
                  fontsize=9.5, loc="left")

    # (c) concentricity
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(d["separation"].to_numpy(), bins=np.linspace(0, 2, 41), color="#4575b4", alpha=0.9, label="x, y")
    ax3.axvline(1.0, color="crimson", ls="--", lw=1.3, label="1.0 = spatially distinct")
    ax3.set_xlabel("centroid distance / mean spread")
    ax3.set_ylabel("cells")
    ax3.legend(fontsize=8, frameon=False)
    ax3.set_title(f"(c) the two pieces are CONCENTRIC, not side by side\n"
                  f"     median {d.separation.median():.2f}; only {100*(d.separation>1).mean():.1f}% separated "
                  f"(z: 0%)", fontsize=9.5, loc="left")

    # (d) multinucleation
    ax4 = fig.add_subplot(gs[1, 0])
    mor = pd.read_parquet(OUT / "xenium_cell_morphology.parquet")
    roi = pd.Index(ax.index.astype(str))
    m = mor.reindex(roi).dropna(subset=["nucleus_count"])
    src = pd.Index(sorted(set(prov["dom_all"].dropna()))).astype(str).intersection(m.index)
    epi_o = pd.Index(ax.index[ax == "epithelial"].astype(str)).intersection(m.index).difference(src)
    fib_o = pd.Index(ax.index[ax == "fibroblast"].astype(str)).intersection(m.index)
    names = ["converted\nsource cells", "other Xenium\nepithelial", "Xenium\nfibroblast"]
    vals = [100 * (m.reindex(i)["nucleus_count"] > 1).mean() for i in (src, epi_o, fib_o)]
    ns = [len(src), len(epi_o), len(fib_o)]
    ax4.bar(names, vals, color=["#984ea3", "#2e7d32", "#d73027"], alpha=0.85)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax4.text(i, v + 0.15, f"{v:.1f}%\nn={n}", ha="center", fontsize=8)
    ax4.set_ylabel("% multi-nucleate (Xenium nucleus_count > 1)")
    ax4.set_title("(d) Xenium did NOT merge two cells: the source cells are\n"
                  "     single-nucleus (OR 0.44, p=0.43) and normal-sized", fontsize=9.5, loc="left")

    # (e) CNV identity
    ax5 = fig.add_subplot(gs[1, 1])
    ad, _ = C.load_arm("tracer_whole")
    X = C.dense_cnv(ad); o = ad.obs
    truth = o["truth_clone"].astype(str).to_numpy(); comp = o["compartment"].astype(str).to_numpy()
    p4 = X[(truth == "4") & (comp == "tumor")].mean(0); p4c = p4 - p4.mean()
    Xc = X - X.mean(1, keepdims=True)
    sim = (Xc @ p4c) / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(p4c) + 1e-12)
    isc = np.isin(o.index.astype(str), conv.astype(str).to_numpy())
    grp = [("converted\nentities", isc), ("other reference\nentities", (comp == "reference") & ~isc),
           ("tumour, truth\nclone 0", (truth == "0") & (comp == "tumor")),
           ("tumour, truth\nclone 4", (truth == "4") & (comp == "tumor"))]
    bp = ax5.boxplot([sim[g] for _, g in grp], patch_artist=True, showfliers=False)
    for b, c in zip(bp["boxes"], ["#984ea3", "#999999", "#4575b4", "#d73027"]):
        b.set_facecolor(c); b.set_alpha(0.85)
    for mm in bp["medians"]:
        mm.set_color("k")
    ax5.set_xticklabels([f"{n}\n(n={int(g.sum())})" for n, g in grp], fontsize=7.5)
    ax5.set_ylabel("correlation with the clone-4 CNV profile")
    ax5.axhline(0, color="k", lw=0.5)
    ax5.set_title("(e) despite being physically tumour transcripts, the converted\n"
                  "     entities carry almost no tumour CNV signal", fontsize=9.5, loc="left")

    # (f) text summary
    ax6 = fig.add_subplot(gs[1, 2]); ax6.axis("off")
    txt = (
        "Verdict\n\n"
        "The Xenium cells are genuinely epithelial:\n"
        "  - their own nuclear transcripts score epithelial\n"
        "    (2.07 vs 0.91 fibroblast; 90% argmax epithelial)\n"
        "  - single nucleus, normal area, normal depth\n\n"
        "What TRACER creates is a NEW entity, not a\n"
        "reclassified cell:\n"
        "  - it takes ~28% of that cell's nuclear transcripts\n"
        "  - 0 of 388 expected epithelial markers\n"
        "  - concentric with the epithelial part in x, y AND z\n"
        "  - so the partition is by gene identity, not geometry\n\n"
        "Consequence: these 82 entities enter the CNV\n"
        "reference. Empirically benign (removing them moves\n"
        "reference~clone4 from 0.037 to 0.020), because the\n"
        "same gene-identity split also strips the epithelial\n"
        "expression program that drives the apparent CNV."
    )
    ax6.text(0, 1, txt, va="top", ha="left", fontsize=8.6, family="monospace")

    fig.suptitle("Are the Xenium epithelial cells that TRACER calls fibroblast really fibroblasts? -- n=82 entities",
                 fontsize=13)
    p = OUT / "fig_epi_to_fibroblast.png"; fig.savefig(p, dpi=155); print("wrote", p)


if __name__ == "__main__":
    main()
