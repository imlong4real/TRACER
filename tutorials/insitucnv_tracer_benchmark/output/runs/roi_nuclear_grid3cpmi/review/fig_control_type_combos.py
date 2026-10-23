"""CONTROL 2: adjacent whole-cell pairs across every cell-type combination.

fig_control_adjacent_cells.py did epithelial-vs-fibroblast only. This widens it
to all pairings of the abundant annotated types, INCLUDING same-type pairs,
which is the informative addition: two adjacent epithelial cells should be hard
to tell apart by gene but trivial by position. That pins down what each rule is
actually measuring before the switcher pairs are read against them.

Same LOO rules and same size-matching as before (imported, not re-implemented).
Every pair is the closest available of its combination, i.e. the hardest control.
"""
from __future__ import annotations
import sys, warnings, itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns
from fig_split_gene_vs_space import bal_acc, gene_loo, pos_loo, K
from fig_control_adjacent_cells import pair_metrics, MIN_TX

OUT = C.Path(__file__).parent
RNG = np.random.default_rng(0)
N_PAIRS = 40
TYPES = ["epithelial", "fibroblast", "myeloid"]
SHORT = {"epithelial": "epi", "fibroblast": "fib", "myeloid": "mye"}


def closest_pairs(cA, cB, n, same):
    """n closest disjoint A-B centroid pairs; `same` excludes self-matches."""
    tree = cKDTree(cB[["x", "y", "z"]].to_numpy())
    k = 2 if same else 1
    d, j = tree.query(cA[["x", "y", "z"]].to_numpy(), k=k)
    if same:
        d, j = d[:, 1], j[:, 1]          # skip self
    cand = pd.DataFrame({"A": cA.index, "B": cB.index[j], "d": d})
    if same:                              # A-B and B-A are the same pair
        key = cand.apply(lambda r: tuple(sorted((r.A, r.B))), axis=1)
        cand = cand.loc[~key.duplicated()]
    cand = cand.sort_values("d")
    used, out = set(), []
    for r in cand.itertuples(index=False):
        if r.A in used or r.B in used:
            continue
        used.update([r.A, r.B]); out.append((r.A, r.B, r.d))
        if len(out) == n:
            break
    return out


def main():
    sw = pd.read_csv(OUT / "switcher_pairs_3d.csv")
    swacc = pd.read_csv(OUT / "split_gene_vs_space.csv")
    ann = pd.read_csv(C.RUN / "annotations_tracer_whole.csv").set_index("entity_id")["annotation"]

    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "_etype", "feature_name", "x", "y", "z"]))
    t["tracer_id"] = t["tracer_id"].astype(str)
    w = t.loc[t["_etype"].astype(str) == "cell"].copy()
    w["ann"] = w["tracer_id"].map(ann)
    w = w.loc[w["ann"].isin(TYPES)]

    cen = w.groupby("tracer_id").agg(n=("x", "size"), x=("x", "mean"), y=("y", "mean"),
                                     z=("z", "mean"), ann=("ann", "first"))
    cen = cen.loc[cen["n"] >= MIN_TX]
    print(cen["ann"].value_counts().to_string())
    by_ent = dict(tuple(w.groupby("tracer_id")))
    sizes = sw[["n_part", "n_rem"]].to_numpy()

    rows = []
    for ta, tb in itertools.combinations_with_replacement(TYPES, 2):
        same = ta == tb
        picks = closest_pairs(cen.loc[cen["ann"] == ta], cen.loc[cen["ann"] == tb],
                              N_PAIRS, same)
        dd = [x[2] for x in picks]
        combo = f"{SHORT[ta]}-{SHORT[tb]}"
        print(f"{combo:10s} {len(picks):3d} pairs, centroid dist "
              f"{min(dd):.1f}-{max(dd):.1f} um (median {np.median(dd):.1f})")
        for i, (a, b, _d) in enumerate(picks):
            ga, gb = by_ent[a], by_ent[b]
            nA, nB = sizes[i % len(sizes)]
            if len(ga) > nA:
                ga = ga.iloc[RNG.choice(len(ga), nA, replace=False)]
            if len(gb) > nB:
                gb = gb.iloc[RNG.choice(len(gb), nB, replace=False)]
            A = ga[["x", "y", "z"]].to_numpy(float); B = gb[["x", "y", "z"]].to_numpy(float)
            m = pair_metrics(A, B)
            y = np.r_[np.ones(len(A), int), np.zeros(len(B), int)]
            genes = np.r_[ga["feature_name"].astype(str).to_numpy(),
                          gb["feature_name"].astype(str).to_numpy()]
            m.update(combo=combo, same_type=same, A=a, B=b, nA=len(A), nB=len(B),
                     acc_gene=bal_acc(y, gene_loo(genes, y)),
                     acc_pos=bal_acc(y, pos_loo(np.vstack([A, B]), y)))
            rows.append(m)

    ct = pd.DataFrame(rows)
    ct.to_csv(OUT / "control_type_combos.csv", index=False)

    order = ["epi-epi", "fib-fib", "mye-mye", "epi-fib", "epi-mye", "fib-mye"]
    pal = {"epi-epi": "#2166ac", "fib-fib": "#4393c3", "mye-mye": "#92c5de",
           "epi-fib": "#b2182b", "epi-mye": "#d6604d", "fib-mye": "#f4a582"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    # (a) the 2x2 design: gene vs position, one point per pair
    ax = axes[0]
    for c in order:
        d = ct.loc[ct["combo"] == c]
        ax.scatter(d["acc_pos"], d["acc_gene"], s=34, color=pal[c], alpha=0.8, linewidths=0,
                   marker="s" if c.split("-")[0] == c.split("-")[1] else "o", label=c)
    ax.scatter(swacc["acc_pos"], swacc["acc_gene"], s=46, color="#984ea3", alpha=0.9,
               linewidths=0, marker="D", label="SWITCHERS")
    ax.plot([0.45, 1.02], [0.45, 1.02], ls=":", color="0.6", lw=0.9)
    ax.axhline(0.5, color="0.85", lw=0.7)
    ax.set_xlim(0.45, 1.02); ax.set_ylim(0.45, 1.02)
    ax.set_xlabel(f"position rule: LOO {K}-NN in 3D (balanced acc.)")
    ax.set_ylabel("gene rule: LOO same-gene majority (balanced acc.)")
    ax.set_title("(a) squares = same type, circles = different types\n"
                 "every real pair separates by position; switchers do not",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=2)

    # (b) position rule by combo
    def bars(ax, col, ylab, title, chance=0.5):
        vals = [ct.loc[ct["combo"] == c, col].values for c in order] + [swacc[col].values]
        labs = order + ["SWITCHERS"]
        cols = [pal[c] for c in order] + ["#984ea3"]
        bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1.5))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(0.4); patch.set_edgecolor(c)
        for k_, (v, c) in enumerate(zip(vals, cols), start=1):
            ax.scatter(np.full(len(v), k_) + RNG.normal(0, 0.06, len(v)), v, s=14,
                       color=c, alpha=0.75, linewidths=0)
        ax.axhline(chance, color="0.5", ls="--", lw=1.0, label="chance")
        ax.set_xticks(range(1, len(labs) + 1))
        ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8.5)
        ax.set_ylabel(ylab); ax.set_ylim(0.45, 1.03)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        ax.set_title(title, fontsize=9.5, loc="left")

    bars(axes[1], "acc_pos", "balanced accuracy",
         "(b) position rule: WHERE did it come from?")
    bars(axes[2], "acc_gene", "balanced accuracy",
         "(c) gene rule: WHAT gene is it?")

    fig.suptitle("Adjacent TRACER whole cells, all type combinations, vs the switcher "
                 "remnant/partial split", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p = OUT / "fig_control_type_combos.png"; fig.savefig(p, dpi=150); print("wrote", p)

    summ = ct.groupby("combo")[["d3", "sep", "frac_A_in_B_hull", "frac_B_in_A_hull",
                                "acc_gene", "acc_pos"]].median().round(3).reindex(order)
    summ.loc["SWITCHERS"] = [sw.d3.median(), sw.sep.median(), sw.frac_part_in_rem_hull.median(),
                             sw.frac_rem_in_part_hull.median(), swacc.acc_gene.median(),
                             swacc.acc_pos.median()]
    print("\n" + summ.round(3).to_string())
    print("\nsame-type vs different-type gene rule:")
    st = ct.loc[ct["same_type"], "acc_gene"]; dt = ct.loc[~ct["same_type"], "acc_gene"]
    print(f"  same {st.median():.3f} (n={len(st)}) vs diff {dt.median():.3f} (n={len(dt)}), "
          f"p={mannwhitneyu(st, dt).pvalue:.1e}")
    print("same-type vs different-type position rule:")
    st = ct.loc[ct["same_type"], "acc_pos"]; dt = ct.loc[~ct["same_type"], "acc_pos"]
    print(f"  same {st.median():.3f} vs diff {dt.median():.3f}, "
          f"p={mannwhitneyu(st, dt).pvalue:.1e}")


if __name__ == "__main__":
    main()
