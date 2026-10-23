"""CONTROL: what does a genuinely adjacent pair of TRACER whole cells look like?

The switcher remnant/partial pairs are ~90% hull-nested and the split is
predicted by gene (0.81) far better than by position (0.59). That is only
meaningful against a baseline. Here the baseline is TRACER's own output: pairs
of distinct `cell` entities, one annotated epithelial and one fibroblast, that
sit as close together as TRACER ever puts two whole cells.

Identical metrics, identical LOO rules (imported, not re-implemented), and the
two components are subsampled to the same transcript counts as a real switcher
pair so neither depth nor density can explain a difference.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, ConvexHull, Delaunay
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns
from fig_split_gene_vs_space import bal_acc, gene_loo, pos_loo, K

OUT = C.Path(__file__).parent
RNG = np.random.default_rng(0)
MIN_TX = 200
N_PAIRS = 40
CA, CB = "#d73027", "#1b7837"


def frac_in_hull(pts, hull_pts):
    try:
        d = Delaunay(hull_pts)
    except Exception:
        return np.nan
    return float((d.find_simplex(pts) >= 0).mean())


def pair_metrics(A, B):
    """Exactly the geometry + LOO metrics used on the switcher pairs."""
    cA, cB = A.mean(0), B.mean(0)
    d3 = float(np.linalg.norm(cA - cB))
    sA = float(np.sqrt(((A - cA) ** 2).sum(1).mean()))
    sB = float(np.sqrt(((B - cB) ** 2).sum(1).mean()))
    return dict(d3=d3, spread_A=sA, spread_B=sB, sep=d3 / ((sA + sB) / 2),
                frac_A_in_B_hull=frac_in_hull(A, B), frac_B_in_A_hull=frac_in_hull(B, A))


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
    w = w.loc[w["ann"].isin(["epithelial", "fibroblast"])]

    cen = w.groupby("tracer_id").agg(n=("x", "size"), x=("x", "mean"), y=("y", "mean"),
                                     z=("z", "mean"), ann=("ann", "first"))
    cen = cen.loc[cen["n"] >= MIN_TX]
    epi = cen.loc[cen["ann"] == "epithelial"]
    fib = cen.loc[cen["ann"] == "fibroblast"]
    print(f"whole cells with >={MIN_TX} tx: {len(epi)} epithelial, {len(fib)} fibroblast")

    # nearest fibroblast for every epithelial cell -> two control tiers
    tree = cKDTree(fib[["x", "y", "z"]].to_numpy())
    d, j = tree.query(epi[["x", "y", "z"]].to_numpy(), k=1)
    cand = pd.DataFrame({"A": epi.index, "B": fib.index[j], "d": d}).sort_values("d")

    def take(c, n):
        used, out = set(), []
        for r in c.itertuples(index=False):
            if r.A in used or r.B in used:
                continue
            used.update([r.A, r.B]); out.append((r.A, r.B, r.d))
            if len(out) == n:
                break
        return out

    # TIER 1 "crowded": the n closest pairs in the whole ROI (hardest control)
    crowded = take(cand, N_PAIRS)
    # TIER 2 "typical": a random draw from all nearest-neighbour pairs
    typical = take(cand.sample(frac=1.0, random_state=0), N_PAIRS)
    for nm, pk in [("crowded", crowded), ("typical", typical)]:
        dd = [x[2] for x in pk]
        print(f"{nm}: {len(pk)} pairs, centroid distance "
              f"{min(dd):.2f}-{max(dd):.2f} um (median {np.median(dd):.2f})")

    by_ent = dict(tuple(w.groupby("tracer_id")))
    # size template: the real switcher pairs, so density cannot explain a difference
    sizes = sw[["n_part", "n_rem"]].to_numpy()

    rows = []
    for tier, picks in [("crowded", crowded), ("typical", typical)]:
      for i, (a, b, dd) in enumerate(picks):
          ga, gb = by_ent[a], by_ent[b]
          nA, nB = sizes[i % len(sizes)]
          capped = (len(ga) < nA) or (len(gb) < nB)
          if len(ga) > nA:
              ga = ga.iloc[RNG.choice(len(ga), nA, replace=False)]
          if len(gb) > nB:
              gb = gb.iloc[RNG.choice(len(gb), nB, replace=False)]
          A = ga[["x", "y", "z"]].to_numpy(float); B = gb[["x", "y", "z"]].to_numpy(float)
          m = pair_metrics(A, B)

          y = np.r_[np.ones(len(A), int), np.zeros(len(B), int)]
          genes = np.r_[ga["feature_name"].astype(str).to_numpy(),
                        gb["feature_name"].astype(str).to_numpy()]
          xyz = np.vstack([A, B])
          m.update(tier=tier, A=a, B=b, nA=len(A), nB=len(B), capped=capped,
                   acc_gene=bal_acc(y, gene_loo(genes, y)),
                   acc_pos=bal_acc(y, pos_loo(xyz, y)))
          rows.append(m)

    ct = pd.DataFrame(rows)
    ct.to_csv(OUT / "control_adjacent_pairs.csv", index=False)
    print(f"size-capped pairs: {int(ct['capped'].sum())}/{len(ct)}")
    TY = ct.loc[ct["tier"] == "typical"]
    CR_ = ct.loc[ct["tier"] == "crowded"]

    # ---- figure: two control tiers vs switchers ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    COL = ["#4575b4", "#74add1", "#984ea3"]
    LAB = [f"CONTROL typical\ntwo whole cells\n(n={len(TY)})",
           f"CONTROL crowded\n40 closest of 668\n(n={len(CR_)})",
           f"SWITCHERS\nremnant vs partial\n(n={len(sw)})"]

    def trio(ax, vals, ylab, title, hline=None, hlab=None, logy=False):
        bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1.6))
        for patch, c in zip(bp["boxes"], COL):
            patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
        for k, (v, c) in enumerate(zip(vals, COL), start=1):
            ax.scatter(np.full(len(v), k) + RNG.normal(0, 0.06, len(v)), v, s=20,
                       color=c, alpha=0.8, linewidths=0)
        if hline is not None:
            ax.axhline(hline, color="0.5", ls="--", lw=1.1, label=hlab)
            ax.legend(fontsize=8, frameon=False)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(LAB, fontsize=8.2)
        ax.set_ylabel(ylab)
        med = "  |  ".join(f"{np.median(v):.2f}" for v in vals)
        u = mannwhitneyu(vals[1], vals[2])
        ax.set_title(f"{title}\nmedians {med}   (crowded vs switcher p={u.pvalue:.1e})",
                     fontsize=9.5, loc="left")

    trio(axes[0], [TY["sep"].values, CR_["sep"].values, sw["sep"].values],
         "3D centroid distance / mean spread", "(a) are the two clouds displaced?",
         hline=1.0, hlab="1.0 = spatially distinct", logy=True)

    def nest(d, a, b):
        return np.r_[d[a].values, d[b].values] * 100
    trio(axes[1], [nest(TY, "frac_A_in_B_hull", "frac_B_in_A_hull"),
                   nest(CR_, "frac_A_in_B_hull", "frac_B_in_A_hull"),
                   nest(sw, "frac_part_in_rem_hull", "frac_rem_in_part_hull")],
         "% of points inside the other hull", "(b) do the hulls contain each other?")
    axes[1].set_ylim(-3, 103)

    ax = axes[2]
    for d, c, lab in [(TY, COL[0], "control typical"), (CR_, COL[1], "control crowded"),
                      (swacc, COL[2], "switchers")]:
        ax.scatter(d["acc_pos"], d["acc_gene"], s=40, color=c, alpha=0.85, linewidths=0,
                   label=lab)
    ax.plot([0.45, 1.02], [0.45, 1.02], ls=":", color="0.6", lw=0.9)
    ax.axhline(0.5, color="0.85", lw=0.7); ax.axvline(0.5, color="0.85", lw=0.7)
    ax.set_xlim(0.45, 1.02); ax.set_ylim(0.45, 1.02)
    ax.set_xlabel(f"position rule: LOO {K}-NN in 3D (balanced acc.)")
    ax.set_ylabel("gene rule: LOO same-gene majority (balanced acc.)")
    ax.set_title("(c) tell them apart by WHERE, or by WHAT?\nposition medians  "
                 f"{TY.acc_pos.median():.2f}  |  {CR_.acc_pos.median():.2f}  |  "
                 f"{swacc.acc_pos.median():.2f}", fontsize=9.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    fig.suptitle("Control: two adjacent TRACER whole cells vs one cell split into "
                 "remnant + partial (transcript counts matched)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p = OUT / "fig_control_adjacent_cells.png"; fig.savefig(p, dpi=150); print("wrote", p)

    cols = ["d3", "sep", "frac_A_in_B_hull", "frac_B_in_A_hull", "spread_A", "spread_B",
            "acc_gene", "acc_pos"]
    print("\nmedians by tier:")
    print(ct.groupby("tier")[cols].median().round(3).to_string())
    print("\nSWITCHER medians:")
    print(pd.Series({"d3": sw.d3.median(), "sep": sw.sep.median(),
                     "frac_A_in_B_hull": sw.frac_part_in_rem_hull.median(),
                     "frac_B_in_A_hull": sw.frac_rem_in_part_hull.median(),
                     "spread_A": sw.spread_part.median(), "spread_B": sw.spread_rem.median(),
                     "acc_gene": swacc.acc_gene.median(),
                     "acc_pos": swacc.acc_pos.median()}).round(3).to_string())


if __name__ == "__main__":
    main()
