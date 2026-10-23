"""What does TRACER's within-cell split track: gene identity, or 3D position?

The switcher pairs' hulls are ~90% nested (fig_3d_hulls.png), which is only
"weird" if you assume the split is geometric. This asks the question directly.

For each of the 40 switcher Xenium cells, every transcript of that cell is
labelled remnant (tracer_id == cell_id) or partial (tracer_id == the epithelial
recipient). Then the SAME leave-one-out protocol predicts that label twice:

  gene rule     : majority label of the other transcripts of the same gene
                  in the same cell
  position rule : majority label of the 10 nearest other transcripts in 3D

Balanced accuracy, so class imbalance cannot inflate either. A within-cell label
permutation gives the empirical chance level for both.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom, wilcoxon
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns

OUT = C.Path(__file__).parent
RNG = np.random.default_rng(0)
K = 10
CR, CP = "#d73027", "#1b7837"


def bal_acc(y, pred):
    """Balanced accuracy; pred may be -1 for 'abstain' (counts as a coin flip)."""
    out = []
    for cls in (0, 1):
        m = y == cls
        if not m.any():
            continue
        p = pred[m]
        out.append(((p == cls).sum() + 0.5 * (p == -1).sum()) / m.sum())
    return float(np.mean(out))


def gene_loo(genes, y):
    """Leave-one-out majority label of the same gene within this cell."""
    df = pd.DataFrame({"g": genes, "y": y})
    tot = df.groupby("g")["y"].transform("size").to_numpy()
    n1 = df.groupby("g")["y"].transform("sum").to_numpy()
    n1_loo = n1 - y                      # partial-count excluding self
    n0_loo = (tot - n1) - (1 - y)        # remnant-count excluding self
    pred = np.where(n1_loo > n0_loo, 1, np.where(n0_loo > n1_loo, 0, -1))
    return pred


def pos_loo(xyz, y, k=K):
    """Leave-one-out majority label of the k nearest other transcripts in 3D."""
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)
    idx = idx[:, 1:]                     # drop self
    v = y[idx].sum(1)
    pred = np.where(v > k / 2, 1, np.where(v < k / 2, 0, -1))
    return pred


def main():
    pairs = pd.read_csv(OUT / "switcher_pairs_3d.csv")
    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "_etype", "feature_name", "x", "y", "z"]))
    t["cell_id"] = t["cell_id"].astype(str); t["tracer_id"] = t["tracer_id"].astype(str)
    t = t.loc[t["cell_id"].isin(set(pairs["cell_id"]))]
    by_cell = dict(tuple(t.groupby("cell_id")))

    rows, gene_frac = [], []
    for _, pr in pairs.iterrows():
        cid, pid = pr["cell_id"], pr["partial"]
        g = by_cell[cid]
        lab = np.where(g["tracer_id"].to_numpy() == pid, 1,
                       np.where(g["tracer_id"].to_numpy() == cid, 0, -1))
        keep = lab >= 0
        n_other = int((~keep).sum())
        g, y = g.loc[keep], lab[keep].astype(int)
        if y.min() == y.max():
            continue
        genes = g["feature_name"].astype(str).to_numpy()
        xyz = g[["x", "y", "z"]].to_numpy().astype(float)

        acc_g = bal_acc(y, gene_loo(genes, y))
        acc_p = bal_acc(y, pos_loo(xyz, y))
        yp = RNG.permutation(y)
        null_g = bal_acc(yp, gene_loo(genes, yp))
        null_p = bal_acc(yp, pos_loo(xyz, yp))

        # per-gene partial fraction, genes with >=10 transcripts in this cell
        d = pd.DataFrame({"g": genes, "y": y}).groupby("g")["y"].agg(["size", "sum"])
        d = d.loc[d["size"] >= 10]
        p_cell = y.mean()
        for n, s in d.itertuples(index=False):
            gene_frac.append((s / n, n, p_cell))

        rows.append(dict(cell_id=cid, partial=pid, n=len(y), n_partial=int(y.sum()),
                         n_other=n_other, p_cell=p_cell, acc_gene=acc_g, acc_pos=acc_p,
                         null_gene=null_g, null_pos=null_p))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "split_gene_vs_space.csv", index=False)
    gf = pd.DataFrame(gene_frac, columns=["frac", "n", "p_cell"])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # (a) per-gene partial fraction vs the within-cell binomial null
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(gf["frac"], bins=bins, weights=np.ones(len(gf)) / len(gf), color="#4575b4",
            alpha=0.9, label=f"observed ({len(gf)} gene x cell, >=10 tx)")
    null = np.concatenate([binom.rvs(int(n), p, size=20, random_state=1) / n
                           for n, p in zip(gf["n"], gf["p_cell"])])
    ax.hist(null, bins=bins, weights=np.ones(len(null)) / len(null), histtype="step",
            color="crimson", lw=1.8, label="null: labels shuffled within cell")
    ax.set_xlabel("fraction of a gene's transcripts sent to the partial")
    ax.set_ylabel("fraction of gene x cell pairs")
    extreme = ((gf["frac"] < 0.02) | (gf["frac"] > 0.98)).mean()
    n_ex = ((null < 0.02) | (null > 0.98)).mean()
    ax.set_title(f"(a) genes go one way or the other\nall-or-nothing: {100*extreme:.0f}% obs vs "
                 f"{100*n_ex:.0f}% null", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)

    # (b) leave-one-out balanced accuracy: gene vs position
    ax = axes[1]
    ax.scatter(res["acc_pos"], res["acc_gene"], s=40, color="#984ea3", alpha=0.85, linewidths=0,
               label="switcher cell")
    ax.scatter(res["null_pos"], res["null_gene"], s=26, color="0.6", alpha=0.7, linewidths=0,
               marker="x", label="same cell, labels shuffled")
    ax.plot([0.4, 1.02], [0.4, 1.02], ls=":", color="0.6", lw=0.9)
    ax.axhline(0.5, color="0.8", lw=0.7); ax.axvline(0.5, color="0.8", lw=0.7)
    ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel(f"position rule: LOO {K}-NN in 3D (balanced acc.)")
    ax.set_ylabel("gene rule: LOO same-gene majority (balanced acc.)")
    st = wilcoxon(res["acc_gene"], res["acc_pos"])
    ax.set_title(f"(b) the split is gene-defined, not geometric\nmedians gene "
                 f"{res.acc_gene.median():.3f} vs position {res.acc_pos.median():.3f} "
                 f"(p={st.pvalue:.1e})", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    # (c) one cell, xy projection, coloured by destination
    ax = axes[2]
    cid = res.sort_values("n", ascending=False)["cell_id"].iloc[0]
    pid = res.set_index("cell_id").loc[cid, "partial"]
    g = by_cell[cid]
    m_p = g["tracer_id"].to_numpy() == pid
    m_r = g["tracer_id"].to_numpy() == cid
    ax.scatter(g.loc[m_r, "x"], g.loc[m_r, "y"], s=7, color=CR, alpha=0.55, linewidths=0,
               label="-> remnant ('fibroblast')")
    ax.scatter(g.loc[m_p, "x"], g.loc[m_p, "y"], s=7, color=CP, alpha=0.75, linewidths=0,
               label="-> epithelial partial")
    ax.set_aspect("equal")
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    r = res.set_index("cell_id").loc[cid]
    ax.set_title(f"(c) Xenium cell {cid}, {int(r.n)} tx\ngene acc {r.acc_gene:.2f}, "
                 f"position acc {r.acc_pos:.2f}", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, markerscale=2)

    fig.suptitle("Why the two components share space: TRACER partitions the cell by gene, "
                 "not by geometry", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / "fig_split_gene_vs_space.png"; fig.savefig(p, dpi=150); print("wrote", p)
    print(res[["n", "n_partial", "n_other", "acc_gene", "acc_pos", "null_gene", "null_pos"]]
          .describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
