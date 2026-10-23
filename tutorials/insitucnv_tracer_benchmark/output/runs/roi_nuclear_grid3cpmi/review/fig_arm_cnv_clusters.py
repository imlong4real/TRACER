"""Cluster-level arm CNV calls (arm_cnv.ArmCNV.score_groups) on InSituCNV's clusters.

Xenium (raw) arm of roi_nuclear_grid3cpmi. Groups = Leiden on InSituCNV's own CNV
neighbour graph (obsp cnv_neighbors_*, as cnv.tl.leiden builds cnv_leiden), re-run at
--resolution (InSituCNV used 0.4: 13 clusters, clones split into several). Each cluster's
cells are summed into one profile and scored against the normal cells OUTSIDE that cluster
(pooled reference): per-gene fold change, whole-profile shift removed, arm = weighted
median, location test against random gene groups (z), BH across arms.
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cnv_review_common as C
from arm_cnv import ArmCNV

OUT = C.Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=float, default=0.2)
    args = p.parse_args()

    a = ad.read_h5ad(C.RUN / "insitucnv_raw/adata_cnv.h5ad")
    sc.tl.leiden(a, resolution=args.resolution, neighbors_key="cnv_neighbors",
                 key_added="cluster", n_iterations=-1, random_state=0, flavor="leidenalg")
    obs = a.obs.copy()
    obs["truth"] = obs.index.map(C.load_truth()).fillna("none").astype(str)
    lab = obs["cluster"].astype(str)

    m = ArmCNV(a, stat="wmedian")                   # median / pooled / z (agreed)
    res = m.score_groups(lab)
    res.to_csv(OUT / "arm_cnv_clusters.csv", index=False)

    # cluster descriptors, ordered tumour clones first
    desc = []
    for g, o in obs.groupby(lab):
        an = o["annotation"].value_counts(normalize=True)
        tr = o["truth"].value_counts(normalize=True)
        tr_epi = tr.drop(["none", "not_epithelial"], errors="ignore")
        clone = tr_epi.idxmax() if len(tr_epi) and tr_epi.max() >= 0.5 else "-"
        desc.append(dict(group=g, n=len(o), top_type=an.idxmax(), top_frac=an.max(),
                         clone=clone, clone_frac=tr.get(clone, 0.0),
                         tumour=(o["compartment"] == "tumor").mean()))
    desc = pd.DataFrame(desc).set_index("group")
    desc["kind"] = np.where(desc["tumour"] >= 0.5, "tumour", "normal/mixed")
    order = desc.sort_values(["kind", "clone", "n"], ascending=[False, True, False]).index.tolist()

    E = res.pivot(index="group", columns="arm", values="effect").loc[order, m.arms]
    K = res.pivot(index="group", columns="arm", values="call").loc[order, m.arms]

    fig, ax = plt.subplots(figsize=(15, 1.6 + 0.42 * len(order)))
    v = np.nanquantile(np.abs(E.values), 0.98)
    im = ax.imshow(E.values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v),
                   interpolation="nearest")
    ax.set_xticks(range(len(m.arms))); ax.set_xticklabels(m.arms, rotation=90, fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"cl {g} | n={desc.loc[g, 'n']} | {desc.loc[g, 'top_type']} "
                        f"{100*desc.loc[g, 'top_frac']:.0f}% | clone {desc.loc[g, 'clone']}"
                        for g in order], fontsize=9)
    ax.axhline(int((desc.loc[order, "kind"] == "tumour").sum()) - 0.5, color="k", lw=1.5)
    ii, jj = np.where(K.values != 0)
    ax.scatter(jj, ii, s=30, marker="o", facecolor="none", edgecolor="k", lw=1.2)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("arm effect (log fold change)")
    ax.set_title(f"Arm calls from counts, Xenium arm, InSituCNV clusters at Leiden resolution "
                 f"{args.resolution:g}\nmedian per-gene log fold change vs normal cells outside "
                 f"the cluster; circles = called (test vs random gene groups, BH q<0.05 within "
                 f"cluster)", fontsize=10.5, loc="left")
    fig.tight_layout()
    out = OUT / "fig_arm_cnv_clusters.png"; fig.savefig(out, dpi=140); print("wrote", out)

    print(desc.loc[order].round(2).to_string())
    print("\ncalled arms per cluster:")
    for g in order:
        c = K.loc[g]
        print(f"  cl {g:>2} ({desc.loc[g, 'kind']:12s} clone {desc.loc[g, 'clone']}): "
              f"gains {list(c[c > 0].index)}  losses {list(c[c < 0].index)}")


if __name__ == "__main__":
    main()
