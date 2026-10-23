"""Task-2 figure: de novo clones in each arm, matched across arms."""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
warnings.filterwarnings("ignore")
import scanpy as sc; sc.settings.verbosity = 0
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as ari, adjusted_mutual_info_score as ami

import cnv_review_common as C
from denovo_clones import call_clones, project_to_cells, clone_profiles, genome_grid
from step1_heatmap import genome_axis

OUT = C.Path(__file__).parent
RES = 0.1
DISP = {"raw": "Xenium (original segmentation)", "tracer_whole": "TRACER-refined (whole cells)"}
PAL = ["#d73027", "#4575b4", "#1a9850", "#984ea3", "#ff7f00", "#a65628", "#999999"]


def main():
    data = {a: C.load_arm(a) for a in ("raw", "tracer_whole")}
    ext, offsets, total = genome_axis({a: w for a, (_, w) in data.items()})
    g = np.arange(0, total, 1e6)

    subs, clones, profs = {}, {}, {}
    for arm, (ad, wins) in data.items():
        sub, cl = call_clones(ad, resolution=RES)
        subs[arm], clones[arm] = sub, cl
        x, _ = genome_grid(sub, wins, ext, offsets, total)
        profs[arm] = clone_profiles(sub, cl, x, g)

    A, B = clones["raw"], clones["tracer_whole"]
    # Every clone in a given arm shares a large common epithelial-vs-reference
    # component, so RAW profile correlation between arms is ~0.8-0.9 for every
    # pair and cannot identify which clone is which (it mismatched a pair when
    # tried). Match on the clone-SPECIFIC residual instead: subtract each arm's
    # overall tumour mean profile, leaving only what distinguishes its clones.
    resid = {}
    for arm in profs:
        w = np.array([(subs[arm].obs["clone"].astype(str) == c).sum() for c in clones[arm]], float)
        base = np.average(np.vstack([profs[arm][c] for c in clones[arm]]), axis=0, weights=w)
        resid[arm] = {c: profs[arm][c] - base for c in clones[arm]}
    M = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            u = resid["raw"][a] - resid["raw"][a].mean(); v = resid["tracer_whole"][b] - resid["tracer_whole"][b].mean()
            M[i, j] = u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    ri, ci = linear_sum_assignment(-M)
    # colour by MATCHED identity so the same colour means the same clone across arms
    colour = {}
    for k, (i, j) in enumerate(zip(ri, ci)):
        colour[("raw", A[i])] = PAL[k]; colour[("tracer_whole", B[j])] = PAL[k]
    for arm, cl in clones.items():
        for c in cl:
            colour.setdefault((arm, c), "#cccccc")

    proj = {a: project_to_cells(a, subs[a]) for a in subs}
    com = proj["raw"].index.intersection(proj["tracer_whole"].index)
    rl = proj["raw"].reindex(com).astype(str); tl = proj["tracer_whole"].reindex(com).astype(str)
    ok = (rl != "filtered_small_clone") & (tl != "filtered_small_clone")
    ARI, AMI = ari(rl[ok], tl[ok]), ami(rl[ok], tl[ok])

    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.85, 0.8], hspace=0.30, wspace=0.18,
                          left=0.07, right=0.97, top=0.92, bottom=0.06)

    for k, arm in enumerate(("raw", "tracer_whole")):
        sub = subs[arm]; xy = sub.obsm["spatial"]; lab = sub.obs["clone"].astype(str).to_numpy()
        ax = fig.add_subplot(gs[0, k])
        for c in clones[arm]:
            m = lab == c
            ax.scatter(xy[m, 0], xy[m, 1], s=12, color=colour[(arm, c)], label=f"clone {c} (n={m.sum()})",
                       linewidths=0)
        m = lab == "filtered_small_clone"
        if m.sum():
            ax.scatter(xy[m, 0], xy[m, 1], s=8, color="#eeeeee", label=f"filtered (n={m.sum()})", linewidths=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, markerscale=1.6, frameon=False, loc="upper left")
        ax.set_title(f"{DISP[arm]}\nde novo clones in space ({len(clones[arm])} clones, Leiden res={RES})",
                     fontsize=10)

        axp = fig.add_subplot(gs[1, k])
        for c in clones[arm]:
            axp.plot(g, profs[arm][c], color=colour[(arm, c)], lw=1.0, label=f"clone {c}")
        for ch in ext.index[1:]:
            axp.axvline(offsets[ch], color="k", lw=0.35, alpha=0.4)
        cen = [offsets[ch] + (ext.loc[ch, "hi"] - ext.loc[ch, "lo"]) / 2 for ch in ext.index]
        axp.set_xticks(cen); axp.set_xticklabels([ch.replace("chr", "") for ch in ext.index], fontsize=6)
        axp.set_xlim(0, total); axp.set_ylim(-0.45, 0.45); axp.axhline(0, color="k", lw=0.4)
        axp.set_ylabel("mean CNV", fontsize=8); axp.tick_params(labelsize=7)
        axp.legend(fontsize=7, ncol=4, frameon=False)
        axp.set_title("clone mean CNV profiles (matched colours across arms)", fontsize=9, loc="left")

    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(M, cmap="RdBu_r", norm=Normalize(-1, 1))
    ax.set_xticks(range(len(B))); ax.set_xticklabels([f"TRACER {b}" for b in B], fontsize=8)
    ax.set_yticks(range(len(A))); ax.set_yticklabels([f"Xenium {a}" for a in A], fontsize=8)
    for i in range(len(A)):
        for j in range(len(B)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="w" if abs(M[i, j]) > 0.6 else "k")
    for i, j in zip(ri, ci):
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="lime", lw=2.5))
    plt.colorbar(im, ax=ax, fraction=0.045, label="clone-specific residual correlation")
    ax.set_title(f"(a) cross-arm correlation of clone-SPECIFIC residual profiles; green = best match\n"
                 f"mean matched r = {np.mean([M[i,j] for i,j in zip(ri,ci)]):+.3f}", fontsize=9, loc="left")

    ax = fig.add_subplot(gs[2, 1])
    ct = pd.crosstab(rl[ok], tl[ok])
    im = ax.imshow(ct.values, cmap="Greys")
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels([f"TRACER {c}" for c in ct.columns], fontsize=8)
    ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels([f"Xenium {c}" for c in ct.index], fontsize=8)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="w" if v > ct.values.max() * 0.55 else "k")
    ax.set_title(f"(b) same 10X cells, which clone did each arm put them in?\n"
                 f"n={int(ok.sum())} cells   ARI={ARI:.3f}   AMI={AMI:.3f}", fontsize=9, loc="left")

    fig.suptitle("De novo clone discovery in each arm independently -- do we find the same clones, "
                 "in the same places?", fontsize=13)
    out = OUT / "denovo_clones.png"; fig.savefig(out, dpi=150); print("wrote", out)

    print(f"\nres={RES}: Xenium {len(A)} clones {[int((subs['raw'].obs['clone'].astype(str)==a).sum()) for a in A]}, "
          f"TRACER {len(B)} clones {[int((subs['tracer_whole'].obs['clone'].astype(str)==b).sum()) for b in B]}")
    print(f"mean matched profile r = {np.mean([M[i,j] for i,j in zip(ri,ci)]):+.3f}")
    print(f"cross-arm ARI={ARI:.3f} AMI={AMI:.3f} on {int(ok.sum())} shared cells")
    for arm in subs:
        t = pd.crosstab(subs[arm].obs["clone"].astype(str), subs[arm].obs["truth_clone"].astype(str))
        print(f"\n{DISP[arm]} de novo clone vs full-slide truth:\n{t.to_string()}")
    for i, j in zip(ri, ci):
        ca = subs["raw"].obsm["spatial"][subs["raw"].obs["clone"].astype(str).to_numpy() == A[i]].mean(0)
        cb = subs["tracer_whole"].obsm["spatial"][subs["tracer_whole"].obs["clone"].astype(str).to_numpy() == B[j]].mean(0)
        print(f"matched Xenium {A[i]} <-> TRACER {B[j]}: centroids {np.hypot(*(ca-cb)):.0f} um apart, r={M[i,j]:+.3f}")


if __name__ == "__main__":
    main()
