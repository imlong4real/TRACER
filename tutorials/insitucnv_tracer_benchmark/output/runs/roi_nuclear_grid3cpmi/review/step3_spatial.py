"""Is the inferred CNV a spatial field rather than a per-cell genotype?

The arms smooth each entity over its 200 nearest SPATIAL neighbours, which in
this 500x500 um ROI is a ~92 um disc holding 10.6% of all entities. If that
dominates, a cell's CNV profile reports its neighbourhood, not its genome.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import cnv_review_common as C
from step1_heatmap import CLONE_COLORS, COMPARTMENT_COLORS

OUT = C.Path(__file__).parent
TITLES = {"raw": "Xenium (original segmentation)", "tracer_whole": "TRACER-refined (whole cells)"}


def clone4_similarity(adata):
    X = C.dense_cnv(adata)
    truth = adata.obs["truth_clone"].astype(str).to_numpy()
    comp = adata.obs["compartment"].astype(str).to_numpy()
    ref = X[(truth == "4") & (comp == "tumor")].mean(axis=0)
    refc = ref - ref.mean()
    Xc = X - X.mean(axis=1, keepdims=True)
    return (Xc @ refc) / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(refc) + 1e-12)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
    for r, arm in enumerate(("raw", "tracer_whole")):
        adata, _ = C.load_arm(arm)
        xy = adata.obsm["spatial"]
        sim = clone4_similarity(adata)

        ax = axes[r, 0]
        for cat in ("not_epithelial", "0", "4", "1", "2", "3"):
            m = (adata.obs["truth_clone"].astype(str) == cat).to_numpy()
            if m.sum():
                ax.scatter(xy[m, 0], xy[m, 1], s=7, color=CLONE_COLORS[cat],
                           label=f"{cat} ({m.sum()})", linewidths=0)
        ax.legend(fontsize=7, markerscale=2, frameon=False, loc="upper left")
        ax.set_title("full-slide truth clone (cnv_epi)", fontsize=10)

        ax = axes[r, 1]
        for cat, col in COMPARTMENT_COLORS.items():
            m = (adata.obs["compartment"].astype(str) == cat).to_numpy()
            if m.sum():
                ax.scatter(xy[m, 0], xy[m, 1], s=7, color=col, label=f"{cat} ({m.sum()})", linewidths=0)
        ax.legend(fontsize=7, markerscale=2, frameon=False, loc="upper left")
        ax.set_title("compartment (from marker expression)", fontsize=10)

        ax = axes[r, 2]
        s = ax.scatter(xy[:, 0], xy[:, 1], c=sim, s=7, cmap="RdBu_r",
                       norm=Normalize(-1, 1), linewidths=0)
        plt.colorbar(s, ax=ax, fraction=0.045, pad=0.02, label="corr. with clone-4 CNV profile")
        ax.set_title("per-entity CNV similarity to the clone-4 profile", fontsize=10)

        for c in range(3):
            axes[r, c].set_aspect("equal")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(TITLES[arm], fontsize=12)

    fig.suptitle("Spatial check after the smoothing fix (expression kNN, k=100): does the CNV readout "
                 "still track position rather than genotype?\n"
                 "previously, with a 200-NN spatial kernel (~92 um, 10.6% of the ROI), the right column "
                 "was a pure spatial gradient",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT / "step3_cnv_is_spatial.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
