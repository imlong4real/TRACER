"""CNV UMAP and marker dotplot, raw vs TRACER.

The committed per-arm plots colour UMAPs by the arm's own `cnv_leiden`, which is
the 15/20-cluster stage-1 artefact. Here we colour by compartment, marker
annotation, full-slide truth clone and depth, and dotplot markers against the
ANNOTATION (not clones) to test whether marker calling survives TRACER's
shallower entities.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import yaml

import cnv_review_common as C
from step1_heatmap import CLONE_COLORS, COMPARTMENT_COLORS

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
OUT = C.Path(__file__).parent
ARM_TITLES = {"raw": "Xenium (original segmentation)", "tracer_whole": "TRACER-refined (whole cells)"}
REFERENCE_LABELS = ["t_cell", "b_cell", "myeloid", "fibroblast", "endothelial"]


def marker_sets() -> dict[str, list[str]]:
    with open(C.BENCH / "config/panels.yaml") as fh:
        return yaml.safe_load(fh)["marker_sets"]


def prep(arm: str):
    adata, _ = C.load_arm(arm)
    # CNV UMAP from the stored CNV PCA (arms carry X_cnv_pca but no X_umap)
    sc.pp.neighbors(adata, use_rep="X_cnv_pca", key_added="cnvu", random_state=0)
    sc.tl.umap(adata, neighbors_key="cnvu", random_state=0)
    # expression object for the dotplot: raw counts, normalised + log1p.
    # adata.X holds the smoothed M matrix by the end of the arm run, so counts
    # must come from the layer.
    expr = adata.copy()
    expr.X = expr.layers["counts"].copy()
    del expr.layers["M"]
    sc.pp.normalize_total(expr, target_sum=1e4)
    sc.pp.log1p(expr)
    return adata, expr


def umap_figure(data):
    keys = [
        ("compartment", "compartment (marker annotation -> reference/tumor)"),
        ("annotation", "marker annotation"),
        ("truth_clone", "full-slide truth clone (cnv_epi)"),
        ("n_genes", "n_genes per entity (log)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    for r, arm in enumerate(("raw", "tracer_whole")):
        adata = data[arm][0]
        xy = adata.obsm["X_umap"]
        for c, (key, title) in enumerate(keys):
            ax = axes[r, c]
            if key == "n_genes":
                v = adata.obs["n_genes"].to_numpy(float)
                s = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=3, cmap="viridis",
                               norm=matplotlib.colors.LogNorm(max(v.min(), 1), v.max()),
                               linewidths=0, rasterized=True)
                plt.colorbar(s, ax=ax, fraction=0.04, pad=0.02)
                ax.set_title(f"{title}\nmedian={int(np.median(v))}", fontsize=9)
            else:
                vals = adata.obs[key].astype(str)
                palette = (COMPARTMENT_COLORS if key == "compartment"
                           else CLONE_COLORS if key == "truth_clone" else None)
                cats = sorted(vals.unique())
                for cat in cats:
                    m = (vals == cat).to_numpy()
                    col = palette.get(cat) if palette else None
                    ax.scatter(xy[m, 0], xy[m, 1], s=3, label=f"{cat} ({m.sum()})",
                               color=col, linewidths=0, rasterized=True)
                ax.legend(fontsize=6, markerscale=2.5, frameon=False, loc="best")
                ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(ARM_TITLES[arm], fontsize=11)
    fig.suptitle("CNV-space UMAP (from X_cnv_pca), raw vs TRACER -- ovary ROI (expression-kNN smoothing k=100)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = OUT / "step0_cnv_umap.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


def dotplot_figure(data):
    ms = marker_sets()
    present = {k: [g for g in v if g in data["raw"][1].var_names and g in data["tracer_whole"][1].var_names]
               for k, v in ms.items()}
    order = ["epithelial"] + REFERENCE_LABELS
    groups = order + ["unknown"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for r, arm in enumerate(("raw", "tracer_whole")):
        expr = data[arm][1]
        expr.obs["annotation"] = pd.Categorical(
            expr.obs["annotation"].astype(str),
            categories=[g for g in groups if (expr.obs["annotation"].astype(str) == g).any()])
        sc.pl.dotplot(expr, {k: present[k] for k in order}, groupby="annotation",
                      standard_scale="var", ax=axes[r], show=False, colorbar_title="scaled mean expr")
        axes[r].set_title(f"{ARM_TITLES[arm]}  (n={expr.n_obs})", fontsize=11, loc="left")
    fig.suptitle("Marker expression by marker-based annotation -- does annotation survive TRACER's shallower entities?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT / "step0_marker_dotplot.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main() -> None:
    data = {arm: prep(arm) for arm in ("raw", "tracer_whole")}
    umap_figure(data)
    dotplot_figure(data)

    lines = []
    for arm in ("raw", "tracer_whole"):
        adata = data[arm][0]
        obs = adata.obs
        lines.append(f"\n=== {arm} (n={adata.n_obs}) ===")
        ct = pd.crosstab(obs["annotation"].astype(str), obs["compartment"].astype(str))
        lines.append("annotation x compartment:\n" + ct.to_string())
        med = obs.groupby(obs["annotation"].astype(str))["n_genes"].agg(["size", "median"])
        lines.append("median n_genes per annotation:\n" + med.to_string())
        truth_epi = obs["truth_clone"].astype(str).isin(["0", "1", "2", "3", "4"])
        lines.append(f"truth-epithelial entities: {int(truth_epi.sum())}")
        lines.append("  where they land (compartment):\n" +
                     obs.loc[truth_epi, "compartment"].astype(str).value_counts().to_string())
        lines.append("  their median n_genes by compartment:\n" +
                     obs.loc[truth_epi].groupby(obs.loc[truth_epi, "compartment"].astype(str))["n_genes"]
                     .median().to_string())
        # depth dependence of the annotation failure
        q = pd.qcut(obs["n_genes"], 5, duplicates="drop")
        rate = obs.assign(q=q, epi=truth_epi).groupby("q", observed=True).apply(
            lambda d: pd.Series({
                "n": len(d),
                "n_truth_epi": int(d["epi"].sum()),
                "frac_truth_epi_called_tumor": float(
                    (d.loc[d["epi"], "compartment"].astype(str) == "tumor").mean())
                if d["epi"].any() else float("nan"),
            }), include_groups=False)
        lines.append("truth-epithelial recall into the tumour compartment, by depth quintile:\n" + rate.to_string())
    txt = "\n".join(lines)
    print(txt)
    (OUT / "step0_summary.txt").write_text(txt)


if __name__ == "__main__":
    main()
