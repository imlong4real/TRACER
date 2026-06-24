#!/usr/bin/env python3
"""Fast RCTD-style metrics for Figure 4 Panel E via Python Poisson-EM.

Re-uses the *exact* Poisson-EM deconvolution from
``scripts/run_rctd_tracer_overlap.py`` (build_lineage_signature +
poisson_em_deconvolution + rctd_metrics) to score all four methods on the
shared 1,656 HVG/NPMI gene panel, with chunking for TRACER 2 µm (~260k
profiles). This is labelled "RCTD-style Poisson-EM deconvolution" (NOT
spacexr) unless the 10x sensitivity validation is strong.

For each method we save: per-profile lineage weights, RCTD-style entropy,
max weight, predicted dominant lineage, plus runtime and peak RSS.

Sensitivity: for 10x — where a real spacexr/RCTD run exists
(results/.../rctd/10x_segmented/rctd_cell_assignments_post.tsv) — we compare
Poisson-EM vs spacexr on entropy correlation, max-weight correlation and
dominant-lineage agreement, and save a validation table + scatter plot.

Usage:
    python scripts/reproducibility/fig4/prep/run_poisson_em_rctd.py
"""
from __future__ import annotations
import json
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fig4_config as C

# import the canonical Poisson-EM implementation
sys.path.insert(0, str(C.ROOT / "scripts"))
from run_rctd_tracer_overlap import (build_lineage_signature,
                                     poisson_em_deconvolution, rctd_metrics)

OUTDIR = C.RCTD.parent / "rctd_poisson_em"
OUTDIR.mkdir(parents=True, exist_ok=True)
EM_ITERS = 80
CHUNK = 20000
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger("poisson_em")


def _peak_rss_gb():
    # macOS ru_maxrss is bytes; Linux is kB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**3) if sys.platform == "darwin" else rss / (1024**2)


def _panel_genes():
    return np.asarray(ad.read_h5ad(
        C.RES / "tracer_noseg/kidney_visiumhd_2um/outputs/profile_by_gene.h5ad",
        backed="r").var_names, dtype=object)


def run_method(method, S, sig_genes, lineages):
    a = ad.read_h5ad(C.WT_H5AD[method])
    y = sp.csr_matrix(a.X).astype(np.float32)
    gene_names = np.asarray(a.var_names, dtype=object)
    t0 = time.time()
    W, counts, active = poisson_em_deconvolution(
        y, gene_names, S, sig_genes, n_iter=EM_ITERS, chunk_size=CHUNK, logger=LOG)
    runtime = time.time() - t0
    met = rctd_metrics(W, lineages, active)
    met.insert(0, "cell_id", np.asarray(a.obs_names, dtype=object))
    met["n_panel_umi"] = counts.astype(np.int32)
    # per-profile weights
    wdf = pd.DataFrame(W, columns=[f"w_{l}" for l in lineages])
    wdf.insert(0, "cell_id", np.asarray(a.obs_names, dtype=object))
    scores_p = OUTDIR / f"{method}_poisson_em_scores.tsv.gz"
    weights_p = OUTDIR / f"{method}_poisson_em_weights.tsv.gz"
    met.to_csv(scores_p, sep="\t", index=False)
    wdf.to_csv(weights_p, sep="\t", index=False)
    stat = {"method": method, "n_profiles": int(a.n_obs),
            "n_active": int(active.sum()), "runtime_s": round(runtime, 1),
            "peak_rss_gb": round(_peak_rss_gb(), 2),
            "median_entropy": float(np.nanmedian(met["RCTD_entropy"])),
            "median_max_weight": float(np.nanmedian(met["RCTD_max_weight"])),
            "scores": str(scores_p), "weights": str(weights_p)}
    LOG.info("[%s] %s", method, stat)
    return stat


def validate_against_spacexr():
    """Poisson-EM vs spacexr on 10x: entropy/max-weight corr + dominant agreement."""
    from scipy.stats import pearsonr, spearmanr
    pe = pd.read_csv(OUTDIR / "10x_segmented_poisson_em_scores.tsv.gz", sep="\t")
    sx_p = C.RCTD / "10x_segmented" / "rctd_cell_assignments_post.tsv"
    if not sx_p.exists():
        LOG.warning("spacexr 10x output missing; skipping validation"); return None
    sx = pd.read_csv(sx_p, sep="\t")
    sx["dominant_celltype"] = sx["dominant_celltype"].replace(C.RCTD_LABEL_FIX)
    m = pe.merge(sx, on="cell_id", how="inner", suffixes=("_pe", "_sx"))
    m = m.dropna(subset=["RCTD_entropy", "entropy", "RCTD_max_weight", "max_weight"])
    res = {
        "n_cells_compared": int(len(m)),
        "entropy_pearson": round(float(pearsonr(m["RCTD_entropy"], m["entropy"])[0]), 4),
        "entropy_spearman": round(float(spearmanr(m["RCTD_entropy"], m["entropy"])[0]), 4),
        "max_weight_pearson": round(float(pearsonr(m["RCTD_max_weight"], m["max_weight"])[0]), 4),
        "max_weight_spearman": round(float(spearmanr(m["RCTD_max_weight"], m["max_weight"])[0]), 4),
        "dominant_lineage_agreement": round(float(
            (m["predicted_dominant_lineage"] == m["dominant_celltype"]).mean()), 4),
    }
    m[["cell_id", "RCTD_entropy", "entropy", "RCTD_max_weight", "max_weight",
       "predicted_dominant_lineage", "dominant_celltype"]].to_csv(
        C.SRCDIR / "panel_E_poisson_vs_spacexr_percell.csv.gz", index=False)
    pd.DataFrame([res]).to_csv(C.SRCDIR / "panel_E_poisson_vs_spacexr_validation.csv", index=False)

    # scatter plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for axi, (xc, yc, lab, r) in zip(ax, [
        ("entropy", "RCTD_entropy", "entropy", res["entropy_pearson"]),
        ("max_weight", "RCTD_max_weight", "max weight", res["max_weight_pearson"])]):
        s = m.sample(min(8000, len(m)), random_state=1)
        axi.scatter(s[xc], s[yc], s=3, alpha=0.25, color="#2E6FB7", linewidths=0, rasterized=True)
        lo = min(s[xc].min(), s[yc].min()); hi = max(s[xc].max(), s[yc].max())
        axi.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        axi.set_xlabel(f"spacexr RCTD {lab}"); axi.set_ylabel(f"Poisson-EM {lab}")
        axi.set_title(f"{lab}: Pearson r={r:.3f}", fontsize=9)
    fig.suptitle(f"10x: Poisson-EM vs spacexr RCTD  (dominant-lineage agreement "
                 f"{res['dominant_lineage_agreement']:.1%}, n={res['n_cells_compared']:,})", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(C.OUTDIR / f"panel_E_supp_poisson_vs_spacexr.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOG.info("[validation] %s", res)
    return res


def main():
    hvgs = _panel_genes()
    ref = ad.read_h5ad(C.REFERENCE_H5AD)
    S, lineages, sig_genes = build_lineage_signature(ref, hvgs, LOG)
    del ref
    stats = [run_method(m, S, sig_genes, lineages) for m in C.METHOD_ORDER]
    val = validate_against_spacexr()
    summary = {"em_iters": EM_ITERS, "chunk_size": CHUNK,
               "n_signature_genes": int(len(sig_genes)),
               "lineages": list(map(str, lineages)),
               "methods": stats, "validation_10x_vs_spacexr": val}
    (OUTDIR / "poisson_em_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("wrote %s", OUTDIR / "poisson_em_summary.json")


if __name__ == "__main__":
    main()
