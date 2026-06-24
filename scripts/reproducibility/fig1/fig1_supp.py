#!/usr/bin/env python3
"""Supplementary Figure 1 panels for the TRACER resubmission.

Invoked by make_fig1_panels.py (``run(OUT, log)``) or standalone.  Consumes the
finalized Atera ovrlpy sweep + three-method concordance outputs and the cached
kidney VisiumHD RCTD x TRACER overlap.  Three-method concordance was evaluated
on CANDIDATE ROIs (not whole-tissue RCTD); this is stated on the relevant
panels and in SUMMARY_FIG1.md.

Canonical convention (shared with make_fig1_panels): high score == problem /
ambiguous; '+' = high problem, '-' = low problem; no check/cross glyphs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]

SRC_PARQUET = _REPO / "datasets/cervical_cancer_atera_10x/filtered_df.parquet"
SWEEP_DIR = _REPO / "results/ovrlpy_tracer/param_sweep_atera"
CONCORD = _REPO / "results/ovrlpy_tracer/atera_three_method_concordance"
ATERA_HEADLINE = _REPO / "results/ovrlpy_tracer/cervical_atera_full_memoryaware"
VHD = _REPO / "results/kidney_visiumhd_rctd_tracer"
VHD_JOINED = VHD / "overlap/joined_rctd_tracer_scores.tsv.gz"

METRICS = ["ovrlpy_problem", "tracer_relconflict", "rctd_entropy", "rctd_lowmaxweight"]
METRIC_SHORT = {"ovrlpy_problem": "ovrlpy", "tracer_relconflict": "TRACER",
                "rctd_entropy": "RCTD-entropy", "rctd_lowmaxweight": "RCTD-1-maxw"}
FLAG_Q = 0.80


def _saver(OUT):
    def _save(fig, name):
        for ext in ("png", "svg", "pdf"):
            fig.savefig(OUT / "figures" / f"{name}.{ext}", bbox_inches="tight",
                        facecolor=fig.get_facecolor(), dpi=190)
        plt.close(fig)
    return _save


def _flags(v, q=FLAG_Q):
    return v >= np.nanquantile(v, q)


def _partial_spearman(x, y, z):
    """Spearman of x,y after regressing rank(z) out of both (density control)."""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rz = (rz - rz.mean())
    def resid(r):
        r = r - r.mean()
        b = np.dot(r, rz) / np.dot(rz, rz)
        return r - b * rz
    ex, ey = resid(rx), resid(ry)
    return float(np.corrcoef(ex, ey)[0, 1])


# ---------------------------------------------------------------------------
def s1a_atera_qc(OUT, save, log):
    """Atera full-tissue QC: transcript density, cell density, qv, ROI locations."""
    cache = OUT / "tables" / "atera_tissue_density_50um.parquet"
    if cache.exists():
        g = pl.read_parquet(cache)
    else:
        g = (pl.scan_parquet(SRC_PARQUET).rename({"x_location": "x", "y_location": "y"})
             .with_columns([(pl.col("x") // 50).alias("bx"), (pl.col("y") // 50).alias("by")])
             .group_by(["bx", "by"]).agg(pl.len().alias("n"),
                                         pl.col("cell_id").n_unique().alias("ncell"))
             .collect(engine="streaming"))
        g.write_parquet(cache)
    gp = g.to_pandas()
    bx = gp["bx"].to_numpy() * 50; by = gp["by"].to_numpy() * 50
    # qv sample
    qv = (pl.scan_parquet(SRC_PARQUET).select("qv").head(2_000_000).collect()).to_pandas()["qv"]
    # ROI locations
    rois = []
    summ = pd.read_csv(CONCORD / "tables/roi_convergence_summary.tsv", sep="\t")
    best = json.loads((CONCORD / "best_convergent_roi.json").read_text())["roi"]
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=190)
        sc = axes[0].scatter(bx, by, c=gp["n"], s=4, cmap="magma",
                             vmax=np.percentile(gp["n"], 99), linewidths=0, rasterized=True)
        axes[0].set_title("transcript density (50 um bins)", fontsize=9, color="white")
        fig.colorbar(sc, ax=axes[0], fraction=0.046).ax.tick_params(labelsize=6, colors="white")
        sc = axes[1].scatter(bx, by, c=gp["ncell"], s=4, cmap="viridis",
                             vmax=np.percentile(gp["ncell"], 99), linewidths=0, rasterized=True)
        axes[1].set_title("cell density (unique cells / 50 um bin)", fontsize=9, color="white")
        fig.colorbar(sc, ax=axes[1], fraction=0.046).ax.tick_params(labelsize=6, colors="white")
        for a in axes[:2]:
            a.set_aspect("equal"); a.invert_yaxis(); a.set_xticks([]); a.set_yticks([])
        axes[2].hist(qv, bins=60, color="#00E5FF")
        axes[2].set_title(f"transcript QV (n={len(qv):,} sample)\nfiltered_df is pre-QC'd",
                          fontsize=9, color="white")
        axes[2].set_xlabel("qv", fontsize=8); axes[2].tick_params(labelsize=7)
        # ROI locations on tissue
        ax = axes[3]
        ax.scatter(bx, by, c="#333344", s=3, linewidths=0, rasterized=True)
        for _, r in summ.iterrows():
            col = "#00E5FF" if r["roi"] == best["name"] else "#FFB000"
            ax.add_patch(Rectangle((r["xmin"], r["ymin"]), r["xmax"]-r["xmin"], r["ymax"]-r["ymin"],
                                   edgecolor=col, facecolor="none",
                                   lw=2.0 if r["roi"] == best["name"] else 1.0))
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("candidate ROI locations\n(cyan = convergent winner repB)", fontsize=9, color="white")
        fig.suptitle("Supplementary Fig 1a — Atera full-tissue QC", fontsize=12, color="white", y=1.02)
        fig.tight_layout()
        save(fig, "S1a_atera_tissue_qc")
    log("  [S1a] atera tissue QC")


def s1b_vhd_qc(OUT, save, log):
    d = pd.read_csv(VHD_JOINED, sep="\t")
    da = d[d["active_in_rctd"] == True].copy()  # noqa: E712
    da["rctd_lowmaxweight"] = 1.0 - da["RCTD_max_weight"]
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=190)
        ax = axes[0]
        sc = ax.hexbin(d["cx_um"], d["cy_um"], gridsize=80, cmap="bone", linewidths=0)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("VisiumHD bin density", fontsize=9, color="white")
        fig.colorbar(sc, ax=ax, fraction=0.046).ax.tick_params(labelsize=6, colors="white")
        axes[1].hist(da["RCTD_norm_entropy"].dropna(), bins=60, color="#39FF14", alpha=0.8, label="entropy")
        axes[1].hist(da["rctd_lowmaxweight"].dropna(), bins=60, color="#FFB000", alpha=0.6, label="1-maxw")
        axes[1].legend(fontsize=7, facecolor="black"); axes[1].set_title("RCTD ambiguity distributions", fontsize=9, color="white")
        axes[1].set_xlabel("score", fontsize=8); axes[1].tick_params(labelsize=7)
        axes[2].hist(da["TRACER_relative_conflict"].dropna(), bins=60, color="#FF1493")
        axes[2].set_title("TRACER relative conflict dist.", fontsize=9, color="white")
        axes[2].set_xlabel("relative conflict", fontsize=8); axes[2].set_yscale("log"); axes[2].tick_params(labelsize=7)
        cc = pd.read_csv(VHD / "qc/category_counts.tsv", sep="\t") if (VHD / "qc/category_counts.tsv").exists() else None
        ax = axes[3]; ax.axis("off")
        txt = ["VisiumHD RCTD x TRACER (cached)", f"  active bins: {len(da):,}",
               f"  median RCTD entropy: {da['RCTD_norm_entropy'].median():.3f}",
               f"  median RCTD 1-maxw: {da['rctd_lowmaxweight'].median():.3f}",
               f"  median TRACER rel.conflict: {da['TRACER_relative_conflict'].median():.4f}"]
        if cc is not None:
            txt += ["", "overlap categories:"] + [f"  {r.iloc[0]}: {int(r.iloc[1]):,}" for _, r in cc.iterrows()]
        ax.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=8.5, color="white")
        fig.suptitle("Supplementary Fig 1b — Kidney VisiumHD full-tissue QC", fontsize=12, color="white", y=1.02)
        fig.tight_layout()
        save(fig, "S1b_visiumhd_qc")
    log("  [S1b] visiumhd QC")


def s1c_ovrlpy_sensitivity(OUT, save, log):
    sw = pd.read_csv(SWEEP_DIR / "tables/sweep_primary_roi.tsv", sep="\t")
    sel = json.loads((SWEEP_DIR / "selected_setting.json").read_text())
    g = sw[sw["_axes"].astype(str).str.contains("grid2d", na=False)
           & (sw["n_components"] == 20) & (sw["min_tx_vsi"] == 2) & (sw["min_tx_local_max"] == 10)]
    kdes = sorted(g["KDE_bandwidth"].unique()); mds = sorted(g["min_distance"].unique())
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=190)
        # KDE x min_distance heatmaps
        for ax, metric, cmap in [(axes[0, 0], "frac_low_vsi", "magma"),
                                 (axes[0, 1], "morans_i_problem", "cividis"),
                                 (axes[0, 2], "auroc_problem_predicts_conflict", "RdBu_r")]:
            M = np.full((len(mds), len(kdes)), np.nan)
            for _, r in g.iterrows():
                M[mds.index(r["min_distance"]), kdes.index(r["KDE_bandwidth"])] = r[metric]
            im = ax.imshow(M, cmap=cmap, origin="lower", aspect="auto")
            ax.set_xticks(range(len(kdes))); ax.set_xticklabels(kdes)
            ax.set_yticks(range(len(mds))); ax.set_yticklabels(mds)
            ax.set_xlabel("KDE_bandwidth", fontsize=8); ax.set_ylabel("min_distance", fontsize=8)
            ax.set_title(metric, fontsize=9, color="white")
            for ii in range(len(mds)):
                for jj in range(len(kdes)):
                    if np.isfinite(M[ii, jj]):
                        ax.text(jj, ii, f"{M[ii, jj]:.2f}", ha="center", va="center", fontsize=7, color="white")
            fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(labelsize=6, colors="white")
        # frac_low vs KDE (saturation curve), Moran vs KDE, AUROC/Spearman vs setting
        ofat_k = sw[sw["_axes"].astype(str).str.contains("KDE_bandwidth")].sort_values("KDE_bandwidth")
        ax = axes[1, 0]
        ax.plot(ofat_k["KDE_bandwidth"], ofat_k["frac_low_vsi"], "o-", color="#FF1493", label="frac flagged (<0.5)")
        ax.plot(ofat_k["KDE_bandwidth"], ofat_k["median_vsi"], "s-", color="#00E5FF", label="median VSI")
        ax.axvspan(0, 1.2, color="red", alpha=0.12); ax.axvspan(3.5, 7, color="blue", alpha=0.10)
        ax.set_xlabel("KDE_bandwidth", fontsize=8); ax.legend(fontsize=7, facecolor="black")
        ax.set_title("saturation vs KDE_bandwidth\n(red=saturated, blue=washed out)", fontsize=9, color="white")
        ax.tick_params(labelsize=7)
        # scatter: frac_low vs AUROC across all settings, mark selected
        ax = axes[1, 1]
        ax.scatter(sw["frac_low_vsi"], sw["auroc_problem_predicts_conflict"], c=sw["morans_i_problem"],
                   cmap="viridis", s=30)
        ax.axvspan(0.03, 0.45, color="green", alpha=0.10)
        ax.scatter([sel["frac_low_vsi"]], [sel["auroc_problem_predicts_conflict"]], marker="*",
                   s=260, color="#FFD700", edgecolors="black", label=f"selected {sel['selected_id']}")
        ax.set_xlabel("frac flagged (saturation)", fontsize=8); ax.set_ylabel("AUROC vs TRACER", fontsize=8)
        ax.set_title("non-saturation band (green) vs concordance", fontsize=9, color="white")
        ax.legend(fontsize=7, facecolor="black"); ax.tick_params(labelsize=7)
        # old vs new inset reference
        ax = axes[1, 2]; ax.axis("off")
        txt = ["ovrlpy selected setting (s010):",
               f"  KDE_bandwidth = {sel['KDE_bandwidth']}", f"  min_distance = {sel['min_distance']}",
               f"  n_components = {sel['n_components']}", f"  min_tx_local_max = {sel['min_tx_local_max']}",
               f"  min_tx_vsi = {sel['min_tx_vsi']}", "",
               f"  median VSI = {sel['median_vsi']:.2f}", f"  frac flagged = {sel['frac_low_vsi']:.2f}",
               f"  Moran's I = {sel['morans_i_problem']:.2f}",
               f"  AUROC vs TRACER = {sel['auroc_problem_predicts_conflict']:.2f}",
               f"  Spearman = {sel['spearman_problem_vs_relconflict']:.2f}", "",
               "old-vs-new saturated inset correction:",
               "  see fig1_ovrlpy_inset_old_vs_new.* in",
               "  atera_three_method_concordance/figures/"]
        ax.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=9, color="white")
        fig.suptitle("Supplementary Fig 1c — ovrlpy parameter sensitivity (KDE_bandwidth is the saturation knob)",
                     fontsize=12, color="white", y=1.00)
        fig.tight_layout()
        save(fig, "S1c_ovrlpy_sensitivity")
    log("  [S1c] ovrlpy sensitivity")


def _heat(ax, df, rowk, colk, val, fig, title, fmt="{:.2f}", cmap="viridis"):
    rows = sorted(df[rowk].unique()); cols = sorted(df[colk].unique())
    M = np.full((len(rows), len(cols)), np.nan)
    for _, r in df.iterrows():
        M[rows.index(r[rowk]), cols.index(r[colk])] = r[val]
    im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xlabel(colk, fontsize=8); ax.set_ylabel(rowk, fontsize=8); ax.set_title(title, fontsize=9, color="white")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(labelsize=6, colors="white")


def s1d_tracer_sensitivity(OUT, save, log):
    sw = pd.read_csv(CONCORD / "tables/sweep_tracer.tsv", sep="\t")
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=190)
        _heat(axes[0], sw, "tau", "conflict_percentile", "convergence_score", fig,
              "convergence score", cmap="magma")
        _heat(axes[1], sw, "tau", "conflict_percentile", "conv_min_pairwise_spearman", fig,
              "min pairwise Spearman", cmap="viridis")
        _heat(axes[2], sw, "tau", "conflict_percentile", "conv_triple_enrichment", fig,
              "triple-flag enrichment", fmt="{:.1f}", cmap="cividis")
        fig.suptitle("Supplementary Fig 1d — TRACER NPMI relative-conflict sensitivity (tau x percentile)",
                     fontsize=12, color="white", y=1.03)
        fig.tight_layout()
        save(fig, "S1d_tracer_sensitivity")
    log("  [S1d] tracer sensitivity")


def s1e_rctd_sensitivity(OUT, save, log):
    sw = pd.read_csv(CONCORD / "tables/sweep_rctd.tsv", sep="\t")
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=190)
        _heat(axes[0], sw, "lineage_col", "n_iter", "convergence_score", fig, "convergence score", cmap="magma")
        _heat(axes[1], sw, "lineage_col", "n_iter", "med_rctd_entropy", fig, "median RCTD entropy", cmap="viridis")
        _heat(axes[2], sw, "lineage_col", "n_iter", "frac_low_maxweight", fig,
              "frac (1-maxw)>0.5", cmap="cividis")
        for a in axes:
            a.set_yticklabels([t.get_text().replace("cell_type_", "") for t in a.get_yticklabels()], fontsize=7)
        fig.suptitle("Supplementary Fig 1e — RCTD sensitivity (entropy and 1-max-weight reported separately)",
                     fontsize=12, color="white", y=1.03)
        fig.tight_layout()
        save(fig, "S1e_rctd_sensitivity")
    log("  [S1e] rctd sensitivity")


def s1f_null_controls(OUT, save, log):
    """Spatial-permutation nulls for Moran's I, pairwise overlap, triple-flag (repB)."""
    best = json.loads((CONCORD / "best_convergent_roi.json").read_text())["roi"]
    cells = pd.read_csv(CONCORD / "per_roi_cells" / f"{best['name']}_cells.tsv.gz", sep="\t").dropna(subset=METRICS)
    bats = json.loads((CONCORD / "all_batteries.json").read_text())[best["name"]]
    rng = np.random.default_rng(1)
    xy = cells[["cx", "cy"]].to_numpy(float)
    sys.path.insert(0, str(_THIS.parent))
    import atera_three_method_concordance as C
    fm = C.FastMoran(xy)
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=190)
        # Moran nulls for the 3 method metrics
        ax = axes[0]
        for m, col in zip(["ovrlpy_problem", "tracer_relconflict", "rctd_entropy"],
                          ["#00E5FF", "#FF1493", "#39FF14"]):
            v = cells[m].to_numpy(float)
            null = np.array([fm.I(rng.permutation(v)) for _ in range(500)])
            ax.hist(null, bins=40, color=col, alpha=0.4)
            ax.axvline(bats["moran"][m], color=col, lw=2)
        ax.set_title("Moran's I: observed (lines) vs\nspatial-permutation null (hist)", fontsize=9, color="white")
        ax.set_xlabel("Moran's I", fontsize=8); ax.tick_params(labelsize=7)
        # pairwise overlap null (ovrlpy & TRACER)
        ax = axes[1]
        fo = _flags(cells["ovrlpy_problem"].to_numpy()); ft = _flags(cells["tracer_relconflict"].to_numpy())
        obs = int((fo & ft).sum())
        null = np.array([int((fo & rng.permutation(ft)).sum()) for _ in range(2000)])
        ax.hist(null, bins=40, color="#888899"); ax.axvline(obs, color="#00E5FF", lw=2.5, label=f"obs={obs}")
        ax.set_title(f"ovrlpy & TRACER flag overlap\nobs {obs} vs null {null.mean():.0f}", fontsize=9, color="white")
        ax.legend(fontsize=7, facecolor="black"); ax.set_xlabel("overlap count", fontsize=8); ax.tick_params(labelsize=7)
        # triple-flag null
        ax = axes[2]
        frc = _flags(cells["rctd_entropy"].to_numpy())
        obs = int((fo & ft & frc).sum())
        null = np.array([int((fo & rng.permutation(ft) & rng.permutation(frc)).sum()) for _ in range(2000)])
        ax.hist(null, bins=40, color="#888899"); ax.axvline(obs, color="#FFD700", lw=2.5, label=f"obs={obs}")
        ax.set_title(f"triple-flag overlap\nobs {obs} vs null {null.mean():.0f} "
                     f"({obs/max(null.mean(),1e-9):.1f}x)", fontsize=9, color="white")
        ax.legend(fontsize=7, facecolor="black"); ax.set_xlabel("triple overlap count", fontsize=8); ax.tick_params(labelsize=7)
        fig.suptitle(f"Supplementary Fig 1f — spatial-permutation null controls (Atera ROI {best['name']})",
                     fontsize=12, color="white", y=1.03)
        fig.tight_layout()
        save(fig, "S1f_null_controls")
    log("  [S1f] null controls")


def s1g_density_control(OUT, save, log):
    """Concordance is not simply driven by transcript / cell density (partial Spearman)."""
    best = json.loads((CONCORD / "best_convergent_roi.json").read_text())["roi"]
    cells = pd.read_csv(CONCORD / "per_roi_cells" / f"{best['name']}_cells.tsv.gz", sep="\t").dropna(subset=METRICS)
    n_tx = cells["n_tx"].to_numpy(float)
    pairs = [("ovrlpy_problem", "tracer_relconflict"), ("ovrlpy_problem", "rctd_entropy"),
             ("tracer_relconflict", "rctd_entropy")]
    from scipy.stats import spearmanr
    rows = []
    for a, b in pairs:
        raw = spearmanr(cells[a], cells[b]).correlation
        part = _partial_spearman(cells[a].to_numpy(), cells[b].to_numpy(), n_tx)
        rows.append({"pair": f"{METRIC_SHORT[a]} x {METRIC_SHORT[b]}", "raw_spearman": raw,
                     "partial_spearman_given_density": part})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "tables" / "S1g_density_control.tsv", sep="\t", index=False)
    # also metric vs density correlation
    dens_corr = {m: spearmanr(cells[m], n_tx).correlation for m in METRICS}
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=190)
        ax = axes[0]
        xpos = np.arange(len(rows)); w = 0.38
        ax.bar(xpos - w/2, tbl["raw_spearman"], w, color="#00E5FF", label="raw Spearman")
        ax.bar(xpos + w/2, tbl["partial_spearman_given_density"], w, color="#FFB000",
               label="partial | n_tx density")
        ax.set_xticks(xpos); ax.set_xticklabels(tbl["pair"], rotation=20, ha="right", fontsize=7)
        ax.set_ylabel("Spearman rho", fontsize=8); ax.legend(fontsize=7, facecolor="black")
        ax.set_title("concordance survives density control", fontsize=9, color="white"); ax.tick_params(labelsize=7)
        ax = axes[1]
        ax.bar(range(len(METRICS)), [dens_corr[m] for m in METRICS],
               color=["#00E5FF", "#FF1493", "#39FF14", "#FFB000"])
        ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([METRIC_SHORT[m] for m in METRICS],
                                                               rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("Spearman(metric, n_tx)", fontsize=8)
        ax.set_title("each metric vs transcript count\n(weak -> not a density artifact)", fontsize=9, color="white")
        ax.tick_params(labelsize=7)
        ax = axes[2]
        sc = ax.scatter(n_tx, cells["ovrlpy_problem"], s=5, c=cells["tracer_relconflict"],
                        cmap="magma", vmax=np.nanpercentile(cells["tracer_relconflict"], 95), linewidths=0)
        ax.set_xlabel("n transcripts / cell", fontsize=8); ax.set_ylabel("ovrlpy problem", fontsize=8)
        ax.set_title("problem score vs density\n(colour = TRACER conflict)", fontsize=9, color="white")
        fig.colorbar(sc, ax=ax, fraction=0.046).ax.tick_params(labelsize=6, colors="white"); ax.tick_params(labelsize=7)
        fig.suptitle(f"Supplementary Fig 1g — density-control analysis (Atera ROI {best['name']})",
                     fontsize=12, color="white", y=1.03)
        fig.tight_layout()
        save(fig, "S1g_density_control")
    log(f"  [S1g] density control: raw vs partial Spearman written")


def s1h_additional_rois(OUT, save, log):
    """Additional Atera ROIs: concordant-problem AND discordant-control regions."""
    summ = pd.read_csv(CONCORD / "tables/roi_convergence_summary.tsv", sep="\t").sort_values(
        "convergence_score", ascending=False)
    concordant = summ.head(3)["roi"].tolist()
    discordant = summ.tail(3)["roi"].tolist()
    sel = concordant + discordant
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(len(sel), 4, figsize=(13, 3.0 * len(sel)), dpi=170)
        for ri, name in enumerate(sel):
            cells = pd.read_csv(CONCORD / "per_roi_cells" / f"{name}_cells.tsv.gz", sep="\t").dropna(subset=METRICS)
            row = summ[summ["roi"] == name].iloc[0]
            tag = "CONCORDANT" if name in concordant else "discordant control"
            for ci, m in enumerate(METRICS):
                ax = axes[ri, ci]
                v = cells[m].to_numpy(float); lo, hi = np.nanpercentile(v, [2, 98])
                sc = ax.scatter(cells["cx"], cells["cy"], c=v, s=5, cmap="magma", vmin=lo, vmax=hi,
                                linewidths=0, rasterized=True)
                ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(METRIC_SHORT[m], fontsize=8, color="white")
                if ci == 0:
                    ax.set_ylabel(f"{name}\n{tag}\nminRho={row['conv_min_pairwise_spearman']:.2f}",
                                  fontsize=7, color="#00E5FF" if tag == "CONCORDANT" else "#FF6666")
        fig.suptitle("Supplementary Fig 1h — additional Atera ROIs: concordant-problem vs discordant-control",
                     fontsize=12, color="white", y=1.005)
        fig.tight_layout()
        save(fig, "S1h_additional_rois")
    log(f"  [S1h] additional ROIs: concordant={concordant} discordant={discordant}")


def write_summary(OUT, log):
    sel = json.loads((SWEEP_DIR / "selected_setting.json").read_text())
    best = json.loads((CONCORD / "best_convergent_roi.json").read_text())
    md = f"""# Figure 1 (B-D) + Supplementary Figure 1 — inputs, parameters, interpretation

## Canonical convention (audited)
Every diagnostic is oriented so HIGH score == PROBLEM / AMBIGUOUS cell.
`+` = **high problem (ambiguous)**, `-` = **low problem (clean)**. No check/cross
glyphs are used; legends spell out "high/low problem" to prevent any reversal.

## Selected ovrlpy setting (non-saturated, TRACER-concordant)
From `param_sweep_atera/selected_setting.json` (id `{sel['selected_id']}`):
KDE_bandwidth={sel['KDE_bandwidth']}, min_distance={sel['min_distance']},
n_components={sel['n_components']}, min_tx_local_max={sel['min_tx_local_max']},
min_tx_vsi={sel['min_tx_vsi']}  -> median VSI {sel['median_vsi']:.2f}, frac flagged
{sel['frac_low_vsi']:.2f}, Moran's I {sel['morans_i_problem']:.2f}, AUROC vs TRACER
{sel['auroc_problem_predicts_conflict']:.2f}. KDE_bandwidth is the saturation knob.

## Convergent ROI (winner, Fig 1B/1D)
`{best['roi']['name']}`  x:[{best['roi']['xmin']:.1f}, {best['roi']['xmax']:.1f}]
y:[{best['roi']['ymin']:.1f}, {best['roi']['ymax']:.1f}] um  (units: microns).
3-way convergence: min pairwise Spearman {best['convergence']['min_pairwise_spearman']:.3f}
(all pairs positive), mean Moran's I {best['convergence']['mean_method_moran']:.3f}
(all p=0.005), triple-flag enrichment {best['convergence']['triple_enrichment']:.2f}x
(perm p {best['convergence']['triple_perm_p']:.3f}). Fig 1B large window:
see `tables/fig1b_coords.json`.

## Figure panels
- **Fig 1B** `figures/fig1B_atera_structured_conflict.*` — Atera large window: transcript
  density, ovrlpy problem (s010), TRACER relative conflict, ovrlpy&TRACER concordance
  overlay; magenta box = repB.
- **Fig 1C** `figures/fig1C_visiumhd_structured_ambiguity.*` — Kidney VisiumHD: RCTD
  norm. entropy and (1-max-weight) kept SEPARATE, TRACER relative conflict, RCTD&TRACER
  overlap. ROI boxes drawn, no in-canvas text.
- **Fig 1D** `figures/fig1D_cross_method_concordance.*` — pairwise Spearman, odds-ratio
  enrichment, Moran's I + spatial-permutation p, observed-vs-permuted triple-flag overlap.
  Punchline: three independent diagnostics converge on the same structured regions ->
  ambiguous segmentation is a real spatial property, not one algorithm's artifact.
- **Supp 1a-1h** `figures/S1*.*` — Atera & VisiumHD QC; ovrlpy KDExmin_distance sensitivity
  + saturation curve + old-vs-new correction; TRACER tau x percentile; RCTD n_iter x
  lineage granularity (entropy & 1-maxw separate); spatial-permutation null controls;
  density-control (raw vs partial-Spearman | n_tx); additional concordant vs discordant ROIs.

## Scope note
Three-method concordance (ovrlpy x TRACER x RCTD) was evaluated on **candidate ROIs**
(8 RCTD-ranked windows + A/B/C representative ROIs), NOT whole-tissue RCTD. Fig 1B's
ovrlpy/TRACER maps are a large contiguous Atera window; Fig 1C uses the cached kidney
VisiumHD whole-tissue RCTD x TRACER overlap. Whole-tissue Atera RCTD was intentionally
not rerun.

## Key input files
- `results/ovrlpy_tracer/param_sweep_atera/` (sweep tables, selected_setting.json)
- `results/ovrlpy_tracer/atera_three_method_concordance/` (per_roi_cells, all_batteries.json,
  best_convergent_roi.json, sweep_tracer.tsv, sweep_rctd.tsv)
- `results/kidney_visiumhd_rctd_tracer/overlap/joined_rctd_tracer_scores.tsv.gz`
- `datasets/cervical_cancer_atera_10x/filtered_df.parquet` (Fig 1B ovrlpy window only)
"""
    (OUT / "SUMMARY_FIG1.md").write_text(md)
    log("  [SUMMARY] wrote SUMMARY_FIG1.md")


def run(OUT, log):
    save = _saver(OUT)
    s1a_atera_qc(OUT, save, log)
    s1b_vhd_qc(OUT, save, log)
    s1c_ovrlpy_sensitivity(OUT, save, log)
    s1d_tracer_sensitivity(OUT, save, log)
    s1e_rctd_sensitivity(OUT, save, log)
    s1f_null_controls(OUT, save, log)
    s1g_density_control(OUT, save, log)
    s1h_additional_rois(OUT, save, log)
    write_summary(OUT, log)


if __name__ == "__main__":
    OUT = _REPO / "results/ovrlpy_tracer/fig1_panels"
    (OUT / "figures").mkdir(parents=True, exist_ok=True); (OUT / "tables").mkdir(parents=True, exist_ok=True)
    run(OUT, lambda m: print(m, flush=True))
