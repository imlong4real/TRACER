#!/usr/bin/env python3
"""Panel E — quantitative benchmarking on QC-filtered profiles.

All metrics are computed on **QC-passing** cells/profiles:
  PRIMARY QC: ≥100 detected genes, ≥200 UMIs, ≥5 contributing 2×2 µm bins.
(The bin criterion is skipped for 10x segmented cells, not bin-derived.)

Metrics (matched 1,656 HVG/NPMI panel / whole-transcriptome):
  (E1) RCTD-style Poisson-EM entropy (lower = purer) — half-violins, all 4 methods.
  (E2) RCTD-style Poisson-EM max weight (higher = purer) — half-violins.
  (E3) NPMI relative purity / conflict — stacked bar (purity + conflict = 1).
  (E4) per-lineage **Kendall τ** of each method's lineage pseudobulk to the scRNA
       reference pseudobulk — magma heatmap (main). Pearson & Spearman heatmaps
       are saved separately as `panel_E_supp_concordance.{png,svg}`.

RCTD-style metrics come from a Python Poisson-EM deconvolution (the exact
implementation in scripts/run_rctd_tracer_overlap.py), validated against real
spacexr/RCTD on 10x (see panel_E_supp_poisson_vs_spacexr + validation table).

Message: TRACER reconstructed profiles are at least as cell-type pure /
reference-consistent as competing approaches, robustly across QC thresholds.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, kendalltau

import fig4_config as C
import utils as U

QC_PRIMARY = {"min_genes": 100, "min_umis": 200, "min_bins": 5}
QC_STRICT = {"min_genes": 200, "min_umis": 500, "min_bins": 10}
POISSON_DIR = C.RCTD.parent / "rctd_poisson_em"


def _panel_genes():
    import anndata as ad
    return list(ad.read_h5ad(
        C.RES / "tracer_noseg/kidney_visiumhd_2um/outputs/profile_by_gene.h5ad",
        backed="r").var_names)


def _load_poisson(method):
    """Poisson-EM RCTD-style per-profile metrics (entropy, max_weight, dominant)."""
    p = POISSON_DIR / f"{method}_poisson_em_scores.tsv.gz"
    if not p.exists():
        return None
    d = pd.read_csv(p, sep="\t", usecols=["cell_id", "RCTD_entropy", "RCTD_max_weight",
                                          "predicted_dominant_lineage"])
    d["cell_id"] = d["cell_id"].astype(str)
    return d.rename(columns={"RCTD_entropy": "entropy", "RCTD_max_weight": "max_weight",
                             "predicted_dominant_lineage": "dominant"})


def _pseudobulk_for_masks(method, genes, ref_index, masks: dict) -> dict:
    import anndata as ad
    a = ad.read_h5ad(C.WT_H5AD[method])
    ids = pd.Index(a.obs_names.astype(str))
    lab = U._wt_labels(method).reindex(a.obs_names).to_numpy()
    gkeep = [g for g in genes if g in set(a.var_names) and g in set(ref_index)]
    sub = sp.csr_matrix(a[:, gkeep].X).astype(np.float64)
    tot = np.asarray(a.X.sum(1)).ravel().astype(np.float64); tot[tot == 0] = 1.0
    Xn = sub.multiply(1.0 / tot[:, None]).tocsr() * 1e4
    Xn.data = np.log1p(Xn.data)
    out = {}
    for name, keep in masks.items():
        m = np.asarray(ids.isin(keep)) & pd.notna(lab)
        df = pd.DataFrame(Xn[m].toarray(), columns=gkeep)
        df["lineage"] = lab[m]
        out[name] = df.groupby("lineage").mean().T          # genes x lineage
    return out


def _concordance(pb_by_method, ref) -> pd.DataFrame:
    """Per method × lineage Pearson / Spearman / Kendall to reference pseudobulk."""
    rows = []
    for m in C.METHOD_ORDER:
        expr = pb_by_method[m]
        shared = [g for g in ref.index if g in expr.index]
        for lin in C.LINEAGES:
            r = {"method": m, "lineage": lin, "pearson": np.nan,
                 "spearman": np.nan, "kendall": np.nan}
            if lin in expr.columns and lin in ref.columns:
                a = expr.loc[shared, lin].to_numpy(); b = ref.loc[shared, lin].to_numpy()
                if a.std() > 0 and b.std() > 0:
                    r["pearson"] = float(np.corrcoef(a, b)[0, 1])
                    r["spearman"] = float(spearmanr(a, b).correlation)
                    r["kendall"] = float(kendalltau(a, b).correlation)
            rows.append(r)
    return pd.DataFrame(rows)


def _pivot(conc, metric):
    return conc.pivot(index="lineage", columns="method", values=metric) \
        .reindex(C.LINEAGES)[C.METHOD_ORDER]


def _load_pc(method):
    p = C.SRCDIR / f"panel_E_purity_conflict_percell_{method}.csv.gz"
    if not p.exists():
        return None
    d = pd.read_csv(p); d["cell_id"] = d["cell_id"].astype(str)
    return d


def _halfviolin(ax, data, positions, colors, width=0.7):
    parts = ax.violinplot(data, positions=positions, showextrema=False, widths=width)
    for b, pos, col in zip(parts["bodies"], positions, colors):
        v = b.get_paths()[0].vertices
        v[:, 0] = np.clip(v[:, 0], pos, np.inf)
        b.set_facecolor(col); b.set_alpha(0.78); b.set_edgecolor("k"); b.set_linewidth(0.4)
    for d, pos in zip(data, positions):
        if len(d):
            ax.plot([pos - 0.16, pos], [np.nanmedian(d)] * 2, color="k", lw=1.2, zorder=6)


def _filt(d, keep, col):
    return d[d["cell_id"].isin(keep)][col].dropna().to_numpy()


def make():
    plt = U.setup_style()
    genes = _panel_genes()
    ref = U.reference_pseudobulk()
    ref = ref.reindex([g for g in genes if g in ref.index]).dropna()

    qc_ids = {lvl: {m: U.qc_pass_ids(m, **thr) for m in C.METHOD_ORDER}
              for lvl, thr in [("primary", QC_PRIMARY), ("strict", QC_STRICT)]}
    totals = {m: len(U.qc_table(m)) for m in C.METHOD_ORDER}

    ret = pd.DataFrame([{
        "method": C.METHOD_DISPLAY[m], "n_total": totals[m],
        "n_pass_primary": len(qc_ids["primary"][m]),
        "retention_primary": round(len(qc_ids["primary"][m]) / totals[m], 4),
        "n_pass_strict": len(qc_ids["strict"][m]),
        "retention_strict": round(len(qc_ids["strict"][m]) / totals[m], 4),
    } for m in C.METHOD_ORDER])
    ret.to_csv(C.SRCDIR / "panel_E_qc_retention.csv", index=False)

    # pseudobulk + concordance (primary + strict)
    pb = {"primary": {}, "strict": {}}
    for m in C.METHOD_ORDER:
        res = _pseudobulk_for_masks(m, genes, ref.index,
                                    {"primary": qc_ids["primary"][m], "strict": qc_ids["strict"][m]})
        pb["primary"][m] = res["primary"]; pb["strict"][m] = res["strict"]
    conc = {lvl: _concordance(pb[lvl], ref) for lvl in ("primary", "strict")}
    conc["primary"].to_csv(C.SRCDIR / "panel_E_concordance_to_reference.csv", index=False)
    kmat = _pivot(conc["primary"], "kendall")

    # RCTD-style Poisson-EM metrics (all methods), filtered to QC
    pem = {m: _load_poisson(m) for m in C.METHOD_ORDER}
    have = [m for m in C.METHOD_ORDER if pem[m] is not None and len(pem[m])]
    pc_raw = {m: _load_pc(m) for m in C.METHOD_ORDER}

    def _pc_qc(m, level):
        d = pc_raw[m]
        if d is None:
            return None
        return d[(d["signal_strength"] > 0) & (d["cell_id"].isin(qc_ids[level][m]))]

    # sensitivity table (primary vs strict)
    sens_rows = []
    for level in ("primary", "strict"):
        cm = conc[level]
        for m in C.METHOD_ORDER:
            r = pem[m]; p = _pc_qc(m, level)
            ent = _filt(r, qc_ids[level][m], "entropy") if r is not None else np.array([])
            mw = _filt(r, qc_ids[level][m], "max_weight") if r is not None else np.array([])
            kk = cm[cm["method"] == m]["kendall"]
            sens_rows.append({
                "qc": level, "method": C.METHOD_DISPLAY[m],
                "n_pass": len(qc_ids[level][m]),
                "retention": round(len(qc_ids[level][m]) / totals[m], 4),
                "poissonEM_median_entropy": round(float(np.median(ent)), 4) if len(ent) else np.nan,
                "poissonEM_median_max_weight": round(float(np.median(mw)), 4) if len(mw) else np.nan,
                "mean_relative_purity": round(float(p["relative_purity"].mean()), 4) if p is not None and len(p) else np.nan,
                "mean_relative_conflict": round(float(p["relative_conflict"].mean()), 4) if p is not None and len(p) else np.nan,
                "mean_kendall": round(float(kk.mean()), 4),
            })
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(C.SRCDIR / "panel_E_sensitivity.csv", index=False)

    pc_means = []
    for m in C.METHOD_ORDER:
        p = _pc_qc(m, "primary")
        pc_means.append({"method": C.METHOD_DISPLAY[m],
                         "mean_relative_purity": float(p["relative_purity"].mean()) if p is not None and len(p) else np.nan,
                         "mean_relative_conflict": float(p["relative_conflict"].mean()) if p is not None and len(p) else np.nan})
    pcm = pd.DataFrame(pc_means).set_index("method")
    pcm.round(4).to_csv(C.SRCDIR / "panel_E_purity_conflict.csv")

    if have:
        pd.DataFrame([{
            "method": C.METHOD_DISPLAY[m], "n_qc_cells": int((pem[m]["cell_id"].isin(qc_ids["primary"][m])).sum()),
            "median_entropy": round(float(np.median(_filt(pem[m], qc_ids["primary"][m], "entropy"))), 4),
            "median_max_weight": round(float(np.median(_filt(pem[m], qc_ids["primary"][m], "max_weight"))), 4),
        } for m in have]).to_csv(C.SRCDIR / "panel_E_rctd_summary.csv", index=False)

    val = None
    vpath = POISSON_DIR / "poisson_em_summary.json"
    if vpath.exists():
        val = json.loads(vpath.read_text()).get("validation_10x_vs_spacexr")

    _render_main(plt, have, pem, qc_ids, pcm, kmat, ret, val)
    _render_sensitivity(plt, sens)
    _render_supp_concordance(plt, conc["primary"])
    print(ret.to_string(index=False))
    print("Kendall τ:\n", kmat.round(3).to_string())
    if val:
        print("validation 10x Poisson-EM vs spacexr:", val)


def _render_main(plt, have, pem, qc_ids, pcm, kmat, ret, val):
    fig = plt.figure(figsize=(17, 4.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.8, 1.0, 1.0, 0.95, 1.3], wspace=0.42)
    ax0, ax1, ax2, axp, ax3 = [fig.add_subplot(gs[0, i]) for i in range(5)]

    xs = np.arange(len(C.METHOD_ORDER))
    rp = ret["retention_primary"].to_numpy() * 100
    rs = ret["retention_strict"].to_numpy() * 100
    ax0.bar(xs - 0.2, rp, width=0.38, color="#4C72B0", label="primary")
    ax0.bar(xs + 0.2, rs, width=0.38, color="#A0AEC8", label="strict")
    ax0.set_xticks(xs); ax0.set_xticklabels([C.METHOD_DISPLAY[m] for m in C.METHOD_ORDER],
                                            rotation=40, ha="right", fontsize=6.5)
    ax0.set_ylabel("% profiles passing QC"); ax0.set_title("QC retention", fontsize=8.5)
    ax0.legend(fontsize=5.5, frameon=False); ax0.set_ylim(0, 105)
    for i, v in enumerate(rp):
        ax0.text(i - 0.2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=5)

    if have:
        pos = list(range(len(have))); cols = [C.METHOD_COLOR[m] for m in have]
        ent = [_filt(pem[m], qc_ids["primary"][m], "entropy") for m in have]
        mw = [_filt(pem[m], qc_ids["primary"][m], "max_weight") for m in have]
        _halfviolin(ax1, ent, pos, cols); _halfviolin(ax2, mw, pos, cols)
        for ax, lab in [(ax1, "RCTD-style entropy (lower=purer)"),
                        (ax2, "RCTD-style max weight (higher=purer)")]:
            ax.set_xticks(pos); ax.set_xticklabels([C.METHOD_DISPLAY[m] for m in have],
                                                   rotation=40, ha="right", fontsize=7)
            ax.set_title(lab, fontsize=8.2)
        ax1.set_ylabel("Shannon entropy"); ax2.set_ylabel("max weight")
        note = "Poisson-EM deconvolution"
        if val:
            note += (f"\nvs spacexr (10x): r$_H$={val['entropy_pearson']:.2f}, "
                     f"r$_{{maxw}}$={val['max_weight_pearson']:.2f}, "
                     f"dom={val['dominant_lineage_agreement']:.0%}")
        ax1.text(0.02, -0.46, note, transform=ax1.transAxes, ha="left", va="top",
                 fontsize=5.6, style="italic", color="#666")

    xs = np.arange(len(pcm))
    pur = pcm["mean_relative_purity"].to_numpy(); con = pcm["mean_relative_conflict"].to_numpy()
    axp.bar(xs, pur, color="#2CA089", width=0.7, label="rel. purity")
    axp.bar(xs, con, bottom=pur, color="#C0392B", width=0.7, label="rel. conflict")
    for i, (p, c) in enumerate(zip(pur, con)):
        if p == p:
            axp.text(i, p / 2, f"{p:.2f}", ha="center", va="center", fontsize=6.3, color="white")
            axp.text(i, p + c / 2, f"{c:.2f}", ha="center", va="center", fontsize=6.3, color="white")
    axp.set_xticks(xs); axp.set_xticklabels(pcm.index, rotation=40, ha="right", fontsize=7)
    axp.set_ylim(0, 1.0); axp.set_ylabel("fraction of NPMI signal")
    axp.set_title("NPMI relative\npurity / conflict", fontsize=8.2)
    axp.legend(fontsize=5.5, frameon=False, loc="lower center")

    km = kmat.to_numpy()
    im = ax3.imshow(km, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax3.set_xticks(range(len(C.METHOD_ORDER)))
    ax3.set_xticklabels([C.METHOD_DISPLAY[m] for m in C.METHOD_ORDER], rotation=40, ha="right", fontsize=7)
    ax3.set_yticks(range(len(C.LINEAGES)))
    ax3.set_yticklabels([C.LINEAGE_DISPLAY[l] for l in C.LINEAGES], fontsize=7)
    ax3.set_title("Per-lineage Kendall τ to\nscRNA pseudobulk", fontsize=8.2)
    for i in range(len(C.LINEAGES)):
        for j in range(len(C.METHOD_ORDER)):
            v = km[i, j]
            if not np.isnan(v):
                ax3.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                         color="white" if v < 0.55 else "black")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="Kendall τ")

    fig.suptitle("Quantitative benchmark on QC-filtered profiles (≥100 genes, ≥200 UMIs, ≥5 bins; "
                 "1,656-gene panel; RCTD-style = Poisson-EM deconvolution)", fontsize=9.5)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.84, bottom=0.26)
    U.save_fig(fig, "panel_E_quantitative_benchmark")


def _render_sensitivity(plt, sens):
    metrics = [("poissonEM_median_entropy", "RCTD-style entropy (↓)"),
               ("poissonEM_median_max_weight", "RCTD-style max weight (↑)"),
               ("mean_relative_purity", "NPMI rel. purity (↑)"),
               ("mean_kendall", "mean Kendall τ (↑)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.1))
    xs = np.arange(len(C.METHOD_ORDER))
    disp = [C.METHOD_DISPLAY[m] for m in C.METHOD_ORDER]
    for ax, (col, lab) in zip(axes, metrics):
        prim = sens[sens["qc"] == "primary"].set_index("method").reindex(disp)[col].to_numpy()
        strict = sens[sens["qc"] == "strict"].set_index("method").reindex(disp)[col].to_numpy()
        ax.bar(xs - 0.2, prim, width=0.38, color="#4C72B0", label="primary")
        ax.bar(xs + 0.2, strict, width=0.38, color="#DD8452", label="strict")
        ax.set_xticks(xs); ax.set_xticklabels(disp, rotation=40, ha="right", fontsize=6.5)
        ax.set_title(lab, fontsize=8.5); ax.margins(y=0.2)
        for i, (a, b) in enumerate(zip(prim, strict)):
            if a == a: ax.text(i - 0.2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=5)
            if b == b: ax.text(i + 0.2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=5)
    axes[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Panel E sensitivity: metrics under primary (≥100/≥200/≥5) vs strict (≥200/≥500/≥10) QC",
                 fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.26, wspace=0.3)
    U.save_fig(fig, "panel_E_sensitivity")


def _render_supp_concordance(plt, conc):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, metric in zip(axes, ["pearson", "spearman"]):
        mat = _pivot(conc, metric).to_numpy()
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(C.METHOD_ORDER)))
        ax.set_xticklabels([C.METHOD_DISPLAY[m] for m in C.METHOD_ORDER], rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(C.LINEAGES)))
        ax.set_yticklabels([C.LINEAGE_DISPLAY[l] for l in C.LINEAGES], fontsize=7)
        ax.set_title(f"{metric.capitalize()} to scRNA pseudobulk", fontsize=9)
        for i in range(len(C.LINEAGES)):
            for j in range(len(C.METHOD_ORDER)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v < 0.55 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Panel E supplement — per-lineage Pearson & Spearman concordance to scRNA reference",
                 fontsize=9)
    fig.tight_layout()
    U.save_fig(fig, "panel_E_supp_concordance")


if __name__ == "__main__":
    make()
