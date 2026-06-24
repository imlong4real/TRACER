#!/usr/bin/env python3
"""Generate TRACER resubmission Figure 1 (B-D) + Supplementary Figure 1.

Consumes the finalized Atera ovrlpy sweep and three-method concordance outputs
(plus the cached kidney VisiumHD RCTD x TRACER overlap) and renders dark
Nature-style panels.  Heavy whole-tissue RCTD is NOT rerun; Fig 1D uses the
cached per-ROI three-method results (candidate-ROI evaluation, labelled as such
in the supplement).  The only fresh compute is one ovrlpy `s010` fit on a large
Atera window for Fig 1B.

================  CANONICAL PROBLEM-POSITIVE CONVENTION  ====================
Every diagnostic is oriented so that a HIGH score == a PROBLEM / AMBIGUOUS cell:
  ovrlpy problem score   = 1 - VSI            (high = vertical signal overlap)
  TRACER relative conflict                    (high = co-expression conflict)
  RCTD normalised entropy                     (high = mixed cell-type posterior)
  RCTD (1 - max weight)                       (high = no dominant cell type)
"+"  ==  PROBLEM-POSITIVE  ==  "high problem (ambiguous)"
"-"  ==  PROBLEM-NEGATIVE  ==  "low problem (clean)"
No check/cross glyphs are used anywhere: legends spell out high/low problem to
remove any possible reversal.
============================================================================

Stages: --stage {b,c,d,supp,all}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_THIS.parent))

# ---------------------------------------------------------------------------
#  Canonical convention
# ---------------------------------------------------------------------------
PROBLEM_POS = "high problem (ambiguous)"
PROBLEM_NEG = "low problem (clean)"
FLAG_Q = 0.80
METRICS = ["ovrlpy_problem", "tracer_relconflict", "rctd_entropy", "rctd_lowmaxweight"]
METRIC_LABEL = {
    "ovrlpy_problem":     "ovrlpy problem (1 - VSI)",
    "tracer_relconflict": "TRACER relative conflict",
    "rctd_entropy":       "RCTD norm. entropy",
    "rctd_lowmaxweight":  "RCTD (1 - max weight)",
}
METRIC_SHORT = {"ovrlpy_problem": "ovrlpy", "tracer_relconflict": "TRACER",
                "rctd_entropy": "RCTD-entropy", "rctd_lowmaxweight": "RCTD-1-maxw"}
# colour for "both methods high problem" overlay
COL_BOTH = "#00E5FF"     # cyan  = both high problem (concordant)
COL_ONE = "#FFB000"      # amber = one method high problem
COL_BG = "#2a2a3a"       # dim   = low problem background
COL_ROI = "#FF1493"      # magenta ROI box

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
SRC_PARQUET = _REPO / "datasets/cervical_cancer_atera_10x/filtered_df.parquet"
SWEEP_DIR = _REPO / "results/ovrlpy_tracer/param_sweep_atera"
CONCORD = _REPO / "results/ovrlpy_tracer/atera_three_method_concordance"
ATERA_HEADLINE = _REPO / "results/ovrlpy_tracer/cervical_atera_full_memoryaware"
ATERA_COMPARE = ATERA_HEADLINE / "tables/ovrlpy_tracer_cell_level_comparison.tsv"
VHD = _REPO / "results/kidney_visiumhd_rctd_tracer"
VHD_JOINED = VHD / "overlap/joined_rctd_tracer_scores.tsv.gz"
VHD_ROIS = VHD / "overlap/representative_rois.json"
OUT = _REPO / "results/ovrlpy_tracer/fig1_panels"

SELECTED = json.loads((SWEEP_DIR / "selected_setting.json").read_text())
OVRLPY_PARAMS = dict(KDE_bandwidth=SELECTED["KDE_bandwidth"], min_distance=SELECTED["min_distance"],
                     n_components=int(SELECTED["n_components"]),
                     min_tx_local_max=SELECTED["min_tx_local_max"], min_tx_vsi=SELECTED["min_tx_vsi"])
BEST_ROI = json.loads((CONCORD / "best_convergent_roi.json").read_text())["roi"]


def _dirs():
    for d in ("figures", "tables"):
        (OUT / d).mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", bbox_inches="tight",
                    facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)


def _flags(v, q=FLAG_Q):
    return v >= np.nanquantile(v, q)


def _scatter(ax, x, y, c, *, vmin=None, vmax=None, cmap="magma", s=5, cb=True, fig=None, label=""):
    if vmin is None:
        vmin, vmax = np.nanpercentile(c, [2, 98])
    sc = ax.scatter(x, y, c=c, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
                    linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    if cb and fig is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6, colors="white")
        if label:
            cbar.set_label(label, fontsize=6, color="white")
    return sc


# ===========================================================================
#  FIG 1B  — Atera structured conflict map (ovrlpy s010 x TRACER)
# ===========================================================================
def _ovrlpy_largewindow(win: dict, log) -> pd.DataFrame:
    """Run ovrlpy s010 on a large Atera window; cache per-cell problem+centroids."""
    import ovrlpy_param_sweep_atera as ops
    cache = OUT / "tables" / "fig1b_ovrlpy_s010_largewindow_cells.tsv.gz"
    if cache.exists():
        log(f"  [1B] cached ovrlpy window cells: {cache.name}")
        return pd.read_csv(cache, sep="\t")
    panel = pd.read_csv(ATERA_HEADLINE / "selected_ovrlpy_gene_panel.tsv", sep="\t")["gene"].astype(str).tolist()
    t = time.time()
    df = (pl.scan_parquet(SRC_PARQUET)
          .rename({"feature_name": "gene", "x_location": "x", "y_location": "y", "z_location": "z"})
          .filter((pl.col("x") >= win["xmin"]) & (pl.col("x") <= win["xmax"])
                  & (pl.col("y") >= win["ymin"]) & (pl.col("y") <= win["ymax"]))
          .select(["x", "y", "z", "gene", "cell_id"]).collect(engine="streaming"))
    log(f"  [1B] window {df.height/1e6:.1f}M tx extracted ({time.time()-t:.0f}s); fitting ovrlpy s010...")
    fr = ops.run_ovrlpy(df, panel, n_workers=4, seed=42, **OVRLPY_PARAMS)
    # per-cell centroids in micron from transcripts (consistent coord system)
    cen = (df.filter(~pl.col("cell_id").is_in(["-1", "UNASSIGNED", "nan", "NA", ""]))
           .group_by("cell_id").agg(pl.col("x").mean().alias("cx"), pl.col("y").mean().alias("cy"),
                                    pl.len().alias("n_tx")).to_pandas())
    cen["cell_id"] = cen["cell_id"].astype(str)
    out = cen.merge(fr.per_cell[["cell_id", "mean_vsi"]], on="cell_id", how="inner")
    out["ovrlpy_problem"] = 1.0 - out["mean_vsi"]
    out.to_csv(cache, sep="\t", index=False)
    log(f"  [1B] ovrlpy s010 done: {len(out)} cells ({time.time()-t:.0f}s)")
    return out


def fig1b(log):
    # Large window containing repB + surrounding upper-right Atera tissue.
    win = {"xmin": 7000.0, "xmax": 8928.0, "ymin": 6700.0, "ymax": 8800.0}
    ov = _ovrlpy_largewindow(win, log)
    # TRACER relative conflict (whole-tissue) restricted to window cells.
    tr = pd.read_csv(ATERA_COMPARE, sep="\t", usecols=["cell_id", "cx", "cy", "relative_conflict"])
    tr["cell_id"] = tr["cell_id"].astype(str)
    tr = tr[(tr.cx >= win["xmin"]) & (tr.cx <= win["xmax"])
            & (tr.cy >= win["ymin"]) & (tr.cy <= win["ymax"])].copy()
    # join the two diagnostics on cell_id (same coordinate system)
    j = ov.merge(tr[["cell_id", "relative_conflict"]], on="cell_id", how="inner")
    j = j.rename(columns={"relative_conflict": "tracer_relconflict"})
    log(f"  [1B] window: ovrlpy={len(ov)} TRACER={len(tr)} joined={len(j)} cells")
    j.to_csv(OUT / "tables" / "fig1b_window_cells.tsv.gz", sep="\t", index=False)

    fo = _flags(j["ovrlpy_problem"].to_numpy())
    ft = _flags(j["tracer_relconflict"].to_numpy())
    both = fo & ft
    # export coords
    (OUT / "tables" / "fig1b_coords.json").write_text(json.dumps(
        {"large_window_um": win, "repB_roi_um": BEST_ROI,
         "ovrlpy_params": OVRLPY_PARAMS, "flag_quantile": FLAG_Q,
         "n_cells_window": int(len(j)), "n_both_high_problem": int(both.sum())}, indent=2))

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), dpi=200)
        # i) transcript-density background
        ax = axes[0]
        ax.hexbin(j["cx"], j["cy"], C=j["n_tx"], gridsize=60, cmap="bone",
                  reduce_C_function=np.sum, linewidths=0)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("transcript density\n(context)", fontsize=9, color="white")
        # ii) ovrlpy problem s010
        _scatter(axes[1], j["cx"], j["cy"], j["ovrlpy_problem"], vmin=0, vmax=1,
                 fig=fig, label="1 - VSI")
        axes[1].set_title("ovrlpy problem score (s010)\nhigh = problem (ambiguous)", fontsize=9, color="white")
        # iii) TRACER relative conflict
        vmax = np.nanpercentile(j["tracer_relconflict"], 98)
        _scatter(axes[2], j["cx"], j["cy"], j["tracer_relconflict"], vmin=0, vmax=vmax,
                 fig=fig, label="rel. conflict")
        axes[2].set_title("TRACER relative conflict\nhigh = problem (ambiguous)", fontsize=9, color="white")
        # iv) concordance overlay
        ax = axes[3]
        ax.scatter(j["cx"], j["cy"], c=COL_BG, s=4, linewidths=0, rasterized=True)
        ax.scatter(j["cx"][fo ^ ft], j["cy"][fo ^ ft], c=COL_ONE, s=7, linewidths=0,
                   label="one method high", rasterized=True)
        ax.scatter(j["cx"][both], j["cy"][both], c=COL_BOTH, s=13,
                   edgecolors="white", linewidths=0.2, label="ovrlpy & TRACER high")
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"concordance overlay\nboth high problem n={int(both.sum())}", fontsize=9, color="white")
        ax.legend(loc="lower left", fontsize=6, facecolor="black", framealpha=0.6)
        # repB bbox on all panels
        for ax in axes:
            r = BEST_ROI
            ax.add_patch(Rectangle((r["xmin"], r["ymin"]), r["xmax"]-r["xmin"], r["ymax"]-r["ymin"],
                                   edgecolor=COL_ROI, facecolor="none", lw=1.6))
        fig.suptitle("Figure 1B — Atera cervical cancer: structured conflict map "
                     "(ovrlpy x TRACER; magenta box = convergent ROI repB)",
                     fontsize=12, color="white", y=1.02)
        fig.tight_layout()
        _save(fig, "fig1B_atera_structured_conflict")
    log("  [1B] wrote fig1B_atera_structured_conflict.{png,svg,pdf}")


# ===========================================================================
#  FIG 1C — VisiumHD structured ambiguity map (RCTD x TRACER)
# ===========================================================================
def fig1c(log):
    d = pd.read_csv(VHD_JOINED, sep="\t")
    d = d[d["active_in_rctd"] == True].copy()  # noqa: E712
    d["rctd_lowmaxweight"] = 1.0 - d["RCTD_max_weight"]
    x, y = d["cx_um"], d["cy_um"]
    rois = json.loads(VHD_ROIS.read_text()) if VHD_ROIS.exists() else {}
    fr = _flags(d["rctd_norm_entropy" if "rctd_norm_entropy" in d else "RCTD_norm_entropy"].to_numpy())
    ft = _flags(d["TRACER_relative_conflict"].to_numpy())
    both = fr & ft
    GS = 110  # hexbin gridsize -> smooth per-bin noise into spatial structure

    def hexmap(ax, c, *, vmin=None, vmax=None, cmap="magma", label="", reduce=np.mean):
        if vmin is None:
            vmin, vmax = np.nanpercentile(c, [2, 98])
        hb = ax.hexbin(x, y, C=c, gridsize=GS, cmap=cmap, reduce_C_function=reduce,
                       vmin=vmin, vmax=vmax, mincnt=1, linewidths=0)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors="white")
        if label:
            cb.set_label(label, fontsize=6, color="white")
        return hb

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), dpi=200)
        # i) anatomical context = dominant lineage (mode per hexbin)
        ax = axes[0]
        lin = d["predicted_dominant_lineage"].astype("category")
        ax.hexbin(x, y, C=lin.cat.codes, gridsize=GS, cmap="tab20",
                  reduce_C_function=lambda v: np.bincount(np.asarray(v, int)).argmax(),
                  mincnt=1, linewidths=0)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("anatomical context\n(RCTD dominant lineage)", fontsize=9, color="white")
        # ii) RCTD entropy   iii) RCTD 1-maxweight (SEPARATE)   iv) TRACER conflict
        hexmap(axes[1], d["RCTD_norm_entropy"].to_numpy(), label="norm. entropy")
        axes[1].set_title("RCTD norm. entropy\nhigh = ambiguous", fontsize=9, color="white")
        hexmap(axes[2], d["rctd_lowmaxweight"].to_numpy(), label="1 - max weight")
        axes[2].set_title("RCTD (1 - max weight)\nhigh = ambiguous", fontsize=9, color="white")
        hexmap(axes[3], d["TRACER_relative_conflict"].to_numpy(), vmin=0,
               vmax=np.nanpercentile(d["TRACER_relative_conflict"], 98), label="rel. conflict")
        axes[3].set_title("TRACER relative conflict\nhigh = problem (ambiguous)", fontsize=9, color="white")
        # v) concordance: fraction of both-high bins per hexbin
        hexmap(axes[4], both.astype(float), vmin=0, vmax=1, cmap="cividis",
               label="frac both-high", reduce=np.mean)
        axes[4].set_title(f"RCTD & TRACER concordance\nboth high n={int(both.sum())} "
                          f"({100*both.mean():.0f}% of bins)", fontsize=9, color="white")
        # ROI boxes (no in-canvas text)
        for ax in axes:
            for cat, rl in (rois.items() if isinstance(rois, dict) else []):
                if not isinstance(rl, list):
                    continue
                for r in rl[:2]:
                    if all(k in r for k in ("xmin", "xmax", "ymin", "ymax")):
                        ax.add_patch(Rectangle((r["xmin"], r["ymin"]), r["xmax"]-r["xmin"],
                                               r["ymax"]-r["ymin"], edgecolor=COL_ROI,
                                               facecolor="none", lw=0.8))
        fig.suptitle("Figure 1C — Kidney VisiumHD: structured ambiguity map "
                     "(RCTD entropy & 1-max-weight x TRACER)", fontsize=12, color="white", y=1.02)
        fig.tight_layout()
        _save(fig, "fig1C_visiumhd_structured_ambiguity")
    log(f"  [1C] wrote fig1C (n bins={len(d)}, both-high={int(both.sum())})")


# ===========================================================================
#  FIG 1D — cross-method quantitative concordance (Atera repB)
# ===========================================================================
def _battery_for(name: str):
    bats = json.loads((CONCORD / "all_batteries.json").read_text())
    return bats[name]


def fig1d(log):
    name = BEST_ROI["name"]
    b = _battery_for(name)
    cells = pd.read_csv(CONCORD / "per_roi_cells" / f"{name}_cells.tsv.gz", sep="\t").dropna(subset=METRICS)
    short = [METRIC_SHORT[m] for m in METRICS]
    # matrices
    Msp = np.full((4, 4), np.nan); Mor = np.full((4, 4), np.nan)
    for i, a in enumerate(METRICS):
        Msp[i, i] = 1.0
        for k, v in b["spearman"].items():
            pass
        for j, bb in enumerate(METRICS):
            if j <= i:
                continue
            key = f"{a}__{bb}"
            Msp[i, j] = Msp[j, i] = b["spearman"][key]
            Mor[i, j] = Mor[j, i] = b["odds_ratio"][key]
    # triple-flag null
    rng = np.random.default_rng(0)
    fo = _flags(cells["ovrlpy_problem"].to_numpy())
    ft = _flags(cells["tracer_relconflict"].to_numpy())
    fr = _flags(cells["rctd_entropy"].to_numpy())
    obs = int((fo & ft & fr).sum())
    null = np.array([int((fo & rng.permutation(ft) & rng.permutation(fr)).sum()) for _ in range(2000)])

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200)
        # i) Spearman heatmap
        ax = axes[0]
        im = ax.imshow(Msp, cmap="RdBu_r", vmin=-0.7, vmax=0.7)
        for i in range(4):
            for j in range(4):
                if np.isfinite(Msp[i, j]):
                    ax.text(j, i, f"{Msp[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="black" if abs(Msp[i, j]) < 0.45 else "white")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7); ax.set_yticklabels(short, fontsize=7)
        ax.set_title("pairwise Spearman rho", fontsize=9, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6, colors="white")
        # ii) odds-ratio heatmap (log2)
        ax = axes[1]
        im = ax.imshow(np.log2(Mor), cmap="magma", vmin=0, vmax=4)
        for i in range(4):
            for j in range(4):
                if np.isfinite(Mor[i, j]):
                    ax.text(j, i, f"{Mor[i, j]:.1f}", ha="center", va="center", fontsize=8, color="white")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7); ax.set_yticklabels(short, fontsize=7)
        ax.set_title("flag enrichment odds ratio", fontsize=9, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6, colors="white")
        # iii) Moran's I + perm p
        ax = axes[2]
        mi = [b["moran"][m] for m in METRICS]; mp = [b["moran_p"][m] for m in METRICS]
        bars = ax.bar(range(4), mi, color=["#00E5FF", "#FF1493", "#39FF14", "#FFB000"])
        for i, (v, p) in enumerate(zip(mi, mp)):
            ax.text(i, v + 0.01, f"p={p:.3f}", ha="center", fontsize=6.5, color="white")
        ax.set_xticks(range(4)); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Moran's I", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_title("spatial structure (Moran's I)\nspatial-permutation p", fontsize=9, color="white")
        # iv) observed vs permuted triple-flag
        ax = axes[3]
        ax.hist(null, bins=40, color="#888899", alpha=0.8, label="permuted null")
        ax.axvline(obs, color=COL_BOTH, lw=2.5, label=f"observed = {obs}")
        ax.axvline(null.mean(), color="white", lw=1, ls="--", label=f"null mean = {null.mean():.0f}")
        enr = obs / max(null.mean(), 1e-9)
        ax.set_title(f"triple-flag overlap\nenrichment = {enr:.1f}x  perm p = {b['convergence']['triple_perm_p']:.3f}",
                     fontsize=9, color="white")
        ax.set_xlabel("# cells flagged by all 3 methods", fontsize=8); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, facecolor="black")
        fig.suptitle(f"Figure 1D — three independent diagnostics converge on the same structured regions "
                     f"(Atera ROI {name}, x[{BEST_ROI['xmin']:.0f},{BEST_ROI['xmax']:.0f}] "
                     f"y[{BEST_ROI['ymin']:.0f},{BEST_ROI['ymax']:.0f}] um)",
                     fontsize=11.5, color="white", y=1.03)
        fig.tight_layout()
        _save(fig, "fig1D_cross_method_concordance")
    # table
    pd.DataFrame({"metric": METRICS, "moran_I": mi, "moran_perm_p": mp}).to_csv(
        OUT / "tables" / "fig1D_moran.tsv", sep="\t", index=False)
    log(f"  [1D] wrote fig1D (triple obs={obs}, null mean={null.mean():.1f}, enr={enr:.1f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("b", "c", "d", "supp", "all"), default="all")
    args = ap.parse_args()
    _dirs()
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    log(f"convention: '+' = {PROBLEM_POS}; '-' = {PROBLEM_NEG}")
    if args.stage in ("b", "all"):
        fig1b(log)
    if args.stage in ("c", "all"):
        fig1c(log)
    if args.stage in ("d", "all"):
        fig1d(log)
    if args.stage in ("supp", "all"):
        import fig1_supp
        fig1_supp.run(OUT, log)
    log("DONE")


if __name__ == "__main__":
    main()
