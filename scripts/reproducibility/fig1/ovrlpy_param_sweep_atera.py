#!/usr/bin/env python3
"""ovrlpy hyperparameter sensitivity sweep for the Atera cervical-cancer sample.

Motivation
==========
The headline Figure-1 ovrlpy inset (``results/ovrlpy_tracer/
cervical_atera_full_memoryaware``) shows a *saturated* problem-score map: the
saved per-cell ``mean_vsi`` spans roughly [-0.9, 1] with a median of ~0.10,
i.e. ~90% of cells are flagged as low-integrity.  That distribution is **not
reproducible** with the currently installed ovrlpy (1.2.0): a direct per-ROI
fit on the identical QC'd input — at the documented hyperparameters and across
tile sizes/positions — yields a clean [0, 1] VSI with median ~0.77 and only
~5-8% of cells below 0.5.  The stale figure used an older *signed* integrity
convention; the current API returns a well-behaved [0, 1] VSI.

This script sweeps the ovrlpy hyperparameters that actually control VSI
structure (``KDE_bandwidth``, ``min_distance``, ``n_components`` and the two
``min_transcripts`` thresholds), on representative Atera ROIs, and for every
setting records:

  * the VSI distribution (median, IQR, saturation fraction),
  * a *structure* score (Moran's I of the per-cell problem score — do flagged
    cells cluster into regions, or are they salt-and-pepper noise?), and
  * TRACER concordance (Spearman / AUROC / Fisher odds of the ovrlpy problem
    score against the pre-computed TRACER ``relative_conflict`` /
    ``is_conflict_relu`` per cell).

It then selects the best **non-saturated, structured, TRACER-concordant**
setting and writes the full sensitivity tables + per-setting problem-score maps
for the supplement.  The chosen setting is consumed by
``regen_fig1_ovrlpy_inset.py`` to replace the saturated headline inset.

Stages (``--stage``)
  extract  : crop ROI transcript subsets from filtered_df.parquet (cached).
  sweep    : run the full OFAT + 2D grid on the primary ROI (+ confirmation
             baseline/best on secondary ROIs); write tables + maps.
  select   : rank settings and write the selection summary.
  all      : extract -> sweep -> select (default).

All heavy ovrlpy fits run on small cached ROI parquets, so the whole sweep is
minutes, not the ~1 h whole-tissue tiled run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[3]
SRC_PARQUET = _REPO / "datasets/cervical_cancer_atera_10x/filtered_df.parquet"
HEADLINE = _REPO / "results/ovrlpy_tracer/cervical_atera_full_memoryaware"
PANEL_TSV = HEADLINE / "selected_ovrlpy_gene_panel.tsv"
TRACER_TSV = HEADLINE / "tables/tracer_cell_npmi_metrics.tsv"
OUTDIR = _REPO / "results/ovrlpy_tracer/param_sweep_atera"

# Representative ROIs (from the headline run's representative_rois.json).  We
# enlarge each canonical 400 um inset ROI to an 800 um context window so the
# sweep sees enough cells (and enough TRACER-conflict signal) to estimate
# concordance robustly while staying cheap to fit.
ROIS: dict[str, dict[str, float]] = {
    # category A: ovrlpy+ / TRACER+  (concordant problem) -> PRIMARY sweep ROI
    "A1": {"cx": 6700.6, "cy": 7134.8, "half": 400.0, "category": "A_ovrlpy+_tracer+"},
    # category B: ovrlpy- / TRACER+  (contamination without vertical structure)
    "B1": {"cx": 8784.3, "cy": 8390.2, "half": 400.0, "category": "B_ovrlpy-_tracer+"},
    # category C: ovrlpy+ / TRACER-  (benign z-overlap)
    "C1": {"cx": 2935.5, "cy": 6100.4, "half": 400.0, "category": "C_ovrlpy+_tracer-"},
}
PRIMARY_ROI = "A1"

# Baseline = the headline run's documented ovrlpy parameters.
BASELINE = dict(KDE_bandwidth=2.5, min_distance=8, n_components=20,
                min_tx_local_max=10, min_tx_vsi=2)

# One-factor-at-a-time sweep axes (baseline value included implicitly).
OFAT = {
    "KDE_bandwidth": [1.0, 1.5, 2.5, 4.0, 6.0],
    "min_distance": [4, 8, 15, 25],
    "n_components": [10, 20, 30, 50],
    "min_tx_vsi": [2, 5, 10, 20],
    "min_tx_local_max": [5, 10, 20, 40],
}
# Focused 2D grid for a heatmap supplement (the two most impactful smoothing
# knobs): KDE_bandwidth x min_distance.
GRID_2D = {"KDE_bandwidth": [1.0, 2.5, 4.0, 6.0], "min_distance": [4, 8, 15, 25]}

VSI_LOW = 0.5  # problem threshold (cell flagged if mean_vsi < VSI_LOW)


# ---------------------------------------------------------------------------
def _log() -> logging.Logger:
    lg = logging.getLogger("ovrlpy_sweep")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    lg.addHandler(h)
    lg.propagate = False
    return lg


def _load_panel() -> list[str]:
    return pd.read_csv(PANEL_TSV, sep="\t")["gene"].astype(str).tolist()


def _load_tracer() -> pd.DataFrame:
    df = pd.read_csv(TRACER_TSV, sep="\t",
                     usecols=["cell_id", "relative_conflict", "cell_conflict_relu",
                              "is_conflict_relu", "relative_purity"])
    df["cell_id"] = df["cell_id"].astype(str)
    return df


# ---------------------------------------------------------------------------
#  Stage: extract
# ---------------------------------------------------------------------------
def extract_rois(log: logging.Logger) -> None:
    cache = OUTDIR / "roi_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name, r in ROIS.items():
        out = cache / f"{name}.parquet"
        if out.exists():
            log.info("[extract] %s cached", name)
            continue
        t0 = time.time()
        cx, cy, half = r["cx"], r["cy"], r["half"]
        df = (pl.scan_parquet(SRC_PARQUET)
              .rename({"feature_name": "gene", "x_location": "x",
                       "y_location": "y", "z_location": "z"})
              .filter((pl.col("x") >= cx - half) & (pl.col("x") <= cx + half)
                      & (pl.col("y") >= cy - half) & (pl.col("y") <= cy + half))
              .select(["x", "y", "z", "gene", "cell_id"])
              .collect(engine="streaming"))
        df.write_parquet(out)
        log.info("[extract] %s: %d tx, %d cells, %d genes (%.1fs)",
                 name, df.height, df["cell_id"].n_unique(), df["gene"].n_unique(),
                 time.time() - t0)


# ---------------------------------------------------------------------------
#  ovrlpy fit + per-cell aggregation
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    per_cell: pd.DataFrame      # cell_id, mean_vsi, cx, cy, n_pixels
    per_pixel: pd.DataFrame     # x_pixel, y_pixel, vsi  (for maps)
    n_pseudocells: int
    runtime_s: float


def run_ovrlpy(df: pl.DataFrame, panel: list[str], *, KDE_bandwidth: float,
               min_distance: float, n_components: int, min_tx_local_max: float,
               min_tx_vsi: float, n_workers: int, seed: int) -> FitResult:
    import ovrlpy
    t0 = time.time()
    present = set(df["gene"].unique().to_list())
    p = [g for g in panel if g in present]
    ov = ovrlpy.Ovrlp(df, KDE_bandwidth=KDE_bandwidth, min_distance=min_distance,
                      n_components=n_components, n_workers=n_workers, random_state=seed)
    ov.process_coordinates()
    ov.fit_transcripts(min_transcripts=min_tx_local_max, genes=p or None)
    ov.compute_VSI(min_transcripts=min_tx_vsi)
    pp = ovrlpy.cell_integrity_from_transcripts(ov, cell_id="cell_id", unassigned="-1")
    pp = pp.to_pandas() if hasattr(pp, "to_pandas") else pp
    ic = "integrity" if "integrity" in pp.columns else "vsi"
    pp = pp.rename(columns={ic: "vsi"})
    pp = pp[pp["cell_id"].notna()
            & ~pp["cell_id"].astype(str).isin({"-1", "nan", "NA", "None", ""})]
    pp["cell_id"] = pp["cell_id"].astype(str)
    xcol = "x_pixel" if "x_pixel" in pp.columns else ("x" if "x" in pp.columns else None)
    ycol = "y_pixel" if "y_pixel" in pp.columns else ("y" if "y" in pp.columns else None)
    g = pp.groupby("cell_id")
    per_cell = pd.DataFrame({
        "mean_vsi": g["vsi"].mean(),
        "n_pixels": g.size(),
    }).reset_index()
    if xcol and ycol:
        cen = g[[xcol, ycol]].mean().rename(columns={xcol: "cx", ycol: "cy"})
        per_cell = per_cell.merge(cen.reset_index(), on="cell_id", how="left")
    n_ps = int(getattr(getattr(ov, "pca", None), "n_samples_", 0) or 0)
    pix_cols = [c for c in (xcol, ycol, "vsi") if c]
    return FitResult(per_cell=per_cell,
                     per_pixel=pp[pix_cols].rename(columns={xcol: "x", ycol: "y"})
                     if xcol and ycol else pp[["vsi"]],
                     n_pseudocells=n_ps, runtime_s=time.time() - t0)


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def morans_i(xy: np.ndarray, vals: np.ndarray, k: int = 6) -> float:
    """Moran's I of `vals` over a kNN graph on centroids `xy` (row-standardised)."""
    n = len(vals)
    if n < k + 2:
        return float("nan")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, idx = nn.kneighbors(xy)
    idx = idx[:, 1:]                      # drop self
    z = vals - np.nanmean(vals)
    denom = np.nansum(z * z)
    if denom == 0:
        return float("nan")
    num = 0.0
    for i in range(n):
        num += z[i] * np.nansum(z[idx[i]])
    W = n * k
    return float((n / W) * (num / denom))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC of `score` predicting boolean `label` (rank-based, no sklearn dep)."""
    label = label.astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def setting_metrics(fr: FitResult, tracer: pd.DataFrame) -> dict[str, Any]:
    from scipy.stats import spearmanr, fisher_exact
    pc = fr.per_cell.copy()
    v = pc["mean_vsi"].to_numpy(float)
    problem = 1.0 - v
    out: dict[str, Any] = {
        "n_cells": int(len(pc)),
        "n_pseudocells": fr.n_pseudocells,
        "median_vsi": float(np.nanmedian(v)),
        "mean_vsi": float(np.nanmean(v)),
        "q25_vsi": float(np.nanpercentile(v, 25)),
        "q75_vsi": float(np.nanpercentile(v, 75)),
        "frac_low_vsi": float(np.nanmean(v < VSI_LOW)),     # saturation indicator
        "runtime_s": round(fr.runtime_s, 2),
    }
    # spatial structure of the problem score
    if {"cx", "cy"}.issubset(pc.columns):
        xy = pc[["cx", "cy"]].to_numpy(float)
        ok = np.isfinite(xy).all(1) & np.isfinite(problem)
        out["morans_i_problem"] = morans_i(xy[ok], problem[ok]) if ok.sum() > 10 else float("nan")
    else:
        out["morans_i_problem"] = float("nan")
    # TRACER concordance
    j = pc.merge(tracer, on="cell_id", how="inner")
    out["n_joined_tracer"] = int(len(j))
    if len(j) > 20:
        pr = 1.0 - j["mean_vsi"].to_numpy(float)
        rc = j["relative_conflict"].to_numpy(float)
        m = np.isfinite(pr) & np.isfinite(rc)
        rho, _ = spearmanr(pr[m], rc[m])
        out["spearman_problem_vs_relconflict"] = float(rho)
        out["auroc_problem_predicts_conflict"] = auroc(pr[m], j["is_conflict_relu"].to_numpy()[m])
        lo = pr >= np.nanpercentile(pr, 80)              # top-20% problem = ovrlpy "low VSI"
        hc = j["is_conflict_relu"].to_numpy().astype(bool)
        ct = np.array([[int((lo & hc).sum()), int((lo & ~hc).sum())],
                       [int((~lo & hc).sum()), int((~lo & ~hc).sum())]])
        try:
            odds, p = fisher_exact(ct, alternative="greater")
        except Exception:
            odds, p = float("nan"), float("nan")
        out["fisher_odds"] = float(odds)
        out["fisher_p"] = float(p)
    else:
        out.update(spearman_problem_vs_relconflict=float("nan"),
                   auroc_problem_predicts_conflict=float("nan"),
                   fisher_odds=float("nan"), fisher_p=float("nan"))
    return out


# ---------------------------------------------------------------------------
#  Maps
# ---------------------------------------------------------------------------
def save_problem_map(fr: FitResult, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pc = fr.per_cell
    if not {"cx", "cy"}.issubset(pc.columns):
        return
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=130)
    sc = ax.scatter(pc["cx"], pc["cy"], c=1.0 - pc["mean_vsi"], s=6,
                    cmap="magma", vmin=0, vmax=1, linewidths=0)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("problem score (1 - VSI)", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Stage: sweep
# ---------------------------------------------------------------------------
def _settings_list() -> list[dict[str, Any]]:
    """Deduplicated baseline + OFAT + 2D-grid settings, each tagged by axis."""
    seen: dict[tuple, dict] = {}

    def key(s):
        return tuple(s[k] for k in ("KDE_bandwidth", "min_distance", "n_components",
                                    "min_tx_local_max", "min_tx_vsi"))

    def add(over: dict, axis: str):
        s = dict(BASELINE); s.update(over)
        k = key(s)
        if k not in seen:
            s = dict(s); s["_axes"] = {axis}; s["_id"] = f"s{len(seen):03d}"
            seen[k] = s
        else:
            seen[k]["_axes"].add(axis)

    add({}, "baseline")
    for axis, vals in OFAT.items():
        for val in vals:
            add({axis: val}, f"ofat:{axis}")
    for kde in GRID_2D["KDE_bandwidth"]:
        for md in GRID_2D["min_distance"]:
            add({"KDE_bandwidth": kde, "min_distance": md}, "grid2d")
    for s in seen.values():
        s["_axes"] = ",".join(sorted(s["_axes"]))
    return list(seen.values())


def sweep(log: logging.Logger, n_workers: int, seed: int) -> None:
    cache = OUTDIR / "roi_cache"
    tabs = OUTDIR / "tables"; tabs.mkdir(parents=True, exist_ok=True)
    figs = OUTDIR / "figures" / "sensitivity_maps"; figs.mkdir(parents=True, exist_ok=True)
    panel = _load_panel()
    tracer = _load_tracer()
    settings = _settings_list()
    log.info("Sweep: %d unique settings on primary ROI '%s'", len(settings), PRIMARY_ROI)

    rows: list[dict] = []
    # ---- full sweep on primary ROI (maps for every setting) ----
    df = pl.read_parquet(cache / f"{PRIMARY_ROI}.parquet")
    for i, s in enumerate(settings, 1):
        kw = {k: s[k] for k in ("KDE_bandwidth", "min_distance", "n_components",
                                "min_tx_local_max", "min_tx_vsi")}
        try:
            fr = run_ovrlpy(df, panel, n_workers=n_workers, seed=seed, **kw)
        except Exception as e:
            log.warning("  [%d/%d] %s FAILED: %s", i, len(settings), s["_id"], e)
            rows.append({"roi": PRIMARY_ROI, **s, "error": str(e)})
            continue
        m = setting_metrics(fr, tracer)
        rows.append({"roi": PRIMARY_ROI, **{k: s[k] for k in
                     ("_id", "_axes", "KDE_bandwidth", "min_distance", "n_components",
                      "min_tx_local_max", "min_tx_vsi")}, **m})
        title = (f"{s['_id']}  KDE={kw['KDE_bandwidth']} md={kw['min_distance']} "
                 f"nc={kw['n_components']} mtl={kw['min_tx_local_max']} mtv={kw['min_tx_vsi']}\n"
                 f"medVSI={m['median_vsi']:.2f} fracLow={m['frac_low_vsi']:.2f} "
                 f"MoranI={m['morans_i_problem']:.2f} AUROC={m.get('auroc_problem_predicts_conflict', float('nan')):.2f}")
        save_problem_map(fr, figs / f"{PRIMARY_ROI}_{s['_id']}.png", title)
        log.info("  [%d/%d] %s medVSI=%.2f fracLow=%.2f MoranI=%.2f AUROC=%.2f rho=%.2f (%.1fs)",
                 i, len(settings), s["_id"], m["median_vsi"], m["frac_low_vsi"],
                 m["morans_i_problem"], m.get("auroc_problem_predicts_conflict", float("nan")),
                 m.get("spearman_problem_vs_relconflict", float("nan")), m["runtime_s"])

    df_primary = pd.DataFrame(rows)
    df_primary.to_csv(tabs / "sweep_primary_roi.tsv", sep="\t", index=False)
    log.info("Wrote %s", tabs / "sweep_primary_roi.tsv")

    # ---- baseline confirmation on secondary ROIs ----
    sec_rows: list[dict] = []
    for roi in [r for r in ROIS if r != PRIMARY_ROI]:
        dfs = pl.read_parquet(cache / f"{roi}.parquet")
        try:
            fr = run_ovrlpy(dfs, panel, n_workers=n_workers, seed=seed,
                            **{k: BASELINE[k] for k in
                               ("KDE_bandwidth", "min_distance", "n_components",
                                "min_tx_local_max", "min_tx_vsi")})
            m = setting_metrics(fr, tracer)
            sec_rows.append({"roi": roi, "category": ROIS[roi]["category"], **BASELINE, **m})
            save_problem_map(fr, figs / f"{roi}_baseline.png",
                             f"{roi} ({ROIS[roi]['category']}) baseline\nmedVSI={m['median_vsi']:.2f} "
                             f"fracLow={m['frac_low_vsi']:.2f} AUROC={m.get('auroc_problem_predicts_conflict', float('nan')):.2f}")
            log.info("[secondary] %s medVSI=%.2f fracLow=%.2f AUROC=%.2f",
                     roi, m["median_vsi"], m["frac_low_vsi"],
                     m.get("auroc_problem_predicts_conflict", float("nan")))
        except Exception as e:
            log.warning("[secondary] %s FAILED: %s", roi, e)
    if sec_rows:
        pd.DataFrame(sec_rows).to_csv(tabs / "sweep_secondary_rois_baseline.tsv",
                                      sep="\t", index=False)

    # ---- 2D-grid heatmaps (from primary-ROI rows tagged grid2d) ----
    _grid_heatmaps(df_primary, OUTDIR / "figures", log)


def _grid_heatmaps(df: pd.DataFrame, figdir: Path, log: logging.Logger) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = df[df["_axes"].astype(str).str.contains("grid2d", na=False)
           & (df["n_components"] == BASELINE["n_components"])
           & (df["min_tx_vsi"] == BASELINE["min_tx_vsi"])
           & (df["min_tx_local_max"] == BASELINE["min_tx_local_max"])]
    if g.empty:
        return
    kdes = sorted(g["KDE_bandwidth"].unique())
    mds = sorted(g["min_distance"].unique())
    for metric, cmap in [("frac_low_vsi", "magma"), ("median_vsi", "viridis"),
                         ("morans_i_problem", "cividis"),
                         ("auroc_problem_predicts_conflict", "RdBu_r")]:
        M = np.full((len(mds), len(kdes)), np.nan)
        for _, r in g.iterrows():
            M[mds.index(r["min_distance"]), kdes.index(r["KDE_bandwidth"])] = r[metric]
        fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=130)
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(kdes))); ax.set_xticklabels(kdes)
        ax.set_yticks(range(len(mds))); ax.set_yticklabels(mds)
        ax.set_xlabel("KDE_bandwidth"); ax.set_ylabel("min_distance")
        ax.set_title(f"{metric}")
        for ii in range(len(mds)):
            for jj in range(len(kdes)):
                if np.isfinite(M[ii, jj]):
                    ax.text(jj, ii, f"{M[ii, jj]:.2f}", ha="center", va="center",
                            fontsize=7, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figdir / f"grid2d_{metric}.png", bbox_inches="tight")
        plt.close(fig)
    log.info("Wrote 2D-grid heatmaps to %s", figdir)


# ---------------------------------------------------------------------------
#  Stage: select
# ---------------------------------------------------------------------------
def select(log: logging.Logger) -> None:
    tabs = OUTDIR / "tables"
    df = pd.read_csv(tabs / "sweep_primary_roi.tsv", sep="\t")
    df = df[df.get("error").isna()] if "error" in df.columns else df
    # Non-saturated + structured band:
    #   * frac_low_vsi in [0.03, 0.45]  (flags a real minority, not ~all cells)
    #   * Moran's I > 0                  (flagged cells cluster into regions)
    cand = df[(df["frac_low_vsi"] >= 0.03) & (df["frac_low_vsi"] <= 0.45)
              & (df["morans_i_problem"] > 0)].copy()
    pool = cand if len(cand) else df.copy()
    # Rank by TRACER concordance: AUROC primary, Spearman secondary.
    pool["concordance"] = pool["auroc_problem_predicts_conflict"].fillna(0.5)
    pool = pool.sort_values(["concordance", "spearman_problem_vs_relconflict"],
                            ascending=False)
    best = pool.iloc[0]
    sel = {
        "selected_id": str(best["_id"]),
        "KDE_bandwidth": float(best["KDE_bandwidth"]),
        "min_distance": float(best["min_distance"]),
        "n_components": int(best["n_components"]),
        "min_tx_local_max": float(best["min_tx_local_max"]),
        "min_tx_vsi": float(best["min_tx_vsi"]),
        "median_vsi": float(best["median_vsi"]),
        "frac_low_vsi": float(best["frac_low_vsi"]),
        "morans_i_problem": float(best["morans_i_problem"]),
        "auroc_problem_predicts_conflict": float(best["auroc_problem_predicts_conflict"]),
        "spearman_problem_vs_relconflict": float(best["spearman_problem_vs_relconflict"]),
        "fisher_odds": float(best.get("fisher_odds", float("nan"))),
        "n_candidates_in_band": int(len(cand)),
        "selection_rule": ("frac_low_vsi in [0.03,0.45] AND Moran's I>0; "
                           "rank by AUROC(problem->conflict) then Spearman"),
    }
    (tabs.parent / "selected_setting.json").write_text(json.dumps(sel, indent=2))
    log.info("SELECTED %s: KDE=%.1f md=%g nc=%d mtl=%g mtv=%g | medVSI=%.2f "
             "fracLow=%.2f MoranI=%.2f AUROC=%.2f",
             sel["selected_id"], sel["KDE_bandwidth"], sel["min_distance"],
             sel["n_components"], sel["min_tx_local_max"], sel["min_tx_vsi"],
             sel["median_vsi"], sel["frac_low_vsi"], sel["morans_i_problem"],
             sel["auroc_problem_predicts_conflict"])
    log.info("Wrote %s", tabs.parent / "selected_setting.json")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("extract", "sweep", "select", "all"),
                    default="all")
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log = _log()
    if args.stage in ("extract", "all"):
        extract_rois(log)
    if args.stage in ("sweep", "all"):
        sweep(log, args.n_workers, args.seed)
    if args.stage in ("select", "all"):
        select(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
