#!/usr/bin/env python3
"""Three-method problem-region concordance on the Atera cervical-cancer sample.

Scores every cell in a set of candidate ROIs with three orthogonal
contamination / vertical-overlap diagnostics, kept as FOUR separate per-cell
metrics (RCTD entropy and RCTD low-max-weight are deliberately *not* merged
into one composite):

  * ovrlpy_problem        = 1 - VSI            (vertical signal overlap)
  * tracer_relconflict    = TRACER NPMI relative-conflict (co-expression prior)
  * rctd_entropy          = RCTD normalised entropy        (mixture ambiguity)
  * rctd_lowmaxweight     = 1 - RCTD max cell-type weight  (no dominant type)

For every ROI it quantifies, for each metric and every metric pair:

  spatial structure   Moran's I (kNN graph) + spatial-permutation null
  monotone agreement  pairwise Spearman rho (+ p)
  detection agreement AUROC (one continuous score predicting another's
                      top-quantile flag), enrichment odds ratio (Fisher 2x2 on
                      flags) + label-permutation null, and Jaccard overlap of
                      flagged sets

It then ranks ROIs by *three-way convergence* (ovrlpy x TRACER x RCTD all
spatially structured AND mutually concordant), exports the winning ROI's
coordinates, renders a dark Nature-style concordance panel, and writes the full
per-ROI / parameter-sensitivity tables and maps for the supplement.

Per-method parameter sweeps (``--stage sweep``) characterise sensitivity:
  ovrlpy : reuses results/ovrlpy_tracer/param_sweep_atera (KDE_bandwidth ...)
  TRACER : tau x conflict_percentile x npmi_min_occurrences on the winning ROI
  RCTD   : EM n_iter x lineage granularity on the winning ROI

All heavy fits run on small cached ROI crops, so the whole study is minutes.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

warnings.filterwarnings("ignore")

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_THIS.parent))

# reuse validated cores
from run_rctd_tracer_overlap import (  # noqa: E402
    build_lineage_signature, poisson_em_deconvolution, rctd_metrics,
)
from tracer.metrics import (  # noqa: E402
    build_npmi_matrix, build_cell_gene_matrix, compute_cell_conflict_relu,
)
import ovrlpy_param_sweep_atera as ops  # noqa: E402  (run_ovrlpy, morans_i, auroc)

# ---------------------------------------------------------------------------
SRC_PARQUET = _REPO / "datasets/cervical_cancer_atera_10x/filtered_df.parquet"
HEADLINE = _REPO / "results/ovrlpy_tracer/cervical_atera_full_memoryaware"
PANEL_TSV = HEADLINE / "selected_ovrlpy_gene_panel.tsv"
REF_H5AD = _REPO / "datasets/dataset/cervical_scrna/h5ad/cervical_scrna_adc_scc_marker_annotated.h5ad"
HVG_TSV = _REPO / "datasets/dataset/atera_cervical/hvg_gene_list.tsv"
NPMI_CSV = _REPO / "datasets/dataset/atera_cervical/npmi_panel.csv.gz"
ROI_RANKING = _REPO / "datasets/dataset/atera_cervical/candidate_roi_ranking.tsv"
OUTDIR = _REPO / "results/ovrlpy_tracer/atera_three_method_concordance"

# Chosen ovrlpy setting (s010 from the param sweep: best non-saturated,
# TRACER-concordant -- median VSI 0.65, frac_low 0.21, Moran's I 0.45).
OVRLPY_PARAMS = dict(KDE_bandwidth=2.5, min_distance=8, n_components=50,
                     min_tx_local_max=10, min_tx_vsi=2)
# Canonical TRACER conflict operating point (recomputed per ROI for self-
# consistency with the RCTD/ovrlpy ROI-local scoring).
TRACER_PARAMS = dict(tau=0.05, conflict_percentile=80.0, min_transcripts=10)
RCTD_PARAMS = dict(n_iter=30, chunk=8192, lineage_col="cell_type_coarse")

FLAG_Q = 0.80          # top-20% of a metric = "flagged problem cell"
N_PERM = 199           # permutation-null replicates
METRICS = ["ovrlpy_problem", "tracer_relconflict", "rctd_entropy", "rctd_lowmaxweight"]
# method-level grouping for the 3-way convergence (RCTD primary = entropy)
METHOD_METRIC = {"ovrlpy": "ovrlpy_problem", "tracer": "tracer_relconflict",
                 "rctd": "rctd_entropy"}


def _log() -> logging.Logger:
    lg = logging.getLogger("concord")
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    lg.addHandler(h)
    lg.propagate = False
    return lg


# ---------------------------------------------------------------------------
#  Shared resources (loaded once)
# ---------------------------------------------------------------------------
class Resources:
    def __init__(self, log: logging.Logger, lineage_col: str = "cell_type_coarse"):
        import h5py
        import anndata as ad
        self.log = log
        self.panel = pd.read_csv(PANEL_TSV, sep="\t")["gene"].astype(str).tolist()
        # --- RCTD reference signature (bypass anndata uns quirk via h5py) ---
        f = h5py.File(REF_H5AD, "r")
        g = f["layers"]["counts"]
        C = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                          shape=tuple(g.attrs["shape"]))
        var = [x.decode() if isinstance(x, bytes) else x for x in f["var"]["symbol"][:]]
        ct = f["obs"][lineage_col]
        cats = [x.decode() for x in ct["categories"][:]]
        codes = ct["codes"][:]
        lineage = np.array([cats[c] if c >= 0 else "Unannotated" for c in codes], dtype=object)
        f.close()
        keep = lineage != "Unannotated"
        ref = ad.AnnData(X=C[keep], obs=pd.DataFrame({"lineage": lineage[keep]}),
                         var=pd.DataFrame(index=np.array(var)))
        ref.layers["counts"] = ref.X
        hvg = pd.read_csv(HVG_TSV, sep="\t")["gene"].astype(str).to_numpy()
        self.S, self.lineages, self.sig_genes = build_lineage_signature(ref, hvg, log)
        self.sig_idx = {g: i for i, g in enumerate(self.sig_genes)}
        log.info("RCTD signature: %d genes x %d lineages", self.S.shape[0], self.S.shape[1])
        # --- TRACER NPMI matrix ---
        npmi_long = pd.read_csv(NPMI_CSV, usecols=["gene_i", "gene_j", "NPMI"])
        npmi_long["gene_i"] = npmi_long["gene_i"].astype(str)
        npmi_long["gene_j"] = npmi_long["gene_j"].astype(str)
        self.npmi_long = npmi_long
        self.npmi_mat, self.npmi_genes = build_npmi_matrix(npmi_long)
        self.npmi_gene_set = set(map(str, self.npmi_genes))
        log.info("TRACER NPMI matrix: %d genes", len(self.npmi_genes))


# ---------------------------------------------------------------------------
#  Per-ROI scoring
# ---------------------------------------------------------------------------
def crop_roi(roi: dict, cache: Path) -> pl.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    fp = cache / f"{roi['name']}.parquet"
    if fp.exists():
        return pl.read_parquet(fp)
    df = (pl.scan_parquet(SRC_PARQUET)
          .rename({"feature_name": "gene", "x_location": "x",
                   "y_location": "y", "z_location": "z"})
          .filter((pl.col("x") >= roi["xmin"]) & (pl.col("x") <= roi["xmax"])
                  & (pl.col("y") >= roi["ymin"]) & (pl.col("y") <= roi["ymax"]))
          .select(["x", "y", "z", "gene", "cell_id"])
          .collect(engine="streaming"))
    df.write_parquet(fp)
    return df


def cell_centroids(df: pl.DataFrame) -> pd.DataFrame:
    c = (df.filter(~pl.col("cell_id").is_in(["-1", "UNASSIGNED", "nan", "NA", ""]))
         .group_by("cell_id")
         .agg(pl.col("x").mean().alias("cx"), pl.col("y").mean().alias("cy"),
              pl.len().alias("n_tx"))
         .to_pandas())
    c["cell_id"] = c["cell_id"].astype(str)
    return c


def score_ovrlpy(df: pl.DataFrame, panel: list[str], params: dict,
                 n_workers: int, seed: int) -> pd.DataFrame:
    fr = ops.run_ovrlpy(df, panel, n_workers=n_workers, seed=seed, **params)
    out = fr.per_cell[["cell_id", "mean_vsi"]].copy()
    out["ovrlpy_problem"] = 1.0 - out["mean_vsi"]
    return out[["cell_id", "ovrlpy_problem"]]


def score_rctd(df: pl.DataFrame, res: Resources, *, n_iter: int, chunk: int) -> pd.DataFrame:
    sub = df.filter(pl.col("gene").is_in(list(res.sig_genes))
                    & ~pl.col("cell_id").is_in(["-1", "UNASSIGNED", "nan", "NA", ""]))
    pdf = sub.group_by(["cell_id", "gene"]).len().to_pandas()
    if pdf.empty:
        return pd.DataFrame(columns=["cell_id", "rctd_entropy", "rctd_lowmaxweight"])
    cids = pdf["cell_id"].astype(str).unique()
    cmap = {c: i for i, c in enumerate(cids)}
    rows = pdf["cell_id"].astype(str).map(cmap).to_numpy()
    cols = pdf["gene"].map(res.sig_idx).to_numpy()
    vals = pdf["len"].to_numpy().astype(np.float32)
    Y = sp.coo_matrix((vals, (rows, cols)), shape=(len(cids), len(res.sig_genes))).tocsr()
    W, _cc, active = poisson_em_deconvolution(
        Y, np.asarray(res.sig_genes, dtype=object), res.S, res.sig_genes,
        n_iter=n_iter, chunk_size=chunk, logger=res.log)
    met = rctd_metrics(W, res.lineages, active)
    out = pd.DataFrame({
        "cell_id": cids.astype(str),
        "rctd_entropy": met["RCTD_norm_entropy"].to_numpy(),
        "rctd_lowmaxweight": 1.0 - met["RCTD_max_weight"].to_numpy(),
    })
    return out


def score_tracer(df: pl.DataFrame, res: Resources, *, tau: float,
                 conflict_percentile: float, min_transcripts: int) -> pd.DataFrame:
    fdf = (df.filter(~pl.col("cell_id").is_in(["-1", "UNASSIGNED", "nan", "NA", ""]))
           .select(["cell_id", "gene"]).to_pandas())
    fdf["cell_id"] = fdf["cell_id"].astype(str)
    fdf = fdf.rename(columns={"gene": "feature_name"})
    cell_ids, _genes, M, col_idx = build_cell_gene_matrix(
        fdf, min_transcripts=min_transcripts, genes_npm=res.npmi_long,
        cell_col="cell_id")
    if len(cell_ids) == 0:
        return pd.DataFrame(columns=["cell_id", "tracer_relconflict"])
    _conf, _is, _thr, cdf = compute_cell_conflict_relu(
        M=M, col_idx=col_idx, npmi_mat=res.npmi_mat, tau=tau,
        cell_ids=cell_ids, conflict_percentile=conflict_percentile)
    out = cdf[["cell_id", "relative_conflict"]].rename(
        columns={"relative_conflict": "tracer_relconflict"})
    out["cell_id"] = out["cell_id"].astype(str)
    return out


def score_roi(roi: dict, res: Resources, cache: Path, *, ovrlpy_params=OVRLPY_PARAMS,
              tracer_params=TRACER_PARAMS, rctd_params=RCTD_PARAMS,
              n_workers=4, seed=42) -> pd.DataFrame:
    df = crop_roi(roi, cache)
    cen = cell_centroids(df)
    ov = score_ovrlpy(df, res.panel, ovrlpy_params, n_workers, seed)
    rc = score_rctd(df, res, n_iter=rctd_params["n_iter"], chunk=rctd_params["chunk"])
    tr = score_tracer(df, res, tau=tracer_params["tau"],
                      conflict_percentile=tracer_params["conflict_percentile"],
                      min_transcripts=tracer_params["min_transcripts"])
    cells = (cen.merge(ov, on="cell_id", how="left")
             .merge(rc, on="cell_id", how="left")
             .merge(tr, on="cell_id", how="left"))
    return cells


# ---------------------------------------------------------------------------
#  Stats battery
# ---------------------------------------------------------------------------
def _flags(v: np.ndarray, q: float) -> np.ndarray:
    thr = np.nanquantile(v, q)
    return v >= thr


class FastMoran:
    """Precompute a kNN graph once; evaluate Moran's I (vectorised) for many
    value vectors -- cheap permutation nulls over a fixed geometry."""

    def __init__(self, xy: np.ndarray, k: int = 6):
        from sklearn.neighbors import NearestNeighbors
        self.n = len(xy)
        self.k = min(k, max(1, self.n - 1))
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(xy)
        _, idx = nn.kneighbors(xy)
        self.idx = idx[:, 1:]
        self.W = self.n * self.k

    def I(self, vals: np.ndarray) -> float:
        z = vals - np.nanmean(vals)
        denom = np.nansum(z * z)
        if denom == 0:
            return float("nan")
        neigh = np.nansum(z[self.idx], axis=1)
        return float((self.n / self.W) * (np.nansum(z * neigh) / denom))


def battery(cells: pd.DataFrame, *, flag_q=FLAG_Q, n_perm=N_PERM, rng=None) -> dict:
    from scipy.stats import spearmanr, fisher_exact
    if rng is None:
        rng = np.random.default_rng(0)
    # restrict to cells scored by ALL methods
    cc = cells.dropna(subset=METRICS).reset_index(drop=True)
    n = len(cc)
    res: dict[str, Any] = {"n_cells_all_methods": int(n)}
    if n < 50:
        res["insufficient"] = True
        return res
    xy = cc[["cx", "cy"]].to_numpy(float)
    vals = {m: cc[m].to_numpy(float) for m in METRICS}
    flags = {m: _flags(vals[m], flag_q) for m in METRICS}

    # ---- Moran's I + spatial-permutation null ----
    fm = FastMoran(xy)
    moran, moran_z, moran_p = {}, {}, {}
    for m in METRICS:
        I = fm.I(vals[m])
        null = np.array([fm.I(rng.permutation(vals[m])) for _ in range(n_perm)])
        mu, sd = np.nanmean(null), np.nanstd(null) + 1e-12
        moran[m] = float(I)
        moran_z[m] = float((I - mu) / sd)
        moran_p[m] = float((np.sum(null >= I) + 1) / (n_perm + 1))
    res["moran"] = moran; res["moran_z"] = moran_z; res["moran_p"] = moran_p

    # ---- pairwise concordance ----
    pairs = [(a, b) for i, a in enumerate(METRICS) for b in METRICS[i + 1:]]
    spear, spear_p, auroc_ab, odds, odds_p, jacc = {}, {}, {}, {}, {}, {}
    for a, b in pairs:
        key = f"{a}__{b}"
        va, vb = vals[a], vals[b]
        rho, p = spearmanr(va, vb)
        spear[key] = float(rho); spear_p[key] = float(p)
        # AUROC: symmetric mean of a->flag_b and b->flag_a
        au1 = ops.auroc(va, flags[b]); au2 = ops.auroc(vb, flags[a])
        auroc_ab[key] = float(np.nanmean([au1, au2]))
        # enrichment odds (Fisher 2x2 on flags) + label-permutation null
        fa, fb = flags[a], flags[b]
        ct = np.array([[int((fa & fb).sum()), int((fa & ~fb).sum())],
                       [int((~fa & fb).sum()), int((~fa & ~fb).sum())]])
        try:
            orr, pf = fisher_exact(ct, alternative="greater")
        except Exception:
            orr, pf = float("nan"), float("nan")
        # permutation p on overlap count
        obs_ov = int((fa & fb).sum())
        null_ov = np.array([int((fa & rng.permutation(fb)).sum()) for _ in range(n_perm)])
        odds[key] = float(orr)
        odds_p[key] = float((np.sum(null_ov >= obs_ov) + 1) / (n_perm + 1))
        union = int((fa | fb).sum())
        jacc[key] = float(obs_ov / union) if union else float("nan")
    res["spearman"] = spear; res["spearman_p"] = spear_p
    res["auroc"] = auroc_ab; res["odds_ratio"] = odds
    res["odds_perm_p"] = odds_p; res["jaccard"] = jacc

    # ---- three-way convergence (ovrlpy x tracer x rctd[entropy]) ----
    method_pairs = [("ovrlpy", "tracer"), ("ovrlpy", "rctd"), ("tracer", "rctd")]
    pair_rho = []
    for ma, mb in method_pairs:
        a, b = METHOD_METRIC[ma], METHOD_METRIC[mb]
        k = f"{a}__{b}" if f"{a}__{b}" in spear else f"{b}__{a}"
        pair_rho.append(spear[k])
    fo, ft, fr_ = flags[METHOD_METRIC["ovrlpy"]], flags[METHOD_METRIC["tracer"]], flags[METHOD_METRIC["rctd"]]
    triple = int((fo & ft & fr_).sum())
    exp_triple = float(fo.mean() * ft.mean() * fr_.mean() * n)
    null_triple = np.array([int((fo & rng.permutation(ft) & rng.permutation(fr_)).sum())
                            for _ in range(n_perm)])
    res["convergence"] = {
        "min_pairwise_spearman": float(np.min(pair_rho)),
        "mean_pairwise_spearman": float(np.mean(pair_rho)),
        "all_pairs_positive": bool(np.all(np.array(pair_rho) > 0)),
        "mean_method_moran": float(np.mean([moran[METHOD_METRIC[m]] for m in ("ovrlpy", "tracer", "rctd")])),
        "all_methods_moran_sig": bool(np.all([moran_p[METHOD_METRIC[m]] < 0.05 for m in ("ovrlpy", "tracer", "rctd")])),
        "triple_flagged": triple,
        "triple_expected": exp_triple,
        "triple_enrichment": float(triple / exp_triple) if exp_triple > 0 else float("nan"),
        "triple_perm_p": float((np.sum(null_triple >= triple) + 1) / (n_perm + 1)),
    }
    return res


def convergence_score(b: dict) -> float:
    """Higher = stronger 3-way convergence. Requires all pairwise rho>0."""
    if b.get("insufficient") or "convergence" not in b:
        return -1e9
    c = b["convergence"]
    if not c["all_pairs_positive"]:
        return -1.0 + c["min_pairwise_spearman"]
    # reward min pairwise agreement + structure + triple enrichment
    return (c["min_pairwise_spearman"]
            + 0.25 * c["mean_method_moran"]
            + 0.10 * np.tanh(max(c["triple_enrichment"], 0.0)))


# ---------------------------------------------------------------------------
#  Candidate ROIs
# ---------------------------------------------------------------------------
def candidate_rois(top_k: int) -> list[dict]:
    rois: list[dict] = []
    rank = pd.read_csv(ROI_RANKING, sep="\t").sort_values("rank").head(top_k)
    for _, r in rank.iterrows():
        rois.append({"name": f"cand{int(r['rank']):02d}",
                     "xmin": float(r["xmin"]), "xmax": float(r["xmax"]),
                     "ymin": float(r["ymin"]), "ymax": float(r["ymax"]),
                     "source": "rctd_candidate_ranking"})
    # A/B/C representative ROIs (ovrlpy x TRACER categories) at 800 um.
    reps = {"repA": (6700.6, 7134.8, "A_ovrlpy+_tracer+"),
            "repB": (8784.3, 8390.2, "B_ovrlpy-_tracer+"),
            "repC": (2935.5, 6100.4, "C_ovrlpy+_tracer-")}
    for name, (cx, cy, cat) in reps.items():
        rois.append({"name": name, "xmin": cx - 400, "xmax": cx + 400,
                     "ymin": cy - 400, "ymax": cy + 400, "source": cat})
    return rois


# ---------------------------------------------------------------------------
#  Dark Nature-style concordance panel
# ---------------------------------------------------------------------------
_METRIC_LABEL = {
    "ovrlpy_problem": "ovrlpy  (1 - VSI)",
    "tracer_relconflict": "TRACER  relative conflict",
    "rctd_entropy": "RCTD  norm. entropy",
    "rctd_lowmaxweight": "RCTD  (1 - max weight)",
}


def render_panel(cells: pd.DataFrame, b: dict, roi: dict, out_base: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    cc = cells.dropna(subset=METRICS).reset_index(drop=True)
    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(13, 7.4), dpi=200)
        gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.92],
                      hspace=0.28, wspace=0.22)
        # row 0: per-cell spatial maps of the 4 metrics
        for i, m in enumerate(METRICS):
            ax = fig.add_subplot(gs[0, i])
            v = cc[m].to_numpy(float)
            lo, hi = np.nanpercentile(v, [2, 98])
            sc = ax.scatter(cc["cx"], cc["cy"], c=v, s=7, cmap="magma",
                            vmin=lo, vmax=hi, linewidths=0, rasterized=True)
            ax.set_aspect("equal"); ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(_METRIC_LABEL[m] + f"\nMoran I={b['moran'][m]:.2f} (p={b['moran_p'][m]:.3f})",
                         fontsize=8.5, color="white")
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6, colors="white")
        # row 1, col 0: triple-flagged convergence map
        axc = fig.add_subplot(gs[1, 0])
        fo = _flags(cc["ovrlpy_problem"].to_numpy(float), FLAG_Q)
        ft = _flags(cc["tracer_relconflict"].to_numpy(float), FLAG_Q)
        frc = _flags(cc["rctd_entropy"].to_numpy(float), FLAG_Q)
        nflag = fo.astype(int) + ft.astype(int) + frc.astype(int)
        axc.scatter(cc["cx"], cc["cy"], c="#222233", s=6, linewidths=0, rasterized=True)
        triple = nflag == 3
        two = nflag == 2
        axc.scatter(cc["cx"][two], cc["cy"][two], c="#FFB000", s=10, linewidths=0,
                    label="2/3 methods", rasterized=True)
        axc.scatter(cc["cx"][triple], cc["cy"][triple], c="#00E5FF", s=16,
                    edgecolors="white", linewidths=0.3, label="all 3 methods")
        axc.set_aspect("equal"); axc.invert_yaxis(); axc.set_xticks([]); axc.set_yticks([])
        axc.set_title(f"convergent flags\ntriple n={int(triple.sum())} "
                      f"({b['convergence']['triple_enrichment']:.1f}x, p={b['convergence']['triple_perm_p']:.3f})",
                      fontsize=8.5, color="white")
        axc.legend(loc="lower right", fontsize=6, facecolor="black", framealpha=0.6)
        # row 1, col 1: pairwise Spearman heatmap
        axh = fig.add_subplot(gs[1, 1])
        Msp = np.full((4, 4), np.nan)
        for i, a in enumerate(METRICS):
            Msp[i, i] = 1.0
            for j, bb in enumerate(METRICS):
                if j <= i:
                    continue
                k = f"{a}__{bb}"
                Msp[i, j] = Msp[j, i] = b["spearman"][k]
        im = axh.imshow(Msp, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
        axh.set_xticks(range(4)); axh.set_yticks(range(4))
        short = ["ovrlpy", "TRACER", "RCTD-ent", "RCTD-lmw"]
        axh.set_xticklabels(short, rotation=45, ha="right", fontsize=6.5)
        axh.set_yticklabels(short, fontsize=6.5)
        for i in range(4):
            for j in range(4):
                if np.isfinite(Msp[i, j]):
                    axh.text(j, i, f"{Msp[i, j]:.2f}", ha="center", va="center",
                             fontsize=6.5, color="black" if abs(Msp[i, j]) < 0.4 else "white")
        axh.set_title("pairwise Spearman", fontsize=8.5, color="white")
        fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6, colors="white")
        # row 1, col 2: enrichment odds heatmap
        axo = fig.add_subplot(gs[1, 2])
        Mo = np.full((4, 4), np.nan)
        for i, a in enumerate(METRICS):
            for j, bb in enumerate(METRICS):
                if j <= i:
                    continue
                k = f"{a}__{bb}"
                Mo[i, j] = Mo[j, i] = b["odds_ratio"][k]
        im2 = axo.imshow(np.log2(Mo), cmap="magma", vmin=0, vmax=4)
        axo.set_xticks(range(4)); axo.set_yticks(range(4))
        axo.set_xticklabels(short, rotation=45, ha="right", fontsize=6.5)
        axo.set_yticklabels(short, fontsize=6.5)
        for i in range(4):
            for j in range(4):
                if np.isfinite(Mo[i, j]):
                    axo.text(j, i, f"{Mo[i, j]:.1f}", ha="center", va="center",
                             fontsize=6.5, color="white")
        axo.set_title("flag enrichment (odds ratio)", fontsize=8.5, color="white")
        fig.colorbar(im2, ax=axo, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6, colors="white")
        # row 1, col 3: text summary
        axt = fig.add_subplot(gs[1, 3]); axt.axis("off")
        c = b["convergence"]
        lines = [
            f"ROI {roi['name']}  ({roi.get('source', '')})",
            f"  x:[{roi['xmin']:.0f},{roi['xmax']:.0f}] y:[{roi['ymin']:.0f},{roi['ymax']:.0f}]",
            f"  n cells = {b['n_cells_all_methods']}",
            "",
            "3-way convergence (ovrlpy x TRACER x RCTD)",
            f"  min pairwise rho   = {c['min_pairwise_spearman']:.3f}",
            f"  mean pairwise rho  = {c['mean_pairwise_spearman']:.3f}",
            f"  all pairs positive = {c['all_pairs_positive']}",
            f"  mean Moran's I     = {c['mean_method_moran']:.3f}",
            f"  all Moran sig.     = {c['all_methods_moran_sig']}",
            f"  triple-flag enrich = {c['triple_enrichment']:.2f}x",
            f"  triple-flag perm p = {c['triple_perm_p']:.3f}",
        ]
        axt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.0,
                 family="monospace", color="white", transform=axt.transAxes)
        fig.suptitle("Atera cervical cancer — three-method problem-region concordance",
                     fontsize=12, color="white", y=0.99)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(f"{out_base}.{ext}", bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        plt.close(fig)


# ---------------------------------------------------------------------------
#  Flatten a battery dict to a one-row table
# ---------------------------------------------------------------------------
def battery_row(name: str, roi: dict, b: dict) -> dict:
    row = {"roi": name, "source": roi.get("source", ""),
           "xmin": roi["xmin"], "xmax": roi["xmax"],
           "ymin": roi["ymin"], "ymax": roi["ymax"],
           "n_cells": b.get("n_cells_all_methods", 0),
           "convergence_score": convergence_score(b)}
    if b.get("insufficient"):
        return row
    for m in METRICS:
        row[f"moran_{m}"] = b["moran"][m]
        row[f"moran_p_{m}"] = b["moran_p"][m]
    for k, v in b["spearman"].items():
        row[f"spearman_{k}"] = v
    for k, v in b["auroc"].items():
        row[f"auroc_{k}"] = v
    for k, v in b["odds_ratio"].items():
        row[f"odds_{k}"] = v
    for k, v in b["jaccard"].items():
        row[f"jaccard_{k}"] = v
    c = b["convergence"]
    for k, v in c.items():
        row[f"conv_{k}"] = v
    return row


# ---------------------------------------------------------------------------
#  Parameter sweeps on the winning ROI
# ---------------------------------------------------------------------------
def sweep_tracer(roi: dict, res: Resources, cache: Path, tabs: Path,
                 log: logging.Logger, n_workers: int, seed: int) -> None:
    df = crop_roi(roi, cache)
    cen = cell_centroids(df)
    ov = score_ovrlpy(df, res.panel, OVRLPY_PARAMS, n_workers, seed)
    rc = score_rctd(df, res, n_iter=RCTD_PARAMS["n_iter"], chunk=RCTD_PARAMS["chunk"])
    rows = []
    for tau in (0.02, 0.05, 0.10):
        for pct in (70.0, 80.0, 90.0):
            tr = score_tracer(df, res, tau=tau, conflict_percentile=pct,
                              min_transcripts=TRACER_PARAMS["min_transcripts"])
            cells = cen.merge(ov, on="cell_id").merge(rc, on="cell_id").merge(tr, on="cell_id")
            b = battery(cells, rng=np.random.default_rng(seed))
            r = {"tau": tau, "conflict_percentile": pct, **battery_row("tracer_sweep", roi, b)}
            rows.append(r)
            log.info("[tracer-sweep] tau=%.2f pct=%.0f convScore=%.3f minRho=%.3f",
                     tau, pct, r["convergence_score"], b.get("convergence", {}).get("min_pairwise_spearman", float("nan")))
    pd.DataFrame(rows).to_csv(tabs / "sweep_tracer.tsv", sep="\t", index=False)


def sweep_rctd(roi: dict, cache: Path, tabs: Path, log: logging.Logger,
               n_workers: int, seed: int) -> None:
    df = crop_roi(roi, cache)
    cen = cell_centroids(df)
    rows = []
    for lineage_col in ("cell_type_coarse", "cell_type_fine"):
        res = Resources(log, lineage_col=lineage_col)
        ov = score_ovrlpy(df, res.panel, OVRLPY_PARAMS, n_workers, seed)
        tr = score_tracer(df, res, tau=TRACER_PARAMS["tau"],
                          conflict_percentile=TRACER_PARAMS["conflict_percentile"],
                          min_transcripts=TRACER_PARAMS["min_transcripts"])
        for n_iter in (15, 30, 60):
            rc = score_rctd(df, res, n_iter=n_iter, chunk=RCTD_PARAMS["chunk"])
            cells = cen.merge(ov, on="cell_id").merge(rc, on="cell_id").merge(tr, on="cell_id")
            b = battery(cells, rng=np.random.default_rng(seed))
            r = {"lineage_col": lineage_col, "n_iter": n_iter,
                 "med_rctd_entropy": float(np.nanmedian(cells["rctd_entropy"])),
                 "frac_low_maxweight": float(np.nanmean(cells["rctd_lowmaxweight"] > 0.5)),
                 **battery_row("rctd_sweep", roi, b)}
            rows.append(r)
            log.info("[rctd-sweep] lineage=%s n_iter=%d convScore=%.3f",
                     lineage_col, n_iter, r["convergence_score"])
    pd.DataFrame(rows).to_csv(tabs / "sweep_rctd.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("score", "sweep", "all"), default="all")
    ap.add_argument("--top-k-rois", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tabs = OUTDIR / "tables"; tabs.mkdir(exist_ok=True)
    figs = OUTDIR / "figures"; figs.mkdir(exist_ok=True)
    cells_dir = OUTDIR / "per_roi_cells"; cells_dir.mkdir(exist_ok=True)
    cache = OUTDIR / "roi_cache"
    log = _log()
    res = Resources(log)

    rois = candidate_rois(args.top_k_rois)
    log.info("Scoring %d candidate ROIs", len(rois))

    summary_rows, batteries = [], {}
    if args.stage in ("score", "all"):
        for roi in rois:
            t0 = time.time()
            cells = score_roi(roi, res, cache, n_workers=args.n_workers, seed=args.seed)
            cells.to_csv(cells_dir / f"{roi['name']}_cells.tsv.gz", sep="\t", index=False)
            b = battery(cells, rng=np.random.default_rng(args.seed))
            batteries[roi["name"]] = b
            summary_rows.append(battery_row(roi["name"], roi, b))
            cv = b.get("convergence", {})
            log.info("[%s] n=%d convScore=%.3f minRho=%.3f triple=%.1fx (%.1fs)",
                     roi["name"], b.get("n_cells_all_methods", 0),
                     convergence_score(b), cv.get("min_pairwise_spearman", float("nan")),
                     cv.get("triple_enrichment", float("nan")), time.time() - t0)
        summ = pd.DataFrame(summary_rows).sort_values("convergence_score", ascending=False)
        summ.to_csv(tabs / "roi_convergence_summary.tsv", sep="\t", index=False)
        (OUTDIR / "all_batteries.json").write_text(json.dumps(batteries, indent=2, default=float))
        # winning ROI
        best_name = summ.iloc[0]["roi"]
        best_roi = next(r for r in rois if r["name"] == best_name)
        best_b = batteries[best_name]
        best_cells = pd.read_csv(cells_dir / f"{best_name}_cells.tsv.gz", sep="\t")
        (OUTDIR / "best_convergent_roi.json").write_text(json.dumps({
            "roi": best_roi, "convergence": best_b["convergence"],
            "ovrlpy_params": OVRLPY_PARAMS, "tracer_params": TRACER_PARAMS,
            "rctd_params": RCTD_PARAMS, "flag_quantile": FLAG_Q,
            "selection_rule": ("max convergence_score = min pairwise Spearman "
                               "(all >0) + 0.25*mean Moran's I + 0.10*tanh(triple enrichment)"),
        }, indent=2, default=float))
        render_panel(best_cells, best_b, best_roi, figs / "concordance_panel_best_roi")
        # per-metric maps for ALL ROIs (supplement)
        smaps = figs / "per_roi_metric_maps"; smaps.mkdir(exist_ok=True)
        for roi in rois:
            cdf = pd.read_csv(cells_dir / f"{roi['name']}_cells.tsv.gz", sep="\t")
            _save_metric_maps(cdf, smaps / f"{roi['name']}.png", roi["name"])
        log.info("WINNER: %s | %s", best_name, summ.iloc[0].to_dict())

    if args.stage in ("sweep", "all"):
        # sweep on the winning ROI (fall back to repA if score stage skipped)
        if (tabs / "roi_convergence_summary.tsv").exists():
            summ = pd.read_csv(tabs / "roi_convergence_summary.tsv", sep="\t")
            best_name = summ.iloc[0]["roi"]
        else:
            best_name = "repA"
        best_roi = next(r for r in rois if r["name"] == best_name)
        log.info("Parameter sweeps on winning ROI: %s", best_name)
        sweep_tracer(best_roi, res, cache, tabs, log, args.n_workers, args.seed)
        sweep_rctd(best_roi, cache, tabs, log, args.n_workers, args.seed)
    return 0


def _save_metric_maps(cells: pd.DataFrame, path: Path, name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cc = cells.dropna(subset=METRICS)
    if len(cc) < 20:
        return
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), dpi=150)
        for ax, m in zip(axes, METRICS):
            v = cc[m].to_numpy(float)
            lo, hi = np.nanpercentile(v, [2, 98])
            sc = ax.scatter(cc["cx"], cc["cy"], c=v, s=6, cmap="magma",
                            vmin=lo, vmax=hi, linewidths=0, rasterized=True)
            ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(_METRIC_LABEL[m], fontsize=8, color="white")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5, colors="white")
        fig.suptitle(f"{name}", fontsize=10, color="white")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
