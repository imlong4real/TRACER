"""Sweep the smoothing-graph width and score each choice on four diagnostics.

Runs the arm pipeline in-process (no h5ad written) for each (graph, k).
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import scanpy as sc
sc.settings.verbosity = 0

import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
import run_insitucnv_arm as R
from common import standardize_transcript_columns


def build(arm):
    spec = C.ARMS[arm]
    df = standardize_transcript_columns(pd.read_parquet(spec["transcripts"]))
    ecol = R.resolve_entity_col(df, arm)
    df = R.filter_arm(df, arm, ecol)
    ann = pd.read_csv(C.RUN / f"annotations_{arm}.csv")
    a = R.build_adata(df, ecol, ann, 20, 10)
    a = R.add_gene_positions(a, C.BENCH / "data/gene_positions_grch38.tsv")
    R.assign_compartment(a, {"t_cell","b_cell","myeloid","fibroblast","endothelial"}, {"epithelial"})
    truth = C.truth_by_entity(arm, a.obs_names)
    for c in truth.columns:
        a.obs[c] = truth[c].to_numpy()
    return a


def score(a, wins):
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    X = C.dense_cnv(a)
    truth = a.obs["truth_clone"].astype(str).to_numpy()
    comp = a.obs["compartment"].astype(str).to_numpy()
    is4 = truth == "4"; is0 = truth == "0"
    t4 = is4 & (comp == "tumor"); t0 = is0 & (comp == "tumor")
    p4 = X[t4].mean(axis=0); p0 = X[t0].mean(axis=0); pr = X[comp == "reference"].mean(axis=0)

    def cc(u, v):
        u = u - u.mean(); v = v - v.mean()
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

    # spatial bleed: zero-epithelial reference entities vs local clone-4 density
    xy = a.obsm["spatial"]
    _, idx = cKDTree(xy).query(xy, k=min(201, a.n_obs))
    frac4 = is4[idx[:, 1:]].mean(axis=1)
    Xc = X - X.mean(axis=1, keepdims=True)
    p4c = p4 - p4.mean()
    sim = (Xc @ p4c) / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(p4c) + 1e-12)
    m = comp == "reference"
    bleed = spearmanr(frac4[m], sim[m]).statistic

    # clone discriminability: per-entity sim to clone4 minus sim to clone0, AUC-style
    p0c = p0 - p0.mean()
    s0 = (Xc @ p0c) / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(p0c) + 1e-12)
    d = sim - s0
    from itertools import product
    x, y = d[t4], d[t0]
    auc = float(np.mean([a_ > b_ for a_, b_ in product(x[:300], y[:300])])) if len(x) and len(y) else np.nan
    return {
        "clone0_vs_clone4_profile_r": cc(p0, p4),
        "clone4_vs_reference_r": cc(p4, pr),
        "clone0_vs_reference_r": cc(p0, pr),
        "reference_RMS": float(np.sqrt((pr ** 2).mean())),
        "clone4_RMS": float(np.sqrt((p4 ** 2).mean())),
        "clone0_RMS": float(np.sqrt((p0 ** 2).mean())),
        "clone4_over_ref_RMS": float(np.sqrt((p4 ** 2).mean()) / (np.sqrt((pr ** 2).mean()) + 1e-12)),
        "spatial_bleed_spearman": float(bleed),
        "clone_AUC": auc,
        "pct_zero": float((X == 0).mean() * 100),
    }


def main():
    combos = [("expression", k) for k in (15, 30, 60, 100, 200)] + [("spatial", 200)]
    rows = []
    for arm in ("raw", "tracer_whole"):
        base = build(arm)
        for graph, k in combos:
            a = base.copy()
            R.run_infercnv(a, 60, 0.4, k, smoothing_graph=graph,
                           expression_neighbors=k, neighbor_n_pcs=50)
            wins = C.window_intervals(a.var)
            s = score(a, wins)
            s.update(arm=arm, graph=graph, k=k)
            rows.append(s)
            print(f"[{arm:13s} {graph:10s} k={k:3d}] "
                  f"clone0~clone4 r={s['clone0_vs_clone4_profile_r']:+.3f}  "
                  f"AUC={s['clone_AUC']:.3f}  bleed={s['spatial_bleed_spearman']:+.3f}  "
                  f"c4/ref RMS={s['clone4_over_ref_RMS']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    out = C.Path(__file__).parent / "sweep_smoothing.csv"
    df.to_csv(out, index=False)
    print("\nwrote", out)
    cols = ["arm","graph","k","clone0_vs_clone4_profile_r","clone_AUC","spatial_bleed_spearman",
            "clone4_over_ref_RMS","reference_RMS","clone4_RMS","clone0_RMS"]
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
