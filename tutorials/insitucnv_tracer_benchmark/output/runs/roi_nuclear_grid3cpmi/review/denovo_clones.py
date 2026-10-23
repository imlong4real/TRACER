"""De novo clone discovery in both arms, then cross-arm comparison.

Recipe = the published ovarian stage 2: restrict to the tumour compartment,
rebuild the CNV PCA + kNN graph ON THE SUBSET, Leiden, drop clones under 0.1%.
No truth labels are used to define clones; truth is only read afterwards.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import scanpy as sc
sc.settings.verbosity = 0
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns

OUT = C.Path(__file__).parent
RES = 0.2
MIN_PCT = 0.1
GRID_BP = 1e6


def call_clones(adata, resolution=RES, min_pct=MIN_PCT):
    import infercnvpy as cnv
    m = (adata.obs["compartment"].astype(str) == "tumor").to_numpy()
    sub = adata[m].copy()                       # copy => pca/graph rebuilt on the subset
    cnv.tl.pca(sub)
    cnv.pp.neighbors(sub)
    cnv.tl.leiden(sub, resolution=resolution, key_added="clone")
    lab = sub.obs["clone"].astype(str)
    pct = lab.value_counts(normalize=True) * 100
    keep = pct[pct > min_pct].index.astype(str).tolist()
    lab = lab.where(lab.isin(keep), "filtered_small_clone")
    sub.obs["clone"] = lab.to_numpy()
    return sub, sorted(c for c in lab.unique() if c != "filtered_small_clone")


def genome_grid(sub, wins, ext, offsets, total):
    x = np.array([offsets[c] + (mid - ext.loc[c, "lo"])
                  for c, mid in zip(wins["chromosome"], wins["bp_mid"])], float)
    g = np.arange(0, total, GRID_BP)
    return x, g


def clone_profiles(sub, clones, x, g):
    X = C.dense_cnv(sub)
    lab = sub.obs["clone"].astype(str).to_numpy()
    out = {}
    for c in clones:
        v = X[lab == c].mean(axis=0)
        out[c] = np.interp(g, x, v)
    return out


def project_to_cells(arm, sub, min_frac=0.5):
    """Give each 10X cell_id the clone of the entity owning most of its transcripts."""
    spec = C.ARMS[arm]
    ecol = spec["entity_col"]
    cols = list(dict.fromkeys([ecol, "cell_id"]))
    df = standardize_transcript_columns(pd.read_parquet(spec["transcripts"], columns=cols))
    df[ecol] = df[ecol].astype(str); df["cell_id"] = df["cell_id"].astype(str)
    if ecol == "cell_id":
        df["_entity"] = df["cell_id"]
    else:
        df["_entity"] = df[ecol]
    df = df.loc[~df["_entity"].isin(UNASSIGNED_TOKENS) & ~df["cell_id"].isin(UNASSIGNED_TOKENS)]
    clone = sub.obs["clone"].astype(str)
    df["_clone"] = df["_entity"].map(clone)
    df = df.dropna(subset=["_clone"])
    vc = df.groupby(["cell_id", "_clone"], observed=True).size().rename("n").reset_index()
    tot = df.groupby("cell_id").size().rename("tot")
    vc = vc.sort_values(["n", "_clone"], ascending=[True, False])
    top = vc.groupby("cell_id").tail(1).set_index("cell_id")
    frac = top["n"] / tot.reindex(top.index)
    return top.loc[frac >= min_frac, "_clone"]


def main():
    from step1_heatmap import genome_axis
    data = {a: C.load_arm(a) for a in ("raw", "tracer_whole")}
    wins_by_arm = {a: w for a, (_, w) in data.items()}
    ext, offsets, total = genome_axis(wins_by_arm)

    subs, clones, profs, xs = {}, {}, {}, {}
    print("=== de novo clone calling (tumour compartment, Leiden res=0.2, >0.1% filter) ===")
    for arm, (adata, wins) in data.items():
        sub, cl = call_clones(adata)
        subs[arm], clones[arm] = sub, cl
        x, g = genome_grid(sub, wins, ext, offsets, total)
        xs[arm] = x
        profs[arm] = clone_profiles(sub, cl, x, g)
        vc = sub.obs["clone"].astype(str).value_counts()
        print(f"\n{arm}: {sub.n_obs} tumour entities -> {len(cl)} clones kept "
              f"({int((sub.obs['clone'].astype(str)=='filtered_small_clone').sum())} entities filtered)")
        print("  sizes:", {k: int(v) for k, v in vc.items()})
        # truth composition per de novo clone (read AFTER calling)
        ct = pd.crosstab(sub.obs["clone"].astype(str), sub.obs["truth_clone"].astype(str))
        print("  vs full-slide truth clone:\n" + ct.to_string())

    # ---- (a) do the arms recover the same clones? match by profile correlation
    A, B = clones["raw"], clones["tracer_whole"]
    M = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            u = profs["raw"][a] - profs["raw"][a].mean()
            v = profs["tracer_whole"][b] - profs["tracer_whole"][b].mean()
            M[i, j] = u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    corr = pd.DataFrame(M, index=[f"raw_{a}" for a in A], columns=[f"tr_{b}" for b in B])
    print("\n=== (a) cross-arm clone profile correlation (1-Mb genomic grid) ===")
    print(corr.round(3).to_string())
    ri, ci = linear_sum_assignment(-M)
    pairs = [(A[i], B[j], M[i, j]) for i, j in zip(ri, ci)]
    print("\nbest 1-to-1 matching (Hungarian):")
    for a, b, r in pairs:
        print(f"  raw clone {a} (n={int((subs['raw'].obs['clone'].astype(str)==a).sum()):4d})  <->  "
              f"tracer clone {b} (n={int((subs['tracer_whole'].obs['clone'].astype(str)==b).sum()):4d})   r={r:+.3f}")
    print(f"  mean matched r = {np.mean([p[2] for p in pairs]):+.3f}")

    # ---- (b) same cells grouped together? project onto shared 10X cell ids
    proj = {a: project_to_cells(a, subs[a]) for a in subs}
    common = proj["raw"].index.intersection(proj["tracer_whole"].index)
    r_lab = proj["raw"].reindex(common).astype(str)
    t_lab = proj["tracer_whole"].reindex(common).astype(str)
    ok = (r_lab != "filtered_small_clone") & (t_lab != "filtered_small_clone")
    print(f"\n=== (b) agreement on the SAME 10X cells ===")
    print(f"  cells assigned a clone in raw: {len(proj['raw'])}, in tracer: {len(proj['tracer_whole'])}, "
          f"common: {len(common)}, both non-filtered: {int(ok.sum())}")
    print(f"  ARI = {adjusted_rand_score(r_lab[ok], t_lab[ok]):.3f}   "
          f"AMI = {adjusted_mutual_info_score(r_lab[ok], t_lab[ok]):.3f}")
    ct = pd.crosstab(r_lab[ok], t_lab[ok])
    print("  contingency (rows = raw clone, cols = tracer clone):\n" + ct.to_string())

    # ---- spatial centroids per clone
    print("\n=== clone spatial centroids (um) and spread ===")
    cen = {}
    for arm in subs:
        s = subs[arm]; xy = s.obsm["spatial"]; lab = s.obs["clone"].astype(str).to_numpy()
        for c in clones[arm]:
            m = lab == c
            cen[(arm, c)] = dict(n=int(m.sum()), x=xy[m, 0].mean(), y=xy[m, 1].mean(),
                                 sd=float(np.sqrt(((xy[m] - xy[m].mean(0)) ** 2).sum(1).mean())))
    cendf = pd.DataFrame(cen).T
    print(cendf.round(1).to_string())
    print("\nmatched-pair centroid distance:")
    for a, b, r in pairs:
        ca, cb = cen[("raw", a)], cen[("tracer_whole", b)]
        d = np.hypot(ca["x"] - cb["x"], ca["y"] - cb["y"])
        print(f"  raw {a} <-> tracer {b}: {d:6.1f} um apart  (clone spread ~{ca['sd']:.0f}/{cb['sd']:.0f} um)  r={r:+.3f}")

    np.save(OUT / "_denovo_grid.npy", np.arange(0, total, GRID_BP))
    import pickle
    with open(OUT / "_denovo.pkl", "wb") as fh:
        pickle.dump({"clones": clones, "profs": profs, "pairs": pairs,
                     "obs": {a: subs[a].obs[["clone", "truth_clone", "n_genes"]] for a in subs},
                     "xy": {a: subs[a].obsm["spatial"] for a in subs},
                     "ext": ext, "offsets": offsets, "total": total, "corr": corr}, fh)
    corr.to_csv(OUT / "denovo_clone_crossarm_corr.csv")
    cendf.to_csv(OUT / "denovo_clone_centroids.csv")
    print("\nwrote denovo_clone_*.csv")


if __name__ == "__main__":
    main()
