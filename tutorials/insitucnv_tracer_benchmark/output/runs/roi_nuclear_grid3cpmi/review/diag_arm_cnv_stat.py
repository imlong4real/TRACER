"""Diagnostic: which reference handling and arm summary can see a CNV in a sparse cell?

Run 1 found two problems: (i) with an empirical p floored at 1/1001, BH across 38 arms
can never reach q < 0.05 for a single arm; (ii) bounds across fibroblast / myeloid / T
zero most genes (the pooled tumour profile had a median of exactly 0 on 37/38 arms).
Run 2 uses z-based p from the random-group null (mean/SD) and crosses
  reference  {envelope (bounds across types), pooled (one mean over all normal cells)}
  summary    {weighted median, 10%-trimmed weighted mean, weighted mean}
with identical folds, references and random gene groups (same seed). Per config:
  A. calibration   unplanted held-out normals at fixed totals: share of arm p < 0.05
                   (~5% if calibrated) and false calls per cell
  B. sensitivity   planted 1.5x gain on 1q / 8q / 20q in the same cells
  C. karyotype     pooled profile of all tumour cells vs InSituCNV's own arm-level tumour
                   profile (X_cnv windows -> arms by bp midpoint), Pearson r across arms
"""
from __future__ import annotations
import numpy as np, pandas as pd, anndata as ad
import cnv_review_common as C
from arm_cnv import ArmCNV, drop_to, CEN

OUT = C.Path(__file__).parent
DEPTHS = [800, 1500, 2500]
PLANT = ["1q", "8q", "20q"]
N_CELLS = 80
CONFIGS = [(r, s) for r in ("envelope", "pooled") for s in ("wmedian", "wtrim", "wmean")]


def xcnv_arm_profile(a, cells):
    """InSituCNV's X_cnv averaged over `cells`, windows mapped to arms by bp midpoint."""
    wins = C.window_intervals(a.var)
    X = C.dense_cnv(a)
    assert X.shape[1] == len(wins)
    cen = pd.read_csv(CEN, sep="\t").set_index("chromosome")
    cm = wins["chromosome"].map((cen["cen_start"] + cen["cen_end"]) / 2)
    arm = wins["chromosome"].str.replace("chr", "", regex=False) + np.where(wins["bp_mid"] < cm, "p", "q")
    prof = pd.Series(X[cells].mean(0), index=arm.values)
    return prof.groupby(level=0).mean()


def main():
    a = ad.read_h5ad(C.RUN / "insitucnv_raw/adata_cnv.h5ad")
    tum = (a.obs["compartment"] == "tumor").to_numpy()
    xprof = xcnv_arm_profile(a, tum)
    rows, karyo = [], {}
    for ref, stat in CONFIGS:
        rng = np.random.default_rng(0)
        m = ArmCNV(a, n_null=1000, stat=stat, reference=ref, pmode="z")
        tag = f"{ref}/{stat}"

        r = m.score_matrix(np.asarray(m.Y[tum].sum(0)))
        k = pd.Series(r["D"][0], index=m.arms)
        common = k.index.intersection(xprof.index)
        karyo[tag] = pd.DataFrame({"effect": k, "q": r["q"][0], "call": r["call"][0]})
        r_x = np.corrcoef(k[common], xprof[common])[0, 1]

        tot = np.asarray(m.Y.sum(1)).ravel()
        r_ = np.random.default_rng(1)                       # same cells for every config
        for d in DEPTHS:
            cand = np.where(m.is_ref & (tot >= d))[0]
            cells = r_.choice(cand, min(N_CELLS, len(cand)), replace=False)
            base = np.vstack([drop_to(m.Y[i].toarray().ravel(), d, rng) for i in cells])
            fo = m.fold[cells]
            rb = m.score_matrix(base, folds=fo)
            row = dict(config=tag, depth=d, n=len(cells), karyo_r_vs_insitucnv=r_x,
                       normal_frac_p05=(rb["p"] < 0.05).mean(),
                       normal_false_calls=rb["n_called"].mean())
            for arm in PLANT:
                ai = m.arms.index(arm); g = m.arm_idx[ai]
                Yp = base.copy(); Yp[:, g] = Yp[:, g] + rng.binomial(Yp[:, g], 0.5)
                rp = m.score_matrix(Yp, folds=fo)
                row[f"sens_{arm}"] = (rp["call"][:, ai] == 1).mean()
                row[f"p05_{arm}"] = (rp["p_gain"][:, ai] < 0.05).mean()
            rows.append(row)
        print(f"done {tag}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "diag_arm_cnv_stat2.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + res.round(3).to_string(index=False))
    K = pd.concat({t: v["effect"] for t, v in karyo.items()}, axis=1)
    K["insitucnv_xcnv"] = xprof.reindex(K.index)
    K.to_csv(OUT / "diag_arm_cnv_karyotype2.csv")
    print("\npooled tumour profile, arm effect per config vs InSituCNV X_cnv arm mean:")
    print(K.round(3).to_string())
    print("\narms CALLED in the pooled tumour profile (q<0.05):")
    for t, v in karyo.items():
        c = v.loc[v["call"] != 0, "call"]
        print(f"  {t:18s} gains {list(c[c > 0].index)}  losses {list(c[c < 0].index)}")


if __name__ == "__main__":
    main()
