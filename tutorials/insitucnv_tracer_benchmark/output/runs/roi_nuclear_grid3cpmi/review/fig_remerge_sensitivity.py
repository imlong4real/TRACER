"""Do the switchers survive TRACER's Phase-1 remerge ΔC floor at -0.5?

Three TRACER runs of the same ROI, same container (tracer_insitucnv.sif), same
panel, seed 1:
  original -0.2   roi_nuclear_grid3cpmi                    (2026-09-17)
  rerun    -0.2   roi_nuclear_grid3cpmi_remerge020_rerun   reproducibility control
  -0.5            roi_nuclear_grid3cpmi_remerge050         phase1.maha_remerge_delta_c_floor
The rerun gives the run-to-run noise floor; the -0.5 change is read against it.

Phase-1 remerge (src/tracer/phase1_rescue.py:phase1_maha_remerge) unions co-located
entity pairs with floor < dC < 0 and Mahalanobis D <= 1. Lowering the floor lets
more compositionally incompatible but overlapping pairs remerge -- the situation of
a switcher's remnant and its epithelial partial.

Same switcher rule as fig_switch_cnv.id_groups (ID join; Xenium epithelial ->
TRACER class), same per-switcher bookkeeping and the SAME inferCNV call as
fig_immune_switchers.py (imported), one cells+partials run per TRACER run.
Runs whose outputs are not there yet are skipped.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns
from fig_immune_switchers import (tracer_plus_partials_cnv, per_entity_r, MK, CLASSES,
                                  CLAB, REFSET, CACHE)

OUT = C.Path(__file__).parent
RNG = np.random.default_rng(0)
RUNS_DIR = C.BENCH / "output/runs"
RUNS = [("original -0.2", "roi_nuclear_grid3cpmi"),
        ("rerun -0.2", "roi_nuclear_grid3cpmi_remerge020_rerun"),
        ("-0.5", "roi_nuclear_grid3cpmi_remerge050")]
ORIG = RUNS[0][0]
FATES = ["same-class switcher", "other non-epithelial", "epithelial (split undone)",
         "unknown", "no whole-cell entity"]
FCOL = ["#984ea3", "#fdae61", "#1b7837", "#bdbdbd", "#525252"]


def paths(name):
    d = RUNS_DIR / name
    return dict(parquet=d / "tracer/outputs/transcripts_tracer_refined.parquet",
                ann=d / "annotations_tracer_whole.csv",
                cache=CACHE if name == RUNS[0][1] else d / "_partials_run_all.pkl")


def load_run(name, xen):
    p = paths(name)
    df = standardize_transcript_columns(pd.read_parquet(p["parquet"]))
    df["cell_id"] = df["cell_id"].astype(str); df["tracer_id"] = df["tracer_id"].astype(str)
    et = df["_etype"].astype(str)
    whole = set(df.loc[et == "cell", "tracer_id"].unique())
    partials = set(df.loc[et == "partial", "tracer_id"].unique())
    at = pd.read_csv(p["ann"], dtype={"entity_id": str}).set_index("entity_id")["annotation"] \
        .astype(str)
    atw = at.loc[at.index.isin(whole)]
    ids = xen.index.intersection(atw.index)
    x, y = xen.reindex(ids), atw.reindex(ids)
    sw = {c: ids[(x == "epithelial") & (y == c)] for c in CLASSES}
    conc_epi = ids[(x == "epithelial") & (y == "epithelial")]

    # per-switcher bookkeeping, identical rule to fig_immune_switchers
    epi = df["feature_name"].astype(str).isin(MK["epithelial"])
    allsw = set().union(*[set(v) for v in sw.values()])
    sub = df.loc[df["cell_id"].isin(allsw)].assign(epi=epi)
    rows = []
    for cls, cids in sw.items():
        for cid, g in sub.loc[sub["cell_id"].isin(set(cids))].groupby("cell_id"):
            rem = g["tracer_id"] == cid
            ep = g.loc[g["epi"] & g["tracer_id"].isin(partials), "tracer_id"].value_counts()
            rows.append(dict(cls=cls, cell_id=cid, epi_rem=int((g["epi"] & rem).sum()),
                             top_epi_partial=ep.index[0] if len(ep) else None))
    res = pd.DataFrame(rows)

    tp = tracer_plus_partials_cnv(df, ann_path=p["ann"], cache=p["cache"])
    rt = per_entity_r(tp["X"], tp["names"], conc_epi)
    res["r_remnant"] = res["cell_id"].map(rt)
    res["r_partial"] = res["top_epi_partial"].map(lambda v: rt.get(v, np.nan) if v else np.nan)

    comp = df.groupby(et)["tracer_id"].nunique().rename("entities")
    return dict(df=df[["transcript_id", "tracer_id"]], sw=sw, res=res, comp=comp,
                label_of=atw.reindex(xen.index), whole_ann=atw.value_counts())


def fate(label, cls):
    if pd.isna(label):
        return "no whole-cell entity"
    if label == cls:
        return "same-class switcher"
    if label == "epithelial":
        return "epithelial (split undone)"
    if label in REFSET:
        return "other non-epithelial"
    return "unknown"


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    m = a.merge(b, on="transcript_id", suffixes=("_a", "_b"))
    return (adjusted_rand_score(pd.factorize(m["tracer_id_a"])[0], pd.factorize(m["tracer_id_b"])[0]),
            float((m["tracer_id_a"] == m["tracer_id_b"]).mean()), len(m))


def main():
    xen = pd.read_csv(C.RUN / "annotations_raw.csv", dtype={"entity_id": str}) \
        .set_index("entity_id")["annotation"].astype(str)
    avail = []
    for lab, name in RUNS:
        p = paths(name)
        if p["parquet"].exists() and p["ann"].exists():
            avail.append((lab, name))
        else:
            print(f"skip {lab}: missing {'parquet' if not p['parquet'].exists() else 'annotations'}")
    runs = {lab: load_run(name, xen) for lab, name in avail}

    # Xenium CNV is shared by all runs; its baseline is the ORIGINAL run's concordant
    # epithelial set, exactly as in fig_immune_switchers
    xa, _ = C.load_arm("raw")
    o = runs[ORIG]
    conc_x = list(o["label_of"].index[(xen == "epithelial") & (o["label_of"] == "epithelial")])
    rx = per_entity_r(C.dense_cnv(xa), xa.obs_names.astype(str).to_numpy(), conc_x)
    for r in runs.values():
        r["res"]["r_xenium"] = r["res"]["cell_id"].map(rx)

    # ---------------- numbers ----------------
    print("\nswitchers by class:")
    print(pd.DataFrame({lab: {c: len(r["sw"][c]) for c in CLASSES} for lab, r in runs.items()}).to_string())
    print("\nentities by type:")
    print(pd.DataFrame({lab: r["comp"] for lab, r in runs.items()}).to_string())
    print("\nmedian CNV r per run (pooled classes): Xenium / remnant / partial; remnant epi markers")
    for lab, r in runs.items():
        d = r["res"]
        print(f"  {lab:14s} {d.r_xenium.median():+.3f} / {d.r_remnant.median():+.3f} / "
              f"{d.r_partial.median():+.3f}   epi in remnants {int(d.epi_rem.sum())}  "
              f"(cells {int((d.epi_rem > 0).sum())}/{len(d)})")
    if len(runs) == 1:
        print("\nonly the original run is available -> sanity output only, no figure")
        return

    sw0 = runs[ORIG]["sw"]
    fates = {}
    for lab, r in runs.items():
        if lab == ORIG:
            continue
        print(f"\n=== {lab} vs original ===")
        a, eq, n = ari(runs[ORIG]["df"], r["df"])
        print(f"  transcript partition ARI {a:.4f}; identical tracer_id {eq:.1%} of {n:,} tx")
        f = pd.DataFrame([(c, cid, fate(r["label_of"].get(cid), c))
                          for c in CLASSES for cid in sw0[c]], columns=["cls", "cell_id", "fate"])
        fates[lab] = f
        print(pd.crosstab(f["cls"], f["fate"]).reindex(index=CLASSES, columns=FATES,
                                                        fill_value=0).to_string())
        for c in CLASSES:
            new = set(r["sw"][c]) - set(sw0[c])
            print(f"  new {c} switchers not in the original: {len(new)}")
    pd.concat([f.assign(run=lab) for lab, f in fates.items()]).to_csv(
        OUT / "remerge_switcher_fates.csv", index=False)

    # ---------------- figure ----------------
    labs = [lab for lab, _ in RUNS if lab in runs]
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))

    a = axs[0]
    yt, yl, yi = [], [], 0
    for c in CLASSES:
        for lab in fates:
            f = fates[lab].loc[fates[lab]["cls"] == c, "fate"].value_counts(normalize=True)
            left = 0.0
            for k, col in zip(FATES, FCOL):
                v = f.get(k, 0.0)
                a.barh(yi, v, left=left, color=col, edgecolor="w", lw=0.5)
                if v > 0.07:
                    a.text(left + v / 2, yi, f"{100*v:.0f}%", ha="center", va="center",
                           fontsize=8, color="w" if k != "unknown" else "k")
                left += v
            yt.append(yi); yl.append(f"{CLAB[c]}  |  {lab}"); yi += 1
        yi += 0.6
    a.set_yticks(yt); a.set_yticklabels(yl, fontsize=8.5); a.invert_yaxis(); a.set_xlim(0, 1)
    a.set_xlabel("fraction of the ORIGINAL switchers of that class")
    a.legend(handles=[Patch(color=c, label=k) for k, c in zip(FATES, FCOL)], fontsize=7.5,
             frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    a.set_title("(a) what happened to the original switchers", fontsize=10, loc="left")

    a = axs[1]
    W, cols = 0.26, {"r_xenium": "#7f7f7f", "r_remnant": "#d73027", "r_partial": "#1b7837"}
    for gi, lab in enumerate(labs):
        d = runs[lab]["res"]
        for k, col in enumerate(["r_xenium", "r_remnant", "r_partial"]):
            v = d[col].dropna().values
            p = gi + (k - 1) * W
            bp = a.boxplot([v], positions=[p], widths=W * .8, patch_artist=True,
                           showfliers=False, medianprops=dict(color="k", lw=1.3))
            bp["boxes"][0].set_facecolor(cols[col]); bp["boxes"][0].set_alpha(.35)
            bp["boxes"][0].set_edgecolor(cols[col])
            a.scatter(np.full(len(v), p) + RNG.normal(0, .03, len(v)), v, s=7, color=cols[col],
                      alpha=.6, linewidths=0)
    a.axhline(0, color="0.6", lw=.7)
    a.set_xticks(range(len(labs)))
    a.set_xticklabels([f"{lab}\n(n={len(runs[lab]['res'])})" for lab in labs], fontsize=9)
    a.set_ylabel("CNV profile r vs tumour profile")
    a.legend(handles=[Patch(color=cols["r_xenium"], alpha=.5, label="Xenium cell"),
                      Patch(color=cols["r_remnant"], alpha=.5, label="TRACER remnant"),
                      Patch(color=cols["r_partial"], alpha=.5, label="top epithelial partial")],
             fontsize=8, frameon=False, loc="lower left")
    a.set_title("(b) does each run's switcher split keep the same CNV pattern?",
                fontsize=10, loc="left")

    a = axs[2]
    bw = 0.8 / len(labs)
    for k, lab in enumerate(labs):
        tot = [len(runs[lab]["sw"][c]) for c in CLASSES]
        kept = [len(set(runs[lab]["sw"][c]) & set(sw0[c])) for c in CLASSES]
        xs = np.arange(len(CLASSES)) + (k - (len(labs) - 1) / 2) * bw
        a.bar(xs, tot, bw * .92, color=f"C{k}", alpha=.35, label=f"{lab}: all switchers")
        a.bar(xs, kept, bw * .92, color=f"C{k}", alpha=.9,
              label=f"{lab}: also in original" if lab != ORIG else None)
        for xv, t in zip(xs, tot):
            a.text(xv, t + 1, str(t), ha="center", fontsize=8)
    a.set_xticks(range(len(CLASSES))); a.set_xticklabels([CLAB[c] for c in CLASSES])
    a.set_ylabel("switcher cells"); a.legend(fontsize=7.5, frameon=False)
    a.set_title("(c) switcher counts, and how many are the same cells", fontsize=10, loc="left")

    fig.suptitle("Switcher robustness to TRACER's Phase-1 remerge ΔC floor "
                 "(-0.2 original, -0.2 rerun, -0.5)", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    p = OUT / "fig_remerge050_switchers.png"; fig.savefig(p, dpi=150); print("\nwrote", p)


if __name__ == "__main__":
    main()
