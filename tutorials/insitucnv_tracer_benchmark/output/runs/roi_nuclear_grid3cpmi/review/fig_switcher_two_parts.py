"""Cell x gene matrix of one switcher's two parts, from the -0.5 remerge run.

Switcher = the myeloid example of fig_switcher_class_separation.py (same selection
rule, re-applied here and asserted), so the spatial plot and this matrix describe the
same cell. Selected in the original run; parts taken from the -0.5 run, where a
one-cell split should have been collapsed by the loosened Phase-1 remerge:
  remnant = same-ID whole-cell entity
  partial = the -0.5 entity holding most of the original partial's transcripts
            (partial IDs are renamed between runs)

Rows
  remnant, partial            TRACER's own cell_by_gene_tracer.h5ad ('counts' layer)
  xenium_cell                 every transcript Xenium assigned to the cell
  remnant_incell, partial_incell   each part's transcripts inside the Xenium cell
Columns: all 5,000 panel genes.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, yaml, matplotlib, anndata as ad, scipy.sparse as sp
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns

OUT = C.Path(__file__).parent
R05 = C.BENCH / "output/runs/roi_nuclear_grid3cpmi_remerge050"
ms = yaml.safe_load(open(C.BENCH / "config/panels.yaml"))["marker_sets"]
MCOL = {"epithelial": "#1b7837", "myeloid": "#ff7f00", "fibroblast": "#984ea3",
        "t_cell": "#e31a1c", "b_cell": "#1f78b4", "endothelial": "#a6761d"}
G2SET = {g: k for k, v in ms.items() for g in map(str, v)}
TOP = 30


def pick_example():
    res = pd.read_csv(OUT / "switcher_class_separation.csv")
    my = res.loc[(res["cls"] == "myeloid") & (res["nA"] >= 250) & (res["nB"] >= 250)]
    med = res.loc[res["cls"] == "myeloid", "auc_best"].median()
    ex = my.iloc[(my["auc_best"] - med).abs().argsort().iloc[0]]
    return ex["cell_id"], ex["partial"]


def load(run_dir):
    t = standardize_transcript_columns(pd.read_parquet(
        run_dir / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["transcript_id", "cell_id", "tracer_id", "_etype", "feature_name"]))
    for c in ("cell_id", "tracer_id", "_etype"):
        t[c] = t[c].astype(str)
    t["feature_name"] = t["feature_name"].astype(str)
    return t


def main():
    cid, pid0 = pick_example()
    assert cid == "hojhmlnf-1", f"example changed: {cid}"
    o, n = load(C.RUN), load(R05)

    # match the -0.5 parts to their original counterparts
    tx_part0 = set(o.loc[o["tracer_id"] == pid0, "transcript_id"])
    tx_rem0 = set(o.loc[o["tracer_id"] == cid, "transcript_id"])
    dest = n.loc[n["transcript_id"].isin(tx_part0), "tracer_id"].value_counts()
    pid = dest.index[0]
    etype_p = n.loc[n["tracer_id"] == pid, "_etype"].iloc[0]
    tx_part = set(n.loc[n["tracer_id"] == pid, "transcript_id"])
    tx_rem = set(n.loc[n["tracer_id"] == cid, "transcript_id"])
    J = lambda a, b: len(a & b) / max(len(a | b), 1)
    print(f"switcher {cid}: original partial {pid0} -> -0.5 entity {pid} ({etype_p}); "
          f"{dest.iloc[0] / len(tx_part0):.1%} of the original partial's tx went there")
    print(f"  Jaccard vs original: remnant {J(tx_rem, tx_rem0):.3f}, partial {J(tx_part, tx_part0):.3f}")

    # TRACER's own cell x gene rows
    h = ad.read_h5ad(R05 / "tracer/outputs/cell_by_gene_tracer.h5ad")
    genes = h.var_names.astype(str)
    gpos = {g: i for i, g in enumerate(genes)}
    Xc = h.layers["counts"]
    rows = {"remnant": cid, "partial": pid}
    mats, obs = [], []
    for lab, eid in rows.items():
        i = int(np.where(h.obs_names.astype(str) == eid)[0][0])
        mats.append(sp.csr_matrix(Xc[i]))
        o_ = h.obs.iloc[i].to_dict()
        obs.append(dict(row=lab, entity_id=eid, source="cell_by_gene_tracer.h5ad", **o_))

    # context rows from the -0.5 parquet
    def count_row(mask):
        vc = n.loc[mask, "feature_name"].value_counts()
        v = np.zeros(len(genes), dtype=np.float32)
        idx = [gpos[g] for g in vc.index if g in gpos]
        v[idx] = vc[[g for g in vc.index if g in gpos]].to_numpy()
        return sp.csr_matrix(v)

    inc = n["cell_id"] == cid
    for lab, mask in [("xenium_cell", inc), ("remnant_incell", inc & (n["tracer_id"] == cid)),
                      ("partial_incell", inc & (n["tracer_id"] == pid))]:
        mats.append(count_row(mask))
        obs.append(dict(row=lab, entity_id=cid if lab == "xenium_cell" else rows[lab.split("_")[0]],
                        source="refined parquet, transcripts inside the Xenium cell"))
    M = sp.vstack(mats).tocsr()

    # verification: row sums vs transcript counts of on-panel features
    on_panel = n["feature_name"].isin(gpos)
    expect = [int((on_panel & (n["tracer_id"] == cid)).sum()),
              int((on_panel & (n["tracer_id"] == pid)).sum()),
              int((on_panel & inc).sum()),
              int((on_panel & inc & (n["tracer_id"] == cid)).sum()),
              int((on_panel & inc & (n["tracer_id"] == pid)).sum())]
    got = np.asarray(M.sum(1)).ravel().astype(int).tolist()
    print("row sums  :", dict(zip([r["row"] for r in obs], got)))
    print("tx counts :", dict(zip([r["row"] for r in obs], expect)))
    assert got == expect, "row sums do not match transcript counts"

    annot = pd.read_csv(R05 / "annotations_tracer_whole.csv").set_index("entity_id")
    od = pd.DataFrame(obs).set_index("row")
    od["n_tx"] = got
    od["n_genes"] = np.asarray((M > 0).sum(1)).ravel()
    for k in ms:
        od[f"{k}_marker_tx"] = [int(M[i, [gpos[g] for g in map(str, ms[k]) if g in gpos]].sum())
                                for i in range(M.shape[0])]
    od["annotation"] = [annot["annotation"].get(e) if r in ("remnant", "partial") else None
                        for r, e in zip(od.index, od["entity_id"])]
    od["jaccard_vs_original"] = [J(tx_rem, tx_rem0), J(tx_part, tx_part0), np.nan, np.nan, np.nan]
    od = od.astype({c: str for c in od.columns if od[c].dtype == object})
    A = ad.AnnData(X=M, obs=od, var=pd.DataFrame(index=genes))
    A.var["marker_set"] = [G2SET.get(g, "") for g in genes]
    A.uns["switcher"] = dict(cell_id=cid, partial_original=pid0, partial_remerge050=pid,
                             run="roi_nuclear_grid3cpmi_remerge050")
    stem = f"switcher_{cid}_two_parts"
    A.write_h5ad(OUT / f"{stem}.h5ad")
    D = pd.DataFrame(M.toarray().T, index=genes, columns=od.index).astype(int)
    D = D.loc[D.sum(1) > 0]
    D.insert(0, "marker_set", [G2SET.get(g, "") for g in D.index])
    D.sort_values(["remnant", "partial"], ascending=False).to_csv(OUT / f"{stem}.csv")
    print(f"wrote {stem}.h5ad ({M.shape[0]} x {M.shape[1]}) and {stem}.csv ({len(D)} genes with >=1 count)")

    # ---------------- figure ----------------
    rem, par = D["remnant"], D["partial"]
    ri, pi_ = D["remnant_incell"], D["partial_incell"]
    only_r, only_p, both = ((rem > 0) & (par == 0)).sum(), ((par > 0) & (rem == 0)).sum(), \
        ((rem > 0) & (par > 0)).sum()
    fig, axs = plt.subplots(1, 3, figsize=(19, 8.2), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))

    def topbar(a, col, other, title):
        s = D[col].sort_values(ascending=False).head(TOP)[::-1]
        cols = [MCOL.get(G2SET.get(g, ""), "0.65") for g in s.index]
        a.barh(range(len(s)), s.values, color=cols)
        a.set_yticks(range(len(s)))
        a.set_yticklabels([f"{g}   ({int(D.loc[g, other])} in {other})" for g in s.index], fontsize=8)
        a.set_xlabel("transcripts")
        a.set_title(title, fontsize=10, loc="left")

    topbar(axs[0], "remnant", "partial",
           f"(a) remnant {cid} -> '{od.loc['remnant', 'annotation']}'\n"
           f"{od.loc['remnant', 'n_tx']} tx, {od.loc['remnant', 'n_genes']} genes; "
           f"top {TOP} (count in the partial in brackets)")
    topbar(axs[1], "partial", "remnant",
           f"(b) partial {pid} -> '{od.loc['partial', 'annotation']}'\n"
           f"{od.loc['partial', 'n_tx']} tx, {od.loc['partial', 'n_genes']} genes; "
           f"top {TOP} (count in the remnant in brackets)")
    handles = [Patch(color=c, label=k) for k, c in MCOL.items()] + [Patch(color="0.65", label="other")]
    axs[1].legend(handles=handles, fontsize=8, frameon=False, loc="lower right", title="marker set")

    a = axs[2]
    # one generator, independent draws for x and y (multiplicative, so it reads
    # evenly on log axes and never pushes a 0-count gene off its axis)
    rng = np.random.default_rng(0)
    jit = lambda v: (v + 1) * np.exp(rng.uniform(-0.12, 0.12, len(v)))
    m = (ri > 0) | (pi_ > 0)
    cols = [MCOL.get(G2SET.get(g, ""), "0.6") for g in D.index[m]]
    sz = [40 if G2SET.get(g) else 10 for g in D.index[m]]
    a.scatter(jit(ri[m].values), jit(pi_[m].values), s=sz, c=cols, alpha=0.6, linewidths=0)
    a.set_xscale("log"); a.set_yscale("log")
    a.plot([1, 300], [1, 300], ls=":", color="0.6")
    for g in D.index[m]:
        if G2SET.get(g) and (ri[g] + pi_[g]) >= 3:
            a.annotate(g, (ri[g] + 1, pi_[g] + 1), fontsize=7, xytext=(3, 2),
                       textcoords="offset points")
    excl = ((ri[m] == 0) | (pi_[m] == 0)).mean()
    a.set_xlabel("transcripts in the remnant, inside the Xenium cell (+1)")
    a.set_ylabel("transcripts in the partial, inside the Xenium cell (+1)")
    a.set_title(f"(c) inside the Xenium cell, per gene: {100*excl:.0f}% of {int(m.sum())} genes\n"
                f"go to ONE side only (on an axis = exclusive, diagonal = shared)\n"
                f"whole entities: {only_r} remnant-only, {only_p} partial-only, {both} shared genes",
                fontsize=10, loc="left")

    fig.suptitle(f"Myeloid switcher {cid}: the two parts TRACER kept apart even at remerge "
                 f"ΔC floor -0.5 (run roi_nuclear_grid3cpmi_remerge050)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = OUT / f"fig_{stem}.png"; fig.savefig(p, dpi=150); print("wrote", p)
    print("\n" + od[["entity_id", "n_tx", "n_genes", "epithelial_marker_tx", "myeloid_marker_tx",
                     "fibroblast_marker_tx", "annotation", "jaccard_vs_original"]].to_string())
    print(f"\nin-cell genes exclusive to one side: {excl:.1%} of {int(m.sum())}")


if __name__ == "__main__":
    main()
