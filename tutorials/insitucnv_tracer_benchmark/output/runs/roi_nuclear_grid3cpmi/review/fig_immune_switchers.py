"""Do IMMUNE switchers split the same way the fibroblast switchers do?

Motivation: the epithelial -> fibroblast switchers (CNV-carrying Xenium cells that
TRACER splits into a fibroblast remnant + a CNV-carrying epithelial partial) were
challenged as possible EMT -- one tumour cell co-expressing an epithelial and a
mesenchymal programme. EMT cannot put a T-cell or myeloid programme into a tumour
cell. So if Xenium-epithelial cells whose TRACER remnant is IMMUNE show the same
split, the phenomenon is not EMT-specific.

Switcher classes use exactly the rule of fig_switch_cnv.id_groups: ID join,
Xenium annotation == epithelial, TRACER same-ID annotation == <class>.

For every switcher cell, identically across classes:
  1. Xenium cell: CNV r vs the tumour (concordant-epithelial) profile, Xenium arm
  2. remnant (same ID under TRACER): own-type markers, epithelial markers, CNV r
  3. top epithelial partial = the partial that received most of the cell's
     epithelial-marker transcripts: CNV r
Remnant and partial r come from ONE inferCNV run over whole cells + partials
(the partial_recipients.py call), so they share a smoothing pool and a baseline.
That run is cached to _partials_run_all.pkl; existing review artefacts are not
touched.
"""
from __future__ import annotations
import sys, warnings, pickle
import numpy as np, pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns

OUT = C.Path(__file__).parent
RNG = np.random.default_rng(0)
CACHE = OUT / "_partials_run_all.pkl"
ms = yaml.safe_load(open(C.BENCH / "config/panels.yaml"))["marker_sets"]
MK = {k: set(map(str, v)) for k, v in ms.items()}
CLASSES = ["fibroblast", "myeloid", "t_cell"]
CLAB = {"fibroblast": "-> fibroblast", "myeloid": "-> myeloid", "t_cell": "-> T cell"}
REFSET = {"t_cell", "b_cell", "myeloid", "fibroblast", "endothelial"}


def id_join():
    ax = pd.read_csv(C.RUN / "annotations_raw.csv", dtype={"entity_id": str}) \
        .set_index("entity_id")["annotation"].astype(str)
    at = pd.read_csv(C.RUN / "annotations_tracer_whole.csv", dtype={"entity_id": str}) \
        .set_index("entity_id")["annotation"].astype(str)
    return ax, at


def tracer_plus_partials_cnv(df, ann_path=None, cache=None):
    """Same inferCNV call as partial_recipients.py: whole cells + partials, one run.

    ann_path / cache default to this run (C.RUN) and CACHE; other TRACER runs pass
    their own so every run is scored by the identical call.
    """
    ann_path = C.RUN / "annotations_tracer_whole.csv" if ann_path is None else ann_path
    cache = CACHE if cache is None else cache
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    import run_insitucnv_arm as R
    et = df["_etype"].astype(str)
    d = df.loc[~df["tracer_id"].isin(UNASSIGNED_TOKENS) & et.isin(["cell", "partial"])].copy()
    et = d["_etype"].astype(str)
    ann = pd.read_csv(ann_path)
    a = R.build_adata(d, "tracer_id", ann, 20, 10)
    a = R.add_gene_positions(a, C.BENCH / "data/gene_positions_grch38.tsv")
    R.assign_compartment(a, REFSET, {"epithelial"})
    whole = set(d.loc[et == "cell", "tracer_id"].unique())
    is_part = ~np.isin(a.obs_names.astype(str), list(whole))
    a.obs.loc[is_part, "compartment"] = "unknown"          # scored, never reference
    R.run_infercnv(a, 60, 0.4, 100, smoothing_graph="expression",
                   expression_neighbors=100, neighbor_n_pcs=50)
    res = {"X": C.dense_cnv(a), "names": a.obs_names.astype(str).to_numpy()}
    with open(cache, "wb") as fh:
        pickle.dump(res, fh)
    return res


def per_entity_r(X, names, base_ids):
    """Per-entity Pearson r of the CNV profile vs the mean profile of base_ids."""
    pos = {v: i for i, v in enumerate(names)}
    b = X[[pos[i] for i in base_ids if i in pos]].mean(0)
    b = b - b.mean()
    Xc = X - X.mean(1, keepdims=True)
    den = np.linalg.norm(Xc, axis=1) * np.linalg.norm(b) + 1e-12
    return pd.Series((Xc @ b) / den, index=names)


def main():
    ax, at = id_join()
    df = standardize_transcript_columns(pd.read_parquet(C.ARMS["tracer_whole"]["transcripts"]))
    df["cell_id"] = df["cell_id"].astype(str); df["tracer_id"] = df["tracer_id"].astype(str)
    et = df["_etype"].astype(str)
    whole = set(df.loc[et == "cell", "tracer_id"].unique())
    partials = set(df.loc[et == "partial", "tracer_id"].unique())
    atw = at.loc[at.index.isin(whole)]
    ids = ax.index.intersection(atw.index)
    x, y = ax.reindex(ids), atw.reindex(ids)

    groups = {c: ids[(x == "epithelial") & (y == c)] for c in CLASSES}
    conc = {c: ids[(x == c) & (y == c)] for c in ["epithelial"] + CLASSES}
    print({f"switch {c}": len(v) for c, v in groups.items()},
          {f"conc {c}": len(v) for c, v in conc.items()})

    # ---- CNV: Xenium arm (per cell) and TRACER + partials (per entity) ----
    xa, _ = C.load_arm("raw")
    rx = per_entity_r(C.dense_cnv(xa), xa.obs_names.astype(str).to_numpy(), conc["epithelial"])
    tp = tracer_plus_partials_cnv(df)
    rt = per_entity_r(tp["X"], tp["names"], conc["epithelial"])

    # ---- per-switcher transcript bookkeeping ----
    feat = df["feature_name"].astype(str)
    df["mk"] = "other"
    for k, s in MK.items():
        df.loc[feat.isin(s), "mk"] = k
    sw_all = set().union(*[set(v) for v in groups.values()])
    sub = df.loc[df["cell_id"].isin(sw_all)]
    by_cell = dict(tuple(sub.groupby("cell_id")))

    rows = []
    for cls, cids in groups.items():
        for cid in cids:
            g = by_cell.get(cid)
            if g is None:
                continue
            rem = g["tracer_id"] == cid
            epi = g["mk"] == "epithelial"
            own = g["mk"] == cls
            dest = np.where(rem, "remnant",
                   np.where(g["tracer_id"].isin(partials), "partial",
                   np.where(g["tracer_id"].isin(whole), "other whole cell", "unassigned")))
            e_dest = pd.Series(dest[epi.to_numpy()]).value_counts()
            ep = g.loc[epi & g["tracer_id"].isin(partials), "tracer_id"].value_counts()
            top = ep.index[0] if len(ep) else None
            rows.append(dict(
                cls=cls, cell_id=cid, n_xen=len(g), n_rem=int(rem.sum()),
                epi_xen=int(epi.sum()), own_xen=int(own.sum()),
                epi_rem=int((epi & rem).sum()), own_rem=int((own & rem).sum()),
                **{f"epi_to_{k.replace(' ', '_')}": int(e_dest.get(k, 0))
                   for k in ["remnant", "partial", "other whole cell", "unassigned"]},
                top_epi_partial=top,
                top_epi_share=(ep.iloc[0] / max(epi.sum(), 1)) if len(ep) else np.nan,
                r_xenium=rx.get(cid, np.nan), r_remnant=rt.get(cid, np.nan),
                r_partial=rt.get(top, np.nan) if top else np.nan))
    res = pd.DataFrame(rows)

    # own-type markers in the XENIUM CELL of concordant cells of the same type --
    # measured on the same footing as the switchers' own_xen (whole Xenium cell),
    # not on the TRACER entity, which can import transcripts from neighbours
    cm = []
    for cls in CLASSES:
        cc = df.loc[df["cell_id"].isin(set(conc[cls]))]
        cnt = cc.assign(own=cc["mk"] == cls).groupby("cell_id")["own"].sum()
        cm.append(pd.DataFrame({"cls": cls, "own_conc": cnt.values}))
    cm = pd.concat(cm, ignore_index=True)
    res.to_csv(OUT / "immune_switchers.csv", index=False)

    # ================= figure =================
    COLS = {"xen": "#7f7f7f", "rem": "#d73027", "par": "#1b7837"}
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))

    # (a) CNV triplet per class
    a = axs[0]
    W = 0.26
    for gi, cls in enumerate(CLASSES):
        d = res.loc[res["cls"] == cls]
        for k, (col, key) in enumerate([("r_xenium", "xen"), ("r_remnant", "rem"),
                                        ("r_partial", "par")]):
            v = d[col].dropna().values
            p = gi + (k - 1) * W
            bp = a.boxplot([v], positions=[p], widths=W * 0.8, patch_artist=True,
                           showfliers=False, medianprops=dict(color="k", lw=1.4))
            bp["boxes"][0].set_facecolor(COLS[key]); bp["boxes"][0].set_alpha(0.35)
            bp["boxes"][0].set_edgecolor(COLS[key])
            a.scatter(np.full(len(v), p) + RNG.normal(0, 0.03, len(v)), v, s=10,
                      color=COLS[key], alpha=0.75, linewidths=0)
    ref_t = rx.reindex(conc["epithelial"]).median()
    a.axhline(ref_t, color="#1b7837", ls="--", lw=1, alpha=0.6,
              label=f"concordant tumour cells, Xenium ({ref_t:.2f})")
    # each remnant class against concordant cells of ITS OWN type, same run
    for gi, cls in enumerate(CLASSES):
        b = rt.reindex(conc[cls]).median()
        v = res.loc[res["cls"] == cls, "r_remnant"].dropna()
        a.hlines(b, gi - W * 0.55, gi + W * 0.55, colors="k", linestyles="-", lw=2.2,
                 label="concordant cells of the remnant's type" if gi == 0 else None)
        a.text(gi + W * 0.45, v.quantile(0.75) + 0.04, f"{v.median() - b:+.2f}",
               fontsize=8.5, ha="center", color="#d73027", fontweight="bold")
    a.axhline(0, color="0.6", lw=0.7)
    a.set_xticks(range(3))
    a.set_xticklabels([f"Xenium epithelial\n{CLAB[c]}\n(n={len(groups[c])})" for c in CLASSES],
                      fontsize=9)
    a.set_ylabel("CNV profile r vs tumour profile")
    a.legend(handles=[Patch(color=COLS["xen"], alpha=.5, label="Xenium cell"),
                      Patch(color=COLS["rem"], alpha=.5, label="TRACER remnant (same ID)"),
                      Patch(color=COLS["par"], alpha=.5, label="top epithelial partial")]
             + a.get_legend_handles_labels()[0], fontsize=8, frameon=False, loc="lower left")
    a.set_title("(a) does the CNV go with the epithelial partial, for every class?",
                fontsize=10, loc="left")

    # (b) where each class's epithelial marker transcripts went
    a = axs[1]
    dk = ["epi_to_remnant", "epi_to_partial", "epi_to_other_whole_cell", "epi_to_unassigned"]
    dl = ["remnant", "partial", "other whole cell", "unassigned"]
    dc = ["#d73027", "#1b7837", "#fdae61", "#bdbdbd"]
    tot = res.groupby("cls")[dk].sum().reindex(CLASSES)
    frac = tot.div(tot.sum(1), axis=0)
    left = np.zeros(len(CLASSES))
    for k, lab, c in zip(dk, dl, dc):
        a.barh(range(len(CLASSES)), frac[k].values, left=left, color=c, label=lab)
        for i, v in enumerate(frac[k].values):
            if v > 0.06:
                a.text(left[i] + v / 2, i, f"{100*v:.0f}%", ha="center", va="center", fontsize=8)
        left += frac[k].values
    a.set_yticks(range(len(CLASSES)))
    a.set_yticklabels([f"{CLAB[c]}\n({int(tot.loc[c].sum())} epi tx)" for c in CLASSES],
                      fontsize=9)
    a.invert_yaxis(); a.set_xlim(0, 1)
    a.set_xlabel("fraction of the Xenium cell's epithelial-marker transcripts")
    a.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
             ncol=4)
    a.set_title("(b) where the epithelial markers went", fontsize=10, loc="left")

    # (c) is the remnant really that cell type?
    a = axs[2]
    vals, cols = [], []
    for cls in CLASSES:
        vals += [res.loc[res["cls"] == cls, "own_xen"].values,
                 cm.loc[cm["cls"] == cls, "own_conc"].values]
        cols += ["#7f7f7f", "#4575b4"]
    pos = [0.8, 1.2, 1.8, 2.2, 2.8, 3.2]
    bp = a.boxplot(vals, positions=pos, widths=0.33, patch_artist=True, showfliers=False,
                   medianprops=dict(color="k", lw=1.3))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    for p, v, c in zip(pos, vals, cols):
        a.scatter(np.full(len(v), p) + RNG.normal(0, 0.03, len(v)), v, s=8, color=c,
                  alpha=0.6, linewidths=0)
    a.set_yscale("symlog", linthresh=5)
    a.set_xticks([1, 2, 3])
    a.set_xticklabels([f"fibroblast\nmarkers", "myeloid\nmarkers", "T-cell\nmarkers"],
                      fontsize=9)
    a.set_ylabel("own-type marker transcripts in the XENIUM cell")
    a.legend(handles=[Patch(color="#7f7f7f", alpha=.5, label="switcher Xenium cell"),
                      Patch(color="#4575b4", alpha=.5, label="concordant Xenium cell of that type")],
             fontsize=8, frameon=False, loc="upper right")
    a.set_title("(c) was the immune / fibroblast signal already in the Xenium cell?",
                fontsize=10, loc="left")

    fig.suptitle("Immune switchers: CNV-carrying Xenium cells that TRACER splits into an immune "
                 "remnant + an epithelial partial  (same rule as the fibroblast switchers)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    p = OUT / "fig_immune_switchers.png"; fig.savefig(p, dpi=150); print("wrote", p)

    # ---- numbers ----
    summ = res.groupby("cls").agg(
        n=("cell_id", "size"),
        r_xenium=("r_xenium", "median"), r_remnant=("r_remnant", "median"),
        r_partial=("r_partial", "median"), has_partial=("top_epi_partial", "count"),
        epi_xen=("epi_xen", "median"), epi_rem_sum=("epi_rem", "sum"),
        cells_rem_any_epi=("epi_rem", lambda s: int((s > 0).sum())),
        own_rem=("own_rem", "median"), n_rem=("n_rem", "median"),
        top_epi_share=("top_epi_share", "median")).reindex(CLASSES)
    print("\n" + summ.round(3).to_string())
    print("\nconcordant own-type markers (median):")
    print(cm.groupby("cls")["own_conc"].median().reindex(CLASSES).to_string())
    print("\nconcordant reference, TRACER+partials run, r vs tumour (median):")
    for c in CLASSES:
        print(f"  {c:10s} {rt.reindex(conc[c]).median():+.3f}  (n={len(conc[c])})")
    print(f"  epithelial {rt.reindex(conc['epithelial']).median():+.3f}")


if __name__ == "__main__":
    main()
