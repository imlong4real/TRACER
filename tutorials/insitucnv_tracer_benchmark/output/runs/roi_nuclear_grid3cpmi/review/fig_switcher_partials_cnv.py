"""CNV profiles of the epithelial partials that receive switcher material.

One inferCNV run (whole cells + partials, cached by fig_immune_switchers.py), so
every row shares the same baseline, smoothing pool and colour scale.

Row blocks:
  concordant tumour cells (random 60)          -- what tumour CNV looks like
  partials <- epithelial->fibroblast switchers
  partials <- epithelial->myeloid switchers
  partials <- epithelial->T-cell switchers
  concordant normal cells (random 60: fib/myeloid/T)  -- what flat looks like

"Partial" = the partial that received the most of a switcher cell's
epithelial-marker transcripts (top_epi_partial in immune_switchers.csv),
deduplicated within a block. The cache stored only X and entity names, so the
window axis is rebuilt from the same build_adata call and checked against both
the entity order and the column count before anything is drawn.
"""
from __future__ import annotations
import sys, warnings, pickle
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns
from fig_switch_cnv import order_within
from fig_immune_switchers import id_join, CACHE, CLASSES

OUT = C.Path(__file__).parent
RNG = np.random.default_rng(0)
N_REF = 60


def rebuild_windows(names, n_cols):
    import run_insitucnv_arm as R
    df = standardize_transcript_columns(pd.read_parquet(C.ARMS["tracer_whole"]["transcripts"]))
    df["cell_id"] = df["cell_id"].astype(str); df["tracer_id"] = df["tracer_id"].astype(str)
    et = df["_etype"].astype(str)
    d = df.loc[~df["tracer_id"].isin(UNASSIGNED_TOKENS) & et.isin(["cell", "partial"])]
    a = R.build_adata(d, "tracer_id", pd.read_csv(C.RUN / "annotations_tracer_whole.csv"), 20, 10)
    a = R.add_gene_positions(a, C.BENCH / "data/gene_positions_grch38.tsv")
    if not np.array_equal(a.obs_names.astype(str).to_numpy(), names):
        raise SystemExit("rebuilt adata entity order differs from the cached CNV run")
    wins = C.window_intervals(a.var)
    if len(wins) != n_cols:
        raise SystemExit(f"rebuilt {len(wins)} windows, cached X has {n_cols}")
    print(f"window axis OK: {n_cols} columns, entity order identical to the cache")
    return wins.reset_index(drop=True)


def main():
    tp = pickle.load(open(CACHE, "rb"))
    X, names = tp["X"], tp["names"]
    pos = {v: i for i, v in enumerate(names)}
    wins = rebuild_windows(names, X.shape[1])

    res = pd.read_csv(OUT / "immune_switchers.csv")
    ax, at = id_join()
    ids = ax.index.intersection(at.index)
    x, y = ax.reindex(ids), at.reindex(ids)
    have = lambda s: [i for i in s if i in pos]

    tum = have(ids[(x == "epithelial") & (y == "epithelial")])
    nrm = have(ids[((x == "fibroblast") & (y == "fibroblast")) | ((x == "myeloid") & (y == "myeloid"))
                   | ((x == "t_cell") & (y == "t_cell"))])
    tum = list(RNG.choice(tum, N_REF, replace=False))
    nrm = list(RNG.choice(nrm, N_REF, replace=False))

    blocks = [("concordant tumour cells\n(random %d)" % N_REF, tum, "#1b7837")]
    PCOL = {"fibroblast": "#984ea3", "myeloid": "#ff7f00", "t_cell": "#e31a1c"}
    PLAB = {"fibroblast": "fibroblast", "myeloid": "myeloid", "t_cell": "T-cell"}
    part_sets = {}
    for c in CLASSES:
        p = have(res.loc[res["cls"] == c, "top_epi_partial"].dropna().unique())
        part_sets[c] = set(p)
        blocks.append((f"partials <- {PLAB[c]}\nswitchers", p, PCOL[c]))
    blocks.append(("concordant normal cells\n(fib / myeloid / T,\nrandom %d)" % N_REF, nrm, "0.45"))
    shared = (part_sets["fibroblast"] & part_sets["myeloid"]) | \
             (part_sets["fibroblast"] & part_sets["t_cell"]) | \
             (part_sets["myeloid"] & part_sets["t_cell"])

    order, bounds = [], [0]
    for _, ids_, _ in blocks:
        sub = X[[pos[i] for i in ids_]]
        order += [ids_[k] for k in order_within(sub)]
        bounds.append(len(order))
    M = X[[pos[i] for i in order]]
    vmax = float(np.quantile(np.abs(M), 0.995))

    # chromosome boundaries on the window axis
    chrom = wins["chromosome"].to_numpy()
    edges = [0] + [k for k in range(1, len(chrom)) if chrom[k] != chrom[k - 1]] + [len(chrom)]
    mids = [(edges[k] + edges[k + 1]) / 2 for k in range(len(edges) - 1)]
    labs = [chrom[edges[k]].replace("chr", "") for k in range(len(edges) - 1)]

    fig = plt.figure(figsize=(19, 12.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.0], width_ratios=[0.13, 1.0],
                          hspace=0.08, wspace=0.01, left=0.02, right=0.985, top=0.92, bottom=0.06)
    hm = fig.add_subplot(gs[0, 1])
    hm.imshow(M, aspect="auto", cmap="RdBu_r", norm=Normalize(-vmax, vmax),
              interpolation="nearest")
    for e in edges[1:-1]:
        hm.axvline(e - 0.5, color="k", lw=0.5)
    for b in bounds[1:-1]:
        hm.axhline(b - 0.5, color="k", lw=1.6)
    hm.set_xticks([]); hm.set_yticks([])
    hm.set_title(f"inferCNV, one run (whole cells + partials), one colour scale   "
                 f"|   {X.shape[1]} windows", fontsize=10.5, loc="left")

    lab = fig.add_subplot(gs[0, 0], sharey=hm)
    lab.axis("off")
    base = X[[pos[i] for i in tum]].mean(0)
    for (name, ids_, col), b0, b1 in zip(blocks, bounds[:-1], bounds[1:]):
        prof = X[[pos[i] for i in ids_]].mean(0)
        r = np.corrcoef(prof - prof.mean(), base - base.mean())[0, 1]
        lab.add_patch(plt.Rectangle((0.90, b0 - 0.5), 0.08, b1 - b0, color=col,
                                    transform=lab.transData, clip_on=False))
        lab.text(0.86, (b0 + b1) / 2, f"{name}\nn={len(ids_)}, r={r:+.2f}", ha="right",
                 va="center", fontsize=9, color=col if col != "0.45" else "0.25")
    lab.set_xlim(0, 1)

    tr = fig.add_subplot(gs[1, 1], sharex=hm)
    for name, ids_, col in blocks:
        prof = X[[pos[i] for i in ids_]].mean(0)
        tr.plot(np.arange(len(prof)), prof, color=col, lw=1.4 if "partial" in name else 2.0,
                alpha=0.9, label=name.split("\n")[0] + " " + name.split("\n")[1]
                if "partial" in name else name.split("\n")[0])
    for e in edges[1:-1]:
        tr.axvline(e - 0.5, color="0.7", lw=0.5)
    tr.axhline(0, color="0.5", lw=0.7)
    tr.set_xticks(mids); tr.set_xticklabels(labs, fontsize=8)
    hm.tick_params(labelbottom=False, bottom=False)       # shared axis: labels on the track only
    tr.set_xlim(-0.5, len(chrom) - 0.5)
    tr.set_xlabel("chromosome"); tr.set_ylabel("block mean CNV")
    tr.legend(fontsize=8, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16))

    cax = fig.add_axes([0.012, 0.17, 0.07, 0.012])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-vmax, vmax), cmap="RdBu_r"), cax=cax,
                 orientation="horizontal", label="inferCNV value")
    cax.tick_params(labelsize=7)
    fig.suptitle("CNV profiles of the epithelial partials fed by switcher cells, by switcher class "
                 "(r = block mean vs concordant-tumour mean)", fontsize=13)
    p = OUT / "fig_switcher_partials_cnv.png"; fig.savefig(p, dpi=140); print("wrote", p)

    print(f"\npartials shared between classes: {len(shared)}")
    for c in CLASSES:
        print(f"  {c:10s} unique partials {len(part_sets[c])}")
    # per-chromosome means of the blocks, for the arm-level events
    cm = pd.DataFrame({name.split("\n")[0] + ("" if "partial" not in name else
                       " " + name.split("\n")[1]):
                       pd.Series(X[[pos[i] for i in ids_]].mean(0)).groupby(chrom).mean()
                       for name, ids_, _ in blocks})
    cm = cm.reindex([c for c in pd.unique(chrom)])
    cm.to_csv(OUT / "switcher_partials_cnv_by_chrom.csv")
    print("\nper-chromosome block means (largest |tumour| events first):")
    top = cm.iloc[:, 0].abs().sort_values(ascending=False).index[:8]
    print(cm.loc[top].round(3).to_string())


if __name__ == "__main__":
    main()
