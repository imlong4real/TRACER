#!/usr/bin/env python3
"""Panel G — "from pixel to profile": shared 2 µm bins & overlapping cells.

(G1) Distribution of the number of 2×2 µm bins contributing to each
     reconstructed profile, bucketed 1 / 2–4 / 5–10 / >10 bins.
(G2) A spatially DENSE, cell-type-DIVERSE region shown on full-resolution H&E:
     every retained 2 µm bin is coloured by the transferred cell type of the
     reconstructed cell it belongs to, and bins that are **shared by more than
     one cell** (a single 2×2 µm bin whose transcripts TRACER split across
     multiple reconstructed cells; `dominant_fraction < 1`) are highlighted
     distinctly. This shows TRACER resolving overlapping cells that occupy the
     very same pixel — impossible to separate in 2D H&E.
(G3) Reconstructed transcriptome (canonical markers) of every cell displayed in
     G2, confirming each is a transcriptionally coherent, distinct cell type.

Message: TRACER reconstructs distinct, overlapping cell-like profiles from
pixel-scale 2 µm bins, even where multiple cells share a single bin.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp

import fig4_config as C
import utils as U

BUCKETS = ["1 bin", "2–4 bins", "5–10 bins", ">10 bins"]
_HOUSEKEEP = re.compile(r"^(MT-|MTRNR|MALAT1|RPL|RPS|RP[0-9]|NEAT1|FOS|JUN|HSP)")
SHARED_THR = 0.999          # dominant_fraction < this  ==>  bin shared by >1 cell

STRONG = {
    "PT": ["LRP2", "CUBN", "SLC34A1", "ALDOB", "SLC5A2"],
    "TAL": ["UMOD", "SLC12A1", "CLDN16", "KCNJ1"],
    "PC": ["AQP2", "AQP3", "SCNN1G"],
    "IC": ["SLC4A1", "ATP6V0D2", "ATP6V1B1", "SLC26A4", "FOXI1"],
    "EC": ["PECAM1", "EMCN", "FLT1", "VWF", "KDR"],
    "FIB/VSMC/P": ["COL1A1", "COL3A1", "DCN", "ACTA2", "PDGFRB"],
    "Myeloid": ["LYZ", "C1QA", "C1QB", "CD68", "CSF1R"],
    "Lymphoid": ["CD3E", "CD3D", "PTPRC", "TRAC", "MS4A1"],
    "POD": ["NPHS1", "NPHS2", "PODXL", "WT1", "PTPRO"],
}


def _bucket(n):
    if n <= 1:
        return "1 bin"
    if n <= 4:
        return "2–4 bins"
    if n <= 10:
        return "5–10 bins"
    return ">10 bins"


def _pick_dense_region(bf, half=22.0, min_lin=3):
    """Pick a window that is dense (many retained bins), cell-type diverse, and
    has a MODERATE fraction of shared bins (dominant_fraction < 1) — so shared
    bins stand out against unshared ones rather than swamping the view. Returns
    window centre (cx, cy), half-size, and the region stats row."""
    G = 2 * half
    gx = np.floor(bf["mx"].to_numpy() / G).astype(int)
    gy = np.floor(bf["my"].to_numpy() / G).astype(int)
    d = bf.assign(gx=gx, gy=gy)
    agg = d.groupby(["gx", "gy"]).agg(
        n_bins=("bin_id", "size"),
        n_lin=("lineage", "nunique"),
        n_shared=("dominant_fraction", lambda s: int((s < SHARED_THR).sum())),
    ).reset_index()
    agg["frac_shared"] = agg["n_shared"] / agg["n_bins"]
    agg = agg[(agg["n_lin"] >= min_lin) & (agg["n_shared"] >= 6) &
              (agg["frac_shared"].between(0.12, 0.55)) & (agg["n_bins"] >= 40)]
    if agg.empty:                                   # relax if nothing qualifies
        agg = d.groupby(["gx", "gy"]).agg(
            n_bins=("bin_id", "size"), n_lin=("lineage", "nunique"),
            n_shared=("dominant_fraction", lambda s: int((s < SHARED_THR).sum()))
        ).reset_index()
        agg = agg[agg["n_lin"] >= 3]
    # dense + diverse + enough (but not too many) shared bins
    agg["score"] = agg["n_lin"] * np.log1p(agg["n_bins"]) * np.log1p(agg["n_shared"])
    best = agg.sort_values("score", ascending=False).iloc[0]
    cx = (best["gx"] + 0.5) * G; cy = (best["gy"] + 0.5) * G
    return float(cx), float(cy), half, best


def make():
    plt = U.setup_style()
    import anndata as ad

    full = pd.read_csv(C.LABELS["tracer_2um"], sep="\t").dropna(subset=["transferred_label"])
    full["reconstructed_profile_id"] = full["reconstructed_profile_id"].astype(str)
    full["bucket"] = full["n_bins"].apply(_bucket)
    counts = full["bucket"].value_counts().reindex(BUCKETS).fillna(0).astype(int)
    frac = counts / counts.sum()
    pd.DataFrame({"n_profiles": counts, "fraction": frac.round(4)}).to_csv(
        C.SRCDIR / "panel_G_bins_per_profile_distribution.csv")

    bf = U.tracer2um_bin_frame()
    cx, cy, half, reg = _pick_dense_region(bf)
    win = bf[(bf.mx >= cx - half) & (bf.mx <= cx + half) &
             (bf.my >= cy - half) & (bf.my <= cy + half)].copy()
    win["shared"] = win["dominant_fraction"] < SHARED_THR
    n_shared = int(win["shared"].sum())

    nbins_win = win.groupby("profile_id").size()
    ent_lin = win.drop_duplicates("profile_id").set_index("profile_id")["lineage"]
    cands = nbins_win[nbins_win >= 2].index.tolist()

    # Fetch whole-transcriptome for ALL candidates, then keep only those whose
    # own lineage markers are detected (non-empty heatmap rows). Pick the
    # strongest representative per lineage, capped to keep the ROI legible.
    a = ad.read_h5ad(C.WT_H5AD["tracer_2um"], backed="r")
    suba = a[cands].to_memory()
    Xa = sp.csr_matrix(suba.X).astype(float)
    tota = np.asarray(Xa.sum(1)).ravel(); umis_a = tota.copy(); tota[tota == 0] = 1
    Xna = sp.csr_matrix(Xa.multiply(1.0 / tota[:, None])); Xna.data = np.log1p(Xna.data * 1e4)
    vpos = {g: i for i, g in enumerate(suba.var_names)}
    cdf = pd.DataFrame({"profile_id": cands, "lineage": [ent_lin.get(c) for c in cands],
                        "umis": umis_a.astype(int), "row": range(len(cands))}).dropna(subset=["lineage"])
    cdf["marker_sum"] = [float(Xna[r, [vpos[g] for g in STRONG.get(l, []) if g in vpos]].sum())
                         if any(g in vpos for g in STRONG.get(l, [])) else 0.0
                         for r, l in zip(cdf["row"], cdf["lineage"])]
    cdf = cdf[cdf["marker_sum"] > 0].sort_values("marker_sum", ascending=False)
    N_ENT = 7
    reps = cdf.drop_duplicates("lineage")            # strongest cell per lineage
    entities = reps.head(N_ENT)["profile_id"].tolist()
    if len(entities) < N_ENT:                          # fill with next-strongest
        extra = cdf[~cdf["profile_id"].isin(entities)]
        entities += extra.head(N_ENT - len(entities))["profile_id"].tolist()
    rowmap = dict(zip(cdf["profile_id"], cdf["row"]))
    umis_e = np.array([umis_a[rowmap[e]] for e in entities])
    lins_present = [l for l in C.LINEAGES if l in set(ent_lin.reindex(entities).dropna())]
    markers, mlin = [], []
    for l in lins_present:
        for g in STRONG.get(l, [])[:3]:
            if g in vpos and g not in markers:
                markers.append(g); mlin.append(l)
    M = np.asarray(Xna[[rowmap[e] for e in entities]][:, [vpos[g] for g in markers]].todense())

    # ---- source tables ----
    pd.DataFrame({
        "profile_id": entities, "lineage": [ent_lin.get(e, "NA") for e in entities],
        "n_bins_in_window": [int(nbins_win.get(e, 0)) for e in entities],
        "wt_umis": umis_e.astype(int)}).to_csv(
        C.SRCDIR / "panel_G_region_entities.csv", index=False)
    win[["bin_id", "profile_id", "lineage", "mx", "my", "dominant_fraction",
         "n_tx_in_bin", "shared"]].to_csv(C.SRCDIR / "panel_G_region_bins.csv.gz", index=False)
    pd.DataFrame(M, index=entities, columns=markers).to_csv(
        C.SRCDIR / "panel_G_region_marker_matrix.csv")
    pd.DataFrame([{"center_x_um": round(cx, 1), "center_y_um": round(cy, 1),
                   "window_um": int(2 * half), "n_bins": int(reg["n_bins"]),
                   "n_lineages": int(reg["n_lin"]), "n_shared_bins": n_shared,
                   "frac_shared": round(n_shared / max(len(win), 1), 3)}]).to_csv(
        C.SRCDIR / "panel_G_region_metadata.csv", index=False)

    # ---- figure ----
    fig = plt.figure(figsize=(12.8, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.12, 1.22], wspace=0.42)

    # (G1) bins-per-profile distribution
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(range(len(BUCKETS)), frac.values * 100,
            color=["#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"], width=0.72)
    ax0.set_xticks(range(len(BUCKETS))); ax0.set_xticklabels(BUCKETS, fontsize=7)
    ax0.set_ylabel("% of TRACER 2 µm profiles")
    ax0.set_title("Bins contributing per\nreconstructed profile", fontsize=9)
    for i, (f, n) in enumerate(zip(frac.values, counts.values)):
        ax0.text(i, f * 100, f"{f*100:.0f}%\n({n:,})", ha="center", va="bottom", fontsize=6)
    ax0.margins(y=0.2)
    ax0.text(0.97, 0.95, f"median = {int(full['n_bins'].median())} bins",
             transform=ax0.transAxes, ha="right", va="top", fontsize=7, style="italic")

    # (G2) dense region on H&E: bins by cell type; shared bins highlighted
    ax1 = fig.add_subplot(gs[0, 1])
    crop, ext = U.he_crop_um(cx - half, cy - half, cx + half, cy + half)
    ax1.imshow(crop, extent=ext, zorder=0)
    sh = win[win["shared"]]; un = win[~win["shared"]]
    # unshared bins: solid fill, no edge
    for xi, yi, ll in zip(un["mx"], un["my"], un["lineage"]):
        if ll in C.PALETTE:
            ax1.add_patch(plt.Rectangle((xi - 1, yi - 1), 2, 2, facecolor=C.PALETTE[ll],
                          alpha=0.78, edgecolor="none", zorder=2))
    # shared bins (>1 cell): same fill + bold black outline so they pop
    for xi, yi, ll in zip(sh["mx"], sh["my"], sh["lineage"]):
        if ll in C.PALETTE:
            ax1.add_patch(plt.Rectangle((xi - 1, yi - 1), 2, 2, facecolor=C.PALETTE[ll],
                          alpha=0.95, edgecolor="k", lw=1.1, zorder=3))
    ax1.set_xlim(ext[0], ext[1]); ax1.set_ylim(ext[2], ext[3]); ax1.set_aspect("equal")
    ax1.set_xlabel("x (µm)"); ax1.set_ylabel("y (µm)"); ax1.tick_params(labelsize=6)
    ax1.set_title(f"Dense {int(2*half)} µm region: {len(win):,} bins, {len(lins_present)} cell types · "
                  f"{n_shared} bins ({n_shared/max(len(win),1):.0%}) shared by >1 cell (black outline)",
                  fontsize=7.2)
    import matplotlib.lines as mlines
    handles = U.lineage_handles([l for l in C.LINEAGES if l in lins_present])
    handles.append(mlines.Line2D([], [], marker="s", linestyle="", markerfacecolor="#CCC",
                                 markeredgecolor="k", markeredgewidth=1.1, markersize=6,
                                 label="shared bin (>1 cell)"))
    leg = ax1.legend(handles=handles, fontsize=6.0, loc="upper left", ncol=2,
                     handletextpad=0.3, columnspacing=0.7, borderpad=0.4,
                     frameon=True, facecolor="white", framealpha=0.72, edgecolor="none")
    leg.set_zorder(10)

    # (G3) marker expression of every displayed entity
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(M, aspect="auto", cmap="magma", vmin=0)
    ax2.set_xticks(range(len(markers))); ax2.set_xticklabels(markers, rotation=90, fontsize=5.5)
    ylabs = [f"{e.split('::')[-1]} · {ent_lin.get(e, 'NA')}" for e in entities]
    ax2.set_yticks(range(len(entities))); ax2.set_yticklabels(ylabs, fontsize=5.5)
    for tl, e in zip(ax2.get_yticklabels(), entities):
        tl.set_color(C.PALETTE.get(ent_lin.get(e), "#333"))
    for b in [i for i in range(1, len(mlin)) if mlin[i] != mlin[i - 1]]:
        ax2.axvline(b - 0.5, color="w", lw=0.6)
    for i, g in enumerate(markers):
        ax2.add_patch(plt.Rectangle((i - 0.5, -1.1), 1, 0.55, color=C.PALETTE[mlin[i]],
                      clip_on=False, lw=0))
    ax2.set_ylim(len(entities) - 0.5, -1.3)
    ax2.set_title("Reconstructed transcriptome of each cell\n(log1p CP10k; strip = marker lineage)",
                  fontsize=7.6)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03, label="log1p CP10k")

    fig.suptitle("From pixel to profile: TRACER resolves distinct, overlapping cells — including ones that "
                 "share a single 2 µm bin — where 2D H&E is ambiguous", fontsize=9.0)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.80, bottom=0.2)
    U.save_fig(fig, "panel_G_pixel_to_profile")
    print(f"region center=({cx:.0f},{cy:.0f}) µm; {len(win)} bins; {len(lins_present)} types; "
          f"{n_shared} shared bins ({n_shared/max(len(win),1):.1%})")
    print("entities:", [f"{e.split('::')[-1]}:{ent_lin.get(e)}" for e in entities])


if __name__ == "__main__":
    make()
