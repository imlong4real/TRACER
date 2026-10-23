"""Is there ANY plane that separates the remnant from the partial?

Per-axis separation (x, y, z) as a 1D AUC, plus the best linear direction found
by Fisher LDA and scored on held-out transcripts -- because x/y/z are arbitrary
lab axes and a cell split along a diagonal would look unseparated on all three.

AUC is folded to [0.5, 1]: 0.5 = the two clouds are indistinguishable along that
direction, 1.0 = perfectly separated by a plane. Controls are the adjacent
whole-cell pairs, which must separate if the metric works at all.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import standardize_transcript_columns

OUT = C.Path(__file__).parent
RNG = np.random.default_rng(0)
CR, CP = "#d73027", "#1b7837"
EXAMPLE = "dplafofh-1"


def auc1d(a, b):
    """Folded AUC of a 1D projection: 0.5 = indistinguishable, 1 = separated."""
    if len(a) < 5 or len(b) < 5:
        return np.nan
    u = mannwhitneyu(a, b).statistic / (len(a) * len(b))
    return max(u, 1 - u)


def lda_auc(A, B, rng, n_rep=5):
    """Fisher LDA direction fit on half the points, AUC scored on the other half."""
    out, ws = [], []
    for _ in range(n_rep):
        ia, ib = rng.permutation(len(A)), rng.permutation(len(B))
        ha, hb = ia[: len(A) // 2], ib[: len(B) // 2]
        ta, tb = ia[len(A) // 2:], ib[len(B) // 2:]
        fa, fb = A[ha], B[hb]
        Sw = np.cov(fa.T) * (len(fa) - 1) + np.cov(fb.T) * (len(fb) - 1)
        Sw /= (len(fa) + len(fb) - 2)
        Sw += np.eye(3) * 1e-6
        w = np.linalg.solve(Sw, fa.mean(0) - fb.mean(0))
        w /= np.linalg.norm(w)
        ws.append(w)
        out.append(auc1d(A[ta] @ w, B[tb] @ w))
    return float(np.nanmean(out)), ws[0]


def load_xyz():
    t = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "_etype", "x", "y", "z"]))
    t["cell_id"] = t["cell_id"].astype(str); t["tracer_id"] = t["tracer_id"].astype(str)
    return t


def main():
    sw = pd.read_csv(OUT / "switcher_pairs_3d.csv")
    ctl = pd.read_csv(OUT / "control_adjacent_pairs.csv")
    t = load_xyz()

    rows = []
    # --- switcher pairs: both components inside ONE Xenium cell ---
    for r in sw.itertuples(index=False):
        g = t.loc[t["cell_id"] == r.cell_id]
        A = g.loc[g["tracer_id"] == r.partial][["x", "y", "z"]].to_numpy(float)
        B = g.loc[g["tracer_id"] == r.cell_id][["x", "y", "z"]].to_numpy(float)
        if len(A) < 20 or len(B) < 20:
            continue
        best, _ = lda_auc(A, B, np.random.default_rng(abs(hash(r.cell_id)) % 2**31))
        yp = RNG.permutation(np.r_[np.zeros(len(A)), np.ones(len(B))]).astype(bool)
        P = np.vstack([A, B])
        null, _ = lda_auc(P[yp], P[~yp], np.random.default_rng(7))
        rows.append(dict(group="switcher", pair=r.cell_id,
                         auc_x=auc1d(A[:, 0], B[:, 0]), auc_y=auc1d(A[:, 1], B[:, 1]),
                         auc_z=auc1d(A[:, 2], B[:, 2]), auc_best=best, auc_null=null))

    # --- control: adjacent whole cells ---
    w = t.loc[t["_etype"].astype(str) == "cell"]
    by = dict(tuple(w.groupby("tracer_id")))
    for r in ctl.itertuples(index=False):
        A = by[r.A][["x", "y", "z"]].to_numpy(float)
        B = by[r.B][["x", "y", "z"]].to_numpy(float)
        if len(A) < 20 or len(B) < 20:
            continue
        best, _ = lda_auc(A, B, np.random.default_rng(abs(hash(r.A)) % 2**31))
        rows.append(dict(group=f"control_{r.tier}", pair=f"{r.A}|{r.B}",
                         auc_x=auc1d(A[:, 0], B[:, 0]), auc_y=auc1d(A[:, 1], B[:, 1]),
                         auc_z=auc1d(A[:, 2], B[:, 2]), auc_best=best, auc_null=np.nan))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "axis_separation.csv", index=False)

    order = ["control_typical", "control_crowded", "switcher"]
    LAB = {"control_typical": "CONTROL typical\ntwo whole cells",
           "control_crowded": "CONTROL crowded\n40 closest",
           "switcher": "SWITCHERS\nremnant vs partial"}
    COL = {"control_typical": "#4575b4", "control_crowded": "#74add1", "switcher": "#984ea3"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    # (a) per-axis AUC
    ax = axes[0]
    W = 0.24
    off = {"auc_x": -W, "auc_y": 0.0, "auc_z": W}
    for ai, (col, nm) in enumerate([("auc_x", "x"), ("auc_y", "y"), ("auc_z", "z")]):
        for gi, gname in enumerate(order):
            v = res.loc[res["group"] == gname, col].dropna().values
            pos = gi + 1 + off[col]
            bp = ax.boxplot([v], positions=[pos], widths=W * 0.8, patch_artist=True,
                            showfliers=False, medianprops=dict(color="k", lw=1.3))
            bp["boxes"][0].set_facecolor(COL[gname]); bp["boxes"][0].set_alpha(0.35)
            bp["boxes"][0].set_edgecolor(COL[gname])
            ax.scatter(np.full(len(v), pos) + RNG.normal(0, 0.03, len(v)), v, s=10,
                       color=COL[gname], alpha=0.7, linewidths=0)
            ax.text(pos, 1.035, nm, ha="center", fontsize=9, color="0.3")
    ax.axhline(0.5, color="0.5", ls="--", lw=1.0, label="0.5 = indistinguishable")
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels([LAB[g] for g in order], fontsize=8.5)
    ax.set_ylim(0.45, 1.06); ax.set_ylabel("1D separation AUC (folded)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("(a) separation along each lab axis", fontsize=10, loc="left")

    # (b) best possible direction
    ax = axes[1]
    vals = [res.loc[res["group"] == g, "auc_best"].dropna().values for g in order]
    vals.append(res.loc[res["group"] == "switcher", "auc_null"].dropna().values)
    cols = [COL[g] for g in order] + ["0.6"]
    labs = [LAB[g] for g in order] + ["switcher,\nlabels shuffled"]
    bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.5))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    for k, (v, c) in enumerate(zip(vals, cols), start=1):
        ax.scatter(np.full(len(v), k) + RNG.normal(0, 0.06, len(v)), v, s=16, color=c,
                   alpha=0.8, linewidths=0)
    ax.axhline(0.5, color="0.5", ls="--", lw=1.0)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(labs, fontsize=8.3)
    ax.set_ylim(0.45, 1.03); ax.set_ylabel("held-out AUC along the BEST direction")
    med = "  |  ".join(f"{np.median(v):.2f}" for v in vals)
    ax.set_title(f"(b) best separating plane, any orientation\nmedians {med}",
                 fontsize=10, loc="left")

    # (c) the example cell projected onto its own best direction
    ax = axes[2]
    g = t.loc[t["cell_id"] == EXAMPLE]
    pid = sw.set_index("cell_id").loc[EXAMPLE, "partial"]
    A = g.loc[g["tracer_id"] == pid][["x", "y", "z"]].to_numpy(float)
    B = g.loc[g["tracer_id"] == EXAMPLE][["x", "y", "z"]].to_numpy(float)
    best, wv = lda_auc(A, B, np.random.default_rng(1))
    pa, pb = A @ wv, B @ wv
    lo, hi = min(pa.min(), pb.min()), max(pa.max(), pb.max())
    bins = np.linspace(lo, hi, 40)
    ax.hist(pb, bins=bins, color=CR, alpha=0.6, label=f"remnant ({len(B)} tx)")
    ax.hist(pa, bins=bins, color=CP, alpha=0.6, label=f"partial ({len(A)} tx)")
    ax.set_xlabel("position along the best separating direction (um)")
    ax.set_ylabel("transcripts")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title(f"(c) {EXAMPLE}, optimally oriented\nheld-out AUC {best:.2f} - the best "
                 f"plane still cannot cut it", fontsize=10, loc="left")

    fig.suptitle("Is there any plane that separates the two components? "
                 "Per-axis and best-direction separation", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p = OUT / "fig_axis_separation.png"; fig.savefig(p, dpi=150); print("wrote", p)

    print(res.groupby("group")[["auc_x", "auc_y", "auc_z", "auc_best", "auc_null"]]
          .median().round(3).to_string())
    print("\nexample", EXAMPLE, "best direction w =", np.round(wv, 3),
          f"held-out AUC {best:.3f}")


if __name__ == "__main__":
    main()
