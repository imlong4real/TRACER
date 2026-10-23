"""Are the Xenium-epithelial cells that TRACER calls fibroblast really fibroblasts?

Decisive evidence: transcripts with overlaps_nucleus=True sit inside a nucleus,
so they belong to that cell regardless of how the cytoplasm was segmented.
Scoring markers on NUCLEAR transcripts only gives a segmentation-independent
identity readout.
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns

OUT = C.Path(__file__).parent
MIN_NUC_TX = 20


def marker_sets():
    with open(C.BENCH / "config/panels.yaml") as fh:
        return yaml.safe_load(fh)["marker_sets"]


def score_by_group(df, group_col, ms, min_tx=MIN_NUC_TX):
    """Per-group log-normalised mean expression of each marker set."""
    cnt = df.groupby([group_col, "feature_name"], observed=True).size().rename("n").reset_index()
    tot = cnt.groupby(group_col)["n"].sum().rename("tot")
    cnt = cnt.join(tot, on=group_col)
    cnt["ln"] = np.log1p(cnt["n"] / cnt["tot"] * 1e4)
    out = pd.DataFrame(index=tot.index)
    out["n_tx"] = tot
    for label, genes in ms.items():
        g = cnt.loc[cnt["feature_name"].isin(genes)]
        s = g.groupby(group_col)["ln"].sum() / max(len(genes), 1)
        out[f"s_{label}"] = s.reindex(out.index).fillna(0.0)
    out = out.loc[out["n_tx"] >= min_tx]
    scols = [c for c in out.columns if c.startswith("s_")]
    out["argmax"] = out[scols].idxmax(axis=1).str[2:]
    out["s_max"] = out[scols].max(axis=1)
    return out


def main():
    ms = marker_sets()
    ax = pd.read_csv(C.RUN / "annotations_raw.csv").set_index("entity_id")["annotation"].astype(str)
    at = pd.read_csv(C.RUN / "annotations_tracer_whole.csv").set_index("entity_id")["annotation"].astype(str)

    tdf = standardize_transcript_columns(pd.read_parquet(
        C.RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        columns=["cell_id", "tracer_id", "_etype", "overlaps_nucleus", "feature_name", "qv", "is_gene"]))
    tdf = tdf.loc[tdf["is_gene"].astype(bool) & (tdf["qv"] >= 20)]
    tdf["cell_id"] = tdf["cell_id"].astype(str); tdf["tracer_id"] = tdf["tracer_id"].astype(str)
    nuc = tdf["overlaps_nucleus"].astype(bool)

    # --- identify the converted set: TRACER fibroblast entity, Xenium-epithelial source ---
    cells = tdf.loc[(tdf["_etype"].astype(str) == "cell") & ~tdf["tracer_id"].isin(UNASSIGNED_TOKENS)].copy()
    cells["x_lab"] = cells["cell_id"].map(ax)
    cells["t_lab"] = cells["tracer_id"].map(at)
    dom = cells.groupby("tracer_id")["x_lab"].agg(lambda s: s.value_counts().idxmax() if s.notna().any() else "na")
    tlab = cells.groupby("tracer_id")["t_lab"].first()
    conv = pd.DataFrame({"x_src": dom, "t_lab": tlab})
    converted = conv.index[(conv.t_lab == "fibroblast") & (conv.x_src == "epithelial")]
    conc_fib = conv.index[(conv.t_lab == "fibroblast") & (conv.x_src == "fibroblast")]
    conc_epi = conv.index[(conv.t_lab == "epithelial") & (conv.x_src == "epithelial")]
    print(f"TRACER fibroblast entities from Xenium-EPITHELIAL cells (the set in question): {len(converted)}")
    print(f"TRACER fibroblast entities from Xenium-fibroblast cells (concordant):          {len(conc_fib)}")
    print(f"TRACER epithelial entities from Xenium-epithelial cells (concordant):          {len(conc_epi)}")

    # --- NUCLEAR-ONLY scores, per TRACER entity ---
    ent_nuc = score_by_group(tdf.loc[nuc & (tdf["_etype"].astype(str) == "cell")
                                     & ~tdf["tracer_id"].isin(UNASSIGNED_TOKENS)], "tracer_id", ms)
    ent_all = score_by_group(tdf.loc[(tdf["_etype"].astype(str) == "cell")
                                     & ~tdf["tracer_id"].isin(UNASSIGNED_TOKENS)], "tracer_id", ms)

    groups = {"converted (X:epi -> T:fib)": converted, "concordant fibroblast": conc_fib,
              "concordant epithelial": conc_epi}
    print("\n=== TRACER entities: NUCLEAR-transcript-only marker scores ===")
    rows = []
    for name, ids in groups.items():
        sub = ent_nuc.reindex(ids.astype(str)).dropna(subset=["n_tx"])
        rows.append({"group": name, "n_with>=20_nuclear_tx": len(sub),
                     "median_nuclear_tx": sub["n_tx"].median(),
                     "s_epithelial": sub["s_epithelial"].median(),
                     "s_fibroblast": sub["s_fibroblast"].median(),
                     "argmax=fibroblast_%": 100 * (sub["argmax"] == "fibroblast").mean(),
                     "argmax=epithelial_%": 100 * (sub["argmax"] == "epithelial").mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # --- the same question asked of the XENIUM cells themselves ---
    xdf = tdf.copy()
    xdf = xdf.loc[~xdf["cell_id"].isin(UNASSIGNED_TOKENS)]
    xnuc = score_by_group(xdf.loc[xdf["overlaps_nucleus"].astype(bool)], "cell_id", ms)
    xall = score_by_group(xdf, "cell_id", ms)
    # Xenium cells that are the dominant source of a converted entity
    src_cells = (cells.loc[cells["tracer_id"].isin(converted)]
                 .groupby("tracer_id")["cell_id"].agg(lambda s: s.value_counts().idxmax()))
    src_cells = pd.Index(sorted(set(src_cells))).astype(str)
    epi_cells = pd.Index(ax.index[ax == "epithelial"].astype(str))
    fib_cells = pd.Index(ax.index[ax == "fibroblast"].astype(str))
    print("\n=== XENIUM cells: whole-cell vs NUCLEAR-only marker scores ===")
    rows = []
    for name, ids in (("Xenium-epi cells feeding a converted entity", src_cells),
                      ("all Xenium epithelial cells", epi_cells),
                      ("all Xenium fibroblast cells", fib_cells)):
        a = xall.reindex(ids).dropna(subset=["n_tx"]); n = xnuc.reindex(ids).dropna(subset=["n_tx"])
        rows.append({"group": name, "n": len(a),
                     "whole_s_epi": a["s_epithelial"].median(), "whole_s_fib": a["s_fibroblast"].median(),
                     "whole_argmax_epi_%": 100 * (a["argmax"] == "epithelial").mean(),
                     "nuc_n": len(n), "nuc_median_tx": n["n_tx"].median(),
                     "nuc_s_epi": n["s_epithelial"].median(), "nuc_s_fib": n["s_fibroblast"].median(),
                     "nuc_argmax_epi_%": 100 * (n["argmax"] == "epithelial").mean(),
                     "nuc_argmax_fib_%": 100 * (n["argmax"] == "fibroblast").mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    ent_nuc.to_csv(OUT / "nuclear_scores_tracer_entities.csv")
    xnuc.to_csv(OUT / "nuclear_scores_xenium_cells.csv")
    pd.DataFrame({"tracer_id": converted}).to_csv(OUT / "converted_entities.csv", index=False)
    print("\nwrote nuclear_scores_*.csv, converted_entities.csv")


if __name__ == "__main__":
    main()
