"""Segmentation-level comparison, raw Xenium vs TRACER. No CNV, no clones.

Q1: do the two arms yield a similar number of cells?
Q2: how do cell-type proportions change?
"""
from __future__ import annotations
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
from common import UNASSIGNED_TOKENS, standardize_transcript_columns

OUT = C.Path(__file__).parent
QC_TX, QC_GENES = 20, 10
LABELS = ["epithelial", "fibroblast", "myeloid", "t_cell", "b_cell", "endothelial", "unknown"]


def load(arm):
    spec = C.ARMS[arm]
    df = standardize_transcript_columns(pd.read_parquet(spec["transcripts"]))
    df["cell_id"] = df["cell_id"].astype(str)
    ecol = spec["entity_col"]
    df[ecol] = df[ecol].astype(str)
    return df, ecol


def qc_table(df, ecol):
    g = df.groupby(ecol, observed=True)
    n_tx = g.size()
    n_gene = g["feature_name"].nunique()
    t = pd.DataFrame({"n_tx": n_tx, "n_genes": n_gene})
    t = t.loc[~t.index.isin(UNASSIGNED_TOKENS)]
    t["passes_qc"] = (t.n_tx >= QC_TX) & (t.n_genes >= QC_GENES)
    return t


def main():
    rows, prop_rows = [], {}
    for arm in ("raw", "tracer_whole"):
        df, ecol = load(arm)
        total_tx = len(df)
        unassigned = int(df[ecol].isin(UNASSIGNED_TOKENS).sum())
        sub = df.loc[~df[ecol].isin(UNASSIGNED_TOKENS)]
        rec = {"arm": arm, "entity_col": ecol, "total_transcripts": total_tx,
               "transcripts_unassigned": unassigned,
               "pct_transcripts_unassigned": 100 * unassigned / total_tx}

        if "_etype" in df.columns:
            et = df["_etype"].astype(str).value_counts()
            rec["tx_by_etype"] = et.to_dict()
            # entity counts per etype (whole cells vs partials)
            for e in ("cell", "partial"):
                m = df["_etype"].astype(str) == e
                rec[f"n_entities_{e}"] = int(df.loc[m & ~df[ecol].isin(UNASSIGNED_TOKENS), ecol].nunique())
            sub = df.loc[(df["_etype"].astype(str) == "cell") & ~df[ecol].isin(UNASSIGNED_TOKENS)]

        t = qc_table(sub, ecol)
        rec.update({
            "n_entities_preQC": int(len(t)),
            "n_entities_postQC": int(t.passes_qc.sum()),
            "pct_lost_to_QC": 100 * (1 - t.passes_qc.mean()),
            "median_tx": float(t.loc[t.passes_qc, "n_tx"].median()),
            "median_genes": float(t.loc[t.passes_qc, "n_genes"].median()),
            "transcripts_in_postQC_entities": int(t.loc[t.passes_qc, "n_tx"].sum()),
        })
        rec["pct_all_transcripts_retained"] = 100 * rec["transcripts_in_postQC_entities"] / total_tx
        rows.append(rec)

        ann = pd.read_csv(C.RUN / f"annotations_{arm}.csv")
        keep = set(t.index[t.passes_qc])
        ann = ann.loc[ann["entity_id"].astype(str).isin(keep)]
        vc = ann["annotation"].astype(str).value_counts()
        prop_rows[arm] = vc

    summary = pd.DataFrame(rows).set_index("arm").T
    print("=== Q1: segmentation output ===")
    print(summary.to_string())

    print("\n=== Q2: cell-type composition (marker annotation, post-QC entities) ===")
    comp = pd.DataFrame(prop_rows).reindex(LABELS).fillna(0).astype(int)
    pct = (comp / comp.sum() * 100).round(1)
    truth = pd.read_csv(C.BENCH / "data/roi_cell_metadata.csv")["cell_class"].value_counts()
    tmap = {"Epithelial": "epithelial", "Fibroblasts": "fibroblast", "Monocytes": "myeloid",
            "T cells": "t_cell", "B cells": "b_cell", "Endothelial": "endothelial",
            "Low expression": "unknown"}
    tr = truth.rename(index=tmap).groupby(level=0).sum().reindex(LABELS).fillna(0).astype(int)
    out = pd.DataFrame({
        "full_slide_cell_class_n": tr,
        "full_slide_%": (tr / tr.sum() * 100).round(1),
        "raw_n": comp["raw"], "raw_%": pct["raw"],
        "tracer_n": comp["tracer_whole"], "tracer_%": pct["tracer_whole"],
    })
    out.loc["TOTAL"] = [tr.sum(), 100.0, comp["raw"].sum(), 100.0, comp["tracer_whole"].sum(), 100.0]
    print(out.to_string())
    print("\nNote: the full-slide `cell_class` column is an expression-Leiden labelling of the ORIGINAL")
    print("segmentation, so it is not an independent yardstick -- it shares raw's contamination.")

    # depth-controlled composition: entities above a common absolute depth
    print("\n=== Q2b: composition restricted to entities with n_genes >= 700 (common threshold) ===")
    dep = {}
    for arm in ("raw", "tracer_whole"):
        df, ecol = load(arm)
        sub = df.loc[~df[ecol].isin(UNASSIGNED_TOKENS)]
        if "_etype" in df.columns:
            sub = sub.loc[sub["_etype"].astype(str) == "cell"]
        t = qc_table(sub, ecol)
        keep = set(t.index[(t.passes_qc) & (t.n_genes >= 700)])
        ann = pd.read_csv(C.RUN / f"annotations_{arm}.csv")
        ann = ann.loc[ann["entity_id"].astype(str).isin(keep)]
        dep[arm] = ann["annotation"].astype(str).value_counts()
    d = pd.DataFrame(dep).reindex(LABELS).fillna(0).astype(int)
    dp = (d / d.sum() * 100).round(1)
    dd = pd.DataFrame({"raw_n": d["raw"], "raw_%": dp["raw"],
                       "tracer_n": d["tracer_whole"], "tracer_%": dp["tracer_whole"]})
    dd.loc["TOTAL"] = [d["raw"].sum(), 100.0, d["tracer_whole"].sum(), 100.0]
    print(dd.to_string())

    summary.to_csv(OUT / "seg_compare_summary.csv")
    out.to_csv(OUT / "seg_compare_composition.csv")
    dd.to_csv(OUT / "seg_compare_composition_depth700.csv")
    print("\nwrote seg_compare_*.csv")


if __name__ == "__main__":
    main()
