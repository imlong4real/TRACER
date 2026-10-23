"""Isolate the PARTIAL entities that receive the excised epithelial material
from the 78 switcher cells, and run inferCNV on them.

Reference stays the arm's whole-cell reference compartment, so the baseline is
unchanged. Partials are scored but never used as reference.
"""
from __future__ import annotations
import sys, warnings, pickle
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
import scanpy as sc; sc.settings.verbosity = 0
import cnv_review_common as C
sys.path.insert(0, str(C.BENCH / "scripts"))
import run_insitucnv_arm as R
from common import UNASSIGNED_TOKENS, standardize_transcript_columns
from fig_switch_cnv import id_groups, CAT

OUT = C.Path(__file__).parent
ms = yaml.safe_load(open(C.BENCH / "config/panels.yaml"))["marker_sets"]
EPI = set(map(str, ms["epithelial"]))


def main():
    g = id_groups()
    SW, CF, CE = list(g[CAT[0]]), list(g[CAT[1]]), list(g[CAT[2]])
    at = pd.read_csv(C.RUN / "annotations_tracer_whole.csv", dtype={"entity_id": str}).set_index("entity_id")["annotation"].astype(str)

    df = standardize_transcript_columns(pd.read_parquet(C.ARMS["tracer_whole"]["transcripts"]))
    df["cell_id"] = df["cell_id"].astype(str); df["tracer_id"] = df["tracer_id"].astype(str)
    et = df["_etype"].astype(str)
    df = df.loc[~df["tracer_id"].isin(UNASSIGNED_TOKENS) & et.isin(["cell", "partial"])].copy()
    et = df["_etype"].astype(str)
    part_ids = set(df.loc[et == "partial", "tracer_id"].unique())

    # excised material from the switchers, and which PARTIALS receive it
    sub = df.loc[df["cell_id"].isin(SW)].copy()
    out = sub.loc[sub["cell_id"] != sub["tracer_id"]].copy()
    out["is_epi"] = out["feature_name"].astype(str).isin(EPI)
    rec = (out.loc[out["tracer_id"].isin(part_ids)]
           .groupby("tracer_id")
           .agg(n_from_switchers=("tracer_id", "size"),
                n_epi_markers=("is_epi", "sum"),
                n_switcher_cells=("cell_id", "nunique")))
    rec["annotation"] = rec.index.map(at)
    tot = df.groupby("tracer_id").size().rename("n_total")
    rec = rec.join(tot)
    rec["frac_from_switchers"] = rec["n_from_switchers"] / rec["n_total"]
    rec = rec.sort_values("n_from_switchers", ascending=False)
    epi_part = rec.loc[rec["annotation"] == "epithelial"]
    print(f"PARTIAL entities receiving excised switcher material : {len(rec)}")
    print(f"  of which annotated epithelial                      : {len(epi_part)}")
    print("\ntop 15 epithelial PARTIAL recipients:")
    print(epi_part.head(15)[["n_total", "n_from_switchers", "frac_from_switchers",
                             "n_epi_markers", "n_switcher_cells"]].round(3).to_string())

    # "big" recipients: enough transcripts to survive QC and a real share from switchers
    big = epi_part.loc[(epi_part.n_total >= 100) & (epi_part.n_from_switchers >= 50)]
    print(f"\n'big' epithelial partial recipients (>=100 tx, >=50 from switchers): {len(big)}")

    ann = pd.read_csv(C.RUN / "annotations_tracer_whole.csv")
    a = R.build_adata(df, "tracer_id", ann, 20, 10)
    a = R.add_gene_positions(a, C.BENCH / "data/gene_positions_grch38.tsv")
    R.assign_compartment(a, {"t_cell", "b_cell", "myeloid", "fibroblast", "endothelial"}, {"epithelial"})
    whole = set(df.loc[et == "cell", "tracer_id"].unique())
    nm = a.obs_names.astype(str)
    is_part = ~np.isin(nm, list(whole))
    a.obs.loc[is_part, "compartment"] = "unknown"      # scored, never reference
    print(f"\nentities in the run: {a.n_obs} ({int(is_part.sum())} partial); "
          f"compartments {a.obs['compartment'].value_counts().to_dict()}")

    R.run_infercnv(a, 60, 0.4, 100, smoothing_graph="expression",
                   expression_neighbors=100, neighbor_n_pcs=50)

    X = C.dense_cnv(a); nm = a.obs_names.astype(str); pos = {v: i for i, v in enumerate(nm)}

    def prof(ids):
        r = [pos[i] for i in ids if i in pos]
        return (X[r].mean(0) if r else np.full(X.shape[1], np.nan)), r

    pe, _ = prof(CE); base = pe - pe.mean()

    def cc(v):
        v = v - v.mean()
        return float(v @ base / (np.linalg.norm(v) * np.linalg.norm(base) + 1e-12))

    other_part = [i for i in nm[is_part] if i not in set(epi_part.index)]
    sets = {
        "concordant epithelial (whole)": CE,
        "concordant fibroblast (whole)": CF,
        "switcher REMNANT (same id -> 'fibroblast')": SW,
        "epithelial PARTIAL recipients (all)": list(epi_part.index),
        "epithelial PARTIAL recipients (big)": list(big.index),
        "all other partials": other_part,
    }
    rows = []
    for k, ids in sets.items():
        p, r = prof(ids)
        rows.append(dict(group=k, n=len(r), RMS=np.sqrt(np.nanmean(p ** 2)),
                         r_vs_tumour=cc(p) if len(r) else np.nan,
                         med_genes=float(np.median(a.obs["n_genes"].to_numpy()[r])) if r else np.nan))
    res = pd.DataFrame(rows)
    print("\nmean CNV profile vs the concordant-epithelial (tumour) profile:")
    print(res.round(4).to_string(index=False))

    with open(OUT / "_partials_run.pkl", "wb") as fh:
        pickle.dump({"X": X, "names": nm, "obs": a.obs[["annotation", "compartment", "n_genes"]],
                     "var": a.var[["chromosome", "start", "end"]],
                     "chr_pos": {k: int(v) for k, v in a.uns["cnv"]["chr_pos"].items()},
                     "spatial": a.obsm["spatial"], "sets": sets,
                     "epi_part": epi_part, "big": list(big.index)}, fh)
    res.to_csv(OUT / "partials_cnv_summary.csv", index=False)
    rec.to_csv(OUT / "partial_recipients.csv")
    print("\nwrote _partials_run.pkl, partials_cnv_summary.csv, partial_recipients.csv")


if __name__ == "__main__":
    main()
