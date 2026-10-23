"""Shared loading for the scratchpad CNV review figures.

Not committed: a review scratchpad. Reads the two finished arm h5ads read-only,
joins the full-slide `cnv_epi` labels onto each arm's entities, and reconstructs
per-window genomic intervals so the two arms (354 vs 348 windows) can be drawn
on a common bp axis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[4]   # .../insitucnv_tracer_benchmark
RUN = BENCH / "output/runs/roi_nuclear_grid3cpmi"
sys.path.insert(0, str(BENCH / "scripts"))

from common import cnv_bin_chromosomes, chromosome_rank  # noqa: E402

ARMS = {
    "raw": {
        "h5ad": RUN / "insitucnv_raw/adata_cnv.h5ad",
        "transcripts": BENCH / "data/roi_transcripts.parquet",
        "entity_col": "cell_id",
    },
    "tracer_whole": {
        "h5ad": RUN / "insitucnv_tracer_whole/adata_cnv.h5ad",
        "transcripts": RUN / "tracer/outputs/transcripts_tracer_refined.parquet",
        "entity_col": "tracer_id",
    },
}
TRUTH_CSV = BENCH / "data/roi_cell_metadata.csv"
TRUTH_COL = "cnv_epi"

# infercnvpy defaults actually used by run_insitucnv_arm.py
WINDOW_SIZE = 60
STEP = 10
EXCLUDE_CHROMOSOMES = ("chrX", "chrY")


def load_truth() -> pd.Series:
    meta = pd.read_csv(TRUTH_CSV, dtype={"cell_id": str})
    return meta.set_index("cell_id")[TRUTH_COL].astype(str)


def truth_by_entity(arm: str, entities: pd.Index) -> pd.DataFrame:
    """Transcript-count-weighted majority vote of cnv_epi onto arm entities."""
    spec = ARMS[arm]
    truth = load_truth()
    ecol = spec["entity_col"]
    cols = [ecol, "cell_id"] if ecol != "cell_id" else ["cell_id"]
    df = pd.read_parquet(spec["transcripts"], columns=list(dict.fromkeys(cols)))
    if ecol == "cell_id":
        df = df.rename(columns={"cell_id": "_entity"})
        df["cell_id"] = df["_entity"]
    else:
        df = df.rename(columns={ecol: "_entity"})
    df["_entity"] = df["_entity"].astype(str)
    df["cell_id"] = df["cell_id"].astype(str)
    df["_truth"] = df["cell_id"].map(truth)
    df = df.dropna(subset=["_truth"])

    vc = df.groupby(["_entity", "_truth"], observed=True).size().rename("n").reset_index()
    tot = vc.groupby("_entity")["n"].sum().rename("tot")
    # deterministic argmax: highest n, then lexicographically smallest label
    vc = vc.sort_values(["n", "_truth"], ascending=[True, False])
    top = vc.groupby("_entity").tail(1).set_index("_entity")
    out = pd.DataFrame(index=entities.astype(str))
    out["truth_clone"] = top["_truth"].reindex(out.index).fillna("no_truth")
    out["truth_purity"] = (top["n"] / tot).reindex(out.index)
    out["truth_n_cells"] = df.groupby("_entity")["cell_id"].nunique().reindex(out.index)
    return out


def window_intervals(var: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct each X_cnv column's genomic interval from var.

    Mirrors infercnvpy._running_mean: per chromosome, genes sorted by start, a
    length-60 pyramid kernel convolved with mode="valid" (one gene per step),
    then subsampled every 10. Column j therefore spans gene ranks
    [10j, 10j+59]. infercnvpy stores no coordinates, so this is the only way to
    put two runs with different gene sets on a common axis.
    """
    # infercnv() drops `exclude_chromosomes` (default chrX, chrY) BEFORE
    # windowing, and _running_mean_by_chromosome additionally skips chrM. var
    # still carries the excluded genes, so filtering here is required.
    chroms = [
        c
        for c in var["chromosome"].unique()
        if str(c).startswith("chr") and c not in EXCLUDE_CHROMOSOMES and c != "chrM"
    ]
    chroms = sorted(chroms, key=chromosome_rank)
    rows = []
    for c in chroms:
        g = var.loc[var["chromosome"] == c].sort_values("start")
        n = len(g)
        starts = g["start"].to_numpy()
        ends = g["end"].to_numpy()
        if n < WINDOW_SIZE:
            # _running_mean's else-branch: window size collapses to the gene
            # count and a single flat-kernel convolution yields ONE column
            # spanning the whole chromosome (chr21 here, 57 genes).
            rows.append(
                {"chromosome": c, "bp_start": int(starts[0]), "bp_end": int(ends[-1]), "n_genes_chrom": n}
            )
            continue
        for j in np.arange(0, n - WINDOW_SIZE + 1, STEP):
            rows.append(
                {
                    "chromosome": c,
                    "bp_start": int(starts[j]),
                    "bp_end": int(ends[j + WINDOW_SIZE - 1]),
                    "n_genes_chrom": n,
                }
            )
    out = pd.DataFrame(rows)
    out["bp_mid"] = (out["bp_start"] + out["bp_end"]) / 2.0
    return out


def check_windows(adata, label: str) -> pd.DataFrame:
    """Reconstruct intervals and assert they reproduce uns['cnv']['chr_pos']."""
    wins = window_intervals(adata.var)
    n_cnv = adata.obsm["X_cnv"].shape[1]
    if len(wins) != n_cnv:
        raise SystemExit(f"{label}: reconstructed {len(wins)} windows, X_cnv has {n_cnv}")
    counts = wins.groupby("chromosome", sort=False).size()
    starts = counts.cumsum().shift(1).fillna(0).astype(int)
    chr_pos = {k: int(v) for k, v in adata.uns["cnv"]["chr_pos"].items()}
    for c, s in starts.items():
        if chr_pos.get(c) != s:
            raise SystemExit(f"{label}: chr_pos mismatch {c}: stored {chr_pos.get(c)} vs rebuilt {s}")
    print(f"[{label}] windows OK: {n_cnv} columns, chr_pos reproduced exactly")
    return wins


def load_arm(arm: str):
    import anndata as ad

    adata = ad.read_h5ad(ARMS[arm]["h5ad"])
    wins = check_windows(adata, arm)
    truth = truth_by_entity(arm, adata.obs_names)
    for col in truth.columns:
        adata.obs[col] = truth[col].to_numpy()
    return adata, wins


def dense_cnv(adata) -> np.ndarray:
    X = adata.obsm["X_cnv"]
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
