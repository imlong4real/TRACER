#!/usr/bin/env python3
"""Re-run TRACER 2um/8um label transfer on the WHOLE-TRANSCRIPTOME matrices.

The original TRACER noseg label transfer used the 1,656-HVG reconstruction
matrix as the query (shared_genes = 1,656). 10x-segmented and bin2cell, by
contrast, were label-transferred on their full ~16k-gene matrices. To put all
four methods on equal footing, we re-run the *same* label-transfer logic
(scripts/label_transfer_spatial.py, cosine-centroid + softmax, clean_marker
anchors) for TRACER using the whole-transcriptome profile-by-gene matrix as
the query.

TRACER-internal NPMI purity/conflict scores remain HVG-derived (merged in from
the original profile_scores.tsv.gz) — only the biological lineage annotation is
upgraded to whole-transcriptome. Outputs go to a *new* directory
``label_transfer_wt/`` so the original HVG label transfer is preserved.

Usage:
    python scripts/reproducibility/fig4/prep/relabel_tracer_whole_transcriptome.py \
        --run kidney_visiumhd_2um kidney_visiumhd_8um
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
REF = ROOT / "results/tracer_noseg/_ref/kidney_ref_noschwann.h5ad"


def run_label_transfer(query_h5ad: Path, outdir: Path, prefix: str, seed: int = 1) -> Path:
    cmd = [
        sys.executable, "scripts/label_transfer_spatial.py",
        "--query_h5ad", str(query_h5ad),
        "--reference_type", "cervical_atera_plus_scrna",
        "--reference_h5ad", str(REF),
        "--label_harmonization", "passthrough",
        "--outdir", str(outdir),
        "--sample_prefix", prefix,
        "--min_transcripts", "5",
        "--max_transcripts", "100000",
        "--anchor_selection", "clean_marker",
        "--max_reference_cells_per_type", "2000",
        "--min_reference_cells_per_type", "50",
        "--low_confidence_threshold", "0.4",
        "--softmax_temperature", "0.05",
        "--random_seed", str(seed),
    ]
    env = {**os.environ, "PYTHONPATH": "src"}
    print("[wt-lt] " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"label_transfer_spatial.py failed (exit {r.returncode})")
    annot = outdir / f"{prefix}_transferred_cell_annotations.csv"
    if not annot.exists():
        raise SystemExit(f"missing annotation output: {annot}")
    return annot


def assemble(annot_csv: Path, scores_tsv: Path, outdir: Path) -> pd.DataFrame:
    annot = pd.read_csv(annot_csv).rename(columns={"cell_id": "reconstructed_profile_id"})
    annot["reconstructed_profile_id"] = annot["reconstructed_profile_id"].astype(str)
    scores = pd.read_csv(scores_tsv, sep="\t")
    scores["reconstructed_profile_id"] = scores["reconstructed_profile_id"].astype(str)
    keep = ["reconstructed_profile_id", "transferred_label", "transfer_confidence",
            "second_label", "second_confidence", "shared_genes"]
    merged = scores.merge(annot[keep], on="reconstructed_profile_id", how="left")
    out = outdir / "reconstructed_profiles_with_labels.tsv.gz"
    merged.to_csv(out, sep="\t", index=False)
    lab = merged.dropna(subset=["transferred_label"])
    (lab["transferred_label"].value_counts(normalize=True)
        .rename_axis("cell_type").reset_index(name="frequency")
        .to_csv(outdir / "celltype_frequency.tsv", sep="\t", index=False))
    print(f"[wt-lt] labeled {len(lab):,}/{len(merged):,}; median shared_genes="
          f"{lab['shared_genes'].median():.0f}; wrote {out}", flush=True)
    print(lab["transferred_label"].value_counts().to_string(), flush=True)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+",
                    default=["kidney_visiumhd_2um", "kidney_visiumhd_8um"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    for run in args.run:
        rdir = ROOT / "results/tracer_noseg" / run
        query = rdir / "outputs/profile_by_gene_whole_transcriptome.h5ad"
        scores = rdir / "outputs/profile_scores.tsv.gz"
        if not query.exists():
            raise SystemExit(f"missing WT matrix: {query} (run build_whole_transcriptome.py)")
        outdir = rdir / "label_transfer_wt"
        outdir.mkdir(parents=True, exist_ok=True)
        annot = run_label_transfer(query, outdir, run, args.seed)
        assemble(annot, scores, outdir)


if __name__ == "__main__":
    main()
