#!/usr/bin/env python
"""Merge per-arm RCTD output into one per-entity table plus arm summaries.

Emits the analysis-ready artefacts:

  rctd_entities.parquet / .csv.gz   one row per entity, every arm stacked
  rctd_weights_<arm>.tsv.gz         full normalised weight matrices (copied)
  rctd_arm_summary.tsv              per-arm medians / counts
  rctd_settings.json                the single settings block used by all arms
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ASSIGN_NAME = "rctd_cell_assignments_post.tsv"
WEIGHTS_NAME = "rctd_weights_post.tsv.gz"

OUT_COLS = [
    "sample", "patient", "section", "arm", "entity_type",
    "entity_id", "original_cell_id", "tracer_id",
    "gbmap_celltype", "max_weight", "entropy",
    "n_tx", "n_umi_used", "x_centroid", "y_centroid", "doublet_status",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-name", required=True)
    p.add_argument("--prep-manifest", required=True)
    p.add_argument("--settings-b64", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--arm-dirs", nargs="+", required=True)
    p.add_argument("--entity-h5ads", nargs="+", required=True)
    return p.parse_args()


def arm_of_dir(path: Path) -> str:
    return path.name.replace("rctd_", "", 1)


def arm_of_h5ad(path: Path) -> str:
    return path.stem.replace("entities_", "", 1)


def main() -> None:
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.prep_manifest).read_text())
    settings = json.loads(base64.b64decode(args.settings_b64).decode())

    h5ads = {arm_of_h5ad(Path(p)): Path(p) for p in args.entity_h5ads}
    frames, summaries = [], []

    for d in sorted(Path(p) for p in args.arm_dirs):
        arm = arm_of_dir(d)
        assign = d / ASSIGN_NAME
        if not assign.exists():
            print(f"[collect] WARNING: {assign} missing; arm {arm} produced no "
                  f"assignments", flush=True)
            continue
        a = pd.read_csv(assign, sep="\t")
        a = a.rename(columns={"cell_id": "entity_id",
                              "dominant_celltype": "gbmap_celltype"})

        obs = pd.DataFrame()
        if arm in h5ads:
            ent = ad.read_h5ad(h5ads[arm], backed="r")
            obs = ent.obs.copy()
            obs.index.name = "entity_id"
            obs = obs.reset_index()
            del ent

        df = a.merge(obs, on="entity_id", how="left") if len(obs) else a
        df["arm"] = arm
        df["sample"] = manifest.get("sample", args.sample_name)
        if "patient" not in df or df["patient"].isna().all():
            df["patient"] = manifest.get("patient", "")
        if "section" not in df or df["section"].isna().all():
            df["section"] = manifest.get("section", "")
        if "entity_type" not in df:
            df["entity_type"] = {"original": "original", "post_whole": "whole",
                                 "post_partial": "partial",
                                 "post_all": "whole_or_partial"}.get(arm, arm)
        # tracer_id is the entity label for every post-TRACER arm; the
        # pre-TRACER arm has no TRACER identity, only the input cell_id.
        df["tracer_id"] = np.where(df["arm"] == "original", "", df["entity_id"])
        if "original_cell_id" not in df:
            df["original_cell_id"] = df["entity_id"]

        w = d / WEIGHTS_NAME
        if w.exists():
            shutil.copyfile(w, out / f"rctd_weights_{arm}.tsv.gz")
            head = pd.read_csv(w, sep="\t", nrows=1)
            n_types = max(len([c for c in head.columns if c != "cell_id"]), 0)
        else:
            n_types = 0
        df["n_umi_used"] = df.get("n_tx", pd.Series(np.nan, index=df.index))

        for c in OUT_COLS:
            if c not in df:
                df[c] = np.nan
        frames.append(df[OUT_COLS])

        ent_total = manifest.get("arms", {}).get(arm, {}).get("n_entities", np.nan)
        summaries.append({
            "sample": manifest.get("sample", args.sample_name),
            "patient": manifest.get("patient", ""),
            "section": manifest.get("section", ""),
            "arm": arm,
            "entity_type": df["entity_type"].iloc[0] if len(df) else "",
            "n_entities_built": ent_total,
            "n_entities_scored": int(len(df)),
            "frac_scored": (len(df) / ent_total) if ent_total else np.nan,
            "n_celltypes": n_types,
            "median_entropy": float(np.nanmedian(df["entropy"])) if len(df) else np.nan,
            "mean_entropy": float(np.nanmean(df["entropy"])) if len(df) else np.nan,
            "median_max_weight": float(np.nanmedian(df["max_weight"])) if len(df) else np.nan,
            "mean_max_weight": float(np.nanmean(df["max_weight"])) if len(df) else np.nan,
            "median_n_tx": float(np.nanmedian(df["n_tx"])) if len(df) else np.nan,
        })
        print(f"[collect] {arm:<13} scored={len(df):>9,}  "
              f"median_entropy={summaries[-1]['median_entropy']:.4f}  "
              f"median_max_w={summaries[-1]['median_max_weight']:.4f}", flush=True)

    if not frames:
        raise SystemExit("no RCTD arm produced assignments; nothing to collect")

    allf = pd.concat(frames, ignore_index=True)
    allf.to_parquet(out / "rctd_entities.parquet", index=False,
                    compression="snappy")
    allf.to_csv(out / "rctd_entities.csv.gz", index=False, compression="gzip")

    summ = pd.DataFrame(summaries)
    summ.to_csv(out / "rctd_arm_summary.tsv", sep="\t", index=False)

    (out / "rctd_settings.json").write_text(json.dumps(
        {"settings_shared_by_all_arms": settings, "prep_manifest": manifest},
        indent=2))
    print(f"[collect] wrote {len(allf):,} entity rows across "
          f"{allf['arm'].nunique()} arms -> {out}", flush=True)


if __name__ == "__main__":
    main()
