#!/usr/bin/env python
"""Build per-arm entity x gene AnnData objects from TRACER refined transcripts.

Four arms are emitted, all on an identical, caller-supplied gene axis so that
RCTD sees exactly the same feature space in every arm:

  original      pre-TRACER cells, grouped by the pristine input ``cell_id``
  post_whole    post-TRACER whole cells   (``_etype == 'cell'``)
  post_partial  post-TRACER partial cells (``_etype == 'partial'``)
  post_all      post-TRACER whole + partial entities (visualisation arm)

Every arm is built from the *same* transcript table, so QV filtering and
control-probe removal are shared by construction and the pre/post contrast is
paired on ``original_cell_id``.

Transcripts are streamed row-group by row-group; nothing larger than one row
group is ever materialised.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

# Counts are stored as float32, not an integer dtype: R's anndata reader
# maps an integer X onto a dgRMatrix whose "x" slot must be double, and
# Matrix rejects the object outright ("'x' slot is not of type \"double\"").
COUNT_DTYPE = np.float32

ENTITY_DELIMITER = "-tr-"
NULL_LABELS = {"UNASSIGNED", "-1", "", "nan", "NaN", "None", "DROP"}
ARMS = ("original", "post_whole", "post_partial", "post_all")

# Columns actually needed; keeps the streamed footprint small.
REQUIRED = ["cell_id", "tracer_id", "_etype", "feature_name"]
# TRACER's refined output names the coordinates x/y; a standardised Xenium
# transcript table names them x_location/y_location. Accept either, and say
# which was used, rather than silently centroid-ing everything to the origin.
X_ALIASES = ("x", "x_location", "x_centroid")
Y_ALIASES = ("y", "y_location", "y_centroid")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcripts", required=True,
                   help="transcripts_tracer_refined.parquet from TRACER Seg")
    p.add_argument("--genes", required=True,
                   help="Newline-delimited panel-matched gene list (the RCTD gene axis)")
    p.add_argument("--sample-name", required=True)
    p.add_argument("--patient", default="")
    p.add_argument("--section", default="")
    p.add_argument("--outdir", required=True)
    p.add_argument("--arms", default=",".join(ARMS))
    return p.parse_args()


def modal_parent(counts: dict[str, int]) -> str:
    """Most common pristine input cell_id among an entity's transcripts."""
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


class ArmAccumulator:
    """Streams (entity, gene) counts and centroid sums into COO fragments."""

    def __init__(self, name: str, gene_index: dict[str, int], n_genes: int):
        self.name = name
        self.gene_index = gene_index
        self.n_genes = n_genes
        self.labels: dict[str, int] = {}
        self.rows: list[np.ndarray] = []
        self.cols: list[np.ndarray] = []
        self.vals: list[np.ndarray] = []
        # centroid / size accumulators, grown lazily
        self.sum_x = np.zeros(0, dtype=np.float64)
        self.sum_y = np.zeros(0, dtype=np.float64)
        self.n_tx = np.zeros(0, dtype=np.int64)
        # entity index -> {pristine cell_id: transcript count}. The parent is
        # taken from the data rather than parsed out of the entity label: the
        # label delimiter is a TRACER-version detail (this pin emits
        # "<cell_id>-<k>", the source constant declares "-tr-"), and a cell_id
        # may itself contain dashes.
        self.parents: dict[int, dict[str, int]] = {}

    def _grow(self, n: int) -> None:
        if n <= self.sum_x.size:
            return
        extra = n - self.sum_x.size
        self.sum_x = np.concatenate([self.sum_x, np.zeros(extra)])
        self.sum_y = np.concatenate([self.sum_y, np.zeros(extra)])
        self.n_tx = np.concatenate([self.n_tx, np.zeros(extra, dtype=np.int64)])

    def add(self, entity: pd.Series, gene: pd.Series,
            x: np.ndarray, y: np.ndarray,
            parent: pd.Series | None = None) -> None:
        if len(entity) == 0:
            return
        col = gene.map(self.gene_index)
        keep = col.notna().to_numpy()
        if not keep.any():
            return
        entity = entity[keep]
        col = col[keep].to_numpy(dtype=np.int32)
        x, y = x[keep], y[keep]
        parent = parent[keep] if parent is not None else None

        # map entity labels -> stable integer ids across row groups.
        # Resolve each *unique* label once, then gather through `inv`; a
        # per-label boolean scan over the row group would be quadratic.
        uniq, inv = np.unique(entity.to_numpy(), return_inverse=True)
        lut = np.empty(len(uniq), dtype=np.int64)
        labels = self.labels
        for i, lab in enumerate(uniq):
            idx = labels.get(lab)
            if idx is None:
                idx = len(labels)
                labels[lab] = idx
            lut[i] = idx
        codes = lut[inv]

        self.rows.append(codes)
        self.cols.append(col)
        self.vals.append(np.ones(len(codes), dtype=np.float32))

        n = len(labels)
        self._grow(n)
        self.sum_x[:n] += np.bincount(codes, weights=x, minlength=n)
        self.sum_y[:n] += np.bincount(codes, weights=y, minlength=n)
        self.n_tx[:n] += np.bincount(codes, minlength=n).astype(np.int64)

        if parent is not None:
            pc = (pd.DataFrame({"e": codes, "p": parent.to_numpy()})
                  .groupby(["e", "p"], sort=False).size())
            for (e, par), cnt in pc.items():
                if par in NULL_LABELS:
                    continue
                d = self.parents.setdefault(int(e), {})
                d[par] = d.get(par, 0) + int(cnt)

    def finalize(self, genes: np.ndarray, entity_type: str,
                 sample: str, patient: str, section: str) -> ad.AnnData:
        n = len(self.labels)
        if n == 0:
            X = sp.csr_matrix((0, self.n_genes), dtype=np.float32)
            obs = pd.DataFrame(index=pd.Index([], name="entity_id"))
        else:
            X = sp.coo_matrix(
                (np.concatenate(self.vals),
                 (np.concatenate(self.rows), np.concatenate(self.cols))),
                shape=(n, self.n_genes), dtype=np.float32,
            ).tocsr()
            X.sum_duplicates()
            labels = np.empty(n, dtype=object)
            for lab, i in self.labels.items():
                labels[i] = lab
            labels = labels.astype(str)
            self._grow(n)
            parents = np.array([modal_parent(self.parents.get(i, {}))
                                for i in range(n)], dtype=object)
            obs = pd.DataFrame(
                {
                    "entity_type": entity_type,
                    "original_cell_id": parents,
                    "x_centroid": self.sum_x[:n] / np.maximum(self.n_tx[:n], 1),
                    "y_centroid": self.sum_y[:n] / np.maximum(self.n_tx[:n], 1),
                    "n_tx": self.n_tx[:n],
                    "sample": sample,
                    "patient": patient,
                    "section": section,
                },
                index=pd.Index(labels, name="entity_id"),
            )
        var = pd.DataFrame(index=pd.Index(genes, name="feature_name"))
        a = ad.AnnData(X=X, obs=obs, var=var)
        a.layers["counts"] = X.copy()
        return a


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    genes = np.array([g.strip() for g in Path(args.genes).read_text().split("\n")
                      if g.strip()], dtype=object)
    gene_index = {g: i for i, g in enumerate(genes)}
    print(f"[prep] gene axis: {len(genes)} panel-matched genes", flush=True)

    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    accs = {a: ArmAccumulator(a, gene_index, len(genes)) for a in wanted}

    pf = pq.ParquetFile(args.transcripts)
    have = set(pf.schema_arrow.names)
    missing = [c for c in REQUIRED if c not in have]
    if missing:
        raise SystemExit(f"transcripts parquet is missing required columns: {missing}")
    xcol = next((c for c in X_ALIASES if c in have), None)
    ycol = next((c for c in Y_ALIASES if c in have), None)
    has_xy = xcol is not None and ycol is not None
    if has_xy:
        print(f"[prep] coordinates from '{xcol}' / '{ycol}'", flush=True)
    else:
        print(f"[prep] WARNING: no coordinate columns among {X_ALIASES} / "
              f"{Y_ALIASES}; centroids will be 0 and spatial maps unusable",
              flush=True)
    cols = REQUIRED + ([xcol, ycol] if has_xy else [])

    total = 0
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        df = t.to_pandas()
        total += len(df)
        for c in ("cell_id", "tracer_id", "_etype", "feature_name"):
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                df[c] = df[c].astype(str)
            else:
                df[c] = df[c].astype(str)
        x = (df[xcol].to_numpy(dtype=np.float64) if has_xy
             else np.zeros(len(df), dtype=np.float64))
        y = (df[ycol].to_numpy(dtype=np.float64) if has_xy
             else np.zeros(len(df), dtype=np.float64))

        if "original" in accs:
            m = (~df["cell_id"].isin(NULL_LABELS)).to_numpy()
            accs["original"].add(df.loc[m, "cell_id"], df.loc[m, "feature_name"],
                                 x[m], y[m], parent=df.loc[m, "cell_id"])
        post_valid = (~df["tracer_id"].isin(NULL_LABELS)).to_numpy()
        et = df["_etype"].to_numpy()
        for arm, sel in (
            ("post_whole", et == "cell"),
            ("post_partial", et == "partial"),
            ("post_all", np.isin(et, ("cell", "partial"))),
        ):
            if arm not in accs:
                continue
            m = post_valid & sel
            accs[arm].add(df.loc[m, "tracer_id"], df.loc[m, "feature_name"],
                          x[m], y[m], parent=df.loc[m, "cell_id"])
        del df, t
        if rg % 10 == 0:
            print(f"[prep]   row group {rg + 1}/{pf.metadata.num_row_groups} "
                  f"({total:,} transcripts)", flush=True)

    manifest = {"sample": args.sample_name, "patient": args.patient,
                "section": args.section, "transcripts_rows": int(total),
                "n_genes": int(len(genes)), "arms": {}}
    for arm, acc in accs.items():
        etype = {"original": "original", "post_whole": "whole",
                 "post_partial": "partial", "post_all": "whole_or_partial"}[arm]
        a = acc.finalize(genes, etype, args.sample_name, args.patient, args.section)
        path = outdir / f"entities_{arm}.h5ad"
        a.write_h5ad(path, compression="gzip")
        med = float(np.median(a.obs["n_tx"])) if a.n_obs else 0.0
        manifest["arms"][arm] = {
            "n_entities": int(a.n_obs),
            "median_tx_per_entity": med,
            "total_tx": int(a.obs["n_tx"].sum()) if a.n_obs else 0,
            "file": path.name,
        }
        print(f"[prep] {arm:<13} n={a.n_obs:>9,}  median tx/entity={med:>7.1f}",
              flush=True)
        del a

    (outdir / "prep_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[prep] wrote", outdir / "prep_manifest.json", flush=True)


if __name__ == "__main__":
    main()
