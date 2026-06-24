#!/usr/bin/env python3
"""Memory-safe whole-transcriptome NPMI panel via top-k-per-gene (deterministic).

Why this exists
---------------
The production gene-row PMI *bootstrap* (`build_npmi_from_scrna.py`) needs ~20 GB
for Atera's whole-transcriptome candidate set (~84M pairs) and OOM-kills on a
16 GB machine. For the TRACER within-cell prune veto we only need each gene's
strongest associations (point estimates), not bootstrap CIs. This builder:

  - builds a binary cell x gene presence matrix from the reference raw counts,
  - processes genes in BLOCKS (bounded memory): for each block computes
    co-occurrence to all genes via a sparse matmul, then PMI/NPMI,
  - keeps the **top-k partners per gene by |PMI|** (positive and negative),
  - writes a long-format panel (gene_i, gene_j, PMI, NPMI, p_*, n_cells_*).

Deterministic, exact point estimates, peak RSS a few GB. The output schema is
what `run_tracer.py` / `run_segmented_pipeline` consume.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-h5ad", required=True, type=Path)
    ap.add_argument("--gene-list", type=Path,
                    help="One gene/line (e.g. npmi_gene_list.tsv with header 'gene'). "
                         "If omitted, --spatial-transcripts is required.")
    ap.add_argument("--spatial-transcripts", type=Path,
                    help="ROI parquet; overlap(ref, ROI feature_name) defines the panel.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--k", type=int, default=100, help="Top partners per gene by |PMI|.")
    ap.add_argument("--min-cooccur", type=int, default=10,
                    help="Drop pairs co-occurring in < this many reference cells.")
    ap.add_argument("--min-cells-expressed", type=int, default=100)
    ap.add_argument("--block-size", type=int, default=1500)
    args = ap.parse_args()

    import anndata as ad
    t0 = time.time()
    a = ad.read_h5ad(args.reference_h5ad)
    cnt = a.layers["counts"] if "counts" in a.layers else a.X
    cnt = sp.csr_matrix(cnt) if not sp.issparse(cnt) else cnt.tocsr()
    var = np.asarray(a.var_names, dtype=str)

    if args.gene_list and args.gene_list.exists():
        gl = pd.read_csv(args.gene_list, sep="\t")
        genes_in = gl["gene"].astype(str).tolist() if "gene" in gl.columns \
            else gl.iloc[:, 0].astype(str).tolist()
    else:
        import duckdb
        roi = duckdb.connect().execute(
            f"SELECT DISTINCT feature_name FROM '{args.spatial_transcripts}'"
        ).fetchnumpy()["feature_name"]
        genes_in = sorted(set(map(str, roi)) & set(var.tolist()))

    idx = pd.Index(var).get_indexer(genes_in)
    idx = idx[idx >= 0]
    B = (cnt[:, idx] > 0).astype(np.float32).tocsc()           # cells x genes presence
    N = B.shape[0]
    n_i = np.asarray(B.sum(0)).ravel().astype(np.int64)
    keep = n_i >= args.min_cells_expressed
    B = B[:, keep].tocsc()
    genes = np.asarray(genes_in)[idx[keep] >= -1][keep] if False else np.asarray(genes_in)[keep]
    n_i = n_i[keep]
    p_i = n_i / N
    G = B.shape[1]
    logp = np.log(p_i)
    print(f"[topk] presence {B.shape} nnz={B.nnz:,}  G={G} genes  N={N} cells  "
          f"({time.time()-t0:.0f}s)", flush=True)

    Bc = B.tocsr()                       # for the right operand of matmul
    rows_i, cols_j, pmis, npmis, cos = [], [], [], [], []
    k = int(args.k); mc = int(args.min_cooccur)
    for bs in range(0, G, args.block_size):
        be = min(bs + args.block_size, G)
        # co-occurrence of this gene block vs all genes: (block x G) dense int
        co = np.asarray((B[:, bs:be].T @ Bc).todense(), dtype=np.float64)  # block x G
        bsz = be - bs
        # PMI where co>=min; -inf elsewhere and on the diagonal (self)
        with np.errstate(divide="ignore", invalid="ignore"):
            pij = co / N
            pmi = np.log(pij) - logp[bs:be][:, None] - logp[None, :]
        valid = co >= mc
        diag_rows = np.arange(bsz); valid[diag_rows, bs + diag_rows] = False
        absp = np.where(valid, np.abs(pmi), -1.0)
        # top-k columns per row by |PMI|
        kk = min(k, G - 1)
        part = np.argpartition(absp, -kk, axis=1)[:, -kk:]      # block x kk
        for r in range(bsz):
            cj = part[r]
            cj = cj[valid[r, cj]]
            if cj.size == 0:
                continue
            gi = bs + r
            pv = pmi[r, cj]; cv = co[r, cj]
            pijv = cv / N
            nv = pv / (-np.log(pijv))
            rows_i.append(np.full(cj.size, gi, dtype=np.int32))
            cols_j.append(cj.astype(np.int32)); pmis.append(pv); npmis.append(nv); cos.append(cv)
        print(f"[topk]   block {bs}-{be}/{G}  ({time.time()-t0:.0f}s)", flush=True)

    ri = np.concatenate(rows_i); cj = np.concatenate(cols_j)
    pmi = np.concatenate(pmis); npmi = np.concatenate(npmis); co = np.concatenate(cos)
    # dedup symmetric duplicates (keep unordered pair once)
    lo = np.minimum(ri, cj); hi = np.maximum(ri, cj)
    key = lo.astype(np.int64) * G + hi.astype(np.int64)
    _, uniq = np.unique(key, return_index=True)
    lo, hi, pmi, npmi, co = lo[uniq], hi[uniq], pmi[uniq], npmi[uniq], co[uniq]
    panel = pd.DataFrame({
        "gene_i": genes[lo], "gene_j": genes[hi],
        "PMI": pmi.astype(np.float32), "NPMI": npmi.astype(np.float32),
        "n_cells_i": n_i[lo], "n_cells_j": n_i[hi], "n_cells_ij": co.astype(np.int64),
        "p_i": p_i[lo], "p_j": p_i[hi], "p_ij": co / N,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, index=False, compression="gzip")
    ng = len(set(panel.gene_i) | set(panel.gene_j))
    print(f"[topk] DONE wrote {len(panel):,} pairs over {ng} genes -> {args.out}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[topk] NPMI median={np.median(npmi):.3f} >=0.2={np.mean(npmi>=0.2)*100:.0f}% "
          f"<=-0.2={np.mean(npmi<=-0.2)*100:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
