#!/usr/bin/env python3
"""Run InSituCNV end-to-end on ONE segmentation "arm" of a TRACER parquet.

Single per-arm script: (1) inferCNV -> (2) cluster ALL cells on the CNV matrix ->
(3) select tumor CNV clusters as **subclones** and profile each one. Run it on
both arms and compare raw vs TRACER by eyeballing the per-subclone outputs; there
is no separate bulk-tumor comparison (pooling all tumor cells averages subclones
away).

Arms are built from the *same* parquet so tissue, transcripts, panel and QC are
identical and the only variable is which per-transcript cell label we group by:

* ``--segmentation raw``    -> group by the original Xenium ``cell_id``
* ``--segmentation tracer`` -> group by TRACER's refined whole-cell label
                               (``cell_id_tracer`` current / ``cell_id_finetuned``
                               legacy / ``stitched`` fallback)

The CNV engine is ``infercnvpy`` (inferCNV); ``insitucnv`` is a thin wrapper we
call via ``insitucnv.tl``. We use the ``tl`` API directly (not ``insitucnv.cli``)
because the CLI's Xenium loader re-reads 10x's official matrix with its own
filtering (a transcript-set/QV confound) and does not let us inject our own gene
coordinates. The GBM panel needs ``tutorials/gbm/data/gene_positions_grch38.tsv``.

**Reference / tumor cells come from an EXISTING annotation** (``--annotations-csv``,
``adata_obs_annotated.csv``), not de-novo clustering -- one source of truth across
arms. The annotation ``cell_id`` carries a per-patient suffix (e.g. ``-P4``) that
the parquet ``cell_id`` lacks; ``--annotation-suffix`` bridges the join. Raw arm:
direct ``cell_id`` lookup. TRACER arm: each refined cell inherits the MAJORITY
annotation of its constituent transcripts' original ``cell_id``. ``compartment``
in {reference, tumor, unknown}: ``cancer_*`` -> tumor; myeloid / T-cell / vascular
/ oligo / neutrophil -> reference; else unknown. inferCNV uses ``reference`` as the
normal baseline.

**Subclones.** ``X_cnv`` (per-cell x genomic-window CNV) is clustered over ALL
cells; a CNV cluster is a **subclone** when its tumor-annotated fraction >=
``--tumor-cluster-frac`` (reference-dominated clusters are the flat/diploid
anchor). Each subclone is profiled by per-chromosome mean CNV and **Cohen's d vs
reference cells**; arm-level **baseline flatness** (reference CNV std) and
**median counts/cell** (depth-confound flag) are also reported.

**Panel resolution.** inferCNV windows are per-chromosome, and this panel is
sparse (366 genes; ~16 of 23 chromosomes have <20 genes), so small chromosomes
collapse to ~1 coarse window. Defaults are ``--window-size 10 --step 3``.
Per-chromosome gene+window counts are reported, and chromosomes with fewer than
``--min-genes-per-chromosome`` genes are flagged ``low_resolution`` and EXCLUDED
from the subclone Cohen's d (so a 4-gene chromosome cannot drive a fake "event").

**Two entry modes (one script):**
* default -- full run from ``--parquet`` (through inferCNV);
* ``--from-h5ad <adata_cnv.h5ad>`` -- skip parquet+inferCNV, reload a previous
  run, and only re-cluster + re-profile subclones (cheap resolution re-sweep).

EXAMPLE
=======
::

    python tutorials/gbm/run_insitucnv_arm.py \
      --parquet tutorials/gbm/output/slide3_tracer/slide3_piece_04_Patient4_tracer.parquet \
      --segmentation raw \
      --annotations-csv /mnt/storage/.../Xenium_Annotations/adata_obs_annotated.csv \
      --output-dir tutorials/gbm/output/insitucnv/piece04/raw

Add ``--dry-run`` to validate plumbing (matrix, coords, gene mapping, annotation
coverage) without importing infercnvpy/insitucnv.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the "not a real cell" token set from the TRACER GBM runner when available.
try:  # run_gbm.py sits next to this script; sys.path[0] is this dir.
    from run_gbm import UNASSIGNED_TOKENS  # type: ignore
except Exception:  # pragma: no cover - keep the arm script self-contained.
    UNASSIGNED_TOKENS = frozenset(
        {"UNASSIGNED", "Unassigned", "unassigned", "DROP", "nan", "None", "", "0", "-1", "NA"}
    )

# TRACER refined whole-cell label, best first. Matches merge_slide3_tracer.py's
# convention (cell_id_tracer current, cell_id_finetuned legacy); `stitched` is a
# last-resort fallback for very old pieces. NOTE: cell_id_stitched is NOT canonical.
TRACER_LABEL_COLUMNS = ("cell_id_tracer", "cell_id_finetuned", "stitched")

# Annotation -> compartment rules (see module docstring). Explicit --tumor-annotations
# / --reference-annotations override these prefix/membership defaults.
TUMOR_PREFIXES = ("cancer",)
REFERENCE_PREFIXES = ("myeloid", "tcell")
REFERENCE_EXACT = {"oligo", "vascular_endothelial", "vascular_pericyte", "neutrophil"}
NEURON_PREFIXES = ("neuron",)  # non-malignant but transcriptionally distinct; opt-in only.

_INSTALL_HINT = (
    "InSituCNV is not importable in this environment. Install it into insitucnv_env:\n"
    "    conda activate insitucnv_env\n"
    "    pip uninstall -y insitucnv\n"
    "    pip install git+https://github.com/Moldia/InSituCNV.git\n"
    "    python -c \"import insitucnv; print(insitucnv.__file__)\"   # must be under the env, not /tmp\n"
    "(A `pip install -e /tmp/InSituCNV` breaks on SGE because /tmp is node-local.)"
)


def _require_deps() -> None:
    """Fail fast if the scientific stack is missing, before any expensive work."""
    import importlib
    for module in ("insitucnv", "infercnvpy"):
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise SystemExit(f"{exc}\n\n{_INSTALL_HINT}")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", default=None,
                   help="TRACER refined transcript parquet (required unless --from-h5ad).")
    p.add_argument("--from-h5ad", default=None,
                   help="Reuse a previous run's adata_cnv.h5ad; skip parquet+inferCNV, only "
                        "re-cluster + re-profile subclones (cheap resolution re-sweep).")
    p.add_argument("--segmentation", choices=["raw", "tracer"],
                   help="Which cell label to group transcripts by (required unless --from-h5ad).")
    p.add_argument("--output-dir", required=True, help="Directory for this arm's outputs.")
    p.add_argument("--gene-positions",
                   default="tutorials/gbm/data/gene_positions_grch38.tsv",
                   help="GRCh38 gene coordinate TSV (columns: gene, chromosome, start, end).")
    p.add_argument("--sample-id", default=None, help="Value for adata.obs['sample'] (default: parquet stem).")

    # Existing annotations (source of truth for reference/tumor compartments).
    p.add_argument("--annotations-csv", default=None,
                   help="adata_obs_annotated.csv with cell_id + annotation (required unless --dry-run).")
    p.add_argument("--annotation-suffix", default="-P4",
                   help="Per-patient suffix on annotation cell_id absent from the parquet (e.g. -P4).")
    p.add_argument("--reference-annotations", default=None,
                   help="Comma-separated annotation labels to force as reference (overrides defaults).")
    p.add_argument("--tumor-annotations", default=None,
                   help="Comma-separated annotation labels to force as tumor (overrides defaults).")
    p.add_argument("--include-neurons-as-reference", action="store_true",
                   help="Also treat neuron_* annotations as reference (default: unknown).")
    p.add_argument("--min-reference-cells", type=int, default=25,
                   help="Abort if fewer than this many reference cells are found.")

    # Transcript-level QC (applied IDENTICALLY to both arms).
    p.add_argument("--min-qv", type=float, default=20.0, help="Drop transcripts below this Phred QV.")
    p.add_argument("--nucleus-only", action="store_true",
                   help="Keep only nucleus-overlapping transcripts (overlaps_nucleus==1).")

    # Depth matching. The separation metric shrinks with per-cell depth, and the
    # arms differ in counts/cell (that is part of the segmentation effect). To tell
    # a genuine de-contamination gain from a mere depth gain, run BOTH arms with the
    # SAME --downsample-to-counts (e.g. the shallower arm's median) and compare at
    # matched depth. 0 = off.
    p.add_argument("--downsample-to-counts", type=int, default=0,
                   help="Cap each cell's total counts to this value (depth matching). 0 = off.")

    # Cell-level QC (shared).
    p.add_argument("--min-counts", type=int, default=20, help="Minimum counts per cell.")
    p.add_argument("--min-genes", type=int, default=5, help="Minimum detected genes per cell.")
    p.add_argument("--min-cells", type=int, default=5, help="Minimum cells per gene.")

    # InSituCNV / infercnvpy parameters. Defaults are smaller than InSituCNV's
    # manuscript 5K-panel defaults (window 60 / step 10) because this GBM panel
    # has only ~366 mappable genes (4-35 per chromosome).
    p.add_argument("--target-sum", type=float, default=1e4, help="normalize_total target sum.")
    p.add_argument("--smoothing-neighbors", type=int, default=100, help="Neighbors for scVelo smoothing.")
    p.add_argument("--window-size", type=int, default=10,
                   help="infercnvpy window_size (small: sparse 366-gene panel, windows are per-chromosome).")
    p.add_argument("--step", type=int, default=3, help="infercnvpy step.")
    p.add_argument("--lfc-clip", type=float, default=4.0, help="infercnvpy lfc_clip.")
    p.add_argument("--cluster-resolutions", default="0.05,0.1,0.2",
                   help="Comma-separated CNV Leiden resolutions to sweep (subclone granularity).")

    # Subclone selection + panel-resolution handling.
    p.add_argument("--tumor-cluster-frac", type=float, default=0.5,
                   help="A CNV cluster is a subclone if this fraction of its cells are tumor-annotated.")
    p.add_argument("--min-genes-per-chromosome", type=int, default=10,
                   help="Chromosomes with fewer panel genes are flagged low_resolution and excluded "
                        "from the subclone Cohen's d (too few genes for a reliable CNV call).")
    p.add_argument("--vmax", type=float, default=0.4, help="CNV heatmap color scale limit (+/-).")

    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate plumbing (matrix, coords, gene mapping, annotation coverage) then stop.")
    return p.parse_args(argv)


def _csl(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


# --------------------------------------------------------------------------- #
# Transcript loading and matrix building
# --------------------------------------------------------------------------- #
def resolve_group_column(columns: list[str], segmentation: str) -> str:
    if segmentation == "raw":
        if "cell_id" not in columns:
            raise SystemExit("Parquet has no 'cell_id' column for the raw arm.")
        return "cell_id"
    for candidate in TRACER_LABEL_COLUMNS:
        if candidate in columns:
            return candidate
    raise SystemExit(
        "Parquet has no TRACER refined-label column. Looked for: " + ", ".join(TRACER_LABEL_COLUMNS)
    )


def load_arm_transcripts(parquet: Path, group_col: str, min_qv: float, nucleus_only: bool) -> pd.DataFrame:
    """Read the needed columns, apply shared transcript QC, keep group_col + orig cell_id.

    ``orig_cell_id`` (the raw Xenium ``cell_id``) is retained for BOTH arms so the
    annotation transfer is identical: majority annotation per group over the
    transcripts' original cell ids. For the raw arm group_col IS cell_id, so the
    majority is trivial.
    """
    import pyarrow.parquet as pq

    schema_names = pq.ParquetFile(parquet).schema_arrow.names
    need = {group_col, "cell_id", "feature_name", "x", "y"}
    missing = need - set(schema_names) - {"cell_id"}  # cell_id checked below
    if missing:
        raise SystemExit(f"Parquet missing required columns: {sorted(missing)}")
    has_cell_id = "cell_id" in schema_names

    wanted = list({group_col, "feature_name", "x", "y"} | ({"cell_id"} if has_cell_id else set()))
    for optional in ("qv", "overlaps_nucleus", "is_gene"):
        if optional in schema_names:
            wanted.append(optional)
    df = pd.read_parquet(parquet, columns=wanted)
    n0 = len(df)

    # Keep only real genes (drop Xenium controls: NegControl/BLANK/antisense/etc.).
    if "is_gene" in df.columns:
        df = df[df["is_gene"].astype(bool)]
    else:
        fname = df["feature_name"].astype(str)
        control = fname.str.contains(
            "NegControl|BLANK|antisense|Unassigned|Deprecated|Intergenic|Genomic", case=False, regex=True
        )
        df = df[~control]

    if "qv" in df.columns and min_qv > 0:
        df = df[df["qv"].astype("float32") >= min_qv]
    if nucleus_only and "overlaps_nucleus" in df.columns:
        df = df[df["overlaps_nucleus"].astype("uint8") == 1]

    df[group_col] = df[group_col].astype(str)
    df = df[~df[group_col].isin(UNASSIGNED_TOKENS)]
    df["orig_cell_id"] = df["cell_id"].astype(str) if has_cell_id else df[group_col]
    df["feature_name"] = df["feature_name"].astype(str).str.strip()
    df = df[[group_col, "orig_cell_id", "feature_name", "x", "y"]].copy()

    print(f"Transcripts: {n0:,} read -> {len(df):,} kept "
          f"(genes only, qv>={min_qv:g}{', nucleus-only' if nucleus_only else ''}, assigned).",
          flush=True)
    if df.empty:
        raise SystemExit("No transcripts left after QC/assignment filtering.")
    return df


def build_matrix(df: pd.DataFrame, group_col: str):
    """Group transcripts into a cell x gene count matrix plus per-cell centroids."""
    from scipy import sparse as sp

    counts = df.groupby([group_col, "feature_name"], observed=True).size().rename("count").reset_index()
    cell_cat = pd.Categorical(counts[group_col])
    gene_cat = pd.Categorical(counts["feature_name"])
    X = sp.csr_matrix(
        (counts["count"].to_numpy(np.float32), (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(gene_cat.categories)),
    )
    cell_ids = cell_cat.categories.astype(str).to_numpy()
    genes = gene_cat.categories.astype(str).to_numpy()

    centroids = (
        df.groupby(group_col, observed=True)[["x", "y"]].mean()
        .reindex(cell_ids).to_numpy(np.float32)
    )
    return cell_ids, genes, X, centroids


# --------------------------------------------------------------------------- #
# Existing-annotation ingestion + compartment assignment
# --------------------------------------------------------------------------- #
def load_annotation_map(csv: Path, suffix: str) -> pd.Series:
    """Return a base-cell_id -> annotation Series from adata_obs_annotated.csv.

    The annotation cell_id carries a per-patient suffix (e.g. ``-P4``) absent from
    the parquet cell_id. We keep only rows with that suffix (avoids cross-patient
    collisions) and strip it to match the parquet.
    """
    df = pd.read_csv(csv, usecols=["cell_id", "annotation"])
    ids = df["cell_id"].astype(str)
    if suffix:
        keep = ids.str.endswith(suffix)
        df, ids = df[keep], ids[keep]
        base = ids.str.slice(0, -len(suffix))
    else:
        base = ids
    ser = pd.Series(df["annotation"].astype(str).to_numpy(), index=base.to_numpy())
    ser = ser[~ser.index.duplicated(keep="first")]
    return ser


def _compartment_fn(reference_override: list[str], tumor_override: list[str], include_neurons: bool):
    ref_set = {s.lower() for s in reference_override}
    tum_set = {s.lower() for s in tumor_override}

    def classify(annotation: str) -> str:
        a = str(annotation).lower()
        if tum_set or ref_set:  # explicit lists take precedence (exact match)
            if a in tum_set:
                return "tumor"
            if a in ref_set:
                return "reference"
            # fall through to prefix defaults for anything not explicitly listed
        if a.startswith(TUMOR_PREFIXES):
            return "tumor"
        if a.startswith(REFERENCE_PREFIXES) or a in REFERENCE_EXACT:
            return "reference"
        if include_neurons and a.startswith(NEURON_PREFIXES):
            return "reference"
        return "unknown"

    return classify


def assign_compartments(df: pd.DataFrame, group_col: str, cell_ids: np.ndarray,
                        ann_map: pd.Series, classify) -> tuple[pd.Series, pd.Series, float]:
    """Majority annotation per group cell, then map to compartment.

    Returns (annotation per cell_id, compartment per cell_id, annotation coverage).
    """
    tmp = df[[group_col, "orig_cell_id"]].copy()
    tmp["annotation"] = tmp["orig_cell_id"].map(ann_map)
    annotated = tmp.dropna(subset=["annotation"])
    # Majority annotation per group = most transcripts.
    if annotated.empty:
        maj = pd.Series(dtype="object")
    else:
        grp = annotated.groupby([group_col, "annotation"], observed=True).size()
        maj = grp.groupby(level=0, observed=True).idxmax().map(lambda t: t[1])
    annotation = maj.reindex(cell_ids)
    compartment = annotation.map(lambda a: classify(a) if pd.notna(a) else "unknown")
    compartment = compartment.fillna("unknown")
    coverage = float(annotation.notna().mean())
    annotation = annotation.fillna("unknown")
    return annotation, compartment, coverage


# --------------------------------------------------------------------------- #
# Gene coordinate reference
# --------------------------------------------------------------------------- #
def load_gene_reference(path: Path) -> pd.DataFrame:
    """Load the GBM panel coordinate table for insitucnv.pp.add_genomic_positions.

    ``add_genomic_positions`` expects ``gene_name``, ``chromosome``, ``start``,
    ``end``. Our file uses ``gene`` and unprefixed chromosomes; rename + add ``chr``.
    """
    ref = pd.read_csv(path, sep="\t")
    if "gene" not in ref.columns:
        raise SystemExit(f"{path} must have a 'gene' column; got {list(ref.columns)}")
    ref = ref.rename(columns={"gene": "gene_name"})
    for col in ("chromosome", "start", "end"):
        if col not in ref.columns:
            raise SystemExit(f"{path} missing '{col}' column.")
    chrom = ref["chromosome"].astype(str)
    ref["chromosome"] = np.where(chrom.str.startswith("chr"), chrom, "chr" + chrom)
    return ref


# --------------------------------------------------------------------------- #
# Per-chromosome CNV summary (plotting-independent)
# --------------------------------------------------------------------------- #
def chromosome_cluster_table(adata, cluster_key: str, layer: str = "gene_values_cnv") -> pd.DataFrame:
    """Mean CNV per (group x chromosome) from the per-gene CNV layer.

    Plotting-independent view of the chromosome heatmap: which cell groups carry
    chr-level gains/losses. Rows = group labels, cols = chromosomes.
    """
    from scipy import sparse as sp

    if layer not in adata.layers or "chromosome" not in adata.var:
        return pd.DataFrame()
    gv = adata.layers[layer]
    gv = gv.toarray() if sp.issparse(gv) else np.asarray(gv)
    chrom = adata.var["chromosome"].astype(str).to_numpy()
    order = [f"chr{c}" for c in list(range(1, 23)) + ["X"]]
    labels = adata.obs[cluster_key].astype(str)
    rows = {}
    for lab in sorted(labels.unique()):
        per_gene = gv[(labels == lab).to_numpy()].mean(axis=0)
        rows[lab] = pd.Series(per_gene, index=chrom).groupby(level=0).mean()
    table = pd.DataFrame(rows).T
    return table.reindex(columns=[c for c in order if c in table.columns])


CHROM_ORDER = [f"chr{c}" for c in list(range(1, 23)) + ["X"]]


def _order_key(label: str):
    try:
        return (0, int(label))
    except (ValueError, TypeError):
        return (1, str(label))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Cohen's d = (mean(a) - mean(b)) / pooled SD. NaN if degenerate."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def per_cell_chromosome_cnv(adata, layer: str = "gene_values_cnv") -> pd.DataFrame:
    """Per-cell mean CNV on each chromosome (cells x chromosomes) from the gene CNV layer."""
    from scipy import sparse as sp

    gv = adata.layers[layer]
    gv = gv.toarray() if sp.issparse(gv) else np.asarray(gv)
    chrom = adata.var["chromosome"].astype(str).to_numpy()
    cols = {c: gv[:, chrom == c].mean(axis=1) for c in pd.unique(chrom)}
    out = pd.DataFrame(cols, index=adata.obs_names)
    return out.reindex(columns=[c for c in CHROM_ORDER if c in out.columns])


def chromosome_resolution_table(adata, min_genes: int, cnv_key: str = "cnv") -> pd.DataFrame:
    """Per-chromosome panel gene count + inferCNV window count + low_resolution flag.

    Window counts come from ``uns[cnv_key]['chr_pos']`` (start column of each
    chromosome block in X_cnv); genes from ``var['chromosome']``.
    """
    chrom = adata.var["chromosome"].astype(str)
    gene_counts = chrom.value_counts()

    win_counts = {}
    chr_pos = adata.uns.get(cnv_key, {}).get("chr_pos", {}) if hasattr(adata, "uns") else {}
    if chr_pos:
        n_cols = adata.obsm[f"X_{cnv_key}"].shape[1]
        items = sorted(chr_pos.items(), key=lambda kv: kv[1])
        for k, (name, start) in enumerate(items):
            end = items[k + 1][1] if k + 1 < len(items) else n_cols
            win_counts[name] = int(end - start)

    rows = []
    for c in [c for c in CHROM_ORDER if c in set(gene_counts.index)]:
        g = int(gene_counts.get(c, 0))
        rows.append({"chromosome": c, "n_genes": g, "n_windows": win_counts.get(c, np.nan),
                     "low_resolution": g < min_genes})
    return pd.DataFrame(rows).set_index("chromosome")


def dominant_events(chrom_row: pd.Series, threshold: float, exclude: set) -> list[str]:
    """Chromosomes with |mean CNV| >= threshold (skipping low-resolution ones), e.g. ['chr7+','chr19-']."""
    out = []
    for chrom, val in chrom_row.items():
        if chrom in exclude:
            continue
        if pd.notna(val) and abs(val) >= threshold:
            out.append(f"{chrom}{'+' if val > 0 else '-'}")
    return out


def subclone_metrics(adata, cluster_key: str, per_cell_chr: pd.DataFrame,
                     low_res_chroms: set, tumor_frac_thresh: float, event_threshold: float = 0.1):
    """Profile every CNV cluster at one resolution.

    Returns (clusters_df, cohensd_df, chrom_cnv_df):
      * clusters_df: per cluster n_cells, tumor/reference/unknown fractions,
        is_subclone, dominant events.
      * cohensd_df: subclones x chromosome, Cohen's d vs reference cells
        (low-resolution chromosomes set to NaN).
      * chrom_cnv_df: subclones x chromosome mean CNV.
    """
    comp = adata.obs["compartment"].astype(str)
    labels = adata.obs[cluster_key].astype(str)
    ref_mask = (comp == "reference").to_numpy()
    usable = [c for c in per_cell_chr.columns if c not in low_res_chroms]

    cluster_rows, cohensd_rows, cnv_rows = [], {}, {}
    for lab in sorted(labels.unique(), key=_order_key):
        m = (labels == lab).to_numpy()
        n = int(m.sum())
        fr = comp[m].value_counts(normalize=True)
        tumor_frac = float(fr.get("tumor", 0.0))
        is_sub = tumor_frac >= tumor_frac_thresh
        chr_means = per_cell_chr.loc[m].mean()
        cluster_rows.append({
            "cluster": lab, "n_cells": n,
            "tumor_frac": round(tumor_frac, 3),
            "reference_frac": round(float(fr.get("reference", 0.0)), 3),
            "unknown_frac": round(float(fr.get("unknown", 0.0)), 3),
            "is_subclone": bool(is_sub),
            "events": ",".join(dominant_events(chr_means, event_threshold, low_res_chroms)),
        })
        if is_sub:
            cnv_rows[lab] = chr_means
            cohensd_rows[lab] = pd.Series(
                {c: (cohens_d(per_cell_chr[c].to_numpy()[m], per_cell_chr[c].to_numpy()[ref_mask])
                     if c in usable else np.nan) for c in per_cell_chr.columns})

    clusters_df = pd.DataFrame(cluster_rows).set_index("cluster")
    cohensd_df = pd.DataFrame(cohensd_rows).T.reindex(columns=per_cell_chr.columns) \
        if cohensd_rows else pd.DataFrame(columns=per_cell_chr.columns)
    cnv_df = pd.DataFrame(cnv_rows).T.reindex(columns=per_cell_chr.columns) \
        if cnv_rows else pd.DataFrame(columns=per_cell_chr.columns)
    return clusters_df, cohensd_df, cnv_df


def render_cnv_heatmap(adata, group_key: str, out_path: Path, vmax: float, cnv_key: str = "cnv") -> None:
    """Classic inferCNV view: cells (rows, blocked by group) x genome (cols). Matplotlib-only."""
    from scipy import sparse as sp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if f"X_{cnv_key}" not in adata.obsm:
        print(f"[heatmap] adata.obsm['X_{cnv_key}'] missing; skipping.", flush=True)
        return
    X = adata.obsm[f"X_{cnv_key}"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    labels = adata.obs[group_key].astype(str)
    order = np.argsort([_order_key(l) for l in labels], kind="stable")
    Xo, labels_o = X[order], labels.to_numpy()[order]

    uniq, first = [], []
    for i, lab in enumerate(labels_o):
        if not uniq or uniq[-1] != lab:
            uniq.append(lab); first.append(i)
    centers = [(first[k] + (first[k + 1] if k + 1 < len(first) else len(labels_o))) / 2
               for k in range(len(uniq))]

    chr_pos = adata.uns.get(cnv_key, {}).get("chr_pos", {})
    items = sorted(chr_pos.items(), key=lambda kv: kv[1])
    col_bounds = [v for _, v in items][1:]
    chr_centers, chr_names = [], []
    for k, (name, start) in enumerate(items):
        end = items[k + 1][1] if k + 1 < len(items) else Xo.shape[1]
        chr_centers.append((start + end) / 2); chr_names.append(name.replace("chr", ""))

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(Xo, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    for b in first[1:]:
        ax.axhline(b - 0.5, color="black", lw=0.6)
    for b in col_bounds:
        ax.axvline(b - 0.5, color="grey", lw=0.4)
    ax.set_yticks(centers); ax.set_yticklabels([f"cl {u}" for u in uniq], fontsize=7)
    ax.set_xticks(chr_centers); ax.set_xticklabels(chr_names, fontsize=7)
    ax.set_xlabel("genomic position (by chromosome)")
    ax.set_title(f"Per-cell CNV grouped by {group_key}  [n={Xo.shape[0]} cells]")
    fig.colorbar(im, ax=ax, shrink=0.6, label="inferred CNV")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _annotate_cells(ax, M: np.ndarray, vmax: float, fmt: str) -> None:
    """Overlay the numeric value in each cell when the grid is small enough to stay legible."""
    if M.size > 400:  # too many cells: labels would be unreadable, skip.
        return
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=6,
                        color="black" if abs(v) < 0.6 * vmax else "white")


def render_subclone_chrom_heatmap(cnv_df: pd.DataFrame, ref_row, low_res: set,
                                  out_path: Path, vmax: float) -> None:
    """Subclone x chromosome MEAN-CNV heatmap (rows = reference baseline + each subclone).

    The compact "which subclone carries which arm-level gain/loss" view: one row per
    subclone plus a leading ``reference`` baseline row (should read ~flat), columns are
    chromosomes. Low-resolution chromosomes (excluded from the Cohen's d) are marked
    with ``*`` on the axis label but still shown here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cnv_df.empty:
        print("[heatmap] no subclones at this resolution; skipping subclone chrom heatmap.", flush=True)
        return
    cols = list(cnv_df.columns)
    rows, data = [], []
    if ref_row is not None:
        rows.append("reference"); data.append(np.asarray(ref_row.reindex(cols), dtype=float))
    for lab in cnv_df.index:
        rows.append(f"subclone {lab}"); data.append(np.asarray(cnv_df.loc[lab].reindex(cols), dtype=float))
    M = np.asarray(data, dtype=float)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")

    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(cols) + 2), max(2.5, 0.5 * len(rows) + 1)))
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("chr", "") + ("*" if c in low_res else "") for c in cols], fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    if ref_row is not None:  # separate the baseline row from the subclones.
        ax.axhline(0.5, color="black", lw=1.0)
    _annotate_cells(ax, M, vmax, ".2f")
    ax.set_xlabel("chromosome  (* = low-resolution, < min genes)")
    ax.set_title("Mean inferred CNV by subclone")
    fig.colorbar(im, ax=ax, shrink=0.7, label="mean inferred CNV")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_cohensd_heatmap(cohensd_df: pd.DataFrame, out_path: Path) -> None:
    """Subclone x chromosome Cohen's-d (vs reference) heatmap.

    Significance-aware companion to the mean-CNV heatmap: separates a real event from
    depth noise. Low-resolution / degenerate chromosomes are NaN and render grey. The
    color scale is symmetric and adapts to the data (capped at +/-3).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cohensd_df.empty:
        print("[heatmap] no subclones at this resolution; skipping Cohen's d heatmap.", flush=True)
        return
    cols = list(cohensd_df.columns)
    rows = [f"subclone {lab}" for lab in cohensd_df.index]
    M = cohensd_df.to_numpy(dtype=float)
    finite = M[np.isfinite(M)]
    vmax = float(min(3.0, max(0.5, np.abs(finite).max()))) if finite.size else 1.0

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")

    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(cols) + 2), max(2.5, 0.5 * len(rows) + 1)))
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace("chr", "") for c in cols], fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    _annotate_cells(ax, M, vmax, ".1f")
    ax.set_xlabel("chromosome  (grey = low-resolution / undefined)")
    ax.set_title("Cohen's d vs reference by subclone")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Cohen's d (subclone − reference)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Mode B: reuse a previous run's adata_cnv.h5ad (re-cluster only) ----
    if args.from_h5ad:
        if args.dry_run:
            raise SystemExit("--dry-run is only for the full (--parquet) mode.")
        _require_deps()
        _run_from_h5ad(args, out_dir)
        return

    # ---- Mode A: full run from a parquet ----
    if not args.parquet or not args.segmentation:
        raise SystemExit("--parquet and --segmentation are required unless --from-h5ad is given.")
    if not args.dry_run:  # --dry-run intentionally stays dependency-light.
        _require_deps()
        if not args.annotations_csv:
            raise SystemExit("--annotations-csv is required (source of reference/tumor labels).")
    parquet = Path(args.parquet)
    sample_id = args.sample_id or parquet.stem

    import pyarrow.parquet as pq
    columns = pq.ParquetFile(parquet).schema_arrow.names
    group_col = resolve_group_column(columns, args.segmentation)
    print(f"Arm='{args.segmentation}'  group_column='{group_col}'  sample='{sample_id}'", flush=True)

    df = load_arm_transcripts(parquet, group_col, args.min_qv, args.nucleus_only)
    ref = load_gene_reference(Path(args.gene_positions))
    ref_genes = set(ref["gene_name"].astype(str))

    classify = _compartment_fn(_csl(args.reference_annotations), _csl(args.tumor_annotations),
                               args.include_neurons_as_reference)
    ann_map = load_annotation_map(Path(args.annotations_csv), args.annotation_suffix) \
        if args.annotations_csv else pd.Series(dtype="object")

    def _stats_base(n_cells, gene_arr, median_counts, median_genes, coverage, comp_counts):
        genes_set = set(np.asarray(gene_arr).astype(str))
        n_map = int(np.isin(np.asarray(gene_arr).astype(str), list(ref_genes)).sum())
        return {
            "segmentation": args.segmentation, "group_column": group_col, "sample": sample_id,
            "parquet": str(parquet),
            "n_cells_raw": int(n_cells), "n_genes_panel": int(len(genes_set)), "n_genes_mappable": n_map,
            "median_counts_per_cell": float(median_counts), "median_genes_per_cell": float(median_genes),
            "annotation_coverage": float(coverage),
            "compartment_counts": {str(k): int(v) for k, v in comp_counts.items()},
            "params": {
                "min_qv": args.min_qv, "nucleus_only": args.nucleus_only,
                "min_counts": args.min_counts, "min_genes": args.min_genes, "min_cells": args.min_cells,
                "window_size": args.window_size, "step": args.step, "lfc_clip": args.lfc_clip,
                "smoothing_neighbors": args.smoothing_neighbors, "target_sum": args.target_sum,
                "cluster_resolutions": args.cluster_resolutions,
                "annotation_suffix": args.annotation_suffix,
                "downsample_to_counts": args.downsample_to_counts,
            },
        }

    # 1. Dry-run: pandas-only plumbing + annotation-coverage check.
    if args.dry_run:
        genes_arr = df["feature_name"].astype(str).unique()
        per_cell_counts = df.groupby(group_col, observed=True).size()
        per_cell_genes = df.groupby(group_col, observed=True)["feature_name"].nunique()
        cell_ids = np.asarray(sorted(df[group_col].unique()))
        if len(ann_map):
            _, comp, cov = assign_compartments(df, group_col, cell_ids, ann_map, classify)
            comp_counts = comp.value_counts()
        else:
            comp_counts, cov = pd.Series(dtype=int), float("nan")
        stats = _stats_base(len(cell_ids), genes_arr, per_cell_counts.median(),
                            per_cell_genes.median(), cov, comp_counts)
        stats["dry_run"] = True
        (out_dir / "arm_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"Cells={stats['n_cells_raw']:,}  genes(panel)={stats['n_genes_panel']}  "
              f"genes(mappable)={stats['n_genes_mappable']}", flush=True)
        if len(ann_map):
            print(f"Annotation coverage={cov:.1%}  compartments: "
                  + ", ".join(f"{k}={v}" for k, v in comp_counts.items()), flush=True)
        else:
            print("No --annotations-csv given; skipped compartment check.", flush=True)
        if not np.isfinite(df[["x", "y"]].to_numpy(dtype="float64")).any():
            raise SystemExit("Spatial coordinates are all NaN — check x/y columns.")
        if stats["n_genes_mappable"] < 50:
            warnings.warn(f"Only {stats['n_genes_mappable']} genes map — CNV will be very coarse.", RuntimeWarning)
        print(f"[dry-run] wrote {out_dir / 'arm_stats.json'} — plumbing OK.", flush=True)
        return

    # 2. Build the matrix + AnnData.
    import anndata as ad
    import scanpy as sc

    cell_ids, genes, X, centroids = build_matrix(df, group_col)
    annotation, compartment, coverage = assign_compartments(df, group_col, cell_ids, ann_map, classify)
    del df

    adata = ad.AnnData(
        X=X.copy(),
        obs=pd.DataFrame({
            "cell_id": cell_ids, "sample": sample_id, "segmentation": args.segmentation,
            "annotation": pd.Categorical(annotation.to_numpy()),
            "compartment": pd.Categorical(compartment.to_numpy(),
                                          categories=["reference", "tumor", "unknown"]),
        }, index=cell_ids),
        var=pd.DataFrame(index=pd.Index(genes, name="feature_name")),
    )
    adata.obsm["spatial"] = centroids

    # Optional depth matching BEFORE raw_counts is frozen (inferCNV reads raw_counts).
    if args.downsample_to_counts and args.downsample_to_counts > 0:
        before = float(np.median(np.asarray(adata.X.sum(axis=1)).ravel()))
        sc.pp.downsample_counts(adata, counts_per_cell=args.downsample_to_counts,
                                random_state=args.seed)
        after = float(np.median(np.asarray(adata.X.sum(axis=1)).ravel()))
        print(f"Depth matching: capped counts/cell at {args.downsample_to_counts} "
              f"(median {before:.0f} -> {after:.0f}).", flush=True)
    adata.layers["raw_counts"] = adata.X.copy()

    sc.pp.filter_cells(adata, min_counts=args.min_counts)
    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    comp_counts = adata.obs["compartment"].value_counts()
    print(f"After cell/gene QC: {adata.n_obs:,} cells x {adata.n_vars} genes", flush=True)
    print(f"Annotation coverage={coverage:.1%}  compartments:\n{comp_counts.to_string()}", flush=True)

    n_ref = int((adata.obs["compartment"].astype(str) == "reference").sum())
    if n_ref < args.min_reference_cells:
        raise SystemExit(
            f"Only {n_ref} reference cells; need >= {args.min_reference_cells}. The inferCNV baseline "
            "would be undefined. Check --annotations-csv / --annotation-suffix / --reference-annotations.")

    # Median depth AFTER QC + any downsampling (what actually feeds inferCNV).
    rc = adata.layers["raw_counts"]
    counts_per_cell = np.asarray(rc.sum(axis=1)).ravel()
    genes_per_cell = np.asarray((rc > 0).sum(axis=1)).ravel()
    stats = _stats_base(adata.n_obs, genes, float(np.median(counts_per_cell)),
                        float(np.median(genes_per_cell)), coverage, comp_counts)

    # Expression embedding + neighbor graph at the smoothing breadth (scVelo
    # smoothing in prepare_cnv_input reads this graph). X is restored from
    # raw_counts inside prepare_cnv_input, so mutating it here is fine.
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1, adata.n_obs - 1), random_state=args.seed)
    sc.pp.neighbors(adata, n_neighbors=min(args.smoothing_neighbors, adata.n_obs - 1),
                    random_state=args.seed)

    # 3. InSituCNV (infercnvpy) pipeline via the tl API, injecting gene coordinates.
    from insitucnv.tl import (cluster_cnv_resolutions, compute_cnv_neighbors,
                              export_mean_cnv_per_gene, prepare_cnv_input, run_infercnv)

    # prepare_cnv_input returns a NEW subset AnnData when it drops genes without
    # genomic coordinates, so the return MUST be captured.
    adata = prepare_cnv_input(adata, raw_layer="raw_counts", target_sum=args.target_sum,
                              smoothing_neighbors=args.smoothing_neighbors,
                              gene_reference=ref, drop_unmapped_genes=True, copy=False)
    print(f"After genomic-position mapping: {adata.n_obs:,} cells x {adata.n_vars} CNV genes", flush=True)

    run_infercnv(adata, reference_key="compartment", reference_categories=["reference"],
                 window_size=args.window_size, step=args.step, lfc_clip=args.lfc_clip,
                 calculate_gene_values=True)
    compute_cnv_neighbors(adata)
    adata.obs["cnv_status"] = adata.obs["compartment"]

    # 4. Cluster all cells on CNV, select tumor subclones, profile them, write outputs.
    _profile_and_write(adata, args, out_dir, stats)


def _run_from_h5ad(args, out_dir: Path) -> None:
    """Mode B: reload a previous run's adata_cnv.h5ad and re-cluster/re-profile only."""
    import anndata as ad
    import infercnvpy as cnv

    adata = ad.read_h5ad(args.from_h5ad)
    if "X_cnv" not in adata.obsm:
        raise SystemExit(f"{args.from_h5ad} has no obsm['X_cnv'] — not an inferCNV output.")
    if "gene_values_cnv" not in adata.layers:
        raise SystemExit(f"{args.from_h5ad} has no layers['gene_values_cnv'] — rerun with calculate_gene_values.")
    if "compartment" not in adata.obs:
        raise SystemExit(f"{args.from_h5ad} has no obs['compartment'].")
    if "cnv_neighbors" not in adata.uns:  # graph needed for clustering
        cnv.tl.pca(adata)
        cnv.pp.neighbors(adata)
    print(f"Reloaded {adata.n_obs:,} cells x {adata.n_vars} CNV genes from {args.from_h5ad}", flush=True)

    comp_counts = adata.obs["compartment"].value_counts()
    rc = adata.layers.get("raw_counts", adata.X)
    counts_per_cell = np.asarray(rc.sum(axis=1)).ravel()
    stats = {
        "segmentation": str(adata.obs.get("segmentation", pd.Series(["?"])).iloc[0]),
        "sample": str(adata.obs.get("sample", pd.Series([Path(args.from_h5ad).stem])).iloc[0]),
        "from_h5ad": str(args.from_h5ad),
        "n_cells_raw": int(adata.n_obs), "n_genes_mappable": int(adata.n_vars),
        "median_counts_per_cell": float(np.median(counts_per_cell)),
        "compartment_counts": {str(k): int(v) for k, v in comp_counts.items()},
        "params": {"cluster_resolutions": args.cluster_resolutions,
                   "tumor_cluster_frac": args.tumor_cluster_frac,
                   "min_genes_per_chromosome": args.min_genes_per_chromosome},
    }
    _profile_and_write(adata, args, out_dir, stats)


def _spatial(adata, key: str, out_path: Path) -> None:
    try:
        from insitucnv.pl import plot_spatial
        plot_spatial(adata, color=key, output_path=out_path, spatial_key="spatial",
                     point_size=4.0, title=f"Spatial {key}")
    except Exception as exc:  # pragma: no cover
        print(f"[plot] spatial plot for {key} FAILED: {exc!r}", flush=True)


def _profile_and_write(adata, args, out_dir: Path, stats: dict) -> None:
    """Common tail: cluster all cells, select subclones, profile them, write outputs."""
    from insitucnv.tl import cluster_cnv_resolutions, export_mean_cnv_per_gene

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Per-chromosome resolution (genes + inferCNV windows); flag low-resolution chromosomes.
    chrom_res = chromosome_resolution_table(adata, args.min_genes_per_chromosome)
    chrom_res.to_csv(out_dir / "chrom_resolution.csv")
    low_res = set(chrom_res.index[chrom_res["low_resolution"]])
    print("Per-chromosome resolution (genes / windows; * = low_resolution, excluded from Cohen's d):",
          flush=True)
    for c, row in chrom_res.iterrows():
        star = " *" if row["low_resolution"] else ""
        print(f"  {c:>5}: {int(row['n_genes']):>3} genes, "
              f"{'' if pd.isna(row['n_windows']) else int(row['n_windows'])} windows{star}", flush=True)
    if low_res:
        print(f"Low-resolution chromosomes (<{args.min_genes_per_chromosome} genes): "
              f"{', '.join(sorted(low_res, key=_order_key))}", flush=True)

    per_cell_chr = per_cell_chromosome_cnv(adata)
    ref_mask = (adata.obs["compartment"].astype(str) == "reference").to_numpy()
    baseline_flatness = float(np.nanstd(per_cell_chr.to_numpy()[ref_mask])) if ref_mask.any() else float("nan")
    # Reference-cell per-chromosome mean CNV = the flat baseline row for the subclone
    # heatmap (same per-cell quantity the Cohen's d compares against).
    ref_chrom_mean = per_cell_chr[ref_mask].mean() if ref_mask.any() else None

    # Bulk sanity: mean CNV by compartment (NOT the subclone analysis).
    comp_tbl = chromosome_cluster_table(adata, "compartment")
    if not comp_tbl.empty:
        comp_tbl.to_csv(out_dir / "chrom_cnv_by_compartment.csv")

    # Cluster ALL cells on X_cnv at each resolution; subclones are the tumor clusters.
    resolutions = [float(r) for r in _csl(args.cluster_resolutions)]
    cluster_keys = cluster_cnv_resolutions(adata, resolutions, dendrogram=False)

    summary = dict(stats)
    summary.update({
        "baseline_flatness_std": baseline_flatness,
        "baseline_flatness_note": "reference-cell CNV std; partly mechanical (shrinks with depth).",
        "depth_confound_note": ("Cohen's d and baseline flatness shrink with per-cell depth; when "
                                "comparing arms, match --downsample-to-counts or read next to median depth."),
        "min_genes_per_chromosome": args.min_genes_per_chromosome,
        "low_resolution_chromosomes": sorted(low_res, key=_order_key),
        "chrom_resolution": chrom_res.reset_index().to_dict("records"),
        "tumor_cluster_frac": args.tumor_cluster_frac,
        "resolutions": {},
    })

    for r, key in zip(resolutions, cluster_keys):
        rtag = f"{r:g}"
        clusters_df, cohensd_df, cnv_df = subclone_metrics(
            adata, key, per_cell_chr, low_res, args.tumor_cluster_frac)
        clusters_df.to_csv(out_dir / f"cnv_clusters_r{rtag}.csv")
        cohensd_df.round(3).to_csv(out_dir / f"subclone_cohensd_r{rtag}.csv")
        cnv_df.round(4).to_csv(out_dir / f"subclone_chrom_cnv_r{rtag}.csv")
        pd.DataFrame({
            "cell_id": adata.obs.get("cell_id", pd.Series(adata.obs_names)).astype(str).to_numpy(),
            "cnv_cluster": adata.obs[key].astype(str).to_numpy(),
            "is_subclone": adata.obs[key].astype(str).isin(
                clusters_df.index[clusters_df["is_subclone"]].astype(str)).to_numpy(),
        }).to_csv(out_dir / f"subclone_assignments_r{rtag}.csv", index=False)

        render_cnv_heatmap(adata, key, plots_dir / f"cnv_heatmap_r{rtag}.png", args.vmax)
        _spatial(adata, key, plots_dir / f"spatial_clusters_r{rtag}.png")
        # Per-subclone CNV summaries (readable views of the two CSVs above).
        render_subclone_chrom_heatmap(cnv_df, ref_chrom_mean, low_res,
                                      plots_dir / f"subclone_chrom_cnv_heatmap_r{rtag}.png", args.vmax)
        render_cohensd_heatmap(cohensd_df, plots_dir / f"subclone_cohensd_heatmap_r{rtag}.png")

        subs = clusters_df.index[clusters_df["is_subclone"]].tolist()
        summary["resolutions"][rtag] = {
            "n_clusters": int(len(clusters_df)),
            "n_subclones": int(len(subs)),
            "subclones": {str(s): {"n_cells": int(clusters_df.loc[s, "n_cells"]),
                                   "tumor_frac": float(clusters_df.loc[s, "tumor_frac"]),
                                   "events": clusters_df.loc[s, "events"]} for s in subs},
        }
        print(f"\n=== resolution {rtag}: {len(clusters_df)} CNV clusters, {len(subs)} subclones ===",
              flush=True)
        for s in subs:
            row = clusters_df.loc[s]
            print(f"  subclone {s}: {int(row['n_cells']):>6} cells "
                  f"(tumor {row['tumor_frac']:.0%})  events: {row['events'] or '(flat)'}", flush=True)

    _spatial(adata, "compartment", plots_dir / "spatial_compartment.png")

    if (adata.obs["compartment"].astype(str) == "tumor").any():
        try:
            export_mean_cnv_per_gene(adata, out_dir / "mean_cnv_per_gene_tumor.tsv",
                                     layer="gene_values_cnv", obs_key="compartment", obs_values=("tumor",))
        except Exception as exc:  # pragma: no cover
            print(f"[export] mean CNV table failed: {exc!r}", flush=True)

    adata.write(out_dir / "adata_cnv.h5ad", compression="gzip")
    (out_dir / "arm_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. Wrote adata_cnv.h5ad, per-resolution subclone tables, and arm_summary.json to {out_dir}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
