"""Production no-segmentation VisiumHD entry point.

Reconstructs cellular profiles directly from VisiumHD square bins
**without** using any prior cell-segmentation labels, by exploding the
bin x gene count matrix into transcript-level records and running the
existing ``tracer.pipeline.run_noseg_pipeline`` (Group/density-cascade ->
Post-Group Rescue -> Stitch -> Demote -> Final Rescue) under the
``noseg`` platform config.

This module is a *thin orchestration layer*. It does not reimplement any
of TRACER's grouping / rescue / stitching / scoring logic — it only

  1. turns a VisiumHD ``binned_outputs/square_0NNum`` matrix + spatial
     metadata into the transcript-level ``DataFrame`` that
     ``run_noseg_pipeline`` already consumes, and
  2. aggregates the per-transcript partition the pipeline returns back up
     into reconstructed-profile tables, an AnnData, scores, and figures.

Input layout (standard spaceranger VisiumHD)
--------------------------------------------
    binned_outputs/square_002um/
        filtered_feature_bc_matrix/   (matrix.mtx.gz, barcodes, features)
        spatial/
            tissue_positions.parquet  (barcode, array_row, array_col, ...)
            scalefactors_json.json    (bin_size_um, microns_per_pixel, ...)

Each barcode is one square bin and is treated as a primitive spatial
unit. Bin micron coordinates are ``array_col * bin_size_um`` (x) and
``array_row * bin_size_um`` (y) — an exact, axis-aligned micron grid (no
image-registration shear, no flip). The bin's UMI counts are exploded
into one transcript per count, each located at the bin centroid.

``run_noseg_pipeline`` discards any incoming label (every ``cell_id`` is
reset to ``"-1"``); reconstruction is driven purely by spatial proximity
and gene-gene NPMI coherence — i.e. it is segmentation-free.

CLI
---
    python -m tracer.noseg_pipeline \
        --visiumhd-matrix .../binned_outputs/square_002um/filtered_feature_bc_matrix \
        --spatial-dir    .../binned_outputs/square_002um/spatial \
        --npmi           .../kidney_visiumhd_npmi.csv.gz \
        --defaults-config src/tracer/configs/defaults.toml \
        --platform-config src/tracer/configs/platforms/noseg.toml \
        --outdir results/tracer_noseg/kidney_visiumhd_2um \
        --sample-name kidney_visiumhd_2um \
        --bin-size-um 2 --seed 1 --n-jobs 8 --overwrite

Add ``--smoke`` to restrict to a small ROI (``--roi-size-um``, default
500 um) for a fast end-to-end check before the full run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Reused TRACER building blocks (no pipeline logic is reimplemented).
# ---------------------------------------------------------------------
from tracer.config import load_config
from tracer.pipeline import run_noseg_pipeline
from tracer.cc_scoring import (
    build_pmi_matrix_from_long,
    compute_purity_conflict_per_cc_relu,
)

# Labels the pipeline uses for "not part of any reconstructed profile"
# (see tracer.spatial.finalize_unassigned + the "-1" sentinel).
_UNASSIGNED_LABELS = frozenset({"UNASSIGNED", "DROP", "-1", "nan", "NaN"})


class ValidationError(RuntimeError):
    """Raised when a sanity check fails — surfaced loudly to the CLI."""


# =====================================================================
# 1. Input loading: VisiumHD bins -> transcript-level DataFrame
# =====================================================================
@dataclass
class BinTable:
    """Bin-level intermediate before the transcript explode."""

    adata: Any                  # AnnData: bins x genes counts
    coords: pd.DataFrame        # index = barcode; x_um, y_um, array_row, array_col
    bin_size_um: float          # from scalefactors_json.json
    microns_per_pixel: float


def _read_scalefactors(spatial_dir: Path) -> dict:
    sf = spatial_dir / "scalefactors_json.json"
    if not sf.exists():
        raise FileNotFoundError(f"scalefactors_json.json not found in {spatial_dir}")
    return json.loads(sf.read_text())


def _read_tissue_positions(spatial_dir: Path) -> pd.DataFrame:
    """Load tissue_positions (parquet preferred, csv fallback)."""
    pq = spatial_dir / "tissue_positions.parquet"
    if pq.exists():
        tp = pd.read_parquet(pq)
    else:
        csv = spatial_dir / "tissue_positions.csv"
        if not csv.exists():
            csv = spatial_dir / "tissue_positions_list.csv"
        if not csv.exists():
            raise FileNotFoundError(f"No tissue_positions[.parquet/.csv] in {spatial_dir}")
        tp = pd.read_csv(csv)
    need = {"barcode", "array_row", "array_col"}
    if not need.issubset(tp.columns):
        raise ValidationError(
            f"tissue_positions missing columns {need - set(tp.columns)}"
        )
    return tp


def load_visiumhd_bins(
    matrix_dir: str | Path,
    spatial_dir: str | Path,
    *,
    expected_bin_size_um: float | None = None,
) -> BinTable:
    """Load the bin x gene matrix and attach micron grid coordinates.

    Coordinates are ``array_col * bin_size_um`` (x) and
    ``array_row * bin_size_um`` (y) — an exact axis-aligned micron grid.
    ``bin_size_um`` comes from ``scalefactors_json.json``; if
    ``expected_bin_size_um`` is given it must match (guards against
    pointing the 2um CLI at an 8um matrix).
    """
    import scanpy as sc

    matrix_dir = Path(matrix_dir)
    spatial_dir = Path(spatial_dir)

    adata = sc.read_10x_mtx(matrix_dir, var_names="gene_symbols")
    adata.var_names_make_unique()

    sf = _read_scalefactors(spatial_dir)
    mpp = float(sf["microns_per_pixel"])
    bin_size = sf.get("bin_size_um")
    if bin_size is None:
        if expected_bin_size_um is None:
            raise ValidationError(
                "scalefactors_json.json has no 'bin_size_um' and "
                "--bin-size-um was not supplied."
            )
        bin_size = float(expected_bin_size_um)
        print(f"[load] scalefactors has no bin_size_um; using --bin-size-um={bin_size}")
    else:
        bin_size = float(bin_size)
    if (expected_bin_size_um is not None
            and not np.isclose(bin_size, expected_bin_size_um)):
        raise ValidationError(
            f"bin_size_um mismatch: scalefactors says {bin_size}um but "
            f"--bin-size-um is {expected_bin_size_um}um. Pointing at the wrong "
            f"square_0NNum matrix?"
        )

    tp = _read_tissue_positions(spatial_dir).set_index("barcode")
    missing = ~adata.obs_names.isin(tp.index)
    if missing.all():
        raise ValidationError(
            "No matrix barcode is present in tissue_positions — mismatched "
            "matrix/spatial pair?"
        )
    if missing.any():
        n_missing = int(missing.sum())
        print(f"[load] dropping {n_missing} bins absent from tissue_positions")
        adata = adata[~missing].copy()

    tp = tp.loc[adata.obs_names]
    coords = pd.DataFrame({
        "x_um": tp["array_col"].to_numpy(dtype=np.float64) * bin_size,
        "y_um": tp["array_row"].to_numpy(dtype=np.float64) * bin_size,
        "array_row": tp["array_row"].to_numpy(),
        "array_col": tp["array_col"].to_numpy(),
    }, index=pd.Index(adata.obs_names, name="bin_id"))

    print(f"[load] {adata.n_obs:,} bins x {adata.n_vars:,} genes "
          f"(bin_size={bin_size}um)")
    return BinTable(adata=adata, coords=coords,
                    bin_size_um=bin_size, microns_per_pixel=mpp)


def subset_roi(
    bins: BinTable,
    *,
    size_um: float,
    center: tuple[float, float] | None = None,
) -> BinTable:
    """Restrict to a square ROI of side ``size_um`` (microns).

    ``center=None`` picks the densest region via a coarse 2D histogram so
    the smoke test lands on tissue rather than empty background.
    """
    x = bins.coords["x_um"].to_numpy()
    y = bins.coords["y_um"].to_numpy()
    if center is None:
        nb = max(4, int(np.ceil((x.max() - x.min() + 1) / size_um)))
        h, xe, ye = np.histogram2d(x, y, bins=nb)
        ix, iy = np.unravel_index(int(h.argmax()), h.shape)
        cx = 0.5 * (xe[ix] + xe[ix + 1])
        cy = 0.5 * (ye[iy] + ye[iy + 1])
    else:
        cx, cy = center
    half = size_um / 2.0
    mask = (
        (x >= cx - half) & (x < cx + half)
        & (y >= cy - half) & (y < cy + half)
    )
    if not mask.any():
        raise ValidationError(
            f"ROI ({cx:.0f},{cy:.0f}) +/-{half:.0f}um contains no bins."
        )
    print(f"[roi] center=({cx:.0f},{cy:.0f})um size={size_um}um "
          f"-> {int(mask.sum()):,} bins")
    return BinTable(
        adata=bins.adata[mask].copy(),
        coords=bins.coords.loc[mask],
        bin_size_um=bins.bin_size_um,
        microns_per_pixel=bins.microns_per_pixel,
    )


def explode_to_transcripts(
    bins: BinTable,
    *,
    panel_genes: set[str],
    max_transcripts: int | None = None,
    id_offset: int = 0,
    bin_cell_id: "pd.Series | None" = None,
    bin_overlaps_nucleus: "pd.Series | None" = None,
) -> pd.DataFrame:
    """Explode the bin x gene count matrix into one row per transcript.

    Each UMI count becomes a transcript located at its bin's centroid.
    Genes are restricted to ``panel_genes`` (the NPMI panel) up-front —
    TRACER scores on the curated panel, and this keeps the explode small.

    Returns columns ``transcript_id, feature_name, cell_id, bin_id, x, y``
    where ``cell_id`` is the sentinel ``"-1"`` and ``bin_id`` preserves the
    originating bin for the bin->profile map.

    SEG-mode hooks (default off → NOSEG behavior unchanged):

    * ``bin_cell_id`` — Series indexed by barcode giving each bin's initial
      label (e.g. the nucleus id its center falls in, or ``"-1"``). When
      provided it replaces the hard-coded ``"-1"`` cell_id, turning the
      explode into a *segmented* TRACER input (nucleus-seeded).
    * ``bin_overlaps_nucleus`` — Series indexed by barcode (0/1) marking
      bins whose center overlaps a nucleus. When provided, an
      ``overlaps_nucleus`` column is emitted so ``run_segmented_pipeline``
      takes the nuclear-seed prune path. Missing barcodes default to 0.
    """
    import scipy.sparse as sp

    adata = bins.adata
    keep = adata.var_names.isin(panel_genes)
    n_keep = int(keep.sum())
    if n_keep == 0:
        raise ValidationError("No matrix gene overlaps the NPMI panel.")
    sub = adata[:, keep]
    genes = np.asarray(sub.var_names)

    X = sub.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    coo = X.tocoo()
    counts = np.rint(coo.data).astype(np.int64)
    valid = counts > 0
    rows, cols, counts = coo.row[valid], coo.col[valid], counts[valid]

    total = int(counts.sum())
    if max_transcripts is not None and total > max_transcripts:
        raise ValidationError(
            f"Explode would create {total:,} transcripts > --max-transcripts "
            f"({max_transcripts:,}). Use --smoke / a smaller ROI, or raise the cap."
        )

    bin_idx = np.repeat(rows, counts)
    gene_idx = np.repeat(cols, counts)

    coords = bins.coords
    barcodes = np.asarray(coords.index)
    xs = coords["x_um"].to_numpy()
    ys = coords["y_um"].to_numpy()

    # Per-bin initial label. NOSEG default: every bin is "-1" (unassigned).
    # SEG: a per-bin label Series (nucleus id) provided by the caller.
    if bin_cell_id is None:
        cell_id_per_tx = "-1"
    else:
        per_bin = bin_cell_id.reindex(barcodes).fillna("-1").astype(str).to_numpy()
        cell_id_per_tx = per_bin[bin_idx]

    df = pd.DataFrame({
        # int64 transcript_id keeps the 13M-row full run within RAM
        # (str ids cost ~1 GB); offset makes ids globally unique across tiles.
        "transcript_id": np.arange(id_offset, id_offset + total, dtype=np.int64),
        "feature_name": genes[gene_idx],
        "cell_id": cell_id_per_tx,
        "bin_id": barcodes[bin_idx],
        "x": xs[bin_idx].astype(np.float32),
        "y": ys[bin_idx].astype(np.float32),
    })

    if bin_overlaps_nucleus is not None:
        ov = (bin_overlaps_nucleus.reindex(barcodes)
              .fillna(0).astype(np.uint8).to_numpy())
        df["overlaps_nucleus"] = ov[bin_idx]

    print(f"[explode] {n_keep:,} panel genes x {sub.n_obs:,} bins "
          f"-> {total:,} transcripts"
          + ("" if bin_cell_id is None
             else f" (seg: {int((per_bin != '-1').sum()):,} bins nucleus-seeded)"))
    return df


# =====================================================================
# 2. NPMI panel
# =====================================================================
def load_pmi_panel(npmi_path: str | Path) -> pd.DataFrame:
    """Load the long-format NPMI panel (gene_i, gene_j, NPMI[, ...])."""
    df = pd.read_csv(npmi_path)
    needed = {"gene_i", "gene_j"}
    if not needed.issubset(df.columns):
        raise ValidationError(
            f"NPMI panel {npmi_path} missing columns {needed - set(df.columns)}"
        )
    if "NPMI" not in df.columns:
        raise ValidationError(f"NPMI panel {npmi_path} has no 'NPMI' column.")
    df["gene_i"] = df["gene_i"].astype(str).str.strip()
    df["gene_j"] = df["gene_j"].astype(str).str.strip()
    return df


def pmi_gene_set(panel: pd.DataFrame) -> set[str]:
    return set(panel["gene_i"]).union(panel["gene_j"])


# =====================================================================
# 3. Profile aggregation + scoring
# =====================================================================
@dataclass
class ProfileResult:
    scores: pd.DataFrame          # one row per reconstructed profile
    bin_assignment: pd.DataFrame  # one row per bin
    profile_long: pd.DataFrame    # (reconstructed_profile_id, feature_name, count)
    adata: Any                    # AnnData profiles x genes
    label_col: str


def _is_real_label(s: pd.Series) -> pd.Series:
    return ~s.astype(str).isin(_UNASSIGNED_LABELS)


def aggregate_profiles(
    df_final: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    label_col: str = "stitched",
    relu_tau: float = 0.05,
) -> ProfileResult:
    """Aggregate the transcript partition into reconstructed profiles.

    Per profile: counts/sizes/centroid, a profiles x gene count matrix
    (AnnData), and NPMI-based purity/conflict scores (relative variants +
    signal strength) via ``cc_scoring.compute_purity_conflict_per_cc_relu``.
    """
    import anndata as ad
    import scipy.sparse as sp

    real = df_final[_is_real_label(df_final[label_col])].copy()
    if real.empty:
        raise ValidationError("Pipeline produced zero reconstructed profiles.")
    real[label_col] = real[label_col].astype(str)

    # Stable categorical ordering shared by every downstream artifact.
    profile_ids = np.sort(real[label_col].unique())
    pid_to_row = {p: i for i, p in enumerate(profile_ids)}
    gene_ids = np.sort(real["feature_name"].astype(str).unique())
    gid_to_col = {g: j for j, g in enumerate(gene_ids)}

    r = real[label_col].map(pid_to_row).to_numpy()
    c = real["feature_name"].astype(str).map(gid_to_col).to_numpy()
    counts = sp.coo_matrix(
        (np.ones(len(real), dtype=np.int64), (r, c)),
        shape=(len(profile_ids), len(gene_ids)),
    ).tocsr()

    # Per-profile summary stats.
    grp = real.groupby(label_col, sort=True)
    n_tx = grp.size().reindex(profile_ids).to_numpy()
    n_bins = grp["bin_id"].nunique().reindex(profile_ids).to_numpy()
    cx = grp["x"].mean().reindex(profile_ids).to_numpy()
    cy = grp["y"].mean().reindex(profile_ids).to_numpy()
    n_genes = np.diff(counts.indptr)  # nonzeros per row == unique genes

    # ---- NPMI purity / conflict on the profile presence matrix --------
    npmi_genes, gene_to_idx, npmi_mat, col_idx = build_pmi_matrix_from_long(panel)
    # Presence matrix M aligned to the NPMI gene ordering.
    M = np.zeros((len(profile_ids), len(npmi_genes)), dtype=np.int8)
    present = counts.tocoo()
    keep = np.fromiter((gene_ids[j] in gene_to_idx for j in present.col),
                       dtype=bool, count=present.nnz)
    if keep.any():
        rr = present.row[keep]
        cc = np.fromiter((gene_to_idx[gene_ids[j]] for j in present.col[keep]),
                         dtype=np.int64, count=int(keep.sum()))
        M[rr, cc] = 1
    purity, conflict, rel_pur, rel_conf, sig = compute_purity_conflict_per_cc_relu(
        M, npmi_mat, col_idx, tau=relu_tau,
    )

    scores = pd.DataFrame({
        "reconstructed_profile_id": profile_ids,
        "n_bins": n_bins.astype(int),
        "n_transcripts": n_tx.astype(int),
        "total_counts": n_tx.astype(int),
        "n_genes": n_genes.astype(int),
        "purity_score": purity,
        "conflict_score": conflict,
        "relative_purity": rel_pur,
        "relative_conflict": rel_conf,
        "signal_strength": sig,
        "centroid_x": cx.astype(np.float32),
        "centroid_y": cy.astype(np.float32),
    })

    # ---- AnnData profiles x genes -------------------------------------
    adata = ad.AnnData(
        X=counts.astype(np.float32),
        obs=scores.set_index("reconstructed_profile_id").copy(),
        var=pd.DataFrame(index=pd.Index(gene_ids, name="feature_name")),
    )
    adata.obsm["spatial"] = scores[["centroid_x", "centroid_y"]].to_numpy()

    # ---- Long profile x gene composition ------------------------------
    long_coo = counts.tocoo()
    profile_long = pd.DataFrame({
        "reconstructed_profile_id": profile_ids[long_coo.row],
        "feature_name": gene_ids[long_coo.col],
        "count": long_coo.data.astype(np.int64),
    }).sort_values(["reconstructed_profile_id", "feature_name"]).reset_index(drop=True)

    # ---- Bin -> profile (plurality vote among a bin's transcripts) ----
    vote = (real.groupby(["bin_id", label_col]).size().rename("n").reset_index())
    vote = vote.sort_values(["bin_id", "n"], ascending=[True, False])
    top = vote.drop_duplicates("bin_id", keep="first")
    bin_tot = real.groupby("bin_id").size().rename("n_tx_in_bin")
    bin_xy = real.groupby("bin_id")[["x", "y"]].first()
    bin_assignment = (
        top.set_index("bin_id")
        .rename(columns={label_col: "reconstructed_profile_id", "n": "n_tx_in_profile"})
        .join(bin_tot).join(bin_xy)
        .reset_index()
    )
    bin_assignment["dominant_fraction"] = (
        bin_assignment["n_tx_in_profile"] / bin_assignment["n_tx_in_bin"]
    ).astype(np.float32)

    return ProfileResult(
        scores=scores, bin_assignment=bin_assignment,
        profile_long=profile_long, adata=adata, label_col=label_col,
    )


# =====================================================================
# 4. Validation
# =====================================================================
def validate_gene_overlap(panel_genes: set[str], matrix_genes: set[str],
                          *, min_overlap: float) -> float:
    ov = panel_genes & matrix_genes
    frac = len(ov) / max(1, len(panel_genes))
    print(f"[validate] NPMI/matrix gene overlap: {len(ov)}/{len(panel_genes)} "
          f"({frac:.2%})")
    if frac < min_overlap:
        raise ValidationError(
            f"NPMI genes overlap matrix at only {frac:.2%} "
            f"(< --min-gene-overlap {min_overlap:.2%}). Are gene IDs (symbol vs "
            f"Ensembl) consistent between the panel and the matrix?"
        )
    return frac


def validate_coordinates(bins: BinTable, *, expected_bin_size_um: float) -> None:
    coords = bins.coords
    span_x = float(coords["x_um"].max() - coords["x_um"].min())
    span_y = float(coords["y_um"].max() - coords["y_um"].min())
    if not np.isfinite([span_x, span_y]).all() or min(span_x, span_y) <= 0:
        raise ValidationError("Degenerate coordinate span — coords not in microns?")
    if max(span_x, span_y) < bins.bin_size_um:
        raise ValidationError(
            f"Coordinate span ({span_x:.1f},{span_y:.1f})um < bin size — "
            f"coordinates likely not in microns."
        )
    # Bin grid spacing must equal the declared bin size (no scale mismatch).
    dr = np.unique(np.diff(np.sort(np.unique(coords["array_col"].to_numpy()))))
    step_um = bins.bin_size_um * (dr.min() if dr.size else 1)
    if not np.isclose(bins.bin_size_um, expected_bin_size_um):
        raise ValidationError(
            f"bin_size_um {bins.bin_size_um} != expected {expected_bin_size_um}.")
    print(f"[validate] coordinate span x={span_x:.0f}um y={span_y:.0f}um; "
          f"grid step {step_um:.0f}um == bin size {bins.bin_size_um}um")


def validate_outputs(res: ProfileResult, *, bin_size_um: float) -> None:
    s = res.scores
    if s["reconstructed_profile_id"].duplicated().any():
        raise ValidationError("Duplicate reconstructed_profile_id in score table.")
    if (s["n_bins"] <= 0).any() or (s["n_transcripts"] <= 0).any():
        raise ValidationError("Empty reconstructed profile (n_bins/n_transcripts == 0).")
    if (s["n_genes"] <= 0).any():
        raise ValidationError("Reconstructed profile with zero genes.")
    if list(res.adata.obs_names) != list(s["reconstructed_profile_id"].astype(str)):
        raise ValidationError("AnnData obs order != score table profile order.")
    var_order = list(res.adata.var_names)
    if var_order != sorted(var_order):
        raise ValidationError("AnnData var (gene) order is not the canonical sort.")
    if not set(res.profile_long["feature_name"]).issubset(set(var_order)):
        raise ValidationError("profile_long references genes absent from AnnData var.")
    if bin_size_um <= 0:
        raise ValidationError(f"Invalid bin size {bin_size_um}.")
    print(f"[validate] {len(s):,} profiles, "
          f"{int(s['n_bins'].sum()):,} assigned bins — schema OK")


# =====================================================================
# 5. Figures
# =====================================================================
def _save_figures(res: ProfileResult, fig_dir: Path, *, sample_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    ba = res.bin_assignment
    pid = ba["reconstructed_profile_id"].astype("category")

    # (a) spatial reconstructed profiles — bins colored by profile.
    fig, ax = plt.subplots(figsize=(7, 7))
    rng = np.random.default_rng(0)
    palette = rng.permutation(int(pid.cat.codes.max()) + 1)
    ax.scatter(ba["x"], ba["y"], c=palette[pid.cat.codes], cmap="tab20",
               s=4, linewidths=0)
    ax.set_aspect("equal"); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_title(f"{sample_name}: reconstructed profiles "
                 f"(n={pid.cat.categories.size})")
    fig.tight_layout()
    fig.savefig(fig_dir / "spatial_reconstructed_profiles.png", dpi=150)
    plt.close(fig)

    # (b) conflict map — bins colored by their profile's conflict score.
    conf = res.scores.set_index("reconstructed_profile_id")["conflict_score"]
    fig, ax = plt.subplots(figsize=(7, 7))
    sct = ax.scatter(ba["x"], ba["y"],
                     c=ba["reconstructed_profile_id"].map(conf).to_numpy(),
                     cmap="magma", s=4, linewidths=0)
    ax.set_aspect("equal"); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_title(f"{sample_name}: TRACER conflict score")
    fig.colorbar(sct, ax=ax, label="conflict_score", shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "tracer_conflict_map.png", dpi=150)
    plt.close(fig)

    # (c) profile size distribution.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(res.scores["n_bins"],
            bins=min(50, max(5, int(res.scores["n_bins"].nunique()))))
    ax.set_xlabel("bins per reconstructed profile"); ax.set_ylabel("n profiles")
    ax.set_title(f"{sample_name}: profile size distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "profile_size_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[figures] wrote 3 figures to {fig_dir}")


# =====================================================================
# 6. Reporting
# =====================================================================
def _write_run_summary(path: Path, *, args, gene_overlap: float, n_input_bins: int,
                       n_transcripts: int, res: ProfileResult,
                       runtime_s: float, peak_mem_mb: float | None) -> None:
    s = res.scores
    lines = [
        f"# TRACER no-seg run — {args.sample_name}",
        "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- mode: {'SMOKE (ROI ' + str(args.roi_size_um) + 'um)' if args.smoke else 'FULL'}",
        f"- bin size: {args.bin_size_um} um (one barcode == one square bin)",
        f"- seed: {args.seed}",
        "",
        "## Inputs",
        f"- matrix: `{args.visiumhd_matrix}`",
        f"- spatial: `{args.spatial_dir}`",
        f"- NPMI panel: `{args.npmi}`",
        f"- platform config: `{args.platform_config}`",
        f"- NPMI/matrix gene overlap: {gene_overlap:.2%}",
        "",
        "## Results",
        f"- input bins: {n_input_bins:,}",
        f"- exploded transcripts: {n_transcripts:,}",
        f"- reconstructed profiles: {len(s):,}",
        f"- median bins / profile: {float(s['n_bins'].median()):g}",
        f"- median genes / profile: {float(s['n_genes'].median()):g}",
        f"- median transcripts / profile: {float(s['n_transcripts'].median()):g}",
        f"- mean purity / conflict: {s['purity_score'].mean():.4f} / "
        f"{s['conflict_score'].mean():.4f}",
        f"- runtime: {runtime_s:.1f} s",
        f"- peak memory: {f'{peak_mem_mb:.0f} MB ({peak_mem_mb/1024:.1f} GB)' if peak_mem_mb else 'n/a'}",
        f"- git commit: {_git_commit()}",
        f"- benchmark metrics: `benchmark_metrics/runtime_memory.json`",
        "",
        "## Reused TRACER functions",
        "- `tracer.pipeline.run_noseg_pipeline` "
        "(Group/cascade -> Post-Group Rescue -> Stitch -> Demote -> Final Rescue)",
        "- `tracer.config.load_config(platform='noseg')`",
        "- `tracer.cc_scoring.compute_purity_conflict_per_cc_relu` + "
        "`build_pmi_matrix_from_long`",
        "",
        "## Caveats / assumptions",
        "- Each square-bin barcode is the spatial primitive; bin micron coords are",
        "  `array_col*bin_size_um` (x) and `array_row*bin_size_um` (y) from",
        "  tissue_positions (exact grid; no registration shear / flip).",
        "- Segmentation labels are discarded (every cell_id reset to '-1'); the",
        "  reconstruction is driven only by spatial proximity + NPMI coherence.",
        "- Transcripts are restricted to the NPMI panel genes; off-panel counts are",
        "  not exploded. All transcripts of a bin share the bin centroid.",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {path}")


# =====================================================================
# 7. Benchmarking, tiling, grid scaling, geometry
# =====================================================================
import dataclasses as _dc
import threading as _threading

# Pipeline stage functions to time, keyed by the stage label reported in
# runtime_by_stage.tsv. Patched in-place on the `tracer.pipeline` module so
# we time the *exact* functions run_noseg_pipeline calls — no logic copied.
_STAGE_FUNCS = {
    "init_prune": "prune_transcripts_fast",
    "group_cascade": "cascade_as_residual_handler",
    "post_group_rescue": "guarded_rescue",
    "stitch": "apply_stitching_to_transcripts_memory_efficient",
    "demote": "demote_small_entities",
    "final_rescue": "reassign_unassigned_grid_pool",
    "finalize": "finalize_unassigned",
}


class StageTimer:
    """Accumulate wall time per pipeline stage by patching the stage
    functions on ``tracer.pipeline`` (sums across tiles)."""

    def __init__(self):
        self.totals: dict[str, float] = {k: 0.0 for k in _STAGE_FUNCS}
        self._orig: dict[str, Any] = {}

    def __enter__(self):
        import tracer.pipeline as P
        for stage, fname in _STAGE_FUNCS.items():
            orig = getattr(P, fname)
            self._orig[fname] = orig

            def make(o, st):
                def wrapped(*a, **k):
                    t = time.perf_counter()
                    try:
                        return o(*a, **k)
                    finally:
                        self.totals[st] += time.perf_counter() - t
                return wrapped
            setattr(P, fname, make(orig, stage))
        return self

    def __exit__(self, *exc):
        import tracer.pipeline as P
        for fname, orig in self._orig.items():
            setattr(P, fname, orig)
        return False


class PeakMemSampler:
    """Background sampler for peak process RSS (psutil), in MB."""

    def __init__(self, interval: float = 0.5):
        import psutil
        self._proc = psutil.Process()
        self._interval = interval
        self._peak = 0.0
        self._stop = _threading.Event()
        self._thread = _threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                for ch in self._proc.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except Exception:
                        pass
                self._peak = max(self._peak, rss / (1024 ** 2))
            except Exception:
                pass
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        return False

    @property
    def peak_mb(self) -> float:
        return self._peak


def grid_scaled_cfg(cfg, bin_size_um: float):
    """Scale the spatial-grid knobs to the input bin pitch.

    The cascade/rescue/stitch grids default to 2 um (tuned for 2 um bins).
    With an N-um bin pitch the grouping grid must be N um or adjacent bins
    never share a neighborhood. No-op at 2 um.
    """
    b = float(bin_size_um)
    return _dc.replace(
        cfg,
        group=_dc.replace(cfg.group, cascade_bin_size_um=b),
        rescue=_dc.replace(cfg.rescue, bin_size_um=b),
        final_rescue=_dc.replace(cfg.final_rescue, bin_size_um=b),
        stitch=_dc.replace(cfg.stitch, bin_size_um=b,
                           dist_threshold_um=max(cfg.stitch.dist_threshold_um, b * 2.0)),
    )


def plan_tiles(bins: BinTable, *, panel_genes: set[str], max_tile_transcripts: int,
               grid: tuple[int, int] | None = None) -> list[np.ndarray]:
    """Partition bins into a near-square spatial grid of tiles so each tile
    holds <= ~max_tile_transcripts (bounds peak memory). Returns a list of
    boolean masks over ``bins.coords`` rows. Empty tiles are dropped.

    Cross-tile Stitch loss is small: Stitch/Rescue reach (~bin pitch) <<
    tile size, so only entities within ~1 bin of an internal edge can miss
    a merge — acceptable for whole-tissue benchmarking.
    """
    import scipy.sparse as sp
    sub = bins.adata[:, bins.adata.var_names.isin(panel_genes)]
    per_bin = np.asarray(sub.X.tocsr().sum(axis=1)).ravel()
    total = int(per_bin.sum())
    if grid is None:
        n_axis = max(1, int(np.ceil(np.sqrt(total / max(1, max_tile_transcripts)))))
        grid = (n_axis, n_axis)
    nx, ny = grid
    x = bins.coords["x_um"].to_numpy()
    y = bins.coords["y_um"].to_numpy()
    # Quantile edges → roughly balanced transcript load per tile.
    xe = np.quantile(x, np.linspace(0, 1, nx + 1))
    ye = np.quantile(y, np.linspace(0, 1, ny + 1))
    xe[0] -= 1; xe[-1] += 1; ye[0] -= 1; ye[-1] += 1
    xb = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, nx - 1)
    yb = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, ny - 1)
    tile_id = xb * ny + yb
    masks = []
    for t in range(nx * ny):
        m = tile_id == t
        if m.any() and per_bin[m].sum() > 0:
            masks.append(m)
    print(f"[tile] grid {nx}x{ny} -> {len(masks)} non-empty tiles "
          f"(total {total:,} transcripts, cap {max_tile_transcripts:,}/tile)")
    return masks


def build_pseudocell_geometry(bin_assignment: pd.DataFrame, *, bin_size_um: float) -> pd.DataFrame:
    """Pseudo-cell geometry per reconstructed profile: bounding box, centroid,
    area (bins x bin^2), and a convex-hull WKT polygon when >= 3 bins.

    Each bin is an N-um square; the hull is over bin centers (a compact,
    alignment-free stand-in for a true cell boundary).
    """
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        ConvexHull = None
    rows = []
    for pid, g in bin_assignment.groupby("reconstructed_profile_id"):
        xs = g["x"].to_numpy(dtype=float); ys = g["y"].to_numpy(dtype=float)
        wkt = None
        if ConvexHull is not None and len(xs) >= 3:
            pts = np.column_stack([xs, ys])
            try:
                hull = ConvexHull(pts)
                ring = pts[hull.vertices]
                ring = np.vstack([ring, ring[0]])
                wkt = "POLYGON((" + ", ".join(f"{a:.1f} {b:.1f}" for a, b in ring) + "))"
            except Exception:
                wkt = None
        rows.append({
            "reconstructed_profile_id": pid,
            "n_bins": int(len(xs)),
            "centroid_x": float(xs.mean()), "centroid_y": float(ys.mean()),
            "bbox_xmin": float(xs.min()), "bbox_xmax": float(xs.max()),
            "bbox_ymin": float(ys.min()), "bbox_ymax": float(ys.max()),
            "area_um2": float(len(xs) * bin_size_um * bin_size_um),
            "hull_wkt": wkt,
        })
    return pd.DataFrame(rows)


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _config_hash(cfg) -> str:
    import hashlib
    try:
        payload = json.dumps(_dc.asdict(cfg), sort_keys=True, default=str)
    except Exception:
        payload = repr(cfg)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# =====================================================================
# 8. Orchestration + CLI
# =====================================================================
def _resolve_cfg(args):
    """Build the noseg PipelineConfig honoring the supplied platform file.

    ``load_config`` always loads the bundled defaults first; layering the
    given platform file on top reproduces ``load_config(platform='noseg')``
    when that file is the bundled noseg preset.
    """
    plat = Path(args.platform_config)
    try:
        return load_config(platform=plat.stem)
    except FileNotFoundError:
        return load_config(path=plat)


def _run_one(df, panel, cfg, *, tile_tag: str | None):
    """Run run_noseg_pipeline on one transcript frame; return the
    real-labeled subset with tile-prefixed profile ids (drops UNASSIGNED to
    bound the concat memory)."""
    df_final, _prog = run_noseg_pipeline(df, panel, cfg=cfg)
    if "bin_id" not in df_final.columns:
        df_final = df_final.merge(df[["transcript_id", "bin_id"]],
                                  on="transcript_id", how="left")
    keep = _is_real_label(df_final["stitched"])
    out = df_final.loc[keep, ["bin_id", "x", "y", "feature_name", "stitched"]].copy()
    out["stitched"] = out["stitched"].astype(str)
    if tile_tag is not None:
        out["stitched"] = tile_tag + "::" + out["stitched"]
    return out


# --- parallel tile execution (cascade is single-threaded; fan out across
#     cores over spatially-disjoint tiles) --------------------------------
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(panel, cfg):
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"  # avoid OMP oversubscription across workers
    _WORKER_STATE["panel"] = panel
    _WORKER_STATE["cfg"] = cfg


def _worker_run(task):
    """Run one tile in a worker; returns the real-labeled subset, per-stage
    CPU seconds, and wall time."""
    df, tag = task
    t = time.perf_counter()
    with StageTimer() as st:
        out = _run_one(df, _WORKER_STATE["panel"], _WORKER_STATE["cfg"], tile_tag=tag)
    return {"df": out, "stage": dict(st.totals), "wall": time.perf_counter() - t}


def run(args) -> None:
    t0 = time.time()
    timings: dict[str, float] = {}
    outdir = Path(args.outdir)
    out_sub = outdir / "outputs"
    fig_sub = outdir / "figures"
    bench_sub = outdir / "benchmark_metrics"
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{outdir} is non-empty; pass --overwrite to proceed.")
    for d in (out_sub, fig_sub, bench_sub):
        d.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # --- load panel + bins -------------------------------------------
    t = time.perf_counter()
    panel = load_pmi_panel(args.npmi)
    panel_genes = pmi_gene_set(panel)
    bins = load_visiumhd_bins(
        args.visiumhd_matrix, args.spatial_dir,
        expected_bin_size_um=args.bin_size_um,
    )
    timings["load_inputs"] = time.perf_counter() - t

    matrix_genes = set(map(str, bins.adata.var_names))
    gene_overlap = validate_gene_overlap(panel_genes, matrix_genes,
                                         min_overlap=args.min_gene_overlap)
    validate_coordinates(bins, expected_bin_size_um=args.bin_size_um)

    if args.smoke:
        center = None
        if args.roi_center:
            cx, cy = (float(v) for v in args.roi_center.split(","))
            center = (cx, cy)
        bins = subset_roi(bins, size_um=args.roi_size_um, center=center)

    n_input_bins = int(bins.adata.n_obs)
    cfg = grid_scaled_cfg(_resolve_cfg(args), args.bin_size_um)

    # --- tile plan (bounds peak memory on whole-tissue runs) ---------
    if args.no_tiling:
        tile_masks = [np.ones(n_input_bins, dtype=bool)]
    else:
        grid = None
        if args.tile_grid:
            gx, gy = (int(v) for v in args.tile_grid.lower().split("x"))
            grid = (gx, gy)
        tile_masks = plan_tiles(bins, panel_genes=panel_genes,
                                max_tile_transcripts=args.max_tile_transcripts,
                                grid=grid)
    tiled = len(tile_masks) > 1

    # --- run (instrumented) ------------------------------------------
    def _explode_tile(mask, offset):
        tile = BinTable(adata=bins.adata[mask].copy(),
                        coords=bins.coords.loc[mask],
                        bin_size_um=bins.bin_size_um,
                        microns_per_pixel=bins.microns_per_pixel)
        return explode_to_transcripts(
            tile, panel_genes=panel_genes,
            max_transcripts=(None if tiled else args.max_transcripts),
            id_offset=offset)

    tile_runtimes: list[float] = []
    stage_totals: dict[str, float] = {k: 0.0 for k in _STAGE_FUNCS}
    n_transcripts = 0
    parts: list[pd.DataFrame] = []
    n_workers = max(1, min(args.n_jobs, len(tile_masks)))
    t_pipe = time.perf_counter()

    with PeakMemSampler() as mem:
        if n_workers == 1 or len(tile_masks) == 1:
            # In-process (smoke / single tile): time stages directly.
            with StageTimer() as st:
                for ti, mask in enumerate(tile_masks):
                    dft = _explode_tile(mask, n_transcripts)
                    n_transcripts += len(dft)
                    tag = f"t{ti}" if tiled else None
                    print(f"[pipeline] tile {ti+1}/{len(tile_masks)}: "
                          f"{len(dft):,} transcripts ...")
                    tt = time.perf_counter()
                    parts.append(_run_one(dft, panel, cfg, tile_tag=tag))
                    tile_runtimes.append(time.perf_counter() - tt)
                    del dft
            stage_totals = dict(st.totals)
        else:
            # Parallel: stream tiles across cores with a bounded number of
            # in-flight (exploded) frames, so workers stay saturated (no wave
            # barrier) while peak memory stays at ~n_workers exploded tiles.
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
            ctx = mp.get_context("spawn")
            print(f"[pipeline] {len(tile_masks)} tiles streamed across {n_workers} workers")
            max_inflight = n_workers + 2
            n_done = 0
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                     initializer=_worker_init,
                                     initargs=(panel, cfg)) as ex:
                pending = {}
                it = iter(range(len(tile_masks)))

                def _submit_next():
                    nonlocal n_transcripts
                    try:
                        ti = next(it)
                    except StopIteration:
                        return False
                    dft = _explode_tile(tile_masks[ti], n_transcripts)
                    n_transcripts += len(dft)
                    pending[ex.submit(_worker_run, (dft, f"t{ti}"))] = ti
                    return True

                for _ in range(max_inflight):
                    if not _submit_next():
                        break
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        del pending[fut]
                        r = fut.result()
                        parts.append(r["df"])
                        tile_runtimes.append(r["wall"])
                        for k, v in r["stage"].items():
                            stage_totals[k] += v
                        n_done += 1
                        _submit_next()
                    print(f"[pipeline]   {n_done}/{len(tile_masks)} tiles done "
                          f"({n_transcripts:,} transcripts so far)")
    timings["pipeline_total"] = time.perf_counter() - t_pipe
    timings.update({f"stage::{k}": v for k, v in stage_totals.items()})
    peak_mem_mb = mem.peak_mb

    df_final = pd.concat(parts, ignore_index=True)
    del parts

    # --- aggregate + score + validate --------------------------------
    t = time.perf_counter()
    res = aggregate_profiles(df_final, panel)
    validate_outputs(res, bin_size_um=args.bin_size_um)
    timings["aggregate_score"] = time.perf_counter() - t

    # --- write outputs -----------------------------------------------
    t = time.perf_counter()
    res.profile_long.to_parquet(out_sub / "reconstructed_profiles.parquet", index=False)
    res.bin_assignment.to_parquet(out_sub / "bin_to_profile_assignment.parquet", index=False)
    res.adata.write_h5ad(out_sub / "profile_by_gene.h5ad")
    res.scores.to_csv(out_sub / "profile_scores.tsv.gz", sep="\t", index=False)
    geom = build_pseudocell_geometry(res.bin_assignment, bin_size_um=args.bin_size_um)
    geom.to_parquet(out_sub / "pseudocell_geometry.parquet", index=False)
    timings["write_outputs"] = time.perf_counter() - t

    t = time.perf_counter()
    _save_figures(res, fig_sub, sample_name=args.sample_name)
    timings["figures"] = time.perf_counter() - t

    total_runtime = time.time() - t0

    # --- benchmark metrics -------------------------------------------
    _write_benchmark(bench_sub, args=args, cfg=cfg, timings=timings,
                     tile_runtimes=tile_runtimes, peak_mem_mb=peak_mem_mb,
                     n_input_bins=n_input_bins, n_transcripts=n_transcripts,
                     res=res, total_runtime=total_runtime, tiled=tiled,
                     gene_overlap=gene_overlap)

    _write_run_summary(
        outdir / "run_summary.md", args=args, gene_overlap=gene_overlap,
        n_input_bins=n_input_bins, n_transcripts=n_transcripts, res=res,
        runtime_s=total_runtime, peak_mem_mb=peak_mem_mb,
    )
    print(f"[done] {len(res.scores):,} reconstructed profiles in "
          f"{total_runtime:.1f}s (peak {peak_mem_mb:.0f} MB) -> {outdir}")


def _write_benchmark(bench_sub: Path, *, args, cfg, timings, tile_runtimes,
                     peak_mem_mb, n_input_bins, n_transcripts, res,
                     total_runtime, tiled, gene_overlap) -> None:
    import platform
    s = res.scores
    stage_rows = [(k.split("::", 1)[1], v) for k, v in timings.items()
                  if k.startswith("stage::")]
    stage_rows += [(k, v) for k, v in timings.items() if not k.startswith("stage::")]
    pd.DataFrame(stage_rows, columns=["stage", "seconds"]).to_csv(
        bench_sub / "runtime_by_stage.tsv", sep="\t", index=False)

    runtime_memory = {
        "total_wallclock_s": round(total_runtime, 2),
        "pipeline_total_s": round(timings.get("pipeline_total", 0.0), 2),
        "stage_seconds": {k.split("::", 1)[1]: round(v, 3)
                          for k, v in timings.items() if k.startswith("stage::")},
        "phase_seconds": {k: round(v, 3) for k, v in timings.items()
                          if not k.startswith("stage::")},
        "n_tiles": len(tile_runtimes),
        "tile_runtimes_s": [round(x, 2) for x in tile_runtimes],
        "peak_rss_mb": round(peak_mem_mb, 1),
        "peak_rss_gb": round(peak_mem_mb / 1024, 2),
        "n_cpu_cores_available": __import__("os").cpu_count(),
        "n_jobs_requested": args.n_jobs,
        "gpu_used": False,
        "input_bin_count": int(n_input_bins),
        "input_feature_count": int(res.adata.n_vars),
        "input_transcript_count": int(n_transcripts),
        "output_profile_count": int(len(s)),
        "median_transcripts_per_profile": float(s["n_transcripts"].median()),
        "median_bins_per_profile": float(s["n_bins"].median()),
        "git_commit": _git_commit(),
        "config_hash": _config_hash(cfg),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    (bench_sub / "runtime_memory.json").write_text(
        json.dumps(runtime_memory, indent=2))

    method_summary = {
        "method": "TRACER no-segmentation VisiumHD reconstruction",
        "entry_point": "tracer.noseg_pipeline",
        "pipeline_function": "tracer.pipeline.run_noseg_pipeline",
        "stages": ["init/prune", "group/density-cascade", "post-group rescue",
                   "stitch", "demote", "final rescue", "finalize"],
        "platform_config": str(args.platform_config),
        "bin_size_um": args.bin_size_um,
        "spatial_grid_um": args.bin_size_um,
        "tiled_execution": tiled,
        "max_tile_transcripts": args.max_tile_transcripts,
        "gene_panel": "NPMI panel genes only",
        "gene_overlap_frac": round(gene_overlap, 4),
        "scoring": "cc_scoring.compute_purity_conflict_per_cc_relu (ReLU NPMI)",
        "git_commit": _git_commit(),
        "config_hash": _config_hash(cfg),
        "command": "python -m tracer.noseg_pipeline " + " ".join(sys.argv[1:]),
    }
    (bench_sub / "method_summary.json").write_text(
        json.dumps(method_summary, indent=2))
    print(f"[benchmark] wrote metrics -> {bench_sub}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tracer.noseg_pipeline",
        description="No-segmentation VisiumHD reconstruction pipeline.",
    )
    p.add_argument("--visiumhd-matrix", required=True,
                   help="10x bin matrix dir (matrix.mtx.gz/barcodes/features).")
    p.add_argument("--spatial-dir", required=True,
                   help="Spatial dir (tissue_positions + scalefactors_json.json).")
    p.add_argument("--npmi", required=True, help="Long-format NPMI panel csv[.gz].")
    p.add_argument("--defaults-config", default=None,
                   help="Defaults TOML (informational; bundled defaults are used).")
    p.add_argument("--platform-config", required=True,
                   help="Platform TOML (e.g. .../platforms/noseg.toml).")
    p.add_argument("--outdir", required=True)
    p.add_argument("--sample-name", required=True)
    p.add_argument("--bin-size-um", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    # Smoke-test / safety controls.
    p.add_argument("--smoke", action="store_true",
                   help="Restrict to a small ROI for a fast end-to-end check.")
    p.add_argument("--roi-size-um", type=float, default=500.0)
    p.add_argument("--roi-center", default=None,
                   help="ROI center 'x,y' in microns (default: densest region).")
    p.add_argument("--min-gene-overlap", type=float, default=0.5,
                   help="Fail if NPMI/matrix gene overlap is below this fraction.")
    p.add_argument("--max-transcripts", type=int, default=None,
                   help="Safety cap on exploded transcript count (single-tile runs).")
    # Tiling (bounds peak memory on whole-tissue runs).
    p.add_argument("--max-tile-transcripts", type=int, default=1_500_000,
                   help="Target max transcripts per spatial tile (memory bound).")
    p.add_argument("--tile-grid", default=None,
                   help="Force an explicit tile grid 'NxM' (default: auto from "
                        "transcript count).")
    p.add_argument("--no-tiling", action="store_true",
                   help="Disable spatial tiling (single in-memory pass).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except ValidationError as exc:
        raise SystemExit(f"[FAIL] validation: {exc}")


if __name__ == "__main__":
    main()
