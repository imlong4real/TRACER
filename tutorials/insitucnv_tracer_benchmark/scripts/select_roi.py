#!/usr/bin/env python3
"""Extract a small Xenium ROI parquet from a full transcript table.

Use this on argos for the full Xenium Prime dataset, then copy the ROI parquet
locally for pilot runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ensure_parent, require_columns, standardize_transcript_columns, write_json


CELL_CLASS_COLORS = {
    "Epithelial": "#1f77b4",
    "Fibroblasts": "#ff7f0e",
    "Monocytes": "#2ca02c",
    "Monocytes/macrophages": "#2ca02c",
    "T cells": "#9467bd",
    "B cells": "#8c564b",
    "B/plasma cells": "#8c564b",
    "Endothelial": "#d62728",
    "Low expression": "#e377c2",
}

CNV_EPI_COLORS = {
    "0": "#1f77b4",
    "1": "#ff7f0e",
    "2": "#2ca02c",
    "3": "#d62728",
    "4": "#9467bd",
    "filtered_small_clone": "#7f7f7f",
    "not_epithelial": "#b0b0b0",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--transcripts",
        required=True,
        type=Path,
        help="Full Xenium transcripts parquet.",
    )
    p.add_argument("--out", required=True, type=Path, help="ROI transcript parquet.")
    p.add_argument(
        "--cell-metadata",
        type=Path,
        default=None,
        help="Optional cell metadata CSV/parquet to subset.",
    )
    p.add_argument("--cell-metadata-out", type=Path, default=None)
    p.add_argument("--xmin", type=float, required=True)
    p.add_argument("--xmax", type=float, required=True)
    p.add_argument("--ymin", type=float, required=True)
    p.add_argument("--ymax", type=float, required=True)
    p.add_argument("--min-qv", type=float, default=None)
    p.add_argument("--remove-controls", action="store_true")
    p.add_argument(
        "--preview-out",
        type=Path,
        default=None,
        help="Optional 3-panel ROI preview PNG.",
    )
    p.add_argument(
        "--preview-counts-out",
        type=Path,
        default=None,
        help="Optional ROI preview counts CSV.",
    )
    p.add_argument(
        "--preview-summary-out",
        type=Path,
        default=None,
        help="Optional ROI preview summary JSON.",
    )
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_roi_transcripts(args: argparse.Namespace) -> pd.DataFrame:
    """Read only the bbox when possible; fall back to pandas for local toy files."""
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(args.transcripts, format="parquet")
        names = set(dataset.schema.names)
        x_col = "x" if "x" in names else "x_location"
        y_col = "y" if "y" in names else "y_location"
        filt = (
            (ds.field(x_col) >= args.xmin)
            & (ds.field(x_col) <= args.xmax)
            & (ds.field(y_col) >= args.ymin)
            & (ds.field(y_col) <= args.ymax)
        )
        if args.min_qv is not None and "qv" in names:
            filt = filt & (ds.field("qv") >= args.min_qv)
        table = dataset.to_table(filter=filt)
        return table.to_pandas()
    except Exception as exc:
        print(
            f"[warn] PyArrow bbox read failed ({exc}); falling back to pandas full read.",
            flush=True,
        )
        df = pd.read_parquet(args.transcripts)
        df = standardize_transcript_columns(df)
        mask = (
            (df["x"].astype(float) >= args.xmin)
            & (df["x"].astype(float) <= args.xmax)
            & (df["y"].astype(float) >= args.ymin)
            & (df["y"].astype(float) <= args.ymax)
        )
        if args.min_qv is not None and "qv" in df.columns:
            mask &= df["qv"].astype(float) >= args.min_qv
        return df.loc[mask].copy()


def bbox_dict(args: argparse.Namespace) -> dict[str, float]:
    return {"xmin": args.xmin, "xmax": args.xmax, "ymin": args.ymin, "ymax": args.ymax}


def standardize_cell_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "x_centroid" in df.columns and "x" not in df.columns:
        rename["x_centroid"] = "x"
    if "y_centroid" in df.columns and "y" not in df.columns:
        rename["y_centroid"] = "y"
    if "cell class" in df.columns and "cell_class" not in df.columns:
        rename["cell class"] = "cell_class"
    if rename:
        df = df.rename(columns=rename)
    return df


def metadata_bbox_mask(meta: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    require_columns(meta, {"x", "y"}, "cell metadata")
    return (
        (meta["x"].astype(float) >= args.xmin)
        & (meta["x"].astype(float) <= args.xmax)
        & (meta["y"].astype(float) >= args.ymin)
        & (meta["y"].astype(float) <= args.ymax)
    )


def label_sort_key(label: str) -> tuple[int, int | str]:
    if label == "not_epithelial":
        return (2, label)
    if label == "filtered_small_clone":
        return (1, label)
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def make_palette(labels: pd.Series, fixed_colors: dict[str, str]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    levels = sorted(labels.dropna().astype(str).unique(), key=label_sort_key)
    cmap = plt.get_cmap("tab20")
    palette = {}
    extra_idx = 0
    for level in levels:
        if level in fixed_colors:
            palette[level] = fixed_colors[level]
        else:
            palette[level] = cmap(extra_idx % cmap.N)
            extra_idx += 1
    return palette


def write_preview_counts(meta_roi: pd.DataFrame, out: Path) -> None:
    rows = []
    for category in ("cell_class", "cnv_epi", "cnv_epi_raw", "cnv_leiden"):
        if category not in meta_roi.columns:
            continue
        counts = meta_roi[category].fillna("missing").astype(str).value_counts()
        for label, n_cells in counts.items():
            rows.append({"category": category, "label": label, "n_cells": int(n_cells)})
    ensure_parent(out)
    pd.DataFrame(rows, columns=["category", "label", "n_cells"]).to_csv(out, index=False)


def plot_points(
    ax,
    df: pd.DataFrame,
    color_by: str,
    fixed_colors: dict[str, str],
    point_size: float,
) -> list:
    from matplotlib.lines import Line2D

    labels = df[color_by].fillna("missing").astype(str)
    palette = make_palette(labels, fixed_colors)
    colors = labels.map(palette)
    ax.scatter(
        df["x"].to_numpy(dtype=float),
        df["y"].to_numpy(dtype=float),
        c=list(colors),
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=4, label=label)
        for label, color in palette.items()
    ]
    return handles


def plot_roi_preview(meta: pd.DataFrame, meta_roi: pd.DataFrame, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    require_columns(meta_roi, {"cell_class", "cnv_epi"}, "cell metadata for preview")
    ensure_parent(args.preview_out)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    ax_context, ax_class, ax_cnv = axes

    if "cell_class" in meta.columns:
        context_labels = meta["cell_class"].fillna("missing").astype(str)
    else:
        context_labels = pd.Series("cell", index=meta.index)
    context_palette = make_palette(context_labels, CELL_CLASS_COLORS)
    ax_context.scatter(
        meta["x"].to_numpy(dtype=float),
        meta["y"].to_numpy(dtype=float),
        c=list(context_labels.map(context_palette)),
        s=0.2,
        linewidths=0,
        alpha=0.45,
        rasterized=True,
    )
    ax_context.add_patch(
        Rectangle(
            (args.xmin, args.ymin),
            args.xmax - args.xmin,
            args.ymax - args.ymin,
            fill=False,
            edgecolor="black",
            linewidth=1,
        )
    )
    ax_context.set_title("Full slide context")

    class_handles = plot_points(ax_class, meta_roi, "cell_class", CELL_CLASS_COLORS, point_size=1.2)
    ax_class.set_title("ROI cell classes")
    ax_class.legend(
        handles=class_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    cnv_handles = plot_points(ax_cnv, meta_roi, "cnv_epi", CNV_EPI_COLORS, point_size=1.2)
    ax_cnv.set_title("ROI epithelial CNV clones")
    ax_cnv.legend(
        handles=cnv_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()

    for ax in (ax_class, ax_cnv):
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymax, args.ymin)

    fig.tight_layout()
    fig.savefig(args.preview_out)
    plt.close(fig)


def write_preview_outputs(
    meta: pd.DataFrame,
    roi: pd.DataFrame,
    args: argparse.Namespace,
    metadata_rows: int | None,
) -> None:
    if args.preview_out is None:
        return
    if args.cell_metadata is None:
        raise SystemExit("--preview-out requires --cell-metadata")

    meta = standardize_cell_metadata_columns(meta)
    require_columns(meta, {"x", "y"}, "cell metadata")
    meta_roi = meta.loc[metadata_bbox_mask(meta, args)].copy()
    plot_roi_preview(meta, meta_roi, args)

    if args.preview_counts_out is not None:
        write_preview_counts(meta_roi, args.preview_counts_out)

    if args.preview_summary_out is not None:
        cell_class_counts = {}
        cnv_epi_counts = {}
        cnv_epi_raw_counts = {}
        if "cell_class" in meta_roi.columns:
            cell_class_counts = (
                meta_roi["cell_class"].fillna("missing").astype(str).value_counts().to_dict()
            )
        if "cnv_epi" in meta_roi.columns:
            cnv_epi_counts = (
                meta_roi["cnv_epi"].fillna("missing").astype(str).value_counts().to_dict()
            )
        if "cnv_epi_raw" in meta_roi.columns:
            cnv_epi_raw_counts = (
                meta_roi["cnv_epi_raw"].fillna("missing").astype(str).value_counts().to_dict()
            )
        if "cell_class" in meta_roi.columns:
            epithelial_mask = meta_roi["cell_class"].fillna("missing").astype(str) == "Epithelial"
        else:
            epithelial_mask = pd.Series(False, index=meta_roi.index)
        write_json(
            args.preview_summary_out,
            {
                "bbox": bbox_dict(args),
                "width_um": float(args.xmax - args.xmin),
                "height_um": float(args.ymax - args.ymin),
                "area_mm2": float((args.xmax - args.xmin) * (args.ymax - args.ymin) / 1_000_000),
                "n_cells": int(len(meta_roi)),
                "n_epithelial_cells": int(epithelial_mask.sum()),
                "cell_class_counts": cell_class_counts,
                "cnv_epi_counts": cnv_epi_counts,
                "cnv_epi_raw_counts": cnv_epi_raw_counts,
                "transcript_rows": int(len(roi)),
                "transcript_genes": int(roi["feature_name"].nunique()),
                "transcript_cells": int(roi["cell_id"].astype(str).nunique()),
                "metadata_rows": metadata_rows,
                "preview_png": str(args.preview_out),
                "counts_csv": (
                    str(args.preview_counts_out) if args.preview_counts_out is not None else None
                ),
            },
        )


def main() -> int:
    args = parse_args()
    if args.preview_out is None and (
        args.preview_counts_out is not None or args.preview_summary_out is not None
    ):
        raise SystemExit("--preview-counts-out and --preview-summary-out require --preview-out")
    if args.preview_out is not None and args.cell_metadata is None:
        raise SystemExit("--preview-out requires --cell-metadata")

    roi = standardize_transcript_columns(read_roi_transcripts(args))
    require_columns(roi, {"x", "y", "feature_name", "cell_id", "transcript_id"}, "transcripts")

    if args.remove_controls:
        if "is_gene" in roi.columns:
            roi = roi.loc[roi["is_gene"].astype(bool)].copy()
        else:
            roi = roi.loc[
                ~roi["feature_name"].astype(str).str.contains(
                    "NegControl|BLANK|antisense|Unassigned|Deprecated|Intergenic|Genomic",
                    case=False,
                    regex=True,
                )
            ].copy()

    ensure_parent(args.out)
    roi.to_parquet(args.out, index=False)

    meta = None
    metadata_rows = None
    if args.cell_metadata and (args.cell_metadata_out or args.preview_out):
        meta = read_table(args.cell_metadata)
        meta = standardize_cell_metadata_columns(meta)
    if meta is not None and args.cell_metadata_out:
        if "cell_id" not in meta.columns:
            raise SystemExit(f"{args.cell_metadata} has no cell_id column")
        keep = set(roi["cell_id"].astype(str))
        roi_meta = meta.loc[meta["cell_id"].astype(str).isin(keep)].copy()
        ensure_parent(args.cell_metadata_out)
        if args.cell_metadata_out.suffix.lower() == ".parquet":
            roi_meta.to_parquet(args.cell_metadata_out, index=False)
        else:
            roi_meta.to_csv(args.cell_metadata_out, index=False)
        metadata_rows = len(roi_meta)

    if meta is not None:
        write_preview_outputs(meta, roi, args, metadata_rows)

    write_json(
        args.out.with_suffix(args.out.suffix + ".summary.json"),
        {
            "input": str(args.transcripts),
            "output": str(args.out),
            "bbox": bbox_dict(args),
            "rows": len(roi),
            "genes": int(roi["feature_name"].nunique()),
            "cells": int(roi["cell_id"].astype(str).nunique()),
            "metadata_rows": metadata_rows,
            "preview": str(args.preview_out) if args.preview_out is not None else None,
        },
    )
    print(f"Wrote {len(roi):,} ROI transcripts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
