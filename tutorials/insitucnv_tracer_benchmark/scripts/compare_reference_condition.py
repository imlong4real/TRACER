#!/usr/bin/env python3
"""Compare raw-segmentation vs TRACER-refined CNV for one reference condition.

"Before" = raw Xenium segmentation arm; "after" = TRACER whole-cell arm. Both
arms are produced by ``run_insitucnv_arm.py`` and each writes
``chrom_cnv_by_compartment.csv`` (mean CNV per compartment x chromosome) and
``arm_stats.json``.

This script quantifies whether TRACER refinement retains/sharpens CNV signal:

  - ``tumor_signal``      : mean |CNV| over chromosomes in the tumor compartment
                            (higher = stronger copy-number signal).
  - ``reference_flatness``: mean |CNV| in the reference compartment
                            (lower = flatter, cleaner diploid baseline).
  - ``signal_to_baseline``: tumor_signal / (reference_flatness + eps).

Outputs:
  - ``compare_metrics.csv`` / ``compare_summary.json`` : per-arm metrics + raw->tracer deltas.
  - ``compare_chrom_cnv.png``  : per-chromosome CNV, raw vs tracer, tumor & reference panels.
  - ``compare_umap_dotplot.png`` (optional) : montage of the two arms' UMAP + dotplot
                            (from plot_cnv_arm.py), i.e. the before/after clone view.

EXAMPLE
=======
::

    python compare_reference_condition.py \\
      --raw-dir output/runs/pilot_full_100/insitucnv_raw \\
      --tracer-dir output/runs/pilot_full_100/insitucnv_tracer_whole \\
      --outdir output/runs/pilot_full_100/compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True, type=Path, help="insitucnv_raw output dir.")
    p.add_argument("--tracer-dir", required=True, type=Path, help="insitucnv_tracer_whole output dir.")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--raw-plots-dir", type=Path, default=None,
                   help="Dir with raw_umap.png / raw_dotplot_clone.png (default: <raw-dir>/plots).")
    p.add_argument("--tracer-plots-dir", type=Path, default=None,
                   help="Dir with tracer_whole_*.png (default: <tracer-dir>/plots).")
    p.add_argument("--tumor-compartment", default="tumor")
    p.add_argument("--reference-compartment", default="reference")
    return p.parse_args()


def _crank(c: str):
    c = str(c).replace("chr", "")
    return (0, int(c)) if c.isdigit() else (1, {"X": 23, "Y": 24, "MT": 25}.get(c, 99))


def load_chrom(arm_dir: Path) -> pd.DataFrame:
    """Load chrom_cnv_by_compartment.csv -> DataFrame indexed by compartment, chromosome columns."""
    path = arm_dir / "chrom_cnv_by_compartment.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Did run_insitucnv_arm.py finish for this arm?")
    df = pd.read_csv(path)
    if "compartment" not in df.columns:
        raise SystemExit(f"{path} has no 'compartment' column; columns={list(df.columns)}")
    df = df.set_index("compartment")
    chrom_cols = sorted(df.columns, key=_crank)
    return df[chrom_cols]


def arm_metrics(chrom: pd.DataFrame, tumor: str, ref: str) -> dict:
    out = {}
    tvec = chrom.loc[tumor].to_numpy(dtype=float) if tumor in chrom.index else None
    rvec = chrom.loc[ref].to_numpy(dtype=float) if ref in chrom.index else None
    out["has_tumor"] = tvec is not None
    out["has_reference"] = rvec is not None
    out["tumor_signal"] = float(np.mean(np.abs(tvec))) if tvec is not None else float("nan")
    out["reference_flatness"] = float(np.mean(np.abs(rvec))) if rvec is not None else float("nan")
    out["signal_to_baseline"] = (out["tumor_signal"] / (out["reference_flatness"] + EPS)
                                 if (tvec is not None and rvec is not None) else float("nan"))
    return out


def read_arm_stats(arm_dir: Path) -> dict:
    p = arm_dir / "arm_stats.json"
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return {}


def plot_chrom_compare(raw: pd.DataFrame, tracer: pd.DataFrame, tumor: str, ref: str, out: Path):
    chroms = [c for c in raw.columns if c in tracer.columns]
    x = np.arange(len(chroms))
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.4 * len(chroms)), 7), sharex=True)
    for ax, comp, title in ((axes[0], tumor, "Tumor compartment"),
                            (axes[1], ref, "Reference compartment")):
        if comp in raw.index:
            ax.plot(x, raw.loc[comp, chroms].to_numpy(float), "-o", ms=3, label="raw", color="#d62728")
        if comp in tracer.index:
            ax.plot(x, tracer.loc[comp, chroms].to_numpy(float), "-s", ms=3, label="tracer_whole", color="#1f77b4")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(title); ax.set_ylabel("mean CNV"); ax.legend(fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(chroms, rotation=90, fontsize=7)
    fig.suptitle("Per-chromosome CNV: raw vs TRACER")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def montage(raw_plots: Path, tracer_plots: Path, out: Path):
    """Assemble the two arms' UMAP + dotplot into a 2x2 before/after montage."""
    tiles = [
        (raw_plots / "raw_umap.png", "raw: UMAP"),
        (tracer_plots / "tracer_whole_umap.png", "tracer_whole: UMAP"),
        (raw_plots / "raw_dotplot_clone.png", "raw: markers by clone"),
        (tracer_plots / "tracer_whole_dotplot_clone.png", "tracer_whole: markers by clone"),
    ]
    present = [(p, t) for p, t in tiles if p.exists()]
    if not present:
        print("No per-arm plot PNGs found; skipping montage.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (p, t) in zip(axes.ravel(), tiles):
        ax.axis("off")
        if p.exists():
            ax.imshow(plt.imread(p)); ax.set_title(t, fontsize=10)
    fig.suptitle("Before (raw) vs After (TRACER): clones", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw_chrom = load_chrom(args.raw_dir)
    tracer_chrom = load_chrom(args.tracer_dir)

    raw_m = arm_metrics(raw_chrom, args.tumor_compartment, args.reference_compartment)
    tracer_m = arm_metrics(tracer_chrom, args.tumor_compartment, args.reference_compartment)

    metrics = pd.DataFrame({"raw": raw_m, "tracer_whole": tracer_m}).T
    metrics.index.name = "arm"
    metrics.to_csv(args.outdir / "compare_metrics.csv")

    delta = {
        "d_tumor_signal": tracer_m["tumor_signal"] - raw_m["tumor_signal"],
        "d_reference_flatness": tracer_m["reference_flatness"] - raw_m["reference_flatness"],
        "d_signal_to_baseline": tracer_m["signal_to_baseline"] - raw_m["signal_to_baseline"],
    }
    summary = {
        "raw": {**raw_m, "stats": read_arm_stats(args.raw_dir)},
        "tracer_whole": {**tracer_m, "stats": read_arm_stats(args.tracer_dir)},
        "delta_tracer_minus_raw": delta,
        "interpretation": (
            "d_tumor_signal>0 and d_reference_flatness<=0 => TRACER sharpens tumor CNV "
            "without adding reference-baseline noise."
        ),
    }
    with open(args.outdir / "compare_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
        fh.write("\n")

    plot_chrom_compare(raw_chrom, tracer_chrom, args.tumor_compartment,
                       args.reference_compartment, args.outdir / "compare_chrom_cnv.png")

    raw_plots = args.raw_plots_dir or (args.raw_dir / "plots")
    tracer_plots = args.tracer_plots_dir or (args.tracer_dir / "plots")
    montage(raw_plots, tracer_plots, args.outdir / "compare_umap_dotplot.png")

    print("Compare metrics (raw vs tracer_whole):")
    print(metrics.to_string())
    print("Delta (tracer - raw):", json.dumps({k: round(float(v), 4) for k, v in delta.items()}))
    print(f"Wrote comparison to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
