#!/usr/bin/env python3
"""Orchestrate generation of all Figure 4 panels (B–G) + manifest + summary.

Panel A (graphical-abstract concept cartoon) is omitted from this pipeline and
added separately. Each code panel (B–G) is run independently and failures are
reported without aborting the others. Writes outputs/fig4_manifest.json and
outputs/fig4_run_summary.md.

Usage:
    python scripts/reproducibility/fig4/make_fig4.py [--panels B C D E F G]
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib
import json
import traceback
from pathlib import Path

import fig4_config as C

PANELS = {
    "B": ("panel_b_whole_tissue_maps", "panel_B_whole_tissue_maps"),
    "C": ("panel_c_roi_validation", "panel_C_roi_validation"),
    "D": ("panel_d_marker_validation", "panel_D_marker_validation"),
    "E": ("panel_e_quantitative_benchmark", "panel_E_quantitative_benchmark"),
    "F": ("panel_f_resolution_tradeoff", "panel_F_resolution_tradeoff"),
    "G": ("panel_g_pixel_to_profile", "panel_G_pixel_to_profile"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", default=list(PANELS), choices=list(PANELS))
    args = ap.parse_args()

    results = {}
    for p in args.panels:
        mod_name, stem = PANELS[p]
        print(f"\n===== Panel {p}: {mod_name} =====", flush=True)
        try:
            mod = importlib.import_module(mod_name)
            importlib.reload(mod)
            mod.make()
            png, svg = C.OUTDIR / f"{stem}.png", C.OUTDIR / f"{stem}.svg"
            results[p] = {"status": "ok" if png.exists() else "no-output",
                          "png": str(png), "svg": str(svg)}
        except Exception as e:
            traceback.print_exc()
            results[p] = {"status": "error", "error": str(e)}

    _write_manifest(results)
    _write_summary(results)
    print("\n[make_fig4] done.")
    for p, r in results.items():
        print(f"  Panel {p}: {r['status']}")


def _collect_outputs():
    pngs = sorted(C.OUTDIR.glob("panel_*.png"))
    svgs = sorted(C.OUTDIR.glob("panel_*.svg"))
    src = sorted(C.SRCDIR.glob("*.csv")) + sorted(C.SRCDIR.glob("*.parquet")) \
        + sorted(C.SRCDIR.glob("*.csv.gz"))
    return pngs, svgs, src


def _write_manifest(results):
    pngs, svgs, src = _collect_outputs()
    manifest = {
        "figure": "Figure 4 — TRACER on VisiumHD kidney (noseg)",
        "generated_utc": dt.datetime.utcnow().isoformat() + "Z",
        "message": "TRACER reconstructs biologically coherent cellular profiles "
                   "from sequencing-based VisiumHD, incl. near-pixel 2µm bins, "
                   "without prior segmentation.",
        "methods": {k: C.METHOD_DISPLAY[k] for k in C.METHOD_ORDER},
        "lineage_palette": C.PALETTE,
        "panels": results,
        "inputs": {
            "he_btf": str(C.HE_BTF),
            "he_hires_png": str(C.HE_HIRES_PNG),
            "cell_segmentations_geojson": str(C.CELL_SEG_GEOJSON),
            "reference_h5ad": str(C.REFERENCE_H5AD),
            "wt_matrices": {k: str(v) for k, v in C.WT_H5AD.items()},
            "labels": {k: str(v) for k, v in C.LABELS.items()},
            "rctd_assignments": {k: str(v) for k, v in C.RCTD_ASSIGN.items()},
            "benchmark_metrics": {k: str(v) for k, v in C.BENCH_METRICS.items()},
        },
        "outputs": {
            "png": [str(p) for p in pngs],
            "svg": [str(p) for p in svgs],
            "source_data": [str(p) for p in src],
        },
    }
    (C.OUTDIR / "fig4_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[make_fig4] wrote {C.OUTDIR/'fig4_manifest.json'}")


def _write_summary(results):
    pngs, svgs, src = _collect_outputs()
    rctd_ready = [k for k in C.METHOD_ORDER if C.RCTD_ASSIGN[k].exists()]
    lines = [
        "# Figure 4 — run summary",
        "",
        f"Generated: {dt.datetime.utcnow().isoformat()}Z",
        "",
        "## Message",
        "TRACER generalizes beyond imaging-based ST and reconstructs biologically",
        "coherent cellular profiles from sequencing-based VisiumHD, including from",
        "very small (2 µm, near-pixel) bins, without prior segmentation.",
        "",
        "## Methods included",
        *[f"- {C.METHOD_DISPLAY[k]}" for k in C.METHOD_ORDER],
        "",
        "## Key files used",
        f"- H&E (full-res BigTIFF): `{C.HE_BTF}`",
        f"- 10x cell segmentations: `{C.CELL_SEG_GEOJSON}`",
        f"- scRNA reference (Schwann-excluded, 9 lineages): `{C.REFERENCE_H5AD}`",
        "- Whole-transcriptome matrices (downstream biology):",
        *[f"    - {C.METHOD_DISPLAY[k]}: `{C.WT_H5AD[k]}`" for k in C.METHOD_ORDER],
        "- Lineage labels (whole-transcriptome label transfer):",
        *[f"    - {C.METHOD_DISPLAY[k]}: `{C.LABELS[k]}`" for k in C.METHOD_ORDER],
        "",
        "## Panel status",
        *[f"- Panel {p}: **{r['status']}**" for p, r in results.items()],
        "- Panel A: **omitted** (graphical-abstract concept cartoon; to be added "
        "separately).",
        "",
        "## Caveats / fallback behavior",
        f"- RCTD ran on the matched **1,656 HVG/NPMI** gene panel for all methods "
        f"(per design). RCTD complete for: {', '.join(C.METHOD_DISPLAY[k] for k in rctd_ready) or 'none yet'}. "
        "If a method is still running, Panel E renders its Pearson heatmap and "
        "fills RCTD violins as runs complete (re-run panel E).",
        "- TRACER reconstruction/NPMI scoring used the 1,656 HVG/NPMI panel; all "
        "downstream biology (label transfer, markers, Pearson, gene/UMI counts) "
        "uses whole-transcriptome matrices re-aggregated from the original bins.",
        "- TRACER whole-transcriptome re-aggregation covers profiles that own ≥1 "
        "bin in bin_to_profile_assignment (2 µm: 260,896/264,964 = 98.5%; 8 µm: "
        "82,044/133,815 = 61%). Minority fragment profiles without an owned bin "
        "are excluded from whole-transcriptome metrics.",
        "- Profile counts / unassigned-bin fractions reflect input bin granularity, "
        "not reconstruction quality (Panel F framing).",
        "",
        f"## Outputs ({len(pngs)} PNG / {len(svgs)} SVG, {len(src)} source tables)",
        f"- Panels: `{C.OUTDIR}`",
        f"- Source data: `{C.SRCDIR}`",
        f"- Manifest: `{C.OUTDIR/'fig4_manifest.json'}`",
    ]
    (C.OUTDIR / "fig4_run_summary.md").write_text("\n".join(lines))
    print(f"[make_fig4] wrote {C.OUTDIR/'fig4_run_summary.md'}")


if __name__ == "__main__":
    main()
