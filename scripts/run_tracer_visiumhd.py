"""Visium HD TRACER runner — the run_tracer.py equivalent for 2um-bin data.

Builds a transcript-level table from the Visium HD 2um bins and runs either the
SEGMENTED or the UNSEGMENTED (NOSEG) TRACER pipeline with the current default
config, then scores + writes outputs.

  SEG   : bins -> cell_id via CELL polygons (cytoplasm bins join their cell, so
          the cell prior acts) + overlaps_nucleus via NUCLEUS polygons ->
          run_segmented_pipeline (nuclear-seed prune).
  NOSEG : bins with cell_id=-1 -> run_noseg_pipeline (density cascade; isolated
          singleton/noise bins fall through to unassigned = "real-bin" gate).

Genes are restricted to the panel up-front (the explode's real-signal filter).

Usage:
  genesis_env/python scripts/run_tracer_visiumhd.py --mode seg \
    --data-dir tutorials/pdac_visiumhd/HC01/data \
    --panel tutorials/pdac_io/output/depthcorr_GSE_edges.csv --value-col cPMI \
    --outdir tutorials/pdac_visiumhd/HC01/output/seg --sample-name HC01_seg \
    [--roi-size-um 2000] [--roi-center X Y]
"""
from __future__ import annotations
import argparse, json, resource, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

SENT = {"-1", "DROP", "UNASSIGNED", "nan", "", "__GUARD_SKIP__", "group_rejected", "demote_rejected"}


def _peak_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r if sys.platform == "darwin" else r * 1024) / (1024 ** 3)


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["seg", "noseg"], required=True)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="HC01/data dir (has binned_outputs/ + segmented_outputs/).")
    p.add_argument("--bin", default="square_002um")
    p.add_argument("--panel", required=True, type=Path, help="long-format panel csv (gene_i,gene_j,<value>).")
    p.add_argument("--value-col", default=None, help="panel value column; auto: cPMI>PMI>NPMI.")
    p.add_argument("--roi-size-um", type=float, default=None, help="square ROI side (um); default whole tissue.")
    p.add_argument("--roi-center", type=float, nargs=2, default=None)
    p.add_argument("--max-transcripts", type=int, default=40_000_000)
    p.add_argument("--multi-rule", default="nearest_centroid")
    p.add_argument("--panel-genes-only", action="store_true",
                   help="Explode ONLY panel genes (legacy). Default: explode ALL data genes so "
                        "off-panel genes (in data, not in the PMI panel) are kept and assigned via "
                        "the off-panel proximity rescue instead of being dropped.")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--sample-name", required=True)
    p.add_argument("--prune-scope", choices=["auto", "cell", "nuclear"], default="auto",
                   help="Override cfg.phase1.prune_scope. 'cell' seeds Phase-1a on the "
                        "WHOLE CELL and admits whole-cell tx; 'nuclear' is the legacy "
                        "nuclear-throughout path; 'auto' keeps the loaded config "
                        "(default: cell).")
    return p.parse_args()


def _cy_rw_shim():
    """Wrap _cy_prune callables so read-only numpy buffers don't crash the direct
    run_*_pipeline path (same shim our PDAC/ovarian A/B drivers use)."""
    import tracer._cy_prune as _cy

    def _rw(x):
        if isinstance(x, np.ndarray):
            return np.array(x) if not x.flags.writeable else x
        if isinstance(x, (list, tuple)):
            return type(x)(_rw(e) for e in x)
        return x

    def _wrap(o):
        def w(*a, **k):
            return o(*(_rw(x) for x in a), **{kk: _rw(v) for kk, v in k.items()})
        return w
    for nm in dir(_cy):
        if not nm.startswith("_") and callable(getattr(_cy, nm)) and not nm.startswith(("set_", "get_", "clear_")):
            try:
                setattr(_cy, nm, _wrap(getattr(_cy, nm)))
            except Exception:
                pass


def main() -> int:
    a = _args()
    t0 = time.time()
    _cy_rw_shim()
    from tracer.noseg_pipeline import (
        load_visiumhd_bins, subset_roi, explode_to_transcripts, _read_tissue_positions,
    )
    from tracer.pipeline import run_segmented_pipeline, run_noseg_pipeline
    from tracer.config import load_config

    data = a.data_dir
    matrix_dir = data / "binned_outputs" / a.bin / "filtered_feature_bc_matrix"
    spatial_dir = data / "binned_outputs" / a.bin / "spatial"

    # --- panel + gene set ---
    panel_raw = (pd.read_parquet(a.panel) if str(a.panel).endswith(".parquet")
                 else pd.read_csv(a.panel))
    vcol = a.value_col or next((c for c in ("cPMI", "PMI", "NPMI") if c in panel_raw.columns), None)
    if vcol is None:
        raise SystemExit(f"panel {a.panel} has no cPMI/PMI/NPMI column: {list(panel_raw.columns)}")
    panel = panel_raw[["gene_i", "gene_j", vcol]].rename(columns={vcol: "PMI"}).dropna()
    panel["gene_i"] = panel.gene_i.astype(str); panel["gene_j"] = panel.gene_j.astype(str)
    panel_genes = set(panel.gene_i) | set(panel.gene_j)
    print(f"[panel] {a.panel.name}  value={vcol}  pairs={len(panel):,}  genes={len(panel_genes)}", flush=True)

    # --- bins (+ optional ROI) ---
    bins = load_visiumhd_bins(matrix_dir, spatial_dir, expected_bin_size_um=2.0)
    print(f"[bins] loaded {bins.adata.n_obs:,} bins x {bins.adata.n_vars:,} genes  [{time.time()-t0:.0f}s]", flush=True)
    if a.roi_size_um is not None:
        center = tuple(a.roi_center) if a.roi_center else None
        bins = subset_roi(bins, size_um=a.roi_size_um, center=center)
        print(f"[roi] {a.roi_size_um}um -> {bins.adata.n_obs:,} bins", flush=True)
    barcodes = np.asarray(bins.coords.index)

    # --- segmentation overlay (SEG only) ---
    bin_cell_id = bin_overlaps = None
    if a.mode == "seg":
        import json as _json
        from tracer.visiumhd_seg import load_nucleus_polygons, assign_bins_to_nuclei
        cell_gj = data / "segmented_outputs" / "cell_segmentations.geojson"
        nuc_gj = data / "segmented_outputs" / "nucleus_segmentations.geojson"
        tp = _read_tissue_positions(spatial_dir).set_index("barcode").loc[barcodes]
        px = tp["pxl_col_in_fullres"].to_numpy(np.float64)
        py = tp["pxl_row_in_fullres"].to_numpy(np.float64)
        # --- registration: bins are in the CytAssist 'fullres' frame; the geojson
        # polygons are in the high-res microscopy frame. Convert bins -> geojson
        # frame via a scale (= um-per-px ratio, exact) + offset (whole-tissue
        # centroid match, so ROI subsets reuse the same global transform). ---
        bsf = _json.load(open(spatial_dir / "scalefactors_json.json"))
        ssf = _json.load(open(data / "segmented_outputs" / "spatial" / "scalefactors_json.json"))
        scale = float(bsf["microns_per_pixel"]) / float(ssf["microns_per_pixel"])
        if abs(scale - 1.0) < 0.02:
            # Version-consistent data (e.g. binned + segmented both from the same
            # spaceranger run): the binned tissue_positions and the segmentation
            # geojson already share the microscopy pixel frame -> overlay directly,
            # NO transform. (Applying a centroid offset here would MIS-register.)
            print(f"[registration] same-frame (scale={scale:.4f}) — direct overlay, no transform", flush=True)
        else:
            # Resolution / spaceranger-version MISMATCH (e.g. SR3.0.1 CytAssist bins at
            # 5.75um/px vs SR4.0.1 microscopy polygons at 0.46um/px): map bins -> geojson
            # frame by scale (um-per-px ratio) + whole-tissue nucleus-centroid offset.
            tp_all = _read_tissue_positions(spatial_dir)
            it = tp_all[tp_all["in_tissue"] == 1]
            _gj = _json.load(open(nuc_gj))
            _gc = np.array([[np.asarray(f["geometry"]["coordinates"][0])[:, 0].mean(),
                             np.asarray(f["geometry"]["coordinates"][0])[:, 1].mean()]
                            for f in _gj["features"]])
            ox = _gc[:, 0].mean() - it["pxl_col_in_fullres"].mean() * scale
            oy = _gc[:, 1].mean() - it["pxl_row_in_fullres"].mean() * scale
            px = px * scale + ox
            py = py * scale + oy
            print(f"[registration] bins->geojson  scale={scale:.4f}  offset=({ox:.0f},{oy:.0f}) [version-mismatch path]", flush=True)
        m = 50.0
        bbox = (px.min() - m, py.min() - m, px.max() + m, py.max() + m)
        # cell_id from CELL bodies (so cytoplasm bins join their cell)
        cells = load_nucleus_polygons(cell_gj, bbox=bbox, id_field="cell_id")
        cell_id_arr, _, cs = assign_bins_to_nuclei(px, py, cells, multi_rule=a.multi_rule)
        # overlaps_nucleus from NUCLEUS polygons (the seed)
        nuclei = load_nucleus_polygons(nuc_gj, bbox=bbox, id_field="cell_id")
        _, overlaps_arr, ns = assign_bins_to_nuclei(px, py, nuclei, multi_rule=a.multi_rule)
        bin_cell_id = pd.Series(cell_id_arr, index=barcodes)
        bin_overlaps = pd.Series(overlaps_arr, index=barcodes)
        print(f"[overlay] cell-assigned bins={cs['n_assigned']:,} ({cs['frac_assigned']:.1%})  "
              f"nucleus-overlap bins={int(overlaps_arr.sum()):,} ({ns['frac_assigned']:.1%})", flush=True)

    # --- explode to transcripts ---
    # Explode ALL data genes by default: the panel genes drive the PMI seed/admission/coherence,
    # while off-panel genes (present in the data but absent from the PMI panel) are exploded too
    # and picked up by the off-panel proximity rescue (offpanel_first_entity). Restricting the
    # explode to panel genes (--panel-genes-only) drops them and idles that rescue path.
    all_genes = set(map(str, bins.adata.var_names))
    explode_genes = panel_genes if a.panel_genes_only else all_genes
    print(f"[explode-genes] data={len(all_genes):,}  in-panel={len(all_genes & panel_genes):,}  "
          f"off-panel={len(all_genes - panel_genes):,}  "
          f"exploding={'panel-only' if a.panel_genes_only else 'ALL (off-panel via proximity rescue)'}",
          flush=True)
    df = explode_to_transcripts(
        bins, panel_genes=explode_genes, max_transcripts=a.max_transcripts,
        bin_cell_id=bin_cell_id, bin_overlaps_nucleus=bin_overlaps,
    )
    df["z"] = np.float32(0.0)
    n_tx = len(df)
    n_prior = int((~df["cell_id"].astype(str).isin(SENT)).sum())
    print(f"[explode] transcripts={n_tx:,}  prior-assigned={n_prior:,} "
          f"({100*n_prior/max(n_tx,1):.1f}%)  overlaps_nucleus? {'overlaps_nucleus' in df.columns}  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # --- run pipeline ---
    cfg = load_config()
    if a.prune_scope != "auto":
        object.__setattr__(cfg.phase1, "prune_scope", a.prune_scope)
    seed_nuclear, admit_nuclear = cfg.phase1.resolve_scope()
    print(f"[cfg] prune_scope={cfg.phase1.prune_scope} (seed source: "
          f"{'NUCLEAR' if seed_nuclear else 'WHOLE CELL'}; admit: "
          f"{'nuclear-only' if admit_nuclear else 'whole-cell'})  "
          f"admit_independent={cfg.phase1.admit_independent} "
          f"rst={cfg.phase1.real_signal_threshold}", flush=True)
    tp0 = time.time()
    if a.mode == "seg":
        out, prog = run_segmented_pipeline(df.copy(), panel, cfg=cfg)
    else:
        out, prog = run_noseg_pipeline(df.copy(), panel, cfg=cfg)
    wall = time.time() - tp0
    print(f"[pipeline] {a.mode} wall {wall:.0f}s ({wall/60:.1f} min)  peak {_peak_gb():.1f} GB", flush=True)

    # --- score / census ---
    col = "tracer_id" if "tracer_id" in out.columns else ("stitched" if "stitched" in out.columns else None)
    o = out.copy(); o[col] = o[col].astype(str)
    if "_etype" in o.columns:
        et = o["_etype"].astype(str)
    else:
        from tracer._etype import infer_etype_from_label
        et = pd.Series(np.asarray(infer_etype_from_label(o[col])).astype(str), index=o.index)
    is_un = o[col].isin(SENT) | o[col].str.endswith("_rejected", na=False)
    per = pd.DataFrame({"l": o.loc[~is_un, col], "e": et.loc[~is_un]}).drop_duplicates("l")
    ec = per["e"].value_counts().to_dict()
    n_cells = int(ec.get("cell", 0)); n_part = int(ec.get("partial", 0)); n_comp = int(ec.get("component", 0))
    cov = round(100 * int((~is_un).sum()) / max(len(o), 1), 2)

    a.outdir.mkdir(parents=True, exist_ok=True)
    keep = ["transcript_id", "cell_id", "x", "y", "z", "feature_name", col] + (["_etype"] if "_etype" in o.columns else [])
    o[keep].to_parquet(a.outdir / "partition.parquet", index=False)
    summary = {
        "sample": a.sample_name, "mode": a.mode, "bin": a.bin,
        "panel": str(a.panel), "panel_value_col": vcol, "panel_genes": len(panel_genes),
        "roi_size_um": a.roi_size_um, "n_bins": int(bins.adata.n_obs),
        "n_input_tx": n_tx, "n_prior_assigned_tx": n_prior,
        "n_cells": n_cells, "n_partials": n_part, "n_components": n_comp,
        "partial_to_cell_ratio": round(n_part / max(n_cells, 1), 3),
        "coverage_pct": cov, "n_unassigned_tx": int(is_un.sum()),
        "wall_seconds": round(wall, 1), "peak_rss_gb": round(_peak_gb(), 2),
        "label_column": col,
        "cfg": {"prune_scope": cfg.phase1.prune_scope,
                "prune_scope_cli": a.prune_scope,
                "seed_nuclear": seed_nuclear, "admit_nuclear": admit_nuclear,
                "admit_independent": cfg.phase1.admit_independent,
                "real_signal_threshold": cfg.phase1.real_signal_threshold},
    }
    (a.outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # --- stagewise census (per-stage entity/tx trajectory) ---
    census = [{k: s.get(k) for k in ("stage", "stage_seconds", "n_cells",
                                     "n_partials", "n_components", "n_unassigned_tx")}
              for s in prog]
    (a.outdir / "stage_census.json").write_text(json.dumps(census, indent=2, default=str))
    print("\n=== STAGE CENSUS ===")
    print(f"  {'stage':<22s} {'secs':>8s} {'cells':>8s} {'partials':>9s} {'unassigned_tx':>13s}")
    for s in census:
        sec = f"{s['stage_seconds']:.1f}" if s.get("stage_seconds") is not None else "t0"
        print(f"  {str(s['stage']):<22s} {sec:>8s} {s.get('n_cells',0):>8,} "
              f"{s.get('n_partials',0):>9,} {s.get('n_unassigned_tx',0):>13,}")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if k != "cfg":
            print(f"  {k}: {v}")
    print(f"  -> {a.outdir}/partition.parquet + summary.json  [total {time.time()-t0:.0f}s]")
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
