#!/usr/bin/env python3
"""Build a PMI/cPMI reference panel from a single-cell reference.

Thin CLI over :mod:`tracer.panel_builder` (draws, presence arms, reference
loading and validation, panel writing) and
:func:`tracer.conflict_reference.build_depth_corrected_reference` (the cPMI
estimator). Same split as ``build_npmi_from_scrna.py`` over
``tracer.metrics.compute_pmi_bootstrap`` — that one drives the resampling
estimator with per-pair CIs and a marginal-product null; this one drives the
deterministic depth-conditioned estimator.

The RECIPE is not in this file. It is resolved from
``tracer/configs/defaults.toml`` ``[panel]`` (with-reference) or
``tracer/configs/panels/nuclear.toml`` (no reference), inferred from the source
flags and overridable per flag. `--panel-preset`/`--panel-config` select it;
the resolved recipe is written to ``panel_receipt.json`` beside the panels.

    # with an scRNA reference -> rep + xgt1 + cPMI + n_genes
    python scripts/build_panels.py --h5ad atlas.h5ad \
        --panel-genes panel_genes.txt --celltype-col cell_type --out panels

    # nuclear / in-situ, no reference -> grid3 + naive PMI
    python scripts/build_panels.py --transcripts tx.parquet --nucleus-only \
        --out panels

REFERENCES BUILT WITH THIS (and the choices that are not recoverable from the
panel files themselves):

  GBM   --h5ad gbm_scrna_qc_with_tumor_archetype_immune_vascular_subtypes.h5ad
        --celltype-col new_consensus        11 levels, granularity comparable to
            the lung reference's 9. NOT tumor_archetype / immune_subtype /
            vascular_subtype: each carries an NA / not_* catch-all covering the
            cells outside its scope, so none is a complete partition and
            balancing on one collapses most of the atlas into one stratum.
        --panel-genes  filter the Xenium transcripts on `is_gene` first. The
            541 distinct feature_name values include 175 non-genes (53
            deprecated, 41 negative-control codewords, 20 negative-control
            probes, 61 unassigned); the real panel is 366 genes, 365 of which
            are in the reference. Unfiltered, the overlap pre-flight reads a
            spurious 67.5% instead of 99.7%.

  cervical  --h5ad cervical_scrna_adc_scc_marker_annotated.h5ad
        --celltype-col cell_type_fine       the only column that splits Tumor
            Epithelial into SCC (3,309) and ADC (1,354), the contrast the atlas
            exists to represent. `cell_type` and `cell_type_coarse` are
            byte-identical (verified), so there is no third option. Note
            `Unannotated` (1,966) is a real stratum and gets sampled up to the
            mean type size under replacement-balancing; add
            --exclude-obs cell_type_fine=Unannotated if that is unwanted.
        --panel-genes  TWO cervical spatial datasets exist and the gene universe
            is a CHOICE, not a property of the reference: atera (17,420 genes)
            or xenium5k (4,863). Pick the platform the panel will be USED on.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import asdict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracer.panel_builder import (  # noqa: E402
    ARM_SPECS, build_reference_panels, load_reference,
    load_transcripts_reference, log, read_panel_genes, _write_frame,
)


def _write_receipt(args, pcfg, *, reference: dict, ps, files: list[Path]) -> Path:
    """Record the resolved recipe and what was written, beside the panels.

    A bare `.csv.gz` carries no provenance -- which is how a panel built with
    an off-by-2x top-k rule ended up mislabelled `topk1000`. This states the
    recipe, which estimator sits in each file's `PMI` column (the only column
    TRACER reads), which file is the TRACER input, the draw, the gene support,
    the depth covariate, and the exact command. Rewritten on every run, so it
    never describes an earlier build.
    """
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec = out / f"{args.prefix}panel_receipt.json"
    tracer_inputs = set(ps.tracer_input.values())
    rec.write_text(json.dumps({
        "panel_preset": args._panel_preset,
        "effective_promote": args._effective_promote,
        "canonical_panel_substring":
            "_cpmi_" if args._effective_promote == "cPMI" else "_pmi_",
        "outputs": [{"file": f.name, "pmi_column": ps.pmi_column[f.name[:-len(".csv.gz")]],
                     "tracer_input": f.name[:-len(".csv.gz")] in tracer_inputs}
                    for f in files],
        "tracer_input": {arm: f"{name}.csv.gz" for arm, name in ps.tracer_input.items()},
        "counts_check": dict(_COUNTS_VERDICT) or None,
        "reference": reference,
        "recipe_facts": ps.facts,
        "resolved_panel_config": asdict(pcfg),
        "effective_args": {k: (str(v) if isinstance(v, Path) else v)
                           for k, v in vars(args).items()
                           if not k.startswith("_")},
        "command": " ".join(sys.argv),
    }, indent=2, default=str) + "\n")
    log(f"receipt -> {rec}")
    return rec


_COUNTS_VERDICT: dict = {}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5ad", type=Path,
                     help="Single-cell reference with a `counts` layer.")
    src.add_argument("--transcripts", type=Path,
                     help="Standardized transcript table (cell_id, "
                          "feature_name). Estimates the panel from the assay's "
                          "own cells instead of an external reference.")
    ap.add_argument("--nucleus-only", action="store_true",
                    help="With --transcripts, keep only overlaps_nucleus == 1 "
                         "— the segmentation-independent subset.")
    ap.add_argument("--panel-genes", type=Path, default=None,
                    help="Transcripts parquet (feature_name column) or a "
                         "one-gene-per-line text file. Required with --h5ad; "
                         "with --transcripts defaults to the observed genes.")
    ap.add_argument("--out", type=Path, required=True, help="Output directory.")
    ap.add_argument("--strategy",
                    choices=("rep", "capped", "vanilla", "grid3", "grid4"),
                    default=None,
                    help="Reference draw. rep/capped need --celltype-col; "
                         "grid3/grid4 are LABEL-FREE (n_genes x depth grid, "
                         "marginal edges, replacement draw); vanilla is none.")
    ap.add_argument("--celltype-col", default=None,
                    help="obs column to balance on. Required unless "
                         "--strategy vanilla.")
    ap.add_argument("--arms", nargs="+", default=None,
                    choices=sorted(ARM_SPECS), help="Presence arms to build.")
    ap.add_argument("--allow-non-integral-counts", action="store_true",
                    help="Proceed when the counts matrix is not integral. Safe "
                         "ONLY for a per-cell RESCALING (CP10k/TPM) under the "
                         "xgt1 + n_genes recipe, where CP10k normalises the "
                         "factor away; NEVER for log-transformed values.")
    ap.add_argument("--panel-preset", default=None,
                    help="Panel RECIPE preset (tracer/configs/panels/*.toml). "
                         "Inferred when omitted: --h5ad -> the with-reference "
                         "recipe (rep + xgt1 + cPMI + n_genes); "
                         "--transcripts --nucleus-only -> 'nuclear' "
                         "(grid3 + naive PMI). Explicit flags always win.")
    ap.add_argument("--panel-config", type=Path, default=None,
                    help="User-override TOML, top of the layer stack.")
    ap.add_argument("--prefix", default="",
                    help="Prepended to every output panel name.")
    ap.add_argument("--emit-cpmi", action="store_true", default=None,
                    help="Also write a `<arm>_cpmi_balanced...` panel per arm, "
                         "the same build with cPMI promoted into the `PMI` "
                         "column so the pipeline consumes the depth-corrected "
                         "estimator. Adds no information — the default panel "
                         "already carries both columns — but drops into a run "
                         "without editing. Implied for --strategy vanilla.")
    ap.add_argument("--no-emit-cpmi", dest="emit_cpmi", action="store_false",
                    help="Suppress the cPMI-promoted panel (it is written by "
                         "default under the with-reference recipe).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min-det-cells", type=int, default=None,
                    help="A gene must be detected in this many cells (default 25).")
    ap.add_argument("--n-depth-bins", type=int, default=None)
    ap.add_argument("--depth-metric", default=None,
                    choices=("total_counts", "n_genes"),
                    help="Covariate the cPMI null is binned on. Inert for the "
                         "naive-PMI column, which never reads E.")
    ap.add_argument("--depth-obs", default=None,
                    help="obs column holding the TRUE library size. Use it when "
                         "X has been gene-filtered (e.g. nCount_RNA).")
    ap.add_argument("--top-k-per-gene", type=int, default=None, metavar="K",
                    help="Builder-native relative cut: keep each gene's K "
                         "strongest partners by |cPMI| (ONE ranking per gene "
                         "over all its edges; an edge survives if in the top-K "
                         "of either endpoint). Pure rank cut — no magnitude "
                         "threshold; combine with --min-abs-value if wanted.")
    ap.add_argument("--min-abs-value", type=float, default=None, metavar="TAU",
                    help="Drop edges with |value| <= TAU (the uninformative "
                         "limbo band). Use 0.2 to match TRACER's PMI_THR, "
                         "making the panel exactly the pairs that can change a "
                         "decision. Applied to each panel's promoted metric.")
    ap.add_argument("--exclude-obs", action="append", metavar="COL=VALUE",
                    help="Drop reference cells where obs[COL] == VALUE. "
                         "Repeatable.")
    ap.add_argument("--x-is-log1p-cp10k", action="store_true",
                    help="X holds log1p(CP10k), not counts; invert it using "
                         "--depth-obs to recover integer counts.")
    args = ap.parse_args()
    _COUNTS_VERDICT.clear()     # one verdict per run, even when called twice

    # ---- resolve the panel recipe -------------------------------------
    # The recipe lives in tracer/configs (defaults.toml [panel], plus
    # configs/panels/<preset>.toml). It is inferred from the SOURCE flags,
    # which already determine it unambiguously, and any explicit CLI flag
    # overrides it. See PanelConfig for why `promote` matters most: the
    # pipeline reads whatever estimator sits in the `PMI` column.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tracer.config import load_config

    preset = args.panel_preset
    if preset is None and args.transcripts and args.nucleus_only:
        preset = "nuclear"
    if preset is None and args.transcripts and not args.nucleus_only:
        ap.error("--transcripts without --nucleus-only is ambiguous: pass "
                 "--panel-preset {reference,nuclear} explicitly")
    pcfg = load_config(path=args.panel_config,
                       panel_preset=None if preset in (None, "reference") else preset).panel

    for field, val in (("strategy", pcfg.strategy), ("arms", list(pcfg.arms)),
                       ("depth_metric", pcfg.depth_metric), ("seed", pcfg.seed),
                       ("min_det_cells", pcfg.min_det_cells),
                       ("n_depth_bins", pcfg.n_depth_bins)):
        if getattr(args, field) is None:
            setattr(args, field, val)
    if args.min_abs_value is None:
        args.min_abs_value = pcfg.min_abs_value
    if args.top_k_per_gene is None:
        args.top_k_per_gene = pcfg.top_k_per_gene
    if args.emit_cpmi is None:
        args.emit_cpmi = (pcfg.promote == "cPMI")

    # Report the RESOLVED promotion, not the config's preference: under
    # --no-emit-cpmi no cPMI panel is written, so pointing at one would lie.
    effective = "cPMI" if (pcfg.promote == "cPMI" and args.emit_cpmi) else "PMI"
    canonical = "_cpmi_" if effective == "cPMI" else "_pmi_"
    log(f"panel recipe: preset={preset or 'reference'}  strategy={args.strategy}  "
        f"arms={args.arms}  promote={effective}  depth_metric={args.depth_metric}")
    log(f"  -> the panel TRACER should consume is the one with '{canonical}' "
        f"in its name (the pipeline reads the PMI column blindly)")
    if pcfg.promote == "cPMI" and not args.emit_cpmi:
        log("  WARNING --no-emit-cpmi: only the NAIVE-PMI panel will be written. "
            "TRACER will consume the depth-CONFOUNDED estimator.")
    if pcfg.promote == "PMI":
        log("  NOTE nuclear recipe: naive PMI is INTERIM here (depth-confounded, "
            "67% of pairs positive). cPMI is better per cell but the nuclear "
            "reference is evidence-starved. Fix with more nuclei, then switch.")
    args._panel_cfg = pcfg
    args._panel_preset = preset or "reference"
    args._effective_promote = effective

    if args.strategy in ("rep", "capped") and not args.celltype_col:
        ap.error("--celltype-col is required for --strategy rep/capped")
    if args.h5ad and not args.panel_genes:
        ap.error("--panel-genes is required with --h5ad")
    if args.nucleus_only and not args.transcripts:
        ap.error("--nucleus-only only applies to --transcripts")
    if args.x_is_log1p_cp10k and not args.depth_obs:
        ap.error("--x-is-log1p-cp10k needs --depth-obs to invert the scaling")

    args.out.mkdir(parents=True, exist_ok=True)

    genes = read_panel_genes(args.panel_genes) if args.panel_genes else None
    reference: dict = {}
    if args.transcripts:
        sub, keep, depth, obs = load_transcripts_reference(
            args.transcripts, nucleus_only=args.nucleus_only, panel_genes=genes)
        reference = {"path": str(args.transcripts), "kind": "transcripts",
                     "nucleus_only": bool(args.nucleus_only),
                     "depth_source": "per-cell transcript count over the panel "
                                     "genes (an in-situ panel has no off-panel "
                                     "genes to sum)"}
    else:
        sub, keep, depth, obs = load_reference(
            args.h5ad, genes, depth_obs=args.depth_obs,
            x_is_log1p_cp10k=args.x_is_log1p_cp10k,
            exclude_obs=args.exclude_obs,
            allow_non_integral=args.allow_non_integral_counts,
            verdict_out=_COUNTS_VERDICT, info_out=reference)
        reference["kind"] = "h5ad"

    # The CLI flags are folded into the recipe so the library runs exactly
    # what the receipt records.
    run_cfg = dataclasses.replace(
        pcfg, strategy=args.strategy, arms=tuple(args.arms),
        depth_metric=args.depth_metric, seed=args.seed,
        min_det_cells=args.min_det_cells, n_depth_bins=args.n_depth_bins,
        min_abs_value=args.min_abs_value, top_k_per_gene=args.top_k_per_gene)
    ps = build_reference_panels(
        sub, keep, depth, obs, cfg=run_cfg, celltype_col=args.celltype_col,
        emit_cpmi=args.emit_cpmi, prefix=args.prefix)

    files = [_write_frame(frame, args.out, name) for name, frame in ps.panels.items()]
    _write_receipt(args, run_cfg, reference=reference, ps=ps, files=files)
    for arm, name in ps.tracer_input.items():
        log(f"TRACER input ({arm}): {name}.csv.gz  [PMI column = "
            f"{ps.pmi_column[name]}]")


if __name__ == "__main__":
    main()
