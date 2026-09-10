# TRACER RCTD label transfer for Cirro

Runs RCTD (`spacexr`) against a single-cell reference over four entity arms
derived from one TRACER Seg run, and reports per-entity cell-type weights with
max weight and Shannon entropy as confidence metrics.

## Arms

All four are rebuilt from the *same* `transcripts_tracer_refined.parquet`, so
QV filtering and control-probe removal are shared by construction and the
pre/post contrast is paired on `original_cell_id`.

| arm | entities | grouping key |
|---|---|---|
| `original` | pre-TRACER Xenium cells | pristine input `cell_id` |
| `post_whole` | post-TRACER whole cells | `tracer_id` where `_etype == 'cell'` |
| `post_partial` | post-TRACER reconstructed partial cells | `tracer_id` where `_etype == 'partial'` |
| `post_all` | whole + partial (visualisation) | `tracer_id` where `_etype in {cell, partial}` |

`original_cell_id` is the entity label with the `-tr-` partial suffix stripped,
so a partial always carries the input cell it was reconstructed from.

## No per-arm tuning

`RCTD_RUN` is a single process. Every arm receives the same value channels for
`celltype_col`, `doublet_mode`, `umi_min`, `umi_min_sigma`, `gene_cutoff`,
`fc_cutoff`, `min_cells_per_celltype_reference` and `seed`; the workflow
exposes no arm-conditional parameter anywhere. The effective block is written
verbatim to `rctd_results/rctd_settings.json`.

The gene axis is read from the reference h5ad's `var_names`, so the spatial
matrices and the reference cannot drift apart, and every arm is scored on an
identical panel-matched feature space.

`doublet_mode` defaults to `full`. Doublet mode constrains each entity to at
most two cell types, which caps entropy and pushes low-UMI partial cells into
`reject`; that would confound exactly the pre/post comparison this workflow
exists to measure.

## Reference contract

The reference h5ad must carry **integer counts** in `layers['counts']`
(`run_rctd.R` prefers that layer, then `raw/X`, then `X`) and the cell-type
column named by `--celltype_col`.

## Outputs

`rctd_results/` is published into the destination dataset:

- `rctd_entities.parquet` / `rctd_entities.csv.gz` — one row per entity:
  `sample`, `patient`, `section`, `arm`, `entity_type`, `entity_id`,
  `original_cell_id`, `tracer_id`, `gbmap_celltype`, `max_weight`, `entropy`,
  `n_tx`, `x_centroid`, `y_centroid`
- `rctd_weights_<arm>.tsv.gz` — full normalised weight matrix per arm
- `rctd_arm_summary.tsv` — per-arm entity counts and median entropy / max weight
- `rctd_settings.json` — the shared settings block plus the prep manifest
- `pipeline_info/` — Nextflow trace, report, timeline, DAG

## Pins

- Analysis image: `ghcr.io/imlong4real/segbench-analysis@sha256:98985cd498937a651cb4c278d562452606c24a9bea7095d1eecd5c7fea6a42a3`
- `spacexr`: `dmcable/spacexr@9f5dc33c8060f946c6072a138b70e189636e1435`
- Upstream TRACER Seg source commit: `ee259003be572581c434dd5bed40d7568f05f906`
