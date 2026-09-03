# TRACER Seg for Cirro

Minimal Cirro/Nextflow adapter for TRACER Seg on Xenium and Xenium 5K
(`atera`) transcript data. The TRACER core source is not copied or modified by
this adapter.

## Reproducibility pins

- TRACER version: `0.1.1`
- TRACER source commit: `ee259003be572581c434dd5bed40d7568f05f906`
- OCI image: `ghcr.io/imlong4real/tracer@sha256:1ab4e0b2704fad56237fb4a9099c4ecbd1d83c71a52b3738386e5fafca90d282`
- The image was produced by the successful GitHub Actions run for the source
  commit above. Cirro users cannot change the pin through the form.

## Inputs

`--transcripts` and `--pmi` are required. Cirro renders the former as a file
inside the selected dataset and the latter as an explicit Cirro reference
selection.

The transcript Parquet may already contain `cell_id` and
`overlaps_nucleus`, as normal Xenium output does. Optional vector
`--cell_boundaries` and `--nucleus_boundaries` inputs support Parquet/CSV
vertex tables (`cell_id`, `vertex_x`, `vertex_y`) and GeoJSON geometry tables.
When provided, cell membership and nucleus overlap are assigned from the
polygons before standard TRACER preprocessing. Raster masks are intentionally
not accepted because a reliable pixel-to-micron transform is not available in
a standalone mask file.

The PMI/cPMI reference must be long-format CSV or CSV.GZ with `gene_i`,
`gene_j`, and `PMI` or `NPMI`. The effective selected source, staged path,
SHA-256, and size are recorded in both `config_receipt.json` and
`provenance/run_manifest.json`.

The complete parameter contract is in `nextflow_schema.json`. The Cirro form
keeps the output directory platform-managed; locally it is set with
`--outdir`.

## Outputs

Cirro publishes `tracer_results/` into the destination dataset:

- `outputs/transcripts_tracer_refined.parquet`
- `outputs/cell_by_gene_tracer.h5ad`
- `outputs/cell_scores.tsv.gz`
- `preprocessing/qc/` summaries and optional mask-assignment summary
- `config_receipt.json`, `run_summary.md`, and `runtime_memory.json`
- `logs/` for preprocessing and TRACER stdout/stderr
- `provenance/resolved_tracer_config.json`
- `provenance/run_manifest.json`, canonical output fingerprints, software
  versions, and SHA-256 checksums
- `pipeline_info/` Nextflow trace, report, timeline, and DAG

## Retry behavior

Exit statuses associated with transient transfer/termination or out-of-memory
conditions (`104`, `137`, `143`) are retried up to `--max_retries`. Memory is
multiplied by the attempt number and capped at `--max_memory_gb`. Other
failures terminate immediately so invalid inputs are not repeatedly billed.

The pinned image does not include the `procps` package. A minimal read-only
`bin/ps` compatibility shim exposes PID/parent-PID data from `/proc`, which is
the only operation Nextflow's task monitor needs. Nothing is installed into or
changed inside the image at runtime. The image also omits the `fastparquet`
package imported by TRACER's Xenium preprocessor; `bin/fastparquet.py` provides
only the three row-group APIs that script uses, backed by the image's pinned
PyArrow runtime.

## Local smoke test

The test fixture is the same deterministic eight-cell synthetic Xenium-like
sample used by the upstream TRACER segmented smoke tests.

```bash
module load gcc/9.3.0 openjdk/17.0.8.1_1 singularity/3.8.7 Nextflow/22.10.0-RC1
mkdir -p workflows/cirro/tests/work
singularity pull workflows/cirro/tests/work/tracer-ee259003.sif \
  docker://ghcr.io/imlong4real/tracer@sha256:1ab4e0b2704fad56237fb4a9099c4ecbd1d83c71a52b3738386e5fafca90d282
singularity exec workflows/cirro/tests/work/tracer-ee259003.sif \
  python workflows/cirro/tests/make_smoke_fixture.py \
  --outdir workflows/cirro/tests/work/input
nextflow run workflows/cirro/main.nf -profile singularity \
  --tracer_container "$PWD/workflows/cirro/tests/work/tracer-ee259003.sif" \
  --transcripts workflows/cirro/tests/work/input/synthetic_xenium_transcripts.parquet \
  --pmi workflows/cirro/tests/work/input/synthetic_cpmi.csv.gz \
  --sample_name tracer_cirro_smoke --platform xenium \
  --qv_min 20 --cpus 2 --memory_gb 8 --max_memory_gb 16 \
  --outdir workflows/cirro/tests/results-local
```

After downloading a Cirro smoke-test result, compare data content rather than
timestamps, paths, gzip headers, or host metadata:

```bash
python workflows/cirro/tests/compare_runs.py \
  workflows/cirro/tests/results-local/tracer_results \
  workflows/cirro/tests/results-cirro/tracer_results \
  --output workflows/cirro/tests/concordance.json
```

## Cirro integration files

The repository root `.cirro/` directory follows Cirro's current custom
pipeline contract. `process-input.json` maps UI selections to Nextflow
parameters and maps `outdir` to the destination dataset. No credential,
registry token, or Cirro token is stored in this repository.

No-seg is not included in this first release: the pinned no-seg entrypoint is
for VisiumHD matrices and spatial metadata rather than the Xenium dataset
contract above, so exposing it here would make the Cirro form ambiguous.
