nextflow.enable.dsl = 2

/*
 * Cirro adapter: RCTD (spacexr) GBmap label transfer over TRACER entity arms.
 *
 * Consumes a TRACER Seg output dataset, rebuilds four entity x gene matrices
 * on one shared panel-matched gene axis, and runs RCTD on each with a single
 * shared settings block. No arm-specific tuning exists anywhere in this
 * workflow: RCTD_RUN receives the same value channels for every arm.
 */

def requiredParam(String name, value) {
    if (value == null || value.toString().trim().isEmpty()) {
        error "Missing required parameter --${name}"
    }
    return value
}

def resolveDatasetPath(value, inputDir) {
    if (value == null || value.toString().trim().isEmpty()) {
        return value
    }
    def candidate = value.toString()
    if (candidate ==~ /^[A-Za-z][A-Za-z0-9+.-]*:\/\/.*/ || candidate.startsWith('/')) {
        return candidate
    }
    if (inputDir == null || inputDir.toString().trim().isEmpty()) {
        return candidate
    }
    def base = inputDir.toString().replaceFirst('/+$', '')
    def relative = candidate.replaceFirst('^/+', '')
    def baseLeaf = base.tokenize('/').last()
    if (relative == baseLeaf) {
        relative = ''
    } else if (relative.startsWith("${baseLeaf}/")) {
        relative = relative.substring(baseLeaf.size() + 1)
    }
    return relative ? "${base}/${relative}" : base
}


process RCTD_PREP {
    tag "${sample_name}"
    label 'prep'

    input:
    path transcripts
    path reference
    path prep_script
    val sample_name
    val patient
    val section
    val arms

    output:
    path 'entities/*.h5ad',            emit: entities
    path 'entities/prep_manifest.json', emit: manifest
    path 'entities/genes.txt',          emit: genes

    script:
    """
    mkdir -p entities

    # The RCTD gene axis is taken from the reference itself, so the spatial
    # matrices and the reference cannot drift apart.
    python - <<'PY'
import anndata as ad
a = ad.read_h5ad("${reference}", backed="r")
with open("entities/genes.txt", "w") as fh:
    fh.write("\\n".join(map(str, a.var_names)))
print(f"[genes] {a.n_vars} panel-matched genes from reference", flush=True)
PY

    python '${prep_script}' \
      --transcripts '${transcripts}' \
      --genes entities/genes.txt \
      --sample-name '${sample_name}' \
      --patient '${patient}' \
      --section '${section}' \
      --arms '${arms}' \
      --outdir entities
    """

    stub:
    """
    mkdir -p entities
    touch entities/entities_original.h5ad entities/prep_manifest.json entities/genes.txt
    """
}


process RCTD_RUN {
    tag "${sample_name}:${arm}"
    label 'rctd'

    input:
    tuple val(arm), path(entities_h5ad)
    path reference
    path rctd_script
    val sample_name
    val celltype_col
    val doublet_mode
    val umi_min
    val umi_min_sigma
    val gene_cutoff
    val fc_cutoff
    val min_cells_ref
    val reference_min_umi
    val seed

    output:
    tuple val(arm), path("rctd_${arm}"), emit: results

    script:
    """
    export OMP_NUM_THREADS='${task.cpus}'
    export OPENBLAS_NUM_THREADS='${task.cpus}'
    export MKL_NUM_THREADS='${task.cpus}'
    mkdir -p 'rctd_${arm}'

    Rscript '${rctd_script}' \
      --spatial-h5ad '${entities_h5ad}' \
      --reference-h5ad '${reference}' \
      --reference-celltype-col '${celltype_col}' \
      --outdir 'rctd_${arm}' \
      --doublet-mode '${doublet_mode}' \
      --umi-min '${umi_min}' \
      --umi-min-sigma '${umi_min_sigma}' \
      --gene-cutoff '${gene_cutoff}' \
      --fc-cutoff '${fc_cutoff}' \
      --min-cells-per-celltype-reference '${min_cells_ref}' \
      --reference-min-umi '${reference_min_umi}' \
      --celltype-name-map 'rctd_${arm}/celltype_name_map.json' \
      --max-cores '${task.cpus}' \
      --seed '${seed}'
    """

    stub:
    """
    mkdir -p 'rctd_${arm}'
    touch 'rctd_${arm}/rctd_cell_assignments_post.tsv'
    """
}


process RCTD_COLLECT {
    tag "${sample_name}"
    label 'collect'

    publishDir params.outdir, mode: 'copy', overwrite: true

    input:
    path arm_dirs
    path entities
    path manifest
    path collect_script
    val sample_name
    val settings_json_b64

    output:
    path 'rctd_results', emit: results

    script:
    """
    python '${collect_script}' \
      --sample-name '${sample_name}' \
      --prep-manifest '${manifest}' \
      --settings-b64 '${settings_json_b64}' \
      --outdir rctd_results \
      --arm-dirs ${arm_dirs} \
      --entity-h5ads ${entities}
    """

    stub:
    """
    mkdir -p rctd_results
    touch rctd_results/rctd_entities.parquet
    """
}


workflow {
    requiredParam('transcripts', params.transcripts)
    requiredParam('reference', params.reference)

    def resolved_transcripts = resolveDatasetPath(params.transcripts, params.input_dir)

    transcripts_ch = Channel.fromPath(resolved_transcripts, checkIfExists: true)
    reference_ch   = Channel.fromPath(params.reference, checkIfExists: true)

    def script_dir = file("${projectDir}/bin").exists()
        ? file("${projectDir}/bin")
        : file("${projectDir}/workflows/cirro/rctd/bin")
    prep_script    = file("${script_dir}/prep_rctd_entities.py")
    collect_script = file("${script_dir}/collect_rctd_results.py")
    rctd_script    = file("${script_dir}/run_rctd.R")

    RCTD_PREP(
        transcripts_ch,
        reference_ch,
        prep_script,
        params.sample_name,
        params.patient,
        params.section,
        params.arms,
    )

    // (arm, h5ad) pairs, derived from the emitted file names.
    arm_inputs = RCTD_PREP.out.entities
        .flatten()
        .map { f -> tuple(f.simpleName.replaceFirst('^entities_', ''), f) }

    RCTD_RUN(
        arm_inputs,
        reference_ch.first(),
        rctd_script,
        params.sample_name,
        params.celltype_col,
        params.doublet_mode,
        params.umi_min,
        params.umi_min_sigma,
        params.gene_cutoff,
        params.fc_cutoff,
        params.min_cells_per_celltype_reference,
        params.reference_min_umi,
        params.seed,
    )

    def settings = groovy.json.JsonOutput.toJson([
        celltype_col : params.celltype_col,
        doublet_mode : params.doublet_mode,
        umi_min      : params.umi_min,
        umi_min_sigma: params.umi_min_sigma,
        gene_cutoff  : params.gene_cutoff,
        fc_cutoff    : params.fc_cutoff,
        min_cells_per_celltype_reference: params.min_cells_per_celltype_reference,
        reference_min_umi: params.reference_min_umi,
        seed         : params.seed,
        arms         : params.arms,
        reference    : params.reference.toString(),
        transcripts  : resolved_transcripts.toString(),
        tracer_source_commit: params.tracer_source_commit,
        analysis_container  : params.analysis_container,
        spacexr_commit      : params.spacexr_commit,
        workflow_commit     : workflow.commitId ?: '',
        workflow_revision   : workflow.revision ?: '',
    ])
    def settings_b64 = settings.bytes.encodeBase64().toString()

    RCTD_COLLECT(
        RCTD_RUN.out.results.map { it[1] }.collect(),
        RCTD_PREP.out.entities.collect(),
        RCTD_PREP.out.manifest,
        collect_script,
        params.sample_name,
        settings_b64,
    )
}
