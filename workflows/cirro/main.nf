nextflow.enable.dsl = 2

/*
 * Cirro adapter for TRACER Seg.
 *
 * The analysis itself is supplied exclusively by the immutable TRACER image
 * configured in nextflow.config.  This workflow only stages Cirro inputs,
 * applies the documented Xenium preprocessing step, and publishes results.
 */

def requiredParam(String name, value) {
    if (value == null || value.toString().trim().isEmpty()) {
        error "Missing required parameter --${name}"
    }
    return value
}

def optionalPathChannel(value) {
    if (value == null || value.toString().trim().isEmpty()) {
        return Channel.value([])
    }
    return Channel.fromPath(value, checkIfExists: true)
}

def sourceB64(value) {
    if (value == null) {
        return ''
    }
    return value.toString().bytes.encodeBase64().toString()
}

process TRACER_SEG {
    tag "${sample_name}"
    label 'tracer_seg'

    publishDir params.outdir, mode: 'copy', overwrite: true

    input:
    path transcripts
    path pmi
    path user_config
    path cell_boundaries
    path nucleus_boundaries
    path runner
    path fastparquet_shim
    val sample_name
    val platform
    val qv_min
    val remove_control_probes
    val drop_unassigned
    val pmi_threshold
    val g_z_um
    val tau
    val min_tx_per_cell_for_scores
    val score_mode
    val seed
    val transcripts_source_b64
    val pmi_source_b64
    val user_config_source_b64
    val cell_boundaries_source_b64
    val nucleus_boundaries_source_b64
    val workflow_commit
    val workflow_revision

    output:
    path 'tracer_results', emit: results

    script:
    def qvArg = qv_min ? "--qv-min ${qv_min}" : ''
    def controlsArg = remove_control_probes ? '--remove-control-probes' : ''
    def dropArg = drop_unassigned ? '--drop-unassigned' : ''
    def pmiThresholdArg = pmi_threshold ? "--pmi-threshold ${pmi_threshold}" : ''
    def gzArg = g_z_um ? "--g-z-um ${g_z_um}" : ''
    def tauArg = tau ? "--tau ${tau}" : ''
    def configArg = user_config ? "--user-config '${user_config}'" : ''
    def cellMaskArg = cell_boundaries ? "--cell-boundaries '${cell_boundaries}'" : ''
    def nucleusMaskArg = nucleus_boundaries ? "--nucleus-boundaries '${nucleus_boundaries}'" : ''
    def taskMemoryB64 = task.memory.toString().bytes.encodeBase64().toString()

    """
    export PYTHONHASHSEED='${seed}'
    export OMP_NUM_THREADS='${task.cpus}'
    export OPENBLAS_NUM_THREADS='${task.cpus}'
    export MKL_NUM_THREADS='${task.cpus}'
    export NUMEXPR_NUM_THREADS='${task.cpus}'

    python '${runner}' \
      --transcripts '${transcripts}' \
      --pmi '${pmi}' \
      --outdir tracer_results \
      --sample-name '${sample_name}' \
      --platform '${platform}' \
      --seed '${seed}' \
      --min-tx-per-cell-for-scores '${min_tx_per_cell_for_scores}' \
      --score-mode '${score_mode}' \
      --transcripts-source-b64 '${transcripts_source_b64}' \
      --pmi-source-b64 '${pmi_source_b64}' \
      --user-config-source-b64 '${user_config_source_b64}' \
      --cell-boundaries-source-b64 '${cell_boundaries_source_b64}' \
      --nucleus-boundaries-source-b64 '${nucleus_boundaries_source_b64}' \
      --fastparquet-shim '${fastparquet_shim}' \
      --container-image '${params.tracer_oci_image}' \
      --execution-container '${params.tracer_container}' \
      --tracer-source-commit '${params.tracer_source_commit}' \
      --tracer-version '${params.tracer_version}' \
      --workflow-commit '${workflow_commit}' \
      --workflow-revision '${workflow_revision}' \
      --task-attempt '${task.attempt}' \
      --task-cpus '${task.cpus}' \
      --task-memory-b64 '${taskMemoryB64}' \
      ${qvArg} \
      ${controlsArg} \
      ${dropArg} \
      ${pmiThresholdArg} \
      ${gzArg} \
      ${tauArg} \
      ${configArg} \
      ${cellMaskArg} \
      ${nucleusMaskArg}
    """

    stub:
    """
    mkdir -p tracer_results/outputs tracer_results/preprocessing/qc tracer_results/logs tracer_results/provenance
    touch tracer_results/outputs/transcripts_tracer_refined.parquet
    touch tracer_results/outputs/cell_by_gene_tracer.h5ad
    touch tracer_results/outputs/cell_scores.tsv.gz
    touch tracer_results/config_receipt.json
    touch tracer_results/provenance/resolved_tracer_config.json
    touch tracer_results/provenance/output_fingerprints.json
    """
}

workflow {
    requiredParam('transcripts', params.transcripts)
    requiredParam('pmi', params.pmi)

    if (!(params.platform in ['xenium', 'atera'])) {
        error "--platform must be one of: xenium, atera"
    }
    if (!(params.sample_name ==~ /[A-Za-z0-9][A-Za-z0-9._-]{0,127}/)) {
        error "--sample_name must contain only letters, numbers, dot, underscore, or dash"
    }
    if ((params.cpus as int) < 1) {
        error "--cpus must be at least 1"
    }
    if ((params.memory_gb as int) < 8 || (params.max_memory_gb as int) < (params.memory_gb as int)) {
        error "--memory_gb must be at least 8 and --max_memory_gb must be >= --memory_gb"
    }
    if (params.g_z_um && !(params.g_z_um.toString() ==~ /(auto|[0-9]+(?:\.[0-9]+)?)/)) {
        error "--g_z_um must be 'auto' or a positive number"
    }

    transcripts_ch = Channel.fromPath(params.transcripts, checkIfExists: true)
    pmi_ch = Channel.fromPath(params.pmi, checkIfExists: true)
    user_config_ch = optionalPathChannel(params.user_config)
    cell_boundaries_ch = optionalPathChannel(params.cell_boundaries)
    nucleus_boundaries_ch = optionalPathChannel(params.nucleus_boundaries)
    runner_path = file("${projectDir}/bin/run_tracer_seg.py")
    if (!runner_path.exists()) {
        runner_path = file("${projectDir}/workflows/cirro/bin/run_tracer_seg.py")
    }
    if (!runner_path.exists()) {
        error "Cannot locate workflows/cirro/bin/run_tracer_seg.py"
    }
    runner_ch = Channel.value(runner_path)
    fastparquet_shim_path = file("${runner_path.parent}/fastparquet.py")
    if (!fastparquet_shim_path.exists()) {
        error "Cannot locate workflows/cirro/bin/fastparquet.py"
    }
    fastparquet_shim_ch = Channel.value(fastparquet_shim_path)

    TRACER_SEG(
        transcripts_ch,
        pmi_ch,
        user_config_ch,
        cell_boundaries_ch,
        nucleus_boundaries_ch,
        runner_ch,
        fastparquet_shim_ch,
        params.sample_name,
        params.platform,
        params.qv_min == null ? '' : params.qv_min.toString(),
        params.remove_control_probes as boolean,
        params.drop_unassigned as boolean,
        params.pmi_threshold == null ? '' : params.pmi_threshold.toString(),
        params.g_z_um == null ? '' : params.g_z_um.toString(),
        params.tau == null ? '' : params.tau.toString(),
        params.min_tx_per_cell_for_scores as int,
        params.score_mode,
        params.seed as int,
        sourceB64(params.transcripts),
        sourceB64(params.pmi),
        sourceB64(params.user_config),
        sourceB64(params.cell_boundaries),
        sourceB64(params.nucleus_boundaries),
        workflow.commitId ?: 'local-uncommitted',
        workflow.revision ?: 'local'
    )
}
