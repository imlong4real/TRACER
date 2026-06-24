"""TRACER package.

Convenience exports for commonly used functions and metadata.
"""

__all__ = [
    "get_confident_nuclei_transcripts",
    # `compute_npmi` retired. The production tutorial caller
    # (process_melanoma_data.py) calls `compute_pmi_bootstrap`
    # directly; the unused imports in 2 other notebooks
    # (lung_cancer/mouse_ileum metrics_umap.ipynb) were stripped.
    "compute_pmi_bootstrap",
    "PmiBootstrapResult",
    "build_cell_gene_matrix",
    "build_sparse_pmi_matrix",
    "build_npmi_matrix",
    "attach_metrics_to_adata",
    "compute_cell_purity",
    "compute_cell_conflict",
    "compute_purity_and_conflict",
    "relu_symmetric",
    "compute_cell_purity_relu",
    "compute_cell_conflict_relu",
    "attach_metrics_to_adata_relu",
    "compute_purity_and_conflict_relu",
    "calculate_rankings",
    "calculate_thresholds",
    "build_graph",
    "add_edge_prob_stats",
    "to_networkx",
    "build_gene_threshold_maps_from_ranked_df",
    "prune_graph",
    "build_npmi_matrix_from_long",
    "diagnostic_npmi_report",
    "compute_purity_conflict_per_cc",
    "compute_purity_conflict_per_cc_relu",
    "purity_conflict_from_cc",
    "build_cc_delaunay_graph",
    "coherence",
    "signal_strength",
    "deltaC",
    "compute_housekeeping_mask",
    "compute_deltaC_stitch",
    "stitch_connected_components",
    "build_dense_pmi_matrix_small_panel",
    "prune_transcripts",
    "prune_transcripts_fast",
    "annotate_unassigned_components",
    "annotate_unassigned_components_fast",
    "apply_stitching_to_transcripts",
    "apply_stitching_to_transcripts_fast",
    "apply_stitching_to_transcripts_memory_efficient",
    "build_entity_table",
    "stitch_entities_hierarchical",
    "delaunay_edges",
    
    "prune_genes_by_npmi_greedy",
    "enforce_spatial_coherence",
    "enforce_spatial_coherence_fast",
    "reassign_unassigned_to_nearby_entities",
    "reassign_unassigned_to_nearby_entities_fast",
    "reassign_unassigned_by_gene_compat",
    "reassign_unassigned_to_nearest_tx_no_neg",
    "reassign_unassigned_grid_pool",
    "pre_stage2_rescue",
    "demote_small_entities",
    "plot_cc",
    "plot_3d_concave_cell",
    "plot_3d_convex_cell",
]

__version__ = "0.1.1"
__author__ = "Long Yuan <lyuan13@jhmi.edu>"
__license__ = "MIT"


from .metrics import (
    get_confident_nuclei_transcripts,
    compute_pmi_bootstrap,
    PmiBootstrapResult,
    build_cell_gene_matrix,
    build_npmi_matrix,
    attach_metrics_to_adata,
    compute_cell_purity,
    compute_cell_conflict,
    compute_purity_and_conflict,
    relu_symmetric,
    compute_cell_purity_relu,
    compute_cell_conflict_relu,
    attach_metrics_to_adata_relu,
    compute_purity_and_conflict_relu,
)

from .core import (
    calculate_rankings,
    calculate_thresholds,
    build_graph,
    add_edge_prob_stats,
    to_networkx,
    build_gene_threshold_maps_from_ranked_df,
    prune_graph,
    build_npmi_matrix_from_long,
    diagnostic_npmi_report,
    compute_purity_conflict_per_cc,
    compute_purity_conflict_per_cc_relu,
    purity_conflict_from_cc,
    build_cc_delaunay_graph,
    coherence,
    signal_strength,
    deltaC,
    compute_housekeeping_mask,
    compute_deltaC_stitch,
    stitch_connected_components,
    build_dense_pmi_matrix_small_panel,
    build_sparse_pmi_matrix,
    prune_transcripts,
    prune_transcripts_fast,
    annotate_unassigned_components,
    annotate_unassigned_components_fast,
    apply_stitching_to_transcripts,
    apply_stitching_to_transcripts_fast,
    apply_stitching_to_transcripts_memory_efficient,
    build_entity_table,
    stitch_entities_hierarchical,
    delaunay_edges,
    prune_genes_by_npmi_greedy,
    enforce_spatial_coherence,
    enforce_spatial_coherence_fast,
    reassign_unassigned_to_nearby_entities,
    reassign_unassigned_to_nearby_entities_fast,
    reassign_unassigned_by_gene_compat,
    reassign_unassigned_to_nearest_tx_no_neg,
    reassign_unassigned_grid_pool,
    pre_stage2_rescue,
    demote_small_entities,
)

# Optional plot module (requires open3d)
try:
    from .plot import (
        plot_cc,
        plot_3d_concave_cell,
        plot_3d_convex_cell,
    )
except ImportError:
    # open3d not installed; plotting functions unavailable
    pass
