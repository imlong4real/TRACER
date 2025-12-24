"""HOT-NERD package.

Convenience exports for commonly used functions and metadata.
"""

__all__ = [
    "metis_partition_cells",
    "build_metis_partition_hulls",
    "plot_metis_partitions",
    "plot_metis_hulls",
    "chunk_transcripts",
    "get_confident_nuclei_transcripts",
    "compute_npmi",
    "build_cell_gene_matrix",
    "build_npmi_matrix",
    "attach_metrics_to_adata",
    "compute_cell_purity",
    "compute_cell_conflict",
    "compute_purity_and_conflict",
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
    "purity_conflict_from_cc",
    "build_cc_delaunay_graph",
    "compute_deltaC_stitch",
    "stitch_connected_components",
    "build_dense_npmi_matrix",
    "prune_transcripts",
    "prune_transcripts_fast",
    "annotate_unassigned_components",
    "annotate_unassigned_components_fast",
    "apply_stitching_to_transcripts",
    "apply_stitching_to_transcripts_fast",
    "build_entity_table",
    "stitch_entities_hierarchical",
    "delaunay_edges",
    "prune_genes_by_npmi_greedy",
    "enforce_spatial_coherence",
    "enforce_spatial_coherence_fast",
    "plot_cc",
    "plot_3d_concave_cell",
    "plot_3d_convex_cell",
]

__version__ = "0.1.1"
__author__ = "Long Yuan <lyuan13@jhmi.edu>"
__license__ = "MIT"

from .tiling import (
    metis_partition_cells,
    build_metis_partition_hulls,
    plot_metis_partitions,
    plot_metis_hulls,
    chunk_transcripts,
)

from .metrics import (
    get_confident_nuclei_transcripts,
    compute_npmi,
    build_cell_gene_matrix,
    build_npmi_matrix,
    attach_metrics_to_adata,
    compute_cell_purity,
    compute_cell_conflict,
    compute_purity_and_conflict,
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
    purity_conflict_from_cc,
    build_cc_delaunay_graph,
    compute_deltaC_stitch,
    stitch_connected_components,
    build_dense_npmi_matrix,
    prune_transcripts,
    prune_transcripts_fast,
    annotate_unassigned_components,
    annotate_unassigned_components_fast,
    apply_stitching_to_transcripts,
    apply_stitching_to_transcripts_fast,
    build_entity_table,
    stitch_entities_hierarchical,
    delaunay_edges,
    prune_genes_by_npmi_greedy,
    enforce_spatial_coherence,
    enforce_spatial_coherence_fast,
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
