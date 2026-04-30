"""Compatibility shim — core.py has been split into per-phase modules.

All public names remain importable as `from tracer.core import X` thanks
to the re-exports below. New code should import from the specific
submodule, e.g. `from tracer.pruning import prune_transcripts_fast`.
"""
from ._repro import (
    set_reproducibility_seed,
    _ensure_reproducibility_seed,
    reproducibility_smoke_test,
)
from ._utils import relu_symmetric
from .pruning import (
    build_dense_npmi_matrix,
    prune_genes_by_npmi_greedy,
    prune_transcripts,
    prune_transcripts_fast,
    pairwise_npmi_stats,
    diagnostic_npmi_report,
)
from .graph import (
    build_graph,
    to_networkx,
    delaunay_edges,
)
from .spatial import (
    annotate_unassigned_components,
    annotate_unassigned_components_fast,
    enforce_spatial_coherence,
    enforce_spatial_coherence_fast,
    reassign_unassigned_to_nearby_entities,
    reassign_unassigned_to_nearby_entities_fast,
)
from .stitching import (
    infer_entity_type,
    build_entity_table,
    coherence_C_from_genes,
    coherence_C_from_genes_relu,
    deltaC_between_clusters,
    deltaC_between_clusters_relu,
    DSU,
    stitch_entities_hierarchical,
    apply_stitching_to_transcripts,
    apply_stitching_to_transcripts_fast,
    apply_stitching_to_transcripts_memory_efficient,
)
from .cc_scoring import (
    calculate_rankings,
    calculate_thresholds,
    add_edge_prob_stats,
    build_gene_threshold_maps_from_ranked_df,
    prune_graph,
    build_npmi_matrix_from_long,
    compute_purity_conflict_per_cc,
    compute_purity_conflict_per_cc_relu,
    purity_conflict_from_cc,
    build_cc_delaunay_graph,
    compute_deltaC_stitch,
    stitch_connected_components,
)
# Re-export Cython accelerators (some scripts reference tracer.core._cy_prune)
from . import _cy_prune, _cy_spatial  # noqa: F401
