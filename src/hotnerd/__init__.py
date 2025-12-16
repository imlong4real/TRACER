"""HOT-NERD package.

Convenience exports for the most commonly used functions.
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
]

__version__ = "0.1.0"
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

