# Lung Cancer Tutorial

**Author:** Long Yuan  
**Email:** lyuan13[at]jhmi.edu

This tutorial demonstrates how TRACER improves cell segmentation quality and enhances downstream analysis for spatial transcriptomics data from a lung cancer biopsy sample.

## Overview

We compare three segmentation approaches:
1. **Original 10X Multimodal Xenium segmentation** - baseline cell segmentation
2. **TRACER Stitched segmentation** - NPMI-aware graph-based cell boundary reconstruction  
3. **TRACER Stitched + Fine-tuned segmentation** - iterative refinement with spatial contraints and NPMI-based quality metrics

## Key Results

### Cell Quality Metrics Improved Significantly

TRACER's NPMI-based purity and conflict scores quantify segmentation quality at the single-cell level:

![Purity and Conflict Scores](plot/lung_cancer_mean_purity_conflict_transcript_scores.png)

**Purity Score** (measures gene co-expression consistency within cells):
- **0.292** - Original 10X Multimodal segmentation
- **0.356** - TRACER Stitched (+22% improvement)
- **0.373** - TRACER Stitched + Fine-tuned (+28% improvement)

**Conflict Score** (measures incompatible gene signatures, lower is better):
- **0.032** - Original 10X Multimodal segmentation  
- **0.009** - TRACER Stitched (-72% reduction)
- **0.008** - TRACER Stitched + Fine-tuned (-75% reduction)

### Enhanced UMAP Interpretability

The improved segmentation quality translates directly to more biologically interpretable UMAP embeddings:

![UMAP Comparison](plot/lung_cancer_umap_whole_cell.png)

TRACER segmentation produces:
- **Better-defined cell clusters** with clearer boundaries between cell types
- **Reduced technical noise** from mis-segmented or contaminated cells
- **Improved cell type separation** enabling more accurate downstream annotation

## Notebooks

- [`lung_cancer.ipynb`](lung_cancer.ipynb) - Main analysis pipeline with TRACER segmentation
- [`metrics_umap.ipynb`](metrics_umap.ipynb) - NPMI-based quality metrics computation and UMAP analysis
- [`scanpy_dotplot.ipynb`](scanpy_dotplot.ipynb) - Cell type marker visualization

## Data

- Input: Xenium lung cancer biopsy transcript data
- NPMI matrix: Pre-computed nucleus-level gene co-occurrence scores
- Output: Segmented cells with quality metrics, UMAP coordinates, and cluster annotations

## Methods

### NPMI-Based Quality Metrics

**Purity Score**: Fraction of positively correlated gene pairs (NPMI > 0.05) within each cell. Higher purity indicates consistent gene expression patterns.

**Conflict Score**: Normalized magnitude of negatively correlated gene pairs (NPMI < 0) within each cell. Higher conflict suggests contamination from multiple cell types.

These metrics leverage gene co-expression patterns learned from high-quality nucleus-overlapping transcripts to assess segmentation quality without requiring ground truth labels.