# Breast Cancer Tissue Analysis with HOT-NERD

**Author:** Long Yuan  
**Email:** lyuan13[at]jhmi.edu

This tutorial demonstrates the HOT-NERD pipeline on a large-scale Xenium v1 breast cancer dataset containing ~28M transcripts.

## Dataset

**Xenium v1 Breast Cancer Dataset**

The data is from:
> Janesick, A., Shelansky, R., Gottscho, A.D. et al. High resolution mapping of the tumor microenvironment using integrated single-cell, spatial and in situ analysis. *Nat Commun* **14**, 8353 (2023). https://doi.org/10.1038/s41467-023-43458-x

This dataset provides high-resolution spatial transcriptomics of the breast tumor microenvironment, enabling detailed analysis of cell-cell interactions and tissue organization.

## Requirements

- Python 3.9+
- HOT-NERD package with Cython acceleration (see main README)
- Required packages: pandas, numpy, scipy, torch, torch-geometric, tqdm

## Data Files

Place the following files in the `data/` directory:
- `breast_cancer_df.parquet` - Transcript coordinates and gene annotations (~28M transcripts)
- `breast_cancer_npmi.csv` - Pre-computed NPMI (normalized pointwise mutual information) values for gene pairs

## Running the Pipeline

```bash
python tutorials/breast_cancer/run_breast_cancer.py
```

The script executes the full HOT-NERD pipeline:
1. **Stage 1**: Conservative NPMI-based pruning (removes low-quality transcripts)
2. **Stage 2**: Graph-based component detection and annotation of unassigned transcripts
3. **Stage 3**: Hierarchical stitching of partial cells and components
4. **Stage 4**: Spatial coherence enforcement
5. **Stage 5**: Final stitching refinement

## Expected Output

```
$ python tutorials/breast_cancer/run_breast_cancer.py
Starting HOT-NERD run on breast cancer tissue
Reading transcripts from: /Users/lyuan13/Desktop/HOT-NERD/tutorials/breast_cancer/data/breast_cancer_df.parquet
Loaded transcripts rows: 28059774 took 0.43671131134033203 s
Reading NPMI table from: /Users/lyuan13/Desktop/HOT-NERD/tutorials/breast_cancer/data/breast_cancer_npmi.csv
Loaded npmi rows: 96721 took 0.07839298248291016 s
Stage 1: prune_transcripts_fast (conservative NPMI)
prune_pass1: 100%|█████████████████████████████████████████| 166217/166217 [00:00<00:00, 2106237546.73it/s]
apply_pass1: 100%|████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 43240.25it/s]
prune_pass2: 100%|██████████████████████████████████████████████| 164337/164337 [00:02<00:00, 77547.89it/s]
Stage 1 done: rows= 28059774 took 127.9855329990387 s
Stage 2: annotate_unassigned_components_fast (build graph + CCs)
Constructed 6,518,980 edges among 5,251,713 transcripts (k≤5, d≤1.5 µm)
post_cc_mapping: : 4it [00:00,  8.54it/s]                                                                  
grouping: 100%|██████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.56s/it]
[INFO] Using Cython-accelerated pruning (7732 components)
prune_comps: 100%|█████████████████████████████████████████████████████| 7732/7732 [50:58<00:00,  2.53it/s]
Stage 2 done: rows= 28059774 took 3103.3299539089203 s
Stage 3: apply_stitching_to_transcripts_fast (initial stitching)
stitching: 100%|████████████████████████████████████████████████████████████| 2/2 [09:38<00:00, 289.04s/it]
Stage 3 done: rows= 28059774 took 601.4606521129608 s
Saving df_stitched to /Users/lyuan13/Desktop/HOT-NERD/tutorials/breast_cancer/output/df_stitched.parquet
Stage 4: enforce_spatial_coherence_fast (split spatially disjoint labels)
Constructed 140,289,974 edges among 28,059,774 transcripts (k≤5, d≤25.0 µm)
spatial_labels: 100%|████████████████████████████████████████████| 290077/290077 [1:00:50<00:00, 79.46it/s]
Stage 4 done: rows= 28059774 took 3824.076679944992 s
Stage 5: apply_stitching_to_transcripts_fast (final stitching on split labels)
stitching: 100%|█████████████████████████████████████████████████████████| 2/2 [1:27:52<00:00, 2636.20s/it]
Stage 5 done: rows= 28059774 took 5313.000539064407 s
Saving df_finetuned to /Users/lyuan13/Desktop/HOT-NERD/tutorials/breast_cancer/output/df_finetuned.parquet
Pipeline complete. Outputs:
 - /Users/lyuan13/Desktop/HOT-NERD/tutorials/breast_cancer/output/df_stitched.parquet
 - /Users/lyuan13/Desktop/HOT-NERD/tutorials/breast_cancer/output/df_finetuned.parquet
```

**Total Runtime**: ~3.5 hours on this dataset with Cython acceleration enabled on Apple M1

## Output Files

Results are saved to the `output/` directory:
- `df_stitched.parquet` - Transcripts with initial cell assignments after stitching
- `df_finetuned.parquet` - Final cell assignments after spatial coherence enforcement

## Performance Notes

The pipeline leverages several optimizations for large datasets:
- **Cython acceleration** for greedy pruning and spatial component detection
- **scipy.spatial.cKDTree** for fast k-nearest neighbor graph construction (the example showcased an user-customized function)
- **scipy.sparse.csgraph** for efficient connected components detection
- **Vectorized DataFrame operations** for bulk transcript assignment

Without Cython acceleration, runtime may be 2-5× longer. To compile Cython modules:
```bash
pip install -e .
```

## Quality Metrics & UMAP Enhancement

HOT-NERD significantly improves cell segmentation quality using NPMI-based purity and conflict scores:

![Purity and Conflict Scores](plot/breast_cancer_mean_purity_conflict_transcript_scores.png)

**Quantitative Improvements:**
- **Purity Score** (gene co-expression consistency): 0.457 (Original Xenium) → 0.686 (HOT-NERD Stitched) → **0.708** (HOT-NERD Stitched + Fine-tuned) — **+55% improvement**
- **Conflict Score** (incompatible gene signatures): 0.055 (Original Xenium) → 0.005 (HOT-NERD Stitched) → **0.004** (HOT-NERD Stitched + Fine-tuned) — **-93% reduction**

**Enhanced UMAP with Author-Annotated Cell Types:**

The improved segmentation produces clearer cell type clustering with better within-lineage cohesion and between-lineage separation:

![UMAP with Cell Type Overlay](plot/breast_cancer_umap_whole_cell_supervised_overlay.png)

The enhanced UMAP demonstrates:
- **Better-defined cell type clusters** reflecting true biological identity
- **Improved lineage separation** between distinct cell populations
- **Reduced contamination** within cell type clusters
- **More accurate downstream annotation** and biological interpretation

## Downstream Analysis

The output Parquet files contain cell assignments and can be used for:
- Cell type identification
- Spatial domain detection
- Cell-cell interaction analysis
- Tissue architecture visualization

See the lung cancer tutorial (`tutorials/lung_cancer/`) for examples of visualization and downstream analysis.
