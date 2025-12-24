# Breast Cancer Tissue Analysis with HOT-NERD

**Author:** Long Yuan  
**Email:** lyuan13@jhmi.edu

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
Loaded transcripts rows: 28059774 took 0.4473731517791748 s
Reading NPMI table from: /Users/lyuan13/Desktop/HOT-NERD/tutorials/breast_cancer/data/breast_cancer_npmi.csv
Loaded npmi rows: 96721 took 0.1528923511505127 s
Stage 1: prune_transcripts_fast (conservative NPMI)
prune_pass1: 100%|████████████████████████████████████████████████████████████████████████████████| 166217/166217 [00:00<00:00, 5202721104.24it/s]
apply_pass1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 52428.80it/s]
prune_pass2: 100%|████████████████████████████████████████████████████████████████████████████████████| 162149/162149 [00:01<00:00, 112007.88it/s]
Stage 1 done: rows= 28059774 took 122.33935189247131 s
Stage 2: annotate_unassigned_components_fast (build graph + CCs)
Constructed 4,725,401 edges among 3,566,280 transcripts (k≤5, d≤1.5 µm)
post_cc_mapping: : 4it [00:00, 11.47it/s]                                                                                                         
grouping: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.12s/it]
[INFO] Using Cython-accelerated pruning (6519 components)
prune_comps: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 6519/6519 [37:44<00:00,  2.88it/s]
Stage 2 done: rows= 28059774 took 2302.975259780884 s
Stage 3: apply_stitching_to_transcripts_fast (initial stitching)
stitching: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [11:41<00:00, 350.93s/it]
apply_labels: 100%|███| 24689/24689 [45:02<00:00,  9.14it/s]
Stage 3 done: rows= 28059774 took 3434.49645113945 s
Saving df_stitched to /Users/lyuan13/Desktop/HOT-NERD/tutorials/breast_cancer/output/df_stitched.parquet
Stage 4: enforce_spatial_coherence_fast (split spatially disjoint labels)
spatial_labels: 100%|██████████████████████████████████████████████████████████████████████████████████| 264722/264722 [56:18<00:00, 78.36it/s]
Stage 4 done: rows= 28059774 took 3544.750870704651 s
Stage 5: apply_stitching_to_transcripts_fast (final stitching on split labels)
stitching: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [1:30:06<00:00, 2703.38s/it]

```

**Total Runtime**: ~2.5 hours on this dataset with Cython acceleration enabled

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

## Downstream Analysis

The output Parquet files contain cell assignments and can be used for:
- Cell type identification
- Spatial domain detection
- Cell-cell interaction analysis
- Tissue architecture visualization

See the lung cancer tutorial (`tutorials/lung_cancer/`) for examples of visualization and downstream analysis.
