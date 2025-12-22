# HOT-NERD

High-Order Transcriptomic with NPMI-Enhanced Reconstruction & Delaunay-stitching (HOT-NERD)

Overview
--------
HOT-NERD provides lightweight utilities for imaging-based spatial transcriptomics 3D segmentation and partial pseudo cell construction. The package includes tools to:

- Partition large tissue spatially using Metis on a kNN graph built from cell centroids.
- Compute gene co-occurrence statistics (PMI / NPMI) and derive per-cell purity and conflict metrics.
- Utilities to refine cell segmentation using a 3D transcript graph and identify 3D partial (pseudo) cells.

Quick start
-----------
Install the package (editable for development):

```bash
python3 -m pip install -e '.[dev]'
```

Import and inspect available functions:

```python
import hotnerd
print(hotnerd.__version__)
print(sorted(hotnerd.__all__))
```

Example
-------
The `examples/` folder contains a runnable demonstration that shows how HOT-NERD can refine an initial segmentation produced by the 10X Xenium V1 platform.

- Original segmentation (10X Xenium V1):

![Original segmentation](examples/output/10X_transcripts.png)

- After refining segmentation with HOT-NERD, we can identify Z-axis overlap at single-cell level:

![Refined segmentation (HOT-NERD)](examples/output/cell_124838_concave_refinement.png)

Run the example locally:

```bash
pip install -e .
python examples/refine_segmentation.py
```

Design notes
------------
- Source layout: `src/` package layout.
- Runtime dependencies include `numpy`, `pandas`, `geopandas`, `shapely`, `scikit-learn`, `pymetis`, `open3d`, and `matplotlib`.

Contact
-------
For questions or collaboration, please contact:
- Long Yuan — lyuan13@jhmi.edu
- Atul Deshpande — adeshpande@jhu.edu

Repository
----------
https://github.com/imlong4real/HOT-NERD

License
-------
MIT (see LICENSE)


