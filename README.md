# HOT-NERD

High-Order Transcriptomic with NPMI-Enhanced Reconstruction & Delaunay-stitching (HOT-NERD)

Overview
--------
HOT-NERD provides lightweight utilities for imaging-based spatial transcriptomics 3D segmentation and partial pseudo cell construction. The package includes tools to:

- Partition large tissue spatially using Metis on a kNN graph built from cell centroids.
- Compute gene co-occurrence statistics (PMI / NPMI) and derive per-cell purity and conflict metrics.
- Utilities for refine cell segmentation using 3D cell transcript graph and identify 3D partial (pseudo) cells.

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

Design notes
------------
- Source layout: `src/` package layout.
- Runtime dependencies include `numpy`, `pandas`, `geopandas`, `shapely`, `scikit-learn`, `pymetis`, and `matplotlib`.

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
MIT (see `LICENSE`)

