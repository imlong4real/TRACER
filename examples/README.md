# Example: refine segmentation

**Author:** Long Yuan  
**Email:** lyuan13@jhmi.edu

This folder contains a runnable example that demonstrates how to refine
an initial segmentation visualization into a clustered/stitched result
using the `tracer` API.

Files
- `refine_segmentation.py`: runnable script which synthesizes data,
  runs the pipeline, and writes `examples/output/10X.png` and
  `examples/output/v1_example.png`.

Quick start

1. From the repository root, install the package in editable mode:

```bash
pip install -e .
```

2. Run the example:

```bash
python examples/refine_segmentation.py
```

3. Output images are written to `examples/output/`.

If you prefer not to install, run with `PYTHONPATH`:

```bash
PYTHONPATH=./src python examples/refine_segmentation.py
```
