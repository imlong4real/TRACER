# Figure 2 — PDAC Xenium transcript-fate & z-stack reconstruction

Source data: `datasets/pancreas_cancer_xenium_10x/pdac_io_partition_sequential.parquet`
(20,734,426 transcripts, one entity-class label per TRACER phase).

Entity-class codes (`etype_at_<phase>`):
`0 = original cell`, `1 = partial`, `3 = unassigned`, `5 = neighboring cell`.

## Scripts

| Script | Outputs |
|--------|---------|
| `alluvial_pdac.py` | `pdac_full_tier_a.{png,svg}`, `pdac_full_tier_b.{png,svg}` |
| `zstack_pdac.py`   | `pdac_zstack_reconstruction.{png,svg}` |

All figures are written to `outputs/` at 300 dpi (PNG + vector SVG).

### Alluvial diagrams
Track transcript fate transitions between entity classes across TRACER phases.
Node columns are phase *states*; the italic labels sit over the ribbons (the
operation that produced the next state).

* **Tier A** (coarse): `input → Prune → Group → Stitch`
  (`input, phase1, group, stitch`).
* **Tier B** (detailed): `Prune, Rescue, Group, Post-Group Rescue, Stitch,
  Demote, Final Rescue` (`input, phase1, rescue, group, post_group_rescue,
  stitch, demote, final_rescue`).

Rendered on a dark charcoal canvas with a vivid, high-contrast entity-class
palette. Ribbons use a **source→target colour gradient**, so a transcript's
fate change is legible in the flow itself. Class slots keep a fixed top→bottom
order so a class occupies the same band in every column. Nodes are luminous
soft-rounded bars; refined light typography and a bottom legend complete the
Nature-style look.

### Z-stack reconstruction
Two transcript slabs in one 3-D scene over a 300 µm ROI
(`x0=4600, y0=3500`):

* lower slab — complete cells (`etype==0`) at native z depth;
* upper slab — TRACER-reconstructed partial cells (`etype==1`), lifted above.

Each cell gets a unique, perceptually-uniform colour from its `cell_id` via a
golden-angle hue walk in CIELAB (`skimage.color.lab2rgb`). Dark background,
Nature-style minimal chrome.

## Reproduce
```bash
python scripts/reproducibility/fig2/alluvial_pdac.py
python scripts/reproducibility/fig2/zstack_pdac.py
```
Requires: numpy, pandas, pyarrow, matplotlib, scikit-image.
