# ROI CNV review — findings and handoff

Written 2026-09-19. Everything in this folder is gitignored scratch analysis; the
figures are the deliverable, the `.py` files beside them are the scripts that made
them (kept for reproducibility, deliberately **not** promoted into `scripts/`).

ROI: public Xenium Prime 5K human ovary, x∈[1750,2250], y∈[14750,15250] (500×500 µm),
run id `roi_nuclear_grid3cpmi`. Arms: `xenium` (raw Xenium segmentation, formerly
called "raw" — the user asked for the rename) and `tracer_whole`.

---

## 1. What was wrong with the arms, and what was fixed

Three defects in `scripts/run_insitucnv_arm.py` were found by comparing against the
recipe that produced the `cnv_epi` ground truth (`make_full_ovary_cnv_preview.py`,
which follows the published ovarian notebook).

### 1.1 Smoothing graph (FIXED — uncommitted, see §5)

The arms smoothed over a **200-NN spatial** graph (`scvelo.pp.neighbors(use_rep="spatial")`).
That is not the published recipe — InSituCNV smooths over an expression/PCA graph —
and for this benchmark specifically it is self-defeating: averaging each cell with its
200 nearest *spatial* neighbours re-mixes transcripts across cell boundaries, which is
exactly the contamination TRACER removes.

Proof it was an artefact (`step3_cnv_is_spatial.png`): under spatial smoothing, the
Spearman correlation between a cell's *local tumour fraction* and its similarity to the
tumour CNV profile was **+0.75**, and held even for cells with **zero** epithelial gene
expression. CNV had become a spatial field, not a cell property.

Fix: `--smoothing-graph {expression,spatial}`, default **expression**, with
`--expression-neighbors 100` and `--neighbor-n-pcs 50`. Compartment bleed dropped from
0.52/0.63 to **0.22/0.04**. Spatial is retained as an opt-in flag only.

Related upstream fact worth remembering: `scvelo`'s `smooth_data_for_cnv` can only
**prune** an existing graph, never widen it (`scvelo/preprocessing/neighbors.py:468`
requires `n_neighbors < get_n_neighs(adata)`). So the manuscript's `n_neighbors=200`
against a stored 15-NN graph was a **no-op** — the truth was built on 15 NN.

### 1.2 `log1p` before smoothing (FIXED — uncommitted)

The arms did `sc.pp.log1p` before smoothing and then passed `layer="M"` to
`cnv.tl.infercnv`, i.e. inferCNV ran on log-scale input while the truth ran on linear.
The canonical chain is counts → `normalize_total` → smooth → `normalize_total` + `log1p`.
The pre-smoothing `log1p` was removed.

### 1.3 Two-stage clone recipe (NOT fixed in the pipeline)

The published ovarian recipe is: `cnv.tl.leiden(resolution=0.4)` over all cells, then an
**epithelial-only** subset with PCA and kNN **rebuilt on the subset**,
`cnv.tl.leiden(resolution=0.2, key_added="cnv_epi")`, then drop clones <0.1% of cells.
The arms stop after stage 1. `--cluster-resolution` still defaults to **0.4** even though
the sweep (`denovo_resolution_sweep.csv`) shows **0.1** is far better (ARI vs truth 0.66
vs 0.38). Stage 2 exists only in the scratchpad (`denovo_clones.py`).

Caveat for whoever implements it: do **not** use
`insitucnv.tl.cnv.cluster_cnv_resolutions(subset_key=...)` — in subset mode it slices the
full-object CNV graph instead of rebuilding PCA/kNN on the subset
(`insitucnv/tl/cnv.py:264-265`), which is not how `cnv_epi` was built.

### 1.4 Still open

- The two arms' CNV column axes differ (354 vs 348 windows). Chromosome-averaged outputs
  are unaffected; any bin-level cross-arm comparison must align on bp intervals
  reconstructed from `var` (see §4).
- No full-slide CNV matrix was ever persisted (`make_full_ovary_cnv_preview.py` writes an
  h5ad only under `--write-h5ad`, which was never set). `cnv_epi` labels survived, the
  per-window values did not. A full-slide reference profile is a re-run, not a lookup.
- `data/roi_cell_metadata.csv` (the `cnv_epi` truth) is read by no committed script; every
  join so far has been ad hoc.

---

## 2. Biological findings (the actual investigation)

All of these were produced **after** the expression-kNN fix. Figures named below are in
this folder.

### 2.1 Segmentation-level comparison — `seg_compare.png`, `seg_spatial_celltype.png`, `fig_spatial_celltype_maps.png`

Cell counts and cell-type proportions before/after TRACER. TRACER's reference compartment
is **~10× less contaminated** by epithelial signal than Xenium's (mean epithelial score
0.037 vs 0.390).

### 2.2 De novo clone recovery — `denovo_clones.png`, `denovo_clone_crossarm_corr.csv`

Both arms recover the same overarching clone structure and the clones sit in the same
places spatially. Clone matching across arms must be done on **clone-specific residuals**,
not raw profile correlation — all pairs correlate 0.74–0.93 because of the shared
epithelial component, and Hungarian matching on raw correlation mismatches a pair.

### 2.3 The ID join — this is where I was wrong once, read this before redoing it

TRACER `cell` entities reuse the Xenium `cell_id` string, so `xenium` and `tracer_whole`
entities can be joined **directly by ID**. I originally used a ≥50%-transcript-ownership
projection instead, which silently dropped 299 cells and produced the false claim
"only 27 cells flip compartment, 98% agreement".

Correct numbers on the ID join (`fig_id_linkage.png`, `fig_disagreement_sankey.png`):
**163** tumour→reference switchers and **6** reference→tumour.

The flips **are** spatially localised: 27.3% flip at low local-epithelial density vs 7.0%
at high; Spearman −0.242, p=1.4e-14. Not depth-driven (p=0.25).
(An earlier "not spatially localised" claim was based on 12 cells and is withdrawn.)

### 2.4 The switchers are genuine tumour cells — `fig_switch_scores.png`, `fig_switch_paired_scores.png`, `fig_switch_cnv.png`

Paired epithelial/fibroblast scores for the same IDs under both segmentations, with
concordant-epithelial and concordant-fibroblast controls.

- Switcher nuclei carry a median of **10** nuclear epithelial marker transcripts; real
  fibroblasts carry **1**.
- Under Xenium segmentation the switchers' CNV profile correlates **r = 0.905** with the
  tumour profile.
- Not circular: force-assigning the switchers into the tumour compartment barely moves the
  result (r 0.121 → 0.122).

So TRACER is moving genuine tumour cells out of the tumour compartment.

### 2.5 Where the epithelial material goes — `fig_switch_cnv_partials.png`, `partial_recipients.csv`

The excised epithelial transcripts land in **`partial` entities**, which carry the tumour
CNV profile at **r = 0.983**.

Careful with the wording: what is established is that the *epithelial-gene-bearing
transcripts* relocated and that inferCNV scores those as tumour. Whether real copy number
travelled is **not** established — see §3.

### 2.6 Why the two components physically overlap — `fig_3d_hulls.png`, `partial_recipient_nuclei.csv`

The user's question: if TRACER carved out two distinct components, why do their convex
hulls overlap rather than separate?

- The epithelial partial takes a median **342 of the Xenium cell's 1,554 nuclear
  transcripts (28.1%)**, in 40/40 pairs.
- Of the cell's median 12 nuclear epithelial markers, the remnant gets **0 in 0/40 pairs**;
  the partial gets 5 in 36/40.
- **76%** of the partial's mass lies *outside* the Xenium cell, and its in-cell portion is
  displaced +1.47 µm toward its out-of-cell mass (33/40 positive, Wilcoxon p = 3.48e-08).
- Decisive: a single epithelial partial recipient draws nuclear transcripts from a median
  of **11 distinct Xenium nuclei** (median 6 nuclei with ≥20, 4 with ≥50). **28 of 30**
  recipients draw ≥50 nuclear transcripts from 2+ nuclei. Largest, `hnomglgj-1-1`: 6,804
  transcripts, 3,175 nuclear, 14 source nuclei, 6 contributing ≥50.

**Conclusion: the hulls overlap because neither component is a cell.** The partial is a
gene-defined regional aggregate spanning ~11 nuclei; the remnant is the polygon-bounded
residue of one cell. Two such sets must overlap.

In fairness to TRACER: it labels these `partial`, and `tracer_whole` excludes them. So the
defect is narrower than "TRACER splits cells wrongly" — it is that **the whole-cell entity
keeps the Xenium cell's ID while its identity-defining transcripts have been moved into a
multi-nucleus partial aggregate.**

(Do **not** "fix" this by running the `tracer_all` arm, which includes partials — I
suggested that and retracted it. Partials are multi-nucleus aggregates; including them
makes the compartment assignment worse, not better, and it also dilutes the 100-NN
smoothing pool: concordant-epithelial RMS is 0.1199 without partials vs 0.0572 with,
because 2,457 shallow partials enter the neighbour pool. **Never compare colours across
panels with different entity sets** — use per-panel colour scales and per-block RMS labels,
as `fig_switch_cnv_partials.png` now does.)

---

## 3. The main unresolved question

**Is the "tumour CNV profile" copy number, or is it an epithelial expression program?**

Clone 0 and clone 4 profiles correlate at 0.82–0.97, which is suspicious for a genuine
subclonal split. Proposed test, not yet run: recompute CNV after excluding genes
differentially expressed between the tumour and reference compartments. If the profile
survives, it is copy number; if it collapses, it was expression.

This matters because §2.5's interpretation depends on it. The user's own framing was that
it does not change the structural conclusion: *"regardless of whether or not inferCNV is
carrying true cnv or epithelial signature, the fact of the matter is we are carving out two
distinct components."*

Other things offered but not started:

- Whether the multi-nucleus property is specific to switcher-neighbourhood partials or
  general to **all 2,457 partials**. This distinguishes a local failure from a general
  property of TRACER's partial construction. **This was the last open question put to the
  user; they never answered it.**
- Extending the paired-score analysis to the other 85 switchers (62 epithelial→myeloid,
  23 epithelial→t_cell); only the 163 epithelial→fibroblast set was analysed in depth.

---

## 4. Reusable technical facts (hard-won, easy to get wrong again)

**`X_cnv` columns are gene-rank windows, not genomic intervals.** infercnvpy's
`_running_mean` convolves a length-60 pyramid kernel with `mode="valid"`, sliding one gene
at a time, then subsamples with `step=10`. Column *j* of chromosome *c* therefore spans
gene ranks `[10j, 10j+59]` of that chromosome's position-sorted gene list.
`uns["cnv"]` stores only `chr_pos` — no coordinates. Intervals are recoverable from
`var[["chromosome","start","end"]]`; `cnv_review_common.window_intervals()` does this.

Two traps that cost a debugging cycle each:

- `exclude_chromosomes=("chrX","chrY")` is the **default** — do not include chrX.
- A chromosome with **fewer than 60 genes** takes a single-convolution branch and yields
  exactly **one** column. Here that is chr21 (57 genes). Including chrX and skipping chr21
  gave 369 reconstructed windows against an actual 354.

Other:

- `dynamic_threshold=1.5` (default) zeroes ~87.6% of `X_cnv` entries.
- Reference-block flatness: 7.3% / 8.0% of windows beyond ±2SD is **chance level**. An
  earlier claim that "the reference compartment is not flat" was withdrawn.
- Per-cell mean `|CNV|` is depth-confounded and dilutes focal events — it is not a usable
  metric. A deletion reads as *low* mean |CNV| relative to a noisier shallow reference,
  which is how clone 0 was once wrongly called "CNV-quiet". Clone 0 is in fact strongly
  aneuploid (chr20 +0.200, chr18 +0.181, chr11 +0.157 under TRACER).
- Correlation distance is NaN on zero-variance CNV rows — cluster only the variable rows
  and append the flat ones.
- InSituCNV refuses to assign clones automatically: `insitucnv/tl/cnv.py:297` raises
  *"Pass tumor_clusters, normal_clusters, or both. No CNV status is inferred
  automatically."* The paper (bioRxiv 10.1101/2025.07.02.662761) states no criterion for
  clone count and defers all parameters to the repo. There is no principled clone count to
  adopt.
- Annotator score (`annotate_entities.marker_scores`) = fraction of an entity's transcripts
  belonging to a marker set. The `--min-marker-hits 2` gate counts *distinct genes*, so it
  is size-dependent. `--rank-by score` is the default.
- TRACER entity types: `cell` (whole; IDs reuse Xenium `cell_id`), `partial`
  (`cascade_*` or `<cell_id>-1`), `unknown` (`-1`).

Environment: conda env `insitucnv_tracer_benchmark`; InSituCNV v0.1.0 commit `44664ff`;
infercnvpy 0.6.1; scanpy 1.11.5. SGE on argos — **argos2 is not a submit host**.

---

## 5. Uncommitted work

Three tracked files are modified and **not committed**:

```
tutorials/insitucnv_tracer_benchmark/scripts/common.py
tutorials/insitucnv_tracer_benchmark/scripts/run_insitucnv_arm.py
tutorials/insitucnv_tracer_benchmark/sge/run_insitucnv_reference.sge
```

They contain the §1.1 and §1.2 fixes:

- `common.py`: new `build_expression_neighbors`, `build_spatial_neighbors` (moved here from
  the arm script), `prepare_smoothing_graph`.
- `run_insitucnv_arm.py`: `--smoothing-graph` (default `expression`),
  `--expression-neighbors` (default 100), `--neighbor-n-pcs` (default 50); pre-smoothing
  `log1p` removed; `arm_stats.json` records the smoothing settings plus a post-run
  `smoothing_graph_realised`.
- `run_insitucnv_reference.sge`: threads `SMOOTHING_GRAPH`, `EXPRESSION_NEIGHBORS`,
  `NEIGHBOR_N_PCS`, `SMOOTH_NEIGHBORS`, `WINDOW_SIZE`, `CLUSTER_RES` through as env vars.

Suggested commit message (no Claude trailers, per `CLAUDE.md`):

```
Smooth CNV input over the expression graph, not spatial neighbours

The arms smoothed each entity against its 200 nearest spatial neighbours,
which re-mixes transcripts across cell boundaries and makes CNV a spatial
field rather than a cell property. Switch the default to the expression/PCA
kNN graph the published recipe uses, keep spatial as an opt-in flag, and
drop the pre-smoothing log1p so inferCNV sees linear-scale input like the
ground truth did.
```

Dead code still present in `run_insitucnv_arm.py`: `TRACER_ALL_ETYPES` and the
`--arm tracer_all` branch are defined but unused.

Older spatial-smoothing outputs are snapshotted at
`../_snapshot_spatial_smoothing_200nn/` for comparison.

Unrelated loose ends: PR #11 (`ci/latest-tag-on-default-branch`) is unmerged; three
untracked `tests/test_*.py` files still need a skip guard; `AGENTS.md` and `CLAUDE.md` are
untracked.
