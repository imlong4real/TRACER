# Design note: one prune for cells and grouped entities

Status: proposal, 2026-11-13. Based on the code at `1e7e1a4` and on prototype runs on an internal Xenium dataset. Results from that dataset are unpublished and are kept out of this repository.

## Problem

TRACER prunes in two places:

- **Phase-1 Prune** works on segmented cells.
- **The Group cascade** works on seed windows of unassigned transcripts.

Conceptually these are the same operation: find the coherent gene program in a set of transcripts, keep it, and carve out or release the rest. The two paths share the greedy kernel, but everything around the kernel was written separately and has drifted apart. Each recent cascade fix has rebuilt something Phase-1 already does.

## What is shared today

Both paths call the same greedy core, `_cy_prune._greedy_prune_to_retained`:

- Conflict scores are weighted by transcript count.
- A gene whose own transcript count exceeds its conflict score is protected from removal.
- Pairs with PMI below `PMI_THR` (0.2) count as conflicts.
- Pairs missing from the panel are skipped.

The callers are:

| Caller | Where |
|---|---|
| Phase 1a seed (nuclear or whole cell) | `_prune_cells_nuclear_seed_core` |
| Phase 1c sub-seed | `_prune_cells_nuclear_seed_core` |
| Non-nuclear whole-cell prune | `prune_cells_retained` (via `prune_transcripts_fast`) |
| Cascade seed | `greedy_prune_retained` (via `density_cascade._greedy_prune_retained`) |

Both paths also treat a pair that is stored with value 0 as a conflict, because 0 < 0.2. So that is a panel-format question, not a difference between the paths.

## Where the paths diverge

| Behaviour | Phase-1 Prune | Cascade seed prune | Observed effect |
|---|---|---|---|
| **W contract** | Builds symmetric CSR arrays itself (`_symmetric_csr_arrays`) | Passes `W.tocsr()` as received. On the nuclear-seed path `aux["W"]` is upper-triangle only. | Conflict bookkeeping goes negative and pruning stops early, so cascade seeds keep conflicting genes. Symmetrizing W fixed this in the prototype and lowered the unassigned fraction. |
| **Membership after the seed** | 1b: every transcript is tested against the seed with `_admission_test` (mean, percentile and veto modes). A gene outside the seed can still be admitted if it fits. | A transcript joins only if its gene is in the retained set. There is no admission test. | Cascade entities are stricter on membership. Not measured separately. |
| **Rejected transcripts** | 1c: the rejected nuclear genes are pruned again into a sub-seed (at least `min_nuclear_genes`), with admission. That produces a carved partial; the rest go to unassigned. | All go back to the pool (window mode, the repo default). | In the prototype, cell-style carving (`TRACER_CASCADE_PRUNE=cell`) raised partial coherence and reduced unassigned transcripts. |
| **Size minimum** | Phase1-QC sends entities under `PHASE1_QC_MIN_TX` (3) to unassigned. | A seed needs at least `min_anchor_tx` (3) committed transcripts. The prototype sibling had no minimum. | The prototype produced many siblings under 3 transcripts. |
| **z-split** | Split-Phase1 splits at gaps over 2 µm. | None. The seed window is 3×3 bins in xy across all z. | Not measured. |
| **Re-ranking the main entity** | Phase1-Rerank picks the main by nuclear transcript count. | Not applicable (no nucleus). | – |
| **Remerging over-splits** | Phase1-Maha-Remerge (D ≤ 1, ΔC > −0.2) | Only later, through Stitch's Mahalanobis rescue | – |
| **Config source** | `cfg.phase1.*` plus module constants | NOSEG path: `cfg.group.*`. SEG path: hard-coded literals (`G=2.0`, `territory_radius_bins=1`, `min_anchor_tx=3`, `PMI_THR`), so `cfg.group` is ignored. | Settings silently differ between SEG and NOSEG. |

There is also a legacy Group path (`annotate_unassigned_components_fast`, used when `PHASE1_SEG_RESIDUAL_CASCADE=False`). It uses a third implementation, the pure-Python `prune_genes_by_pmi_greedy`, which:

- counts each conflicting gene once instead of weighting by transcripts;
- uses a different default threshold (`ANNOTATE_NEG_THR`).

It is off by default.

## Proposal

One orchestration function that both paths call:

```python
prune_entity(tx_idx, gene_idx, coords, W_sym, cfg, *, seed_mask=None)
    -> PruneResult(main: idx, carved: list[idx], rejected: idx)
```

1. **One W contract.** `W_sym` is a symmetric CSR built once, at panel load, by a single helper. `aux["W"]` always carries this form, with the convention stated in its docstring. Every consumer reads the same object: Prune, cascade, Rescue, Mid-QC, Stitch and Reassign-1c. Stitch's `W + W.T` becomes a no-op, or an assertion.
2. **Seed.** Greedy prune over the seed genes:
   - Phase-1 sets `seed_mask` to nuclear or whole-cell transcripts.
   - The cascade uses the whole window.
   - The kernel is unchanged.
3. **Admission (1b).** Each transcript is tested against the seed with `_admission_test` under `cfg.prune.admission`. The cascade gains admission; whether it should use the same veto settings as Phase-1 is an open question.
4. **Carve (1c).** The prune is repeated on the rejected genes to give a sub-seed with admission, up to `cfg.prune.max_carves` (default 1). The prototype's cell-style sibling becomes this step.
5. **Shared post-steps, applied to main and carved:** z-split (`cfg.prune.split_dz`), then the size minimum (`cfg.prune.min_tx`), then the optional Maha remerge.
6. **The caller decides what happens to rejected transcripts.**
   - Phase-1 sends them to unassigned, for Rescue.
   - The cascade returns them to the pool for later seeds.

   This is the only intended behavioural difference between the two paths.

Nuclear-specific steps stay outside `prune_entity` as Phase-1 wrappers: the seed mask, nuclear-only admission, Reassign-1c and Rerank.

### Settings that collapse

| Today | After |
|---|---|
| `PHASE1_QC_MIN_TX`, `group.cascade_min_anchor_tx`, prototype `TRACER_CASCADE_SIB_MIN` | `prune.min_tx` |
| `SPLIT_PHASE1_DZ`, plus a missing cascade equivalent | `prune.split_dz` |
| Prototype `TRACER_CASCADE_PRUNE` | `prune.max_carves`: 0 (window behaviour) for the cascade, by decision; 1 for Phase 1 |
| SEG-path hard-coded cascade literals | `cfg.group.*`, read on both paths |

## Migration plan (each step is its own PR)

1. **Lock in current output.** Add bit-identical regression fixtures for Phase-1 and Group outputs on a small Xenium ROI, with the nucleus column, and on a NOSEG ROI.
   *Done:* `tests/test_stage_label_regression.py` hashes every transcript's label at every recorded stage for three variants (SEG with nucleus, SEG whole-cell, NOSEG) on a 30-cell synthetic section with simulated DAPI segmentation. It uses synthetic data rather than a real ROI so the fixture can live in the repo. Checked: the hashes are stable across `PYTHONHASHSEED`; raising the Phase1-QC minimum first diverges at Phase1-QC; symmetrizing the cascade W first diverges at Group, with every Phase-1 stage unchanged.
2. **One W contract.** Symmetrize at panel load. Audit every `aux["W"]` consumer: Stitch, Rescue, Final Rescue, Mid-QC, Reassign-1c, coherence scoring. Expected diff: the cascade changes; Stitch must stay bit-identical. This is the bug fix and can land on its own.
   *Done, in two commits.* The audit found two mirror-image bugs: on the nuclear-seed path the consumers that read W as given (cascade seed prune, Rescue's negative-gene veto, Mid-QC) saw only the upper triangle, and on the whole-cell and NOSEG paths, whose W was already symmetric, the consumers that symmetrized on their own (Stitch `W + W.T`, the coherence slice, the housekeeping count) doubled every value. 2a makes those consumers idempotent through `tracer._utils.symmetric_csr` (nuclear path bit-identical; whole-cell and NOSEG first change at Stitch). 2b makes the nuclear-seed path pass a symmetric W (Phase-1 bit-identical; first change at Rescue). `tests/test_w_contract.py` pins the contract. The prediction above that only the cascade would change was wrong: Rescue and Mid-QC also read the triangle.
3. **Extract `prune_entity` from the Phase-1 path, behaviour unchanged.** Phase-1 output must stay bit-identical.
   *Done, at the kernel level, as two layers.* The seed core `_cy_prune._seed_entity` (count transcripts per gene with one `np.unique`, then the tx-weighted greedy; returns the seed genes ascending) is shared by every seed in TRACER: Phase-1 1a (nuclear or whole cell), the 1c sub-seed, and each cascade window. Phase-1's per-cell body (seed, admission, one carve) is `_cy_prune._prune_entity`, built on that core and called once per cell from compiled code; its Python entry is `_cy_prune.prune_entities(entity_tx_idx_lists, tx_gene_idx, tx_seed_mask, W_csr…, min_seed_genes, max_carves, seed_from_mask, admit_mask_only, …)`, which `pruning.prune_transcripts_nuclear_seed` calls. **Deviations from the proposal:** (1) layered rather than one flag-driven `prune_entity`: after the decision in step 5, the cascade needs only the seed, so it calls the seed core directly instead of `_prune_entity` with admission and carving switched off (an interim flag-driven version cost about 0.2 s per ROI in per-window bookkeeping); (2) the post-steps (z-split, size minimum, Maha remerge) are not shared yet. In Phase 1, Rerank (nuclear-only) runs between Split-Phase1 and Phase1-QC, so bundling them would reorder Phase 1.
4. **Move the cascade onto `prune_entity` with `max_carves=0` and no admission.** Group output must stay bit-identical to step 2.
   *Done.* Each seed window in `density_cascade_phase1` calls `_cy_prune.seed_entity` (the shared seed core) and commits the transcripts of the seed genes; the rest return to the pool. The CSR arrays are built once per cascade call (`_csr_arrays_for_prune`; a dense W keeps its zeros as stored entries) instead of once per seed. This covers both the SEG residual cascade and NOSEG, which call the same function. Merge note: Alex's TRACER-MT fork (multi-process tiling) edits the cascade's threshold block and pool loop and relies on the hot-bin visit order and anchor naming; this step changes only the per-window body.
5. **Change the cascade's behaviour, one flag at a time:** carving (`max_carves=1`), the size minimum on carved entities, the z-split, then admission. Measure each on a real ROI (coverage, coherence, RCTD max weight), and check runtime and peak memory with `scripts/bench_stages.py`.
   *Decision (2026-11-13): the cascade keeps one partial per window.* Its job is: take the densest 3×3 bin neighbourhood of unassigned transcripts, extract one coherent partial, return the rest to the pool, repeat. So carving, the carved-entity size minimum and admission of non-seed genes are dropped: the cascade uses only the shared seed core (see step 4). The prototype's case for carving came from the broken W: before step 2, window-mode partials had low coherence and carving raised it; after step 2 they are already coherent. The remaining candidate is the **z-split**: a window spans all z, so stacked cells can fuse into one partial. Measure it before changing code.
   *Done (z-split + size QC).* The cascade's partials now go through the same post-steps as Phase-1 entities, right after Group: `_spatial_split_phase1_entities` (Δz > `CASCADE_SPLIT_DZ` = `SPLIT_PHASE1_DZ` = 2 µm) and then `_qc_demote_small_phase1_entities` with `CASCADE_QC_MIN_TX` = 5 (Demote's minimum; Phase-1 uses 3), both restricted to `cascade_` labels so Phase-1 entities enlarged by Rescue are not re-split. New stages `Cascade-Split` and `Cascade-QC`; `None` turns either off. Before this, nothing z-split cascade partials: Mid-QC's `_split_unassigned_components` only matches the legacy `UNASSIGNED_*` labels. Effect on internal ROIs (numbers kept out of the repo): identical through Group everywhere; no-op on Visium HD (2D); on Xenium a small rise in unassigned transcripts with partials that are compact in z; on MERFISH (discrete 1.5 µm planes) the largest effect, where many final cascade partials had spanned a z gap. The 2 µm threshold is fixed across platforms; on the MERFISH and CosMx inputs it cuts under ~1% of input cells, so a plane-aware threshold was not needed.
   Runtime: the two stages add well under a second per run; total runtime and peak RSS are unchanged within noise (`bench_stages.py`, all three ROIs). Decision (2026-11-14): keep both.
6. **SEG path reads `cfg.group`.** Remove the hard-coded literals. In the same change, move the post-step switches into config so each can be turned on or off separately: `GroupConfig.cascade_split_dz_um` and `GroupConfig.cascade_qc_min_tx` (`None` = off) replace `CASCADE_SPLIT_DZ` / `CASCADE_QC_MIN_TX`, and wire the existing but unread `SplitPhase1Config` and `Phase1QcConfig` into Split-Phase1 and Phase1-QC (today they read `SPLIT_PHASE1_DZ` / `PHASE1_QC_MIN_TX`, so a YAML value is silently ignored). One change to `run_segmented_pipeline`, to keep the TRACER-MT merge to one conflict.
   *Done.* One `_group_stage(df, aux, cfg, progression, seg=)` now runs Group for both runners: backend switch (`group.seg_residual_cascade` / `noseg_cascade`), every cascade setting, then Cascade-Split and Cascade-QC, each switched by its own `group` field (0 or `None` = off; 0 is the TOML spelling, since TOML has no null). The two module constants added in step 5 are gone. Split-Phase1 reads `cfg.split_phase1` and Phase1-QC reads `cfg.phase1_qc`. With `cfg=None`, `_resolve_pipeline_cfg` builds these three sections from the module constants, so tests that patch constants still work and `cfg=None` equals the bundled defaults (pinned in `tests/test_group_config_wiring.py`). `run_tracer.py --pmi-threshold` now also writes the value into `phase1.pmi_threshold` and `group.cascade_pmi_threshold`: the SEG cascade used to read `PMI_THR` and now reads config, and the receipt now shows the override. The legacy backend keeps its literals (`min_comp_size` 4 SEG / 5 NOSEG, versus `legacy_min_comp_size` = 5 in config); step 7 retires it. Labels identical to step 5 at every stage on the fixtures and on the internal ROIs.
7. **Retire the legacy Group path** (`annotate_unassigned_components_fast` with `prune_genes_by_pmi_greedy`), or route it through `prune_entity`.
   *Done (2026-11-14): retired.* Group is always the density cascade. Removed from the pipeline: the spatial-CC branch of `_group_stage`, its graph helper `_grid_self_graph_fn`, `ANNOTATE_NEG_THR`, the backend constants `PHASE1_SEG_RESIDUAL_CASCADE` / `PHASE1_NOSEG_CASCADE`, and Mid-QC's `_split_unassigned_components` with `MID_SPLIT_UNASSIGNED_DZ` (it matched only `UNASSIGNED_<n>` components, so it was a no-op on the cascade path; Mid-QC is now the coherence floor only). Removed from config: `group.seg_residual_cascade`, `group.noseg_cascade` and the six `group.legacy_*` fields; a config that still sets one fails with a migration hint (`_RETIRED_KEYS`). Kept for one release with a `DeprecationWarning`: the exported `annotate_unassigned_components`, `annotate_unassigned_components_fast` and `prune_genes_by_pmi_greedy`. Not touched: `UNASSIGNED` handling elsewhere (`_etype`, QC skip lists, plotting, metrics), since `UNASSIGNED` is also the Finalize sentinel and old outputs carry `UNASSIGNED_<n>` labels. Labels identical at every stage on the fixtures and on the internal ROIs.

## Open questions

- Should the cascade use Phase-1's admission settings (`veto_mode`, `mean_admit_threshold`, `aggregator_percentile`)? The cascade has no nucleus to anchor identity, so a stricter admission may be warranted.
- Should 1c be allowed to recurse more than once (`max_carves > 1`)?
- Explicit zeros: should a pair stored as 0 count as a conflict, or be skipped like a missing pair? This affects both paths equally, for example on panels with zeroed unsupported pairs.
- Phase-1 sends rejected transcripts to unassigned while the cascade sends them back to its pool. Should Phase-1 rejects instead feed the cascade pool directly?

## Evidence

Prototype runs on an internal dataset compared cell-style carving, cell-style carving with a sibling size minimum, and the repo's window mode. Cell-style carving gave more partials with higher coherence and fewer unassigned transcripts, and left whole-cell coherence unchanged. The numbers are kept outside this repository because the dataset is unpublished.

The prototype patches are environment flags in a scratch copy of TRACER; none are in the repo. The flags are:

- `TRACER_CASCADE_SYM`
- `TRACER_CASCADE_PRUNE`
- `TRACER_CASCADE_SIB_MIN`
- `STITCH_CUNION_MIN`
- `STITCH_CELL_DCMIN`
- `STITCH_CUM_DCMIN`
- `TRACER_SIM_REEXPAND`
